from __future__ import annotations

import json
import os
import io
import hashlib
import re
from datetime import date
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st
from openai import OpenAI
from pydantic import BaseModel, Field
from streamlit.errors import StreamlitSecretNotFoundError
from streamlit_ace import st_ace
from streamlit_chat import message

from utils.data_loader import load_interview_data
from utils.interviewer_store import (
    InterviewerProfile,
    delete_interviewer,
    get_interviewers_by_jd,
    list_interviewers_with_paths,
    save_interviewer,
)
from utils.personas import build_system_prompts
from utils.security import validate_input
from utils.profile_health import (
    DEFAULT_PROFILE_LABEL as AUDIT_DEFAULT_PROFILE_LABEL,
    audit_interview_profiles,
    recommend_interviewers,
)


def get_api_key() -> str | None:
    try:
        secret_key = st.secrets.get("OPENAI_API_KEY")
    except StreamlitSecretNotFoundError:
        secret_key = None
    return secret_key or os.getenv("OPENAI_API_KEY")

try:
    from annotated_text import annotated_text
except ImportError:
    annotated_text = None

try:
    from pypdf import PdfReader
except ImportError:  # pragma: no cover - optional runtime dependency
    PdfReader = None


def load_local_env() -> None:
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


load_local_env()

DATA_DIR = Path(__file__).resolve().parent / "data"
PROFILES_DIR = DATA_DIR / "profiles"
INTERVIEWEE_PROFILE_PATH = DATA_DIR / "interviewee_profile.json"
JD_PROFILE_SESSION_KEY = "session_interviewers"
TEMPERATURE_RULES_PATH = Path(__file__).resolve().parent / "utils" / "model_temperature_constraints.json"
CHAT_CAPABLE_MODELS = [
    "gpt-4o-mini-2024-07-18",
    "gpt-4o-mini",
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-0125",
    "gpt-3.5-turbo-1106",
    "gpt-4.1-mini",
    "gpt-4.1-mini-2025-04-14",
    "gpt-4.1-nano",
    "gpt-5-nano",
    "gpt-5-mini",
]
CHAT_FALLBACK_ORDER = [
    "gpt-4o-mini",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
    "gpt-5-mini",
    "gpt-5-nano",
    "gpt-3.5-turbo",
]
TEMPERATURE_RULES_FALLBACK = {
    "default": {"min": 0.0, "max": 2.0, "fallback": 1.0},
    "rules": [],
}


def _coerce_to_float(value: Any, fallback: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _load_temperature_rules() -> Dict[str, Any]:
    if not TEMPERATURE_RULES_PATH.exists():
        return TEMPERATURE_RULES_FALLBACK

    try:
        raw = json.loads(TEMPERATURE_RULES_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return TEMPERATURE_RULES_FALLBACK

    if not isinstance(raw, dict):
        return TEMPERATURE_RULES_FALLBACK

    default_policy = raw.get("default") if isinstance(raw.get("default"), dict) else {}
    default_min = _coerce_to_float(
        default_policy.get("min"),
        TEMPERATURE_RULES_FALLBACK["default"]["min"],
    )
    default_max = _coerce_to_float(
        default_policy.get("max"),
        TEMPERATURE_RULES_FALLBACK["default"]["max"],
    )
    default_fallback = _coerce_to_float(
        default_policy.get("fallback"),
        TEMPERATURE_RULES_FALLBACK["default"]["fallback"],
    )
    normalized = {
        "default": {
            "min": default_min,
            "max": default_max,
            "fallback": default_fallback,
        },
        "rules": [
            rule for rule in raw.get("rules", []) if isinstance(rule, dict)
        ],
    }
    return normalized


TEMPERATURE_RULES = _load_temperature_rules()


def _coerce_temperature_for_model(model: str, temperature: float) -> float:
    """Return a model-compatible temperature based on external rule config."""
    default_policy = TEMPERATURE_RULES.get("default", TEMPERATURE_RULES_FALLBACK["default"])
    min_temperature = float(default_policy.get("min", 0.0))
    max_temperature = float(default_policy.get("max", 2.0))
    fallback_temperature = float(default_policy.get("fallback", 1.0))

    model_name = str(model).strip().lower()
    for rule in TEMPERATURE_RULES.get("rules", []):
        patterns = rule.get("patterns") or []
        if isinstance(patterns, str):
            patterns = [patterns]
        if not isinstance(patterns, list) or not patterns:
            continue

        match_mode = str(rule.get("match_mode", "contains")).strip().lower()
        if not any(
            (pattern and isinstance(pattern, str))
            and (
                (match_mode == "equals" and pattern.strip().lower() == model_name)
                or (
                    match_mode in {"contains", ""} and pattern.strip().lower() in model_name
                )
                or (
                    match_mode == "startswith" and model_name.startswith(pattern.strip().lower())
                )
            )
            for pattern in patterns
        ):
            continue

        rule_min = _coerce_to_float(rule.get("min"), min_temperature)
        rule_max = _coerce_to_float(rule.get("max"), max_temperature)
        rule_temperature = rule.get("temperature")
        if isinstance(rule_temperature, (int, float)):
            return max(rule_min, min(float(rule_temperature), rule_max))

        if not isinstance(temperature, (int, float)):
            return fallback_temperature
        return max(rule_min, min(float(temperature), rule_max))

    if not isinstance(temperature, (int, float)):
        return fallback_temperature

    return max(min_temperature, min(float(temperature), max_temperature))


def _sync_temperature_to_model_constraint() -> None:
    current_temperature = st.session_state.get("temperature", 0.4)
    st.session_state["temperature"] = _coerce_temperature_for_model(
        st.session_state.get("selected_model", ""),
        current_temperature,
    )

APP_TITLE = "AI Interview Coach"

st.set_page_config(page_title=APP_TITLE, page_icon="🎯", layout="wide")

st.sidebar.title(APP_TITLE)

GENERIC_INTERVIEWER_KEY = "generic_ai_interviewer"
GENERIC_INTERVIEWER_LABEL = "Generic AI Interviewer"
MANDATORY_SESSION_INTERVIEWERS = [GENERIC_INTERVIEWER_LABEL]


def _speaker_key(name: str) -> str:
    lowered = re.sub(r"[^a-z0-9]+", "_", name.strip().lower())
    lowered = lowered.strip("_")
    if not lowered:
        return "interviewer"
    if "generic" in lowered:
        return GENERIC_INTERVIEWER_KEY
    return lowered


def _speaker_label(name: str) -> str:
    return name.strip() or "Interviewer"


def _build_session_interviewer_profiles(data: Dict[str, Any], jd_title: str) -> List[Dict[str, Any]]:
    profiles: List[Dict[str, Any]] = []
    seen: set[str] = set()

    for interviewer in data.get("interviewers", []):
        if not isinstance(interviewer, dict):
            continue
        profile = dict(interviewer)
        profile.setdefault("role", "")
        profile.setdefault("expertise", [])
        profile.setdefault("background", "")
        key = _speaker_key(str(profile.get("name", "")))
        if key in seen:
            continue
        profile["is_generic_ai"] = bool(profile.get("is_generic_ai", key == GENERIC_INTERVIEWER_KEY))
        seen.add(key)
        profiles.append(profile)

    for profile in get_interviewers_by_jd(jd_title):
        if hasattr(profile, "model_dump"):
            payload = profile.model_dump()
        else:
            payload = dict(profile)
        payload["is_generic_ai"] = bool(payload.get("is_generic_ai", False))
        key = _speaker_key(str(payload.get("name", "")))
        if key in seen:
            continue
        if key == GENERIC_INTERVIEWER_KEY:
            payload["is_generic_ai"] = True
        seen.add(key)
        profiles.append(payload)

    generic_profile = {
        "name": GENERIC_INTERVIEWER_LABEL,
        "role": "AI Recruiter",
        "background": "Standard Capgemini hiring recruiter.",
        "expertise": [],
        "is_generic_ai": True,
    }
    generic_key = _speaker_key(generic_profile["name"])
    if generic_key not in seen:
        profiles.append(generic_profile)
        seen.add(generic_key)
    return profiles


def _build_speaker_options_from_profiles(
    profiles: List[Dict[str, Any]],
) -> tuple[List[str], Dict[str, str]]:
    options: List[str] = [GENERIC_INTERVIEWER_LABEL]
    label_to_key = {GENERIC_INTERVIEWER_LABEL: GENERIC_INTERVIEWER_KEY}
    seen = {GENERIC_INTERVIEWER_KEY}

    for profile in profiles:
        key = _speaker_key(profile.get("name", ""))
        if key in seen:
            continue

        label = _speaker_label(str(profile.get("name", "")))
        options.append(label)
        label_to_key[label] = key
        seen.add(key)

    return options, label_to_key


def _selected_profile_path(selected_profile: str) -> Path:
    if selected_profile == DEFAULT_PROFILE_LABEL:
        return DATA_DIR / "interview_data.json"
    return PROFILES_DIR / selected_profile


def _normalize_label_list(values: List[str]) -> List[str]:
    output: List[str] = []
    seen: set[str] = set()
    for value in values:
        candidate = str(value).strip()
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        output.append(candidate)
    return output


def _normalize_text_list(values: Any) -> List[str]:
    if not isinstance(values, list):
        return []

    output: List[str] = []
    seen: set[str] = set()
    for value in values:
        item = str(value).strip()
        if not item:
            continue
        item = re.sub(r"\s{2,}", " ", item).strip()
        item_key = item.lower()
        if item_key in seen:
            continue
        seen.add(item_key)
        output.append(item)
    return output


class ParsedIntervieweeProfile(BaseModel):
    name: str = Field(default="")
    status: str = Field(default="")
    core_strengths: List[str] = Field(default_factory=list)
    focus_phrases: List[str] = Field(default_factory=list)


def _load_interviewee_profile() -> Dict[str, Any]:
    if not INTERVIEWEE_PROFILE_PATH.exists():
        return {}

    try:
        raw_profile = json.loads(INTERVIEWEE_PROFILE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        st.error("Failed to load interviewee profile JSON.")
        return {}

    if not isinstance(raw_profile, dict):
        st.error("Interviewee profile JSON must be an object.")
        return {}

    return raw_profile


def _attach_interviewee_profile(profile_data: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(profile_data, dict):
        return {}

    interviewee_profile = _load_interviewee_profile()
    if interviewee_profile:
        profile_data["candidate_profile"] = interviewee_profile
    return profile_data


def _extract_text_from_pdf(uploaded_file) -> str:
    if uploaded_file is None:
        return ""

    if PdfReader is None:
        raise RuntimeError(
            "PDF parsing library is not available. Add `pypdf` to requirements and restart."
        )

    uploaded_file.seek(0)
    pdf_binary = uploaded_file.read()
    if not pdf_binary:
        return ""

    try:
        reader = PdfReader(io.BytesIO(pdf_binary))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages)
    except Exception as exc:
        raise RuntimeError(f"Failed to parse PDF content: {exc}") from exc


def _as_interviewee_payload(parsed: ParsedIntervieweeProfile) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    payload["name"] = str(parsed.name).strip() or "Unknown"
    payload["status"] = str(parsed.status).strip()
    payload["core_strengths"] = _normalize_text_list(parsed.core_strengths)
    payload["probes"] = {"focus_phrases": _normalize_text_list(parsed.focus_phrases)}

    if not payload["core_strengths"]:
        payload["core_strengths"] = []
    if not payload["probes"]["focus_phrases"]:
        payload["probes"]["focus_phrases"] = [
            (
                "Specifically probe the candidate on how their ADAS Perception experience "
                "directly solves the requirements in this Job Description."
            )
        ]
    return payload


def parse_interviewee_profile(
    api_key: str,
    source_text: str,
    model: str,
    source_label: str = "resume",
) -> ParsedIntervieweeProfile:
    client = OpenAI(api_key=api_key)
    base_prompt = (
        "You are a strict extraction assistant.\n"
        "Extract candidate profile fields and return valid content in English only.\n"
        "If input includes non-English text, translate it into natural English first.\n"
    )
    if source_label == "cover_letter":
        system_prompt = (
            f"{base_prompt}"
            "Only extract additional strengths and interview-probing phrases from the cover letter.\n"
            "Do not alter name/status unless explicitly stated in the cover letter."
        )
    else:
        system_prompt = (
            f"{base_prompt}"
            "From the resume/CV text, extract the full candidate profile fields in JSON-compatible form.\n"
            "Fill all available fields: name, status, core_strengths, focus_phrases."
        )

    request_text = (
        f"Document source: {source_label}\n\n"
        f"Text:\n{source_text}"
    )

    models_to_try = [model] + [m for m in CHAT_FALLBACK_ORDER if m != model]
    last_error: Exception | None = None

    for model_name in models_to_try:
        try:
            resolved_temperature = _coerce_temperature_for_model(model_name, 0.2)
            response = client.beta.chat.completions.parse(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": request_text},
                ],
                temperature=resolved_temperature,
                response_format=ParsedIntervieweeProfile,
            )
            parsed = response.choices[0].message.parsed
            if parsed is None:
                raise ValueError("Failed to parse structured interviewee profile.")
            return parsed
        except Exception as exc:
            last_error = exc
            if is_model_access_error(exc):
                continue
            try:
                fallback_response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": (
                            "Extract fields and output JSON only with keys: "
                            "name, status, core_strengths, focus_phrases."
                        )},
                        {"role": "user", "content": request_text},
                    ],
                    temperature=resolved_temperature,
                )
                raw_content = fallback_response.choices[0].message.content or ""
                json_payload = _extract_json_payload(raw_content)
                if not json_payload:
                    raise ValueError("Could not extract JSON block from model response.")
                parsed = ParsedIntervieweeProfile.model_validate_json(json_payload)
                return parsed
            except Exception:
                continue

    if last_error is not None:
        raise last_error
    raise ValueError("Failed to parse interviewee profile with available models.")


def _save_interviewee_profile(profile_payload: Dict[str, Any]) -> None:
    payload = dict(profile_payload) if isinstance(profile_payload, dict) else {}
    payload["name"] = str(payload.get("name", "")).strip() or "Unknown"
    payload["status"] = str(payload.get("status", "")).strip()

    payload["core_strengths"] = _normalize_text_list(payload.get("core_strengths", []))
    probes = payload.get("probes")
    if not isinstance(probes, dict):
        probes = {}
    probes["focus_phrases"] = _normalize_text_list(probes.get("focus_phrases", []))
    payload["probes"] = probes
    if not payload["core_strengths"]:
        payload["core_strengths"] = []
    if not payload["probes"]["focus_phrases"]:
        payload["probes"]["focus_phrases"] = [
            "Probe the candidate's ADAS Perception achievements against the JD requirements."
        ]

    INTERVIEWEE_PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    INTERVIEWEE_PROFILE_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _merge_interviewee_cover_letter(base: Dict[str, Any], addition: ParsedIntervieweeProfile) -> Dict[str, Any]:
    merged = dict(base)
    parsed_payload = addition.model_dump()
    incoming_name = str(parsed_payload.get("name", "")).strip()
    incoming_status = str(parsed_payload.get("status", "")).strip()
    incoming_strengths = _normalize_text_list(parsed_payload.get("core_strengths", []))
    incoming_focus = _normalize_text_list(parsed_payload.get("focus_phrases", []))

    if incoming_name:
        merged["name"] = incoming_name
    if incoming_status:
        merged["status"] = incoming_status

    merged.setdefault("core_strengths", [])
    merged.setdefault("probes", {})
    merged["core_strengths"] = _normalize_text_list(
        _normalize_text_list(merged.get("core_strengths", [])) + incoming_strengths
    )
    merged["probes"]["focus_phrases"] = _normalize_text_list(
        _normalize_text_list(merged["probes"].get("focus_phrases", [])) + incoming_focus
    )

    if not merged["probes"]["focus_phrases"]:
        merged["probes"]["focus_phrases"] = [
            "Probe the candidate's ADAS Perception achievements against the JD requirements."
        ]

    return merged


def _build_default_session_interviewers(data: Dict[str, Any]) -> List[str]:
    candidates = list(MANDATORY_SESSION_INTERVIEWERS)
    return _normalize_label_list(candidates)


def _filter_session_interviewers(
    stored: List[str],
    available: List[str],
    mandatory: List[str],
    enforce_mandatory: bool = True,
) -> List[str]:
    filtered = [value for value in stored if value in available]
    if not filtered:
        if enforce_mandatory:
            return _normalize_label_list(list(mandatory))
        return _normalize_label_list([GENERIC_INTERVIEWER_LABEL])
    if not enforce_mandatory:
        return _normalize_label_list(filtered)
    return _normalize_label_list(filtered + mandatory)


def _persist_session_interviewers(profile_path: Path, profile_data: Dict[str, Any], session_interviewers: List[str]) -> None:
    write_payload = dict(profile_data)
    write_payload[JD_PROFILE_SESSION_KEY] = session_interviewers
    write_payload.pop("candidate_profile", None)
    try:
        profile_path.write_text(json.dumps(write_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        st.error(f"Failed to persist session interviewer candidates to '{profile_path.name}'.")


def _render_profile_status_panel() -> None:
    report = audit_interview_profiles(DATA_DIR)
    if not report:
        st.sidebar.caption("No profile files were found.")
        return

    total = len(report)
    healthy = sum(1 for item in report if item["status"] == "ok")
    needs_attention = sum(1 for item in report if item["status"] in {"invalid_file", "missing_required"})

    st.sidebar.divider()
    st.sidebar.subheader("JD Profile Status")
    metric_col_total, metric_col_healthy, metric_col_needs = st.sidebar.columns(3)
    metric_col_total.metric("Total", total)
    metric_col_healthy.metric("Healthy", healthy)
    metric_col_needs.metric("Needs", needs_attention)

    selected = st.session_state.get("selected_jd_profile", AUDIT_DEFAULT_PROFILE_LABEL)
    for item in report:
        missing_required = item.get("missing_required", [])
        recommendations = item.get("recommendations", [])
        top_recommendations = recommendations[:5]

        if item["status"] == "ok":
            icon = "✅"
            status_label = "Complete"
        else:
            icon = "⚠️"
            status_label = "Needs attention"

        file_label = item["file"]
        if item["file"] == "interview_data.json":
            file_label = AUDIT_DEFAULT_PROFILE_LABEL

        session_list = ", ".join(item.get("session_interviewers", []))
        if not session_list:
            session_list = "(empty)"

        if file_label == selected:
            st.sidebar.markdown(f"**{icon} {file_label}**")
            st.sidebar.caption(f"Status: {status_label}")
            st.sidebar.caption(f"Current session_interviewers: {session_list}")
            if not top_recommendations:
                st.sidebar.caption("No interviewer correlation score found.")
            else:
                st.sidebar.caption("Interviewer-JD Correlation")
                for rec in top_recommendations:
                    label = rec.get("name", "Interviewer")
                    score = rec.get("score", 0.0)
                    status = "recommended" if rec.get("recommended") else "candidate"
                    st.sidebar.caption(f"{label} | {status} | {score:.1f}%")
        else:
            st.sidebar.markdown(f"{icon} {file_label}")

        if missing_required:
            st.sidebar.caption(f"Missing required: {', '.join(missing_required)}")


def _normalize_list_text(value: str) -> List[str]:
    return [item.strip() for item in re.split(r"[\r\n,]+", value or "") if item.strip()]


def _collect_interviewer_records(selected_jd: str) -> List[tuple[Path, InterviewerProfile]]:
    return list_interviewers_with_paths(selected_jd)


def _collect_interviewer_names(records: List[tuple[Path, InterviewerProfile]]) -> List[str]:
    names: List[str] = []
    seen = set()
    for _, profile in records:
        name = str(profile.name).strip()
        key = name.lower()
        if not key or key in seen:
            continue
        seen.add(key)
        names.append(name)
    return names


def _interviewer_path_options(records: List[tuple[Path, InterviewerProfile]]) -> Dict[str, InterviewerProfile]:
    out: Dict[str, InterviewerProfile] = {}
    for path, profile in records:
        out[str(path)] = profile
    return out


def _interviewer_display_label(profile: InterviewerProfile) -> str:
    return profile.name


def _bootstrap_default_interviewers(current_data: Dict[str, Any]) -> None:
    if not isinstance(current_data, dict):
        return

    profile_interviewers = current_data.get("interviewers", [])
    if not isinstance(profile_interviewers, list):
        return

    existing = {
        str(profile.name).strip().lower()
        for _, profile in list_interviewers_with_paths()
    }

    for raw in profile_interviewers:
        if not isinstance(raw, dict):
            continue
        name = str(raw.get("name", "")).strip()
        if not name or name.lower() in existing:
            continue
        profile = InterviewerProfile(
            name=name,
            background=str(raw.get("background", "")),
            is_generic_ai=bool(raw.get("is_generic_ai", False)),
            role=str(raw.get("role", "")),
            expertise=_normalize_list_text(",".join(raw.get("expertise", []))) if isinstance(raw.get("expertise"), list) else [],
            potential_questions=_normalize_list_text(",".join(raw.get("potential_questions", []))) if isinstance(raw.get("potential_questions"), list) else [],
        )
        save_interviewer(profile)


class ParsedInterviewerProfile(BaseModel):
    name: str = Field(description="Interviewer name.")
    background: str = Field(description="Clear English background summary.")
    role: str = Field(description="Current primary role/title.")
    expertise: List[str] = Field(description="Key technical expertise areas.")
    potential_questions: List[str] = Field(description="Suggested interview probing questions.")


def _extract_json_payload(raw: str) -> str | None:
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return raw[start : end + 1]


def parse_interviewer_background(
    api_key: str,
    interviewer_name: str,
    background_text: str,
    model: str,
) -> ParsedInterviewerProfile:
    client = OpenAI(api_key=api_key)
    prompt = (
        "You are an information extractor for interview coach profiles.\n"
        "Normalize the pasted LinkedIn background into structured English profile fields.\n"
        "Output strict JSON-compatible fields with concise, professional language.\n"
        "Infer role and up to 5 core expertise bullets and up to 5 potential interview questions.\n"
        "Use only the provided background text; do not invent facts.\n\n"
        "Preserve the candidate's name exactly as given for the new profile."
    )
    models_to_try = [model] + [m for m in CHAT_FALLBACK_ORDER if m != model]
    last_error: Exception | None = None

    for model_name in models_to_try:
        try:
            resolved_temperature = _coerce_temperature_for_model(model_name, 0.2)
            response = client.beta.chat.completions.parse(
                model=model_name,
                messages=[
                    {"role": "system", "content": prompt},
                    {
                        "role": "user",
                        "content": f"Name: {interviewer_name}\nBackground:\n{background_text}",
                    },
                ],
                temperature=resolved_temperature,
                response_format=ParsedInterviewerProfile,
            )
            parsed = response.choices[0].message.parsed
            if parsed is None:
                raise ValueError("Failed to parse structured interviewer background.")
            parsed_dict = parsed.model_dump()
            parsed_dict["name"] = interviewer_name.strip()
            return ParsedInterviewerProfile.model_validate(parsed_dict)
        except Exception as exc:
            last_error = exc
            if is_model_access_error(exc):
                continue
            try:
                fallback_json_prompt = (
                    "Extract interviewer profile fields and return valid JSON only with keys: "
                    "name, background, role, expertise, potential_questions."
                )
                fallback_response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": fallback_json_prompt},
                        {
                            "role": "user",
                            "content": f"Name: {interviewer_name}\nBackground:\n{background_text}",
                        },
                    ],
                    temperature=resolved_temperature,
                )
                raw_content = fallback_response.choices[0].message.content or ""
                json_payload = _extract_json_payload(raw_content)
                if not json_payload:
                    raise ValueError("Could not extract JSON block from model response.")
                parsed = ParsedInterviewerProfile.model_validate_json(json_payload)
                parsed_dict = parsed.model_dump()
                parsed_dict["name"] = interviewer_name.strip()
                return ParsedInterviewerProfile.model_validate(parsed_dict)
            except Exception:
                # If JSON fallback fails for this model, try the next model when available.
                continue
    if last_error is not None:
        raise last_error
    raise ValueError("Failed to parse interviewer profile with available models.")

PROMPT_TECHNIQUES = {
    "Zero-Shot": "zero_shot",
    "Few-Shot": "few_shot",
    "Chain-of-Thought": "chain_of_thought",
    "Persona-Conditioning": "persona_conditioning",
    "Knowledge-Paucity (ISO 26262 Focus)": "knowledge_paucity",
}
if "selected_technique" not in st.session_state:
    st.session_state.selected_technique = "Zero-Shot"

st.sidebar.selectbox(
    "Prompting Technique",
    list(PROMPT_TECHNIQUES.keys()),
    key="selected_technique",
)

if "selected_model" not in st.session_state:
    st.session_state.selected_model = "gpt-4o-mini"
if st.session_state.selected_model not in CHAT_CAPABLE_MODELS:
    st.session_state.selected_model = "gpt-4o-mini"
if "temperature" not in st.session_state:
    st.session_state.temperature = 0.4
if "interview_started" not in st.session_state:
    st.session_state.interview_started = False
if "messages" not in st.session_state:
    st.session_state.messages = []

st.sidebar.selectbox(
    "OpenAI Model",
    CHAT_CAPABLE_MODELS,
    key="selected_model",
    on_change=_sync_temperature_to_model_constraint,
)
st.sidebar.slider(
    "Temperature",
    min_value=0.0,
    max_value=1.0,
    value=st.session_state.temperature,
    step=0.1,
    key="temperature",
    on_change=_sync_temperature_to_model_constraint,
)

DEFAULT_PROFILE_LABEL = "(Default) interview_data.json"
profile_files = sorted([path.name for path in PROFILES_DIR.glob("*.json")]) if PROFILES_DIR.exists() else []
profile_options = [DEFAULT_PROFILE_LABEL, *profile_files]
if "selected_jd_profile" not in st.session_state:
    st.session_state.selected_jd_profile = DEFAULT_PROFILE_LABEL
if st.session_state.selected_jd_profile not in profile_options:
    st.session_state.selected_jd_profile = DEFAULT_PROFILE_LABEL

st.sidebar.selectbox(
    "Select Target JD Profile",
    profile_options,
    key="selected_jd_profile",
)

selected_profile = st.session_state.selected_jd_profile
if selected_profile == DEFAULT_PROFILE_LABEL:
    st.session_state.current_interview_data = _attach_interviewee_profile(load_interview_data())
else:
    profile_path = PROFILES_DIR / selected_profile
    try:
        st.session_state.current_interview_data = _attach_interviewee_profile(
            load_interview_data(profile_path)
        )
    except Exception as exc:
        st.sidebar.error(f"Failed to load profile '{selected_profile}': {exc}")
        st.session_state.current_interview_data = _attach_interviewee_profile(load_interview_data())

if st.session_state.get("active_jd_profile") != selected_profile:
    st.session_state.active_jd_profile = selected_profile
    st.session_state.jd_profile_changed = True
    st.session_state.interview_started = False
else:
    st.session_state.jd_profile_changed = False

if st.session_state.jd_profile_changed and st.session_state.get("chat_histories"):
    st.session_state.chat_histories.pop(GENERIC_INTERVIEWER_KEY, None)

_bootstrap_default_interviewers(st.session_state.current_interview_data)

if selected_profile == DEFAULT_PROFILE_LABEL:
    job_positions = st.session_state.current_interview_data.get("job_positions", [])
    if isinstance(job_positions, list) and job_positions and isinstance(job_positions[0], dict):
        selected_jd_title = str(job_positions[0].get("title", DEFAULT_PROFILE_LABEL))
    else:
        selected_jd_title = DEFAULT_PROFILE_LABEL
else:
    selected_jd_title = str(st.session_state.current_interview_data.get("position", selected_profile))
st.session_state.selected_jd = selected_jd_title
jd_profile_path = _selected_profile_path(selected_profile)

interviewer_records = _collect_interviewer_records(st.session_state.selected_jd)
custom_names = _collect_interviewer_names(interviewer_records)

default_session_interviewers = _build_default_session_interviewers(st.session_state.current_interview_data)
session_interviewer_catalog = _normalize_label_list([*default_session_interviewers, *custom_names])
stored_session_interviewers = st.session_state.current_interview_data.get(JD_PROFILE_SESSION_KEY, default_session_interviewers)
if not isinstance(stored_session_interviewers, list):
    stored_session_interviewers = default_session_interviewers
active_interviewers = _filter_session_interviewers(
    stored_session_interviewers,
    session_interviewer_catalog,
    default_session_interviewers,
)
st.session_state.active_interviewers = active_interviewers

_render_profile_status_panel()

if "current_interviewer_label" not in st.session_state:
    st.session_state.current_interviewer_label = GENERIC_INTERVIEWER_LABEL

session_interviewers = _build_session_interviewer_profiles(st.session_state.current_interview_data, st.session_state.selected_jd)
speaker_options, label_to_key = _build_speaker_options_from_profiles(session_interviewers)

active_interviewers = st.session_state.get("active_interviewers", [GENERIC_INTERVIEWER_LABEL])
if not active_interviewers:
    active_interviewers = [GENERIC_INTERVIEWER_LABEL]

available_interviewers = [label for label in speaker_options if label in active_interviewers]
if not available_interviewers and GENERIC_INTERVIEWER_LABEL in speaker_options:
    available_interviewers = [GENERIC_INTERVIEWER_LABEL]
elif not available_interviewers:
    available_interviewers = speaker_options[:1]

if st.session_state.current_interviewer_label not in available_interviewers:
    st.session_state.current_interviewer_label = available_interviewers[0]

selected_label = st.sidebar.selectbox(
    "Select interviewer",
    available_interviewers,
    index=available_interviewers.index(st.session_state.current_interviewer_label)
    if st.session_state.current_interviewer_label in available_interviewers
    else 0,
)
st.session_state.current_interviewer_label = selected_label
selected_interviewer = label_to_key[selected_label]
if "active_interviewer_key" not in st.session_state:
    st.session_state.active_interviewer_key = selected_interviewer
elif st.session_state.active_interviewer_key != selected_interviewer:
    st.session_state.interview_started = False
    st.session_state.active_interviewer_key = selected_interviewer

with st.sidebar.expander("Add New Interviewer", expanded=False):
    st.markdown("**Manage Interviewers**")
    interviewer_options = _interviewer_path_options(interviewer_records)
    new_interviewer_key = "__new_interviewer__"
    selector_key = f"selected_interviewer_path_{st.session_state.selected_jd}"
    name_input_key = f"interviewer_name_input_{st.session_state.selected_jd}"
    background_input_key = f"interviewer_background_input_{st.session_state.selected_jd}"
    path_keys = sorted(interviewer_options.keys())

    if selector_key not in st.session_state:
        st.session_state[selector_key] = new_interviewer_key

    if st.session_state[selector_key] != new_interviewer_key and st.session_state[selector_key] not in interviewer_options:
        st.session_state[selector_key] = new_interviewer_key

    if name_input_key not in st.session_state:
        st.session_state[name_input_key] = ""
    if background_input_key not in st.session_state:
        st.session_state[background_input_key] = ""

    def _sync_interviewer_form() -> None:
        selected_interviewer_profile_path = st.session_state.get(selector_key)
        if selected_interviewer_profile_path == new_interviewer_key:
            st.session_state[name_input_key] = ""
            st.session_state[background_input_key] = ""
            return
        profile = interviewer_options.get(selected_interviewer_profile_path)
        if profile is None:
            st.session_state[name_input_key] = ""
            st.session_state[background_input_key] = ""
            st.session_state[selector_key] = new_interviewer_key
            return
        st.session_state[name_input_key] = profile.name
        st.session_state[background_input_key] = profile.background

    if not path_keys:
        st.session_state[selector_key] = new_interviewer_key
        _sync_interviewer_form()
    st.selectbox(
        "Select interviewer (or create new)",
        options=[new_interviewer_key, *path_keys],
        format_func=lambda path: "➕ Add New Interviewer" if path == new_interviewer_key else _interviewer_display_label(interviewer_options[path]),
        key=selector_key,
        on_change=_sync_interviewer_form,
    )

    selected_interviewer_profile_path = st.session_state.get(selector_key, new_interviewer_key)
    is_new_interviewer = selected_interviewer_profile_path == new_interviewer_key
    selected_profile_obj = None if is_new_interviewer else interviewer_options.get(selected_interviewer_profile_path)

    if not is_new_interviewer and selected_profile_obj is None:
        selected_interviewer_profile_path = new_interviewer_key
        st.session_state[selector_key] = new_interviewer_key
        is_new_interviewer = True

    interviewer_name = st.text_input(
        "Name",
        key=name_input_key,
        value=st.session_state.get(name_input_key, ""),
    )
    interviewer_background = st.text_area(
        "Background/Experience",
        key=background_input_key,
        value=st.session_state.get(background_input_key, ""),
        height=120,
    )

    action_col1, action_col2 = st.columns(2)
    with action_col1:
        save_label = "Save Interviewer" if is_new_interviewer else "Update Interviewer"
        if st.button(save_label, key="save_interviewer_button", use_container_width=True):
            if not interviewer_name.strip() or not interviewer_background.strip():
                st.warning("Name and Background/Experience are required.")
            else:
                api_key = get_api_key()
                if not api_key:
                    st.error("OPENAI_API_KEY is not set. Add it in Streamlit secrets or environment variables.")
                else:
                    try:
                        parsed = parse_interviewer_background(
                            api_key=api_key,
                            interviewer_name=interviewer_name.strip(),
                            background_text=interviewer_background.strip(),
                            model=st.session_state.selected_model,
                        )
                        profile = InterviewerProfile(
                            name=parsed.name,
                            background=parsed.background,
                            role=parsed.role,
                            expertise=parsed.expertise,
                            potential_questions=parsed.potential_questions,
                            is_generic_ai=False,
                        )
                        saved_path = save_interviewer(
                            profile,
                            existing_path=None if is_new_interviewer else selected_interviewer_profile_path,
                        )
                        st.success(
                            (
                                f"Saved interviewer: {profile.name} ({saved_path.name})"
                                if is_new_interviewer
                                else f"Updated interviewer: {profile.name}"
                            )
                        )
                        if is_new_interviewer:
                            st.session_state[selector_key] = saved_path.as_posix()
                            _sync_interviewer_form()
                    except Exception as exc:
                        st.error(f"Failed to parse and save interviewer: {exc}")
                st.rerun()

    with action_col2:
        if not is_new_interviewer:
            if st.button("Delete Interviewer", key="delete_interviewer_button", use_container_width=True):
                if selected_profile_obj and selected_profile_obj.name in st.session_state.get("active_interviewers", []):
                    active_interviewers = _filter_session_interviewers(
                        [name for name in st.session_state.active_interviewers if name != selected_profile_obj.name],
                        session_interviewer_catalog,
                        default_session_interviewers,
                        enforce_mandatory=True,
                    )
                    st.session_state.current_interview_data[JD_PROFILE_SESSION_KEY] = active_interviewers
                    st.session_state.active_interviewers = active_interviewers
                    _persist_session_interviewers(jd_profile_path, st.session_state.current_interview_data, active_interviewers)
                if selected_profile_obj:
                    delete_interviewer(selected_interviewer_profile_path)
                    st.success(f"Deleted interviewer: {selected_profile_obj.name}")
                    if st.session_state.current_interviewer_label == selected_profile_obj.name:
                        st.session_state.current_interviewer_label = GENERIC_INTERVIEWER_LABEL
                st.session_state[selector_key] = new_interviewer_key
                st.session_state[name_input_key] = ""
                st.session_state[background_input_key] = ""
                st.rerun()

    if not interviewer_options:
        st.caption("No interviewer JSON files yet.")

    st.divider()
    st.markdown("**Select Interviewers for this Session**")
    session_options = session_interviewer_catalog

    session_widget_key = f"active_interviewers_widget_{st.session_state.selected_jd}"
    selected_for_session = st.multiselect(
        "Select interviewers for this session",
        options=session_options,
        default=active_interviewers,
        key=session_widget_key,
    )
    if st.button("Save Session Interviewers", key="save_session_interviewers_button", use_container_width=True):
        filtered_session_interviewers = _filter_session_interviewers(
            selected_for_session,
            session_interviewer_catalog,
            default_session_interviewers,
            enforce_mandatory=False,
        )
        st.session_state.current_interview_data[JD_PROFILE_SESSION_KEY] = filtered_session_interviewers
        st.session_state.active_interviewers = filtered_session_interviewers
        _persist_session_interviewers(
            jd_profile_path,
            st.session_state.current_interview_data,
            filtered_session_interviewers,
        )
        st.success(f"Saved session interviewers: {', '.join(filtered_session_interviewers)}")

if "current_interviewer" not in st.session_state:
    st.session_state.current_interviewer = selected_interviewer

if "chat_histories" not in st.session_state:
    st.session_state.chat_histories = {}
    st.session_state.chat_histories[GENERIC_INTERVIEWER_KEY] = []
if selected_interviewer not in st.session_state.chat_histories:
    st.session_state.chat_histories[selected_interviewer] = []
st.session_state.current_interviewer = selected_interviewer

current_interviewer = st.session_state.current_interviewer

data = st.session_state.current_interview_data
technique_key = PROMPT_TECHNIQUES[st.session_state.selected_technique]
system_prompts = build_system_prompts(
    data=data,
    jd_profile=st.session_state.current_interview_data,
    technique=technique_key,
    interviewers=session_interviewers,
)
history: List[Dict[str, str]] = st.session_state.chat_histories[current_interviewer]
st.session_state.messages = history

st.title(APP_TITLE)
if st.session_state.jd_profile_changed:
    st.success(f"Rebuilt personas for JD profile: {selected_profile}")
st.caption(f"Active interviewer: {selected_label}")
if st.session_state.jd_profile_changed:
    st.caption(f"Loaded JD profile: {st.session_state.selected_jd_profile}")

CRITIQUE_PERSONA = """You are a Capgemini ADAS Interview Critique Persona.
You evaluate a candidate's latest answer for a Capgemini technical interview.
Focus strictly on:
1) ASPICE CL3 process maturity evidence
2) Technical accuracy for ADAS / embedded C++ context

Provide concise, actionable feedback using this exact structure:
- Overall verdict (2-3 lines)
- ASPICE CL3 score: <1-5> with one-sentence rationale
- Technical accuracy score: <1-5> with one-sentence rationale
- Strengths: 2-3 bullets
- Gaps / risks: 2-3 bullets
- Improvement plan for next answer: 3 bullets with concrete phrasing examples
"""

ADAS_KEY_TERMS = [
    "ASPICE CL3",
    "ASPICE",
    "ISO 26262",
    "ASIL",
    "AUTOSAR",
    "MISRA",
    "Functional Safety",
    "Safety Case",
    "Radar",
    "Camera",
    "Lidar",
    "Sensor Fusion",
    "UDS",
    "DMA",
    "Unit Testing",
    "Static Analysis",
    "Determinism",
]
JD_KEYWORD_CATALOG_KEY = "jd_keyword_catalog"
JD_KEYWORD_CATEGORY_RULES: List[tuple[str, List[str]]] = [
    ("Safety & Compliance", ["aspice", "iso 26262", "sotif", "asil", "safety", "fmea"]),
    ("Testing & Quality", ["test", "testing", "validation", "verification", "traceability", "quality"]),
    ("ADAS Core", ["adas", "radar", "camera", "lidar", "fusion", "perception", "localization", "tracking", "mapping", "lidar"]),
    ("Runtime & Implementation", ["c++", "cpp", "misra", "autosar", "dma", "determinism", "real-time", "concurrency", "memory", "thread"]),
    ("Tools & Framework", ["python", "docker", "can", "lin", "ethernet", "git", "jenkins", "jira", "matlab", "simulink", "ros", "linux"]),
]

_REDUNDANT_TERM_PREFIXES = (
    "experience in",
    "experience with",
    "knowledge of",
    "knowledge on",
    "proven experience in",
    "ability to",
    "proven ability to",
    "strong focus on",
    "good understanding of",
    "solid understanding of",
    "expertise in",
    "required to have",
)


def _normalize_term_item(raw: Any) -> str:
    term = str(raw).strip()
    if not term:
        return ""

    # Keep slash-containing technology names (for example C/C++) and split only on real separators.
    parts = [item.strip() for item in re.split(r"[;\n|,]+", term) if item.strip()]
    if not parts:
        parts = [term]

    normalized: List[str] = []
    for part in parts:
        part = part.replace("C/C++", "C++").replace("c/c++", "C++")
        part = re.sub(r"\s{2,}", " ", part).strip(" -")
        for prefix in _REDUNDANT_TERM_PREFIXES:
            pattern = re.compile(rf"^{re.escape(prefix)}\s+", re.IGNORECASE)
            cleaned = pattern.sub("", part).strip()
            if cleaned != part:
                part = cleaned
                break
        part = part[0:1].upper() + part[1:] if part else part
        if not part:
            continue
        normalized.append(part)

    return normalized[0] if normalized else ""


def _normalize_terms(values: Any) -> List[str]:
    if not isinstance(values, list):
        return []
    normalized: List[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        if not text:
            continue
        parts = [part.strip() for part in re.split(r"[;\n|,]+", text) if part.strip()]
        if not parts:
            continue
        for part in parts:
            term = _normalize_term_item(part)
            if not term:
                continue
            term = term.strip()
            if len(term) < 2:
                continue
            key = term.lower()
            if key in seen:
                continue
            seen.add(key)
            normalized.append(term)
    return normalized


def _classify_jd_keyword(term: str) -> str:
    candidate = term.lower()
    for category, hints in JD_KEYWORD_CATEGORY_RULES:
        if any(hint in candidate for hint in hints):
            return category
    return "Role Requirements"


def _normalize_jd_keyword_catalog(values: Dict[str, List[str]]) -> Dict[str, List[str]]:
    output: Dict[str, List[str]] = {}
    for category, items in values.items():
        cleaned = _normalize_terms(items)
        if cleaned:
            output[category] = cleaned
    return output


def _build_jd_keyword_catalog(extracted: JDExtraction) -> Dict[str, List[str]]:
    requirements = _normalize_terms(extracted.key_requirements)
    tech_stack = _normalize_terms(extracted.tech_stack)

    catalog: Dict[str, List[str]] = {
        "Role Requirements": [],
        "Tech Stack": [],
        "Safety & Compliance": [],
        "Testing & Quality": [],
        "ADAS Core": [],
        "Runtime & Implementation": [],
        "Tools & Framework": [],
    }

    for term in requirements:
        inferred_category = _classify_jd_keyword(term)
        if inferred_category in {"Safety & Compliance", "Testing & Quality", "ADAS Core", "Runtime & Implementation", "Tools & Framework"}:
            catalog[inferred_category].append(term)
            continue
        catalog["Role Requirements"].append(term)

    for term in tech_stack:
        catalog["Tech Stack"].append(term)

    return _normalize_jd_keyword_catalog(catalog)

class JDExtraction(BaseModel):
    company: str = Field(description="Company name in English.")
    position: str = Field(description="Job position title in English.")
    job_description: str = Field(description="Full job description rewritten in English.")
    key_requirements: List[str] = Field(description="Top key requirements from the job description.")
    tech_stack: List[str] = Field(description="Technologies and tools required for the role.")


def is_model_access_error(exc: Exception) -> bool:
    error_text = str(exc).lower()
    return "model_not_found" in error_text or "does not have access to model" in error_text


def slugify_for_filename(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9]+", "-", value.strip())
    sanitized = sanitized.strip("-")
    return sanitized or "unknown"


def build_profile_filename(company: str, position: str) -> str:
    company_part = slugify_for_filename(company)
    position_part = slugify_for_filename(position)
    return f"{company_part}-{position_part}-{date.today().isoformat()}.json"


def extract_jd_fields(api_key: str, jd_text: str, company: str, position: str) -> tuple[JDExtraction, str]:
    client = OpenAI(api_key=api_key)
    extraction_prompt = (
        "Extract hiring signals from this job description.\n"
        "If the content is in Korean or German, translate everything to natural English first.\n"
        "Output every field in English.\n"
        "Use concise phrases, deduplicate items, and only include content supported by the JD.\n"
        f"Company: {company}\n"
        f"Position: {position}\n\n"
        "Job Description:\n"
        f"{jd_text}"
    )
    models_to_try = CHAT_FALLBACK_ORDER
    last_error: Exception | None = None
    for model_name in models_to_try:
        try:
            response = client.beta.chat.completions.parse(
                model=model_name,
                messages=[
                    {"role": "system", "content": "Return structured hiring insights."},
                    {"role": "user", "content": extraction_prompt},
                ],
                temperature=_coerce_temperature_for_model(model_name, 0.1),
                response_format=JDExtraction,
            )
            parsed = response.choices[0].message.parsed
            if parsed is None:
                raise ValueError("Failed to parse structured output for JD extraction.")
            return parsed, model_name
        except Exception as exc:
            last_error = exc
            if is_model_access_error(exc):
                continue
            raise
    if last_error is not None:
        raise last_error
    raise ValueError("Failed to extract JD fields with available models.")


def create_chat_completion_with_fallback(
    client: OpenAI,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
):
    try:
        resolved_temperature = _coerce_temperature_for_model(model, temperature)
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=resolved_temperature,
        )
        return response, model
    except Exception as exc:
        if model != CHAT_FALLBACK_ORDER[0] and is_model_access_error(exc):
            for fallback_model in CHAT_FALLBACK_ORDER:
                if fallback_model == model:
                    continue
                fallback_temperature = _coerce_temperature_for_model(
                    fallback_model,
                    temperature,
                )
                try:
                    fallback_response = client.chat.completions.create(
                        model=fallback_model,
                        messages=messages,
                        temperature=fallback_temperature,
                    )
                    return fallback_response, fallback_model
                except Exception as fallback_exc:
                    if is_model_access_error(fallback_exc):
                        continue
                    raise
        raise


def build_interview_profile(extracted: JDExtraction) -> Dict[str, object]:
    keyword_catalog = _build_jd_keyword_catalog(extracted)
    return {
        "generated_at": date.today().isoformat(),
        "company": extracted.company,
        "position": extracted.position,
        "job_description": extracted.job_description,
        "key_requirements": extracted.key_requirements,
        "tech_stack": extracted.tech_stack,
        JD_KEYWORD_CATALOG_KEY: keyword_catalog,
        "session_interviewers": [GENERIC_INTERVIEWER_LABEL],
        "interviewers": [],
    }


def get_last_user_response(messages: List[Dict[str, str]]) -> str | None:
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return msg.get("content")
    return None


def get_last_assistant_message(messages: List[Dict[str, str]]) -> str | None:
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            return msg.get("content")
    return None


def _build_opening_prompt(interviewer_name: str, jd_title: str) -> str:
    return (
        f"You are {interviewer_name}. "
        f"Start the interview by briefly introducing yourself and asking the first question based on the JD: {jd_title}."
    )


def _build_support_prompt(role: str, target_question: str, response_style: str) -> str:
    if response_style == "hint":
        return (
            "You are a helpful interview coach in ADAS/automotive engineering interviews.\n"
            "Provide 3 concise hints only, no full answer. "
            "Keep hints practical, interview-ready, and specific to the question below.\n\n"
            f"Interviewer role: {role}\n\nQuestion:\n{target_question}"
        )
    return (
        "You are a helpful interview coach in ADAS/automotive engineering interviews.\n"
        "Provide one strong, concise model answer in English that a good candidate could deliver.\n"
        "Keep it practical, specific, and interview-ready. "
        "Include one implementation detail and one risk/quality consideration.\n\n"
        f"Interviewer role: {role}\n\nQuestion:\n{target_question}"
    )


def _start_interview_for_active_interviewer(
    interviewer_key: str,
    interviewer_label: str,
    jd_title: str,
    system_prompts: Dict[str, str],
) -> None:
    st.session_state.interview_started = True
    st.session_state.messages = []
    st.session_state.interview_started_interviewer = interviewer_label

    api_key = get_api_key()
    if not api_key:
        st.error("OPENAI_API_KEY is not set. Add it in Streamlit secrets or environment variables.")
        st.session_state.interview_started = False
        return

    if "chat_histories" not in st.session_state:
        st.session_state.chat_histories = {}

    if interviewer_key not in st.session_state.chat_histories:
        st.session_state.chat_histories[interviewer_key] = []
    history = st.session_state.chat_histories[interviewer_key]
    history.clear()
    st.session_state.chat_histories[interviewer_key] = history
    st.session_state.messages = history

    opening_prompt = _build_opening_prompt(interviewer_label, jd_title)
    system_prompt = system_prompts.get(interviewer_key, f"You are {interviewer_label}.")
    client = OpenAI(api_key=api_key)
    typing_placeholder = st.empty()
    typing_placeholder.info(f"{interviewer_label} is typing...")
    try:
        response, used_model = create_chat_completion_with_fallback(
            client=client,
            model=st.session_state.selected_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": opening_prompt},
            ],
            temperature=st.session_state.temperature,
        )
        assistant_reply = response.choices[0].message.content or "I could not generate an opening question."
        if used_model != st.session_state.selected_model:
            assistant_reply = f"[Model fallback: {used_model}]\n\n{assistant_reply}"
        history.append({"role": "assistant", "content": assistant_reply})
        st.session_state.chat_histories[interviewer_key] = history
        st.session_state.messages = history
        st.rerun()
    except Exception as exc:
        st.error(f"Error from OpenAI API: {exc}")
        st.session_state.interview_started = False
    finally:
        typing_placeholder.empty()


def _get_feedback_highlight_catalog(profile_data: Dict[str, Any]) -> Dict[str, List[str]]:
    if not isinstance(profile_data, dict):
        return {}
    catalog = profile_data.get(JD_KEYWORD_CATALOG_KEY, {})
    if not isinstance(catalog, dict):
        return {}

    output: Dict[str, List[str]] = {}
    for category, raw_terms in catalog.items():
        normalized = _normalize_terms(raw_terms)
        if normalized:
            output[str(category)] = normalized

    if output:
        return output

    return {"ADAS Core": ADAS_KEY_TERMS}


def render_feedback_with_adas_terms(feedback: str, profile_data: Dict[str, Any]) -> None:
    if not feedback:
        return

    if annotated_text is None:
        st.markdown(feedback)
        st.info("Install `streamlit-annotated-text` to enable in-line ADAS term highlighting.")
        return

    catalog = _get_feedback_highlight_catalog(profile_data)
    term_to_label: Dict[str, str] = {}
    ordered_terms: List[tuple[str, str]] = []

    for label, terms in catalog.items():
        for term in terms:
            normalized_term = term.strip()
            if not normalized_term:
                continue
            ordered_terms.append((normalized_term, label))
            term_to_label[normalized_term.lower()] = label

    if not ordered_terms:
        st.markdown(feedback)
        return

    sorted_terms = sorted(ordered_terms, key=lambda item: len(item[0]), reverse=True)
    pattern = re.compile("|".join(re.escape(term) for term, _ in sorted_terms), flags=re.IGNORECASE)
    parts: List[str | tuple] = []
    cursor = 0
    for match in pattern.finditer(feedback):
        if match.start() > cursor:
            parts.append(feedback[cursor:match.start()])
        label = term_to_label.get(match.group(0).lower(), "JD Term")
        parts.append((match.group(0), label))
        cursor = match.end()
    if cursor < len(feedback):
        parts.append(feedback[cursor:])

    if parts:
        annotated_text(*parts)
    else:
        st.markdown(feedback)


PANEL_CHAT = "Interview Chat"
PANEL_CODING = "Coding Challenge"


def render_sidebar_profile_creator() -> None:
    st.sidebar.divider()
    with st.sidebar.expander("Create Interview Profile from JD", expanded=False):
        jd_company = st.text_input("Company Name", key="jd_company_name")
        jd_position = st.text_input("Job Position Title", key="jd_position_title")
        jd_text = st.text_area("Paste Job Description", key="jd_text", height=180)

        if not st.button("Generate & Save Interview Profile", use_container_width=True):
            return

        company = jd_company.strip()
        position = jd_position.strip()
        description = jd_text.strip()
        if not company or not position or not description:
            st.error("Company, position, and JD are all required.")
            return

        api_key = get_api_key()
        if not api_key:
            st.error("OPENAI_API_KEY is not set. Add it in Streamlit secrets or environment variables.")
            return

        try:
            extracted, used_model = extract_jd_fields(
                api_key=api_key,
                jd_text=description,
                company=company,
                position=position,
            )
            profile = build_interview_profile(extracted=extracted)
            filename = build_profile_filename(extracted.company, extracted.position)
            PROFILES_DIR.mkdir(parents=True, exist_ok=True)
            output_path = PROFILES_DIR / filename
            output_path.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")
            st.success(f"Saved profile: {output_path.name} (model: {used_model})")
        except Exception as exc:
            st.error(f"Failed to generate profile: {exc}")


def render_sidebar_interviewee_profile_loader() -> None:
    st.sidebar.divider()
    with st.sidebar.expander("Load Interviewee Profile (CV / Cover Letter)", expanded=False):
        uploaded_pdf = st.file_uploader(
            "Upload Resume/CV (PDF)",
            type=["pdf"],
            key="interviewee_resume_pdf",
        )
        if uploaded_pdf is not None:
            pdf_signature = f"{uploaded_pdf.name}|{hashlib.md5(uploaded_pdf.getvalue()).hexdigest()}"
            previous_signature = st.session_state.get("interviewee_resume_signature", "")
            if previous_signature != pdf_signature:
                try:
                    api_key = get_api_key()
                    if not api_key:
                        st.error(
                            "OPENAI_API_KEY is not set. Add it in Streamlit secrets or environment variables."
                        )
                    else:
                        extracted_text = _extract_text_from_pdf(uploaded_pdf)
                        if not extracted_text.strip():
                            st.warning("No extractable text found in uploaded PDF.")
                        else:
                            parsed = parse_interviewee_profile(
                                api_key=api_key,
                                source_text=extracted_text,
                                model=st.session_state.selected_model,
                                source_label="resume",
                            )
                            _save_interviewee_profile(_as_interviewee_payload(parsed))
                            st.session_state.interviewee_resume_signature = pdf_signature
                            st.success("Interviewee profile reset from uploaded CV.")
                            st.session_state.current_interview_data = _attach_interviewee_profile(
                                st.session_state.current_interview_data
                            )
                            st.rerun()
                except Exception as exc:
                    st.error(f"Failed to reset interviewee profile from PDF: {exc}")

        cover_letter = st.text_area(
            "Paste Cover Letter",
            key="interviewee_cover_letter",
            height=140,
        )

        if st.button("Apply Cover Letter", key="apply_cover_letter_button", use_container_width=True):
            cover_text = cover_letter.strip()
            if not cover_text:
                st.warning("Paste a cover letter first.")
                return

            api_key = get_api_key()
            if not api_key:
                st.error(
                    "OPENAI_API_KEY is not set. Add it in Streamlit secrets or environment variables."
                )
                return

            current_profile = _load_interviewee_profile()
            if not current_profile:
                st.warning("No interviewee profile exists yet. Upload a CV first.")
                return

            try:
                parsed = parse_interviewee_profile(
                    api_key=api_key,
                    source_text=cover_text,
                    model=st.session_state.selected_model,
                    source_label="cover_letter",
                )
                merged = _merge_interviewee_cover_letter(current_profile, parsed)
                _save_interviewee_profile(merged)
                st.success("Cover letter information merged into interviewee profile.")
                st.session_state.current_interview_data = _attach_interviewee_profile(
                    st.session_state.current_interview_data
                )
                st.rerun()
            except Exception as exc:
                st.error(f"Failed to merge cover letter: {exc}")


def render_floating_coding_tab_button() -> None:
    st.markdown(
        f"""
<style>
[data-testid="stAppViewContainer"] .main .block-container {{
  padding-right: 96px;
}}
.st-key-open_coding_tab {{
  position: fixed;
  right: 12px;
  top: 50%;
  transform: translateY(-50%);
  z-index: 9999;
  margin: 0 !important;
  padding: 0 !important;
}}
.st-key-open_coding_tab button {{
  width: 42px;
  height: 190px;
  border: 1px solid #444;
  border-radius: 12px;
  background: #0e1117;
  color: #fff !important;
  writing-mode: vertical-rl;
  text-orientation: mixed;
  letter-spacing: 0.3px;
  font-size: 12px;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 0 !important;
}}
.st-key-open_coding_tab button:hover {{
  border-color: #999;
}}
@media (max-width: 640px) {{
  .st-key-open_coding_tab button {{
    right: 8px;
    width: 36px;
    height: 160px;
    font-size: 11px;
  }}
}}
</style>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Coding Challenge", key="open_coding_tab"):
        st.session_state.tabs_instance_nonce = st.session_state.get("tabs_instance_nonce", 0) + 1
        st.session_state.pending_tabs_default = PANEL_CODING
        st.rerun()


def get_tab_labels() -> tuple[str, str]:
    nonce = st.session_state.get("tabs_instance_nonce", 0)
    suffix = "\u200b" * nonce
    return PANEL_CHAT + suffix, PANEL_CODING + suffix


def get_tabs_default_once(tab_chat_label: str, tab_code_label: str) -> str | None:
    pending = st.session_state.pop("pending_tabs_default", None)
    if pending == PANEL_CODING:
        return tab_code_label
    if pending == PANEL_CHAT:
        return tab_chat_label
    return None


def render_chat_panel(
    current_interviewer: str,
    history: List[Dict[str, str]],
    system_prompts: Dict[str, str],
) -> None:
    st.subheader(PANEL_CHAT)
    if not st.session_state.get("interview_started", False):
        st.info("Press '🚀 Start Interview' in the sidebar to let the interviewer begin.")
        return

    for idx, msg in enumerate(history):
        message(msg["content"], is_user=msg["role"] == "user", key=f"{current_interviewer}-{idx}")

    user_input = st.chat_input("Type your interview answer or ask a question...")
    if not user_input:
        pass
    else:
        is_valid, validation_error = validate_input(user_input)
        if not is_valid:
            st.error(validation_error or "Invalid input.")
        else:
            history.append({"role": "user", "content": user_input})
            message(user_input, is_user=True, key=f"{current_interviewer}-user-live")

            api_key = get_api_key()
            if not api_key:
                st.error("OPENAI_API_KEY is not set. Add it in Streamlit secrets or environment variables.")
            else:
                client = OpenAI(api_key=api_key)
                typing_placeholder = st.empty()
                typing_placeholder.info(f"{current_interviewer.title()} is typing...")
                try:
                    response, used_model = create_chat_completion_with_fallback(
                        client=client,
                        model=st.session_state.selected_model,
                        messages=[{"role": "system", "content": system_prompts[current_interviewer]}, *history],
                        temperature=st.session_state.temperature,
                    )
                    assistant_reply = response.choices[0].message.content or "I could not generate a response."
                    if used_model != st.session_state.selected_model:
                        assistant_reply = f"[Model fallback: {used_model}]\n\n{assistant_reply}"
                except Exception as exc:
                    assistant_reply = f"Error from OpenAI API: {exc}"
                finally:
                    typing_placeholder.empty()

                history.append({"role": "assistant", "content": assistant_reply})
                st.session_state.chat_histories[current_interviewer] = history
                st.rerun()

    analyze_key = f"critique_feedback_{current_interviewer}"
    hint_key = f"interview_hint_{current_interviewer}"
    answer_key = f"model_answer_{current_interviewer}"

    if st.button("Analyze My Answer", use_container_width=True):
        last_user_answer = get_last_user_response(history)
        if not last_user_answer:
            st.warning("No user answer found yet. Submit an answer first.")
        else:
            is_valid, validation_error = validate_input(last_user_answer)
            if not is_valid:
                st.error(validation_error or "Invalid input.")
            else:
                api_key = get_api_key()
                if not api_key:
                    st.error("OPENAI_API_KEY is not set. Add it in Streamlit secrets or environment variables.")
                else:
                    client = OpenAI(api_key=api_key)
                    critique_user_prompt = f"""Interviewer: {current_interviewer.title()}
Candidate answer:
{last_user_answer}

Evaluate this answer for ASPICE CL3 evidence and technical accuracy."""
                    try:
                        critique_temperature = max(0.1, st.session_state.temperature - 0.2)
                        critique_response, used_model = create_chat_completion_with_fallback(
                            client=client,
                            model=st.session_state.selected_model,
                            messages=[
                                {"role": "system", "content": CRITIQUE_PERSONA},
                                {"role": "user", "content": critique_user_prompt},
                            ],
                            temperature=critique_temperature,
                        )
                        critique_text = critique_response.choices[0].message.content or "No critique generated."
                        if used_model != st.session_state.selected_model:
                            critique_text = f"[Model fallback: {used_model}]\n\n{critique_text}"
                        st.session_state[analyze_key] = critique_text
                    except Exception as exc:
                        st.session_state[analyze_key] = f"Error from OpenAI API: {exc}"

    last_question = get_last_assistant_message(history)
    st.divider()
    col_hint, col_model = st.columns(2)
    with col_hint:
        if st.button("Need a Hint", use_container_width=True, key=f"hint_btn_{current_interviewer}"):
            if not last_question:
                st.session_state[hint_key] = "Please ask a question first, then request a hint."
            else:
                api_key = get_api_key()
                if not api_key:
                    st.session_state[hint_key] = "OPENAI_API_KEY is not set. Add it in Streamlit secrets or environment variables."
                else:
                    client = OpenAI(api_key=api_key)
                    try:
                        hint_response, used_model = create_chat_completion_with_fallback(
                            client=client,
                            model=st.session_state.selected_model,
                            messages=[{"role": "user", "content": _build_support_prompt(current_interviewer.title(), last_question, "hint")}],
                            temperature=max(0.1, st.session_state.temperature - 0.1),
                        )
                        hint_text = hint_response.choices[0].message.content or "No hint generated."
                        if used_model != st.session_state.selected_model:
                            hint_text = f"[Model fallback: {used_model}]\n\n{hint_text}"
                        st.session_state[hint_key] = hint_text
                    except Exception as exc:
                        st.session_state[hint_key] = f"Failed to generate hint: {exc}"
    with col_model:
        if st.button("Show Model Answer", use_container_width=True, key=f"model_answer_btn_{current_interviewer}"):
            if not last_question:
                st.session_state[answer_key] = "Please ask a question first, then request a model answer."
            else:
                api_key = get_api_key()
                if not api_key:
                    st.session_state[answer_key] = "OPENAI_API_KEY is not set. Add it in Streamlit secrets or environment variables."
                else:
                    client = OpenAI(api_key=api_key)
                    try:
                        answer_response, used_model = create_chat_completion_with_fallback(
                            client=client,
                            model=st.session_state.selected_model,
                            messages=[{"role": "user", "content": _build_support_prompt(current_interviewer.title(), last_question, "sample")}],
                            temperature=st.session_state.temperature,
                        )
                        answer_text = answer_response.choices[0].message.content or "No model answer generated."
                        if used_model != st.session_state.selected_model:
                            answer_text = f"[Model fallback: {used_model}]\n\n{answer_text}"
                        st.session_state[answer_key] = answer_text
                    except Exception as exc:
                        st.session_state[answer_key] = f"Failed to generate model answer: {exc}"

    feedback = st.session_state.get(analyze_key)
    if feedback:
        st.subheader("Critique Feedback")
        render_feedback_with_adas_terms(feedback, st.session_state.current_interview_data)

    hint = st.session_state.get(hint_key)
    if hint:
        st.subheader("Hint")
        st.write(hint)

    model_answer = st.session_state.get(answer_key)
    if model_answer:
        st.subheader("Model Answer")
        st.write(model_answer)


def render_coding_panel(current_interviewer: str) -> None:
    st.subheader(f"{PANEL_CODING} (C++)")
    st.caption("Use this area for implementation-focused questions. Syntax highlighting is set to C++.")
    if "editor_input_mode" not in st.session_state:
        st.session_state.editor_input_mode = "Normal"
    st.selectbox("Editor Input Mode", ["Normal", "Vim"], key="editor_input_mode")
    keybinding_mode = "vim" if st.session_state.editor_input_mode == "Vim" else "vscode"

    ace_key = f"coding_challenge_cpp_{current_interviewer}"
    default_cpp = """#include <iostream>
#include <vector>

int main() {
    std::vector<int> data{1, 2, 3, 4, 5};
    int sum = 0;
    for (int x : data) {
        sum += x;
    }
    std::cout << "Sum: " << sum << std::endl;
    return 0;
}
"""
    if ace_key not in st.session_state:
        st.session_state[ace_key] = default_cpp

    st_ace(
        value=st.session_state[ace_key],
        language="c_cpp",
        theme="tomorrow_night",
        key=ace_key,
        height=420,
        auto_update=True,
        font_size=14,
        wrap=True,
        keybinding=keybinding_mode,
    )


render_sidebar_profile_creator()
render_sidebar_interviewee_profile_loader()
render_floating_coding_tab_button()

if st.sidebar.button("🚀 Start Interview", use_container_width=True, type="primary"):
    _start_interview_for_active_interviewer(
        interviewer_key=current_interviewer,
        interviewer_label=selected_label,
        jd_title=st.session_state.selected_jd,
        system_prompts=system_prompts,
    )

chat_tab_label, code_tab_label = get_tab_labels()
default_tab = get_tabs_default_once(chat_tab_label, code_tab_label)
tab_chat, tab_code = st.tabs([chat_tab_label, code_tab_label], default=default_tab)

with tab_chat:
    render_chat_panel(
        current_interviewer=current_interviewer,
        history=history,
        system_prompts=system_prompts,
    )

with tab_code:
    render_coding_panel(current_interviewer=current_interviewer)
