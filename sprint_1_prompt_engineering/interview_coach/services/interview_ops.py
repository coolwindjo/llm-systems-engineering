from __future__ import annotations

import hashlib
import io
import json
import os
import re
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Tuple

from openai import OpenAI
from pydantic import BaseModel, Field

from utils.interviewer_store import (
    InterviewerProfile,
    get_interviewers_by_jd,
    list_interviewers_with_paths,
    save_interviewer,
)

try:
    from pypdf import PdfReader
except ImportError:  # pragma: no cover
    PdfReader = None

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
PROFILES_DIR = DATA_DIR / "profiles"
INTERVIEWEE_PROFILE_PATH = DATA_DIR / "interviewee_profile.json"
TEMPERATURE_RULES_PATH = ROOT_DIR / "utils" / "model_temperature_constraints.json"
JD_PROFILE_SESSION_KEY = "session_interviewers"
JD_KEYWORD_CATALOG_KEY = "jd_keyword_catalog"
DEFAULT_PROFILE_LABEL = "(Default) interview_data.json"

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

GENERIC_INTERVIEWER_KEY = "generic_ai_interviewer"
GENERIC_INTERVIEWER_LABEL = "Generic AI Interviewer"
MANDATORY_SESSION_INTERVIEWERS = [GENERIC_INTERVIEWER_LABEL]

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

JD_KEYWORD_CATEGORY_RULES: List[tuple[str, List[str]]] = [
    ("Safety & Compliance", ["aspice", "iso 26262", "sotif", "asil", "safety", "fmea"]),
    ("Testing & Quality", ["test", "testing", "validation", "verification", "traceability", "quality"]),
    (
        "ADAS Core",
        [
            "adas",
            "radar",
            "camera",
            "lidar",
            "fusion",
            "perception",
            "localization",
            "tracking",
            "mapping",
        ],
    ),
    (
        "Runtime & Implementation",
        [
            "c++",
            "cpp",
            "misra",
            "autosar",
            "dma",
            "determinism",
            "real-time",
            "concurrency",
            "memory",
            "thread",
        ],
    ),
    (
        "Tools & Framework",
        [
            "python",
            "docker",
            "can",
            "lin",
            "ethernet",
            "git",
            "jenkins",
            "jira",
            "matlab",
            "simulink",
            "ros",
            "linux",
        ],
    ),
]

_REDUNDANT_PREFIXES = (
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


class ParsedIntervieweeProfile(BaseModel):
    name: str = Field(default="")
    status: str = Field(default="")
    core_strengths: List[str] = Field(default_factory=list)
    focus_phrases: List[str] = Field(default_factory=list)


class ParsedInterviewerProfile(BaseModel):
    name: str = Field(description="Interviewer name.")
    background: str = Field(description="Clear English background summary.")
    role: str = Field(description="Current primary role/title.")
    expertise: List[str] = Field(description="Key technical expertise areas.")
    potential_questions: List[str] = Field(description="Suggested interview probing questions.")


class JDExtraction(BaseModel):
    company: str = Field(description="Company name in English.")
    position: str = Field(description="Job position title in English.")
    job_description: str = Field(description="Full job description rewritten in English.")
    key_requirements: List[str] = Field(description="Top key requirements from the job description.")
    tech_stack: List[str] = Field(description="Technologies and tools required for the role.")


def load_local_env(env_path: Path | None = None) -> None:
    target = env_path if env_path is not None else (ROOT_DIR / ".env")
    if not target.exists():
        return

    for raw_line in target.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


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
    default_min = _coerce_to_float(default_policy.get("min"), TEMPERATURE_RULES_FALLBACK["default"]["min"])
    default_max = _coerce_to_float(default_policy.get("max"), TEMPERATURE_RULES_FALLBACK["default"]["max"])
    default_fallback = _coerce_to_float(
        default_policy.get("fallback"),
        TEMPERATURE_RULES_FALLBACK["default"]["fallback"],
    )

    return {
        "default": {
            "min": default_min,
            "max": default_max,
            "fallback": default_fallback,
        },
        "rules": [rule for rule in raw.get("rules", []) if isinstance(rule, dict)],
    }


TEMPERATURE_RULES = _load_temperature_rules()


def _coerce_temperature_for_model(model: str, temperature: float) -> float:
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
                or (match_mode in {"contains", ""} and pattern.strip().lower() in model_name)
                or (match_mode == "startswith" and model_name.startswith(pattern.strip().lower()))
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
        payload = profile.model_dump()
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


def _selected_profile_path(selected_profile: str, default_label: str = DEFAULT_PROFILE_LABEL) -> Path:
    if selected_profile == default_label:
        return DATA_DIR / "interview_data.json"
    return PROFILES_DIR / selected_profile


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


def _persist_session_interviewers(profile_path: Path, profile_data: Dict[str, Any], session_interviewers: List[str]) -> bool:
    write_payload = dict(profile_data)
    write_payload[JD_PROFILE_SESSION_KEY] = session_interviewers
    write_payload.pop("candidate_profile", None)
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profile_path.write_text(json.dumps(write_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return True


def _load_interviewee_profile() -> Dict[str, Any]:
    if not INTERVIEWEE_PROFILE_PATH.exists():
        return {}

    raw_profile = None
    try:
        raw_profile = json.loads(INTERVIEWEE_PROFILE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}

    if not isinstance(raw_profile, dict):
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
        raise RuntimeError("PDF parsing library is not available. Add `pypdf` to requirements and restart.")

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

    request_text = f"Document source: {source_label}\n\nText:\n{source_text}"

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
                        {"role": "system", "content": "Extract fields and output JSON only with keys: name, status, core_strengths, focus_phrases."},
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
    INTERVIEWEE_PROFILE_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


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
    merged["core_strengths"] = _normalize_text_list(merged.get("core_strengths", []) + incoming_strengths)
    merged["probes"]["focus_phrases"] = _normalize_text_list(
        merged["probes"].get("focus_phrases", []) + incoming_focus,
    )

    if not merged["probes"]["focus_phrases"]:
        merged["probes"]["focus_phrases"] = [
            "Probe the candidate's ADAS Perception achievements against the JD requirements."
        ]

    return merged


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
            expertise=_normalize_text_list(raw.get("expertise", [])),
            potential_questions=_normalize_text_list(raw.get("potential_questions", [])),
        )
        save_interviewer(profile)


def _normalize_term_item(raw: Any) -> str:
    term = str(raw).strip()
    if not term:
        return ""

    split_items = [item.strip() for item in re.split(r"[;\n|,]+", term) if item.strip()]
    if not split_items:
        split_items = [term]

    normalized_items: List[str] = []
    for item in split_items:
        item = item.replace("C/C++", "C++").replace("c/c++", "C++")
        item = re.sub(r"\s{2,}", " ", item).strip(" -")
        item = item.replace("  ", " ")
        for prefix in _REDUNDANT_PREFIXES:
            pattern = re.compile(rf"^{re.escape(prefix)}\s+", re.IGNORECASE)
            cleaned = pattern.sub("", item).strip()
            if cleaned != item:
                item = cleaned
                break
        if not item:
            continue
        normalized_items.append(item)
    return normalized_items[0] if normalized_items else ""


def _normalize_terms(values: Any) -> List[str]:
    if not isinstance(values, list):
        return []
    normalized: List[str] = []
    seen: set[str] = set()

    for raw in values:
        text = str(raw).strip()
        if not text:
            continue
        items = [part.strip() for part in re.split(r"[;\n|,]+", text) if part.strip()]
        if not items:
            continue

        for item in items:
            term = _normalize_term_item(item)
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
        if inferred_category in catalog:
            catalog[inferred_category].append(term)
            continue
        catalog["Role Requirements"].append(term)

    for term in tech_stack:
        catalog["Tools & Framework"].append(term)

    return _normalize_jd_keyword_catalog(catalog)


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
        JD_PROFILE_SESSION_KEY: [GENERIC_INTERVIEWER_LABEL],
        "interviewers": [],
    }


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
                continue

    if last_error is not None:
        raise last_error
    raise ValueError("Failed to parse interviewer profile with available models.")


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
                fallback_temperature = _coerce_temperature_for_model(fallback_model, temperature)
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
        "Provide one strong, concise model answer in English that a good candidate could deliver. "
        "Keep it practical, specific, and interview-ready. "
        "Include one implementation detail and one risk/quality consideration.\n\n"
        f"Interviewer role: {role}\n\nQuestion:\n{target_question}"
    )


def get_feedback_highlight_catalog(profile_data: Dict[str, Any]) -> Dict[str, List[str]]:
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


def hash_file_contents(source_data: bytes) -> str:
    return hashlib.md5(source_data).hexdigest()
