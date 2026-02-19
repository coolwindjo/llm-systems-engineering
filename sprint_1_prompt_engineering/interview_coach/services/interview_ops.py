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
from pydantic import BaseModel, Field, ValidationError

try:
    from openai import APIError
except (ImportError, ModuleNotFoundError):
    class APIError(Exception):
        """Fallback APIError when openai dependency is unavailable."""

from utils.interviewer_store import (
    InterviewerProfile,
    get_interviewers_by_jd,
    list_interviewers_with_paths,
    save_interviewer,
)
from services.jd_keyword_catalog import (
    JD_KEYWORD_CATEGORY_RULES,
    _normalize_term_item as _normalize_term_item_from_catalog,
    _normalize_terms as _normalize_terms_from_catalog,
    build_jd_keyword_catalog as _build_jd_keyword_catalog_from_catalog,
    classify_term as _classify_jd_keyword_from_catalog,
    normalize_catalog as _normalize_jd_keyword_catalog_from_catalog,
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
PROMPTS_PATH = ROOT_DIR / "prompts" / "system_prompts.json"
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

DEFAULT_CRITIQUE_PERSONA = """You are the {company} {technical_domain} Interview Critique Persona.\nYou evaluate a candidate's latest answer for a {company} technical interview as {interviewer_name} for the role of {position}.\nFocus strictly on:\n1) JD-specific technical and process terms: {focus_terms}.\n2) Technical accuracy for {technical_scope} context\n\nProvide concise, actionable feedback using this exact structure:\n- Overall verdict (2-3 lines)\n- {quality_metric} score: <1-5> with one-sentence rationale\n- Technical accuracy score: <1-5> with one-sentence rationale\n- Strengths: 2-3 bullets\n- Gaps / risks: 2-3 bullets\n- Improvement plan for next answer: 3 bullets with concrete phrasing examples""".strip()

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

_CRITIQUE_PROFILE_STYLE_RULES: List[Tuple[str, Tuple[str, ...]]] = [
    (
        "ai_ml",
        (
            "ai",
            "machine learning",
            "deep learning",
            "neural",
            "computer vision",
            "ml",
            "founder",
            "generative",
            "model",
        ),
    ),
    (
        "safety_management",
        (
            "safety",
            "aspice",
            "iso 26262",
            "asil",
            "functional safety",
            "compliance",
            "program manager",
            "team",
            "quality",
        ),
    ),
    (
        "integration_testability",
        (
            "diagnostic",
            "testing",
            "integration",
            "traceability",
            "c/c++",
            "cpp",
            "c++",
            "quality",
            "automotive",
            "uds",
        ),
    ),
]

_ESSENTIAL_FOCUS_TERMS: tuple[str, ...] = (
    "ASPICE CL3",
    "ISO 26262",
    "SOTIF",
    "Safety Case",
    "ASIL",
    "Functional Safety",
)

_CRITIQUE_TECHNICAL_DOMAIN_HINTS: List[Tuple[str, Tuple[str, ...]]] = [
    ("ADAS / automotive", ("adas", "automotive", "driver assistance", "autonomous", "perception", "planning", "control", "radar", "lidar", "camera")),
    ("AI/ML", ("machine learning", "deep learning", "ai", "artificial intelligence", "computer vision", "model", "neural")),
    ("Safety-critical software", ("safety", "functional safety", "iso 26262", "aspice", "asil", "safety case", "sotif")),
    (
        "Embedded systems",
        (
            "embedded",
            "c++",
            "cpp",
            "autosar",
            "misra",
            "realtime",
            "real-time",
            "thread",
            "concurrency",
        ),
    ),
    ("General software", ("software", "architecture", "development", "implementation", "testing")),
]

_CRITIQUE_TECHNICAL_SCOPE_HINTS: List[Tuple[str, Tuple[str, ...]]] = [
    ("embedded C++ and diagnostics", ("c++", "cpp", "embedded", "diagnostic", "autosar", "realtime", "real-time")),
    ("AI/ML systems", ("machine learning", "deep learning", "ai", "computer vision", "model", "training", "inference")),
    ("safety-critical control systems", ("safety", "safety case", "functional safety", "iso 26262", "aspice")),
    ("software engineering", ("software", "implementation", "architecture", "testing", "validation", "verification")),
]

_CRITIQUE_PROCESS_METRIC_HINTS: List[Tuple[str, Tuple[str, ...]]] = [
    ("ASPICE CL3", ("aspice cl3",)),
    ("ASPICE", ("aspice",)),
    ("ISO 26262", ("iso 26262", "iso26262")),
    ("SOTIF", ("sotif",)),
    ("Quality", ("quality", "testing", "traceability")),
]

JD_KEYWORD_CATEGORY_RULES = list(JD_KEYWORD_CATEGORY_RULES)


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


def _prompt_context_key(value: Any) -> str:
    text = str(value).strip().lower()
    if not text:
        return ""
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")


def _resolve_critique_profile_style_candidates(raw_profile: Dict[str, Any] | None) -> List[str]:
    profile = raw_profile if isinstance(raw_profile, dict) else {}
    if bool(profile.get("is_generic_ai")):
        return ["technical_general"]

    explicit_styles = profile.get("critique_style")
    if isinstance(explicit_styles, str):
        explicit_styles = explicit_styles.strip()
    if isinstance(explicit_styles, str) and explicit_styles:
        return [_prompt_context_key(explicit_styles)]

    if isinstance(explicit_styles, list):
        output = []
        for style_item in explicit_styles:
            candidate = _prompt_context_key(style_item)
            if candidate and candidate not in output:
                output.append(candidate)
        if output:
            return output

    critique_profile = profile.get("critique_profile")
    if isinstance(critique_profile, dict):
        explicit_profile_styles = critique_profile.get("style")
        if isinstance(explicit_profile_styles, str):
            explicit_profile_styles = explicit_profile_styles.strip()
        if isinstance(explicit_profile_styles, str) and explicit_profile_styles:
            return [_prompt_context_key(explicit_profile_styles)]
        if isinstance(explicit_profile_styles, list):
            output = []
            for style_item in explicit_profile_styles:
                candidate = _prompt_context_key(style_item)
                if candidate and candidate not in output:
                    output.append(candidate)
            if output:
                return output

    profile_text = " ".join(
        str(value).strip().lower()
        for value in [
            profile.get("role"),
            profile.get("background"),
            " ".join([str(item).strip().lower() for item in profile.get("expertise", []) if str(item).strip()]),
        ]
        if str(value).strip()
    )

    candidate_scores: List[Tuple[int, int, str]] = []
    for rank, (style_key, tokens) in enumerate(_CRITIQUE_PROFILE_STYLE_RULES):
        score = sum(1 for token in tokens if token in profile_text)
        if score > 0:
            candidate_scores.append((score, -rank, style_key))

    if not candidate_scores:
        return ["technical_general"]

    candidate_scores.sort(reverse=True)
    return [style for _, _, style in candidate_scores]


def _collect_critique_template_keys(
    jd_profile: Dict[str, Any] | None = None,
    selected_jd_title: str | None = None,
) -> tuple[List[str], List[str]]:
    profile = jd_profile if isinstance(jd_profile, dict) else {}
    context = _build_jd_context(profile, selected_jd_title=selected_jd_title)

    company = context["company"]
    position = context["position"]
    domain = context["domain"]

    position_keys: List[str] = []
    company_keys: List[str] = []
    seen_position: set[str] = set()
    seen_company: set[str] = set()

    for value in [position, selected_jd_title, domain]:
        slug = _prompt_context_key(value)
        if not slug or slug in seen_position:
            continue
        seen_position.add(slug)
        position_keys.append(slug)

    position_text = str(position).lower()
    position_tokens = ["adas", "safety", "ai", "cpp", "embedded", "software", "automotive"]
    for token in position_tokens:
        if token in position_text and token not in seen_position:
            seen_position.add(token)
            position_keys.append(token)

    company_slug = _prompt_context_key(company)
    if company_slug:
        if company_slug not in seen_company:
            seen_company.add(company_slug)
            company_keys.append(company_slug)
        company_short = company_slug.replace("_engineering", "").replace("_gmbh", "")
        if company_short and company_short not in seen_company:
            seen_company.add(company_short)
            company_keys.append(company_short)

    return position_keys, company_keys


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
    except (APIError, AttributeError, IndexError, OSError, TypeError, ValueError) as exc:
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
        except (APIError, AttributeError, IndexError, KeyError, OSError, TypeError, ValueError, ValidationError) as exc:
            last_error = exc
            if is_model_access_error(exc):
                continue
            try:
                fallback_response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": "Extract fields and output JSON only with keys: name, status, core_strengths, focus_phrases.",
                        },
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
            except (APIError, AttributeError, KeyError, OSError, TypeError, ValueError, ValidationError):
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
    return _normalize_term_item_from_catalog(raw)


def _normalize_terms(values: Any) -> List[str]:
    return _normalize_terms_from_catalog(values)


def _classify_jd_keyword(term: str) -> str:
    return _classify_jd_keyword_from_catalog(term)


def _normalize_jd_keyword_catalog(values: Dict[str, List[str]]) -> Dict[str, List[str]]:
    return _normalize_jd_keyword_catalog_from_catalog(values)


def _build_jd_keyword_catalog(extracted: JDExtraction) -> Dict[str, List[str]]:
    return _build_jd_keyword_catalog_from_catalog(extracted.model_dump())


def is_model_access_error(exc: Exception) -> bool:
    error_text = str(exc).lower()
    return (
        "model_not_found" in error_text
        or "does not have access" in error_text
        or "does not have access to model" in error_text
        or "does not have access to the model" in error_text
    )


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
        except (APIError, AttributeError, IndexError, KeyError, OSError, TypeError, ValueError, ValidationError) as exc:
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
        except (APIError, AttributeError, IndexError, KeyError, OSError, TypeError, ValueError, ValidationError) as exc:
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
            except (APIError, AttributeError, KeyError, OSError, TypeError, ValueError, ValidationError):
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
    except (APIError, AttributeError, IndexError, KeyError, OSError, RuntimeError, TypeError, ValueError, ValidationError) as exc:
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
                except (APIError, AttributeError, IndexError, KeyError, OSError, RuntimeError, TypeError, ValueError, ValidationError) as fallback_exc:
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


def _load_prompt_templates(path: Path | None = None) -> Dict[str, object]:
    target = path or PROMPTS_PATH
    if not target.exists():
        return {}
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(raw, dict):
        return {}

    templates: Dict[str, object] = {}
    for key, value in raw.items():
        if isinstance(key, str):
            templates[key] = value
    return templates


def get_prompt_template(
    template_name: str,
    fallback: str,
    *,
    path: Path | None = None,
    context: Dict[str, object] | None = None,
) -> str:
    templates = _load_prompt_templates(path=path)
    template = templates.get(template_name, fallback)

    if not isinstance(template, str):
        template = fallback

    template = template.strip()
    if not template:
        return fallback.strip()

    template_text = template.strip()
    if not context:
        return template_text

    string_context = {key: str(value) for key, value in context.items()}
    try:
        return template_text.format(**string_context)
    except KeyError:
        return template_text


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _first_non_empty(values: List[Any]) -> str:
    for value in values:
        text = _safe_text(value)
        if text:
            return text
    return ""


def _build_jd_context(
    jd_profile: Dict[str, Any] | None,
    selected_jd_title: str | None = None,
) -> Dict[str, str]:
    profile = jd_profile if isinstance(jd_profile, dict) else {}
    company = _first_non_empty([
        profile.get("company"),
        profile.get("target_company"),
        profile.get("organization"),
        profile.get("company_name"),
    ])

    position = _first_non_empty([
        profile.get("position"),
        profile.get("title"),
        profile.get("role"),
    ])

    if not position and isinstance(profile.get("job_positions"), list):
        raw_job_positions = profile.get("job_positions")
        if raw_job_positions:
            first_job = raw_job_positions[0]
            if isinstance(first_job, dict):
                position = _safe_text(first_job.get("title"))

    selected_title = _safe_text(selected_jd_title)
    if selected_title:
        position = position or selected_title

    return {
        "organization": company or "Capgemini",
        "company": company or "Capgemini",
        "position": position or "this role",
        "jd_title": position or selected_title or "this role",
        "domain": _first_non_empty([_safe_text(selected_jd_title), position]) or "interview",
    }


def _collect_critique_prompt_context_corpus(
    jd_profile: Dict[str, Any] | None = None,
    interviewer_profile: Dict[str, Any] | None = None,
) -> str:
    profile = jd_profile if isinstance(jd_profile, dict) else {}
    raw_interviewer_profile = interviewer_profile if isinstance(interviewer_profile, dict) else {}
    catalog = profile.get(JD_KEYWORD_CATALOG_KEY, {})

    catalog_terms: List[str] = []
    preferred_categories = [
        "Safety & Compliance",
        "Testing & Quality",
        "ADAS Core",
        "Runtime & Implementation",
        "Tools & Framework",
        "Tech Stack",
        "Role Requirements",
    ]
    if isinstance(catalog, dict):
        for category in preferred_categories:
            raw_terms = catalog.get(category, [])
            if isinstance(raw_terms, list):
                catalog_terms.extend(raw_terms)

    if not catalog_terms:
        catalog_terms.extend(profile.get("key_requirements", []) or [])
        catalog_terms.extend(profile.get("tech_stack", []) or [])

    raw_focus_terms: List[str] = []
    raw_focus_terms.extend(_safe_text(profile.get("job_description")).split())
    raw_focus_terms.extend(_safe_text(profile.get("description")).split())
    raw_focus_terms.extend(_safe_text(profile.get("position")).split())
    raw_focus_terms.extend(_safe_text(profile.get("title")).split())
    raw_focus_terms.extend(profile.get("role") if isinstance(profile.get("role"), list) else [profile.get("role", "")])
    raw_focus_terms.extend(catalog_terms)

    raw_reviewer_focus = raw_interviewer_profile.get("critique_focus_terms")
    if isinstance(raw_reviewer_focus, list):
        raw_focus_terms.extend(raw_reviewer_focus)
    critique_profile = raw_interviewer_profile.get("critique_profile")
    if isinstance(critique_profile, dict):
        raw_profile_focus_terms = critique_profile.get("focus_terms")
        if isinstance(raw_profile_focus_terms, list):
            raw_focus_terms.extend(raw_profile_focus_terms)

    normalized_terms = _normalize_terms(raw_focus_terms)
    normalized_company = re.sub(r"\s{2,}", " ", _first_non_empty([profile.get("company"), profile.get("organization")]))
    raw_profile_fields = [
        _safe_text(profile.get("job_description")),
        _safe_text(profile.get("description")),
        _safe_text(profile.get("title")),
        _safe_text(profile.get("position")),
        _safe_text(profile.get("role")),
        normalized_company,
    ]

    normalized_values = [term.lower() for term in normalized_terms if isinstance(term, str)]
    raw_values = [value.lower() for value in raw_profile_fields if value]
    corpus = " ".join(normalized_values + raw_values)
    return corpus.replace("c/c++", "C++").replace("C/C++", "C++")


def _pick_critique_prompt_hint(
    corpus: str,
    hint_rules: List[Tuple[str, Tuple[str, ...]]],
    default: str,
) -> str:
    if not isinstance(hint_rules, list):
        return default

    lowered_corpus = (corpus or "").lower()
    for phrase, tokens in hint_rules:
        if not isinstance(tokens, tuple):
            continue
        for token in tokens:
            token_text = str(token).strip().lower()
            if not token_text:
                continue
            if token_text in lowered_corpus:
                return phrase
    return default


def _resolve_critique_prompt_context(
    jd_profile: Dict[str, Any] | None = None,
    interviewer_profile: Dict[str, Any] | None = None,
) -> Dict[str, str]:
    corpus = _collect_critique_prompt_context_corpus(
        jd_profile=jd_profile,
        interviewer_profile=interviewer_profile,
    )
    return {
        "technical_domain": _pick_critique_prompt_hint(
            corpus=corpus,
            hint_rules=_CRITIQUE_TECHNICAL_DOMAIN_HINTS,
            default="General software",
        ),
        "technical_scope": _pick_critique_prompt_hint(
            corpus=corpus,
            hint_rules=_CRITIQUE_TECHNICAL_SCOPE_HINTS,
            default="software engineering",
        ),
        "quality_metric": _pick_critique_prompt_hint(
            corpus=corpus,
            hint_rules=_CRITIQUE_PROCESS_METRIC_HINTS,
            default="Technical quality",
        ),
    }


def _collect_critique_focus_terms(
    jd_profile: Dict[str, Any] | None,
    interviewer_profile: Dict[str, Any] | None = None,
    fallback_title: str | None = None,
) -> str:
    profile = jd_profile if isinstance(jd_profile, dict) else {}
    raw_interviewer_profile = interviewer_profile if isinstance(interviewer_profile, dict) else {}
    catalog = profile.get(JD_KEYWORD_CATALOG_KEY, {})

    preferred_categories = [
        "Safety & Compliance",
        "Testing & Quality",
        "ADAS Core",
        "Runtime & Implementation",
        "Tools & Framework",
        "Tech Stack",
        "Role Requirements",
    ]

    candidate_terms: List[str] = []
    if isinstance(catalog, dict):
        for category in preferred_categories:
            raw_terms = catalog.get(category, [])
            if isinstance(raw_terms, list):
                candidate_terms.extend(raw_terms)

    if not candidate_terms:
        candidate_terms.extend(profile.get("key_requirements", []) or [])
        candidate_terms.extend(profile.get("tech_stack", []) or [])

    reviewer_focus_terms: List[str] = []
    raw_focus_terms = raw_interviewer_profile.get("critique_focus_terms")
    if isinstance(raw_focus_terms, list):
        reviewer_focus_terms.extend(raw_focus_terms)
    critique_profile = raw_interviewer_profile.get("critique_profile")
    if isinstance(critique_profile, dict):
        raw_profile_focus_terms = critique_profile.get("focus_terms")
        if isinstance(raw_profile_focus_terms, list):
            reviewer_focus_terms.extend(raw_profile_focus_terms)

    normalized_profile_terms = _normalize_terms(reviewer_focus_terms)
    if normalized_profile_terms:
        candidate_terms.extend(normalized_profile_terms)

    normalized_terms = _normalize_terms(candidate_terms)
    normalized_term_index = {term.lower() for term in normalized_terms}
    raw_profile_text = [
        str(value).strip().lower()
        for value in [
            profile.get("job_description"),
            profile.get("description"),
            profile.get("position"),
            profile.get("role"),
            profile.get("title"),
        ]
        if str(value).strip()
    ]
    raw_profile_text.extend(
        str(value).strip().lower()
        for value in candidate_terms + (profile.get("key_requirements", []) or [])
        if str(value).strip()
    )
    profile_text = " ".join(raw_profile_text)
    for injected_term in _ESSENTIAL_FOCUS_TERMS:
        if injected_term.lower() in normalized_term_index:
            continue
        if injected_term.lower() in profile_text:
            normalized_terms.append(injected_term)
            normalized_term_index.add(injected_term.lower())

    if not normalized_terms:
        prompt_context = _resolve_critique_prompt_context(jd_profile=profile, interviewer_profile=raw_interviewer_profile)
        fallback_title_text = _safe_text(fallback_title)
        if fallback_title_text:
            return ", ".join(
                [
                    fallback_title_text,
                    prompt_context["quality_metric"],
                    f"{prompt_context['technical_domain']} implementation quality",
                ]
            )
        return ", ".join([prompt_context["quality_metric"], f"{prompt_context['technical_domain']} implementation quality"])

    top_terms = normalized_terms[:6]
    return ", ".join(top_terms)


def _resolve_critique_persona_template_entry(
    raw: Any,
    interviewer_key: str | None,
    technique_key: str | None,
    profile_styles: List[str] | None = None,
    position_keys: List[str] | None = None,
    company_keys: List[str] | None = None,
) -> str | None:
    if not isinstance(raw, dict):
        if isinstance(raw, str):
            return raw.strip() or None
        return None

    # 1) Specific combo: "{interviewer}.{technique}"
    if interviewer_key and technique_key:
        by_combo = raw.get("by_combo")
        if isinstance(by_combo, dict):
            candidate = by_combo.get(f"{interviewer_key}.{technique_key}")
            if isinstance(candidate, str):
                value = candidate.strip()
                if value:
                    return value

    if profile_styles and isinstance(profile_styles, list):
        by_combo_profile = raw.get("by_style_and_technique")
        by_profile_style = raw.get("by_style")
        if isinstance(by_combo_profile, dict) and technique_key:
            for style_key in profile_styles:
                candidate = by_combo_profile.get(f"{style_key}.{technique_key}")
                if isinstance(candidate, str):
                    value = candidate.strip()
                    if value:
                        return value
        if isinstance(by_profile_style, dict):
            for style_key in profile_styles:
                candidate = by_profile_style.get(style_key)
                if isinstance(candidate, str):
                    value = candidate.strip()
                    if value:
                        return value

    # 2) Interviewer-scoped
    if interviewer_key:
        by_interviewer = raw.get("by_interviewer")
        if isinstance(by_interviewer, dict):
            candidate = by_interviewer.get(interviewer_key)
            if isinstance(candidate, str):
                value = candidate.strip()
                if value:
                    return value

    # 3) Technique-scoped
    if technique_key:
        by_technique = raw.get("by_technique")
        if isinstance(by_technique, dict):
            candidate = by_technique.get(technique_key)
            if isinstance(candidate, str):
                value = candidate.strip()
                if value:
                    return value

    if position_keys:
        by_position = raw.get("by_position")
        if isinstance(by_position, dict):
            for key in position_keys:
                candidate = by_position.get(key)
                if isinstance(candidate, str):
                    value = candidate.strip()
                    if value:
                        return value

    if company_keys:
        by_company = raw.get("by_company")
        if isinstance(by_company, dict):
            for key in company_keys:
                candidate = by_company.get(key)
                if isinstance(candidate, str):
                    value = candidate.strip()
                    if value:
                        return value

    # 4) Default under structured form
    default_value = raw.get("default")
    if isinstance(default_value, str):
        value = default_value.strip()
        if value:
            return value

    return None


def get_critique_persona_prompt(
    *,
    interviewer_name: str | None = None,
    interviewer_key: str | None = None,
    technique: str | None = None,
    interviewer_profile: Dict[str, Any] | None = None,
    jd_profile: Dict[str, Any] | None = None,
    jd_title: str | None = None,
    path: Path | None = None,
) -> str:
    jd_context = _build_jd_context(jd_profile, selected_jd_title=jd_title)
    templates = _load_prompt_templates(path=path)
    raw_critique = templates.get("critique_persona", DEFAULT_CRITIQUE_PERSONA)

    normalized_key: str | None = _speaker_key(interviewer_key) if interviewer_key else None
    normalized_key = normalized_key or None
    normalized_technique = technique.strip() if isinstance(technique, str) else None
    profile_styles = _resolve_critique_profile_style_candidates(interviewer_profile)
    position_keys, company_keys = _collect_critique_template_keys(jd_profile, selected_jd_title=jd_title)
    critique_prompt_context = _resolve_critique_prompt_context(
        jd_profile=jd_profile,
        interviewer_profile=interviewer_profile,
    )
    selected = _resolve_critique_persona_template_entry(
        raw_critique,
        interviewer_key=normalized_key,
        technique_key=normalized_technique,
        profile_styles=profile_styles,
        position_keys=position_keys,
        company_keys=company_keys,
    )
    if selected is None:
        selected = DEFAULT_CRITIQUE_PERSONA

    return get_prompt_template(
        "critique_persona",
        selected,
        path=path,
        context={
            "interviewer_name": interviewer_name or "Interviewer",
            "interviewer_key": normalized_key or "",
            "technique": normalized_technique or "",
            "organization": jd_context["organization"],
            "company": jd_context["company"],
            "position": jd_context["position"],
            "jd_title": jd_context["jd_title"],
            "jd_domain": jd_context["domain"],
            "technical_domain": critique_prompt_context["technical_domain"],
            "technical_scope": critique_prompt_context["technical_scope"],
            "quality_metric": critique_prompt_context["quality_metric"],
            "focus_terms": _collect_critique_focus_terms(
                jd_profile,
                interviewer_profile=interviewer_profile,
                fallback_title=jd_title,
            ),
            "interviewer_style": _safe_text(
                profile_styles[0] if profile_styles else "technical_general"
            ),
        },
    )


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
