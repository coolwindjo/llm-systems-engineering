from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List

from utils.interviewer_store import get_interviewers_by_jd


DEFAULT_PROFILE_LABEL = "(Default) interview_data.json"
REQUIRED_SESSION_INTERVIEWERS = ["Generic AI Interviewer"]
RECOMMENDATION_SCORE_THRESHOLD = 25.0
RECOMMENDATION_TOKEN_WEIGHT = 0.55
RECOMMENDATION_PHRASE_WEIGHT = 0.45
RECOMMENDATION_MIN_SCORE_BOTH_MATCH = 80.0
RECOMMENDATION_MIN_SCORE_ONE_MATCH = 40.0
DEPARTMENT_MATCH_STOP_WORDS = {
    "in",
    "on",
    "of",
    "to",
    "for",
    "with",
    "your",
    "you",
    "and",
    "the",
    "a",
    "an",
    "is",
    "are",
    "as",
    "at",
    "by",
    "from",
    "their",
    "them",
    "this",
    "that",
    "these",
    "those",
    "it",
    "its",
    "team",
    "teams",
    "position",
    "role",
    "senior",
    "lead",
    "member",
    "member",
    "junior",
    "developer",
    "engineering",
    "engineer",
    "software",
}


def _read_json_file(profile_path: Path) -> Dict[str, Any] | None:
    try:
        data = json.loads(profile_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return None
    return data if isinstance(data, dict) else None


def _to_label_list(values: Any) -> List[str]:
    if not isinstance(values, list):
        return []
    normalized = []
    seen = set()
    for value in values:
        candidate = str(value).strip()
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        normalized.append(candidate)
    return normalized


def _tokenize(value: Any) -> List[str]:
    normalized = re.sub(r"[^a-z0-9\s\+\-]", " ", str(value).lower())
    tokens = [token for token in re.split(r"\s+", normalized) if len(token) > 1]
    return tokens


def _to_set(tokens: List[str]) -> set[str]:
    stop_words = {
        "and",
        "a",
        "an",
        "the",
        "for",
        "you",
        "your",
        "with",
        "including",
        "experience",
        "requirements",
        "responsibilities",
        "required",
        "ability",
        "strong",
        "knowledge",
        "understanding",
        "company",
        "team",
        "teams",
        "position",
        "ability",
        "work",
        "working",
        "including",
        "across",
        "into",
        "that",
        "this",
        "these",
        "their",
        "them",
        "there",
        "when",
        "while",
        "where",
    }
    return {token for token in tokens if token and token not in stop_words}


def _build_jd_profile_text(profile_data: Dict[str, Any]) -> str:
    extracted_requirements: List[str] = []
    for item in _extract_jd_requirements(profile_data):
        extracted_requirements.append(item)

    for requirements in profile_data.get("job_positions", []):
        if not isinstance(requirements, dict):
            continue
        for item in requirements.get("key_requirements", []) or []:
            text = str(item).strip()
            if text and text not in extracted_requirements:
                extracted_requirements.append(text)
        for item in requirements.get("stack", []) or []:
            text = str(item).strip()
            if text and text not in extracted_requirements:
                extracted_requirements.append(text)

    keyword_catalog = profile_data.get("jd_keyword_catalog", {})
    if isinstance(keyword_catalog, dict):
        for terms in keyword_catalog.values():
            if not isinstance(terms, list):
                continue
            for text in terms:
                value = str(text).strip()
                if value and value not in extracted_requirements:
                    extracted_requirements.append(value)

    sections = [
        profile_data.get("company", ""),
        profile_data.get("position", ""),
        " ".join(profile_data.get("tech_stack", []) or []),
        " ".join(extracted_requirements),
    ]
    return " ".join(str(item) for item in sections if item)


def _build_candidate_text(candidate: Dict[str, Any]) -> str:
    fields = [
        candidate.get("name", ""),
        candidate.get("background", ""),
        candidate.get("role", ""),
        " ".join(candidate.get("expertise", []) or []),
        " ".join(candidate.get("potential_questions", []) or []),
    ]
    return " ".join(str(item) for item in fields if item)


def _extract_jd_requirements(profile_data: Dict[str, Any]) -> List[str]:
    requirements = profile_data.get("key_requirements", [])
    if isinstance(requirements, list) and requirements:
        return [str(item) for item in requirements if str(item).strip()]

    requirements_out: List[str] = []
    job_positions = profile_data.get("job_positions", [])
    if isinstance(job_positions, list):
        for position in job_positions:
            if not isinstance(position, dict):
                continue
            for item in position.get("key_requirements", []) or []:
                text = str(item).strip()
                if text and text not in requirements_out:
                    requirements_out.append(text)
    return requirements_out


def _collect_candidate_interviewers(profile_data: Dict[str, Any], jd_title: str) -> List[Dict[str, Any]]:
    candidates = []
    seen = set()

    default_interviewers: List[Dict[str, Any]] = []
    default_path = Path(__file__).resolve().parent.parent / "data" / "interview_data.json"
    raw_default = _read_json_file(default_path)
    if isinstance(raw_default, dict):
        for profile in raw_default.get("interviewers", []):
            if isinstance(profile, dict):
                default_interviewers.append(profile)

    for profile in [*profile_data.get("interviewers", []), *default_interviewers]:
        if not isinstance(profile, dict):
            continue
        name = str(profile.get("name", "")).strip()
        if not name or name.lower() in seen:
            continue
        seen.add(name.lower())
        candidates.append(
            {
                "name": name,
                "source": "profile",
                "role": str(profile.get("role", "")),
                "expertise": profile.get("expertise", []),
                "potential_questions": profile.get("potential_questions", []),
                "background": str(profile.get("background", "")),
                "text": _build_candidate_text(profile),
            }
        )

    for custom_profile in get_interviewers_by_jd(jd_title):
        name = custom_profile.name.strip()
        key = name.lower()
        if not name or key in seen:
            continue
        seen.add(key)
        candidates.append(
            {
                "name": name,
                "source": "custom",
                "background": custom_profile.background,
                "role": str(custom_profile.role),
                "expertise": custom_profile.expertise,
                "potential_questions": custom_profile.potential_questions,
                "text": _build_candidate_text(custom_profile.model_dump()),
            }
        )

    return candidates


def _extract_company_signal(profile_data: Dict[str, Any]) -> str:
    for key in ("company", "target_company"):
        value = str(profile_data.get(key, "")).strip()
        if value:
            return value
    return ""


def _build_department_signals(profile_data: Dict[str, Any]) -> List[str]:
    signals: List[str] = []

    for value in [profile_data.get("position", "")]:
        if isinstance(value, str):
            normalized = value.strip()
            if normalized and normalized not in signals:
                signals.append(normalized)

    for values in profile_data.get("key_requirements", []) or []:
        if isinstance(values, str):
            normalized = values.strip()
            if normalized and normalized not in signals:
                signals.append(normalized)

    for requirements in profile_data.get("job_positions", []):
        if not isinstance(requirements, dict):
            continue
        for title in [requirements.get("title", "")]:
            normalized = str(title).strip()
            if normalized and normalized not in signals:
                signals.append(normalized)
        for text in requirements.get("key_requirements", []) or []:
            normalized = str(text).strip()
            if normalized and normalized not in signals:
                signals.append(normalized)

    return signals


def _is_company_match(profile_data: Dict[str, Any], candidate_text: str) -> bool:
    company = _extract_company_signal(profile_data)
    if not company:
        return False

    company_tokens = _to_set(_tokenize(company))
    candidate_terms = _to_set(_tokenize(candidate_text))
    if not company_tokens or not candidate_terms:
        return False

    if company_tokens.issubset(candidate_terms):
        return True

    candidate_flat = " ".join(_tokenize(candidate_text))
    return all(token in candidate_flat for token in company_tokens)


def _is_department_match(profile_data: Dict[str, Any], candidate_text: str, candidate_terms: set[str]) -> bool:
    department_signals = _build_department_signals(profile_data)
    if not department_signals or not candidate_terms:
        return False

    department_tokens: set[str] = set()
    normalized_candidate = " ".join(_tokenize(candidate_text))

    for signal in department_signals:
        tokens = _to_set(_tokenize(signal))
        if tokens and tokens.issubset(candidate_terms):
            return True
        if len(tokens) >= 2:
            if " ".join(_tokenize(signal)) in normalized_candidate:
                return True
        department_tokens.update(tokens)

    department_tokens = department_tokens - DEPARTMENT_MATCH_STOP_WORDS - {"manager", "specialist"}
    if not department_tokens:
        return False

    overlap_ratio = len(department_tokens & candidate_terms) / len(department_tokens)
    return overlap_ratio >= 0.35


def _extract_jd_phrase_terms(profile_data: Dict[str, Any]) -> List[str]:
    phrases = _extract_jd_requirements(profile_data)
    job_positions = profile_data.get("job_positions", [])
    if isinstance(job_positions, list):
        for position in job_positions:
            if not isinstance(position, dict):
                continue
            for text in position.get("key_requirements", []) or []:
                candidate = str(text).strip()
                if candidate and candidate not in phrases:
                    phrases.append(candidate)
            for text in position.get("stack", []) or []:
                candidate = str(text).strip()
                if candidate and candidate not in phrases:
                    phrases.append(candidate)

    for text in profile_data.get("tech_stack", []) or []:
        candidate = str(text).strip()
        if candidate and candidate not in phrases:
            phrases.append(candidate)

    keyword_catalog = profile_data.get("jd_keyword_catalog", {})
    if isinstance(keyword_catalog, dict):
        for terms in keyword_catalog.values():
            if not isinstance(terms, list):
                continue
            for text in terms:
                candidate = str(text).strip()
                if candidate and candidate not in phrases:
                    phrases.append(candidate)
    return phrases


def _phrase_matches(candidate_terms: set[str], candidate_phrase_text: str, phrase: str) -> bool:
    phrase_tokens = _to_set(_tokenize(phrase))
    if not phrase_tokens:
        return False
    if phrase_tokens.issubset(candidate_terms):
        return True
    normalized_phrase = " ".join(_tokenize(phrase))
    return bool(normalized_phrase) and normalized_phrase in candidate_phrase_text


def _normalize_recommendation_name(name: str) -> str:
    return name.strip() or "Interviewer"


def _compute_correlation_score(
    jd_text: str,
    candidate_text: str,
    profile_data: Dict[str, Any],
    candidate: Dict[str, Any],
) -> float:
    jd_terms = _to_set(_tokenize(jd_text))
    candidate_terms = _to_set(_tokenize(candidate_text))
    if not jd_terms:
        return 0.0
    if not candidate_terms:
        return 0.0

    term_score = len(jd_terms & candidate_terms) / len(jd_terms) * 100

    phrase_terms = [term for term in _extract_jd_phrase_terms(profile_data) if term.strip()]
    if not phrase_terms:
        return round(term_score, 1)

    candidate_phrase_text = " ".join(_tokenize(candidate_text))
    hits = 0
    for phrase in phrase_terms:
        if _phrase_matches(candidate_terms, candidate_phrase_text, phrase):
            hits += 1
    phrase_score = hits / len(phrase_terms) * 100

    score = round(
        (RECOMMENDATION_TOKEN_WEIGHT * term_score) + (RECOMMENDATION_PHRASE_WEIGHT * phrase_score),
        1,
    )

    company_match = _is_company_match(profile_data, candidate.get("text", ""))
    department_match = _is_department_match(profile_data, candidate_text, candidate_terms)
    if company_match and department_match:
        return max(score, RECOMMENDATION_MIN_SCORE_BOTH_MATCH)
    if company_match or department_match:
        return max(score, RECOMMENDATION_MIN_SCORE_ONE_MATCH)
    return score


def _recommend_interviewers(profile_data: Dict[str, Any], jd_title: str, score_threshold: float = RECOMMENDATION_SCORE_THRESHOLD) -> List[Dict[str, Any]]:
    jd_text = _build_jd_profile_text(profile_data)
    candidates = _collect_candidate_interviewers(profile_data, jd_title)
    recommendations = []
    for candidate in candidates:
        score = _compute_correlation_score(
            jd_text,
            candidate["text"],
            profile_data,
            candidate,
        )
        recommendations.append(
            {
                "name": _normalize_recommendation_name(candidate["name"]),
                "raw_name": str(candidate["name"]),
                "score": score,
                "recommended": score >= score_threshold,
                "source": candidate["source"],
            }
        )
    recommendations.sort(key=lambda item: item["score"], reverse=True)
    return recommendations


def recommend_interviewers(profile_data: Dict[str, Any], jd_title: str) -> List[Dict[str, Any]]:
    return _recommend_interviewers(profile_data, jd_title)


def _extract_jd_title(profile_data: Dict[str, Any], profile_path: Path) -> str:
    if profile_path.name == "interview_data.json":
        return DEFAULT_PROFILE_LABEL
    return str(profile_data.get("position", profile_path.stem))


def _profile_status(profile_path: Path, profile_data: Dict[str, Any]) -> Dict[str, Any]:
    jd_title = _extract_jd_title(profile_data, profile_path)
    session_interviewers = _to_label_list(profile_data.get("session_interviewers"))

    session_set = set(session_interviewers)
    missing_required = [name for name in REQUIRED_SESSION_INTERVIEWERS if name not in session_set]
    status = "ok" if not missing_required else "missing_required"

    custom_interviewers = sorted(
        {
            interviewer.name
            for interviewer in get_interviewers_by_jd(jd_title)
            if interviewer.name.strip()
        }
    )

    return {
        "file": profile_path.name,
        "jd_title": jd_title,
        "status": status,
        "session_interviewers": session_interviewers,
        "missing_required": missing_required,
        "recommendations": _recommend_interviewers(profile_data, jd_title),
        "custom_interviewers": custom_interviewers,
        "missing_defaults": missing_required,
    }


def audit_interview_profiles(data_dir: Path) -> List[Dict[str, Any]]:
    profile_dir = data_dir / "profiles"
    default_profile_path = data_dir / "interview_data.json"

    reports: List[Dict[str, Any]] = []
    if default_profile_path.exists():
        default_data = _read_json_file(default_profile_path)
        if default_data is None:
            reports.append(
                {
                    "file": default_profile_path.name,
                    "jd_title": DEFAULT_PROFILE_LABEL,
                    "status": "invalid_file",
                    "session_interviewers": [],
                    "missing_required": REQUIRED_SESSION_INTERVIEWERS.copy(),
                    "missing_defaults": REQUIRED_SESSION_INTERVIEWERS.copy(),
                    "custom_interviewers": [],
                }
            )
        else:
            reports.append(_profile_status(default_profile_path, default_data))

    if profile_dir.exists():
        for path in sorted(profile_dir.glob("*.json")):
            profile_data = _read_json_file(path)
            if profile_data is None:
                reports.append(
                    {
                        "file": path.name,
                        "jd_title": _extract_jd_title({}, path),
                        "status": "invalid_file",
                        "session_interviewers": [],
                        "missing_required": REQUIRED_SESSION_INTERVIEWERS.copy(),
                        "missing_defaults": REQUIRED_SESSION_INTERVIEWERS.copy(),
                        "custom_interviewers": [],
                    }
                )
                continue
            reports.append(_profile_status(path, profile_data))

    return reports
