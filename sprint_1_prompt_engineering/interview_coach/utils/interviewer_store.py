from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List, Tuple

from pydantic import BaseModel, Field


INTERVIEWERS_DIR = Path(__file__).resolve().parent.parent / "data" / "interviewers"


class InterviewerProfile(BaseModel):
    name: str
    background: str
    is_generic_ai: bool
    role: str = ""
    expertise: List[str] = Field(default_factory=list)
    potential_questions: List[str] = Field(default_factory=list)


def _slugify(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9]+", "-", value.strip())
    normalized = normalized.strip("-").lower()
    return normalized or "unknown"


def _build_filename(profile: InterviewerProfile) -> str:
    name_part = _slugify(profile.name)
    return f"{name_part}.json"


def _coerce_list(values) -> List[str]:
    if not isinstance(values, list):
        return []
    return [str(item).strip() for item in values if str(item).strip()]


def save_interviewer(profile: InterviewerProfile, existing_path: str | None = None) -> Path:
    """Save one interviewer profile as JSON under data/interviewers/."""
    INTERVIEWERS_DIR.mkdir(parents=True, exist_ok=True)
    payload = profile.model_dump()
    payload["role"] = str(profile.role).strip()
    payload["expertise"] = _coerce_list(profile.expertise)
    payload["potential_questions"] = _coerce_list(profile.potential_questions)
    payload["is_generic_ai"] = bool(profile.is_generic_ai)

    if existing_path:
        output_path = Path(existing_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        output_path = INTERVIEWERS_DIR / _build_filename(profile)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def delete_interviewer(profile_path: str) -> None:
    """Delete one interviewer JSON file if it exists."""
    target = Path(profile_path)
    if target.exists():
        target.unlink()


def load_interviewer(profile_path: str | Path) -> InterviewerProfile | None:
    """Load one interviewer profile from a JSON file."""
    return _load_interviewer(Path(profile_path))


def _load_interviewer(path: Path) -> InterviewerProfile | None:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return InterviewerProfile.model_validate(raw)
    except Exception:
        return None


def _all_profile_files() -> List[Path]:
    if not INTERVIEWERS_DIR.exists():
        return []
    return sorted(INTERVIEWERS_DIR.glob("*.json"))


def list_interviewers_with_paths(jd_title: str | None = None) -> List[Tuple[Path, InterviewerProfile]]:
    """Return all interviewer JSON files.

    `jd_title` is kept for backward compatibility and is ignored now that scope is no longer persisted.
    """
    profiles: List[Tuple[Path, InterviewerProfile]] = []
    for path in _all_profile_files():
        profile = _load_interviewer(path)
        if profile is None:
            continue
        profiles.append((path, profile))
    return profiles


def list_all_interviewers() -> List[Tuple[Path, InterviewerProfile]]:
    """Return all persisted interviewer profiles with paths."""
    return [(path, profile) for path, profile in ((path, _load_interviewer(path)) for path in _all_profile_files()) if profile is not None]


def get_interviewers_by_jd(jd_title: str | None = None) -> List[InterviewerProfile]:
    """Return all interviewer profiles (backward-compatible JD-based API)."""
    return [profile for _, profile in list_interviewers_with_paths(jd_title)]


def get_interviewer_path(profile: InterviewerProfile) -> Path:
    return INTERVIEWERS_DIR / _build_filename(profile)


def get_profile_file_options_by_jd(jd_title: str) -> List[Path]:
    """Return persisted profile file paths for the JD and global scope."""
    return [path for path, _ in list_interviewers_with_paths(jd_title)]


def find_interviewer_file(profile: InterviewerProfile, allow_scope_fallback: bool = True) -> Path | None:
    """Find existing file for profile name."""
    target_name = _slugify(profile.name)
    direct_path = INTERVIEWERS_DIR / _build_filename(profile)
    if direct_path.exists():
        return direct_path

    if not allow_scope_fallback:
        return None

    fallback_path = get_profile_file_options_by_jd(None)
    for path in fallback_path:
        loaded = _load_interviewer(path)
        if loaded is not None and _slugify(loaded.name) == target_name:
            return path
    return None
