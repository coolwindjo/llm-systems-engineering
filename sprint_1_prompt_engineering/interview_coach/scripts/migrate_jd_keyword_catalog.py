from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
DEFAULT_JD_KEY = "jd_keyword_catalog"
PROFILE_DIR = DATA_DIR / "profiles"
DEFAULT_PROFILE = DATA_DIR / "interview_data.json"

JD_KEYWORD_CATEGORY_RULES: List[tuple[str, List[str]]] = [
    ("Safety & Compliance", ["aspice", "iso 26262", "sotif", "asil", "safety", "fmea"]),
    ("Testing & Quality", ["test", "testing", "validation", "verification", "traceability", "quality"]),
    ("ADAS Core", ["adas", "radar", "camera", "lidar", "fusion", "perception", "localization", "tracking", "mapping"]),
    ("Runtime & Implementation", ["c++", "cpp", "misra", "autosar", "dma", "determinism", "real-time", "concurrency", "memory", "thread"]),
    ("Tools & Framework", ["python", "docker", "can", "lin", "ethernet", "git", "jenkins", "jira", "matlab", "simulink", "ros", "linux"]),
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


def _normalize_term_item(raw: Any) -> str:
    term = str(raw).strip()
    if not term:
        return ""

    # Keep slash in technology identifiers (like C/C++), but split on list delimiters.
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


def normalize_terms(values: Any) -> List[str]:
    if not isinstance(values, list):
        return []
    normalized: List[str] = []
    seen: set[str] = set()

    for raw in values:
        base = str(raw).strip()
        if not base:
            continue
        items = [part.strip() for part in re.split(r"[;\n|,]+", base) if part.strip()]
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


def classify_term(term: str) -> str:
    candidate = term.lower()
    for category, hints in JD_KEYWORD_CATEGORY_RULES:
        if any(hint in candidate for hint in hints):
            return category
    return "Role Requirements"


def normalize_catalog(values: Dict[str, List[str]]) -> Dict[str, List[str]]:
    output: Dict[str, List[str]] = {}
    for category, items in values.items():
        cleaned = normalize_terms(items)
        if cleaned:
            output[category] = cleaned
    return output


def build_jd_keyword_catalog(profile: Dict[str, Any]) -> Dict[str, List[str]]:
    requirements = normalize_terms(profile.get("key_requirements", []))
    tech_stack = normalize_terms(profile.get("tech_stack", []))

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
        category = classify_term(term)
        if category in catalog:
            catalog[category].append(term)
            continue
        catalog["Role Requirements"].append(term)

    catalog["Tools & Framework"].extend(tech_stack)
    return normalize_catalog(catalog)


def collect_profile_paths(explicit: List[Path] | None) -> List[Path]:
    if explicit:
        return explicit

    files = [DEFAULT_PROFILE]
    if PROFILE_DIR.exists():
        files.extend(sorted(PROFILE_DIR.glob("*.json")))

    return files


def migrate_profile(path: Path, dry_run: bool) -> bool:
    if not path.exists():
        print(f"[skip] missing file: {path}")
        return False

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        print(f"[skip] non-object payload: {path.name}")
        return False

    if "key_requirements" not in payload and "tech_stack" not in payload:
        print(f"[skip] no JD fields: {path.name}")
        return False

    next_catalog = build_jd_keyword_catalog(payload)
    current_catalog = payload.get(DEFAULT_JD_KEY)
    if isinstance(current_catalog, dict) and json.dumps(current_catalog, ensure_ascii=False, sort_keys=True) == json.dumps(
        next_catalog, ensure_ascii=False, sort_keys=True
    ):
        print(f"[skip] already up-to-date: {path.name}")
        return False

    payload[DEFAULT_JD_KEY] = next_catalog
    if not dry_run:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[updated] {path.name}")
    else:
        print(f"[dry-run] {path.name}: would update jd_keyword_catalog")
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild jd_keyword_catalog from existing JD fields in profile JSON files."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="Data directory containing interview_data.json and profiles/",
    )
    parser.add_argument(
        "--profile",
        action="append",
        type=Path,
        help="Optional specific profile path (repeatable).",
        default=[],
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show planned updates without writing files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir
    global DATA_DIR, PROFILE_DIR, DEFAULT_PROFILE

    DATA_DIR = data_dir
    PROFILE_DIR = DATA_DIR / "profiles"
    DEFAULT_PROFILE = DATA_DIR / "interview_data.json"

    explicit_paths: List[Path] = []
    for path in args.profile:
        explicit_paths.append(path if path.is_absolute() else (DATA_DIR / path))

    profiles = collect_profile_paths(explicit_paths)
    if not profiles:
        print("No profile files found.")
        return

    changed = 0
    for path in profiles:
        if migrate_profile(path, dry_run=args.dry_run):
            changed += 1

    print(f"\nTotal updated: {changed} / {len(profiles)}")
    if args.dry_run:
        print("Run again without --dry-run to write files.")


if __name__ == "__main__":
    main()
