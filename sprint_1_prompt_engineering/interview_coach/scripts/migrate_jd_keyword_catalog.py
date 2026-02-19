from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

try:
    from ..services.jd_keyword_catalog import (
        JD_KEYWORD_CATEGORY_RULES as _SHARED_KEYWORD_CATEGORIES,
        normalize_terms as _normalize_terms_from_catalog,
        build_jd_keyword_catalog,
    )
except (ImportError, ModuleNotFoundError):
    from sprint_1_prompt_engineering.interview_coach.services.jd_keyword_catalog import (
        JD_KEYWORD_CATEGORY_RULES as _SHARED_KEYWORD_CATEGORIES,
        normalize_terms as _normalize_terms_from_catalog,
        build_jd_keyword_catalog,
    )


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
DEFAULT_JD_KEY = "jd_keyword_catalog"

JD_KEYWORD_CATEGORY_RULES: List[tuple[str, List[str]]] = list(_SHARED_KEYWORD_CATEGORIES)


def normalize_terms(values: List[str]) -> List[str]:
    return _normalize_terms_from_catalog(values)


def collect_profile_paths(
    explicit: List[Path] | None,
    data_dir: Path | None = None,
) -> List[Path]:
    if explicit:
        return explicit

    search_root = data_dir if data_dir is not None else DATA_DIR

    files = [search_root / "interview_data.json"]
    profile_dir = search_root / "profiles"
    if profile_dir.exists():
        files.extend(sorted(profile_dir.glob("*.json")))

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

    explicit_paths: List[Path] = []
    for path in args.profile:
        explicit_paths.append(path if path.is_absolute() else (data_dir / path))

    profiles = collect_profile_paths(explicit_paths, data_dir=data_dir)
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
