from __future__ import annotations

import json

from scripts.migrate_jd_keyword_catalog import (
    build_jd_keyword_catalog,
    collect_profile_paths,
    migrate_profile,
    normalize_terms,
)


def test_normalize_terms_removes_noise_and_duplicates() -> None:
    values = [
        "  Experience in C++  ",
        "experience in C++",
        "C/C++",
        "Python, testing",
        "",
        "  ",
    ]
    assert normalize_terms(values) == ["C++", "Python", "testing"]


def test_build_jd_keyword_catalog_classifies_terms() -> None:
    profile = {
        "key_requirements": ["Safety", "ISO 26262", "ADAS stack"],
        "tech_stack": ["C++", "Docker", "Python"],
    }
    catalog = build_jd_keyword_catalog(profile)

    assert catalog["Safety & Compliance"] == ["Safety", "ISO 26262"]
    assert catalog["ADAS Core"] == ["ADAS stack"]
    assert catalog["Tools & Framework"] == ["C++", "Docker", "Python"]


def test_collect_profile_paths_respects_explicit_overrides(tmp_path) -> None:
    profile1 = tmp_path / "p1.json"
    profile2 = tmp_path / "p2.json"
    explicit = [profile1, profile2]

    collected = collect_profile_paths(explicit)
    assert collected == explicit


def test_collect_profile_paths_defaults(tmp_path) -> None:
    from scripts import migrate_jd_keyword_catalog as migrate

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    profile_dir = data_dir / "profiles"
    profile_dir.mkdir()

    default_profile = data_dir / "interview_data.json"
    default_profile.write_text("{}", encoding="utf-8")
    p1 = profile_dir / "a.json"
    p2 = profile_dir / "b.json"
    p1.write_text("{}", encoding="utf-8")
    p2.write_text("{}", encoding="utf-8")

    migrate.DATA_DIR = data_dir
    migrate.PROFILE_DIR = profile_dir
    migrate.DEFAULT_PROFILE = default_profile

    collected = collect_profile_paths([])
    assert collected == [default_profile, p1, p2]


def test_migrate_profile_writes_catalog_and_skips_when_up_to_date(tmp_path, monkeypatch) -> None:
    profile = tmp_path / "sample.json"
    profile.write_text(json.dumps({"key_requirements": ["Safety", "ADAS"], "tech_stack": ["Python"]}), encoding="utf-8")

    changed = migrate_profile(profile, dry_run=True)
    assert changed is True
    assert "jd_keyword_catalog" not in profile.read_text(encoding="utf-8")

    changed = migrate_profile(profile, dry_run=False)
    payload = json.loads(profile.read_text(encoding="utf-8"))
    assert changed is True
    assert "jd_keyword_catalog" in payload
    assert payload["jd_keyword_catalog"]

    assert migrate_profile(profile, dry_run=False) is False


def test_migrate_profile_skips_when_no_target_fields(tmp_path) -> None:
    profile = tmp_path / "missing.json"
    profile.write_text("{}", encoding="utf-8")

    changed = migrate_profile(profile, dry_run=False)
    assert changed is False


def test_migrate_profile_skips_missing_file(tmp_path) -> None:
    assert migrate_profile(tmp_path / "missing.json", dry_run=False) is False
