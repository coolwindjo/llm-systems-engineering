from __future__ import annotations

from pathlib import Path

import pytest

from utils.interviewer_store import (
    InterviewerProfile,
    _build_filename,
    _coerce_list,
    _slugify,
    delete_interviewer,
    find_interviewer_file,
    list_all_interviewers,
    list_interviewers_with_paths,
    load_interviewer,
    save_interviewer,
)


@pytest.fixture
def interviewers_dir(tmp_path, monkeypatch):
    monkeypatch.setattr("utils.interviewer_store.INTERVIEWERS_DIR", tmp_path)
    return tmp_path


def test_slugify_handles_ascii_and_punctuation(interviewers_dir) -> None:
    assert _slugify("  John   Doe  ") == "john-doe"
    assert _slugify("Jane/Doe/III") == "jane-doe-iii"
    assert _slugify("    ") == "unknown"


def test_build_filename_matches_slugify_profile() -> None:
    profile = InterviewerProfile(name="Jane Doe", background="", is_generic_ai=False)
    assert _build_filename(profile) == "jane-doe.json"


def test_coerce_list_filters_empty_entries() -> None:
    assert _coerce_list(["a", "", " b ", 0, None, "a"]) == ["a", "b", "0"]
    assert _coerce_list("not-a-list") == []


def test_save_load_delete_interviewer_flow(interviewers_dir) -> None:
    profile = InterviewerProfile(
        name="Alice Kim",
        background="ADAS expert",
        is_generic_ai=False,
        role="Engineer",
        expertise=["C++", "  "],
        potential_questions=["Q1", "", "Q2"],
    )

    saved_path = save_interviewer(profile)
    assert saved_path == interviewers_dir / "alice-kim.json"
    assert saved_path.exists()

    loaded = load_interviewer(saved_path)
    assert loaded is not None
    assert loaded.name == "Alice Kim"
    assert loaded.expertise == ["C++"]
    assert loaded.potential_questions == ["Q1", "Q2"]

    delete_interviewer(saved_path)
    assert not saved_path.exists()
    assert load_interviewer(saved_path) is None
    # Deleting again should remain a no-op.
    delete_interviewer(saved_path)


def test_list_interviewers_with_paths_filters_invalid_json(interviewers_dir) -> None:
    valid = InterviewerProfile(name="Bob", background="", is_generic_ai=False)
    invalid_path = interviewers_dir / "invalid.json"
    valid_path = save_interviewer(valid)
    invalid_path.write_text("{not-json}", encoding="utf-8")

    records = list_interviewers_with_paths()
    assert len(records) == 1
    assert records[0][0] == valid_path
    assert records[0][1].name == "Bob"


def test_list_all_interviewers_matches_valid_records(interviewers_dir) -> None:
    save_interviewer(InterviewerProfile(name="Ann", background="", is_generic_ai=False))
    save_interviewer(InterviewerProfile(name="Ben", background="", is_generic_ai=False))

    all_profiles = list_all_interviewers()
    names = {profile.name for _, profile in all_profiles}
    assert names == {"Ann", "Ben"}


def test_find_interviewer_file_with_fallback(monkeypatch, interviewers_dir) -> None:
    direct = InterviewerProfile(name="Direct Match", background="", is_generic_ai=False)
    save_interviewer(direct)

    fallback_profile = InterviewerProfile(name="Fallback Match", background="", is_generic_ai=False)
    fallback_path = interviewers_dir / "fallback-file.json"
    fallback_path.write_text(fallback_profile.model_dump_json(), encoding="utf-8")

    assert find_interviewer_file(direct) == interviewers_dir / "direct-match.json"
    assert find_interviewer_file(fallback_profile) == fallback_path
    assert (
        find_interviewer_file(
            InterviewerProfile(name="Missing Person", background="", is_generic_ai=False),
            allow_scope_fallback=False,
        )
        is None
    )
