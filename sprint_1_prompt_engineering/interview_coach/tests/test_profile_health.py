from __future__ import annotations

import json
from pathlib import Path

from services import profile_health as ph
from utils.interviewer_store import InterviewerProfile


def test_read_json_file_validates_dict_payload(tmp_path) -> None:
    valid_path = tmp_path / "valid.json"
    valid_path.write_text('{"a": 1}', encoding="utf-8")
    assert ph._read_json_file(valid_path) == {"a": 1}

    missing = tmp_path / "missing.json"
    assert ph._read_json_file(missing) is None

    non_dict = tmp_path / "list.json"
    non_dict.write_text("[1,2]", encoding="utf-8")
    assert ph._read_json_file(non_dict) is None

    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{", encoding="utf-8")
    assert ph._read_json_file(bad_json) is None


def test_label_list_and_tokenization() -> None:
    assert ph._to_label_list(["A", "", "A", " B ", "b"]) == ["A", "B", "b"]

    tokens = ph._tokenize("ADAS Perception, C++ & safety")
    assert tokens == ["adas", "perception", "c++", "safety"]

    filtered = ph._to_set(tokens)
    assert "and" not in filtered
    assert "adas" in filtered
    assert "c++" in filtered


def test_extract_jd_requirements_prefers_key_requirements() -> None:
    profile = {
        "key_requirements": ["Req 1", "Req 2", ""],
        "job_positions": [{"key_requirements": ["From Position"]}],
    }
    assert ph._extract_jd_requirements(profile) == ["Req 1", "Req 2"]


def test_extract_jd_requirements_falls_back_to_positions() -> None:
    profile = {
        "job_positions": [
            {"title": "Senior", "key_requirements": ["Req 1", "Req 2"]},
            {"key_requirements": ["Req 3"]},
        ]
    }
    assert ph._extract_jd_requirements(profile) == ["Req 1", "Req 2", "Req 3"]


def test_collect_candidate_interviewers_merges_profile_and_custom(monkeypatch, tmp_path) -> None:
    default_payload = {
        "interviewers": [
            {"name": "Default A", "role": "Role", "expertise": ["A"], "background": "", "potential_questions": []},
            {"name": "Duplicate", "role": "Role", "expertise": [], "background": "", "potential_questions": []},
        ]
    }

    default_path = Path(ph.__file__).resolve().parent.parent / "data" / "interview_data.json"

    def read_json_file(path):
        if path == default_path:
            return default_payload
        return None

    monkeypatch.setattr(ph, "_read_json_file", read_json_file)

    custom = [
        InterviewerProfile(name="Custom A", background="", is_generic_ai=False),
        InterviewerProfile(name="Duplicate", background="", is_generic_ai=False),
    ]
    monkeypatch.setattr(
        "utils.interviewer_store.get_interviewers_by_jd",
        lambda jd_title: custom,
    )

    profile = {
        "interviewers": [
            {"name": "Local A", "role": "Role", "expertise": ["L"], "background": "", "potential_questions": []},
            {"name": "Duplicate", "role": "Role", "expertise": ["X"], "background": "", "potential_questions": []},
        ]
    }

    candidates = ph._collect_candidate_interviewers(profile, "JD")
    names = [candidate["name"] for candidate in candidates]
    assert names == ["Local A", "Duplicate", "Default A", "Custom A"]


def test_correlation_score_thresholds() -> None:
    profile = {
        "company": "Acme",
        "position": "ADAS Lead",
        "key_requirements": ["lidar", "safety"],
    }
    jd_text = ph._build_jd_profile_text(profile)
    candidate = ph._build_candidate_text(
        {
            "name": "Ada",
            "background": "Adas lead at ACME",
            "role": "Engineer",
            "expertise": ["Lidar"],
            "potential_questions": ["How do you ensure safety?"]
        }
    )

    both_match = ph._compute_correlation_score(jd_text, candidate, profile, {
        "name": "Ada",
        "text": candidate,
    })
    assert both_match >= ph.RECOMMENDATION_MIN_SCORE_BOTH_MATCH

    no_match = ph._compute_correlation_score("", "", profile, {"name": "X", "text": ""})
    assert no_match == 0.0


def test_recommend_interviewers_applies_threshold_and_sorts(monkeypatch) -> None:
    profile = {
        "interviewers": [
            {"name": "Alice", "background": "", "role": "", "expertise": [], "potential_questions": []},
            {"name": "Bob", "background": "", "role": "", "expertise": [], "potential_questions": []},
        ]
    }

    monkeypatch.setattr(
        ph,
        "_collect_candidate_interviewers",
        lambda profile_data, jd_title: [
            {"name": "Bob", "text": "", "source": "profile", "role": "", "expertise": [], "potential_questions": [], "background": ""},
            {"name": "Alice", "text": "", "source": "profile", "role": "", "expertise": [], "potential_questions": [], "background": ""},
            {"name": "Charlie", "text": "", "source": "profile", "role": "", "expertise": [], "potential_questions": [], "background": ""},
        ],
    )
    monkeypatch.setattr(
        ph,
        "_compute_correlation_score",
        lambda jd_text, candidate_text, profile_data, candidate: {
            "Alice": 90.0,
            "Bob": 10.0,
            "Charlie": 40.0,
        }[candidate["name"]],
    )

    recommendations = ph.recommend_interviewers(profile, "JD")
    assert recommendations == [
        {"name": "Alice", "raw_name": "Alice", "score": 90.0, "recommended": True, "source": "profile"},
        {"name": "Charlie", "raw_name": "Charlie", "score": 40.0, "recommended": True, "source": "profile"},
        {"name": "Bob", "raw_name": "Bob", "score": 10.0, "recommended": False, "source": "profile"},
    ]


def test_audit_interview_profiles_reports_statuses(tmp_path, monkeypatch) -> None:
    data_dir = tmp_path
    default_profile = data_dir / "interview_data.json"
    default_profile.write_text(
        json.dumps({"interviewers": [], "session_interviewers": []}, ensure_ascii=False),
        encoding="utf-8",
    )

    profiles_dir = data_dir / "profiles"
    profiles_dir.mkdir()

    valid_profile = profiles_dir / "senior.json"
    valid_profile.write_text(
        json.dumps({"position": "Senior ADAS", "session_interviewers": ["Generic AI Interviewer"]}),
        encoding="utf-8",
    )

    invalid_profile = profiles_dir / "broken.json"
    invalid_profile.write_text("[]", encoding="utf-8")

    monkeypatch.setattr(ph, "_recommend_interviewers", lambda profile_data, jd_title: [])
    monkeypatch.setattr(
        "utils.interviewer_store.get_interviewers_by_jd",
        lambda jd_title: [
            InterviewerProfile(name="Generic AI Interviewer", background="", is_generic_ai=True),
        ],
    )

    report = ph.audit_interview_profiles(data_dir)
    assert len(report) == 3
    file_to_status = {item["file"]: item["status"] for item in report}
    assert file_to_status["interview_data.json"] == "missing_required"
    assert file_to_status["senior.json"] == "ok"
    assert file_to_status["broken.json"] == "invalid_file"
