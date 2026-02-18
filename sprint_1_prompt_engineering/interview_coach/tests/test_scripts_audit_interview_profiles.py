from __future__ import annotations

import sys

from scripts import audit_interview_profiles as audit_mod


def test_print_summary_prints_expected_lines(capsys) -> None:
    report = [
        {
            "file": "interview_data.json",
            "status": "ok",
            "missing_required": [],
            "recommendations": [
                {"name": "Alice", "recommended": True, "score": 87.2},
                {"name": "Bob", "recommended": False, "score": 22.1},
            ],
        },
        {
            "file": "broken.json",
            "status": "invalid_file",
            "missing_required": ["Generic AI Interviewer"],
            "recommendations": [],
        },
    ]

    audit_mod._print_summary(report)
    output = capsys.readouterr().out

    assert "Scanned profiles: 2" in output
    assert "Healthy: 1" in output
    assert "Needs attention: 1" in output
    assert "OK interview_data.json (ok)" in output
    assert "WARN broken.json (invalid_file)" in output
    assert "missing_required: Generic AI Interviewer" in output
    assert "recommended: Alice" in output


def test_main_default_returns_success_when_healthy(monkeypatch, capsys) -> None:
    monkeypatch.setattr(audit_mod, "audit_interview_profiles", lambda _: [
        {
            "file": "interview_data.json",
            "status": "ok",
            "missing_required": [],
            "recommendations": [],
        }
    ])
    monkeypatch.setattr(sys, "argv", ["audit_interview_profiles.py"])

    assert audit_mod.main() == 0
    assert "Scanned profiles" in capsys.readouterr().out


def test_main_json_mode_prints_payload_and_returns_error(monkeypatch) -> None:
    monkeypatch.setattr(audit_mod, "audit_interview_profiles", lambda _: [
        {
            "file": "broken.json",
            "status": "missing_required",
            "missing_required": ["Generic AI Interviewer"],
            "recommendations": [],
        }
    ])
    monkeypatch.setattr(sys, "argv", ["audit_interview_profiles.py", "--json"])

    result = audit_mod.main()
    assert result == 1
