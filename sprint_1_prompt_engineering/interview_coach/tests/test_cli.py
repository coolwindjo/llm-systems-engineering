from __future__ import annotations

from pathlib import Path

from types import SimpleNamespace
from typing import Any

import cli


def test_extract_jd_command_writes_profile(tmp_path, monkeypatch) -> None:
    captured = {}

    def fake_extract_jd_fields(api_key: str, jd_text: str, company: str, position: str):
        captured["api_key"] = api_key
        captured["company"] = company
        captured["position"] = position
        captured["jd_text"] = jd_text
        return SimpleNamespace(company="Acme", position="Engineer"), "gpt-4o-mini"

    monkeypatch.setattr(cli, "extract_jd_fields", fake_extract_jd_fields)
    monkeypatch.setattr(cli, "build_profile_filename", lambda company, position: "acme-engineer.json")
    monkeypatch.setattr(cli, "build_interview_profile", lambda extracted: {"generated": "ok"})

    output_path = tmp_path / "interview.json"
    result = cli.main(
        [
            "--api-key",
            "test-key",
            "extract-jd",
            "--company",
            "Acme",
            "--position",
            "Engineer",
            "--jd-text",
            "Role text",
            "--output",
            str(output_path),
        ]
    )

    assert result == 0
    assert output_path.exists()
    assert captured["api_key"] == "test-key"
    assert output_path.read_text(encoding="utf-8") == '{\n  "generated": "ok",\n  "_model": "gpt-4o-mini"\n}'


def test_parse_interviewee_command_merges_cover_letter(tmp_path, monkeypatch) -> None:
    base_profile = {"name": "Dana", "core_strengths": ["ADAS"]}

    def fake_load_interviewee_profile() -> dict[str, Any]:
        return base_profile

    monkeypatch.setattr(cli, "_load_interviewee_profile", fake_load_interviewee_profile)
    monkeypatch.setattr(
        cli,
        "parse_interviewee_profile",
        lambda *args, **kwargs: SimpleNamespace(
            model_dump=lambda: {"name": "Dana", "status": "Experienced", "core_strengths": ["Safety"], "focus_phrases": ["Proof"]},
            name="Dana",
            status="Experienced",
            core_strengths=["Safety"],
            focus_phrases=["Proof"],
        ),
    )

    output_path = tmp_path / "interviewee.json"
    result = cli.main(
        [
            "--api-key",
            "test-key",
            "parse-interviewee",
            "--source",
            "cover_letter",
            "--input-text",
            "cover letter content",
            "--output",
            str(output_path),
        ]
    )
    assert result == 0
    payload = output_path.read_text(encoding="utf-8")
    assert '"name": "Dana"' in payload
    assert "Safety" in payload


def test_parse_interviewer_command_save(tmp_path, monkeypatch) -> None:
    captured = {}

    class FakeParsed:
        name = "Inter"
        background = "bg"
        role = "Role"
        expertise = ["A"]
        potential_questions = ["Q"]

        def model_dump(self):
            return {"name": "Inter", "background": "bg", "role": "Role", "expertise": ["A"], "potential_questions": ["Q"]}

    def fake_parse(*args, **kwargs) -> FakeParsed:
        return FakeParsed()

    def fake_save(profile):
        captured["profile"] = profile
        saved_path = tmp_path / "interviewer.json"
        saved_path.write_text("{}", encoding="utf-8")
        return saved_path

    monkeypatch.setattr(cli, "parse_interviewer_background", fake_parse)
    monkeypatch.setattr(cli, "save_interviewer", fake_save)

    result = cli.main(
        [
            "--api-key",
            "test-key",
            "parse-interviewer",
            "--name",
            "Inter",
            "--input-text",
            "some background",
            "--save",
        ]
    )

    assert result == 0
    assert captured["profile"].name == "Inter"


def test_delete_interviewer_command(tmp_path, monkeypatch, capsys) -> None:
    called = {"path": None}

    def fake_delete(path: str) -> None:
        called["path"] = path

    monkeypatch.setattr(cli, "delete_interviewer", fake_delete)
    code = cli.main(["delete-interviewer", str(tmp_path / "p.json")])
    assert code == 0
    assert called["path"] == str(tmp_path / "p.json")
