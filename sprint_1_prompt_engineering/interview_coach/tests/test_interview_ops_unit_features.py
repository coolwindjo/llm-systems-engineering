from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from services import interview_ops as ops
from utils.interviewer_store import InterviewerProfile


def test_coerce_temperature_for_model_applies_rule_and_clamps(monkeypatch) -> None:
    monkeypatch.setattr(
        ops,
        "TEMPERATURE_RULES",
        {
            "default": {"min": 0.0, "max": 1.0, "fallback": 0.7},
            "rules": [
                {
                    "patterns": ["gpt-4o-mini"],
                    "match_mode": "equals",
                    "min": 0.1,
                    "max": 0.9,
                    "temperature": 0.3,
                }
            ],
        },
    )

    assert ops._coerce_temperature_for_model("gpt-4o-mini", 0.9) == 0.3
    assert ops._coerce_temperature_for_model("other-model", 2.0) == 1.0
    assert ops._coerce_temperature_for_model("other-model", "bad") == 0.7


def test_resolve_critique_profile_style_candidates_uses_generic_and_explicit_fields() -> None:
    assert ops._resolve_critique_profile_style_candidates({"is_generic_ai": True}) == ["technical_general"]

    assert ops._resolve_critique_profile_style_candidates(
        {"critique_style": "  AI/ML, Safety  "}
    ) == ["ai_ml_safety"]

    assert ops._resolve_critique_profile_style_candidates(
        {
            "critique_profile": {"style": ["safety_management", "integration_testability"]},
        }
    ) == ["safety_management", "integration_testability"]


def test_collect_critique_template_keys_from_position_and_company() -> None:
    position_keys, company_keys = ops._collect_critique_template_keys(
        {"position": "Senior ADAS Developer", "company": "Capgemini Engineering"},
        selected_jd_title="ADAS Lead",
    )

    assert "senior_adas_developer" in position_keys
    assert "adas" in position_keys
    assert "capgemini_engineering" in company_keys
    assert "capgemini" in company_keys


def test_normalize_text_list_strips_and_deduplicates() -> None:
    assert ops._normalize_text_list(["  C++  ", "c++", "C++", "Python", ""]) == ["C++", "Python"]


def test_build_session_interviewer_profiles_deduplicates_and_includes_generic(monkeypatch) -> None:
    base_data = {
        "interviewers": [
            {"name": "Alice", "background": "ADAS", "role": "Lead", "expertise": [], "potential_questions": []},
            {"name": "Generic AI Interviewer", "background": "Recruiter", "role": "AI", "expertise": [], "potential_questions": []},
            {"name": "alice", "background": "dup", "role": "Lead", "expertise": [], "potential_questions": []},
        ]
    }

    def fake_custom_interviewers(_: str) -> list[InterviewerProfile]:
        return [
            InterviewerProfile(
                name="Alice",
                background="dup",
                is_generic_ai=False,
                role="",
                expertise=[],
                potential_questions=[],
            ),
            InterviewerProfile(
                name="Bob",
                background="Perception",
                is_generic_ai=False,
                role="",
                expertise=[],
                potential_questions=[],
            ),
        ]

    monkeypatch.setattr(ops, "get_interviewers_by_jd", fake_custom_interviewers)

    profiles = ops._build_session_interviewer_profiles(base_data, "ADAS Profile")
    keys = [ops._speaker_key(profile.get("name", "")) for profile in profiles]

    assert keys == ["alice", "generic_ai_interviewer", "bob"]


def test_build_speaker_options_from_profiles_deduplicates() -> None:
    profiles = [
        {"name": "Alice", "is_generic_ai": False},
        {"name": "Alice", "is_generic_ai": False},
        {"name": "Bob", "is_generic_ai": False},
        {"name": "Generic AI Interviewer", "is_generic_ai": True},
    ]

    labels, mapping = ops._build_speaker_options_from_profiles(profiles)

    assert labels == [ops.GENERIC_INTERVIEWER_LABEL, "Alice", "Bob"]
    assert mapping[ops.GENERIC_INTERVIEWER_LABEL] == ops.GENERIC_INTERVIEWER_KEY
    assert mapping["Alice"] == "alice"
    assert mapping["Bob"] == "bob"


def test_filter_session_interviewers_preserves_mandatory_when_missing() -> None:
    available = ["Alice", "Bob", ops.GENERIC_INTERVIEWER_LABEL]
    mandatory = [ops.GENERIC_INTERVIEWER_LABEL]

    assert ops._filter_session_interviewers([], available, mandatory) == [ops.GENERIC_INTERVIEWER_LABEL]
    assert ops._filter_session_interviewers(["Alice"], available, mandatory) == ["Alice", ops.GENERIC_INTERVIEWER_LABEL]
    assert ops._filter_session_interviewers([], available, mandatory, enforce_mandatory=False) == [
        ops.GENERIC_INTERVIEWER_LABEL,
    ]


def test_persist_session_interviewers_stores_without_candidate_profile(tmp_path: Path) -> None:
    profile_path = tmp_path / "profile.json"
    source = {
        "name": "ADAS Role",
        "candidate_profile": {"name": "Alice"},
        "session_interviewers": ["Alice"],
    }

    persisted = ops._persist_session_interviewers(profile_path, source, ["A", "B"])
    assert persisted is True

    written = Path(profile_path)
    assert written.exists()
    payload = json.loads(written.read_text(encoding="utf-8"))
    assert payload == {
        "name": "ADAS Role",
        "session_interviewers": ["A", "B"],
    }


def test_create_chat_completion_with_fallback_uses_requested_model_when_available() -> None:
    class FakeCompletions:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def create(self, *, model: str, messages: list[dict[str, str]], temperature: float) -> SimpleNamespace:
            self.calls.append(model)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
            )

    completions = FakeCompletions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

    response, used_model = ops.create_chat_completion_with_fallback(
        client=client,
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": "hi"}],
        temperature=0.2,
    )

    assert used_model == "gpt-4.1-mini"
    assert response.choices[0].message.content == "ok"
    assert completions.calls == ["gpt-4.1-mini"]


def test_create_chat_completion_with_fallback_uses_fallback_on_access_error() -> None:
    class AccessDeniedError(RuntimeError):
        pass

    class FakeCompletions:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def create(self, *, model: str, messages: list[dict[str, str]], temperature: float) -> SimpleNamespace:
            self.calls.append(model)
            if len(self.calls) == 1:
                raise AccessDeniedError("Model does not have access")
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="fallback ok"))],
            )

    completions = FakeCompletions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

    response, used_model = ops.create_chat_completion_with_fallback(
        client=client,
        model="unknown-model",
        messages=[{"role": "user", "content": "test"}],
        temperature=0.2,
    )

    assert response.choices[0].message.content == "fallback ok"
    assert used_model == ops.CHAT_FALLBACK_ORDER[0]
    assert completions.calls[0] == "unknown-model"


def test_get_prompt_template_loads_and_formats_json_context(tmp_path: Path) -> None:
    template_path = tmp_path / "prompts.json"
    template_path.write_text(
        '{"greeting": "Hello {name} from {company}"}',
        encoding="utf-8",
    )

    rendered = ops.get_prompt_template(
        "greeting",
        fallback="Hello candidate",
        path=template_path,
        context={"name": "Alice", "company": "Capgemini"},
    )
    assert rendered == "Hello Alice from Capgemini"
