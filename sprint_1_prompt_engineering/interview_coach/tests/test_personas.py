from __future__ import annotations

from utils.personas import (
    _append_jd_context_and_probe_instructions,
    _candidate_profile,
    _candidate_probe_phrases,
    _interviewer_key,
    build_system_prompts,
    build_generic_ai_prompt,
    _get_candidate_highlights,
)


def test_get_candidate_highlights_prefers_core_strengths() -> None:
    payload = {
        "core_strengths": ["Safety engineering", "Embedded C++", ""],
        "highlights": ["Backup"],
    }
    highlights = _get_candidate_highlights(payload)

    assert highlights["experience"] == "Safety engineering"
    assert highlights["mpc"] == "Embedded C++"
    assert highlights["dma"] == "Embedded C++"


def test_candidate_profile_returns_default_when_not_dict() -> None:
    assert _candidate_profile(None) == {}
    assert _candidate_profile([1, 2]) == {}


def test_candidate_probe_phrases_from_profile_or_fallback() -> None:
    custom = {"probes": {"focus_phrases": ["A", "", "B"]}}
    assert _candidate_probe_phrases(custom) == ["A", "B"]

    fallback = {"core_strengths": ["S"]}
    phrases = _candidate_probe_phrases(fallback)
    assert len(phrases) == 2
    assert all(isinstance(item, str) and item for item in phrases)


def test_interviewer_key_normalizes_generic_and_duplicates() -> None:
    assert _interviewer_key("Generic AI Interviewer") == "generic_ai_interviewer"
    assert _interviewer_key("Senior AI Reviewer") == "senior_ai_reviewer"
    assert _interviewer_key("   ") == "interviewer"


def test_build_system_prompts_includes_fallback_generic_interviewer() -> None:
    data = {
        "candidate_profile": {"name": "Iris", "core_strengths": ["ADAS"], "probes": {"focus_phrases": ["focus phrase"]}},
        "interviewers": [
            {
                "name": "Alice Johnson",
                "background": "Perception",
                "role": "Lead",
                "expertise": ["C++"],
                "potential_questions": ["How did you"],
                "is_generic_ai": False,
            }
        ],
    }

    prompts = build_system_prompts(data, technique="chain_of_thought")

    assert "alice_johnson" in prompts
    assert "generic_ai_interviewer" in prompts
    assert "Context for this specific role:" in prompts["alice_johnson"]
    assert "focus phrase" in prompts["alice_johnson"]
    assert "Interviewing flow" in prompts["alice_johnson"]
    assert "Generic AI Interviewer" in prompts["generic_ai_interviewer"]


def test_build_system_prompts_deduplicates_by_normalized_name() -> None:
    data = {
        "interviewers": [
            {"name": "Alice", "background": "", "expertise": [], "potential_questions": []},
            {"name": "  alice  ", "background": "", "expertise": [], "potential_questions": []},
        ]
    }

    prompts = build_system_prompts(data, technique="few_shot")
    assert list(prompts.keys()) == ["alice", "generic_ai_interviewer"]


def test_append_context_includes_technique_directive_and_candidate_name() -> None:
    data = {
        "candidate_profile": {
            "name": "Jordan",
            "highlights": ["A strong engineer"],
            "core_strengths": ["Signal processing"],
            "probes": {"focus_phrases": ["Ask for proof"]},
        },
    }

    prompt = build_generic_ai_prompt(data)
    assert "Jordan" in _append_jd_context_and_probe_instructions(prompt, data, "Generic AI Interviewer", True)

    zero_shot_prompt = _append_jd_context_and_probe_instructions(prompt, data, "Security Lead", False)
    assert "Apply your professional perspective" in zero_shot_prompt
    assert "Ask for proof" in zero_shot_prompt
    assert "JD-Specific Probing Directive" in zero_shot_prompt
