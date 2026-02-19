from __future__ import annotations

from components.sidebar import (
    _parse_experience_block,
    _build_interviewer_profile_payload,
    _coerce_interviewer_form_list,
)


def test_coerce_interviewer_form_list_supports_newline_comma_and_semicolon_inputs() -> None:
    raw_input = "ADAS, ISO 26262\nC++, unit testing;C++"
    assert _coerce_interviewer_form_list(raw_input) == ["ADAS", "ISO 26262", "C++", "unit testing"]


def test_build_interviewer_profile_payload_normalizes_fields() -> None:
    profile = _build_interviewer_profile_payload(
        name="  Alex  ",
        role="  Recruiter  ",
        experience=(
            "Background:   background  \n"
            "Expertise: AI, C++\nC++ ; ML\n"
            "Potential Questions: Q1, Q2\nQ1"
        ),
    )

    assert profile is not None
    assert profile.name == "Alex"
    assert profile.background == "background"
    assert profile.role == "Recruiter"
    assert profile.expertise == ["AI", "C++", "ML"]
    assert profile.potential_questions == ["Q1", "Q2"]
    assert profile.is_generic_ai is False


def test_build_interviewer_profile_payload_requires_name() -> None:
    profile = _build_interviewer_profile_payload(
        name="   ",
        role="",
        experience="",
    )
    assert profile is None


def test_parse_experience_block_classifies_sections_by_label() -> None:
    draft = _parse_experience_block(
        "Background: worked on ADAS for 5 years.\n"
        "Expertise: C++, Safety, ISO 26262\n"
        "Potential Questions:\n"
        "How do you design diagnostics?\n"
        "How do you handle ASIL?\n"
    )
    assert draft.background == "worked on ADAS for 5 years."
    assert draft.expertise == ["C++", "Safety", "ISO 26262"]
    assert draft.potential_questions == [
        "How do you design diagnostics?",
        "How do you handle ASIL?",
    ]
