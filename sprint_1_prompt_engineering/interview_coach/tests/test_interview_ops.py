from __future__ import annotations

import json
from pathlib import Path

from services.interview_ops import DEFAULT_CRITIQUE_PERSONA, get_critique_persona_prompt
from services.interview_ops import _speaker_key


def test_get_critique_persona_prompt_from_template(tmp_path: Path) -> None:
    template_path = tmp_path / "system_prompts.json"
    template_path.write_text(
        json.dumps(
            {
                "critique_persona": "Critique for {interviewer_name}: evaluate answer quality strictly.",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    text = get_critique_persona_prompt(interviewer_name="Jordan", path=template_path)
    assert "Critique for Jordan" in text
    assert text == "Critique for Jordan: evaluate answer quality strictly."


def test_get_critique_persona_prompt_selects_interviewer_technique_preference(tmp_path: Path) -> None:
    template_path = tmp_path / "system_prompts.json"
    template_path.write_text(
        json.dumps(
            {
                "critique_persona": {
                    "default": "default {interviewer_name} {technique} {interviewer_key}",
                    "by_technique": {"chain_of_thought": "technique {interviewer_name}"},
                    "by_interviewer": {_speaker_key("Generic AI Interviewer"): "interviewer {interviewer_name}"},
                    "by_combo": {"generic_ai_interviewer.chain_of_thought": "combo {interviewer_name}"},
                }
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    combo = get_critique_persona_prompt(
        interviewer_name="Jordan",
        interviewer_key="Generic AI Interviewer",
        technique="chain_of_thought",
        path=template_path,
    )
    assert combo == "combo Jordan"

    by_interviewer = get_critique_persona_prompt(
        interviewer_name="Jordan",
        interviewer_key="Generic AI Interviewer",
        technique="few_shot",
        path=template_path,
    )
    assert by_interviewer == "interviewer Jordan"

    by_technique = get_critique_persona_prompt(
        interviewer_name="Jordan",
        interviewer_key="Other",
        technique="chain_of_thought",
        path=template_path,
    )
    assert by_technique == "technique Jordan"

    default = get_critique_persona_prompt(
        interviewer_name="Jordan",
        interviewer_key="Other",
        technique="zero_shot",
        path=template_path,
    )
    assert default == "default Jordan zero_shot other"


def test_get_critique_persona_prompt_fallback_when_template_missing(tmp_path: Path) -> None:
    text = get_critique_persona_prompt(
        interviewer_name="Jordan",
        path=tmp_path / "does_not_exist.json",
    )
    assert text == DEFAULT_CRITIQUE_PERSONA
