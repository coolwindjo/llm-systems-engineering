from __future__ import annotations

import json
from pathlib import Path

from services.interview_ops import DEFAULT_CRITIQUE_PERSONA, get_critique_persona_prompt


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


def test_get_critique_persona_prompt_fallback_when_template_missing(tmp_path: Path) -> None:
    text = get_critique_persona_prompt(
        interviewer_name="Jordan",
        path=tmp_path / "does_not_exist.json",
    )
    assert text == DEFAULT_CRITIQUE_PERSONA
