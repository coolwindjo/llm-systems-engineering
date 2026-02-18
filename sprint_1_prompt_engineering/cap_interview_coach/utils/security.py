from __future__ import annotations

from typing import Tuple

MAX_INPUT_LENGTH = 2000
PROMPT_INJECTION_KEYWORDS = [
    "ignore previous instructions",
    "ignore all previous instructions",
    "system prompt",
]


def validate_input(text: str) -> Tuple[bool, str | None]:
    if len(text) > MAX_INPUT_LENGTH:
        return False, f"Input is too long. Please keep it under {MAX_INPUT_LENGTH} characters."

    normalized_text = text.lower()
    for keyword in PROMPT_INJECTION_KEYWORDS:
        if keyword in normalized_text:
            return False, "Input contains disallowed prompt-injection phrasing. Please rephrase your answer."

    return True, None
