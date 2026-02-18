from __future__ import annotations

from utils.security import MAX_INPUT_LENGTH, PROMPT_INJECTION_KEYWORDS, validate_input


def test_validate_input_accepts_clean_text() -> None:
    ok, error = validate_input("Hello, I am preparing for this interview.")
    assert ok
    assert error is None


def test_validate_input_rejects_overly_long_text() -> None:
    too_long = "a" * (MAX_INPUT_LENGTH + 1)
    ok, error = validate_input(too_long)
    assert not ok
    assert error == (
        f"Input is too long. Please keep it under {MAX_INPUT_LENGTH} characters."
    )


def test_validate_input_rejects_prompt_injection_phrases() -> None:
    for keyword in PROMPT_INJECTION_KEYWORDS:
        ok, error = validate_input(f"Could you {keyword} and ignore the rules?")
        assert not ok
        assert error == "Input contains disallowed prompt-injection phrasing. Please rephrase your answer."
