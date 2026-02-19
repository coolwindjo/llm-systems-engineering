from __future__ import annotations

from typing import Callable, Dict

import streamlit as st
from openai import OpenAI
try:
    from openai import APIError
except (ImportError, ModuleNotFoundError):
    class APIError(Exception):
        """Fallback APIError when openai dependency is unavailable."""

from services.interview_ops import _build_opening_prompt, create_chat_completion_with_fallback


def start_interview_for_active_interviewer(
    *,
    interviewer_key: str,
    interviewer_label: str,
    jd_title: str,
    system_prompts: Dict[str, str],
    get_api_key: Callable[[], str | None],
) -> None:
    st.session_state.interview_started = True
    st.session_state.messages = []
    st.session_state.interview_started_interviewer = interviewer_label

    api_key = get_api_key()
    if not api_key:
        st.error("OPENAI_API_KEY is not set. Add it in Streamlit secrets or environment variables.")
        st.session_state.interview_started = False
        return

    if "chat_histories" not in st.session_state:
        st.session_state.chat_histories = {}

    if interviewer_key not in st.session_state.chat_histories:
        st.session_state.chat_histories[interviewer_key] = []
    history = st.session_state.chat_histories[interviewer_key]
    history.clear()
    st.session_state.chat_histories[interviewer_key] = history
    st.session_state.messages = history

    opening_prompt = _build_opening_prompt(interviewer_label, jd_title)
    system_prompt = system_prompts.get(interviewer_key, f"You are {interviewer_label}.")
    client = OpenAI(api_key=api_key)
    typing_placeholder = st.empty()
    typing_placeholder.info(f"{interviewer_label} is typing...")
    try:
        response, used_model = create_chat_completion_with_fallback(
            client=client,
            model=st.session_state.selected_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": opening_prompt},
            ],
            temperature=st.session_state.temperature,
        )
        assistant_reply = response.choices[0].message.content or "I could not generate an opening question."
        if used_model != st.session_state.selected_model:
            assistant_reply = f"[Model fallback: {used_model}]\n\n{assistant_reply}"
        history.append({"role": "assistant", "content": assistant_reply})
        st.session_state.chat_histories[interviewer_key] = history
        st.session_state.messages = history
        st.rerun()
    except (APIError, AttributeError, IndexError, KeyError, TypeError, ValueError) as exc:
        st.error(f"Error from OpenAI API: {exc}")
        st.session_state.interview_started = False
    finally:
        typing_placeholder.empty()
