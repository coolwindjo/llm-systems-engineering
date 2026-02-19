from __future__ import annotations

import re
from typing import Any, Callable, Dict, List

import streamlit as st
from openai import OpenAI
try:
    from openai import APIError
except (ImportError, ModuleNotFoundError):
    class APIError(Exception):
        """Fallback APIError when openai dependency is unavailable."""
from streamlit_chat import message

from services.interview_ops import (
    _build_support_prompt,
    create_chat_completion_with_fallback,
    get_critique_persona_prompt,
    get_feedback_highlight_catalog,
    get_last_assistant_message,
    get_last_user_response,
)
from utils.security import validate_input

try:
    from annotated_text import annotated_text
except ImportError:
    annotated_text = None


def render_feedback_with_adas_terms(feedback: str, profile_data: Dict[str, Any]) -> None:
    if not feedback:
        return

    if annotated_text is None:
        st.markdown(feedback)
        st.info("Install `streamlit-annotated-text` to enable in-line ADAS term highlighting.")
        return

    catalog = get_feedback_highlight_catalog(profile_data)
    term_to_label: Dict[str, str] = {}
    ordered_terms: List[tuple[str, str]] = []

    for label, terms in catalog.items():
        for term in terms:
            normalized_term = term.strip()
            if not normalized_term:
                continue
            ordered_terms.append((normalized_term, label))
            term_to_label[normalized_term.lower()] = label

    if not ordered_terms:
        st.markdown(feedback)
        return

    sorted_terms = sorted(ordered_terms, key=lambda item: len(item[0]), reverse=True)
    pattern = re.compile("|".join(re.escape(term) for term, _ in sorted_terms), flags=re.IGNORECASE)
    parts: List[str | tuple] = []
    cursor = 0
    for match in pattern.finditer(feedback):
        if match.start() > cursor:
            parts.append(feedback[cursor : match.start()])
        label = term_to_label.get(match.group(0).lower(), "JD Term")
        parts.append((match.group(0), label))
        cursor = match.end()
    if cursor < len(feedback):
        parts.append(feedback[cursor:])

    if parts:
        annotated_text(*parts)


def _safe_model_reply(
    api_key: str,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    fallback_prefix: str,
) -> str:
    client = OpenAI(api_key=api_key)
    try:
        response, used_model = create_chat_completion_with_fallback(
            client=client,
            model=model,
            messages=messages,
            temperature=temperature,
        )
        reply = response.choices[0].message.content or "No response."
        if used_model != model:
            reply = f"[Model fallback: {used_model}]\\n\\n{reply}"
        return reply
    except (APIError, AttributeError, IndexError, KeyError, TypeError, ValueError) as exc:
        return f"{fallback_prefix}: {exc}"


def _render_user_and_assistant_reply(
    current_interviewer: str,
    api_key: str,
    history: List[Dict[str, str]],
    system_prompts: Dict[str, str],
    state_key: str,
) -> None:
    typing_placeholder = st.empty()
    typing_placeholder.info(f"{current_interviewer.title()} is typing...")
    try:
        assistant_reply = _safe_model_reply(
            api_key=api_key,
            model=st.session_state.selected_model,
            messages=[{"role": "system", "content": system_prompts[current_interviewer]}, *history],
            temperature=st.session_state.temperature,
            fallback_prefix="Error from OpenAI API",
        )
    finally:
        typing_placeholder.empty()

    history.append({"role": "assistant", "content": assistant_reply})
    st.session_state.chat_histories[state_key] = history
    st.rerun()


def _build_critique_message(
    current_interviewer: str,
    last_user_answer: str,
) -> str:
    return (
        f"Interviewer: {current_interviewer.title()}\\n"
        f"Candidate answer:\\n{last_user_answer}\\n\\n"
        "Evaluate this answer for concrete, evidence-based technical quality."
    )


def _run_critique_analysis(
    current_interviewer: str,
    current_interviewer_key: str,
    interviewer_profile: Dict[str, Any] | None,
    technique_key: str,
    jd_profile: Dict[str, Any],
    jd_title: str | None,
    last_user_answer: str,
    api_key: str,
) -> str:
    critique_persona = get_critique_persona_prompt(
        interviewer_name=current_interviewer,
        interviewer_key=current_interviewer_key,
        interviewer_profile=interviewer_profile,
        technique=technique_key,
        jd_profile=jd_profile,
        jd_title=jd_title,
    )
    return _safe_model_reply(
        api_key=api_key,
        model=st.session_state.selected_model,
        messages=[
            {"role": "system", "content": critique_persona},
            {"role": "user", "content": _build_critique_message(current_interviewer, last_user_answer)},
        ],
        temperature=max(0.1, st.session_state.temperature - 0.2),
        fallback_prefix="Error from OpenAI API",
    )


def _build_support_and_model(
    *,
    current_interviewer: str,
    last_question: str,
    support_type: str,
    api_key: str,
) -> str:
    prompt = _build_support_prompt(current_interviewer.title(), last_question, support_type)
    fallback = "Failed to generate hint" if support_type == "hint" else "No model answer generated"
    temperature = max(0.1, st.session_state.temperature - 0.1) if support_type == "hint" else st.session_state.temperature
    return _safe_model_reply(
        api_key=api_key,
        model=st.session_state.selected_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        fallback_prefix=fallback,
    )


def render_chat_panel(
    *,
    panel_title: str,
    current_interviewer: str,
    history: List[Dict[str, str]],
    system_prompts: Dict[str, str],
    interviewer_name: str,
    technique_key: str,
    jd_profile: Dict[str, Any],
    get_api_key: Callable[[], str | None],
    interviewer_profile: Dict[str, Any] | None = None,
) -> None:
    st.subheader(panel_title)
    if not st.session_state.get("interview_started", False):
        st.info("Press '🚀 Start Interview' in the sidebar to let the interviewer begin.")
        return

    for idx, msg in enumerate(history):
        message(msg["content"], is_user=msg["role"] == "user", key=f"{current_interviewer}-{idx}")

    user_input = st.chat_input("Type your interview answer or ask a question...")
    if user_input:
        is_valid, validation_error = validate_input(user_input)
        if not is_valid:
            st.error(validation_error or "Invalid input.")
        else:
            history.append({"role": "user", "content": user_input})
            message(user_input, is_user=True, key=f"{current_interviewer}-user-live")

            api_key = get_api_key()
            if not api_key:
                st.error("OPENAI_API_KEY is not set. Add it in Streamlit secrets or environment variables.")
            else:
                _render_user_and_assistant_reply(
                    current_interviewer=current_interviewer,
                    api_key=api_key,
                    history=history,
                    system_prompts=system_prompts,
                    state_key=current_interviewer,
                )

    analyze_key = f"critique_feedback_{current_interviewer}"
    hint_key = f"interview_hint_{current_interviewer}"
    answer_key = f"model_answer_{current_interviewer}"

    if st.button("Analyze My Answer", use_container_width=True):
        last_user_answer = get_last_user_response(history)
        if not last_user_answer:
            st.warning("No user answer found yet. Submit an answer first.")
        else:
            is_valid, validation_error = validate_input(last_user_answer)
            if not is_valid:
                st.error(validation_error or "Invalid input.")
            else:
                api_key = get_api_key()
                if not api_key:
                    st.error("OPENAI_API_KEY is not set. Add it in Streamlit secrets or environment variables.")
                else:
                    st.session_state[analyze_key] = _run_critique_analysis(
                        current_interviewer=interviewer_name,
                        current_interviewer_key=current_interviewer,
                        interviewer_profile=interviewer_profile,
                        technique_key=technique_key,
                        jd_profile=jd_profile,
                        jd_title=st.session_state.get("selected_jd"),
                        last_user_answer=last_user_answer,
                        api_key=api_key,
                    )

    last_question = get_last_assistant_message(history)
    st.divider()
    col_hint, col_model = st.columns(2)
    with col_hint:
        if st.button("Need a Hint", use_container_width=True, key=f"hint_btn_{current_interviewer}"):
            if not last_question:
                st.session_state[hint_key] = "Please ask a question first, then request a hint."
            else:
                api_key = get_api_key()
                if not api_key:
                    st.session_state[hint_key] = "OPENAI_API_KEY is not set. Add it in Streamlit secrets or environment variables."
                else:
                    st.session_state[hint_key] = _build_support_and_model(
                        current_interviewer=current_interviewer,
                        last_question=last_question,
                        support_type="hint",
                        api_key=api_key,
                    )

    with col_model:
        if st.button("Show Model Answer", use_container_width=True, key=f"model_answer_btn_{current_interviewer}"):
            if not last_question:
                st.session_state[answer_key] = "Please ask a question first, then request a model answer."
            else:
                api_key = get_api_key()
                if not api_key:
                    st.session_state[answer_key] = "OPENAI_API_KEY is not set. Add it in Streamlit secrets or environment variables."
                else:
                    st.session_state[answer_key] = _build_support_and_model(
                        current_interviewer=current_interviewer,
                        last_question=last_question,
                        support_type="sample",
                        api_key=api_key,
                    )

    feedback = st.session_state.get(analyze_key)
    if feedback:
        st.subheader("Critique Feedback")
        render_feedback_with_adas_terms(feedback, st.session_state.current_interview_data)

    hint = st.session_state.get(hint_key)
    if hint:
        st.subheader("Hint")
        st.write(hint)

    model_answer = st.session_state.get(answer_key)
    if model_answer:
        st.subheader("Model Answer")
        st.write(model_answer)
