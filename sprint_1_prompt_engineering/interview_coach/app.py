from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List

import streamlit as st
try:
    from annotated_text import annotated_text
except ImportError:
    annotated_text = None
from openai import OpenAI
from streamlit.errors import StreamlitSecretNotFoundError
from streamlit_ace import st_ace
from streamlit_chat import message

from services.interview_ops import (
    DEFAULT_PROFILE_LABEL as AUDIT_DEFAULT_PROFILE_LABEL,
    CHAT_CAPABLE_MODELS,
    DATA_DIR,
    GENERIC_INTERVIEWER_KEY,
    GENERIC_INTERVIEWER_LABEL,
    INTERVIEWEE_PROFILE_PATH,
    JD_PROFILE_SESSION_KEY,
    PROFILES_DIR,
    _attach_interviewee_profile,
    _build_opening_prompt,
    _build_session_interviewer_profiles,
    _build_speaker_options_from_profiles,
    _build_default_session_interviewers,
    _coerce_temperature_for_model,
    _collect_interviewer_names,
    _collect_interviewer_records,
    _filter_session_interviewers,
    _extract_text_from_pdf,
    _load_interviewee_profile,
    _as_interviewee_payload,
    _merge_interviewee_cover_letter,
    _normalize_label_list,
    _save_interviewee_profile,
    _bootstrap_default_interviewers,
    build_interview_profile,
    build_profile_filename,
    create_chat_completion_with_fallback,
    extract_jd_fields,
    get_feedback_highlight_catalog,
    get_last_assistant_message,
    get_last_user_response,
    hash_file_contents,
    load_local_env,
    parse_interviewee_profile,
    get_critique_persona_prompt,
    _build_support_prompt,
)
from utils.data_loader import load_interview_data
from utils.personas import build_system_prompts
from utils.security import validate_input
from utils.profile_health import audit_interview_profiles


load_local_env()

def get_api_key() -> str | None:
    try:
        secret_key = st.secrets.get("OPENAI_API_KEY")
    except StreamlitSecretNotFoundError:
        secret_key = None
    return secret_key or os.getenv("OPENAI_API_KEY")


APP_TITLE = "AI Interview Coach"
st.set_page_config(page_title=APP_TITLE, page_icon="🎯", layout="wide")
st.sidebar.title(APP_TITLE)


def _sync_temperature_to_model_constraint() -> None:
    current_temperature = st.session_state.get("temperature", 0.4)
    st.session_state["temperature"] = _coerce_temperature_for_model(
        st.session_state.get("selected_model", ""),
        current_temperature,
    )


PROMPT_TECHNIQUES = {
    "Zero-Shot": "zero_shot",
    "Few-Shot": "few_shot",
    "Chain-of-Thought": "chain_of_thought",
    "Persona-Conditioning": "persona_conditioning",
    "Knowledge-Paucity (ISO 26262 Focus)": "knowledge_paucity",
}
if "selected_technique" not in st.session_state:
    st.session_state.selected_technique = "Zero-Shot"

st.sidebar.selectbox(
    "Prompting Technique",
    list(PROMPT_TECHNIQUES.keys()),
    key="selected_technique",
)

if "selected_model" not in st.session_state:
    st.session_state.selected_model = "gpt-4o-mini"
if st.session_state.selected_model not in CHAT_CAPABLE_MODELS:
    st.session_state.selected_model = "gpt-4o-mini"
if "temperature" not in st.session_state:
    st.session_state.temperature = 0.4
if "interview_started" not in st.session_state:
    st.session_state.interview_started = False
if "messages" not in st.session_state:
    st.session_state.messages = []

st.sidebar.selectbox(
    "OpenAI Model",
    CHAT_CAPABLE_MODELS,
    key="selected_model",
    on_change=_sync_temperature_to_model_constraint,
)
st.sidebar.slider(
    "Temperature",
    min_value=0.0,
    max_value=1.0,
    value=st.session_state.temperature,
    step=0.1,
    key="temperature",
    on_change=_sync_temperature_to_model_constraint,
)

DEFAULT_PROFILE_LABEL = AUDIT_DEFAULT_PROFILE_LABEL
profile_files = sorted([path.name for path in PROFILES_DIR.glob("*.json")]) if PROFILES_DIR.exists() else []
profile_options = [DEFAULT_PROFILE_LABEL, *profile_files]
if "selected_jd_profile" not in st.session_state:
    st.session_state.selected_jd_profile = DEFAULT_PROFILE_LABEL
if st.session_state.selected_jd_profile not in profile_options:
    st.session_state.selected_jd_profile = DEFAULT_PROFILE_LABEL

st.sidebar.selectbox(
    "Select Target JD Profile",
    profile_options,
    key="selected_jd_profile",
)

selected_profile = st.session_state.selected_jd_profile
if selected_profile == DEFAULT_PROFILE_LABEL:
    st.session_state.current_interview_data = _attach_interviewee_profile(load_interview_data())
else:
    profile_path = PROFILES_DIR / selected_profile
    try:
        st.session_state.current_interview_data = _attach_interviewee_profile(
            load_interview_data(profile_path)
        )
    except Exception as exc:
        st.sidebar.error(f"Failed to load profile '{selected_profile}': {exc}")
        st.session_state.current_interview_data = _attach_interviewee_profile(load_interview_data())

if st.session_state.get("active_jd_profile") != selected_profile:
    st.session_state.active_jd_profile = selected_profile
    st.session_state.jd_profile_changed = True
    st.session_state.interview_started = False
else:
    st.session_state.jd_profile_changed = False

if st.session_state.jd_profile_changed and st.session_state.get("chat_histories"):
    st.session_state.chat_histories.pop(GENERIC_INTERVIEWER_KEY, None)

_bootstrap_default_interviewers(st.session_state.current_interview_data)

if selected_profile == DEFAULT_PROFILE_LABEL:
    job_positions = st.session_state.current_interview_data.get("job_positions", [])
    if isinstance(job_positions, list) and job_positions and isinstance(job_positions[0], dict):
        selected_jd_title = str(job_positions[0].get("title", DEFAULT_PROFILE_LABEL))
    else:
        selected_jd_title = DEFAULT_PROFILE_LABEL
else:
    selected_jd_title = str(st.session_state.current_interview_data.get("position", selected_profile))
st.session_state.selected_jd = selected_jd_title

interviewer_records = _collect_interviewer_records(st.session_state.selected_jd)
custom_names = _collect_interviewer_names(interviewer_records)

def _build_profile_status_panel() -> None:
    report = audit_interview_profiles(DATA_DIR)
    if not report:
        st.sidebar.caption("No profile files were found.")
        return

    total = len(report)
    healthy = sum(1 for item in report if item["status"] == "ok")
    needs_attention = sum(1 for item in report if item["status"] in {"invalid_file", "missing_required"})

    st.sidebar.divider()
    st.sidebar.subheader("JD Profile Status")
    metric_col_total, metric_col_healthy, metric_col_needs = st.sidebar.columns(3)
    metric_col_total.metric("Total", total)
    metric_col_healthy.metric("Healthy", healthy)
    metric_col_needs.metric("Needs", needs_attention)

    selected = st.session_state.get("selected_jd_profile", AUDIT_DEFAULT_PROFILE_LABEL)
    for item in report:
        missing_required = item.get("missing_required", [])
        recommendations = item.get("recommendations", [])
        top_recommendations = recommendations[:5]

        if item["status"] == "ok":
            icon = "✅"
            status_label = "Complete"
        else:
            icon = "⚠️"
            status_label = "Needs attention"

        file_label = item["file"]
        if item["file"] == "interview_data.json":
            file_label = AUDIT_DEFAULT_PROFILE_LABEL

        session_list = ", ".join(item.get("session_interviewers", []))
        if not session_list:
            session_list = "(empty)"

        if file_label == selected:
            st.sidebar.markdown(f"**{icon} {file_label}**")
            st.sidebar.caption(f"Status: {status_label}")
            st.sidebar.caption(f"Current session_interviewers: {session_list}")
            if not top_recommendations:
                st.sidebar.caption("No interviewer correlation score found.")
            else:
                st.sidebar.caption("Interviewer-JD Correlation")
                for rec in top_recommendations:
                    label = rec.get("name", "Interviewer")
                    score = rec.get("score", 0.0)
                    status = "recommended" if rec.get("recommended") else "candidate"
                    st.sidebar.caption(f"{label} | {status} | {score:.1f}%")
        else:
            st.sidebar.markdown(f"{icon} {file_label}")

        if missing_required:
            st.sidebar.caption(f"Missing required: {', '.join(missing_required)}")


default_session_interviewers = _build_default_session_interviewers(st.session_state.current_interview_data)
session_interviewer_catalog = _normalize_label_list([*default_session_interviewers, *custom_names])
stored_session_interviewers = st.session_state.current_interview_data.get(JD_PROFILE_SESSION_KEY, default_session_interviewers)
if not isinstance(stored_session_interviewers, list):
    stored_session_interviewers = default_session_interviewers
active_interviewers = _filter_session_interviewers(
    stored_session_interviewers,
    session_interviewer_catalog,
    default_session_interviewers,
)
st.session_state.active_interviewers = active_interviewers

_build_profile_status_panel()

if "current_interviewer_label" not in st.session_state:
    st.session_state.current_interviewer_label = GENERIC_INTERVIEWER_LABEL

session_interviewers = _build_session_interviewer_profiles(st.session_state.current_interview_data, st.session_state.selected_jd)
speaker_options, label_to_key = _build_speaker_options_from_profiles(session_interviewers)

active_interviewers = st.session_state.get("active_interviewers", [GENERIC_INTERVIEWER_LABEL])
if not active_interviewers:
    active_interviewers = [GENERIC_INTERVIEWER_LABEL]

available_interviewers = [label for label in speaker_options if label in active_interviewers]
if not available_interviewers and GENERIC_INTERVIEWER_LABEL in speaker_options:
    available_interviewers = [GENERIC_INTERVIEWER_LABEL]
elif not available_interviewers:
    available_interviewers = speaker_options[:1]

if st.session_state.current_interviewer_label not in available_interviewers:
    st.session_state.current_interviewer_label = available_interviewers[0]

selected_label = st.sidebar.selectbox(
    "Select interviewer",
    available_interviewers,
    index=available_interviewers.index(st.session_state.current_interviewer_label)
    if st.session_state.current_interviewer_label in available_interviewers
    else 0,
)
st.session_state.current_interviewer_label = selected_label
selected_interviewer = label_to_key[selected_label]
if "active_interviewer_key" not in st.session_state:
    st.session_state.active_interviewer_key = selected_interviewer
elif st.session_state.active_interviewer_key != selected_interviewer:
    st.session_state.interview_started = False
    st.session_state.active_interviewer_key = selected_interviewer


def _start_interview_for_active_interviewer(
    interviewer_key: str,
    interviewer_label: str,
    jd_title: str,
    system_prompts: Dict[str, str],
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
    except Exception as exc:
        st.error(f"Error from OpenAI API: {exc}")
        st.session_state.interview_started = False
    finally:
        typing_placeholder.empty()


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
    else:
        st.markdown(feedback)


PANEL_CHAT = "Interview Chat"
PANEL_CODING = "Coding Challenge"


def render_sidebar_profile_creator() -> None:
    st.sidebar.divider()
    with st.sidebar.expander("Create Interview Profile from JD", expanded=False):
        jd_company = st.text_input("Company Name", key="jd_company_name")
        jd_position = st.text_input("Job Position Title", key="jd_position_title")
        jd_text = st.text_area("Paste Job Description", key="jd_text", height=180)

        if st.button("Generate & Save Interview Profile", use_container_width=True):
            company = jd_company.strip()
            position = jd_position.strip()
            description = jd_text.strip()
            if not company or not position or not description:
                st.error("Company, position, and JD are all required.")
                return

            api_key = get_api_key()
            if not api_key:
                st.error("OPENAI_API_KEY is not set. Add it in Streamlit secrets or environment variables.")
                return

            try:
                extracted, used_model = extract_jd_fields(
                    api_key=api_key,
                    jd_text=description,
                    company=company,
                    position=position,
                )
                profile = build_interview_profile(extracted=extracted)
                filename = build_profile_filename(extracted.company, extracted.position)
                PROFILES_DIR.mkdir(parents=True, exist_ok=True)
                output_path = PROFILES_DIR / filename
                output_path.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")
                st.success(f"Saved profile: {output_path.name} (model: {used_model})")
            except Exception as exc:
                st.error(f"Failed to generate profile: {exc}")


def render_sidebar_interviewee_profile_loader() -> None:
    st.sidebar.divider()
    with st.sidebar.expander("Load Interviewee Profile (CV / Cover Letter)", expanded=False):
        uploaded_pdf = st.file_uploader(
            "Upload Resume/CV (PDF)",
            type=["pdf"],
            key="interviewee_resume_pdf",
        )
        if uploaded_pdf is not None:
            pdf_signature = f"{uploaded_pdf.name}|{hash_file_contents(uploaded_pdf.getvalue())}"
            previous_signature = st.session_state.get("interviewee_resume_signature", "")
            if previous_signature != pdf_signature:
                try:
                    api_key = get_api_key()
                    if not api_key:
                        st.error(
                            "OPENAI_API_KEY is not set. Add it in Streamlit secrets or environment variables."
                        )
                    else:
                        extracted_text = _extract_text_from_pdf(uploaded_pdf)
                        if not extracted_text.strip():
                            st.warning("No extractable text found in uploaded PDF.")
                        else:
                            parsed = parse_interviewee_profile(
                                api_key=api_key,
                                source_text=extracted_text,
                                model=st.session_state.selected_model,
                                source_label="resume",
                            )
                            _save_interviewee_profile(_as_interviewee_payload(parsed))
                            st.session_state.interviewee_resume_signature = pdf_signature
                            st.success("Interviewee profile reset from uploaded CV.")
                            st.session_state.current_interview_data = _attach_interviewee_profile(
                                st.session_state.current_interview_data
                            )
                            st.rerun()
                except Exception as exc:
                    st.error(f"Failed to reset interviewee profile from PDF: {exc}")

        cover_letter = st.text_area(
            "Paste Cover Letter",
            key="interviewee_cover_letter",
            height=140,
        )

        if st.button("Apply Cover Letter", key="apply_cover_letter_button", use_container_width=True):
            cover_text = cover_letter.strip()
            if not cover_text:
                st.warning("Paste a cover letter first.")
                return

            api_key = get_api_key()
            if not api_key:
                st.error(
                    "OPENAI_API_KEY is not set. Add it in Streamlit secrets or environment variables."
                )
                return

            current_profile = _load_interviewee_profile()
            if not current_profile:
                st.warning("No interviewee profile exists yet. Upload a CV first.")
                return

            try:
                parsed = parse_interviewee_profile(
                    api_key=api_key,
                    source_text=cover_text,
                    model=st.session_state.selected_model,
                    source_label="cover_letter",
                )
                merged = _merge_interviewee_cover_letter(current_profile, parsed)
                _save_interviewee_profile(merged)
                st.success("Cover letter information merged into interviewee profile.")
                st.session_state.current_interview_data = _attach_interviewee_profile(
                    st.session_state.current_interview_data
                )
                st.rerun()
            except Exception as exc:
                st.error(f"Failed to merge cover letter: {exc}")


def render_floating_coding_tab_button() -> None:
    st.markdown(
        f"""
<style>
[data-testid="stAppViewContainer"] .main .block-container {{
  padding-right: 96px;
}}
.st-key-open_coding_tab {{
  position: fixed;
  right: 12px;
  top: 50%;
  transform: translateY(-50%);
  z-index: 9999;
  margin: 0 !important;
  padding: 0 !important;
}}
.st-key-open_coding_tab button {{
  width: 42px;
  height: 190px;
  border: 1px solid #444;
  border-radius: 12px;
  background: #0e1117;
  color: #fff !important;
  writing-mode: vertical-rl;
  text-orientation: mixed;
  letter-spacing: 0.3px;
  font-size: 12px;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 0 !important;
}}
.st-key-open_coding_tab button:hover {{
  border-color: #999;
}}
@media (max-width: 640px) {{
  .st-key-open_coding_tab button {{
    right: 8px;
    width: 36px;
    height: 160px;
    font-size: 11px;
  }}
}}
</style>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Coding Challenge", key="open_coding_tab"):
        st.session_state.tabs_instance_nonce = st.session_state.get("tabs_instance_nonce", 0) + 1
        st.session_state.pending_tabs_default = PANEL_CODING
        st.rerun()


def get_tab_labels() -> tuple[str, str]:
    nonce = st.session_state.get("tabs_instance_nonce", 0)
    suffix = "\u200b" * nonce
    return PANEL_CHAT + suffix, PANEL_CODING + suffix


def get_tabs_default_once(tab_chat_label: str, tab_code_label: str) -> str | None:
    pending = st.session_state.pop("pending_tabs_default", None)
    if pending == PANEL_CODING:
        return tab_code_label
    if pending == PANEL_CHAT:
        return tab_chat_label
    return None


def render_chat_panel(
    current_interviewer: str,
    history: List[Dict[str, str]],
    system_prompts: Dict[str, str],
) -> None:
    st.subheader(PANEL_CHAT)
    if not st.session_state.get("interview_started", False):
        st.info("Press '🚀 Start Interview' in the sidebar to let the interviewer begin.")
        return

    for idx, msg in enumerate(history):
        message(msg["content"], is_user=msg["role"] == "user", key=f"{current_interviewer}-{idx}")

    user_input = st.chat_input("Type your interview answer or ask a question...")
    if not user_input:
        pass
    else:
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
                client = OpenAI(api_key=api_key)
                typing_placeholder = st.empty()
                typing_placeholder.info(f"{current_interviewer.title()} is typing...")
                try:
                    response, used_model = create_chat_completion_with_fallback(
                        client=client,
                        model=st.session_state.selected_model,
                        messages=[{"role": "system", "content": system_prompts[current_interviewer]}, *history],
                        temperature=st.session_state.temperature,
                    )
                    assistant_reply = response.choices[0].message.content or "I could not generate a response."
                    if used_model != st.session_state.selected_model:
                        assistant_reply = f"[Model fallback: {used_model}]\n\n{assistant_reply}"
                except Exception as exc:
                    assistant_reply = f"Error from OpenAI API: {exc}"
                finally:
                    typing_placeholder.empty()

                history.append({"role": "assistant", "content": assistant_reply})
                st.session_state.chat_histories[current_interviewer] = history
                st.rerun()

    analyze_key = f"critique_feedback_{current_interviewer}"
    hint_key = f"interview_hint_{current_interviewer}"
    answer_key = f"model_answer_{current_interviewer}"

    critique_persona = get_critique_persona_prompt(interviewer_name=current_interviewer.title())

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
                    client = OpenAI(api_key=api_key)
                    critique_user_prompt = f"""Interviewer: {current_interviewer.title()}\nCandidate answer:\n{last_user_answer}\n\nEvaluate this answer for ASPICE CL3 evidence and technical accuracy."""
                    try:
                        critique_temperature = max(0.1, st.session_state.temperature - 0.2)
                        critique_response, used_model = create_chat_completion_with_fallback(
                            client=client,
                            model=st.session_state.selected_model,
                            messages=[
                                {"role": "system", "content": critique_persona},
                                {"role": "user", "content": critique_user_prompt},
                            ],
                            temperature=critique_temperature,
                        )
                        critique_text = critique_response.choices[0].message.content or "No critique generated."
                        if used_model != st.session_state.selected_model:
                            critique_text = f"[Model fallback: {used_model}]\n\n{critique_text}"
                        st.session_state[analyze_key] = critique_text
                    except Exception as exc:
                        st.session_state[analyze_key] = f"Error from OpenAI API: {exc}"

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
                    client = OpenAI(api_key=api_key)
                    try:
                        hint_response, used_model = create_chat_completion_with_fallback(
                            client=client,
                            model=st.session_state.selected_model,
                            messages=[{"role": "user", "content": _build_support_prompt(current_interviewer.title(), last_question, "hint")}],
                            temperature=max(0.1, st.session_state.temperature - 0.1),
                        )
                        hint_text = hint_response.choices[0].message.content or "No hint generated."
                        if used_model != st.session_state.selected_model:
                            hint_text = f"[Model fallback: {used_model}]\n\n{hint_text}"
                        st.session_state[hint_key] = hint_text
                    except Exception as exc:
                        st.session_state[hint_key] = f"Failed to generate hint: {exc}"
    with col_model:
        if st.button("Show Model Answer", use_container_width=True, key=f"model_answer_btn_{current_interviewer}"):
            if not last_question:
                st.session_state[answer_key] = "Please ask a question first, then request a model answer."
            else:
                api_key = get_api_key()
                if not api_key:
                    st.session_state[answer_key] = "OPENAI_API_KEY is not set. Add it in Streamlit secrets or environment variables."
                else:
                    client = OpenAI(api_key=api_key)
                    try:
                        answer_response, used_model = create_chat_completion_with_fallback(
                            client=client,
                            model=st.session_state.selected_model,
                            messages=[{"role": "user", "content": _build_support_prompt(current_interviewer.title(), last_question, "sample")}],
                            temperature=st.session_state.temperature,
                        )
                        answer_text = answer_response.choices[0].message.content or "No model answer generated."
                        if used_model != st.session_state.selected_model:
                            answer_text = f"[Model fallback: {used_model}]\n\n{answer_text}"
                        st.session_state[answer_key] = answer_text
                    except Exception as exc:
                        st.session_state[answer_key] = f"Failed to generate model answer: {exc}"

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


def render_coding_panel(current_interviewer: str) -> None:
    st.subheader(f"{PANEL_CODING} (C++)")
    st.caption("Use this area for implementation-focused questions. Syntax highlighting is set to C++.")
    if "editor_input_mode" not in st.session_state:
        st.session_state.editor_input_mode = "Normal"
    st.selectbox("Editor Input Mode", ["Normal", "Vim"], key="editor_input_mode")
    keybinding_mode = "vim" if st.session_state.editor_input_mode == "Vim" else "vscode"

    ace_key = f"coding_challenge_cpp_{current_interviewer}"
    default_cpp = """#include <iostream>\n#include <vector>\n\nint main() {\n    std::vector<int> data{1, 2, 3, 4, 5};\n    int sum = 0;\n    for (int x : data) {\n        sum += x;\n    }\n    std::cout << \"Sum: \" << sum << std::endl;\n    return 0;\n}\n"""
    if ace_key not in st.session_state:
        st.session_state[ace_key] = default_cpp

    st_ace(
        value=st.session_state[ace_key],
        language="c_cpp",
        theme="tomorrow_night",
        key=ace_key,
        height=420,
        auto_update=True,
        font_size=14,
        wrap=True,
        keybinding=keybinding_mode,
    )


render_sidebar_profile_creator()
render_sidebar_interviewee_profile_loader()
render_floating_coding_tab_button()

if "current_interviewer" not in st.session_state:
    st.session_state.current_interviewer = selected_interviewer

if "chat_histories" not in st.session_state:
    st.session_state.chat_histories = {}
    st.session_state.chat_histories[GENERIC_INTERVIEWER_KEY] = []
if selected_interviewer not in st.session_state.chat_histories:
    st.session_state.chat_histories[selected_interviewer] = []
st.session_state.current_interviewer = selected_interviewer

current_interviewer = st.session_state.current_interviewer

data = st.session_state.current_interview_data
technique_key = PROMPT_TECHNIQUES[st.session_state.selected_technique]
system_prompts = build_system_prompts(
    data=data,
    jd_profile=st.session_state.current_interview_data,
    technique=technique_key,
    interviewers=session_interviewers,
)
history: List[Dict[str, str]] = st.session_state.chat_histories[current_interviewer]
st.session_state.messages = history

st.title(APP_TITLE)
if st.session_state.jd_profile_changed:
    st.success(f"Rebuilt personas for JD profile: {selected_profile}")
st.caption(f"Active interviewer: {selected_label}")
if st.session_state.jd_profile_changed:
    st.caption(f"Loaded JD profile: {st.session_state.selected_jd_profile}")

if st.sidebar.button("🚀 Start Interview", use_container_width=True, type="primary"):
    _start_interview_for_active_interviewer(
        interviewer_key=current_interviewer,
        interviewer_label=selected_label,
        jd_title=st.session_state.selected_jd,
        system_prompts=system_prompts,
    )

chat_tab_label, code_tab_label = get_tab_labels()
default_tab = get_tabs_default_once(chat_tab_label, code_tab_label)
tab_chat, tab_code = st.tabs([chat_tab_label, code_tab_label], default=default_tab)

with tab_chat:
    render_chat_panel(
        current_interviewer=current_interviewer,
        history=history,
        system_prompts=system_prompts,
    )

with tab_code:
    render_coding_panel(current_interviewer=current_interviewer)
