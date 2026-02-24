from __future__ import annotations

import os
import json
from typing import Any, Dict, List

import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError

from components.chat import render_chat_panel
from components.coding import (
    PANEL_CHAT,
    get_tab_labels,
    get_tabs_default_once,
    render_coding_panel,
    render_floating_coding_tab_button,
)
from components.evaluation_dashboard import render_evaluation_dashboard
from components.interview_session import start_interview_for_active_interviewer
from components.sidebar import (
    render_profile_status_panel,
    render_sidebar_interviewee_profile_loader,
    render_sidebar_profile_creator,
)
from services.interview_ops import (
    DEFAULT_PROFILE_LABEL as AUDIT_DEFAULT_PROFILE_LABEL,
    CHAT_CAPABLE_MODELS,
    DATA_DIR,
    GENERIC_INTERVIEWER_KEY,
    GENERIC_INTERVIEWER_LABEL,
    JD_PROFILE_SESSION_KEY,
    PROFILES_DIR,
    _attach_interviewee_profile,
    _bootstrap_default_interviewers,
    _build_default_session_interviewers,
    _build_session_interviewer_profiles,
    _build_speaker_options_from_profiles,
    _coerce_temperature_for_model,
    _collect_interviewer_names,
    _collect_interviewer_records,
    _filter_session_interviewers,
    _normalize_label_list,
    _speaker_key,
    load_local_env,
)
from services.personas import build_system_prompts
from utils.data_loader import load_interview_data


APP_TITLE = "AI Interview Coach"


PROMPT_TECHNIQUES = {
    "Zero-Shot": "zero_shot",
    "Few-Shot": "few_shot",
    "Chain-of-Thought": "chain_of_thought",
    "Persona-Conditioning": "persona_conditioning",
    "Knowledge-Paucity (ISO 26262 Focus)": "knowledge_paucity",
}
_SIDEBAR_SELECTED_JD_PROFILE_PENDING_KEY = "selected_jd_profile_pending_update"


def get_api_key() -> str | None:
    try:
        secret_key = st.secrets.get("OPENAI_API_KEY")
    except StreamlitSecretNotFoundError:
        secret_key = None
    return secret_key or os.getenv("OPENAI_API_KEY")


def _sync_temperature_to_model_constraint() -> None:
    current_temperature = st.session_state.get("temperature", 0.4)
    st.session_state["temperature"] = _coerce_temperature_for_model(
        st.session_state.get("selected_model", ""),
        current_temperature,
    )


def run_interview_app() -> None:
    load_local_env()

    st.set_page_config(page_title=APP_TITLE, page_icon="🎯", layout="wide")
    st.sidebar.title(APP_TITLE)

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

    pending_profile = st.session_state.pop(_SIDEBAR_SELECTED_JD_PROFILE_PENDING_KEY, None)
    if isinstance(pending_profile, str) and pending_profile in profile_options:
        st.session_state.selected_jd_profile = pending_profile

    if "selected_jd_profile" not in st.session_state:
        st.session_state.selected_jd_profile = DEFAULT_PROFILE_LABEL
    if st.session_state.selected_jd_profile not in profile_options:
        st.session_state.selected_jd_profile = DEFAULT_PROFILE_LABEL

    selected_profile = st.sidebar.selectbox(
        "Select Target JD Profile",
        profile_options,
        key="selected_jd_profile",
    )
    if selected_profile == DEFAULT_PROFILE_LABEL:
        st.session_state.current_interview_data = _attach_interviewee_profile(load_interview_data())
    else:
        profile_path = PROFILES_DIR / selected_profile
        try:
            st.session_state.current_interview_data = _attach_interviewee_profile(
                load_interview_data(profile_path)
            )
        except (FileNotFoundError, json.JSONDecodeError, ValueError, TypeError, OSError) as exc:
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

    default_session_interviewers = _build_default_session_interviewers(st.session_state.current_interview_data)
    session_interviewer_catalog = _normalize_label_list([*default_session_interviewers, *custom_names])
    stored_session_interviewers = st.session_state.current_interview_data.get(
        JD_PROFILE_SESSION_KEY,
        default_session_interviewers,
    )
    if not isinstance(stored_session_interviewers, list):
        stored_session_interviewers = default_session_interviewers
    active_interviewers = _filter_session_interviewers(
        stored_session_interviewers,
        session_interviewer_catalog,
        default_session_interviewers,
    )
    st.session_state.active_interviewers = active_interviewers

    if "current_interviewer_label" not in st.session_state:
        st.session_state.current_interviewer_label = GENERIC_INTERVIEWER_LABEL

    session_interviewers = _build_session_interviewer_profiles(
        st.session_state.current_interview_data,
        st.session_state.selected_jd,
    )
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

    render_profile_status_panel(
        DATA_DIR,
        selected_label=st.session_state.get("selected_jd_profile", AUDIT_DEFAULT_PROFILE_LABEL),
        default_profile_label=AUDIT_DEFAULT_PROFILE_LABEL,
    )

    render_sidebar_profile_creator(get_api_key)
    render_sidebar_interviewee_profile_loader(get_api_key, st.session_state.current_interview_data)
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
        start_interview_for_active_interviewer(
            interviewer_key=current_interviewer,
            interviewer_label=selected_label,
            jd_title=st.session_state.selected_jd,
            system_prompts=system_prompts,
            get_api_key=get_api_key,
        )

    chat_tab_label, code_tab_label = get_tab_labels()
    dashboard_tab_label = "Admin Dashboard"
    default_tab = get_tabs_default_once(chat_tab_label, code_tab_label)
    tab_chat, tab_code, tab_dashboard = st.tabs(
        [chat_tab_label, code_tab_label, dashboard_tab_label],
        default=default_tab,
    )

    with tab_chat:
        active_interviewer_profile = next(
            (
                profile
                for profile in session_interviewers
                if _speaker_key(profile.get("name", "")) == current_interviewer
            ),
            None,
        )

        render_chat_panel(
            panel_title=PANEL_CHAT,
            current_interviewer=current_interviewer,
            history=history,
            system_prompts=system_prompts,
            interviewer_name=selected_label,
            technique_key=technique_key,
            interviewer_profile=active_interviewer_profile,
            jd_profile=st.session_state.current_interview_data,
            get_api_key=get_api_key,
        )

    with tab_code:
        render_coding_panel(current_interviewer=current_interviewer)

    with tab_dashboard:
        render_evaluation_dashboard(get_api_key)
