from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable, Dict, List

import streamlit as st
from pydantic import BaseModel, Field

from services.interview_ops import (
    PROFILES_DIR,
    _as_interviewee_payload,
    _attach_interviewee_profile,
    _extract_text_from_pdf,
    _load_interviewee_profile,
    _merge_interviewee_cover_letter,
    _save_interviewee_profile,
    build_interview_profile,
    build_profile_filename,
    extract_jd_fields,
    _collect_interviewer_records,
    _interviewer_display_label,
    hash_file_contents,
    parse_interviewee_profile,
)
from utils.interviewer_store import (
    InterviewerProfile,
    delete_interviewer,
    find_interviewer_file,
    save_interviewer,
)
from services.profile_health import audit_interview_profiles


_INTERVIEWER_MANAGE_NEW_ENTRY = "Add new interviewer"
_INTERVIEWER_DELETE_CONFIRM_KEY = "interviewer_delete_confirm"


class _ExperienceDraft(BaseModel):
    background: str = ""
    expertise: List[str] = Field(default_factory=list)
    potential_questions: List[str] = Field(default_factory=list)


def _coerce_interviewer_form_list(raw: Any) -> List[str]:
    if raw is None:
        return []

    if isinstance(raw, (list, tuple, set)):
        candidates = list(raw)
    else:
        if not str(raw).strip():
            return []
        candidates = re.split(r"[;,\n]", str(raw))

    normalized: List[str] = []
    seen = set[str]()
    for value in candidates:
        item = str(value).strip()
        if not item or item in seen:
            continue
        seen.add(item)
        normalized.append(item)

    return normalized


def _parse_experience_block(raw: Any) -> _ExperienceDraft:
    lines = str(raw).splitlines() if raw is not None else []
    background_chunks: List[str] = []
    expertise_chunks: List[str] = []
    question_chunks: List[str] = []

    active_field = "background"
    for raw_line in lines:
        line = str(raw_line).strip()
        if not line:
            continue

        lower = line.lower()
        if lower.startswith("background:"):
            active_field = "background"
            payload = line[len("background:"):].strip()
            if payload:
                background_chunks.append(payload)
            continue
        if lower.startswith("expertise:") or lower.startswith("expertises:"):
            active_field = "expertise"
            payload = line.split(":", 1)[1].strip()
            if payload:
                expertise_chunks.append(payload)
            continue
        if lower.startswith("potential questions:") or lower.startswith("questions:"):
            active_field = "questions"
            payload = line.split(":", 1)[1].strip()
            if payload:
                question_chunks.append(payload)
            continue

        if active_field == "expertise":
            expertise_chunks.append(line)
        elif active_field == "questions":
            question_chunks.append(line)
        else:
            background_chunks.append(line)

    if not background_chunks and not expertise_chunks and not question_chunks:
        raw_text = str(raw).strip()
        if raw_text and any(ch in raw_text for ch in ";,\n"):
            # Treat unlabeled structured input as plain experience tags.
            expertise_chunks.append(raw_text)
        elif raw_text:
            background_chunks.append(raw_text)

    payload = {
        "background": "\n".join(chunk.strip() for chunk in background_chunks if chunk.strip()),
        "expertise": _coerce_interviewer_form_list(
            ",".join(chunk for chunk in expertise_chunks if chunk.strip())
        ),
        "potential_questions": _coerce_interviewer_form_list(
            ",".join(chunk for chunk in question_chunks if chunk.strip())
        ),
    }
    return _ExperienceDraft.model_validate(payload)


def _serialize_experience_for_form(profile: InterviewerProfile | None) -> str:
    if profile is None:
        return ""

    lines = []
    if str(profile.background).strip():
        lines.append(f"Background: {str(profile.background).strip()}")
    if profile.expertise:
        lines.append(f"Expertise: {', '.join(profile.expertise)}")
    if profile.potential_questions:
        lines.append(f"Potential Questions: {', '.join(profile.potential_questions)}")
    return "\n".join(lines)


def _build_interviewer_profile_payload(
    name: str,
    role: str,
    experience: str,
) -> InterviewerProfile | None:
    normalized_name = str(name).strip()
    if not normalized_name:
        return None

    parsed = _parse_experience_block(experience)
    return InterviewerProfile(
        name=normalized_name,
        background=parsed.background,
        is_generic_ai=False,
        role=str(role).strip(),
        expertise=parsed.expertise,
        potential_questions=parsed.potential_questions,
    )


def _interviewer_path_labels(selected_jd: str) -> Dict[str, Path]:
    records = _collect_interviewer_records(selected_jd)
    return {
        f"{_interviewer_display_label(profile)} ({path.name})": path
        for path, profile in records
    }


def render_sidebar_interviewer_manager(selected_jd: str) -> None:
    st.sidebar.divider()
    with st.sidebar.expander("Manage Interviewers (Add / Edit / Remove)", expanded=False):
        path_options = _interviewer_path_labels(selected_jd)
        selected_path = list(path_options.keys())
        profile_records = {path: profile for path, profile in _collect_interviewer_records(selected_jd)}

        options = [_INTERVIEWER_MANAGE_NEW_ENTRY, *selected_path]
        selected_label = st.selectbox(
            "Interviewer",
            options,
            index=0,
            key="interviewer_manage_selection",
        )

        selected_profile: InterviewerProfile | None = None
        selected_profile_path: Path | None = None
        if selected_label != _INTERVIEWER_MANAGE_NEW_ENTRY:
            selected_profile_path = path_options.get(selected_label)
            if selected_profile_path is not None:
                selected_profile = profile_records.get(selected_profile_path)

        form_name = st.text_input("Name", value=selected_profile.name if selected_profile else "")
        form_role = st.text_input("Role", value=selected_profile.role if selected_profile else "")
        form_experience = st.text_area(
            "Experience",
            value=_serialize_experience_for_form(selected_profile),
            help="Use sections: Background:, Expertise:, Potential Questions:."
            " Separate expertise/questions with comma/newline/semicolon.",
            height=180,
        )
        st.caption("Examples: `Background: ...`, `Expertise: ADAS, C++`, `Potential Questions: Explain ...`")

        action_cols = st.columns([6, 1])
        with action_cols[0]:
            save_clicked = st.button(
                "Save interviewer profile",
                key="interviewer_save_button",
                use_container_width=True,
            )
        with action_cols[1]:
            delete_icon_key = "interviewer_delete_icon_button"
            selected_profile_for_delete = selected_profile_path is not None
            if selected_profile_for_delete:
                if st.button(
                    "🗑️",
                    key=delete_icon_key,
                    help="Remove selected interviewer",
                    use_container_width=True,
                ):
                    st.session_state[_INTERVIEWER_DELETE_CONFIRM_KEY] = (
                        str(selected_profile_path)
                        if selected_profile_path is not None
                        else ""
                    )
                    st.rerun()
            else:
                st.button(" ", key=delete_icon_key, disabled=True)

        if save_clicked:
            payload = _build_interviewer_profile_payload(
                name=form_name,
                role=form_role,
                experience=form_experience,
            )
            if payload is None:
                st.error("Interviewer name is required.")
            else:
                existing_path = selected_profile_path
                if existing_path is None:
                    existing_path = find_interviewer_file(payload, allow_scope_fallback=True)
                if existing_path is not None:
                    existing_path = str(existing_path)
                save_interviewer(profile=payload, existing_path=existing_path)
                st.success(f"Saved interviewer: {payload.name}")
                st.rerun()

        if selected_profile is not None and selected_profile_path is not None:
            selected_profile_path_key = str(selected_profile_path)
            confirm_target = st.session_state.get(_INTERVIEWER_DELETE_CONFIRM_KEY)
            if confirm_target != selected_profile_path_key:
                st.session_state.pop(_INTERVIEWER_DELETE_CONFIRM_KEY, None)
                confirm_target = None

            if confirm_target == selected_profile_path_key:
                confirm_cols = st.columns(2)
                if confirm_cols[0].button("Delete", key="confirm_interviewer_delete_button"):
                    delete_interviewer(selected_profile_path_key)
                    st.session_state.pop(_INTERVIEWER_DELETE_CONFIRM_KEY, None)
                    st.success("Interviewer deleted.")
                    st.rerun()
                if confirm_cols[1].button("Cancel", key="cancel_interviewer_delete_button"):
                    st.session_state.pop(_INTERVIEWER_DELETE_CONFIRM_KEY, None)
                    st.rerun()
        elif not path_options:
            st.info("No stored interviewers yet.")


def render_profile_status_panel(data_dir, selected_label: str, default_profile_label: str) -> None:
    report = audit_interview_profiles(data_dir)
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
            file_label = default_profile_label

        session_list = ", ".join(item.get("session_interviewers", []))
        if not session_list:
            session_list = "(empty)"

        if file_label == selected_label:
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


def render_sidebar_profile_creator(get_api_key: Callable[[], str | None]) -> None:
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


def render_sidebar_interviewee_profile_loader(
    get_api_key: Callable[[], str | None],
    current_interview_data: Dict[str, Any],
) -> None:
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
                                current_interview_data
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
                    current_interview_data
                )
                st.rerun()
            except Exception as exc:
                st.error(f"Failed to merge cover letter: {exc}")
