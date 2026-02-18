from __future__ import annotations

import json
import os
import re
from datetime import date
from pathlib import Path
from typing import Dict, List

import streamlit as st
from openai import OpenAI
from pydantic import BaseModel, Field
from streamlit.errors import StreamlitSecretNotFoundError
from streamlit_ace import st_ace
from streamlit_chat import message

from utils.data_loader import load_interview_data
from utils.personas import build_system_prompts
from utils.security import validate_input

try:
    from annotated_text import annotated_text
except ImportError:
    annotated_text = None


def load_local_env() -> None:
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


load_local_env()

DATA_DIR = Path(__file__).resolve().parent / "data"
PROFILES_DIR = DATA_DIR / "profiles"
CHAT_CAPABLE_MODELS = [
    "gpt-4o-mini-2024-07-18",
    "gpt-4o-mini",
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-0125",
    "gpt-3.5-turbo-1106",
    "gpt-4.1-mini",
    "gpt-4.1-mini-2025-04-14",
    "gpt-4.1-nano",
    "gpt-5-nano",
    "gpt-5-mini",
]
CHAT_FALLBACK_ORDER = [
    "gpt-4o-mini",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
    "gpt-5-mini",
    "gpt-5-nano",
    "gpt-3.5-turbo",
]

st.set_page_config(page_title="Capgemini AI Interview Coach", page_icon="🎯", layout="wide")

st.sidebar.title("Capgemini AI Interview Coach")

PROMPT_TECHNIQUES = {
    "Zero-Shot": "zero_shot",
    "Few-Shot (Denis)": "few_shot",
    "Chain-of-Thought (Denis)": "chain_of_thought",
    "Persona-Conditioning (Aymen)": "persona_conditioning",
    "Knowledge-Paucity (ISO 26262 Focus)": "knowledge_paucity",
}
if "selected_technique" not in st.session_state:
    st.session_state.selected_technique = "Zero-Shot"

st.sidebar.selectbox(
    "Prompting Technique",
    list(PROMPT_TECHNIQUES.keys()),
    key="selected_technique",
)

interviewer_labels = {"Denis": "denis", "Aymen": "aymen"}
selected_label = st.sidebar.selectbox("Select interviewer", list(interviewer_labels.keys()))
selected_interviewer = interviewer_labels[selected_label]

if "selected_model" not in st.session_state:
    st.session_state.selected_model = "gpt-4o-mini"
if st.session_state.selected_model not in CHAT_CAPABLE_MODELS:
    st.session_state.selected_model = "gpt-4o-mini"
if "temperature" not in st.session_state:
    st.session_state.temperature = 0.4

st.sidebar.selectbox(
    "OpenAI Model",
    CHAT_CAPABLE_MODELS,
    key="selected_model",
)
st.sidebar.slider(
    "Temperature",
    min_value=0.0,
    max_value=1.0,
    value=0.4,
    step=0.1,
    key="temperature",
)

DEFAULT_PROFILE_LABEL = "(Default) interview_data.json"
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
    st.session_state.current_interview_data = load_interview_data()
else:
    profile_path = PROFILES_DIR / selected_profile
    try:
        st.session_state.current_interview_data = load_interview_data(profile_path)
    except Exception as exc:
        st.sidebar.error(f"Failed to load profile '{selected_profile}': {exc}")
        st.session_state.current_interview_data = load_interview_data()

if "current_interviewer" not in st.session_state:
    st.session_state.current_interviewer = selected_interviewer

if "chat_histories" not in st.session_state:
    st.session_state.chat_histories = {"denis": [], "aymen": []}

if selected_interviewer != st.session_state.current_interviewer:
    st.session_state.current_interviewer = selected_interviewer

current_interviewer = st.session_state.current_interviewer

data = st.session_state.current_interview_data
technique_key = PROMPT_TECHNIQUES[st.session_state.selected_technique]
system_prompts = build_system_prompts(
    data=data,
    jd_profile=st.session_state.current_interview_data,
    technique=technique_key,
)
history: List[Dict[str, str]] = st.session_state.chat_histories[current_interviewer]

st.title("Capgemini AI Interview Coach")
st.caption(f"Active interviewer: {current_interviewer.title()}")
if st.session_state.selected_jd_profile != DEFAULT_PROFILE_LABEL:
    st.success(f"Personas recalibrated for JD profile: {st.session_state.selected_jd_profile}")

CRITIQUE_PERSONA = """You are a Capgemini ADAS Interview Critique Persona.
You evaluate a candidate's latest answer for a Capgemini technical interview.
Focus strictly on:
1) ASPICE CL3 process maturity evidence
2) Technical accuracy for ADAS / embedded C++ context

Provide concise, actionable feedback using this exact structure:
- Overall verdict (2-3 lines)
- ASPICE CL3 score: <1-5> with one-sentence rationale
- Technical accuracy score: <1-5> with one-sentence rationale
- Strengths: 2-3 bullets
- Gaps / risks: 2-3 bullets
- Improvement plan for next answer: 3 bullets with concrete phrasing examples
"""

ADAS_KEY_TERMS = [
    "ASPICE CL3",
    "ASPICE",
    "ISO 26262",
    "ASIL",
    "AUTOSAR",
    "MISRA",
    "Functional Safety",
    "Safety Case",
    "Radar",
    "Camera",
    "Lidar",
    "Sensor Fusion",
    "UDS",
    "DMA",
    "Unit Testing",
    "Static Analysis",
    "Determinism",
]

class JDExtraction(BaseModel):
    company: str = Field(description="Company name in English.")
    position: str = Field(description="Job position title in English.")
    job_description: str = Field(description="Full job description rewritten in English.")
    key_requirements: List[str] = Field(description="Top key requirements from the job description.")
    tech_stack: List[str] = Field(description="Technologies and tools required for the role.")


def is_model_access_error(exc: Exception) -> bool:
    error_text = str(exc).lower()
    return "model_not_found" in error_text or "does not have access to model" in error_text


def get_api_key() -> str | None:
    try:
        secret_key = st.secrets.get("OPENAI_API_KEY")
    except StreamlitSecretNotFoundError:
        secret_key = None
    return secret_key or os.getenv("OPENAI_API_KEY")


def slugify_for_filename(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9]+", "-", value.strip())
    sanitized = sanitized.strip("-")
    return sanitized or "unknown"


def build_profile_filename(company: str, position: str) -> str:
    company_part = slugify_for_filename(company)
    position_part = slugify_for_filename(position)
    return f"{company_part}-{position_part}-{date.today().isoformat()}.json"


def extract_jd_fields(api_key: str, jd_text: str, company: str, position: str) -> tuple[JDExtraction, str]:
    client = OpenAI(api_key=api_key)
    extraction_prompt = (
        "Extract hiring signals from this job description.\n"
        "If the content is in Korean or German, translate everything to natural English first.\n"
        "Output every field in English.\n"
        "Use concise phrases, deduplicate items, and only include content supported by the JD.\n"
        f"Company: {company}\n"
        f"Position: {position}\n\n"
        "Job Description:\n"
        f"{jd_text}"
    )
    models_to_try = CHAT_FALLBACK_ORDER
    last_error: Exception | None = None
    for model_name in models_to_try:
        try:
            response = client.beta.chat.completions.parse(
                model=model_name,
                messages=[
                    {"role": "system", "content": "Return structured hiring insights."},
                    {"role": "user", "content": extraction_prompt},
                ],
                temperature=0.1,
                response_format=JDExtraction,
            )
            parsed = response.choices[0].message.parsed
            if parsed is None:
                raise ValueError("Failed to parse structured output for JD extraction.")
            return parsed, model_name
        except Exception as exc:
            last_error = exc
            if is_model_access_error(exc):
                continue
            raise
    if last_error is not None:
        raise last_error
    raise ValueError("Failed to extract JD fields with available models.")


def create_chat_completion_with_fallback(
    client: OpenAI,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
        )
        return response, model
    except Exception as exc:
        if model != CHAT_FALLBACK_ORDER[0] and is_model_access_error(exc):
            for fallback_model in CHAT_FALLBACK_ORDER:
                if fallback_model == model:
                    continue
                try:
                    fallback_response = client.chat.completions.create(
                        model=fallback_model,
                        messages=messages,
                        temperature=temperature,
                    )
                    return fallback_response, fallback_model
                except Exception as fallback_exc:
                    if is_model_access_error(fallback_exc):
                        continue
                    raise
        raise


def build_interview_profile(extracted: JDExtraction) -> Dict[str, object]:
    return {
        "generated_at": date.today().isoformat(),
        "company": extracted.company,
        "position": extracted.position,
        "job_description": extracted.job_description,
        "key_requirements": extracted.key_requirements,
        "tech_stack": extracted.tech_stack,
        "interviewers": [
            {
                "name": "Denis",
                "role": "Program Manager",
                "background": "TBD",
                "expertise": [],
                "potential_questions": [],
            },
            {
                "name": "Aymen",
                "role": "Senior Embedded SW Consultant",
                "background": "TBD",
                "expertise": [],
                "potential_questions": [],
            },
        ],
    }


def get_last_user_response(messages: List[Dict[str, str]]) -> str | None:
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return msg.get("content")
    return None


def render_feedback_with_adas_terms(feedback: str) -> None:
    if not feedback:
        return

    if annotated_text is None:
        st.markdown(feedback)
        st.info("Install `streamlit-annotated-text` to enable in-line ADAS term highlighting.")
        return

    sorted_terms = sorted(set(ADAS_KEY_TERMS), key=len, reverse=True)
    pattern = re.compile("|".join(re.escape(term) for term in sorted_terms), flags=re.IGNORECASE)
    parts: List[str | tuple] = []
    cursor = 0
    for match in pattern.finditer(feedback):
        if match.start() > cursor:
            parts.append(feedback[cursor:match.start()])
        parts.append((match.group(0), "ADAS"))
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
    st.sidebar.subheader("Create Interview Profile from JD")
    jd_company = st.sidebar.text_input("Company Name", key="jd_company_name")
    jd_position = st.sidebar.text_input("Job Position Title", key="jd_position_title")
    jd_text = st.sidebar.text_area("Paste Job Description", key="jd_text", height=180)

    if not st.sidebar.button("Generate & Save Interview Profile", use_container_width=True):
        return

    company = jd_company.strip()
    position = jd_position.strip()
    description = jd_text.strip()
    if not company or not position or not description:
        st.sidebar.error("Company, position, and JD are all required.")
        return

    api_key = get_api_key()
    if not api_key:
        st.sidebar.error("OPENAI_API_KEY is not set. Add it in Streamlit secrets or environment variables.")
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
        st.sidebar.success(f"Saved profile: {output_path.name} (model: {used_model})")
    except Exception as exc:
        st.sidebar.error(f"Failed to generate profile: {exc}")


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
    for idx, msg in enumerate(history):
        message(msg["content"], is_user=msg["role"] == "user", key=f"{current_interviewer}-{idx}")

    analyze_key = f"critique_feedback_{current_interviewer}"
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
                    critique_user_prompt = f"""Interviewer: {current_interviewer.title()}
Candidate answer:
{last_user_answer}

Evaluate this answer for ASPICE CL3 evidence and technical accuracy."""
                    try:
                        critique_temperature = max(0.1, st.session_state.temperature - 0.2)
                        critique_response, used_model = create_chat_completion_with_fallback(
                            client=client,
                            model=st.session_state.selected_model,
                            messages=[
                                {"role": "system", "content": CRITIQUE_PERSONA},
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

    feedback = st.session_state.get(analyze_key)
    if feedback:
        st.subheader("Critique Feedback")
        render_feedback_with_adas_terms(feedback)

    user_input = st.chat_input("Type your interview answer or ask a question...")
    if not user_input:
        return

    is_valid, validation_error = validate_input(user_input)
    if not is_valid:
        st.error(validation_error or "Invalid input.")
        return

    history.append({"role": "user", "content": user_input})
    message(user_input, is_user=True, key=f"{current_interviewer}-user-live")

    api_key = get_api_key()
    if not api_key:
        st.error("OPENAI_API_KEY is not set. Add it in Streamlit secrets or environment variables.")
        return

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


def render_coding_panel(current_interviewer: str) -> None:
    st.subheader(f"{PANEL_CODING} (C++)")
    st.caption("Use this area for implementation-focused questions. Syntax highlighting is set to C++.")
    if "editor_input_mode" not in st.session_state:
        st.session_state.editor_input_mode = "Normal"
    st.selectbox("Editor Input Mode", ["Normal", "Vim"], key="editor_input_mode")
    keybinding_mode = "vim" if st.session_state.editor_input_mode == "Vim" else "vscode"

    ace_key = f"coding_challenge_cpp_{current_interviewer}"
    default_cpp = """#include <iostream>
#include <vector>

int main() {
    std::vector<int> data{1, 2, 3, 4, 5};
    int sum = 0;
    for (int x : data) {
        sum += x;
    }
    std::cout << "Sum: " << sum << std::endl;
    return 0;
}
"""
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
render_floating_coding_tab_button()
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
