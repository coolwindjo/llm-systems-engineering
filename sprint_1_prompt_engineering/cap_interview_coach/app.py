from __future__ import annotations

import os
from typing import Dict, List

import streamlit as st
from openai import OpenAI
from streamlit_ace import st_ace
from streamlit_chat import message

from utils.data_loader import load_interview_data
from utils.personas import build_system_prompts


st.set_page_config(page_title="Capgemini AI Interview Coach", page_icon="🎯", layout="wide")

st.sidebar.title("Capgemini AI Interview Coach")

interviewer_labels = {"Denis": "denis", "Aymen": "aymen"}
selected_label = st.sidebar.selectbox("Select interviewer", list(interviewer_labels.keys()))
selected_interviewer = interviewer_labels[selected_label]

if "current_interviewer" not in st.session_state:
    st.session_state.current_interviewer = selected_interviewer

if "chat_histories" not in st.session_state:
    st.session_state.chat_histories = {"denis": [], "aymen": []}

if selected_interviewer != st.session_state.current_interviewer:
    st.session_state.current_interviewer = selected_interviewer

current_interviewer = st.session_state.current_interviewer

data = load_interview_data()
system_prompts = build_system_prompts(data)
history: List[Dict[str, str]] = st.session_state.chat_histories[current_interviewer]

st.title("Capgemini AI Interview Coach")
st.caption(f"Active interviewer: {current_interviewer.title()}")

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


def get_api_key() -> str | None:
    return st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")


def get_last_user_response(messages: List[Dict[str, str]]) -> str | None:
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return msg.get("content")
    return None


tab_chat, tab_code = st.tabs(["Interview Chat", "Coding Challenge"])

with tab_chat:
    for idx, msg in enumerate(history):
        message(msg["content"], is_user=msg["role"] == "user", key=f"{current_interviewer}-{idx}")

    analyze_key = f"critique_feedback_{current_interviewer}"
    if st.button("Analyze My Answer", use_container_width=True):
        last_user_answer = get_last_user_response(history)
        if not last_user_answer:
            st.warning("No user answer found yet. Submit an answer first.")
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
                    critique_response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": CRITIQUE_PERSONA},
                            {"role": "user", "content": critique_user_prompt},
                        ],
                        temperature=0.2,
                    )
                    st.session_state[analyze_key] = (
                        critique_response.choices[0].message.content or "No critique generated."
                    )
                except Exception as exc:
                    st.session_state[analyze_key] = f"Error from OpenAI API: {exc}"

    feedback = st.session_state.get(analyze_key)
    if feedback:
        st.subheader("Critique Feedback")
        st.markdown(feedback)

    user_input = st.chat_input("Type your interview answer or ask a question...")

    if user_input:
        history.append({"role": "user", "content": user_input})
        message(user_input, is_user=True, key=f"{current_interviewer}-user-live")

        api_key = get_api_key()
        if not api_key:
            st.error("OPENAI_API_KEY is not set. Add it in Streamlit secrets or environment variables.")
        else:
            client = OpenAI(api_key=api_key)
            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "system", "content": system_prompts[current_interviewer]}, *history],
                    temperature=0.4,
                )
                assistant_reply = response.choices[0].message.content or "I could not generate a response."
            except Exception as exc:
                assistant_reply = f"Error from OpenAI API: {exc}"

            history.append({"role": "assistant", "content": assistant_reply})
            st.session_state.chat_histories[current_interviewer] = history
            st.rerun()

with tab_code:
    st.subheader("Coding Challenge (C++)")
    st.caption("Use this area for implementation-focused questions. Syntax highlighting is set to C++.")
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
    )
