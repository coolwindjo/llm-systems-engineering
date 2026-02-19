from __future__ import annotations

import streamlit as st
from streamlit_ace import st_ace

PANEL_CHAT = "Interview Chat"
PANEL_CODING = "Coding Challenge"


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

