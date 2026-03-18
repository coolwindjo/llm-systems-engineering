from __future__ import annotations

from typing import Any

import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError

from services.config import (
    APP_TITLE,
    DEFAULT_MODEL_NAME,
    SUPPORTED_CHAT_MODELS,
    get_logger,
    load_local_env,
)
from services.copilot import ConfigurationCopilot, create_copilot

LOGGER = get_logger("app_runtime")
DEMO_PROMPTS = [
    "Add a cyclic WheelSpeedBroadcast service with id 0x120 running every 10 ms for wheel speed signals.",
    "Need a one-shot DiagnosticSnapshot service, class event, id 42, triggered after fault confirmation.",
    "Configure TorqueCommandSync as a cyclic service at 5 Hz, but also say it should be on-change only.",
    "Create an on-request method called FlashReadbackService with id 0x220 for maintenance diagnostics.",
    "Define BatteryStateField as an on-change field service with id 0x98 to publish state-of-charge updates.",
    "Set up CabinThermalBroadcast for HVAC data every 100 ms, but I forgot the service id.",
    "Build a request-response service named VinReadService for plant tools to read VIN data on demand.",
    "I need a steering torque service for chassis control, probably cyclic, but I did not specify the exact frequency or play type.",
]


def _render_assistant_message(result: dict[str, Any]) -> None:
    if result.get("error"):
        error = result["error"]
        st.error(error["message"])
        with st.expander("Failure Details", expanded=True):
            st.write("Stage", error["stage"])
            st.code(error["detail"])
        if result.get("tool_trace"):
            with st.expander("Tool Trace Before Failure", expanded=False):
                _render_tool_trace(result["tool_trace"])
        return

    validation = result["validation"]
    extraction = result["extraction"]
    analysis = result["analysis"]

    st.markdown(f"**Overall status:** `{validation['status']}`")
    st.markdown(validation["summary"])

    with st.expander("Supporting Step: analyze_requirement_text", expanded=True):
        st.markdown(f"**Intent summary**: {analysis['summary']}")
        st.write("Ambiguities", analysis["ambiguities"] or ["None"])
        st.write("Contradictions", analysis["contradictions"] or ["None"])
        st.write("Missing information", analysis["missing_information"] or ["None"])
        st.write("Follow-up questions", analysis["follow_up_questions"] or ["None"])

    with st.expander("Tool Result: extract_configuration_parameters", expanded=True):
        st.json(extraction)

    with st.expander("Tool Result: retrieve_reference_context", expanded=True):
        st.write("Translated queries", result["translated_queries"])
        for chunk in result["retrieved_chunks"]:
            st.markdown(
                f"- **{chunk['title']}** (`{chunk['source']}` / query: `{chunk['query']}`)"
            )
            st.caption(chunk["excerpt"])

    with st.expander("Tool Result: validate_configuration_json", expanded=True):
        st.write("Schema valid", validation["schema_valid"])
        st.write("Missing required fields", validation["missing_required_fields"] or ["None"])
        st.write("Basic rule findings", validation["basic_rule_findings"] or ["None"])
        st.write("Reference guidance notes", validation["compliance_notes"] or ["None"])
        st.write("Suggested actions", validation["suggested_actions"] or ["None"])
        st.write("Follow-up questions", validation["follow_up_questions"] or ["None"])
        st.write("Referenced sections", validation["referenced_sections"] or ["None"])

    with st.expander("Tool Call Trace", expanded=False):
        _render_tool_trace(result["tool_trace"])


def _render_tool_trace(tool_trace: list[dict[str, Any]]) -> None:
    for step in tool_trace:
        st.markdown(f"**{step.get('tool', 'unknown_tool')}**")
        st.caption(step.get("purpose", ""))
        st.write("Status", step.get("status", "unknown"))
        if step.get("output_summary") is not None:
            st.json({"output_summary": step["output_summary"]})
        st.json(
            {
                "input": step.get("input", {}),
                "output": step.get("output", {}),
            }
        )


def _run_kb_operation(
    *,
    spinner_message: str,
    success_message: str,
    operation,
) -> None:
    with st.spinner(spinner_message):
        try:
            result = operation()
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("kb_operation_failed")
            st.error(f"Knowledge base operation failed: {exc}")
        else:
            st.success(success_message)
            st.json(result)


def _render_kb_reset_controls(copilot: ConfigurationCopilot) -> None:
    if st.button("Reset and rebuild knowledge base", key="kb-reset"):
        _run_kb_operation(
            spinner_message="Resetting and rebuilding the vectorstore...",
            success_message="Knowledge base reset completed.",
            operation=copilot.kb.reset_knowledge_base,
        )


def _render_kb_document_inventory(copilot: ConfigurationCopilot) -> None:
    st.write("Stored reference files")
    documents = copilot.kb.list_reference_documents()
    if not documents:
        st.caption("No reference files are stored in `data/standards/`.")
        return

    for document in documents:
        cols = st.columns([3, 2, 1])
        with cols[0]:
            st.markdown(f"`{document['name']}`")
        with cols[1]:
            st.caption(
                f"type={document['file_type']}, indexed={document['indexed_source']}"
            )
            if document.get("paired_markdown"):
                st.caption(f"paired markdown: {document['paired_markdown']}")
            if document.get("paired_pdf"):
                st.caption(f"paired pdf: {document['paired_pdf']}")
        with cols[2]:
            if st.button(
                "🗑️",
                key=f"kb-delete::{document['name']}",
                help=f"Delete {document['name']}",
            ):
                _run_kb_operation(
                    spinner_message=f"Deleting {document['name']} and rebuilding the vectorstore...",
                    success_message=f"Deleted {document['name']} from the knowledge base.",
                    operation=lambda name=document["name"]: copilot.kb.delete_document_and_rebuild(
                        name
                    ),
                )


def _render_kb_update_panel(copilot: ConfigurationCopilot) -> None:
    st.divider()
    st.header("Knowledge Base Updates")
    st.caption(
        "Import standards PDF/Markdown files, reset the indexed collection, or delete stored files."
    )
    st.caption("Indexed sources: " + ", ".join(copilot.kb.corpus_sources()))
    _render_kb_reset_controls(copilot)
    uploaded_reference = st.file_uploader(
        "Import standards PDF or Markdown",
        type=["pdf", "md"],
        accept_multiple_files=False,
        key="kb_reference_upload",
    )
    if uploaded_reference is not None and st.button(
        "Import document and rebuild knowledge base",
        key=f"kb-import::{uploaded_reference.name}",
    ):
        _run_kb_operation(
            spinner_message="Importing document and rebuilding the vectorstore...",
            success_message="Knowledge base updated successfully.",
            operation=lambda: copilot.kb.import_document_bytes_and_rebuild(
                uploaded_reference.name,
                uploaded_reference.getvalue(),
            ),
        )
    _render_kb_document_inventory(copilot)


@st.cache_resource(show_spinner=False)
def _get_copilot(model_name: str, api_key: str) -> ConfigurationCopilot:
    return create_copilot(model_name=model_name, api_key=api_key)


def _get_api_key() -> str | None:
    try:
        secret_key = st.secrets.get("OPENAI_API_KEY")
    except StreamlitSecretNotFoundError:
        secret_key = None

    if secret_key:
        return secret_key

    import os

    return os.getenv("OPENAI_API_KEY")


def run_config_copilot_app() -> None:
    load_local_env()

    st.set_page_config(page_title=APP_TITLE, page_icon="🚗", layout="wide")
    st.title(APP_TITLE)
    st.caption(
        "Natural-language configuration requirement to reference retrieval, configuration JSON drafting, and basic validation."
    )

    with st.sidebar:
        st.header("Runtime Settings")
        selected_model = st.selectbox("OpenAI model", SUPPORTED_CHAT_MODELS, index=SUPPORTED_CHAT_MODELS.index(DEFAULT_MODEL_NAME))
        st.markdown(
            """
            **MVP flow**
            1. Optional requirement analysis
            2. `retrieve_reference_context`
            3. `extract_configuration_parameters`
            4. `validate_configuration_json`
            """
        )
        st.write("Demo prompts")
        for sample in DEMO_PROMPTS:
            if st.button(sample, key=f"sample::{sample[:24]}"):
                st.session_state["pending_prompt"] = sample

    api_key = _get_api_key() or st.session_state.get("OPENAI_API_KEY")

    if not api_key:
        st.error("OPENAI_API_KEY is required. Add it to Streamlit secrets or /workspace/.env.")
        st.stop()

    copilot = _get_copilot(selected_model, api_key)
    with st.sidebar:
        _render_kb_update_panel(copilot)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for item in st.session_state.chat_history:
        with st.chat_message(item["role"]):
            if item["role"] == "user":
                st.markdown(item["content"])
            else:
                _render_assistant_message(item["content"])

    pending_prompt = st.session_state.pop("pending_prompt", None)
    prompt = st.chat_input("Describe the configuration requirement.") or pending_prompt

    if prompt:
        prompt = prompt.strip()
        if not prompt:
            LOGGER.warning("empty_prompt_submitted")
            st.warning("Enter a non-empty configuration requirement before running the copilot.")
            return

        LOGGER.info("user_prompt_submitted chars=%s model=%s", len(prompt), selected_model)
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Running tool calls: retrieve context, draft configuration JSON, validate output..."):
                try:
                    result = copilot.run(prompt)
                except Exception as exc:  # noqa: BLE001
                    LOGGER.exception("copilot_run_failed")
                    result = {
                        "error": {
                            "stage": "runtime",
                            "message": "The configuration copilot failed during execution. Review the error details and retry.",
                            "detail": str(exc),
                        },
                        "tool_trace": [],
                    }
            _render_assistant_message(result)
        st.session_state.chat_history.append({"role": "assistant", "content": result})
