from __future__ import annotations

from contextlib import contextmanager

import pytest
from streamlit.errors import StreamlitSecretNotFoundError

from components import app_runtime


class DummyContextManager:
    def __init__(self, value=None) -> None:
        self.value = value

    def __enter__(self):
        return self.value if self.value is not None else self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


class SessionState(dict):
    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc

    def __setattr__(self, key, value) -> None:
        self[key] = value


class MissingSecrets:
    def get(self, _key):
        raise StreamlitSecretNotFoundError("missing streamlit secrets")


class FakeCopilot:
    def __init__(self) -> None:
        self.prompts: list[str] = []
        self.kb = type(
            "FakeKnowledgeBase",
            (),
            {
                "corpus_sources": lambda self: ["autosar_error_handling.md"],
                "list_reference_documents": lambda self: [
                    {
                        "name": "autosar_error_handling.md",
                        "file_type": "md",
                        "indexed_source": True,
                        "paired_pdf": None,
                        "paired_markdown": None,
                        "delete_behavior": "Deletes only this file.",
                    }
                ],
                "reset_knowledge_base": lambda self: {
                    "collection_name": "automotive_config_standards",
                    "document_count": 1,
                    "chunk_count": 5,
                    "sources": ["autosar_error_handling.md"],
                },
                "delete_document_and_rebuild": lambda self, name: {
                    "requested_document": name,
                    "deleted_documents": [name],
                    "deleted_count": 1,
                    "document_count": 0,
                    "chunk_count": 0,
                    "sources": [],
                },
                "import_document_bytes_and_rebuild": lambda self, name, content: {
                    "stored_document_path": f"/tmp/{name}",
                    "markdown_path": f"/tmp/{name}.md",
                    "document_count": 3,
                    "chunk_count": 12,
                },
            },
        )()

    def run(self, prompt: str) -> dict:
        self.prompts.append(prompt)
        return {
            "analysis": {
                "summary": "Safe summary",
                "ambiguities": [],
                "contradictions": [],
                "missing_information": [],
                "follow_up_questions": [],
                "risk_level": "low",
            },
            "extraction": {
                "ServiceName": "WheelSpeedBroadcast",
                "ID": 288,
                "Class": "Event",
                "Frequency": "10 ms",
                "PlayType": "Cyclic",
            },
            "translated_queries": ["AUTOSAR timing note"],
            "retrieved_chunks": [
                {
                    "source": "standards_reference.md",
                    "title": "Standards Reference",
                    "query": "AUTOSAR timing note",
                    "excerpt": "Cyclic services should declare a period.",
                }
            ],
            "validation": {
                "status": "ready",
                "summary": "Validation completed.",
                "schema_valid": True,
                "missing_required_fields": [],
                "basic_rule_findings": [],
                "compliance_notes": ["Review timing note."],
                "suggested_actions": [],
                "follow_up_questions": [],
                "referenced_sections": ["Standards Reference"],
            },
            "tool_trace": [
                {
                    "tool": "retrieve_reference_context",
                    "status": "success",
                    "purpose": "Retrieve context.",
                    "input": {},
                    "output_summary": {"retrieved_chunk_count": 1},
                    "output": {},
                }
            ],
        }


def test_demo_prompts_are_reviewer_ready() -> None:
    assert 5 <= len(app_runtime.DEMO_PROMPTS) <= 10
    assert len(app_runtime.DEMO_PROMPTS) == len(set(app_runtime.DEMO_PROMPTS))


def test_get_api_key_prefers_streamlit_secrets(monkeypatch) -> None:
    monkeypatch.setattr(app_runtime.st, "secrets", {"OPENAI_API_KEY": "secret-key"})
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")

    api_key = app_runtime._get_api_key()

    assert api_key == "secret-key"


def test_get_api_key_falls_back_to_environment(monkeypatch) -> None:
    monkeypatch.setattr(app_runtime.st, "secrets", MissingSecrets())
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")

    api_key = app_runtime._get_api_key()

    assert api_key == "env-key"


def test_render_assistant_message_shows_required_review_sections(monkeypatch) -> None:
    expanders: list[str] = []
    markdown_calls: list[str] = []

    monkeypatch.setattr(app_runtime.st, "markdown", lambda text: markdown_calls.append(text))
    monkeypatch.setattr(app_runtime.st, "write", lambda *args, **kwargs: None)
    monkeypatch.setattr(app_runtime.st, "json", lambda *args, **kwargs: None)
    monkeypatch.setattr(app_runtime.st, "caption", lambda *args, **kwargs: None)
    monkeypatch.setattr(app_runtime.st, "error", lambda *args, **kwargs: None)
    monkeypatch.setattr(app_runtime.st, "code", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        app_runtime.st,
        "expander",
        lambda label, expanded=False: expanders.append(label) or DummyContextManager(),
    )
    monkeypatch.setattr(app_runtime, "_render_tool_trace", lambda trace: expanders.append("ToolTraceRendered"))

    app_runtime._render_assistant_message(FakeCopilot().run("Create service"))

    assert "**Overall status:** `ready`" in markdown_calls
    assert "Supporting Step: analyze_requirement_text" in expanders
    assert "Tool Result: extract_configuration_parameters" in expanders
    assert "Tool Result: retrieve_reference_context" in expanders
    assert "Tool Result: validate_configuration_json" in expanders
    assert "Tool Call Trace" in expanders


def test_run_app_stops_with_visible_error_when_api_key_is_missing(monkeypatch) -> None:
    errors: list[str] = []

    class StopCalled(Exception):
        pass

    monkeypatch.setattr(app_runtime, "load_local_env", lambda: None)
    monkeypatch.setattr(app_runtime, "_get_api_key", lambda: None)
    monkeypatch.setattr(app_runtime.st, "session_state", SessionState())
    monkeypatch.setattr(app_runtime.st, "set_page_config", lambda **kwargs: None)
    monkeypatch.setattr(app_runtime.st, "title", lambda *args, **kwargs: None)
    monkeypatch.setattr(app_runtime.st, "caption", lambda *args, **kwargs: None)
    monkeypatch.setattr(app_runtime.st, "sidebar", DummyContextManager())
    monkeypatch.setattr(app_runtime.st, "header", lambda *args, **kwargs: None)
    monkeypatch.setattr(app_runtime.st, "selectbox", lambda *args, **kwargs: app_runtime.DEFAULT_MODEL_NAME)
    monkeypatch.setattr(app_runtime.st, "markdown", lambda *args, **kwargs: None)
    monkeypatch.setattr(app_runtime.st, "write", lambda *args, **kwargs: None)
    monkeypatch.setattr(app_runtime.st, "button", lambda *args, **kwargs: False)
    monkeypatch.setattr(app_runtime.st, "error", lambda message: errors.append(message))
    monkeypatch.setattr(app_runtime.st, "stop", lambda: (_ for _ in ()).throw(StopCalled()))
    monkeypatch.setattr(app_runtime, "_render_kb_update_panel", lambda copilot: None)

    with pytest.raises(StopCalled):
        app_runtime.run_config_copilot_app()

    assert errors == ["OPENAI_API_KEY is required. Add it to Streamlit secrets or /workspace/.env."]


def test_run_app_uses_spinner_and_appends_chat_history(monkeypatch) -> None:
    spinner_messages: list[str] = []
    fake_copilot = FakeCopilot()
    session_state = SessionState()

    @contextmanager
    def spinner(message: str):
        spinner_messages.append(message)
        yield

    monkeypatch.setattr(app_runtime, "load_local_env", lambda: None)
    monkeypatch.setattr(app_runtime, "_get_api_key", lambda: "test-key")
    monkeypatch.setattr(app_runtime, "_get_copilot", lambda model_name, api_key: fake_copilot)
    monkeypatch.setattr(app_runtime.st, "session_state", session_state)
    monkeypatch.setattr(app_runtime.st, "set_page_config", lambda **kwargs: None)
    monkeypatch.setattr(app_runtime.st, "title", lambda *args, **kwargs: None)
    monkeypatch.setattr(app_runtime.st, "caption", lambda *args, **kwargs: None)
    monkeypatch.setattr(app_runtime.st, "sidebar", DummyContextManager())
    monkeypatch.setattr(app_runtime.st, "header", lambda *args, **kwargs: None)
    monkeypatch.setattr(app_runtime.st, "selectbox", lambda *args, **kwargs: app_runtime.DEFAULT_MODEL_NAME)
    monkeypatch.setattr(app_runtime.st, "markdown", lambda *args, **kwargs: None)
    monkeypatch.setattr(app_runtime.st, "write", lambda *args, **kwargs: None)
    monkeypatch.setattr(app_runtime.st, "button", lambda *args, **kwargs: False)
    monkeypatch.setattr(app_runtime.st, "chat_input", lambda *args, **kwargs: "Create a wheel speed service.")
    monkeypatch.setattr(app_runtime.st, "chat_message", lambda *args, **kwargs: DummyContextManager())
    monkeypatch.setattr(app_runtime.st, "spinner", spinner)
    monkeypatch.setattr(app_runtime.st, "warning", lambda *args, **kwargs: None)
    monkeypatch.setattr(app_runtime.st, "error", lambda *args, **kwargs: None)
    monkeypatch.setattr(app_runtime, "_render_assistant_message", lambda result: None)
    monkeypatch.setattr(app_runtime, "_render_kb_update_panel", lambda copilot: None)

    app_runtime.run_config_copilot_app()

    assert spinner_messages == [
        "Running tool calls: retrieve context, draft configuration JSON, validate output..."
    ]
    assert fake_copilot.prompts == ["Create a wheel speed service."]
    assert len(session_state.chat_history) == 2


def test_render_kb_update_panel_imports_document_and_reports_success(monkeypatch) -> None:
    fake_copilot = FakeCopilot()
    json_payloads: list[dict] = []
    success_messages: list[str] = []

    class UploadedDocument:
        name = "sample_standard.md"

        @staticmethod
        def getvalue() -> bytes:
            return b"# sample standard\n"

    monkeypatch.setattr(app_runtime.st, "divider", lambda: None)
    monkeypatch.setattr(app_runtime.st, "header", lambda *args, **kwargs: None)
    monkeypatch.setattr(app_runtime.st, "caption", lambda *args, **kwargs: None)
    monkeypatch.setattr(app_runtime.st, "file_uploader", lambda *args, **kwargs: UploadedDocument())
    monkeypatch.setattr(
        app_runtime.st,
        "button",
        lambda *args, **kwargs: kwargs.get("key") == "kb-import::sample_standard.md",
    )
    monkeypatch.setattr(
        app_runtime.st,
        "columns",
        lambda *args, **kwargs: [DummyContextManager(), DummyContextManager(), DummyContextManager()],
    )
    monkeypatch.setattr(app_runtime.st, "write", lambda *args, **kwargs: None)
    monkeypatch.setattr(app_runtime.st, "markdown", lambda *args, **kwargs: None)
    monkeypatch.setattr(app_runtime.st, "spinner", lambda *args, **kwargs: DummyContextManager())
    monkeypatch.setattr(app_runtime.st, "success", lambda message: success_messages.append(message))
    monkeypatch.setattr(app_runtime.st, "json", lambda payload: json_payloads.append(payload))
    monkeypatch.setattr(app_runtime.st, "error", lambda *args, **kwargs: None)

    app_runtime._render_kb_update_panel(fake_copilot)

    assert success_messages == ["Knowledge base updated successfully."]
    assert json_payloads[-1]["document_count"] == 3


def test_render_kb_update_panel_resets_vectorstore(monkeypatch) -> None:
    fake_copilot = FakeCopilot()
    json_payloads: list[dict] = []
    success_messages: list[str] = []

    monkeypatch.setattr(app_runtime.st, "divider", lambda: None)
    monkeypatch.setattr(app_runtime.st, "header", lambda *args, **kwargs: None)
    monkeypatch.setattr(app_runtime.st, "caption", lambda *args, **kwargs: None)
    monkeypatch.setattr(app_runtime.st, "file_uploader", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        app_runtime.st,
        "button",
        lambda *args, **kwargs: kwargs.get("key") == "kb-reset",
    )
    monkeypatch.setattr(
        app_runtime.st,
        "columns",
        lambda *args, **kwargs: [DummyContextManager(), DummyContextManager(), DummyContextManager()],
    )
    monkeypatch.setattr(app_runtime.st, "write", lambda *args, **kwargs: None)
    monkeypatch.setattr(app_runtime.st, "markdown", lambda *args, **kwargs: None)
    monkeypatch.setattr(app_runtime.st, "spinner", lambda *args, **kwargs: DummyContextManager())
    monkeypatch.setattr(app_runtime.st, "success", lambda message: success_messages.append(message))
    monkeypatch.setattr(app_runtime.st, "json", lambda payload: json_payloads.append(payload))
    monkeypatch.setattr(app_runtime.st, "error", lambda *args, **kwargs: None)

    app_runtime._render_kb_update_panel(fake_copilot)

    assert success_messages == ["Knowledge base reset completed."]
    assert json_payloads[-1]["collection_name"] == "automotive_config_standards"


def test_render_kb_update_panel_deletes_document_and_reports_success(monkeypatch) -> None:
    fake_copilot = FakeCopilot()
    json_payloads: list[dict] = []
    success_messages: list[str] = []

    monkeypatch.setattr(app_runtime.st, "divider", lambda: None)
    monkeypatch.setattr(app_runtime.st, "header", lambda *args, **kwargs: None)
    monkeypatch.setattr(app_runtime.st, "caption", lambda *args, **kwargs: None)
    monkeypatch.setattr(app_runtime.st, "file_uploader", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        app_runtime.st,
        "button",
        lambda *args, **kwargs: kwargs.get("key") == "kb-delete::autosar_error_handling.md",
    )
    monkeypatch.setattr(
        app_runtime.st,
        "columns",
        lambda *args, **kwargs: [DummyContextManager(), DummyContextManager(), DummyContextManager()],
    )
    monkeypatch.setattr(app_runtime.st, "write", lambda *args, **kwargs: None)
    monkeypatch.setattr(app_runtime.st, "markdown", lambda *args, **kwargs: None)
    monkeypatch.setattr(app_runtime.st, "spinner", lambda *args, **kwargs: DummyContextManager())
    monkeypatch.setattr(app_runtime.st, "success", lambda message: success_messages.append(message))
    monkeypatch.setattr(app_runtime.st, "json", lambda payload: json_payloads.append(payload))
    monkeypatch.setattr(app_runtime.st, "error", lambda *args, **kwargs: None)

    app_runtime._render_kb_update_panel(fake_copilot)

    assert success_messages == ["Deleted autosar_error_handling.md from the knowledge base."]
    assert json_payloads[-1]["deleted_documents"] == ["autosar_error_handling.md"]
