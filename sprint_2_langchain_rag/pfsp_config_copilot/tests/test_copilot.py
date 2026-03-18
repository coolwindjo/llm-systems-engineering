from __future__ import annotations

import json

from services.copilot import ConfigurationCopilot
from services.schemas import ConfigurationDraft, RequirementAnalysis, RetrievedStandardChunk, StandardsValidation


class DummyKnowledgeBase:
    def __init__(self, *, chunks: list[RetrievedStandardChunk] | None = None, error: Exception | None = None) -> None:
        self.chunks = chunks or []
        self.error = error

    def retrieve(self, requirement_text: str, analysis: RequirementAnalysis, extraction=None):
        if self.error:
            raise self.error
        return ["AUTOSAR timing note"], list(self.chunks)


class DummyValidationTool:
    name = "validate_configuration_json"

    def __init__(self, payload: str | None = None, *, error: Exception | None = None) -> None:
        self.payload = payload
        self.error = error

    def invoke(self, _payload: dict) -> str:
        if self.error:
            raise self.error
        assert self.payload is not None
        return self.payload


def _analysis() -> RequirementAnalysis:
    return RequirementAnalysis(
        summary="The requirement is clear enough for extraction.",
        contradictions=[],
        ambiguities=[],
        missing_information=[],
        follow_up_questions=[],
        risk_level="low",
    )


def _analysis_trace() -> dict:
    return {
        "tool": "analyze_requirement_text",
        "status": "success",
        "purpose": "Summarize ambiguity, contradictions, and missing inputs before extraction.",
        "input": {"requirement_text": "Create a wheel speed service."},
        "output_summary": {
            "risk_level": "low",
            "ambiguity_count": 0,
            "contradiction_count": 0,
            "missing_information_count": 0,
        },
        "output": _analysis().model_dump(),
    }


def _extraction() -> ConfigurationDraft:
    return ConfigurationDraft(
        ServiceName="WheelSpeedBroadcast",
        ID=0x120,
        Class="Event",
        Frequency="10 ms",
        PlayType="Cyclic",
    )


def _extraction_trace() -> dict:
    extraction = _extraction()
    return {
        "tool": "extract_configuration_parameters",
        "status": "success",
        "purpose": "Generate the configuration JSON draft from the requirement and retrieved reference context.",
        "input": {
            "requirement_text": "Create a wheel speed service.",
            "analysis_summary": "The requirement is clear enough for extraction.",
        },
        "output_summary": {
            "service_name": extraction.ServiceName,
            "class": extraction.Class,
            "play_type": extraction.PlayType,
            "missing_fields": [],
        },
        "output": extraction.model_dump(),
    }


def _validation(status: str = "ready") -> StandardsValidation:
    return StandardsValidation(
        status=status,
        summary="Validation completed.",
        schema_valid=(status != "incomplete"),
        missing_required_fields=[] if status != "incomplete" else ["Frequency"],
        basic_rule_findings=[],
        compliance_notes=["Check timing guidance."],
        suggested_actions=[],
        follow_up_questions=[],
        referenced_sections=["Standards Reference"],
    )


def test_copilot_rejects_empty_input() -> None:
    copilot = ConfigurationCopilot(model_name="gpt-4o-mini", api_key="test-key", kb=DummyKnowledgeBase())

    result = copilot.run("   ")

    assert result["error"]["stage"] == "input_validation"
    assert result["tool_trace"] == []


def test_copilot_success_path_returns_structured_result(monkeypatch) -> None:
    chunks = [
        RetrievedStandardChunk(
            source="standards_reference.md",
            title="Standards Reference",
            query="AUTOSAR timing note",
            excerpt="Wheel speed services can publish every 10 ms.",
            score=0.1,
        )
    ]
    kb = DummyKnowledgeBase(chunks=chunks)
    validation = _validation()
    copilot = ConfigurationCopilot(model_name="gpt-4o-mini", api_key="test-key", kb=kb)

    monkeypatch.setattr("services.copilot.run_analysis_tool", lambda *args, **kwargs: (_analysis(), _analysis_trace()))
    monkeypatch.setattr(
        "services.copilot.run_extraction_tool",
        lambda *args, **kwargs: (_extraction(), _extraction_trace()),
    )
    monkeypatch.setattr(
        "services.copilot.build_validation_tool",
        lambda: DummyValidationTool(validation.model_dump_json(indent=2)),
    )

    result = copilot.run("Create a wheel speed service.")

    assert result["validation"]["status"] == "ready"
    assert result["translated_queries"] == ["AUTOSAR timing note"]
    assert [step["tool"] for step in result["tool_trace"]] == [
        "analyze_requirement_text",
        "retrieve_reference_context",
        "extract_configuration_parameters",
        "validate_configuration_json",
    ]
    assert result["retrieved_chunks"][0]["title"] == "Standards Reference"


def test_copilot_returns_analysis_error(monkeypatch) -> None:
    copilot = ConfigurationCopilot(model_name="gpt-4o-mini", api_key="test-key", kb=DummyKnowledgeBase())
    monkeypatch.setattr(
        "services.copilot.run_analysis_tool",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("analysis failed")),
    )

    result = copilot.run("Create a wheel speed service.")

    assert result["error"]["stage"] == "analyze_requirement_text"
    assert result["tool_trace"] == []


def test_copilot_returns_retrieval_error(monkeypatch) -> None:
    kb = DummyKnowledgeBase(error=RuntimeError("chroma unavailable"))
    copilot = ConfigurationCopilot(model_name="gpt-4o-mini", api_key="test-key", kb=kb)
    monkeypatch.setattr("services.copilot.run_analysis_tool", lambda *args, **kwargs: (_analysis(), _analysis_trace()))

    result = copilot.run("Create a wheel speed service.")

    assert result["error"]["stage"] == "retrieve_reference_context"
    assert result["tool_trace"][0]["tool"] == "analyze_requirement_text"


def test_copilot_returns_error_when_retrieval_has_no_chunks(monkeypatch) -> None:
    copilot = ConfigurationCopilot(model_name="gpt-4o-mini", api_key="test-key", kb=DummyKnowledgeBase(chunks=[]))
    monkeypatch.setattr("services.copilot.run_analysis_tool", lambda *args, **kwargs: (_analysis(), _analysis_trace()))

    result = copilot.run("Create a wheel speed service.")

    assert result["error"]["stage"] == "retrieve_reference_context"
    assert result["error"]["detail"] == "The retrieval step returned zero chunks."


def test_copilot_returns_extraction_error(monkeypatch) -> None:
    chunks = [
        RetrievedStandardChunk(
            source="standards_reference.md",
            title="Standards Reference",
            query="AUTOSAR timing note",
            excerpt="Wheel speed services can publish every 10 ms.",
            score=0.1,
        )
    ]
    copilot = ConfigurationCopilot(model_name="gpt-4o-mini", api_key="test-key", kb=DummyKnowledgeBase(chunks=chunks))
    monkeypatch.setattr("services.copilot.run_analysis_tool", lambda *args, **kwargs: (_analysis(), _analysis_trace()))
    monkeypatch.setattr(
        "services.copilot.run_extraction_tool",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("extract failed")),
    )

    result = copilot.run("Create a wheel speed service.")

    assert result["error"]["stage"] == "extract_configuration_parameters"
    assert [step["tool"] for step in result["tool_trace"]] == [
        "analyze_requirement_text",
        "retrieve_reference_context",
    ]


def test_copilot_returns_validation_error(monkeypatch) -> None:
    chunks = [
        RetrievedStandardChunk(
            source="standards_reference.md",
            title="Standards Reference",
            query="AUTOSAR timing note",
            excerpt="Wheel speed services can publish every 10 ms.",
            score=0.1,
        )
    ]
    copilot = ConfigurationCopilot(model_name="gpt-4o-mini", api_key="test-key", kb=DummyKnowledgeBase(chunks=chunks))
    monkeypatch.setattr("services.copilot.run_analysis_tool", lambda *args, **kwargs: (_analysis(), _analysis_trace()))
    monkeypatch.setattr(
        "services.copilot.run_extraction_tool",
        lambda *args, **kwargs: (_extraction(), _extraction_trace()),
    )
    monkeypatch.setattr(
        "services.copilot.build_validation_tool",
        lambda: DummyValidationTool(error=RuntimeError("validation failed")),
    )

    result = copilot.run("Create a wheel speed service.")

    assert result["error"]["stage"] == "validate_configuration_json"
    assert [step["tool"] for step in result["tool_trace"]] == [
        "analyze_requirement_text",
        "retrieve_reference_context",
        "extract_configuration_parameters",
    ]
