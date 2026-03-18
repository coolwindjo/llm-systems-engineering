from __future__ import annotations

import json

from services.copilot import build_retrieval_tool
from services.extraction import build_analysis_tool, build_extraction_tool
from services.schemas import ConfigurationDraft, RequirementAnalysis, RetrievedStandardChunk
from services.validation import build_validation_tool


class DummyKnowledgeBase:
    def retrieve(self, requirement_text: str, analysis: RequirementAnalysis, extraction=None):
        assert requirement_text.startswith("Create")
        assert analysis.summary == "Safe summary"
        assert extraction is None
        return [
            "AUTOSAR wheel speed timing",
            "project-specific broadcast rules",
        ], [
            RetrievedStandardChunk(
                source="standards_reference.md",
                title="Standards Reference",
                query="AUTOSAR wheel speed timing",
                excerpt="Cyclic wheel speed services should declare a nominal period.",
                score=0.12,
            )
        ]


class FakeStructuredLLM:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def invoke(self, prompt: str) -> ConfigurationDraft:
        self.prompts.append(prompt)
        return ConfigurationDraft(
            ServiceName="WheelSpeedBroadcast",
            ID="0x120",
            Class="signal",
            Frequency="10ms",
            PlayType="periodic",
        )


class FakeLLM:
    def __init__(self, structured_llm: FakeStructuredLLM) -> None:
        self.structured_llm = structured_llm

    def with_structured_output(self, _schema):
        return self.structured_llm


def test_retrieve_reference_context_tool_has_reviewer_friendly_name_and_payload() -> None:
    tool = build_retrieval_tool(DummyKnowledgeBase())
    analysis = RequirementAnalysis(
        summary="Safe summary",
        contradictions=[],
        ambiguities=[],
        missing_information=[],
        follow_up_questions=[],
        risk_level="low",
    )

    payload = json.loads(
        tool.invoke(
            {
                "requirement_text": "Create a wheel speed broadcast service.",
                "analysis_json": analysis.model_dump_json(indent=2),
            }
        )
    )

    assert tool.name == "retrieve_reference_context"
    assert payload["translated_queries"] == [
        "AUTOSAR wheel speed timing",
        "project-specific broadcast rules",
    ]
    assert payload["retrieved_chunks"][0]["title"] == "Standards Reference"


def test_analyze_requirement_text_tool_exposes_expected_name() -> None:
    tool = build_analysis_tool(model_name="gpt-4o-mini", api_key="test-key")

    assert tool.name == "analyze_requirement_text"


def test_extract_configuration_parameters_tool_uses_context_and_analysis(monkeypatch) -> None:
    structured_llm = FakeStructuredLLM()
    monkeypatch.setattr(
        "services.extraction.create_chat_model",
        lambda model_name, api_key: FakeLLM(structured_llm),
    )
    tool = build_extraction_tool(model_name="gpt-4o-mini", api_key="test-key")
    analysis = RequirementAnalysis(
        summary="The request is mostly clear but timing must stay cyclic.",
        contradictions=[],
        ambiguities=[],
        missing_information=[],
        follow_up_questions=[],
        risk_level="low",
    )
    retrieved_context_json = json.dumps(
        [
            {
                "title": "Standards Timing Note",
                "source": "standards_reference.md",
                "excerpt": "Wheel speed broadcasts are usually periodic at a fixed interval.",
            }
        ]
    )

    output = ConfigurationDraft.model_validate_json(
        tool.invoke(
            {
                "requirement_text": "Create a cyclic wheel speed service every 10 ms.",
                "retrieved_context_json": retrieved_context_json,
                "analysis_json": analysis.model_dump_json(indent=2),
            }
        )
    )

    assert tool.name == "extract_configuration_parameters"
    assert output.ServiceName == "WheelSpeedBroadcast"
    assert output.ID == 0x120
    assert output.Class == "Event"
    assert output.PlayType == "Cyclic"
    assert "Extract service configuration fields from the requirement." in structured_llm.prompts[0]
    assert "Standards Timing Note" in structured_llm.prompts[0]
    assert "Analysis summary" in structured_llm.prompts[0]


def test_validate_configuration_json_tool_returns_structured_validation_result() -> None:
    tool = build_validation_tool()
    analysis = RequirementAnalysis(
        summary="The request is missing timing information.",
        contradictions=[],
        ambiguities=["timing is implicit"],
        missing_information=["frequency"],
        follow_up_questions=["What frequency should be used?"],
        risk_level="medium",
    )
    extraction = ConfigurationDraft(
        ServiceName="WheelSpeedBroadcast",
        ID=0x120,
        Class="Event",
        Frequency=None,
        PlayType="Cyclic",
    )
    retrieved_context_json = json.dumps(
        [
            {
                "source": "standards_reference.md",
                "title": "Standards Reference",
                "query": "wheel speed cyclic timing",
                "excerpt": "Cyclic services should define a nominal period.",
                "score": 0.2,
            }
        ]
    )

    validation = json.loads(
        tool.invoke(
            {
                "config_json": extraction.model_dump_json(indent=2),
                "requirement_text": "Create a cyclic wheel speed broadcast service.",
                "analysis_json": analysis.model_dump_json(indent=2),
                "retrieved_context_json": retrieved_context_json,
            }
        )
    )

    assert tool.name == "validate_configuration_json"
    assert validation["status"] == "incomplete"
    assert validation["schema_valid"] is False
    assert "Frequency" in validation["missing_required_fields"]
