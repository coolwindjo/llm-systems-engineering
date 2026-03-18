from __future__ import annotations

from services.schemas import ConfigurationDraft, RequirementAnalysis, RetrievedStandardChunk
from services.validation import build_validation_result


def test_validation_marks_cyclic_without_frequency_as_incomplete() -> None:
    analysis = RequirementAnalysis(
        summary="A cyclic service was requested without explicit timing.",
        contradictions=[],
        ambiguities=["frequency not fully specified"],
        missing_information=["frequency"],
        follow_up_questions=[],
        risk_level="medium",
    )
    extraction = ConfigurationDraft(
        ServiceName="WheelSpeedBroadcast",
        ID=0x120,
        Class="Event",
        Frequency=None,
        PlayType="Cyclic",
    )
    chunks = [
        RetrievedStandardChunk(
            source="autosar_error_handling.md",
            title="Autosar Error Handling",
            query="WheelSpeedBroadcast timing",
            excerpt="For cyclic publication, the sender should document the nominal update period.",
        )
    ]

    validation = build_validation_result(
        requirement_text="Create a cyclic wheel speed broadcast service.",
        analysis=analysis,
        extraction=extraction,
        retrieved_chunks=chunks,
    )

    assert validation.status == "incomplete"
    assert "Frequency" in validation.missing_required_fields
    assert validation.schema_valid is False


def test_validation_marks_conflicting_trigger_hints_for_review() -> None:
    analysis = RequirementAnalysis(
        summary="The requirement mixes cyclic and on-change behavior.",
        contradictions=["cyclic timing conflicts with on-change behavior"],
        ambiguities=[],
        missing_information=[],
        follow_up_questions=["Should the service be cyclic or on-change?"],
        risk_level="high",
    )
    extraction = ConfigurationDraft(
        ServiceName="TorqueCommandSync",
        ID=42,
        Class="Event",
        Frequency="10 ms",
        PlayType="Cyclic",
    )

    validation = build_validation_result(
        requirement_text="Configure TorqueCommandSync as cyclic every 10 ms, but also say it should be on-change.",
        analysis=analysis,
        extraction=extraction,
        retrieved_chunks=[],
    )

    assert validation.status == "needs_review"
    assert any("cyclic and on-change" in finding for finding in validation.basic_rule_findings)


def test_validation_marks_request_response_class_mismatch_for_review() -> None:
    analysis = RequirementAnalysis(
        summary="The request implies on-demand RPC behavior.",
        contradictions=[],
        ambiguities=[],
        missing_information=[],
        follow_up_questions=[],
        risk_level="medium",
    )
    extraction = ConfigurationDraft(
        ServiceName="VinReadService",
        ID=0x220,
        Class="Event",
        Frequency=None,
        PlayType="OnRequest",
    )

    validation = build_validation_result(
        requirement_text="Build a request-response service for VIN readout on demand.",
        analysis=analysis,
        extraction=extraction,
        retrieved_chunks=[],
    )

    assert validation.status == "needs_review"
    assert any("request/response behavior" in finding for finding in validation.basic_rule_findings)


def test_validation_marks_clean_complete_result_as_ready() -> None:
    analysis = RequirementAnalysis(
        summary="The requirement is explicit and complete.",
        contradictions=[],
        ambiguities=[],
        missing_information=[],
        follow_up_questions=[],
        risk_level="low",
    )
    extraction = ConfigurationDraft(
        ServiceName="BatteryStateField",
        ID=0x98,
        Class="Field",
        Frequency=None,
        PlayType="OnChange",
    )
    chunks = [
        RetrievedStandardChunk(
            source="project_notes.md",
            title="Project Notes",
            query="battery state field on-change",
            excerpt="Readable state fields can use on-change publication if timing is event-driven.",
        )
    ]

    validation = build_validation_result(
        requirement_text="Define BatteryStateField as an on-change field service with id 0x98.",
        analysis=analysis,
        extraction=extraction,
        retrieved_chunks=chunks,
    )

    assert validation.status == "ready"
    assert validation.schema_valid is True
    assert validation.missing_required_fields == []
