from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import tool

from services.schemas import ConfigurationDraft, RequirementAnalysis, RetrievedStandardChunk, StandardsValidation


def _has_any(text: str, phrases: list[str]) -> bool:
    lowered = text.lower()
    return any(phrase in lowered for phrase in phrases)


def build_validation_result(
    *,
    requirement_text: str,
    analysis: RequirementAnalysis,
    extraction: ConfigurationDraft,
    retrieved_chunks: list[RetrievedStandardChunk],
) -> StandardsValidation:
    missing_required_fields: list[str] = []
    basic_rule_findings: list[str] = []
    suggested_actions: list[str] = []
    follow_up_questions = list(analysis.follow_up_questions)
    referenced_sections = []

    for field_name in ["ServiceName", "ID", "Class", "PlayType"]:
        value = getattr(extraction, field_name)
        if value in [None, "", "Unknown"]:
            missing_required_fields.append(field_name)

    if extraction.PlayType == "Cyclic" and not extraction.Frequency:
        missing_required_fields.append("Frequency")
        basic_rule_findings.append(
            "PlayType is Cyclic but Frequency is missing, so timing cannot be verified."
        )

    if analysis.contradictions:
        for contradiction in analysis.contradictions:
            basic_rule_findings.append(f"Requirement contradiction detected: {contradiction}")

    lowered = requirement_text.lower()
    mentions_cyclic = _has_any(lowered, ["cyclic", "periodic", " every ", " hz", " ms"])
    mentions_on_change = _has_any(lowered, ["on-change", "on change", "event-driven"])
    mentions_one_shot = _has_any(lowered, ["one-shot", "one shot", "single trigger"])
    mentions_request = _has_any(lowered, ["on-request", "on request", "request", "rpc", "response"])
    mentions_broadcast = _has_any(lowered, ["broadcast", "publish", "signal"])

    if mentions_cyclic and mentions_on_change:
        basic_rule_findings.append(
            "The requirement mixes cyclic and on-change timing hints; one primary triggering mode should be chosen."
        )

    if mentions_cyclic and extraction.PlayType not in [None, "Cyclic"]:
        basic_rule_findings.append(
            "The text contains cyclic timing hints, but the extracted PlayType is not Cyclic."
        )

    if mentions_on_change and extraction.PlayType not in [None, "OnChange"]:
        basic_rule_findings.append(
            "The text contains on-change hints, but the extracted PlayType is not OnChange."
        )

    if mentions_one_shot and extraction.PlayType not in [None, "OneShot"]:
        basic_rule_findings.append(
            "The text contains one-shot hints, but the extracted PlayType is not OneShot."
        )

    if mentions_request and extraction.Class == "Event":
        basic_rule_findings.append(
            "The text suggests request/response behavior, but the extracted Class is Event."
        )

    if mentions_broadcast and extraction.Class == "Method":
        basic_rule_findings.append(
            "The text suggests broadcast/publication behavior, but the extracted Class is Method."
        )

    if analysis.missing_information:
        for missing_item in analysis.missing_information:
            suggested_actions.append(f"Clarify missing input: {missing_item}")

    if missing_required_fields:
        for field_name in missing_required_fields:
            suggested_actions.append(f"Provide or correct `{field_name}` before accepting the JSON draft.")

    for chunk in retrieved_chunks[:4]:
        if chunk.title not in referenced_sections:
            referenced_sections.append(chunk.title)

    reference_notes = []
    for chunk in retrieved_chunks[:3]:
        reference_notes.append(
            f"Reference context from '{chunk.title}' should be reviewed for timing, triggering, and fallback wording."
        )

    if missing_required_fields:
        status = "incomplete"
        summary = (
            "The configuration JSON draft is incomplete. Required fields are still missing or unresolved and should be clarified before use."
        )
    elif basic_rule_findings or analysis.ambiguities:
        status = "needs_review"
        summary = (
            "The configuration JSON draft was generated, but it still needs review because ambiguities or rule-check findings remain."
        )
    else:
        status = "ready"
        summary = (
            "The configuration JSON draft is structurally complete and passed the current MVP-level rule checks."
        )

    if not follow_up_questions and missing_required_fields:
        follow_up_questions = [
            f"What is the intended value for `{field_name}`?"
            for field_name in missing_required_fields[:3]
        ]

    return StandardsValidation(
        status=status,
        summary=summary,
        schema_valid=not missing_required_fields,
        missing_required_fields=missing_required_fields,
        basic_rule_findings=basic_rule_findings,
        compliance_notes=reference_notes,
        suggested_actions=suggested_actions,
        follow_up_questions=follow_up_questions,
        referenced_sections=referenced_sections,
    )


def build_validation_tool():
    @tool
    def validate_configuration_json(
        config_json: str,
        requirement_text: str,
        analysis_json: str,
        retrieved_context_json: str,
    ) -> str:
        """Validate a configuration JSON draft with Pydantic parsing and basic rule checks."""
        extraction = ConfigurationDraft.model_validate_json(config_json)
        analysis = RequirementAnalysis.model_validate_json(analysis_json)
        raw_chunks = json.loads(retrieved_context_json)
        retrieved_chunks = [RetrievedStandardChunk.model_validate(item) for item in raw_chunks]
        validation = build_validation_result(
            requirement_text=requirement_text,
            analysis=analysis,
            extraction=extraction,
            retrieved_chunks=retrieved_chunks,
        )
        return validation.model_dump_json(indent=2)

    return validate_configuration_json
