from __future__ import annotations

import json

from langchain_core.tools import tool

from services.config import get_logger
from services.extraction import run_analysis_tool, run_extraction_tool
from services.knowledge_base import StandardsKnowledgeBase
from services.schemas import (
    CopilotResult,
    RequirementAnalysis,
    RetrievedStandardChunk,
)
from services.validation import build_validation_tool


LOGGER = get_logger("copilot")


def build_retrieval_tool(kb: StandardsKnowledgeBase):
    @tool
    def retrieve_reference_context(
        requirement_text: str,
        analysis_json: str,
    ) -> str:
        """Retrieve standards and reference context with translated search queries."""
        analysis = RequirementAnalysis.model_validate_json(analysis_json)
        translated_queries, chunks = kb.retrieve(requirement_text, analysis, extraction=None)
        payload = {
            "translated_queries": translated_queries,
            "retrieved_chunks": [chunk.model_dump() for chunk in chunks],
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)

    return retrieve_reference_context


def _build_error_result(stage: str, user_message: str, tool_trace: list[dict], detail: str) -> dict:
    return {
        "error": {
            "stage": stage,
            "message": user_message,
            "detail": detail,
        },
        "tool_trace": tool_trace,
    }


class ConfigurationCopilot:
    def __init__(self, *, model_name: str, api_key: str, kb: StandardsKnowledgeBase) -> None:
        self.model_name = model_name
        self.api_key = api_key
        self.kb = kb

    def run(self, requirement_text: str) -> dict:
        requirement_text = requirement_text.strip()
        tool_trace: list[dict] = []

        if not requirement_text:
            LOGGER.warning("request_rejected empty_requirement_text")
            return _build_error_result(
                "input_validation",
                "The requirement text is empty. Enter a configuration requirement before running the copilot.",
                tool_trace,
                "Empty or whitespace-only input was received.",
            )

        LOGGER.info("request_started model=%s chars=%s", self.model_name, len(requirement_text))

        try:
            analysis, analysis_trace = run_analysis_tool(
                requirement_text,
                model_name=self.model_name,
                api_key=self.api_key,
            )
            tool_trace.append(analysis_trace)
            LOGGER.info(
                "tool_completed tool=%s ambiguity_count=%s contradiction_count=%s",
                analysis_trace["tool"],
                analysis_trace["output_summary"]["ambiguity_count"],
                analysis_trace["output_summary"]["contradiction_count"],
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("tool_failed tool=analyze_requirement_text")
            return _build_error_result(
                "analyze_requirement_text",
                "The app could not analyze the requirement text. Try simplifying the wording and retry.",
                tool_trace,
                str(exc),
            )

        try:
            retrieval_tool = build_retrieval_tool(self.kb)
            retrieval_output = retrieval_tool.invoke(
                {
                    "requirement_text": requirement_text,
                    "analysis_json": analysis.model_dump_json(indent=2),
                }
            )
            parsed_search = json.loads(retrieval_output)
            translated_queries = parsed_search["translated_queries"]
            retrieved_chunks = [
                RetrievedStandardChunk.model_validate(item)
                for item in parsed_search["retrieved_chunks"]
            ]
            retrieval_trace = {
                "tool": retrieval_tool.name,
                "status": "success",
                "purpose": "Retrieve standards and reference context using query translation and similarity search.",
                "input": {
                    "requirement_text": requirement_text,
                    "analysis_summary": analysis.summary,
                },
                "output_summary": {
                    "translated_query_count": len(translated_queries),
                    "retrieved_chunk_count": len(retrieved_chunks),
                },
                "output": parsed_search,
            }
            tool_trace.append(retrieval_trace)
            LOGGER.info(
                "tool_completed tool=%s translated_queries=%s retrieved_chunks=%s",
                retrieval_trace["tool"],
                len(translated_queries),
                len(retrieved_chunks),
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("tool_failed tool=retrieve_reference_context")
            return _build_error_result(
                "retrieve_reference_context",
                "Reference retrieval failed. Check the local knowledge base and API setup, then retry.",
                tool_trace,
                str(exc),
            )

        if not retrieved_chunks:
            LOGGER.warning("retrieval_returned_no_chunks")
            return _build_error_result(
                "retrieve_reference_context",
                "No reference context was retrieved for this requirement. Try a more specific request with timing, class, or validation details.",
                tool_trace,
                "The retrieval step returned zero chunks.",
            )

        try:
            retrieved_context_json = json.dumps(
                [chunk.model_dump() for chunk in retrieved_chunks],
                ensure_ascii=False,
                indent=2,
            )
            extraction, extraction_trace = run_extraction_tool(
                requirement_text,
                analysis,
                retrieved_context_json,
                model_name=self.model_name,
                api_key=self.api_key,
            )
            tool_trace.append(extraction_trace)
            LOGGER.info(
                "tool_completed tool=%s service=%s missing_fields=%s",
                extraction_trace["tool"],
                extraction.ServiceName,
                extraction.unresolved_fields(),
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("tool_failed tool=extract_configuration_parameters")
            return _build_error_result(
                "extract_configuration_parameters",
                "The app could not generate a configuration JSON draft from the current input and reference context.",
                tool_trace,
                str(exc),
            )

        try:
            validation_tool = build_validation_tool()
            validation_output = validation_tool.invoke(
                {
                    "config_json": extraction.model_dump_json(indent=2),
                    "requirement_text": requirement_text,
                    "analysis_json": analysis.model_dump_json(indent=2),
                    "retrieved_context_json": retrieved_context_json,
                }
            )
            validation = self._parse_validation_output(validation_output)
            validation_trace = {
                "tool": validation_tool.name,
                "status": "success",
                "purpose": "Validate the configuration JSON draft with schema parsing and basic rule checks.",
                "input": {
                    "requirement_text": requirement_text,
                    "config_preview": extraction.model_dump(),
                },
                "output_summary": {
                    "status": validation.status,
                    "schema_valid": validation.schema_valid,
                    "missing_required_fields": validation.missing_required_fields,
                    "basic_rule_findings_count": len(validation.basic_rule_findings),
                },
                "output": json.loads(validation_output),
            }
            tool_trace.append(validation_trace)
            LOGGER.info(
                "tool_completed tool=%s status=%s schema_valid=%s",
                validation_trace["tool"],
                validation.status,
                validation.schema_valid,
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("tool_failed tool=validate_configuration_json")
            return _build_error_result(
                "validate_configuration_json",
                "The configuration JSON draft was generated, but validation could not be completed.",
                tool_trace,
                str(exc),
            )

        result = CopilotResult(
            analysis=analysis,
            extraction=extraction,
            translated_queries=translated_queries,
            retrieved_chunks=retrieved_chunks,
            validation=validation,
            tool_trace=tool_trace,
        )
        LOGGER.info(
            "request_completed model=%s validation_status=%s",
            self.model_name,
            validation.status,
        )
        return result.model_dump()

    @staticmethod
    def _parse_validation_output(validation_output: str):
        from services.schemas import StandardsValidation

        return StandardsValidation.model_validate_json(validation_output)


def create_copilot(*, model_name: str, api_key: str) -> ConfigurationCopilot:
    kb = StandardsKnowledgeBase(api_key=api_key)
    return ConfigurationCopilot(model_name=model_name, api_key=api_key, kb=kb)
