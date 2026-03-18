from __future__ import annotations

import json

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from services.schemas import ConfigurationDraft, RequirementAnalysis


def create_chat_model(model_name: str, api_key: str) -> ChatOpenAI:
    return ChatOpenAI(model=model_name, api_key=api_key, temperature=0)


def analyze_requirement_text(
    requirement_text: str,
    *,
    model_name: str,
    api_key: str,
) -> RequirementAnalysis:
    llm = create_chat_model(model_name=model_name, api_key=api_key)
    structured_llm = llm.with_structured_output(RequirementAnalysis)
    prompt = (
        "You are reviewing a service configuration requirement for an automotive software platform.\n"
        "Produce a concise, safe-to-display analysis summary.\n"
        "Do not reveal hidden chain-of-thought.\n"
        "Identify contradictions, ambiguities, and missing information before extraction.\n\n"
        f"Requirement:\n{requirement_text}"
    )
    return structured_llm.invoke(prompt)


def build_analysis_tool(*, model_name: str, api_key: str):
    @tool
    def analyze_requirement_text_tool(requirement_text: str) -> str:
        """Analyze a service configuration requirement for ambiguity, contradiction, and missing information."""
        analysis = analyze_requirement_text(
            requirement_text,
            model_name=model_name,
            api_key=api_key,
        )
        return analysis.model_dump_json(indent=2)

    analyze_requirement_text_tool.name = "analyze_requirement_text"
    return analyze_requirement_text_tool


def run_analysis_tool(
    requirement_text: str,
    *,
    model_name: str,
    api_key: str,
) -> tuple[RequirementAnalysis, dict]:
    analysis_tool = build_analysis_tool(model_name=model_name, api_key=api_key)
    tool_output = analysis_tool.invoke({"requirement_text": requirement_text})
    analysis = RequirementAnalysis.model_validate_json(tool_output)
    trace = {
        "tool": analysis_tool.name,
        "status": "success",
        "purpose": "Summarize ambiguity, contradictions, and missing inputs before extraction.",
        "input": {"requirement_text": requirement_text},
        "output_summary": {
            "risk_level": analysis.risk_level,
            "ambiguity_count": len(analysis.ambiguities),
            "contradiction_count": len(analysis.contradictions),
            "missing_information_count": len(analysis.missing_information),
        },
        "output": json.loads(tool_output),
    }
    return analysis, trace


def build_extraction_tool(*, model_name: str, api_key: str):
    llm = create_chat_model(model_name=model_name, api_key=api_key)
    structured_llm = llm.with_structured_output(ConfigurationDraft)

    @tool
    def extract_configuration_parameters(
        requirement_text: str,
        retrieved_context_json: str,
        analysis_json: str,
    ) -> str:
        """Extract configuration parameters from a requirement using a Pydantic schema."""
        analysis = RequirementAnalysis.model_validate_json(analysis_json)
        retrieved_context = json.loads(retrieved_context_json)
        context_text = "\n\n".join(
            f"[{item.get('title', 'Reference')} | {item.get('source', 'unknown')}]\n{item.get('excerpt', '')}"
            for item in retrieved_context[:4]
        )
        prompt = (
            "Extract service configuration fields from the requirement.\n"
            "Use the retrieved reference context and the analysis to avoid copying contradictory information blindly.\n"
            "Only populate fields that are sufficiently supported.\n"
            "Normalize play type and class when possible.\n\n"
            "Classification hints:\n"
            "- broadcast, publish, signal update, cyclic notification -> Class=Event\n"
            "- request, response, RPC, explicit call -> Class=Method\n"
            "- field, property, readable state value -> Class=Field\n"
            "- cyclic, periodic, every N ms/Hz -> PlayType=Cyclic\n"
            "- on-change, event-driven -> PlayType=OnChange\n"
            "- on-request, on-demand -> PlayType=OnRequest\n"
            "- one-shot, single trigger -> PlayType=OneShot\n\n"
            f"Requirement:\n{requirement_text}\n\n"
            f"Retrieved context:\n{context_text}\n\n"
            f"Analysis summary:\n{analysis.summary}\n"
            f"Contradictions: {analysis.contradictions}\n"
            f"Ambiguities: {analysis.ambiguities}\n"
            f"Missing information: {analysis.missing_information}\n"
        )
        extracted = structured_llm.invoke(prompt)
        return extracted.model_dump_json(indent=2)

    return extract_configuration_parameters


def run_extraction_tool(
    requirement_text: str,
    analysis: RequirementAnalysis,
    retrieved_context_json: str,
    *,
    model_name: str,
    api_key: str,
) -> tuple[ConfigurationDraft, dict]:
    extraction_tool = build_extraction_tool(model_name=model_name, api_key=api_key)
    payload = {
        "requirement_text": requirement_text,
        "retrieved_context_json": retrieved_context_json,
        "analysis_json": analysis.model_dump_json(indent=2),
    }
    tool_output = extraction_tool.invoke(payload)
    extraction = ConfigurationDraft.model_validate_json(tool_output)
    trace = {
        "tool": extraction_tool.name,
        "status": "success",
        "purpose": "Generate the configuration JSON draft from the requirement and retrieved reference context.",
        "input": {
            "requirement_text": requirement_text,
            "analysis_summary": analysis.summary,
        },
        "output_summary": {
            "service_name": extraction.ServiceName,
            "class": extraction.Class,
            "play_type": extraction.PlayType,
            "missing_fields": extraction.unresolved_fields(),
        },
        "output": json.loads(tool_output),
    }
    return extraction, trace
