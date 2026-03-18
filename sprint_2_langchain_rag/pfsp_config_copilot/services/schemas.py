from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


ServiceClassValue = Literal["Event", "Method", "Field", "Unknown"]
PlayTypeValue = Literal["Cyclic", "OnChange", "OnRequest", "OneShot", "Unknown"]
RiskLevelValue = Literal["low", "medium", "high"]
ValidationStatusValue = Literal["ready", "needs_review", "incomplete"]


class RequirementAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str = Field(..., description="Safe-to-display reasoning summary, not hidden chain-of-thought.")
    contradictions: list[str] = Field(default_factory=list)
    ambiguities: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    follow_up_questions: list[str] = Field(default_factory=list)
    risk_level: RiskLevelValue = "medium"


class ConfigurationDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ServiceName: str | None = None
    ID: int | None = None
    Class: ServiceClassValue | None = None
    Frequency: str | None = None
    PlayType: PlayTypeValue | None = None

    @field_validator("ID", mode="before")
    @classmethod
    def _normalize_id(cls, value):
        if value in [None, ""]:
            return None
        if isinstance(value, int):
            return value
        text = str(value).strip().lower()
        if text.startswith("0x"):
            return int(text, 16)
        return int(text)

    @field_validator("Class", mode="before")
    @classmethod
    def _normalize_class(cls, value):
        if value in [None, ""]:
            return None
        text = str(value).strip().lower()
        mapping = {
            "event": "Event",
            "signal": "Event",
            "broadcast": "Event",
            "method": "Method",
            "request": "Method",
            "rpc": "Method",
            "field": "Field",
            "property": "Field",
            "unknown": "Unknown",
        }
        return mapping.get(text, "Unknown")

    @field_validator("PlayType", mode="before")
    @classmethod
    def _normalize_play_type(cls, value):
        if value in [None, ""]:
            return None
        text = str(value).strip().lower().replace("-", "").replace("_", "").replace(" ", "")
        mapping = {
            "cyclic": "Cyclic",
            "periodic": "Cyclic",
            "timerdriven": "Cyclic",
            "onchange": "OnChange",
            "eventdriven": "OnChange",
            "ondemand": "OnRequest",
            "onrequest": "OnRequest",
            "requestresponse": "OnRequest",
            "oneshot": "OneShot",
            "singletrigger": "OneShot",
            "unknown": "Unknown",
        }
        return mapping.get(text, "Unknown")

    @field_validator("Frequency", mode="before")
    @classmethod
    def _normalize_frequency(cls, value):
        if value in [None, ""]:
            return None
        if isinstance(value, (int, float)):
            return f"{value} Hz"

        text = " ".join(str(value).strip().split())
        lower = text.lower()

        hz_match = re.match(r"^(\d+(?:\.\d+)?)\s*hz$", lower)
        if hz_match:
            return f"{hz_match.group(1)} Hz"

        ms_match = re.match(r"^(\d+(?:\.\d+)?)\s*ms$", lower)
        if ms_match:
            return f"{ms_match.group(1)} ms"

        if lower in {"on change", "onchange", "on request", "onrequest", "event driven"}:
            return text
        return text

    def unresolved_fields(self) -> list[str]:
        return [
            field_name
            for field_name in ["ServiceName", "ID", "Class", "Frequency", "PlayType"]
            if getattr(self, field_name) in [None, "", "Unknown"]
        ]


class RetrievedStandardChunk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str
    title: str
    query: str
    excerpt: str
    score: float | None = None


class StandardsValidation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: ValidationStatusValue = "needs_review"
    summary: str
    schema_valid: bool = True
    missing_required_fields: list[str] = Field(default_factory=list)
    basic_rule_findings: list[str] = Field(default_factory=list)
    compliance_notes: list[str] = Field(default_factory=list)
    suggested_actions: list[str] = Field(default_factory=list)
    follow_up_questions: list[str] = Field(default_factory=list)
    referenced_sections: list[str] = Field(default_factory=list)


class CopilotResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    analysis: RequirementAnalysis
    extraction: ConfigurationDraft
    translated_queries: list[str]
    retrieved_chunks: list[RetrievedStandardChunk]
    validation: StandardsValidation
    tool_trace: list[dict]
