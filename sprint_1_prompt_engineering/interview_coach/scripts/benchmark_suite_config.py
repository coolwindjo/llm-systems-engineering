"""Shared constants and prompts for ADAS interview benchmark suite."""

from __future__ import annotations

from pathlib import Path

MODEL_LIST = ["gpt-5-mini", "gpt-4o-mini", "gpt-4.1-mini"]
TECHNIQUE_LIST = [
    "chain_of_thought",
    "few_shot",
    "persona_conditioning",
    "knowledge_paucity",
]

JUDGE_MODEL_DEFAULT = "gpt-5-mini"
LOG_ROOT = Path(__file__).resolve().parent.parent / "benchmark_logs"
RESULTS_CSV = "benchmark_summary.csv"
RAW_RESULTS_JSON = "raw_results.json"
RADAR_CHART_PATH = "benchmark_radar_chart.png"
GROUPED_BAR_PATH = "benchmark_grouped_overall_bar.png"

SCENARIO_TURNS: list[str] = [
    "Hi, I'm a senior embedded software engineer with 10 years of ADAS/perception experience, mostly in camera and radar pipelines in safety-related projects.",
    "My technical stack includes Embedded C++, Python, and C++ sensor processing tools. I have implemented camera, radar, and lidar fusion modules with AUTOSAR integration, static analysis, and CI/CD verification.",
    "In one ADAS project, object tracking was unstable in low-light scenes. I solved this by adding confidence-weighted cross-sensor reconciliation, temporal filtering stabilization, and edge-case regression tests before launch.",
]

JUDGE_SYSTEM_PROMPT = """You are an expert ADAS interviewer evaluator.

Task:
Evaluate the AI interviewer quality for each turn from a mock interview transcript.
Use the rubric below and return strict JSON only.

Scoring dimensions (integer 1-10):
- technical_depth: How well the interviewer probes ASPICE CL3 and ADAS practical knowledge with depth, specificity, and rigor.
- context_awareness: Whether the interviewer naturally continues the conversation and asks follow-up questions using prior candidate answers.
- professionalism: How consistently the interviewer maintains a realistic senior interview tone (clear, concise, non-redundant, high bar, and respectful).

ASPICE evidence rule (mandatory):
If the context is ASPICE CL3-relevant and the interviewer did not request concrete evidence artifacts (e.g., requirement updates, verification strategy, traceability matrix, test cases/results, review records, or change control artifacts) when asking technical claims, subtract 1 point from technical_depth (minimum 1).
If the interviewer repeatedly ignores an opportunity for traceability or process checks in context, add a note in evidence_check.

Input is one interview turn as JSON with fields:
- interviewer_turn: Interviewer question or feedback text.
- candidate_answer: Candidate answer text.
- context_turn_history: Prior conversation turns for continuity checks.

Return exactly this JSON schema:
{
  "technical_depth": {"score": 1, "reasoning": "", "evidence_check": ""},
  "context_awareness": {"score": 1, "reasoning": ""},
  "professionalism": {"score": 1, "reasoning": ""},
  "aspice_evidence_penalty_applied": {"technical_depth": false, "notes": ""},
  "overall_reasoning": ""
}
"""
