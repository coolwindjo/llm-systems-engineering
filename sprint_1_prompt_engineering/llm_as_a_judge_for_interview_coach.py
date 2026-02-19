#!/usr/local/bin/python3
"""Automated LLM-as-a-Judge evaluator for ADAS/ASPICE interview logs.

Workflow:
1) Read interview samples (question, feedback, and candidate answer).
2) Send each sample to Judge LLM (gpt-5-mini) with rubric prompt.
3) Aggregate per-dimension and weighted score.
4) Save detailed results to evaluation_report.csv.
5) Optional: compare CoT-style and non-CoT judge prompts.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
from openai import OpenAI

MODEL_DEFAULT = "gpt-5-mini"
DEFAULT_OUTPUT_CSV = "evaluation_report.csv"

RUBRIC_WEIGHTS = {
    "technical_depth": 0.45,
    "logical_flow": 0.30,
    "constructive_feedback": 0.25,
}


PENALTY_NOTES = [
    "If ASPICE CL3-relevant content is discussed, lack of concrete evidence artifacts (requirements IDs, "
    "test evidence, traceability, change records, verification outcomes) should reduce score."
]


INTERVIEW_CASES = [
    {
        "id": "ADAS-01",
        "scenario": "ADAS Perception - sensor fusion",
        "jd_context": "ADAS Perception engineer interview, ASPICE CL3 process-oriented.",
        "interviewer_question": (
            "You mentioned reducing misses under backlight. Explain how you'd combine camera and radar features "
            "to reduce false negatives, and specify which ASPICE CL3 work products you would update."
        ),
        "interviewer_feedback": (
            "Nice answer. It covers calibration and redundancy. But be specific about the exact evidence artifacts "
            "you would produce before deployment (requirements, test evidence, and review records)."
        ),
        "candidate_answer": (
            "I would lower the confidence weight of camera under backlight and raise radar weighting in the short horizon, "
            "with temporal consistency constraints and NMS gating by track age. I'd update system requirements in SWRS, "
            "component-level interface definitions, verification scenarios, and map the updates in the traceability matrix."
        ),
    },
    {
        "id": "ADAS-02",
        "scenario": "ADAS Perception - evidence traceability",
        "jd_context": "ADAS Perception engineer interview, ASPICE CL3 process-oriented.",
        "interviewer_question": (
            "A customer issue showed missed crosswalk detection at dusk. Explain your corrective cycle from bug report "
            "to release decision in CL3 terms."
        ),
        "interviewer_feedback": (
            "You stated 'train more data' broadly, but did not map that to a CL3 change/review flow or verify objective KPIs. "
            "Which evidence would convince a release decision board?"
        ),
        "candidate_answer": (
            "First I would classify the issue severity and impact on ASIL goals, create a change request, and add regression scenarios. "
            "Then we retrain and run A/B validation with miss rate and latency KPIs, then hold a review before release."
        ),
    },
    {
        "id": "ADAS-03",
        "scenario": "ADAS Perception - interview continuity",
        "jd_context": "ADAS Perception engineer interview, ASPICE CL3 process-oriented.",
        "interviewer_question": (
            "In the previous answer you proposed sensor redundancy. Given that, how would you validate a rare-fog edge case "
            "without increasing false positive rate too much?"
        ),
        "interviewer_feedback": (
            "Good continuity. You mentioned scene selection but I still need concrete pass/fail gates and ownership."
        ),
        "candidate_answer": (
            "I would define a confidence-band gate, require scene-stratified KPIs, and use a shadow mode before fleet rollout. "
            "Ownership goes to the perception lead for design, validation for KPI definition, and safety manager for approval."
        ),
    },
]


INTERVIEW_JUDGE_PROMPT_COT = """You are an expert technical interviewer assessor for ADAS/automotive interview sessions.

Evaluate only the AI interviewer quality based on:
1) AI interviewer's question/feedback
2) Candidate answer
3) Interview context

Scoring dimensions (1-5 each, integers):
- technical_depth: validate ASPICE CL3 and ADAS practical knowledge accuracy.
- logical_flow: continuity from candidate answer and smooth question progression.
- constructive_feedback: whether feedback is actionable and specific.

Evidence rule (mandatory):
- If interview context is ASPICE CL3-relevant and the interviewer did not demand concrete artifacts/evidence
  (requirements, test cases/results, design docs, review records, change requests, traceability matrix),
  then apply -1 penalty (minimum 1) to technical_depth.
- If ASPICE evidence is not demanded but candidate answer is opinion-based, also apply -1 penalty (minimum 1) to constructive_feedback.

Use this process explicitly in your reasoning for each dimension:
Step 1) Check factual/technical correctness.
Step 2) Check context continuity.
Step 3) Check actionability and evidence-driving advice.
Step 4) Apply evidence penalties where needed.

Return strict JSON exactly in this format:
{{
  "technical_depth": {{
    "score": 1,
    "reasoning": "...",
    "evidence_check": "..."
  }},
  "logical_flow": {{
    "score": 1,
    "reasoning": "..."
  }},
  "constructive_feedback": {{
    "score": 1,
    "reasoning": "..."
  }},
  "aspice_evidence_penalty_applied": {{
    "technical_depth": false,
    "constructive_feedback": false,
    "notes": "..."
  }},
  "overall_reasoning": "..."
}}

Scoring anchors:
technical_depth 1-5 from basic inaccuracy to expert safety/process depth.
logical_flow 1-5 from topic jumps to seamless continuation.
constructive_feedback 1-5 from vague praise/criticism to specific action plans with measurable improvements.
"""


INTERVIEW_JUDGE_PROMPT_NON_COT = """You are an expert judge for AI interviewer quality in ADAS/ASPICE interviews.
Evaluate technical_depth, logical_flow, constructive_feedback on 1-5 integer scale.
Apply ASPICE evidence penalties when interviewer does not request concrete artifacts while CL3 context is relevant.
Return strict JSON in this exact schema:
{{
  "technical_depth": {{"score": 1, "reasoning": "", "evidence_check": ""}},
  "logical_flow": {{"score": 1, "reasoning": ""}},
  "constructive_feedback": {{"score": 1, "reasoning": ""}},
  "aspice_evidence_penalty_applied": {{"technical_depth": false, "constructive_feedback": false, "notes": ""}},
  "overall_reasoning": ""
}}
"""


def _coerce_int_score(value: Any, fallback: int = 3) -> int:
    try:
        score = int(value)
    except (TypeError, ValueError):
        return fallback
    if score < 1:
        return 1
    if score > 5:
        return 5
    return score


def _coerce_bool(value: Any) -> bool:
    return bool(value) if isinstance(value, bool) else str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def _extract_json(raw: str) -> dict[str, Any] | None:
    text = raw.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None


def _weighted_overall(scores: dict[str, int]) -> float:
    raw = (
        scores["technical_depth"] * RUBRIC_WEIGHTS["technical_depth"]
        + scores["logical_flow"] * RUBRIC_WEIGHTS["logical_flow"]
        + scores["constructive_feedback"] * RUBRIC_WEIGHTS["constructive_feedback"]
    )
    return round(raw, 2)


def _rating_label(score: float) -> str:
    if score <= 2.0:
        return "Needs major improvement"
    if score <= 3.3:
        return "Needs improvement"
    if score <= 4.1:
        return "Good"
    return "Excellent"


def _normalize_judge_payload(payload: dict[str, Any], parse_error: bool) -> dict[str, Any]:
    tech = payload.get("technical_depth", {})
    lf = payload.get("logical_flow", {})
    cf = payload.get("constructive_feedback", {})
    penalties = payload.get("aspice_evidence_penalty_applied", {})

    technical_depth_score = _coerce_int_score(
        tech.get("score", 1 if parse_error else 3),
        1 if parse_error else 3,
    )
    logical_flow_score = _coerce_int_score(
        lf.get("score", 1 if parse_error else 3),
        1 if parse_error else 3,
    )
    constructive_score = _coerce_int_score(
        cf.get("score", 1 if parse_error else 3),
        1 if parse_error else 3,
    )

    overall = _weighted_overall(
        {
            "technical_depth": technical_depth_score,
            "logical_flow": logical_flow_score,
            "constructive_feedback": constructive_score,
        }
    )

    return {
        "technical_depth_score": technical_depth_score,
        "technical_depth_reasoning": str(tech.get("reasoning", "")) or "No reasoning provided.",
        "technical_depth_evidence_check": str(tech.get("evidence_check", "")),
        "logical_flow_score": logical_flow_score,
        "logical_flow_reasoning": str(lf.get("reasoning", "")) or "No reasoning provided.",
        "constructive_feedback_score": constructive_score,
        "constructive_feedback_reasoning": str(cf.get("reasoning", "")) or "No reasoning provided.",
        "aspice_penalty_technical_depth": _coerce_bool(penalties.get("technical_depth", False)),
        "aspice_penalty_constructive_feedback": _coerce_bool(
            penalties.get("constructive_feedback", False)
        ),
        "aspice_penalty_notes": str(penalties.get("notes", "")),
        "overall_weighted_score": overall,
        "overall_rating": _rating_label(overall),
        "overall_reasoning": str(payload.get("overall_reasoning", "")) or "No overall reasoning provided.",
        "parse_error": parse_error,
    }


def call_judge_llm(
    client: OpenAI,
    model: str,
    system_prompt: str,
    sample: dict[str, Any],
) -> str:
    user_payload = json.dumps(
        {
            "jd_context": sample.get("jd_context", ""),
            "interview_turn": {
                "interviewer_question": sample.get("interviewer_question", ""),
                "interviewer_feedback": sample.get("interviewer_feedback", ""),
            },
            "candidate_answer": sample.get("candidate_answer", ""),
        },
        ensure_ascii=False,
        indent=2,
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_payload},
        ],
    )
    return response.choices[0].message.content or ""


def evaluate_one_sample(
    client: OpenAI,
    model: str,
    system_prompt: str,
    sample: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    judge_raw = call_judge_llm(client, model, system_prompt, sample)
    parsed = _extract_json(judge_raw)
    parse_error = parsed is None
    if parse_error:
        normalized = {
            "technical_depth_score": 1,
            "technical_depth_reasoning": "Failed to parse judge JSON.",
            "technical_depth_evidence_check": "",
            "logical_flow_score": 1,
            "logical_flow_reasoning": "Failed to parse judge JSON.",
            "constructive_feedback_score": 1,
            "constructive_feedback_reasoning": "Failed to parse judge JSON.",
            "aspice_penalty_technical_depth": False,
            "aspice_penalty_constructive_feedback": False,
            "aspice_penalty_notes": "Judge output was not parseable as JSON.",
            "overall_weighted_score": 1.0,
            "overall_rating": _rating_label(1.0),
            "overall_reasoning": "Parsing failed. Defaulted to minimum score.",
            "parse_error": True,
        }
    else:
        normalized = _normalize_judge_payload(parsed, parse_error=False)

    row = {
        "case_id": sample.get("id", ""),
        "scenario": sample.get("scenario", ""),
        "judge_model": model,
        "jd_context": sample.get("jd_context", ""),
        "interviewer_question": sample.get("interviewer_question", ""),
        "interviewer_feedback": sample.get("interviewer_feedback", ""),
        "candidate_answer": sample.get("candidate_answer", ""),
        "judge_raw_output": judge_raw,
    }
    row.update(normalized)
    return row, judge_raw


def evaluate_cases(
    client: OpenAI,
    model: str,
    samples: list[dict[str, Any]],
    prompt_mode: str,
    system_prompt: str,
) -> pd.DataFrame:
    records = []
    for idx, sample in enumerate(samples, 1):
        print(f"[{idx}/{len(samples)}] evaluating case: {sample.get('id', idx)}")
        record, _ = evaluate_one_sample(client, model, system_prompt, sample)
        record["prompt_mode"] = prompt_mode
        records.append(record)
    return pd.DataFrame(records)


def summarize_by_prompt(df: pd.DataFrame) -> pd.DataFrame:
    score_cols = [
        "technical_depth_score",
        "logical_flow_score",
        "constructive_feedback_score",
        "overall_weighted_score",
    ]
    return (
        df.groupby("prompt_mode")[score_cols]
        .mean(numeric_only=True)
        .round(2)
        .sort_values("overall_weighted_score", ascending=False)
    )


def load_cases_from_json(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    if not isinstance(loaded, list):
        raise ValueError("Input JSON must be a list of interview samples.")
    return loaded


def run_evaluation(
    client: OpenAI,
    model: str,
    cases: list[dict[str, Any]],
    output_csv: str,
    compare: bool = False,
) -> pd.DataFrame:
    if compare:
        df_cot = evaluate_cases(client, model, cases, "cot", INTERVIEW_JUDGE_PROMPT_COT)
        df_no_cot = evaluate_cases(client, model, cases, "no_cot", INTERVIEW_JUDGE_PROMPT_NON_COT)
        df = pd.concat([df_cot, df_no_cot], ignore_index=True)
    else:
        df = evaluate_cases(client, model, cases, "single_prompt", INTERVIEW_JUDGE_PROMPT_COT)

    df.to_csv(output_csv, index=False)
    print(f"\nSaved report: {output_csv}")

    avg_cols = [
        "technical_depth_score",
        "logical_flow_score",
        "constructive_feedback_score",
    ]
    print("\nPer-dimension average:")
    print(df[avg_cols].mean(numeric_only=True).round(2).to_string())

    overall_avg = df["overall_weighted_score"].mean()
    print(f"\nOverall weighted average: {overall_avg:.2f}")

    if compare:
        summary = summarize_by_prompt(df)
        print("\nPrompt comparison (higher is better):")
        print(summary.to_string())
        best = summary.index[0]
        worst = summary.index[-1]
        print(f"\nBest prompt mode: {best}")
        print(f"Worst prompt mode: {worst}")

    return df


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "LLM-as-a-Judge automation for ADAS interview coaching logs "
            "(technical_depth / logical_flow / constructive_feedback)."
        )
    )
    parser.add_argument(
        "--mode",
        default="evaluate",
        choices=["evaluate", "compare"],
        help="evaluate: single prompt (CoT-style). compare: evaluate both CoT and non-CoT prompts.",
    )
    parser.add_argument(
        "--model",
        default=MODEL_DEFAULT,
        help="Judge model name (default: gpt-5-mini).",
    )
    parser.add_argument(
        "--input-json",
        default=None,
        help=(
            "Optional path to interview logs JSON list. "
            "Required keys per entry: interviewer_question, interviewer_feedback, candidate_answer, jd_context."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of cases to evaluate.",
    )
    parser.add_argument(
        "--output-csv",
        default=DEFAULT_OUTPUT_CSV,
        help="Output CSV path (default: evaluation_report.csv).",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("OPENAI_API_KEY"),
        help="OpenAI API key (or set OPENAI_API_KEY).",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.api_key:
        raise SystemExit("Missing OPENAI API key. Set OPENAI_API_KEY env or pass --api-key.")

    if args.mode not in {"evaluate", "compare"}:
        raise SystemExit(f"Unsupported mode: {args.mode}")

    cases: list[dict[str, Any]]
    if args.input_json:
        if not Path(args.input_json).exists():
            raise SystemExit(f"Input file not found: {args.input_json}")
        cases = load_cases_from_json(args.input_json)
        if not cases:
            raise SystemExit("Input JSON has no records.")
    else:
        cases = INTERVIEW_CASES

    if args.limit is not None:
        cases = cases[: args.limit]

    client = OpenAI(api_key=args.api_key)

    run_evaluation(
        client=client,
        model=args.model,
        cases=cases,
        output_csv=args.output_csv,
        compare=args.mode == "compare",
    )


if __name__ == "__main__":
    main()
