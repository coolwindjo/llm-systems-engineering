#!/usr/bin/env python3
"""Benchmark ADAS interview simulations and evaluate logs with an LLM judge."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from itertools import product
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Dict, List

from openai import OpenAI

try:
    from matplotlib import pyplot as plt
except Exception:  # pragma: no cover - optional visualization dependency
    plt = None

try:
    import numpy as np
except Exception:  # pragma: no cover - optional dependency
    np = None

try:
    import pandas as pd
except Exception:  # pragma: no cover - optional dependency
    pd = None

from services.interview_ops import load_local_env
from services.personas import build_system_prompts
from utils.data_loader import load_interview_data


MODEL_LIST = ["gpt-5-mini", "gpt-4o-mini", "gpt-4.1-mini"]
TECHNIQUE_LIST = [
    "chain_of_thought",
    "few_shot",
    "persona_conditioning",
    "knowledge_paucity",
]

JUDGE_MODEL_DEFAULT = "gpt-5-mini"
LOG_ROOT = Path(__file__).resolve().parent / "benchmark_logs"
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


@dataclass
class TurnRecord:
    turn_index: int
    role: str
    started_at: str
    ended_at: str
    elapsed_ms: float
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    content: str
    model: str
    technique: str
    error: str


class UserProxy:
    """Deterministic interviewee simulator with fixed scripted responses."""

    def __init__(self, turns: list[str] | None = None) -> None:
        self._turns = turns or SCENARIO_TURNS
        self._index = 0

    def next_response(self) -> str:
        if self._index < len(self._turns):
            response = self._turns[self._index]
            self._index += 1
            return response
        return "Thank you for the interview."


def _coerce_score(value: Any, minimum: int = 1, maximum: int = 10) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return minimum
    return max(minimum, min(maximum, parsed))


def _extract_usage(response: Any) -> tuple[int | None, int | None, int | None]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None, None, None
    return (
        int(getattr(usage, "prompt_tokens", 0) or 0),
        int(getattr(usage, "completion_tokens", 0) or 0),
        int(getattr(usage, "total_tokens", 0) or 0),
    )


def _build_system_prompt(data: Dict[str, Any], technique: str) -> str:
    prompts = build_system_prompts(data=data, jd_profile=data, technique=technique)
    system_prompt = prompts.get("generic_ai_interviewer")
    if not system_prompt:
        raise RuntimeError("Cannot build interviewer system prompt.")
    return system_prompt


def _call_model(
    client: OpenAI,
    model: str,
    messages: list[dict[str, str]],
) -> tuple[str, float, int | None, int | None, int | None]:
    started = time.perf_counter()
    response = client.chat.completions.create(model=model, messages=messages)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    content = response.choices[0].message.content or ""
    return content, elapsed_ms, *_extract_usage(response)


def run_one_simulation(
    client: OpenAI,
    model: str,
    technique: str,
    data: Dict[str, Any],
    output_dir: Path,
    progress_callback: Callable[[int, int, str], None] | None = None,
    step: int = 1,
    total_steps: int = 1,
) -> Dict[str, Any]:
    user_proxy = UserProxy()
    system_prompt = _build_system_prompt(data=data, technique=technique)
    run_id = f"{model.replace('.', '_')}__{technique}__{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}"

    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
    transcript: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
    turn_records: list[TurnRecord] = []

    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0
    assistant_calls = 0
    assistant_elapsed_total = 0.0
    error_message = ""

    run_start = datetime.utcnow().isoformat()

    messages.append(
        {
            "role": "user",
            "content": "Start a mock ADAS interview and ask your first question.",
        }
    )
    transcript.append(messages[-1])

    for turn_idx in range(len(SCENARIO_TURNS)):
        call_start = datetime.utcnow().isoformat()
        try:
            assistant_reply, elapsed_ms, tokens_prompt, tokens_completion, tokens_total = _call_model(
                client=client,
                model=model,
                messages=messages,
            )
            assistant_ended = datetime.utcnow().isoformat()

            messages.append({"role": "assistant", "content": assistant_reply})
            transcript.append({"role": "assistant", "content": assistant_reply})

            if tokens_prompt is not None:
                total_prompt_tokens += tokens_prompt
            if tokens_completion is not None:
                total_completion_tokens += tokens_completion
            if tokens_total is not None:
                total_tokens += tokens_total

            assistant_calls += 1
            assistant_elapsed_total += elapsed_ms

            turn_records.append(
                TurnRecord(
                    turn_index=turn_idx,
                    role="assistant",
                    started_at=call_start,
                    ended_at=assistant_ended,
                    elapsed_ms=elapsed_ms,
                    prompt_tokens=tokens_prompt,
                    completion_tokens=tokens_completion,
                    total_tokens=tokens_total,
                    content=assistant_reply,
                    model=model,
                    technique=technique,
                    error="",
                )
            )
        except Exception as exc:  # pragma: no cover
            error_message = str(exc)
            turn_records.append(
                TurnRecord(
                    turn_index=turn_idx,
                    role="assistant",
                    started_at=call_start,
                    ended_at=datetime.utcnow().isoformat(),
                    elapsed_ms=0.0,
                    prompt_tokens=None,
                    completion_tokens=None,
                    total_tokens=None,
                    content="",
                    model=model,
                    technique=technique,
                    error=error_message,
                )
            )
            break

        candidate_answer = user_proxy.next_response()
        messages.append({"role": "user", "content": candidate_answer})
        transcript.append({"role": "user", "content": candidate_answer})

    # Final interviewer follow-up after all fixed candidate turns.
    if not error_message:
        call_start = datetime.utcnow().isoformat()
        try:
            final_reply, elapsed_ms, tokens_prompt, tokens_completion, tokens_total = _call_model(
                client=client,
                model=model,
                messages=messages,
            )
            final_ended = datetime.utcnow().isoformat()

            messages.append({"role": "assistant", "content": final_reply})
            transcript.append({"role": "assistant", "content": final_reply})

            if tokens_prompt is not None:
                total_prompt_tokens += tokens_prompt
            if tokens_completion is not None:
                total_completion_tokens += tokens_completion
            if tokens_total is not None:
                total_tokens += tokens_total

            assistant_calls += 1
            assistant_elapsed_total += elapsed_ms

            turn_records.append(
                TurnRecord(
                    turn_index=len(SCENARIO_TURNS),
                    role="assistant",
                    started_at=call_start,
                    ended_at=final_ended,
                    elapsed_ms=elapsed_ms,
                    prompt_tokens=tokens_prompt,
                    completion_tokens=tokens_completion,
                    total_tokens=tokens_total,
                    content=final_reply,
                    model=model,
                    technique=technique,
                    error="",
                )
            )
        except Exception as exc:  # pragma: no cover
            error_message = str(exc)
            turn_records.append(
                TurnRecord(
                    turn_index=len(SCENARIO_TURNS),
                    role="assistant",
                    started_at=call_start,
                    ended_at=datetime.utcnow().isoformat(),
                    elapsed_ms=0.0,
                    prompt_tokens=None,
                    completion_tokens=None,
                    total_tokens=None,
                    content="",
                    model=model,
                    technique=technique,
                    error=error_message,
                )
            )

    run_end = datetime.utcnow().isoformat()
    avg_latency = (
        round(assistant_elapsed_total / assistant_calls, 2)
        if assistant_calls
        else 0.0
    )

    summary = {
        "run_id": run_id,
        "model": model,
        "technique": technique,
        "run_start": run_start,
        "run_end": run_end,
        "error_message": error_message,
        "assistant_turn_count": assistant_calls,
        "user_turn_count": len(SCENARIO_TURNS),
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "total_tokens": total_tokens,
        "avg_assistant_latency_ms": avg_latency,
        "system_prompt": system_prompt,
        "transcript": transcript,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / f"{run_id}.json"
    log_payload = {
        "meta": summary,
        "turn_records": [asdict(item) for item in turn_records],
    }
    log_path.write_text(json.dumps(log_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    if progress_callback:
        progress_callback(step, total_steps, f"Completed {model} / {technique}")

    return summary


def run_benchmark_matrix(
    client: OpenAI,
    models: list[str],
    techniques: list[str],
    data: Dict[str, Any],
    output_dir: Path,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> list[Dict[str, Any]]:
    combos = list(product(models, techniques))
    total = len(combos)
    results: list[Dict[str, Any]] = []

    for idx, (model, technique) in enumerate(combos, start=1):
        if progress_callback:
            progress_callback(idx - 1, total, f"Running {model} | {technique}")
        results.append(
            run_one_simulation(
                client=client,
                model=model,
                technique=technique,
                data=data,
                output_dir=output_dir,
                progress_callback=progress_callback,
                step=idx,
                total_steps=total,
            )
        )

    if progress_callback:
        progress_callback(total, total, "Simulation complete")

    return results


def _extract_turn_pairs_from_transcript(log_data: Dict[str, Any], model: str, technique: str, run_id: str) -> list[Dict[str, Any]]:
    transcript = log_data.get("meta", {}).get("transcript", []) or log_data.get("transcript", [])

    if not isinstance(transcript, list):
        return []

    pairs: list[Dict[str, Any]] = []
    for idx, message in enumerate(transcript):
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue

        if idx == 0:
            continue

        candidate_answer = ""
        if idx + 1 < len(transcript) and isinstance(transcript[idx + 1], dict) and transcript[idx + 1].get("role") == "user":
            candidate_answer = str(transcript[idx + 1].get("content", "")).strip()
        elif idx > 0 and isinstance(transcript[idx - 1], dict) and transcript[idx - 1].get("role") == "user":
            candidate_answer = str(transcript[idx - 1].get("content", "")).strip()

        interviewer_turn = str(message.get("content", "")).strip()
        if not interviewer_turn or not candidate_answer:
            continue

        history_start = max(0, idx - 6)
        history = [
            {
                "role": item.get("role", ""),
                "content": str(item.get("content", "")),
            }
            for item in transcript[history_start:idx]
            if isinstance(item, dict)
        ]

        pairs.append(
            {
                "run_id": run_id,
                "model": model,
                "technique": technique,
                "turn_index": idx,
                "interviewer_turn": interviewer_turn,
                "candidate_answer": candidate_answer,
                "context_turn_history": history,
            }
        )

    return pairs


def load_benchmark_logs(output_dir: Path) -> list[Dict[str, Any]]:
    if not output_dir.exists():
        return []

    loaded: list[Dict[str, Any]] = []
    for path in sorted(output_dir.glob("*.json")):
        if path.name in {RAW_RESULTS_JSON, RESULTS_CSV, RADAR_CHART_PATH, GROUPED_BAR_PATH}:
            continue
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue

        if not isinstance(raw, dict):
            continue

        meta = raw.get("meta", {})
        if not isinstance(meta, dict):
            continue

        run_id = str(meta.get("run_id", path.stem))
        model = str(meta.get("model", ""))
        technique = str(meta.get("technique", ""))

        pairs = _extract_turn_pairs_from_transcript(raw, model=model, technique=technique, run_id=run_id)
        if not pairs:
            continue

        loaded.extend(pairs)

    return loaded


def _parse_judge_json(raw: str) -> Dict[str, Any] | None:
    text = (raw or "").strip()
    if not text:
        return None

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        first = text.find("{")
        last = text.rfind("}")
        if first == -1 or last <= first:
            return None
        try:
            return json.loads(text[first : last + 1])
        except json.JSONDecodeError:
            return None


def _call_judge(
    client: OpenAI,
    model: str,
    payload: Dict[str, Any],
) -> Dict[str, Any] | None:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content or ""
    return _parse_judge_json(raw)


def _normalize_judge_payload(raw: Dict[str, Any] | None, parse_failed: bool) -> Dict[str, Any]:
    if parse_failed or not raw:
        return {
        "technical_depth_score": 1,
        "context_awareness_score": 1,
        "professionalism_score": 1,
            "overall_reasoning": "Failed to parse judge output.",
            "technical_reasoning": "Failed to parse judge output.",
            "context_reasoning": "Failed to parse judge output.",
            "professionalism_reasoning": "Failed to parse judge output.",
            "aspice_penalty_applied": False,
            "aspice_penalty_notes": "Judge output not parseable.",
            "penalty": False,
        }

    tech = raw.get("technical_depth", {})
    ctx = raw.get("context_awareness", {})
    pro = raw.get("professionalism", {})
    penalty = raw.get("aspice_evidence_penalty_applied", {})

    technical = _coerce_score(tech.get("score", 1), 1, 10)
    context = _coerce_score(ctx.get("score", 1), 1, 10)
    professional = _coerce_score(pro.get("score", 1), 1, 10)

    return {
        "technical_depth_score": technical,
        "context_awareness_score": context,
        "professionalism_score": professional,
        "overall_reasoning": str(raw.get("overall_reasoning", "")) or "No overall reasoning provided.",
        "technical_reasoning": str(tech.get("reasoning", "")) or "No technical reasoning provided.",
        "context_reasoning": str(ctx.get("reasoning", "")) or "No context reasoning provided.",
        "professionalism_reasoning": str(pro.get("reasoning", "")) or "No professionalism reasoning provided.",
        "aspice_penalty_applied": bool(penalty.get("technical_depth", False)),
        "aspice_penalty_notes": str(penalty.get("notes", "")),
        "evidence_check": str(tech.get("evidence_check", "")),
    }


def evaluate_benchmark_logs(
    client: OpenAI,
    output_dir: Path,
    judge_model: str = JUDGE_MODEL_DEFAULT,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]]]:
    pairs = load_benchmark_logs(output_dir)
    if not pairs:
        raise RuntimeError("No benchmark logs found. Run simulation first or provide valid benchmark_logs directory.")

    # Group pairs by run_id for stable aggregation.
    grouped_pairs: Dict[str, list[Dict[str, Any]]] = {}
    for pair in pairs:
        key = str(pair.get("run_id", "unknown"))
        grouped_pairs.setdefault(key, []).append(pair)

    ordered_run_ids = list(grouped_pairs.keys())
    total = len(ordered_run_ids)
    raw_results: list[Dict[str, Any]] = []
    detailed_records: list[Dict[str, Any]] = []

    for idx, run_id in enumerate(ordered_run_ids, start=1):
        run_pairs = grouped_pairs[run_id]
        if not run_pairs:
            continue

        run_model = str(run_pairs[0].get("model", ""))
        run_technique = str(run_pairs[0].get("technique", ""))

        turn_scores: list[Dict[str, Any]] = []
        for pair in run_pairs:
            payload = {
                "interviewer_turn": pair.get("interviewer_turn", ""),
                "candidate_answer": pair.get("candidate_answer", ""),
                "context_turn_history": pair.get("context_turn_history", []),
            }

            try:
                raw = _call_judge(client=client, model=judge_model, payload=payload)
                parse_failed = raw is None
            except Exception:
                raw = None
                parse_failed = True

            normalized = _normalize_judge_payload(raw, parse_failed=parse_failed)

            detailed_records.append(
                {
                    "run_id": pair.get("run_id", run_id),
                    "model": run_model,
                    "technique": run_technique,
                    "turn_index": pair.get("turn_index"),
                    "technical_depth_score": normalized["technical_depth_score"],
                    "context_awareness_score": normalized["context_awareness_score"],
                    "professionalism_score": normalized["professionalism_score"],
                    "overall_reasoning": normalized["overall_reasoning"],
                    "aspice_penalty_applied": normalized["aspice_penalty_applied"],
                    "aspice_penalty_notes": normalized["aspice_penalty_notes"],
                    "technical_reasoning": normalized["technical_reasoning"],
                    "context_reasoning": normalized["context_reasoning"],
                    "professionalism_reasoning": normalized["professionalism_reasoning"],
                    "interviewer_turn": pair.get("interviewer_turn", ""),
                    "candidate_answer": pair.get("candidate_answer", ""),
                }
            )

            turn_scores.append(normalized)

        if not turn_scores:
            avg_scores = {
                "technical_depth": 1,
                "context_awareness": 1,
                "professionalism": 1,
                "overall": 1,
            }
            reasoning = "No valid turn-level scores were produced."
        else:
            avg_tech = mean(item["technical_depth_score"] for item in turn_scores)
            avg_ctx = mean(item["context_awareness_score"] for item in turn_scores)
            avg_pro = mean(item["professionalism_score"] for item in turn_scores)
            avg_overall = round((avg_tech + avg_ctx + avg_pro) / 3, 2)
            avg_scores = {
                "technical_depth": round(avg_tech, 2),
                "context_awareness": round(avg_ctx, 2),
                "professionalism": round(avg_pro, 2),
                "overall": avg_overall,
            }
            reasoning = (
                f"Technical depth avg: {avg_scores['technical_depth']:.2f}; "
                f"Context awareness avg: {avg_scores['context_awareness']:.2f}; "
                f"Professionalism avg: {avg_scores['professionalism']:.2f}."
            )

        raw_results.append(
            {
                "model": run_model,
                "technique": run_technique,
                "scores": avg_scores,
                "reasoning": reasoning,
            }
        )

        if progress_callback:
            progress_callback(idx, total, f"Evaluated run_id={run_id}")

    output_path = output_dir / RAW_RESULTS_JSON
    output_path.write_text(
        json.dumps(raw_results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return raw_results, detailed_records


def _to_long_df(raw_results: list[Dict[str, Any]]) -> "pd.DataFrame":
    if pd is None:
        raise RuntimeError("pandas is required for analysis output.")

    if not raw_results:
        return pd.DataFrame(columns=["model", "technique", "technical_depth", "context_awareness", "professionalism", "overall"])

    normalized_rows: list[Dict[str, Any]] = []
    for item in raw_results:
        scores = item.get("scores", {})
        normalized_rows.append(
            {
                "model": item.get("model", ""),
                "technique": item.get("technique", ""),
                "technical_depth": float(scores.get("technical_depth", scores.get("technical_depth_score", 0) or 0) or 0),
                "context_awareness": float(scores.get("context_awareness", scores.get("context_awareness_score", 0) or 0) or 0),
                "professionalism": float(scores.get("professionalism", scores.get("professionalism_score", 0) or 0) or 0),
                "overall": float(scores.get("overall", 0) or 0),
                "reasoning": item.get("reasoning", ""),
            }
        )
    return pd.DataFrame(normalized_rows)


def _plot_radar_chart(
    by_technique: "pd.DataFrame",
    output_path: Path,
) -> None:
    if plt is None or np is None:
        return

    if by_technique.empty:
        return

    categories = ["Technical Depth", "Context Awareness", "Professionalism"]
    value_cols = ["technical_depth", "context_awareness", "professionalism"]

    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, polar=True)

    for technique in by_technique.index:
        values = by_technique.loc[technique, value_cols].tolist()
        if not values or len(values) != len(value_cols):
            continue
        values += values[:1]
        ax.plot(angles, values, linewidth=1.8, label=technique)
        ax.fill(angles, values, alpha=0.12)

    ax.set_thetagrids(np.degrees(angles[:-1]), categories)
    ax.set_ylim(1, 10)
    ax.set_title("Benchmark Radar Comparison by Prompt Technique")
    ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.05))
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_grouped_overall_bar(
    by_combo: "pd.DataFrame",
    output_path: Path,
) -> None:
    if plt is None:
        return

    if by_combo.empty:
        return

    fig = by_combo.plot(
        kind="bar",
        figsize=(8, 4),
        title="Overall Score by Model and Prompt Technique",
    )
    plt.xlabel("Model")
    plt.ylabel("Average Overall Score")
    plt.ylim(1, 10)
    plt.legend(title="Technique")
    plt.tight_layout()
    fig.get_figure().savefig(output_path)
    plt.close(fig.get_figure())


def analyze_raw_results(
    raw_results: list[Dict[str, Any]],
    output_dir: Path,
) -> tuple["pd.DataFrame", "pd.DataFrame", "pd.DataFrame", str, Path, Path]:
    if pd is None:
        raise RuntimeError("pandas is required for benchmark analysis.")

    df = _to_long_df(raw_results)
    if df.empty:
        raise ValueError("No evaluation results to analyze.")

    by_combo = (
        df.groupby(["model", "technique"])[
            ["technical_depth", "context_awareness", "professionalism", "overall"]
        ]
        .mean(numeric_only=True)
        .round(2)
    )
    by_technique = by_combo.groupby("technique")[[
        "technical_depth",
        "context_awareness",
        "professionalism",
    ]].mean(numeric_only=True)

    model_tech_table = by_combo["overall"].unstack("technique")

    radar_path = output_dir / RADAR_CHART_PATH
    bar_path = output_dir / GROUPED_BAR_PATH

    _plot_radar_chart(by_technique=by_technique, output_path=radar_path)
    _plot_grouped_overall_bar(by_combo=by_combo["overall"].unstack("technique"), output_path=bar_path)

    best_idx = by_combo["overall"].idxmax()
    best_model, best_technique = best_idx
    best_score = by_combo.loc[best_idx, "overall"]

    worst_idx = by_combo["overall"].idxmin()
    worst_model, worst_technique = worst_idx
    worst_score = by_combo.loc[worst_idx, "overall"]

    summary = (
        f"### Benchmark Conclusion\n"
        f"- Best for senior engineer interview: **{best_model} + {best_technique}** (overall = {best_score:.2f})\n"
        f"- Weakest combination: **{worst_model} + {worst_technique}** (overall = {worst_score:.2f})\n"
    )

    return df, by_combo, model_tech_table, summary, radar_path, bar_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run multi-turn benchmark on interview coach prompts and evaluate results with judge LLM.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=MODEL_LIST,
        help="Model list to benchmark.",
    )
    parser.add_argument(
        "--techniques",
        nargs="*",
        default=TECHNIQUE_LIST,
        help="Prompt techniques to benchmark.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(LOG_ROOT),
        help="Directory for benchmark JSON logs and analysis artifacts.",
    )
    parser.add_argument(
        "--judge-model",
        default=JUDGE_MODEL_DEFAULT,
        help="Judge model name (default: gpt-5-mini).",
    )
    parser.add_argument(
        "--results-csv",
        default=RESULTS_CSV,
        help="Summary CSV path filename under output directory.",
    )
    parser.add_argument(
        "--no-evaluate",
        action="store_true",
        help="Skip judge evaluation and only run benchmark simulation.",
    )
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Skip simulation and evaluate existing logs in output-dir only.",
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Analyze existing raw_results.json in output-dir only.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.analyze_only:
        output_dir = Path(args.output_dir)
        raw_json = output_dir / RAW_RESULTS_JSON
        if not raw_json.exists():
            raise SystemExit(f"raw_results.json not found in {output_dir}")

        raw_results = json.loads(raw_json.read_text(encoding="utf-8"))
        if not isinstance(raw_results, list):
            raise SystemExit("Invalid raw_results.json format.")

        _, by_combo, _, conclusion, radar_path, bar_path = analyze_raw_results(
            raw_results=raw_results,
            output_dir=output_dir,
        )
        print("" + conclusion)
        print("Average by model/technique:")
        print(by_combo.to_string())
        print(f"Saved radar chart: {radar_path}")
        print(f"Saved grouped bar chart: {bar_path}")
        return

    load_local_env()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is missing.")

    client = OpenAI(api_key=api_key)
    data = load_interview_data()
    output_dir = Path(args.output_dir)

    simulation_results: list[Dict[str, Any]] = []

    if not args.evaluate_only:
        simulation_results = run_benchmark_matrix(
            client=client,
            models=args.models,
            techniques=args.techniques,
            data=data,
            output_dir=output_dir,
        )

        summary_path = output_dir / args.results_csv
        output_dir.mkdir(parents=True, exist_ok=True)
        with summary_path.open("w", encoding="utf-8", newline="") as fp:
            writer = csv.DictWriter(
                fp,
                fieldnames=[
                    "run_id",
                    "model",
                    "technique",
                    "run_start",
                    "run_end",
                    "assistant_turn_count",
                    "user_turn_count",
                    "error_message",
                    "total_prompt_tokens",
                    "total_completion_tokens",
                    "total_tokens",
                    "avg_assistant_latency_ms",
                ],
            )
            writer.writeheader()
            writer.writerows(simulation_results)
        print(f"Simulation summary saved: {summary_path}")

    if args.no_evaluate:
        print("Skipping judge evaluation due to --no-evaluate")
        return

    raw_results, detailed_records = evaluate_benchmark_logs(
        client=client,
        output_dir=output_dir,
        judge_model=args.judge_model,
    )
    raw_json_path = output_dir / RAW_RESULTS_JSON
    print(f"Saved raw judge results: {raw_json_path}")
    print(f"Saved detailed judge records: {output_dir / 'judge_detailed_records.json'}")

    try:
        detail_path = output_dir / "judge_detailed_records.json"
        detail_path.write_text(json.dumps(detailed_records, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass

    if pd is not None:
        try:
            df, by_combo, model_tech, conclusion, radar_path, bar_path = analyze_raw_results(
                raw_results=raw_results,
                output_dir=output_dir,
            )
            print("\n" + conclusion)
            print("\nAverage by model/technique:")
            print(by_combo.to_string())
            print(f"Saved radar chart: {radar_path}")
            print(f"Saved grouped bar chart: {bar_path}")
        except Exception as exc:
            print(f"Analysis skipped due to error: {exc}")


if __name__ == "__main__":
    main()
