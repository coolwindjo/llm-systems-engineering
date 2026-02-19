"""LLM judge evaluation for benchmark transcripts."""

from __future__ import annotations

import json
from statistics import mean
from typing import Any, Callable, Dict, List
from pathlib import Path

from openai import OpenAI
try:
    from openai import APIError
except (ImportError, ModuleNotFoundError):
    class APIError(Exception):
        """Fallback APIError when openai is unavailable."""

try:
    from .benchmark_suite_config import JUDGE_SYSTEM_PROMPT, RAW_RESULTS_JSON
except ImportError:  # Fallback when running as a top-level module.
    from scripts.benchmark_suite_config import JUDGE_SYSTEM_PROMPT, RAW_RESULTS_JSON


def _coerce_score(value: Any, minimum: int = 1, maximum: int = 10) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return minimum
    return max(minimum, min(maximum, parsed))


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
    if parse_failed or not isinstance(raw, dict):
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
            "evidence_check": "",
        }

    tech = raw.get("technical_depth", {})
    ctx = raw.get("context_awareness", {})
    pro = raw.get("professionalism", {})
    penalty = raw.get("aspice_evidence_penalty_applied", {})

    return {
        "technical_depth_score": _coerce_score(tech.get("score", 1), 1, 10),
        "context_awareness_score": _coerce_score(ctx.get("score", 1), 1, 10),
        "professionalism_score": _coerce_score(pro.get("score", 1), 1, 10),
        "overall_reasoning": str(raw.get("overall_reasoning", "")) or "No overall reasoning provided.",
        "technical_reasoning": str(tech.get("reasoning", "")) or "No technical reasoning provided.",
        "context_reasoning": str(ctx.get("reasoning", "")) or "No context reasoning provided.",
        "professionalism_reasoning": str(pro.get("reasoning", "")) or "No professionalism reasoning provided.",
        "aspice_penalty_applied": bool(penalty.get("technical_depth", False)),
        "aspice_penalty_notes": str(penalty.get("notes", "")),
        "evidence_check": str(tech.get("evidence_check", "")),
    }


def _extract_turn_pairs_from_transcript(
    log_data: Dict[str, Any],
    model: str,
    technique: str,
    run_id: str,
) -> list[Dict[str, Any]]:
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
        if (
            idx + 1 < len(transcript)
            and isinstance(transcript[idx + 1], dict)
            and transcript[idx + 1].get("role") == "user"
        ):
            candidate_answer = str(transcript[idx + 1].get("content", "")).strip()
        elif (
            idx > 0
            and isinstance(transcript[idx - 1], dict)
            and transcript[idx - 1].get("role") == "user"
        ):
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


def load_benchmark_logs(
    output_dir: Path,
    raw_file_names: set[str] | None = None,
) -> list[Dict[str, Any]]:
    if not output_dir.exists():
        return []
    skip_files = raw_file_names or {RAW_RESULTS_JSON, "benchmark_summary.csv", "benchmark_radar_chart.png", "benchmark_grouped_overall_bar.png"}
    loaded: list[Dict[str, Any]] = []
    for path in sorted(output_dir.glob("*.json")):
        if path.name in skip_files:
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
        pairs = _extract_turn_pairs_from_transcript(
            raw,
            model=model,
            technique=technique,
            run_id=run_id,
        )
        if not pairs:
            continue
        loaded.extend(pairs)
    return loaded


def _compute_run_averages(turn_scores: list[Dict[str, Any]]) -> tuple[dict[str, float], str]:
    if not turn_scores:
        avg_scores = {
            "technical_depth": 1,
            "context_awareness": 1,
            "professionalism": 1,
            "overall": 1,
        }
        reasoning = "No valid turn-level scores were produced."
        return avg_scores, reasoning

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
    return avg_scores, reasoning


def evaluate_benchmark_logs(
    client: OpenAI,
    output_dir: Path,
    judge_model: str,
    progress_callback: Callable[[int, int, str], None] | None = None,
    call_judge: Callable[[OpenAI, str, Dict[str, Any]], Dict[str, Any] | None] | None = None,
) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]]]:
    pairs = load_benchmark_logs(output_dir)
    if not pairs:
        raise RuntimeError("No benchmark logs found. Run simulation first or provide valid benchmark_logs directory.")

    judge_caller = call_judge or _call_judge

    grouped_pairs: Dict[str, list[Dict[str, Any]]] = {}
    for pair in pairs:
        key = str(pair.get("run_id", "unknown"))
        grouped_pairs.setdefault(key, []).append(pair)

    ordered_run_ids = list(grouped_pairs.keys())
    raw_results: list[Dict[str, Any]] = []
    detailed_records: list[Dict[str, Any]] = []

    total = len(ordered_run_ids)
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
                raw = judge_caller(client=client, model=judge_model, payload=payload)
                parse_failed = raw is None
            except (APIError, TypeError, ValueError, AttributeError, KeyError):
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

        avg_scores, reasoning = _compute_run_averages(turn_scores)
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
    output_path.write_text(json.dumps(raw_results, ensure_ascii=False, indent=2), encoding="utf-8")
    return raw_results, detailed_records
