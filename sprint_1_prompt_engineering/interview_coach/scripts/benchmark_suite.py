#!/usr/bin/env python3
"""Benchmark ADAS interview simulations and evaluate logs with an LLM judge."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, Callable, Dict

from openai import OpenAI

try:
    from ..services.interview_ops import load_local_env
    from ..utils.data_loader import load_interview_data
    from .benchmark_suite_analysis import (
        analyze_raw_results as _analyze_raw_results,
        _plot_grouped_overall_bar,
        _plot_radar_chart,
        _to_long_df,
        pd,
    )
    from .benchmark_suite_config import (
        GROUPED_BAR_PATH,
        JUDGE_MODEL_DEFAULT,
        LOG_ROOT,
        MODEL_LIST,
        RADAR_CHART_PATH,
        RAW_RESULTS_JSON,
        RESULTS_CSV,
        TECHNIQUE_LIST,
    )
    from .benchmark_suite_judge import (
        _call_judge as _call_judge_impl,
        _coerce_score,
        _extract_turn_pairs_from_transcript,
        _normalize_judge_payload,
        _parse_judge_json,
        evaluate_benchmark_logs as _evaluate_benchmark_logs,
        load_benchmark_logs as _load_benchmark_logs,
    )
    from .benchmark_suite_simulation import (
        TurnRecord,
        UserProxy,
        run_benchmark_matrix,
        run_one_simulation,
    )
except ImportError:  # Fallback when running as a top-level module.
    from services.interview_ops import load_local_env
    from utils.data_loader import load_interview_data
    from scripts.benchmark_suite_analysis import (
        analyze_raw_results as _analyze_raw_results,
        _plot_grouped_overall_bar,
        _plot_radar_chart,
        _to_long_df,
        pd,
    )
    from scripts.benchmark_suite_config import (
        GROUPED_BAR_PATH,
        JUDGE_MODEL_DEFAULT,
        LOG_ROOT,
        MODEL_LIST,
        RADAR_CHART_PATH,
        RAW_RESULTS_JSON,
        RESULTS_CSV,
        TECHNIQUE_LIST,
    )
    from scripts.benchmark_suite_judge import (
        _call_judge as _call_judge_impl,
        _coerce_score,
        _extract_turn_pairs_from_transcript,
        _normalize_judge_payload,
        _parse_judge_json,
        evaluate_benchmark_logs as _evaluate_benchmark_logs,
        load_benchmark_logs as _load_benchmark_logs,
    )
    from scripts.benchmark_suite_simulation import (
        TurnRecord,
        UserProxy,
        run_benchmark_matrix,
        run_one_simulation,
    )


def _call_judge(
    client: OpenAI,
    model: str,
    payload: Dict[str, Any],
) -> Dict[str, Any] | None:
    return _call_judge_impl(client=client, model=model, payload=payload)


def analyze_raw_results(
    raw_results: list[dict[str, Any]],
    output_dir: Path,
) -> tuple["pd.DataFrame", "pd.DataFrame", "pd.DataFrame", str, Path, Path]:
    return _analyze_raw_results(
        raw_results=raw_results,
        output_dir=output_dir,
        plot_radar_chart=_plot_radar_chart,
        plot_grouped_overall_bar=_plot_grouped_overall_bar,
    )


def evaluate_benchmark_logs(
    client: OpenAI,
    output_dir: Path,
    judge_model: str,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]]]:
    return _evaluate_benchmark_logs(
        client=client,
        output_dir=output_dir,
        judge_model=judge_model,
        progress_callback=progress_callback,
        call_judge=_call_judge,
    )


def load_benchmark_logs(
    output_dir: Path,
    raw_file_names: set[str] | None = None,
) -> list[Dict[str, Any]]:
    return _load_benchmark_logs(output_dir=output_dir, raw_file_names=raw_file_names)


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
    detail_path = output_dir / "judge_detailed_records.json"
    detail_path.write_text(
        json.dumps(detailed_records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved raw judge results: {output_dir / RAW_RESULTS_JSON}")
    print(f"Saved detailed judge records: {detail_path}")

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
    except (RuntimeError, ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"Analysis skipped due to error: {exc}")


# Backward-compatible exports for tests and existing call sites.
__all__ = [
    "TurnRecord",
    "UserProxy",
    "run_one_simulation",
    "run_benchmark_matrix",
    "evaluate_benchmark_logs",
    "load_benchmark_logs",
    "analyze_raw_results",
    "_coerce_score",
    "_parse_judge_json",
    "_normalize_judge_payload",
    "_extract_turn_pairs_from_transcript",
    "_call_judge",
    "_plot_radar_chart",
    "_plot_grouped_overall_bar",
    "_to_long_df",
    "pd",
    "JUDGE_MODEL_DEFAULT",
    "MODEL_LIST",
    "TECHNIQUE_LIST",
    "RESULTS_CSV",
    "RAW_RESULTS_JSON",
    "RADAR_CHART_PATH",
    "GROUPED_BAR_PATH",
    "load_interview_data",
]


if __name__ == "__main__":
    main()
