"""Analysis and visualization utilities for benchmark judge results."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

try:
    from matplotlib import pyplot as plt
except (ImportError, ModuleNotFoundError):  # pragma: no cover - optional visualization dependency
    plt = None

try:
    import numpy as np
except (ImportError, ModuleNotFoundError):  # pragma: no cover - optional dependency
    np = None

try:
    import pandas as pd
except (ImportError, ModuleNotFoundError):  # pragma: no cover - optional dependency
    pd = None

try:
    from .benchmark_suite_config import (
        GROUPED_BAR_PATH,
        RADAR_CHART_PATH,
    )
except ImportError:  # Fallback when running as a top-level module.
    from scripts.benchmark_suite_config import (
        GROUPED_BAR_PATH,
        RADAR_CHART_PATH,
    )


def _coerce_score(value: Any, fallback: float = 0.0) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return fallback
    return max(0.0, min(10.0, score))


def _to_long_df(raw_results: list[dict[str, Any]]) -> "pd.DataFrame":
    if pd is None:
        raise RuntimeError("pandas is required for analysis output.")
    if not raw_results:
        return pd.DataFrame(columns=["model", "technique", "technical_depth", "context_awareness", "professionalism", "overall"])

    normalized_rows: list[dict[str, Any]] = []
    for item in raw_results:
        scores = item.get("scores", {})
        normalized_rows.append(
            {
                "model": item.get("model", ""),
                "technique": item.get("technique", ""),
                "technical_depth": _coerce_score(
                    scores.get("technical_depth", scores.get("technical_depth_score", 0) or 0) or 0
                ),
                "context_awareness": _coerce_score(
                    scores.get("context_awareness", scores.get("context_awareness_score", 0) or 0) or 0
                ),
                "professionalism": _coerce_score(
                    scores.get("professionalism", scores.get("professionalism_score", 0) or 0) or 0
                ),
                "overall": _coerce_score(scores.get("overall", 0) or 0),
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
    if plt is None or by_combo.empty:
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
    raw_results: list[dict[str, Any]],
    output_dir: Path,
    plot_radar_chart: Callable[["pd.DataFrame", Path], None] | None = None,
    plot_grouped_overall_bar: Callable[["pd.DataFrame", Path], None] | None = None,
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
    by_technique = by_combo.groupby("technique")[
        ["technical_depth", "context_awareness", "professionalism"]
    ].mean(numeric_only=True)
    model_tech_table = by_combo["overall"].unstack("technique")

    radar_path = output_dir / RADAR_CHART_PATH
    bar_path = output_dir / GROUPED_BAR_PATH

    radar_plot = plot_radar_chart or _plot_radar_chart
    bar_plot = plot_grouped_overall_bar or _plot_grouped_overall_bar

    radar_plot(by_technique, radar_path)
    bar_plot(by_combo["overall"].unstack("technique"), bar_path)

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
