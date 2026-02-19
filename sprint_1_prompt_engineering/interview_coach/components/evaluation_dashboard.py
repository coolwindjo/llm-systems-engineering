from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import streamlit as st

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None

from tests import benchmark_suite as bench


def _default_log_dir() -> Path:
    project_root = Path(__file__).resolve().parents[1]
    return project_root / "tests" / "benchmark_logs"


def _parse_optional_json_file(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if isinstance(raw, list):
        return raw
    return []


def _to_df(raw_results: list[dict[str, Any]]) -> Any:
    if pd is None:
        raise RuntimeError("pandas is required to render tables in the dashboard.")

    return pd.DataFrame(raw_results)


def _expand_scores(df: "pd.DataFrame") -> "pd.DataFrame":
    if "scores" not in df.columns:
        return df

    expanded = pd.json_normalize(df["scores"])
    expanded = expanded.rename(
        columns={
            "technical_depth": "technical_depth",
            "context_awareness": "context_awareness",
            "professionalism": "professionalism",
            "overall": "overall",
        }
    )
    return pd.concat([df[["model", "technique"]].reset_index(drop=True), expanded], axis=1)


def _load_results(raw_path: Path) -> list[dict[str, Any]]:
    if not raw_path.exists():
        return []

    return _parse_optional_json_file(raw_path)


def _render_score_chart(path: Path, caption: str) -> None:
    if path.exists():
        st.image(str(path), caption=caption)
    else:
        st.info(f"Waiting to generate: {caption}")


def render_evaluation_dashboard(get_api_key: Callable[[], str | None] | None = None) -> None:
    st.subheader("Admin Dashboard")
    st.caption("Run benchmark matrix, evaluate with judge model, and visualize results.")

    if pd is None:
        st.error("pandas is required. Install with `pip install pandas`. ")
        return

    output_dir = Path(st.text_input("Benchmark output directory", value=str(_default_log_dir()), key="admin_output_dir"))
    model_default = list(bench.MODEL_LIST)
    technique_default = list(bench.TECHNIQUE_LIST)

    selected_models = st.multiselect(
        "Select models to benchmark",
        options=model_default,
        default=model_default,
        key="admin_models",
    )
    selected_techniques = st.multiselect(
        "Select prompt techniques",
        options=technique_default,
        default=technique_default,
        key="admin_techniques",
    )
    judge_model = st.text_input(
        "Judge model",
        value=bench.JUDGE_MODEL_DEFAULT,
        key="admin_judge_model",
    )

    run_cols = st.columns([1, 1, 2])
    run_clicked = run_cols[0].button("Start Benchmark + Judge", use_container_width=True)
    refresh_clicked = run_cols[1].button("Reload Existing Results", use_container_width=True)

    if run_clicked:
        if not selected_models:
            st.warning("Select at least one model.")
        elif not selected_techniques:
            st.warning("Select at least one technique.")
        elif OpenAI is None:
            st.error("OpenAI SDK is not installed.")
        else:
            api_key = get_api_key() if get_api_key else None
            if not api_key:
                st.error("OpenAI API key is required. Check STREAMLIT secrets or OPENAI_API_KEY.")
            else:
                client = OpenAI(api_key=api_key)
                output_dir.mkdir(parents=True, exist_ok=True)

                data = bench.load_interview_data()
                total_steps = len(selected_models) * len(selected_techniques)
                progress = st.progress(0.0)
                status = st.empty()

                def cb(done: int, total: int, message: str) -> None:
                    ratio = min(1.0, done / max(1, total))
                    status.text(message)
                    progress.progress(ratio)

                status.text("Running benchmark simulations...")
                cb(0, total_steps, "Running benchmark simulations...")
                bench.run_benchmark_matrix(
                    client=client,
                    models=selected_models,
                    techniques=selected_techniques,
                    data=data,
                    output_dir=output_dir,
                    progress_callback=cb,
                )

                status.text("Running judge evaluation...")
                cb(0, max(1, total_steps), "Running judge evaluation...")
                raw_results, detailed_records = bench.evaluate_benchmark_logs(
                    client=client,
                    output_dir=output_dir,
                    judge_model=judge_model or bench.JUDGE_MODEL_DEFAULT,
                    progress_callback=cb,
                )

                detailed_path = output_dir / "judge_detailed_records.json"
                detailed_path.write_text(
                    json.dumps(detailed_records, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

                status.text("Generating reports and charts...")
                df, by_combo, model_tech, conclusion, radar_path, bar_path = bench.analyze_raw_results(
                    raw_results=raw_results,
                    output_dir=output_dir,
                )
                st.success("Benchmark and evaluation finished.")
                st.markdown(conclusion)

                st.markdown("### Average Scores by Model + Technique")
                st.dataframe(by_combo.reset_index(), use_container_width=True)
                st.markdown("### Model x Technique Overall Table")
                st.dataframe(model_tech.reset_index(), use_container_width=True)

                st.markdown("### Detailed judge summaries")
                expanded_df = _expand_scores(_to_df(raw_results))
                st.dataframe(expanded_df, use_container_width=True)

                st.markdown("### Visualizations")
                vis_cols = st.columns(2)
                with vis_cols[0]:
                    _render_score_chart(radar_path, "Radar Chart by Prompt Technique")
                with vis_cols[1]:
                    _render_score_chart(bar_path, "Overall Score by Model and Technique")

                status.empty()
                progress.empty()
                st.session_state["admin_raw_path"] = str(output_dir / bench.RAW_RESULTS_JSON)
                return

    if refresh_clicked:
        raw_path = output_dir / bench.RAW_RESULTS_JSON
        if not raw_path.exists():
            st.warning("No raw_results.json exists yet in selected output path.")
        else:
            raw_results = _load_results(raw_path)
            if not raw_results:
                st.warning("raw_results.json is empty or malformed.")
            else:
                st.success(f"Loaded {len(raw_results)} result records from {raw_path}")
                try:
                    df, by_combo, model_tech, conclusion, radar_path, bar_path = bench.analyze_raw_results(
                        raw_results=raw_results,
                        output_dir=output_dir,
                    )
                    st.markdown(conclusion)
                    st.markdown("### Average Scores by Model + Technique")
                    st.dataframe(by_combo.reset_index(), use_container_width=True)
                    st.markdown("### Model x Technique Overall Table")
                    st.dataframe(model_tech.reset_index(), use_container_width=True)
                    st.markdown("### Detailed judge summaries")
                    expanded_df = _expand_scores(_to_df(raw_results))
                    st.dataframe(expanded_df, use_container_width=True)

                    st.markdown("### Visualizations")
                    vis_cols = st.columns(2)
                    with vis_cols[0]:
                        _render_score_chart(radar_path, "Radar Chart by Prompt Technique")
                    with vis_cols[1]:
                        _render_score_chart(bar_path, "Overall Score by Model and Technique")
                except Exception as exc:
                    st.error(f"Failed to analyze cached results: {exc}")

    raw_path = output_dir / bench.RAW_RESULTS_JSON
    st.session_state.setdefault("admin_raw_path", str(raw_path))

    if raw_path.exists():
        try:
            raw_results = _load_results(raw_path)
            if raw_results:
                st.caption(f"Last run file: {raw_path}")
        except Exception:
            pass
