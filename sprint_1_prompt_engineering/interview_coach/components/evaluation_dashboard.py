from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st

try:
    import pandas as pd
except ImportError:  # pragma: no cover - graceful fallback for environments without pandas
    pd = None

DIMENSION_SCORES = {
    "technical_depth": "technical_depth_score",
    "logical_flow": "logical_flow_score",
    "constructive_feedback": "constructive_feedback_score",
}
DIMENSION_LABELS = {
    "technical_depth": "전문성 (Technical Depth)",
    "logical_flow": "논리적 흐름 (Logical Flow)",
    "constructive_feedback": "피드백의 유용성 (Constructive Feedback)",
}
REASONING_FIELDS = {
    "technical_depth": "technical_depth_reasoning",
    "logical_flow": "logical_flow_reasoning",
    "constructive_feedback": "constructive_feedback_reasoning",
}


def _default_report_candidates() -> list[Path]:
    project_root = Path(__file__).resolve().parents[1]
    candidates = [
        Path.cwd() / "evaluation_report.csv",
        project_root / "evaluation_report.csv",
    ]
    # De-duplicate while preserving order
    uniq: list[Path] = []
    for path in candidates:
        if path not in uniq:
            uniq.append(path)
    return uniq


def _coerce_numeric(series: pd.Series) -> pd.Series:
    if pd is None:
        raise RuntimeError("pandas is required for dashboard rendering.")
    return pd.to_numeric(series, errors="coerce")


def _load_report(
    uploaded_file: Any | None,
    explicit_path: str,
) -> tuple[pd.DataFrame | None, str | None]:
    if pd is None:
        raise RuntimeError("pandas is required for CSV loading.")
    if uploaded_file is not None:
        try:
            return pd.read_csv(uploaded_file), uploaded_file.name
        except Exception:
            return None, None

    candidate_paths = _default_report_candidates()
    explicit = explicit_path.strip()
    if explicit:
        candidate_paths = [Path(explicit)] + candidate_paths

    for path in candidate_paths:
        if path.exists():
            try:
                return pd.read_csv(path), str(path)
            except Exception:
                return None, None
    return None, None


def _build_improvement_rows(df: pd.DataFrame, low_threshold: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for _, raw_row in df.iterrows():
        row = raw_row.to_dict()
        for dim, score_col in DIMENSION_SCORES.items():
            if score_col not in df.columns:
                continue
            score = _coerce_numeric(pd.Series([row.get(score_col)])).iloc[0]
            if pd.isna(score) or score > low_threshold:
                continue

            reason_col = REASONING_FIELDS.get(dim)
            reasoning = ""
            if reason_col and reason_col in df.columns:
                reasoning = str(row.get(reason_col, "")).strip()
            if not reasoning:
                reasoning = "No detailed reasoning provided by judge output."

            rows.append(
                {
                    "Case ID": row.get("case_id", ""),
                    "Scenario": row.get("scenario", ""),
                    "Prompt Mode": row.get("prompt_mode", "default"),
                    "평가 항목": DIMENSION_LABELS.get(dim, dim),
                    "점수": int(score),
                    "판사 피드백": reasoning,
                }
            )

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values(["점수", "Case ID"]).reset_index(drop=True)


def render_evaluation_dashboard() -> None:
    if pd is None:
        st.error("pandas가 설치되어 있지 않습니다. `pip install pandas`를 실행해 주세요.")
        return
    st.subheader("평가 대시보드")
    st.caption("면접 평가 스크립트로 생성된 `evaluation_report.csv`를 불러와 성능을 확인합니다.")

    default_path = str(_default_report_candidates()[0]) if _default_report_candidates() else ""
    report_path = st.text_input("evaluation_report.csv 경로", value=default_path, key="eval_report_path")
    uploaded_report = st.file_uploader("또는 CSV 파일 업로드", type=["csv"], key="eval_report_uploader")

    if not uploaded_report and not report_path.strip():
        st.info("CSV 경로를 입력하거나 업로드해주세요.")
        return

    report_df, report_source = _load_report(uploaded_report, report_path)
    if report_df is None or report_df.empty:
        st.warning("유효한 평가 CSV를 찾지 못했습니다. 경로 또는 업로드한 파일을 확인해주세요.")
        return

    st.success(f"로드 완료: {report_source}")
    st.write("")

    for col in list(DIMENSION_SCORES.values()) + ["overall_weighted_score"]:
        if col in report_df.columns:
            report_df[col] = _coerce_numeric(report_df[col])

    available_dimension_cols = [
        score_col
        for score_col in DIMENSION_SCORES.values()
        if score_col in report_df.columns
    ]
    if not available_dimension_cols:
        st.error("평가 점수 컬럼이 없어 차트를 그릴 수 없습니다.")
        return

    dimension_means = report_df[available_dimension_cols].mean(axis=0, skipna=True).round(2)
    dimension_means.index = [
        key for key, value in DIMENSION_SCORES.items() if value in dimension_means.index
    ]
    dimension_means = dimension_means.rename(
        index={key: DIMENSION_LABELS[key] for key in dimension_means.index}
    )

    col_avg_overall, col_highlights = st.columns([2, 1])
    with col_avg_overall:
        st.subheader("평균 점수 (항목별)")
        st.bar_chart(dimension_means.to_frame("평균 점수"), use_container_width=True)
    with col_highlights:
        st.subheader("총점 요약")
        if "overall_weighted_score" in report_df.columns:
            st.metric("평균 종합 점수", f"{report_df['overall_weighted_score'].mean():.2f}")
        st.metric("총 샘플 수", len(report_df))
        if "prompt_mode" in report_df.columns:
            st.metric("프롬프트 모드 수", report_df["prompt_mode"].nunique(dropna=True))

    if "prompt_mode" in report_df.columns and report_df["prompt_mode"].nunique(dropna=True) > 1:
        st.subheader("프롬프트 모드별 점수 비교")
        rename_map = {
            col: DIMENSION_LABELS[dim]
            for dim, col in DIMENSION_SCORES.items()
            if col in report_df.columns
        }
        compare_cols = available_dimension_cols.copy()
        if "overall_weighted_score" in report_df.columns:
            rename_map["overall_weighted_score"] = "종합 점수"
            compare_cols.append("overall_weighted_score")
        by_prompt = (
            report_df.groupby("prompt_mode")[compare_cols]
            .mean(numeric_only=True)
            .round(2)
        )
        by_prompt = by_prompt.rename(columns=rename_map)
        st.bar_chart(by_prompt.T, use_container_width=True)

    st.subheader("개선 필요 사항 (낮은 점수 항목)")
    low_threshold = int(st.slider("임계값 (이 점수 이하를 개선 필요로 표시)", 1, 5, 2))
    improvements_df = _build_improvement_rows(report_df, low_threshold)
    if improvements_df.empty:
        st.success("현재 보고서에서 임계값 이하 항목이 없습니다. 개선 포인트가 거의 없습니다.")
    else:
        st.dataframe(improvements_df, use_container_width=True)

    st.subheader("상세 레코드")
    sort_cols = [
        col
        for col in ["case_id", "overall_weighted_score"]
        if col in report_df.columns
    ]
    detail_df = (
        report_df.sort_values(
            by=sort_cols,
            ascending=[True] * len(sort_cols),
            na_position="last",
        )
        if sort_cols
        else report_df
    )
    st.dataframe(
        detail_df,
        use_container_width=True,
    )
