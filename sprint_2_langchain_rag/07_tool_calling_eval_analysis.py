import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from openai import OpenAI
from ragas.llms import llm_factory
from ragas.metrics import DiscreteMetric


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "analysis_outputs"
CACHE_DIR = PROJECT_ROOT / ".cache"
MPL_CACHE_DIR = CACHE_DIR / "matplotlib"
DOCS_DIR = PROJECT_ROOT / "docs"
NOTEBOOK_GUIDE_PATH = DOCS_DIR / "07_tool_calling_eval_results_ko.md"

DEFAULT_AGENT_MODELS = [
    "gpt-4o-mini",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
    "gpt-5-mini",
]
DEFAULT_JUDGE_MODELS = [
    "gpt-4o-mini",
    "gpt-5-mini",
]

OUTCOME_ORDER = [
    "strict_match",
    "dubious_correct_match",
    "judge_challenge",
    "true_error",
    "execution_error",
    "judge_error",
]
OUTCOME_COLORS = {
    "strict_match": "#4C78A8",
    "dubious_correct_match": "#F58518",
    "judge_challenge": "#E45756",
    "true_error": "#72B7B2",
    "execution_error": "#B279A2",
    "judge_error": "#9D755D",
}


@dataclass(frozen=True)
class TestCase:
    case_id: str
    query: str
    expected_tool: str | None
    expected_params: dict[str, str] | None
    difficulty: str
    ambiguity_focus: bool
    note: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run multi-model tool-calling evaluation with deterministic scoring and "
            "DiscreteMetric judge scoring, then save tables and visualizations."
        )
    )
    parser.add_argument(
        "--agent-models",
        nargs="+",
        default=DEFAULT_AGENT_MODELS,
        help="Agent models to evaluate.",
    )
    parser.add_argument(
        "--judge-models",
        nargs="+",
        default=DEFAULT_JUDGE_MODELS,
        help="Judge models used by DiscreteMetric.",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Optional limit for the number of test cases.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Retries for agent or judge API calls.",
    )
    return parser.parse_args()


def configure_runtime() -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))
    MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")


def load_api_key() -> str:
    load_dotenv(PROJECT_ROOT.parent / ".env")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set. Add it to /workspace/.env before running this script.")
    return api_key


@tool
def search_jobs(role: str, location: str) -> str:
    """Search for job openings matching a specific role and location."""
    return json.dumps(
        {
            "jobs": [
                {"title": f"Senior {role}", "company": "TechCorp", "location": location},
                {"title": role, "company": "DataInc", "location": location},
            ]
        }
    )


@tool
def compare_salaries(role: str, location: str) -> str:
    """Compare average salaries for a given role in a specific location."""
    return json.dumps(
        {
            "role": role,
            "location": location,
            "average_salary": 78500,
            "currency": "EUR",
            "range": {"min": 65000, "max": 92000},
        }
    )


@tool
def analyze_resume(resume_text: str) -> str:
    """Analyze a resume and provide improvement suggestions."""
    return json.dumps(
        {
            "score": 7,
            "suggestions": [
                "Add more quantifiable achievements",
                "Include relevant certifications",
            ],
        }
    )


TOOLS = [search_jobs, compare_salaries, analyze_resume]


def build_test_cases() -> list[TestCase]:
    return [
        TestCase(
            case_id="salary_berlin_clear",
            query="Average salary for data scientist in Berlin?",
            expected_tool="compare_salaries",
            expected_params={"role": "data scientist", "location": "berlin"},
            difficulty="easy",
            ambiguity_focus=False,
            note="Clear salary lookup.",
        ),
        TestCase(
            case_id="jobs_london_clear",
            query="Find me ML engineer jobs in London",
            expected_tool="search_jobs",
            expected_params={"role": "ml engineer", "location": "london"},
            difficulty="easy",
            ambiguity_focus=False,
            note="Clear job search request.",
        ),
        TestCase(
            case_id="resume_review_clear",
            query=(
                "Review my resume? Here it is: Experienced data scientist with 5 years "
                "in ML. Skills: Python, TensorFlow, SQL. Previous role: Senior Analyst at DataCorp."
            ),
            expected_tool="analyze_resume",
            expected_params=None,
            difficulty="easy",
            ambiguity_focus=False,
            note="Direct resume review.",
        ),
        TestCase(
            case_id="berlin_salary_indirect",
            query="Thinking about moving to Berlin, wondering what people like me make — I do data science",
            expected_tool="compare_salaries",
            expected_params=None,
            difficulty="medium",
            ambiguity_focus=True,
            note="Salary intent is indirect.",
        ),
        TestCase(
            case_id="cv_fix_hint",
            query="Friend told me to fix my CV. 3 years Python/ML at Google.",
            expected_tool="analyze_resume",
            expected_params=None,
            difficulty="medium",
            ambiguity_focus=False,
            note="Resume help via indirect phrasing.",
        ),
        TestCase(
            case_id="salary_worded_as_search",
            query="Can you search for how much a product manager earns?",
            expected_tool="compare_salaries",
            expected_params=None,
            difficulty="medium",
            ambiguity_focus=True,
            note="Uses the verb 'search' but asks for salary.",
        ),
        TestCase(
            case_id="skills_advice_no_tool",
            query="What skills should I learn for ML engineering?",
            expected_tool=None,
            expected_params=None,
            difficulty="medium",
            ambiguity_focus=False,
            note="General advice should not use tools.",
        ),
        TestCase(
            case_id="positions_and_pay",
            query="Need to know about ML engineer positions and their pay in Amsterdam",
            expected_tool="compare_salaries",
            expected_params=None,
            difficulty="hard",
            ambiguity_focus=True,
            note="Blends jobs and salary; canonical label prefers salary.",
        ),
        TestCase(
            case_id="compare_job_markets",
            query="Compare the job markets in Berlin and London for data scientists",
            expected_tool="search_jobs",
            expected_params=None,
            difficulty="hard",
            ambiguity_focus=True,
            note="Market comparison may invite salary instead of job search.",
        ),
        TestCase(
            case_id="resume_salary_blend",
            query="Resume: Python, SQL, 2 years. What salary should I expect?",
            expected_tool="compare_salaries",
            expected_params=None,
            difficulty="hard",
            ambiguity_focus=True,
            note="Resume context can pull the agent toward resume analysis.",
        ),
        TestCase(
            case_id="everything_munich",
            query="Tell me everything about data engineering in Munich",
            expected_tool="search_jobs",
            expected_params=None,
            difficulty="hard",
            ambiguity_focus=True,
            note="Broad market query could plausibly trigger salary comparison.",
        ),
        TestCase(
            case_id="help_my_job",
            query="I hate my job. Help me.",
            expected_tool=None,
            expected_params=None,
            difficulty="edge_case",
            ambiguity_focus=False,
            note="Emotional support request.",
        ),
        TestCase(
            case_id="hot_market",
            query="What's hot in the tech job market right now?",
            expected_tool=None,
            expected_params=None,
            difficulty="edge_case",
            ambiguity_focus=False,
            note="Trend advice without a specific retrieval need.",
        ),
        TestCase(
            case_id="tell_joke",
            query="Tell me a joke",
            expected_tool=None,
            expected_params=None,
            difficulty="edge_case",
            ambiguity_focus=False,
            note="No tool should be used.",
        ),
        TestCase(
            case_id="city_pay_and_openings",
            query="I'm choosing between Berlin and London for ML roles. Which city pays better and has more openings?",
            expected_tool="compare_salaries",
            expected_params=None,
            difficulty="hard",
            ambiguity_focus=True,
            note="Salary and openings are both reasonable primary intents.",
        ),
        TestCase(
            case_id="resume_strength_and_pay",
            query="Does my resume look strong enough for senior data science jobs in Paris, and what do those roles usually pay?",
            expected_tool="analyze_resume",
            expected_params=None,
            difficulty="hard",
            ambiguity_focus=True,
            note="Resume assessment and salary lookup are both plausible first steps.",
        ),
        TestCase(
            case_id="roles_barcelona_comp",
            query="Could you look up product roles in Barcelona? I'm mainly trying to gauge compensation.",
            expected_tool="compare_salaries",
            expected_params=None,
            difficulty="hard",
            ambiguity_focus=True,
            note="Search phrasing plus compensation goal.",
        ),
        TestCase(
            case_id="better_for_mlops",
            query="I want to know whether Munich or Amsterdam is better for MLOps careers.",
            expected_tool="search_jobs",
            expected_params=None,
            difficulty="hard",
            ambiguity_focus=True,
            note="Could mean openings, salary, or general advice.",
        ),
        TestCase(
            case_id="cv_market_ready",
            query="Before I apply anywhere, tell me if my CV is market-ready for ML engineer roles in Berlin.",
            expected_tool="analyze_resume",
            expected_params=None,
            difficulty="hard",
            ambiguity_focus=True,
            note="Resume review with location-specific role context.",
        ),
        TestCase(
            case_id="berlin_expectation",
            query="If I moved to Berlin as a data engineer, what should I expect out there?",
            expected_tool="search_jobs",
            expected_params=None,
            difficulty="hard",
            ambiguity_focus=True,
            note="Open-ended market expectation often yields reasonable alternative tools.",
        ),
    ]


def build_agent(model_name: str) -> Any:
    model = ChatOpenAI(model=model_name, temperature=0, timeout=120)
    return create_agent(
        model=model,
        tools=TOOLS,
        system_prompt=(
            "You are a career consultant. Use your tools when you need real data. "
            "For general advice questions, respond directly without calling tools."
        ),
    )


def build_metric() -> DiscreteMetric:
    return DiscreteMetric(
        name="tool_selection",
        allowed_values=["correct", "incorrect"],
        prompt=(
            "Evaluate whether the AI agent selected an appropriate tool plan.\n\n"
            "Available tools:\n"
            "- search_jobs: search for job openings by role and location\n"
            "- compare_salaries: compare salary data for a role in a location\n"
            "- analyze_resume: analyze resume text and suggest improvements\n"
            "- NO TOOL: for general advice, off-topic, or conversational queries\n\n"
            "Canonical benchmark tool (used for deterministic scoring, but not always the only valid choice): {expected_tool}\n"
            "User query: {user_query}\n"
            "Actual primary tool: {actual_tool}\n"
            "Actual tool sequence: {actual_tool_sequence}\n"
            "Case note: {case_note}\n\n"
            "Judging rules:\n"
            "1. If the query is ambiguous or multi-intent, an alternative tool can still be CORRECT if it reasonably addresses a major part of the user's request.\n"
            "2. If no tool is needed for a conversational/advice query, selecting NO TOOL can be CORRECT.\n"
            "3. Mark INCORRECT only when the chosen tool is clearly irrelevant, obviously incomplete as a first step, or contradicts the request.\n\n"
            "Answer with only 'correct' or 'incorrect'."
        ),
    )


def call_with_retry(func, max_retries: int):
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            return func()
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt < max_retries:
                time.sleep(1.5 * (attempt + 1))
    raise last_error


def normalize_text(value: str) -> str:
    return " ".join(value.strip().lower().split())


def normalize_params(params: dict[str, Any] | None) -> dict[str, Any] | None:
    if params is None:
        return None
    normalized: dict[str, Any] = {}
    for key, value in params.items():
        if isinstance(value, str):
            normalized[key] = normalize_text(value)
        else:
            normalized[key] = value
    return normalized


def stringify_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    return str(content)


def extract_tool_calls(agent_response: dict[str, Any]) -> tuple[list[dict[str, Any]], str]:
    messages = agent_response.get("messages", [])
    tool_calls: list[dict[str, Any]] = []
    final_response = ""

    for msg in messages:
        calls = getattr(msg, "tool_calls", None)
        if calls:
            tool_calls.extend(calls)
        content = getattr(msg, "content", None)
        if content:
            final_response = stringify_content(content)

    return tool_calls, final_response


def evaluate_agent_case(agent: Any, case: TestCase, max_retries: int) -> dict[str, Any]:
    response = call_with_retry(
        lambda: agent.invoke({"messages": [{"role": "user", "content": case.query}]}),
        max_retries=max_retries,
    )
    tool_calls, final_response = extract_tool_calls(response)

    actual_tool = tool_calls[0]["name"] if tool_calls else None
    actual_params = tool_calls[0]["args"] if tool_calls else None
    tool_sequence = [call["name"] for call in tool_calls]

    tool_correct = actual_tool == case.expected_tool
    expected_params = normalize_params(case.expected_params)
    normalized_actual_params = normalize_params(actual_params if isinstance(actual_params, dict) else None)

    if case.expected_tool is None:
        params_correct = tool_correct
    elif case.expected_params is None:
        params_correct = tool_correct
    elif tool_correct and normalized_actual_params is not None:
        params_correct = normalized_actual_params == expected_params
    else:
        params_correct = False

    return {
        "case_id": case.case_id,
        "query": case.query,
        "difficulty": case.difficulty,
        "ambiguity_focus": case.ambiguity_focus,
        "case_note": case.note,
        "expected_tool": case.expected_tool,
        "expected_params_json": json.dumps(expected_params, ensure_ascii=False) if expected_params is not None else None,
        "actual_tool": actual_tool,
        "actual_tool_sequence": json.dumps(tool_sequence, ensure_ascii=False),
        "actual_tool_count": len(tool_sequence),
        "actual_params_json": json.dumps(normalized_actual_params, ensure_ascii=False) if normalized_actual_params is not None else None,
        "tool_correct": tool_correct,
        "params_correct": params_correct,
        "expected_tool_in_sequence": case.expected_tool in tool_sequence if case.expected_tool else pd.NA,
        "response_text": final_response,
        "agent_error": None,
    }


def score_case_with_judge(
    metric: DiscreteMetric,
    judge_llm: Any,
    row: dict[str, Any],
    max_retries: int,
) -> tuple[str, str]:
    score = call_with_retry(
        lambda: metric.score(
            llm=judge_llm,
            user_query=row["query"],
            expected_tool=str(row["expected_tool"]),
            actual_tool=str(row["actual_tool"]),
            actual_tool_sequence=row["actual_tool_sequence"],
            case_note=row["case_note"],
        ),
        max_retries=max_retries,
    )
    return score.value, score.reason


def classify_outcome(tool_correct: Any, judge_value: str | None, agent_error: Any, judge_error: Any) -> str:
    if pd.notna(agent_error):
        return "execution_error"
    if pd.notna(judge_error):
        return "judge_error"
    if judge_value is None or pd.isna(tool_correct):
        return "judge_error"

    judge_correct = judge_value == "correct"
    if bool(tool_correct) and judge_correct:
        return "strict_match"
    if (not bool(tool_correct)) and judge_correct:
        return "dubious_correct_match"
    if bool(tool_correct) and (not judge_correct):
        return "judge_challenge"
    return "true_error"


def safe_rate(numerator: int | float, denominator: int | float) -> float:
    if not denominator:
        return float("nan")
    return float(numerator) / float(denominator)


def summarize_pair_results(results_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (agent_model, judge_model), group in results_df.groupby(["agent_model", "judge_model"], dropna=False):
        valid = group[group["outcome_bucket"].isin(["strict_match", "dubious_correct_match", "judge_challenge", "true_error"])]
        ambiguous = valid[valid["ambiguity_focus"]]

        tool_correct_count = int(valid["tool_correct"].fillna(False).sum())
        param_correct_count = int(valid["params_correct"].fillna(False).sum())
        judge_pass_count = int((valid["judge_value"] == "correct").sum())
        agreement_count = int((valid["tool_correct"] == (valid["judge_value"] == "correct")).sum())
        dubious_count = int((valid["outcome_bucket"] == "dubious_correct_match").sum())
        ambiguous_dubious_count = int((ambiguous["outcome_bucket"] == "dubious_correct_match").sum())

        rows.append(
            {
                "agent_model": agent_model,
                "judge_model": judge_model,
                "total_rows": len(group),
                "scored_rows": len(valid),
                "execution_errors": int((group["outcome_bucket"] == "execution_error").sum()),
                "judge_errors": int((group["outcome_bucket"] == "judge_error").sum()),
                "det_tool_accuracy": safe_rate(tool_correct_count, len(valid)),
                "det_param_accuracy": safe_rate(param_correct_count, len(valid)),
                "judge_pass_rate": safe_rate(judge_pass_count, len(valid)),
                "agreement_rate": safe_rate(agreement_count, len(valid)),
                "dubious_correct_matches": dubious_count,
                "dubious_correct_rate": safe_rate(dubious_count, len(valid)),
                "ambiguous_rows": len(ambiguous),
                "ambiguous_dubious_matches": ambiguous_dubious_count,
                "ambiguous_dubious_rate": safe_rate(ambiguous_dubious_count, len(ambiguous)),
                "strict_matches": int((valid["outcome_bucket"] == "strict_match").sum()),
                "judge_challenges": int((valid["outcome_bucket"] == "judge_challenge").sum()),
                "true_errors": int((valid["outcome_bucket"] == "true_error").sum()),
            }
        )

    summary_df = pd.DataFrame(rows)
    if summary_df.empty:
        return summary_df
    return summary_df.sort_values(
        by=["ambiguous_dubious_rate", "dubious_correct_matches", "agreement_rate"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def summarize_agent_results(results_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for agent_model, group in results_df.groupby("agent_model"):
        valid = group[group["outcome_bucket"].isin(["strict_match", "dubious_correct_match", "judge_challenge", "true_error"])]
        ambiguous = valid[valid["ambiguity_focus"]]

        rows.append(
            {
                "agent_model": agent_model,
                "scored_rows": len(valid),
                "judge_models_used": group["judge_model"].nunique(dropna=True),
                "det_tool_accuracy": safe_rate(valid["tool_correct"].fillna(False).sum(), len(valid)),
                "judge_pass_rate": safe_rate((valid["judge_value"] == "correct").sum(), len(valid)),
                "agreement_rate": safe_rate(
                    (valid["tool_correct"] == (valid["judge_value"] == "correct")).sum(),
                    len(valid),
                ),
                "dubious_correct_matches": int((valid["outcome_bucket"] == "dubious_correct_match").sum()),
                "ambiguous_dubious_matches": int((ambiguous["outcome_bucket"] == "dubious_correct_match").sum()),
                "ambiguous_dubious_rate": safe_rate(
                    (ambiguous["outcome_bucket"] == "dubious_correct_match").sum(),
                    len(ambiguous),
                ),
            }
        )

    summary_df = pd.DataFrame(rows)
    if summary_df.empty:
        return summary_df
    return summary_df.sort_values(
        by=["ambiguous_dubious_rate", "dubious_correct_matches", "agreement_rate"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def build_case_summary(results_df: pd.DataFrame) -> pd.DataFrame:
    valid = results_df[results_df["outcome_bucket"].isin(["strict_match", "dubious_correct_match", "judge_challenge", "true_error"])]
    rows = []
    for case_id, group in valid.groupby("case_id"):
        first = group.iloc[0]
        dubious_count = int((group["outcome_bucket"] == "dubious_correct_match").sum())
        judge_pass_count = int((group["judge_value"] == "correct").sum())
        rows.append(
            {
                "case_id": case_id,
                "query": first["query"],
                "difficulty": first["difficulty"],
                "ambiguity_focus": bool(first["ambiguity_focus"]),
                "expected_tool": first["expected_tool"],
                "runs": len(group),
                "det_tool_accuracy": safe_rate(group["tool_correct"].fillna(False).sum(), len(group)),
                "judge_pass_rate": safe_rate(judge_pass_count, len(group)),
                "dubious_correct_matches": dubious_count,
                "dubious_correct_rate": safe_rate(dubious_count, len(group)),
            }
        )

    case_df = pd.DataFrame(rows)
    if case_df.empty:
        return case_df
    return case_df.sort_values(
        by=["dubious_correct_matches", "dubious_correct_rate", "difficulty"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def make_heatmap(summary_df: pd.DataFrame, output_path: Path) -> None:
    if summary_df.empty:
        return

    pivot = summary_df.pivot(index="agent_model", columns="judge_model", values="ambiguous_dubious_rate")
    matrix = pivot.to_numpy(dtype=float)
    finite_values = matrix[np.isfinite(matrix)]
    vmax = max(0.01, float(finite_values.max())) if finite_values.size else 0.01

    fig, ax = plt.subplots(figsize=(8, 5))
    image = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=vmax)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=25, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title("Ambiguous-Case Dubious Correct Rate by Agent/Judge Model")
    ax.set_xlabel("Judge model")
    ax.set_ylabel("Agent model")

    for row_idx, agent_model in enumerate(pivot.index):
        for col_idx, judge_model in enumerate(pivot.columns):
            value = pivot.loc[agent_model, judge_model]
            label = "NA" if pd.isna(value) else f"{value:.0%}"
            ax.text(col_idx, row_idx, label, ha="center", va="center", color="black", fontsize=10)

    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Rate")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def make_agent_outcome_plot(results_df: pd.DataFrame, output_path: Path) -> None:
    plot_df = results_df[
        (results_df["ambiguity_focus"])
        & (results_df["outcome_bucket"].isin(["strict_match", "dubious_correct_match", "judge_challenge", "true_error"]))
    ]
    if plot_df.empty:
        return

    counts = (
        plot_df.groupby(["agent_model", "outcome_bucket"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["strict_match", "dubious_correct_match", "judge_challenge", "true_error"], fill_value=0)
    )
    shares = counts.div(counts.sum(axis=1), axis=0)

    fig, ax = plt.subplots(figsize=(10, 6))
    bottom = np.zeros(len(shares))
    x = np.arange(len(shares))

    for bucket in shares.columns:
        values = shares[bucket].to_numpy()
        ax.bar(
            x,
            values,
            bottom=bottom,
            color=OUTCOME_COLORS[bucket],
            label=bucket.replace("_", " "),
        )
        bottom += values

    ax.set_xticks(x)
    ax.set_xticklabels(shares.index, rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Share of ambiguous-case outcomes")
    ax.set_title("Ambiguous-Case Outcome Mix by Agent Model")
    ax.legend(loc="upper center", ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def make_case_barh(case_summary_df: pd.DataFrame, output_path: Path) -> None:
    plot_df = case_summary_df[case_summary_df["dubious_correct_matches"] > 0].head(10)
    if plot_df.empty:
        return

    labels = [query[:58] + "..." if len(query) > 61 else query for query in plot_df["query"]]
    y = np.arange(len(plot_df))

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.barh(y, plot_df["dubious_correct_matches"], color=OUTCOME_COLORS["dubious_correct_match"])
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Dubious correct matches across model pairs")
    ax.set_title("Queries Most Likely to Reveal Reasonable Tool Flexibility")

    for idx, value in enumerate(plot_df["dubious_correct_matches"]):
        ax.text(value + 0.05, idx, str(int(value)), va="center")

    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def write_markdown_report(
    pair_summary_df: pd.DataFrame,
    dubious_df: pd.DataFrame,
    output_path: Path,
) -> None:
    lines = [
        "# Tool Calling Evaluation Analysis",
        "",
        "## Model Pair Summary",
        "",
    ]

    if pair_summary_df.empty:
        lines.append("No successful evaluations were produced.")
    else:
        display_cols = [
            "agent_model",
            "judge_model",
            "det_tool_accuracy",
            "judge_pass_rate",
            "agreement_rate",
            "dubious_correct_matches",
            "ambiguous_dubious_rate",
        ]
        formatted = pair_summary_df[display_cols].copy()
        for col in ["det_tool_accuracy", "judge_pass_rate", "agreement_rate", "ambiguous_dubious_rate"]:
            formatted[col] = formatted[col].map(lambda value: "NA" if pd.isna(value) else f"{value:.1%}")
        lines.extend(["```text", formatted.to_string(index=False), "```"])

    lines.extend(["", "## Dubious Correct Match Samples", ""])
    if dubious_df.empty:
        lines.append("No `deterministic FAIL / judge PASS` cases were found.")
    else:
        top_rows = dubious_df.head(12)
        for _, row in top_rows.iterrows():
            lines.append(f"- Query: {row['query']}")
            lines.append(f"  Agent/Judge: {row['agent_model']} / {row['judge_model']}")
            lines.append(f"  Expected vs actual: {row['expected_tool']} -> {row['actual_tool']}")
            lines.append(f"  Reason: {row['judge_reason']}")
            lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def format_pct(value: Any, digits: int = 1) -> str:
    if pd.isna(value):
        return "NA"
    return f"{float(value):.{digits}%}"


def escape_md_cell(value: Any) -> str:
    text = str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def truncate_text(value: str, limit: int = 72) -> str:
    if len(value) <= limit:
        return value
    return value[: limit - 3].rstrip() + "..."


def display_tool_name(value: Any) -> str:
    if pd.isna(value) or value in [None, "None", "nan"]:
        return "NO TOOL"
    return str(value)


def build_markdown_table(
    frame: pd.DataFrame,
    columns: list[tuple[str, str]],
) -> list[str]:
    header = "| " + " | ".join(title for _, title in columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    rows = [header, divider]
    for _, row in frame.iterrows():
        rows.append(
            "| "
            + " | ".join(escape_md_cell(row[column]) for column, _ in columns)
            + " |"
        )
    return rows


def write_korean_notebook_guide(
    pair_summary_df: pd.DataFrame,
    agent_summary_df: pd.DataFrame,
    case_summary_df: pd.DataFrame,
    dubious_df: pd.DataFrame,
    output_path: Path,
) -> None:
    total_pairs = len(pair_summary_df)
    total_agents = int(agent_summary_df["agent_model"].nunique()) if not agent_summary_df.empty else 0
    total_judges = int(pair_summary_df["judge_model"].nunique()) if not pair_summary_df.empty else 0
    total_cases = len(case_summary_df)
    total_dubious = len(dubious_df)
    ambiguous_dubious = int(dubious_df["ambiguity_focus"].fillna(False).sum()) if "ambiguity_focus" in dubious_df.columns else 0

    top_pair = pair_summary_df.iloc[0] if not pair_summary_df.empty else None
    top_agent = agent_summary_df.iloc[0] if not agent_summary_df.empty else None
    top_cases = (
        case_summary_df[case_summary_df["dubious_correct_matches"] > 0]
        .head(5)
        .copy()
    )
    top_cases["query"] = top_cases["query"].map(lambda value: truncate_text(value, limit=68))
    top_cases["expected_tool"] = top_cases["expected_tool"].map(display_tool_name)
    top_cases["judge_pass_rate"] = top_cases["judge_pass_rate"].map(format_pct)

    top_pairs = pair_summary_df.head(4).copy()
    if not top_pairs.empty:
        for column in ["det_tool_accuracy", "judge_pass_rate", "agreement_rate", "ambiguous_dubious_rate"]:
            top_pairs[column] = top_pairs[column].map(format_pct)

    representative = dubious_df.copy()
    if not representative.empty:
        representative = representative.sort_values(
            by=["ambiguity_focus", "difficulty", "agent_model", "judge_model"],
            ascending=[False, True, True, True],
        )
        representative = representative.drop_duplicates(subset=["query"]).head(5)

    lines = [
        "# Tool Calling Evaluation 결과 해설 (07)",
        "",
        "이 문서는 `07_tool_calling_eval_analysis.py` 실행 결과를 바탕으로,",
        "`Deterministic` 평가와 `DiscreteMetric` 기반 judge 평가가 왜 다르게 나왔는지 설명하기 위한 notebook용 해설입니다.",
        "",
        "다음 결과 파일과 그림을 함께 보면 해석이 가장 쉽습니다.",
        "",
        "- [모델 조합 요약](../analysis_outputs/07_tool_calling_eval_pair_summary.csv)",
        "- [에이전트 요약](../analysis_outputs/07_tool_calling_eval_agent_summary.csv)",
        "- [질의별 요약](../analysis_outputs/07_tool_calling_eval_case_summary.csv)",
        "- [Dubious correct match 상세](../analysis_outputs/07_tool_calling_eval_dubious_matches.csv)",
        "- [Ambiguous heatmap](../analysis_outputs/07_tool_calling_eval_dubious_heatmap.png)",
        "- [Outcome mix chart](../analysis_outputs/07_tool_calling_eval_ambiguous_outcome_mix.png)",
        "- [Case frequency chart](../analysis_outputs/07_tool_calling_eval_case_frequency.png)",
        "",
        "---",
        "",
        "## 1. 한눈에 보는 결론",
        "",
        f"이번 실험은 **{total_cases}개 질의**, **{total_agents}개 agent 모델**, **{total_judges}개 judge 모델**, 총 **{total_pairs}개 모델 조합**으로 수행했습니다.",
        f"그 결과 `Deterministic FAIL / Judge PASS`에 해당하는 **dubious correct match가 {total_dubious}건** 발견되었고, 이 중 **{ambiguous_dubious}건**은 의도적으로 넣은 모호한 질문에서 나왔습니다.",
        "",
        "즉, 단일 정답 라벨만 두고 `==` 비교하는 방식은 에이전트의 합리적인 유연성을 자주 놓쳤고,",
        "judge 평가는 다음과 같은 경우를 '충분히 맞는 전략'으로 인정했습니다.",
        "",
        "- 사용자의 질문이 **복수 의도**를 포함해, 정답 도구 외의 다른 도구도 주요 의도를 해결하는 경우",
        "- 질문이 **넓고 대화형(advisory)** 이라서 `NO TOOL` 응답이 오히려 자연스러운 경우",
        "- 필요한 슬롯 정보가 부족해서, 바로 도구를 호출하기보다 일반 답변이나 후속 질문이 더 타당한 경우",
        "",
    ]

    if top_pair is not None and top_agent is not None:
        lines.extend(
            [
                f"가장 disagreement를 잘 드러낸 모델 조합은 **`{top_pair['agent_model']}` / `{top_pair['judge_model']}`** 이었고,",
                f"ambiguous-case dubious rate는 **{format_pct(top_pair['ambiguous_dubious_rate'])}** 였습니다.",
                f"agent 전체 기준으로는 **`{top_agent['agent_model']}`** 가 ambiguous dubious match를 가장 많이 만들었으며,",
                f"judge 두 종류를 합쳐 **{int(top_agent['ambiguous_dubious_matches'])}건**을 기록했습니다.",
                "",
            ]
        )

    lines.extend(
        [
            "## 2. 모델 조합별 해석",
            "",
            "상위 모델 조합을 보면, `Deterministic` 정확도가 높다고 해서 judge와의 disagreement가 적은 것은 아니었습니다.",
            "오히려 작은 모델이나 미니 계열 모델은 canonical label과는 다른 첫 선택을 자주 했고,",
            "judge는 그 선택이 질문의 한 축을 충분히 해결하면 `correct`로 받아들였습니다.",
            "",
        ]
    )

    if not top_pairs.empty:
        lines.extend(
            build_markdown_table(
                top_pairs[
                    [
                        "agent_model",
                        "judge_model",
                        "det_tool_accuracy",
                        "judge_pass_rate",
                        "dubious_correct_matches",
                        "ambiguous_dubious_rate",
                    ]
                ],
                [
                    ("agent_model", "Agent"),
                    ("judge_model", "Judge"),
                    ("det_tool_accuracy", "Det Accuracy"),
                    ("judge_pass_rate", "Judge Pass"),
                    ("dubious_correct_matches", "Dubious Matches"),
                    ("ambiguous_dubious_rate", "Ambiguous Dubious Rate"),
                ],
            )
        )
        lines.extend(
            [
                "",
                "![Ambiguous-case heatmap](../analysis_outputs/07_tool_calling_eval_dubious_heatmap.png)",
                "",
                "위 heatmap은 어떤 agent/judge 조합에서 모호한 질문의 '합리적 불일치'가 많이 나왔는지 보여줍니다.",
                "",
            ]
        )

    lines.extend(
        [
            "## 3. 어떤 질문이 disagreement를 만들었는가?",
            "",
            "가장 중요한 포인트는 **질문 설계 자체**였습니다. 아래 케이스들은 single-label 정답만으로는 포착하기 어려운 의미적 유연성을 반복적으로 드러냈습니다.",
            "",
        ]
    )

    if not top_cases.empty:
        lines.extend(
            build_markdown_table(
                top_cases[
                    [
                        "query",
                        "expected_tool",
                        "dubious_correct_matches",
                        "judge_pass_rate",
                    ]
                ],
                [
                    ("query", "Query"),
                    ("expected_tool", "Canonical Tool"),
                    ("dubious_correct_matches", "Dubious Matches"),
                    ("judge_pass_rate", "Judge Pass"),
                ],
            )
        )
        lines.extend(
            [
                "",
                "특히 아래 패턴에서 disagreement가 자주 발생했습니다.",
                "",
                "1. **Multi-intent 질의**",
                "예: `positions and their pay in Amsterdam`, `which city pays better and has more openings`",
                "canonical label은 하나만 고르지만, 실제 agent는 `search_jobs -> compare_salaries` 같은 순서를 택했고 judge는 이를 합리적이라고 봤습니다.",
                "",
                "2. **Broad / advisory 질의**",
                "예: `What should I expect out there?`, `Tell me everything about data engineering in Munich`",
                "benchmark는 `search_jobs`를 기대했지만, judge는 이 질의들을 일반적인 시장 조언 질문으로 해석해 `NO TOOL`도 정답으로 인정했습니다.",
                "",
                "3. **정보가 덜 주어진 질의**",
                "예: `Fix my CV`, `What salary should I expect?`",
                "resume 전문이나 지역 정보가 충분하지 않으면, 바로 도구를 쓰기보다 일반 조언을 주거나 clarification을 유도하는 것이 더 자연스러울 수 있습니다.",
                "",
                "![Case frequency chart](../analysis_outputs/07_tool_calling_eval_case_frequency.png)",
                "",
            ]
        )

    lines.extend(
        [
            "## 4. 대표 disagreement 사례",
            "",
            "아래 예시들은 notebook 본문에서 그대로 설명하기 좋은 케이스입니다.",
            "",
        ]
    )

    if representative.empty:
        lines.append("- 이번 실행에서는 representative disagreement 사례가 없었습니다.")
    else:
        for _, row in representative.iterrows():
            lines.extend(
                [
                    f"### `{truncate_text(row['query'], 90)}`",
                    f"- Canonical tool: `{display_tool_name(row['expected_tool'])}`",
                    f"- Agent choice: `{display_tool_name(row['actual_tool'])}`",
                    f"- Judge interpretation: {row['judge_reason']}",
                    "",
                ]
            )

    lines.extend(
        [
            "## 5. Notebook에서 강조하면 좋은 메시지",
            "",
            "이 실험이 주는 가장 실무적인 교훈은 다음과 같습니다.",
            "",
            "- `Deterministic` 평가는 회귀 테스트와 baseline 측정에 매우 유용하지만, **single-label benchmark**로 쓰면 모호한 질문에서 agent를 과하게 벌점 줄 수 있습니다.",
            "- `DiscreteMetric` 같은 judge 평가는 '완전히 같은 정답'이 아니라 '**질문의 주요 의도를 얼마나 합리적으로 다뤘는가**'를 볼 수 있게 해줍니다.",
            "- 따라서 tool-calling benchmark를 설계할 때는 `expected_tool` 하나만 두기보다, **허용 가능한 대안 도구**, **NO TOOL 허용 조건**, **multi-tool sequence 허용 여부**를 함께 정의하는 것이 더 타당합니다.",
            "",
            "![Outcome mix chart](../analysis_outputs/07_tool_calling_eval_ambiguous_outcome_mix.png)",
            "",
            "> 한 줄 요약: **도구 선택 평가에서 strict exact match만 보면 agent의 유연한 문제 해결 능력을 놓치고, judge 기반 평가는 그 '애매하지만 합리적인 선택'을 드러내 준다.**",
            "",
        ]
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def format_rate_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    formatted = df.copy()
    for column in columns:
        if column in formatted.columns:
            formatted[column] = formatted[column].map(lambda value: "NA" if pd.isna(value) else f"{value:.0%}")
    return formatted


def main() -> None:
    args = parse_args()
    configure_runtime()
    api_key = load_api_key()
    client = OpenAI(api_key=api_key)

    cases = build_test_cases()
    if args.max_cases is not None:
        cases = cases[: args.max_cases]

    metric = build_metric()

    agent_rows: list[dict[str, Any]] = []
    agent_failures: list[dict[str, str]] = []
    for agent_model in args.agent_models:
        print(f"\n[agent] Running cases with {agent_model}")
        try:
            agent = build_agent(agent_model)
        except Exception as exc:  # noqa: BLE001
            agent_failures.append({"agent_model": agent_model, "error": str(exc)})
            print(f"  Skipped agent model due to setup error: {exc}")
            continue

        for index, case in enumerate(cases, start=1):
            print(f"  - [{index:02d}/{len(cases)}] {case.case_id}")
            try:
                row = evaluate_agent_case(agent=agent, case=case, max_retries=args.max_retries)
            except Exception as exc:  # noqa: BLE001
                row = {
                    "case_id": case.case_id,
                    "query": case.query,
                    "difficulty": case.difficulty,
                    "ambiguity_focus": case.ambiguity_focus,
                    "case_note": case.note,
                    "expected_tool": case.expected_tool,
                    "expected_params_json": json.dumps(case.expected_params, ensure_ascii=False) if case.expected_params is not None else None,
                    "actual_tool": None,
                    "actual_tool_sequence": json.dumps([], ensure_ascii=False),
                    "actual_tool_count": 0,
                    "actual_params_json": None,
                    "tool_correct": pd.NA,
                    "params_correct": pd.NA,
                    "expected_tool_in_sequence": pd.NA,
                    "response_text": "",
                    "agent_error": str(exc),
                }
                print(f"    agent error: {exc}")

            row["agent_model"] = agent_model
            agent_rows.append(row)

    agent_df = pd.DataFrame(agent_rows)
    if agent_df.empty:
        raise RuntimeError("No agent results were collected.")

    judge_rows: list[dict[str, Any]] = []
    judge_failures: list[dict[str, str]] = []
    for judge_model in args.judge_models:
        print(f"\n[judge] Scoring with {judge_model}")
        try:
            judge_llm = llm_factory(judge_model, client=client)
        except Exception as exc:  # noqa: BLE001
            judge_failures.append({"judge_model": judge_model, "error": str(exc)})
            print(f"  Skipped judge model due to setup error: {exc}")
            continue

        judge_model_failed = False
        for _, agent_row in agent_df.iterrows():
            row = agent_row.to_dict()
            row["judge_model"] = judge_model
            row["judge_value"] = None
            row["judge_reason"] = None
            row["judge_error"] = None

            if pd.isna(row["agent_error"]):
                try:
                    value, reason = score_case_with_judge(
                        metric=metric,
                        judge_llm=judge_llm,
                        row=row,
                        max_retries=args.max_retries,
                    )
                    row["judge_value"] = value
                    row["judge_reason"] = reason
                except Exception as exc:  # noqa: BLE001
                    row["judge_error"] = str(exc)
                    if "model" in str(exc).lower() and "not" in str(exc).lower():
                        judge_model_failed = True
                        judge_failures.append({"judge_model": judge_model, "error": str(exc)})
                        print(f"  Judge model unavailable, skipping remaining cases: {exc}")
                        break

            row["outcome_bucket"] = classify_outcome(
                tool_correct=row["tool_correct"],
                judge_value=row["judge_value"],
                agent_error=row["agent_error"],
                judge_error=row["judge_error"],
            )
            judge_rows.append(row)

        if judge_model_failed:
            judge_rows = [row for row in judge_rows if row["judge_model"] != judge_model]

    results_df = pd.DataFrame(judge_rows)
    if results_df.empty:
        raise RuntimeError("No judge-scored results were collected.")

    pair_summary_df = summarize_pair_results(results_df)
    agent_summary_df = summarize_agent_results(results_df)
    case_summary_df = build_case_summary(results_df)
    dubious_df = results_df[results_df["outcome_bucket"] == "dubious_correct_match"].copy()
    dubious_df = dubious_df.sort_values(
        by=["ambiguity_focus", "difficulty", "agent_model", "judge_model"],
        ascending=[False, True, True, True],
    ).reset_index(drop=True)

    details_csv = OUTPUT_DIR / "07_tool_calling_eval_details.csv"
    pair_summary_csv = OUTPUT_DIR / "07_tool_calling_eval_pair_summary.csv"
    agent_summary_csv = OUTPUT_DIR / "07_tool_calling_eval_agent_summary.csv"
    case_summary_csv = OUTPUT_DIR / "07_tool_calling_eval_case_summary.csv"
    dubious_csv = OUTPUT_DIR / "07_tool_calling_eval_dubious_matches.csv"
    heatmap_png = OUTPUT_DIR / "07_tool_calling_eval_dubious_heatmap.png"
    outcome_png = OUTPUT_DIR / "07_tool_calling_eval_ambiguous_outcome_mix.png"
    case_png = OUTPUT_DIR / "07_tool_calling_eval_case_frequency.png"
    report_md = OUTPUT_DIR / "07_tool_calling_eval_report.md"

    results_df.to_csv(details_csv, index=False)
    pair_summary_df.to_csv(pair_summary_csv, index=False)
    agent_summary_df.to_csv(agent_summary_csv, index=False)
    case_summary_df.to_csv(case_summary_csv, index=False)
    dubious_df.to_csv(dubious_csv, index=False)

    make_heatmap(pair_summary_df, heatmap_png)
    make_agent_outcome_plot(results_df, outcome_png)
    make_case_barh(case_summary_df, case_png)
    write_markdown_report(pair_summary_df, dubious_df, report_md)
    write_korean_notebook_guide(
        pair_summary_df=pair_summary_df,
        agent_summary_df=agent_summary_df,
        case_summary_df=case_summary_df,
        dubious_df=dubious_df,
        output_path=NOTEBOOK_GUIDE_PATH,
    )

    if agent_failures:
        print("\nAgent model setup failures")
        for failure in agent_failures:
            print(f"  - {failure['agent_model']}: {failure['error']}")

    if judge_failures:
        print("\nJudge model setup failures")
        for failure in judge_failures:
            print(f"  - {failure['judge_model']}: {failure['error']}")

    print("\nModel pair summary")
    if pair_summary_df.empty:
        print("  No successful model pairs.")
    else:
        display_cols = [
            "agent_model",
            "judge_model",
            "det_tool_accuracy",
            "judge_pass_rate",
            "agreement_rate",
            "dubious_correct_matches",
            "ambiguous_dubious_rate",
        ]
        print(
            format_rate_columns(
                pair_summary_df[display_cols],
                ["det_tool_accuracy", "judge_pass_rate", "agreement_rate", "ambiguous_dubious_rate"],
            ).to_string(index=False)
        )

    print("\nTop dubious correct matches")
    if dubious_df.empty:
        print("  No deterministic FAIL / judge PASS cases found.")
    else:
        top_rows = dubious_df[
            ["query", "agent_model", "judge_model", "expected_tool", "actual_tool", "judge_reason"]
        ].head(10)
        for idx, row in enumerate(top_rows.itertuples(index=False), start=1):
            preview = row.query[:90] + ("..." if len(row.query) > 90 else "")
            print(f"  {idx}. {preview}")
            print(f"     {row.agent_model} / {row.judge_model} | {row.expected_tool} -> {row.actual_tool}")
            print(f"     {row.judge_reason}")

    print("\nSaved outputs")
    print(f"  details: {details_csv}")
    print(f"  pair summary: {pair_summary_csv}")
    print(f"  agent summary: {agent_summary_csv}")
    print(f"  case summary: {case_summary_csv}")
    print(f"  dubious matches: {dubious_csv}")
    print(f"  heatmap: {heatmap_png}")
    print(f"  outcome mix: {outcome_png}")
    print(f"  case frequency: {case_png}")
    print(f"  report: {report_md}")
    print(f"  notebook guide: {NOTEBOOK_GUIDE_PATH}")


if __name__ == "__main__":
    main()
