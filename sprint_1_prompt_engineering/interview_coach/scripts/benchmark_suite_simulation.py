"""Simulation phase for benchmarking interviewers."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Dict, List

try:
    from openai import APIError
except (ImportError, ModuleNotFoundError):
    class APIError(Exception):
        """Fallback APIError when openai is unavailable."""

from openai import OpenAI

try:
    from ..services.personas import build_system_prompts
    from .benchmark_suite_config import SCENARIO_TURNS
except ImportError:  # Fallback when running as a top-level module.
    from services.personas import build_system_prompts
    from scripts.benchmark_suite_config import SCENARIO_TURNS


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


def _build_initial_messages(system_prompt: str) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    messages = [{"role": "system", "content": system_prompt}]
    transcript = [{"role": "system", "content": system_prompt}]
    start_prompt = {"role": "user", "content": "Start a mock ADAS interview and ask your first question."}
    messages.append(start_prompt)
    transcript.append(start_prompt)
    return messages, transcript


def _record_turn(
    turn_records: list[TurnRecord],
    turn_index: int,
    role: str,
    model: str,
    technique: str,
    started_at: str,
    ended_at: str,
    elapsed_ms: float,
    tokens_prompt: int | None,
    tokens_completion: int | None,
    tokens_total: int | None,
    content: str,
    error: str,
) -> None:
    turn_records.append(
        TurnRecord(
            turn_index=turn_index,
            role=role,
            started_at=started_at,
            ended_at=ended_at,
            elapsed_ms=elapsed_ms,
            prompt_tokens=tokens_prompt,
            completion_tokens=tokens_completion,
            total_tokens=tokens_total,
            content=content,
            model=model,
            technique=technique,
            error=error,
        )
    )


def _append_assistant_turn(
    client: OpenAI,
    model: str,
    technique: str,
    turn_records: list[TurnRecord],
    messages: list[dict[str, str]],
    turn_idx: int,
) -> tuple[str, bool]:
    started = datetime.utcnow().isoformat()
    try:
        assistant_reply, elapsed_ms, tokens_prompt, tokens_completion, tokens_total = _call_model(
            client=client,
            model=model,
            messages=messages,
        )
        ended = datetime.utcnow().isoformat()
        _record_turn(
            turn_records=turn_records,
            turn_index=turn_idx,
            role="assistant",
            model=model,
            technique=technique,
            started_at=started,
            ended_at=ended,
            elapsed_ms=elapsed_ms,
            tokens_prompt=tokens_prompt,
            tokens_completion=tokens_completion,
            tokens_total=tokens_total,
            content=assistant_reply,
            error="",
        )
        return assistant_reply, True
    except (APIError, TypeError, ValueError, AttributeError, KeyError, IndexError) as exc:
        ended = datetime.utcnow().isoformat()
        _record_turn(
            turn_records=turn_records,
            turn_index=turn_idx,
            role="assistant",
            model=model,
            technique=technique,
            started_at=started,
            ended_at=ended,
            elapsed_ms=0.0,
            tokens_prompt=None,
            tokens_completion=None,
            tokens_total=None,
            content="",
            error=str(exc),
        )
        return "", False


def _append_user_reply(proxy: UserProxy, messages: list[dict[str, str]], transcript: list[dict[str, str]]) -> str:
    candidate_answer = proxy.next_response()
    messages.append({"role": "user", "content": candidate_answer})
    transcript.append({"role": "user", "content": candidate_answer})
    return candidate_answer


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
    run_id = f"{model.replace('.', '_')}__{technique}__{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"

    messages, transcript = _build_initial_messages(system_prompt)
    turn_records: list[TurnRecord] = []
    run_start = datetime.now(UTC).isoformat()

    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0
    assistant_calls = 0
    assistant_elapsed_total = 0.0
    error_message = ""

    for turn_idx in range(len(SCENARIO_TURNS)):
        assistant_reply, ok = _append_assistant_turn(
            client=client,
            model=model,
            technique=technique,
            turn_records=turn_records,
            messages=messages,
            turn_idx=turn_idx,
        )
        if not ok:
            error_message = turn_records[-1].error
            break

        latest = turn_records[-1]
        if latest.prompt_tokens is not None:
            total_prompt_tokens += latest.prompt_tokens
        if latest.completion_tokens is not None:
            total_completion_tokens += latest.completion_tokens
        if latest.total_tokens is not None:
            total_tokens += latest.total_tokens
        assistant_calls += 1
        assistant_elapsed_total += latest.elapsed_ms

        messages.append({"role": "assistant", "content": assistant_reply})
        transcript.append({"role": "assistant", "content": assistant_reply})

        _append_user_reply(user_proxy, messages, transcript)

    if not error_message:
        assistant_reply, ok = _append_assistant_turn(
            client=client,
            model=model,
            technique=technique,
            turn_records=turn_records,
            messages=messages,
            turn_idx=len(SCENARIO_TURNS),
        )
        if ok:
            latest = turn_records[-1]
            if latest.prompt_tokens is not None:
                total_prompt_tokens += latest.prompt_tokens
            if latest.completion_tokens is not None:
                total_completion_tokens += latest.completion_tokens
            if latest.total_tokens is not None:
                total_tokens += latest.total_tokens
            assistant_calls += 1
            assistant_elapsed_total += latest.elapsed_ms

            messages.append({"role": "assistant", "content": assistant_reply})
            transcript.append({"role": "assistant", "content": assistant_reply})
        else:
            error_message = turn_records[-1].error

    run_end = datetime.now(UTC).isoformat()
    avg_latency = round(assistant_elapsed_total / assistant_calls, 2) if assistant_calls else 0.0

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
    log_payload = {"meta": summary, "turn_records": [asdict(item) for item in turn_records]}
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
    total = len(models) * len(techniques)
    results: list[Dict[str, Any]] = []
    for idx, (model, technique) in enumerate(
        ((model, technique) for model in models for technique in techniques),
        start=1,
    ):
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
