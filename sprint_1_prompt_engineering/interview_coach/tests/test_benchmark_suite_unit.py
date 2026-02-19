from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from sprint_1_prompt_engineering.interview_coach.scripts import benchmark_suite as bench


def test_coerce_score_bounds_and_coercion() -> None:
    assert bench._coerce_score("7") == 7
    assert bench._coerce_score(11) == 10
    assert bench._coerce_score(-3) == 1
    assert bench._coerce_score("bad") == 1


def test_user_proxy_cycles_three_turns_then_default() -> None:
    proxy = bench.UserProxy(["A", "B", "C"])
    assert proxy.next_response() == "A"
    assert proxy.next_response() == "B"
    assert proxy.next_response() == "C"
    assert proxy.next_response() == "Thank you for the interview."
    assert proxy.next_response() == "Thank you for the interview."


def test_parse_judge_json_recover_embedded_block() -> None:
    raw = 'Leading text {"technical_depth": {"score": 1}, "context_awareness": {}, "professionalism": {}} trailing text'
    parsed = bench._parse_judge_json(raw)
    assert parsed is not None
    assert parsed["technical_depth"]["score"] == 1


def test_parse_judge_json_invalid_payload() -> None:
    assert bench._parse_judge_json("not json") is None


def test_normalize_judge_payload_parse_failed_returns_default_penalty_notes() -> None:
    normalized = bench._normalize_judge_payload(None, parse_failed=True)
    assert normalized["technical_depth_score"] == 1
    assert normalized["context_awareness_score"] == 1
    assert normalized["professionalism_score"] == 1
    assert normalized["aspice_penalty_applied"] is False
    assert "parse" in normalized["overall_reasoning"].lower()


def test_normalize_judge_payload_maps_scores_and_text() -> None:
    normalized = bench._normalize_judge_payload(
        {
            "technical_depth": {"score": 11, "reasoning": "deep", "evidence_check": "checked"},
            "context_awareness": {"score": 0, "reasoning": "continuous"},
            "professionalism": {"score": 7, "reasoning": "balanced"},
            "aspice_evidence_penalty_applied": {"technical_depth": True, "notes": "missing evidence"},
            "overall_reasoning": "good",
        },
        parse_failed=False,
    )

    assert normalized["technical_depth_score"] == 10
    assert normalized["context_awareness_score"] == 1
    assert normalized["professionalism_score"] == 7
    assert normalized["technical_reasoning"] == "deep"
    assert normalized["evidence_check"] == "checked"
    assert normalized["aspice_penalty_applied"] is True
    assert normalized["aspice_penalty_notes"] == "missing evidence"


def test_extract_turn_pairs_from_transcript_keeps_history_for_followup() -> None:
    transcript = [
        {"role": "system", "content": "init"},
        {"role": "assistant", "content": "Q1"},
        {"role": "user", "content": "A1"},
        {"role": "assistant", "content": "Q2"},
        {"role": "user", "content": "A2"},
    ]

    pairs = bench._extract_turn_pairs_from_transcript({"meta": {"transcript": transcript}}, "m", "t", "run-1")
    assert len(pairs) == 2
    assert pairs[0]["turn_index"] == 1
    assert pairs[0]["interviewer_turn"] == "Q1"
    assert pairs[0]["candidate_answer"] == "A1"
    assert pairs[0]["context_turn_history"] == [{"role": "system", "content": "init"}]


def test_load_benchmark_logs_reads_only_turn_pair_transcripts(tmp_path: Path) -> None:
    valid = {
        "meta": {"run_id": "run-1", "model": "gpt-5-mini", "technique": "few_shot"},
        "transcript": [
            {"role": "system", "content": "init"},
            {"role": "assistant", "content": "Q1"},
            {"role": "user", "content": "A1"},
        ],
    }

    (tmp_path / "run.json").write_text(json.dumps(valid), encoding="utf-8")
    (tmp_path / "invalid.json").write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    (tmp_path / "broken.json").write_text("not json", encoding="utf-8")

    pairs = bench.load_benchmark_logs(tmp_path)
    assert len(pairs) == 1
    assert pairs[0]["run_id"] == "run-1"
    assert pairs[0]["model"] == "gpt-5-mini"


def test_evaluate_benchmark_logs_aggregates_scores_and_saves_results(tmp_path: Path, monkeypatch) -> None:
    log_payload = {
        "meta": {"run_id": "run-1", "model": "gpt-5-mini", "technique": "few_shot"},
        "transcript": [
            {"role": "system", "content": "init"},
            {"role": "assistant", "content": "Q1"},
            {"role": "user", "content": "A1"},
            {"role": "assistant", "content": "Q2"},
            {"role": "user", "content": "A2"},
        ],
    }
    (tmp_path / "run.json").write_text(json.dumps(log_payload), encoding="utf-8")

    def fake_call_judge(client, model: str, payload: dict[str, object]) -> dict[str, object]:
        assert payload["candidate_answer"].startswith("A")
        return {
            "technical_depth": {"score": 9, "reasoning": "good technical", "evidence_check": "ok"},
            "context_awareness": {"score": 8, "reasoning": "good context"},
            "professionalism": {"score": 7, "reasoning": "good tone"},
            "aspice_evidence_penalty_applied": {"technical_depth": False, "notes": ""},
            "overall_reasoning": "avg",
        }

    monkeypatch.setattr(bench, "_call_judge", fake_call_judge)

    raw_results, detailed = bench.evaluate_benchmark_logs(
        client=SimpleNamespace(),
        output_dir=tmp_path,
        judge_model="gpt-5-mini",
    )

    assert len(raw_results) == 1
    assert raw_results[0]["model"] == "gpt-5-mini"
    assert raw_results[0]["scores"]["technical_depth"] == 9.0
    assert raw_results[0]["scores"]["overall"] == 8.0
    assert len(detailed) == 2
    assert (tmp_path / bench.RAW_RESULTS_JSON).exists()


def test_analyze_raw_results_returns_summary_and_writes_charts(tmp_path: Path, monkeypatch) -> None:
    if bench.pd is None:
        pytest.skip("pandas is required for benchmark analysis tests")

    raw_results = [
        {
            "model": "gpt-5-mini",
            "technique": "few_shot",
            "scores": {
                "technical_depth": 8.0,
                "context_awareness": 8.5,
                "professionalism": 7.5,
                "overall": 8.0,
            },
            "reasoning": "good",
        },
        {
            "model": "gpt-4o-mini",
            "technique": "knowledge_paucity",
            "scores": {
                "technical_depth": 6.0,
                "context_awareness": 6.5,
                "professionalism": 7.0,
                "overall": 6.5,
            },
            "reasoning": "okay",
        },
    ]

    def fake_radar_chart(data: object, output_path: Path) -> None:
        output_path.write_text("radar", encoding="utf-8")

    def fake_bar_chart(data: object, output_path: Path) -> None:
        output_path.write_text("bar", encoding="utf-8")

    monkeypatch.setattr(bench, "_plot_radar_chart", fake_radar_chart)
    monkeypatch.setattr(bench, "_plot_grouped_overall_bar", fake_bar_chart)

    _, _, _, conclusion, radar_path, bar_path = bench.analyze_raw_results(
        raw_results=raw_results,
        output_dir=tmp_path,
    )

    assert "### Benchmark Conclusion" in conclusion
    assert "Best for senior engineer interview" in conclusion
    assert radar_path.exists()
    assert bar_path.exists()
