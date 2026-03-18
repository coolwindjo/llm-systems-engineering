from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import run_config_copilot_app  # noqa: E402
from components.app_runtime import run_config_copilot_app as runtime_entry  # noqa: E402
from services.config import DEFAULT_MODEL_NAME, get_openai_api_key, load_local_env  # noqa: E402
from services.copilot import ConfigurationCopilot, create_copilot  # noqa: E402
from services.knowledge_base import load_standard_documents  # noqa: E402


REPORT_JSON_PATH = PROJECT_ROOT / "docs" / "smoke_test_report.json"
REPORT_MD_PATH = PROJECT_ROOT / "docs" / "smoke_test_report.md"
SAMPLE_REQUIREMENT = (
    "Add a cyclic WheelSpeedBroadcast service with id 0x120 running every 10 ms "
    "for wheel speed signals and diagnostics publication."
)


@dataclass
class SmokeCheck:
    name: str
    passed: bool
    detail: str


def _record(name: str, passed: bool, detail: str) -> SmokeCheck:
    return SmokeCheck(name=name, passed=passed, detail=detail)


def _write_report(report: dict) -> None:
    REPORT_JSON_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Smoke Test Report",
        "",
        f"- Timestamp: `{report['timestamp_utc']}`",
        f"- Model: `{report['model_name']}`",
        f"- Overall status: `{report['overall_status']}`",
        "",
        "## Checks",
    ]
    for check in report["checks"]:
        status = "PASS" if check["passed"] else "FAIL"
        lines.append(f"- `{status}` {check['name']}: {check['detail']}")

    lines.extend(
        [
            "",
            "## Runtime Summary",
            f"- Sample requirement: `{report['sample_requirement']}`",
            f"- Corpus sources: `{', '.join(report['corpus_sources'])}`",
            f"- Translated query count: `{report['translated_query_count']}`",
            f"- Retrieved chunk count: `{report['retrieved_chunk_count']}`",
            f"- Validation status: `{report['validation_status']}`",
            "",
            "## Extraction Preview",
            "```json",
            json.dumps(report["extraction_preview"], ensure_ascii=False, indent=2),
            "```",
        ]
    )
    REPORT_MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    load_local_env()
    checks: list[SmokeCheck] = []
    model_name = DEFAULT_MODEL_NAME
    extraction_preview: dict = {}
    translated_query_count = 0
    retrieved_chunk_count = 0
    validation_status = "not_run"
    corpus_sources = [doc.metadata["source"] for doc in load_standard_documents()]

    try:
        dependency_ok = run_config_copilot_app is runtime_entry
        checks.append(
            _record(
                "app runtime dependency path",
                dependency_ok,
                "app.py resolves to components.app_runtime.run_config_copilot_app",
            )
        )

        api_key = get_openai_api_key()
        copilot = create_copilot(model_name=model_name, api_key=api_key)
        checks.append(
            _record(
                "copilot creation",
                isinstance(copilot, ConfigurationCopilot),
                f"Created {type(copilot).__name__} with model {model_name}",
            )
        )

        result = copilot.run(SAMPLE_REQUIREMENT)
        if result.get("error"):
            checks.append(
                _record(
                    "sample requirement input",
                    False,
                    f"Copilot returned error at stage {result['error']['stage']}: {result['error']['message']}",
                )
            )
        else:
            checks.append(
                _record(
                    "sample requirement input",
                    True,
                    "Sample requirement produced a structured runtime result.",
                )
            )

            translated_query_count = len(result["translated_queries"])
            retrieved_chunk_count = len(result["retrieved_chunks"])
            extraction_preview = result["extraction"]
            validation_status = result["validation"]["status"]

            checks.append(
                _record(
                    "retrieval",
                    retrieved_chunk_count > 0 and translated_query_count > 0,
                    f"{translated_query_count} translated queries, {retrieved_chunk_count} retrieved chunks",
                )
            )

            extraction_ok = all(
                extraction_preview.get(field) not in [None, "", "Unknown"]
                for field in ["ServiceName", "ID", "Class", "PlayType"]
            )
            checks.append(
                _record(
                    "extraction",
                    extraction_ok,
                    json.dumps(extraction_preview, ensure_ascii=False),
                )
            )

            validation = result["validation"]
            validation_ok = validation.get("status") in {"ready", "needs_review", "incomplete"} and isinstance(
                validation.get("schema_valid"), bool
            )
            checks.append(
                _record(
                    "validation response",
                    validation_ok,
                    json.dumps(
                        {
                            "status": validation.get("status"),
                            "schema_valid": validation.get("schema_valid"),
                            "missing_required_fields": validation.get("missing_required_fields"),
                        },
                        ensure_ascii=False,
                    ),
                )
            )
    except Exception as exc:  # noqa: BLE001
        checks.append(_record("smoke test runtime", False, str(exc)))

    overall_status = "passed" if all(check.passed for check in checks) else "failed"
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": model_name,
        "overall_status": overall_status,
        "sample_requirement": SAMPLE_REQUIREMENT,
        "corpus_sources": corpus_sources,
        "translated_query_count": translated_query_count,
        "retrieved_chunk_count": retrieved_chunk_count,
        "validation_status": validation_status,
        "extraction_preview": extraction_preview,
        "checks": [asdict(check) for check in checks],
    }
    _write_report(report)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if overall_status == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
