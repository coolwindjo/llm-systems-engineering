from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.profile_health import audit_interview_profiles


def _print_summary(report):
    total = len(report)
    healthy = sum(1 for item in report if item["status"] == "ok")
    needs_attention = sum(1 for item in report if item["status"] in {"invalid_file", "missing_required"})
    print(f"Scanned profiles: {total}")
    print(f"Healthy: {healthy}")
    print(f"Needs attention: {needs_attention}")
    print("")

    for item in report:
        icon = "OK" if item["status"] == "ok" else "WARN"
        line = f"{icon} {item['file']} ({item['status']})"
        missing_required = ", ".join(item.get("missing_required", []))
        recommendations = item.get("recommendations", [])
        recommended_names = ", ".join([rec.get("name", "") for rec in recommendations if rec.get("recommended")])
        top_scores = ", ".join(
            [f"{rec.get('name', '')} {rec.get('score', 0.0):.1f}%" for rec in recommendations[:3]]
        )
        if missing_required:
            line += f" | missing_required: {missing_required}"
        if recommended_names:
            line += f" | recommended: {recommended_names}"
        if top_scores:
            line += f" | top_scores: {top_scores}"
        print(line)


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit JD profile session_interviewers completeness.")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON report.",
    )
    args = parser.parse_args()

    report = audit_interview_profiles(PROJECT_ROOT / "data")

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        _print_summary(report)

    return 0 if all(item["status"] != "missing_required" and item["status"] != "invalid_file" for item in report) else 1


if __name__ == "__main__":
    raise SystemExit(main())
