#!/usr/bin/env python3
"""Apply Gemma-equivalence, E19 no-regression, and CT2 parity promotion gates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.verbalizer.distillation import normal_form, promotion_gate  # noqa: E402
from src.verbalizer.training.manifest import write_json  # noqa: E402


def report_rows(path: Path) -> list[dict[str, Any]]:
    report = json.loads(path.read_text(encoding="utf-8"))
    rows = report.get("rows", report.get("examples"))
    if not isinstance(rows, list):
        raise ValueError(f"{path} has no rows/examples list")
    return rows


def ct2_parity(fp32_rows: list[dict[str, Any]], ct2_rows: list[dict[str, Any]]) -> dict[str, Any]:
    def keyed(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        return {str(row.get("id", row.get("dataset_index"))): row for row in rows}

    fp32 = keyed(fp32_rows)
    ct2 = keyed(ct2_rows)
    if set(fp32) != set(ct2):
        return {"passed": False, "reason": "row IDs differ", "mismatches": None}
    mismatches = [
        key
        for key in fp32
        if normal_form(str(fp32[key].get("prediction", "")))
        != normal_form(str(ct2[key].get("prediction", "")))
    ]
    return {"passed": not mismatches, "mismatches": len(mismatches), "sample_ids": mismatches[:20]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-report", required=True, type=Path)
    parser.add_argument("--student-report", required=True, type=Path)
    parser.add_argument("--baseline-report", type=Path)
    parser.add_argument("--fp32-report", type=Path)
    parser.add_argument("--ct2-report", type=Path)
    parser.add_argument("--allowed-gap-pp", type=float, default=1.0)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--require-pass", action="store_true")
    args = parser.parse_args()
    if bool(args.fp32_report) != bool(args.ct2_report):
        parser.error("fp32-report and ct2-report must be provided together")
    report = promotion_gate(
        report_rows(args.teacher_report),
        report_rows(args.student_report),
        allowed_gap_pp=args.allowed_gap_pp,
        baseline_rows=report_rows(args.baseline_report) if args.baseline_report else None,
    )
    if args.fp32_report:
        report["ct2_parity"] = ct2_parity(report_rows(args.fp32_report), report_rows(args.ct2_report))
        if not report["ct2_parity"]["passed"]:
            report["status"] = "fail"
    write_json(args.output, report)
    print(json.dumps({"status": report["status"], "checks": report["checks"]}, ensure_ascii=False, indent=2))
    if args.require_pass and report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
