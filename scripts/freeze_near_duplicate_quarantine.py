"""Freeze source-pool IDs implicated by a near-duplicate screening report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = ROOT / "reports/current_dataset_near_duplicates.json"
DEFAULT_OUTPUT = ROOT / "data/manifests/current_near_duplicate_quarantine.json"
SOURCE_POOL_SPLITS = {"train", "validation", "test_id"}


def freeze(report: dict[str, Any], report_path: Path) -> dict[str, Any]:
    ids = {
        pair[f"{side}_id"]
        for pair in report["cross_split_pairs"]
        for side in ("left", "right")
        if pair[f"{side}_split"] in SOURCE_POOL_SPLITS
    }
    return {
        "dataset_version": "current-hybrid-2026-07-27",
        "reason": "cross-split-near-duplicate-screening",
        "source_report": str(report_path),
        "screening_algorithm": report["algorithm"],
        "ids": sorted(ids),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    policy = freeze(report, args.report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(policy, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "quarantined_ids": len(policy["ids"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
