#!/usr/bin/env python3
"""Export quarantined teacher labels to an auditable human-review CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


FIELDS = [
    "source",
    "target",
    "teacher_output",
    "provenance",
    "rejection_flags",
    "review_flags",
    "quarantine_reason",
    "review_decision",
    "reviewer_note",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quarantine", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    rows: list[dict[str, Any]] = [
        json.loads(line) for line in args.quarantine.read_text(encoding="utf-8").splitlines() if line
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "source": row.get("source", ""),
                    "target": row.get("target", ""),
                    "teacher_output": row.get("teacher_output", row.get("target", "")),
                    "provenance": row.get("provenance", "gemma_teacher"),
                    "rejection_flags": ";".join(row.get("rejection_flags") or []),
                    "review_flags": ";".join(row.get("review_flags") or []),
                    "quarantine_reason": row.get("quarantine_reason", ""),
                    "review_decision": "",
                    "reviewer_note": "",
                }
            )
    print(json.dumps({"rows": len(rows), "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
