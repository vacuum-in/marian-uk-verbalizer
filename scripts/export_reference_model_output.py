#!/usr/bin/env python3
"""Export source/reference/model output rows from a Marian evaluation report."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    label = args.model_label.strip().casefold()
    if not label or not label.replace("_", "").isalnum():
        raise ValueError("model label must contain only letters, digits, and underscores")
    report = json.loads(args.report.read_text(encoding="utf-8"))
    rows = report.get("examples")
    if not isinstance(rows, list):
        raise ValueError("evaluation report has no examples")
    fields = ("id", "category", "source", "reference", label)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "id": row["id"],
                    "category": " | ".join(str(x) for x in row.get("categories", [])[1:]),
                    "source": row["source"],
                    "reference": row["reference"],
                    label: row["prediction"],
                }
            )
    print(json.dumps({"rows": len(rows), "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
