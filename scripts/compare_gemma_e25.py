#!/usr/bin/env python3
"""Create an aligned source/reference/Gemma/Marian comparison artifact."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GEMMA = ROOT / "reports/gemma-3-270m-uk-verbalizer-6771e190-full-pipeline.json"
DEFAULT_E25 = ROOT / "reports/hf-uktn-numeric-release-3597-e25-wsl.json"
DEFAULT_OUTPUT = ROOT / "reports/evaluations/gemma-vs-e25-numeric-3597.csv"
DEFAULT_SUMMARY = ROOT / "reports/evaluations/gemma-vs-e25-numeric-3597-summary.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def split_and_index(identifier: str) -> tuple[str, int]:
    split, raw_index = identifier.rsplit("-", 1)
    return split, int(raw_index)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gemma-report", type=Path, default=DEFAULT_GEMMA)
    parser.add_argument("--e25-report", type=Path, default=DEFAULT_E25)
    parser.add_argument("--model-label", default="e25")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args()
    model_label = args.model_label.strip().casefold()
    if not model_label or not model_label.replace("_", "").isalnum():
        raise ValueError("model label must contain only letters, digits, and underscores")

    gemma = load_json(args.gemma_report)
    e25 = load_json(args.e25_report)
    gemma_rows = gemma.get("rows")
    e25_rows = e25.get("examples")
    if not isinstance(gemma_rows, list) or not isinstance(e25_rows, list):
        raise ValueError("both reports must contain row-level results")
    if len(gemma_rows) != len(e25_rows):
        raise ValueError(f"row count differs: Gemma={len(gemma_rows)}, E25={len(e25_rows)}")

    rows: list[dict[str, Any]] = []
    for position, (gemma_row, e25_row) in enumerate(zip(gemma_rows, e25_rows, strict=True)):
        source = str(e25_row["source"])
        if source != str(gemma_row["source"]):
            raise ValueError(f"source mismatch at position {position}: {source!r}")
        reference = str(e25_row["reference"])
        gemma_reference = str(gemma_row["primary_reference"])
        if reference != gemma_reference:
            raise ValueError(
                "reference mismatch at position "
                f"{position} for source "
                f"{source!r}: E25={reference!r}, Gemma={gemma_reference!r}"
            )
        split, index = split_and_index(str(e25_row["id"]))
        if split != str(gemma_row["split"]):
            raise ValueError(f"split mismatch at position {position} for source {source!r}")
        rows.append(
            {
                "id": e25_row["id"],
                "split": split,
                "index": index,
                "category": " | ".join(str(x) for x in e25_row.get("categories", [])[1:]),
                "source": source,
                "reference": reference,
                "gemma": str(gemma_row["pipeline_output"]),
                model_label: str(e25_row["prediction"]),
                "gemma_exact": str(gemma_row["pipeline_output"]) == reference,
                f"{model_label}_exact": str(e25_row["prediction"]) == reference,
                "gemma_pipeline_accepted": bool(gemma_row.get("pipeline_accepted")),
                f"{model_label}_numeric_model_rejected": bool(
                    e25_row.get("numeric_model_rejected")
                ),
                f"{model_label}_protected_model_rejected": bool(
                    e25_row.get("protected_model_rejected")
                ),
            }
        )
    fields = list(rows[0]) if rows else []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "rows": len(rows),
        "gemma_report": str(args.gemma_report),
        f"{model_label}_report": str(args.e25_report),
        "output": str(args.output),
        "gemma_exact": sum(row["gemma_exact"] for row in rows),
        f"{model_label}_exact": sum(row[f"{model_label}_exact"] for row in rows),
        "both_exact": sum(row["gemma_exact"] and row[f"{model_label}_exact"] for row in rows),
        "gemma_only_exact": sum(
            row["gemma_exact"] and not row[f"{model_label}_exact"] for row in rows
        ),
        f"{model_label}_only_exact": sum(
            not row["gemma_exact"] and row[f"{model_label}_exact"] for row in rows
        ),
        "neither_exact": sum(
            not row["gemma_exact"] and not row[f"{model_label}_exact"] for row in rows
        ),
        f"{model_label}_numeric_model_rejected": sum(
            row[f"{model_label}_numeric_model_rejected"] for row in rows
        ),
        f"{model_label}_rejections_by_split": dict(
            sorted(
                Counter(
                    row["split"]
                    for row in rows
                    if row[f"{model_label}_numeric_model_rejected"]
                ).items()
            )
        ),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
