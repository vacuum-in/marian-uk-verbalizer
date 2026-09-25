#!/usr/bin/env python3
"""Join external predictions to the pinned public uk-text-normalization tests.

This is intentionally evaluation-only.  It does not write the downloaded test
rows to a training directory and its output is accepted by
``evaluate_distillation_gate.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.verbalizer.distillation import GEMMA_REVISION, score_evaluation_rows  # noqa: E402
from src.verbalizer.training.manifest import write_json  # noqa: E402


DATASET = "skypro1111/uk-text-normalization"
FILES = {
    "main": "data/test_domains.jsonl",
    "anchor": "data/test_anchor.jsonl",
    "transliteration": "data/eval_translit.jsonl",
    "agreement": "data/eval_agreement.jsonl",
    "misc": "data/eval_misc.jsonl",
}


def load_predictions(path: Path, prediction_key: str) -> dict[str, str]:
    values: dict[str, str] = {}
    if path.suffix == ".json":
        report = json.loads(path.read_text(encoding="utf-8"))
        raw = report.get("rows", report.get("examples", []))
    else:
        raw = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    for row in raw:
        source = str(row["source"])
        value = row.get(prediction_key, row.get("prediction", row.get("teacher_output")))
        if value is None:
            raise ValueError(f"prediction missing for source {source!r}")
        if source in values and values[source] != str(value):
            raise ValueError(f"conflicting predictions for source {source!r}")
        values[source] = str(value)
    return values


def download(split: str, revision: str) -> list[dict[str, Any]]:
    url = f"https://huggingface.co/datasets/{DATASET}/resolve/{revision}/{FILES[split]}"
    request = urllib.request.Request(url, headers={"User-Agent": "marian-distillation-evaluation"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return [json.loads(line) for line in response.read().decode("utf-8").splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--prediction-key", default="teacher_output")
    parser.add_argument("--revision", default="6576c0825199f40617c3c0cfe7f2c886020f1dc9")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    predictions = load_predictions(args.predictions, args.prediction_key)
    rows: list[dict[str, Any]] = []
    index = 0
    for split in FILES:
        for reference in download(split, args.revision):
            source = str(reference["original"])
            if source not in predictions:
                raise ValueError(f"missing prediction for {split} row {source!r}")
            rows.append(
                {
                    "id": f"{split}-{index:05d}",
                    "split": split,
                    "source": source,
                    "primary_reference": str(reference["verbalized"]),
                    "alternative_references": list(reference.get("verbalized_alts") or []),
                    "prediction": predictions[source],
                    "categories": reference.get("combo_names") or [reference.get("category") or "uncategorized"],
                }
            )
            index += 1
    unused = len(set(predictions) - {row["source"] for row in rows})
    report = {
        "dataset": DATASET,
        "revision": args.revision,
        "teacher_revision": GEMMA_REVISION if args.prediction_key == "teacher_output" else None,
        "rows": rows,
        "metrics": score_evaluation_rows(rows),
        "unused_prediction_sources": unused,
    }
    write_json(args.output, report)
    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
