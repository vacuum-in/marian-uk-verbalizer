#!/usr/bin/env python3
"""Refresh numeric gates/artifacts from stored evaluator examples without inference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.verbalizer.evaluation.metrics import evaluate_examples  # noqa: E402
from src.verbalizer.numeric import (  # noqa: E402
    enforce_numeric_fidelity,
    numeric_fidelity_failures,
    span_as_dict,
)
from src.verbalizer.training.manifest import write_json  # noqa: E402


def refresh(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    examples = list(report["examples"])
    for example in examples:
        stored_failures = list(example.get("numeric_failures") or [])
        stored_offsets = {(item["start"], item["end"]) for item in stored_failures}
        for span in numeric_fidelity_failures(
            str(example["source"]), str(example["prediction"])
        ):
            if (span.start, span.end) not in stored_offsets:
                stored_failures.append(span_as_dict(span))
        example["numeric_failures"] = stored_failures
        numeric_rejected = bool(stored_failures)
        # Re-apply the current deterministic fallback to stored predictions.
        # The original failure metadata remains the model-rejection evidence.
        example["prediction"] = enforce_numeric_fidelity(
            str(example["source"]), str(example["prediction"])
        ).text
        example["numeric_model_rejected"] = numeric_rejected
        example["protected_model_rejected"] = bool(example.get("numeric_warnings"))
        example["numeric_warnings"] = ["NUMERIC_MODEL_REJECTED"] if numeric_rejected else (
            ["PROTECTED_MODEL_REJECTED"] if example["protected_model_rejected"] else []
        )
    report["metrics"] = evaluate_examples(examples)
    overall = report["metrics"]["overall"]
    checks = {
        "digit_sequence_fidelity": overall["digit_sequence_fidelity"] in {None, 1.0},
        "numeric_value_fidelity": overall["numeric_value_fidelity"] in {None, 1.0},
        "identifier_group_fidelity": overall["identifier_group_fidelity"] in {None, 1.0},
        "contextual_morphology_accuracy": overall["contextual_morphology_accuracy"] is None
        or overall["contextual_morphology_accuracy"] >= 0.99,
        "zero_numeric_model_rejections": overall["numeric_model_rejected"] == 0,
    }
    report["numeric_release_gate"] = {"passed": all(checks.values()), "checks": checks}
    failure_path = path.with_name(path.stem + "-numeric-failures.jsonl")
    failure_path.write_text(
        "".join(
            json.dumps(example, ensure_ascii=False) + "\n"
            for example in examples
            if example["numeric_model_rejected"]
        ),
        encoding="utf-8",
    )
    report["numeric_failure_artifact"] = str(failure_path)
    write_json(path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    report = refresh(args.report.resolve())
    print(
        json.dumps(
            {
                "rows": report["metrics"]["overall"]["rows"],
                "numeric_model_rejected": report["metrics"]["overall"]["numeric_model_rejected"],
                "numeric_release_gate": report["numeric_release_gate"],
                "numeric_failure_artifact": report["numeric_failure_artifact"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
