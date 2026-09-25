"""Apply the sibling news project's semantic validators to a Marian report."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SIBLING = ROOT.parent / "hybrid-ukrainian-tts-verbalizer-openspec"


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    sibling = args.sibling.resolve()
    sys.path.insert(0, str(sibling))
    from verbolisation.scripts.validate import (  # noqa: PLC0415
        NUM_SCORE_MIN,
        WORD_SCORE_MIN,
        has_bad_day_grammar,
        has_bad_decimal_grammar,
        has_bad_year_grammar,
        num_score,
        word_score,
    )

    report = json.loads(args.report.read_text(encoding="utf-8"))
    examples = report.get("examples")
    if not isinstance(examples, list) or not examples:
        raise ValueError(f"report has no examples: {args.report}")

    audited: list[dict[str, Any]] = []
    for example in examples:
        source = str(example["source"])
        prediction = str(example["prediction"])
        prediction_low = prediction.lower()
        numeric = num_score(source, prediction_low)
        content = word_score(source, prediction_low)
        checks = {
            "numeric_fidelity": numeric >= NUM_SCORE_MIN,
            "content_fidelity": content >= WORD_SCORE_MIN,
            "no_digit_leak": not bool(re.search(r"\d", prediction)),
            "year_grammar": not has_bad_year_grammar(source, prediction_low),
            "day_grammar": not has_bad_day_grammar(source, prediction_low),
            "decimal_grammar": not has_bad_decimal_grammar(source, prediction_low),
        }
        audited.append(
            {
                "id": str(example["id"]),
                "source": source,
                "prediction": prediction,
                "numeric_score": numeric,
                "content_score": content,
                "sibling_semantic_ok": checks["numeric_fidelity"]
                and checks["content_fidelity"],
                "strict_semantic_ok": all(checks.values()),
                "failed_checks": sorted(key for key, value in checks.items() if not value),
            }
        )

    rows = len(audited)
    failure_counts = {
        name: sum(name in row["failed_checks"] for row in audited)
        for name in (
            "numeric_fidelity",
            "content_fidelity",
            "no_digit_leak",
            "year_grammar",
            "day_grammar",
            "decimal_grammar",
        )
    }
    result = {
        "report": str(args.report.resolve()),
        "checkpoint": report.get("checkpoint"),
        "rows": rows,
        "policy": {
            "validator_source": str(
                sibling / "verbolisation/scripts/validate.py"
            ),
            "numeric_score_min": NUM_SCORE_MIN,
            "content_score_min": WORD_SCORE_MIN,
            "sibling_semantic_ok": "numeric_fidelity and content_fidelity",
            "strict_semantic_ok": "all six listed checks",
        },
        "sibling_semantic_accuracy": sum(
            row["sibling_semantic_ok"] for row in audited
        )
        / rows,
        "strict_semantic_accuracy": sum(row["strict_semantic_ok"] for row in audited)
        / rows,
        "mean_numeric_score": sum(row["numeric_score"] for row in audited) / rows,
        "mean_content_score": sum(row["content_score"] for row in audited) / rows,
        "failure_counts": failure_counts,
        "failed_rows": sum(not row["strict_semantic_ok"] for row in audited),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.errors.parent.mkdir(parents=True, exist_ok=True)
    with args.errors.open("w", encoding="utf-8") as handle:
        for row in audited:
            if not row["strict_semantic_ok"]:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--errors", type=Path, required=True)
    parser.add_argument("--sibling", type=Path, default=DEFAULT_SIBLING)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(evaluate(parse_args()), ensure_ascii=False, indent=2, sort_keys=True))
