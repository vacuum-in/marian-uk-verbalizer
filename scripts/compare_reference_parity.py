"""Compare the local Marian model, frozen targets, and public reference API.

The reference capture may be incomplete.  Only captured IDs are evaluated, in
capture order, which makes this useful both for progress checks and the final
15k comparison.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import unicodedata
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.verbalizer.evaluation.metrics import (
    evaluate_examples,
    normalize_evaluation_text,
)
from src.verbalizer.data.semantic import semantic_flags


DEFAULT_CAPTURE = ROOT / "data/reference/verbalizer_skravchenko_gemma3_15k.jsonl"
DEFAULT_LOCAL_REPORT = ROOT / "reports/e5-final-acceptance-15k-segmented.json"
DEFAULT_JSON = ROOT / "reports/reference-gemma3-parity.json"
DEFAULT_MARKDOWN = ROOT / "reports/reference-gemma3-parity.md"
DEFAULT_DISAGREEMENTS = ROOT / "reports/reference-gemma3-disagreements.jsonl"
APOSTROPHE_TRANSLATION = str.maketrans({"’": "'", "ʼ": "'", "`": "'"})


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
    return rows


def canonical_apostrophes(value: str) -> str:
    return unicodedata.normalize("NFC", value).translate(APOSTROPHE_TRANSLATION)


def metric_rows(
    joined: Iterable[dict[str, Any]],
    prediction_key: str,
    reference_key: str,
    *,
    apostrophe_canonical: bool = False,
) -> list[dict[str, Any]]:
    rows = []
    for row in joined:
        source = row["source"]
        reference = row[reference_key]
        prediction = row[prediction_key]
        if apostrophe_canonical:
            source = canonical_apostrophes(source)
            reference = canonical_apostrophes(reference)
            prediction = canonical_apostrophes(prediction)
        rows.append(
            {
                "id": row["id"],
                "source": source,
                "reference": reference,
                "prediction": prediction,
                "categories": row.get("categories") or [],
                "terminated_with_eos": True,
            }
        )
    return rows


def exact(left: str, right: str, *, apostrophe_canonical: bool = False) -> bool:
    if apostrophe_canonical:
        left = canonical_apostrophes(left)
        right = canonical_apostrophes(right)
    return normalize_evaluation_text(left) == normalize_evaluation_text(right)


def classify_output(source: str, target: str, output: str) -> dict[str, Any]:
    """Classify one output against the target without claiming human adjudication."""

    target_flags = set(semantic_flags(source, target))
    output_flags = set(semantic_flags(source, output))
    excess_flags = sorted(output_flags - target_flags)
    primary_exact = exact(output, target)
    canonical_exact = exact(output, target, apostrophe_canonical=True)
    difference_class = (
        "exact"
        if primary_exact
        else "apostrophe_only"
        if canonical_exact
        else "semantic_risk"
        if excess_flags
        else "other_target_difference"
    )
    return {
        "difference_class": difference_class,
        "primary_exact": primary_exact,
        "apostrophe_canonical_exact": canonical_exact,
        "target_floor_flags": sorted(target_flags),
        "output_flags": sorted(output_flags),
        "excess_semantic_flags": excess_flags,
    }


def classify_disagreements(joined: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    local_counts: Counter[str] = Counter()
    website_counts: Counter[str] = Counter()
    paired_counts: Counter[str] = Counter()
    category_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in joined:
        local = classify_output(row["source"], row["dataset_target"], row["local_output"])
        website = classify_output(row["source"], row["dataset_target"], row["remote_output"])
        local_counts[local["difference_class"]] += 1
        website_counts[website["difference_class"]] += 1
        pair = f"local:{local['difference_class']}|website:{website['difference_class']}"
        paired_counts[pair] += 1
        for category in row.get("categories") or ["uncategorized"]:
            category_counts[str(category)][website["difference_class"]] += 1
        if (
            local["difference_class"] != "exact"
            or website["difference_class"] != "exact"
            or not exact(row["local_output"], row["remote_output"])
        ):
            rows.append(
                {
                    "index": row["index"],
                    "id": row["id"],
                    "source": row["source"],
                    "dataset_target": row["dataset_target"],
                    "local_output": row["local_output"],
                    "website_output": row["remote_output"],
                    "categories": row.get("categories") or [],
                    "local_classification": local,
                    "website_classification": website,
                }
            )
    summary = {
        "classified_rows": len(joined),
        "written_disagreement_rows": len(rows),
        "local_difference_classes": dict(sorted(local_counts.items())),
        "website_difference_classes": dict(sorted(website_counts.items())),
        "paired_difference_classes": dict(sorted(paired_counts.items())),
        "website_classes_by_category": {
            category: dict(sorted(counts.items()))
            for category, counts in sorted(category_counts.items())
        },
        "taxonomy_note": (
            "semantic_risk means the conservative checker added flags beyond the "
            "target row's own floor; other_target_difference requires policy or "
            "morphological review and is not automatically a semantic error"
        ),
    }
    return rows, summary


def compact_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    overall = metrics["overall"]
    return {
        "rows": overall["rows"],
        "exact": overall["exact"],
        "exact_match": overall["exact_match"],
        "cer": overall["cer"],
        "wer": overall["wer"],
        "character_3gram_f1": overall["character_3gram_f1"],
        "semantic_flag_rate": overall["semantic_flag_rate"],
        "empty_predictions": overall["empty_predictions"],
        "likely_deletions": overall["likely_deletions"],
        "likely_insertions": overall["likely_insertions"],
        "repetition_loops": overall["repetition_loops"],
        "punctuation_corruptions": overall["punctuation_corruptions"],
        "invalid_unicode": overall["invalid_unicode"],
        "output_truncations": overall["output_truncations"],
        "repetition_loop_rate": overall["repetition_loop_rate"],
        "output_truncation_rate": overall["output_truncation_rate"],
    }


def critical_failure_count(metrics: dict[str, Any]) -> int:
    overall = metrics["overall"]
    return sum(
        int(overall[key])
        for key in (
            "empty_predictions",
            "likely_deletions",
            "likely_insertions",
            "repetition_loops",
            "punctuation_corruptions",
            "invalid_unicode",
            "output_truncations",
        )
    )


def parity_gate(
    comparisons: dict[str, dict[str, Any]],
    semantic_reviews: dict[str, dict[str, Any]],
    *,
    capture_complete: bool,
    require_local_exact: bool = True,
) -> dict[str, Any]:
    local = comparisons["local_vs_dataset_target"]["overall"]
    website = comparisons["remote_vs_dataset_target"]["overall"]
    local_critical = critical_failure_count(comparisons["local_vs_dataset_target"])
    website_critical = critical_failure_count(comparisons["remote_vs_dataset_target"])
    checks = {
        "target_exact_no_worse": local["exact_match"] >= website["exact_match"],
        "target_cer_no_worse": local["cer"] <= website["cer"],
        "target_wer_no_worse": local["wer"] <= website["wer"],
        "excess_semantic_flags_no_worse": (
            semantic_reviews["local"]["excess_flagged_rows"]
            <= semantic_reviews["website"]["excess_flagged_rows"]
        ),
        "critical_failures_no_worse": local_critical <= website_critical,
    }
    if require_local_exact:
        checks = {
            "local_target_exact_is_100_percent": local["exact_match"] == 1.0,
            **checks,
        }
    all_checks_pass = all(checks.values())
    return {
        "status": (
            "pass"
            if capture_complete and all_checks_pass
            else "fail"
            if capture_complete
            else "provisional"
        ),
        "capture_complete": capture_complete,
        "require_local_target_exact": require_local_exact,
        "all_comparative_checks_pass": all_checks_pass,
        "checks": checks,
        "values": {
            "local": {
                "target_exact_match": local["exact_match"],
                "target_cer": local["cer"],
                "target_wer": local["wer"],
                "excess_semantic_flagged_rows": semantic_reviews["local"][
                    "excess_flagged_rows"
                ],
                "critical_failures": local_critical,
            },
            "website": {
                "target_exact_match": website["exact_match"],
                "target_cer": website["cer"],
                "target_wer": website["wer"],
                "excess_semantic_flagged_rows": semantic_reviews["website"][
                    "excess_flagged_rows"
                ],
                "critical_failures": website_critical,
            },
        },
    }


def category_comparison(joined: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in joined:
        local_ok = exact(row["local_output"], row["dataset_target"], apostrophe_canonical=True)
        remote_ok = exact(row["remote_output"], row["dataset_target"], apostrophe_canonical=True)
        outcome = (
            "both_exact"
            if local_ok and remote_ok
            else "local_only_exact"
            if local_ok
            else "remote_only_exact"
            if remote_ok
            else "neither_exact"
        )
        for category in row.get("categories") or ["uncategorized"]:
            counts[str(category)]["rows"] += 1
            counts[str(category)][outcome] += 1
    return {category: dict(values) for category, values in sorted(counts.items())}


def semantic_review(joined: list[dict[str, Any]], output_key: str) -> dict[str, Any]:
    """Report conservative flags relative to the dataset target's own floor."""
    output_flag_counts: Counter[str] = Counter()
    target_flag_counts: Counter[str] = Counter()
    excess_flag_counts: Counter[str] = Counter()
    output_flagged_rows = 0
    target_flagged_rows = 0
    excess_flagged_rows = 0
    for row in joined:
        target_flags = set(semantic_flags(row["source"], row["dataset_target"]))
        output_flags = set(semantic_flags(row["source"], row[output_key]))
        excess_flags = output_flags - target_flags
        target_flag_counts.update(target_flags)
        output_flag_counts.update(output_flags)
        excess_flag_counts.update(excess_flags)
        target_flagged_rows += int(bool(target_flags))
        output_flagged_rows += int(bool(output_flags))
        excess_flagged_rows += int(bool(excess_flags))
    rows = len(joined)
    return {
        "rows": rows,
        "target_floor_flagged_rows": target_flagged_rows,
        "target_floor_flag_rate": target_flagged_rows / rows if rows else 0.0,
        "output_flagged_rows": output_flagged_rows,
        "output_flag_rate": output_flagged_rows / rows if rows else 0.0,
        "excess_flagged_rows": excess_flagged_rows,
        "excess_flag_rate": excess_flagged_rows / rows if rows else 0.0,
        "target_floor_flag_counts": dict(sorted(target_flag_counts.items())),
        "output_flag_counts": dict(sorted(output_flag_counts.items())),
        "excess_flag_counts": dict(sorted(excess_flag_counts.items())),
    }


def comparison(args: argparse.Namespace) -> dict[str, Any]:
    capture = read_jsonl(args.capture)
    with args.local_report.open(encoding="utf-8") as handle:
        local_report = json.load(handle)
    local_examples = local_report.get("examples")
    if not isinstance(local_examples, list):
        raise ValueError(f"{args.local_report} has no examples list")
    by_id = {str(row["id"]): row for row in local_examples}
    if len(by_id) != len(local_examples):
        raise ValueError("local report contains duplicate IDs")

    joined: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for expected_index, remote in enumerate(capture):
        if remote.get("index") != expected_index:
            raise ValueError(f"capture is non-contiguous at index {expected_index}")
        row_id = str(remote["id"])
        if row_id in seen_ids:
            raise ValueError(f"capture contains duplicate ID {row_id}")
        seen_ids.add(row_id)
        local = by_id.get(row_id)
        if local is None:
            raise ValueError(f"capture ID is absent from local report: {row_id}")
        if remote["source"] != local["source"]:
            raise ValueError(f"source mismatch for ID {row_id}")
        joined.append(
            {
                "index": expected_index,
                "id": row_id,
                "source": local["source"],
                "dataset_target": local["reference"],
                "local_output": local["prediction"],
                "remote_output": remote["reference_output"],
                "categories": local.get("categories") or [],
            }
        )

    metric_inputs = {
        "primary:local_vs_dataset_target": metric_rows(
            joined, "local_output", "dataset_target"
        ),
        "primary:remote_vs_dataset_target": metric_rows(
            joined, "remote_output", "dataset_target"
        ),
        "primary:local_vs_remote": metric_rows(
            joined, "local_output", "remote_output"
        ),
        "apostrophe:local_vs_dataset_target": metric_rows(
            joined, "local_output", "dataset_target", apostrophe_canonical=True
        ),
        "apostrophe:remote_vs_dataset_target": metric_rows(
            joined, "remote_output", "dataset_target", apostrophe_canonical=True
        ),
        "apostrophe:local_vs_remote": metric_rows(
            joined, "local_output", "remote_output", apostrophe_canonical=True
        ),
    }
    if args.metric_workers == 1:
        evaluated = {
            key: evaluate_examples(rows) for key, rows in metric_inputs.items()
        }
    else:
        with ProcessPoolExecutor(max_workers=args.metric_workers) as executor:
            futures = {
                key: executor.submit(evaluate_examples, rows)
                for key, rows in metric_inputs.items()
            }
            evaluated = {key: future.result() for key, future in futures.items()}
    comparisons = {
        key.removeprefix("primary:"): value
        for key, value in evaluated.items()
        if key.startswith("primary:")
    }
    apostrophe_comparisons = {
        key.removeprefix("apostrophe:"): value
        for key, value in evaluated.items()
        if key.startswith("apostrophe:")
    }

    outcomes: Counter[str] = Counter()
    disagreements: list[dict[str, Any]] = []
    for row in joined:
        local_ok = exact(row["local_output"], row["dataset_target"], apostrophe_canonical=True)
        remote_ok = exact(row["remote_output"], row["dataset_target"], apostrophe_canonical=True)
        local_remote_ok = exact(
            row["local_output"], row["remote_output"], apostrophe_canonical=True
        )
        outcome = (
            "both_exact"
            if local_ok and remote_ok
            else "local_only_exact"
            if local_ok
            else "remote_only_exact"
            if remote_ok
            else "same_non_target"
            if local_remote_ok
            else "all_differ"
        )
        outcomes[outcome] += 1
        if outcome != "both_exact" and len(disagreements) < args.sample_limit:
            disagreements.append({**row, "outcome": outcome})

    classified_disagreements, classification_summary = classify_disagreements(joined)
    semantic_reviews = {
        "local": semantic_review(joined, "local_output"),
        "website": semantic_review(joined, "remote_output"),
    }
    capture_complete = len(joined) == len(local_examples)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "captured_rows": len(joined),
            "frozen_dataset_rows": len(local_examples),
            "capture_complete": capture_complete,
            "capture_path": str(args.capture),
            "local_report_path": str(args.local_report),
            "reference_url": capture[0].get("reference_url") if capture else None,
            "reference_endpoint": capture[0].get("reference_endpoint") if capture else None,
            "reference_model": capture[0].get("reference_model") if capture else None,
        },
        "metric_policy": {
            "primary": "NFC, collapsed whitespace, and no space before sentence punctuation",
            "apostrophe_canonical": "primary plus ASCII/curly/modifier apostrophes mapped to ASCII",
            "dataset_target_note": "Frozen dataset target; remote output is evaluation evidence, not training data.",
        },
        "primary": {key: compact_metrics(value) for key, value in comparisons.items()},
        "apostrophe_canonical": {
            key: compact_metrics(value) for key, value in apostrophe_comparisons.items()
        },
        "outcomes_apostrophe_canonical": dict(sorted(outcomes.items())),
        "categories_apostrophe_canonical": category_comparison(joined),
        "semantic_review_relative_to_target_floor": semantic_reviews,
        "quality_gate": parity_gate(
            comparisons,
            semantic_reviews,
            capture_complete=capture_complete,
            require_local_exact=not getattr(args, "allow_nonexact_local", False),
        ),
        "disagreement_classification": classification_summary,
        "disagreement_samples": disagreements,
        "_classified_disagreements": classified_disagreements,
    }


def percent(value: float) -> str:
    return f"{value * 100:.3f}%"


def markdown(report: dict[str, Any]) -> str:
    scope = report["scope"]
    lines = [
        "# Reference API parity",
        "",
        f"Captured **{scope['captured_rows']:,} / {scope['frozen_dataset_rows']:,}** frozen rows. "
        f"Complete: **{scope['capture_complete']}**.",
        "",
        "The frozen dataset target and the website output are reported separately. "
        "The website is a model-generated comparator, not ground-truth training data.",
        "",
        "## Metrics",
        "",
        "| Comparison | Exact | CER | WER | Apostrophe-normalized exact |",
        "|---|---:|---:|---:|---:|",
    ]
    labels = {
        "local_vs_dataset_target": "Local → dataset target",
        "remote_vs_dataset_target": "Website → dataset target",
        "local_vs_remote": "Local → website output",
    }
    for key, label in labels.items():
        primary = report["primary"][key]
        canonical = report["apostrophe_canonical"][key]
        lines.append(
            f"| {label} | {percent(primary['exact_match'])} | "
            f"{percent(primary['cer'])} | {percent(primary['wer'])} | "
            f"{percent(canonical['exact_match'])} |"
        )
    lines.extend(["", "## Paired target outcomes", ""])
    for key, value in report["outcomes_apostrophe_canonical"].items():
        lines.append(f"- `{key}`: {value:,}")
    lines.extend(["", "## Per-category exact wins", ""])
    lines.extend(
        [
            "| Category | Rows | Both | Local only | Website only | Neither |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for category, values in report["categories_apostrophe_canonical"].items():
        lines.append(
            f"| {category} | {values.get('rows', 0)} | {values.get('both_exact', 0)} | "
            f"{values.get('local_only_exact', 0)} | {values.get('remote_only_exact', 0)} | "
            f"{values.get('neither_exact', 0)} |"
        )
    lines.extend(["", "## Conservative semantic-review flags", ""])
    lines.append(
        "The checker has false positives even on dataset targets, so parity uses "
        "only flags added beyond each target row's own flag set."
    )
    lines.extend(
        [
            "",
            "| System | Output flagged | Added beyond target floor |",
            "|---|---:|---:|",
        ]
    )
    for key, label in (("local", "Local"), ("website", "Website")):
        values = report["semantic_review_relative_to_target_floor"][key]
        lines.append(
            f"| {label} | {values['output_flagged_rows']:,} "
            f"({percent(values['output_flag_rate'])}) | {values['excess_flagged_rows']:,} "
            f"({percent(values['excess_flag_rate'])}) |"
        )
    classes = report["disagreement_classification"]
    lines.extend(["", "## Exhaustive disagreement classification", ""])
    lines.append(
        f"Every captured row was classified; **{classes['written_disagreement_rows']:,}** "
        "non-identical rows were written to the disagreement JSONL artifact."
    )
    lines.extend(["", "| Website class | Rows |", "|---|---:|"])
    for key, value in classes["website_difference_classes"].items():
        lines.append(f"| `{key}` | {value:,} |")
    gate = report["quality_gate"]
    lines.extend(["", "## Quality gate", ""])
    lines.append(
        f"Status: **{gate['status']}**. Comparative checks currently pass: "
        f"**{gate['all_comparative_checks_pass']}**; capture complete: "
        f"**{gate['capture_complete']}**."
    )
    lines.extend(["", "| Check | Pass |", "|---|---:|"])
    for key, value in gate["checks"].items():
        lines.append(f"| `{key}` | {value} |")
    return "\n".join(lines) + "\n"


def enforce_requirements(
    report: dict[str, Any],
    *,
    require_complete: bool,
    require_gate_pass: bool,
) -> None:
    if require_complete and not report["scope"]["capture_complete"]:
        raise RuntimeError(
            f"reference capture is incomplete: {report['scope']['captured_rows']} / "
            f"{report['scope']['frozen_dataset_rows']} rows"
        )
    if require_gate_pass and report["quality_gate"]["status"] != "pass":
        raise RuntimeError(
            f"external parity gate status is {report['quality_gate']['status']!r}, "
            "expected 'pass'"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, default=DEFAULT_CAPTURE)
    parser.add_argument("--local-report", type=Path, default=DEFAULT_LOCAL_REPORT)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output-markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--output-disagreements", type=Path, default=DEFAULT_DISAGREEMENTS)
    parser.add_argument("--sample-limit", type=int, default=100)
    parser.add_argument(
        "--metric-workers",
        type=int,
        default=min(6, os.cpu_count() or 1),
        help="parallel processes used for the six independent metric views",
    )
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--require-gate-pass", action="store_true")
    parser.add_argument(
        "--allow-nonexact-local",
        action="store_true",
        help="Gate only on no-worse parity; do not require 100%% local target exact.",
    )
    args = parser.parse_args()
    if args.sample_limit < 0 or args.metric_workers < 1:
        parser.error("sample-limit must be non-negative; metric-workers must be positive")
    return args


def main() -> None:
    args = parse_args()
    report = comparison(args)
    classified_disagreements = report.pop("_classified_disagreements")
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_disagreements.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.output_markdown.write_text(markdown(report), encoding="utf-8")
    with args.output_disagreements.open("w", encoding="utf-8") as handle:
        for row in classified_disagreements:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps(report["scope"], ensure_ascii=False, sort_keys=True))
    enforce_requirements(
        report,
        require_complete=args.require_complete,
        require_gate_pass=args.require_gate_pass,
    )


if __name__ == "__main__":
    main()
