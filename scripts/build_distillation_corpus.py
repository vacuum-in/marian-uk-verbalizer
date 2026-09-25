#!/usr/bin/env python3
"""Build a leakage-safe Marian corpus from canonical and Gemma-labelled text.

The frozen test sets are passed explicitly and are excluded by normalized
source text before any split is written.  Validation is canonical-only, so
checkpoint selection never depends on a teacher-generated label.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.verbalizer.data.audit import canonical_source, row_flags  # noqa: E402
from src.verbalizer.distillation import stable_row_id  # noqa: E402
from src.verbalizer.numeric import formula_supervision_valid  # noqa: E402
from src.verbalizer.training.manifest import sha256_file, write_json  # noqa: E402


def read_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return pq.read_table(path).to_pylist()


def source_from_row(row: dict[str, Any]) -> str:
    for name in ("source", "original", "inputs"):
        if name in row:
            return str(row[name])
    raise ValueError("source row needs source, original, or inputs column")


def category(source: str) -> list[str]:
    lowered = source.casefold()
    categories: list[str] = []
    if any(marker in source for marker in ("@", ".com", ".ua")):
        categories.append("internet_or_email")
    if any(marker in lowered for marker in ("грн", "долар", "євро", "₴", "$", "€")):
        categories.append("currency")
    if any(marker in source for marker in (":", "/", ".")) and any(char.isdigit() for char in source):
        categories.append("structured_numeric")
    if any(char.isdigit() for char in source):
        categories.append("numeric")
    if any("A" <= char <= "Z" or "a" <= char <= "z" for char in source):
        categories.append("latin")
    return categories or ["general"]


def canonical_pairs(
    path: Path,
    source_column: str,
    target_column: str,
    provenance: str = "canonical",
) -> list[dict[str, Any]]:
    rows = pq.read_table(path, columns=[source_column, target_column]).to_pylist()
    return [
        {
            "source": str(row[source_column]),
            "target": str(row[target_column]),
            "provenance": provenance,
        }
        for row in rows
    ]


def teacher_pairs(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    approved: list[dict[str, Any]] = []
    quarantine: list[dict[str, Any]] = []
    for row in read_records(path):
        if row.get("status") != "approved":
            quarantine.append(row)
            continue
        approved.append(
            {
                "source": str(row["source"]),
                "target": str(row["teacher_output"]),
                "provenance": "gemma_teacher",
                "review_flags": list(row.get("review_flags") or []),
            }
        )
    return approved, quarantine


def write_parquet(path: Path, rows: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")
    return sha256_file(path)


def build(args: argparse.Namespace) -> dict[str, Any]:
    frozen_sources: set[str] = set()
    frozen_hashes: dict[str, str] = {}
    for path in args.frozen:
        records = read_records(path)
        frozen_sources.update(canonical_source(source_from_row(row)) for row in records)
        frozen_hashes[str(path)] = sha256_file(path)
    canonical = canonical_pairs(
        args.canonical,
        args.canonical_source_column,
        args.canonical_target_column,
    )
    additional_canonical = list(getattr(args, "additional_canonical", []))
    for path in additional_canonical:
        canonical.extend(canonical_pairs(path, "source", "target", "canonical_hf"))
    teacher, quarantine = teacher_pairs(args.teacher_capture) if args.teacher_capture else ([], [])

    canonical_by_source: dict[str, dict[str, Any]] = {}
    drops: Counter[str] = Counter()
    for row in canonical:
        key = canonical_source(row["source"])
        if not formula_supervision_valid(row["source"], row["target"]):
            drops["canonical_formula_identity_failure"] += 1
            quarantine.append({**row, "quarantine_reason": "formula_identity_failure"})
            continue
        if not key or key in frozen_sources:
            drops["canonical_frozen_overlap"] += 1
            continue
        if key in canonical_by_source and canonical_by_source[key]["target"] != row["target"]:
            drops["canonical_conflicting_target"] += 1
            continue
        canonical_by_source[key] = row

    teacher_by_source: dict[str, dict[str, Any]] = {}
    for row in teacher:
        key = canonical_source(row["source"])
        if not formula_supervision_valid(row["source"], row["target"]):
            drops["teacher_formula_identity_failure"] += 1
            quarantine.append({**row, "quarantine_reason": "formula_identity_failure"})
            continue
        if not key or key in frozen_sources:
            drops["teacher_frozen_overlap"] += 1
            quarantine.append({**row, "quarantine_reason": "frozen_evaluation_source"})
            continue
        if key in canonical_by_source:
            # Canonical supervision wins any policy disagreement and avoids duplicate sources.
            drops["teacher_canonical_overlap"] += 1
            if canonical_by_source[key]["target"] != row["target"]:
                quarantine.append({**row, "quarantine_reason": "canonical_teacher_disagreement"})
            continue
        # Teacher rows with non-critical semantic/formatting flags require the
        # explicit human-review path instead of silently entering the corpus.
        if row.get("review_flags") or row_flags(row["source"], row["target"]):
            drops["teacher_quality_review"] += 1
            quarantine.append({**row, "quarantine_reason": "teacher_quality_flags"})
            continue
        teacher_by_source.setdefault(key, row)

    ordered_canonical = sorted(
        canonical_by_source.values(), key=lambda row: stable_row_id("e21-canonical", row["source"], row["target"])
    )
    if args.validation_rows >= len(ordered_canonical):
        raise ValueError("validation-rows must leave canonical rows for training")
    validation_pairs = ordered_canonical[: args.validation_rows]
    train_pairs = [*ordered_canonical[args.validation_rows :], *teacher_by_source.values()]
    train_keys = {canonical_source(row["source"]) for row in train_pairs}
    validation_keys = {canonical_source(row["source"]) for row in validation_pairs}
    leakage = {
        "train_validation_source_overlap": len(train_keys & validation_keys),
        "train_frozen_source_overlap": len(train_keys & frozen_sources),
        "validation_frozen_source_overlap": len(validation_keys & frozen_sources),
    }
    if any(leakage.values()):
        raise RuntimeError(f"distillation leakage detected: {leakage}")

    def materialize(split: str, pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "id": stable_row_id(f"e21-{split}", row["source"], row["target"]),
                "source": row["source"],
                "target": row["target"],
                "categories": category(row["source"]),
                "dataset_version": "e21-gemma-distillation-2026-08-20",
                "quality_flags": list(row.get("review_flags") or []),
                "provenance": row["provenance"],
            }
            for row in pairs
        ]

    train_rows = materialize("train", train_pairs)
    validation_rows = materialize("validation", validation_pairs)
    output_hashes = {
        "train": write_parquet(args.output / "train.parquet", train_rows),
        "validation": write_parquet(args.output / "validation.parquet", validation_rows),
    }
    quarantine_path = args.output / "quarantine.jsonl"
    quarantine_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in quarantine), encoding="utf-8"
    )
    manifest = {
        "dataset_version": "e21-gemma-distillation-2026-08-20",
        "canonical": {"path": str(args.canonical), "sha256": sha256_file(args.canonical)},
        "teacher_capture": (
            {"path": str(args.teacher_capture), "sha256": sha256_file(args.teacher_capture)}
            if args.teacher_capture
            else None
        ),
        "additional_canonical": [
            {"path": str(path), "sha256": sha256_file(path)} for path in additional_canonical
        ],
        "frozen_evaluations": frozen_hashes,
        "rows": {
            "train": len(train_rows),
            "validation": len(validation_rows),
            "canonical_train": sum(row["provenance"] == "canonical" for row in train_rows),
            "teacher_train": sum(row["provenance"] == "gemma_teacher" for row in train_rows),
            "quarantine": len(quarantine),
        },
        "drops": dict(sorted(drops.items())),
        "leakage": leakage,
        "output_sha256": output_hashes,
        "policy": [
            "Every supplied frozen evaluation source is excluded from train and validation.",
            "Validation contains canonical labels only; teacher outputs cannot select checkpoints.",
            "Canonical targets win source collisions; reviewed teacher data is used only for novel sources.",
            "Teacher capture is optional for a canonical-only supervised continuation run.",
        ],
    }
    write_json(args.output / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", required=True, type=Path)
    parser.add_argument("--teacher-capture", type=Path)
    parser.add_argument(
        "--additional-canonical",
        type=Path,
        action="append",
        default=[],
        help="Additional prepared source/target parquet, such as the pinned HF train split.",
    )
    parser.add_argument("--frozen", required=True, type=Path, action="append")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--canonical-source-column", default="inputs")
    parser.add_argument("--canonical-target-column", default="labels")
    parser.add_argument("--validation-rows", type=int, default=20_000)
    args = parser.parse_args()
    print(json.dumps(build(args), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
