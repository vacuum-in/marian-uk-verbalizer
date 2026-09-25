"""Build leakage-checked structured-domain adaptation splits.

The synthetic raw files are produced by the verified generator in the sibling
hybrid project.  The frozen 15k diverse corpus and user regressions are never
eligible for training or validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from prepare_current_dataset import (
    canonical_rows,
    canonical_text,
    normalize_text,
    sha256_file,
    write_rows,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_RAW = ROOT / "data/target_domain/raw/train.parquet"
DEFAULT_VALIDATION_RAW = ROOT / "data/target_domain/raw/validation.parquet"
DEFAULT_ACCEPTANCE = ROOT / "data/processed/test_template_ood.parquet"
DEFAULT_REGRESSIONS = ROOT / "data/challenge/user_target_regressions.jsonl"
DEFAULT_IDENTITY = ROOT / "data/pilot/train.parquet"
DEFAULT_OUTPUT = ROOT / "data/target_domain"
DATASET_VERSION = "structured-spoken-v1-2026-07-31"


def read_raw_pairs(path: Path) -> list[tuple[str, str]]:
    table = pq.read_table(path, columns=["inputs", "labels"])
    return [
        (normalize_text(source), normalize_text(target))
        for source, target in zip(
            table.column("inputs").to_pylist(),
            table.column("labels").to_pylist(),
            strict=True,
        )
    ]


def read_canonical_pairs(path: Path) -> list[tuple[str, str]]:
    table = pq.read_table(path, columns=["source", "target"])
    return [
        (normalize_text(source), normalize_text(target))
        for source, target in zip(
            table.column("source").to_pylist(),
            table.column("target").to_pylist(),
            strict=True,
        )
    ]


def read_regressions(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row.get("source"), str) or not isinstance(row.get("target"), str):
                raise ValueError(f"invalid regression row at {path}:{line_number}")
            rows.append(row)
    return rows


def source_key(source: str) -> str:
    return canonical_text(normalize_text(source))


def pair_key(pair: tuple[str, str]) -> tuple[str, str]:
    return source_key(pair[0]), canonical_text(normalize_text(pair[1]))


def unique_pairs(
    pairs: list[tuple[str, str]],
    forbidden_sources: set[str],
    seen_sources: set[str],
) -> tuple[list[tuple[str, str]], Counter[str]]:
    retained: list[tuple[str, str]] = []
    counts: Counter[str] = Counter()
    for pair in pairs:
        key = source_key(pair[0])
        if key in forbidden_sources:
            counts["forbidden_source"] += 1
            continue
        if key in seen_sources:
            counts["duplicate_source"] += 1
            continue
        seen_sources.add(key)
        retained.append(pair)
    return retained, counts


def stamp_rows(pairs: list[tuple[str, str]], split: str) -> list[dict[str, Any]]:
    rows = canonical_rows(pairs)
    for row in rows:
        value = f"{DATASET_VERSION}\0{split}\0{row['source']}\0{row['target']}"
        row["id"] = hashlib.sha256(value.encode("utf-8")).hexdigest()
        row["dataset_version"] = DATASET_VERSION
        row["quality_flags"] = []
        row["quality_tier"] = "A"
    return rows


def build(args: argparse.Namespace) -> dict[str, Any]:
    regression_records = read_regressions(args.regressions)
    regression_pairs = [
        (normalize_text(row["source"]), normalize_text(row["target"]))
        for row in regression_records
    ]
    acceptance_pairs = read_canonical_pairs(args.acceptance)
    forbidden_sources = {
        source_key(source) for source, _ in acceptance_pairs + regression_pairs
    }

    validation_pairs, validation_drops = unique_pairs(
        read_raw_pairs(args.validation_raw),
        forbidden_sources,
        set(),
    )
    validation_sources = {source_key(source) for source, _ in validation_pairs}
    train_pairs, train_drops = unique_pairs(
        read_raw_pairs(args.train_raw),
        forbidden_sources | validation_sources,
        set(),
    )

    identity_table = pq.read_table(
        args.identity,
        columns=["source", "target", "requires_change"],
    ).to_pylist()
    identity_pool = [
        (normalize_text(row["source"]), normalize_text(row["target"]))
        for row in identity_table
        if not bool(row["requires_change"])
    ]
    rng = random.Random(args.seed)
    rng.shuffle(identity_pool)
    identity_pairs, identity_drops = unique_pairs(
        identity_pool,
        forbidden_sources | validation_sources,
        {source_key(source) for source, _ in train_pairs},
    )
    identity_pairs = identity_pairs[: args.identity_rows]
    train_pairs.extend(identity_pairs)
    rng.shuffle(train_pairs)

    train_rows = stamp_rows(train_pairs, "train")
    validation_rows = stamp_rows(validation_pairs, "validation")
    regression_rows = stamp_rows(regression_pairs, "regression")
    for row, record in zip(regression_rows, regression_records, strict=True):
        row["id"] = record["id"]
        row["categories"] = list(record.get("categories") or row["categories"])

    outputs = {
        "train": train_rows,
        "validation": validation_rows,
        "regressions": regression_rows,
    }
    output_hashes = {
        name: write_rows(args.output / f"{name}.parquet", rows)
        for name, rows in outputs.items()
    }

    train_source_keys = {source_key(row["source"]) for row in train_rows}
    validation_source_keys = {source_key(row["source"]) for row in validation_rows}
    acceptance_source_keys = {source_key(source) for source, _ in acceptance_pairs}
    regression_source_keys = {source_key(source) for source, _ in regression_pairs}
    leakage = {
        "train_validation_source_overlap": len(train_source_keys & validation_source_keys),
        "train_acceptance_source_overlap": len(train_source_keys & acceptance_source_keys),
        "train_regression_source_overlap": len(train_source_keys & regression_source_keys),
        "validation_acceptance_source_overlap": len(validation_source_keys & acceptance_source_keys),
        "validation_regression_source_overlap": len(validation_source_keys & regression_source_keys),
        "train_acceptance_pair_overlap": len(
            {pair_key(pair) for pair in train_pairs} & {pair_key(pair) for pair in acceptance_pairs}
        ),
    }
    if any(leakage.values()):
        raise RuntimeError(f"dataset leakage detected: {leakage}")

    manifest = {
        "dataset_version": DATASET_VERSION,
        "policy_profile": "spoken-structured-domain",
        "seed": args.seed,
        "sources": {
            "train_raw": {"path": str(args.train_raw), "sha256": sha256_file(args.train_raw)},
            "validation_raw": {
                "path": str(args.validation_raw),
                "sha256": sha256_file(args.validation_raw),
            },
            "identity": {"path": str(args.identity), "sha256": sha256_file(args.identity)},
            "acceptance_frozen": {
                "path": str(args.acceptance),
                "sha256": sha256_file(args.acceptance),
                "rows": len(acceptance_pairs),
            },
            "user_regressions": {
                "path": str(args.regressions),
                "sha256": sha256_file(args.regressions),
                "rows": len(regression_pairs),
            },
        },
        "rows": {name: len(rows) for name, rows in outputs.items()},
        "identity_rows": len(identity_pairs),
        "drops": {
            "train": dict(train_drops),
            "validation": dict(validation_drops),
            "identity": dict(identity_drops),
        },
        "leakage": leakage,
        "output_sha256": output_hashes,
        "notes": [
            "The frozen 15k acceptance corpus is read only for exclusion checks.",
            "User regression sources are excluded from both train and validation.",
            "Zero exact-source overlap demonstrates held-out instances, not universal generalization.",
        ],
    }
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-raw", type=Path, default=DEFAULT_TRAIN_RAW)
    parser.add_argument("--validation-raw", type=Path, default=DEFAULT_VALIDATION_RAW)
    parser.add_argument("--acceptance", type=Path, default=DEFAULT_ACCEPTANCE)
    parser.add_argument("--regressions", type=Path, default=DEFAULT_REGRESSIONS)
    parser.add_argument("--identity", type=Path, default=DEFAULT_IDENTITY)
    parser.add_argument("--identity-rows", type=int, default=15_000)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(build(parse_args()), ensure_ascii=False, indent=2, sort_keys=True))
