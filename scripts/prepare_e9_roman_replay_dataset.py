"""Mix Roman-numeral training with the complete E7 news/structured replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from prepare_current_dataset import canonical_text, sha256_file, write_rows  # noqa: E402


DEFAULT_REPLAY = ROOT / "data/news_domain_replay"
DEFAULT_ROMAN = ROOT / "data/roman_numerals"
DEFAULT_OUTPUT = ROOT / "data/e9_roman_replay"
DATASET_VERSION = "e9-roman-news-structured-replay-v1-2026-07-31"


def rows(path: Path) -> list[dict[str, Any]]:
    return pq.read_table(path).to_pylist()


def source_keys(values: list[dict[str, Any]]) -> set[str]:
    return {canonical_text(str(row["source"])) for row in values}


def build(args: argparse.Namespace) -> dict[str, Any]:
    replay_train = rows(args.replay / "train.parquet")
    replay_validation = rows(args.replay / "validation.parquet")
    roman_train = rows(args.roman / "train.parquet")
    roman_validation = rows(args.roman / "validation.parquet")
    roman_test = rows(args.roman / "test.parquet")

    train = [*replay_train, *roman_train]
    validation = [*replay_validation, *roman_validation]
    random.Random(args.seed).shuffle(train)

    train_sources = source_keys(train)
    validation_sources = source_keys(validation)
    test_sources = source_keys(roman_test)
    leakage = {
        "train_validation_source_overlap": len(train_sources & validation_sources),
        "train_roman_test_source_overlap": len(train_sources & test_sources),
        "validation_roman_test_source_overlap": len(validation_sources & test_sources),
    }
    if any(leakage.values()):
        raise RuntimeError(f"E9 replay leakage: {leakage}")

    args.output.mkdir(parents=True, exist_ok=True)
    hashes = {
        "train": write_rows(args.output / "train.parquet", train),
        "validation": write_rows(args.output / "validation.parquet", validation),
    }
    manifest = {
        "dataset_version": DATASET_VERSION,
        "seed": args.seed,
        "sources": {
            "replay_train": {"path": str(args.replay / "train.parquet"), "sha256": sha256_file(args.replay / "train.parquet"), "rows": len(replay_train)},
            "replay_validation": {"path": str(args.replay / "validation.parquet"), "sha256": sha256_file(args.replay / "validation.parquet"), "rows": len(replay_validation)},
            "roman_train": {"path": str(args.roman / "train.parquet"), "sha256": sha256_file(args.roman / "train.parquet"), "rows": len(roman_train)},
            "roman_validation": {"path": str(args.roman / "validation.parquet"), "sha256": sha256_file(args.roman / "validation.parquet"), "rows": len(roman_validation)},
            "roman_test_exclusion_only": {"path": str(args.roman / "test.parquet"), "sha256": sha256_file(args.roman / "test.parquet"), "rows": len(roman_test)},
        },
        "rows": {"train": len(train), "validation": len(validation)},
        "mixture": {"replay_train": len(replay_train), "roman_train": len(roman_train)},
        "leakage": leakage,
        "output_sha256": hashes,
        "policy": [
            "All E7 news/structured replay rows are retained to resist forgetting.",
            "Roman test targets are never copied into train or validation.",
            "Roman train, validation, and test values and source templates are disjoint.",
        ],
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, default=DEFAULT_REPLAY)
    parser.add_argument("--roman", type=Path, default=DEFAULT_ROMAN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=97)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(build(parse_args()), ensure_ascii=False, indent=2, sort_keys=True))
