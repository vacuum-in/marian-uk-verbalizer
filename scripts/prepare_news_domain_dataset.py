"""Prepare leakage-safe Marian splits from the sibling 95k/news corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
SIBLING_DATA = (
    ROOT.parent
    / "hybrid-ukrainian-tts-verbalizer-openspec"
    / "verbolisation"
    / "data"
)
DEFAULT_TRAIN = SIBLING_DATA / "main_train_95k.parquet"
DEFAULT_TEST = SIBLING_DATA / "golden_5k.parquet"
DEFAULT_OUTPUT = ROOT / "data/news_domain"
DATASET_VERSION = "news-95k-v1-2026-07-31"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFC", str(value)).split())


def source_key(value: str) -> str:
    return normalize_text(value).casefold()


def read_pairs(path: Path) -> list[tuple[str, str]]:
    rows = pq.read_table(path, columns=["inputs", "labels"]).to_pylist()
    return [(normalize_text(row["inputs"]), normalize_text(row["labels"])) for row in rows]


def stamp(pairs: list[tuple[str, str]], split: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source, target in pairs:
        identity = hashlib.sha256(
            f"{DATASET_VERSION}\0{split}\0{source}\0{target}".encode("utf-8")
        ).hexdigest()
        rows.append(
            {
                "id": identity,
                "source": source,
                "target": target,
                "categories": ["news_domain"],
                "dataset_version": DATASET_VERSION,
                "quality_flags": [],
            }
        )
    return rows


def write_rows(path: Path, rows: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")
    return sha256_file(path)


def build(args: argparse.Namespace) -> dict[str, Any]:
    raw_train = read_pairs(args.train)
    raw_test = read_pairs(args.test)
    test_sources = {source_key(source) for source, _ in raw_test}

    source_counts = Counter(source_key(source) for source, _ in raw_train)
    conflicting_sources = {key for key, count in source_counts.items() if count > 1}
    unique_train = [
        pair
        for pair in raw_train
        if source_key(pair[0]) not in conflicting_sources
        and source_key(pair[0]) not in test_sources
    ]
    ranked = sorted(
        unique_train,
        key=lambda pair: hashlib.sha256(
            f"{args.seed}\0{pair[0]}\0{pair[1]}".encode("utf-8")
        ).hexdigest(),
    )
    if not 0 < args.validation_rows < len(ranked):
        raise ValueError("validation_rows must leave non-empty train and validation splits")
    validation_pairs = ranked[: args.validation_rows]
    train_pairs = ranked[args.validation_rows :]

    train_sources = {source_key(source) for source, _ in train_pairs}
    validation_sources = {source_key(source) for source, _ in validation_pairs}
    leakage = {
        "train_validation_source_overlap": len(train_sources & validation_sources),
        "train_test_source_overlap": len(train_sources & test_sources),
        "validation_test_source_overlap": len(validation_sources & test_sources),
    }
    if any(leakage.values()):
        raise RuntimeError(f"dataset leakage detected: {leakage}")

    outputs = {
        "train": stamp(train_pairs, "train"),
        "validation": stamp(validation_pairs, "validation"),
        "test": stamp(raw_test, "test"),
    }
    output_hashes = {
        split: write_rows(args.output / f"{split}.parquet", rows)
        for split, rows in outputs.items()
    }
    manifest = {
        "dataset_version": DATASET_VERSION,
        "seed": args.seed,
        "sources": {
            "main_train_95k": {
                "path": str(args.train.resolve()),
                "rows": len(raw_train),
                "sha256": sha256_file(args.train),
            },
            "golden_5k": {
                "path": str(args.test.resolve()),
                "rows": len(raw_test),
                "sha256": sha256_file(args.test),
            },
        },
        "rows": {split: len(rows) for split, rows in outputs.items()},
        "drops": {
            "conflicting_duplicate_source_rows": sum(
                source_counts[key] for key in conflicting_sources
            ),
            "test_source_overlap_rows": sum(
                source_key(source) in test_sources for source, _ in raw_train
            ),
        },
        "leakage": leakage,
        "output_sha256": output_hashes,
        "policy": [
            "golden_5k is final test only and is never used for checkpoint selection",
            "validation is a deterministic hash split from main_train_95k",
            "ambiguous duplicate sources are excluded rather than selecting a target",
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
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--validation-rows", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=71)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(build(parse_args()), ensure_ascii=False, indent=2, sort_keys=True))
