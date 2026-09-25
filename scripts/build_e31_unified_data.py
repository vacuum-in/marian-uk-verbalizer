#!/usr/bin/env python3
"""Build the E31 unified corpus: full labelled HF split + English transfer.

Two losses accumulated at the E24 build step and have been inherited by every
experiment since:

1.  ``data/hf_uktn_e23/train.parquet`` holds 167,893 rows carrying 169 category
    labels.  ``data/e24_pipeline_hf_plus_synthetic/train.parquet`` holds 118,007
    rows carrying 9.  49,886 rows and 160 labels were dropped.
2.  96,760 of those rows were dropped as ``deterministic_pipeline_sentence`` --
    sentences the *router* decided did not need Marian.  That filter encodes the
    routed architecture into the training data.  A single unified model has to
    see them, or it never learns what to leave alone.

This builder goes back to the labelled E23 split, keeps the categories, adds the
English transfer corpus, and applies the E30 apostrophe canonicalization.  It
also carves a category-stratified dev split so checkpoint selection can finally
see the evaluation distribution.

Frozen evaluation sources are excluded by exact match and the manifest records
the leakage count, which must be zero.  No verbalization rule is introduced:
every source/target pair is carried through literally.
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.verbalizer.training.manifest import sha256_file, write_json  # noqa: E402

CANONICAL_APOSTROPHE = "'"
APOSTROPHE_VARIANTS = "'’‘ʼ´`＇"
APOSTROPHE_TABLE = dict.fromkeys(map(ord, APOSTROPHE_VARIANTS), CANONICAL_APOSTROPHE)
AUGMENT_VARIANT = "’"

SPLIT_LABELS = {"main", "anchor", "transliteration", "agreement", "misc", "feedback", "uncategorized", "?"}


def canonicalize(text: str) -> str:
    return text.translate(APOSTROPHE_TABLE)


def read_rows(path: Path, provenance: str) -> list[dict[str, Any]]:
    table = pq.read_table(path, columns=["id", "source", "target", "categories"])
    return [
        {
            "id": str(row["id"]),
            "source": canonicalize(str(row["source"])),
            "target": canonicalize(str(row["target"])),
            "categories": [str(c) for c in (row.get("categories") or [])],
            "provenance": provenance,
        }
        for row in table.to_pylist()
    ]


def read_jsonl_rows(path: Path, provenance: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        record = json.loads(line)
        source = record.get("source") or record.get("original") or record.get("src")
        target = record.get("target") or record.get("verbalized") or record.get("tgt")
        if source is None or target is None:
            continue
        categories = record.get("categories") or record.get("combo_names") or [provenance]
        rows.append(
            {
                "id": f"{provenance}:{record.get('id', index)}",
                "source": canonicalize(str(source)),
                "target": canonicalize(str(target)),
                "categories": [str(c) for c in categories],
                "provenance": provenance,
            }
        )
    return rows


def augment_sources(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    out: list[dict[str, Any]] = []
    added = 0
    for row in rows:
        out.append(row)
        if CANONICAL_APOSTROPHE in row["source"]:
            out.append(
                {
                    **row,
                    "id": f"{row['id']}:apos-variant",
                    "source": row["source"].replace(CANONICAL_APOSTROPHE, AUGMENT_VARIANT),
                    "provenance": f"{row['provenance']}+apostrophe_variant",
                }
            )
            added += 1
    return out, added


def stratified_dev(
    rows: list[dict[str, Any]], per_category: int, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Hold out up to ``per_category`` rows for each real category label."""
    rng = random.Random(seed)
    by_category: dict[str, list[int]] = collections.defaultdict(list)
    for index, row in enumerate(rows):
        for category in row["categories"]:
            if category not in SPLIT_LABELS:
                by_category[category].append(index)

    held: set[int] = set()
    for category in sorted(by_category):
        candidates = [i for i in by_category[category] if i not in held]
        rng.shuffle(candidates)
        held.update(candidates[:per_category])

    dev = [rows[i] for i in sorted(held)]
    train = [row for i, row in enumerate(rows) if i not in held]
    return train, dev


def write_rows(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")
    return {"path": str(path), "rows": len(rows), "sha256": sha256_file(path)}


def category_histogram(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: collections.Counter[str] = collections.Counter()
    for row in rows:
        for category in row["categories"]:
            counts[category] += 1
    return dict(counts.most_common())


def build(args: argparse.Namespace) -> dict[str, Any]:
    frozen_sources = {
        json.loads(line)["source"]
        for line in args.frozen.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }

    ukrainian = read_rows(args.hf_train, "hf_uktn_e23_train")
    english = read_rows(args.english_train, "en_uk_transducer_train")
    synthetic = read_jsonl_rows(args.synthetic, "synthetic_failure_repair") if args.synthetic.exists() else []
    curated = read_jsonl_rows(args.curated, "abbreviation_supervision") if args.curated.exists() else []

    combined = [*ukrainian, *synthetic, *curated]
    before = len(combined)
    combined = [row for row in combined if row["source"] not in frozen_sources]
    leaked = before - len(combined)

    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for row in combined:
        if row["source"] in seen:
            continue
        seen.add(row["source"])
        deduped.append(row)
    duplicates = len(combined) - len(deduped)

    train_rows, dev_rows = stratified_dev(deduped, args.dev_per_category, args.seed)

    # The English transfer corpus is a separate source and has to clear the same
    # two exclusions as everything else: nothing from the frozen benchmark, and
    # nothing that collides with a held-out dev source.
    dev_sources = {row["source"] for row in dev_rows}
    english_before = len(english)
    english = [
        row for row in english if row["source"] not in frozen_sources and row["source"] not in dev_sources
    ]
    english_excluded = english_before - len(english)

    train_rows = [*train_rows, *english]
    train_rows, augmented = augment_sources(train_rows)

    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "canonical_apostrophe": f"U+{ord(CANONICAL_APOSTROPHE):04X}",
        "inputs": {
            "hf_train": {"path": str(args.hf_train), "rows": len(ukrainian), "sha256": sha256_file(args.hf_train)},
            "english_train": {
                "path": str(args.english_train),
                "rows": len(english),
                "sha256": sha256_file(args.english_train),
            },
            "synthetic": {"path": str(args.synthetic), "rows": len(synthetic)},
            "curated": {"path": str(args.curated), "rows": len(curated)},
        },
        "frozen_leakage_removed": leaked,
        "english_rows_excluded": english_excluded,
        "duplicate_sources_removed": duplicates,
        "apostrophe_variants_added": augmented,
        "dev_per_category": args.dev_per_category,
        "distinct_categories": {
            "train": len(category_histogram(train_rows)),
            "dev": len(category_histogram(dev_rows)),
        },
        "output": {
            "train": write_rows(args.output / "train.parquet", train_rows),
            "dev": write_rows(args.output / "dev-stratified.parquet", dev_rows),
        },
        "dev_category_histogram": category_histogram(dev_rows),
    }
    write_json(args.output / "manifest.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-train", type=Path, default=ROOT / "data/hf_uktn_e23/train.parquet")
    parser.add_argument("--english-train", type=Path, default=ROOT / "data/e28_english_transfer/english-train.parquet")
    parser.add_argument("--synthetic", type=Path, default=ROOT / "data/e24_synthetic_failure_repair/train.jsonl")
    parser.add_argument("--curated", type=Path, default=ROOT / "data/curated/abbreviation-supervision.jsonl")
    parser.add_argument("--frozen", type=Path, default=ROOT / "reports/hf-uktn-frozen-3597.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "data/e31_unified_labelled")
    parser.add_argument("--dev-per-category", type=int, default=12)
    parser.add_argument("--seed", type=int, default=3101)
    args = parser.parse_args()
    report = build(args)
    summary = {k: v for k, v in report.items() if k != "dev_category_histogram"}
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
