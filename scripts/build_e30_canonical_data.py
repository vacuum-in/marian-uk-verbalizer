#!/usr/bin/env python3
"""Build E30 training splits with a single, audited apostrophe convention.

The E27/E28 corpora write every Ukrainian apostrophe as U+2019 (curly).  The
frozen evaluation references and the English-transducer corpus write it as
U+0027 (straight).  A model trained on the former can never produce an exact
match on the 35.7% of evaluation rows that contain one, which caps exact match
at 64.28% regardless of model quality.

This script canonicalizes the *training corpus* so the model learns the target
convention.  It is a dataset-build transform: nothing here runs at inference
time, and no prediction is ever rewritten.  Source text additionally gets a
both-glyph augmentation so the model stays robust to either input convention.

No verbalization rule, dictionary, or reading is introduced.  The apostrophe is
orthography of the same word, not a spoken form.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.verbalizer.training.manifest import sha256_file, write_json  # noqa: E402


# Every apostrophe-like glyph observed in the corpora, mapped to the convention
# used by the frozen references and the English-transducer corpus.
CANONICAL_APOSTROPHE = "'"
APOSTROPHE_VARIANTS = "'’‘ʼ´`＇"
APOSTROPHE_TABLE = dict.fromkeys(map(ord, APOSTROPHE_VARIANTS), CANONICAL_APOSTROPHE)

# The variant to emit as an extra source-side spelling, so the model sees both
# input conventions for the same target.
AUGMENT_VARIANT = "’"


def canonicalize(text: str) -> str:
    return text.translate(APOSTROPHE_TABLE)


def glyph_histogram(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts: collections.Counter[str] = collections.Counter()
    for row in rows:
        for char in row[field]:
            if char in APOSTROPHE_VARIANTS:
                counts[f"U+{ord(char):04X}"] += 1
    return dict(sorted(counts.items()))


def read_parquet_rows(path: Path) -> list[dict[str, Any]]:
    return [
        {
            "id": str(row["id"]),
            "source": str(row["source"]),
            "target": str(row["target"]),
            "categories": list(row.get("categories") or []),
            "provenance": str(row.get("provenance") or "base_raw"),
        }
        for row in pq.read_table(
            path, columns=["id", "source", "target", "categories", "provenance"]
        ).to_pylist()
    ]


def canonicalize_rows(rows: list[dict[str, Any]], augment: bool) -> tuple[list[dict[str, Any]], int, int]:
    """Return canonicalized rows plus the number of edits and augmented copies."""
    out: list[dict[str, Any]] = []
    edited = 0
    augmented = 0
    for row in rows:
        source = canonicalize(row["source"])
        target = canonicalize(row["target"])
        if source != row["source"] or target != row["target"]:
            edited += 1
        out.append({**row, "source": source, "target": target})
        if augment and CANONICAL_APOSTROPHE in source:
            # Same target, alternate source spelling: teaches glyph invariance
            # on the input side without inventing a new reading.
            out.append(
                {
                    **row,
                    "id": f"{row['id']}:apos-variant",
                    "source": source.replace(CANONICAL_APOSTROPHE, AUGMENT_VARIANT),
                    "target": target,
                    "provenance": f"{row['provenance']}+apostrophe_variant",
                }
            )
            augmented += 1
    return out, edited, augmented


def write_rows(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")
    return {"path": str(path), "rows": len(rows), "sha256": sha256_file(path)}


def build(base_dir: Path, output_dir: Path, augment: bool) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "canonical_apostrophe": f"U+{ord(CANONICAL_APOSTROPHE):04X}",
        "source_augmentation": augment,
        "base_dir": str(base_dir),
        "splits": {},
    }

    for name, augment_split in (("train", augment), ("validation", False)):
        src = base_dir / f"{name}.parquet"
        rows = read_parquet_rows(src)
        before = {
            "source": glyph_histogram(rows, "source"),
            "target": glyph_histogram(rows, "target"),
        }
        canonical, edited, augmented = canonicalize_rows(rows, augment_split)
        after = {
            "source": glyph_histogram(canonical, "source"),
            "target": glyph_histogram(canonical, "target"),
        }
        report["splits"][name] = {
            "input": {"path": str(src), "rows": len(rows), "sha256": sha256_file(src)},
            "rows_edited": edited,
            "rows_augmented": augmented,
            "apostrophe_glyphs_before": before,
            "apostrophe_glyphs_after": after,
            "output": write_rows(output_dir / f"{name}.parquet", canonical),
        }

    # The English held-out test set is copied through canonicalization only so
    # the two-domain report stays comparable; it is never trained on.
    english_test = base_dir / "english-test.parquet"
    if english_test.exists():
        rows = read_parquet_rows(english_test)
        canonical, edited, _ = canonicalize_rows(rows, augment=False)
        report["english_test"] = {
            "input": {"path": str(english_test), "rows": len(rows), "sha256": sha256_file(english_test)},
            "rows_edited": edited,
            "output": write_rows(output_dir / "english-test.parquet", canonical),
        }

    write_json(output_dir / "manifest.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=ROOT / "data/e28_english_transfer")
    parser.add_argument("--output", type=Path, default=ROOT / "data/e30_canonical_orthography")
    parser.add_argument("--no-augment", action="store_true", help="skip both-glyph source augmentation")
    args = parser.parse_args()
    report = build(args.base_dir, args.output, augment=not args.no_augment)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
