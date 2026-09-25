#!/usr/bin/env python3
"""Build the E32 corpus: E31 plus broad-vocabulary transliteration supervision.

E31 reached 66.33% on the frozen benchmark, but the error is now lopsided:
Latin-bearing rows score 47.78% against 78.39% for everything else, and 51.9%
of all misses are a single mistransliterated word.

The existing English corpus is name-heavy (17,837 person + 2,751 place against
19,643 ordinary words), while the benchmark needs common vocabulary.  The
sibling repository's bronze shards hold exactly that: OpenSubtitles-sourced
words rendered by two independent providers.  Only pairs where both providers
agree are used -- disagreement marks a genuinely ambiguous reading, and training
on one arbitrary side of it would teach a coin flip.

Latin-heavy categories are additionally oversampled.  They already have training
rows; what they lacked was weight inside a 265k corpus.

No reading is authored here.  Every target comes from reviewed or
provider-agreed data, carried through literally.
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import re
import sys
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.verbalizer.training.manifest import sha256_file, write_json  # noqa: E402

CANONICAL_APOSTROPHE = "'"
APOSTROPHE_TABLE = dict.fromkeys(map(ord, "'’‘ʼ´`＇"), CANONICAL_APOSTROPHE)

# Categories whose rows carry Latin script; these are where the residual error
# concentrates, so they are sampled more often.
LATIN_CATEGORY_PATTERN = re.compile(
    r"англійськ|транслітерац|бренд|латиниц|латинськ|домен|email|пошта|хештег|"
    r"url|веб-адрес|шлях|собачка|Пакет V",
    re.IGNORECASE,
)


def canonicalize(text: str) -> str:
    return text.translate(APOSTROPHE_TABLE)


def load_bronze_agreement(bronze_dir: Path, patterns: list[str]) -> tuple[dict[str, str], dict[str, int]]:
    """Return {word: reading} for words both providers rendered identically."""
    readings: dict[str, dict[str, str]] = collections.defaultdict(dict)
    counts: collections.Counter[str] = collections.Counter()
    for provider, pattern in (("a", patterns[0]), ("b", patterns[1])):
        for path in sorted(glob.glob(str(bronze_dir / pattern))):
            with open(path, encoding="utf-8") as handle:
                for line in handle:
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    parsed = record.get("parsed") or {}
                    if parsed.get("status") != "accepted":
                        continue
                    reading = parsed.get("uk_reading")
                    record_id = record.get("record_id", "")
                    if not reading or ":" not in record_id:
                        continue
                    readings[record_id.split(":", 1)[1]][provider] = reading.strip()
                    counts[provider] += 1

    agreed: dict[str, str] = {}
    for word, by_provider in readings.items():
        if len(by_provider) != 2:
            continue
        first, second = by_provider["a"], by_provider["b"]
        if first.lower() == second.lower():
            agreed[word] = first
    stats = {
        "provider_a_readings": counts["a"],
        "provider_b_readings": counts["b"],
        "distinct_words": len(readings),
        "both_providers": sum(1 for d in readings.values() if len(d) == 2),
        "agreed": len(agreed),
    }
    return agreed, stats


def read_rows(path: Path) -> list[dict[str, Any]]:
    table = pq.read_table(path, columns=["id", "source", "target", "categories", "provenance"])
    return [
        {
            "id": str(row["id"]),
            "source": str(row["source"]),
            "target": str(row["target"]),
            "categories": [str(c) for c in (row.get("categories") or [])],
            "provenance": str(row.get("provenance") or "unknown"),
        }
        for row in table.to_pylist()
    ]


def oversample_latin(rows: list[dict[str, Any]], factor: int) -> tuple[list[dict[str, Any]], int]:
    if factor <= 1:
        return rows, 0
    extra: list[dict[str, Any]] = []
    for row in rows:
        if any(LATIN_CATEGORY_PATTERN.search(c) for c in row["categories"]):
            for copy_index in range(factor - 1):
                extra.append({**row, "id": f"{row['id']}:latin-x{copy_index + 1}"})
    return [*rows, *extra], len(extra)


def write_rows(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")
    return {"path": str(path), "rows": len(rows), "sha256": sha256_file(path)}


def build(args: argparse.Namespace) -> dict[str, Any]:
    frozen_sources = {
        json.loads(line)["source"]
        for line in args.frozen.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    dev_sources = set(
        pq.read_table(args.base_dir / "dev-stratified.parquet", columns=["source"]).column("source").to_pylist()
    )

    base_rows = read_rows(args.base_dir / "train.parquet")
    existing_sources = {row["source"] for row in base_rows}

    agreed, bronze_stats = load_bronze_agreement(
        args.bronze_dir,
        ["v3-words-deepseek-v3.2-shard-*.jsonl", "v3-words-deepseek-v4-pro-shard-*.jsonl"],
    )

    bronze_rows: list[dict[str, Any]] = []
    skipped = collections.Counter()
    for word, reading in sorted(agreed.items()):
        source = canonicalize(word)
        target = canonicalize(reading)
        if not source or not target:
            skipped["empty"] += 1
            continue
        if source in frozen_sources or source in dev_sources:
            skipped["held_out"] += 1
            continue
        if source in existing_sources:
            skipped["already_present"] += 1
            continue
        bronze_rows.append(
            {
                "id": f"bronze-agree:{source}",
                "source": source,
                "target": target,
                "categories": ["english_transducer_transfer", "english_word", "bronze_agreement"],
                "provenance": "deepseek_two_provider_agreement",
            }
        )

    combined, oversampled = oversample_latin([*base_rows, *bronze_rows], args.latin_factor)

    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "base": {
            "path": str(args.base_dir / "train.parquet"),
            "rows": len(base_rows),
            "sha256": sha256_file(args.base_dir / "train.parquet"),
        },
        "bronze": {**bronze_stats, "rows_added": len(bronze_rows), "skipped": dict(skipped)},
        "latin_oversample_factor": args.latin_factor,
        "latin_rows_added": oversampled,
        "output": {"train": write_rows(args.output / "train.parquet", combined)},
    }
    # The dev split is carried through unchanged so E31 and E32 stay comparable.
    dev = read_rows(args.base_dir / "dev-stratified.parquet")
    report["output"]["dev"] = write_rows(args.output / "dev-stratified.parquet", dev)
    write_json(args.output / "manifest.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=ROOT / "data/e31_unified_labelled")
    parser.add_argument(
        "--bronze-dir", type=Path, default=ROOT.parent / "en-uk-transducer/data/bronze"
    )
    parser.add_argument("--frozen", type=Path, default=ROOT / "reports/hf-uktn-frozen-3597.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "data/e32_transliteration_scale")
    parser.add_argument("--latin-factor", type=int, default=3)
    args = parser.parse_args()
    print(json.dumps(build(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
