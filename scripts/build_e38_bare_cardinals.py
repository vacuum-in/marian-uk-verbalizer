#!/usr/bin/env python3
"""Add bare-cardinal supervision for numbers the corpus never shows as quantities.

Bare four-digit numbers in this corpus are almost entirely years (2023, 2022,
...) or digit-wise identifiers (card and phone fragments such as 1234, 5678).
Genuine four-digit cardinals are rare, so the model learned "bare 4-digit =>
year or digit string" and misread 1001 and 5001 as magnitudes.

E37 tried to close this by stripping thousands separators from existing sources
and keeping their reviewed targets. That is the cleanest possible fix, but it
only exposed 1,024 bare four-digit numbers, because four-digit values are
rarely written with a separator in the first place. It fixed 3001 and nothing
else.

This builder therefore generates the missing pairs with num2words, which is a
declared dependency of this project and is used here only to construct TRAINING
DATA -- never at inference. Two safeguards keep it honest:

  * only nominative frames are emitted, where the citation form num2words
    returns is the correct reading and no case agreement is at stake;
  * every generated pair is cross-checked with src/verbalizer/_audit_numparse,
    an independently written parser, and dropped unless both agree on the value.

Frames mirror bare-number patterns already present in the corpus.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from num2words import num2words

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.verbalizer._audit_numparse import parse_numbers  # noqa: E402
from src.verbalizer.training.manifest import sha256_file, write_json  # noqa: E402

FRAMES = [("{n}", "{w}"), ("Показник: {n}.", "Показник: {w}.")]


def read_rows(path: Path) -> list[dict[str, Any]]:
    table = pq.read_table(path, columns=["id", "source", "target", "categories", "provenance"])
    return [
        {
            "id": str(r["id"]),
            "source": str(r["source"]),
            "target": str(r["target"]),
            "categories": [str(c) for c in (r.get("categories") or [])],
            "provenance": str(r.get("provenance") or "unknown"),
        }
        for r in table.to_pylist()
    ]


def write_rows(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")
    return {"path": str(path), "rows": len(rows), "sha256": sha256_file(path)}


def candidate_values(count: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    vals: set[int] = set()
    # every X00Y and X0YZ shape -- the patterns that actually fail
    for a in range(1, 10):
        for c in range(0, 10):
            for d in range(0, 10):
                vals.add(a * 1000 + c * 10 + d)      # X0YZ
                vals.add(a * 1000 + c * 100 + d)     # XY0Z
    while len(vals) < count:
        vals.add(rng.randint(1000, 99999))
    return sorted(vals)


def build(args: argparse.Namespace) -> dict[str, Any]:
    frozen = {
        json.loads(l)["source"]
        for l in args.frozen.read_text(encoding="utf-8").splitlines()
        if l.strip()
    }
    dev = set(
        pq.read_table(args.base_dir / "dev-stratified.parquet", columns=["source"]).column("source").to_pylist()
    )
    base = read_rows(args.base_dir / "train.parquet")
    seen = {r["source"] for r in base}

    added: list[dict[str, Any]] = []
    stats = {"considered": 0, "cross_check_failed": 0, "frozen": 0, "dev": 0, "duplicate": 0}
    for n in candidate_values(args.count, args.seed):
        if 1900 <= n <= 2100:            # leave years to the year distribution
            continue
        word = num2words(n, lang="uk")
        if parse_numbers(word) != [n]:   # independent parser must agree
            stats["cross_check_failed"] += 1
            continue
        for src_f, tgt_f in FRAMES:
            stats["considered"] += 1
            source, target = src_f.format(n=n), tgt_f.format(w=word)
            if source in frozen:
                stats["frozen"] += 1
            elif source in dev:
                stats["dev"] += 1
            elif source in seen:
                stats["duplicate"] += 1
            else:
                seen.add(source)
                added.append(
                    {
                        "id": f"bare-cardinal:{n}:{src_f}",
                        "source": source,
                        "target": target,
                        "categories": ["Голий фрагмент: число", "bare_cardinal"],
                        "provenance": "generated_num2words_nominative_crosschecked",
                    }
                )

    combined = [*base, *added]
    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "base": {
            "path": str(args.base_dir / "train.parquet"),
            "rows": len(base),
            "sha256": sha256_file(args.base_dir / "train.parquet"),
        },
        "generation": {**stats, "added": len(added)},
        "output": {
            "train": write_rows(args.output / "train.parquet", combined),
            "dev": write_rows(
                args.output / "dev-stratified.parquet", read_rows(args.base_dir / "dev-stratified.parquet")
            ),
        },
    }
    write_json(args.output / "manifest.json", report)
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-dir", type=Path, default=ROOT / "data/e37_numeric_fixes")
    p.add_argument("--frozen", type=Path, default=ROOT / "reports/hf-uktn-frozen-3597.jsonl")
    p.add_argument("--output", type=Path, default=ROOT / "data/e38_bare_cardinals")
    p.add_argument("--count", type=int, default=6000)
    p.add_argument("--seed", type=int, default=3801)
    a = p.parse_args()
    print(json.dumps(build(a), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
