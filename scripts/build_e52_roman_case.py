#!/usr/bin/env python3
"""E52: Roman ordinals agreeing in case and gender with the following noun.

User report (2026-09-25): "Потрібно прочитати XX розділ" -> "двадцятого розділ"
instead of "двадцятий розділ". E51 also gives "дев'ятої главу" and "в третій
акті". Adds, on top of data/e51_khvylyna, rows from scripts/roman_case.py
(num2words ordinal + pymorphy3 agreement, value cross-checked) in training-only
frames covering all cases and genders, I..XXXIX only (user rule).
TRAINING DATA ONLY.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from build_e41_domain_growth import read_rows  # noqa: E402
from roman_case import TRAIN_FRAMES, e54_frames, make_rows, rich_frames  # noqa: E402
from src.verbalizer.training.manifest import sha256_file, write_json  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=ROOT / "data/e51_khvylyna")
    parser.add_argument("--eval-dir", type=Path, default=ROOT / "reports/domain-eval")
    parser.add_argument("--frozen", type=Path, default=ROOT / "reports/hf-uktn-frozen-3597.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "data/e52_roman_case")
    parser.add_argument("--seed", type=int, default=52052)
    parser.add_argument("--rows", type=int, default=8000)
    parser.add_argument("--rich", action="store_true", help="E53: many governors per case, weighted to datv/ablt/gent")
    parser.add_argument("--tag", default="e52")
    parser.add_argument("--rebalance", action="store_true", help="E54: accusative + possessive genitive, light refresher of the rest")
    args = parser.parse_args()

    blocked = {json.loads(l)["source"] for l in args.frozen.read_text(encoding="utf-8").splitlines() if l.strip()}
    for path in args.eval_dir.glob("*.jsonl"):
        blocked |= {json.loads(l)["source"] for l in path.read_text(encoding="utf-8").splitlines() if l.strip()}
    base = read_rows(args.base_dir / "train.parquet")
    seen = {r["source"] for r in base}
    added, stats = [], Counter()
    rng = random.Random(args.seed)
    if args.rebalance:
        frames = e54_frames()
        quotas = {"accs": 4000, "gent": 3000, "datv": 800, "ablt": 800, "loct": 400, "nomn": 400}
        generated = [row for case, count in quotas.items() for row in make_rows(rng, {case: frames[case]}, count)]
        rng.shuffle(generated)
    elif args.rich:
        frames = rich_frames()
        quotas = {"datv": 4000, "ablt": 3500, "gent": 3500, "accs": 2000, "loct": 1000, "nomn": 500}
        generated = [row for case, count in quotas.items() for row in make_rows(rng, {case: frames[case]}, count)]
        rng.shuffle(generated)
    else:
        generated = make_rows(rng, TRAIN_FRAMES, args.rows)
    for row in generated:
        if row["source"] in blocked or row["source"] in seen:
            stats["skipped"] += 1
            continue
        seen.add(row["source"])
        stats[row["case"]] += 1
        added.append({"id": f"{args.tag}-roman-case:{len(added)}", "source": row["source"], "target": row["target"],
                      "categories": ["Римські: узгодження з іменником", f"{args.tag}_roman_case"],
                      "provenance": "generated_e52_num2words_pymorphy3_crosschecked"})
    args.output.mkdir(parents=True, exist_ok=True)
    combined = [*base, *added]
    pq.write_table(pa.Table.from_pylist(combined), args.output / "train.parquet", compression="zstd")
    pq.write_table(pa.Table.from_pylist(read_rows(args.base_dir / "dev-stratified.parquet")),
                   args.output / "dev-stratified.parquet", compression="zstd")
    report = {"base_rows": len(base), "added": len(added), "total_rows": len(combined), "stats": dict(stats),
              "train_sha256": sha256_file(args.output / "train.parquet"),
              "samples": [{"source": r["source"], "target": r["target"]} for r in added[:8]]}
    write_json(args.output / "manifest.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "train_sha256"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
