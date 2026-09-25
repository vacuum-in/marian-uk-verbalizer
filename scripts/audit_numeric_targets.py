#!/usr/bin/env python3
"""Audit numeric verbalization in a dataset by reading targets back to integers.

Offline QA only. This never runs in the serving pipeline and produces no spoken
form: it parses targets that already exist and reports where the value in the
target disagrees with the value in the source.

Rows read digit-wise by design (phones, cards, plates, codes), decimals, ranges,
times/dates and abbreviated scales are bucketed separately rather than checked,
because their targets legitimately do not encode a single cardinal value.
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import re
import sys
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.verbalizer._audit_numparse import parse_numbers  # noqa: E402

GROUP = re.compile(r"(?<![\d.,])(\d{1,3}(?:[   ]\d{3})+)(?![\d,])")
DIGITWISE = re.compile(
    r"\+\d|№|@|IBAN|CVV|картк|[Тт]елефон|[Кк]од |[Пп]ередзвони|[Рр]ахунок|[Пп]ароль|"
    r"(?<!\d)0[\s\d]|[A-ZА-ЯІЇЄ]{2}\s?\d{3,4}\s?[A-ZА-ЯІЇЄ]{2}|\d{4}\s\d{4}|[A-Za-z]\s*\d+\s*[A-Za-z]"
)
DECIMAL = re.compile(r"\d[.,]\d")
RANGE = re.compile(r"\d\s*[-–—]\s*\d")
TIMEDATE = re.compile(r"\d{1,2}[:.]\d{2}|\d{1,2}\.\d{1,2}\.\d{2,4}")
MULT = re.compile(r"\b(млн|млрд|тис|трлн)\b\.?|\d\s+(тисяч|мільйон|мільярд)")
PLAIN = re.compile(r"(?<![\d.,:/\-])(\d{1,15})(?![\d.,:/\-])")


def normalize(text: str) -> str:
    return GROUP.sub(lambda m: re.sub(r"[   ]", "", m.group(1)), text)


def bucket(source: str) -> str:
    if DIGITWISE.search(source):
        return "digit-wise"
    if TIMEDATE.search(source):
        return "time/date"
    if DECIMAL.search(source):
        return "decimal"
    if RANGE.search(source):
        return "range"
    if MULT.search(source):
        return "abbrev-scale"
    return "plain"


def audit(rows, sample_size, seed):
    random.seed(seed)
    if sample_size and sample_size < len(rows):
        rows = random.sample(rows, sample_size)
    counts = collections.Counter()
    mismatches = []
    for source, target in rows:
        kind = bucket(source)
        counts[kind] += 1
        if kind != "plain":
            continue
        values = [int(v) for v in PLAIN.findall(normalize(source))]
        if not values:
            counts["plain:no-int"] += 1
            continue
        parsed = parse_numbers(target)
        if [v for v in values if v not in parsed]:
            counts["plain:mismatch"] += 1
            mismatches.append({"source": source, "target": target, "source_values": values, "target_parsed": parsed})
        else:
            counts["plain:ok"] += 1
    return counts, mismatches


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("--sample", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.data.suffix == ".jsonl":
        recs = [json.loads(l) for l in args.data.read_text(encoding="utf-8").splitlines() if l.strip()]
        pairs = [(r["source"], r["target"]) for r in recs]
    else:
        table = pq.read_table(args.data, columns=["source", "target"])
        pairs = list(zip(table.column("source").to_pylist(), table.column("target").to_pylist()))
    pairs = [(a, b) for a, b in pairs if re.search(r"\d", a)]

    counts, mismatches = audit(pairs, args.sample, args.seed)
    checked = counts["plain:ok"] + counts["plain:mismatch"]
    report = {
        "data": str(args.data),
        "rows_with_digits": len(pairs),
        "sampled": min(args.sample, len(pairs)) if args.sample else len(pairs),
        "buckets": {k: counts[k] for k in ("plain", "digit-wise", "decimal", "time/date", "range", "abbrev-scale")},
        "plain_checked": checked,
        "plain_ok": counts["plain:ok"],
        "plain_mismatch": counts["plain:mismatch"],
        "value_accuracy": round(counts["plain:ok"] / checked, 4) if checked else None,
        "mismatches": mismatches,
    }
    if args.output:
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {k: v for k, v in report.items() if k != "mismatches"}
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
