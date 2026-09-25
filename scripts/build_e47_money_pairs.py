#!/usr/bin/env python3
"""E47: value errors E46 still makes on book text.

E46 reads "3.50$" as "три тисячі п'ятсот доларів", merges "2456-2781" into
"два мільярди ...", and turns "15-20 центів" into "п'ятнадцяте центів". Adds,
on top of data/e46_spans:

  * money with dot decimals and currency signs ("$61.50", "3.50$", "€4.99"),
    plus English-style thousands ("$1,200");
  * dash-joined pairs of 4-digit numbers ("серія 2456-2781");
  * quantity ranges followed by a currency word ("15-20 центів").

Readings come from num2words (+ the E45 genitive helper) and are cross-checked
by value with src/verbalizer/_audit_numparse -- TRAINING DATA ONLY.
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

from build_e41_domain_growth import plural, read_rows, words  # noqa: E402
from build_e45_english_ranges import genitive_number  # noqa: E402
from src.verbalizer._audit_numparse import parse_numbers  # noqa: E402
from src.verbalizer.training.manifest import sha256_file, write_json  # noqa: E402

CURRENCIES = {
    "$": (("долар", "долари", "доларів"), ("цент", "центи", "центів"), False),
    "€": (("євро", "євро", "євро"), ("цент", "центи", "центів"), False),
    "£": (("фунт", "фунти", "фунтів"), ("пенс", "пенси", "пенсів"), False),
}
MONEY_FRAMES = ["Квиток коштував {m}.", "Вона заплатила {m} за вечерю.", "Кімната коштувала {m} на тиждень.",
                "Це обійшлося в {m}, не більше.", "Ціна — {m}.", "Кілька примірників по {m} за штуку."]
PAIR_FRAMES = ["Він обрав проєкт серії {p}.", "Номери {p} вже зайняті.", "Партія {p} пройшла перевірку.",
               "Квитки {p} продано."]
CURRENCY_RANGE_FRAMES = ["Ціна становила {r}.", "Обід коштує {r}.", "За це просили {r}."]
CURRENCY_WORDS = ["центів", "гривень", "доларів", "копійок", "євро"]


def money_reading(sign: str, whole: int, cents: int) -> str:
    major, minor, _ = CURRENCIES[sign]
    parts = [f"{words(whole)} {plural(whole, major)}"]
    if cents:
        feminine = False
        parts.append(f"{words(cents, feminine)} {plural(cents, minor)}")
    return " ".join(parts)


def build(args: argparse.Namespace) -> dict:
    rng = random.Random(args.seed)
    blocked = {json.loads(l)["source"] for l in args.frozen.read_text(encoding="utf-8").splitlines() if l.strip()}
    for path in [*args.eval_dir.glob("*.jsonl"), args.book_identity, args.book_numeric]:
        blocked |= {json.loads(l)["source"] for l in path.read_text(encoding="utf-8").splitlines() if l.strip()}
    base = read_rows(args.base_dir / "train.parquet")
    seen = {row["source"] for row in base}
    added, stats = [], Counter()

    def keep(kind: str, source: str, target: str, category: str) -> None:
        stats[f"{kind}:considered"] += 1
        if source in blocked:
            stats[f"{kind}:blocked"] += 1
        elif source in seen:
            stats[f"{kind}:duplicate"] += 1
        else:
            seen.add(source)
            added.append({"id": f"e47-{kind}:{len(added)}", "source": source, "target": target,
                          "categories": [category, f"e47_{kind}"], "provenance": "generated_e47_value_crosschecked"})

    for _ in range(args.money):
        sign = rng.choice(["$", "$", "€", "£"])
        whole = rng.choice([rng.randint(1, 99), rng.randint(1, 999), rng.randint(1000, 9999)])
        cents = rng.choice([0, rng.randint(1, 99), rng.choice([50, 99, 25])])
        if whole >= 1000:
            amount = f"{whole:,}" if rng.random() < 0.5 else str(whole)
        else:
            amount = str(whole)
        if cents:
            amount += f".{cents:02d}"
        text = rng.choice([f"{sign}{amount}", f"{amount}{sign}", f"{amount} {sign}"])
        reading = money_reading(sign, whole, cents)
        expected = [whole] + ([cents] if cents else [])
        if parse_numbers(reading) != expected:
            stats["money:check_failed"] += 1
            continue
        frame = rng.choice(MONEY_FRAMES)
        keep("money", frame.format(m=text), frame.format(m=reading), "Гроші/Валюта")

    for _ in range(args.pairs):
        a = rng.randint(1000, 9989)
        b = a + rng.randint(1, 900)
        if b > 9999 or parse_numbers(words(a)) != [a] or parse_numbers(words(b)) != [b]:
            continue
        frame = rng.choice(PAIR_FRAMES)
        text = f"{a}{rng.choice(['-', '–'])}{b}"
        keep("pair", frame.format(p=text), frame.format(p=f"{words(a)} — {words(b)}"), "Діапазони")

    for _ in range(args.currency_ranges):
        a = rng.randint(1, 90)
        b = a + rng.choice([2, 5, 10, rng.randint(1, 30)])
        noun = rng.choice(CURRENCY_WORDS)
        reading = f"від {genitive_number(a)} до {genitive_number(b)} {noun}"
        if parse_numbers(reading) != [a, b]:
            stats["currency_range:check_failed"] += 1
            continue
        frame = rng.choice(CURRENCY_RANGE_FRAMES)
        keep("currency_range", frame.format(r=f"{a}-{b} {noun}"), frame.format(r=reading), "Діапазони")

    rng.shuffle(added)
    args.output.mkdir(parents=True, exist_ok=True)
    combined = [*base, *added]
    pq.write_table(pa.Table.from_pylist(combined), args.output / "train.parquet", compression="zstd")
    pq.write_table(pa.Table.from_pylist(read_rows(args.base_dir / "dev-stratified.parquet")),
                   args.output / "dev-stratified.parquet", compression="zstd")
    by_kind = Counter(r["id"].split(":")[0].removeprefix("e47-") for r in added)
    report = {"base_rows": len(base), "added": len(added), "added_by_kind": dict(by_kind), "total_rows": len(combined),
              "stats": dict(stats), "train_sha256": sha256_file(args.output / "train.parquet"),
              "samples": {k: [{"source": r["source"], "target": r["target"]} for r in added if r["id"].startswith(f"e47-{k}:")][:4]
                          for k in by_kind}}
    write_json(args.output / "manifest.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=ROOT / "data/e46_spans")
    parser.add_argument("--frozen", type=Path, default=ROOT / "reports/hf-uktn-frozen-3597.jsonl")
    parser.add_argument("--eval-dir", type=Path, default=ROOT / "reports/domain-eval")
    parser.add_argument("--book-identity", type=Path, default=ROOT / "reports/book-prose-identity.jsonl")
    parser.add_argument("--book-numeric", type=Path, default=ROOT / "reports/book-numeric.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "data/e47_money_pairs")
    parser.add_argument("--seed", type=int, default=47047)
    parser.add_argument("--money", type=int, default=4000)
    parser.add_argument("--pairs", type=int, default=2000)
    parser.add_argument("--currency-ranges", type=int, default=2000)
    report = build(parser.parse_args())
    print(json.dumps({k: v for k, v in report.items() if k != "train_sha256"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
