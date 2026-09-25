#!/usr/bin/env python3
"""E46: number pairs joined by a dash that are not quantity ranges.

E45 reads book ranges at 50%. The misses are lifespans after a name
("Рікард (1907–1940)" loses the opening bracket and the first word of the
year), scores ("з рахунками 14-1, 24-3" read as "від ... до ..."), and article
numbers ("статті 122-1"). Adds, on top of data/e45_english_ranges:

  * lifespans after names mined from the training-split books;
  * sports scores, alone and in lists;
  * hyphenated article/section numbers.

Each pair is read as two cardinals joined by a dash ("чотирнадцять — один").
Readings come from num2words and are cross-checked chunk by chunk with the
independent parser -- TRAINING DATA ONLY.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from build_e41_domain_growth import read_rows, words  # noqa: E402
from src.verbalizer._audit_numparse import parse_numbers  # noqa: E402
from src.verbalizer.data.books import iter_book_sentences  # noqa: E402
from src.verbalizer.training.manifest import sha256_file, write_json  # noqa: E402

NAME_RE = re.compile(r"(?<=[а-яіїєґ,] )[А-ЯІЇЄҐ][а-яіїєґ'’]{2,11}(?=[ ,.])")
LIFESPAN_FRAMES = ["{n} ({a}{d}{b})", "{n} ({a}{d}{b}) — {rel}.", "{rel} {n} ({a}{d}{b})",
                   "{N} {a}{d}{b}", "Портрет: {n} ({a}{d}{b})."]
RELATIONS = ["дружина", "чоловік", "син", "донька", "брат", "сестра", "дід", "онука"]
SCORE_FRAMES = ["Матч завершився з рахунком {s}.", "Це були розгроми з рахунками {s}.", "Рахунок {s} нікого не здивував.",
                "Фінал: {s}."]
ARTICLE_FRAMES = ["Статті {s} стосуються цієї справи.", "Дивіться пункт {s}.", "Згідно з параграфом {s}, це заборонено.",
                  "Розділ {s} доповнено."]


def pair_reading(a: int, b: int) -> str:
    return f"{words(a)} — {words(b)}"


def pair_ok(a: int, b: int) -> bool:
    return parse_numbers(words(a)) == [a] and parse_numbers(words(b)) == [b]


def mine_names(books_dir: Path, train_ids: set[str], limit: int = 3000) -> list[str]:
    counts: Counter = Counter()
    for path in sorted(books_dir.glob("book-*.json")):
        if path.stem.removeprefix("book-") not in train_ids:
            continue
        for sentence in iter_book_sentences(path):
            counts.update(NAME_RE.findall(sentence.text))
        if len(counts) > 50000:
            break
    return [name for name, n in counts.most_common(limit) if n >= 3]


def build(args: argparse.Namespace) -> dict:
    rng = random.Random(args.seed)
    split = json.loads(args.split.read_text(encoding="utf-8"))
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
            added.append({"id": f"e46-{kind}:{len(added)}", "source": source, "target": target,
                          "categories": [category, f"e46_{kind}"], "provenance": "generated_e46_value_crosschecked"})

    names = mine_names(args.books_dir, set(split["train"]))
    stats["names_mined"] = len(names)
    for _ in range(args.lifespans):
        a = rng.randint(1500, 2005)
        b = a + rng.randint(1, 95)
        if b > 2099 or not pair_ok(a, b):
            continue
        name = rng.choice(names)
        if rng.random() < 0.4:
            name = f"{name} {rng.choice(names)}"
        frame = rng.choice(LIFESPAN_FRAMES)
        fill = {"n": name, "N": name.upper(), "rel": rng.choice(RELATIONS), "d": rng.choice(["–", "—", "-"])}
        source = frame.format(a=a, b=b, **fill)
        target = frame.replace("{a}{d}{b}", "{r}").format(r=pair_reading(a, b), **fill)
        keep("lifespan", source, target, "Діапазони")

    for _ in range(args.scores):
        count = rng.choice([1, 1, 2, 3])
        pairs = [(rng.randint(0, 30), rng.randint(0, 30)) for _ in range(count)]
        sep = rng.choice(["-", ":", "–"])
        text = ", ".join(f"{a}{sep}{b}" for a, b in pairs)
        reading = ", ".join(pair_reading(a, b) for a, b in pairs)
        frame = rng.choice(SCORE_FRAMES)
        keep("score", frame.format(s=text), frame.format(s=reading), "Спорт")

    for _ in range(args.articles):
        a, b = rng.randint(1, 400), rng.randint(1, 9)
        frame = rng.choice(ARTICLE_FRAMES)
        keep("article", frame.format(s=f"{a}-{b}"), frame.format(s=pair_reading(a, b)), "Право/документи")

    rng.shuffle(added)
    args.output.mkdir(parents=True, exist_ok=True)
    combined = [*base, *added]
    pq.write_table(pa.Table.from_pylist(combined), args.output / "train.parquet", compression="zstd")
    pq.write_table(pa.Table.from_pylist(read_rows(args.base_dir / "dev-stratified.parquet")),
                   args.output / "dev-stratified.parquet", compression="zstd")
    by_kind = Counter(r["id"].split(":")[0].removeprefix("e46-") for r in added)
    report = {"base_rows": len(base), "added": len(added), "added_by_kind": dict(by_kind), "total_rows": len(combined),
              "stats": dict(stats), "train_sha256": sha256_file(args.output / "train.parquet"),
              "samples": {k: [{"source": r["source"], "target": r["target"]} for r in added if r["id"].startswith(f"e46-{k}:")][:3]
                          for k in by_kind}}
    write_json(args.output / "manifest.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=ROOT / "data/e45_english_ranges")
    parser.add_argument("--books-dir", type=Path, default=Path("/home/devops/ukr-books/books"))
    parser.add_argument("--split", type=Path, default=ROOT / "data/book_domain/split.json")
    parser.add_argument("--frozen", type=Path, default=ROOT / "reports/hf-uktn-frozen-3597.jsonl")
    parser.add_argument("--eval-dir", type=Path, default=ROOT / "reports/domain-eval")
    parser.add_argument("--book-identity", type=Path, default=ROOT / "reports/book-prose-identity.jsonl")
    parser.add_argument("--book-numeric", type=Path, default=ROOT / "reports/book-numeric.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "data/e46_spans")
    parser.add_argument("--seed", type=int, default=46046)
    parser.add_argument("--lifespans", type=int, default=4000)
    parser.add_argument("--scores", type=int, default=2000)
    parser.add_argument("--articles", type=int, default=1500)
    report = build(parser.parse_args())
    print(json.dumps({k: v for k, v in report.items() if k != "train_sha256"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
