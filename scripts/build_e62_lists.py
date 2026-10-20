#!/usr/bin/env python3
"""E62: numbered lists, Roman section headings and words written in capitals (offline data generation only).

On an order document E60 read list numbers at the start of a line at random ("7. користуватися" -> "Сент",
"9." -> "дев'ятого"), Roman section numbers in any case ("I." -> "Перша", "V." -> "п'ятого"), and a word written in
capitals as an acronym ("НАКАЗ" -> "НакАЗ"). Conventions taught here:

  * "N. text" / "N) text" at the start of a line: the cardinal, "7." -> "Сім";
  * "IV. Heading": the masculine ordinal (a section), "IV." -> "Четвертий";
  * an ordinary word in capitals ("НАКАЗ", "ІНСТРУКЦІЯ"): read as itself.

Item text comes from training-split books only. Output: span rows (source with [[span]], target).

    build_e62_lists.py --out data/e62_spans/lists.parquet
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from num2words import num2words

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.roman_case import roman  # noqa: E402
from src.verbalizer.data.books import book_id_of, iter_book_sentences  # noqa: E402
from src.verbalizer.spans import detect_spans, model_input  # noqa: E402

PLAIN = re.compile(r"^[А-Яа-яІіЇїЄєҐґ'’ʼ ,;:\-–—()«»]+[.;:]?$")


def sentences(books_dir: Path, holdout: set[str], want: int, rng: random.Random) -> list[str]:
    out = []
    for path in sorted(books_dir.glob("book-*.json")):
        if book_id_of(path) in holdout:
            continue
        for bs in iter_book_sentences(path):
            t = bs.text.replace(" ", " ").strip()
            if t[:1] in "—–-" or not PLAIN.match(t) or not (4 <= len(t.split()) <= 28):
                continue
            if rng.random() < 0.15:
                out.append(t)
        if len(out) >= want:
            break
    rng.shuffle(out)
    return out


def row(line: str, span_text: str, target: str, kind: str):
    for sp in detect_spans(line):
        if sp.text == span_text and sp.start == line.index(span_text):
            return {"id": f"e62-{kind}-{abs(hash(line)) % 10**12}", "source": model_input(line, sp), "target": target,
                    "span": sp.text, "sentence_id": kind}
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--books-dir", type=Path, default=Path("/home/devops/ukr-books/books"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--lists", type=int, default=24000)
    ap.add_argument("--romans", type=int, default=6000)
    ap.add_argument("--caps", type=int, default=8000)
    args = ap.parse_args()
    rng = random.Random(62)
    holdout = set(json.loads((ROOT / "data/book_domain/split.json").read_text(encoding="utf-8"))["holdout"])
    texts = sentences(args.books_dir, holdout, args.lists + args.romans + args.caps + 5000, rng)
    rows, kinds, it = [], {"list": 0, "roman": 0, "caps": 0}, iter(texts)

    def take():
        return next(it)

    # numbered list items
    while kinds["list"] < args.lists:
        t = take()
        n = rng.choice(list(range(1, 13)) * 4 + list(range(13, 41)))
        body = t[0].lower() + t[1:] if rng.random() < 0.6 else t
        body = re.sub(r"[.;:]?$", rng.choice([";", ".", ".", ":", ""]), body, count=1)
        line = f"{n}{rng.choice(['.', '.', '.', ')'])} {body}"
        r = row(line, str(n), num2words(n, lang="uk").capitalize(), "list")
        if r:
            rows.append(r)
            kinds["list"] += 1
    # Roman section headings
    while kinds["roman"] < args.romans:
        t = take()
        words = t.rstrip(".;:").split()
        head = " ".join(words[: rng.randint(2, min(9, len(words)))])
        head = head[0].upper() + head[1:]
        n = rng.choice(list(range(1, 11)) * 3 + list(range(11, 21)))
        line = f"{roman(n)}. {head}"
        r = row(line, roman(n), num2words(n, lang="uk", to="ordinal").capitalize(), "roman")
        if r:
            rows.append(r)
            kinds["roman"] += 1
    # an ordinary word in capitals, read as itself
    while kinds["caps"] < args.caps:
        t = take()
        words = [w for w in re.findall(r"[А-Яа-яІіЇїЄєҐґ'’]{4,}", t) if not w[0].isupper() or len(w) > 4]
        if not words:
            continue
        w = rng.choice(words).upper()
        rest = t.rstrip(".;:")
        line = f"{w} {rest[0].lower() + rest[1:]}" if rng.random() < 0.5 else f"{w}"
        r = row(line, w, w, "caps")
        if r:
            rows.append(r)
            kinds["caps"] += 1
    rng.shuffle(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), args.out)
    print(len(rows), kinds)
    for r in rows[:9]:
        print("  ", r["source"][:90], "->", r["target"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
