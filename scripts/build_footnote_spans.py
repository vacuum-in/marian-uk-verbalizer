#!/usr/bin/env python3
"""Span rows teaching that a footnote mark is not read: "[93]", "[*]", a digit glued to a word ("референдарій9"),
superscript digits. Sentences come from the training-split books only (data/book_domain/split.json). Offline data
construction; the target of every row is the empty string.

    build_footnote_spans.py --out data/e56_spans/footnotes.parquet [--max 30000]
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.verbalizer.data.books import book_id_of, iter_book_sentences  # noqa: E402
from src.verbalizer.spans import GLUED_NOTE, detect_spans, model_input  # noqa: E402

NOTE = re.compile(r"\[\s*(?:\d{1,4}|\*+)\s*\]|[¹²³⁰⁴⁵⁶⁷⁸⁹]+")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--books-dir", type=Path, default=Path("/home/devops/ukr-books/books"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max", type=int, default=30000)
    ap.add_argument("--synthetic", type=int, default=24000)
    args = ap.parse_args()
    holdout = set(json.loads((ROOT / "data/book_domain/split.json").read_text(encoding="utf-8"))["holdout"])
    rng = random.Random(56)
    rows, kinds = [], {"bracket": 0, "glued": 0, "superscript": 0}
    for path in sorted(args.books_dir.glob("book-*.json")):
        if book_id_of(path) in holdout:
            continue
        for bs in iter_book_sentences(path):
            s = bs.text
            if not (12 <= len(s) <= 300) or not (NOTE.search(s) or GLUED_NOTE.search(s)):
                continue
            for sp in detect_spans(s):
                if NOTE.fullmatch(sp.text):
                    kind = "superscript" if sp.text[0] in "¹²³⁰⁴⁵⁶⁷⁸⁹" else "bracket"
                elif sp.text.isdigit() and sp.start > 0 and s[sp.start - 1].isalpha():
                    kind = "glued"
                else:
                    continue
                kinds[kind] += 1
                rows.append({"id": f"fn-{len(rows)}", "source": model_input(s, sp), "target": "", "span": sp.text,
                             "sentence_id": f"book-{bs.book_id}"})
    # Real marks are rare in these books, so most rows are made: a mark of each shape put after a random word of a
    # sentence that has no digits ("слово[12],", "слово [12]", "слово9", "слово¹²").
    sup = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")
    made = 0
    for path in sorted(args.books_dir.glob("book-*.json")):
        if made >= args.synthetic or book_id_of(path) in holdout:
            continue
        for bs in iter_book_sentences(path):
            s = bs.text
            if made >= args.synthetic or not (20 <= len(s) <= 250) or re.search(r"\d|\[", s) or rng.random() > 0.05:
                continue
            words = [m for m in re.finditer(r"[А-Яа-яІіЇїЄєҐґ'’]{4,}", s)]
            if not words:
                continue
            w = rng.choice(words)
            n = str(rng.choice([rng.randint(1, 9), rng.randint(10, 99), rng.randint(100, 999)]))
            kind = rng.choice(["bracket", "bracket", "spaced", "glued", "superscript"])
            mark = {"bracket": f"[{n}]", "spaced": f" [{n}]", "glued": n[:2], "superscript": n.translate(sup)}[kind]
            t = s[: w.end()] + mark + s[w.end():]
            for sp in detect_spans(t):
                if sp.start <= w.end() + (1 if kind == "spaced" else 0) < sp.end or sp.start == w.end() + (1 if kind == "spaced" else 0):
                    if sp.text.strip() != mark.strip():
                        break
                    rows.append({"id": f"fs-{made}", "source": model_input(t, sp), "target": "", "span": sp.text,
                                 "sentence_id": f"book-{bs.book_id}"})
                    kinds[kind] = kinds.get(kind, 0) + 1
                    made += 1
                    break
    rng.shuffle(rows)
    rows = rows[: args.max]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), args.out)
    print(f"{len(rows)} rows written; found {kinds}")
    for r in rows[:6]:
        print("  ", r["source"][:120])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
