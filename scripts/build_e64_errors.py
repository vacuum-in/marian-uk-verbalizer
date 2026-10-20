#!/usr/bin/env python3
"""E64: span rows for the error types E62 makes most on real book text (offline data generation only).

The five-judge review of the real-text references (real_v2) showed E62 reading, against the references:

  * "3 мене", "— 3 тих пір": an OCR'd capital З read as the number three -> "З мене";
  * "(1797)", "(1873–1945)": a year or life span in parentheses dropped or read as an ordinal -> cardinals,
    "(тисяча вісімсот сімдесят три — тисяча дев'ятсот сорок п'ять)"; "(624 — 546 до н. е.)" -> "до нашої ери";
  * "18—21 серпня", "29 й 30 квітня": day ranges and pairs -> "з вісімнадцятого по двадцять перше серпня",
    "двадцять дев'ятого й тридцятого квітня";
  * "[6]", "[52] І": footnote numbers read aloud -> left out.

Each row is built by inserting (or swapping in) the pattern into a sentence from a training-split book, finding the
span the shape detector makes of it, and writing the span's reading. Rows whose span holds other digits are skipped.

    build_e64_errors.py --out data/e64_spans/errors.parquet [--per-kind 5000]
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
from scripts.roman_case import ordinal_form  # noqa: E402
from src.verbalizer.data.books import book_id_of, iter_book_sentences  # noqa: E402
from src.verbalizer.spans import detect_spans, model_input  # noqa: E402

MONTHS = ["січня", "лютого", "березня", "квітня", "травня", "червня", "липня", "серпня", "вересня", "жовтня",
          "листопада", "грудня"]
DAY = re.compile(r"\b([1-9]|[12]\d|3[01]) (" + "|".join(MONTHS) + r")\b(?! \d)")
NO_DIGITS = re.compile(r"^[^\d]*$")
CAP_WORD = re.compile(r"(?<=[a-zа-яіїєґ'’] )[А-ЯІЇЄҐ][а-яіїєґ'’]{3,}(?: [А-ЯІЇЄҐ][а-яіїєґ'’]{3,})?(?=[ ,])")
WORD_END = re.compile(r"[а-яіїєґ'’]{4,}(?=[ ,.;:])")


def cardinal(n: int) -> str:
    return re.sub(r"^одна тисяча", "тисяча", num2words(n, lang="uk"))


def ordinal(n: int, gender: str = "masc", case: str = "gent") -> str | None:
    w = ordinal_form(n, gender, case)
    return re.sub(r"^одна тисяча", "тисяча", w) if w else None


def sentences(books_dir: Path, holdout: set[str], want: int, rng: random.Random) -> list[str]:
    out = []
    for path in sorted(books_dir.glob("book-*.json")):
        if book_id_of(path) in holdout:
            continue
        for bs in iter_book_sentences(path):
            t = bs.text.replace(" ", " ").strip()
            if 5 <= len(t.split()) <= 35 and rng.random() < 0.2:
                out.append(t)
        if len(out) >= want:
            break
    rng.shuffle(out)
    return out


def row(line: str, start: int, end: int, reading: str, kind: str) -> dict | None:
    """The span covering line[start:end]; its target is the span with that piece replaced by `reading`."""
    for sp in detect_spans(line):
        if sp.start <= start and end <= sp.start + len(sp.text):
            a, b = start - sp.start, end - sp.start
            rest = sp.text[:a] + sp.text[b:]
            if not NO_DIGITS.match(rest):
                return None
            target = re.sub(r"\s+", " ", sp.text[:a] + reading + sp.text[b:]).strip()
            target = re.sub(r"\s+([,.;:!?)»])", r"\1", target)
            return {"id": f"e64-{kind}-{abs(hash((line, start))) % 10**12}", "source": model_input(line, sp),
                    "target": target, "span": sp.text, "sentence_id": kind}
    return None


def ocr3(t: str, rng: random.Random):
    m = re.search(r"(^|— |– |« )З (?=[а-яіїєґ])", t)
    if not m:
        return None
    i = m.end() - 2
    return row(t[:i] + "3" + t[i + 1:], i, i + 1, "З", "ocr3")


def paren_year(t: str, rng: random.Random):
    m = rng.choice(list(CAP_WORD.finditer(t)) or [None])
    if not m:
        return None
    r = rng.random()
    if r < 0.35:
        y = rng.randint(1500, 2025)
        ins, reading = f"{y}", cardinal(y)
    elif r < 0.9:
        a = rng.randint(1000, 1990)
        b = a + rng.randint(25, 95)
        dash = rng.choice(["–", "–", "—", "-", " — ", " – "])
        sep = " — " if rng.random() < 0.8 else "–"
        ins, reading = f"{a}{dash}{b}", f"{cardinal(a)}{sep}{cardinal(b)}"
    else:
        a = rng.randint(120, 799)
        b = a - rng.randint(20, 90)
        ins = f"{a} — {b} {rng.choice(['до н. е.', 'до н.е.'])}"
        reading = f"{cardinal(a)} — {cardinal(b)} до нашої ери"
    i = m.end()
    line = t[:i] + " (" + ins + ")" + t[i:]
    return row(line, i + 2, i + 2 + len(ins), reading, "paren-year")


def day_range(t: str, rng: random.Random):
    is_range = rng.random() < 0.65
    m = DAY.search(t)
    if m and is_range and re.search(r"\b(з|із|від|до|по)\s*$", t[:m.start()]):
        return None  # "з 18—21 серпня" would read "з з вісімнадцятого"
    if not m:  # no date in the sentence: put one after a word ("... до 4—6 вересня ...")
        w = rng.choice(list(WORD_END.finditer(t)) or [None])
        if not w:
            return None
        t = t[:w.end()] + f" {rng.choice([',', '', '—'] if is_range else ['до', 'на', 'з', 'від', 'у', ''])} 1 {rng.choice(MONTHS)}".replace("  ", " ").replace(" ,", ",") + t[w.end():]
        m = DAY.search(t)
        if not m:
            return None
    month = m.group(2)
    a = rng.randint(1, 27)
    b = min(31 if month != "лютого" else 28, a + rng.randint(1, 4))
    if is_range:
        ins = f"{a}{rng.choice(['—', '–', '-', ' — '])}{b}"
        ga, nb = ordinal(a), ordinal(b, "neut", "nomn")
        if not (ga and nb):
            return None
        reading = f"з {ga} по {nb}"
    else:
        conj = rng.choice(["й", "і", "та"])
        ins = f"{a} {conj} {b}"
        ga, gb = ordinal(a), ordinal(b)
        if not (ga and gb):
            return None
        reading = f"{ga} {conj} {gb}"
    line = t[:m.start(1)] + ins + t[m.end(1):]
    return row(line, m.start(1), m.start(1) + len(ins), reading, "day-range")


def bc_year(t: str, rng: random.Random):
    m = rng.choice(list(WORD_END.finditer(t)) or [None])
    if not m:
        return None
    y = rng.randint(30, 800)
    g = ordinal(y)
    if not g:
        return None
    ins = f"{y} {rng.choice(['р.', 'р.', 'року'])} {rng.choice(['до н. е.', 'до н.е.'])}"
    i = m.end()
    line = t[:i] + " у " + ins + t[i:]
    return row(line, i + 3, i + 3 + len(ins), f"{g} року до нашої ери", "bc-year")


def footnote(t: str, rng: random.Random):
    ms = list(WORD_END.finditer(t))
    if not ms:
        return None
    m = rng.choice(ms)
    ins = f"{rng.choice(['', ' '])}[{rng.randint(1, 400)}]"
    line = t[:m.end()] + ins + t[m.end():]
    return row(line, m.end(), m.end() + len(ins), "", "footnote")


KINDS = {"ocr3": ocr3, "paren-year": paren_year, "day-range": day_range, "bc-year": bc_year, "footnote": footnote}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--books-dir", type=Path, default=Path("/home/devops/ukr-books/books"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--per-kind", type=int, default=5000)
    ap.add_argument("--show", type=int, default=4)
    args = ap.parse_args()
    rng = random.Random(64)
    holdout = set(json.loads((ROOT / "data/book_domain/split.json").read_text(encoding="utf-8"))["holdout"])
    texts = sentences(args.books_dir, holdout, 60 * args.per_kind, rng)
    rows, counts = [], {}
    for kind, make in KINDS.items():
        want = args.per_kind // 2 if kind == "bc-year" else args.per_kind
        got = []
        for t in texts:
            if len(got) >= want:
                break
            r = make(t, rng)
            if r:
                got.append(r)
        counts[kind] = len(got)
        rows += got
        for r in got[: args.show]:
            print(f"  [{kind}] {r['source'][:110]}\n      -> {r['target']!r}")
    rng.shuffle(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), args.out)
    print(len(rows), counts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
