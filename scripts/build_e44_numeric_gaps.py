#!/usr/bin/env python3
"""E44: numeric gaps that the book set exposes (plan WS4).

E43 inflates short bare numbers after the long-identifier training:
"[265]" -> "дві тисячі двісті шістдесят п'ять", "1997 року" -> "один мільйон ...".
None of E41-E43 teach footnote markers, runs of separate small numbers, or
years inflected in prose. Adds, on top of data/e43_context_growth:

  * footnote markers [n] and {n}, alone and after a word in book sentences,
    read as the plain number (user decision 2026-09-23: read them aloud);
  * runs of 3-5 separate small numbers ("83 72 91"), each read on its own;
  * years in prose: "1997 року", "у 1782 році", "2008-го", "1984 рік";
  * sentence-final decimals ("0,75." -> "нуль цілих сімдесят п'ять сотих.").

Readings come from num2words plus pymorphy3 inflection of the final ordinal
word -- TRAINING DATA ONLY. Every target is cross-checked by value with the
independent parser in src/verbalizer/_audit_numparse and the domain checker.
Only training-split books are read.
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
import pymorphy3
from num2words import num2words

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from build_domain_eval import make_row  # noqa: E402
from build_e41_domain_growth import read_rows, words  # noqa: E402
from src.verbalizer._audit_numparse import parse_numbers  # noqa: E402
from src.verbalizer.data.books import identity_target, iter_book_sentences, prose_identity_stratum  # noqa: E402
from src.verbalizer.evaluation.domain_check import score_row  # noqa: E402
from src.verbalizer.training.manifest import sha256_file, write_json  # noqa: E402

MORPH = pymorphy3.MorphAnalyzer(lang="uk")
YEAR_FRAMES = {
    "gent": ["Це сталося {y} року.", "Будинок збудували {y} року.", "{y} року вони переїхали до Львова.",
             "Навесні {y} року почалася війна.", "Лист датовано осінню {y} року."],
    "loct": ["У {y} році все змінилося.", "Він народився в {y} році.", "Ще у {y} році про це писали газети.",
             "Про ці події згадували в {y} році."],
    "gent_short": ["Відкинувся {y}-го.", "Того самого року, {y}-го, вона повернулася.", "Ще з {y}-го ми не бачилися."],
    "nomn": ["{y} рік видався важким.", "Настав {y} рік.", "Це був {y} рік."],
}
SEQUENCE_FRAMES = ["Він назвав числа {0}.", "На табличці було написано {0}.", "Комбінація замка — {0}.",
                   "— {0}… — рахувала вона.", "Номери квитків: {0}.", "{0}"]
DECIMAL_FRAMES = ["Коефіцієнт — {0}.", "Середній бал: {0}.", "Індекс виріс до {0}.", "Показник склав {0}."]


def ordinal_form(year: int, case: str) -> str | None:
    """num2words ordinal with the last word inflected; masculine, singular."""

    text = num2words(year, lang="uk", to="ordinal")
    if text.startswith("одна тисяча"):
        text = text.replace("одна тисяча", "тисяча", 1)
    if case == "nomn":
        return text
    head, _, last = text.rpartition(" ")
    parse = MORPH.parse(last)[0]
    forms = [f.word for f in parse.lexeme
             if {case, "masc"} <= set(f.tag.grammemes) and "ADJF" in f.tag.grammemes]
    preferred = [w for w in forms if w.endswith("ому")] if case == "loct" else [w for w in forms if w.endswith("ого")]
    form = (preferred or forms or [None])[0]
    if not form:
        return None
    return f"{head} {form}".strip()


def decimal_reading(whole: int, frac: int, places: int) -> str:
    denominator = {1: ("десята", "десятих"), 2: ("сота", "сотих"), 3: ("тисячна", "тисячних")}[places]
    whole_word = "ціла" if whole % 10 == 1 and whole % 100 != 11 else "цілих"
    frac_word = denominator[0] if frac % 10 == 1 and frac % 100 != 11 else denominator[1]
    return f"{words(whole, True)} {whole_word} {words(frac, True)} {frac_word}"


def footnote_number(rng: random.Random) -> int:
    return rng.choice([rng.randint(1, 30), rng.randint(1, 99), rng.randint(100, 999)])


def build(args: argparse.Namespace) -> dict:
    rng = random.Random(args.seed)
    split = json.loads(args.split.read_text(encoding="utf-8"))
    blocked = {json.loads(l)["source"] for l in args.frozen.read_text(encoding="utf-8").splitlines() if l.strip()}
    blocked |= set(pq.read_table(args.base_dir / "dev-stratified.parquet", columns=["source"]).column("source").to_pylist())
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
            added.append({"id": f"e44-{kind}:{len(added)}", "source": source, "target": target,
                          "categories": [category, f"e44_{kind}"], "provenance": "generated_e44_value_crosschecked"})

    train_ids = set(split["train"])
    sentences = []
    for path in sorted(args.books_dir.glob("book-*.json")):
        if path.stem.removeprefix("book-") in train_ids and len(sentences) < 60000:
            sentences += [s.text for s in iter_book_sentences(path)
                          if 25 <= len(s.text) <= 160 and s.text.endswith(".") and len(s.text.split()) >= 4
                          and prose_identity_stratum(s.text) == "plain"]
    rng.shuffle(sentences)

    # Footnote markers inside sentences and on their own.
    for index in range(args.footnotes):
        n = footnote_number(rng)
        mark = rng.choice(["[{}]", "[{}]", "{{{}}}"]).format(n)
        reading = words(n)
        if parse_numbers(reading) != [n]:
            stats["footnote:check_failed"] += 1
            continue
        if index % 3 == 0:
            source, target = mark, reading
        else:
            tokens = sentences[index % len(sentences)][:-1].split(" ")
            cut = rng.randint(1, len(tokens))
            left, right = " ".join(tokens[:cut]), " ".join(tokens[cut:])
            source = left + mark + (" " + right if right else "") + "."
            target = identity_target(left + " " + reading + (" " + right if right else "") + ".")
        keep("footnote", source, target, "Виноски в дужках")

    # Runs of separate small numbers.
    for _ in range(args.sequences):
        values = [rng.randint(1, 99) for _ in range(rng.randint(3, 5))]
        text = " ".join(str(v) for v in values)
        reading = " ".join(words(v) for v in values)
        template = rng.choice(SEQUENCE_FRAMES)
        slot = {"kind": "digits", "digits": "".join(str(v) for v in values)}
        source_row = make_row("sequence", "", template, [text], [slot])
        target = make_row("sequence", "", template, [reading], [slot])["source"]
        if not score_row(source_row, target)["ok"]:
            stats["sequence:check_failed"] += 1
            continue
        keep("sequence", source_row["source"], target, "Послідовність окремих чисел")

    # Years in prose.
    for _ in range(args.years):
        year = rng.randint(1100, 2099)
        if year % 100 == 0:
            continue
        case_key = rng.choices(list(YEAR_FRAMES), weights=[4, 3, 2, 1])[0]
        case = {"gent": "gent", "gent_short": "gent", "loct": "loct", "nomn": "nomn"}[case_key]
        reading = ordinal_form(year, case)
        if not reading or parse_numbers(reading) != [year]:
            stats["year:check_failed"] += 1
            continue
        template = rng.choice(YEAR_FRAMES[case_key])
        if template.count("{y}") != 1:
            continue
        source = template.replace("{y}", str(year))
        target = template.replace("{y}-го", reading).replace("{y}", reading)
        if source[0].isdigit():
            target = target[0].upper() + target[1:]
        keep("year", source, target, "Дати")

    # Sentence-final decimals.
    for _ in range(args.decimals):
        places = rng.choice([1, 2, 2, 3])
        whole, frac = rng.randint(0, 99), rng.randint(1, 10 ** places - 1)
        text = f"{whole},{frac:0{places}d}"
        if str(frac) != f"{frac:0{places}d}":
            continue
        reading = decimal_reading(whole, frac, places)
        if parse_numbers(reading)[:1] != [whole]:
            stats["decimal:check_failed"] += 1
            continue
        template = rng.choice(DECIMAL_FRAMES)
        keep("decimal", template.replace("{0}", text), template.replace("{0}", reading), "Десяткові дроби")

    rng.shuffle(added)
    args.output.mkdir(parents=True, exist_ok=True)
    combined = [*base, *added]
    pq.write_table(pa.Table.from_pylist(combined), args.output / "train.parquet", compression="zstd")
    pq.write_table(pa.Table.from_pylist(read_rows(args.base_dir / "dev-stratified.parquet")),
                   args.output / "dev-stratified.parquet", compression="zstd")
    by_kind = Counter(r["id"].split(":")[0].removeprefix("e44-") for r in added)
    report = {"base_rows": len(base), "added": len(added), "added_by_kind": dict(by_kind), "total_rows": len(combined),
              "stats": dict(stats), "train_sha256": sha256_file(args.output / "train.parquet"),
              "samples": {k: [{"source": r["source"], "target": r["target"]} for r in added if r["id"].startswith(f"e44-{k}:")][:3]
                          for k in by_kind}}
    write_json(args.output / "manifest.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=ROOT / "data/e43_context_growth")
    parser.add_argument("--books-dir", type=Path, default=Path("/home/devops/ukr-books/books"))
    parser.add_argument("--split", type=Path, default=ROOT / "data/book_domain/split.json")
    parser.add_argument("--frozen", type=Path, default=ROOT / "reports/hf-uktn-frozen-3597.jsonl")
    parser.add_argument("--eval-dir", type=Path, default=ROOT / "reports/domain-eval")
    parser.add_argument("--book-identity", type=Path, default=ROOT / "reports/book-prose-identity.jsonl")
    parser.add_argument("--book-numeric", type=Path, default=ROOT / "reports/book-numeric.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "data/e44_numeric_gaps")
    parser.add_argument("--seed", type=int, default=44044)
    parser.add_argument("--footnotes", type=int, default=6000)
    parser.add_argument("--sequences", type=int, default=3000)
    parser.add_argument("--years", type=int, default=6000)
    parser.add_argument("--decimals", type=int, default=2000)
    report = build(parser.parse_args())
    print(json.dumps({k: v for k, v in report.items() if k != "train_sha256"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
