#!/usr/bin/env python3
"""E45: more English, and number ranges in prose.

English slipped over three rounds (held-out set 87.2% -> 86.6% -> 86.2%) while
identity and numeric rows grew -- the E34b pattern -- and book ranges read at
36%. Adds, on top of data/e44_numeric_gaps (training-split books and training
English words only):

  * English words inserted into real book sentences, plus lists and pairs;
  * quantity ranges "5-10 днів" -> "від п'яти до десяти днів" (every numeral
    word put in the genitive with pymorphy3);
  * year spans "(1960–2006)" -> "тисяча дев'ятсот шістдесят — дві тисячі шість".

Readings are generated offline (num2words, pymorphy3, two-provider English
readings) and cross-checked by value with src/verbalizer/_audit_numparse or the
domain checker -- TRAINING DATA ONLY.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from build_domain_eval import make_row  # noqa: E402
from build_e41_domain_growth import ENGLISH_LIST, english_pool, read_rows, words  # noqa: E402
from src.verbalizer._audit_numparse import parse_numbers  # noqa: E402
from src.verbalizer.data.books import identity_target, iter_book_sentences, prose_identity_stratum  # noqa: E402
from src.verbalizer.data.english_readings import load_readings  # noqa: E402
from src.verbalizer.evaluation.domain_check import score_row  # noqa: E402
from src.verbalizer.training.manifest import sha256_file, write_json  # noqa: E402

MORPH = pymorphy3.MorphAnalyzer(lang="uk")
GENITIVE_FIXES = {"сорок": "сорока", "сто": "ста"}
RANGE_FRAMES = [
    ("Це займе {r}.", ["днів", "тижнів", "місяців", "хвилин", "годин", "років"]),
    ("Ремонт триває {r}.", ["днів", "тижнів", "місяців"]),
    ("Черга тривала {r}.", ["хвилин", "годин"]),
    ("Там працювало {r}.", ["людей", "робітників"]),
    ("Відстань становить {r}.", ["кілометрів", "метрів"]),
    ("Прочитай {r} до завтра.", ["сторінок"]),
    ("На зустріч прийшло {r}.", ["людей", "гостей"]),
]
SPAN_FRAMES = ["Роки життя: {s}.", "Фестиваль ({s}) залишив по собі добрі спогади.", "Правління ({s}) було неспокійним.",
               "Архів за {s} роки загубився.", "{s}"]
ENGLISH_PAIR = ["Вона прошепотіла «{0} {1}».", "На обкладинці: {0} {1}.", "Пісня {0} {1} звучала всю ніч."]


def genitive_number(n: int) -> str:
    out = []
    for word in words(n).split():
        if word in GENITIVE_FIXES:
            out.append(GENITIVE_FIXES[word])
            continue
        parses = [p for p in MORPH.parse(word) if "NUMR" in p.tag] or MORPH.parse(word)
        parse = parses[0]
        if word.startswith(("тисяч", "мільйон")):
            form = parse.inflect({"gent", "plur"}) if n // 1000 % 10 != 1 or n // 1000 % 100 == 11 else parse.inflect({"gent"})
            out.append(form.word if form else word)
            continue
        # Standard genitive of 5-9 and tens is "п'яти", "восьми", not the collective "п'ятьох", "вісьмох".
        candidates = [f.word for f in parse.lexeme if "gent" in f.tag.grammemes]
        standard = [w for w in candidates if not w.endswith("ох")] or candidates
        if word in ("два", "дві", "три", "чотири"):
            standard = candidates
        form = parse.inflect({"gent"})
        out.append(standard[0] if standard else (form.word if form else word))
    return " ".join(out)


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
            added.append({"id": f"e45-{kind}:{len(added)}", "source": source, "target": target,
                          "categories": [category, f"e45_{kind}"], "provenance": "generated_e45_checked"})

    train_ids = set(split["train"])
    sentences = []
    for path in sorted(args.books_dir.glob("book-*.json")):
        if path.stem.removeprefix("book-") in train_ids and len(sentences) < 80000:
            sentences += [s.text for s in iter_book_sentences(path)
                          if 25 <= len(s.text) <= 160 and s.text.endswith(".") and len(s.text.split()) >= 5
                          and prose_identity_stratum(s.text) == "plain"]
    rng.shuffle(sentences)

    readings = load_readings()
    pool = english_pool(readings)
    vocab = sorted(pool)

    def english_row(template: str, count: int, kind: str) -> None:
        chosen = rng.sample(vocab, count)
        shown = [w.capitalize() if rng.random() < 0.35 else w for w in chosen]
        slots = [{"kind": "english", "variants": sorted(set().union(*readings[w].values()))} for w in chosen]
        source_row = make_row("english", kind, template, shown, slots)
        target = make_row("english", kind, identity_target(template), [pool[w] for w in chosen], slots)["source"]
        if not score_row(source_row, target)["ok"]:
            stats[f"{kind}:check_failed"] += 1
            return
        keep(kind, source_row["source"], target, "Транслітерація")

    for index in range(args.english_in_sentence):
        tokens = sentences[index % len(sentences)].split(" ")
        count = rng.choice([1, 1, 2, 3])
        cut = rng.randint(1, len(tokens) - 1)
        slots = " ".join("{%d}" % i for i in range(count))
        template = " ".join(tokens[:cut]) + " " + slots + " " + " ".join(tokens[cut:])
        if template.count("{0}") == 1:
            english_row(template, count, "english_in_sentence")
    for _ in range(args.english_lists):
        count = rng.randint(3, 10)
        listed = ", ".join("{%d}" % i for i in range(count))
        english_row(rng.choice(ENGLISH_LIST).replace("{list}", listed), count, "english_list")
    for _ in range(args.english_pairs):
        english_row(rng.choice(ENGLISH_PAIR), 2, "english_pair")

    for _ in range(args.ranges):
        a = rng.choice([rng.randint(1, 20), rng.randint(1, 99), rng.randint(100, 999)])
        b = a + rng.choice([1, 2, 3, 5, 10, rng.randint(2, 50)])
        frame, nouns = rng.choice(RANGE_FRAMES)
        noun = rng.choice(nouns)
        dash = rng.choice(["-", "–", "—"])
        reading = f"від {genitive_number(a)} до {genitive_number(b)} {noun}"
        if parse_numbers(reading) != [a, b]:
            stats["range:check_failed"] += 1
            continue
        keep("range", frame.replace("{r}", f"{a}{dash}{b} {noun}"), frame.replace("{r}", reading), "Діапазони")

    for _ in range(args.spans):
        a = rng.randint(1500, 2020)
        b = a + rng.randint(1, 90)
        if b > 2099:
            continue
        dash = rng.choice(["–", "—", "-"])
        reading = f"{words(a)} — {words(b)}"
        if parse_numbers(words(a)) != [a] or parse_numbers(words(b)) != [b]:
            stats["span:check_failed"] += 1
            continue
        frame = rng.choice(SPAN_FRAMES)
        target = frame.replace("{s}", reading)
        keep("span", frame.replace("{s}", f"{a}{dash}{b}"), target[0].upper() + target[1:], "Діапазони")

    rng.shuffle(added)
    args.output.mkdir(parents=True, exist_ok=True)
    combined = [*base, *added]
    pq.write_table(pa.Table.from_pylist(combined), args.output / "train.parquet", compression="zstd")
    pq.write_table(pa.Table.from_pylist(read_rows(args.base_dir / "dev-stratified.parquet")),
                   args.output / "dev-stratified.parquet", compression="zstd")
    by_kind = Counter(r["id"].split(":")[0].removeprefix("e45-") for r in added)
    report = {"base_rows": len(base), "added": len(added), "added_by_kind": dict(by_kind), "total_rows": len(combined),
              "stats": dict(stats), "train_sha256": sha256_file(args.output / "train.parquet"),
              "samples": {k: [{"source": r["source"], "target": r["target"]} for r in added if r["id"].startswith(f"e45-{k}:")][:3]
                          for k in by_kind}}
    write_json(args.output / "manifest.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=ROOT / "data/e44_numeric_gaps")
    parser.add_argument("--books-dir", type=Path, default=Path("/home/devops/ukr-books/books"))
    parser.add_argument("--split", type=Path, default=ROOT / "data/book_domain/split.json")
    parser.add_argument("--frozen", type=Path, default=ROOT / "reports/hf-uktn-frozen-3597.jsonl")
    parser.add_argument("--eval-dir", type=Path, default=ROOT / "reports/domain-eval")
    parser.add_argument("--book-identity", type=Path, default=ROOT / "reports/book-prose-identity.jsonl")
    parser.add_argument("--book-numeric", type=Path, default=ROOT / "reports/book-numeric.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "data/e45_english_ranges")
    parser.add_argument("--seed", type=int, default=45045)
    parser.add_argument("--english-in-sentence", type=int, default=8000)
    parser.add_argument("--english-lists", type=int, default=5000)
    parser.add_argument("--english-pairs", type=int, default=2000)
    parser.add_argument("--ranges", type=int, default=5000)
    parser.add_argument("--spans", type=int, default=2000)
    report = build(parser.parse_args())
    print(json.dumps({k: v for k, v in report.items() if k != "train_sha256"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
