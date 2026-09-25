#!/usr/bin/env python3
"""E48: combined, rebalanced run -- NBSP fix plus an English boost.

The tokenizer's SentencePiece normalizer is `identity`, so a no-break space
(U+00A0, used by books after one-letter words: "в\xa0окулярах") becomes <unk>.
Training targets kept the NBSP, so the model learned to emit <unk>, which
decodes to nothing and glues words ("вокулярах"). 114 of E46's 125 book-prose
failures are exactly this. English also slipped whenever non-English rows grew
(E44 86.2%, E47 85.2%).

On top of data/e47_money_pairs (training-split books and words only):
  * every target's NBSP / narrow NBSP / thin space becomes an ordinary space
    (same sound; inputs keep their NBSP as books write them);
  * NBSP-augmented copies of prose rows (NBSP after one-letter words in the
    source, ordinary spaces in the target);
  * new English rows: words inside book sentences, lists and pairs.
Dataset construction only; nothing here runs at inference.
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

from build_domain_eval import make_row  # noqa: E402
from build_e41_domain_growth import ENGLISH_LIST, english_pool, read_rows  # noqa: E402
from src.verbalizer.data.books import identity_target, iter_book_sentences, prose_identity_stratum  # noqa: E402
from src.verbalizer.data.english_readings import load_readings  # noqa: E402
from src.verbalizer.evaluation.domain_check import score_row  # noqa: E402
from src.verbalizer.training.manifest import sha256_file, write_json  # noqa: E402

SPACE_VARIANTS = re.compile("[\u00a0\u202f\u2009]")
ONE_LETTER = re.compile(r"(?<![\wА-Яа-яІіЇїЄєҐґ'’])([вузаіоякВУЗАІОЯК]) (?=\S)")
ENGLISH_PAIR = ["Вона прошепотіла «{0} {1}».", "На обкладинці: {0} {1}.", "Пісня {0} {1} звучала всю ніч.",
                "Табличка «{0} {1}» висіла над входом."]


def build(args: argparse.Namespace) -> dict:
    rng = random.Random(args.seed)
    split = json.loads(args.split.read_text(encoding="utf-8"))
    blocked = {json.loads(l)["source"] for l in args.frozen.read_text(encoding="utf-8").splitlines() if l.strip()}
    for path in [*args.eval_dir.glob("*.jsonl"), args.book_identity, args.book_numeric]:
        blocked |= {json.loads(l)["source"] for l in path.read_text(encoding="utf-8").splitlines() if l.strip()}
    base = read_rows(args.base_dir / "train.parquet")
    stats = Counter()
    for row in base:
        if SPACE_VARIANTS.search(row["target"]):
            row["target"] = SPACE_VARIANTS.sub(" ", row["target"])
            stats["targets_space_fixed"] += 1
    seen = {row["source"] for row in base}
    added = []

    def keep(kind: str, source: str, target: str, category: str) -> None:
        stats[f"{kind}:considered"] += 1
        if source in blocked:
            stats[f"{kind}:blocked"] += 1
        elif source in seen:
            stats[f"{kind}:duplicate"] += 1
        else:
            seen.add(source)
            added.append({"id": f"e48-{kind}:{len(added)}", "source": source, "target": target,
                          "categories": [category, f"e48_{kind}"], "provenance": "generated_e48"})

    # NBSP augmentation of prose-like rows (identity and context rows from training books).
    prose = [r for r in base if r["id"].startswith(("e42-identity", "e42-context", "e43-context"))
             and "\u00a0" not in r["source"] and ONE_LETTER.search(r["source"])]
    rng.shuffle(prose)
    for row in prose[: args.nbsp_rows]:
        keep("nbsp", ONE_LETTER.sub("\\1\u00a0", row["source"]), row["target"], "Нерозривні пробіли")

    # English boost.
    train_ids = set(split["train"])
    sentences = []
    for path in sorted(args.books_dir.glob("book-*.json")):
        if path.stem.removeprefix("book-") in train_ids and len(sentences) < 60000:
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
        target = SPACE_VARIANTS.sub(" ", target)
        if not score_row(source_row, target)["ok"]:
            stats[f"{kind}:check_failed"] += 1
            return
        keep(kind, source_row["source"], target, "Транслітерація")

    for index in range(args.english_in_sentence):
        tokens = sentences[index % len(sentences)].split(" ")
        count = rng.choice([1, 1, 2, 3])
        cut = rng.randint(1, len(tokens) - 1)
        template = " ".join(tokens[:cut]) + " " + " ".join("{%d}" % i for i in range(count)) + " " + " ".join(tokens[cut:])
        if template.count("{0}") == 1:
            english_row(template, count, "english_in_sentence")
    for _ in range(args.english_lists):
        count = rng.randint(3, 12)
        listed = ", ".join("{%d}" % i for i in range(count))
        if rng.random() < 0.3:
            listed = ", ".join("{%d}" % i for i in range(count - 1)) + " і {%d}" % (count - 1)
        english_row(rng.choice(ENGLISH_LIST).replace("{list}", listed), count, "english_list")
    for _ in range(args.english_pairs):
        english_row(rng.choice(ENGLISH_PAIR), 2, "english_pair")

    rng.shuffle(added)
    args.output.mkdir(parents=True, exist_ok=True)
    combined = [*base, *added]
    pq.write_table(pa.Table.from_pylist(combined), args.output / "train.parquet", compression="zstd")
    dev = read_rows(args.base_dir / "dev-stratified.parquet")
    for row in dev:
        row["target"] = SPACE_VARIANTS.sub(" ", row["target"])
    pq.write_table(pa.Table.from_pylist(dev), args.output / "dev-stratified.parquet", compression="zstd")
    by_kind = Counter(r["id"].split(":")[0].removeprefix("e48-") for r in added)
    report = {"base_rows": len(base), "added": len(added), "added_by_kind": dict(by_kind), "total_rows": len(combined),
              "stats": dict(stats), "train_sha256": sha256_file(args.output / "train.parquet"),
              "samples": {k: [{"source": r["source"], "target": r["target"]} for r in added if r["id"].startswith(f"e48-{k}:")][:2]
                          for k in by_kind}}
    write_json(args.output / "manifest.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=ROOT / "data/e47_money_pairs")
    parser.add_argument("--books-dir", type=Path, default=Path("/home/devops/ukr-books/books"))
    parser.add_argument("--split", type=Path, default=ROOT / "data/book_domain/split.json")
    parser.add_argument("--frozen", type=Path, default=ROOT / "reports/hf-uktn-frozen-3597.jsonl")
    parser.add_argument("--eval-dir", type=Path, default=ROOT / "reports/domain-eval")
    parser.add_argument("--book-identity", type=Path, default=ROOT / "reports/book-prose-identity.jsonl")
    parser.add_argument("--book-numeric", type=Path, default=ROOT / "reports/book-numeric.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "data/e48_rebalanced")
    parser.add_argument("--seed", type=int, default=48048)
    parser.add_argument("--nbsp-rows", type=int, default=10000)
    parser.add_argument("--english-in-sentence", type=int, default=8000)
    parser.add_argument("--english-lists", type=int, default=6000)
    parser.add_argument("--english-pairs", type=int, default=2000)
    report = build(parser.parse_args())
    print(json.dumps({k: v for k, v in report.items() if k != "train_sha256"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
