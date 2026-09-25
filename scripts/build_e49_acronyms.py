#!/usr/bin/env python3
"""E49: Cyrillic acronyms spelled letter by letter.

A user found "Я пропрацював у ФБР двадцять років," -> "у еф еф бе ер". Probing
E46/E48 shows the skill is unstable: ФСБ -> "уили" / "в місті", ЦРУ sometimes
left unread, doubled letters at sentence start. The benchmark has ~20 such rows.

Adds, on top of data/e48_rebalanced, random letter combinations shaped like
letter-spelled acronyms (all consonants, or consonants ending in У/А/І as in
СБУ), read with the Ukrainian letter names, in training-only frames and inside
training-split book sentences. Random combinations teach the skill without a
word list; the real acronyms in reports/domain-eval/acronyms.jsonl are never
generated. TRAINING DATA ONLY.
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

from build_domain_eval import CONSONANTS, EVAL_ACRONYMS, make_row  # noqa: E402
from build_e41_domain_growth import read_rows  # noqa: E402
from src.verbalizer.data.books import identity_target, iter_book_sentences, prose_identity_stratum  # noqa: E402
from src.verbalizer.evaluation.domain_check import LETTER_NAMES, score_row  # noqa: E402
from src.verbalizer.training.manifest import sha256_file, write_json  # noqa: E402

FRAMES = ["{0} оголосила про нові правила.", "Він працював у {0} до пенсії, а потім поїхав.", "Лист від {0} прийшов учора.",
          "Слідство веде {0}, але деталей немає.", "Про це заявили в {0}, і новину підхопили всі.", "Вона звернулася до {0},",
          "Представники {0} прибули на місце.", "— Хто з {0}? — спитала вона.", "Керівник {0} подав у відставку."]
INSERTS = ["у {0}", "в {0}", "від {0}", "до {0}", "(за даними {0})", "з {0},"]


def random_acronym(rng: random.Random) -> str:
    length = rng.choice([2, 3, 3, 3, 4])
    letters = "".join(rng.choice(CONSONANTS) for _ in range(length))
    if rng.random() < 0.35:
        letters = letters[:-1] + rng.choice("УАІ")
    return letters


def reading(acronym: str) -> str:
    return " ".join(LETTER_NAMES[ch][0] for ch in acronym)


def build(args: argparse.Namespace) -> dict:
    rng = random.Random(args.seed)
    split = json.loads(args.split.read_text(encoding="utf-8"))
    blocked = {json.loads(l)["source"] for l in args.frozen.read_text(encoding="utf-8").splitlines() if l.strip()}
    for path in [*args.eval_dir.glob("*.jsonl"), args.book_identity, args.book_numeric]:
        blocked |= {json.loads(l)["source"] for l in path.read_text(encoding="utf-8").splitlines() if l.strip()}
    base = read_rows(args.base_dir / "train.parquet")
    seen = {row["source"] for row in base}
    added, stats = [], Counter()
    reserved = set(EVAL_ACRONYMS)

    sentences = []
    for path in sorted(args.books_dir.glob("book-*.json")):
        if path.stem.removeprefix("book-") in set(split["train"]) and len(sentences) < 20000:
            sentences += [s.text for s in iter_book_sentences(path)
                          if 25 <= len(s.text) <= 140 and s.text.endswith(".") and len(s.text.split()) >= 5
                          and prose_identity_stratum(s.text) == "plain"]
    rng.shuffle(sentences)

    for index in range(args.rows):
        acronym = random_acronym(rng)
        if acronym in reserved:
            stats["reserved_skipped"] += 1
            continue
        if index % 2 == 0:
            template = rng.choice(FRAMES)
        else:
            tokens = sentences[index % len(sentences)].split(" ")
            cut = rng.randint(1, len(tokens) - 1)
            template = " ".join(tokens[:cut]) + " " + rng.choice(INSERTS) + " " + " ".join(tokens[cut:])
        if template.count("{0}") != 1:
            continue
        spoken = reading(acronym)
        if template.startswith("{0}"):
            spoken = spoken[0].upper() + spoken[1:]
        slot = {"kind": "letters", "letters": acronym}
        source_row = make_row("acronym", "", template, [acronym], [slot])
        target = identity_target(make_row("acronym", "", template, [spoken], [slot])["source"]).replace(" ", " ")
        if not score_row(source_row, target)["ok"]:
            stats["check_failed"] += 1
            continue
        stats["considered"] += 1
        if source_row["source"] in blocked or source_row["source"] in seen:
            stats["skipped"] += 1
            continue
        seen.add(source_row["source"])
        added.append({"id": f"e49-acronym:{len(added)}", "source": source_row["source"], "target": target,
                      "categories": ["Кириличні абревіатури політерно", "e49_acronym"], "provenance": "generated_e49_checked"})

    rng.shuffle(added)
    args.output.mkdir(parents=True, exist_ok=True)
    combined = [*base, *added]
    pq.write_table(pa.Table.from_pylist(combined), args.output / "train.parquet", compression="zstd")
    pq.write_table(pa.Table.from_pylist(read_rows(args.base_dir / "dev-stratified.parquet")),
                   args.output / "dev-stratified.parquet", compression="zstd")
    report = {"base_rows": len(base), "added": len(added), "total_rows": len(combined), "stats": dict(stats),
              "train_sha256": sha256_file(args.output / "train.parquet"),
              "samples": [{"source": r["source"], "target": r["target"]} for r in added[:6]]}
    write_json(args.output / "manifest.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=ROOT / "data/e48_rebalanced")
    parser.add_argument("--books-dir", type=Path, default=Path("/home/devops/ukr-books/books"))
    parser.add_argument("--split", type=Path, default=ROOT / "data/book_domain/split.json")
    parser.add_argument("--frozen", type=Path, default=ROOT / "reports/hf-uktn-frozen-3597.jsonl")
    parser.add_argument("--eval-dir", type=Path, default=ROOT / "reports/domain-eval")
    parser.add_argument("--book-identity", type=Path, default=ROOT / "reports/book-prose-identity.jsonl")
    parser.add_argument("--book-numeric", type=Path, default=ROOT / "reports/book-numeric.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "data/e49_acronyms")
    parser.add_argument("--seed", type=int, default=49049)
    parser.add_argument("--rows", type=int, default=8000)
    report = build(parser.parse_args())
    print(json.dumps({k: v for k, v in report.items() if k != "train_sha256"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
