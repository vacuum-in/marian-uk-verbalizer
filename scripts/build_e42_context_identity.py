#!/usr/bin/env python3
"""E42: identity prose, identifiers inside real sentences, longer English lists.

E41 failures on the held-out sets: the number is read correctly but the words
after it are damaged ("до з'ясування" -> "до з'ятої години"), English lists
lose or mangle items, and contiguous cards with zero runs miscount. On book
prose E41 keeps only 64% of sentences unchanged because the corpus has few
identity rows.

Adds, on top of data/e41_domain_growth (training-split books only; held-out
books in data/book_domain/split.json are never read):
  * identity pairs from book prose (target = source, straight apostrophes),
    stratified like the book evaluation;
  * card / IBAN / phone / coordinate phrases inserted into real sentences;
  * English lists of 5-12 words; contiguous cards with long zero runs.
Every generated row is verified with the evaluation checker. Offline
dataset construction only; nothing here runs at inference.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from build_domain_eval import digits_string, make_row  # noqa: E402
from build_e41_domain_growth import (  # noqa: E402
    ENGLISH_LIST, card_pair, coordinate_pair, digitwise, english_pool, iban_pair, phone_pair, read_rows,
)
from src.verbalizer.data.books import identity_target, iter_book_sentences, prose_identity_stratum  # noqa: E402
from src.verbalizer.data.english_readings import load_readings  # noqa: E402
from src.verbalizer.evaluation.domain_check import score_row  # noqa: E402
from src.verbalizer.training.manifest import sha256_file, write_json  # noqa: E402

IDENTITY_QUOTAS = {"plain": 12000, "dialogue": 10000, "one_letter_start": 8000, "allcaps": 4000}
INSERTS = {
    "card": (card_pair, ["на картку {0}", "з картки {0}", "карткою {0}", "(картка {0})"]),
    "iban": (iban_pair, ["на рахунок {0}", "з рахунку {0}", "(рахунок {0})"]),
    "phone": (phone_pair, ["за номером {0}", "на номер {0}", "(телефон {0})", "— дзвоніть {0} —"]),
    "coordinates": (coordinate_pair, ["у точці {0}", "на координатах {0}", "(координати {0})"]),
}
EXTRA_CONNECTORS = {
    "card": ["переказав на {0}", "— картка {0} —", "номер картки {0}", "карта {0},"],
    "iban": ["реквізити {0}", "на рахунку {0}", "рахунок {0},"],
    "phone": ["номер {0},", "телефонуйте {0}", "мобільний {0}", "подзвонила на {0}"],
    "coordinates": ["в районі {0}", "координати {0},", "за координатами {0}"],
}
LEADERS = {
    "card": ["Картка {0}:", "Картка {0} —"],
    "iban": ["Рахунок {0}:", "Рахунок {0} —"],
    "phone": ["Телефон {0}:", "Номер {0} —"],
    "coordinates": ["Точка {0}:", "Координати {0} —"],
}
TRAILERS = {
    "card": ["Номер картки: {0}.", "Картка для переказу — {0}."],
    "iban": ["Рахунок: {0}.", "Реквізити — {0}."],
    "phone": ["Телефон: {0}.", "Дзвоніть на {0}."],
    "coordinates": ["Координати: {0}.", "Місце — {0}."],
}


def collect_book_sentences(books_dir: Path, split: dict, rng: random.Random, max_context: int):
    """Reservoir-sample identity sentences per stratum and plain sentences for context frames."""

    train_ids = set(split["train"])
    identity, seen_counts, context, seen_context = defaultdict(list), Counter(), [], 0
    for path in sorted(books_dir.glob("book-*.json")):
        if path.stem.removeprefix("book-") not in train_ids:
            continue
        for sentence in iter_book_sentences(path):
            text = sentence.text
            if not 12 <= len(text) <= 250:
                continue
            stratum = prose_identity_stratum(text)
            if not stratum:
                continue
            seen_counts[stratum] += 1
            bucket, quota = identity[stratum], IDENTITY_QUOTAS[stratum]
            if len(bucket) < quota:
                bucket.append(text)
            elif (j := rng.randrange(seen_counts[stratum])) < quota:
                bucket[j] = text
            if stratum == "plain" and 30 <= len(text) <= 160 and text.endswith(".") and len(text.split()) >= 5:
                seen_context += 1
                if len(context) < max_context:
                    context.append(text)
                elif (j := rng.randrange(seen_context)) < max_context:
                    context[j] = text
    return identity, dict(seen_counts), context


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
            added.append({"id": f"{args.tag}-{kind}:{len(added)}", "source": source, "target": target,
                          "categories": [category, f"{args.tag}_{kind}"], "provenance": "generated_e42_checker_verified"})

    identity, available, context = collect_book_sentences(args.books_dir, split, rng, args.context_rows)
    if not args.no_identity:
        for stratum, sentences in identity.items():
            for text in sentences:
                keep(f"identity_{stratum}", text, identity_target(text), "Pass-through (без нормалізації)")

    per_domain = {"card": 0.35, "iban": 0.2, "phone": 0.3, "coordinates": 0.15}
    for text in context:
        domain = rng.choices(list(per_domain), weights=list(per_domain.values()))[0]
        generate, connectors = INSERTS[domain]
        if args.extra_connectors:
            connectors = connectors + EXTRA_CONNECTORS[domain]
        value, reading, slot = generate(rng)
        placement = rng.random()
        if placement < (0.55 if args.extra_connectors else 0.7):
            words = text.split(" ")
            cut = rng.randint(1, len(words) - 1)
            left, right = " ".join(words[:cut]), " ".join(words[cut:])
            template = left + " " + rng.choice(connectors) + " " + right
        elif args.extra_connectors and placement < 0.8:
            template = rng.choice(LEADERS[domain]) + " " + text[0].lower() + text[1:]
        else:
            template = text + " " + rng.choice(TRAILERS[domain])
        if template.count("{0}") != 1:
            continue
        source_row = make_row(domain, "context", template, [value], [slot])
        target_row = make_row(domain, "context", identity_target(template), [reading], [slot])
        if not score_row(source_row, target_row["source"])["ok"]:
            stats[f"context_{domain}:checker_rejected"] += 1
            continue
        keep(f"context_{domain}", source_row["source"], target_row["source"], "Контекст навколо ідентифікатора")

    readings = load_readings()
    pool = english_pool(readings)
    vocab = sorted(pool)
    for _ in range(args.english_lists):
        count = rng.randint(5, 12)
        chosen = rng.sample(vocab, count)
        shown = [w.capitalize() if rng.random() < 0.35 else w for w in chosen]
        slots = [{"kind": "english", "variants": sorted(set().union(*readings[w].values()))} for w in chosen]
        listed = ", ".join("{%d}" % i for i in range(count))
        if rng.random() < 0.3:
            listed = ", ".join("{%d}" % i for i in range(count - 1)) + " і {%d}" % (count - 1)
        template = rng.choice(ENGLISH_LIST).replace("{list}", listed)
        source_row = make_row("english", "list", template, shown, slots)
        target_row = make_row("english", "list", template, [pool[w] for w in chosen], slots)
        if not score_row(source_row, target_row["source"])["ok"]:
            stats["english_list:checker_rejected"] += 1
            continue
        keep("english_list", source_row["source"], target_row["source"], "Транслітерація")

    for _ in range(args.zero_cards):
        digits = rng.choice("2345") + digits_string(rng, 15, zero_run=0.95)
        groups = [digits[i:i + 4] for i in range(0, 16, 4)]
        frame = rng.choice(["{0}", "Картка {0}.", "Номер картки {0}, перевірте."])
        slot = {"kind": "digits", "digits": digits}
        source_row = make_row("card", "zero", frame, [digits], [slot])
        target_row = make_row("card", "zero", frame, [", ".join(digitwise(g) for g in groups)], [slot])
        if score_row(source_row, target_row["source"])["ok"]:
            keep("zero_card", source_row["source"], target_row["source"], "Числові рядки — банківські картки")

    rng.shuffle(added)
    args.output.mkdir(parents=True, exist_ok=True)
    combined = [*base, *added]
    pq.write_table(pa.Table.from_pylist(combined), args.output / "train.parquet", compression="zstd")
    pq.write_table(pa.Table.from_pylist(read_rows(args.base_dir / "dev-stratified.parquet")),
                   args.output / "dev-stratified.parquet", compression="zstd")
    by_kind = Counter(r["id"].split(":")[0].removeprefix(f"{args.tag}-") for r in added)
    report = {
        "base_rows": len(base), "added": len(added), "added_by_kind": dict(by_kind), "total_rows": len(combined),
        "identity_available": available, "stats": dict(stats),
        "identity_fraction": round(sum(1 for r in combined if r["source"].replace("’", "'") == r["target"]) / len(combined), 4),
        "train_sha256": sha256_file(args.output / "train.parquet"),
        "samples": {k: next(({"source": r["source"], "target": r["target"]} for r in added
                             if r["id"].startswith(f"{args.tag}-{k}:")), None) for k in by_kind},
    }
    write_json(args.output / "manifest.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=ROOT / "data/e41_domain_growth")
    parser.add_argument("--books-dir", type=Path, default=Path("/home/devops/ukr-books/books"))
    parser.add_argument("--split", type=Path, default=ROOT / "data/book_domain/split.json")
    parser.add_argument("--frozen", type=Path, default=ROOT / "reports/hf-uktn-frozen-3597.jsonl")
    parser.add_argument("--eval-dir", type=Path, default=ROOT / "reports/domain-eval")
    parser.add_argument("--book-identity", type=Path, default=ROOT / "reports/book-prose-identity.jsonl")
    parser.add_argument("--book-numeric", type=Path, default=ROOT / "reports/book-numeric.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "data/e42_context_identity")
    parser.add_argument("--seed", type=int, default=42042)
    parser.add_argument("--context-rows", type=int, default=7000)
    parser.add_argument("--english-lists", type=int, default=8000)
    parser.add_argument("--zero-cards", type=int, default=3000)
    parser.add_argument("--tag", default="e42")
    parser.add_argument("--no-identity", action="store_true", help="skip identity prose (already in the base)")
    parser.add_argument("--extra-connectors", action="store_true", help="E43: more connectors and sentence-initial identifiers")
    report = build(parser.parse_args())
    print(json.dumps({k: v for k, v in report.items() if k != "train_sha256"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
