"""Build the held-out book split and reference-free book-domain evaluation sets.

Outputs:
  * data/book_domain/split.json          -- which book ids are held out
  * reports/book-prose-identity.jsonl    -- sentences whose correct output is the input
  * reports/book-numeric.jsonl           -- sentences with numeric spans (round-trip scored)

Held-out books must never feed training data.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.verbalizer.data.books import (  # noqa: E402
    book_id_of,
    identity_target,
    is_holdout,
    iter_book_sentences,
    prose_identity_stratum,
)
from src.verbalizer.numeric import detect_numeric_spans  # noqa: E402
from src.verbalizer.training.manifest import sha256_file, write_json  # noqa: E402

BRACKET_RE = re.compile(r"[\[{(]\s*\d+\s*[\]})]")
SEQUENCE_RE = re.compile(r"(?<!\d)(?<!\d )\d{1,2}(?: \d{1,2}){2,}(?! ?\d)")
IDENTITY_QUOTAS = {"plain": 600, "dialogue": 500, "one_letter_start": 500, "allcaps": 400}
NUMERIC_QUOTA_PER_STRATUM = 80


def numeric_stratum(sentence: str) -> str | None:
    spans = detect_numeric_spans(sentence)
    if not any(span.digit_sequence for span in spans):
        return None
    if BRACKET_RE.search(sentence):
        return "bracket"
    if SEQUENCE_RE.search(sentence):
        return "digit_sequence"
    kinds = [span.kind.value for span in spans if span.digit_sequence]
    return kinds[0]


def reservoir_add(bucket: list, seen: int, item, quota: int, rng: random.Random) -> None:
    if len(bucket) < quota:
        bucket.append(item)
    else:
        index = rng.randrange(seen)
        if index < quota:
            bucket[index] = item


def build(args: argparse.Namespace) -> dict:
    rng = random.Random(args.seed)
    paths = sorted(args.books_dir.glob("book-*.json"), key=lambda p: int(book_id_of(p)))
    holdout = [book_id_of(p) for p in paths if is_holdout(book_id_of(p), args.holdout_fraction)]
    holdout_set = set(holdout)
    train = [book_id_of(p) for p in paths if book_id_of(p) not in holdout_set]

    identity: dict[str, list] = defaultdict(list)
    numeric: dict[str, list] = defaultdict(list)
    seen_identity: Counter = Counter()
    seen_numeric: Counter = Counter()
    unique: set[str] = set()
    for path in paths:
        if book_id_of(path) not in holdout_set:
            continue
        for sentence in iter_book_sentences(path):
            text = sentence.text
            if len(text) > args.max_chars or text in unique:
                continue
            unique.add(text)
            row = {"book_id": sentence.book_id, "source": text}
            stratum = prose_identity_stratum(text) if len(text) >= args.min_chars else None
            if stratum:
                seen_identity[stratum] += 1
                reservoir_add(identity[stratum], seen_identity[stratum],
                              {**row, "stratum": stratum, "target": identity_target(text),
                               "nbsp": " " in text},
                              IDENTITY_QUOTAS[stratum], rng)
                continue
            stratum = numeric_stratum(text)
            if stratum:
                seen_numeric[stratum] += 1
                reservoir_add(numeric[stratum], seen_numeric[stratum], {**row, "stratum": stratum},
                              NUMERIC_QUOTA_PER_STRATUM, rng)

    def write(path: Path, buckets: dict[str, list], prefix: str) -> dict[str, int]:
        rows = [row for name in sorted(buckets) for row in buckets[name]]
        with path.open("w", encoding="utf-8") as handle:
            for index, row in enumerate(rows):
                handle.write(json.dumps({"id": f"{prefix}-{index:05d}", **row}, ensure_ascii=False) + "\n")
        return {name: len(items) for name, items in sorted(buckets.items())}

    args.split.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.split, {"method": "sha256('book-domain-split-v1:<id>') < fraction",
                            "holdout_fraction": args.holdout_fraction,
                            "holdout": holdout, "train": train})
    identity_counts = write(args.identity, identity, "prose")
    numeric_counts = write(args.numeric, numeric, "num")
    summary = {
        "books": len(paths), "holdout_books": len(holdout), "train_books": len(train),
        "identity": identity_counts, "identity_available": dict(seen_identity),
        "numeric": numeric_counts, "numeric_available": dict(seen_numeric),
        "sha256": {"identity": sha256_file(args.identity), "numeric": sha256_file(args.numeric)},
    }
    write_json(args.split.with_name("eval_sets_manifest.json"), summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--books-dir", type=Path, default=Path("/home/devops/ukr-books/books"))
    parser.add_argument("--holdout-fraction", type=float, default=0.09)
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--min-chars", type=int, default=12)
    parser.add_argument("--max-chars", type=int, default=250)
    parser.add_argument("--split", type=Path, default=ROOT / "data/book_domain/split.json")
    parser.add_argument("--identity", type=Path, default=ROOT / "reports/book-prose-identity.jsonl")
    parser.add_argument("--numeric", type=Path, default=ROOT / "reports/book-numeric.jsonl")
    print(json.dumps(build(parser.parse_args()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
