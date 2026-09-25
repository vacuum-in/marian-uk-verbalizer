#!/usr/bin/env python3
"""E50: acronyms that are single tokens in the 16k vocabulary.

E49 lifted random acronyms (55.7% -> 73.4%) but not real ones (64% -> 66%):
common acronyms such as ФБР, ФСБ, СБУ, МВС are whole tokens ("▁ФСБ"), so the
model cannot see their letters and must learn each token's spelling, while
random combinations split into single letters. ФСБ has 2 training rows, both
leaving it unread; 424 rows leave such an acronym unread in the target.

On top of data/e49_acronyms:
  * every all-caps Cyrillic token in the tokenizer's own vocabulary whose shape
    is clearly letter-spelled (all consonants, or consonants + one final vowel
    as in СБУ), and that is not a dictionary word (ПРО), gets letter-name rows
    in training-only frames and book sentences;
  * targets that leave one of those acronyms unread are corrected to the
    letter reading.
The token list comes from the tokenizer vocabulary, not a hand-written list.
TRAINING DATA ONLY.
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
from sentencepiece import SentencePieceProcessor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from build_domain_eval import make_row  # noqa: E402
from build_e41_domain_growth import read_rows  # noqa: E402
from build_e49_acronyms import FRAMES, INSERTS, reading  # noqa: E402
from src.verbalizer.data.books import caps_token_kind, identity_target, iter_book_sentences, prose_identity_stratum  # noqa: E402
from src.verbalizer.evaluation.domain_check import score_row  # noqa: E402
from src.verbalizer.training.manifest import sha256_file, write_json  # noqa: E402

VOWELS = "АЕЄИІЇОУЮЯ"


def vocab_acronyms(spm_path: Path) -> list[str]:
    sp = SentencePieceProcessor(model_file=str(spm_path))
    pieces = {sp.id_to_piece(i).lstrip("▁") for i in range(sp.get_piece_size())}
    caps = [p for p in pieces if re.fullmatch(r"[А-ЯІЇЄҐ]{2,5}", p)]
    spelled = [p for p in caps
               if not any(c in VOWELS for c in p) or re.fullmatch(rf"[^{VOWELS}]{{2,}}[{VOWELS}]", p)]
    return sorted(p for p in spelled if caps_token_kind(p) != "word")


def build(args: argparse.Namespace) -> dict:
    rng = random.Random(args.seed)
    split = json.loads(args.split.read_text(encoding="utf-8"))
    blocked = {json.loads(l)["source"] for l in args.frozen.read_text(encoding="utf-8").splitlines() if l.strip()}
    for path in [*args.eval_dir.glob("*.jsonl"), args.book_identity, args.book_numeric]:
        blocked |= {json.loads(l)["source"] for l in path.read_text(encoding="utf-8").splitlines() if l.strip()}
    acronyms = vocab_acronyms(args.spm)
    token_re = re.compile(r"(?<![А-ЯІЇЄҐA-Za-zа-яіїєґ'’])(" + "|".join(acronyms) + r")(?![А-ЯІЇЄҐA-Za-zа-яіїєґ'’])")
    stats = Counter({"vocab_acronyms": len(acronyms)})

    base = read_rows(args.base_dir / "train.parquet")
    for row in base:
        if token_re.search(row["source"]) and token_re.search(row["target"]):
            def spoken(match: re.Match) -> str:
                text = reading(match.group(1))
                at_start = match.start() == 0 or row["target"][:match.start()].rstrip().endswith((".", "!", "?", "—"))
                return text[0].upper() + text[1:] if at_start else text
            row["target"] = token_re.sub(spoken, row["target"])
            stats["targets_corrected"] += 1
    seen = {row["source"] for row in base}

    sentences = []
    for path in sorted(args.books_dir.glob("book-*.json")):
        if path.stem.removeprefix("book-") in set(split["train"]) and len(sentences) < 20000:
            sentences += [s.text for s in iter_book_sentences(path)
                          if 25 <= len(s.text) <= 140 and s.text.endswith(".") and len(s.text.split()) >= 5
                          and prose_identity_stratum(s.text) == "plain"]
    rng.shuffle(sentences)

    added = []
    for acronym in acronyms:
        for index in range(args.per_token):
            if index % 2 == 0:
                template = rng.choice(FRAMES)
            else:
                tokens = rng.choice(sentences).split(" ")
                cut = rng.randint(1, len(tokens) - 1)
                template = " ".join(tokens[:cut]) + " " + rng.choice(INSERTS) + " " + " ".join(tokens[cut:])
            if template.count("{0}") != 1:
                continue
            said = reading(acronym)
            if template.startswith("{0}"):
                said = said[0].upper() + said[1:]
            slot = {"kind": "letters", "letters": acronym}
            source_row = make_row("acronym", "", template, [acronym], [slot])
            target = identity_target(make_row("acronym", "", template, [said], [slot])["source"]).replace(" ", " ")
            if not score_row(source_row, target)["ok"]:
                stats["check_failed"] += 1
                continue
            source = source_row["source"]
            if source in blocked or source in seen:
                stats["skipped"] += 1
                continue
            seen.add(source)
            added.append({"id": f"e50-acronym:{len(added)}", "source": source, "target": target,
                          "categories": ["Кириличні абревіатури політерно", "e50_token_acronym"],
                          "provenance": "generated_e50_vocab_acronyms"})

    rng.shuffle(added)
    args.output.mkdir(parents=True, exist_ok=True)
    combined = [*base, *added]
    pq.write_table(pa.Table.from_pylist(combined), args.output / "train.parquet", compression="zstd")
    pq.write_table(pa.Table.from_pylist(read_rows(args.base_dir / "dev-stratified.parquet")),
                   args.output / "dev-stratified.parquet", compression="zstd")
    report = {"base_rows": len(base), "added": len(added), "total_rows": len(combined), "stats": dict(stats),
              "acronyms": acronyms, "train_sha256": sha256_file(args.output / "train.parquet"),
              "samples": [{"source": r["source"], "target": r["target"]} for r in added[:6]]}
    write_json(args.output / "manifest.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=ROOT / "data/e49_acronyms")
    parser.add_argument("--spm", type=Path, default=ROOT / "artifacts/tokenizers/spm-16k/source.spm")
    parser.add_argument("--books-dir", type=Path, default=Path("/home/devops/ukr-books/books"))
    parser.add_argument("--split", type=Path, default=ROOT / "data/book_domain/split.json")
    parser.add_argument("--frozen", type=Path, default=ROOT / "reports/hf-uktn-frozen-3597.jsonl")
    parser.add_argument("--eval-dir", type=Path, default=ROOT / "reports/domain-eval")
    parser.add_argument("--book-identity", type=Path, default=ROOT / "reports/book-prose-identity.jsonl")
    parser.add_argument("--book-numeric", type=Path, default=ROOT / "reports/book-numeric.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "data/e50_token_acronyms")
    parser.add_argument("--seed", type=int, default=50050)
    parser.add_argument("--per-token", type=int, default=60)
    report = build(parser.parse_args())
    print(json.dumps({k: v for k, v in report.items() if k != "train_sha256"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
