#!/usr/bin/env python3
"""E51: angular minutes as "хвилина", and acronyms no longer left unread.

User correction (2026-09-25): "мінута" is not correct Ukrainian; the angular
minute is "хвилина / хвилини / хвилин". 9,809 training targets used "мінут*",
mostly from the E41 coordinate generator (since fixed) plus 113 base rows.
Also, 111 letter-spelled acronyms are left unread in targets (ЦРУ 3x), the
likely reason E50 leaves "ЦРУ" unread at the start of a sentence.

On top of data/e50_token_acronyms (targets only; sources untouched):
  * every "мінут*" form becomes the matching "хвилин*" form, unless the source
    itself contains "мінут" (identity prose must stay as written);
  * a letter-spelled acronym (all consonants, or consonants + one final vowel,
    not a dictionary word) left unread in the target gets its letter reading.
Dataset construction only.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from build_e41_domain_growth import read_rows  # noqa: E402
from build_e49_acronyms import reading  # noqa: E402
from src.verbalizer.data.books import caps_token_kind  # noqa: E402
from src.verbalizer.training.manifest import sha256_file, write_json  # noqa: E402

MINUTE = re.compile(r"(?<![а-яіїєґА-ЯІЇЄҐ'’])([Мм])інут(а|и|у|ою|і|ам|ами|ах)?(?![а-яіїєґ'’])")
KHVYLYNA = {None: "хвилин", "а": "хвилина", "и": "хвилини", "у": "хвилину", "ою": "хвилиною", "і": "хвилині",
            "ам": "хвилинам", "ами": "хвилинами", "ах": "хвилинах"}
ACRONYM = re.compile(r"(?<![А-ЯІЇЄҐA-Za-zа-яіїєґ'’])([А-ЯІЇЄҐ]{2,5})(?![А-ЯІЇЄҐA-Za-zа-яіїєґ'’])")
VOWELS = "АЕЄИІЇОУЮЯ"


def fix_minutes(row: dict, stats: Counter) -> None:
    if MINUTE.search(row["source"]) or not MINUTE.search(row["target"]):
        return

    def replace(match: re.Match) -> str:
        word = KHVYLYNA[match.group(2)]
        return word.capitalize() if match.group(1) == "М" else word

    row["target"] = MINUTE.sub(replace, row["target"])
    stats["minutes_fixed"] += 1


def spelled(acronym: str) -> bool:
    shaped = not any(c in VOWELS for c in acronym) or re.fullmatch(rf"[^{VOWELS}]{{2,}}[{VOWELS}]", acronym)
    return bool(shaped) and caps_token_kind(acronym) != "word"


def fix_unread_acronyms(row: dict, stats: Counter) -> None:
    left = [a for a in set(ACRONYM.findall(row["source"])) if spelled(a)
            and re.search(rf"(?<![А-ЯІЇЄҐ]){a}(?![А-ЯІЇЄҐ])", row["target"])]
    for acronym in left:
        def replace(match: re.Match) -> str:
            text = reading(acronym)
            before = row["target"][:match.start()].rstrip()
            return text[0].upper() + text[1:] if not before or before.endswith((".", "!", "?", "—")) else text
        row["target"] = re.sub(rf"(?<![А-ЯІЇЄҐA-Za-zа-яіїєґ'’]){acronym}(?![А-ЯІЇЄҐA-Za-zа-яіїєґ'’])", replace, row["target"])
        stats["acronyms_fixed"] += 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=ROOT / "data/e50_token_acronyms")
    parser.add_argument("--output", type=Path, default=ROOT / "data/e51_khvylyna")
    args = parser.parse_args()

    stats = Counter()
    train = read_rows(args.base_dir / "train.parquet")
    for row in train:
        fix_minutes(row, stats)
        fix_unread_acronyms(row, stats)
    dev = read_rows(args.base_dir / "dev-stratified.parquet")
    dev_stats = Counter()
    for row in dev:
        fix_minutes(row, dev_stats)
    args.output.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(train), args.output / "train.parquet", compression="zstd")
    pq.write_table(pa.Table.from_pylist(dev), args.output / "dev-stratified.parquet", compression="zstd")
    left = sum(1 for r in train if MINUTE.search(r["target"]) and not MINUTE.search(r["source"]))
    samples = [{"source": r["source"], "target": r["target"]} for r in train if "хвилин" in r["target"] and "°" in r["source"]][:3]
    report = {"rows": len(train), "train_fixes": dict(stats), "dev_fixes": dict(dev_stats), "minutes_left": left,
              "train_sha256": sha256_file(args.output / "train.parquet"), "samples": samples}
    write_json(args.output / "manifest.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "train_sha256"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
