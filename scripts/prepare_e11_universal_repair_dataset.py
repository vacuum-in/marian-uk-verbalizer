"""Build a balanced final repair mix without copying frozen test sources."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from num2words import num2words


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from prepare_current_dataset import canonical_rows, canonical_text, sha256_file, write_rows  # noqa: E402


DEFAULT_OUTPUT = ROOT / "data/e11_universal_repair"
DATASET_VERSION = "e11-universal-repair-v1-2026-07-31"


def read(path: Path) -> list[dict[str, Any]]:
    return pq.read_table(path).to_pylist()


def key(value: str) -> str:
    return canonical_text(value)


def spoken(number: int) -> str:
    return str(num2words(number, lang="uk")).replace("'", "’")


def unit(number: int) -> str:
    tail100 = number % 100
    tail10 = number % 10
    if 11 <= tail100 <= 14:
        return "градусів"
    if tail10 == 1:
        return "градус"
    if 2 <= tail10 <= 4:
        return "градуси"
    return "градусів"


def signed_rows(count: int, seed: int, split: str) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    pairs: list[tuple[str, str]] = []
    seen: set[str] = set()
    places = ("вранці", "увечері", "на узбережжі", "у долині", "в області")
    subjects = ("вітер", "потік повітря", "північний вітер")
    while len(pairs) < count:
        low = rng.randint(1, 39)
        high = rng.randint(0, 39)
        speed = rng.randint(1, 150)
        place = rng.choice(places)
        subject = rng.choice(subjects)
        source = f"Температура {place} -{low}–+{high} °C, {subject} сягав {speed} км/год."
        target = (
            f"Температура {place} від мінус {spoken(low)} до плюс {spoken(high)} "
            f"{unit(high)} Цельсія, {subject} сягав {spoken(speed)} кілометрів на годину."
        )
        if key(source) in seen:
            continue
        seen.add(key(source))
        pairs.append((source, target))
    values = canonical_rows(pairs)
    for row in values:
        payload = f"{DATASET_VERSION}\0{split}\0{row['source']}\0{row['target']}"
        row["id"] = hashlib.sha256(payload.encode()).hexdigest()
        row["dataset_version"] = DATASET_VERSION
        row["categories"] = ["signed_range", "temperature", "measurement_unit"]
        row["quality_tier"] = "A"
        row["quality_flags"] = []
    return values


def sample(values: list[dict[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    copy = list(values)
    random.Random(seed).shuffle(copy)
    return copy[: min(count, len(copy))]


def build(args: argparse.Namespace) -> dict[str, Any]:
    news_train = read(ROOT / "data/news_domain/train.parquet")
    news_validation = read(ROOT / "data/news_domain/validation.parquet")
    hard_train = sample(read(ROOT / "data/target_domain_hard_v3/train.parquet"), args.hard_rows, args.seed)
    hard_validation = sample(read(ROOT / "data/target_domain_hard_v3/validation.parquet"), args.hard_validation_rows, args.seed + 1)
    roman_train = read(ROOT / "data/roman_numerals/train.parquet")
    roman_validation = read(ROOT / "data/roman_numerals/validation.parquet")
    roman_test = read(ROOT / "data/roman_numerals/test.parquet")
    signed_train = signed_rows(args.signed_rows, args.seed + 2, "train")
    signed_train_keys = {key(str(row["source"])) for row in signed_train}
    signed_validation = [
        row
        for row in signed_rows(args.signed_validation_rows + 2_000, args.seed + 3, "validation")
        if key(str(row["source"])) not in signed_train_keys
    ][: args.signed_validation_rows]
    if len(signed_validation) != args.signed_validation_rows:
        raise RuntimeError("could not generate disjoint signed-range validation rows")
    regressions = [json.loads(line) for line in (ROOT / "data/challenge/user_target_regressions.jsonl").read_text(encoding="utf-8").splitlines() if line]

    roman_core = [row for row in roman_train if int(row.get("roman_value", 0)) <= 100]
    repeated_roman_core = [
        {**row, "id": f"{row['id']}-core-{repeat}"}
        for repeat in range(args.roman_core_repeats)
        for row in roman_core
    ]
    train = [*news_train, *hard_train, *roman_train, *repeated_roman_core, *signed_train]
    validation = [*news_validation, *hard_validation, *roman_validation, *signed_validation]
    random.Random(args.seed + 4).shuffle(train)
    forbidden = {key(str(row["source"])) for row in [*roman_test, *regressions]}
    train_keys = {key(str(row["source"])) for row in train}
    validation_keys = {key(str(row["source"])) for row in validation}
    leakage = {
        "train_validation_source_overlap": len(train_keys & validation_keys),
        "train_frozen_source_overlap": len(train_keys & forbidden),
        "validation_frozen_source_overlap": len(validation_keys & forbidden),
    }
    if any(leakage.values()):
        raise RuntimeError(f"E11 leakage: {leakage}")
    args.output.mkdir(parents=True, exist_ok=True)
    hashes = {"train": write_rows(args.output / "train.parquet", train), "validation": write_rows(args.output / "validation.parquet", validation)}
    components = {"news": len(news_train), "hard_structured": len(hard_train), "roman": len(roman_train), "roman_core_replay": len(repeated_roman_core), "signed_range": len(signed_train)}
    manifest = {
        "dataset_version": DATASET_VERSION,
        "seed": args.seed,
        "rows": {"train": len(train), "validation": len(validation)},
        "train_components": components,
        "leakage": leakage,
        "output_sha256": hashes,
        "sources": {
            "news_train_sha256": sha256_file(ROOT / "data/news_domain/train.parquet"),
            "hard_train_sha256": sha256_file(ROOT / "data/target_domain_hard_v3/train.parquet"),
            "roman_train_sha256": sha256_file(ROOT / "data/roman_numerals/train.parquet"),
        },
        "policy": ["Frozen Roman test and user-regression sources are exclusion-only.", "No inference-time Roman-number rewrite is used."],
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--hard-rows", type=int, default=120_000)
    parser.add_argument("--hard-validation-rows", type=int, default=5_000)
    parser.add_argument("--signed-rows", type=int, default=20_000)
    parser.add_argument("--signed-validation-rows", type=int, default=2_000)
    parser.add_argument("--roman-core-repeats", type=int, default=0)
    parser.add_argument("--seed", type=int, default=113)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(build(parse_args()), ensure_ascii=False, indent=2, sort_keys=True))
