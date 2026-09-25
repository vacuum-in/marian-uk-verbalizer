"""Build an E13 mix that unlearns large-Roman bias in letter contexts."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from prepare_current_dataset import canonical_rows, canonical_text, sha256_file, write_rows  # noqa: E402
from prepare_e11_universal_repair_dataset import signed_rows  # noqa: E402


DEFAULT_OUTPUT = ROOT / "data/e13_letter_repair"
DATASET_VERSION = "letter-vs-roman-v2-2026-08-01"
LETTER_WORDS = {"C": "це", "D": "де"}


def read(path: Path) -> list[dict[str, Any]]:
    return pq.read_table(path).to_pylist()


def key(value: str) -> str:
    return canonical_text(value)


def sample(values: list[dict[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    copy = list(values)
    random.Random(seed).shuffle(copy)
    return copy[: min(count, len(copy))]


def letter_rows(split: str) -> list[dict[str, Any]]:
    cities = (
        "Києва", "Львова", "Одеси", "Харкова", "Дніпра", "Варшави",
        "Праги", "Берліна", "Відня", "Лондона", "Парижа", "Рима",
        "Мадрида", "Лісабона", "Стокгольма", "Осло", "Гельсінкі",
        "Таллінна", "Риги", "Вільнюса", "Кракова", "Брюсселя",
    )
    qualifiers = (
        "ранковий", "денний", "вечірній", "нічний", "регулярний",
        "додатковий", "чартерний", "міжнародний", "наступний", "останній",
    )
    pairs: list[tuple[str, str]] = []
    if split == "train":
        airport_templates = (
            "{q} рейс до {city} вилітає з терміналу {letter}.",
            "Пасажири на {q} рейс до {city} прямують до терміналу {letter}.",
            "Посадку на {q} рейс до {city} оголосили в терміналі {letter}.",
            "Для рейсу до {city} відкрито {q} вихід у терміналі {letter}.",
            "Табло направляє пасажирів до терміналу {letter} на {q} рейс до {city}.",
        )
        items = (
            "седан", "хетчбек", "універсал", "кросовер", "автомобіль",
            "пристрій", "сервер", "модуль", "компонент", "протокол",
            "документ", "сертифікат", "проєкт", "курс", "квиток",
            "вагон", "каюта", "сектор", "корпус", "матеріал",
            "препарат", "аналіз", "зразок", "пакет", "тариф",
        )
        generic_templates = (
            "{q} {item} належить до класу {letter}.",
            "Цей {q} {item} має категорію {letter}.",
            "Для об’єкта {item} встановлено рівень {letter}; статус {q}.",
            "У документації вказано клас {letter} для об’єкта {item}; тип {q}.",
            "Обрано варіант {letter} для об’єкта {item}; режим {q}.",
            "Позначення {letter} застосовано до об’єкта {item}; профіль {q}.",
        )
        for template in airport_templates:
            for qualifier in qualifiers:
                for city in cities:
                    for letter, spoken in LETTER_WORDS.items():
                        source = template.format(q=qualifier, city=city, letter=letter)
                        target = template.format(q=qualifier, city=city, letter=spoken)
                        pairs.append((source[:1].upper() + source[1:], target[:1].upper() + target[1:]))
        for template in generic_templates:
            for qualifier in qualifiers:
                for item in items:
                    for letter, spoken in LETTER_WORDS.items():
                        source = template.format(q=qualifier, item=item, letter=letter)
                        target = template.format(q=qualifier, item=item, letter=spoken)
                        pairs.append((source[:1].upper() + source[1:], target[:1].upper() + target[1:]))
        medical = (
            ("Лікар призначив вітамін D після огляду.", "Лікар призначив вітамін де після огляду."),
            ("Аналіз на гепатит C передали до лабораторії.", "Аналіз на гепатит це передали до лабораторії."),
            ("У висновку згадано вітамін C.", "У висновку згадано вітамін це."),
            ("Тест на гепатит D повторять завтра.", "Тест на гепатит де повторять завтра."),
        )
        pairs.extend(medical)
    else:
        templates = (
            "Відправлення до {city} відбудеться з терміналу {letter}; рейс {q}.",
            "До терміналу {letter} запросили пасажирів, що летять до {city}; рейс {q}.",
            "Модель у виконанні {q} віднесено до класу {letter}; напрямок {city}.",
            "Для категорії {letter} обрали профіль {q}; центр у місті {city}.",
        )
        for template in templates:
            for qualifier in qualifiers:
                for city in cities:
                    for letter, spoken in LETTER_WORDS.items():
                        source = template.format(q=qualifier, city=city, letter=letter)
                        target = template.format(q=qualifier, city=city, letter=spoken)
                        pairs.append((source[:1].upper() + source[1:], target[:1].upper() + target[1:]))
    rows = canonical_rows(pairs)
    for row in rows:
        payload = f"{DATASET_VERSION}\0{split}\0{row['source']}\0{row['target']}"
        row["id"] = hashlib.sha256(payload.encode()).hexdigest()
        row["dataset_version"] = DATASET_VERSION
        row["categories"] = ["mixed_latin", "letter_context"]
        row["quality_tier"] = "A"
        row["quality_flags"] = []
    return rows


def repeated(values: list[dict[str, Any]], repeats: int) -> list[dict[str, Any]]:
    return [
        {**row, "id": f"{row['id']}-repeat-{repeat}"}
        for repeat in range(repeats)
        for row in values
    ]


def build(args: argparse.Namespace) -> dict[str, Any]:
    news_train = sample(read(ROOT / "data/news_domain/train.parquet"), args.news_rows, args.seed)
    news_validation = read(ROOT / "data/news_domain/validation.parquet")
    hard_train = sample(read(ROOT / "data/target_domain_hard_v3/train.parquet"), args.hard_rows, args.seed + 1)
    hard_validation = sample(read(ROOT / "data/target_domain_hard_v3/validation.parquet"), args.hard_validation_rows, args.seed + 2)
    roman_train = read(ROOT / "data/roman_small/train.parquet")
    roman_validation = read(ROOT / "data/roman_small/validation.parquet")
    roman_test = read(ROOT / "data/roman_small/test.parquet")
    letters_train = letter_rows("train")
    letters_validation = letter_rows("validation")
    signed_train = signed_rows(args.signed_rows, args.seed + 3, "e13-train")
    signed_train_keys = {key(str(row["source"])) for row in signed_train}
    signed_validation = [
        row
        for row in signed_rows(
            args.signed_validation_rows + 2_000,
            args.seed + 4,
            "e13-validation",
        )
        if key(str(row["source"])) not in signed_train_keys
    ][: args.signed_validation_rows]
    if len(signed_validation) != args.signed_validation_rows:
        raise RuntimeError("could not generate disjoint signed-range validation rows")

    train = [
        *news_train,
        *hard_train,
        *repeated(roman_train, args.roman_repeats),
        *repeated(letters_train, args.letter_repeats),
        *signed_train,
    ]
    validation = [*news_validation, *hard_validation, *roman_validation, *letters_validation, *signed_validation]
    random.Random(args.seed + 5).shuffle(train)

    regressions = [
        json.loads(line)
        for line in (ROOT / "data/challenge/letter_context_regressions.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    forbidden = {key(str(row["source"])) for row in [*roman_test, *regressions]}
    train_keys = {key(str(row["source"])) for row in train}
    validation_keys = {key(str(row["source"])) for row in validation}
    leakage = {
        "train_validation_source_overlap": len(train_keys & validation_keys),
        "train_frozen_source_overlap": len(train_keys & forbidden),
        "validation_frozen_source_overlap": len(validation_keys & forbidden),
    }
    if any(leakage.values()):
        raise RuntimeError(f"E13 leakage: {leakage}")

    args.output.mkdir(parents=True, exist_ok=True)
    hashes = {
        "train": write_rows(args.output / "train.parquet", train),
        "validation": write_rows(args.output / "validation.parquet", validation),
    }
    manifest = {
        "dataset_version": DATASET_VERSION,
        "seed": args.seed,
        "rows": {"train": len(train), "validation": len(validation)},
        "train_components": {
            "news": len(news_train), "hard_structured": len(hard_train),
            "roman_small_replay": len(roman_train) * args.roman_repeats,
            "letter_context_replay": len(letters_train) * args.letter_repeats,
            "signed_range": len(signed_train),
        },
        "leakage": leakage,
        "output_sha256": hashes,
        "sources": {
            "news": sha256_file(ROOT / "data/news_domain/train.parquet"),
            "hard": sha256_file(ROOT / "data/target_domain_hard_v3/train.parquet"),
            "roman_small": sha256_file(ROOT / "data/roman_small/train.parquet"),
        },
        "policy": [
            "Roman training is capped at 39 and contains only I, V, and X glyphs.",
            "Explicit terminal-letter regressions and Roman-small test are exclusion-only.",
        ],
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--news-rows", type=int, default=50_000)
    parser.add_argument("--hard-rows", type=int, default=100_000)
    parser.add_argument("--hard-validation-rows", type=int, default=5_000)
    parser.add_argument("--roman-repeats", type=int, default=3)
    parser.add_argument("--letter-repeats", type=int, default=10)
    parser.add_argument("--signed-rows", type=int, default=20_000)
    parser.add_argument("--signed-validation-rows", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=131)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(build(parse_args()), ensure_ascii=False, indent=2, sort_keys=True))
