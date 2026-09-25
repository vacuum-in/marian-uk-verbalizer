"""Build a leakage-safe long-form curriculum from the verified synthetic oracle.

The curriculum uses new random seeds and generic family weighting.  Frozen
acceptance and user-regression rows are loaded only to reject source overlap;
their targets and model predictions are never added to training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
SIBLING_ROOT = ROOT.parent / "hybrid-ukrainian-tts-verbalizer-openspec"
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(SIBLING_ROOT))
sys.path.insert(0, str(SIBLING_ROOT / "src"))

from prepare_current_dataset import (  # noqa: E402
    canonical_rows,
    canonical_text,
    normalize_text,
    sha256_file,
    write_rows,
)
from training.generate_synthetic_diverse import (  # noqa: E402
    COUNT_NOUNS,
    DIVERSE_GENERATORS,
    FLIGHT_PREFIXES,
    LETTER_NAMES,
    Clause,
    GeneratedRow,
    _plural_form,
    _spell_identifier,
    _validate_row,
    noun_phrase,
)
from uk_tts_verbalizer.rules.numbers import verbalize_cardinal  # noqa: E402


DEFAULT_EXISTING_TRAIN = ROOT / "data/target_domain/train.parquet"
DEFAULT_EXISTING_VALIDATION = ROOT / "data/target_domain/validation.parquet"
DEFAULT_ACCEPTANCE = ROOT / "data/processed/test_template_ood.parquet"
DEFAULT_REGRESSIONS = ROOT / "data/challenge/user_target_regressions.jsonl"
DEFAULT_OUTPUT = ROOT / "data/target_domain_hard"
DATASET_VERSION = "structured-spoken-hard-v2-2026-07-31"
SEQUENCE_FAMILIES = ("phone", "flight", "order_identifier", "batch_code")


def repeated_sequence_clause(rng: random.Random) -> Clause:
    """Generate an identifier with a guaranteed run of repeated digits."""
    kind = rng.choices(("phone", "flight", "batch"), weights=(60, 20, 20), k=1)[0]
    repeated = str(rng.randint(0, 9))
    if kind == "phone":
        operator = rng.choice((50, 63, 67, 68, 73, 93, 95, 96, 97, 98, 99))
        tail = [str(rng.randint(0, 9)) for _ in range(7)]
        start = rng.randint(0, 4)
        tail[start : start + 3] = [repeated] * 3
        tail_text = "".join(tail)
        raw = f"+380 ({operator}) {tail_text[:3]}-{tail_text[3:5]}-{tail_text[5:]}"
        digits = f"380{operator}{tail_text}"
        spoken = "плюс " + " ".join(verbalize_cardinal(int(digit)) for digit in digits)
        return Clause(
            f"Телефон для зв’язку: {raw}.",
            f"Телефон для зв’язку: {spoken}.",
            "phone_repetition",
            (spoken,),
            ("numeric_sequence_exact", "repeated_digit_run"),
        )
    if kind == "flight":
        prefix, spoken_prefix = rng.choice(tuple(FLIGHT_PREFIXES.items()))
        width = rng.randint(3, 5)
        digits = [str(rng.randint(0, 9)) for _ in range(width)]
        start = rng.randint(0, width - 3)
        digits[start : start + 3] = [repeated] * 3
        if digits[0] == "0":
            digits[0] = str(rng.randint(1, 9))
        number = "".join(digits)
        delay = rng.choice((5, 15, 20, 45, 90, 120))
        delay_words = noun_phrase(
            delay, ("хвилина", "хвилини", "хвилин"), feminine=True
        )
        spoken_number = " ".join(verbalize_cardinal(int(digit)) for digit in number)
        return Clause(
            f"Рейс {prefix}{number} затримується на {delay} хв.",
            f"Рейс {spoken_prefix} {spoken_number} затримується на {delay_words}.",
            "flight_repetition",
            (spoken_prefix, spoken_number, delay_words),
            ("numeric_sequence_exact", "repeated_digit_run"),
        )
    prefix = rng.choice(tuple(key for key in LETTER_NAMES if key.isascii()))
    code = f"{prefix}{repeated * 3}-{rng.randint(0, 9)}"
    spoken = _spell_identifier(code)
    return Clause(
        f"Код партії: {code}.",
        f"Код партії: {spoken}.",
        "batch_repetition",
        (spoken,),
        ("numeric_sequence_exact", "repeated_digit_run"),
    )


def low_grouped_count_clause(rng: random.Random) -> Clause:
    value = rng.randint(1_000, 99_999)
    forms, feminine = rng.choice(COUNT_NOUNS)
    style = rng.choice(("space", "nbsp", "comma"))
    separator = {"space": " ", "nbsp": "\u00a0", "comma": ","}[style]
    digits = f"{value:,}".replace(",", separator)
    source_noun = _plural_form(value, forms)
    spoken = noun_phrase(value, forms, feminine=feminine)
    return Clause(
        f"У системі збережено {digits} {source_noun}.",
        f"У системі збережено {spoken}.",
        "low_grouped_count",
        (spoken,),
        (f"low_grouping_{style}", "grouped_commas") if style == "comma" else (f"low_grouping_{style}",),
    )


def low_hryvnia_clause(rng: random.Random) -> Clause:
    value = rng.choice((2, 2, 2, 1, 3, 4, 5, *range(6, 100)))
    spoken = noun_phrase(
        value, ("гривня", "гривні", "гривень"), feminine=True
    )
    return Clause(
        f"Клієнт оплатив ₴{value} за партію обладнання.",
        f"Клієнт оплатив {spoken} за партію обладнання.",
        "low_hryvnia",
        (spoken,),
        ("currency_hryvnia", "low_value_morphology"),
    )


def source_key(value: str) -> str:
    return canonical_text(normalize_text(value))


def parquet_rows(path: Path) -> list[dict[str, Any]]:
    return pq.read_table(path).to_pylist()


def parquet_sources(path: Path) -> set[str]:
    return {source_key(str(row["source"])) for row in parquet_rows(path)}


def regression_sources(path: Path) -> set[str]:
    sources: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                sources.add(source_key(str(json.loads(line)["source"])))
    return sources


def generated_id(split: str, source: str, target: str) -> str:
    value = f"{DATASET_VERSION}\0{split}\0{source}\0{target}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def generate_hard_rows(
    count: int,
    seed: int,
    forbidden_sources: set[str],
    anchor_policy: str,
) -> tuple[list[GeneratedRow], dict[str, int]]:
    """Generate four-clause rows anchored by coordinates or exact sequences."""
    rng = random.Random(seed)
    all_families = tuple(DIVERSE_GENERATORS)
    generated: list[GeneratedRow] = []
    seen = set(forbidden_sources)
    attempts = 0
    drops: Counter[str] = Counter()
    while len(generated) < count:
        attempts += 1
        if attempts > count * 40:
            raise RuntimeError("could not generate enough unique hard-curriculum rows")
        if anchor_policy == "residual":
            primary = all_families[len(generated) % len(all_families)]
            clauses = [
                repeated_sequence_clause(rng),
                low_grouped_count_clause(rng),
                low_hryvnia_clause(rng),
                DIVERSE_GENERATORS[primary](rng),
            ]
            rng.shuffle(clauses)
        elif anchor_policy == "both":
            primary = all_families[len(generated) % len(all_families)]
            anchors = ["coordinates", rng.choice(SEQUENCE_FAMILIES)]
            if primary not in anchors:
                anchors.append(primary)
        elif len(generated) % 2 == 0:
            anchors = ["coordinates", rng.choice(SEQUENCE_FAMILIES)]
        else:
            primary = all_families[(len(generated) // 2) % len(all_families)]
            anchor = rng.choice(("coordinates", *SEQUENCE_FAMILIES))
            anchors = [primary, anchor] if primary != anchor else [primary]
        if anchor_policy != "residual":
            candidates = tuple(name for name in all_families if name not in anchors)
            selected = [*anchors, *rng.sample(candidates, k=4 - len(anchors))]
            rng.shuffle(selected)
            clauses = [DIVERSE_GENERATORS[name](rng) for name in selected]
        row = GeneratedRow(
            inputs=" ".join(clause.source for clause in clauses),
            labels=" ".join(clause.target for clause in clauses),
            families=tuple(clause.family for clause in clauses),
            tags=tuple(tag for clause in clauses for tag in clause.tags),
        )
        key = source_key(row.inputs)
        if key in seen:
            drops["forbidden_or_duplicate_source"] += 1
            continue
        _validate_row(row, clauses)
        seen.add(key)
        generated.append(row)
    drops["attempts"] = attempts
    return generated, dict(drops)


def stamp_generated(rows: list[GeneratedRow], split: str) -> list[dict[str, Any]]:
    stamped = canonical_rows(
        [(normalize_text(row.inputs), normalize_text(row.labels)) for row in rows]
    )
    for output, generated in zip(stamped, rows, strict=True):
        output["id"] = generated_id(split, output["source"], output["target"])
        output["dataset_version"] = DATASET_VERSION
        output["quality_flags"] = []
        output["quality_tier"] = "A"
        output["generator_families"] = list(generated.families)
        output["generator_tags"] = list(generated.tags)
    return stamped


def build(args: argparse.Namespace) -> dict[str, Any]:
    if args.train_rows < 1 or args.validation_rows < 1:
        raise ValueError("train/validation row counts must be positive")
    existing_train = parquet_rows(args.existing_train)
    existing_validation = parquet_rows(args.existing_validation)
    train_sources = {source_key(str(row["source"])) for row in existing_train}
    validation_sources = {source_key(str(row["source"])) for row in existing_validation}
    acceptance_sources = parquet_sources(args.acceptance)
    user_regression_sources = regression_sources(args.regressions)
    protected_sources = acceptance_sources | user_regression_sources

    hard_validation, validation_drops = generate_hard_rows(
        args.validation_rows,
        args.validation_seed,
        protected_sources | train_sources | validation_sources,
        args.anchor_policy,
    )
    hard_validation_rows = stamp_generated(hard_validation, "validation")
    hard_validation_sources = {
        source_key(str(row["source"])) for row in hard_validation_rows
    }
    hard_train, train_drops = generate_hard_rows(
        args.train_rows,
        args.train_seed,
        protected_sources | train_sources | validation_sources | hard_validation_sources,
        args.anchor_policy,
    )
    hard_train_rows = stamp_generated(hard_train, "train")

    combined_train = [*existing_train, *hard_train_rows]
    combined_validation = [*existing_validation, *hard_validation_rows]
    rng = random.Random(args.shuffle_seed)
    rng.shuffle(combined_train)

    output_hashes = {
        "train": write_rows(args.output / "train.parquet", combined_train),
        "validation": write_rows(args.output / "validation.parquet", combined_validation),
        "hard_train": write_rows(args.output / "hard_train.parquet", hard_train_rows),
        "hard_validation": write_rows(
            args.output / "hard_validation.parquet", hard_validation_rows
        ),
    }

    final_train_sources = {source_key(str(row["source"])) for row in combined_train}
    final_validation_sources = {
        source_key(str(row["source"])) for row in combined_validation
    }
    leakage = {
        "train_validation_source_overlap": len(
            final_train_sources & final_validation_sources
        ),
        "train_acceptance_source_overlap": len(final_train_sources & acceptance_sources),
        "train_regression_source_overlap": len(
            final_train_sources & user_regression_sources
        ),
        "validation_acceptance_source_overlap": len(
            final_validation_sources & acceptance_sources
        ),
        "validation_regression_source_overlap": len(
            final_validation_sources & user_regression_sources
        ),
    }
    if any(leakage.values()):
        raise RuntimeError(f"dataset leakage detected: {leakage}")

    family_counts = Counter(
        family for row in hard_train for family in row.families
    )
    validation_family_counts = Counter(
        family for row in hard_validation for family in row.families
    )
    manifest = {
        "dataset_version": DATASET_VERSION,
        "policy_profile": "spoken-structured-domain-hard-curriculum",
        "anchor_policy": args.anchor_policy,
        "seeds": {
            "train": args.train_seed,
            "validation": args.validation_seed,
            "shuffle": args.shuffle_seed,
        },
        "sources": {
            "existing_train": {
                "path": str(args.existing_train),
                "sha256": sha256_file(args.existing_train),
                "rows": len(existing_train),
            },
            "existing_validation": {
                "path": str(args.existing_validation),
                "sha256": sha256_file(args.existing_validation),
                "rows": len(existing_validation),
            },
            "acceptance_exclusion_only": {
                "path": str(args.acceptance),
                "sha256": sha256_file(args.acceptance),
                "rows": len(acceptance_sources),
            },
            "regressions_exclusion_only": {
                "path": str(args.regressions),
                "sha256": sha256_file(args.regressions),
                "rows": len(user_regression_sources),
            },
            "oracle": str(
                SIBLING_ROOT / "training/generate_synthetic_diverse.py"
            ),
        },
        "rows": {
            "hard_train": len(hard_train_rows),
            "hard_validation": len(hard_validation_rows),
            "combined_train": len(combined_train),
            "combined_validation": len(combined_validation),
        },
        "hard_train_family_clause_counts": dict(sorted(family_counts.items())),
        "hard_validation_family_clause_counts": dict(
            sorted(validation_family_counts.items())
        ),
        "drops": {"train": train_drops, "validation": validation_drops},
        "leakage": leakage,
        "output_sha256": output_hashes,
        "notes": [
            "Every generated curriculum row has four clauses.",
            (
                "Every row includes both coordinate and exact numeric-sequence anchors."
                if args.anchor_policy == "both"
                else "Every row includes repeated digits, a low grouped count, low-value hryvnia, and one coverage clause."
                if args.anchor_policy == "residual"
                else "Rows alternate generic coordinate and exact numeric-sequence anchors."
            ),
            "Frozen acceptance/regression targets and predictions are never read.",
            "Frozen sources are read only to enforce zero canonical-source overlap.",
        ],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--existing-train", type=Path, default=DEFAULT_EXISTING_TRAIN)
    parser.add_argument(
        "--existing-validation", type=Path, default=DEFAULT_EXISTING_VALIDATION
    )
    parser.add_argument("--acceptance", type=Path, default=DEFAULT_ACCEPTANCE)
    parser.add_argument("--regressions", type=Path, default=DEFAULT_REGRESSIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train-rows", type=int, default=60_000)
    parser.add_argument("--validation-rows", type=int, default=6_000)
    parser.add_argument("--train-seed", type=int, default=202_607_311)
    parser.add_argument("--validation-seed", type=int, default=202_607_312)
    parser.add_argument("--shuffle-seed", type=int, default=202_607_313)
    parser.add_argument(
        "--anchor-policy",
        choices=("alternating", "both", "residual"),
        default="alternating",
    )
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(build(parse_args()), ensure_ascii=False, indent=2, sort_keys=True))
