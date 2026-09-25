"""Prepare Marian-ready splits from the existing local Ukrainian corpus.

This script is deliberately offline. It never calls Hugging Face or downloads
anything. The default inputs point at the already checked-in artifacts in the
hybrid verbalizer project.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.verbalizer.data.audit import canonical_source, quality_tier, row_flags

DEFAULT_SOURCE = (
    Path("/home/devops/verbolizer/hybrid-ukrainian-tts-verbalizer-openspec")
    / "verbolisation/data/main_clean_excluding_golden.parquet"
)
DEFAULT_GOLDEN = (
    Path("/home/devops/verbolizer/hybrid-ukrainian-tts-verbalizer-openspec")
    / "verbolisation/data/golden_5k.parquet"
)
DEFAULT_OOD = (
    Path("/home/devops/verbolizer/hybrid-ukrainian-tts-verbalizer-openspec")
    / "verbolisation/data/synthetic_verified_diverse_15k.parquet"
)
DEFAULT_OUTPUT = ROOT / "data/processed"
DEFAULT_MANIFEST = ROOT / "data/manifests/current_dataset.json"
DEFAULT_NEAR_DUPLICATE_QUARANTINE = ROOT / "data/manifests/current_near_duplicate_quarantine.json"
DATASET_VERSION = "current-hybrid-2026-07-27"
SEED = 42

PREFIX_RE = re.compile(r"^<verbalization>:?\s*", re.IGNORECASE)
NUMBER_RE = re.compile(r"(?<!\w)[+-−]?\d+(?:[.,]\d+)?(?!\w)")
DATE_RE = re.compile(r"(?<!\w)\d{1,2}[./-]\d{1,2}(?:[./-]\d{2,4})?(?!\w)")
TIME_RE = re.compile(r"(?<!\w)\d{1,2}:\d{2}(?!\w)")
RANGE_RE = re.compile(r"\d+(?:[.,]\d+)?\s*(?:-|–|—)\s*\d+(?:[.,]\d+)?")
ROMAN_RE = re.compile(r"\b[IVXLCDM]{1,8}\b")
LATIN_RE = re.compile(r"[A-Za-z]")
EMAIL_RE = re.compile(r"\b[^\s@]+@[^\s@]+\b")
UNIT_RE = re.compile(
    r"(?<!\w)(?:%|°\s*[CFС]|(?:к|м|с|г|л|мл|см|мм|км|кг|мг|Вт|кВт|МБ|ГБ|ТБ|MB|GB|TB|W|kW))(?!\w)",
    re.IGNORECASE,
)
MONTH_RE = re.compile(
    r"\b(?:січня|лютого|березня|квітня|травня|червня|липня|серпня|вересня|жовтня|листопада|грудня)\b",
    re.IGNORECASE,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = PREFIX_RE.sub("", text)
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def canonical_text(value: str) -> str:
    value = value.casefold()
    value = re.sub(r"[\W_]+", " ", value, flags=re.UNICODE)
    return re.sub(r"\s+", " ", value).strip()


def template_fingerprint(source: str) -> str:
    value = source.casefold()
    value = DATE_RE.sub("<DATE>", value)
    value = TIME_RE.sub("<TIME>", value)
    value = NUMBER_RE.sub("<NUM>", value)
    value = re.sub(r"\s+", " ", value).strip()
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def row_id(source: str, target: str) -> str:
    value = f"{DATASET_VERSION}\0{source}\0{target}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def categories(source: str, target: str) -> list[str]:
    found: list[str] = []
    if NUMBER_RE.search(source):
        found.append("integer" if not re.search(r"[.,]\d+", source) else "decimal")
    if re.search(r"(?<!\w)[−-]\d", source):
        found.append("negative_number")
    if "%" in source:
        found.append("percentage")
    if re.search(r"\b\d{4}\b", source):
        found.append("year")
    if DATE_RE.search(source):
        found.append("numeric_date")
    if MONTH_RE.search(source):
        found.append("full_date")
    if TIME_RE.search(source):
        found.append("time")
    if RANGE_RE.search(source):
        found.append("numeric_range")
    if re.search(r"[$€₴]|\b(?:USD|EUR|UAH|грн)\b", source, re.IGNORECASE):
        found.append("currency")
    if UNIT_RE.search(source):
        found.append("measurement_unit")
    if re.search(r"\b(?:м2|м3|кв\. м|куб\. м)\b", source, re.IGNORECASE):
        found.append("area_volume")
    if re.search(r"(?:°\s*[CFС]|\bградус)", source, re.IGNORECASE):
        found.append("temperature")
    if re.search(r"\b\d+(?:-й|й|ий|ий)\b|\b(?:I|II|III|IV|V|VI|VII|VIII|IX|X)\b", source):
        found.append("ordinal")
    if re.search(r"(?<!\w)\+?\d[\d ()-]{7,}\d(?!\w)", source):
        found.append("phone_number")
    if re.search(r"\b[A-Za-z]+[-_/]?\d|\b\d+[-_/][A-Za-z]", source):
        found.append("identifier")
    if re.search(r"\bv?\d+(?:\.\d+)+\b", source, re.IGNORECASE):
        found.append("version")
    if ROMAN_RE.search(source):
        found.append("roman_numeral")
    if re.search(r"\b[A-ZА-ЯІЇЄ][A-ZА-ЯІЇЄ0-9_-]{1,}\b", source):
        found.append("abbreviation")
    if LATIN_RE.search(source):
        found.append("mixed_latin")
    if len(found) > 1:
        found.append("multiple_spans")
    if source == target:
        found.append("no_change")
    return sorted(set(found))


def read_pairs(path: Path) -> list[tuple[str, str]]:
    table = pq.read_table(path, columns=["inputs", "labels"])
    inputs = table.column("inputs").to_pylist()
    labels = table.column("labels").to_pylist()
    return [(normalize_text(source), normalize_text(target)) for source, target in zip(inputs, labels, strict=True)]


def canonical_rows(pairs: list[tuple[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source, target in pairs:
        source_key = canonical_text(source)
        rows.append(
            {
                "id": row_id(source, target),
                "source": source,
                "target": target,
                "source_nfc": source,
                "target_nfc": target,
                "categories": categories(source, target),
                "requires_change": source != target,
                "protected_spans": [],
                "group_id": template_fingerprint(source),
                "source_sha256": hashlib.sha256(source_key.encode("utf-8")).hexdigest(),
                "quality_flags": [],
                "dataset_version": DATASET_VERSION,
            }
        )
    return rows


def write_rows(path: Path, rows: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path, compression="zstd")
    return sha256_file(path)


def load_quarantine_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    policy = json.loads(path.read_text(encoding="utf-8"))
    ids = policy.get("ids", [])
    if not isinstance(ids, list) or not all(isinstance(value, str) for value in ids):
        raise ValueError(f"invalid near-duplicate quarantine policy: {path}")
    return set(ids)


def choose_groups(rows: list[dict[str, Any]], target: int, excluded: set[str]) -> set[str]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        if row["group_id"] not in excluded:
            groups[row["group_id"]].append(index)
    ordered = sorted(
        groups.items(),
        key=lambda item: hashlib.sha256(f"{SEED}:{item[0]}".encode()).hexdigest(),
    )
    selected: set[str] = set()
    count = 0
    for group_id, members in ordered:
        if count >= target:
            break
        selected.add(group_id)
        count += len(members)
    return selected


def prepare(
    source: Path,
    golden: Path,
    ood: Path,
    output: Path,
    manifest_path: Path,
    near_duplicate_quarantine: Path,
) -> dict[str, Any]:
    source_rows = canonical_rows(read_pairs(source))
    source_pool_rows = len(source_rows)
    quarantine_ids = load_quarantine_ids(near_duplicate_quarantine)
    source_rows = [row for row in source_rows if row["id"] not in quarantine_ids]
    quarantined_rows = source_pool_rows - len(source_rows)
    weak_golden = canonical_rows(read_pairs(golden))
    template_ood = canonical_rows(read_pairs(ood))
    for row in source_rows + weak_golden + template_ood:
        flags = row_flags(row["source"], row["target"])
        row["quality_flags"] = flags
        row["quality_tier"] = quality_tier(flags)
    heldout_rows = weak_golden + template_ood
    heldout_groups = {row["group_id"] for row in heldout_rows}
    heldout_sources = {canonical_source(row["source"]) for row in heldout_rows}
    eligible_source_rows = [
        row
        for row in source_rows
        if row["group_id"] not in heldout_groups
        and canonical_source(row["source"]) not in heldout_sources
    ]

    validation_groups = choose_groups(eligible_source_rows, 10_000, set())
    test_groups = choose_groups(eligible_source_rows, 10_000, validation_groups)
    validation = [row for row in eligible_source_rows if row["group_id"] in validation_groups]
    test_id = [row for row in eligible_source_rows if row["group_id"] in test_groups]
    train = [
        row
        for row in eligible_source_rows
        if row["group_id"] not in validation_groups and row["group_id"] not in test_groups
    ]

    outputs = {
        "train": train,
        "validation": validation,
        "test_id": test_id,
        "test_golden_weak": weak_golden,
        "test_template_ood": template_ood,
    }
    output_hashes = {
        name: write_rows(output / f"{name}.parquet", rows) for name, rows in outputs.items()
    }
    manifest = {
        "dataset_version": DATASET_VERSION,
        "source_dataset": "skypro1111/ubertext-2-news-verbalized",
        "source_mode": "existing-local-artifacts-no-download",
        "source_files": {
            "clean_pool": {
                "path": str(source),
                "sha256": sha256_file(source),
                "original_rows": source_pool_rows,
                "retained_rows": len(source_rows),
                "quarantined_rows": quarantined_rows,
            },
            "weak_golden": {"path": str(golden), "sha256": sha256_file(golden), "rows": len(weak_golden)},
            "template_ood": {"path": str(ood), "sha256": sha256_file(ood), "rows": len(template_ood)},
        },
        "split_policy": {
            "seed": SEED,
            "group_key": "template_fingerprint(source)",
            "validation_target": 10_000,
            "test_id_target": 10_000,
            "reserved_group_count": len(heldout_groups),
            "reserved_source_count": len(heldout_sources),
            "held_out_weak_golden": True,
            "held_out_template_ood": True,
            "near_duplicate_quarantine": {
                "path": str(near_duplicate_quarantine),
                "sha256": sha256_file(near_duplicate_quarantine) if near_duplicate_quarantine.is_file() else None,
                "requested_ids": len(quarantine_ids),
                "applied_rows": quarantined_rows,
            },
            "human_gold_available": False,
        },
        "rows": {name: len(rows) for name, rows in outputs.items()},
        "output_sha256": output_hashes,
        "notes": [
            "Parquet only was copied/prepared; original raw shards remain in the source project.",
            "test_golden_weak is automatically validated and is not human gold.",
            "test_template_ood is generated verification data and is kept outside training.",
        ],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN)
    parser.add_argument("--ood", type=Path, default=DEFAULT_OOD)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--near-duplicate-quarantine", type=Path, default=DEFAULT_NEAR_DUPLICATE_QUARANTINE)
    args = parser.parse_args()
    for path in (args.source, args.golden, args.ood):
        if not path.is_file():
            raise FileNotFoundError(path)
    result = prepare(
        args.source,
        args.golden,
        args.ood,
        args.output,
        args.manifest,
        args.near_duplicate_quarantine,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
