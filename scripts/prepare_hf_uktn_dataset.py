"""Prepare leakage-safe Marian data from the pinned uk-text-normalization revision."""

from __future__ import annotations

import argparse
import hashlib
import json
import unicodedata
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.verbalizer.numeric import formula_supervision_valid  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
DATASET = "skypro1111/uk-text-normalization"
DEFAULT_REVISION = "576351fd5f63989ace9d92c3904580c94964e471"
FILES = {
    "train": "data/train.jsonl",
    "main": "data/test_domains.jsonl",
    "anchor": "data/test_anchor.jsonl",
    "transliteration": "data/eval_translit.jsonl",
    "agreement": "data/eval_agreement.jsonl",
    "misc": "data/eval_misc.jsonl",
    "feedback": "data/eval_feedback.jsonl",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_source(value: str) -> str:
    return " ".join(unicodedata.normalize("NFC", value).split()).casefold()


def download(name: str, path: Path, revision: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    url = (
        f"https://huggingface.co/datasets/{DATASET}/resolve/{revision}/{FILES[name]}"
    )
    request = urllib.request.Request(url, headers={"User-Agent": "marian-uk-verbalizer"})
    with urllib.request.urlopen(request, timeout=120) as response:
        path.write_bytes(response.read())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def row_id(dataset_version: str, split: str, source: str, target: str) -> str:
    value = f"{dataset_version}\0{split}\0{source}\0{target}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def write_rows(path: Path, rows: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")
    return sha256_file(path)


def build(args: argparse.Namespace) -> dict[str, Any]:
    dataset_version = f"hf-uk-text-normalization-e23-{args.revision[:12]}"
    raw_dir = args.output / "raw"
    paths = {name: raw_dir / f"{name}.jsonl" for name in FILES}
    for name, path in paths.items():
        download(name, path, args.revision)
    raw = {name: read_jsonl(path) for name, path in paths.items()}
    test_sources = {
        canonical_source(str(row["original"]))
        for name, rows in raw.items()
        if name != "train"
        for row in rows
    }
    targets_by_source: dict[str, set[str]] = defaultdict(set)
    categories_by_source: dict[str, set[str]] = defaultdict(set)
    original_by_source: dict[str, str] = {}
    for row in raw["train"]:
        source = str(row["original"])
        target = str(row["verbalized"])
        key = canonical_source(source)
        targets_by_source[key].add(target)
        category_values = row.get("combo_names") or [row.get("category") or "uncategorized"]
        categories_by_source[key].update(str(value) for value in category_values)
        original_by_source.setdefault(key, source)

    retained: list[tuple[str, str, list[str]]] = []
    dropped_test_overlap = 0
    dropped_ambiguous = 0
    dropped_formula_identity = 0
    for key, targets in targets_by_source.items():
        if key in test_sources:
            dropped_test_overlap += 1
            continue
        if len(targets) != 1:
            dropped_ambiguous += 1
            continue
        target = next(iter(targets))
        if not formula_supervision_valid(original_by_source[key], target):
            dropped_formula_identity += 1
            continue
        retained.append(
            (
                original_by_source[key],
                target,
                sorted(categories_by_source[key]),
            )
        )
    retained.sort(
        key=lambda value: hashlib.sha256(
            f"{args.seed}\0{value[0]}\0{value[1]}".encode("utf-8")
        ).hexdigest()
    )
    if not 0 < args.validation_rows < len(retained):
        raise ValueError("validation_rows must leave non-empty train and validation data")
    split_pairs = {
        "validation": retained[: args.validation_rows],
        "train": retained[args.validation_rows :],
    }
    split_rows = {
        split: [
            {
                "id": row_id(dataset_version, split, source, target),
                "source": source,
                "target": target,
                "categories": categories,
                "dataset_version": dataset_version,
                "quality_flags": [],
            }
            for source, target, categories in pairs
        ]
        for split, pairs in split_pairs.items()
    }
    train_keys = {canonical_source(row["source"]) for row in split_rows["train"]}
    validation_keys = {canonical_source(row["source"]) for row in split_rows["validation"]}
    leakage = {
        "train_validation_source_overlap": len(train_keys & validation_keys),
        "train_frozen_test_source_overlap": len(train_keys & test_sources),
        "validation_frozen_test_source_overlap": len(validation_keys & test_sources),
    }
    if any(leakage.values()):
        raise RuntimeError(f"source leakage detected: {leakage}")
    output_hashes = {
        split: write_rows(args.output / f"{split}.parquet", rows)
        for split, rows in split_rows.items()
    }
    manifest = {
        "dataset": DATASET,
        "revision": args.revision,
        "dataset_version": dataset_version,
        "seed": args.seed,
        "raw_files": {
            name: {"path": str(path), "rows": len(raw[name]), "sha256": sha256_file(path)}
            for name, path in paths.items()
        },
        "rows": {split: len(rows) for split, rows in split_rows.items()},
        "frozen_test_rows": sum(len(rows) for name, rows in raw.items() if name != "train"),
        "drops": {
            "test_source_overlap": dropped_test_overlap,
            "ambiguous_source_targets": dropped_ambiguous,
            "formula_identity_failure": dropped_formula_identity,
        },
        "leakage": leakage,
        "output_sha256": output_hashes,
        "policy": [
            "All published test files are frozen and excluded from training and validation.",
            "Conflicting targets for the same normalized source are excluded.",
            "The prepared corpus retains raw source text; production-pipeline materialization is a separate stage.",
        ],
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data/hf_uktn_e23")
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--validation-rows", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260819)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(build(parse_args()), ensure_ascii=False, indent=2, sort_keys=True))
