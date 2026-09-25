"""Build a deterministic, category-aware Marian pilot with identity examples."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.verbalizer.config import load_yaml  # noqa: E402
from src.verbalizer.training.manifest import sha256_file, write_json  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs/data/pilot.yaml"
IDENTITY_DISALLOWED_RE = re.compile(r"\d|[A-Za-z]|[%$€₴@]")


def stable_score(seed: int, namespace: str, value: str) -> str:
    return hashlib.sha256(f"{seed}:{namespace}:{value}".encode("utf-8")).hexdigest()


def select_stratified(
    rows: list[dict[str, Any]],
    count: int,
    quota: int,
    seed: int,
) -> list[dict[str, Any]]:
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        for category in row.get("categories") or []:
            by_category[str(category)].append(row)
    selected_ids: set[str] = set()
    selected: list[dict[str, Any]] = []
    categories = sorted(by_category, key=lambda category: (len(by_category[category]), category))
    for category in categories:
        candidates = sorted(
            by_category[category],
            key=lambda row: stable_score(seed, f"category:{category}", str(row["id"])),
        )
        added = 0
        for row in candidates:
            if row["id"] in selected_ids:
                continue
            selected.append(row)
            selected_ids.add(row["id"])
            added += 1
            if added >= quota or len(selected) >= count:
                break
        if len(selected) >= count:
            break
    if len(selected) < count:
        remaining = sorted(
            (row for row in rows if row["id"] not in selected_ids),
            key=lambda row: stable_score(seed, "fill", str(row["id"])),
        )
        selected.extend(remaining[: count - len(selected)])
    return selected[:count]


def identity_rows(
    rows: list[dict[str, Any]],
    count: int,
    seed: int,
    namespace: str,
) -> list[dict[str, Any]]:
    candidates = sorted(
        rows,
        key=lambda row: stable_score(seed, namespace, str(row["id"])),
    )
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in candidates:
        text = str(row["target"]).strip()
        canonical = re.sub(r"\s+", " ", text.casefold())
        if (
            not text
            or len(text) < 20
            or len(text) > 1000
            or IDENTITY_DISALLOWED_RE.search(text)
            or canonical in seen
        ):
            continue
        seen.add(canonical)
        identity_id = hashlib.sha256(f"pilot-identity\0{row['id']}\0{text}".encode("utf-8")).hexdigest()
        result.append(
            {
                "id": identity_id,
                "source": text,
                "target": text,
                "categories": ["no_change"],
                "requires_change": False,
                "is_identity": True,
                "quality_tier": "A",
                "origin_id": str(row["id"]),
            }
        )
        if len(result) >= count:
            break
    if len(result) != count:
        raise ValueError(f"needed {count} identity rows, found {len(result)}")
    return result


def training_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(row["id"]),
        "source": str(row["source"]),
        "target": str(row["target"]),
        "categories": list(row.get("categories") or []),
        "requires_change": True,
        "is_identity": False,
        "quality_tier": str(row.get("quality_tier", "")),
        "origin_id": str(row["id"]),
    }


def write_parquet(path: Path, rows: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")
    return sha256_file(path)


def category_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(category for row in rows for category in row["categories"])
    return dict(sorted(counts.items()))


def build(config_path: Path) -> dict[str, Any]:
    config = load_yaml(config_path)
    seed = int(config["seed"])
    train_input = ROOT / config["train_input"]
    validation_input = ROOT / config["validation_input"]
    output_dir = ROOT / config["output_dir"]
    accepted = set(config["accepted_quality_tiers"])
    total_train = int(config["train_rows"])
    identity_count = round(total_train * float(config["identity_fraction"]))
    transformed_count = total_train - identity_count

    train_rows = pq.read_table(train_input).to_pylist()
    eligible_train = [
        row
        for row in train_rows
        if row.get("quality_tier") in accepted and bool(row.get("requires_change"))
    ]
    transformed = [
        training_row(row)
        for row in select_stratified(
            eligible_train,
            transformed_count,
            int(config["category_quota"]),
            seed,
        )
    ]
    identities = identity_rows(eligible_train, identity_count, seed, "train-identity")
    pilot_train = transformed + identities
    pilot_train.sort(key=lambda row: stable_score(seed, "train-order", row["id"]))

    validation_rows = pq.read_table(validation_input).to_pylist()
    eligible_validation = [
        row
        for row in validation_rows
        if row.get("quality_tier") in accepted and bool(row.get("requires_change"))
    ]
    pilot_validation = [training_row(row) for row in eligible_validation]
    pilot_validation.extend(
        identity_rows(
            eligible_validation,
            int(config["validation_identity_rows"]),
            seed,
            "validation-identity",
        )
    )
    pilot_validation.sort(key=lambda row: stable_score(seed, "validation-order", row["id"]))

    train_path = output_dir / "train.parquet"
    validation_path = output_dir / "validation.parquet"
    output_hashes = {
        "train": write_parquet(train_path, pilot_train),
        "validation": write_parquet(validation_path, pilot_validation),
    }
    manifest = {
        "config": config,
        "config_path": str(config_path),
        "source_files": {
            "train": {"path": str(train_input), "sha256": sha256_file(train_input)},
            "validation": {"path": str(validation_input), "sha256": sha256_file(validation_input)},
        },
        "outputs": {
            "train": {"path": str(train_path), "rows": len(pilot_train), "sha256": output_hashes["train"]},
            "validation": {
                "path": str(validation_path),
                "rows": len(pilot_validation),
                "sha256": output_hashes["validation"],
            },
        },
        "train": {
            "transformed_rows": len(transformed),
            "identity_rows": len(identities),
            "identity_fraction": len(identities) / len(pilot_train),
            "categories": category_counts(pilot_train),
        },
        "validation": {
            "transformed_rows": len(eligible_validation),
            "identity_rows": int(config["validation_identity_rows"]),
            "categories": category_counts(pilot_validation),
        },
        "policy": [
            "Only configured accepted quality tiers are eligible.",
            "Identity rows are derived only from targets in the same source split.",
            "Pilot evaluation does not use weak-golden or template-OOD labels for checkpoint selection.",
        ],
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    print(json.dumps(build(args.config), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

