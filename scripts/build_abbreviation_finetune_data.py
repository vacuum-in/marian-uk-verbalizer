#!/usr/bin/env python3
"""Build a Marian fine-tuning mix from raw pairs and reviewed supervision.

The script has no pronunciation logic.  It copies reviewed source/target pairs
verbatim and only changes their sampling weight for fine-tuning.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.verbalizer.training.manifest import sha256_file, write_json  # noqa: E402


def read_reviewed_pairs(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def canonical_rows(path: Path) -> list[dict[str, Any]]:
    return [
        {
            "id": str(row["id"]),
            "source": str(row["source"]),
            "target": str(row["target"]),
            "categories": list(row.get("categories") or []),
            "provenance": "base_raw",
        }
        for row in pq.read_table(path, columns=["id", "source", "target", "categories"]).to_pylist()
    ]


def build(
    base_train: Path,
    base_validation: Path,
    reviewed_path: Path,
    output_dir: Path,
    supervision_repeat: int,
) -> dict[str, Any]:
    if supervision_repeat < 1:
        raise ValueError("supervision_repeat must be at least 1")
    reviewed = read_reviewed_pairs(reviewed_path)
    train = canonical_rows(base_train)
    validation = canonical_rows(base_validation)
    train_reviewed = [row for row in reviewed if row.get("split") == "train"]
    validation_reviewed = [row for row in reviewed if row.get("split") == "validation"]
    for row in train_reviewed:
        for repeat in range(supervision_repeat):
            train.append(
                {
                    "id": f"{row['id']}:repeat-{repeat:04d}",
                    "source": str(row["source"]),
                    "target": str(row["target"]),
                    "categories": list(row.get("categories") or []),
                    "provenance": "reviewed_supervision",
                }
            )
    validation.extend(
        {
            "id": str(row["id"]),
            "source": str(row["source"]),
            "target": str(row["target"]),
            "categories": list(row.get("categories") or []),
            "provenance": "reviewed_supervision",
        }
        for row in validation_reviewed
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.parquet"
    validation_path = output_dir / "validation.parquet"
    pq.write_table(pa.Table.from_pylist(train), train_path, compression="zstd")
    pq.write_table(pa.Table.from_pylist(validation), validation_path, compression="zstd")
    report = {
        "base": {
            "train": {"path": str(base_train), "rows": len(train) - len(train_reviewed) * supervision_repeat, "sha256": sha256_file(base_train)},
            "validation": {"path": str(base_validation), "rows": len(validation) - len(validation_reviewed), "sha256": sha256_file(base_validation)},
        },
        "reviewed_supervision": {
            "path": str(reviewed_path),
            "sha256": sha256_file(reviewed_path),
            "train_rows": len(train_reviewed),
            "validation_rows": len(validation_reviewed),
            "train_repeat": supervision_repeat,
        },
        "output": {
            "train": {"path": str(train_path), "rows": len(train), "sha256": sha256_file(train_path)},
            "validation": {"path": str(validation_path), "rows": len(validation), "sha256": sha256_file(validation_path)},
        },
    }
    write_json(output_dir / "manifest.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-train", type=Path, default=ROOT / "data/e24_pipeline_hf_plus_synthetic/train.parquet")
    parser.add_argument("--base-validation", type=Path, default=ROOT / "data/e24_pipeline_hf_plus_synthetic/validation.parquet")
    parser.add_argument("--reviewed", type=Path, default=ROOT / "data/curated/abbreviation-supervision.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "data/e27_abbreviation_routing")
    parser.add_argument("--supervision-repeat", type=int, default=512)
    args = parser.parse_args()
    print(json.dumps(build(args.base_train, args.base_validation, args.reviewed, args.output, args.supervision_repeat), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
