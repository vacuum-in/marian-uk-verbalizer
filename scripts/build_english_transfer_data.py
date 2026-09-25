#!/usr/bin/env python3
"""Build Marian training splits from the reviewed English-transducer corpus.

The transfer is intentionally literal: every transducer ``src`` and ``tgt``
pair becomes a Marian source/target pair.  No pronunciation, token, or lookup
rule is recreated here.
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


def read_parquet_rows(path: Path) -> list[dict[str, Any]]:
    return [
        {
            "id": str(row["id"]),
            "source": str(row["source"]),
            "target": str(row["target"]),
            "categories": list(row.get("categories") or []),
            "provenance": str(row.get("provenance") or "base_raw"),
        }
        for row in pq.read_table(path, columns=["id", "source", "target", "categories", "provenance"]).to_pylist()
    ]


def read_transducer_rows(path: Path, split: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        record = json.loads(line)
        rows.append(
            {
                "id": f"english-transducer:{split}:{record['example_id']}",
                "source": str(record["src"]),
                "target": str(record["tgt"]),
                "categories": [
                    "english_transducer_transfer",
                    str(record.get("entity_type") or "unknown"),
                    str(record.get("rendering_type") or "unknown"),
                ],
                "provenance": f"en_uk_transducer:{split}",
            }
        )
    return rows


def write_rows(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")
    return {"path": str(path), "rows": len(rows), "sha256": sha256_file(path)}


def build(
    base_train: Path,
    base_validation: Path,
    transducer_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    base_train_rows = read_parquet_rows(base_train)
    base_validation_rows = read_parquet_rows(base_validation)
    transducer = {
        split: read_transducer_rows(transducer_dir / f"{split}.jsonl", split)
        for split in ("train", "validation", "test")
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "base": {
            "train": {"path": str(base_train), "rows": len(base_train_rows), "sha256": sha256_file(base_train)},
            "validation": {"path": str(base_validation), "rows": len(base_validation_rows), "sha256": sha256_file(base_validation)},
        },
        "transducer": {
            split: {
                "path": str(transducer_dir / f"{split}.jsonl"),
                "rows": len(rows),
                "sha256": sha256_file(transducer_dir / f"{split}.jsonl"),
            }
            for split, rows in transducer.items()
        },
        "output": {
            "train": write_rows(output_dir / "train.parquet", [*base_train_rows, *transducer["train"]]),
            "validation": write_rows(output_dir / "validation.parquet", [*base_validation_rows, *transducer["validation"]]),
            "english_train": write_rows(output_dir / "english-train.parquet", transducer["train"]),
            "english_validation": write_rows(
                output_dir / "english-validation.parquet", transducer["validation"]
            ),
            "english_test": write_rows(output_dir / "english-test.parquet", transducer["test"]),
        },
    }
    write_json(output_dir / "manifest.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-train", type=Path, default=ROOT / "data/e27_abbreviation_routing/train.parquet")
    parser.add_argument("--base-validation", type=Path, default=ROOT / "data/e27_abbreviation_routing/validation.parquet")
    parser.add_argument("--transducer-dir", type=Path, default=ROOT.parent / "en-uk-transducer/data/splits/en-uk-combined-v1")
    parser.add_argument("--output", type=Path, default=ROOT / "data/e28_english_transfer")
    args = parser.parse_args()
    print(json.dumps(build(args.base_train, args.base_validation, args.transducer_dir, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
