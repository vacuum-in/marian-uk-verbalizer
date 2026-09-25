#!/usr/bin/env python3
"""Materialize raw training pairs for the routing-only pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.verbalizer.numeric import (  # noqa: E402
    NUMERIC_POLICY_VERSION,
    detect_numeric_spans,
    numeric_component_hashes,
    numeric_policy_hash,
    preprocessing_hash,
    prepare_training_pair,
)
from src.verbalizer.training.manifest import sha256_file, write_json  # noqa: E402


def materialize(input_path: Path, output_path: Path, quarantine_path: Path) -> dict[str, Any]:
    rows = pq.read_table(input_path).to_pylist()
    accepted: list[dict[str, Any]] = []
    quarantine: list[dict[str, Any]] = []
    kinds: Counter[str] = Counter()
    for row in rows:
        source = str(row["source"])
        target = str(row["target"])
        marked_source, marked_target, missing = prepare_training_pair(source, target)
        spans = detect_numeric_spans(source)
        kinds.update(span.kind.value for span in spans)
        accepted.append(
            {
                **row,
                "source": marked_source,
                "target": marked_target,
                "raw_source": source,
                "raw_target": target,
                "numeric_policy_version": NUMERIC_POLICY_VERSION,
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(accepted), output_path, compression="zstd")
    quarantine_path.parent.mkdir(parents=True, exist_ok=True)
    quarantine_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in quarantine),
        encoding="utf-8",
    )
    return {
        "input": {"path": str(input_path), "sha256": sha256_file(input_path), "rows": len(rows)},
        "output": {"path": str(output_path), "sha256": sha256_file(output_path), "rows": len(accepted)},
        "quarantine": {"path": str(quarantine_path), "rows": len(quarantine)},
        "numeric_kinds": dict(sorted(kinds.items())),
        "numeric_policy_version": NUMERIC_POLICY_VERSION,
        "numeric_policy_hash": numeric_policy_hash(),
        "preprocessing_hash": preprocessing_hash(),
        "component_hashes": numeric_component_hashes(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--quarantine", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args()
    report = materialize(args.input, args.output, args.quarantine)
    write_json(args.manifest, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
