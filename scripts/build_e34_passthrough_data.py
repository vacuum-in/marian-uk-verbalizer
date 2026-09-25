#!/usr/bin/env python3
"""Build the E34 corpus: E32 plus verified pass-through supervision.

E32 reached 79.48% but regressed on text that needs nothing done to it,
turning "дефіциту" into "еф ес терабайт" and "дід" into "ді".  Pass-through
accuracy fell 36/36 to 31/36.

The cause is corpus balance.  Only 0.23% of E32's training rows are identity
pairs, against 1.00% of the benchmark -- and in real input the large majority
of text needs no verbalization at all.  A model that has almost never been
shown "leave this alone" learns to always transform.

`data/target_domain/train.parquet` carries 15,000 rows explicitly marked
`requires_change=False`, all quality tier A, all self-consistent (no row marked
False has source != target), and none overlapping the frozen benchmark.  They
are already-verbalized Ukrainian: exactly the output-shaped input the model
must learn to leave untouched.

Nothing here decides what needs verbalizing.  The `requires_change` labels are
read from reviewed data; no heuristic is applied and no reading is authored.
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

CANONICAL_APOSTROPHE = "'"
APOSTROPHE_TABLE = dict.fromkeys(map(ord, "'’‘ʼ´`＇"), CANONICAL_APOSTROPHE)


def canonicalize(text: str) -> str:
    return text.translate(APOSTROPHE_TABLE)


def read_base(path: Path) -> list[dict[str, Any]]:
    table = pq.read_table(path, columns=["id", "source", "target", "categories", "provenance"])
    return [
        {
            "id": str(row["id"]),
            "source": str(row["source"]),
            "target": str(row["target"]),
            "categories": [str(c) for c in (row.get("categories") or [])],
            "provenance": str(row.get("provenance") or "unknown"),
        }
        for row in table.to_pylist()
    ]


def read_passthrough(path: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    table = pq.read_table(path, columns=["id", "source", "target", "requires_change", "quality_tier"])
    rows: list[dict[str, Any]] = []
    stats = {"scanned": 0, "requires_change_false": 0, "inconsistent": 0, "kept": 0}
    for record in table.to_pylist():
        stats["scanned"] += 1
        if record.get("requires_change") is not False:
            continue
        stats["requires_change_false"] += 1
        source = canonicalize(str(record["source"]))
        target = canonicalize(str(record["target"]))
        if source != target:
            # A row labelled "no change needed" whose target differs is a data
            # bug; drop it rather than teach a contradiction.
            stats["inconsistent"] += 1
            continue
        rows.append(
            {
                "id": f"passthrough:{record['id']}",
                "source": source,
                "target": target,
                "categories": ["Pass-through (без нормалізації)", "requires_change_false"],
                "provenance": "target_domain:requires_change_false",
            }
        )
        stats["kept"] += 1
    return rows, stats


def build(args: argparse.Namespace) -> dict[str, Any]:
    frozen_sources = {
        json.loads(line)["source"]
        for line in args.frozen.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    dev_sources = set(
        pq.read_table(args.base_dir / "dev-stratified.parquet", columns=["source"]).column("source").to_pylist()
    )

    base_rows = read_base(args.base_dir / "train.parquet")
    existing = {row["source"] for row in base_rows}

    passthrough, stats = read_passthrough(args.passthrough)
    filtered: list[dict[str, Any]] = []
    excluded = {"frozen": 0, "dev": 0, "duplicate": 0}
    for row in passthrough:
        if row["source"] in frozen_sources:
            excluded["frozen"] += 1
        elif row["source"] in dev_sources:
            excluded["dev"] += 1
        elif row["source"] in existing:
            excluded["duplicate"] += 1
        else:
            filtered.append(row)
            existing.add(row["source"])

    combined = [*base_rows, *filtered]
    identity = sum(1 for row in combined if row["source"] == row["target"])

    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "base": {
            "path": str(args.base_dir / "train.parquet"),
            "rows": len(base_rows),
            "sha256": sha256_file(args.base_dir / "train.parquet"),
        },
        "passthrough_source": {
            "path": str(args.passthrough),
            "sha256": sha256_file(args.passthrough),
            **stats,
            "excluded": excluded,
            "added": len(filtered),
        },
        "identity_rows_total": identity,
        "identity_fraction": round(identity / len(combined), 5),
        "output": {
            "train": write_rows(args.output / "train.parquet", combined),
            "dev": write_rows(args.output / "dev-stratified.parquet", read_base(args.base_dir / "dev-stratified.parquet")),
        },
    }
    write_json(args.output / "manifest.json", report)
    return report


def write_rows(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")
    return {"path": str(path), "rows": len(rows), "sha256": sha256_file(path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=ROOT / "data/e32_transliteration_scale")
    parser.add_argument("--passthrough", type=Path, default=ROOT / "data/target_domain/train.parquet")
    parser.add_argument("--frozen", type=Path, default=ROOT / "reports/hf-uktn-frozen-3597.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "data/e34_passthrough_balance")
    args = parser.parse_args()
    print(json.dumps(build(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
