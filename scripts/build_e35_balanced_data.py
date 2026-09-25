#!/usr/bin/env python3
"""Build the E35 corpus: pass-through and English supervision grown together.

E34b traded domains against each other. Adding 15,000 Ukrainian pass-through
rows bought +1.78 on the Ukrainian benchmark and gave back 5.38 on the English
holdout: the same rows that teach restraint also dilute transliteration.

The fix is not to choose. Three reviewed transducer splits were never pulled
into any Marian corpus -- v18-clean-three-provider, v17-three-provider and the
v3flashpro disagreement-resolution set -- and between them they carry tens of
thousands of pairs the model has never seen. Adding them alongside the
pass-through rows keeps both signals growing instead of competing.

Sources are reviewed transducer splits; nothing is authored here and no reading
is invented. Frozen and dev sources are excluded and the counts are recorded.
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

APOSTROPHE_TABLE = dict.fromkeys(map(ord, "'’‘ʼ´`＇"), "'")

EXTRA_SPLITS = [
    "en-uk-v18-clean-three-provider-prelim",
    "en-uk-v17-three-provider-prelim",
    "en-uk-v3flashpro-disagreements-fifth-1k",
]


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


def build(args: argparse.Namespace) -> dict[str, Any]:
    frozen = {
        json.loads(line)["source"]
        for line in args.frozen.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    dev = set(
        pq.read_table(args.base_dir / "dev-stratified.parquet", columns=["source"]).column("source").to_pylist()
    )
    base_rows = read_base(args.base_dir / "train.parquet")
    seen = {row["source"] for row in base_rows}

    added: list[dict[str, Any]] = []
    per_split: dict[str, dict[str, int]] = {}
    for split in EXTRA_SPLITS:
        path = args.splits_dir / split / "train.jsonl"
        stats = {"rows": 0, "added": 0, "frozen": 0, "dev": 0, "duplicate": 0}
        if not path.exists():
            per_split[split] = {**stats, "missing": 1}
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            src, tgt = record.get("src"), record.get("tgt")
            if not src or not tgt:
                continue
            stats["rows"] += 1
            source, target = canonicalize(str(src)), canonicalize(str(tgt))
            if source in frozen:
                stats["frozen"] += 1
            elif source in dev:
                stats["dev"] += 1
            elif source in seen:
                stats["duplicate"] += 1
            else:
                seen.add(source)
                stats["added"] += 1
                added.append(
                    {
                        "id": f"{split}:{record.get('example_id', len(added))}",
                        "source": source,
                        "target": target,
                        "categories": [
                            "english_transducer_transfer",
                            str(record.get("entity_type") or "unknown"),
                            split,
                        ],
                        "provenance": f"en_uk_transducer:{split}",
                    }
                )
        per_split[split] = stats

    combined = [*base_rows, *added]
    identity = sum(1 for row in combined if row["source"] == row["target"])
    english = sum(1 for row in combined if "english_transducer_transfer" in row["categories"])

    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "base": {
            "path": str(args.base_dir / "train.parquet"),
            "rows": len(base_rows),
            "sha256": sha256_file(args.base_dir / "train.parquet"),
        },
        "extra_splits": per_split,
        "english_rows_added": len(added),
        "totals": {
            "rows": len(combined),
            "identity_rows": identity,
            "identity_fraction": round(identity / len(combined), 5),
            "english_rows": english,
            "english_fraction": round(english / len(combined), 5),
        },
        "output": {
            "train": write_rows(args.output / "train.parquet", combined),
            "dev": write_rows(
                args.output / "dev-stratified.parquet", read_base(args.base_dir / "dev-stratified.parquet")
            ),
        },
    }
    write_json(args.output / "manifest.json", report)
    return report


def write_rows(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")
    return {"path": str(path), "rows": len(rows), "sha256": sha256_file(path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=ROOT / "data/e34_passthrough_balance")
    parser.add_argument(
        "--splits-dir", type=Path, default=ROOT.parent / "en-uk-transducer/data/splits"
    )
    parser.add_argument("--frozen", type=Path, default=ROOT / "reports/hf-uktn-frozen-3597.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "data/e35_balanced_growth")
    args = parser.parse_args()
    print(json.dumps(build(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
