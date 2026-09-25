#!/usr/bin/env python3
"""Evaluate all 3,597 frozen Hugging Face rows with numeric P0 gates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_marian import evaluate  # noqa: E402


SPLITS = ("main", "anchor", "transliteration", "agreement", "misc")
EXPECTED_ROWS = 3_597


def materialize_frozen_rows(raw_dir: Path, output: Path) -> int:
    rows: list[dict[str, object]] = []
    for split in SPLITS:
        path = raw_dir / f"{split}.jsonl"
        for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            raw = json.loads(line)
            rows.append(
                {
                    "id": f"{split}-{index:05d}",
                    "source": str(raw["original"]),
                    "target": str(raw["verbalized"]),
                    "categories": [split, str(raw.get("category") or "uncategorized")],
                }
            )
    if len(rows) != EXPECTED_ROWS:
        raise RuntimeError(f"frozen HF row count changed: expected {EXPECTED_ROWS}, got {len(rows)}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data/hf_uktn_e23/raw")
    parser.add_argument(
        "--data-artifact",
        type=Path,
        default=ROOT / "reports/hf-uktn-frozen-3597.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports/hf-uktn-numeric-release-3597.json",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--allow-incompatible-checkpoint", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()
    count = materialize_frozen_rows(args.raw_dir.resolve(), args.data_artifact.resolve())
    report = evaluate(
        args.checkpoint.resolve(),
        args.data_artifact.resolve(),
        args.output.resolve(),
        0,
        args.batch_size,
        args.num_beams,
        True,
        require_compatible_manifest=not args.allow_incompatible_checkpoint,
        allow_cpu=args.allow_cpu,
    )
    summary = {
        "rows": count,
        "runtime": report["runtime"],
        "numeric_release_gate": report["numeric_release_gate"],
        "numeric_model_rejected": report["metrics"]["overall"]["numeric_model_rejected"],
        "exact_match": report["metrics"]["overall"]["exact_match"],
        "numeric_failure_artifact": report["numeric_failure_artifact"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
