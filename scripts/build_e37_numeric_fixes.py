#!/usr/bin/env python3
"""Build the E37 corpus: drop ungrammatical agreement rows, close the bare-number gap.

Two defects were found by auditing 1,000 random numeric rows (see
scripts/audit_numeric_targets.py). Numeric *values* were already correct --
99.21% measured, and every residual mismatch was an artefact of the audit tool
rather than the data. The defects are elsewhere:

1.  A synthetic template dropped a random number into a fixed string carrying
    "людей" plus a plural verb: "У черзі 171 людей чекають...". That is only
    grammatical when the number ends in 5-9 or 0. The bad rows are DROPPED
    rather than rewritten -- correcting them would mean authoring a reading,
    and dropping removes the wrong supervision without inventing anything.

2.  Bare four-digit numbers in the corpus are overwhelmingly years (2023, 2022,
    2021, ...), while cardinals are written with thousands separators. The model
    therefore learned "bare 4-digit => year" and misread 1001 and 3001 as
    magnitudes. This is closed by re-emitting existing rows with the separators
    stripped from the SOURCE while keeping the already-reviewed TARGET, so the
    model sees both spellings for the same verified reading. No target is
    generated here.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.verbalizer.training.manifest import sha256_file, write_json  # noqa: E402

GEN_PL = (
    r"(людей|осіб|дітей|учнів|студентів|працівників|мешканців|пасажирів|"
    r"спортсменів|гравців|відвідувачів|солдатів|глядачів|учасників)"
)
# "один"/"одна" + genitive plural, and 2-4 + genitive plural, in subject position.
BAD_ONE = re.compile(r"\bодин\s+" + GEN_PL + r"\b")
BAD_TWO_FOUR = re.compile(r"\b(два|дві|три|чотири)\s+" + GEN_PL + r"\b")
SUBJECT_VERB = re.compile(r"\b(чекают[ья]|стоять|прийшли|мешкають|працюють|живуть|голосують)\b")

# thousands groups: 1 001, 25 100 000
GROUP = re.compile(r"(?<![\d.,])(\d{1,3}(?:[   ]\d{3})+)(?![\d,])")


def strip_groups(text: str) -> str:
    return GROUP.sub(lambda m: re.sub(r"[   ]", "", m.group(1)), text)


def read_rows(path: Path) -> list[dict[str, Any]]:
    table = pq.read_table(path, columns=["id", "source", "target", "categories", "provenance"])
    return [
        {
            "id": str(r["id"]),
            "source": str(r["source"]),
            "target": str(r["target"]),
            "categories": [str(c) for c in (r.get("categories") or [])],
            "provenance": str(r.get("provenance") or "unknown"),
        }
        for r in table.to_pylist()
    ]


def write_rows(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")
    return {"path": str(path), "rows": len(rows), "sha256": sha256_file(path)}


def build(args: argparse.Namespace) -> dict[str, Any]:
    frozen = {
        json.loads(l)["source"]
        for l in args.frozen.read_text(encoding="utf-8").splitlines()
        if l.strip()
    }
    dev = set(
        pq.read_table(args.base_dir / "dev-stratified.parquet", columns=["source"]).column("source").to_pylist()
    )
    rows = read_rows(args.base_dir / "train.parquet")

    kept, dropped = [], []
    for r in rows:
        t = r["target"]
        if SUBJECT_VERB.search(t) and (BAD_ONE.search(t) or BAD_TWO_FOUR.search(t)):
            dropped.append(r)
            continue
        kept.append(r)

    seen = {r["source"] for r in kept}
    added, skipped = [], {"frozen": 0, "dev": 0, "duplicate": 0, "unchanged": 0}
    for r in kept:
        stripped = strip_groups(r["source"])
        if stripped == r["source"]:
            skipped["unchanged"] += 1
            continue
        if stripped in frozen:
            skipped["frozen"] += 1
        elif stripped in dev:
            skipped["dev"] += 1
        elif stripped in seen:
            skipped["duplicate"] += 1
        else:
            seen.add(stripped)
            added.append(
                {
                    **r,
                    "id": f"{r['id']}:nosep",
                    "source": stripped,
                    "provenance": f"{r['provenance']}+separator_stripped",
                }
            )

    combined = [*kept, *added]
    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "base": {
            "path": str(args.base_dir / "train.parquet"),
            "rows": len(rows),
            "sha256": sha256_file(args.base_dir / "train.parquet"),
        },
        "agreement_rows_dropped": len(dropped),
        "dropped_examples": [{"source": d["source"], "target": d["target"]} for d in dropped[:10]],
        "separator_stripped_added": len(added),
        "separator_skipped": skipped,
        "output": {
            "train": write_rows(args.output / "train.parquet", combined),
            "dev": write_rows(
                args.output / "dev-stratified.parquet", read_rows(args.base_dir / "dev-stratified.parquet")
            ),
        },
    }
    write_json(args.output / "manifest.json", report)
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-dir", type=Path, default=ROOT / "data/e36_cmudict_growth")
    p.add_argument("--frozen", type=Path, default=ROOT / "reports/hf-uktn-frozen-3597.jsonl")
    p.add_argument("--output", type=Path, default=ROOT / "data/e37_numeric_fixes")
    a = p.parse_args()
    r = build(a)
    print(json.dumps({k: v for k, v in r.items() if k != "dropped_examples"}, ensure_ascii=False, indent=2))
    print("\ndropped rows:")
    for d in r["dropped_examples"]:
        print(f"  SRC {d['source'][:88]}\n  TGT {d['target'][:88]}")


if __name__ == "__main__":
    main()
