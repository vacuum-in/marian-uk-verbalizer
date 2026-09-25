#!/usr/bin/env python3
"""Break the people-noun / large-magnitude association in the corpus.

"На 1001 людину" was read as 10 001, 100 101 or 1 000 001 depending on frame,
in both spaced and unspaced input. The cause is a co-occurrence bias, measured
over the E38 corpus: numbers immediately preceding a people-noun break down as

    <100        400 rows
    100-999     192
    1000-9999    21      <-- the gap
    10k-999k    132
    >=1M        483

The model has essentially never seen a four-digit count of people, but has seen
483 population-scale ones, so a people-noun pulls the magnitude upward.

This adds colon-readout frames pairing people-nouns with four-digit values. The
frame is chosen so the numeral stands in citation form, where num2words' output
is correct and no case or gender agreement is at stake -- the same safeguard
used in build_e38_bare_cardinals.py, and every pair is again cross-checked
against the independent parser in src/verbalizer/_audit_numparse.

This is deliberately an indirect fix. It supplies the missing magnitude
association; it does not teach oblique inflection such as
"На 1001 людину" -> "На тисячу одну людину", which would require authoring
case-inflected readings.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from num2words import num2words

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.verbalizer._audit_numparse import parse_numbers  # noqa: E402
from src.verbalizer.training.manifest import sha256_file, write_json  # noqa: E402

LABELS = ["Людей", "Осіб", "Учасників", "Мешканців", "Відвідувачів", "Кількість людей",
          "Кількість осіб", "Кількість учасників", "Пасажирів", "Працівників"]


def read_rows(path: Path) -> list[dict[str, Any]]:
    tb = pq.read_table(path, columns=["id", "source", "target", "categories", "provenance"])
    return [{"id": str(r["id"]), "source": str(r["source"]), "target": str(r["target"]),
             "categories": [str(c) for c in (r.get("categories") or [])],
             "provenance": str(r.get("provenance") or "unknown")} for r in tb.to_pylist()]


def write_rows(path: Path, rows) -> dict[str, Any]:
    pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")
    return {"path": str(path), "rows": len(rows), "sha256": sha256_file(path)}


def build(a: argparse.Namespace) -> dict[str, Any]:
    frozen = {json.loads(l)["source"] for l in a.frozen.read_text(encoding="utf-8").splitlines() if l.strip()}
    dev = set(pq.read_table(a.base_dir / "dev-stratified.parquet", columns=["source"]).column("source").to_pylist())
    base = read_rows(a.base_dir / "train.parquet")
    seen = {r["source"] for r in base}

    rng = random.Random(a.seed)
    values = sorted({rng.randint(1000, 9999) for _ in range(a.count)})
    added, stats = [], {"considered": 0, "cross_check_failed": 0, "frozen": 0, "dev": 0, "duplicate": 0}
    for n in values:
        word = num2words(n, lang="uk")
        if parse_numbers(word) != [n]:
            stats["cross_check_failed"] += 1
            continue
        for label in rng.sample(LABELS, 3):
            stats["considered"] += 1
            src, tgt = f"{label}: {n}.", f"{label}: {word}."
            if src in frozen: stats["frozen"] += 1
            elif src in dev: stats["dev"] += 1
            elif src in seen: stats["duplicate"] += 1
            else:
                seen.add(src)
                added.append({"id": f"people-count:{label}:{n}", "source": src, "target": tgt,
                              "categories": ["Голий фрагмент: число", "people_count_readout"],
                              "provenance": "generated_num2words_nominative_crosschecked"})
    combined = [*base, *added]
    a.output.mkdir(parents=True, exist_ok=True)
    rep = {"base": {"path": str(a.base_dir / "train.parquet"), "rows": len(base),
                    "sha256": sha256_file(a.base_dir / "train.parquet")},
           "generation": {**stats, "added": len(added), "distinct_values": len(values)},
           "output": {"train": write_rows(a.output / "train.parquet", combined),
                      "dev": write_rows(a.output / "dev-stratified.parquet",
                                        read_rows(a.base_dir / "dev-stratified.parquet"))}}
    write_json(a.output / "manifest.json", rep)
    return rep


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-dir", type=Path, default=ROOT / "data/e38_bare_cardinals")
    p.add_argument("--frozen", type=Path, default=ROOT / "reports/hf-uktn-frozen-3597.jsonl")
    p.add_argument("--output", type=Path, default=ROOT / "data/e39_people_counts")
    p.add_argument("--count", type=int, default=3000)
    p.add_argument("--seed", type=int, default=3901)
    print(json.dumps(build(p.parse_args()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
