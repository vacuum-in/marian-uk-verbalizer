#!/usr/bin/env python3
"""Add accusative-case supervision for "на N людину"-style counts.

E39 fixed every four-digit frame except one: a numeral ending in 1 governed by
an accusative preposition and followed by an animate noun --
"На 1001 людину." was still read as 101 / 10 001 / 1 000 001.

The corpus has no such rows to copy and no correct form the model already
produces, so supervision has to be generated. Readings are NOT authored here:

  * the numeral phrase comes from num2words (a declared project dependency);
  * only the TRAILING unit word is inflected, by pymorphy3 with the Ukrainian
    VESUM-derived dictionary. Inflecting every word breaks numeral-scale
    agreement -- it yields "три тисячу" and "п'ять тисячі" -- whereas leaving
    the scale in num2words' form preserves "три тисячі" and "п'ять тисяч";
  * scope is restricted to values ending in 1 (excluding the teens) with
    ANIMATE nouns, because masculine inanimate accusative takes "один день",
    not "одного день", and that distinction is not safely derivable here;
  * every generated pair is cross-checked against the independent parser in
    src/verbalizer/_audit_numparse and dropped unless the value round-trips.

Everything produced here is training data. Nothing runs at inference.
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
import pymorphy3
from num2words import num2words

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.verbalizer._audit_numparse import parse_numbers  # noqa: E402
from src.verbalizer.training.manifest import sha256_file, write_json  # noqa: E402

FEMININE = ["людину", "особу", "дитину", "жінку", "дівчину", "пацієнтку", "студентку"]
MASCULINE = ["учасника", "пасажира", "мешканця", "чоловіка", "працівника", "студента", "пацієнта"]
FRAMES = [
    "На {n} {noun}.", "на {n} {noun}", "Виділили на {n} {noun}.",
    "Розраховано на {n} {noun}.", "Кошти на {n} {noun}.", "Понад {n} {noun}.",
    "Субсидію нарахували на {n} {noun}.", "Квоту збільшили на {n} {noun}.",
]
MORPH = pymorphy3.MorphAnalyzer(lang="uk")
TAIL = {"один", "одна"}


def accusative(n: int, gender: str) -> str | None:
    words = num2words(n, lang="uk").split()
    if words[-1] not in TAIL:
        return None
    for parse in MORPH.parse(words[-1]):
        form = parse.inflect({"accs", gender})
        if form:
            return " ".join(words[:-1] + [form.word])
    return None


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
    values = [v for v in range(1001, 100000) if v % 10 == 1 and v % 100 != 11]
    rng.shuffle(values)
    values = sorted(values[: a.count])

    added, stats = [], {"considered": 0, "no_form": 0, "cross_check_failed": 0,
                        "frozen": 0, "dev": 0, "duplicate": 0}
    for n in values:
        for gender, nouns in (("femn", FEMININE), ("masc", MASCULINE)):
            phrase = accusative(n, gender)
            if not phrase:
                stats["no_form"] += 1
                continue
            if parse_numbers(phrase) != [n]:
                stats["cross_check_failed"] += 1
                continue
            for frame in rng.sample(FRAMES, a.frames_per_value):
                noun = rng.choice(nouns)
                stats["considered"] += 1
                src = frame.format(n=n, noun=noun)
                tgt = frame.format(n=phrase, noun=noun)
                if src in frozen: stats["frozen"] += 1
                elif src in dev: stats["dev"] += 1
                elif src in seen: stats["duplicate"] += 1
                else:
                    seen.add(src)
                    added.append({"id": f"oblique-count:{n}:{gender}:{len(added)}",
                                  "source": src, "target": tgt,
                                  "categories": ["Складені числівники (відмінки)", "oblique_animate_count"],
                                  "provenance": "generated_num2words_pymorphy3_accusative_crosschecked"})
    combined = [*base, *added]
    a.output.mkdir(parents=True, exist_ok=True)
    rep = {"base": {"path": str(a.base_dir / "train.parquet"), "rows": len(base),
                    "sha256": sha256_file(a.base_dir / "train.parquet")},
           "generation": {**stats, "added": len(added), "distinct_values": len(values)},
           "samples": [{"source": r["source"], "target": r["target"]} for r in added[:8]],
           "output": {"train": write_rows(a.output / "train.parquet", combined),
                      "dev": write_rows(a.output / "dev-stratified.parquet",
                                        read_rows(a.base_dir / "dev-stratified.parquet"))}}
    write_json(a.output / "manifest.json", rep)
    return rep


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-dir", type=Path, default=ROOT / "data/e39_people_counts")
    p.add_argument("--frozen", type=Path, default=ROOT / "reports/hf-uktn-frozen-3597.jsonl")
    p.add_argument("--output", type=Path, default=ROOT / "data/e40_oblique_counts")
    p.add_argument("--count", type=int, default=1500)
    p.add_argument("--frames-per-value", type=int, default=3)
    p.add_argument("--seed", type=int, default=4001)
    print(json.dumps(build(p.parse_args()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
