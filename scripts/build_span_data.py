#!/usr/bin/env python3
"""Span-to-span training data from sentence pairs (offline; nothing here runs in serving).

Each (written, spoken) pair is aligned word by word, ignoring case. Every changed region of the alignment must lie
inside the spans the router detects in the written sentence; the spoken words of that region become the span's
target. A span whose words are unchanged gets itself as the target, so the model learns to leave such spans alone. A
pair is skipped when a change falls outside every span (a detector miss, counted) or one changed region spans two
detected spans with prose between them (ambiguous, counted).

    build_span_data.py --pairs data/e54_roman_balance/train.parquet --out data/e55_spans/train.parquet
"""
from __future__ import annotations

import argparse
import collections
import difflib
import hashlib
import json
import random
import re
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.verbalizer.spans import Span, detect_spans, model_input  # noqa: E402

TOKEN_WORD = re.compile(r"[^\W_]")
FOOTNOTE = re.compile(r"\[\s*(?:\d{1,4}|\*+)\s*\]|[¹²³⁰⁴⁵⁶⁷⁸⁹]+")
TOKEN = re.compile(r"[^\W_]+(?:['’ʼ][^\W_]+)*|[^\w\s]", re.UNICODE)


def tokens(text: str):
    return [(m.group(), m.start(), m.end()) for m in TOKEN.finditer(text)]


def key(tok: str) -> str:
    return tok.casefold().replace("’", "'").replace("ʼ", "'")


def align(source: str, target: str):
    """[(span, spoken)] for one pair, or (None, reason)."""
    spans = detect_spans(source)
    st, tt = tokens(source), tokens(target)
    owner = []  # span index of each source token, or None for prose
    for _, a, b in st:
        owner.append(next((k for k, sp in enumerate(spans) if a >= sp.start and b <= sp.end), None))
    ranges = collections.defaultdict(list)  # span -> target token index ranges
    matcher = difflib.SequenceMatcher(a=[key(t) for t, _, _ in st], b=[key(t) for t, _, _ in tt], autojunk=False)
    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op == "equal":
            for i, j in zip(range(i1, i2), range(j1, j2)):
                if owner[i] is not None:
                    ranges[owner[i]].append((j, j + 1))
            continue
        # punctuation next to a span (a dropped final period, an emoji) is not a change "outside" it
        keep = [i for i in range(i1, i2) if owner[i] is not None or TOKEN_WORD.match(st[i][0])]
        if not keep and i1 != i2:
            near = {owner[i1 - 1] if i1 > 0 else None, owner[i2] if i2 < len(owner) else None} - {None}
            if j1 == j2 or len(near) == 1:
                if j2 > j1 and near:
                    ranges[near.pop()].append((j1, j2))
                continue
        owners = {owner[i] for i in keep} if keep else {owner[i] for i in range(i1, i2)}
        if i1 == i2:  # insertion: belongs to the span next to it
            near = {owner[i1 - 1] if i1 > 0 else None, owner[i1] if i1 < len(owner) else None} - {None}
            if len(near) != 1:
                return None, "change outside spans"
            ranges[near.pop()].append((j1, j2))
            continue
        if None in owners:
            return None, "change outside spans"
        if len(owners) > 1:
            return None, "one change across two spans"
        ranges[owners.pop()].append((j1, j2))
    out = []
    for k, sp in enumerate(spans):
        rs = [r for r in ranges.get(k, []) if r[1] > r[0]]
        if not rs:
            spoken = ""
        else:
            a = min(r[0] for r in rs)
            b = max(r[1] for r in rs)
            spoken = target[tt[a][1]: tt[b - 1][2]]
            # punctuation the span does not have belongs to the sentence, not to the span's reading
            if sp.text[-1].isalnum():
                spoken = spoken.rstrip(".,;:!?…")
            if sp.text[0].isalnum():
                spoken = spoken.lstrip(".,;:!?…")
        out.append((sp, spoken))
    return out, None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dev-share", type=float, default=0.01)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--identity-keep", type=float, default=0.35,
                    help="share of unchanged spans kept (they teach 'leave it' but are plentiful)")
    ap.add_argument("--upsample", default="", help="provenance substring:times, e.g. e52:2,nominative:2")
    ap.add_argument("--extra", type=Path, nargs="*", default=[], help="ready span rows (parquet or jsonl) added to train")
    ap.add_argument("--copy-share", type=float, default=0.15, help="pairs that also give one 'read as itself' word")
    args = ap.parse_args()
    rows = pq.read_table(args.pairs, columns=["id", "source", "target", "provenance"]).to_pylist()
    boost = {k: int(v) for k, v in (x.split(":") for x in args.upsample.split(",") if x)}
    if args.limit:
        rows = rows[: args.limit]
    rng = random.Random(55)
    stats = collections.Counter()
    train, dev = [], []
    for r in rows:
        pairs, why = align(r["source"], r["target"])
        if pairs is None:
            stats[why] += 1
            continue
        stats["pairs used"] += 1
        bucket = dev if int(hashlib.sha1(r["id"].encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < args.dev_share else train
        # one policy for footnote marks across all sources: they are not read (some generated rows read "[26]")
        pairs = [(sp, "" if FOOTNOTE.fullmatch(sp.text) else spoken) for sp, spoken in pairs]
        # a span read as nothing must be a footnote mark or an emoji; otherwise the alignment lost its words
        if any(not spoken and re.search(r"[A-Za-zА-Яа-яІіЇїЄєҐґ]|\d(?![\d\]])", sp.text) and not re.fullmatch(r"\[\s*\d+\s*\]", sp.text)
               for sp, spoken in pairs):
            stats["lost reading"] += 1
            continue
        # "leave it alone": an ordinary word of the same sentence marked as a span, read as itself
        if rng.random() < args.copy_share:
            taken = [(sp.start, sp.end) for sp, _ in pairs]
            words = [m for m in re.finditer(r"[А-Яа-яІіЇїЄєҐґ'’ʼ-]{2,}\.?", r["source"])
                     if not any(m.start() < b and m.end() > a for a, b in taken) and m.group().rstrip(".") in r["target"]]
            if words:
                m = rng.choice(words)
                cp = Span(m.start(), m.end(), m.group())
                bucket.append({"id": f"{hashlib.sha1(r['id'].encode()).hexdigest()[:16]}-c", "source": model_input(r["source"], cp), "target": m.group(),
                               "span": m.group(), "sentence_id": r["id"]})
                stats["copy rows"] += 1
        for n, (sp, spoken) in enumerate(pairs):
            identity = spoken == sp.text
            stats["spans identity" if identity else ("spans empty" if not spoken else "spans changed")] += 1
            if identity and bucket is train and rng.random() > args.identity_keep:
                continue
            row = {"id": f"{hashlib.sha1(r['id'].encode()).hexdigest()[:16]}-{n}", "source": model_input(r["source"], sp),
                   "target": spoken, "span": sp.text, "sentence_id": r["id"]}
            times = max([t for k, t in boost.items() if k in (r.get("provenance") or "")] or [1]) if bucket is train else 1
            bucket.extend([row] * times)
    for path in args.extra:
        extra = (pq.read_table(path).to_pylist() if path.suffix == ".parquet"
                 else [json.loads(l) for l in path.open(encoding="utf-8")])
        train.extend({k: x[k] for k in ("id", "source", "target", "span", "sentence_id")} for x in extra)
        stats[f"extra {path.name}"] = len(extra)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rng.shuffle(train)
    pq.write_table(pa.Table.from_pylist(train), args.out)
    pq.write_table(pa.Table.from_pylist(dev), args.out.with_name("dev.parquet"))
    stats["train rows"], stats["dev rows"] = len(train), len(dev)
    (args.out.parent / "manifest.json").write_text(json.dumps({"pairs": str(args.pairs), **stats}, indent=2,
                                                                ensure_ascii=False), encoding="utf-8")
    print(json.dumps(stats, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
