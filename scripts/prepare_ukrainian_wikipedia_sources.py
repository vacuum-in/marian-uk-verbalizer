#!/usr/bin/env python3
"""Extract a pinned, attributable Ukrainian Wikipedia sentence corpus.

Input must be an official ``ukwiki-*-pages-articles*.xml.bz2`` dump.  The
caller records the dump URL/date and optional expected SHA-256 in the manifest;
this script deliberately does not use the live Wikipedia API.
"""

from __future__ import annotations

import argparse
import bz2
import json
import re
import sys
import xml.etree.ElementTree as element_tree
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.verbalizer.data.audit import canonical_source  # noqa: E402
from src.verbalizer.distillation import stable_row_id  # noqa: E402
from src.verbalizer.training.manifest import sha256_file, write_json  # noqa: E402


TAG_RE = re.compile(r"<[^>]+>")
TEMPLATE_RE = re.compile(r"\{\{.*?\}\}", re.DOTALL)
REF_RE = re.compile(r"<ref[^>]*>.*?</ref>|<ref[^>]*/>", re.IGNORECASE | re.DOTALL)
LINK_RE = re.compile(r"\[\[(?:[^\]|]+\|)?([^\]]+)\]\]")
SPACE_RE = re.compile(r"\s+")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[А-ЯІЇЄҐ])")
UKRAINIAN_RE = re.compile(r"[А-Яа-яІіЇїЄєҐґ]")


def clean_wikitext(value: str) -> str:
    value = REF_RE.sub(" ", value)
    value = TEMPLATE_RE.sub(" ", value)
    value = LINK_RE.sub(r"\1", value)
    value = TAG_RE.sub(" ", value)
    value = re.sub(r"={2,}[^=]+={2,}", " ", value)
    value = value.replace("'''", "").replace("''", "")
    return SPACE_RE.sub(" ", value).strip()


def acceptable_sentence(value: str) -> bool:
    if not 30 <= len(value) <= 384:
        return False
    if not UKRAINIAN_RE.search(value):
        return False
    if value.startswith(("Категорія:", "Файл:", "Див. також")):
        return False
    return not value.count("{") and not value.count("}")


def build(args: argparse.Namespace) -> dict[str, object]:
    if args.expected_sha256 and sha256_file(args.dump) != args.expected_sha256:
        raise ValueError("Wikipedia dump SHA-256 does not match --expected-sha256")
    seen: set[str] = set()
    rows: list[dict[str, str]] = []
    pages = 0
    with bz2.open(args.dump, "rb") as handle:
        for _event, element in element_tree.iterparse(handle, events=("end",)):
            if not element.tag.endswith("page"):
                continue
            pages += 1
            text_node = next((child for child in element.iter() if child.tag.endswith("text")), None)
            raw = text_node.text if text_node is not None and text_node.text else ""
            for sentence in SENTENCE_RE.split(clean_wikitext(raw)):
                sentence = sentence.strip()
                key = canonical_source(sentence)
                if not acceptable_sentence(sentence) or key in seen:
                    continue
                seen.add(key)
                rows.append(
                    {
                        "id": stable_row_id("ukwiki", sentence, ""),
                        "source": sentence,
                        "license": "CC-BY-SA-4.0",
                        "origin": "ukrainian_wikipedia",
                    }
                )
                if len(rows) >= args.max_rows:
                    break
            element.clear()
            if len(rows) >= args.max_rows:
                break
    if len(rows) < args.min_rows:
        raise RuntimeError(f"only extracted {len(rows)} rows; expected at least {args.min_rows}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), args.output, compression="zstd")
    manifest = {
        "source": {
            "dump": str(args.dump),
            "sha256": sha256_file(args.dump),
            "url": args.dump_url,
            "license": "CC-BY-SA-4.0",
        },
        "output": {"path": str(args.output), "sha256": sha256_file(args.output), "rows": len(rows)},
        "extraction": {"pages_seen": pages, "min_rows": args.min_rows, "max_rows": args.max_rows},
        "policy": [
            "Official pinned dump only; no live API data.",
            "Sentence-level deterministic cleaning, length filtering, and normalized-source deduplication.",
            "Attribution, URL, license, and input checksum are retained in this manifest.",
        ],
    }
    write_json(args.output.with_suffix(".manifest.json"), manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump", required=True, type=Path)
    parser.add_argument("--dump-url", required=True)
    parser.add_argument("--expected-sha256")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-rows", type=int, default=250_000)
    parser.add_argument("--min-rows", type=int, default=100_000)
    args = parser.parse_args()
    if not 0 < args.min_rows <= args.max_rows:
        parser.error("min-rows must be positive and no larger than max-rows")
    print(json.dumps(build(args), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
