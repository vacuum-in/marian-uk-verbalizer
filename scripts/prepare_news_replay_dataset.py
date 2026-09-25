"""Mix the 95k news domain with deterministic structured replay rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import unicodedata
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NEWS = ROOT / "data/news_domain"
DEFAULT_STRUCTURED = ROOT / "data/target_domain_hard_v3"
DEFAULT_OUTPUT = ROOT / "data/news_domain_replay"
DEFAULT_TEST = ROOT / "data/news_domain/test.parquet"
DEFAULT_REGRESSIONS = ROOT / "data/target_domain/regressions.parquet"
DATASET_VERSION = "news-95k-structured-replay-v1-2026-07-31"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_key(value: str) -> str:
    return " ".join(unicodedata.normalize("NFC", value).split()).casefold()


def read_rows(path: Path, domain: str) -> list[dict[str, str]]:
    return [
        {
            "id": str(row["id"]),
            "source": str(row["source"]),
            "target": str(row["target"]),
            "domain": domain,
        }
        for row in pq.read_table(path, columns=["id", "source", "target"]).to_pylist()
    ]


def ranked_unique(
    rows: list[dict[str, str]],
    seed: int,
    forbidden: set[str],
) -> list[dict[str, str]]:
    ranked = sorted(
        rows,
        key=lambda row: hashlib.sha256(
            f"{seed}\0{row['id']}\0{row['source']}\0{row['target']}".encode("utf-8")
        ).hexdigest(),
    )
    retained: list[dict[str, str]] = []
    seen = set(forbidden)
    for row in ranked:
        key = source_key(row["source"])
        if key in seen:
            continue
        seen.add(key)
        retained.append(row)
    return retained


def stamp(rows: list[dict[str, str]], split: str, seed: int) -> list[dict[str, Any]]:
    stamped: list[dict[str, Any]] = []
    for row in rows:
        identity = hashlib.sha256(
            f"{DATASET_VERSION}\0{split}\0{row['domain']}\0{row['id']}".encode("utf-8")
        ).hexdigest()
        stamped.append(
            {
                "id": identity,
                "source": row["source"],
                "target": row["target"],
                "categories": [row["domain"]],
                "dataset_version": DATASET_VERSION,
                "quality_flags": [],
            }
        )
    return sorted(
        stamped,
        key=lambda row: hashlib.sha256(
            f"{seed}\0{split}\0{row['id']}".encode("utf-8")
        ).hexdigest(),
    )


def write_rows(path: Path, rows: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")
    return sha256_file(path)


def build(args: argparse.Namespace) -> dict[str, Any]:
    news_train_path = args.news / "train.parquet"
    news_validation_path = args.news / "validation.parquet"
    structured_train_path = args.structured / "train.parquet"
    structured_validation_path = args.structured / "validation.parquet"

    news_train = read_rows(news_train_path, "news_domain")
    news_validation = read_rows(news_validation_path, "news_domain")
    test = read_rows(args.test, "news_test")
    regressions = read_rows(args.regressions, "user_regression")
    final_forbidden = {
        source_key(row["source"]) for row in [*test, *regressions]
    }

    news_train = ranked_unique(news_train, args.seed, final_forbidden)
    train_keys = final_forbidden | {source_key(row["source"]) for row in news_train}
    structured_train = ranked_unique(
        read_rows(structured_train_path, "structured_replay"),
        args.seed + 1,
        train_keys,
    )[: len(news_train)]
    combined_train = [*news_train, *structured_train]
    combined_train_keys = {source_key(row["source"]) for row in combined_train}

    news_validation = ranked_unique(
        news_validation,
        args.seed + 2,
        final_forbidden | combined_train_keys,
    )
    validation_forbidden = (
        final_forbidden
        | combined_train_keys
        | {source_key(row["source"]) for row in news_validation}
    )
    structured_validation = ranked_unique(
        read_rows(structured_validation_path, "structured_replay"),
        args.seed + 3,
        validation_forbidden,
    )[: args.structured_validation_rows]
    combined_validation = [*news_validation, *structured_validation]

    train_sources = {source_key(row["source"]) for row in combined_train}
    validation_sources = {source_key(row["source"]) for row in combined_validation}
    test_sources = {source_key(row["source"]) for row in test}
    regression_sources = {source_key(row["source"]) for row in regressions}
    leakage = {
        "train_validation_source_overlap": len(train_sources & validation_sources),
        "train_test_source_overlap": len(train_sources & test_sources),
        "validation_test_source_overlap": len(validation_sources & test_sources),
        "train_regression_source_overlap": len(train_sources & regression_sources),
        "validation_regression_source_overlap": len(validation_sources & regression_sources),
    }
    if any(leakage.values()):
        raise RuntimeError(f"dataset leakage detected: {leakage}")

    outputs = {
        "train": stamp(combined_train, "train", args.seed),
        "validation": stamp(combined_validation, "validation", args.seed),
    }
    output_hashes = {
        split: write_rows(args.output / f"{split}.parquet", rows)
        for split, rows in outputs.items()
    }
    manifest = {
        "dataset_version": DATASET_VERSION,
        "seed": args.seed,
        "rows": {
            "train": len(outputs["train"]),
            "validation": len(outputs["validation"]),
            "train_news": len(news_train),
            "train_structured_replay": len(structured_train),
            "validation_news": len(news_validation),
            "validation_structured_replay": len(structured_validation),
        },
        "sources": {
            str(path): sha256_file(path)
            for path in (
                news_train_path,
                news_validation_path,
                structured_train_path,
                structured_validation_path,
                args.test,
                args.regressions,
            )
        },
        "leakage": leakage,
        "output_sha256": output_hashes,
        "policy": [
            "news and structured replay have equal train row counts",
            "official golden test and user regressions are exclusion-only",
            "all split membership and row ordering are deterministic",
        ],
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--news", type=Path, default=DEFAULT_NEWS)
    parser.add_argument("--structured", type=Path, default=DEFAULT_STRUCTURED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--regressions", type=Path, default=DEFAULT_REGRESSIONS)
    parser.add_argument("--structured-validation-rows", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=79)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(build(parse_args()), ensure_ascii=False, indent=2, sort_keys=True))
