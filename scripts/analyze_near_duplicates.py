"""Find approximate near-duplicate source rows across materialized splits.

This is a bounded screening pass, not a proof of non-duplication. SimHash
bands shortlist candidates and SequenceMatcher confirms pairs above the
configured threshold. Very common buckets are capped to keep runtime bounded.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.verbalizer.data.near_duplicates import band_keys, simhash, similarity  # noqa: E402


DEFAULT_SPLITS = {
    "train": ROOT / "data/processed/train.parquet",
    "validation": ROOT / "data/processed/validation.parquet",
    "test_id": ROOT / "data/processed/test_id.parquet",
    "test_golden_weak": ROOT / "data/processed/test_golden_weak.parquet",
    "test_template_ood": ROOT / "data/processed/test_template_ood.parquet",
}
DEFAULT_REPORT = ROOT / "reports/current_dataset_near_duplicates.json"
DEFAULT_MARKDOWN = ROOT / "reports/current_dataset_near_duplicates.md"
MIN_LENGTH = 40
THRESHOLD = 0.92
MAX_BUCKET = 8


def analyze(split_paths: dict[str, Path], threshold: float = THRESHOLD) -> dict[str, Any]:
    buckets: dict[tuple[int, int], list[tuple[str, str, str, int]]] = defaultdict(list)
    candidate_pairs = 0
    confirmed: dict[tuple[str, str], dict[str, Any]] = {}
    row_count = 0

    for split, path in split_paths.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(columns=["id", "source"], batch_size=8192):
            for row_id, source in zip(batch.column("id").to_pylist(), batch.column("source").to_pylist(), strict=True):
                row_count += 1
                source = str(source or "")
                if len(source) < MIN_LENGTH:
                    continue
                fingerprint = simhash(source)
                seen: set[str] = set()
                for band_index, band in enumerate(band_keys(fingerprint)):
                    bucket_key = (band_index, band)
                    candidates = buckets[bucket_key]
                    for other_id, other_split, other_source, _ in candidates:
                        if other_split == split or other_id in seen:
                            continue
                        seen.add(other_id)
                        candidate_pairs += 1
                        score = similarity(source, other_source)
                        if score < threshold:
                            continue
                        pair = tuple(sorted((str(row_id), other_id)))
                        confirmed[pair] = {
                            "left_id": pair[0],
                            "right_id": pair[1],
                            "left_split": split if pair[0] == str(row_id) else other_split,
                            "right_split": other_split if pair[0] == str(row_id) else split,
                            "similarity": round(score, 6),
                            "left_source": source if pair[0] == str(row_id) else other_source,
                            "right_source": other_source if pair[0] == str(row_id) else source,
                        }
                    if len(candidates) < MAX_BUCKET:
                        candidates.append((str(row_id), split, source, fingerprint))

    pairs = sorted(confirmed.values(), key=lambda item: (-item["similarity"], item["left_id"], item["right_id"]))
    cross_split = [pair for pair in pairs if pair["left_split"] != pair["right_split"]]
    return {
        "algorithm": {
            "name": "32-bit sampled character-shingle SimHash with four bands",
            "threshold": threshold,
            "minimum_source_length": MIN_LENGTH,
            "maximum_bucket_size": MAX_BUCKET,
            "bounded_screening": True,
        },
        "rows_scanned": row_count,
        "candidate_pairs_examined": candidate_pairs,
        "near_duplicate_pairs": len(pairs),
        "cross_split_near_duplicate_pairs": len(cross_split),
        "cross_split_pairs": cross_split[:100],
        "passed": not cross_split,
    }


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Near-Duplicate Screening",
        "",
        "This bounded screening pass uses SimHash to shortlist candidates and "
        "SequenceMatcher to confirm likely near duplicates.",
        "",
        f"- Rows scanned: **{report['rows_scanned']}**",
        f"- Candidate pairs examined: **{report['candidate_pairs_examined']}**",
        f"- Confirmed near-duplicate pairs: **{report['near_duplicate_pairs']}**",
        f"- Cross-split pairs: **{report['cross_split_near_duplicate_pairs']}**",
        f"- Result: **{'PASS' if report['passed'] else 'REVIEW REQUIRED'}**",
        "",
        "## Cross-split examples",
        "",
    ]
    for pair in report["cross_split_pairs"][:20]:
        lines.append(
            f"- `{pair['left_split']}` vs `{pair['right_split']}` "
            f"(similarity {pair['similarity']}, ids `{pair['left_id'][:12]}` / `{pair['right_id'][:12]}`)"
        )
    if not report["cross_split_pairs"]:
        lines.append("- None detected.")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name, path in DEFAULT_SPLITS.items():
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path, default=path)
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--check", action="store_true", help="exit non-zero when cross-split pairs are found")
    args = parser.parse_args()
    split_paths = {name: getattr(args, name.replace("-", "_")) for name in DEFAULT_SPLITS}
    report = analyze(split_paths, threshold=args.threshold)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(markdown_report(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if args.check and not report["passed"]:
        raise SystemExit("near-duplicate screening found cross-split pairs; see the generated report")


if __name__ == "__main__":
    main()
