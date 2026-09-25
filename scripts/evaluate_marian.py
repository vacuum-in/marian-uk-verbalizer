"""Generate predictions and evaluate a trained Marian checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import torch
from transformers import MarianMTModel, MarianTokenizer


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.verbalizer.evaluation.metrics import evaluate_examples  # noqa: E402
from src.verbalizer.text import (  # noqa: E402
    RoutedText,
    TextRoute,
    prepare_model_input,
    route_text,
    split_model_sentences,
)
from src.verbalizer.serving import validate_serving_manifest  # noqa: E402
from src.verbalizer.training.manifest import sha256_file, write_json  # noqa: E402


def chunks(rows: list[dict[str, Any]], size: int):
    for index in range(0, len(rows), size):
        yield rows[index : index + size]


def load_rows(
    data_path: Path,
    source_column: str,
    target_column: str,
) -> list[dict[str, Any]]:
    """Load canonical or external source/target parquet and JSONL schemas."""

    if data_path.suffix.lower() == ".jsonl":
        raw_rows = [
            json.loads(line)
            for line in data_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        schema_names = set().union(*(row.keys() for row in raw_rows)) if raw_rows else set()
        missing = {source_column, target_column} - schema_names
        if missing:
            raise ValueError(
                f"missing evaluation columns in {data_path}: {', '.join(sorted(missing))}"
            )
        return [
            {
                "id": str(row.get("id") or f"row-{index:06d}"),
                "source": str(row[source_column]),
                "target": str(row[target_column]),
                "categories": list(row.get("categories") or []),
            }
            for index, row in enumerate(raw_rows)
        ]

    schema_names = set(pq.read_schema(data_path).names)
    missing = {source_column, target_column} - schema_names
    if missing:
        raise ValueError(
            f"missing evaluation columns in {data_path}: {', '.join(sorted(missing))}"
        )
    optional_columns = [name for name in ("id", "categories") if name in schema_names]
    raw_rows = pq.read_table(
        data_path,
        columns=[source_column, target_column, *optional_columns],
    ).to_pylist()
    return [
        {
            "id": str(row.get("id") or f"row-{index:06d}"),
            "source": str(row[source_column]),
            "target": str(row[target_column]),
            "categories": list(row.get("categories") or []),
        }
        for index, row in enumerate(raw_rows)
    ]


def evaluate(
    checkpoint: Path,
    data_path: Path,
    output_path: Path,
    limit: int,
    batch_size: int,
    num_beams: int,
    segment_sentences: bool,
    source_column: str = "source",
    target_column: str = "target",
    require_compatible_manifest: bool = True,
    allow_cpu: bool = False,
    force_marian: bool = False,
    csv_path: Path | None = None,
) -> dict[str, Any]:
    if not torch.cuda.is_available() and not allow_cpu:
        raise RuntimeError("CUDA GPU is required for checkpoint evaluation")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = load_rows(data_path, source_column, target_column)
    if limit > 0:
        rows = rows[:limit]
    serving = validate_serving_manifest(checkpoint) if require_compatible_manifest else None
    tokenizer = MarianTokenizer.from_pretrained(checkpoint)
    model = MarianMTModel.from_pretrained(checkpoint).to(device).eval()
    model.config.use_cache = True
    examples: list[dict[str, Any]] = []
    started = time.monotonic()
    with torch.inference_mode():
        for batch in chunks(rows, batch_size):
            segments_by_row = [
                split_model_sentences(str(row["source"]))
                if segment_sentences
                else [str(row["source"])]
                for row in batch
            ]
            flat_segments = [segment for segments in segments_by_row for segment in segments]
            routed = [
                RoutedText(TextRoute.MARIAN, segment) if force_marian else route_text(segment)
                for segment in flat_segments
            ]
            flat_predictions = [item.text for item in routed]
            generated_lists: list[list[int]] = [[] for _ in flat_segments]
            marian_indices = [
                index for index, item in enumerate(routed) if item.route is TextRoute.MARIAN
            ]
            if marian_indices:
                encoded = tokenizer(
                    [prepare_model_input(routed[index].text) for index in marian_indices],
                    max_length=384,
                    truncation=True,
                    padding=True,
                    return_tensors="pt",
                ).to(device)
                source_tokens = int(encoded["attention_mask"].sum(dim=1).max())
                dynamic_max_new_tokens = min(
                    384,
                    max(128, source_tokens * 3)
                    if segment_sentences
                    else max(64, source_tokens * 2),
                )
                precision = (
                    torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                    if device.type == "cuda" and torch.cuda.is_bf16_supported()
                    else nullcontext()
                )
                with precision:
                    generated = model.generate(
                        **encoded,
                        max_new_tokens=dynamic_max_new_tokens,
                        num_beams=num_beams,
                        do_sample=False,
                        early_stopping=num_beams > 1,
                    )
                for index, prediction, token_ids in zip(
                    marian_indices,
                    tokenizer.batch_decode(generated, skip_special_tokens=True),
                    generated.tolist(),
                    strict=True,
                ):
                    flat_predictions[index] = prediction
                    generated_lists[index] = token_ids
            offset = 0
            for row, segments in zip(batch, segments_by_row, strict=True):
                count = len(segments)
                prediction = " ".join(flat_predictions[offset : offset + count])
                segment_token_lists = generated_lists[offset : offset + count]
                offset += count
                token_ids = [token_id for item in segment_token_lists for token_id in item]
                examples.append(
                    {
                        "id": str(row["id"]),
                        "source": str(row["source"]),
                        "reference": str(row["target"]),
                        "prediction": prediction,
                        "categories": list(row.get("categories") or []),
                        "generated_tokens": sum(
                            token_id != tokenizer.pad_token_id for token_id in token_ids
                        ),
                        "terminated_with_eos": all(
                            not item or tokenizer.eos_token_id in item
                            for item in segment_token_lists
                        ),
                        "numeric_model_rejected": False,
                        "protected_model_rejected": False,
                        "numeric_warnings": [],
                        "numeric_failures": [],
                    }
                )
            if offset != len(flat_predictions):
                raise RuntimeError("segmented prediction reconstruction mismatch")
    elapsed = time.monotonic() - started
    report = {
        "checkpoint": str(checkpoint),
        "data": {
            "path": str(data_path),
            "sha256": sha256_file(data_path),
            "rows": len(rows),
            "source_column": source_column,
            "target_column": target_column,
        },
        "generation": {
            "batch_size": batch_size,
            "num_beams": num_beams,
            "early_stopping": num_beams > 1,
            "segment_sentences": segment_sentences,
            "raw_model_output": True,
            "force_marian": force_marian,
            "numeric_policy_manifest": serving,
            "compute_precision": (
                "bf16"
                if device.type == "cuda" and torch.cuda.is_bf16_supported()
                else "fp32"
            ),
            "device": device.type,
        },
        "runtime": {
            "seconds": round(elapsed, 3),
            "sentences_per_second": round(len(rows) / max(elapsed, 1e-9), 3),
        },
        "metrics": evaluate_examples(examples),
        "examples": examples,
    }
    overall = report["metrics"]["overall"]
    report["numeric_release_gate"] = {
        "enabled": False,
        "reason": "routing-only runtime returns raw selected-model output",
    }
    write_json(output_path, report)
    numeric_failure_path = output_path.with_name(output_path.stem + "-numeric-failures.jsonl")
    numeric_failure_path.write_text(
        "".join(
            json.dumps(example, ensure_ascii=False) + "\n"
            for example in examples
            if example["numeric_model_rejected"]
        ),
        encoding="utf-8",
    )
    report["numeric_failure_artifact"] = str(numeric_failure_path)
    write_json(output_path, report)
    if csv_path:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=("id", "source", "reference", "prediction", "exact", "categories"),
            )
            writer.writeheader()
            for example in examples:
                writer.writerow(
                    {
                        "id": example["id"],
                        "source": example["source"],
                        "reference": example["reference"],
                        "prediction": example["prediction"],
                        "exact": str(example["prediction"] == example["reference"]).lower(),
                        "categories": "|".join(example["categories"]),
                    }
                )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--data",
        type=Path,
        default=ROOT / "data/pilot/validation.parquet",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--limit", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--segment-sentences", action="store_true")
    parser.add_argument("--source-column", default="source")
    parser.add_argument("--target-column", default="target")
    parser.add_argument("--allow-incompatible-checkpoint", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--force-marian", action="store_true")
    parser.add_argument("--csv", type=Path, help="write source/reference/prediction rows to CSV")
    args = parser.parse_args()
    output = args.output or args.checkpoint.parent / f"evaluation-{args.checkpoint.name}.json"
    report = evaluate(
        args.checkpoint.resolve(),
        args.data.resolve(),
        output.resolve(),
        args.limit,
        args.batch_size,
        args.num_beams,
        args.segment_sentences,
        args.source_column,
        args.target_column,
        not args.allow_incompatible_checkpoint,
        args.allow_cpu,
        args.force_marian,
        args.csv.resolve() if args.csv else None,
    )
    print(json.dumps({**report["runtime"], **report["metrics"]["overall"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
