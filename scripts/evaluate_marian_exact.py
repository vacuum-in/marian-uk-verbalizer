"""Fast normalized exact-match evaluation for checkpoint selection."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import torch
from transformers import MarianMTModel, MarianTokenizer


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.verbalizer.evaluation.metrics import normalize_evaluation_text  # noqa: E402
from src.verbalizer.text import (  # noqa: E402
    TextRoute,
    enforce_output_fidelity_detailed,
    prepare_model_input,
    route_text,
    split_model_sentences,
)
from src.verbalizer.training.manifest import sha256_file, write_json  # noqa: E402


def chunks(rows: list[dict[str, Any]], size: int):
    for index in range(0, len(rows), size):
        yield rows[index : index + size]


def evaluate_exact(
    checkpoint: Path,
    data_path: Path,
    output_path: Path,
    batch_size: int,
    segment_sentences: bool,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for checkpoint evaluation")
    rows = pq.read_table(
        data_path, columns=["id", "source", "target", "categories"]
    ).to_pylist()
    tokenizer = MarianTokenizer.from_pretrained(checkpoint)
    model = MarianMTModel.from_pretrained(checkpoint).cuda().eval()
    model.config.use_cache = True
    exact = 0
    errors: list[dict[str, Any]] = []
    started = time.monotonic()
    with torch.inference_mode():
        for batch_number, batch in enumerate(chunks(rows, batch_size), start=1):
            segments_by_row = [
                split_model_sentences(str(row["source"]))
                if segment_sentences
                else [str(row["source"])]
                for row in batch
            ]
            flat_segments = [segment for segments in segments_by_row for segment in segments]
            routed = [route_text(segment) for segment in flat_segments]
            flat_predictions = [item.text for item in routed]
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
                ).to("cuda")
                source_tokens = int(encoded["attention_mask"].sum(dim=1).max())
                dynamic_max_new_tokens = min(
                    384,
                    max(128, source_tokens * 3)
                    if segment_sentences
                    else max(64, source_tokens * 2),
                )
                generated = model.generate(
                    **encoded,
                    max_new_tokens=dynamic_max_new_tokens,
                    num_beams=1,
                    do_sample=False,
                )
                for index, prediction in zip(
                    marian_indices,
                    tokenizer.batch_decode(generated, skip_special_tokens=True),
                    strict=True,
                ):
                    flat_predictions[index] = prediction
            predictions: list[str] = []
            offset = 0
            for row, segments in zip(batch, segments_by_row, strict=True):
                count = len(segments)
                predictions.append(
                    enforce_output_fidelity_detailed(
                        str(row["source"]),
                        " ".join(flat_predictions[offset : offset + count]),
                    ).text
                )
                offset += count
            if offset != len(flat_predictions):
                raise RuntimeError("segmented prediction reconstruction mismatch")
            for row, prediction in zip(batch, predictions, strict=True):
                reference = str(row["target"])
                if normalize_evaluation_text(prediction) == normalize_evaluation_text(reference):
                    exact += 1
                else:
                    errors.append(
                        {
                            "id": str(row["id"]),
                            "source": str(row["source"]),
                            "reference": reference,
                            "prediction": prediction,
                            "categories": list(row.get("categories") or []),
                        }
                    )
            if batch_number % 25 == 0:
                done = min(batch_number * batch_size, len(rows))
                print(
                    json.dumps(
                        {
                            "checkpoint": checkpoint.name,
                            "completed": done,
                            "total": len(rows),
                            "errors": len(errors),
                        }
                    ),
                    flush=True,
                )
    elapsed = time.monotonic() - started
    report = {
        "checkpoint": str(checkpoint),
        "data": {
            "path": str(data_path),
            "sha256": sha256_file(data_path),
            "rows": len(rows),
        },
        "metric_policy": "normalize_evaluation_text exact match only",
        "generation": {
            "batch_size": batch_size,
            "num_beams": 1,
            "segment_sentences": segment_sentences,
        },
        "runtime": {
            "seconds": round(elapsed, 3),
            "sentences_per_second": round(len(rows) / max(elapsed, 1e-9), 3),
        },
        "metrics": {
            "rows": len(rows),
            "exact": exact,
            "errors": len(errors),
            "exact_match": exact / len(rows) if rows else 0.0,
        },
        "errors": errors,
    }
    write_json(output_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--segment-sentences", action="store_true")
    args = parser.parse_args()
    report = evaluate_exact(
        args.checkpoint.resolve(),
        args.data.resolve(),
        args.output.resolve(),
        args.batch_size,
        args.segment_sentences,
    )
    print(json.dumps({**report["runtime"], **report["metrics"]}, indent=2))


if __name__ == "__main__":
    main()
