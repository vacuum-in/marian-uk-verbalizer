#!/usr/bin/env python3
"""Capture deterministic Gemma system outputs for sequence-level distillation.

Run this script in an isolated teacher environment with transformers >= 4.57.
It is append-only and resumable; failed teacher outputs are retained as
quarantine records rather than silently becoming training labels.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.verbalizer.distillation import (  # noqa: E402
    GEMMA_MAX_NEW_TOKENS,
    GEMMA_MODEL_ID,
    GEMMA_REVISION,
    sha256_text,
    teacher_input,
    teacher_validation_flags,
)
from src.verbalizer.training.manifest import sha256_file, write_json  # noqa: E402


def load_rows(path: Path, source_column: str, limit: int) -> list[dict[str, str]]:
    if path.suffix == ".jsonl":
        raw = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    else:
        columns = [source_column]
        names = set(pq.read_schema(path).names)
        if "id" in names:
            columns.append("id")
        raw = pq.read_table(path, columns=columns).to_pylist()
    if limit:
        raw = raw[:limit]
    return [
        {
            "id": str(row.get("id") or f"teacher-{index:09d}"),
            "source": str(row[source_column]),
        }
        for index, row in enumerate(raw)
    ]


def resume_index(path: Path, rows: list[dict[str, str]]) -> int:
    if not path.exists():
        return 0
    completed = 0
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line:
            continue
        record = json.loads(line)
        if completed >= len(rows) or record.get("index") != completed:
            raise ValueError(f"non-contiguous capture at {path}:{number}")
        expected = rows[completed]
        if record.get("id") != expected["id"] or record.get("source") != expected["source"]:
            raise ValueError(f"source mismatch at {path}:{number}")
        completed += 1
    return completed


def require_teacher_runtime() -> tuple[Any, Any, Any]:
    version = tuple(int(part) for part in importlib.metadata.version("transformers").split(".")[:2])
    if version < (4, 57):
        raise RuntimeError(
            "Gemma teacher requires transformers >= 4.57. Use an isolated teacher venv; "
            "do not upgrade the Marian runtime in place."
        )
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    return torch, AutoModelForCausalLM, AutoTokenizer


def capture(args: argparse.Namespace) -> dict[str, Any]:
    rows = load_rows(args.data, args.source_column, args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed = resume_index(args.output, rows)
    torch, model_cls, tokenizer_cls = require_teacher_runtime()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    tokenizer = tokenizer_cls.from_pretrained(args.model, revision=args.revision)
    model = model_cls.from_pretrained(args.model, revision=args.revision, torch_dtype=dtype).to(device).eval()
    bad_words = [[index] for token, index in tokenizer.get_vocab().items() if any(char.isdigit() for char in token)]
    captured_at = datetime.now(timezone.utc).isoformat()

    with args.output.open("a", encoding="utf-8") as handle, torch.inference_mode():
        for index, row in enumerate(rows[completed:], start=completed):
            handled, prepared = teacher_input(row["source"])
            if handled:
                output = prepared
            else:
                ids = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prepared}],
                    add_generation_prompt=True,
                    return_tensors="pt",
                ).to(device)
                generated = model.generate(
                    ids,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    bad_words_ids=bad_words,
                    pad_token_id=tokenizer.pad_token_id,
                )
                output = tokenizer.decode(generated[0, ids.shape[-1] :], skip_special_tokens=True).strip()
            rejected, review = teacher_validation_flags(row["source"], output)
            record = {
                "index": index,
                **row,
                "source_sha256": sha256_text(row["source"]),
                "teacher_input": prepared,
                "teacher_output": output,
                "status": "quarantine" if rejected else "approved",
                "rejection_flags": rejected,
                "review_flags": review,
                "teacher": {
                    "model": args.model,
                    "revision": args.revision,
                    "max_new_tokens": args.max_new_tokens,
                    "digit_tokens_banned": True,
                    "deterministic_route_handled": handled,
                },
                "captured_at": captured_at,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            if (index + 1) % args.progress_every == 0 or index + 1 == len(rows):
                print(json.dumps({"captured": index + 1, "total": len(rows)}, ensure_ascii=False), flush=True)
    manifest = {
        "created_at": captured_at,
        "source": {"path": str(args.data), "sha256": sha256_file(args.data), "rows": len(rows)},
        "capture": {"path": str(args.output), "sha256": sha256_file(args.output)},
        "teacher": {"model": args.model, "revision": args.revision, "max_new_tokens": args.max_new_tokens},
        "policy": [
            "Greedy generation with all digit-containing vocabulary tokens banned.",
            "Bare digit identifiers and safe long-number spacing follow the pinned teacher pipeline.",
            "Quarantined rows are never eligible for Marian training.",
        ],
    }
    write_json(args.output.with_suffix(".manifest.json"), manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-column", default="source")
    parser.add_argument("--model", default=GEMMA_MODEL_ID)
    parser.add_argument("--revision", default=GEMMA_REVISION)
    parser.add_argument("--device")
    parser.add_argument("--max-new-tokens", type=int, default=GEMMA_MAX_NEW_TOKENS)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=100)
    args = parser.parse_args()
    print(json.dumps(capture(args), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
