"""Snapshot the public reference verbalizer on a frozen local dataset.

The capture is ordered, append-only, resumable, and deliberately uses one
remote job at a time. Newline batches are recursively split if the service does
not preserve one output line per input line.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from gradio_client import Client


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data/processed/test_template_ood.parquet"
DEFAULT_OUTPUT = ROOT / "data/reference/verbalizer_skravchenko_gemma3_15k.jsonl"
DEFAULT_URL = "https://verbalizer.skravchenko.dev"
DEFAULT_ENDPOINT = "/normalize"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_rows(path: Path, max_rows: int) -> list[dict[str, str]]:
    rows = pq.read_table(path, columns=["id", "source"]).to_pylist()
    if max_rows > 0:
        rows = rows[:max_rows]
    normalized = [{"id": str(row["id"]), "source": str(row["source"])} for row in rows]
    if any("\n" in row["source"] or "\r" in row["source"] for row in normalized):
        raise ValueError("reference batching requires one physical line per source row")
    return normalized


def load_resume(path: Path, rows: list[dict[str, str]]) -> int:
    if not path.exists():
        return 0
    completed = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if completed >= len(rows):
                raise ValueError(f"capture has more rows than requested: {path}")
            expected = rows[completed]
            if record.get("index") != completed:
                raise ValueError(f"non-contiguous capture index at line {line_number}")
            if record.get("id") != expected["id"] or record.get("source") != expected["source"]:
                raise ValueError(f"capture/data mismatch at line {line_number}")
            completed += 1
    return completed


def predict_with_retry(
    client: Client,
    sources: list[str],
    endpoint: str,
    retries: int,
) -> list[str]:
    payload = "\n".join(sources)
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            response = client.predict(payload, api_name=endpoint)
            if not isinstance(response, str):
                raise TypeError(f"reference returned {type(response).__name__}, expected str")
            outputs = [line.rstrip("\r") for line in response.split("\n")]
            if len(outputs) == len(sources):
                return outputs
            if len(sources) == 1:
                raise ValueError(
                    f"one input produced {len(outputs)} physical lines: {response!r}"
                )
            middle = len(sources) // 2
            return [
                *predict_with_retry(client, sources[:middle], endpoint, retries),
                *predict_with_retry(client, sources[middle:], endpoint, retries),
            ]
        except Exception as exc:  # remote transport and queue errors vary by version
            last_error = exc
            if attempt >= retries:
                break
            time.sleep(min(60.0, 2.0**attempt * 5.0))
    assert last_error is not None
    raise last_error


def capture(args: argparse.Namespace) -> dict[str, Any]:
    rows = load_rows(args.data, args.max_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed = load_resume(args.output, rows)
    client = Client(args.url, verbose=False)
    started = time.monotonic()
    captured_at = datetime.now(timezone.utc).isoformat()

    with args.output.open("a", encoding="utf-8") as handle:
        for start in range(completed, len(rows), args.batch_size):
            batch = rows[start : start + args.batch_size]
            batch_started = time.monotonic()
            outputs = predict_with_retry(
                client,
                [row["source"] for row in batch],
                args.endpoint,
                args.retries,
            )
            for offset, (row, output) in enumerate(zip(batch, outputs, strict=True)):
                record = {
                    "index": start + offset,
                    "id": row["id"],
                    "source": row["source"],
                    "reference_output": output,
                    "reference_url": args.url,
                    "reference_endpoint": args.endpoint,
                    "reference_model": args.model_name,
                    "captured_at": captured_at,
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            elapsed = time.monotonic() - started
            done = start + len(batch)
            print(
                json.dumps(
                    {
                        "completed": done,
                        "total": len(rows),
                        "batch_seconds": round(time.monotonic() - batch_started, 3),
                        "elapsed_seconds": round(elapsed, 3),
                        "rows_per_second": round((done - completed) / max(elapsed, 1e-9), 4),
                    }
                ),
                flush=True,
            )
            if args.delay_seconds > 0 and done < len(rows):
                time.sleep(args.delay_seconds)

    manifest = {
        "captured_at": captured_at,
        "rows": len(rows),
        "data": {"path": str(args.data), "sha256": sha256_file(args.data)},
        "capture": {"path": str(args.output), "sha256": sha256_file(args.output)},
        "reference": {
            "url": args.url,
            "endpoint": args.endpoint,
            "model": args.model_name,
        },
        "policy": [
            "One queued remote job at a time.",
            "Physical newline count must match input row count.",
            "Capture is append-only and resume validates source IDs and text.",
            "Frozen outputs are evaluation evidence and are not training data.",
        ],
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--model-name", default="Gemma 3 270M")
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--delay-seconds", type=float, default=1.0)
    args = parser.parse_args()
    if args.batch_size < 1 or args.max_rows < 0 or args.retries < 0:
        parser.error("batch-size must be positive; max-rows/retries must be non-negative")
    return args


if __name__ == "__main__":
    try:
        print(json.dumps(capture(parse_args()), ensure_ascii=False, indent=2, sort_keys=True))
    except KeyboardInterrupt:
        print("capture interrupted; rerun the same command to resume", file=sys.stderr)
        raise SystemExit(130)
