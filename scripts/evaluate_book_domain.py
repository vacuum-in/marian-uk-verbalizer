"""Score a Marian checkpoint on the reference-free book-domain sets.

Every row is sent to the checkpoint (forced Marian) so the model itself is
measured. The ``routed`` figures additionally show what the current router
would produce: rows it passes through keep the source unchanged.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from contextlib import nullcontext
from pathlib import Path

import torch
from transformers import MarianMTModel, MarianTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.verbalizer.evaluation.book_domain import identity_verdict, numeric_roundtrip  # noqa: E402
from src.verbalizer.evaluation.metrics import has_repetition_loop  # noqa: E402
from src.verbalizer.text import TextRoute, route_text  # noqa: E402
from src.verbalizer.training.manifest import sha256_file, write_json  # noqa: E402


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def generate(checkpoint: Path, sources: list[str], batch_size: int, device: torch.device) -> list[str]:
    tokenizer = MarianTokenizer.from_pretrained(checkpoint)
    model = MarianMTModel.from_pretrained(checkpoint).to(device).eval()
    model.config.use_cache = True
    precision = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda" and torch.cuda.is_bf16_supported()
        else nullcontext()
    )
    order = sorted(range(len(sources)), key=lambda index: len(sources[index]))
    predictions: list[str] = [""] * len(sources)
    with torch.inference_mode():
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            encoded = tokenizer([sources[i] for i in indices], max_length=384, truncation=True,
                                padding=True, return_tensors="pt").to(device)
            source_tokens = int(encoded["attention_mask"].sum(dim=1).max())
            with precision:
                # Same budget as scripts/gradio_app.py: digit strings expand ~4x when spelled out.
                generated = model.generate(**encoded, max_new_tokens=min(384, max(128, source_tokens * 3)),
                                           num_beams=1, do_sample=False)
            for index, text in zip(indices, tokenizer.batch_decode(generated, skip_special_tokens=True)):
                predictions[index] = text
    return predictions


def rate(counter: Counter, key: str) -> float:
    total = sum(counter.values())
    return round(100 * counter[key] / total, 2) if total else 0.0


def score_identity(rows: list[dict]) -> dict:
    by_stratum: dict[str, Counter] = defaultdict(Counter)
    routed: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        verdict = identity_verdict(row["source"], row["prediction"])
        row["verdict"] = verdict
        row["repetition_loop"] = has_repetition_loop(row["prediction"])
        passthrough = row["route"] == TextRoute.PASSTHROUGH.value
        for name in (row["stratum"], "overall"):
            by_stratum[name][verdict] += 1
            routed[name]["exact" if passthrough else verdict] += 1
    return {
        name: {
            "rows": sum(counts.values()),
            "exact_pct": rate(counts, "exact"),
            "content_preserved_pct": round(rate(counts, "exact") + rate(counts, "whitespace_only"), 2),
            "changed_pct": rate(counts, "changed"),
            "routed_content_preserved_pct": round(rate(routed[name], "exact") + rate(routed[name], "whitespace_only"), 2),
        }
        for name, counts in sorted(by_stratum.items())
    }


def score_numeric(rows: list[dict]) -> dict:
    by_stratum: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        result = numeric_roundtrip(row["source"], row["prediction"])
        row.update(result)
        row["repetition_loop"] = has_repetition_loop(row["prediction"])
        for name in (row["stratum"], "overall"):
            by_stratum[name]["ok" if result["ok"] else "fail"] += 1
            by_stratum[name]["digits_left" if result["digits_left"] else "no_digits"] += 1
    return {
        name: {
            "rows": counts["ok"] + counts["fail"],
            "roundtrip_ok_pct": rate(Counter(ok=counts["ok"], fail=counts["fail"]), "ok"),
            "digits_left": counts["digits_left"],
        }
        for name, counts in sorted(by_stratum.items())
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--identity", type=Path, default=ROOT / "reports/book-prose-identity.jsonl")
    parser.add_argument("--numeric", type=Path, default=ROOT / "reports/book-numeric.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    identity_rows = read_jsonl(args.identity)
    numeric_rows = read_jsonl(args.numeric)
    rows = identity_rows + numeric_rows
    started = time.monotonic()
    predictions = generate(args.checkpoint, [row["source"] for row in rows], args.batch_size, device)
    for row, prediction in zip(rows, predictions, strict=True):
        row["prediction"] = prediction
        row["route"] = route_text(row["source"]).route.value

    report = {
        "checkpoint": str(args.checkpoint),
        "device": device.type,
        "seconds": round(time.monotonic() - started, 1),
        "data": {"identity": {"path": str(args.identity), "sha256": sha256_file(args.identity)},
                 "numeric": {"path": str(args.numeric), "sha256": sha256_file(args.numeric)}},
        "identity": score_identity(identity_rows),
        "numeric": score_numeric(numeric_rows),
        "repetition_loops": sum(bool(row["repetition_loop"]) for row in rows),
    }
    write_json(args.output, report)
    failures = [row for row in identity_rows if row["verdict"] == "changed"]
    failures += [row for row in numeric_rows if not row["ok"]]
    for suffix, selected in (("-predictions.jsonl", rows), ("-failures.jsonl", failures)):
        with args.output.with_name(args.output.stem + suffix).open("w", encoding="utf-8") as handle:
            for row in selected:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({key: report[key] for key in ("seconds", "identity", "numeric", "repetition_loops")},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
