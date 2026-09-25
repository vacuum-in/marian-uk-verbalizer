"""Score a checkpoint on the generated per-domain evaluation sets (reports/domain-eval)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_book_domain import generate  # noqa: E402
from src.verbalizer.evaluation.domain_check import score_row  # noqa: E402
from src.verbalizer.training.manifest import write_json  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--sets", type=Path, default=ROOT / "reports/domain-eval")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    rows = [json.loads(line) for path in sorted(args.sets.glob("*.jsonl"))
            for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    started = time.monotonic()
    predictions = generate(args.checkpoint, [row["source"] for row in rows], args.batch_size, device)

    by_domain: dict[str, Counter] = defaultdict(Counter)
    by_shape: dict[str, Counter] = defaultdict(Counter)
    reasons: dict[str, Counter] = defaultdict(Counter)
    for row, prediction in zip(rows, predictions, strict=True):
        result = score_row(row, prediction)
        row["prediction"], row["ok"], row["reason"] = prediction, result["ok"], result["reason"]
        by_domain[row["domain"]][result["ok"]] += 1
        by_shape[f"{row['domain']}/{row['shape']}"][result["ok"]] += 1
        if not result["ok"]:
            reasons[row["domain"]][result["reason"].split(" read as")[0]] += 1

    pct = lambda c: round(100 * c[True] / (c[True] + c[False]), 2)
    report = {
        "checkpoint": str(args.checkpoint), "device": device.type, "seconds": round(time.monotonic() - started, 1),
        "domains": {d: {"rows": sum(c.values()), "correct_pct": pct(c)} for d, c in sorted(by_domain.items())},
        "shapes": {s: {"rows": sum(c.values()), "correct_pct": pct(c)} for s, c in sorted(by_shape.items())},
        "failure_reasons": {d: dict(c.most_common()) for d, c in reasons.items()},
    }
    write_json(args.output, report)
    with args.output.with_name(args.output.stem + "-failures.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            if not row["ok"]:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({k: report[k] for k in ("seconds", "domains", "shapes", "failure_reasons")}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
