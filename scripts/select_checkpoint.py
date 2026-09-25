"""Rank checkpoint evaluation reports using the frozen validation composite."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.verbalizer.config import load_yaml  # noqa: E402
from src.verbalizer.evaluation.selection import composite_score  # noqa: E402
from src.verbalizer.training.manifest import write_json  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", type=Path, nargs="+")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/evaluate/default.yaml")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_yaml(args.config)
    ranked = []
    for path in args.reports:
        report = json.loads(path.read_text(encoding="utf-8"))
        ranked.append(
            {
                "checkpoint": report["checkpoint"],
                "report": str(path.resolve()),
                **composite_score(report["metrics"]["overall"], config),
            }
        )
    ranked.sort(key=lambda row: (bool(row["eligible"]), float(row["score"])), reverse=True)
    result = {
        "selection_config": str(args.config.resolve()),
        "selection_name": config["name"],
        "selected": ranked[0]["checkpoint"] if ranked and ranked[0]["eligible"] else None,
        "ranking": ranked,
    }
    write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
