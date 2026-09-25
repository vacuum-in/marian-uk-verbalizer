"""Write one JSON file per manual-review row for seeding the review page's `items`.

Items are keyed "<round>~<row id>" so rounds never collide; the page groups
them by `round`. Output: <out-dir>/items/*.json and <out-dir>/ids.txt.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.verbalizer.evaluation.domains import domains_of  # noqa: E402

JUDGE_KEYS = {"deepseek-v4-pro": "deepseek", "gpt-5.4": "gpt"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", type=Path, required=True, help="manual_review.jsonl from llm_judge.py judge-new")
    parser.add_argument("--predictions", type=Path, required=True, help="the checkpoint's benchmark report (for categories)")
    parser.add_argument("--round", required=True)
    parser.add_argument("--start-order", type=int, default=1000)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    categories = {e["id"]: e.get("categories", []) for e in json.loads(args.predictions.read_text(encoding="utf-8"))["examples"]}
    items_dir = args.out_dir / "items"
    items_dir.mkdir(parents=True, exist_ok=True)
    ids = []
    for index, line in enumerate(args.review.read_text(encoding="utf-8").splitlines()):
        row = json.loads(line)
        doc_id = f"{args.round.lower()}~{row['id']}"
        item = {
            "id": doc_id, "row_id": row["id"], "round": args.round, "order": args.start_order + index,
            "domains": sorted(domains_of(categories.get(row["id"], []), row["source"])) or ["other"],
            "source": row["source"], "output": row["prediction"], "reference": row["reference"],
            "judges": {"claude": None, **{JUDGE_KEYS[m]: v for m, v in row["votes"].items() if m in JUDGE_KEYS}},
            "reasons": {JUDGE_KEYS[m]: (r or "")[:400] for m, r in row["reasons"].items() if m in JUDGE_KEYS},
        }
        (items_dir / f"{doc_id}.json").write_text(json.dumps(item, ensure_ascii=False), encoding="utf-8")
        ids.append(doc_id)
    (args.out_dir / "ids.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")
    print(f"{len(ids)} items for round {args.round} in {items_dir}")


if __name__ == "__main__":
    main()
