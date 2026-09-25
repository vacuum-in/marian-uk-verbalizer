"""Merge consensus-accepted rows and manual review votes into final verdicts.

Precedence: a manual reviewer's decision > unanimous judges > match with the
reference. Rows whose reference contradicts a user-set domain rule are kept
but marked excluded, so they do not count toward any score.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.verbalizer.evaluation.domains import DOMAINS, domains_of  # noqa: E402

ROMAN_TOKEN = re.compile(r"(?<![A-Za-zА-Яа-яІіЇїЄєҐґ])[IVXLCDM]{2,12}(?![A-Za-zА-Яа-яІіЇїЄєҐґ])")
NUMERAL_WORD = re.compile(r"сорок|п'ятдесят|сто|сот|тисяч", re.I)


def roman_rule_conflict(source: str, reference: str) -> bool:
    """Reference reads a Roman numeral containing L, C, D or M as a number.

    User rule (2026-09-24): Roman numerals use only I, V, X (1-39).
    """

    tokens = [t for t in ROMAN_TOKEN.findall(source) if re.search(r"[LCDM]", t) and re.fullmatch(r"[IVXLCDM]+", t)]
    return bool(tokens) and bool(NUMERAL_WORD.search(reference)) and not re.search(r"\b(ел|сі|це|ді|де|ем)\b", reference.lower())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=ROOT / "reports/hf-uktn-e40-full-raw-marian.json")
    parser.add_argument("--accepted", type=Path, default=ROOT / "reports/consensus-e40/accepted.jsonl")
    parser.add_argument("--judged", type=Path, default=ROOT / "reports/hf-uktn-e40-judged.json",
                        help="in-session three-class labels, used to split 'correct' into ok/minor")
    parser.add_argument("--votes", type=Path, nargs="+", required=True, help="exported votes documents")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/hf-uktn-e40-final-verdicts.json")
    args = parser.parse_args()

    examples = json.loads(args.predictions.read_text(encoding="utf-8"))["examples"]
    accepted = {json.loads(line)["id"]: json.loads(line) for line in args.accepted.read_text(encoding="utf-8").splitlines() if line.strip()}
    session = json.loads(args.judged.read_text(encoding="utf-8"))["verdicts"]
    manual: dict[str, dict] = {}
    for path in args.votes:
        doc = json.loads(path.read_text(encoding="utf-8"))
        for row_id, vote in (doc.get("data") or doc).get("decisions", {}).items():
            manual.setdefault(row_id, {})[path.stem] = vote

    verdicts = {}
    for example in examples:
        row_id = example["id"]
        if row_id in manual:
            decisions = Counter(v["d"] for v in manual[row_id].values())
            label, basis = decisions.most_common(1)[0][0], "manual_review"
        elif row_id in accepted:
            row = accepted[row_id]
            if row["label"] == "wrong":
                label = "wrong"
            elif row["basis"] == "matches_reference":
                label = "ok"
            else:
                label = session.get(row_id, {}).get("verdict", "ok")
                label = label if label in ("ok", "minor") else "ok"
            basis = row["basis"]
        else:
            label, basis = "unreviewed", "missing"
        excluded = roman_rule_conflict(example["source"], example["reference"])
        verdicts[row_id] = {
            "label": label, "basis": basis, "excluded": excluded,
            "domains": sorted(domains_of(example.get("categories", []), example["source"])) or ["other"],
            "source": example["source"], "prediction": example["prediction"], "reference": example["reference"],
        }

    table = defaultdict(Counter)
    for verdict in verdicts.values():
        if verdict["excluded"]:
            continue
        for domain in [*verdict["domains"], "ALL"]:
            table[domain][verdict["label"]] += 1
    summary = {}
    for domain in [*DOMAINS, "other", "ALL"]:
        counts = table.get(domain)
        if not counts:
            continue
        total = sum(counts.values())
        summary[domain] = {
            "rows": total, "ok": counts["ok"], "minor": counts["minor"], "wrong": counts["wrong"],
            "lenient_pct": round(100 * (counts["ok"] + counts["minor"]) / total, 2),
            "strict_pct": round(100 * counts["ok"] / total, 2),
        }
    report = {
        "rows": len(verdicts), "excluded": sum(v["excluded"] for v in verdicts.values()),
        "by_basis": dict(Counter(v["basis"] for v in verdicts.values())),
        "domains": summary, "verdicts": verdicts,
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"{'domain':15} {'rows':>5} {'lenient':>8} {'strict':>7} {'wrong':>6}")
    for domain, s in summary.items():
        print(f"{domain:15} {s['rows']:5} {s['lenient_pct']:8.1f} {s['strict_pct']:7.1f} {s['wrong']:6}")
    print(f"\nexcluded by Roman rule: {report['excluded']}   basis: {report['by_basis']}")


if __name__ == "__main__":
    main()
