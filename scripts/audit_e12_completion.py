"""Audit the evidence for the selected learned-Roman E12 checkpoint."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports/e12-completion-audit.json"


def load(path: str) -> dict[str, Any]:
    value = json.loads((ROOT / path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rows(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for line in handle if line.strip())


def audit() -> dict[str, Any]:
    failures: list[str] = []
    checks: dict[str, Any] = {}

    def require(name: str, condition: bool, evidence: Any) -> None:
        checks[name] = {"pass": bool(condition), "evidence": evidence}
        if not condition:
            failures.append(name)

    roman = load("data/roman_numerals/manifest.json")
    mix = load("data/e12_roman_core_repair/manifest.json")
    require(
        "training_data_isolation",
        all(value == 0 for value in roman["leakage"].values())
        and all(value == 0 for value in mix["leakage"].values())
        and roman["rows"] == {"train": 150_000, "validation": 1_400, "test": 2_500},
        {"roman": roman["leakage"], "e12_mix": mix["leakage"]},
    )

    stages = [
        ("e9-marian-small-roman-replay", "e8-marian-small-structured-repair"),
        ("e9b-marian-small-roman-replay", "e9-marian-small-roman-replay"),
        ("e9c-marian-small-roman-replay", "e9b-marian-small-roman-replay"),
        ("e10-marian-small-roman-structured-repair", "e9c-marian-small-roman-replay"),
        ("e11-marian-small-universal-repair", "e10-marian-small-roman-structured-repair"),
        ("e12-marian-small-roman-core-repair", "e11-marian-small-universal-repair"),
    ]
    lineage = []
    lineage_ok = True
    for stage, parent in stages:
        manifest = load(f"outputs/{stage}/run_manifest.json")
        initialize_from = str(manifest["training"]["initialize_from"])
        lineage.append({"stage": stage, "initialize_from": initialize_from})
        lineage_ok &= initialize_from.endswith(f"outputs/{parent}/final")
    e12_run = load("outputs/e12-marian-small-roman-core-repair/run_manifest.json")
    lineage_ok &= e12_run["data"]["train"]["sha256"] == mix["output_sha256"]["train"]
    require("training_lineage", lineage_ok, lineage)

    roman_report = load("reports/e12-roman-holdout-2500-greedy.json")
    roman_metrics = roman_report["metrics"]["overall"]
    roman_generation = roman_report["generation"]
    require(
        "learned_roman_holdout",
        roman_report["data"]["rows"] == 2_500
        and roman_report["data"]["sha256"] == roman["output_sha256"]["test"]
        and roman_metrics["exact_match"] >= 0.975
        and roman_metrics["output_truncations"] == 0
        and roman_metrics["invalid_unicode"] == 0
        and roman_generation["num_beams"] == 1
        and roman_generation["compute_precision"] == "bf16",
        {"generation": roman_generation, "metrics": roman_metrics},
    )

    structured = load("reports/e12-structured-acceptance-15k-greedy.json")["metrics"]["overall"]
    regressions = load("reports/e12-user-regressions-production.json")["metrics"]["overall"]
    require(
        "structured_and_user_regressions",
        structured["rows"] == 15_000
        and structured["exact_match"] >= 0.998
        and structured["output_truncations"] == 0
        and regressions["rows"] == 4
        and regressions["exact"] == 4,
        {"structured": structured, "regressions": regressions},
    )

    news = load("reports/e12-news-golden5k-greedy.json")
    semantics = load("reports/e12-news-golden5k-semantics.json")
    require(
        "news_semantic_quality",
        news["data"]["rows"] == 5_000
        and news["generation"]["compute_precision"] == "bf16"
        and semantics["sibling_semantic_accuracy"] >= 0.98
        and semantics["strict_semantic_accuracy"] >= 0.98
        and semantics["failure_counts"]["no_digit_leak"] == 0,
        semantics,
    )

    capture = ROOT / "data/reference/verbalizer_skravchenko_gemma3_news_golden5k.jsonl"
    capture_manifest = load("data/reference/verbalizer_skravchenko_gemma3_news_golden5k.manifest.json")
    require(
        "complete_external_capture",
        rows(capture) == 5_000
        and capture_manifest["rows"] == 5_000
        and capture_manifest["capture"]["sha256"] == sha256(capture),
        {"rows": rows(capture), "sha256": sha256(capture)},
    )

    parity = load("reports/e12-news-golden5k-reference-parity.json")
    gate = parity["quality_gate"]
    require(
        "public_reference_no_worse",
        parity["scope"]["capture_complete"] is True
        and parity["scope"]["captured_rows"] == 5_000
        and gate["status"] == "pass"
        and gate["all_comparative_checks_pass"] is True
        and all(gate["checks"].values()),
        gate,
    )

    report = {"status": "pass" if not failures else "fail", "failures": failures, "checks": checks}
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    result = audit()
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    raise SystemExit(0 if result["status"] == "pass" else 1)
