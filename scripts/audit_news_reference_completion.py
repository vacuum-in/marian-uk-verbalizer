"""Audit evidence that E8 is no worse than the public model on news golden 5k."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_TEST_ROWS = 5_000
EXPECTED_STRUCTURED_ROWS = 15_000
NEWS_MANIFEST = ROOT / "data/news_domain/manifest.json"
REPLAY_MANIFEST = ROOT / "data/news_domain_replay/manifest.json"
LOCAL_REPORT = ROOT / "reports/e8-news-golden5k.json"
STRUCTURED_REPORT = ROOT / "reports/e8-structured-acceptance-15k.json"
REGRESSION_REPORT = ROOT / "reports/e8-user-regressions.json"
SEMANTIC_REPORT = ROOT / "reports/e8-news-golden5k-semantics.json"
REGRESSION_DATA = ROOT / "data/challenge/user_target_regressions.jsonl"
CAPTURE = ROOT / "data/reference/verbalizer_skravchenko_gemma3_news_golden5k.jsonl"
CAPTURE_MANIFEST = CAPTURE.with_suffix(".manifest.json")
PARITY = ROOT / "reports/e8-news-golden5k-reference-parity.json"
E7_RUN = ROOT / "outputs/e7-marian-small-news-structured-replay/run_manifest.json"
E8_RUN = ROOT / "outputs/e8-marian-small-structured-repair/run_manifest.json"
OUTPUT = ROOT / "reports/e8-news-reference-completion-audit.json"


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def nonempty_lines(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for line in handle if line.strip())


def audit() -> dict[str, Any]:
    failures: list[str] = []
    checks: dict[str, Any] = {}

    def require(name: str, condition: bool, evidence: Any) -> None:
        checks[name] = {"pass": bool(condition), "evidence": evidence}
        if not condition:
            failures.append(name)

    news = load_json(NEWS_MANIFEST)
    replay = load_json(REPLAY_MANIFEST)
    require(
        "dataset_isolation",
        news["rows"]["test"] == EXPECTED_TEST_ROWS
        and all(value == 0 for value in news["leakage"].values())
        and all(value == 0 for value in replay["leakage"].values()),
        {"news": news["leakage"], "replay": replay["leakage"]},
    )

    e7 = load_json(E7_RUN)
    e8 = load_json(E8_RUN)
    require(
        "training_lineage",
        e7["data"]["train"]["sha256"] == replay["output_sha256"]["train"]
        and str(e8["training"]["initialize_from"]).endswith(
            "outputs/e7-marian-small-news-structured-replay/final"
        ),
        {
            "e7_train": e7["data"]["train"],
            "e8_initialize_from": e8["training"]["initialize_from"],
        },
    )

    local = load_json(LOCAL_REPORT)
    local_overall = local["metrics"]["overall"]
    require(
        "local_news_report",
        local["data"]["rows"] == EXPECTED_TEST_ROWS
        and local["data"]["sha256"] == news["output_sha256"]["test"]
        and str(local["checkpoint"]).endswith(
            "outputs/e8-marian-small-structured-repair/final"
        )
        and local["generation"].get("enforce_output_fidelity") is True
        and local_overall["output_truncations"] == 0
        and local_overall["invalid_unicode"] == 0,
        {"data": local["data"], "overall": local_overall},
    )

    semantics = load_json(SEMANTIC_REPORT)
    require(
        "local_news_semantic_quality",
        semantics["rows"] == EXPECTED_TEST_ROWS
        and semantics["sibling_semantic_accuracy"] >= 0.98
        and semantics["strict_semantic_accuracy"] >= 0.98
        and semantics["failure_counts"]["no_digit_leak"] == 0,
        semantics,
    )

    structured = load_json(STRUCTURED_REPORT)["metrics"]["overall"]
    regressions = load_json(REGRESSION_REPORT)["metrics"]["overall"]
    expected_regressions = nonempty_lines(REGRESSION_DATA)
    require(
        "retained_structured_quality",
        structured["rows"] == EXPECTED_STRUCTURED_ROWS
        and structured["exact"] == EXPECTED_STRUCTURED_ROWS
        and structured["cer"] == 0.0
        and structured["wer"] == 0.0
        and regressions["rows"] == expected_regressions
        and regressions["exact"] == expected_regressions,
        {"structured": structured, "regressions": regressions},
    )

    capture_manifest = load_json(CAPTURE_MANIFEST)
    capture_evidence = {
        "rows": nonempty_lines(CAPTURE),
        "declared_rows": capture_manifest["rows"],
        "declared_sha256": capture_manifest["capture"]["sha256"],
        "actual_sha256": sha256_file(CAPTURE),
        "data_sha256": capture_manifest["data"]["sha256"],
    }
    require(
        "complete_reference_capture",
        capture_evidence["rows"] == EXPECTED_TEST_ROWS
        and capture_evidence["declared_rows"] == EXPECTED_TEST_ROWS
        and capture_evidence["declared_sha256"] == capture_evidence["actual_sha256"]
        and capture_evidence["data_sha256"] == news["output_sha256"]["test"],
        capture_evidence,
    )

    parity = load_json(PARITY)
    gate = parity["quality_gate"]
    require(
        "public_reference_no_worse_gate",
        parity["scope"]["capture_complete"] is True
        and parity["scope"]["captured_rows"] == EXPECTED_TEST_ROWS
        and gate["status"] == "pass"
        and gate["require_local_target_exact"] is False
        and gate["all_comparative_checks_pass"] is True
        and all(gate["checks"].values()),
        {"scope": parity["scope"], "quality_gate": gate},
    )

    report = {
        "status": "pass" if not failures else "fail",
        "failures": failures,
        "checks": checks,
    }
    OUTPUT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


if __name__ == "__main__":
    result = audit()
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    raise SystemExit(0 if result["status"] == "pass" else 1)
