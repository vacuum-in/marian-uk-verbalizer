"""Audit every artifact required to claim complete external 15k parity."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ROWS = 15_000
EXPECTED_DATA_SHA256 = "77cff9a6c1c724c2699e8a2a2350ab09de4f3dd1624a447544512508235c7d0a"
ACCEPTANCE = ROOT / "reports/e5-final-acceptance-15k-segmented.json"
REGRESSIONS = ROOT / "reports/e5-final-regressions-segmented.json"
REGRESSION_DATA = ROOT / "data/challenge/user_target_regressions.jsonl"
CAPTURE = ROOT / "data/reference/verbalizer_skravchenko_gemma3_15k.jsonl"
CAPTURE_MANIFEST = CAPTURE.with_suffix(".manifest.json")
PARITY = ROOT / "reports/reference-gemma3-parity.json"
DISAGREEMENTS = ROOT / "reports/reference-gemma3-disagreements.jsonl"
OUTPUT = ROOT / "reports/external-parity-completion-audit.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def jsonl_summary(path: Path, index_key: str = "index") -> dict[str, Any]:
    rows = 0
    indexes: set[int] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            rows += 1
            if index_key in record:
                index = int(record[index_key])
                if index in indexes:
                    raise ValueError(f"duplicate index {index} at {path}:{line_number}")
                indexes.add(index)
    return {
        "rows": rows,
        "unique_indexes": len(indexes),
        "minimum_index": min(indexes) if indexes else None,
        "maximum_index": max(indexes) if indexes else None,
    }


def audit() -> dict[str, Any]:
    failures: list[str] = []
    checks: dict[str, Any] = {}

    def require(name: str, condition: bool, evidence: Any) -> None:
        checks[name] = {"pass": bool(condition), "evidence": evidence}
        if not condition:
            failures.append(name)

    acceptance = load_json(ACCEPTANCE)
    acceptance_overall = acceptance["metrics"]["overall"]
    require(
        "frozen_dataset_identity",
        acceptance["data"]["rows"] == EXPECTED_ROWS
        and acceptance["data"]["sha256"] == EXPECTED_DATA_SHA256,
        acceptance["data"],
    )
    require(
        "local_segmented_generation_policy",
        acceptance["generation"].get("segment_sentences") is True
        and acceptance["generation"].get("num_beams") == 1,
        acceptance["generation"],
    )
    require(
        "local_frozen_quality",
        acceptance_overall["rows"] == EXPECTED_ROWS
        and acceptance_overall["exact"] == EXPECTED_ROWS
        and acceptance_overall["cer"] == 0.0
        and acceptance_overall["wer"] == 0.0
        and acceptance_overall["output_truncations"] == 0
        and acceptance_overall["repetition_loops"] == 0
        and acceptance_overall["invalid_unicode"] == 0,
        {
            key: acceptance_overall[key]
            for key in (
                "rows",
                "exact",
                "cer",
                "wer",
                "output_truncations",
                "repetition_loops",
                "invalid_unicode",
            )
        },
    )

    regressions = load_json(REGRESSIONS)
    regression_overall = regressions["metrics"]["overall"]
    expected_regressions = sum(
        1
        for line in REGRESSION_DATA.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    require(
        "user_regressions",
        regression_overall["rows"] == expected_regressions
        and regression_overall["exact"] == expected_regressions
        and regression_overall["cer"] == 0.0
        and regression_overall["wer"] == 0.0,
        {
            key: regression_overall[key]
            for key in ("rows", "exact", "cer", "wer")
        },
    )

    capture_summary = jsonl_summary(CAPTURE)
    require(
        "complete_ordered_capture",
        capture_summary == {
            "rows": EXPECTED_ROWS,
            "unique_indexes": EXPECTED_ROWS,
            "minimum_index": 0,
            "maximum_index": EXPECTED_ROWS - 1,
        },
        capture_summary,
    )

    if CAPTURE_MANIFEST.exists():
        manifest = load_json(CAPTURE_MANIFEST)
        manifest_evidence = {
            "rows": manifest.get("rows"),
            "data_sha256": manifest.get("data", {}).get("sha256"),
            "declared_capture_sha256": manifest.get("capture", {}).get("sha256"),
            "actual_capture_sha256": sha256_file(CAPTURE),
            "reference": manifest.get("reference"),
        }
        manifest_ok = (
            manifest_evidence["rows"] == EXPECTED_ROWS
            and manifest_evidence["data_sha256"] == EXPECTED_DATA_SHA256
            and manifest_evidence["declared_capture_sha256"]
            == manifest_evidence["actual_capture_sha256"]
            and manifest_evidence["reference"].get("endpoint") == "/normalize"
            and manifest_evidence["reference"].get("model") == "Gemma 3 270M"
        )
    else:
        manifest_evidence = {"missing": str(CAPTURE_MANIFEST)}
        manifest_ok = False
    require("capture_manifest_integrity", manifest_ok, manifest_evidence)

    parity = load_json(PARITY)
    gate = parity["quality_gate"]
    require(
        "strict_external_quality_gate",
        parity["scope"]["capture_complete"] is True
        and parity["scope"]["captured_rows"] == EXPECTED_ROWS
        and gate["status"] == "pass"
        and gate["all_comparative_checks_pass"] is True
        and all(gate["checks"].values()),
        {"scope": parity["scope"], "quality_gate": gate},
    )

    disagreement_summary = jsonl_summary(DISAGREEMENTS)
    classification = parity["disagreement_classification"]
    class_total = sum(classification["website_difference_classes"].values())
    require(
        "exhaustive_disagreement_classification",
        classification["classified_rows"] == EXPECTED_ROWS
        and class_total == EXPECTED_ROWS
        and disagreement_summary["rows"]
        == classification["written_disagreement_rows"]
        and disagreement_summary["unique_indexes"] == disagreement_summary["rows"],
        {
            "classification": classification,
            "jsonl": disagreement_summary,
            "website_class_total": class_total,
        },
    )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if not failures else "fail",
        "failures": failures,
        "checks": checks,
    }


def main() -> None:
    try:
        report = audit()
    except Exception as exc:
        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "status": "fail",
            "failures": [f"audit_exception: {type(exc).__name__}: {exc}"],
            "checks": {},
        }
    OUTPUT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": report["status"], "failures": report["failures"]}))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
