import pytest

from scripts.compare_reference_parity import classify_output, enforce_requirements, parity_gate


def test_difference_classifier_distinguishes_apostrophe_only() -> None:
    result = classify_output(
        "Потрібно 15 одиниць.",
        "Потрібно п’ятнадцять одиниць.",
        "Потрібно п'ятнадцять одиниць.",
    )
    assert result["difference_class"] == "apostrophe_only"
    assert result["apostrophe_canonical_exact"] is True


def test_difference_classifier_marks_excess_semantic_risk() -> None:
    result = classify_output(
        "Температура -24 °C, ціна $25.",
        "Температура мінус двадцять чотири градуси Цельсія, ціна двадцять п’ять доларів.",
        "Температура двадцять чотири градуси, ціна двадцять п’ять.",
    )
    assert result["difference_class"] == "semantic_risk"
    assert result["excess_semantic_flags"] == [
        "currency_unverified",
        "sign_not_preserved",
    ]


def _metric(exact: float, cer: float, wer: float, critical: int = 0) -> dict:
    return {
        "overall": {
            "exact_match": exact,
            "cer": cer,
            "wer": wer,
            "empty_predictions": critical,
            "likely_deletions": 0,
            "likely_insertions": 0,
            "repetition_loops": 0,
            "punctuation_corruptions": 0,
            "invalid_unicode": 0,
            "output_truncations": 0,
        }
    }


def test_parity_gate_requires_complete_capture_for_pass() -> None:
    comparisons = {
        "local_vs_dataset_target": _metric(1.0, 0.0, 0.0),
        "remote_vs_dataset_target": _metric(0.5, 0.1, 0.2, critical=2),
    }
    reviews = {
        "local": {"excess_flagged_rows": 0},
        "website": {"excess_flagged_rows": 3},
    }
    provisional = parity_gate(comparisons, reviews, capture_complete=False)
    complete = parity_gate(comparisons, reviews, capture_complete=True)
    assert provisional["status"] == "provisional"
    assert provisional["all_comparative_checks_pass"] is True
    assert complete["status"] == "pass"


def test_parity_gate_can_measure_no_worse_quality_without_perfect_target_exact() -> None:
    comparisons = {
        "local_vs_dataset_target": _metric(0.2, 0.08, 0.15),
        "remote_vs_dataset_target": _metric(0.1, 0.13, 0.24, critical=2),
    }
    reviews = {
        "local": {"excess_flagged_rows": 1},
        "website": {"excess_flagged_rows": 3},
    }
    strict = parity_gate(comparisons, reviews, capture_complete=True)
    no_worse = parity_gate(
        comparisons,
        reviews,
        capture_complete=True,
        require_local_exact=False,
    )
    assert strict["status"] == "fail"
    assert no_worse["status"] == "pass"
    assert no_worse["require_local_target_exact"] is False


def test_required_final_gate_rejects_partial_report() -> None:
    report = {
        "scope": {"capture_complete": False, "captured_rows": 10, "frozen_dataset_rows": 15_000},
        "quality_gate": {"status": "provisional"},
    }
    with pytest.raises(RuntimeError, match="10 / 15000"):
        enforce_requirements(report, require_complete=True, require_gate_pass=True)
