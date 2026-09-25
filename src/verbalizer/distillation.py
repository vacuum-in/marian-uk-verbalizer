"""Shared, deterministic primitives for Gemma-to-Marian distillation.

The teacher and student do not share a tokenizer, so this project uses
sequence-level distillation.  This module deliberately contains no Gemma
runtime dependency: data preparation, validation, and promotion gates remain
available in the Marian environment.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Iterable

from .data.audit import canonical_source, row_flags
from .numeric import NumericKind, detect_numeric_spans


GEMMA_MODEL_ID = "skypro1111/gemma-3-270m-uk-verbalizer"
GEMMA_REVISION = "6781c62ace365e586371705321901eeacf2d9053"
GEMMA_MAX_NEW_TOKENS = 192
GEMMA_MIN_BARE_DIGITS = 7

WHITESPACE_RE = re.compile(r"\s+")
BARE_DIGITS_RE = re.compile(r"\A\s*(\d+)\s*\Z")
BIG_NUMBER_RE = re.compile(r"(?<![\w:/.\-№])(\d{10,13})(?![\w:/.\-])")
GROUPED_NUMBER_RE = re.compile(r"(?<![\w:/.\-№])(\d{1,4}(?:[ \u00a0]\d{1,4})+)(?![\w:/.\-])")
OTHER_NUMBER_CLASS_RE = re.compile(
    r"картк|карта|iban|айбан|телефон|подзвон|дзвон|код\b|на рахунок|"
    r"cvv|імей|imei|паспорт|ідентифікатор",
    re.IGNORECASE,
)
WORD_RE = re.compile(r"[А-Яа-яІіЇїЄєҐґ’'ʼ]+")
DIGIT_WORDS = {
    "0": "нуль",
    "1": "один",
    "2": "два",
    "3": "три",
    "4": "чотири",
    "5": "п'ять",
    "6": "шість",
    "7": "сім",
    "8": "вісім",
    "9": "дев'ять",
}
WORD_TO_DIGIT = {word.replace("'", "’"): digit for digit, word in DIGIT_WORDS.items()}
CRITICAL_OUTPUT_FLAGS = {
    "empty_target",
    "control_character",
    "target_digit_residue",
    "target_introduces_digit",
    "possible_truncation",
}


@dataclass(frozen=True)
class GateResult:
    """One named promotion-gate result."""

    name: str
    passed: bool
    value: float | int | bool
    threshold: float | int | bool


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def stable_row_id(namespace: str, source: str, target: str) -> str:
    return sha256_text(f"{namespace}\0{canonical_source(source)}\0{target}")


def route_bare_digits(value: str) -> str | None:
    """Compatibility no-op: digit readings belong to the selected model."""

    return None


def _group_thousands(digits: str) -> str:
    groups: list[str] = []
    while digits:
        groups.append(digits[-3:])
        digits = digits[:-3]
    return " ".join(reversed(groups))


def teacher_big_number_spacing(value: str) -> str:
    """Compatibility no-op: teachers receive the original source text."""

    return value


def teacher_input(value: str) -> tuple[bool, str]:
    """Return raw source; no deterministic teacher preprocessing is permitted."""

    return False, (value or "").strip()


def teacher_validation_flags(source: str, output: str) -> tuple[list[str], list[str]]:
    """Separate hard rejects from rows requiring human policy review."""

    flags = set(row_flags(source, output))
    hard = sorted(flags & CRITICAL_OUTPUT_FLAGS)
    review = sorted(flags - set(hard))
    if output != unicodedata.normalize("NFC", output):
        hard.append("teacher_output_not_nfc")
    if output and not WORD_RE.search(output):
        hard.append("teacher_output_has_no_words")
    return sorted(set(hard)), sorted(set(review))


def normal_form(value: str, *, apostrophe_tolerant: bool = False) -> str:
    value = unicodedata.normalize("NFC", value)
    value = WHITESPACE_RE.sub(" ", value).strip()
    value = re.sub(r"\s+([.,;:!?])", r"\1", value)
    if apostrophe_tolerant:
        value = value.translate(str.maketrans({"’": "'", "ʼ": "'", "`": "'"}))
    return value


def exact_digit_sequence_fidelity(source: str, prediction: str) -> bool | None:
    """Verify phone/card/IBAN digit order, independent of grouping policy."""

    expected = [
        span.digit_sequence
        for span in detect_numeric_spans(source)
        if span.kind in {NumericKind.PHONE, NumericKind.CARD, NumericKind.IBAN}
    ]
    if not expected:
        return None
    runs: list[str] = []
    current = ""
    for word in WORD_RE.findall(prediction.casefold()):
        digit = WORD_TO_DIGIT.get(word.replace("'", "’").replace("ʼ", "’"))
        if digit is None:
            if current:
                runs.append(current)
                current = ""
        else:
            current += digit
    if current:
        runs.append(current)
    return all(digits in runs for digits in expected)


def _row_reference(row: dict[str, Any]) -> tuple[str, list[str]]:
    primary = str(row.get("primary_reference", row.get("reference", row.get("target", ""))))
    alternatives = row.get("alternative_references", row.get("alternatives", [])) or []
    return primary, [str(value) for value in alternatives]


def _row_prediction(row: dict[str, Any]) -> str:
    for key in ("prediction", "teacher_output", "output"):
        if key in row:
            return str(row[key])
    raise ValueError("evaluation row has no prediction, teacher_output, or output")


def score_evaluation_rows(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Score generic evaluator rows by split using the public-set convention."""

    totals: dict[str, Counter[str]] = defaultdict(Counter)
    critical: Counter[str] = Counter()
    identifier: Counter[str] = Counter()
    for row in rows:
        split = str(row.get("split", "all"))
        prediction = _row_prediction(row)
        primary, alternatives = _row_reference(row)
        refs = [primary, *alternatives]
        normalized_prediction = normal_form(prediction)
        normalized_refs = [normal_form(item) for item in refs]
        tolerant_prediction = normal_form(prediction, apostrophe_tolerant=True)
        tolerant_refs = [normal_form(item, apostrophe_tolerant=True) for item in refs]
        values = totals[split]
        values["rows"] += 1
        values["primary_exact"] += int(normalized_prediction == normalized_refs[0])
        values["accepted_exact"] += int(normalized_prediction in normalized_refs)
        values["tolerant_accepted_exact"] += int(tolerant_prediction in tolerant_refs)
        for flag in row_flags(str(row.get("source", "")), prediction):
            if flag in CRITICAL_OUTPUT_FLAGS:
                critical[flag] += 1
        fidelity = exact_digit_sequence_fidelity(str(row.get("source", "")), prediction)
        if fidelity is not None:
            identifier["rows"] += 1
            identifier["exact"] += int(fidelity)

    def finalize(values: Counter[str]) -> dict[str, float | int]:
        rows_count = values["rows"]
        return {
            "rows": rows_count,
            "primary_exact": values["primary_exact"],
            "accepted_exact": values["accepted_exact"],
            "tolerant_accepted_exact": values["tolerant_accepted_exact"],
            "primary_exact_rate": values["primary_exact"] / rows_count if rows_count else 0.0,
            "accepted_exact_rate": values["accepted_exact"] / rows_count if rows_count else 0.0,
            "tolerant_accepted_exact_rate": (
                values["tolerant_accepted_exact"] / rows_count if rows_count else 0.0
            ),
        }

    all_values: Counter[str] = Counter()
    for values in totals.values():
        all_values.update(values)
    return {
        "overall": finalize(all_values),
        "by_split": {split: finalize(values) for split, values in sorted(totals.items())},
        "critical_output_flags": dict(sorted(critical.items())),
        "identifier_fidelity": {
            **identifier,
            "rate": identifier["exact"] / identifier["rows"] if identifier["rows"] else 1.0,
        },
    }


def promotion_gate(
    teacher_rows: Iterable[dict[str, Any]],
    student_rows: Iterable[dict[str, Any]],
    *,
    allowed_gap_pp: float = 1.0,
    baseline_rows: Iterable[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Compare student quality to teacher and optional E19 baseline by split."""

    teacher = score_evaluation_rows(teacher_rows)
    student = score_evaluation_rows(student_rows)
    baseline = score_evaluation_rows(baseline_rows) if baseline_rows is not None else None
    checks: list[GateResult] = []
    teacher_splits = set(teacher["by_split"])
    if teacher_splits != set(student["by_split"]):
        raise ValueError("teacher and student reports do not contain the same splits")
    for split in sorted(teacher_splits):
        teacher_score = float(teacher["by_split"][split]["primary_exact_rate"])
        student_score = float(student["by_split"][split]["primary_exact_rate"])
        checks.append(
            GateResult(
                f"{split}: within {allowed_gap_pp:.2f}pp of teacher strict exact",
                student_score + allowed_gap_pp / 100 >= teacher_score,
                student_score,
                teacher_score - allowed_gap_pp / 100,
            )
        )
        if baseline is not None:
            if split not in baseline["by_split"]:
                raise ValueError(f"baseline report is missing split {split}")
            baseline_score = float(baseline["by_split"][split]["primary_exact_rate"])
            checks.append(
                GateResult(
                    f"{split}: no E19 strict-exact regression",
                    student_score >= baseline_score,
                    student_score,
                    baseline_score,
                )
            )
    checks.append(
        GateResult(
            "identifier digit sequence fidelity is 100%",
            float(student["identifier_fidelity"]["rate"]) == 1.0,
            float(student["identifier_fidelity"]["rate"]),
            1.0,
        )
    )
    checks.append(
        GateResult(
            "no critical output flags",
            not bool(student["critical_output_flags"]),
            sum(student["critical_output_flags"].values()),
            0,
        )
    )
    return {
        "status": "pass" if all(check.passed for check in checks) else "fail",
        "allowed_gap_pp": allowed_gap_pp,
        "teacher": teacher,
        "student": student,
        "baseline": baseline,
        "checks": [check.__dict__ for check in checks],
    }
