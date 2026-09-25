"""Routing-only structured-span detection shared by verbalizer runtimes.

This module classifies input; it never converts, repairs, rejects, or restores
spoken text.  Numerical and formula rendering belongs to the selected model.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
from enum import Enum
from pathlib import Path


NUMERIC_POLICY_VERSION = "model-routing-v1"

SUBSCRIPT_DIGITS = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
SUPERSCRIPT_DIGITS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")
ALL_SCRIPT_DIGITS = {**SUBSCRIPT_DIGITS, **SUPERSCRIPT_DIGITS}


class NumericKind(str, Enum):
    ROMAN = "roman"
    ARITHMETIC = "arithmetic"
    FORMULA = "formula"
    IBAN = "iban"
    CARD = "card"
    PHONE = "phone"
    IDENTIFIER_SUFFIX = "identifier_suffix"
    CODE = "code"
    DATE = "date"
    TIME = "time"
    RANGE = "range"
    MONEY = "money"
    PERCENT = "percent"
    MEASUREMENT = "measurement"
    VERSION = "version"
    DECIMAL = "decimal"
    ORDINAL = "ordinal"
    CARDINAL = "cardinal"


@dataclass(frozen=True)
class NumericSpan:
    start: int
    end: int
    text: str
    kind: NumericKind
    digit_sequence: str
    groups: tuple[str, ...]
    separators: tuple[str, ...]
    sign: str | None
    value: str | None
    candidates: tuple[str, ...] = ()
    deterministic: bool = False

    @property
    def safe_reading(self) -> str:
        """Compatibility property; routing never produces a spoken reading."""

        return self.text


@dataclass(frozen=True)
class FidelityResult:
    text: str
    accepted: bool
    warnings: tuple[str, ...]
    failures: tuple[NumericSpan, ...]


@dataclass(frozen=True)
class PreparedInput:
    text: str


FORMULA_RE = re.compile(r"(?<![A-Za-zА-Яа-яІіЇїЄєҐґ])(?:[A-Z][a-z]?[₀-₉⁰-⁹0-9]*){2,}(?![A-Za-z])")
SCRIPT_FORMULA_RE = re.compile(
    r"(?<![A-Za-zА-Яа-яІіЇїЄєҐґ])(?=[A-Za-zА-Яа-яІіЇїЄєҐґ₀-₉⁰-⁹]*[₀-₉⁰-⁹])"
    r"(?:[A-ZА-ЯІЇЄҐ][a-zа-яіїєґ]?[₀-₉⁰-⁹]*)+(?![A-Za-zА-Яа-яІіЇїЄєҐґ])"
)
IBAN_RE = re.compile(r"(?<!\w)UA\d{2}(?:[ ]?\d{4}){4,7}(?:[ ]?\d{1,4})?(?!\w)", re.I)
CARD_RE = re.compile(r"(?<!\d)\d{4}(?:[ -]\d{4}){3}(?!\d)")
PHONE_RE = re.compile(r"(?<!\w)(?:\+?380(?:[ \u00a0()\-]*\d){9}|0(?:[ \u00a0()\-]*\d){9})(?!\w)")
IDENTIFIER_SUFFIX_RE = re.compile(r"\b(?:IBAN|картк\w*|рахун\w*|account)\b[^.!?\n]{0,35}?(?:закінчу\w*\s+на|останні\w*)\s+(?P<number>\d{2,12})(?!\d)", re.I)
CODE_RE = re.compile(r"\b(?:код(?:\s+партії)?|замовлення|рейс|CVV|IMEI|ID)\s*[:№#-]?\s*(?P<number>[A-Za-zА-Яа-яІіЇїЄєҐґ]*\d[\w/-]*)", re.I)
DATE_RE = re.compile(r"(?<!\d)\d{1,2}[./-]\d{1,2}[./-]\d{2,4}(?!\d)")
TIME_RE = re.compile(r"(?<!\d)\d{1,2}:\d{2}(?::\d{2})?(?!\d)")
# Routing-only: valid standalone Roman numerals I..XLIX are model-owned.
# This does not parse or verbalize their values; it only selects Marian.
ROMAN_RE = re.compile(
    r"(?<![A-Za-zА-Яа-яІіЇїЄєҐґ])"
    r"(?:XL(?:IX|IV|V(?:I{0,3})|I{1,3})?|X{1,3}(?:IX|IV|V(?:I{0,3})|I{1,3})?|IX|IV|V(?:I{0,3})|I{1,3})"
    r"(?![A-Za-zА-Яа-яІіЇїЄєҐґ])"
)
ARITHMETIC_RE = re.compile(r"(?<![\w.])[+\-−]?\d+(?:[.,]\d+)?(?:\s*(?:[+\-−]|[×*÷/])\s*[+\-−]?\d+(?:[.,]\d+)?)+\s*=\s*[+\-−]?\d+(?:[.,]\d+)?(?!\w)")
RANGE_RE = re.compile(r"(?<!\w)[+\-−]?\d+(?:[.,]\d+)?\s*[–—-]\s*[+\-−]?\d+(?:[.,]\d+)?(?!\w)")
MONEY_RE = re.compile(r"(?<!\w)[+\-−]?\d+(?:[ .]\d{3})*(?:[.,]\d+)?\s*(?:₴|\$|€|£|грн\.?\b|UAH\b|USD\b|EUR\b|GBP\b|PLN\b|CHF\b)", re.I)
PERCENT_RE = re.compile(r"(?<!\w)[+\-−]?\d+(?:[.,]\d+)?\s*%")
MEASUREMENT_RE = re.compile(r"(?<!\w)[+\-−]?\d+(?:[.,]\d+)?\s*(?:км|см|мм|м|кг|г|мг|л|мл|°C|°F|Мбіт/с|км/год)\b", re.I)
VERSION_RE = re.compile(r"(?<!\w)[vV]?\d+(?:\.\d+){1,3}(?!\w|\.\d)")
DECIMAL_RE = re.compile(r"(?<!\w)[+\-−]?\d+[.,]\d+(?![\w.,]\d)")
ORDINAL_RE = re.compile(r"(?<!\w)\d+[-‑]?(?:й|го|му|м|та|ту|гу|ої|ї|ю|ше|ша|ге|тє|те|є|х)(?!\w)", re.I)
CARDINAL_RE = re.compile(r"(?<![\w₀-₉⁰-⁹])(?:[+\-−]?\d{1,3}(?:[ \u00a0]\d{3})+|[+\-−]?\d+)(?![\w₀-₉⁰-⁹])")
DIGIT_RUN_RE = re.compile(r"\d+")


def _ascii_digits(value: str) -> str:
    return "".join(character for character in value.translate(ALL_SCRIPT_DIGITS) if character.isdigit())


def _groups(value: str) -> tuple[str, ...]:
    return tuple(item for item in re.split(r"\D+", value.translate(ALL_SCRIPT_DIGITS)) if item)


def _separators(value: str) -> tuple[str, ...]:
    return tuple(item for item in re.findall(r"\D+", value) if item)


def _value(text: str) -> str | None:
    candidate = re.sub(r"[^0-9+.-]", "", text.strip().replace(" ", "").replace("−", "-").replace(",", "."))
    try:
        return str(Decimal(candidate)) if candidate and candidate not in {"+", "-", "."} else None
    except InvalidOperation:
        return None


def detect_numeric_spans(source: str) -> tuple[NumericSpan, ...]:
    """Classify non-overlapping spans for model routing and offline reporting."""

    source = unicodedata.normalize("NFC", source)
    specs: tuple[tuple[NumericKind, re.Pattern[str], str | None], ...] = (
        (NumericKind.FORMULA, SCRIPT_FORMULA_RE, None), (NumericKind.FORMULA, FORMULA_RE, None),
        (NumericKind.IBAN, IBAN_RE, None), (NumericKind.CARD, CARD_RE, None),
        (NumericKind.PHONE, PHONE_RE, None), (NumericKind.IDENTIFIER_SUFFIX, IDENTIFIER_SUFFIX_RE, "number"),
        (NumericKind.CODE, CODE_RE, "number"), (NumericKind.DATE, DATE_RE, None),
        (NumericKind.TIME, TIME_RE, None), (NumericKind.ROMAN, ROMAN_RE, None),
        (NumericKind.ARITHMETIC, ARITHMETIC_RE, None),
        (NumericKind.RANGE, RANGE_RE, None), (NumericKind.MONEY, MONEY_RE, None),
        (NumericKind.PERCENT, PERCENT_RE, None), (NumericKind.MEASUREMENT, MEASUREMENT_RE, None),
        (NumericKind.VERSION, VERSION_RE, None), (NumericKind.DECIMAL, DECIMAL_RE, None),
        (NumericKind.ORDINAL, ORDINAL_RE, None), (NumericKind.CARDINAL, CARDINAL_RE, None),
        (NumericKind.CODE, DIGIT_RUN_RE, None),
    )
    accepted: list[NumericSpan] = []
    for kind, pattern, group in specs:
        for match in pattern.finditer(source):
            start, end = match.span(group) if group else match.span()
            if any(start < span.end and span.start < end for span in accepted):
                continue
            text = source[start:end]
            digits = _ascii_digits(text)
            if digits or kind is NumericKind.ROMAN:
                accepted.append(NumericSpan(start, end, text, kind, digits, _groups(text), _separators(text), text[:1] if text[:1] in "+-−" else None, _value(text)))
    return tuple(sorted(accepted, key=lambda span: span.start))


def prepare_protected_input(source: str) -> PreparedInput:
    """Return the original text; routing never inserts markers or readings."""

    return PreparedInput(unicodedata.normalize("NFC", source))


def prepare_training_pair(source: str, target: str) -> tuple[str, str, tuple[str, ...]]:
    """Return model-supervised text unchanged; no rule fabricates training targets."""

    return unicodedata.normalize("NFC", source), unicodedata.normalize("NFC", target), ()


def numeric_model_input(source: str) -> str:
    return unicodedata.normalize("NFC", source)


def numeric_fidelity_failures(source: str, prediction: str) -> tuple[NumericSpan, ...]:
    """No lexical candidate policy exists in routing-only mode."""

    return ()


def formula_supervision_valid(source: str, target: str) -> bool:
    """Dataset references are model supervision, not rule-checked replacements."""

    return True


def enforce_numeric_fidelity(source: str, prediction: str) -> FidelityResult:
    """Return raw model output without correction, rejection, or validation gates."""

    return FidelityResult(prediction, True, (), ())


def numeric_policy_hash() -> str:
    payload = {"version": NUMERIC_POLICY_VERSION, "kinds": [kind.value for kind in NumericKind]}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def numeric_component_hashes() -> dict[str, str]:
    return {"router": numeric_policy_hash()}


def preprocessing_hash() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def span_as_dict(span: NumericSpan) -> dict[str, object]:
    result = asdict(span)
    result["kind"] = span.kind.value
    return result
