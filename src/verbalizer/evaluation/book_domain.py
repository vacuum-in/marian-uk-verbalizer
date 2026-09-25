"""Reference-free book-domain metrics: prose identity and numeric round-trip.

Offline QA only. These functions score model output; they never alter it.
"""

from __future__ import annotations

import re
import unicodedata

from ..numeric import NumericKind, detect_numeric_spans
from .._audit_numparse import SCALES, UNITS, WORD, parse_numbers

APOSTROPHES = str.maketrans({"’": "'", "ʼ": "'"})
ROMAN_VALUES = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}


def strict_text(value: str) -> str:
    return unicodedata.normalize("NFC", value).translate(APOSTROPHES).strip()


def content_text(value: str) -> str:
    """Ignore whitespace-only differences, including around dashes."""

    text = strict_text(value).replace(" ", " ")
    text = re.sub(r"\s*([—–])\s*", r" \1 ", text)
    return re.sub(r"\s+", " ", text).strip()


def identity_verdict(source: str, prediction: str) -> str:
    if strict_text(source) == strict_text(prediction):
        return "exact"
    if content_text(source) == content_text(prediction):
        return "whitespace_only"
    return "changed"


def roman_value(text: str) -> int | None:
    values = [ROMAN_VALUES.get(character) for character in text.upper()]
    if not values or None in values:
        return None
    total = 0
    for index, value in enumerate(values):
        total += -value if index + 1 < len(values) and value < values[index + 1] else value
    return total


FINAL_WORD_SUFFIXES = ("ий", "ого", "ому", "их", "ими", "ій", "ої", "ою", "им")


def _magnitude(value: int) -> int:
    return 2 if value >= 100 else 1 if value >= 20 else 0


def segmented_numbers(text: str) -> list[int]:
    """Like ``parse_numbers`` but split digit-wise readings into separate values.

    ``вісімдесят три сімдесят два`` is two numbers (83, 72): a new number
    starts whenever a numeral word does not drop in magnitude. Ordinal and
    oblique unit forms always end a number (``шістдесятих один`` is 60, 1).
    """

    out: list[int] = []
    total = current = 0
    last: int | None = None
    active = False

    def flush() -> None:
        nonlocal total, current, last, active
        if active:
            out.append(total + current)
        total = current = 0
        last, active = None, False

    for raw in WORD.findall(text):
        word = raw.lower().replace("’", "'")
        if word in SCALES:
            total += (current or 1) * SCALES[word]
            current, last, active = 0, 3, True
        elif word in UNITS:
            magnitude = _magnitude(UNITS[word])
            if last is not None and last != 3 and magnitude >= last:
                flush()
            current += UNITS[word]
            last, active = magnitude, True
            if word.endswith(FINAL_WORD_SUFFIXES):
                flush()
        else:
            flush()
    flush()
    return out


def _read_digitwise(digits: str, segments: list[int]) -> bool:
    """True when consecutive spoken values spell ``digits`` (``32 0 27`` for 32027)."""

    for start in range(len(segments)):
        spelled = ""
        for value in segments[start:]:
            spelled += str(value)
            if spelled == digits:
                return True
            if not digits.startswith(spelled):
                break
    return False


def _merge_thousands_groups(source: str, spans) -> list[tuple[object, str]]:
    """Pair each span with its digits, joining ``2`` + ``500 м`` split by the detector."""

    merged: list[tuple[object, str]] = []
    for span in spans:
        if merged:
            previous, digits = merged[-1]
            gap = source[previous.end : span.start]
            if gap in {" ", " "} and span.groups and len(span.groups[0]) == 3 and len(digits) <= 3:
                merged[-1] = (previous, digits + span.digit_sequence)
                continue
        merged.append((span, span.digit_sequence))
    return merged


def _span_ok(span, digits: str, spoken_values: set[int], segments: list[int]) -> bool:
    if span.kind is NumericKind.ROMAN:
        value = roman_value(span.text)
        return value is not None and value in spoken_values
    groups = [group for group in span.groups if group]
    if not groups:
        return True
    if span.kind is NumericKind.TIME:
        groups = [groups[0], *[group for group in groups[1:] if int(group) != 0]]
    if span.kind is NumericKind.DATE and len(groups) == 3:
        groups = [groups[0], groups[2]]
    if int(digits) in spoken_values or _read_digitwise(digits, segments):
        return True
    return all(int(group) in spoken_values or _read_digitwise(group, segments) for group in groups)


def numeric_roundtrip(source: str, prediction: str) -> dict[str, object]:
    """Check that every numeric span in ``source`` is spoken in ``prediction``.

    A span passes when its whole digit sequence, or each of its digit groups,
    appears among the values parsed back from the prediction. Leftover ASCII
    digits in the prediction are reported separately.
    """

    spans = detect_numeric_spans(source)
    # Punctuation separates numbers ("тисяча дев'ятсот сорок — тисяча дев'ятсот сорок один").
    chunks = [c for c in re.split(r"[—–,;:()«»\"]|\s-\s", prediction) if c.strip()]
    segments = [v for c in chunks for v in segmented_numbers(c)]
    spoken = set(v for c in chunks for v in parse_numbers(c)) | set(segments)
    failed = [
        span.text
        for span, digits in _merge_thousands_groups(source, spans)
        if not _span_ok(span, digits, spoken, segments)
    ]
    return {
        "spans": [span.text for span in spans],
        "failed_spans": failed,
        "digits_left": bool(re.search(r"\d", prediction)),
        "ok": not failed and not re.search(r"\d", prediction),
    }
