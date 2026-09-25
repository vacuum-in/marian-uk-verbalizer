"""Model-routing helpers for the verbalization service.

This module deliberately does not normalize, verbalize, repair, reject, or
post-process text.  It chooses one owner for the complete input:

* Marian for anything that may need verbalization;
* pass-through for plain Cyrillic prose that needs none.

The decision looks only at the shape of the text (character classes,
capitalisation, position); it uses no word lists and never produces a
spoken form.
"""

from __future__ import annotations

import unicodedata
import re
from dataclasses import dataclass
from enum import Enum

from .numeric import FidelityResult, enforce_numeric_fidelity, prepare_protected_input


CAPITAL_ABBREVIATION_RE = re.compile(
    r"(?<![A-Za-zА-Яа-яІіЇїЄєҐґ])[A-ZА-ЯІЇЄҐ]{1,4}(?![A-Za-zА-Яа-яІіЇїЄєҐґ])"
)
# Characters that plain Ukrainian prose is made of. Anything else -- digits,
# Latin letters, symbols, emoji -- sends the whole input to Marian.
PLAIN_PROSE_RE = re.compile(r"[А-Яа-яІіЇїЄєҐґ'’ʼ\s.,!?…:;«»\"„“”()\[\]\-–—*]+")
EMOTICON_RE = re.compile(r"[:;]-?[()DPp]|\([:;]")
ELONGATION_RE = re.compile(r"([А-Яа-яІіЇїЄєҐґ])\1\1")
# A short lowercase token ending in a period followed by more text ("вул. ", "т. зв.",
# "м. Київ"). Over-routing a sentence end like "ліс. Потім" to Marian is harmless.
SHORT_ABBREVIATION_RE = re.compile(
    r"(?<![А-Яа-яІіЇїЄєҐґ'’])(?:[а-яіїєґ]{1,5}\.(?=\s*[а-яіїєґ\d,])|[а-яіїєґ]{1,5}\.(?=\s*[А-ЯІЇЄҐ])"
    r"|[а-яіїєґ]{1,4}\.$|[А-ЯІЇЄҐ][а-яіїєґ]?\.(?=\s*[А-ЯІЇЄҐ]))"
)
LONG_CAPS_RE = re.compile(r"(?<![А-Яа-яІіЇїЄєҐґ])[А-ЯІЇЄҐ]{5,}(?![А-Яа-яІіЇїЄєҐґ])")
CLAUSE_START_RE = re.compile(r"(?:^|[.!?…:;«\"„“(]\s*|[—–-]\s*)$")
WORD_RE = re.compile(r"[А-Яа-яІіЇїЄєҐґ'’ʼ]+")


class TextRoute(str, Enum):
    PASSTHROUGH = "passthrough"
    MARIAN = "marian"


@dataclass(frozen=True)
class RoutedText:
    route: TextRoute
    text: str


def _is_caps_run(source: str) -> bool:
    """Shouted text: most words upper-case (emphasis, not abbreviations)."""

    words = WORD_RE.findall(source)
    caps = [w for w in words if len(w) > 1 and w.isupper()]
    return len(caps) >= 3 and len(caps) / max(1, len(words)) > 0.5


def _needs_marian(source: str) -> bool:
    if not PLAIN_PROSE_RE.fullmatch(source) or EMOTICON_RE.search(source):
        return True
    if ELONGATION_RE.search(source) or SHORT_ABBREVIATION_RE.search(source):
        return True
    if _is_caps_run(source):
        return False
    if LONG_CAPS_RE.search(source):
        return True
    for match in CAPITAL_ABBREVIATION_RE.finditer(source):
        after = source[match.end():match.end() + 1]
        # A one-letter word opening a clause ("Я", "А", "У") is ordinary prose.
        if len(match.group()) == 1 and CLAUSE_START_RE.search(source[:match.start()]) and after in ("", " ", " ", ","):
            continue
        return True
    return False


def route_text(value: str) -> RoutedText:
    """Choose an owner without changing the text."""

    source = unicodedata.normalize("NFC", value)
    return RoutedText(TextRoute.MARIAN if _needs_marian(source) else TextRoute.PASSTHROUGH, source)


def prepare_model_input(value: str) -> str:
    """Return original input for Marian; no rule-based preprocessing exists."""

    return prepare_protected_input(value).text


def split_model_sentences(value: str) -> list[str]:
    """Compatibility helper: a route owns the complete submitted text."""

    return [unicodedata.normalize("NFC", value).strip()]


def enforce_output_fidelity(source: str, prediction: str) -> str:
    """Return raw model output; routing performs no output validation."""

    return enforce_numeric_fidelity(source, prediction).text


def enforce_output_fidelity_detailed(source: str, prediction: str) -> FidelityResult:
    """Return raw model output with an empty validation result."""

    return enforce_numeric_fidelity(source, prediction)
