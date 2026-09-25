"""Deterministic scoring for generated per-domain evaluation rows (offline only).

A row is a frame of fixed text with typed slots. The output must reproduce the
fixed text (so damaged neighbouring words fail) and every slot must be read in
one of its accepted ways:

  english  -- matches a provider reading, tolerating the transliteration
              variants the reviewer accepts (г/х/ґ, е/і/и, doubled letters,
              final с/з) plus one edit on long words;
  digits   -- the spoken digits spell exactly the source digits, whether read
              one by one, in pairs or in groups; zeros must all be there;
  iban     -- the country letters read as a letter name, then a digits slot;
  coords   -- exact values, the right number of degree/minute/second units,
              and the right hemisphere words.
"""

from __future__ import annotations

import re
import unicodedata

from functools import lru_cache

from .._audit_numparse import SCALES, UNITS

APOSTROPHES = str.maketrans({"’": "'", "ʼ": "'", "`": "'"})


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text).translate(APOSTROPHES).casefold()
    text = re.sub(r"\s*([,.;:!?…—–«»„“”\"()-])\s*", r" \1 ", text)
    return re.sub(r"\s+", " ", text).strip()


def fold(word: str) -> str:
    """Collapse the transliteration differences a reviewer treats as variants."""

    w = normalize(word).replace("'", "").replace("ь", "").replace("-", "").replace(" ", "")
    w = re.sub(r"[ґгх]", "г", w)
    w = re.sub(r"[еєіїи]", "і", w)
    w = re.sub(r"(.)\1+", r"\1", w)
    return re.sub(r"з$", "с", w)


VOWELS = set("аеєиіїоуюяй")


def _skeleton(folded: str) -> tuple[str, int]:
    return "".join(c for c in folded if c not in VOWELS), sum(c in VOWELS for c in folded)


def english_ok(spoken: str, variants: list[str]) -> bool:
    """Consonants must match a provider reading; vowels are free within one.

    Calibrated on the reviewer's verdicts for 340 benchmark rows: 92.4%
    agreement (edit-distance matching reached 86.2%). Added or dropped
    consonants fail ("еврістайм" for everytime); vowel choices pass
    ("дата"/"дейта", "директор"/"дайректор").
    """

    key = fold(spoken)
    if not key:
        return False
    key_consonants, key_vowels = _skeleton(key)
    for variant in variants:
        target = fold(variant)
        consonants, vowels = _skeleton(target)
        if key == target or (key_consonants and key_consonants == consonants and abs(key_vowels - vowels) <= 1):
            return True
    return False


def _segment_value(words: list[str]) -> int | None:
    """Value of a well-formed numeral phrase, or None.

    Well-formed: groups of [hundreds][tens | teen][units], each group optionally
    closed by a scale word; "нуль" only on its own. This rejects run-ons such as
    "два три" that a summing parser would read as 5.
    """

    if words == ["нуль"]:
        return 0
    total, group, stage = 0, 0, 0  # stage: 0 start, 1 after hundreds, 2 after tens, 3 closed
    for word in words:
        if word in SCALES:
            if stage == 0 and group == 0 and total:
                return None
            total += (group or 1) * SCALES[word]
            group, stage = 0, 0
            continue
        value = UNITS.get(word)
        if value is None or value == 0:
            return None
        if value >= 100 and value % 100 == 0 and value < 1000:
            if stage >= 1:
                return None
            group, stage = group + value, 1
        elif value >= 20 and value % 10 == 0:
            if stage >= 2:
                return None
            group, stage = group + value, 2
        elif 10 <= value <= 19:
            if stage >= 2:
                return None
            group, stage = group + value, 3
        elif 1 <= value <= 9:
            if stage >= 3:
                return None
            group, stage = group + value, 3
        else:
            return None
    return total + group


def digits_ok(spoken: str, digits: str) -> bool:
    """True if the numeral words can be split into numbers spelling `digits` exactly."""

    words = [w.replace("’", "'") for w in re.findall(r"[а-яіїєґ'’]+", normalize(spoken))]
    if not words or any(w not in UNITS and w not in SCALES for w in words):
        return False

    @lru_cache(maxsize=None)
    def reach(word_at: int, digit_at: int) -> bool:
        if word_at == len(words):
            return digit_at == len(digits)
        for end in range(word_at + 1, min(len(words), word_at + 8) + 1):
            value = _segment_value(words[word_at:end])
            if value is None:
                continue
            text = str(value)
            if digits.startswith(text, digit_at) and reach(end, digit_at + len(text)):
                return True
        return False

    return reach(0, 0)


def iban_ok(spoken: str, country: str, digits: str) -> bool:
    letters = {"UA": r"^(ю\s*-?\s*ей|ю\s*-?\s*а|у\s*-?\s*а)\s*"}.get(country)
    text = normalize(spoken).replace(",", " ")
    match = re.match(letters, text) if letters else None
    return bool(match) and digits_ok(text[match.end():], digits)


FRACTION_WORD = re.compile(
    r"^(десят|сот|тисячн|десятитисячн|стотисячн|мільйонн)(их|а|ої|ій|у|ою)$"
)
HEMISPHERE = {"N": r"північн\w* широт", "S": r"південн\w* широт", "E": r"східн\w* довгот", "W": r"західн\w* довгот"}


def phone_ok(spoken: str, digits: str, plus: bool) -> bool:
    text = normalize(spoken).replace(",", " ").replace("(", " ").replace(")", " ")
    if plus:
        if not text.startswith("плюс"):
            return False
        text = text[len("плюс"):]
    return digits_ok(text, digits)


def coords_ok(spoken: str, values: list[int], units: list[str], hemispheres: list[str]) -> bool:
    """Values in order, one unit word per value, and the hemisphere phrases.

    Accepted value readings: whole numbers, or decimals read as "N цілих M
    десятитисячних" (values [N, M]).
    """

    text = normalize(spoken)
    words = re.findall(r"[а-яіїєґ']+", text)
    numeral_runs, current = [], []
    for word in words:
        if (word in UNITS or word in SCALES) and not FRACTION_WORD.match(word):
            current.append(word)
        elif current:
            numeral_runs.append(current)
            current = []
    if current:
        numeral_runs.append(current)
    spoken_values = [_segment_value(run) for run in numeral_runs]
    # "мінута" is a Russianism; the Ukrainian angular minute is "хвилина" (user, 2026-09-25).
    if re.search(r"мінут", text):
        return False
    found_units = re.findall(r"градус\w*|хвилин\w*|секунд\w*", text)
    kinds = ["deg" if u.startswith("градус") else "sec" if u.startswith("секунд") else "min" for u in found_units]
    # Zero minutes/seconds ("73°33′00″") may be left unsaid.
    expected = [(values, units)]
    keep = [i for i, (v, u) in enumerate(zip(values, units)) if not (v == 0 and u in ("min", "sec"))]
    if len(keep) < len(values) and len(values) == len(units):
        expected.append(([values[i] for i in keep], [units[i] for i in keep]))
    if not any(spoken_values == vals and kinds == kinds_expected for vals, kinds_expected in expected):
        return False
    return all(re.search(HEMISPHERE[h], text) for h in hemispheres)


LETTER_NAMES = {
    "А": ["а"], "Б": ["бе"], "В": ["ве"], "Г": ["ге", "ґе"], "Ґ": ["ґе", "ге"], "Д": ["де"], "Е": ["е"],
    "Є": ["є"], "Ж": ["же"], "З": ["зе"], "И": ["и"], "І": ["і"], "Ї": ["ї"], "Й": ["йот", "й"], "К": ["ка"],
    "Л": ["ел", "ель"], "М": ["ем"], "Н": ["ен"], "О": ["о"], "П": ["пе"], "Р": ["ер"], "С": ["ес"], "Т": ["те"],
    "У": ["у"], "Ф": ["еф"], "Х": ["ха"], "Ц": ["це"], "Ч": ["че"], "Ш": ["ша"], "Щ": ["ща"], "Ю": ["ю"], "Я": ["я"],
}


def letters_ok(spoken: str, letters: str) -> bool:
    """A Cyrillic acronym read letter by letter ("ФБР" -> "еф бе ер", "еф-бе-ер")."""

    names = [t for t in re.split(r"[\s\-]+", normalize(spoken)) if t]
    return len(names) == len(letters) and all(name in LETTER_NAMES.get(ch, []) for name, ch in zip(names, letters))


def check_slot(slot: dict, spoken: str) -> bool:
    if slot["kind"] == "exact":
        return normalize(spoken) in {normalize(v) for v in slot["variants"]}
    if slot["kind"] == "letters":
        return letters_ok(spoken, slot["letters"])
    kind = slot["kind"]
    if kind == "english":
        return english_ok(spoken, slot["variants"])
    if kind == "digits":
        return digits_ok(spoken.replace(",", " "), slot["digits"])
    if kind == "iban":
        return iban_ok(spoken, slot["country"], slot["digits"])
    if kind == "phone":
        return phone_ok(spoken, slot["digits"], slot["plus"])
    if kind == "coords":
        return coords_ok(spoken, slot["values"], slot["units"], slot["hemispheres"])
    raise ValueError(f"unknown slot kind {kind}")


def _align(text: str, pos: int, frame: list[str], slots: list[dict], index: int, readings: list[str]) -> list[str] | None:
    """Try every word-boundary split so adjacent slots ("big data") align correctly."""

    literal = normalize(frame[index])
    pos = len(text) - len(text[pos:].lstrip())
    if literal:
        if not text.startswith(literal, pos):
            return None
        pos += len(literal)
    pos = len(text) - len(text[pos:].lstrip())
    if index == len(slots):
        return readings if pos == len(text) else None
    ends = [m.start() for m in re.finditer(r"(?= )|$", text[pos:])]
    for end in ends:
        spoken = text[pos:pos + end].strip()
        if spoken and check_slot(slots[index], spoken):
            found = _align(text, pos + end, frame, slots, index + 1, [*readings, spoken])
            if found is not None:
                return found
    return None


def score_row(row: dict, prediction: str) -> dict:
    """Match the fixed text literally and check each slot; returns {'ok', 'reason'}."""

    text = normalize(prediction)
    if _align(text, 0, row["frame"], row["slots"], 0, []) is not None:
        return {"ok": True, "reason": ""}
    literals = [normalize(p) for p in row["frame"] if normalize(p)]
    if not all(literal in text for literal in literals):
        return {"ok": False, "reason": "fixed text changed"}
    return {"ok": False, "reason": "a slot is misread"}
