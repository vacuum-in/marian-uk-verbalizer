"""Rewrite the few signs the span model cannot see, before span detection.

The Marian tokenizer has no piece for "½ ⅓ ⅔ ⅛", most superscript and subscript digits, "⁻ ⁺ ^", "√ ≤ ≥ ≠ ≈ ∞ ‰"
or the arc marks "′ ″": they all reach the model as the same <unk>, so it cannot tell "≤" from "≥". Only these signs
are rewritten here, into shapes the model reads; every number stays a number for the model to read
("33⅓%" -> "33 і 1/3%", "10⁻³" -> "10 у степені -3", "≤0,01%" -> "менше або дорівнює 0,01%"). General numbers, Roman
numerals, dates and units are never touched: the model reads them.
"""

from __future__ import annotations

import re

VULGAR = {"½": "1/2", "⅓": "1/3", "⅔": "2/3", "¼": "1/4", "¾": "3/4", "⅕": "1/5", "⅖": "2/5", "⅗": "3/5", "⅘": "4/5",
          "⅙": "1/6", "⅚": "5/6", "⅛": "1/8", "⅜": "3/8", "⅝": "5/8", "⅞": "7/8"}
SUPER = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺", "0123456789-+")
SUB = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
WORDS = {"√": "квадратний корінь з ", "≤": " менше або дорівнює ", "≥": " більше або дорівнює ", "≠": " не дорівнює ",
         "≈": " приблизно ", "∞": " нескінченність ", "‰": " проміле"}
CYR = "А-Яа-яІіЇїЄєҐґ"


def rewrite_signs(text: str) -> str:
    if not re.search(r"[½⅓⅔¼¾⅕-⅞⁰¹⁴-⁹⁻⁺₀-₉√≤≥≠≈∞‰′″^²³]", text):
        return text
    # a vulgar fraction after a whole number is "і": "33⅓" -> "33 і 1/3"
    text = re.sub(r"(\d)\s?([" + "".join(VULGAR) + r"])", lambda m: f"{m.group(1)} і {VULGAR[m.group(2)]}", text)
    text = re.sub("[" + "".join(VULGAR) + "]", lambda m: VULGAR[m.group()], text)
    # superscripts after a digit, a Latin letter or a bracket are a power; after a Cyrillic unit ("м²") they stay
    text = re.sub(r"(?<=[\dA-Za-z)])([⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺]+)", lambda m: " у степені " + m.group(1).translate(SUPER), text)
    text = re.sub(r"(?<![" + CYR + r"])([⁰¹⁴⁵⁶⁷⁸⁹⁻⁺]+)", lambda m: m.group(1).translate(SUPER), text)
    text = re.sub(r"\s*\^\s*", " у степені ", text)
    text = text.translate(SUB)
    # arc minutes and seconds after a number: "27′13,2″" -> "27 хв 13,2 с"; elsewhere they are quote marks
    text = re.sub(r"(?<=\d)′", " хв ", text)
    text = re.sub(r"(?<=\d)″", " с ", text)
    text = text.replace("′", "'").replace("″", '"')
    for sign, words in WORDS.items():
        text = text.replace(sign, words)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return re.sub(r"\s+([,.;:)])", r"\1", text).strip()
