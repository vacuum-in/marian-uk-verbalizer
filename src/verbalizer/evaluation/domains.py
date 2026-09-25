"""Evaluation domains for per-domain scoring (offline only; never used at runtime)."""

from __future__ import annotations

import re

CATEGORY_DOMAINS = {
    "dates": r"Дат|Дні тижня",
    "time": r"^Час|Голий фрагмент: час",
    "ranges": r"Діапазон",
    "roman": r"Римськ|Століття",
    "currency": r"Гроші|Валют|гроші",
    "english_words": r"Пакет V|англійськ|Транслітерац|transliteration|бренд|Бренд|Латиниця|латиниця",
    "math": r"Математичн|Арифметичн|Формул|Дроби|дроби|множення|Двокрапка",
    "coordinates": r"Координат|Градуси|Географічн",
    "numbers": r"Великі числа|Кількісні|Складені числівники|Пакет W|Голий фрагмент: число|Десяткові|Порядкові|Збірні|числівник",
    "phone": r"Телефон|телефон",
    "email_url": r"Email|email|URL|Веб-адреси|домен|пошта",
    "card_iban": r"Картк|IBAN|банківські",
    "symbols": r"Символ|Відсотк|відсотк|Спецсимвол|Спеціальні символи|хештег|Хештег",
    "abbreviations": r"Скорочення|Абревіатур|скорочення|Акронім|Ініціал|США",
}
DOMAINS = [*CATEGORY_DOMAINS, "chemistry"]

ELEMENTS = set("""
H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr
Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm
Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No
Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og
""".split())
WORD_COLLISIONS = {"In", "As", "No", "He", "Be", "At", "Am", "Co", "Ho", "Pa", "La", "Ne", "Re", "Os", "Ar", "Es", "Ga", "Ta", "Rh"}
SUBSCRIPTS = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
TOKEN_RE = re.compile(r"(?<![A-Za-z])[A-Z][A-Za-z0-9₀-₉()]*(?![A-Za-z])")
PART_RE = re.compile(r"([A-Z][a-z]?)(\d*)")


def _element_parts(token: str) -> list[str] | None:
    body = token.translate(SUBSCRIPTS).replace("(", "").replace(")", "")
    parts, position = [], 0
    for match in PART_RE.finditer(body):
        if match.start() != position or match.group(1) not in ELEMENTS:
            return None
        parts.append(match.group(1))
        position = match.end()
    return parts if position == len(body) and parts else None


def chemistry_tokens(source: str) -> list[str]:
    """Tokens that read as chemical symbols or formulas.

    Single-letter symbols (C, B, V, I, N, ...) collide with vitamins, Roman
    numerals and acronyms (II, BBC, CNN), so a token qualifies only with a
    two-letter symbol (Mg, NaCl) or a count (H₂O, CO₂). A lone two-letter
    token that is also a common English word (In, As, No) does not count.
    """

    found = []
    for token in TOKEN_RE.findall(source):
        parts = _element_parts(token)
        if not parts:
            continue
        counts = re.findall(r"[0-9₀-₉]+", token)
        if any(len(count) > 2 for count in counts):
            continue
        two_letter = any(len(p) == 2 for p in parts)
        if len(parts) == 1:
            if token in WORD_COLLISIONS:
                continue
            if not two_letter and not re.search(r"[₀-₉]", token):
                continue
        if counts or two_letter:
            found.append(token)
    return found


def domains_of(categories: list[str], source: str = "") -> set[str]:
    found = {d for d, pattern in CATEGORY_DOMAINS.items() if any(re.search(pattern, c) for c in categories)}
    if source and chemistry_tokens(source):
        found.add("chemistry")
    return found
