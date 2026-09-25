"""Roman ordinal + noun rows with case/gender agreement (offline data generation only).

User report (2026-09-25): "Потрібно прочитати XX розділ" -> "двадцятого розділ"
(should be accusative = "двадцятий розділ"). E51 also fails "IX главу" and
"в III акті". Roman numerals stay within I..XXXIX (user rule: I, V, X only).
The noun sets the gender, the frame sets the case; num2words gives the ordinal,
pymorphy3 inflects both, and the value is cross-checked with the parser.
"""

from __future__ import annotations

import random
from functools import lru_cache

import pymorphy3
from num2words import num2words

from src.verbalizer._audit_numparse import parse_numbers

MORPH = pymorphy3.MorphAnalyzer(lang="uk")
NOUNS = {
    "masc": ["розділ", "том", "акт", "параграф", "пункт", "етап", "сезон", "конгрес", "з'їзд", "тур", "поверх", "клас", "курс", "ряд"],
    "femn": ["глава", "частина", "книга", "серія", "симфонія", "олімпіада", "сесія", "редакція", "зміна", "лекція"],
    "neut": ["століття", "видання", "скликання", "тисячоліття", "засідання"],
}


def roman(n: int) -> str:
    out = ""
    for value, mark in ((10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")):
        while n >= value:
            out += mark
            n -= value
    return out


@lru_cache(maxsize=None)
def noun_form(noun: str, case: str, gender: str) -> str | None:
    """Singular inanimate form of `noun` in `case` (the gender picks the right homograph)."""

    if case == "nomn":
        return noun
    parse = next((p for p in MORPH.parse(noun) if "NOUN" in p.tag and gender in p.tag
                  and "plur" not in p.tag and (gender == "femn" or "anim" not in p.tag)), None)
    form = parse.inflect({case}) if parse else None
    return form.word if form and "plur" not in form.tag else None


@lru_cache(maxsize=None)
def ordinal_form(n: int, gender: str, case: str) -> str | None:
    """Ordinal agreeing with an inanimate singular noun (accusative masc/neut = nominative)."""

    words = num2words(n, lang="uk", to="ordinal").split()
    head, last = words[:-1], words[-1]
    parse = next((p for p in MORPH.parse(last) if "ADJF" in p.tag), None)
    if not parse:
        return None
    grammemes = {gender, case}
    if case == "accs" and gender in ("masc", "neut"):
        grammemes = {gender, "nomn"}
    candidates = [f.word for f in parse.lexeme if grammemes <= set(f.tag.grammemes)]
    if case == "loct":  # prefer "-ому"/"-ій" over the rarer "-ім"
        candidates = sorted(candidates, key=lambda w: not w.endswith(("ому", "ій")))
    if not candidates:
        return None
    text = " ".join(head + [candidates[0]])
    return text if parse_numbers(text) == [n] else None


def make_rows(rng: random.Random, frames: dict[str, list[str]], n: int) -> list[dict]:
    """Rows with source, target, and the expected ordinal (for exact checking)."""

    rows, attempts = [], 0
    while len(rows) < n and attempts < n * 50:
        attempts += 1
        case = rng.choice(list(frames))
        gender = rng.choice(list(NOUNS))
        noun = rng.choice(NOUNS[gender])
        value = rng.randint(1, 39)
        noun_text = noun_form(noun, "nomn" if case == "accs" and gender != "femn" else case, gender)
        ordinal = ordinal_form(value, gender, case)
        if not noun_text or not ordinal:
            continue
        frame = rng.choice(frames[case])
        source = frame.format(r=roman(value), n=noun_text)
        target = frame.format(r=ordinal, n=noun_text)
        if source.startswith(roman(value)):
            target = target[0].upper() + target[1:]
        rows.append({"source": source, "target": target, "case": case, "gender": gender, "value": value,
                     "ordinal": ordinal, "frame": frame, "noun": noun_text})
    return rows


TRAIN_FRAMES = {
    "nomn": ["{r} {n} починається з цитати.", "{r} {n} почнеться завтра.", "{r} {n} закінчується несподівано."],
    "accs": ["Прочитай {r} {n}.", "Ми почали {r} {n} учора.", "Вивчи {r} {n} до п'ятниці.", "Треба переглянути {r} {n} ще раз."],
    "gent": ["До {r} {n} ми не дійшли.", "Зміст {r} {n} короткий.", "Після {r} {n} буде перерва."],
    "datv": ["Завдяки {r} {n} все стало ясно.", "Присвятили увагу {r} {n}."],
    "ablt": ["Перед {r} {n} є примітка.", "Разом із {r} {n} ми отримали додаток."],
    "loct": ["У {r} {n} описано війну.", "Про це йдеться в {r} {n}.", "На {r} {n} ми зупинимося."],
}
EVAL_FRAMES = {
    "nomn": ["{r} {n} стартує в понеділок."],
    "accs": ["Потрібно прочитати {r} {n} до понеділка.", "Прочитайте {r} {n}.", "Відкрий {r} {n}."],
    "gent": ["До {r} {n} ще далеко.", "Автор {r} {n} невідомий."],
    "datv": ["Згідно з планом, увагу приділили {r} {n}."],
    "ablt": ["За {r} {n} іде епілог."],
    "loct": ["Сцена відбувається в {r} {n}.", "У {r} {n} все змінюється."],
}

# E53: many governors per case so the case must be read from the noun ending,
# not memorised from one preposition. Held-out EVAL_FRAMES are not repeated.
_TAILS = ["", " минулого року", " без пояснень", " ще раз", " у підсумку"]
RICH_FRAMES = {
    "nomn": ["{r} {n} починається з цитати.", "{r} {n} почнеться завтра.", "Нарешті вийшов друком {r} {n}."],
    "accs": ["Прочитай {r} {n}.", "Ми почали {r} {n} учора.", "Вивчи {r} {n} до п'ятниці.", "Покажіть {r} {n}.",
             "Він обрав {r} {n}{t}.", "Перенесли на {r} {n}.", "Подивіться {r} {n}{t}.", "Запам'ятай {r} {n}.",
             "Через {r} {n} все змінилося.", "Вони пропустили {r} {n}{t}."],
    "gent": ["До {r} {n} ми не дійшли.", "Після {r} {n} буде перерва.", "Без {r} {n} нічого не зрозуміло.",
             "Зміст {r} {n} короткий.", "Початок {r} {n} затягнувся.", "Кінець {r} {n} вражає.", "Учасники {r} {n} зібралися{t}.",
             "Підсумки {r} {n} оприлюднили.", "Замість {r} {n} показали інше.", "Біля {r} {n} стояла черга.",
             "Для {r} {n} потрібен час.", "Протягом {r} {n} нічого не сталося.", "Від {r} {n} залежить усе."],
    "datv": ["Завдяки {r} {n} все стало ясно.", "Особливу увагу приділили {r} {n}{t}.", "Подяку висловили {r} {n}.",
             "Всупереч {r} {n} ми продовжили.", "Нагороду вручили {r} {n}{t}.", "Ми раді {r} {n}.", "Назустріч {r} {n} вийшли гості.",
             "Кінець покладено {r} {n}.", "Критики дорікали {r} {n}{t}.", "Лист адресовано {r} {n}."],
    "ablt": ["Перед {r} {n} є примітка.", "Разом із {r} {n} ми отримали додаток.", "Між {r} {n} і вступом є пауза.",
             "Над {r} {n} працювали роками.", "Під {r} {n} стоїть підпис.", "Ми захоплюємося {r} {n}{t}.",
             "Слідом за {r} {n} вийшла збірка.", "Поруч із {r} {n} лежала записка.", "Вони пишаються {r} {n}.",
             "Разом з {r} {n} прийшли зміни."],
    "loct": ["У {r} {n} описано війну.", "Про це йдеться в {r} {n}.", "На {r} {n} ми зупинимося.",
             "При {r} {n} все було інакше.", "У {r} {n} є помилка{t}."],
}


def rich_frames() -> dict[str, list[str]]:
    return {case: [f.replace("{t}", t) for f in frames for t in (_TAILS if "{t}" in f else [""])]
            for case, frames in RICH_FRAMES.items()}

# E54: E53 over-applied dative/locative to accusatives ("прочитати XX розділ" ->
# "двадцятому"). Masculine/neuter inanimate accusatives look nominative, so the
# verb is the only cue: many verb structures here; plus possessive genitives.
ACCS_FRAMES = ["Треба прочитати {r} {n} до вечора.", "Слід переглянути {r} {n}.", "Мені потрібно дочитати {r} {n}.",
               "Не забудь повторити {r} {n}.", "Учні мають вивчити {r} {n} напам'ять.", "Хочу перечитати {r} {n}.",
               "Просимо здати {r} {n} вчасно.", "Ми відклали {r} {n} на потім.", "Вона швидко закінчила {r} {n}.",
               "Варто уважно прочитати {r} {n}.", "Завтра розберемо {r} {n}.", "Я вже переглянув {r} {n}.",
               "Необхідно підготувати {r} {n} до зустрічі.", "Потрібно обговорити {r} {n} з колегами."]
GENT_POSSESSIVE = ["Редактор {r} {n} подав у відставку.", "Організатори {r} {n} задоволені.", "Доля {r} {n} невідома.",
                   "Тема {r} {n} складна.", "Назва {r} {n} змінилася.", "Сторінки {r} {n} пожовкли.",
                   "Текст {r} {n} загубився.", "Героїня {r} {n} живе в місті.", "Мета {r} {n} очевидна.",
                   "Обсяг {r} {n} невеликий."]


def e54_frames() -> dict[str, list[str]]:
    frames = rich_frames()
    frames["accs"] = frames["accs"] + ACCS_FRAMES
    frames["gent"] = frames["gent"] + GENT_POSSESSIVE
    return frames
