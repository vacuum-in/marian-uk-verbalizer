#!/usr/bin/env python3
"""E66: span rows for business and technical text, where E60-E65 fail (offline data generation only).

A user's test page of business text (eval-data/business_v1.jsonl) showed the span model, trained on book text, reading
"1 001" as "один мільйон один", the version "4.12.3" as "четвертого грудня", "+12,5 °C" with "мінус", "€875,20" with
"пенсів", "0,0035" as "нуль цілих нуль сотих", "00:00" as "нульовій", "01.01.2000" as "...двадцятого року", and
"08:30–17:45" with "мінус"; it never saw "÷" or "пн–пт" (the detector now marks both). Each kind below writes random
values into short business sentences of its own (none taken from the test page) and reads them by rule.

E67 added amounts in a row ("€875,20 та £99,95"), exchange rates ("41,35 грн за $1 та ..."), ratios and simple
fractions ("3:2", "1/4"), and document numbers with a year ("№ 15/2026"). E68 (after a second page, business_v2):
dates after "з/до/на", times after "на/о/до", contract numbers with a date, units (мс, А, В, м³, мм рт. ст., ...), IP
addresses, versions in a list, numbers like "101 101" and "10,01", one weekday with hours, and number case after
prepositions ("у 25 коробках", "на 1,8%"). E69: grouped whole part with a fraction and unit ("1 047,6 км"), durations
("3 год 45 хв 20 с"), temperature ranges, three-part ratios, "N Гбіт/с", numbers with leading zeros.

    build_e66_business.py --out data/e66_spans/business.parquet [--per-kind 4000] [--kinds a,b]
"""
from __future__ import annotations

import argparse
import random
import re
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from num2words import num2words

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.roman_case import ordinal_form  # noqa: E402
from src.verbalizer.spans import detect_spans, model_input  # noqa: E402

MONTHS = ["січня", "лютого", "березня", "квітня", "травня", "червня", "липня", "серпня", "вересня", "жовтня",
          "листопада", "грудня"]
GEN = {"один": "одного", "одна": "однієї", "два": "двох", "дві": "двох", "три": "трьох", "чотири": "чотирьох",
       "сорок": "сорока", "дев'яносто": "дев'яноста", "сто": "ста", "двісті": "двохсот", "триста": "трьохсот",
       "чотириста": "чотирьохсот", "п'ятсот": "п'ятисот", "шістсот": "шестисот", "сімсот": "семисот",
       "вісімсот": "восьмисот", "дев'ятсот": "дев'ятисот", "тисяча": "тисячі", "тисячі": "тисяч", "тисяч": "тисяч",
       "нуль": "нуля", "шість": "шести", "сім": "семи", "вісім": "восьми", "ціла": "цілої", "цілих": "цілих"}
FRAC = ["десят", "сот", "тисячн", "десятитисячн", "стотисячн", "мільйонн", "десятимільйонн", "стомільйонн",
        "мільярдн"]


def card(n: int, fem: bool = False) -> str:
    w = re.sub(r"^одна тисяча", "тисяча", num2words(n, lang="uk"))
    if fem:
        w = re.sub(r"(^|\s)один$", r"\1одна", w)
        w = re.sub(r"(^|\s)два$", r"\1дві", w)
    return w


def gen_word(w: str) -> str:
    if w in GEN:
        return GEN[w]
    if w.endswith("ять") or w.endswith("сять"):
        return w[:-1] + "и"
    if w.endswith("десят"):
        return w + "и"
    if w.endswith(("а", "ої")) and w[-4:] in ("сята", "сота"):
        return w[:-1] + "ої"
    return w


def genitive(phrase: str) -> str:
    return " ".join(gen_word(w) for w in phrase.split())


def plural(n: int, one: str, few: str, many: str) -> str:
    if n % 10 == 1 and n % 100 != 11:
        return one
    if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
        return few
    return many


def decimal(intpart: int, digits: str) -> str:
    """"12", "5" -> "дванадцять цілих п'ять десятих"; the numerator is feminine."""
    k, num = len(digits), int(digits)
    whole = card(intpart, fem=True) + " " + plural(intpart, "ціла", "цілих", "цілих")
    tail = FRAC[k - 1] + ("а" if num % 10 == 1 and num % 100 != 11 else "их")
    return f"{whole} {card(num, fem=True)} {tail}"


def group(n: int) -> str:
    return f"{n:,}".replace(",", " ")


def ord_gen(n: int) -> str:
    if n == 2000:
        return "двохтисячного"
    return re.sub(r"^одна тисяча", "тисяча", ordinal_form(n, "masc", "gent"))


def ord_fem(n: int, case: str) -> str:
    if n == 0:
        return {"nomn": "нульова", "gent": "нульової", "loct": "нульовій"}[case]
    return ordinal_form(n, "femn", case)


def minutes(m: int) -> str:
    return "" if m == 0 else ("нуль " + card(m, fem=True) if m < 10 else card(m, fem=True))


def row(line: str, piece: str, reading: str, kind: str) -> dict | None:
    start = line.index(piece)
    end = start + len(piece)
    for sp in detect_spans(line):
        if sp.start <= start and end <= sp.start + len(sp.text):
            a, b = start - sp.start, end - sp.start
            rest = sp.text[:a] + sp.text[b:]
            if re.search(r"\d", rest):
                return None
            target = re.sub(r"\s+", " ", sp.text[:a] + reading + sp.text[b:]).strip()
            return {"id": f"e66-{kind}-{abs(hash(line)) % 10**12}", "source": model_input(line, sp), "target": target,
                    "span": sp.text, "sentence_id": kind}
    return None


def row_multi(line: str, pieces: list[tuple[str, str]], kind: str) -> dict | None:
    """The span holding the most pieces; each piece inside it is replaced by its reading, nothing else may be a digit."""
    best = None
    for sp in detect_spans(line):
        end, text, inside, pos = sp.start + len(sp.text), sp.text, 0, 0
        for piece, reading in pieces:
            i = line.find(piece, pos)
            pos = i + len(piece)
            if sp.start <= i and i + len(piece) <= end:
                j = text.find(piece)
                text = text[:j] + reading + text[j + len(piece):]
                inside += 1
        if inside and not re.search(r"\d", text) and (not best or inside > best[0]):
            best = (inside, sp, re.sub(r"\s+", " ", text).strip())
    if not best:
        return None
    _, sp, target = best
    return {"id": f"e66-{kind}-{abs(hash(line)) % 10**12}", "source": model_input(line, sp), "target": target,
            "span": sp.text, "sentence_id": kind}


def k_grouped(rng):
    n = rng.choice([rng.randint(1000, 99999), rng.randint(100000, 999999), rng.randint(1000000, 99999999),
                    rng.choice([1000, 1001, 10010, 100000, 1000001, 10000, 2000, 1010, 100100])])
    piece = group(n)
    if rng.random() < 0.5:
        others = "; ".join(str(rng.randint(0, 999)) for _ in range(rng.randint(0, 3)))
        line = rng.choice(["Перевірте числа: {}.", "Значення: {}.", "Список: {}.", "Контрольні суми: {}."]).format(
            "; ".join(x for x in [others, piece] if x))
        return row(line, piece, card(n), "grouped")
    noun = rng.choice([("особа", "особи", "осіб"), ("примірник", "примірники", "примірників"),
                       ("користувач", "користувачі", "користувачів"), ("запис", "записи", "записів")])
    line = rng.choice(["За місяць зареєстровано {} {}.", "На складі залишилося {} {}.", "Тираж становить {} {}."])
    line = line.format(piece, plural(n, *noun))
    return row(line, piece, card(n, fem=noun[0] == "особа"), "grouped")


def k_version(rng):
    parts = [rng.randint(0, 20), rng.randint(0, 30)] + ([rng.randint(0, 40)] if rng.random() < 0.6 else [])
    v = ".".join(map(str, parts))
    reading = " крапка ".join(card(p) for p in parts)
    line = rng.choice(["Встановіть версію {} або новішу.", "Оновлення {} вийде наступного тижня.",
                       "У збірці {} виправлено помилки.", "Поточна версія застосунку — {}.",
                       "Реліз {} містить нові функції.", "Сервер працює на версії {}, а клієнт — на старішій."]).format(v)
    return row(line, v, reading, "version")


def k_temp(rng):
    sign = rng.choice(["+", "−", "-", "+"])
    whole = rng.randint(1, 40)
    frac = str(rng.randint(1, 9)) if rng.random() < 0.4 else ""
    num = f"{sign}{whole}" + (f",{frac}" if frac else "")
    unit = rng.choice([" °C", "°C", " °С"])
    piece = num + unit
    word = "плюс" if sign == "+" else "мінус"
    value = decimal(whole, frac) if frac else card(whole)
    if rng.random() < 0.5:
        deg = "градуса" if frac else plural(whole, "градус", "градуси", "градусів")
        reading = f"{word} {value} {deg} Цельсія"
        line = rng.choice(["Вночі було {}.", "Температура повітря становить {}.", "Зараз на вулиці {}.",
                           "У прогнозі на завтра {}."]).format(piece)
    else:
        deg = "градуса" if frac or whole % 10 == 1 and whole % 100 != 11 else "градусів"
        reading = f"{word} {genitive(value)} {deg} Цельсія"
        line = rng.choice(["Удень потеплішає до {}.", "Вода нагрілася до {}.", "Температура опуститься до {}."]).format(piece)
    return row(line, piece, reading, "temp")


def k_small_decimal(rng):
    k = rng.randint(3, 6)
    num = rng.randint(1, 10 ** (k - 1) - 1) if rng.random() < 0.7 else rng.randint(1, 10 ** k - 1)
    digits = str(num).zfill(k)
    if digits.endswith("0"):
        return None
    sign = rng.choice(["", "", "±"])
    unit, gen = rng.choice([("мм", "міліметра"), ("г", "грама"), ("мг", "міліграма"), ("с", "секунди"),
                            ("м", "метра"), ("", "")])
    piece = f"{sign}0,{digits}" + (f" {unit}" if unit else "")
    reading = ("плюс-мінус " if sign else "") + decimal(0, digits) + (f" {gen}" if gen else "")
    line = rng.choice(["Похибка становить {}.", "Отримане значення — {}.", "Допуск: {}.", "Зміна склала {}."]).format(piece)
    return row(line, piece, reading, "small-decimal")


CURR = {"$": ("долар", "долари", "доларів", "цент", "центи", "центів"),
        "€": ("євро", "євро", "євро", "цент", "центи", "центів"),
        "£": ("фунт", "фунти", "фунтів", "пенс", "пенси", "пенсів")}


def money(sym: str, whole: int, cents: int | None) -> tuple[str, str]:
    c = CURR[sym]
    text = f"{sym}{group(whole)}" + (f",{cents:02d}" if cents is not None else "")
    reading = f"{card(whole)} {plural(whole, *c[:3])}"
    if cents:
        reading += f" {card(cents)} {plural(cents, *c[3:])}"
    return text, reading


def k_currency(rng):
    sym = rng.choice(list(CURR))
    whole = rng.choice([rng.randint(1, 99), rng.randint(100, 9999), rng.randint(10000, 999999)])
    cents = rng.choice([None, rng.randint(1, 99), rng.randint(10, 99)])
    piece, reading = money(sym, whole, cents)
    if rng.random() < 0.3:
        rate = f"{rng.randint(20, 60)},{rng.randint(10, 99)}"
        r_whole, r_kop = map(int, rate.split(","))
        r_read = f"{card(r_whole, fem=True)} {plural(r_whole, 'гривня', 'гривні', 'гривень')} " \
                 f"{card(r_kop, fem=True)} {plural(r_kop, 'копійка', 'копійки', 'копійок')}"
        line = f"Банк продає {sym}1 за {rate} грн."
        unit = {"$": "один долар", "€": "один євро", "£": "один фунт"}[sym]
        return row(line, f"{sym}1", unit, "currency") if rng.random() < 0.5 else row(line, f"{rate} грн", r_read + "", "currency")
    line = rng.choice(["На картці залишилося {}.", "Ціна послуги — {}.", "Переказ на суму {} надійшов.",
                       "Баланс рахунку: {}."]).format(piece)
    return row(line, piece, reading, "currency")


def k_midnight_time(rng):
    h = rng.choice([0, 0, 0, rng.randint(1, 23)])
    m = rng.choice([0, rng.randint(1, 9), rng.randint(10, 59)])
    piece = f"{h:02d}:{m:02d}"
    if h == 0:
        reading = "нуль годин" + ("" if m == 0 else " " + card(m, fem=True) + " " + plural(m, "хвилина", "хвилини", "хвилин"))
    else:
        reading = ord_fem(h, "nomn") + ("" if m == 0 else " " + minutes(m))
    line = rng.choice(["Відмітка часу: {}.", "Час запису: {}.",
                       "Початок трансляції: {}.", "Система показує {}."]).format(piece)
    return row(line, piece, reading, "time")


def k_time_range(rng):
    h1 = rng.randint(6, 13)
    m1 = rng.choice([0, 0, 15, 30, 45, rng.randint(1, 59)])
    h2 = rng.randint(h1 + 1, 22)
    m2 = rng.choice([0, 0, 15, 30, 45, rng.randint(1, 59)])
    dash = rng.choice(["–", "—", "-", " – "])
    piece = f"{h1:02d}:{m1:02d}{dash}{h2:02d}:{m2:02d}"
    a = ord_fem(h1, "gent") + (" " + minutes(m1) if m1 else "")
    b = ord_fem(h2, "gent") + (" " + minutes(m2) if m2 else "")
    line = rng.choice(["Прийом громадян: {}.", "Ми працюємо {}, без вихідних.", "Обідня перерва — {}.",
                       "Години роботи каси: {}.", "Доставка можлива {}."]).format(piece)
    return row(line, piece, f"з {a} до {b}", "time-range")


DAYS = {"пн": ("понеділка", "понеділок"), "вт": ("вівторка", "вівторок"), "ср": ("середи", "середу"),
        "чт": ("четверга", "четвер"), "пт": ("п'ятниці", "п'ятницю"), "сб": ("суботи", "суботу"),
        "нд": ("неділі", "неділю")}


def k_weekdays(rng):
    keys = list(DAYS)
    i = rng.randint(0, 4)
    j = rng.randint(i + 1, 6)
    piece = f"{keys[i]}{rng.choice(['–', '—'])}{keys[j]}"
    line = rng.choice(["Графік: {}, з дев'ятої до шостої.", "Відділення відкрите {}.", "Консультації — {}.",
                       "Офіс працює {}, у вихідні зачинено."]).format(piece)
    return row(line, piece, f"з {DAYS[keys[i]][0]} по {DAYS[keys[j]][1]}", "weekdays")


def k_dash_words(rng):
    """Two two-letter words joined by a dash are not always weekdays: they stay as written."""
    a, b = rng.choice([("ну", "це"), ("ой", "що"), ("та", "ні"), ("як", "це"), ("он", "де"), ("ну", "то")])
    piece = f"{a}{rng.choice(['—', '–'])}{b}"
    line = rng.choice(["Він протягнув: «{}, побачимо».", "А вона йому: {}, не знаю.", "— Та {}, — махнув він рукою."]).format(piece)
    return row(line, piece, piece, "dash-words")


def k_arith(rng):
    op, word = rng.choice([("÷", "поділити на"), (":", "поділити на"), ("×", "помножити на"), ("+", "плюс"),
                           ("−", "мінус")])
    a = rng.choice([rng.randint(2, 99), rng.randint(100, 999), rng.randint(1000, 99999)])
    b = rng.randint(2, 99)
    c = {"÷": a // b, ":": a // b, "×": a * b, "+": a + b, "−": a - b}[op]
    if op in "÷:":
        a = c * b
    piece = f"{group(a)} {op} {b} = {group(c)}"
    line = rng.choice(["Перевірка: {}.", "Обчислення: {}.", "У задачі вийшло {}."]).format(piece)
    return row(line, piece, f"{card(a)} {word} {card(b)} дорівнює {card(c)}", "arith")


def k_dims(rng):
    nums = [rng.randint(2, 200) for _ in range(rng.choice([2, 3]))]
    unit, gen = rng.choice([("см", ("сантиметр", "сантиметри", "сантиметрів")), ("мм", ("міліметр", "міліметри", "міліметрів")),
                            ("м", ("метр", "метри", "метрів"))])
    sep = rng.choice([" × ", "×", " x "])
    piece = sep.join(map(str, nums)) + f" {unit}"
    reading = " на ".join(card(n) for n in nums) + " " + plural(nums[-1], *gen)
    line = rng.choice(["Габарити коробки: {}.", "Розмір плитки — {}.", "Панель має розміри {}."]).format(piece)
    return row(line, piece, reading, "dims")


def k_date(rng):
    y = rng.choice([2000, rng.randint(2000, 2009), rng.randint(2000, 2009), rng.randint(1990, 2039)])
    mth = rng.randint(1, 12)
    d = rng.randint(1, 28)
    piece = f"{d:02d}.{mth:02d}.{y}"
    reading = f"{ordinal_form(d, 'masc', 'gent')} {MONTHS[mth - 1]} {ord_gen(y)} року"
    line = rng.choice(["Договір укладено {}.", "Дата видачі: {}.", "Термін дії — до {}.", "Платіж надійшов {}.",
                       "Контрольні дати: {}; {}.".format("{}", f"{rng.randint(1, 28):02d}.{rng.randint(1, 12):02d}.2024")])
    line = line.format(piece)
    return row(line, piece, reading, "date")


def k_money_list(rng):
    """Two or three amounts in a row: the detector often makes "€875,20 та £99,95" one span."""
    syms = rng.sample(list(CURR), rng.choice([2, 3]))
    parts = [money(sm, rng.choice([rng.randint(1, 99), rng.randint(100, 9999)]), rng.choice([None, rng.randint(1, 99)]))
             for sm in syms]
    conj = rng.choice(["та", "і", "й"])
    text = ", ".join(p[0] for p in parts[:-1]) + f" {conj} " + parts[-1][0]
    reading = ", ".join(p[1] for p in parts[:-1]) + f" {conj} " + parts[-1][1]
    line = rng.choice(["У гаманці лежало {}.", "Залишок на рахунках: {}.", "Ми отримали {}."]).format(text)
    return row(line, text, reading, "money-list")


def k_rate(rng):
    pieces, texts = [], []
    for sm in rng.sample(list(CURR), 2):
        w, k = rng.randint(20, 60), rng.randint(1, 99)
        uah = f"{w},{k:02d} грн"
        pieces += [(uah, f"{card(w, fem=True)} {plural(w, 'гривня', 'гривні', 'гривень')} {card(k, fem=True)} "
                         f"{plural(k, 'копійка', 'копійки', 'копійок')}"),
                   (f"{sm}1", dict(zip(CURR, ["один долар", "один євро", "один фунт"]))[sm])]
        texts.append(f"{uah} за {sm}1")
    line = rng.choice(["Курс на сьогодні: {}.", "Обмінник пропонує {}.", "Офіційний курс — {}."]).format(" та ".join(texts))
    return row_multi(line, pieces, "rate")


def k_ratio(rng):
    if rng.random() < 0.5:
        a, b = rng.randint(1, 9), rng.randint(1, 9)
        piece = f"{a}:{b}"
        reading = f"{card(a)} до {genitive(card(b))}"
        line = rng.choice(["Пропорція суміші — {}.", "Співвідношення сторін {}.", "Змішайте у співвідношенні {}."]).format(piece)
        return row(line, piece, reading, "ratio")
    b = rng.choice([2, 3, 4, 5, 8, 10, 100])
    a = rng.randint(1, b - 1)
    den = {2: "друг", 3: "трет", 4: "четверт", 5: "п'ят", 8: "восьм", 10: "десят", 100: "сот"}[b]
    reading = f"{card(a, fem=True)} {den}{'а' if a == 1 else 'і' if 2 <= a <= 4 else 'их'}"
    piece = f"{a}/{b}"
    line = rng.choice(["Імовірність виграшу — {}.", "Заповнено {} баку.", "Частка ринку становить {}."]).format(piece)
    return row(line, piece, reading, "ratio")


def k_doc_number(rng):
    n, y = rng.randint(1, 999), rng.randint(1995, 2030)
    sep = rng.choice(["/", "/", "-"])
    piece = f"№ {n}{sep}{y}"
    reading = f"номер {card(n)} {'слеш' if sep == '/' else 'дефіс'} {card(y)}"
    line = rng.choice(["Наказ {} підписано.", "Рішення {} оскаржено.", "Протокол {} додається."]).format(piece)
    return row(line, piece, reading, "doc-number")



# ---- E68: after E67's errors on a second business page (business_v2) ----

def acc_phrase(words: str) -> str:
    """Accusative of a feminine count phrase: "одна ціла вісім десятих" -> "одну цілу вісім десятих"."""
    words = re.sub(r"\bодна ціла\b", "одну цілу", words)
    return re.sub(r"\bодна (\w+)а$", r"одну \1у", words)


def date_words(d: int, mth: int, y: int, case: str = "gent") -> str:
    day = ordinal_form(d, "masc", "gent") if case == "gent" else ordinal_form(d, "neut", "nomn")
    return f"{day} {MONTHS[mth - 1]} {ord_gen(y)} року"


def rand_date(rng):
    return rng.randint(1, 28), rng.randint(1, 12), rng.choice([rng.randint(1990, 2039), rng.randint(2020, 2030)])


def k_date_prep(rng):
    d1, d2 = rand_date(rng), rand_date(rng)
    t1, t2 = "{:02d}.{:02d}.{}".format(*d1), "{:02d}.{:02d}.{}".format(*d2)
    if rng.random() < 0.6:
        line = rng.choice(["Звітний період: з {} до {}.", "Договір діє з {} до {}.", "Акція триває з {} по {}.",
                           "Відпустка з {} до {} включно."]).format(t1, t2)
        return row_multi(line, [(t1, date_words(*d1)), (t2, date_words(*d2))], "date-prep")
    line = rng.choice(["Засідання перенесли на {}.", "Перевірку призначено на {}.", "Квитки продають на {}."]).format(t1)
    return row(line, t1, date_words(*d1, case="acc"), "date-prep")


def k_time_prep(rng):
    h, m = rng.randint(1, 23), rng.choice([0, rng.randint(1, 9), rng.randint(10, 59)])
    piece = f"{h:02d}:{m:02d}"
    prep, case = rng.choice([("на", "accs"), ("о", "loct"), ("до", "gent"), ("після", "gent"), ("з", "gent")])
    hour = ordinal_form(h, "femn", case)
    if not hour:
        return None
    line = rng.choice(["Зустріч призначено {} {}.", "Виїзд {} {}, не спізнюйтеся.", "Магазин працює {} {}."]).format(prep, piece)
    return row(line, piece, hour + (" " + minutes(m) if m else ""), "time-prep")


def k_contract(rng):
    n, mth, y = rng.randint(1, 99), rng.randint(1, 12), rng.randint(2015, 2030)
    num = f"№ {n}/{mth:02d}-{y}"
    num_r = f"номер {card(n)} слеш {'нуль ' + card(mth) if mth < 10 else card(mth)} дефіс {card(y)}"
    d = rand_date(rng)
    dt = "{:02d}.{:02d}.{}".format(*d)
    line = rng.choice(["Договір {} підписано {}.", "Угоду {} укладено {}.", "Рахунок {} виставлено {}."]).format(num, dt)
    return row_multi(line, [(num, num_r), (dt, date_words(*d))], "contract")


UNITS = [("мс", "мілісекунда", "мілісекунди", "мілісекунд", "мілісекунди", True),
         ("А", "ампер", "ампери", "амперів", "ампера", False), ("мА", "міліампер", "міліампери", "міліамперів", "міліампера", False),
         ("В", "вольт", "вольти", "вольтів", "вольта", False), ("кВт", "кіловат", "кіловати", "кіловатів", "кіловата", False),
         ("Вт", "ват", "вати", "ватів", "вата", False), ("Гц", "герц", "герци", "герців", "герца", False),
         ("кПа", "кілопаскаль", "кілопаскалі", "кілопаскалів", "кілопаскаля", False),
         ("м³", "кубічний метр", "кубічні метри", "кубічних метрів", "кубічного метра", False),
         ("см³", "кубічний сантиметр", "кубічні сантиметри", "кубічних сантиметрів", "кубічного сантиметра", False),
         ("мм рт. ст.", "міліметр ртутного стовпчика", "міліметри ртутного стовпчика", "міліметрів ртутного стовпчика",
          "міліметра ртутного стовпчика", False),
         ("мкм", "мікрометр", "мікрометри", "мікрометрів", "мікрометра", False),
         ("т", "тонна", "тонни", "тонн", "тонни", True)]


def k_unit(rng):
    sym, one, few, many, gsg, fem = rng.choice(UNITS)
    if rng.random() < 0.5:
        n = rng.choice([rng.randint(1, 30), rng.randint(31, 999)])
        val, reading = str(n), f"{card(n, fem=fem)} {plural(n, one, few, many)}"
    else:
        w, f = rng.randint(0, 99), str(rng.randint(1, 99))
        if f.endswith("0"):
            return None
        val, reading = f"{w},{f}", f"{decimal(w, f)} {gsg}"
    piece = f"{val} {sym}"
    line = rng.choice(["Показник становить {}.", "Виміряне значення — {}.", "Номінал: {}.", "За паспортом — {}, не більше."])
    return row(line.format(piece), piece, reading, "unit")


def k_ip(rng):
    parts = [rng.choice([10, 172, 192, rng.randint(1, 255)]), rng.randint(0, 255), rng.randint(0, 255), rng.randint(1, 254)]
    ip = ".".join(map(str, parts))
    line = rng.choice(["IP-адреса шлюзу: {}.", "Сервер доступний за адресою {}.", "Пристрій отримав адресу {}."]).format(ip)
    return row(line, ip, " крапка ".join(card(p) for p in parts), "ip")


def k_version_list(rng):
    vs, reads = [], []
    for _ in range(rng.randint(2, 5)):
        r = rng.random()
        if r < 0.25:
            parts = [str(rng.randint(2019, 2030)), f"{rng.randint(1, 12):02d}", str(rng.randint(1, 300))]
        else:
            parts = [str(rng.randint(0, 15)), str(rng.randint(0, 30))] + ([str(rng.randint(0, 40))] if r < 0.8 else [])
        vs.append(".".join(parts))
        reads.append(" крапка ".join(("нуль " + card(int(p)) if len(p) == 2 and p[0] == "0" else card(int(p))) for p in parts))
    k = rng.randrange(len(vs))
    line = rng.choice(["Сумісні збірки: {}.", "Підтримувані релізи: {}.", "Тестували на: {}."]).format("; ".join(vs))
    return row(line, vs[k], reads[k], "version-list") if line.count(vs[k]) == 1 else None


def k_grouped_more(rng):
    a = rng.randint(100, 999)
    n = rng.choice([a * 1001, a * 1000 + rng.randint(0, 9), rng.randint(100, 999) * 1000 + rng.randint(100, 999),
                    rng.randint(1, 9) * 1000000 + rng.randint(0, 999) * 1000 + rng.randint(0, 9)])
    nums = [group(rng.randint(1, 999999)) for _ in range(rng.randint(1, 3))] + [group(n)]
    rng.shuffle(nums)
    line = "Ряд значень: " + "; ".join(nums) + "."
    return row(line, group(n), card(n), "grouped") if line.count(group(n)) == 1 else None


def k_mixed_decimal(rng):
    w = rng.choice([rng.randint(0, 9), rng.randint(10, 99), rng.randint(100, 999)])
    k = rng.randint(2, 4)
    f = str(rng.randint(1, 9)).zfill(k) if rng.random() < 0.6 else str(rng.randint(1, 10 ** k - 1)).zfill(k)
    if f.endswith("0"):
        return None
    piece = f"{w},{f}"
    others = "; ".join(f"{rng.randint(0, 99)},{rng.randint(1, 9)}" for _ in range(rng.randint(0, 3)))
    line = "Контрольні значення: " + "; ".join(x for x in [others, piece] if x) + "."
    return row(line, piece, decimal(w, f), "mixed-decimal") if line.count(piece) == 1 else None


def k_weekday_schedule(rng):
    keys = list(DAYS)
    single = rng.random() < 0.5
    i = rng.randint(0, 5 if single else 4)
    j = i if single else rng.randint(i + 1, 6)
    wd = keys[i] if single else f"{keys[i]}–{keys[j]}"
    wd_r = {"пн": "понеділок", "вт": "вівторок", "ср": "середа", "чт": "четвер", "пт": "п'ятниця", "сб": "субота",
            "нд": "неділя"}[keys[i]] if single else f"з {DAYS[keys[i]][0]} по {DAYS[keys[j]][1]}"
    h1, h2 = rng.randint(7, 11), rng.randint(13, 20)
    m1, m2 = rng.choice([0, 15, 30, 45]), rng.choice([0, 15, 30, 45])
    tr = f"{h1:02d}:{m1:02d}–{h2:02d}:{m2:02d}"
    tr_r = f"з {ord_fem(h1, 'gent')}{' ' + minutes(m1) if m1 else ''} до {ord_fem(h2, 'gent')}{' ' + minutes(m2) if m2 else ''}"
    line = rng.choice(["Розклад: {} — {}.", "Прийом: {} — {}; в інші дні зачинено.", "Режим роботи: {} — {}."]).format(wd, tr)
    return row_multi(line, [(wd, wd_r), (tr, tr_r)], "weekday-schedule")


NOUNS_LOC = [("коробках", False), ("містах", False), ("кімнатах", False), ("будинках", False), ("областях", False),
             ("школах", False), ("країнах", False), ("випадках", False)]


def k_prep_case(rng):
    if rng.random() < 0.5:
        n = rng.randint(2, 999)
        if n % 10 == 1 and n % 100 != 11:
            return None
        noun, _ = rng.choice(NOUNS_LOC)
        piece = f"{n} {noun}"
        line = rng.choice(["Товар розміщено у {}.", "Опитування провели в {}.", "Це траплялося у {}."]).format(piece)
        loc = " ".join({"двісті": "двохстах", "триста": "трьохстах", "чотириста": "чотирьохстах", "п'ятсот": "п'ятистах",
                        "шістсот": "шестистах", "сімсот": "семистах", "вісімсот": "восьмистах",
                        "дев'ятсот": "дев'ятистах"}.get(w, gen_word(w)) for w in card(n).split())
        return row(line, piece, f"{loc} {noun}", "prep-case")
    verb = rng.choice(["зріс на", "зменшився на", "впав на", "подорожчав на", "скоротився на"])
    if rng.random() < 0.6:
        w, f = rng.randint(0, 30), str(rng.randint(1, 99))
        if f.endswith("0"):
            return None
        piece, reading = f"{w},{f}%", f"{acc_phrase(decimal(w, f))} відсотка"
    else:
        n = rng.randint(1, 99)
        piece, reading = f"{n}%", f"{card(n)} {plural(n, 'відсоток', 'відсотки', 'відсотків')}"
    line = f"За рік показник {verb} {piece}."
    return row(line, piece, reading, "prep-case")



# ---- E69: what E68 lost or never had ----

MEASURES = [("км", "кілометр", "кілометри", "кілометрів", "кілометра"), ("м²", "квадратний метр", "квадратні метри",
            "квадратних метрів", "квадратного метра"), ("кг", "кілограм", "кілограми", "кілограмів", "кілограма"),
            ("л", "літр", "літри", "літрів", "літра"), ("га", "гектар", "гектари", "гектарів", "гектара"),
            ("т", "тонна", "тонни", "тонн", "тонни"), ("м", "метр", "метри", "метрів", "метра")]


def k_grouped_decimal(rng):
    """"1 047,6 км", "12 345,75 м²": a space-grouped whole part with a fraction and a unit."""
    sym, one, few, many, gsg = rng.choice(MEASURES)
    w = rng.choice([rng.randint(1000, 9999), rng.randint(10000, 999999), rng.randint(100, 999)])
    f = str(rng.randint(1, 99))
    if f.endswith("0"):
        return None
    piece = f"{group(w)},{f} {sym}"
    line = rng.choice(["Загальна довжина — {}.", "Площа ділянки становить {}.", "Обсяг постачання: {}.",
                       "За рік пройдено {}."]).format(piece)
    return row(line, piece, f"{decimal(w, f)} {gsg}", "grouped-decimal")


def k_duration(rng):
    h, m, sec = rng.randint(1, 72), rng.randint(1, 59), rng.randint(1, 59)
    parts = [(f"{h} год", f"{card(h, fem=True)} {plural(h, 'година', 'години', 'годин')}"),
             (f"{m} хв", f"{card(m, fem=True)} {plural(m, 'хвилина', 'хвилини', 'хвилин')}"),
             (f"{sec} с", f"{card(sec, fem=True)} {plural(sec, 'секунда', 'секунди', 'секунд')}")]
    parts = parts[: rng.choice([2, 3])] if rng.random() < 0.7 else parts[1:]
    piece = " ".join(p[0] for p in parts)
    line = rng.choice(["Тривалість: {}.", "Процес зайняв {}.", "Час у дорозі — {}.", "Рекорд траси: {}."]).format(piece)
    return row(line, piece, " ".join(p[1] for p in parts), "duration")


def k_temp_range(rng):
    def t(v, frac):
        sign = "+" if v > 0 and rng.random() < 0.7 else ("−" if v < 0 else "")
        txt = f"{sign}{abs(v)}" + (f",{frac}" if frac else "")
        word = ("плюс " if sign == "+" else "мінус " if sign == "−" else "")
        val = decimal(abs(v), frac) if frac else card(abs(v))
        g = genitive(val)
        deg = "градуса" if frac or abs(v) % 10 == 1 and abs(v) % 100 != 11 else "градусів"
        return txt, f"{word}{g} {deg} Цельсія"
    a = rng.randint(-30, 10)
    b = a + rng.randint(3, 25)
    if a == 0 or b == 0:
        return None
    fa = str(rng.randint(1, 9)) if rng.random() < 0.3 else ""
    fb = str(rng.randint(1, 9)) if rng.random() < 0.3 else ""
    (ta, ra), (tb, rb) = t(a, fa), t(b, fb)
    pa, pb = f"{ta} °C", f"{tb} °C"
    line = rng.choice(["Температура коливалася від {} до {}.", "Прогноз: від {} до {}.", "Вночі від {} до {}."]).format(pa, pb)
    # the two temperatures are separate spans: teach either one ("до +3 °C" -> "до плюс трьох градусів Цельсія")
    return row_multi(line, [(pa, ra), (pb, rb)] if rng.random() < 0.5 else [(pb, rb)], "temp-range")


def k_multi_ratio(rng):
    nums = [rng.randint(1, 20) for _ in range(rng.choice([3, 3, 4]))]
    piece = ":".join(map(str, nums))
    reading = card(nums[0]) + "".join(f" до {genitive(card(n))}" for n in nums[1:])
    line = rng.choice(["Співвідношення компонентів — {}.", "Пропорція суміші: {}.", "Змішати у пропорції {}."]).format(piece)
    return row(line, piece, reading, "multi-ratio")


def k_rate_unit(rng):
    n = rng.choice([1, rng.randint(2, 999)])
    sym, one, few, many = rng.choice([("Гбіт/с", "гігабіт", "гігабіти", "гігабітів"), ("Мбіт/с", "мегабіт", "мегабіти", "мегабітів"),
                                      ("МБ/с", "мегабайт", "мегабайти", "мегабайтів"), ("км/год", "кілометр", "кілометри", "кілометрів")])
    per = "на годину" if sym == "км/год" else "на секунду"
    piece = f"{n} {sym}"
    line = rng.choice(["Швидкість: {} на прийом.", "Канал дає {} у піку.", "Ліміт — {} для всіх."]).format(piece)
    return row(line, piece, f"{card(n)} {plural(n, one, few, many)} {per}", "rate-unit")


def k_leading_zeros(rng):
    k = rng.randint(1, 5)
    rest = str(rng.randint(1, 9999))
    code = "0" * k + rest
    piece = f"№ {code}"
    reading = "номер " + " ".join(["нуль"] * k) + " " + " ".join(card(int(c)) for c in rest)
    line = rng.choice(["Замовлення {} відправлено.", "Квитанція {} сплачена.", "Посилка {} прибула."]).format(piece)
    return row(line, piece, reading, "leading-zeros")



# ---- E70: abbreviations, times with seconds, dates without a year, long decimals, case after з/зі,
#      ordinals with suffixes, digit strings with leading zeros ----

ROMAN = {1: "I", 2: "II", 3: "III", 4: "IV", 5: "V", 6: "VI", 7: "VII", 8: "VIII", 9: "IX", 10: "X", 11: "XI", 12: "XII"}


def k_quarter(rng):
    q, y = rng.randint(1, 4), rng.randint(2015, 2030)
    piece = f"{ROMAN[q]} кв. {y} р."
    reading = f"{ordinal_form(q, 'masc', 'nomn')} квартал {ord_gen(y)} року"
    line = rng.choice(["Звіт за {} готовий.", "Показники за {} гірші.", "Порівнюємо з даними за {}"]).format(piece)
    return row(line, piece, reading, "abbrev")


def k_pp(rng):
    w, f = rng.randint(0, 9), str(rng.randint(1, 9))
    if rng.random() < 0.5:
        piece, reading = f"{w},{f} в. п.", f"{decimal(w, f)} відсоткового пункту"
    else:
        n = rng.randint(1, 30)
        piece, reading = f"{n} в. п.", f"{card(n)} {plural(n, 'відсотковий пункт', 'відсоткові пункти', 'відсоткових пунктів')}"
    line = rng.choice(["Ставка — базова + {}.", "Різниця становить {}.", "Рейтинг зріс на {}."]).format(piece)
    if "+" in line:
        return row(line, "+ " + piece, "плюс " + reading, "abbrev")
    return row(line, piece, reading, "abbrev")


LEGAL = [("ст.", "стаття", "статті", "fem"), ("ч.", "частина", "частини", "fem"), ("п.", "пункт", "пункту", "masc"),
         ("пп.", "підпункт", "підпункту", "masc"), ("абз.", "абзац", "абзацу", "masc"), ("гл.", "глава", "глави", "fem"),
         ("розд.", "розділ", "розділу", "masc")]


def k_legal(rng):
    parts = rng.sample(LEGAL, rng.randint(1, 3))
    gen = rng.random() < 0.5
    pieces = []
    for ab, nom, gn, _ in parts:
        roman = ab == "розд." and rng.random() < 0.6
        n = rng.randint(1, 12) if roman else rng.randint(1, 40)
        txt = f"{ab} {ROMAN[n] if roman else n}"
        num = ordinal_form(n, "masc", "gent" if gen else "nomn") if roman else card(n)
        pieces.append((txt, f"{gn if gen else nom} {num}"))
    body = ", ".join(p[0] for p in pieces)
    line = ("Відповідно до " if gen else rng.choice(["Підстава: ", "Див. ", "Посилання: "])) + body + "."
    return row_multi(line, [rng.choice(pieces)], "abbrev")  # each piece is its own span: teach any of them


def k_address(rng):
    b, k, o = rng.randint(1, 150), rng.randint(1, 9), rng.randint(1, 500)
    lit = rng.choice(["", "-А", "-Б", "-В"])
    o_txt = str(o).zfill(rng.choice([1, 3]))
    pieces = [(f"буд. {b}{lit}", f"будинок {card(b)}{' ' + lit[1:].lower() if lit else ''}"),
              (f"корп. {k}", f"корпус {card(k)}"),
              (f"оф. {o_txt}" if rng.random() < 0.5 else f"кв. {o_txt}", None)]
    is_of = pieces[2][0].startswith("оф.")
    digits = " ".join(["нуль"] * (len(o_txt) - len(str(o)))) + (" " if len(o_txt) > len(str(o)) else "") + card(o)
    pieces[2] = (pieces[2][0], f"{'офіс' if is_of else 'квартира'} {digits}")
    line = "Адреса: вул. Садова, " + ", ".join(p[0] for p in pieces) + "."
    return row_multi(line, pieces, "abbrev")


def k_ext(rng):
    n = rng.randint(1, 999)
    return row(f"Телефон приймальні, доб. {n}.", f"доб. {n}", f"додатковий {card(n)}", "abbrev")


def k_big_abbr(rng):
    kind = rng.choice(["тис.", "млн", "млрд"])
    forms = {"тис.": ("тисяча", "тисячі", "тисяч", "тисячі"), "млн": ("мільйон", "мільйони", "мільйонів", "мільйона"),
             "млрд": ("мільярд", "мільярди", "мільярдів", "мільярда")}[kind]
    if rng.random() < 0.5:
        n = rng.randint(1, 999)
        val, reading = str(n), f"{card(n, fem=kind == 'тис.')} {plural(n, *forms[:3])}"
    else:
        w, f = rng.randint(1, 99), str(rng.randint(1, 9))
        val, reading = f"{w},{f}", f"{decimal(w, f)} {forms[3]}"
    cur = rng.random() < 0.5
    piece = f"{val} {kind}" + (" грн" if cur else "")
    reading += " гривень" if cur else ""
    line = rng.choice(["У плані — {} витрат.", "Оборот сягнув {}.", "Заплановано {} заявок."]).format(piece)
    return row(line, piece, reading, "abbrev")


def k_etc(rng):
    piece, reading = rng.choice([("т. ін.", "таке інше"), ("т. д.", "так далі"), ("та ін.", "та інші"), ("і т. п.", "і тому подібне")])
    line = rng.choice(["Довідник містить коди, назви, адреси {}.", "Ми закупили папір, ручки, теки {}."]).format(piece)
    return row(line, piece.split(" ", 1)[1] if piece.startswith(("та ", "і ")) else piece,
               reading.split(" ", 1)[1] if piece.startswith(("та ", "і ")) else reading, "abbrev")


def k_days_duration(rng):
    d, h, m, sec = rng.randint(1, 30), rng.randint(0, 23), rng.randint(0, 59), rng.randint(1, 59)
    piece = f"{d} д {h:02d} год {m:02d} хв {sec:02d} с"
    reading = (f"{card(d)} {plural(d, 'день', 'дні', 'днів')} {card(h, fem=True)} {plural(h, 'година', 'години', 'годин')} "
               f"{card(m, fem=True)} {plural(m, 'хвилина', 'хвилини', 'хвилин')} {card(sec, fem=True)} "
               f"{plural(sec, 'секунда', 'секунди', 'секунд')}")
    return row(f"Тривалість сеансу — {piece}.", piece, reading, "duration")


def k_time_seconds(rng):
    h, m, sec = rng.randint(0, 23), rng.randint(0, 59), rng.randint(0, 59)
    piece = f"{h:02d}:{m:02d}:{sec:02d}"
    reading = (f"{card(h, fem=True)} {plural(h, 'година', 'години', 'годин')} {card(m, fem=True)} "
               f"{plural(m, 'хвилина', 'хвилини', 'хвилин')} {card(sec, fem=True)} {plural(sec, 'секунда', 'секунди', 'секунд')}")
    line = rng.choice(["Мітка часу: {}.", "Запис створено о {}.", "Останній вхід — {}."]).format(piece)
    return row(line, piece, reading, "time-seconds")


def k_date_noyear(rng):
    days = sorted(rng.sample(range(1, 29), rng.randint(2, 3)))
    mth = rng.randint(1, 12)
    pieces, parts = [], []
    for d in days:
        txt = f"{d:02d}.{mth:02d}"
        rd = f"{ordinal_form(d, 'neut', 'nomn')} {MONTHS[mth - 1]}"
        if rng.random() < 0.5:
            h1, h2 = rng.randint(8, 11), rng.randint(13, 18)
            tr = f"{h1:02d}:00–{h2:02d}:00"
            pieces += [(txt, rd), (tr, f"з {ord_fem(h1, 'gent')} до {ord_fem(h2, 'gent')}")]
            parts.append(f"{txt} — {tr}")
        else:
            pieces.append((txt, rd))
            parts.append(f"{txt} — вихідний")
    line = "Графік у святковий період: " + "; ".join(parts) + "."
    k = rng.randrange(len(pieces))
    return row_multi(line, pieces, "date-noyear")


def k_long_decimal(rng):
    k = rng.randint(5, 9)
    w = rng.choice([0, 0, rng.randint(1, 9), rng.randint(10, 99), rng.randint(1000, 99999999)])
    f = str(rng.randint(1, 9 if rng.random() < 0.5 else 10 ** k - 1)).zfill(k)
    if f.endswith("0"):
        return None
    piece = f"{group(w)},{f}"
    line = rng.choice(["Мінімальне значення становило {}.", "Коефіцієнт — {}.", "Отримали {} у підсумку."]).format(piece)
    return row(line, piece, decimal(w, f), "long-decimal")


def k_from_to_pct(rng):
    a, b = rng.randint(1, 100), rng.randint(1, 100)
    def g(n):
        return f"{genitive(card(n))} {'відсотка' if n % 10 == 1 and n % 100 != 11 else 'відсотків'}"
    prep = "зі" if a in (100,) or str(a)[0] in "7" else rng.choice(["з", "від"])
    line = f"Заряд змінився {prep} {a}% до {b}%."
    return row_multi(line, [(f"{a}%", g(a)), (f"{b}%", g(b))], "prep-case")


def k_ordinal_suffix(rng):
    sufs = [("й", "masc", "nomn"), ("га", "femn", "nomn"), ("тє", "neut", "nomn"), ("те", "neut", "nomn"), ("ті", "plur", "nomn"),
            ("го", "masc", "gent"), ("му", "masc", "loct")]
    items = []
    for _ in range(rng.randint(1, 4)):
        n = rng.choice([rng.randint(1, 30), rng.randint(31, 200), rng.choice([1001, 1000, 2001])])
        suf, gnd, case = rng.choice(sufs)
        w = ordinal_form(n, gnd, case) if gnd != "plur" else ordinal_form(n, "plur", "nomn")
        if not w:
            return None
        items.append((f"{group(n)}-{suf}", re.sub(r"^одна тисяча", "тисяча", w)))
    line = "Позначення: " + ", ".join(i[0] for i in items) + "."
    k = rng.randrange(len(items))
    return row(line, items[k][0], items[k][1], "ordinal-suffix") if line.count(items[k][0]) == 1 else None


def k_zero_digits(rng):
    k = rng.randint(4, 8)
    code = "".join(rng.choice("0001") for _ in range(k)) if rng.random() < 0.6 else str(rng.randint(0, 10 ** k - 1)).zfill(k)
    if code[0] != "0":
        code = "0" + code[1:]
    line = rng.choice(["Контрольна послідовність: {}.", "Код: {}; повторіть його.", "Введіть {} у поле."]).format(code)
    return row(line, code, " ".join(card(int(c)) for c in code), "zero-digits")


def k_no_zero(rng):
    n = rng.randint(1, 99)
    piece = f"№ {n:02d}" if n < 10 else f"№ {n:03d}"
    zeros = len(piece) - 2 - len(str(n))
    reading = "номер " + " ".join(["нуль"] * zeros + [card(n)])
    line = rng.choice(["Сегмент {} зріс.", "Сервер {} перезавантажено.", "Вікно {} зачинене."]).format(piece)
    return row(line, piece, reading, "zero-digits")


KINDS = {"grouped": k_grouped, "version": k_version, "temp": k_temp, "small-decimal": k_small_decimal,
         "currency": k_currency, "time": k_midnight_time, "time-range": k_time_range, "weekdays": k_weekdays,
         "dash-words": k_dash_words, "arith": k_arith, "dims": k_dims, "date": k_date, "money-list": k_money_list,
         "rate": k_rate, "ratio": k_ratio, "doc-number": k_doc_number, "date-prep": k_date_prep,
         "time-prep": k_time_prep, "contract": k_contract, "unit": k_unit, "ip": k_ip, "version-list": k_version_list,
         "grouped-more": k_grouped_more, "mixed-decimal": k_mixed_decimal, "weekday-schedule": k_weekday_schedule,
         "prep-case": k_prep_case, "grouped-decimal": k_grouped_decimal, "duration": k_duration,
         "temp-range": k_temp_range, "multi-ratio": k_multi_ratio, "rate-unit": k_rate_unit,
         "leading-zeros": k_leading_zeros, "quarter": k_quarter, "pp": k_pp, "legal": k_legal, "address": k_address,
         "ext": k_ext, "big-abbr": k_big_abbr, "etc": k_etc, "days-duration": k_days_duration,
         "time-seconds": k_time_seconds, "date-noyear": k_date_noyear, "long-decimal": k_long_decimal,
         "from-to-pct": k_from_to_pct, "ordinal-suffix": k_ordinal_suffix, "zero-digits": k_zero_digits,
         "no-zero": k_no_zero}
SHARE = {"dash-words": 0.2, "weekdays": 0.5, "etc": 0.01}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--per-kind", type=int, default=4000)
    ap.add_argument("--show", type=int, default=3)
    ap.add_argument("--kinds", help="only these kinds (comma-separated)")
    args = ap.parse_args()
    rng = random.Random(66)
    rows, counts = [], {}
    for kind, make in KINDS.items():
        if args.kinds and kind not in args.kinds.split(","):
            continue
        want, got, seen, tries = int(args.per_kind * SHARE.get(kind, 1)), [], set(), 0
        while len(got) < want and tries < want * 20:
            tries += 1
            r = make(rng)
            if r and r["source"] not in seen:
                seen.add(r["source"])
                got.append(r)
        counts[kind] = len(got)
        rows += got
        for r in got[: args.show]:
            print(f"  [{kind}] {r['source'][:100]}\n      -> {r['target']!r}")
    rng.shuffle(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), args.out)
    print(len(rows), counts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
