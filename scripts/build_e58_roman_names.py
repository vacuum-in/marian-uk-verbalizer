#!/usr/bin/env python3
"""E58: a ruler's name followed by a Roman regnal number, in every case (offline data generation only).

E57 read "з Філіппом II" as "другому" and "Людовік XIII" as "тринадцяту": the Roman training rows put the noun after
the numeral ("XX розділ"), so a numeral agreeing with a name before it was rare. The name and the frame set the case
and gender; pymorphy3 inflects the name and the ordinal (animate: the masculine accusative is the genitive), and every
ordinal is cross-checked by parsing it back to its value.

    build_e58_roman_names.py --out data/e58_roman_names/pairs.parquet [--rows 9000]
"""
from __future__ import annotations

import argparse
import random
import sys
from functools import lru_cache
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pymorphy3
from num2words import num2words

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.roman_case import roman  # noqa: E402
from src.verbalizer._audit_numparse import parse_numbers  # noqa: E402

MORPH = pymorphy3.MorphAnalyzer(lang="uk")
NAMES = {
    "masc": ["Людовік", "Філіпп", "Генріх", "Карл", "Петро", "Іван", "Олександр", "Микола", "Казимир", "Болеслав",
             "Ярослав", "Едуард", "Річард", "Георг", "Вільгельм", "Фрідріх", "Яків", "Павло", "Лев", "Данило", "Пій",
             "Бенедикт", "Густав", "Сигізмунд", "Владислав", "Стефан", "Йосиф", "Рудольф", "Альфонс", "Фердинанд",
             "Леопольд", "Максиміліан", "Мехмед", "Сулейман", "Іоанн", "Інокентій", "Климент", "Гуго", "Оттон"],
    "femn": ["Єлизавета", "Катерина", "Марія", "Анна", "Вікторія", "Ізабелла", "Маргарита", "Христина", "Софія"],
}
FRAMES = {
    "nomn": ["{name} {r} підписав указ.", "Тоді країною правив {name} {r}.", "{name} {r} помер навесні.",
             "Того року на престол зійшов {name} {r}{t}."],
    "gent": ["Під час правління {name} {r} збудували собор.", "Указ {name} {r} скасували.", "Двір {name} {r} був розкішним.",
             "Портрет {name} {r} висить у залі.", "Після смерті {name} {r} почалася війна.", "Біля {name} {r} стояли радники."],
    "accs": ["Історики добре знають {name} {r}.", "Народ любив {name} {r}{t}.", "Змовники вбили {name} {r}.",
             "Посли бачили {name} {r} лише раз.", "Хроніст описав {name} {r} як жорстокого правителя."],
    "datv": ["Пам'ятник присвятили {name} {r}.", "Посол вручив листа {name} {r}.", "Це не сподобалося {name} {r}.",
             "Корону передали {name} {r}{t}."],
    "ablt": ["Мир уклали з {name} {r}.", "Він був знайомий з {name} {r}.", "Шлюб з {name} {r} викликав незадоволення.",
             "Країна розквітла за {name} {r}.", "Між {name} {r} і папою спалахнула суперечка."],
    "loct": ["При {name} {r} країна розквітла.", "Ідеться про подію при {name} {r}.", "У хроніці писали о {name} {r}."],
}
TAILS = ["", " після довгої хвороби", " у столиці", " без вагань"]


@lru_cache(maxsize=None)
def name_form(name: str, gender: str, case: str) -> str | None:
    if case == "nomn":
        return name
    parse = next((p for p in MORPH.parse(name) if "NOUN" in p.tag and gender in p.tag and "plur" not in p.tag), None)
    form = parse.inflect({case}) if parse else None
    return form.word.capitalize() if form else None


@lru_cache(maxsize=None)
def ordinal(n: int, gender: str, case: str) -> str | None:
    words = num2words(n, lang="uk", to="ordinal").split()
    parse = next((p for p in MORPH.parse(words[-1]) if "ADJF" in p.tag), None)
    if not parse:
        return None
    want = {gender, "gent" if case == "accs" and gender == "masc" else case}
    found = [f.word for f in parse.lexeme if want <= set(f.tag.grammemes)]
    if case == "loct":
        found.sort(key=lambda w: not w.endswith(("ому", "ій")))
    if not found:
        return None
    text = " ".join(words[:-1] + [found[0]])
    return text if parse_numbers(text) == [n] else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rows", type=int, default=9000)
    args = ap.parse_args()
    rng = random.Random(58)
    rows, seen = [], set()
    attempts = 0
    while len(rows) < args.rows and attempts < args.rows * 40:
        attempts += 1
        gender = "femn" if rng.random() < 0.2 else "masc"
        name = rng.choice(NAMES[gender])
        case = rng.choice(list(FRAMES))
        n = rng.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 21, 23])
        nf, od = name_form(name, gender, case), ordinal(n, gender, case)
        if not nf or not od:
            continue
        frame = rng.choice(FRAMES[case]).replace("{t}", rng.choice(TAILS))
        if gender == "femn":
            frame = frame.replace("підписав", "підписала").replace("помер", "померла").replace("зійшов", "зійшла").replace("правив", "правила")
        src = frame.format(name=nf, r=roman(n))
        if src in seen:
            continue
        seen.add(src)
        rows.append({"id": f"e58-roman-name-{len(rows)}", "source": src, "target": frame.format(name=nf, r=od),
                     "categories": [f"roman-name:{case}:{gender}"], "provenance": "generated_e58_roman_names"})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), args.out)
    print(len(rows), "rows")
    for r in rng.sample(rows, 8):
        print("  ", r["source"], "->", r["target"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
