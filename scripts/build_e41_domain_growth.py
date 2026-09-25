#!/usr/bin/env python3
"""E41: training rows for the domains the held-out domain sets exposed.

E40 on reports/domain-eval: English words 75.6%, phone 74.4%, card/IBAN 28.8%,
coordinates 16.6%. The generators here emit one canonical reading per row
(num2words for numerals, the two-provider bronze readings for English) --
TRAINING DATA ONLY, never used at inference. Every target is scored with the
same checker the evaluation uses and dropped unless it passes, and every
source that appears in an evaluation set, the dev set or the frozen benchmark
is skipped. Frames, English words and the random seed are disjoint from
scripts/build_domain_eval.py.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from num2words import num2words

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from build_domain_eval import digits_string, english_is_holdout, make_row  # noqa: E402
from src.verbalizer.data.english_readings import load_readings  # noqa: E402
from src.verbalizer.evaluation.domain_check import score_row  # noqa: E402
from src.verbalizer.training.manifest import sha256_file, write_json  # noqa: E402

DIGIT = ["нуль", "один", "два", "три", "чотири", "п'ять", "шість", "сім", "вісім", "дев'ять"]

CARD_FRAMES = ["Номер моєї картки {0}.", "Переказуйте на {0}.", "Картка {0} прив'язана до рахунку.",
               "Оплата з картки {0} не пройшла.", "Карта клієнта: {0}", "{0}", "Дані картки: {0}, дякую."]
IBAN_FRAMES = ["Рахунок {0} відкрито в банку.", "Сплатіть на рахунок {0}.", "{0}",
               "Кошти зараховано на {0} сьогодні вранці.", "Повний номер рахунку: {0}."]
PHONE_FRAMES = ["Мій номер {0}.", "Подзвоніть мені на {0} після обіду.", "Контактний телефон: {0}.",
                "Служба підтримки працює за номером {0}.", "{0}", "Запишіть номер {0}, будь ласка."]
COORD_FRAMES = ["Координати місця: {0}.", "Група вийшла на точку {0}.", "Літак зник з радарів у районі {0}.",
                "{0}", "Об'єкт розташований на {0}."]
ENGLISH_SINGLE = ["Він написав {0} у чаті.", "Слово {0} часто трапляється в піснях.", "Бренд {0} відкрив новий офіс.",
                  "На вивісці було {0}.", "Сестра прошепотіла {0}.", "{0}"]
ENGLISH_PAIR = ["Назва альбому — {0} {1}.", "Вивіска «{0} {1}» світилася вночі.", "{0} {1}", "Гурт заспівав {0} {1}."]
ENGLISH_LIST = ["Теги: {list}.", "Серед слів були {list}.", "Вкладки браузера: {list}.", "Перелічую: {list}.",
                "У нотатках записано {list}."]
LAT = {"N": ("пн. ш.", "північної широти"), "S": ("пд. ш.", "південної широти")}
LON = {"E": ("сх. д.", "східної довготи"), "W": ("зх. д.", "західної довготи")}


def words(n: int, feminine: bool = False) -> str:
    text = num2words(n, lang="uk", gender="feminine") if feminine else num2words(n, lang="uk")
    return text.replace("одна тисяча", "тисяча", 1) if text.startswith("одна тисяча") else text


def plural(n: int, forms: tuple[str, str, str]) -> str:
    if n % 10 == 1 and n % 100 != 11:
        return forms[0]
    if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
        return forms[1]
    return forms[2]


def digitwise(text: str) -> str:
    return " ".join(DIGIT[int(c)] for c in text)


def group_reading(group: str) -> str:
    """Phone group: every leading zero is "нуль", the rest is read as a number."""

    zeros = len(group) - len(group.lstrip("0"))
    rest = group[zeros:]
    return " ".join(["нуль"] * zeros + ([words(int(rest))] if rest else []))


def card_pair(rng: random.Random) -> tuple[str, str, dict]:
    digits = rng.choice("2345") + digits_string(rng, 15, zero_run=0.3)
    groups = [digits[i:i + 4] for i in range(0, 16, 4)]
    text = rng.choice([" ".join(groups), "-".join(groups), digits, " ".join(groups)])
    return text, ", ".join(digitwise(g) for g in groups), {"kind": "digits", "digits": digits}


def iban_pair(rng: random.Random) -> tuple[str, str, dict]:
    digits = digits_string(rng, 27, zero_run=0.7)
    body = "UA" + digits
    groups = [body[i:i + 4] for i in range(0, len(body), 4)]
    text = " ".join(groups) if rng.random() < 0.6 else body
    reading = "ю ей " + ", ".join(digitwise(g.replace("UA", "")) for g in groups)
    return text, reading, {"kind": "iban", "country": "UA", "digits": digits}


def phone_pair(rng: random.Random) -> tuple[str, str, dict]:
    op = rng.choice(["50", "63", "66", "67", "68", "73", "93", "95", "96", "97", "98", "99", "44", "32", "56", "57", "62", "48", "31"])
    rest = digits_string(rng, 7, zero_run=0.15)
    a, b, c = rest[:3], rest[3:5], rest[5:]
    shape = rng.choice(["plus_spaced", "plus_brackets", "local_spaced", "local_dashes", "local_brackets",
                        "local_brackets", "free", "tollfree"])
    if shape == "tollfree":
        rest = digits_string(rng, 6)
        text, digits, plus = f"0 800 {rest[:2]} {rest[2:4]} {rest[4:]}", "0800" + rest, False
        reading = " ".join(["нуль", "вісімсот", group_reading(rest[:2]), group_reading(rest[2:4]), group_reading(rest[4:])])
    elif shape.startswith("plus"):
        text = f"+380 {op} {a} {b} {c}" if shape == "plus_spaced" else f"+380 ({op}) {a}-{b}-{c}"
        digits, plus = "380" + op + rest, True
        reading = " ".join(["плюс", digitwise("380"), group_reading(op), group_reading(a), group_reading(b), group_reading(c)])
    else:
        text = {"local_spaced": f"0{op} {a} {b} {c}", "local_dashes": f"0{op}-{a}-{b}-{c}",
                "local_brackets": f"(0{op}) {a}-{b}-{c}", "free": f"0{op}{rest}"}[shape]
        digits, plus = "0" + op + rest, False
        reading = " ".join([group_reading("0" + op), group_reading(a), group_reading(b), group_reading(c)])
    return text, reading, {"kind": "phone", "digits": digits, "plus": plus}


def angle_reading(value: int, unit: str) -> str:
    if unit == "deg":
        return f"{words(value)} {plural(value, ('градус', 'градуси', 'градусів'))}"
    forms = ("хвилина", "хвилини", "хвилин") if unit == "min" else ("секунда", "секунди", "секунд")
    return f"{words(value, feminine=True)} {plural(value, forms)}"


def coordinate_pair(rng: random.Random) -> tuple[str, str, dict]:
    style = rng.choice(["dms", "dms", "dm", "decimal"])
    both = rng.random() < 0.6
    texts, readings, values, units, hemis = [], [], [], [], []
    for table, hemi, max_deg in ((LAT, rng.choice("NNNS"), 89), (LON, rng.choice("EEEW"), 179)):
        if table is LON and not both:
            break
        mark, phrase = table[hemi]
        if style == "decimal":
            whole, places = rng.randint(0, max_deg), rng.choice([1, 2, 3, 4, 4, 4])
            frac = rng.randint(10 ** (places - 1), 10 ** places - 1)
            denominator = {1: ("десята", "десятих"), 2: ("сота", "сотих"), 3: ("тисячна", "тисячних"),
                           4: ("десятитисячна", "десятитисячних")}[places]
            texts.append(f"{whole},{frac}° {mark}")
            whole_word = "ціла" if whole % 10 == 1 and whole % 100 != 11 else "цілих"
            frac_word = denominator[0] if frac % 10 == 1 and frac % 100 != 11 else denominator[1]
            readings.append(f"{words(whole, True)} {whole_word} {words(frac, True)} {frac_word} градуса {phrase}")
            values += [whole, frac]
            units.append("deg")
        else:
            deg, minute, second = rng.randint(0, max_deg), rng.randint(0, 59), rng.randint(0, 59)
            ascii_marks = rng.random() < 0.25
            m, s = ("'", '"') if ascii_marks else ("′", "″")
            parts = [(deg, "deg"), (minute, "min")] + ([(second, "sec")] if style == "dms" else [])
            text = f"{deg}°{minute:02d}{m}" + (f"{second:02d}{s}" if style == "dms" else "")
            texts.append(f"{text} {mark}")
            readings.append(" ".join(angle_reading(v, u) for v, u in parts) + " " + phrase)
            values += [v for v, _ in parts]
            units += [u for _, u in parts]
        hemis.append(hemi)
    slot = {"kind": "coords", "values": values, "units": units, "hemispheres": hemis}
    return ", ".join(texts), ", ".join(readings), slot


def english_pool(readings: dict) -> dict[str, str]:
    """Training words: not held out, and the providers share at least one reading."""

    pool = {}
    for word, by_provider in readings.items():
        if not word.isalpha() or english_is_holdout(word) or len(by_provider) != 2:
            continue
        shared = set.intersection(*by_provider.values())
        if shared:
            pool[word] = sorted(shared)[0]
    return pool


def english_rows(rng: random.Random, pool: dict[str, str], readings: dict, n: int) -> list[tuple[str, str, dict]]:
    vocab = sorted(pool)
    out = []
    while len(out) < n:
        shape = rng.choice(["single", "pair", "list", "list", "list"])
        count = {"single": 1, "pair": 2, "list": rng.randint(3, 9)}[shape]
        chosen = rng.sample(vocab, count)
        shown = [w.capitalize() if rng.random() < 0.35 else w for w in chosen]
        slots = [{"kind": "english", "variants": sorted(set().union(*readings[w].values()))} for w in chosen]
        if shape == "list":
            listed = ", ".join("{%d}" % i for i in range(count))
            if rng.random() < 0.3:
                listed = ", ".join("{%d}" % i for i in range(count - 1)) + " і {%d}" % (count - 1)
            template = rng.choice(ENGLISH_LIST).replace("{list}", listed)
        else:
            template = rng.choice(ENGLISH_SINGLE if shape == "single" else ENGLISH_PAIR)
        out.append((template, shown, [pool[w] for w in chosen], slots))
    return out


def read_rows(path: Path) -> list[dict[str, Any]]:
    table = pq.read_table(path, columns=["id", "source", "target", "categories", "provenance"])
    return [{"id": str(r["id"]), "source": str(r["source"]), "target": str(r["target"]),
             "categories": [str(c) for c in (r.get("categories") or [])],
             "provenance": str(r.get("provenance") or "unknown")} for r in table.to_pylist()]


def build(args: argparse.Namespace) -> dict[str, Any]:
    rng = random.Random(args.seed)
    blocked = {json.loads(l)["source"] for l in args.frozen.read_text(encoding="utf-8").splitlines() if l.strip()}
    blocked |= set(pq.read_table(args.base_dir / "dev-stratified.parquet", columns=["source"]).column("source").to_pylist())
    for path in args.eval_dir.glob("*.jsonl"):
        blocked |= {json.loads(l)["source"] for l in path.read_text(encoding="utf-8").splitlines() if l.strip()}
    base = read_rows(args.base_dir / "train.parquet")
    seen = {row["source"] for row in base}

    readings = load_readings()
    pool = english_pool(readings)
    added, stats = [], Counter()

    def emit(domain: str, template: str, fillers: list[str], targets: list[str], slots: list[dict], category: str) -> None:
        source_row = make_row(domain, "", template, fillers, slots)
        target_row = make_row(domain, "", template, targets, slots)
        source, target = source_row["source"], target_row["source"]
        stats[f"{domain}:considered"] += 1
        if not score_row(source_row, target)["ok"]:
            stats[f"{domain}:checker_rejected"] += 1
            return
        if source in blocked:
            stats[f"{domain}:blocked"] += 1
            return
        if source in seen:
            stats[f"{domain}:duplicate"] += 1
            return
        seen.add(source)
        added.append({"id": f"e41-{domain}:{len(added)}", "source": source, "target": target,
                      "categories": [category, f"e41_{domain}"],
                      "provenance": "generated_e41_domain_checker_verified"})

    generators = [
        ("card", args.cards, card_pair, CARD_FRAMES, "Числові рядки — банківські картки"),
        ("iban", args.ibans, iban_pair, IBAN_FRAMES, "Числові рядки — IBAN"),
        ("phone", args.phones, phone_pair, PHONE_FRAMES, "Номери телефонів"),
        ("coordinates", args.coordinates, coordinate_pair, COORD_FRAMES, "Географічні координати"),
    ]
    for domain, count, generate, frames, category in generators:
        for _ in range(count):
            text, reading, slot = generate(rng)
            emit(domain, rng.choice(frames), [text], [reading], [slot], category)
    for template, shown, targets, slots in english_rows(rng, pool, readings, args.english):
        emit("english", template, shown, targets, slots, "Транслітерація")

    rng.shuffle(added)
    args.output.mkdir(parents=True, exist_ok=True)
    combined = [*base, *added]
    pq.write_table(pa.Table.from_pylist(combined), args.output / "train.parquet", compression="zstd")
    dev = read_rows(args.base_dir / "dev-stratified.parquet")
    pq.write_table(pa.Table.from_pylist(dev), args.output / "dev-stratified.parquet", compression="zstd")
    added_by_domain = Counter(r["id"].split(":")[0].removeprefix("e41-") for r in added)
    report = {
        "base": {"path": str(args.base_dir / "train.parquet"), "rows": len(base)},
        "english_pool_words": len(pool), "added": len(added), "added_by_domain": dict(added_by_domain),
        "stats": dict(stats), "total_rows": len(combined),
        "train_sha256": sha256_file(args.output / "train.parquet"),
        "samples": {d: next(({"source": r["source"], "target": r["target"]} for r in added
                             if r["id"].startswith(f"e41-{d}:")), None) for d in added_by_domain},
    }
    write_json(args.output / "manifest.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=ROOT / "data/e40_oblique_counts")
    parser.add_argument("--frozen", type=Path, default=ROOT / "reports/hf-uktn-frozen-3597.jsonl")
    parser.add_argument("--eval-dir", type=Path, default=ROOT / "reports/domain-eval")
    parser.add_argument("--output", type=Path, default=ROOT / "data/e41_domain_growth")
    parser.add_argument("--seed", type=int, default=41041)
    parser.add_argument("--cards", type=int, default=5000)
    parser.add_argument("--ibans", type=int, default=4000)
    parser.add_argument("--phones", type=int, default=7000)
    parser.add_argument("--coordinates", type=int, default=8000)
    parser.add_argument("--english", type=int, default=12000)
    report = build(parser.parse_args())
    print(json.dumps({k: report[k] for k in ("english_pool_words", "added", "added_by_domain", "stats", "total_rows", "samples")},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
