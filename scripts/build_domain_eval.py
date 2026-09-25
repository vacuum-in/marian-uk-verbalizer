"""Build held-out per-domain evaluation sets with automatic, variant-tolerant scoring.

Each row is a frame of fixed Ukrainian text with typed slots (see
src/verbalizer/evaluation/domain_check.py). Nothing here produces training data:
the English words, sentence frames and random seed are reserved for evaluation.

  english_words  held-out words (10% hash split of the two-provider bronze words)
  card_iban      16-digit cards and UA IBANs, including zero runs
  phone          Ukrainian phone formats
  coordinates    DMS and decimal latitude/longitude with Ukrainian hemisphere marks
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.verbalizer.data.english_readings import accepted_variants, load_readings  # noqa: E402
from src.verbalizer.training.manifest import sha256_file, write_json  # noqa: E402

ENGLISH_HOLDOUT_SALT = "domain-eval-english-v1"


def english_is_holdout(word: str, fraction: float = 0.10) -> bool:
    digest = hashlib.sha256(f"{ENGLISH_HOLDOUT_SALT}:{word}".encode()).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF < fraction


ENGLISH_SINGLE = [
    "Колега згадав {0} на нараді.", "У звіті з'явилося слово {0}.", "Назва проєкту — {0}.",
    "Вона тихо сказала {0} і вийшла.", "На футболці було написано {0}.", "Пароль від вайфаю — {0}.",
    "Команда обрала назву {0} для нового релізу.", "Він двічі повторив слово {0}.", "{0}",
    "Учитель попросив прочитати слово {0} вголос.", "Пошуковий запит {0} нічого не дав.",
]
ENGLISH_PAIR = [
    "Вона сказала «{0} {1}» і засміялася.", "Слоган кампанії — {0} {1}.", "Сторінка називалася {0} {1}.",
    "Нова серія має назву {0} {1}.", "{0} {1}",
]
ENGLISH_LIST = ["Список тегів: {list}.", "Нові терміни тижня: {list}.", "У словнику уроку: {list}.",
                "На дошці були слова {list}.", "Ключові слова статті: {list}."]

CARD_FRAMES = ["Номер картки для переказу: {0}.", "Гроші надішліть на картку {0}, дякую!",
               "Карта {0} заблокована до з'ясування.", "{0}", "Перевірте, будь ласка, картку {0} ще раз."]
IBAN_FRAMES = ["Реквізити для оплати: {0}.", "Рахунок отримувача {0}, призначення — благодійний внесок.",
               "Переказ на {0} надійшов учора.", "{0}"]
PHONE_FRAMES = ["Телефонуйте за номером {0}.", "Мій новий номер {0}, збережи його.", "Гаряча лінія: {0}.",
                "{0}", "Якщо щось трапиться, дзвоніть на {0} у будь-який час."]
COORD_FRAMES = ["Судно передало координати {0}.", "Точка збору: {0}.", "Маяк стоїть на {0}.",
                "{0}", "Дрон зафіксував ціль у точці {0} і повернувся."]
LAT_MARK = {"N": "пн. ш.", "S": "пд. ш."}
LON_MARK = {"E": "сх. д.", "W": "зх. д."}


def english_rows(rng: random.Random, readings: dict, n: int) -> list[dict]:
    words = sorted(w for w in readings if english_is_holdout(w) and accepted_variants(readings, w) and w.isalpha())
    rows = []
    shapes = ["single"] * 4 + ["pair"] * 2 + ["list"] * 4
    while len(rows) < n:
        shape = rng.choice(shapes)
        count = {"single": 1, "pair": 2, "list": rng.randint(3, 8)}[shape]
        chosen = rng.sample(words, count)
        shown = [w.capitalize() if rng.random() < 0.35 else w for w in chosen]
        slots = [{"kind": "english", "word": w, "variants": accepted_variants(readings, w)} for w in chosen]
        if shape == "list":
            template = rng.choice(ENGLISH_LIST)
            joiner = rng.choice([", ", ", "])
            listed = joiner.join(["{%d}" % i for i in range(count)])
            if rng.random() < 0.3:
                listed = ", ".join(["{%d}" % i for i in range(count - 1)]) + " і {%d}" % (count - 1)
            template = template.replace("{list}", listed)
        else:
            template = rng.choice(ENGLISH_SINGLE if shape == "single" else ENGLISH_PAIR)
        rows.append(make_row("english_words", shape, template, shown, slots))
    return rows


def make_row(domain: str, shape: str, template: str, fillers: list[str], slots: list[dict]) -> dict:
    frame, source, rest = [], "", template
    for index, filler in enumerate(fillers):
        before, rest = rest.split("{%d}" % index, 1)
        frame.append(before)
        source += before + filler
        if filler.endswith(".") and rest.startswith("."):
            rest = rest[1:]
    frame.append(rest)
    source += rest
    return {"domain": domain, "shape": shape, "source": source, "frame": frame, "slots": slots}


def digits_string(rng: random.Random, n: int, zero_run: float = 0.0) -> str:
    digits = [str(rng.randint(0, 9)) for _ in range(n)]
    if zero_run and rng.random() < zero_run:
        start, length = rng.randint(0, n - 4), rng.randint(3, 7)
        for i in range(start, min(n, start + length)):
            digits[i] = "0"
    return "".join(digits)


def card_iban_rows(rng: random.Random, n: int) -> list[dict]:
    rows = []
    while len(rows) < n:
        if rng.random() < 0.55:
            digits = rng.choice("2345") + digits_string(rng, 15, zero_run=0.25)
            groups = [digits[i:i + 4] for i in range(0, 16, 4)]
            text = rng.choice([" ".join(groups), "-".join(groups), digits])
            slot = {"kind": "digits", "digits": digits}
            rows.append(make_row("card_iban", "card", rng.choice(CARD_FRAMES), [text], [slot]))
        else:
            digits = digits_string(rng, 27, zero_run=0.6)
            body = "UA" + digits
            text = " ".join(body[i:i + 4] for i in range(0, len(body), 4)) if rng.random() < 0.6 else body
            slot = {"kind": "iban", "country": "UA", "digits": digits}
            rows.append(make_row("card_iban", "iban", rng.choice(IBAN_FRAMES), [text], [slot]))
    return rows


def phone_rows(rng: random.Random, n: int) -> list[dict]:
    rows = []
    operators = ["50", "63", "66", "67", "68", "73", "93", "95", "96", "97", "98", "99", "44", "32", "56", "57", "62"]
    while len(rows) < n:
        op = rng.choice(operators)
        rest = digits_string(rng, 7, zero_run=0.1)
        shape = rng.choice(["plus_spaced", "plus_brackets", "local_spaced", "local_dashes", "local_brackets", "free", "tollfree"])
        if shape == "tollfree":
            rest = digits_string(rng, 6)
            text, digits, plus = f"0 800 {rest[:2]} {rest[2:4]} {rest[4:]}", "0800" + rest, False
        elif shape == "plus_spaced":
            text, digits, plus = f"+380 {op} {rest[:3]} {rest[3:5]} {rest[5:]}", "380" + op + rest, True
        elif shape == "plus_brackets":
            text, digits, plus = f"+380 ({op}) {rest[:3]}-{rest[3:5]}-{rest[5:]}", "380" + op + rest, True
        elif shape == "local_spaced":
            text, digits, plus = f"0{op} {rest[:3]} {rest[3:5]} {rest[5:]}", "0" + op + rest, False
        elif shape == "local_dashes":
            text, digits, plus = f"0{op}-{rest[:3]}-{rest[3:5]}-{rest[5:]}", "0" + op + rest, False
        elif shape == "local_brackets":
            text, digits, plus = f"(0{op}) {rest[:3]}-{rest[3:5]}-{rest[5:]}", "0" + op + rest, False
        else:
            text, digits, plus = f"0{op}{rest}", "0" + op + rest, False
        slot = {"kind": "phone", "digits": digits, "plus": plus}
        rows.append(make_row("phone", shape, rng.choice(PHONE_FRAMES), [text], [slot]))
    return rows


def dms(rng: random.Random, max_deg: int, seconds: bool, ascii_marks: bool) -> tuple[str, list[int], list[str]]:
    deg, minute, second = rng.randint(0, max_deg), rng.randint(0, 59), rng.randint(0, 59)
    d, m, s = ("°", "'", '"') if ascii_marks else ("°", "′", "″")
    if seconds:
        return f"{deg}{d}{minute:02d}{m}{second:02d}{s}", [deg, minute, second], ["deg", "min", "sec"]
    return f"{deg}{d}{minute:02d}{m}", [deg, minute], ["deg", "min"]


def coordinate_rows(rng: random.Random, n: int) -> list[dict]:
    rows = []
    while len(rows) < n:
        ns, ew = rng.choice("NNNS"), rng.choice("EEEW")
        style = rng.choice(["dms", "dms", "dm", "decimal"])
        both = rng.random() < 0.6
        parts, values, units, hemis = [], [], [], []
        for axis, hemi, max_deg, marks in (("lat", ns, 89, LAT_MARK), ("lon", ew, 179, LON_MARK)):
            if axis == "lon" and not both:
                break
            if style == "decimal":
                whole, frac = rng.randint(0, max_deg), rng.randint(1, 9999)
                text, vals, unit = f"{whole},{frac:04d}°", [whole, frac], ["deg"]
                if frac < 1000:
                    continue
            else:
                text, vals, unit = dms(rng, max_deg, style == "dms", rng.random() < 0.2)
            parts.append(f"{text} {marks[hemi]}")
            values += vals
            units += unit
            hemis.append(hemi)
        if not parts:
            continue
        text = ", ".join(parts)
        slot = {"kind": "coords", "values": values, "units": units, "hemispheres": hemis}
        rows.append(make_row("coordinates", style + ("_pair" if len(parts) == 2 else ""), rng.choice(COORD_FRAMES), [text], [slot]))
    return rows


CONTEXT_CONNECTORS = {
    "card": ["на картку {0}", "з картки {0}", "(картка {0})"],
    "iban": ["на рахунок {0}", "(рахунок {0})"],
    "phone": ["за номером {0}", "(телефон {0})"],
    "coordinates": ["у точці {0}", "(координати {0})"],
}


def context_rows(rng: random.Random, n: int, books_dir: Path, split_path: Path) -> list[dict]:
    """Identifiers inserted into sentences from the held-out books (never used for training)."""

    from src.verbalizer.data.books import iter_book_sentences, prose_identity_stratum

    holdout = set(json.loads(split_path.read_text(encoding="utf-8"))["holdout"])
    sentences = []
    for path in sorted(books_dir.glob("book-*.json")):
        if path.stem.removeprefix("book-") in holdout:
            sentences += [s.text for s in iter_book_sentences(path)
                          if 30 <= len(s.text) <= 160 and s.text.endswith(".") and len(s.text.split()) >= 6
                          and prose_identity_stratum(s.text) == "plain"]
    rng.shuffle(sentences)
    makers = {
        "card": lambda: card_iban_rows(rng, 1)[0],
        "iban": None, "phone": lambda: phone_rows(rng, 1)[0], "coordinates": lambda: coordinate_rows(rng, 1)[0],
    }
    rows = []
    for text in sentences:
        if len(rows) >= n:
            break
        domain = rng.choice(list(CONTEXT_CONNECTORS))
        while True:
            sample = (makers[domain] or makers["card"])()
            if domain in ("card", "iban"):
                want = "card" if domain == "card" else "iban"
                if sample["shape"] != want:
                    continue
            break
        value = sample["source"][len(sample["frame"][0]):len(sample["source"]) - len(sample["frame"][1]) or None]
        words = text.split(" ")
        cut = rng.randint(2, len(words) - 3)
        template = " ".join(words[:cut]) + " " + rng.choice(CONTEXT_CONNECTORS[domain]) + " " + " ".join(words[cut:])
        if template.count("{0}") != 1:
            continue
        rows.append(make_row("context", domain, template, [value], sample["slots"]))
    return rows


# Real acronyms read letter by letter; reserved for evaluation (E49+ training excludes them).
EVAL_ACRONYMS = ["ФБР", "СБУ", "ЗСУ", "МВС", "ЦРУ", "НБУ", "ДТП", "ЗМІ", "ВРУ", "КМДА", "ЦВК", "СНД", "ФСБ", "ДБР",
                 "ДСНС", "ПДР", "ТЦК", "ВЛК", "ЄС", "США", "РНБО", "ДФС", "ОДА", "КПІ", "ПФУ", "МЗС", "ДПСУ", "ЗНО"]
ACRONYM_FRAMES = ["Я пропрацював у {0} двадцять років,", "{0} розпочала розслідування.", "Про це повідомили в {0}.",
                  "Представник {0} відмовився від коментарів.", "Звернення до {0} надіслали вчора.", "{0}",
                  "— {0}? — перепитав він.", "За даними {0}, ситуація стабільна."]
CONSONANTS = "БВГДЖЗКЛМНПРСТФХЦЧШЩ"


def acronym_rows(rng: random.Random, n: int) -> list[dict]:
    rows = []
    while len(rows) < n:
        if rng.random() < 0.7:
            acronym = rng.choice(EVAL_ACRONYMS)
        else:
            acronym = "".join(rng.choice(CONSONANTS) for _ in range(rng.randint(2, 4)))
        slot = {"kind": "letters", "letters": acronym}
        rows.append(make_row("acronyms", "real" if acronym in EVAL_ACRONYMS else "random",
                             rng.choice(ACRONYM_FRAMES), [acronym], [slot]))
    return rows


def roman_case_rows(rng: random.Random, n: int) -> list[dict]:
    """Roman ordinals whose case/gender must agree with the following noun."""

    sys.path.insert(0, str(ROOT / "scripts"))
    from roman_case import EVAL_FRAMES, make_rows

    rows = []
    user = {"frame": ["Потрібно прочитати ", " розділ до понеділка."], "value": 20, "ordinal": "двадцятий"}
    rows.append({"domain": "roman_case", "shape": "accs", "source": "Потрібно прочитати XX розділ до понеділка.",
                 "frame": user["frame"], "slots": [{"kind": "exact", "variants": [user["ordinal"]]}]})
    for r in make_rows(rng, EVAL_FRAMES, n):
        before, after = r["frame"].split("{r}")
        after = after.replace("{n}", r["noun"])
        rows.append({"domain": "roman_case", "shape": r["case"], "source": r["source"],
                     "frame": [before, after], "slots": [{"kind": "exact", "variants": [r["ordinal"]]}]})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=500)
    parser.add_argument("--seed", type=int, default=90241)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/domain-eval")
    parser.add_argument("--books-dir", type=Path, default=Path("/home/devops/ukr-books/books"))
    parser.add_argument("--split", type=Path, default=ROOT / "data/book_domain/split.json")
    parser.add_argument("--only", nargs="*", default=None, help="rebuild just these sets")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    readings = load_readings()
    builders = {
        "english_words": lambda rng: english_rows(rng, readings, args.rows),
        "card_iban": lambda rng: card_iban_rows(rng, args.rows),
        "phone": lambda rng: phone_rows(rng, args.rows),
        "coordinates": lambda rng: coordinate_rows(rng, args.rows),
        "context": lambda rng: context_rows(rng, args.rows, args.books_dir, args.split),
        "acronyms": lambda rng: acronym_rows(rng, 300),
        "roman_case": lambda rng: roman_case_rows(rng, 300),
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest.update({"seed": args.seed, "english_holdout_salt": ENGLISH_HOLDOUT_SALT})
    manifest.setdefault("sets", {})
    for index, (name, build) in enumerate(builders.items()):
        if args.only and name not in args.only:
            continue
        rng = random.Random(args.seed * 100 + index)
        rows, seen = [], set()
        for row in build(rng):
            if row["source"] not in seen:
                seen.add(row["source"])
                rows.append({"id": f"{name}-{len(rows):04d}", **row})
        path = args.output_dir / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        manifest["sets"][name] = {"rows": len(rows), "sha256": sha256_file(path)}
        print(f"{name:14} {len(rows):4} rows  e.g. {rows[1]['source'][:80]}")
    write_json(manifest_path, manifest)


if __name__ == "__main__":
    main()
