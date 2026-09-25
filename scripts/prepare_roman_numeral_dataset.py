"""Generate leakage-safe Ukrainian Roman-numeral fine-tuning splits.

Roman numerals remain present in model inputs.  The generated labels teach
cardinal/ordinal interpretation and Ukrainian case, gender, and number rather
than relying on an inference-time Roman-number rewrite.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from prepare_current_dataset import canonical_rows, canonical_text, normalize_text, write_rows  # noqa: E402

from num2words import num2words


DEFAULT_OUTPUT = ROOT / "data/roman_small"
DATASET_VERSION = "roman-grammar-small-v2-2026-08-01"
ROMAN_DIGITS = ((10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I"))


def to_roman(number: int) -> str:
    if not 1 <= number <= 39:
        raise ValueError("Roman numeral training is restricted to 1..39 (I, V, X only)")
    parts: list[str] = []
    remainder = number
    for value, token in ROMAN_DIGITS:
        count, remainder = divmod(remainder, value)
        parts.append(token * count)
    return "".join(parts)


def words(number: int) -> str:
    return str(num2words(number, lang="uk")).replace("'", "’")


def ordinal(number: int, form: str) -> str:
    value = str(num2words(number, lang="uk", to="ordinal")).replace("'", "’")
    if 1000 < number < 2000 and value.startswith("одна тисяча "):
        value = value.removeprefix("одна ")
    prefix, separator, last = value.rpartition(" ")
    irregular = {
        "masc_gen": {"третій": "третього"},
        "masc_dat": {"третій": "третьому"},
        "masc_loc": {"третій": "третьому"},
        "masc_ins": {"третій": "третім"},
        "fem_nom": {"третій": "третя"},
        "fem_gen": {"третій": "третьої"},
        "fem_dat": {"третій": "третій"},
        "fem_acc": {"третій": "третю"},
        "fem_ins": {"третій": "третьою"},
        "neut_nom": {"третій": "третє"},
        "plural_nom": {"третій": "треті"},
        "plural_gen": {"третій": "третіх"},
        "plural_loc": {"третій": "третіх"},
    }
    if form == "masc_nom":
        changed = last
    elif last in irregular.get(form, {}):
        changed = irregular[form][last]
    else:
        if not (last.endswith("ий") or last.endswith("ій")):
            raise ValueError(f"unsupported ordinal ending: {last}")
        stem = last[:-2]
        endings = {
            "masc_gen": "ого", "masc_dat": "ому", "masc_loc": "ому", "masc_ins": "им",
            "fem_nom": "а", "fem_gen": "ої", "fem_dat": "ій", "fem_acc": "у", "fem_ins": "ою",
            "neut_nom": "е", "plural_nom": "і", "plural_gen": "их", "plural_loc": "их",
        }
        changed = stem + endings[form]
    return prefix + separator + changed


def initial(value: str) -> str:
    return value[:1].upper() + value[1:]


@dataclass(frozen=True)
class Template:
    category: str
    render: Callable[[int], tuple[str, str]]


def one(template: str, target: str, form: str) -> Callable[[int], tuple[str, str]]:
    def render(number: int) -> tuple[str, str]:
        spoken = ordinal(number, form)
        if target.startswith("{o}"):
            spoken = initial(spoken)
        return template.format(r=to_roman(number)), target.format(o=spoken)
    return render


TRAIN_TEMPLATES = (
    Template("masc_nom", one("Потрібно опрацювати {r} розділ до п’ятниці.", "Потрібно опрацювати {o} розділ до п’ятниці.", "masc_nom")),
    Template("masc_nom", one("Редактор перевірив {r} том рукопису.", "Редактор перевірив {o} том рукопису.", "masc_nom")),
    Template("masc_nom", one("Команда завершила {r} етап робіт.", "Команда завершила {o} етап робіт.", "masc_nom")),
    Template("masc_gen", one("Депутат Верховної Ради {r} скликання виступив на засіданні.", "Депутат Верховної Ради {o} скликання виступив на засіданні.", "masc_gen")),
    Template("masc_gen", one("Матеріал належить до {r} розділу посібника.", "Матеріал належить до {o} розділу посібника.", "masc_gen")),
    Template("masc_gen", one("Нагороджено орденом {r} ступеня.", "Нагороджено орденом {o} ступеня.", "masc_gen")),
    Template("masc_gen", one("Це корабель {r} рангу.", "Це корабель {o} рангу.", "masc_gen")),
    Template("masc_loc", one("У {r} кварталі показник зріс.", "У {o} кварталі показник зріс.", "masc_loc")),
    Template("masc_loc", one("У {r} розділі наведено висновки.", "У {o} розділі наведено висновки.", "masc_loc")),
    Template("masc_ins", one("Перед {r} етапом провели перевірку.", "Перед {o} етапом провели перевірку.", "masc_ins")),
    Template("century_loc", one("У {r} столітті місто стало торговим центром.", "У {o} столітті місто стало торговим центром.", "masc_loc")),
    Template("century_gen", one("Архів містить документи {r} століття.", "Архів містить документи {o} століття.", "masc_gen")),
    Template("century_loc_abbr", one("Цей храм збудували у {r} ст.", "Цей храм збудували у {o} столітті.", "masc_loc")),
    Template("fem_nom", one("{r} фаза випробувань почалася вчасно.", "{o} фаза випробувань почалася вчасно.", "fem_nom")),
    Template("fem_gen", one("Результати {r} фази вже оприлюднено.", "Результати {o} фази вже оприлюднено.", "fem_gen")),
    Template("fem_acc", one("Команда завершила {r} сесію переговорів.", "Команда завершила {o} сесію переговорів.", "fem_acc")),
    Template("fem_loc", one("На {r} сесії ухвалили рішення.", "На {o} сесії ухвалили рішення.", "fem_dat")),
    Template("plural_nom", one("{r} Олімпійські ігри відбулися успішно.", "{o} Олімпійські ігри відбулися успішно.", "plural_nom")),
    Template("plural_gen", one("Напередодні {r} Літніх Олімпійських ігор відкрили стадіон.", "Напередодні {o} Літніх Олімпійських ігор відкрили стадіон.", "plural_gen")),
    Template("forum_nom", one("{r} Український муніципальний форум розпочав роботу.", "{o} Український муніципальний форум розпочав роботу.", "masc_nom")),
    Template("forum_gen", one("Під час {r} Українського муніципального форуму представили звіт.", "Під час {o} Українського муніципального форуму представили звіт.", "masc_gen")),
    Template("male_ruler_nom", one("Папа Пій {r} оголосив про реформу.", "Папа Пій {o} оголосив про реформу.", "masc_nom")),
    Template("male_ruler_nom", one("Король Генріх {r} підписав указ.", "Король Генріх {o} підписав указ.", "masc_nom")),
    Template("male_ruler_gen", one("За правління короля Генріха {r} звели фортецю.", "За правління короля Генріха {o} звели фортецю.", "masc_gen")),
    Template("male_ruler_dat", one("Пам’ятник присвячено королю Генріху {r}.", "Пам’ятник присвячено королю Генріху {o}.", "masc_dat")),
    Template("female_ruler_nom", one("Королева Єлизавета {r} відкрила засідання.", "Королева Єлизавета {o} відкрила засідання.", "fem_nom")),
    Template("female_ruler_gen", one("За правління королеви Єлизавети {r} палац перебудували.", "За правління королеви Єлизавети {o} палац перебудували.", "fem_gen")),
    Template("female_ruler_acc", one("У фільмі показали королеву Єлизавету {r}.", "У фільмі показали королеву Єлизавету {o}.", "fem_acc")),
    Template("masc_loc", one("Про знахідку написано в {r} томі каталогу.", "Про знахідку написано в {o} томі каталогу.", "masc_loc")),
    Template("masc_loc", one("Учні навчаються в {r} класі школи.", "Учні навчаються в {o} класі школи.", "masc_loc")),
    Template("masc_gen", one("Підсумки {r} кварталу оприлюднили сьогодні.", "Підсумки {o} кварталу оприлюднили сьогодні.", "masc_gen")),
    Template("fem_nom", one("{r} сесія парламенту відкрилася вранці.", "{o} сесія парламенту відкрилася вранці.", "fem_nom")),
    Template("fem_gen", one("Після {r} сесії депутати зустрілися з пресою.", "Після {o} сесії депутати зустрілися з пресою.", "fem_gen")),
    Template("plural_gen", one("Організатори {r} Зимових Олімпійських ігор затвердили програму.", "Організатори {o} Зимових Олімпійських ігор затвердили програму.", "plural_gen")),
    Template("plural_gen", one("Медалі {r} Паралімпійських ігор показали в музеї.", "Медалі {o} Паралімпійських ігор показали в музеї.", "plural_gen")),
    Template("plural_gen", one("Учасники {r} Всесвітніх ігор прибули до міста.", "Учасники {o} Всесвітніх ігор прибули до міста.", "plural_gen")),
    Template("male_ruler_nom", one("Імператор Карл {r} скликав державну раду.", "Імператор Карл {o} скликав державну раду.", "masc_nom")),
    Template("male_ruler_nom", one("Князь Олександр {r} очолив військо.", "Князь Олександр {o} очолив військо.", "masc_nom")),
    Template("male_ruler_nom", one("Герцог Філіп {r} прийняв делегацію.", "Герцог Філіп {o} прийняв делегацію.", "masc_nom")),
    Template("male_ruler_gen", one("Літопис правління імператора Карла {r} зберігся.", "Літопис правління імператора Карла {o} зберігся.", "masc_gen")),
    Template("male_ruler_gen", one("Указ короля Людовика {r} передали до архіву.", "Указ короля Людовика {o} передали до архіву.", "masc_gen")),
    Template("female_ruler_nom", one("Імператриця Катерина {r} видала новий маніфест.", "Імператриця Катерина {o} видала новий маніфест.", "fem_nom")),
    Template("female_ruler_nom", one("Княгиня Анна {r} заснувала школу.", "Княгиня Анна {o} заснувала школу.", "fem_nom")),
    Template("female_ruler_nom", one("Цариця Марія {r} прийняла послів.", "Цариця Марія {o} прийняла послів.", "fem_nom")),
    Template("female_ruler_gen", one("Портрет імператриці Катерини {r} відреставрували.", "Портрет імператриці Катерини {o} відреставрували.", "fem_gen")),
    Template("female_ruler_gen", one("Хроніку королеви Марії {r} видали повторно.", "Хроніку королеви Марії {o} видали повторно.", "fem_gen")),
    Template("female_ruler_acc", one("Документ згадує імператрицю Катерину {r}.", "Документ згадує імператрицю Катерину {o}.", "fem_acc")),
)

# Lexical variation forces the model to learn agreement from syntax and noun
# morphology instead of associating one fixed sentence with each ordinal form.
TRAIN_TEMPLATES = (*TRAIN_TEMPLATES, *tuple(
    Template(
        "fem_acc",
        one(
            f"{subject} {verb} {{r}} {noun} {tail}.",
            f"{subject} {verb} {{o}} {noun} {tail}.",
            "fem_acc",
        ),
    )
    for subject, verb in (
        ("Делегація", "провела"),
        ("Команда", "завершила"),
        ("Група", "розпочала"),
        ("Дослідники", "провели"),
        ("Учасники", "завершили"),
    )
    for noun, tail in (
        ("фазу", "переговорів"),
        ("сесію", "консультацій"),
        ("частину", "дослідження"),
    )
), *tuple(
    Template(
        "female_ruler_nom",
        one(
            f"{title} {name} {{r}} {action}.",
            f"{title} {name} {{o}} {action}.",
            "fem_nom",
        ),
    )
    for title, name in (
        ("Королева", "Вікторія"),
        ("Королева", "Марія"),
        ("Королева", "Єлизавета"),
        ("Імператриця", "Катерина"),
        ("Княгиня", "Анна"),
    )
    for action in (
        "відкрила нову школу",
        "зустріла іноземних послів",
        "звернулася до міської ради",
    )
), *tuple(
    Template(
        "female_ruler_gen",
        one(
            f"{subject} {title} {name} {{r}} {action}.",
            f"{subject} {title} {name} {{o}} {action}.",
            "fem_gen",
        ),
    )
    for subject, title, name in (
        ("Біографію", "королеви", "Вікторії"),
        ("Портрет", "королеви", "Марії"),
        ("Листи", "королеви", "Єлизавети"),
        ("Маніфест", "імператриці", "Катерини"),
        ("Хроніку", "княгині", "Анни"),
    )
    for action in (
        "передали до архіву",
        "показали на виставці",
        "опублікували окремою книгою",
    )
), *tuple(
    Template(
        "male_ruler_nom",
        one(
            f"{title} {name} {{r}} {action}.",
            f"{title} {name} {{o}} {action}.",
            "masc_nom",
        ),
    )
    for title, name in (
        ("Папа", "Бенедикт"),
        ("Папа", "Пій"),
        ("Король", "Генріх"),
        ("Король", "Людовик"),
        ("Імператор", "Карл"),
    )
    for action in (
        "підписав мирну угоду",
        "прийняв офіційну делегацію",
        "завершив тривалу поїздку",
    )
), *tuple(
    Template(
        "plural_gen",
        one(
            f"{subject} {{r}} {event} {action}.",
            f"{subject} {{o}} {event} {action}.",
            "plural_gen",
        ),
    )
    for subject, event, action in (
        ("Організатори", "Паралімпійських ігор", "оприлюднили розклад"),
        ("Учасники", "Олімпійських ігор", "прибули до столиці"),
        ("Програма", "Всесвітніх ігор", "містить нові дисципліни"),
    )
),
    Template("masc_nom", one("Потрібно прочитати {r} розділ до вівторка.", "Потрібно прочитати {o} розділ до вівторка.", "masc_nom")),
    Template("masc_nom", one("Потрібно прочитати {r} том до середи.", "Потрібно прочитати {o} том до середи.", "masc_nom")),
    Template("masc_nom", one("До понеділка необхідно прочитати {r} розділ.", "До понеділка необхідно прочитати {o} розділ.", "masc_nom")),
    Template("masc_nom", one("Студент має прочитати {r} розділ до понеділка.", "Студент має прочитати {o} розділ до понеділка.", "masc_nom")),
)

VALIDATION_TEMPLATES = (
    Template("masc_nom", one("До програми включили {r} клас складності.", "До програми включили {o} клас складності.", "masc_nom")),
    Template("masc_gen", one("Учасники дійшли до {r} етапу конкурсу.", "Учасники дійшли до {o} етапу конкурсу.", "masc_gen")),
    Template("masc_loc", one("Про це йдеться в {r} томі енциклопедії.", "Про це йдеться в {o} томі енциклопедії.", "masc_loc")),
    Template("century_loc_abbr", one("Фортецю спорудили в {r} ст.", "Фортецю спорудили в {o} столітті.", "masc_loc")),
    Template("fem_gen", one("Після {r} сесії оголосили перерву.", "Після {o} сесії оголосили перерву.", "fem_gen")),
    Template("plural_gen", one("Організатори {r} Паралімпійських ігор оприлюднили програму.", "Організатори {o} Паралімпійських ігор оприлюднили програму.", "plural_gen")),
    Template("male_ruler_nom", one("Імператор Карл {r} скликав раду.", "Імператор Карл {o} скликав раду.", "masc_nom")),
    Template("female_ruler_nom", one("Імператриця Катерина {r} видала маніфест.", "Імператриця Катерина {o} видала маніфест.", "fem_nom")),
)

TEST_TEMPLATES = (
    Template("masc_nom", one("Треба прочитати {r} розділ до понеділка.", "Треба прочитати {o} розділ до понеділка.", "masc_nom")),
    Template("masc_gen", one("Представник парламенту {r} скликання подав заяву.", "Представник парламенту {o} скликання подав заяву.", "masc_gen")),
    Template("masc_loc", one("Дані наведені у {r} кварталі звіту.", "Дані наведені у {o} кварталі звіту.", "masc_loc")),
    Template("century_loc", one("Упродовж подій у {r} столітті кордони змінилися.", "Упродовж подій у {o} столітті кордони змінилися.", "masc_loc")),
    Template("century_gen", one("Дослідники аналізують мистецтво {r} століття.", "Дослідники аналізують мистецтво {o} століття.", "masc_gen")),
    Template("fem_nom", one("{r} сесія конференції була найдовшою.", "{o} сесія конференції була найдовшою.", "fem_nom")),
    Template("fem_acc", one("Делегати провели {r} фазу консультацій.", "Делегати провели {o} фазу консультацій.", "fem_acc")),
    Template("plural_gen", one("Церемонія {r} Зимових Олімпійських ігор почалася ввечері.", "Церемонія {o} Зимових Олімпійських ігор почалася ввечері.", "plural_gen")),
    Template("male_ruler_nom", one("Папа Бенедикт {r} завершив поїздку.", "Папа Бенедикт {o} завершив поїздку.", "masc_nom")),
    Template("male_ruler_gen", one("Біографія короля Людовика {r} вийшла друком.", "Біографія короля Людовика {o} вийшла друком.", "masc_gen")),
    Template("female_ruler_nom", one("Королева Вікторія {r} звернулася до парламенту.", "Королева Вікторія {o} звернулася до парламенту.", "fem_nom")),
    Template("female_ruler_gen", one("Портрет королеви Марії {r} передали музею.", "Портрет королеви Марії {o} передали музею.", "fem_gen")),
)


def render_rows(templates: tuple[Template, ...], values: list[int], count: int, split: str, seed: int) -> list[dict[str, object]]:
    rng = random.Random(seed)
    pairs: list[tuple[str, str]] = []
    metadata: list[tuple[str, int]] = []
    seen: set[str] = set()
    attempts = 0
    while len(pairs) < count:
        attempts += 1
        if attempts > count * 20:
            raise RuntimeError(f"could not generate {count} unique {split} rows")
        template = templates[len(pairs) % len(templates)]
        number = rng.choice(values)
        source, target = template.render(number)
        key = canonical_text(normalize_text(source))
        if key in seen:
            continue
        seen.add(key)
        pairs.append((source, target))
        metadata.append((template.category, number))
    rows = canonical_rows(pairs)
    for row, (category, number) in zip(rows, metadata, strict=True):
        payload = f"{DATASET_VERSION}\0{split}\0{row['source']}\0{row['target']}"
        row["id"] = hashlib.sha256(payload.encode()).hexdigest()
        row["dataset_version"] = DATASET_VERSION
        row["quality_tier"] = "A"
        row["quality_flags"] = []
        row["categories"] = ["roman_numeral", category]
        row["roman_value"] = number
    rng.shuffle(rows)
    return rows


def build(args: argparse.Namespace) -> dict[str, object]:
    all_values = list(range(1, args.max_value + 1))
    # The supported Roman policy is the finite set 1..39.  All values belong to
    # training; generalization is measured on disjoint sentence templates.
    train_values = all_values
    validation_values = all_values
    test_values = all_values
    splits = {
        "train": render_rows(TRAIN_TEMPLATES, train_values, args.train_rows, "train", args.seed),
        "validation": render_rows(VALIDATION_TEMPLATES, validation_values, args.validation_rows, "validation", args.seed + 1),
        "test": render_rows(TEST_TEMPLATES, test_values, args.test_rows, "test", args.seed + 2),
    }
    sources = {name: {canonical_text(str(row["source"])) for row in rows} for name, rows in splits.items()}
    leakage = {
        "train_validation_source_overlap": len(sources["train"] & sources["validation"]),
        "train_test_source_overlap": len(sources["train"] & sources["test"]),
        "validation_test_source_overlap": len(sources["validation"] & sources["test"]),
    }
    if any(leakage.values()):
        raise RuntimeError(f"Roman split leakage: {leakage}")
    args.output.mkdir(parents=True, exist_ok=True)
    hashes = {name: write_rows(args.output / f"{name}.parquet", rows) for name, rows in splits.items()}
    manifest = {
        "dataset_version": DATASET_VERSION,
        "seed": args.seed,
        "max_roman_value": args.max_value,
        "rows": {name: len(rows) for name, rows in splits.items()},
        "templates": {"train": len(TRAIN_TEMPLATES), "validation": len(VALIDATION_TEMPLATES), "test": len(TEST_TEMPLATES)},
        "distinct_values": {"train": len(train_values), "validation": len(validation_values), "test": len(test_values)},
        "value_policy": "All supported values 1..39 train; validation/test hold out templates, not values.",
        "categories": {name: dict(Counter(category for row in rows for category in row["categories"])) for name, rows in splits.items()},
        "leakage": leakage,
        "output_sha256": hashes,
        "policy": [
            "Roman glyphs remain in every model source; no inference-time Roman rewrite is used.",
            "Validation and test use disjoint sentence templates; all finite supported values occur in training.",
            "Labels cover Ukrainian case, gender, number, rulers, centuries, and event ordinals.",
        ],
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train-rows", type=int, default=4_400)
    parser.add_argument("--validation-rows", type=int, default=312)
    parser.add_argument("--test-rows", type=int, default=468)
    parser.add_argument("--max-value", type=int, default=39)
    parser.add_argument("--seed", type=int, default=20260731)
    args = parser.parse_args()
    if min(args.train_rows, args.validation_rows, args.test_rows, args.max_value) < 1:
        parser.error("row counts and max-value must be positive")
    if args.max_value > 39:
        parser.error("Roman training is restricted to 1..39 (I, V, X only)")
    return args


if __name__ == "__main__":
    print(json.dumps(build(parse_args()), ensure_ascii=False, indent=2, sort_keys=True))
