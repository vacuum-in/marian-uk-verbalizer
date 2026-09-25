from src.verbalizer._audit_numparse import parse_numbers
from src.verbalizer.data.books import caps_token_kind, is_holdout, prose_identity_stratum
from src.verbalizer.evaluation.book_domain import identity_verdict, numeric_roundtrip, roman_value


def test_holdout_split_is_deterministic_and_near_fraction() -> None:
    ids = [str(index) for index in range(1, 2001)]
    first = [book for book in ids if is_holdout(book, 0.09)]
    assert first == [book for book in ids if is_holdout(book, 0.09)]
    assert 140 < len(first) < 220


def test_caps_tokens_split_dictionary_words_from_abbreviations() -> None:
    assert caps_token_kind("НЕГАЙНО") == "word"
    assert caps_token_kind("ТОБІ") == "word"
    for token in ("ООН", "СБУ", "НАТО", "КТ"):
        assert caps_token_kind(token) == "abbr"
    assert caps_token_kind("ЖПТ") == "unknown"


def test_prose_identity_strata() -> None:
    assert prose_identity_stratum("Шерсть на спині стала дибки.") == "plain"
    assert prose_identity_stratum("— Вона витримає?") == "dialogue"
    assert prose_identity_stratum("— Я вже знову молодець, — сказала вона.") == "one_letter_start"
    assert prose_identity_stratum("СЕСТРА НЕ ЗНАЄ, ДЕ ПЕРЕБУВАЄ ГЕНІЙ") == "allcaps"


def test_prose_identity_rejects_text_that_needs_verbalization() -> None:
    for sentence in (
        "Він живе в м. Львові вже десять років.",
        "Аналізи показали, що це гепатит С.",
        "Твори Л. М. Толстого перечитуємо.",
        "Це було нереальнооо круто!",
        "Гарного дня :)",
        "Заклад входить до ЮНЕСКО разом із ООН.",
        "О 15:00 почалося слухання.",
    ):
        assert prose_identity_stratum(sentence) is None, sentence


def test_identity_verdict_levels() -> None:
    assert identity_verdict("Тримай язик на прив’язі.", "Тримай язик на прив'язі.") == "exact"
    assert identity_verdict("— Тоді слухай сюди.", "—Тоді слухай сюди.") == "whitespace_only"
    assert identity_verdict("— Стривай, — сказав він.", "— Квадратвай, — сказав він.") == "changed"


def test_parser_reads_ordinal_plurals_and_hundreds_ordinals() -> None:
    assert parse_numbers("у шістдесятих роках") == [60]
    assert parse_numbers("тисяча дев'ятсот двадцятих") == [1920]
    assert parse_numbers("двохсотий раз") == [200]


def test_numeric_roundtrip() -> None:
    assert numeric_roundtrip("{937}", "дев'ятсот тридцять сім")["ok"]
    assert not numeric_roundtrip("{937}", "квадратний корінь з дев'яноста семи")["ok"]
    assert numeric_roundtrip("Зустріч о 15:00.", "Зустріч о п'ятнадцятій.")["ok"]
    assert not numeric_roundtrip("Зараз 11:50.", "Зараз одинадцять.")["ok"]
    assert numeric_roundtrip("Фонд розподілив 5 100 000 гривень.", "Фонд розподілив п'ять мільйонів сто тисяч гривень.")["ok"]
    assert numeric_roundtrip("— 83 72 91", "— вісімдесят три сімдесят два дев'яносто один")["ok"]
    assert not numeric_roundtrip("— 83 72 91", "— два трильйони триста сімдесят два мільйони")["ok"]
    left = numeric_roundtrip("Рейс 266.", "Рейс 266.")
    assert left["digits_left"] and not left["ok"]


def test_numeric_roundtrip_accepts_legitimate_readings() -> None:
    assert numeric_roundtrip("на висоті 2 500 м", "на висоті дві тисячі п'ятсот метрів")["ok"]
    assert numeric_roundtrip("Номер 32027.", "Номер тридцять два нуль двадцять сім.")["ok"]
    assert numeric_roundtrip("Дата: 7.01.80", "Дата: сьомого січня вісімдесятого року.")["ok"]
    assert numeric_roundtrip("На початку 60-х один службовець", "На початку шістдесятих один службовець")["ok"]
    assert numeric_roundtrip("на початку 2000-х", "на початку двотисячних")["ok"]


def test_numeric_roundtrip_rejects_wrong_readings() -> None:
    assert not numeric_roundtrip("«П-014»", "«П-один чотири»")["ok"]
    assert not numeric_roundtrip("[275]", "[ двадцять сім тисяч п'ятсот вісімдесят п'ять]")["ok"]
    assert not numeric_roundtrip("навесні 2008 року", "навесні двадцять восьмого року")["ok"]
    assert not numeric_roundtrip("о 8.02", "о восьмій лютого")["ok"]


def test_roman_value() -> None:
    assert roman_value("XIV") == 14
    assert roman_value("IV") == 4
    assert numeric_roundtrip("Людовік XIV", "Людовік чотирнадцятий")["ok"]


def test_chemistry_domain_detection() -> None:
    from src.verbalizer.evaluation.domains import chemistry_tokens, domains_of

    assert chemistry_tokens("Mg і Zn") == ["Mg", "Zn"]
    assert chemistry_tokens("Формула H₂O та NaCl, осад Fe₂O₃, Ca(OH)₂") == ["H₂O", "NaCl", "Fe₂O₃", "Ca(OH)₂"]
    for text in ("вітамін B12", "діагноз I10", "двигун V6", "рейс PS452", "Rh−", "BBC і CNN", "Акт II", "In the"):
        assert chemistry_tokens(text) == [], text
    assert "chemistry" in domains_of([], "Оксид MgO")
    assert domains_of(["Гроші/Валюта"]) == {"currency"}
