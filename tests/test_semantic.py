from src.verbalizer.data.semantic import semantic_flags


def test_correct_integer_and_percent_have_no_semantic_flags() -> None:
    assert semantic_flags(
        "Показник зріс на 15,5%.",
        "Показник зріс на п'ятнадцять цілих п'ять десятих відсотка.",
    ) == []


def test_wrong_decimal_place_is_flagged() -> None:
    flags = semantic_flags("Показник становить 15,50%.", "Показник становить п'ятнадцять цілих п'ятдесят тисячних відсотка.")
    assert "decimal_precision_unverified" in flags


def test_sign_and_currency_are_checked() -> None:
    flags = semantic_flags("Температура -24 °C, ціна $25.", "Температура двадцять чотири градуси, ціна двадцять п'ять.")
    assert "sign_not_preserved" in flags
    assert "currency_unverified" in flags


def test_unit_abbreviations_require_token_boundaries() -> None:
    assert semantic_flags("Хомутовський, 12.", "Хомутовський, дванадцятий.") == []
    assert "unit_unverified" in semantic_flags("Відстань 12 м.", "Відстань дванадцять.")


def test_numeric_range_is_not_a_date_or_negative_sign() -> None:
    flags = semantic_flags(
        "Тариф зросте на 12%-24%.",
        "Тариф зросте на дванадцять відсотків - двадцять чотири відсотки.",
    )
    assert "date_unverified" not in flags
    assert "sign_not_preserved" not in flags
