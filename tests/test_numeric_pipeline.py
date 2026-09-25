from src.verbalizer.numeric import (
    NumericKind,
    detect_numeric_spans,
    enforce_numeric_fidelity,
    numeric_model_input,
    prepare_protected_input,
    prepare_training_pair,
)
from src.verbalizer.text import TextRoute, route_text


def test_formula_is_classified_but_never_rewritten() -> None:
    source = "Застосунок Chemistry 1.5 миттєво розпізнає H₂SO₄."
    formula = next(span for span in detect_numeric_spans(source) if span.kind is NumericKind.FORMULA)
    assert formula.digit_sequence == "24"
    assert formula.groups == ("2", "4")
    assert formula.candidates == ()
    assert formula.safe_reading == "H₂SO₄"
    assert numeric_model_input(source) == source


def test_numeric_input_routes_to_marian_unchanged() -> None:
    source = "Відомо, що 100 - 45 = 55."
    span = detect_numeric_spans(source)[0]
    assert span.kind is NumericKind.ARITHMETIC
    assert span.digit_sequence == "1004555"
    routed = route_text(source)
    assert routed.route is TextRoute.MARIAN
    assert routed.text == source


def test_roman_numerals_route_to_marian_unchanged() -> None:
    source = "Акт II драми відтворює побут XIX ст."
    spans = detect_numeric_spans(source)
    assert [(span.kind, span.text) for span in spans] == [
        (NumericKind.ROMAN, "II"),
        (NumericKind.ROMAN, "XIX"),
    ]
    routed = route_text(source)
    assert routed.route is TextRoute.MARIAN
    assert routed.text == source


def test_roman_router_is_limited_to_values_below_fifty() -> None:
    for numeral in ("I", "IV", "V", "IX", "X", "XIX", "XL", "XLIX"):
        assert route_text(f"Акт {numeral} драми.").route is TextRoute.MARIAN
    for numeral in ("L", "LI", "C", "D", "M"):
        assert not any(
            span.kind is NumericKind.ROMAN
            for span in detect_numeric_spans(f"Акт {numeral} драми.")
        )


def test_labelled_identifier_is_routing_metadata_only() -> None:
    source = "Ооо, мій IBAN закінчується на 6008."
    span = detect_numeric_spans(source)[0]
    assert span.kind is NumericKind.IDENTIFIER_SUFFIX
    assert span.groups == ("6008",)
    assert span.candidates == ()
    assert route_text(source).text == source


def test_training_and_inference_do_not_insert_markers() -> None:
    source = "Chemistry пояснює H₂SO₄."
    target = "Кемістрі пояснює аш два ес о чотири."
    assert prepare_protected_input(source).text == source
    assert prepare_training_pair(source, target) == (source, target, ())


def test_raw_model_output_is_never_rejected_or_corrected() -> None:
    output = "Формула аш два ес о два."
    result = enforce_numeric_fidelity("Формула H₂SO₄.", output)
    assert result.accepted is True
    assert result.text == output
    assert result.warnings == ()
    assert result.failures == ()


def test_ordinary_ukrainian_text_passes_through() -> None:
    source = "Це звичайне українське речення без вербалізації."
    routed = route_text(source)
    assert routed.route is TextRoute.PASSTHROUGH
    assert routed.text == source


def test_english_words_route_to_marian_unchanged() -> None:
    source = "Samsung, Alphabet, NVIDIA та WhatsApp."
    routed = route_text(source)
    assert routed.route is TextRoute.MARIAN
    assert routed.text == source


def test_plain_prose_passes_through() -> None:
    for source in (
        "Я не хочу редьки.",
        "— А це навіщо? — спитала Джулія.",
        "У ній зберігалося кількадесят шоколадних батончиків.",
        "СЕСТРА НЕ ЗНАЄ, ДЕ ПЕРЕБУВАЄ ГЕНІЙ",
    ):
        routed = route_text(source)
        assert routed.route is TextRoute.PASSTHROUGH, source
        assert routed.text == source


def test_cyrillic_text_that_may_need_verbalization_routes_to_marian() -> None:
    for source in (
        "Кав'ярня розташована на вул. Незалежності.",
        "Це було нереальнооо круто!",
        "Заклад входить до ЮНЕСКО.",
        "Гарного дня :)",
        "Аналізи показали, що це гепатит С.",
        "Твори Л. М. Толстого перечитуємо.",
        "Реєстрація в ЦНАП.",
    ):
        assert route_text(source).route is TextRoute.MARIAN, source


def test_short_capital_abbreviations_route_to_marian() -> None:
    for token in ("HDD", "LTE", "BRT", "ДТП"):
        source = f"Позначення {token} потребує вербалізації."
        routed = route_text(source)
        assert routed.route is TextRoute.MARIAN
        assert routed.text == source


def test_latin_word_ending_in_digits_routes_to_marian_as_one_input() -> None:
    source = "Новий iPhone14 надійшов у продаж."
    routed = route_text(source)
    assert routed.route is TextRoute.MARIAN
    assert routed.text == source
