from src.verbalizer.text import (
    TextRoute,
    enforce_output_fidelity,
    prepare_model_input,
    route_text,
    split_model_sentences,
)


def test_marian_input_is_the_original_text() -> None:
    source = "1\u00a0234 ≤ 2\u202f000; 8 ÷ 2 ≈ 4; 20 ℃; 3 ≥ 1"
    assert prepare_model_input(source) == source


def test_model_output_is_returned_raw() -> None:
    assert enforce_output_fidelity("Значення 4,33.", "Значення 4чотири цілих.") == (
        "Значення 4чотири цілих."
    )


def test_route_text_uses_only_model_ownership() -> None:
    assert route_text("Потрібно прочитати XX розділ.").route is TextRoute.MARIAN
    assert route_text("Телефон: +380 (67) 777-17-69.").route is TextRoute.MARIAN


def test_sentence_helper_does_not_rewrite_or_split_content() -> None:
    source = "Перше речення. Друге речення: 42."
    assert split_model_sentences(source) == [source]
