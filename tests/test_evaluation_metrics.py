from src.verbalizer.evaluation.metrics import (
    edit_distance,
    evaluate_examples,
    has_repetition_loop,
    normalize_evaluation_text,
)


def test_category_aggregation_reuses_identical_row_metrics() -> None:
    examples = [
        {
            "source": "Значення 2.",
            "reference": "Значення два.",
            "prediction": "Значення два.",
            "categories": ["integer", "multiple_spans"],
            "terminated_with_eos": True,
        }
    ]
    report = evaluate_examples(examples)
    assert report["overall"]["exact"] == 1
    assert report["categories"]["integer"] == report["categories"]["multiple_spans"]
    assert report["categories"]["integer"] == report["overall"]


def test_normalization_and_edit_distance() -> None:
    assert normalize_evaluation_text("  П’ять\tгривень.  ") == "П’ять гривень."
    assert normalize_evaluation_text("п’ять гривень .") == "п’ять гривень."
    assert edit_distance("кіт", "кит") == 1
    assert edit_distance(["п'ять", "гривень"], ["шість", "гривень"]) == 1


def test_identity_and_category_metrics() -> None:
    report = evaluate_examples(
        [
            {
                "source": "Це звичайне речення.",
                "reference": "Це звичайне речення.",
                "prediction": "Це звичайне речення.",
                "categories": ["no_change"],
            },
            {
                "source": "Є 5 книг.",
                "reference": "Є п'ять книг.",
                "prediction": "Є п'ять книг.",
                "categories": ["integer"],
            },
        ]
    )
    assert report["overall"]["exact_match"] == 1.0
    assert report["overall"]["identity_accuracy"] == 1.0
    assert report["overall"]["character_3gram_f1"] == 1.0
    assert report["categories"]["integer"]["rows"] == 1
    assert report["category_macro"]["exact_match"] == 1.0


def test_generation_safety_metrics() -> None:
    assert has_repetition_loop("місто " * 8)
    report = evaluate_examples(
        [
            {
                "source": "Було 12 людей.",
                "reference": "Було дванадцять людей.",
                "prediction": "місто " * 20,
                "categories": ["integer"],
                "terminated_with_eos": False,
            }
        ]
    )
    overall = report["overall"]
    assert overall["likely_insertion_rate"] == 1.0
    assert overall["repetition_loop_rate"] == 1.0
    assert overall["output_truncation_rate"] == 1.0


def test_numeric_release_metrics_are_reported_separately() -> None:
    report = evaluate_examples(
        [
            {
                "source": "Формула H₂SO₄.",
                "reference": "Формула аш два ес о чотири.",
                "prediction": "Формула аш два ес о чотири.",
                "categories": ["formula"],
            },
            {
                "source": "У неї 4 малят.",
                "reference": "У неї четверо малят.",
                "prediction": "У неї четверо малят.",
                "categories": ["collective"],
            },
        ]
    )
    overall = report["overall"]
    assert overall["digit_sequence_fidelity"] == 1.0
    assert overall["numeric_value_fidelity"] == 1.0
    assert overall["identifier_group_fidelity"] is None
    assert overall["contextual_morphology_accuracy"] is None


def test_spoken_match_ignores_only_what_is_not_heard() -> None:
    from src.verbalizer.evaluation.metrics import add_example, empty_totals, finalize, spoken_form

    assert spoken_form("За добу сталося 3-4 Де те пе.") == spoken_form("за добу сталося 3 4 де-те-пе")
    assert spoken_form("п’ять") == spoken_form("п'ять")
    assert spoken_form("дванадцять дверей") != spoken_form("дванадцяте дверей")
    totals = empty_totals()
    add_example(totals, "у ДТП", "у Де те пе", "у де-те-пе.")
    report = finalize(totals)
    assert report["exact_match"] == 0.0
    assert report["spoken_match"] == 1.0
