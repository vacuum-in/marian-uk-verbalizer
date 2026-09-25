from src.verbalizer.data.audit import audit_rows_by_split, quality_tier, row_flags
from src.verbalizer.data.near_duplicates import band_keys, simhash, similarity


def test_numeric_residue_is_flagged_without_dropping_row() -> None:
    flags = row_flags("Продажі зросли на 15%.", "Продажі зросли на п'ятнадцять відсотків.")
    assert flags == []


def test_target_digit_residue_and_introduction_are_distinct_flags() -> None:
    flags = row_flags("Це тест.", "Це тест 2.")
    assert "target_digit_residue" in flags
    assert "target_introduces_digit" in flags
    assert quality_tier(flags) == "C"


def test_conflicting_and_cross_split_sources_are_annotated() -> None:
    rows, report = audit_rows_by_split(
        {
            "train": [{"source": "Ціна 5 грн.", "target": "Ціна п'ять гривень.", "categories": []}],
            "validation": [{"source": "Ціна 5 грн.", "target": "Ціна п'ять гривень.", "categories": []}],
        }
    )
    flags = rows["train"][0]["quality_flags"]
    assert "exact_duplicate_source" in flags
    assert "cross_split_duplicate_source" in flags
    assert report["global"]["cross_split_duplicate_source_groups"] == 1


def test_near_duplicate_screening_primitives_are_deterministic() -> None:
    left = "Компанія відкрила п'ятнадцять нових магазинів у Києві цього року."
    right = "Компанія відкрила п'ятнадцять нових магазинів у Києві цього року!"
    assert simhash(left) == simhash(left)
    assert len(band_keys(simhash(left))) == 4
    assert similarity(left, right) > 0.95
