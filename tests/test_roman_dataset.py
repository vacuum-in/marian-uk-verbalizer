import re
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from scripts.prepare_roman_numeral_dataset import ordinal, to_roman


ROOT = Path(__file__).resolve().parents[1]


def test_roman_encoder_is_limited_to_ivx_policy() -> None:
    assert to_roman(4) == "IV"
    assert to_roman(39) == "XXXIX"
    with pytest.raises(ValueError, match="restricted to 1..39"):
        to_roman(40)


def test_active_roman_splits_contain_only_i_v_x() -> None:
    for split in ("train", "validation", "test"):
        rows = pq.read_table(ROOT / "data/roman_small" / f"{split}.parquet").to_pylist()
        assert rows
        for row in rows:
            assert 1 <= int(row["roman_value"]) <= 39
            tokens = re.findall(r"\b[IVXLCDM]+\b", str(row["source"]))
            assert tokens
            assert all(set(token) <= {"I", "V", "X"} for token in tokens)


def test_ukrainian_ordinal_inflection_covers_case_gender_and_number() -> None:
    assert ordinal(23, "masc_nom") == "двадцять третій"
    assert ordinal(23, "masc_gen") == "двадцять третього"
    assert ordinal(23, "masc_loc") == "двадцять третьому"
    assert ordinal(23, "fem_nom") == "двадцять третя"
    assert ordinal(23, "fem_gen") == "двадцять третьої"
    assert ordinal(23, "fem_acc") == "двадцять третю"
    assert ordinal(23, "plural_gen") == "двадцять третіх"


def test_one_thousand_ordinal_matches_project_policy() -> None:
    assert ordinal(1045, "masc_loc") == "тисяча сорок п’ятому"
