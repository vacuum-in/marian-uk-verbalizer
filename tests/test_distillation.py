from src.verbalizer.distillation import (
    exact_digit_sequence_fidelity,
    promotion_gate,
    route_bare_digits,
    teacher_big_number_spacing,
    teacher_input,
    teacher_validation_flags,
)
from scripts.prepare_ukrainian_wikipedia_sources import acceptable_sentence, clean_wikitext
from scripts.build_distillation_corpus import build


def test_teacher_input_does_not_own_or_rewrite_numeric_text() -> None:
    assert route_bare_digits("0000") is None
    assert route_bare_digits("1234567") is None
    assert route_bare_digits("42") is None
    assert route_bare_digits("Код 0000") is None


def test_teacher_input_preserves_source_text() -> None:
    assert teacher_big_number_spacing("Борг 7000000000 гривень.") == "Борг 7000000000 гривень."
    assert teacher_big_number_spacing("Картка 1234567890123456 заблокована.") == (
        "Картка 1234567890123456 заблокована."
    )
    handled, value = teacher_input("000000")
    assert handled is False
    assert value == "000000"


def test_wikipedia_cleaning_keeps_readable_ukrainian_sentences() -> None:
    cleaned = clean_wikitext("[[Київ|Києві]] {{citation}} <ref>source</ref> розташований музей.")
    assert cleaned == "Києві розташований музей."
    assert acceptable_sentence("У Києві розташований відомий музей українського мистецтва.") is True


def test_teacher_validation_quarantines_digit_residue() -> None:
    rejected, review = teacher_validation_flags("Ціна 12 грн.", "Ціна 12 гривень.")
    assert "target_digit_residue" in rejected
    assert review


def test_identifier_fidelity_checks_digit_order() -> None:
    source = "Телефон: 067 123 45 67."
    assert exact_digit_sequence_fidelity(
        source,
        "Телефон: нуль шість сім один два три чотири п’ять шість сім.",
    ) is True
    assert exact_digit_sequence_fidelity(
        source,
        "Телефон: нуль шість сім один два три чотири п’ять шість шість.",
    ) is False


def test_promotion_gate_allows_one_point_teacher_gap_and_blocks_regression() -> None:
    teacher = [
        {
            "split": "main",
            "source": "Значення 1.",
            "primary_reference": "Значення один.",
            "prediction": "Значення один.",
        }
        for _ in range(100)
    ]
    student = [dict(row) for row in teacher]
    student[-1]["prediction"] = "Значення два."
    passing = promotion_gate(teacher, student, allowed_gap_pp=1.0)
    assert passing["status"] == "pass"
    baseline = [dict(row) for row in teacher]
    failing = promotion_gate(teacher, student, allowed_gap_pp=1.0, baseline_rows=baseline)
    assert failing["status"] == "fail"
    assert any(not check["passed"] for check in failing["checks"])


def test_corpus_builder_excludes_frozen_sources_and_keeps_validation_canonical(tmp_path) -> None:
    import argparse
    import json

    import pyarrow as pa
    import pyarrow.parquet as pq

    canonical = tmp_path / "canonical.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {"inputs": "Канонічний один.", "labels": "Канонічний один."},
                {"inputs": "Канонічний два.", "labels": "Канонічний два."},
                {"inputs": "Заморожений рядок.", "labels": "Не повинен увійти."},
            ]
        ),
        canonical,
    )
    teacher = tmp_path / "teacher.jsonl"
    teacher.write_text(
        "\n".join(
            json.dumps(row, ensure_ascii=False)
            for row in (
                {"source": "Новий учительський рядок.", "teacher_output": "Новий учительський рядок.", "status": "approved", "review_flags": []},
                {"source": "Заморожений рядок.", "teacher_output": "Заморожений рядок.", "status": "approved", "review_flags": []},
            )
        )
        + "\n",
        encoding="utf-8",
    )
    frozen = tmp_path / "frozen.jsonl"
    frozen.write_text(json.dumps({"original": "Заморожений рядок."}, ensure_ascii=False) + "\n", encoding="utf-8")
    output = tmp_path / "out"
    manifest = build(
        argparse.Namespace(
            canonical=canonical,
            teacher_capture=teacher,
            frozen=[frozen],
            output=output,
            canonical_source_column="inputs",
            canonical_target_column="labels",
            validation_rows=1,
        )
    )
    train = pq.read_table(output / "train.parquet").to_pylist()
    validation = pq.read_table(output / "validation.parquet").to_pylist()
    assert all(row["source"] != "Заморожений рядок." for row in [*train, *validation])
    assert all(row["provenance"] == "canonical" for row in validation)
    assert any(row["provenance"] == "gemma_teacher" for row in train)
    assert manifest["leakage"] == {
        "train_validation_source_overlap": 0,
        "train_frozen_source_overlap": 0,
        "validation_frozen_source_overlap": 0,
    }


def test_corpus_builder_accepts_multiple_canonical_sources_without_teacher(tmp_path) -> None:
    import argparse

    import pyarrow as pa
    import pyarrow.parquet as pq

    previous = tmp_path / "previous.parquet"
    hf = tmp_path / "hf.parquet"
    frozen = tmp_path / "frozen.parquet"
    pq.write_table(pa.Table.from_pylist([{"inputs": "Попередній рядок.", "labels": "Попередній рядок."}, {"inputs": "Ще один рядок.", "labels": "Ще один рядок."}]), previous)
    pq.write_table(pa.Table.from_pylist([{"source": "HF рядок.", "target": "HF рядок."}]), hf)
    pq.write_table(pa.Table.from_pylist([{"source": "Не використовується."}]), frozen)
    output = tmp_path / "out"
    manifest = build(
        argparse.Namespace(
            canonical=previous,
            teacher_capture=None,
            additional_canonical=[hf],
            frozen=[frozen],
            output=output,
            canonical_source_column="inputs",
            canonical_target_column="labels",
            validation_rows=1,
        )
    )
    train = pq.read_table(output / "train.parquet").to_pylist()
    assert manifest["teacher_capture"] is None
    assert any(row["provenance"] == "canonical_hf" for row in train)
