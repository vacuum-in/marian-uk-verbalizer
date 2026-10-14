"""String, identity, and category metrics for generated verbalizations."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict
from typing import Any, Sequence

from src.verbalizer.data.semantic import semantic_flags
from src.verbalizer.numeric import NumericKind, detect_numeric_spans, numeric_fidelity_failures


WHITESPACE_RE = re.compile(r"\s+")
WORD_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)
PUNCTUATION_RE = re.compile(r"[^\w\s]", re.UNICODE)


def normalize_evaluation_text(value: str) -> str:
    value = unicodedata.normalize("NFC", value)
    value = WHITESPACE_RE.sub(" ", value).strip()
    return re.sub(r"\s+([.,;:!?])", r"\1", value)


def spoken_form(value: str) -> str:
    """What a listener can tell apart: case, punctuation and hyphens are not heard.

    Evaluation only; nothing at runtime uses it. "Де те пе." and "де-те-пе" are the same
    utterance, "дванадцяте" and "дванадцять" are not.
    """
    value = unicodedata.normalize("NFC", value).casefold().replace("ʼ", "'").replace("’", "'")
    return " ".join(re.findall(r"[^\W_]+(?:'[^\W_]+)*", value))


def edit_distance(left: Sequence[Any], right: Sequence[Any]) -> int:
    if left == right:
        return 0
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for left_index, left_value in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_value in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_value != right_value),
                )
            )
        previous = current
    return previous[-1]


def empty_totals() -> dict[str, int]:
    return {
        "rows": 0,
        "exact": 0,
        "character_edits": 0,
        "reference_characters": 0,
        "word_edits": 0,
        "reference_words": 0,
        "identity_rows": 0,
        "identity_exact": 0,
        "semantic_flagged": 0,
        "character_3gram_overlap": 0,
        "reference_character_3grams": 0,
        "prediction_character_3grams": 0,
        "empty_predictions": 0,
        "likely_deletions": 0,
        "likely_insertions": 0,
        "repetition_loops": 0,
        "punctuation_corruptions": 0,
        "invalid_unicode": 0,
        "output_truncations": 0,
        "digit_fidelity_rows": 0,
        "digit_fidelity_exact": 0,
        "numeric_value_rows": 0,
        "numeric_value_exact": 0,
        "identifier_group_rows": 0,
        "identifier_group_exact": 0,
        "contextual_morphology_rows": 0,
        "contextual_morphology_exact": 0,
        "numeric_model_rejected": 0,
        "spoken_exact": 0,
    }


def ngrams(value: str, size: int) -> Counter[str]:
    if len(value) < size:
        return Counter({value: 1}) if value else Counter()
    return Counter(value[index : index + size] for index in range(len(value) - size + 1))


def has_repetition_loop(value: str) -> bool:
    tokens = [token.casefold() for token in WORD_TOKEN_RE.findall(value)]
    if any(tokens[index : index + 5] == [tokens[index]] * 5 for index in range(max(0, len(tokens) - 4))):
        return True
    if len(tokens) >= 3:
        token_ngrams = Counter(tuple(tokens[index : index + 3]) for index in range(len(tokens) - 2))
        if token_ngrams and max(token_ngrams.values()) >= 5:
            return True
    return re.search(r"([^\w\s])\1{4,}", value) is not None


def has_invalid_unicode(value: str) -> bool:
    return "\ufffd" in value or any(
        unicodedata.category(character) == "Cc" and character not in "\t\n\r"
        for character in value
    )


def punctuation_corrupted(reference: str, prediction: str) -> bool:
    expected = len(PUNCTUATION_RE.findall(reference))
    actual = len(PUNCTUATION_RE.findall(prediction))
    return abs(actual - expected) > max(3, round(expected * 0.5))


def add_example(
    totals: dict[str, int],
    source: str,
    reference: str,
    prediction: str,
    terminated_with_eos: bool = True,
    numeric_model_rejected: bool = False,
) -> list[str]:
    source = normalize_evaluation_text(source)
    reference = normalize_evaluation_text(reference)
    prediction = normalize_evaluation_text(prediction)
    reference_words = reference.split()
    prediction_words = prediction.split()
    reference_ngrams = ngrams(reference, 3)
    prediction_ngrams = ngrams(prediction, 3)
    ngram_overlap = sum((reference_ngrams & prediction_ngrams).values())
    flags = semantic_flags(source, prediction)
    numeric_spans = detect_numeric_spans(source)
    numeric_failures = numeric_fidelity_failures(source, prediction)
    failed_offsets = {(span.start, span.end) for span in numeric_failures}
    identity_kinds = {
        NumericKind.FORMULA,
        NumericKind.IBAN,
        NumericKind.CARD,
        NumericKind.PHONE,
        NumericKind.IDENTIFIER_SUFFIX,
        NumericKind.CODE,
    }
    identity_spans = [span for span in numeric_spans if span.kind in identity_kinds]
    grouped_identifier_spans = [
        span for span in identity_spans if span.kind != NumericKind.FORMULA
    ]
    contextual_spans = [
        span
        for span in numeric_spans
        if span.kind == NumericKind.CARDINAL and len(span.candidates) > 1
    ]
    totals["rows"] += 1
    totals["exact"] += int(reference == prediction)
    totals["spoken_exact"] += int(spoken_form(reference) == spoken_form(prediction))
    totals["character_edits"] += edit_distance(reference, prediction)
    totals["reference_characters"] += len(reference)
    totals["word_edits"] += edit_distance(reference_words, prediction_words)
    totals["reference_words"] += len(reference_words)
    is_identity = source == reference
    totals["identity_rows"] += int(is_identity)
    totals["identity_exact"] += int(is_identity and prediction == source)
    totals["semantic_flagged"] += int(bool(flags))
    totals["character_3gram_overlap"] += ngram_overlap
    totals["reference_character_3grams"] += sum(reference_ngrams.values())
    totals["prediction_character_3grams"] += sum(prediction_ngrams.values())
    totals["empty_predictions"] += int(not prediction)
    length_ratio = len(prediction) / max(1, len(reference))
    totals["likely_deletions"] += int(length_ratio < 0.5)
    totals["likely_insertions"] += int(length_ratio > 2.0)
    totals["repetition_loops"] += int(has_repetition_loop(prediction))
    totals["punctuation_corruptions"] += int(punctuation_corrupted(reference, prediction))
    totals["invalid_unicode"] += int(has_invalid_unicode(prediction))
    totals["output_truncations"] += int(not terminated_with_eos)
    totals["numeric_value_rows"] += int(bool(numeric_spans))
    totals["numeric_value_exact"] += int(bool(numeric_spans) and not numeric_failures)
    totals["digit_fidelity_rows"] += int(bool(identity_spans))
    totals["digit_fidelity_exact"] += int(
        bool(identity_spans)
        and not any((span.start, span.end) in failed_offsets for span in identity_spans)
    )
    totals["identifier_group_rows"] += int(bool(grouped_identifier_spans))
    totals["identifier_group_exact"] += int(
        bool(grouped_identifier_spans)
        and not any(
            (span.start, span.end) in failed_offsets for span in grouped_identifier_spans
        )
    )
    totals["contextual_morphology_rows"] += int(bool(contextual_spans))
    normalized_prediction = prediction.casefold().replace("'", "’").replace("ʼ", "’")
    totals["contextual_morphology_exact"] += int(
        bool(contextual_spans)
        and all(span.safe_reading.casefold() in normalized_prediction for span in contextual_spans)
    )
    totals["numeric_model_rejected"] += int(numeric_model_rejected)
    return flags


def finalize(totals: dict[str, int]) -> dict[str, int | float | None]:
    rows = totals["rows"]
    identity_rows = totals["identity_rows"]
    ngram_precision = totals["character_3gram_overlap"] / max(1, totals["prediction_character_3grams"])
    ngram_recall = totals["character_3gram_overlap"] / max(1, totals["reference_character_3grams"])
    return {
        **totals,
        "exact_match": totals["exact"] / rows if rows else 0.0,
        "spoken_match": totals.get("spoken_exact", 0) / rows if rows else 0.0,
        "cer": totals["character_edits"] / max(1, totals["reference_characters"]),
        "wer": totals["word_edits"] / max(1, totals["reference_words"]),
        "identity_accuracy": totals["identity_exact"] / identity_rows if identity_rows else None,
        "semantic_flag_rate": totals["semantic_flagged"] / rows if rows else 0.0,
        "empty_prediction_rate": totals["empty_predictions"] / rows if rows else 0.0,
        "likely_deletion_rate": totals["likely_deletions"] / rows if rows else 0.0,
        "likely_insertion_rate": totals["likely_insertions"] / rows if rows else 0.0,
        "repetition_loop_rate": totals["repetition_loops"] / rows if rows else 0.0,
        "punctuation_corruption_rate": totals["punctuation_corruptions"] / rows if rows else 0.0,
        "invalid_unicode_rate": totals["invalid_unicode"] / rows if rows else 0.0,
        "output_truncation_rate": totals["output_truncations"] / rows if rows else 0.0,
        "digit_sequence_fidelity": (
            totals["digit_fidelity_exact"] / totals["digit_fidelity_rows"]
            if totals["digit_fidelity_rows"] else None
        ),
        "numeric_value_fidelity": (
            totals["numeric_value_exact"] / totals["numeric_value_rows"]
            if totals["numeric_value_rows"] else None
        ),
        "identifier_group_fidelity": (
            totals["identifier_group_exact"] / totals["identifier_group_rows"]
            if totals["identifier_group_rows"] else None
        ),
        "contextual_morphology_accuracy": (
            totals["contextual_morphology_exact"] / totals["contextual_morphology_rows"]
            if totals["contextual_morphology_rows"] else None
        ),
        "character_3gram_f1": (
            2 * ngram_precision * ngram_recall / (ngram_precision + ngram_recall)
            if ngram_precision + ngram_recall
            else 0.0
        ),
    }


def evaluate_examples(examples: list[dict[str, Any]]) -> dict[str, Any]:
    overall = empty_totals()
    by_category: dict[str, dict[str, int]] = defaultdict(empty_totals)
    flag_counts: dict[str, int] = defaultdict(int)
    for row in examples:
        row_totals = empty_totals()
        flags = add_example(
            row_totals,
            row["source"],
            row["reference"],
            row["prediction"],
            bool(row.get("terminated_with_eos", True)),
            bool(row.get("numeric_model_rejected", False)),
        )
        for key, value in row_totals.items():
            overall[key] += value
        for flag in flags:
            flag_counts[flag] += 1
        for category in row.get("categories") or []:
            category_totals = by_category[str(category)]
            for key, value in row_totals.items():
                category_totals[key] += value
    finalized_categories = {
        category: finalize(values) for category, values in sorted(by_category.items())
    }
    macro_keys = ("exact_match", "cer", "wer", "character_3gram_f1", "semantic_flag_rate")
    macro = {
        key: (
            sum(float(values[key]) for values in finalized_categories.values())
            / max(1, len(finalized_categories))
        )
        for key in macro_keys
    }
    return {
        "overall": finalize(overall),
        "category_macro": macro,
        "categories": finalized_categories,
        "semantic_flags": dict(sorted(flag_counts.items())),
    }
