from src.verbalizer.numeric import NUMERIC_POLICY_VERSION
from scripts.train_marian import Collator


def test_training_accepts_the_routing_only_preprocessing_version() -> None:
    assert NUMERIC_POLICY_VERSION == "model-routing-v1"


def test_collator_rejects_removed_rule_based_preprocessing() -> None:
    collator = object.__new__(Collator)
    collator.input_preprocessing = "legacy_normalize"
    try:
        collator.prepare_source("Потрібно прочитати XL розділ.")
    except ValueError as error:
        assert "legacy_normalize" not in str(error)
    else:
        raise AssertionError("removed preprocessing mode was accepted")
