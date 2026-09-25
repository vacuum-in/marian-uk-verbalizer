from src.verbalizer.evaluation.selection import composite_score


def test_safety_gate_makes_higher_scoring_checkpoint_ineligible() -> None:
    config = {
        "weights": {"exact_match": 1.0},
        "safety_gates": {"repetition_loop_rate": 0.0},
    }
    unsafe = composite_score(
        {"exact_match": 0.9, "repetition_loop_rate": 0.01},
        config,
    )
    safe = composite_score(
        {"exact_match": 0.8, "repetition_loop_rate": 0.0},
        config,
    )
    assert not unsafe["eligible"]
    assert safe["eligible"]
