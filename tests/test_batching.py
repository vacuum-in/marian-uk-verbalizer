from src.verbalizer.training.batching import TokenBudgetBatchSampler, estimated_pair_tokens


def test_token_budget_sampler_is_deterministic_and_bounded() -> None:
    lengths = [10, 12, 20, 30, 35, 40, 50]
    sampler = TokenBudgetBatchSampler(
        lengths,
        max_batch_tokens=80,
        max_batch_examples=3,
        seed=17,
        bucket_size=4,
        shuffle=True,
    )
    first = list(sampler)
    second = list(sampler)
    assert first == second
    assert sorted(index for batch in first for index in batch) == list(range(len(lengths)))
    for batch in first:
        assert len(batch) <= 3
        assert max(lengths[index] for index in batch) * len(batch) <= 80


def test_estimated_tokens_respects_limits() -> None:
    assert estimated_pair_tokens("а" * 900, "б" * 1500, 256, 384) == 640

