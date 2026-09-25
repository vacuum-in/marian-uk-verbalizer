"""Length-bucketed, token-budgeted batch sampling."""

from __future__ import annotations

import math
import random
from collections.abc import Iterator, Sequence

from torch.utils.data import Sampler


def estimated_pair_tokens(source: str, target: str, source_limit: int, target_limit: int) -> int:
    source_tokens = min(source_limit, max(4, math.ceil(len(source) / 3)))
    target_tokens = min(target_limit, max(4, math.ceil(len(target) / 3)))
    return source_tokens + target_tokens


class TokenBudgetBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        lengths: Sequence[int],
        max_batch_tokens: int,
        max_batch_examples: int,
        seed: int,
        bucket_size: int = 256,
        shuffle: bool = True,
    ) -> None:
        if not lengths or max_batch_tokens <= 0 or max_batch_examples <= 0:
            raise ValueError("lengths and batch limits must be positive")
        self.lengths = list(lengths)
        self.max_batch_tokens = max_batch_tokens
        self.max_batch_examples = max_batch_examples
        self.seed = seed
        self.bucket_size = bucket_size
        self.shuffle = shuffle
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _ordered_indices(self) -> list[int]:
        ordered = sorted(range(len(self.lengths)), key=self.lengths.__getitem__)
        if not self.shuffle:
            return ordered
        rng = random.Random(self.seed + self.epoch)
        buckets = [ordered[index : index + self.bucket_size] for index in range(0, len(ordered), self.bucket_size)]
        for bucket in buckets:
            rng.shuffle(bucket)
        rng.shuffle(buckets)
        return [index for bucket in buckets for index in bucket]

    def _batches(self) -> list[list[int]]:
        batches: list[list[int]] = []
        current: list[int] = []
        current_max = 0
        for index in self._ordered_indices():
            candidate_max = max(current_max, self.lengths[index])
            candidate_count = len(current) + 1
            if current and (
                candidate_max * candidate_count > self.max_batch_tokens
                or candidate_count > self.max_batch_examples
            ):
                batches.append(current)
                current = []
                current_max = 0
            current.append(index)
            current_max = max(current_max, self.lengths[index])
        if current:
            batches.append(current)
        return batches

    def __iter__(self) -> Iterator[list[int]]:
        yield from self._batches()

    def __len__(self) -> int:
        return len(self._batches())

