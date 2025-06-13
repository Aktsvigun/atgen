import numpy as np
from datasets import Dataset

from .base_strategy import BaseStrategy


def random_strategy(
    unlabeled_pool: Dataset,
    num_to_label: int,
    seed: int = 42,
) -> list[int]:
    rng = np.random.default_rng(seed=seed)
    ids = list(unlabeled_pool["id"])
    rng.shuffle(ids)
    return ids[:num_to_label]


class RandomStrategy(BaseStrategy):
    def __init__(self, subsample_size: int = -1, seed: int = 42):
        super().__init__()
        self.seed = seed
        self.random_init = True

    def __call__(
        self, unlabeled_pool: Dataset, num_to_label: int, *args, **kwargs
    ) -> list[int]:
        return random_strategy(unlabeled_pool, num_to_label, self.seed)
