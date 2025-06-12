from abc import ABC, abstractmethod
import random

from datasets import Dataset
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
)


class BaseStrategy(ABC):
    def __init__(self, subsample_size: int | float = -1):
        self.subsample_size = subsample_size

    @abstractmethod
    def __call__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        unlabeled_pool: Dataset,
        labeled_pool: Dataset,
        input_column_name: str,
        output_column_name: str,
        num_to_label: int,
    ) -> list[int]:
        pass

    def _select_subsample_if_necessary(
        self, unlabeled_pool: Dataset, random_subsample_seed: int = 42
    ):
        if (subsample_size := self.subsample_size) > 0:
            if isinstance(subsample_size, float):
                subsample_size = int(len(unlabeled_pool) * subsample_size)
            # Ensure the subsample size does not exceed the dataset size
            subsample_size = min(subsample_size, len(unlabeled_pool))
            # Select random indices for the subsample
            random.seed(random_subsample_seed)
            indices = random.sample(range(len(unlabeled_pool)), subsample_size)
            # Return the subsampled dataset
            return unlabeled_pool.select(indices)
        return unlabeled_pool
