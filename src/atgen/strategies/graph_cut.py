from typing import Literal

import numpy as np
from transformers import PreTrainedModel, PreTrainedTokenizer

from submodlib import GraphCutFunction

from .base_strategy import BaseStrategy
from ..utils.get_embeddings import get_embeddings

"""
Refer to
https://decile-team-distil.readthedocs.io/en/latest/ActStrategy/distil.active_learning_strategies.html#distil.active_learning_strategies.submod_sampling.SubmodularSampling
for parameter disambiguation
"""


class GraphCutStrategy(BaseStrategy):
    def __init__(
        self,
        unlabeled_pool: list[str],
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        subsample_size: int = -1,
        lambda_: float = 0.5,
        batch_size: int = 64,
    ):
        super().__init__(subsample_size)
        self.unlabeled_pool_embeddings = (
            get_embeddings(  # Specify if model should be before AL or after
                unlabeled_pool,
                model=model,
                tokenizer=tokenizer,
                batch_size=batch_size,
                model_is_causallm=True,
            )
        )
        self.lambda_ = lambda_
        self.random_init = False

    def __call__(self, num_to_label: int, *args, **kwargs) -> list[int]:
        unlabeled_pool = self._select_subsample_if_necessary(
            self.unlabeled_pool_embeddings
        )
        return graph_cut(unlabeled_pool, num_to_label, self.lambda_)


def graph_cut(
    unlabeled_pool_embeddings: np.ndarray,
    num_to_label: int,
    lambda_: float,
    metric: Literal["cosine", "euclidean"] = "cosine",
    optimizer: Literal[
        "NaiveGreedy", "StochasticGreedy", "LazyGreedy", "LazierThanLazyGreedy"
    ] = "LazierThanLazyGreedy",
    stop_if_zero_gain: bool = False,
    stop_if_negative_gain: bool = False,
    verbose=False,
) -> list[int]:
    submod_function = GraphCutFunction(
        n=unlabeled_pool_embeddings.shape[0],
        mode="dense",
        lambdaVal=lambda_,
        data=unlabeled_pool_embeddings,
        metric=metric,
    )
    greedy_list = submod_function.maximize(
        budget=num_to_label,
        optimizer=optimizer,
        stopIfZeroGain=stop_if_zero_gain,
        stopIfNegativeGain=stop_if_negative_gain,
        verbose=verbose,
    )

    return [x[0] for x in greedy_list]
