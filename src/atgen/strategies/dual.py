from datasets import Dataset
from transformers import PreTrainedModel

from .base_strategy import BaseStrategy
from .idds import idds_sampling
from .bleuvar import bleuvar
from .random_strategy import random_strategy

"""
https://arxiv.org/pdf/2503.00867v1
"""


class DualStrategy(BaseStrategy):
    def __init__(self, seed=None):
        super().__init__()
        self.seed = seed
        self.p = 0.5  # as per https://github.com/pgiouroukis/dual/blob/main/src/active_learning/active_learning_strategy_dual.py#L29

    def __call__(
        self,
        model: PreTrainedModel,
        unlabeled_pool: Dataset,
        labeled_pool: Dataset,
        num_to_label: int,
        *args,
        **kwargs,
    ) -> list[int]:
        return dual(
            model, unlabeled_pool, num_to_label, labeled_pool, p=self.p, seed=self.seed
        )


def dual(model, X_pool, n_instances, X_train, p=0.5, seed=None, device=None, **kwargs):
    bleuvar_n_instances = max(1, int(n_instances * p))
    random_n_instances = max(0, n_instances - bleuvar_n_instances)

    idds_query_idxs, idds_query, _ = idds_sampling(
        model, X_pool, n_instances, X_train, seed=seed, device=device, **kwargs
    )
    idds_query_idxs = [int(x) for x in list(idds_query_idxs)]

    bleuvar_query_idxs, _, _ = bleuvar(model, idds_query, bleuvar_n_instances, **kwargs)
    bleuvar_query_idxs = [idds_query_idxs[int(x)] for x in list(bleuvar_query_idxs)]

    leftover_pool = X_pool.filter(
        lambda _, idx: idx not in bleuvar_query_idxs,
        with_indices=True,
    )
    random_query_idxs = random_strategy(
        leftover_pool,
        random_n_instances,
        seed,
    )

    return bleuvar_query_idxs + [int(x) for x in list(random_query_idxs)]
