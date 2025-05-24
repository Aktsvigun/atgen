from .base_strategy import Strategy

from datasets import Dataset
from transformers import PreTrainedModel

import numpy as np
import torch


import logging


class IDDSStrategy(Strategy):
    def __init__(self, seed=None):
        super().__init__()
        self.seed = seed

    def __call__(
        self,
        model: PreTrainedModel,
        unlabeled_pool: Dataset,
        labeled_pool: Dataset,
        num_to_label: int,
        *args,
        **kwargs,
    ) -> list[int]:
        return idds_sampling(
            model, unlabeled_pool, num_to_label, labeled_pool, seed=self.seed
        )[0]


def idds_sampling(
    model, X_pool, n_instances, X_train, seed=None, device=None, **kwargs
):
    cache_dir = kwargs.get("cache_dir")
    model_name = kwargs.get("embeddings_model_name", "bert-base-uncased")
    text_name = kwargs.get("text_name", "document")
    subsample_ratio = kwargs.get("subsample_ratio", 1)
    lamb = kwargs.get("lambda", 0.667)
    u_top = kwargs.get("u_top", None)
    l_top = kwargs.get("l_top", None)
    average = kwargs.get("average", False)
    sims_func = kwargs.get("sims_func", "scalar_product")
    filter_outliers = kwargs.get("filter_outliers", None)
    filtering_mode = kwargs.get("filtering_mode", None)
    batch_size = kwargs.get("embeddings_batch_size", 100)

    logging.info(
        f"Used similarities function: {sims_func}; u-top: {u_top}; l-top: {l_top}"
    )

    if filtering_mode is not None:
        uncertainty_threshold = kwargs.get("uncertainty_threshold", 0.0)
        uncertainty_mode = kwargs.get(
            "uncertainty_mode", "absolute"
        )  # "relative" or "absolute"
        generation_output = model.generate(X_pool, to_numpy=True)
        scores = generation_output["sequences_scores"]
        sequences_ids = generation_output["sequences"]

        # if filtering_mode == "uncertainty":
        #     query_idx, uncertainty_estimates = filter_by_uncertainty(
        #         uncertainty_estimates=-scores,
        #         uncertainty_threshold=uncertainty_threshold,
        #         uncertainty_mode=uncertainty_mode,
        #         n_instances=n_instances,
        #     )
        #
        # elif filtering_mode in ["rouge1", "rouge2", "rougeL", "sacrebleu"]:
        #     query_idx, uncertainty_estimates = filter_by_metric(
        #         uncertainty_threshold=uncertainty_threshold,
        #         uncertainty_mode=uncertainty_mode,
        #         texts=X_pool[model.data_config["text_name"]],
        #         generated_sequences_ids=sequences_ids,
        #         tokenizer=model.tokenizer,
        #         metric_cache_dir=model.cache_dir / "metrics",
        #         metric_name=filtering_mode,
        #         agg=kwargs.get("filtering_aggregation", "precision"),
        #         modify_uncertainties=False,
        #     )

    # subsample size = pool size / subsample_ratio
    if device is None:
        device = model.model.device
    if seed is None:
        seed = model.seed

    if subsample_ratio is not None:
        X_pool_subsample, subsample_indices = get_X_pool_subsample(
            X_pool, subsample_ratio, seed
        )  # subsample_indices indicated the indices of the subsample in the original data
    else:
        X_pool_subsample = X_pool

    similarities, counts, embeddings = get_similarities(
        model_name,
        X_pool_subsample,
        X_train,
        sims_func=sims_func,
        average=average,
        text_name=text_name,
        device=device,
        cache_dir=cache_dir,
        return_embeddings=True,
        batch_size=batch_size,
    )
    num_obs = len(similarities)
    if X_train is None:
        X_train = []

    labeled_indices = list(range(num_obs - len(X_train), num_obs))
    unlabeled_indices = list(range(num_obs - len(X_train)))

    unlabeled_indices_without_queries = list(unlabeled_indices)
    top_scores_indices = []
    top_scores = []

    if filter_outliers is not None:
        outliers_idx = []
        num_outliers = round(filter_outliers * num_obs)

    for i_query in range(n_instances):
        # Calculate similarities
        if u_top is None:
            similarities_with_unlabeled = (
                similarities[unlabeled_indices][
                    :, unlabeled_indices_without_queries
                ].sum(dim=1)
                - 1
            ) / (len(unlabeled_indices_without_queries) - 1)
        else:
            similarities_with_unlabeled = (
                similarities[unlabeled_indices][:, unlabeled_indices_without_queries]
                .topk(u_top + 1)[0]
                .sum(dim=1)
                - 1
            ) / u_top
        if len(labeled_indices) == 0:
            similarities_with_labeled = torch.zeros(len(unlabeled_indices)).to(
                similarities_with_unlabeled
            )
        elif l_top is None:
            similarities_with_labeled = similarities[unlabeled_indices][
                :, labeled_indices
            ].mean(dim=1)
        else:
            similarities_with_labeled = (
                similarities[unlabeled_indices][:, labeled_indices]
                .topk(min(len(labeled_indices), l_top))[0]
                .mean(dim=1)
            )
        scores = (
            (
                similarities_with_unlabeled * lamb
                - similarities_with_labeled * (1 - lamb)
            )
            .cpu()
            .detach()
            .numpy()
        )
        scores[top_scores_indices] = -np.inf
        if filter_outliers is not None and len(outliers_idx) > 0:
            scores[outliers_idx.cpu().numpy()] = -np.inf

        # TODO: BUG when subsample_ratio is not None
        most_similar_idx = np.argmax(scores)
        labeled_indices.append(most_similar_idx)
        if most_similar_idx in unlabeled_indices_without_queries:
            unlabeled_indices_without_queries.remove(most_similar_idx)
        top_scores_indices.append(most_similar_idx)
        top_scores.append(scores[most_similar_idx])

        if filter_outliers is not None and i_query > 0:
            outliers_idx = (
                calculate_unicentroid_mahalanobis_distance(embeddings, labeled_indices)
                .topk(num_outliers)
                .indices
            )

        scores[top_scores_indices] = top_scores
        top_scores_idx = [counts.index(i) for i in top_scores_indices]
        scores = scores[counts]

        if subsample_ratio is not None:
            query_idx = subsample_indices[top_scores_idx]
            uncertainty_estimates = assign_ue_scores_for_unlabeled_data(
                len(X_pool), subsample_indices, scores
            )
        else:
            query_idx = np.array(top_scores_idx)
            uncertainty_estimates = scores

        query = X_pool.select(query_idx)

        return query_idx, query, uncertainty_estimates
