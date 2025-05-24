from .base_strategy import Strategy

import numpy as np
from datasets import Dataset
from transformers import set_seed, PreTrainedModel


class BLEUVarStrategy(Strategy):
    def __init__(self):
        super().__init__()

    def __call__(
        self,
        model: PreTrainedModel,
        unlabeled_pool: Dataset,
        num_to_label: int,
        *args,
        **kwargs,
    ) -> list[int]:
        return bleuvar(model, unlabeled_pool, num_to_label)[0]


# https://arxiv.org/pdf/2006.08344.pdf
def bleuvar(
    model,
    X_pool: Dataset,
    n_instances: int,
    **kwargs,
):
    mc_iterations = kwargs.get("mc_iterations", 10)
    metric_name = kwargs.get("var_metric", "bleu")

    filtering_mode = kwargs.get("filtering_mode", None)
    bleuvar_threshold = kwargs.get("uncertainty_threshold", 1.0)
    uncertainty_mode = kwargs.get(
        "uncertainty_mode", "absolute"
    )  # "relative" or "absolute"
    seed = kwargs.get("seed", 42)

    if kwargs.get("subsample_size_mc_dropout", True):
        X_pool_subsample, subsample_indices = get_X_pool_subsample(
            X_pool, mc_iterations, model.seed
        )
    else:
        X_pool_subsample, subsample_indices = X_pool, np.arange(len(X_pool))

    generate_kwargs = dict(
        return_decoded_preds=True, do_sample=False, to_eval_mode=False
    )
    if kwargs.get("enable_dropout", False):
        if seed is not None:
            set_seed(seed)
        model.enable_dropout()  # model.model.train()
    else:
        model.model.eval()
        generate_kwargs["do_sample"] = True
        generate_kwargs["top_p"] = kwargs.get("generate_top_p", 0.95)

    summaries = []  # mc_iterations x len(X_pool_subsample) of str
    for _ in range(mc_iterations):
        generated_texts = model.generate(X_pool_subsample, **generate_kwargs)[
            "predictions"
        ]
        generated_texts = [
            text if len(text) > 0 else "CLS" for text in generated_texts
        ]  # fix empty texts
        summaries.append(generated_texts)

    # sacrebleu is normally more robust than bleu
    if metric_name == "bleu":
        bleu_vars = calculate_bleuvar_scores(summaries)[0]
    else:
        bleu_vars = calculate_pairwise_metric_score(
            summaries,
            metric_name=metric_name,
            cache_dir=model.cache_dir / "metrics",
            tokenizer=model.tokenizer,
        )

    if filtering_mode == "uncertainty":
        subsample_query_idx, bleu_vars = filter_by_uncertainty(
            bleu_vars, bleuvar_threshold, uncertainty_mode, n_instances
        )
    else:
        subsample_query_idx = np.argsort(-bleu_vars)[:n_instances]

    query = X_pool_subsample.select(subsample_query_idx)
    query_idx = subsample_indices[subsample_query_idx]

    uncertainty_estimates = assign_ue_scores_for_unlabeled_data(
        len(X_pool), subsample_indices, bleu_vars
    )

    return query_idx, query, uncertainty_estimates
