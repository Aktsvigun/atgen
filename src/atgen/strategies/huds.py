import numpy as np
from math import ceil
from omegaconf import DictConfig
from typing import Optional
from tqdm import tqdm

from datasets import Dataset
import torch
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    GenerationMixin,
    PreTrainedTokenizerBase,
)

from .base_strategy import Strategy
from ..utils.get_embeddings import get_embeddings
from ..utils.constants import MESSAGES_COLUMN_NAME
from ..utils.data.prepare_conversational_data import prepare_conversational_data


class HudsStrategy(Strategy):
    def __init__(
        self,
        unlabeled_pool: list[str],
        embeddings_model_checkpoint: str,
        subsample_size: int = 1_000,
        embeddings_batch_size: int = 64,
        model_is_causallm: bool = False,
        model_max_length: int = 1024,
        torch_dtype: str = "float16",
        cache_dir: str = None,
        data_config: DictConfig = None,
    ):
        super().__init__(subsample_size)
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.unlabeled_pool_embeddings = get_embeddings(
            texts=unlabeled_pool,
            model=embeddings_model_checkpoint,
            batch_size=embeddings_batch_size,
            model_is_causallm=model_is_causallm,
            model_max_length=model_max_length,
            torch_dtype=torch_dtype,
            cache_dir=cache_dir,
            device=device,
        )
        self.random_init = False
        self.data_config = data_config

    def __call__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        unlabeled_pool: Dataset,
        num_to_label: int,
        batch_size: int = 4,
        max_new_tokens: int = 20,
        few_shot_examples: Optional[Dataset] = None,
        *args,
        **kwargs,
    ) -> list[int]:
        unlabeled_pool = self._select_subsample_if_necessary(unlabeled_pool)
        if not MESSAGES_COLUMN_NAME in unlabeled_pool.column_names:
            unlabeled_pool = prepare_conversational_data(
                dataset=unlabeled_pool,
                data_config=self.data_config,
                split="test",
                few_shot_examples=few_shot_examples,
                model_name=model.name_or_path,
            )
        return huds(
            model=model,
            tokenizer=tokenizer,
            unlabeled_pool=unlabeled_pool,
            unlabeled_pool_embeddings=self.unlabeled_pool_embeddings,
            num_to_label=num_to_label,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
        )


def huds(
    model: GenerationMixin,
    tokenizer: PreTrainedTokenizerBase,
    unlabeled_pool: Dataset,
    unlabeled_pool_embeddings: np.ndarray,  # len(init_unlabeled_pool) x emb_size
    num_to_label: int,
    batch_size: int = 4,
    max_new_tokens: int = 20,
    n_strata: int = 10,
    lambda_: float = 0.5,
    **generation_kwargs,
) -> list[int]:
    # Assert `unlabeled_pool_embeddings` and `unlabeled_pool` are of the same length;
    # If not - take only embeddings with ids from unlabeled_pool
    if len(unlabeled_pool) != len(unlabeled_pool_embeddings):
        unlabeled_pool_embeddings = unlabeled_pool_embeddings[unlabeled_pool["id"]]
    nnlls = normalized_negative_log_likelihoods(
        model,
        tokenizer,
        unlabeled_pool[MESSAGES_COLUMN_NAME],
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        **generation_kwargs,
    )
    strata = stratify(nnlls, n_strata)
    distances = get_strata_centroid_distances(
        torch.from_numpy(unlabeled_pool_embeddings), strata
    )

    scores = lambda_ * distances + (1 - lambda_) * nnlls
    query_idx = np.argsort(-scores)[:num_to_label].tolist()
    return query_idx


def normalized_negative_log_likelihoods(
    model: GenerationMixin,
    tokenizer: PreTrainedTokenizerBase,
    unlabeled_pool_inputs: list[str],
    batch_size: int = 4,
    max_new_tokens: int = 20,
    **generation_kwargs,
) -> np.ndarray:

    nnlls = []
    num_batches = ceil(len(unlabeled_pool_inputs) / batch_size)
    for idx_beginning_batch in tqdm(
        range(num_batches), desc="HUDS: getting NNLLs of the unlabeled_pool..."
    ):
        batch_texts = unlabeled_pool_inputs[
            idx_beginning_batch * batch_size : (idx_beginning_batch + 1) * batch_size
        ]
        inputs = {
            "input_ids": tokenizer.apply_chat_template(
                batch_texts,
                batched=True,
                add_generation_prompt=True,
                padding="longest",
                truncation=True,
                return_tensors="pt",
            ),
        }
        # Add attention mask if it doesn't exist (1 for tokens, 0 for padding)
        inputs["attention_mask"] = (
            inputs["input_ids"] != tokenizer.pad_token_id
        ).long()

        batch_outputs = model.generate(
            **{k: v.to(model.device) for k, v in inputs.items()},
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True,
            output_scores=True,
            **generation_kwargs,
        )
        transition_scores = model.compute_transition_scores(
            batch_outputs.sequences, batch_outputs.scores, normalize_logits=True
        )

        batch_nnlls = []
        padding_counts = (batch_outputs.sequences == tokenizer.pad_token_id).sum(dim=-1)
        for i, seq_transition_scores in enumerate(transition_scores):
            if padding_counts[i] == 0:
                batch_nnlls.append(-torch.mean(seq_transition_scores).item())
            else:
                batch_nnlls.append(
                    -torch.mean(seq_transition_scores[: -padding_counts[i]]).item()
                )
        nnlls += batch_nnlls
    return np.array(nnlls)


def stratify(
    nnlls: np.ndarray,
    n_strata: int,
) -> list[list[int]]:
    scored_texts = [(score, i) for i, score in enumerate(nnlls)]
    scored_texts.sort()

    smin = scored_texts[0][0]
    smax = scored_texts[-1][0]
    r = smax - smin
    strata = []
    current_text = 0
    for i in range(n_strata):
        current_stratum = []
        left = smin + i / n_strata * r
        right = smin + (i + 1) / n_strata * r
        while (
            current_text < len(scored_texts)
            and left <= scored_texts[current_text][0] <= right
        ):
            current_stratum.append(scored_texts[current_text][1])
            current_text += 1

        if len(current_stratum) > 0:
            strata.append(current_stratum)

    return strata


def get_strata_centroid_distances(
    unlabeled_pool_embeddings: torch.Tensor, strata: list[list[int]]
) -> np.ndarray:
    distances = [0.0] * unlabeled_pool_embeddings.size(0)
    for stratum in strata:
        stratum_embeddings = unlabeled_pool_embeddings[stratum]
        centroid = torch.mean(
            stratum_embeddings, dim=0
        )  # Centroid for K-Means with 1 cluster
        for idx, emb in zip(stratum, stratum_embeddings):
            distances[idx] = (
                1 - torch.nn.functional.cosine_similarity(emb, centroid, dim=0).item()
            )

    return np.array(distances)


"""
https://arxiv.org/pdf/2403.09259.pdf
"""
