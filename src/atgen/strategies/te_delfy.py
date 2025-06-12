from collections import Counter
from omegaconf import DictConfig
import logging
from math import ceil
from tqdm import tqdm

import numpy as np
import torch
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from torch.distributions import Categorical
from datasets import Dataset
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    GenerationMixin,
    PreTrainedTokenizerBase,
)

from .base_strategy import BaseStrategy


log = logging.getLogger()


class TeDelfyStrategy(BaseStrategy):
    def __init__(self, subsample_size: int = -1, inference_config: DictConfig = None):
        super().__init__(subsample_size)
        self.random_init = True
        self.inference_config = inference_config

    def __call__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        unlabeled_pool: Dataset,
        labeled_pool: Dataset,
        input_column_name: str,
        num_to_label: int,
        *args,
        **kwargs,
    ) -> list[int]:
        unlabeled_pool = self._select_subsample_if_necessary(unlabeled_pool)
        return te_delfy(
            model,
            tokenizer,
            unlabeled_pool,
            labeled_pool,
            input_column_name,
            num_to_label,
            config=self.inference_config,
        )


"""
https://aclanthology.org/2020.findings-emnlp.162.pdf
"""


def te_delfy(
    model: GenerationMixin,
    tokenizer: PreTrainedTokenizerBase,
    unlabeled_pool: Dataset,
    labeled_pool: Dataset,
    input_column_name: str,
    num_to_label: int,
    config: DictConfig,
    lambda1: float = 1.0,
    lambda2: float = 1.0,
    **generation_kwargs,
) -> list[int]:
    preprocessed_unlabeled_pool = preprocess_delfy(unlabeled_pool[input_column_name])
    preprocessed_labeled_pool = preprocess_delfy(labeled_pool[input_column_name])

    ranks = token_entropy_ranks(
        model,
        tokenizer,
        unlabeled_pool[MESSAGES_COLUMN_NAME],
        config,
        **generation_kwargs,
    ) + delfy_ranks(
        preprocessed_unlabeled_pool, preprocessed_labeled_pool, lambda1, lambda2
    )
    return np.argsort(ranks)[:num_to_label].tolist()


def token_entropy_ranks(
    model: GenerationMixin,
    tokenizer: PreTrainedTokenizerBase,
    unlabeled_pool_inputs: list[str],
    config: DictConfig,
    **generation_kwargs,
) -> np.ndarray:
    token_entropies = []

    batch_size = config.batch_size
    num_batches = ceil(len(unlabeled_pool_inputs) / batch_size)
    log.info("Starting generating outputs for the unlabeled pool...")

    for idx_beginning_batch in tqdm(
        range(num_batches), desc="HUDS: getting NNLLs of the unlabeled_pool..."
    ):
        batch_texts = unlabeled_pool_inputs[
            idx_beginning_batch * batch_size : (idx_beginning_batch + 1) * batch_size
        ]
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        batch_outputs = model.generate(
            **{k: v.to(model.device) for k, v in inputs.items()},
            max_new_tokens=config.max_new_tokens,
            return_dict_in_generate=True,
            output_scores=True,
            **generation_kwargs,
        )
        log.info("Done generating outputs for the unlabeled pool.")

        scores = torch.stack(batch_outputs.scores).transpose(0, 1)
        entropies = Categorical(logits=scores).entropy()

        batch_token_entropies = []
        padding_counts = (batch_outputs.sequences == tokenizer.pad_token_id).sum(dim=-1)
        for i, seq_entropies in enumerate(entropies):
            if padding_counts[i] == 0:
                batch_token_entropies.append(torch.mean(seq_entropies).item())
            else:
                batch_token_entropies.append(
                    torch.mean(seq_entropies[: -padding_counts[i]]).item()
                )

        token_entropies += batch_token_entropies
    return np.argsort(np.argsort(-np.array(token_entropies)))


def preprocess_delfy(pool: list[str]) -> list[list[str]]:
    """
    Paper does not state whether preprocessing is necessary
    We apply the following:
    1. Make text lowercase
    2. Split using RegexpTokenizer from nltk with r"\\w+"
    3. Apply PorterStemmer from nltk to each word
    """
    tokenizer = RegexpTokenizer(r"\w+")
    stemmer = PorterStemmer()
    return [
        [stemmer.stem(word) for word in tokenizer.tokenize(text.lower())]
        for text in pool
    ]


def delfy_C(pool: list[list[str]]) -> dict[str, int]:
    C = Counter()
    for words in pool:
        C.update(words)
    return C


def lf_score(
    text: list[str],
    unlabeled_F: dict[str, float],
    labeled_C: dict[str, int],
    lambda1: float,
) -> float:
    return np.mean(
        [unlabeled_F[word] * np.exp(-lambda1 * labeled_C[word]) for word in text]
    )


def delfy_score(
    text: list[str],
    unlabeled_F: dict[str, float],
    labeled_C: dict[str, int],
    U_hat_C: dict[str, int],
    lambda1: float,
    lambda2: float,
) -> float:
    return np.mean(
        [
            unlabeled_F[word]
            * np.exp(-lambda1 * labeled_C[word])
            * np.exp(-lambda2 * U_hat_C[word])
            for word in text
        ]
    )


def delfy_ranks(
    unlabeled_pool: list[list[str]],
    labeled_pool: list[list[str]],
    lambda1: float,
    lambda2: float,
) -> np.ndarray:
    unlabeled_C = delfy_C(unlabeled_pool)
    words, counts = list(unlabeled_C.keys()), np.array(list(unlabeled_C.values()))
    Gvals = np.log1p(counts)
    Fvals = Gvals / np.sum(Gvals)
    unlabeled_F = {word: F for word, F in zip(words, Fvals)}

    labeled_C = delfy_C(labeled_pool)
    neg_lfs = [
        -lf_score(text, unlabeled_F, labeled_C, lambda1) for text in unlabeled_pool
    ]

    U_hat_C = Counter()

    delfys = []
    for idx in np.argsort(neg_lfs):
        text = unlabeled_pool[idx]
        delfys.append(
            (idx, delfy_score(text, unlabeled_F, labeled_C, U_hat_C, lambda1, lambda2))
        )
        U_hat_C.update(text)

    delfys.sort()  # put scores in original order based on the indices
    delfys = [x[1] for x in delfys]
    return np.argsort(np.argsort(-np.array(delfys)))
