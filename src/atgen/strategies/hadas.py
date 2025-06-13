from typing import Optional, Union
import numpy as np
from evaluate import EvaluationModule, load
from scipy.spatial.distance import jensenshannon
from omegaconf import DictConfig, ListConfig
import logging

from datasets import Dataset
import torch
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    GenerationMixin,
    PreTrainedTokenizerBase,
    RobertaForSequenceClassification,
)

from .unieval import SumEvaluator, convert_to_json
from .base_strategy import BaseStrategy
from ..utils.generate import generate
from ..utils.data import prepare_conversational_data, get_output_column_name_for_phase
from ..utils.constants import MESSAGES_COLUMN_NAME


log = logging.getLogger()


class HadasStrategy(BaseStrategy):
    def __init__(
        self,
        subsample_size: int | float = -1,
        cache_dir: str | None = None,
        inference_config: DictConfig = None,
        data_config: DictConfig = None,
        model_config: DictConfig = None,
    ):
        super().__init__(subsample_size)
        self.entailment_tokenizer = AutoTokenizer.from_pretrained(
            "roberta-base", padding="max_length", truncation=True, cache_dir=cache_dir
        )
        self.entailment_model = AutoModelForSequenceClassification.from_pretrained(
            "bunsenfeng/FactKB", num_labels=2, cache_dir=cache_dir
        )
        self.unieval = make_sumevaluator(cache_dir=cache_dir)
        self.bertscore = load("bertscore", cache_dir=cache_dir)

        self.model_config = model_config
        self.data_config = data_config
        self.inference_config = inference_config
        self.random_init = True

    def __call__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        unlabeled_pool: Dataset,
        labeled_pool: Dataset,
        num_to_label: int,
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
        return hadas(
            model,
            tokenizer,
            self.entailment_model,
            self.entailment_tokenizer,
            self.unieval,
            self.bertscore,
            unlabeled_pool=unlabeled_pool,
            labeled_pool=labeled_pool,
            input_column_name=self.data_config.input_column_name,
            output_column_name=self.data_config.train_output_column_name,
            num_to_label=num_to_label,
            inference_config=self.inference_config,
            model_config=self.model_config,
            data_config=self.data_config,
        )


def semantic_frame_score(
    entailment_model: RobertaForSequenceClassification,
    entailment_tokenizer: PreTrainedTokenizerBase,
    documents: list[str],
    summaries: list[str],
    batch_size: int,
) -> torch.Tensor:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    entailment_model = entailment_model.to(device)

    batch_results = []
    for i_start in range(0, len(documents), batch_size):
        inputs = entailment_tokenizer(
            list(
                zip(
                    documents[i_start : i_start + batch_size],
                    summaries[i_start : i_start + batch_size],
                )
            ),
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        logits = entailment_model(**{k: v.to(device) for k, v in inputs.items()}).logits
        with torch.no_grad():
            batch_results.append(torch.softmax(logits, dim=1)[:, -1].cpu())

    return torch.hstack(batch_results)


def discourse_score(
    unieval: SumEvaluator,
    documents: list[str],
    summaries: list[str],
) -> torch.Tensor:
    """
    Paper does not state which UniEval task is used for discourse score calculation.
    We assume that the "summarization" task is used, specifically, the "overall" score.
    We omit the "relevance" dimension, since it requires a reference summary list, which
    we do not have access to during active learning.
    """
    data = convert_to_json(src_list=documents, output_list=summaries)
    scores = [
        s["overall"]
        for s in unieval.evaluate(data, dims=["coherence", "consistency", "fluency"])
    ]
    return torch.tensor(scores, dtype=torch.float)


def content_verifiability_score(
    bertscore: EvaluationModule,
    documents: list[str],
    summaries: list[str],
) -> torch.Tensor:
    """
    Paper does not state specifically which BERTScore base model should be used.
    We assume "roberta-large".
    """
    results = bertscore.compute(
        predictions=summaries, references=documents, model_type="roberta-large"
    )
    return torch.tensor(results["precision"])


def hallucination_distribution(
    entailment_model: RobertaForSequenceClassification,
    entailment_tokenizer: PreTrainedTokenizerBase,
    unieval: SumEvaluator,
    bertscore: EvaluationModule,
    documents: list[str],
    summaries: list[str],
    batch_size: int = 64,
) -> torch.Tensor:
    h_sf = semantic_frame_score(
        entailment_model, entailment_tokenizer, documents, summaries, batch_size
    ).view(-1, 1)
    h_disc = discourse_score(unieval, documents, summaries).view(-1, 1)
    h_cv = content_verifiability_score(bertscore, documents, summaries).view(-1, 1)
    return torch.hstack([h_sf, h_disc, h_cv])


def make_sumevaluator(**kwargs):
    return SumEvaluator(**kwargs)


"""
https://arxiv.org/pdf/2404.01588.pdf
"""


def hadas(
    model: GenerationMixin,
    tokenizer: PreTrainedTokenizerBase,
    entailment_model: RobertaForSequenceClassification,  # e.g., https://huggingface.co/bunsenfeng/FactKB
    entailment_tokenizer: PreTrainedTokenizerBase,
    unieval: SumEvaluator,  # e.g., make_sumevaluator(cache_dir="./cache")
    bertscore: EvaluationModule,  # e.g, evaluate.load("bertscore", cache_dir="./cache")
    unlabeled_pool: Dataset,
    labeled_pool: Dataset,
    input_column_name: str,
    output_column_name: str,
    num_to_label: int,
    inference_config: DictConfig,
    w1: float = 0.33,
    w2: float = 0.33,
    w3: float = 0.33,
    lambda_: float = 0.33,
    **generation_kwargs,
) -> list[int]:
    log.info("Starting generating outputs for the unlabeled pool...")
    generations = generate(
        inference_config=inference_config,
        data=unlabeled_pool.select_columns([MESSAGES_COLUMN_NAME]),
        model=model,
        tokenizer=tokenizer,
        **generation_kwargs,
    )
    log.info("Done generating outputs for the unlabeled pool.")

    weights = torch.tensor([w1, w2, w3]).view(-1, 1)
    U_unlabeled = hallucination_distribution(
        entailment_model,
        entailment_tokenizer,
        unieval,
        bertscore,
        unlabeled_pool[input_column_name],
        generations,
        inference_config.batch_size,
    )
    h_halu = (U_unlabeled @ weights).flatten().numpy()

    U_labeled = hallucination_distribution(
        entailment_model=entailment_model,
        entailment_tokenizer=entailment_tokenizer,
        unieval=unieval,
        bertscore=bertscore,
        documents=labeled_pool[input_column_name],
        summaries=labeled_pool[output_column_name],
        batch_size=inference_config.batch_size,
    )
    h_div = np.array(
        [
            np.mean(
                [
                    jensenshannon(unlabeled_row, labeled_row) ** 2
                    for labeled_row in U_labeled
                ]
            )
            for unlabeled_row in U_unlabeled
        ]
    )

    scores = lambda_ * h_div + (1 - lambda_) * h_halu
    return np.argsort(-scores)[:num_to_label].tolist()
