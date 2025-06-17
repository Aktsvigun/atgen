from math import ceil
import os
import sys
from openai import OpenAI, AsyncOpenAI
from typing import Union
from urllib.request import urlretrieve
import logging
from pathlib import Path
import nltk
import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset
from nltk import ngrams
from nltk.stem import porter
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.translate.bleu_score import corpus_bleu
from rouge_score import tokenize
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoModel,
    DataCollatorWithPadding,
)

log = logging.getLogger(__name__)

try:
    from .bart_score import BARTScorer

    is_bart_score_available = True
except ImportError:
    log.warning(
        "BARTScorer not found, please install it (see `install.sh`). Skipping the BARTScore metric."
    )
    is_bart_score_available = False

try:
    from alignscore import AlignScore

    is_alignscore_available = True
except ImportError:
    log.warning(
        "AlignScore not found, please install it (see `install.sh`). Skipping the AlignScore metric."
    )
    is_alignscore_available = False

from deepeval import evaluate
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    SummarizationMetric,
    PromptAlignmentMetric,
)


ALIGNSCORE_CHECKPOINT_PATH = os.getenv(
    "ALIGNSCORE_CHECKPOINT_PATH",
    # Going up 3 levels from metrics.py: src/atgen/metrics -> repository root
    os.path.join(
        Path(__file__).parents[3],
        "cache/AlignScore-base.ckpt",
    ),
)


def decode(eval_preds, tokenizer):
    predictions, labels, *inputs = eval_preds
    predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [label.strip() for label in decoded_labels]

    if len(inputs) > 0:
        input_ids = inputs[0]
        input_ids = np.where(input_ids != -100, input_ids, tokenizer.pad_token_id)
        decoded_texts = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        decoded_texts = [text.strip() for text in decoded_texts]
        return decoded_preds, decoded_labels, decoded_texts

    return decoded_preds, decoded_labels


def smoothing_function(p_n, references, hypothesis, hyp_len):
    """
    Smooth-BLEU (BLEUS) as proposed in the paper:
    Chin-Yew Lin, Franz Josef Och. ORANGE: a method for evaluating automatic
    evaluation metrics for machine translation. COLING 2004.
    """
    smoothed_p_n = []
    for i, p_i in enumerate(p_n, start=1):
        # Smoothing is not applied for unigrams
        if i > 1:
            # If hypothesis length is lower than the current order, its value equals (0 + 1) / (0 + 1) = 0
            if hyp_len < i:
                assert p_i.denominator == 1
                smoothed_p_n.append(1)
            # Otherwise apply smoothing
            else:
                smoothed_p_i = (p_i.numerator + 1) / (p_i.denominator + 1)
                smoothed_p_n.append(smoothed_p_i)
        else:
            smoothed_p_n.append(p_i)
    return smoothed_p_n


def pair_bleu(references: list[str] | str, prediction: str):
    """
    Compute the bleu score between two given texts.
    A smoothing function is used to avoid zero scores when
    there are no common higher order n-grams between the
    texts.
    """
    if isinstance(references, str):
        tok_ref = [[word_tokenize(references)]]
    else:
        tok_ref = [[word_tokenize(ref) for ref in references]]
    tok_pred = [word_tokenize(prediction)]
    try:
        return corpus_bleu(tok_ref, tok_pred, smoothing_function=smoothing_function)
    except (KeyError, ZeroDivisionError):
        return 0.0


def calculate_bart_score(
    preds,
    refs=None,
    texts=None,
    scorer=None,
    batch_size=4,
    aggregate=True,
    cache_dir: str = "cache",
):
    if not is_bart_score_available:
        return None
    if scorer is None:
        scorer = BARTScorer(cache_dir=cache_dir)
    scores = {}
    if texts is not None:
        scores["BARTScore-sh"] = np.array(
            scorer.score(texts, preds, batch_size=batch_size)
        )
    if refs is not None:
        # scores["BARTScore-rh"] = np.array(scorer.score(refs, preds, batch_size=batch_size))
        if isinstance(refs[0], list):
            scores_hr = []
            for ref, pred in zip(refs, preds):
                inst_pred = [pred for _ in range(len(ref))]
                # Take a maximum within the observation similar to ROUGE
                inst_score_hr = max(scorer.score(inst_pred, ref, batch_size=batch_size))
                scores_hr.append(inst_score_hr)
            scores["BARTScore-hr"] = np.array(scores_hr)
        else:
            scores["BARTScore-hr"] = np.array(
                scorer.score(preds, refs, batch_size=batch_size)
            )
        # scores["BARTScore-fa"] = (scores["BARTScore-rh"] + scores["BARTScore-hr"]) / 2

    if aggregate:
        scores = {key: np.mean(value) for key, value in scores.items()}
    return scores


def calculate_abstractiveness_scores(
    predictions, texts, references=None, aggregate: bool = True
):
    stemmer = porter.PorterStemmer()
    tokenized_preds = [tokenize.tokenize(x, stemmer) for x in predictions]
    tokenized_texts = [tokenize.tokenize(x, stemmer) for x in texts]
    if references is not None:
        tokenized_refs = [tokenize.tokenize(x, stemmer) for x in references]
    else:
        tokenized_refs = tokenized_preds

    result = {}
    for use_modified in [False, True]:
        for n in range(1, 5):
            pred_ngram_overlaps = []
            label_ngram_overlaps = []
            for pred, label, text in zip(
                tokenized_preds, tokenized_refs, tokenized_texts
            ):
                pred_pair_ngram_overlap = calculate_ngram_overlap(
                    pred, text, n, use_modified
                )
                pred_ngram_overlaps.append(pred_pair_ngram_overlap)
                if references is not None:
                    label_pair_ngram_overlap = calculate_ngram_overlap(
                        label, text, n, use_modified
                    )
                    label_ngram_overlaps.append(label_pair_ngram_overlap)
            key = f"ngram_overlap_{n}" if use_modified else f"novel_ngrams_{n}"

            pred_ngram_overlaps = np.array(pred_ngram_overlaps)
            cond_abs = ~np.isnan(pred_ngram_overlaps)
            result[key + "_abs"] = pred_ngram_overlaps[cond_abs]

            if references is not None:
                label_ngram_overlaps = np.array(label_ngram_overlaps)
                cond_rel = cond_abs & ~np.isnan(label_ngram_overlaps)
                result[key + "_rel"] = (
                    pred_ngram_overlaps[cond_rel] / label_ngram_overlaps[cond_rel]
                )

    if aggregate:
        for key, value in result.items():
            result[key] = np.mean(value)

    return result


def calculate_ngram_overlap(summary, text, n=1, use_modified=True):
    summary_ngrams = list(ngrams(summary, n))
    text_ngrams = list(ngrams(text, n))

    if len(summary_ngrams) > 0:
        ngrams_intersection = set(summary_ngrams).intersection(set(text_ngrams))
        if use_modified:
            word_is_part_of_ngram_copied = [
                any((x in ngram for ngram in ngrams_intersection)) for x in summary
            ]
            return 1 - sum(word_is_part_of_ngram_copied) / len(
                word_is_part_of_ngram_copied
            )
        else:
            return sum([x not in ngrams_intersection for x in summary_ngrams]) / len(
                summary_ngrams
            )
    return np.nan


class SentBert:
    def __init__(
        self,
        checkpoint: str = "sentence-transformers/all-mpnet-base-v2",
        device: str = "cuda",
        cache_dir: str = "cache",
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, cache_dir=cache_dir)
        self.model = AutoModel.from_pretrained(checkpoint, cache_dir=cache_dir).to(
            device
        )
        self.device = device

    def __call__(
        self, source_texts: list[str], ref_texts: list[str], batch_size: int = 32
    ) -> np.ndarray:
        assert len(source_texts) == len(ref_texts)
        # Make batch_size an even number
        if batch_size % 2 == 0:
            batch_size -= 1
        half_batch_size = batch_size // 2
        n_texts = len(source_texts)
        scores = np.empty(n_texts, dtype=np.float32)
        start = 0
        end = 0

        while end < n_texts:
            end += half_batch_size
            batch_idx = slice(start, end)
            # Tokenize sentences
            encoded_input = self.tokenizer(
                source_texts[batch_idx] + ref_texts[batch_idx],
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            encoded_input = {
                key: value.to(self.device) for key, value in encoded_input.items()
            }
            # Calculate the probability of belonging to the positive class
            model_output = self.model(**encoded_input)
            # Perform pooling
            sent_embs = self.mean_pooling(model_output, encoded_input["attention_mask"])
            # Normalize embeddings
            sent_embs = F.normalize(sent_embs, p=2, dim=1)
            n_source_embs = len(sent_embs) // 2
            scores[batch_idx] = (
                (sent_embs[:n_source_embs] * sent_embs[n_source_embs:])
                .sum(-1)
                .cpu()
                .detach()
                .numpy()
            )
            start = end

        return scores

    @staticmethod
    def mean_pooling(model_output, attention_mask):
        """
        Mean Pooling - Take attention mask into account for correct averaging
        """
        token_embeddings = model_output[
            0
        ]  # First element of model_output contains all token embeddings
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )


def calculate_alignscore(
    predictions: list[str],
    references: Union[list[str], list[list[str]]],
    original_texts: list[str],
    batch_size: int = 32,
    device: str = "cuda",
    cache_dir: str = "cache",
):
    if not is_alignscore_available:
        return None
    if isinstance(references[0], list):
        log.error("AlignScore does not support multiple references. Skipping...")
        return None
    if not os.path.exists(ALIGNSCORE_CHECKPOINT_PATH):
        urlretrieve(
            "https://huggingface.co/yzha/AlignScore/resolve/main/AlignScore-base.ckpt",
            ALIGNSCORE_CHECKPOINT_PATH,
        )

    scorer = AlignScore(
        model="roberta-base",
        batch_size=batch_size,
        device=device,
        ckpt_path=ALIGNSCORE_CHECKPOINT_PATH,
        evaluation_mode="nli_sp",
    )
    # Fix: alignscore outputs an error if a text is empty, so we need to add some content to such texts
    original_texts = [text if text else " " for text in original_texts]
    predictions = [text if text else " " for text in predictions]
    references = [text if text else " " for text in references]

    scores_ref = scorer.score(contexts=original_texts, claims=predictions)
    if isinstance(references[0], list):
        scores_baseline = []
        for orig_text, refs in zip(original_texts, references):
            inst_baseline_scores = scorer.score(
                contexts=[orig_text] * len(refs), claims=refs
            )
            scores_baseline.append(max(inst_baseline_scores))
    else:
        scores_baseline = scorer.score(contexts=original_texts, claims=references)
    scores_rel = np.array(scores_ref) / np.array(scores_baseline)
    return {"alignscore": scores_ref, "alignscore_rel": scores_rel}


class EvaluationLLM(DeepEvalBaseLLM):
    """
    Custom Evaluation LLM implementation for DeepEval.

    This class implements the DeepEvalBaseLLM interface to allow using
    custom models with DeepEval metrics.
    """

    def __init__(
        self,
        api_key=None,
        model="openai/gpt-4o-2024-11-20",
        base_url="https://openrouter.ai/api/v1",
    ):
        """
        Initialize the Evaluation LLM.

        Args:
            api_key: Evaluation API key
            model: Model identifier (e.g., "openai/gpt-4o-2024-11-20")
            base_url: Evaluation API base URL
        """
        self.api_key = api_key

        self.model_name = model
        self.base_url = base_url
        self.client = None
        self.async_client = None
        self.OpenAI = OpenAI
        self.AsyncOpenAI = AsyncOpenAI

    def load_model(self):
        """Load and return the client."""
        if self.client is None:
            self.client = self.OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
            )
        return self.client

    def load_async_model(self):
        """Load and return the async client."""
        if self.async_client is None:
            self.async_client = self.AsyncOpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
            )
        return self.async_client

    def generate(self, prompt: str) -> str:
        """
        Generate a response from the evaluation model.

        Args:
            prompt: The prompt to send to the model

        Returns:
            The model's response as a string
        """
        client = self.load_model()
        response = client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content

    async def a_generate(self, prompt: str) -> str:
        """
        Asynchronously generate a response from the evaluation model.

        Args:
            prompt: The prompt to send to the model

        Returns:
            The model's response as a string
        """
        # Use the async client for async operations
        client = self.load_async_model()
        response = await client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content

    def get_model_name(self):
        """Return the name of the model."""
        return f"EvaluationLLM: {self.model_name}"


def calculate_deepeval_metrics(
    predictions,
    references,
    original_texts,
    metrics_to_calculate=None,
    base_url: str = "https://openrouter.ai/api/v1",
    api_key: str = None,
    model="openai/gpt-4o-2024-11-20",
    threshold=0.5,
    include_reason=False,
    strict_mode=False,
    async_mode=True,
    verbose_mode=False,
    truths_extraction_limit=None,
):
    """
    Calculate DeepEval metrics using EvaluationLLM.

    Args:
        predictions: list of generated texts
        references: list of reference texts
        original_texts: list of source texts
        metrics_to_calculate: list of metrics to calculate. Options:
            ["deepeval_answer_relevance", "deepeval_faithfulness", "deepeval_summarization", "deepeval_prompt_alignment"]
        api_key: Evaluation API key
        base_url: Evaluation API base URL
        model: Evaluation model to use
        threshold: Threshold for metrics (default: 0.5)
        include_reason: Include reason for evaluation score (default: False)
        strict_mode: Enforce binary metric score (1 for perfection, 0 otherwise) (default: False)
        async_mode: Enable concurrent execution (default: True)
        verbose_mode: Print intermediate steps (default: False)
        truths_extraction_limit: Maximum number of factual truths to extract (default: None)

    Returns:
        dictionary with metric scores
    """

    if not metrics_to_calculate:
        metrics_to_calculate = [
            "deepeval_answer_relevance",
            "deepeval_faithfulness",
            "deepeval_summarization",
            "deepeval_prompt_alignment",
        ]

    # Create EvaluationLLM instance
    llm = EvaluationLLM(
        base_url=base_url,
        api_key=api_key,
        model=model,
    )

    results = {}

    # Create metrics based on selected options
    metrics = []
    metric_name_mapping = {}  # Maps metric class name to the deepeval metric name

    # Dictionary to store test cases for each metric
    metric_test_cases = {}

    if "deepeval_answer_relevance" in metrics_to_calculate:
        metric = AnswerRelevancyMetric(
            threshold=threshold,
            model=llm,
            include_reason=include_reason,
            strict_mode=strict_mode,
            async_mode=async_mode,
        )
        metrics.append(metric)
        metric_name_mapping[metric.__class__.__name__] = "deepeval_answer_relevance"

        # Create specific test cases for AnswerRelevancy metric
        answer_relevance_test_cases = []
        for i, (pred, src) in enumerate(zip(predictions, original_texts)):
            test_case = LLMTestCase(
                input=src,
                actual_output=pred,
            )
            answer_relevance_test_cases.append(test_case)
        metric_test_cases[metric.__class__.__name__] = answer_relevance_test_cases

    if "deepeval_faithfulness" in metrics_to_calculate:
        metric = FaithfulnessMetric(
            threshold=threshold,
            model=llm,
            include_reason=include_reason,
            strict_mode=strict_mode,
            async_mode=async_mode,
            truths_extraction_limit=truths_extraction_limit,
        )
        metrics.append(metric)
        metric_name_mapping[metric.__class__.__name__] = "deepeval_faithfulness"

        # Create specific test cases for Faithfulness metric
        faithfulness_test_cases = []
        for i, (pred, src) in enumerate(zip(predictions, original_texts)):
            test_case = LLMTestCase(
                input=src,
                actual_output=pred,
                retrieval_context=[src],
            )
            faithfulness_test_cases.append(test_case)
        metric_test_cases[metric.__class__.__name__] = faithfulness_test_cases

    if "deepeval_summarization" in metrics_to_calculate:
        metric = SummarizationMetric(
            threshold=threshold,
            model=llm,
            include_reason=include_reason,
            strict_mode=strict_mode,
            async_mode=async_mode,
        )
        metrics.append(metric)
        metric_name_mapping[metric.__class__.__name__] = "deepeval_summarization"

        # Create specific test cases for Summarization metric
        summarization_test_cases = []
        for i, (pred, src) in enumerate(zip(predictions, original_texts)):
            test_case = LLMTestCase(
                input=src,
                actual_output=pred,
            )
            summarization_test_cases.append(test_case)
        metric_test_cases[metric.__class__.__name__] = summarization_test_cases

    if "deepeval_prompt_alignment" in metrics_to_calculate:
        metric = PromptAlignmentMetric(
            threshold=threshold,
            model=llm,
            prompt_instructions=["Do what you are told to do in the prompt"],
            include_reason=include_reason,
            strict_mode=strict_mode,
            async_mode=async_mode,
        )
        metrics.append(metric)
        metric_name_mapping[metric.__class__.__name__] = "deepeval_prompt_alignment"

        # Create specific test cases for PromptAlignment metric
        prompt_alignment_test_cases = []
        for i, (pred, ref, src) in enumerate(
            zip(predictions, references, original_texts)
        ):
            test_case = LLMTestCase(
                input=src,
                actual_output=pred,
                expected_output=ref,
            )
            prompt_alignment_test_cases.append(test_case)
        metric_test_cases[metric.__class__.__name__] = prompt_alignment_test_cases

    # Run evaluation for each metric separately
    for metric in metrics:
        metric_class_name = metric.__class__.__name__
        test_cases = metric_test_cases.get(metric_class_name, [])

        if test_cases:
            # Disable printing to console during evaluation if not verbose
            original_stdout = sys.stdout
            if not verbose_mode:
                sys.stdout = open(os.devnull, "w")

            try:
                # Run evaluation for this specific metric
                evaluation_results = evaluate(
                    test_cases=test_cases,
                    metrics=[metric],
                    run_async=async_mode,
                )

                deepeval_metric_name = metric_name_mapping.get(metric_class_name)
                scores = []
                reasons = []

                # Process results for this metric
                for result in evaluation_results.test_results:
                    scores.append(1 if result.success else 0)

                # Calculate average score
                if scores:
                    results[deepeval_metric_name] = np.mean(scores)

            finally:
                # Restore stdout
                if not verbose_mode:
                    sys.stdout.close()
                    sys.stdout = original_stdout
    print("================================================")
    print("Results:")
    print(results)
    print("================================================")

    return results
