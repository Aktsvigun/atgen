import string
from time import time
import logging
from typing import Literal, Optional
from omegaconf import DictConfig
import re

import numpy as np
from evaluate import load
from tqdm import tqdm

from .metrics import (
    pair_bleu,
    calculate_bart_score,
    calculate_alignscore,
    calculate_deepeval_metrics,
    is_bart_score_available,
    is_alignscore_available,
)
from .deepeval_supported_models_and_metrics import API_MODELS, DEEPEVAL_METRICS


log = logging.getLogger()


def compute_metrics(
    generated_texts,
    reference_texts,
    original_texts,
    task: Literal["summarization", "open-qa", "multi-choice-qa", "translation", "math"],
    config: DictConfig,
    cache_dir: str = "cache",
) -> dict[str, float]:
    """
    Compute various metrics for generated texts.

    Args:
        generated_texts: List of generated texts to evaluate
        reference_texts: List of reference texts (ground truth) or list of lists of reference texts
        original_texts: List of source texts
        task: Task type (summarization, open-qa, multi-choice-qa, translation)
        config: Configuration for evaluation
            - additional_metrics: List of additional metrics to use. Options include:
                - "bartscore": BARTScore metrics
                - "alignscore": AlignScore metrics
                - DeepEval metrics (requires API key):
                    - "deepeval_answer_relevance": Evaluates how well the output answers the input
                    - "deepeval_faithfulness": Evaluates factual consistency with the input
                    - "deepeval_summarization": Evaluates summarization quality
                    - "deepeval_prompt_alignment": Evaluates alignment with the expected output
            - provider: API key for the provider
            - api_key: Model identifier to use
            - model: Provider name (openai, anthropic, openrouter, or custom)
            - base_url: API base URL (if None, uses default for the provider)
            - deepeval_threshold: Threshold for DeepEval metrics (default: 0.5)
            - deepeval_include_reason: Include reason for evaluation score (default: False)
            - deepeval_strict_mode: Enforce binary metric score (default: False)
            - deepeval_async_mode: Enable concurrent execution (default: True)
            - deepeval_verbose_mode: Print intermediate steps (default: False)
            - deepeval_truths_extraction_limit: Maximum number of factual truths to extract (default: None)
    Returns:
        Dictionary with metric scores

    Note:
        The OpenRouterLLM class is also available for direct use with DeepEval metrics:

        ```python
        from atgen.metrics import OpenRouterLLM
        from deepeval.metrics import AnswerRelevanceMetric

        llm = OpenRouterLLM(
            api_key="your_openrouter_api_key",
            model="openai/gpt-4o-2024-11-20"
        )

        metric = AnswerRelevanceMetric(model=llm)
        ```
    """
    if task == "multi-choice-qa":
        metrics_to_calculate = ["exact_match"] + list(config.additional_metrics)
    elif task == "open-qa":
        metrics_to_calculate = ["exact_match"] + list(config.additional_metrics)
    elif task == "summarization":
        metrics_to_calculate = ["exact_match", "sacrebleu", "bleu", "rouge", "word_length"] + list(config.additional_metrics)
    elif task == "translation":
        metrics_to_calculate = ["exact_match", "sacrebleu", "bleu", "word_length"] + list(config.additional_metrics)
    elif task == "math":
        metrics_to_calculate = ["exact_match_math"] + list(config.additional_metrics)
    else:
        raise NotImplementedError(f"Task {task} not implemented")   

    if "sacrebleu" in metrics_to_calculate:
        sacrebleu = load("sacrebleu", cache_dir=cache_dir)
    if "rouge" in metrics_to_calculate:
        rouge = load("rouge", cache_dir=cache_dir)

    result = {}
    if "word_length" in metrics_to_calculate:
        result["word_length_gen"] = np.array(
            [len(text.split()) for text in generated_texts]
        )

    time_dict = {}

    # Metrics that use both the generated texts and the original texts and
    # those that do not require reference texts
    if "bartscore" in metrics_to_calculate and is_bart_score_available:
        log.info("Calculating BARTScore scores...")
        start_time = time()
        result.update(
            calculate_bart_score(
                preds=generated_texts,
                texts=original_texts,
                refs=reference_texts,
                batch_size=4,
                cache_dir=cache_dir,
            )
        )
        time_dict["time_bartscore"] = time() - start_time
    # Metrics that use both the generated texts and the reference texts
    if reference_texts is not None:
        # Exact match
        if "exact_match" in metrics_to_calculate:
            if isinstance(reference_texts[0], list):
                result["exact_match"] = np.array(
                    [
                        any(_preprocess_text(pred) == _preprocess_text(one_ref) for one_ref in ref)
                        for pred, ref in zip(generated_texts, reference_texts)
                    ]
                )
            else:
                result["exact_match"] = np.array(
                    [_preprocess_text(pred) == _preprocess_text(ref) for pred, ref in zip(generated_texts, reference_texts)]
                )
        if "exact_match_math" in metrics_to_calculate:
            # result["exact_match_math"] = np.array(
            #     [
            #         pred.split("Answer: ")[-1].lower() == ref.lower()
            #         for pred, ref in zip(generated_texts, reference_texts)
            #     ]
            # )
            result["exact_match_math"] = np.array(
                [
                    pred.split("#### ")[-1].lower() == ref.split("#### ")[-1].lower()
                    for pred, ref in zip(generated_texts, reference_texts)
                ]
            )
        if "bleu" in metrics_to_calculate:
            # BLEU
            start_time = time()
            result["bleu"] = np.array(
                [
                    pair_bleu(references=ref, prediction=pred)
                    for pred, ref in tqdm(zip(generated_texts, reference_texts))
                ]
            )
            time_dict["time_bleu"] = time() - start_time
        if "rouge" in metrics_to_calculate:
            # ROUGE
            start_time = time()
            result.update(
                rouge.compute(
                    predictions=generated_texts,
                    references=reference_texts,
                    use_stemmer=True,
                )
            )
            time_dict["time_rouge"] = time() - start_time
        if "sacrebleu" in metrics_to_calculate:
            # Sacrebleu
            start_time = time()
            if not isinstance(reference_texts[0], list):
                sacrebleu_references = [[ref] for ref in reference_texts]
                sacrebleu_result = sacrebleu.compute(
                    predictions=generated_texts, references=sacrebleu_references
                )
                result["sacrebleu"] = sacrebleu_result.pop("score")
            else:
                sacrebleu_scores = []
                for pred, ref in zip(generated_texts, reference_texts):
                    sacrebleu_result = sacrebleu.compute(
                        predictions=[pred], references=[ref]
                    )
                    sacrebleu_scores.append(sacrebleu_result.pop("score"))
                result["sacrebleu"] = sacrebleu_scores

            time_dict["time_sacrebleu"] = time() - start_time
        if "word_length" in metrics_to_calculate:
            # Lengths
            if isinstance(reference_texts[0], list):
                ref_word_lengths = np.array(
                    [
                        np.mean([len(text.split()) for text in ref])
                        for ref in reference_texts
                    ]
                )
            else:
                ref_word_lengths = np.array([len(ref.split()) for ref in reference_texts])
            # Avoid division by zero
            ref_word_lengths_safe = np.where(ref_word_lengths > 0, ref_word_lengths, 1)
            result["word_length_rel"] = result["word_length_gen"] / ref_word_lengths_safe

        # AlignScore
        if "alignscore" in metrics_to_calculate and is_alignscore_available:
            log.info("Calculating AlignScore scores...")
            start_time = time()
            alignscores = calculate_alignscore(
                generated_texts, reference_texts, original_texts
            )
            if alignscores is not None:
                result.update(alignscores)
            time_dict["time_alignscore"] = time() - start_time

        # DeepEval metrics
        deepeval_metrics_to_calculate = [
            metric for metric in DEEPEVAL_METRICS if metric in config.additional_metrics
        ]

        if deepeval_metrics_to_calculate:
            if isinstance(reference_texts[0], list):
                log.error("DeepEval does not support multiple references. Skipping...")
            else:
                # Validate OpenRouter model - only warn if not in predefined list, but still use it
                provider = config["provider"]
                if config.model not in API_MODELS.get(provider):
                    log.warning(
                        f"Using custom model: {config.model}. "
                        + (
                            f"Available models: {API_MODELS[provider]}"
                            if provider in API_MODELS
                            else ""
                        )
                    )
                log.info(
                    f"Calculating DeepEval metrics: {', '.join(deepeval_metrics_to_calculate)}..."
                )
                start_time = time()
                result.update(
                    calculate_deepeval_metrics(
                        predictions=generated_texts,
                        references=reference_texts,
                        original_texts=original_texts,
                        metrics_to_calculate=deepeval_metrics_to_calculate,
                        base_url=config.base_url,
                        api_key=config.api_key,
                        model=config.model,
                        threshold=config.deepeval_threshold,
                        include_reason=config.deepeval_include_reason,
                        strict_mode=config.deepeval_strict_mode,
                        async_mode=config.deepeval_async_mode,
                        verbose_mode=config.deepeval_verbose_mode,
                        truths_extraction_limit=config.deepeval_truths_extraction_limit,
                    )
                )
                time_dict["time_deepeval"] = time() - start_time

    for key, value in result.items():
        if isinstance(value, np.ndarray):
            result[key] = float(np.mean(value))
        elif isinstance(value, (int, float)):
            # Ensure numerical values are converted to float
            result[key] = float(value)
        # Make sure non-numerical values that aren't reasons are preserved
        elif not key.endswith("_reasons") and not "_reason" in key.lower():
            continue

    # Filter out reason fields from the final aggregated results - more robust filtering
    result = {
        key: value
        for key, value in sorted(result.items())
        if not key.endswith("_reasons")
        and not "_reason" in key.lower()
        and isinstance(value, (int, float))  # Ensure we only keep numerical metrics
    }

    return result

def _preprocess_text(text: str, do_lowercase: bool = True, do_remove_punctuation: bool = True, do_remove_extra_spaces: bool = True, do_remove_stopwords: bool = False, stopwords: Optional[list[str]] = None) -> str:
        # Convert to lowercase
        if do_lowercase:
            text = text.lower()
        
        # Remove punctuation
        if do_remove_punctuation:
            # Keep hyphens within words, remove other punctuation
            text = re.sub(r'(?<!\w)-|-(?!\w)', ' ', text)  # Replace standalone hyphens
            translator = str.maketrans('', '', string.punctuation.replace('-', ''))
            text = text.translate(translator)
            text = re.sub(r'(?<!\w)-(?!\w)', '', text)  # Remove remaining standalone hyphens
        
        # Normalize whitespace
        if do_remove_extra_spaces:
            text = ' '.join(text.split())
        
        # Remove stopwords
        if do_remove_stopwords:
            if stopwords is None:
                import nltk
                nltk.download('stopwords')
                stopwords = nltk.corpus.stopwords.words('english')
            words = text.split()
            words = [w for w in words if w not in stopwords]
            text = ' '.join(words)
        
        return text.strip()
