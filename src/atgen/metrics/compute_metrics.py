from time import time
from typing import List, Dict, Union, Optional
import logging
import numpy as np
from omegaconf import DictConfig

from .factory import MetricsFactory, MetricsConfig, get_metric_requirements
from .base import MetricConfig

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
    generated_texts: List[str],
    reference_texts: Optional[List[Union[str, List[str]]]] = None,
    original_texts: Optional[List[str]] = None,
    config: Optional[Union[Dict, DictConfig, MetricsConfig]] = None,
    model=None,
    tokenizer=None,
    cache_dir: Optional[str] = None,
) -> Dict[str, float]:
    """
    Compute various metrics for generated texts using the new architecture.
    
    Args:
        generated_texts: List of generated texts to evaluate
        reference_texts: List of reference texts (ground truth) or list of lists for multiple references
        original_texts: List of source texts
        config: Configuration for metrics (dict, DictConfig, or MetricsConfig)
        model: Model instance (required for BigBenchHard)
        tokenizer: Tokenizer instance (required for BigBenchHard)
        cache_dir: Cache directory for storing intermediate results
    
    Returns:
        Dictionary with metric scores and timing information
    """
    start_total = time()
    
    predictions = generated_texts
    references = reference_texts
    
    # Handle cache_dir in the config
    if cache_dir is not None and isinstance(config, dict):
        config = dict(config)  # Make a copy
        config["cache_dir"] = cache_dir
    
    # Handle different config types
    if config is None:
        # Default configuration
        metrics_config = MetricsConfig(
            metrics=["bleu", "rouge1"],
            batch_size=32,
            device="cuda",
            aggregate=True
        )
    elif isinstance(config, dict):
        metrics_config = MetricsConfig(**config)
    elif isinstance(config, DictConfig):
        # Convert OmegaConf to dict then to MetricsConfig
        config_dict = dict(config)
        metrics_config = MetricsConfig(**config_dict)
    else:
        metrics_config = config
    
    # Validate inputs
    if not predictions:
        raise ValueError("predictions cannot be empty")
    
    # Get metric requirements
    requirements = get_metric_requirements()
    
    # Create metrics using factory
    metrics = MetricsFactory.create_metrics(metrics_config)
    
    if not metrics:
        log.warning("No metrics were successfully created")
        return {}
    
    results = {}
    timing_info = {}
    
    # Basic statistics
    results["word_length_gen"] = float(np.mean([len(text.split()) for text in predictions]))
    
    if original_texts:
        src_lengths = np.array([len(text.split()) for text in original_texts])
        gen_lengths = np.array([len(text.split()) for text in predictions])
        # Avoid division by zero
        src_lengths_safe = np.where(src_lengths > 0, src_lengths, 1)
        results["word_length_src_rel"] = float(np.mean(gen_lengths / src_lengths_safe))
    
    if references:
        if isinstance(references[0], list):
            ref_lengths = np.array([len(refs[0].split()) for refs in references])
        else:
            ref_lengths = np.array([len(text.split()) for text in references])
        
        gen_lengths = np.array([len(text.split()) for text in predictions])
        if isinstance(references[0], list):
            exact_matches = [
                any(pred == ref for ref in ref_list)
                for pred, ref_list in zip(predictions, references)
            ]
        else:
            exact_matches = [pred == ref for pred, ref in zip(predictions, references)]
        results["exact_match"] = float(np.mean(exact_matches))
        
        ref_lengths_safe = np.where(ref_lengths > 0, ref_lengths, 1)
        results["word_length_rel"] = float(np.mean(gen_lengths / ref_lengths_safe))
    
    for metric_name, metric in metrics.items():
        log.info(f"Computing {metric_name}...")
        start_time = time()
        
        try:
            req = requirements.get(metric_name, {})
            
            if req.get("requires_references", False) and references is None:
                log.warning(f"Skipping {metric_name}: requires references but none provided")
                continue
                
            if req.get("requires_original_texts", False) and original_texts is None:
                log.warning(f"Skipping {metric_name}: requires original texts but none provided")
                continue
            
            if metric_name in ["bigbench_hard", "big_bench_hard"]:
                if model is None or tokenizer is None:
                    log.warning(f"Skipping {metric_name}: requires model and tokenizer")
                    continue
                
                metric.set_model_and_tokenizer(model, tokenizer)
                metric_results = metric.calculate([], [], [])
            else:
                metric_results = metric.calculate(predictions, references, original_texts)
            
            for key, value in metric_results.items():
                results[key] = value
            
            timing_info[f"time_{metric_name}"] = time() - start_time
            log.info(f"Completed {metric_name} in {timing_info[f'time_{metric_name}']:.2f}s")
            
        except Exception as e:
            log.error(f"Error computing {metric_name}: {e}")
            timing_info[f"time_{metric_name}"] = time() - start_time
    
    # Add total timing
    timing_info["time_total"] = time() - start_total
    results.update(timing_info)
    
    log.info(f"Computed {len(metrics)} metrics in {timing_info['time_total']:.2f}s")
    return results


def compute_metrics_from_config(
    predictions: List[str],
    references: Optional[List[Union[str, List[str]]]] = None,
    original_texts: Optional[List[str]] = None,
    config_dict: Dict = None,
    model=None,
    tokenizer=None,
) -> Dict[str, float]:
    """
    Convenience function to compute metrics from a configuration dictionary.
    
    Args:
        predictions: List of generated texts
        references: List of reference texts  
        original_texts: List of source texts
        config_dict: Configuration dictionary
        model: Model instance (for BigBenchHard)
        tokenizer: Tokenizer instance (for BigBenchHard)
    
    Returns:
        Dictionary with metric scores
    """
    return compute_metrics(
        generated_texts=predictions,
        reference_texts=references,
        original_texts=original_texts,
        config=config_dict,
        model=model,
        tokenizer=tokenizer
    )


def get_default_config() -> MetricsConfig:
    """Get a default metrics configuration."""
    return MetricsConfig(
        metrics=["bleu", "rouge1"],
        batch_size=32,
        device="cuda",
        cache_dir="cache",
        aggregate=True
    )


def get_comprehensive_config() -> MetricsConfig:
    """Get a comprehensive metrics configuration with all metrics."""
    return MetricsConfig(
        metrics=[
            "bleu", "rouge1", "rouge2", "rougeL",
            "sentbert",
            "cola",
        ],
        batch_size=16,
        device="cuda",
        cache_dir="cache",
        aggregate=True
    )


def get_deepeval_config(api_key: str, model: str = "openai/gpt-4o-mini") -> MetricsConfig:
    return MetricsConfig(
        metrics=[
            "deepeval_answer_relevance",
            "deepeval_faithfulness", 
            "deepeval_summarization",
            "deepeval_prompt_alignment"
        ],
        batch_size=8,
        device="cpu",
        cache_dir="cache",
        aggregate=True,
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        model=model,
        threshold=0.5,
        async_mode=True,
        verbose_mode=False
    )
