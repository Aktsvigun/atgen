from time import time
from typing import List, Dict, Union, Optional
import logging
import numpy as np
from omegaconf import DictConfiga

from .factory import MetricsFactory, MetricsConfig, get_metric_requirements
from .base import MetricConfig

logger = logging.getLogger(__name__)


def compute_metrics(
    predictions: List[str],
    references: Optional[List[Union[str, List[str]]]] = None,
    original_texts: Optional[List[str]] = None,
    metrics_config: Optional[Union[Dict, DictConfig, MetricsConfig]] = None,
    model=None,
    tokenizer=None,
) -> Dict[str, float]:
    """
    Compute various metrics for generated texts using the new architecture.
    
    Args:
        predictions: List of generated texts to evaluate
        references: List of reference texts (ground truth) or list of lists for multiple references
        original_texts: List of source texts
        metrics_config: Configuration for metrics (dict, DictConfig, or MetricsConfig)
        model: Model instance (required for BigBenchHard)
        tokenizer: Tokenizer instance (required for BigBenchHard)
    
    Returns:
        Dictionary with metric scores and timing information
    """
    start_total = time()
    
    # Handle different config types
    if metrics_config is None:
        # Default configuration
        config = MetricsConfig(
            metrics=["bleu", "rouge"],
            batch_size=32,
            device="cuda",
            aggregate=True
        )
    elif isinstance(metrics_config, dict):
        config = MetricsConfig(**metrics_config)
    elif isinstance(metrics_config, DictConfig):
        # Convert OmegaConf to dict then to MetricsConfig
        config_dict = dict(metrics_config)
        config = MetricsConfig(**config_dict)
    else:
        config = metrics_config
    
    # Validate inputs
    if not predictions:
        raise ValueError("predictions cannot be empty")
    
    # Get metric requirements
    requirements = get_metric_requirements()
    
    # Create metrics using factory
    metrics = MetricsFactory.create_metrics(config)
    
    if not metrics:
        logger.warning("No metrics were successfully created")
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
        logger.info(f"Computing {metric_name}...")
        start_time = time()
        
        try:
            req = requirements.get(metric_name, {})
            
            if req.get("requires_references", False) and references is None:
                logger.warning(f"Skipping {metric_name}: requires references but none provided")
                continue
                
            if req.get("requires_original_texts", False) and original_texts is None:
                logger.warning(f"Skipping {metric_name}: requires original texts but none provided")
                continue
            
            if metric_name in ["bigbench_hard", "big_bench_hard"]:
                if model is None or tokenizer is None:
                    logger.warning(f"Skipping {metric_name}: requires model and tokenizer")
                    continue
                
                metric.set_model_and_tokenizer(model, tokenizer)
                metric_results = metric.calculate([], [], [])
            else:
                metric_results = metric.calculate(predictions, references, original_texts)
            
            for key, value in metric_results.items():
                results[key] = value
            
            timing_info[f"time_{metric_name}"] = time() - start_time
            logger.info(f"Completed {metric_name} in {timing_info[f'time_{metric_name}']:.2f}s")
            
        except Exception as e:
            logger.error(f"Error computing {metric_name}: {e}")
            timing_info[f"time_{metric_name}"] = time() - start_time
    
    # Add total timing
    timing_info["time_total"] = time() - start_total
    results.update(timing_info)
    
    logger.info(f"Computed {len(metrics)} metrics in {timing_info['time_total']:.2f}s")
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
        predictions=predictions,
        references=references,
        original_texts=original_texts,
        metrics_config=config_dict,
        model=model,
        tokenizer=tokenizer
    )


def get_default_config() -> MetricsConfig:
    """Get a default metrics configuration."""
    return MetricsConfig(
        metrics=["bleu", "rouge"],
        batch_size=32,
        device="cuda",
        cache_dir="cache",
        aggregate=True
    )


def get_comprehensive_config() -> MetricsConfig:
    """Get a comprehensive metrics configuration with all metrics."""
    return MetricsConfig(
        metrics=[
            "bleu", "rouge",
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