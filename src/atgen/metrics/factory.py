from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, field
import logging

from .base import BaseMetric, MetricConfig
from .lexical import BleuMetric, RougeMetric
from .semantic import BartScoreMetric, AlignScoreMetric, SentBertMetric
from .linguistic import ColaMetric
from .llm_based import (
    DeepEvalAnswerRelevancyMetric,
    DeepEvalFaithfulnessMetric,
    DeepEvalSummarizationMetric,
    DeepEvalPromptAlignmentMetric,
    BigBenchHardMetric,
)

logger = logging.getLogger(__name__)


@dataclass
class MetricsConfig:
    
    batch_size: int = 32
    device: str = "cuda"
    cache_dir: str = "cache"
    aggregate: bool = True
    
    metrics: List[str] = field(default_factory=list)
    
    additional_metrics: List[str] = field(default_factory=list)
    
    checkpoint: Optional[str] = None
    model_name: Optional[str] = None
    
    provider: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model: Optional[str] = None
    
    threshold: float = 0.5
    include_reason: bool = False
    strict_mode: bool = False
    async_mode: bool = True
    verbose_mode: bool = False
    truths_extraction_limit: Optional[int] = None
    
    deepeval_threshold: Optional[float] = None
    deepeval_include_reason: Optional[bool] = None
    deepeval_strict_mode: Optional[bool] = None
    deepeval_async_mode: Optional[bool] = None
    deepeval_verbose_mode: Optional[bool] = None
    deepeval_truths_extraction_limit: Optional[int] = None
    
    benchmark_params: Dict[str, Any] = field(default_factory=dict)
    generation_params: Dict[str, Any] = field(default_factory=dict)
    
    custom_params: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-initialization to handle legacy parameter mappings."""
        # Merge additional_metrics into metrics for backward compatibility
        if self.additional_metrics:
            self.metrics.extend(self.additional_metrics)
            # Remove duplicates while preserving order
            seen = set()
            self.metrics = [x for x in self.metrics if not (x in seen or seen.add(x))]
            
        # Handle DeepEval legacy parameter mappings
        if self.deepeval_threshold is not None:
            self.threshold = self.deepeval_threshold
        if self.deepeval_include_reason is not None:
            self.include_reason = self.deepeval_include_reason
        if self.deepeval_strict_mode is not None:
            self.strict_mode = self.deepeval_strict_mode
        if self.deepeval_async_mode is not None:
            self.async_mode = self.deepeval_async_mode
        if self.deepeval_verbose_mode is not None:
            self.verbose_mode = self.deepeval_verbose_mode
        if self.deepeval_truths_extraction_limit is not None:
            self.truths_extraction_limit = self.deepeval_truths_extraction_limit


class MetricsFactory:
    """Factory for creating metric instances."""
    
    _metric_registry = {
        "bleu": BleuMetric,
        "rouge": RougeMetric,
        
        "bartscore": BartScoreMetric,
        "alignscore": AlignScoreMetric,
        "sentbert": SentBertMetric,
        "sentence_bert": SentBertMetric,  # Alias
        
        "cola": ColaMetric,
        "grammaticality": ColaMetric,  # Alias
        
        "deepeval_answer_relevance": DeepEvalAnswerRelevancyMetric,
        "deepeval_faithfulness": DeepEvalFaithfulnessMetric,
        "deepeval_summarization": DeepEvalSummarizationMetric,
        "deepeval_prompt_alignment": DeepEvalPromptAlignmentMetric,
        "bigbench_hard": BigBenchHardMetric,
        "big_bench_hard": BigBenchHardMetric,  # Alias
    }
    
    @classmethod
    def register_metric(cls, name: str, metric_class: type):
        """Register a new metric class."""
        cls._metric_registry[name] = metric_class
        logger.info(f"Registered metric: {name} -> {metric_class.__name__}")
    
    @classmethod
    def get_available_metrics(cls) -> List[str]:
        """Get list of all available metric names."""
        return list(cls._metric_registry.keys())
    
    @classmethod
    def create_metric(cls, name: str, config: Optional[MetricConfig] = None) -> BaseMetric:

        name = name.lower().strip()
        
        if name not in cls._metric_registry:
            available = ", ".join(cls.get_available_metrics())
            raise ValueError(f"Unknown metric '{name}'. Available metrics: {available}")
        
        metric_class = cls._metric_registry[name]
        return metric_class(config)
    
    @classmethod
    def create_metrics(cls, metrics_config: MetricsConfig) -> Dict[str, BaseMetric]:
        base_config = MetricConfig(
            batch_size=metrics_config.batch_size,
            device=metrics_config.device,
            cache_dir=metrics_config.cache_dir,
            aggregate=metrics_config.aggregate,
            checkpoint=metrics_config.checkpoint,
            model_name=metrics_config.model_name,
            provider=metrics_config.provider,
            api_key=metrics_config.api_key,
            base_url=metrics_config.base_url,
            model=metrics_config.model,
            threshold=metrics_config.threshold,
            include_reason=metrics_config.include_reason,
            strict_mode=metrics_config.strict_mode,
            async_mode=metrics_config.async_mode,
            verbose_mode=metrics_config.verbose_mode,
            truths_extraction_limit=metrics_config.truths_extraction_limit,
            benchmark_params=metrics_config.benchmark_params,
            generation_params=metrics_config.generation_params,
        )
        
        metrics = {}
        for metric_name in metrics_config.metrics:
            try:
                metric = cls.create_metric(metric_name, base_config)
                metrics[metric_name] = metric
                logger.info(f"Created metric: {metric_name}")
            except Exception as e:
                logger.error(f"Failed to create metric '{metric_name}': {e}")
                
        return metrics
    
    @classmethod
    def create_from_config(cls, config: Union[Dict[str, Any], MetricsConfig]) -> Dict[str, BaseMetric]:

        if isinstance(config, dict):
            metrics_config = MetricsConfig(**config)
        else:
            metrics_config = config
            
        return cls.create_metrics(metrics_config)


def get_metric_categories() -> Dict[str, List[str]]:
    return {
        "lexical": ["bleu", "rouge"],
        "semantic": ["bartscore", "alignscore", "sentbert"],
        "linguistic": ["cola"],
        "llm_based": [
            "deepeval_answer_relevance",
            "deepeval_faithfulness", 
            "deepeval_summarization",
            "deepeval_prompt_alignment"
        ],
        "benchmark": ["bigbench_hard"]
    }


def get_metric_requirements() -> Dict[str, Dict[str, bool]]:

    return {
        "bleu": {"requires_references": True, "requires_original_texts": False},
        "rouge": {"requires_references": True, "requires_original_texts": False},
        "bartscore": {"requires_references": True, "requires_original_texts": True},
        "alignscore": {"requires_references": True, "requires_original_texts": True},
        "sentbert": {"requires_references": False, "requires_original_texts": False},
        "cola": {"requires_references": False, "requires_original_texts": False},
        "deepeval_answer_relevance": {"requires_references": False, "requires_original_texts": True},
        "deepeval_faithfulness": {"requires_references": False, "requires_original_texts": True},
        "deepeval_summarization": {"requires_references": False, "requires_original_texts": True},
        "deepeval_prompt_alignment": {"requires_references": True, "requires_original_texts": True},
        "bigbench_hard": {"requires_references": False, "requires_original_texts": False},
    } 