"""
Metrics module for evaluating text generation tasks.

This module provides a comprehensive collection of metrics organized by category:
- Lexical: BLEU, ROUGE
- Semantic: BARTScore, AlignScore, SentBERT  
- Linguistic: CoLA
- LLM-based: DeepEval metrics, BigBenchHard

The module uses a factory pattern for creating metrics and supports
configuration-based instantiation.
"""

# Base classes
from .base import BaseMetric, MetricConfig

# Lexical metrics
from .lexical import BleuMetric, RougeMetric

# Semantic metrics  
from .semantic import BartScoreMetric, AlignScoreMetric, SentBertMetric

# Linguistic metrics
from .linguistic import ColaMetric

# LLM-based metrics
from .llm_based import (
    EvaluationLLM,
    BaseDeepEvalMetric,
    DeepEvalAnswerRelevancyMetric,
    DeepEvalFaithfulnessMetric,
    DeepEvalSummarizationMetric,
    DeepEvalPromptAlignmentMetric,
    BigBenchHardMetric,
)

# Factory system
from .factory import (
    MetricsFactory,
    MetricsConfig,
    get_metric_categories,
    get_metric_requirements,
)

# Compute system
from .compute_metrics_v2 import (
    compute_metrics_v2,
    compute_metrics_from_config,
    get_default_config,
    get_comprehensive_config,
    get_deepeval_config,
)

# Legacy compute system (for backward compatibility)
from .compute_metrics import compute_metrics

# Legacy metrics functions (for backward compatibility)
from .metrics import (
    compute_bleu,
    compute_rouge,
    compute_bartscore,
    compute_alignscore,
    compute_sentbert,
    compute_cola,
)


# Version
__version__ = "2.0.0"

# Main exports
__all__ = [
    # Base classes
    "BaseMetric",
    "MetricConfig",
    
    # Lexical metrics
    "BleuMetric",
    "RougeMetric",
    
    # Semantic metrics
    "BartScoreMetric",
    "AlignScoreMetric", 
    "SentBertMetric",
    
    # Linguistic metrics
    "ColaMetric",
    
    # LLM-based metrics
    "EvaluationLLM",
    "BaseDeepEvalMetric",
    "DeepEvalAnswerRelevancyMetric",
    "DeepEvalFaithfulnessMetric",
    "DeepEvalSummarizationMetric",
    "DeepEvalPromptAlignmentMetric",
    "BigBenchHardMetric",
    
    # Factory system
    "MetricsFactory",
    "MetricsConfig",
    "get_metric_categories",
    "get_metric_requirements",
    
    # New compute system  
    "compute_metrics_v2",
    "compute_metrics_from_config",
    "get_default_config",
    "get_comprehensive_config",
    "get_deepeval_config",
    
    # Legacy compatibility
    "compute_metrics",
    "compute_bleu",
    "compute_rouge",
    "compute_bartscore",
    "compute_alignscore",
    "compute_sentbert",
    "compute_cola",
]


def get_available_metrics():
    """Get all available metrics."""
    return MetricsFactory.get_available_metrics()


def create_metric(name: str, config: MetricConfig = None):
    """Create a metric by name."""
    return MetricsFactory.create_metric(name, config)


def create_metrics_from_config(config):
    """Create metrics from configuration."""
    return MetricsFactory.create_from_config(config) 