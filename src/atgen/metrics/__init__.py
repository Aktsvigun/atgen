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
from .compute_metrics import (
    compute_metrics,
    compute_metrics_from_config,
    get_default_config,
    get_comprehensive_config,
    get_deepeval_config,
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
    "compute_metrics",
    "compute_metrics_from_config",
    "get_default_config",
    "get_comprehensive_config",
    "get_deepeval_config",
    
    # Helper functions
    "get_available_metrics",
    "get_all_possible_metric_keys",
    "create_metric",
    "create_metrics_from_config",
    
    "AVAILABLE_METRICS",
    "METRICS",
]


def get_available_metrics():
    """Get all available metric names."""
    return MetricsFactory.get_available_metrics()


def get_all_possible_metric_keys():
    """Get all possible metric keys that can be returned by compute_metrics."""
    return MetricsFactory.get_all_possible_metric_keys()


def create_metric(name: str, config: MetricConfig = None):
    """Create a metric by name."""
    return MetricsFactory.create_metric(name, config)


def create_metrics_from_config(config):
    """Create metrics from configuration."""
    return MetricsFactory.create_from_config(config)

# For backward compatibility, still provide the basic metric names
METRICS = MetricsFactory.get_available_metrics()

# But AVAILABLE_METRICS should include all possible keys for performance checking
AVAILABLE_METRICS = MetricsFactory.get_all_possible_metric_keys() 
