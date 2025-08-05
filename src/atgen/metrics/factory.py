import logging
from omegaconf import DictConfig, OmegaConf
from .registry import METRICS_REGISTRY
from .base_metric import BaseMetric
from typing import Optional
log = logging.getLogger(__name__)

class MetricFactory:
    def __init__(self, config: DictConfig, cache_dir: Optional[str] = None):
        """
        Initializes the factory with the global configuration.
        
        Args:
            config: The main OmegaConf DictConfig object.
        """
        self.config = config
        self.cache_dir = cache_dir
    def get_metric(self, metric_name: str) -> BaseMetric:
        """
        Instantiates and returns a metric runner based on its name.

        It automatically creates the specific config for the metric
        by extracting relevant parameters from the main config object.

        Args:
            metric_name: The name of the metric to instantiate.

        Returns:
            An instance of a class derived from BaseMetric.
        """
        if metric_name not in METRICS_REGISTRY:
            if "deepeval" in metric_name:
                metric_name = "deepeval"
            else:
                raise ValueError(f"Metric '{metric_name}' not found in registry.")

        MetricClass, ConfigClass = METRICS_REGISTRY[metric_name]
        metric_config = ConfigClass()
        
        return MetricClass(config=metric_config)
