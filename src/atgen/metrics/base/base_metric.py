from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import logging
from dataclasses import dataclass, field


@dataclass
class MetricConfig:
    """Base configuration for metrics."""
    # General parameters
    batch_size: int = 32
    device: str = "cuda"
    cache_dir: str = "cache"
    aggregate: bool = True
    
    # Model-specific parameters (for local models)
    checkpoint: Optional[str] = None
    model_name: Optional[str] = None
    
    # API-based parameters (for LLM-based metrics)
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model: Optional[str] = None
    
    # DeepEval specific parameters
    threshold: float = 0.5
    include_reason: bool = False
    strict_mode: bool = False
    async_mode: bool = True
    verbose_mode: bool = False
    truths_extraction_limit: Optional[int] = None
    
    # BigBenchHard specific parameters
    benchmark_params: Dict[str, Any] = field(default_factory=dict)
    generation_params: Dict[str, Any] = field(default_factory=dict)


class BaseMetric(ABC):
    def __init__(self, config: Optional[MetricConfig] = None):
        self.config = config or MetricConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        self._is_available = None
        self._model = None
        
        # Validate configuration
        self._validate_config()
        
        # Check dependencies
        if not self.is_available():
            self.logger.warning(f"{self.__class__.__name__} dependencies not available")
    
    @property
    def name(self) -> str:
        """Return the metric name (defaults to class name)."""
        return self.__class__.__name__.replace("Metric", "").lower()
    
    def _validate_config(self):
        """Validate configuration - can be overridden by subclasses."""
        pass
    
    @abstractmethod
    def calculate(self, predictions, references, original_texts):
        pass
    