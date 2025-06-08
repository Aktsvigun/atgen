from abc import ABC, abstractmethod
from typing import Optional
import logging
from dataclasses import dataclass


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
    
    def is_available(self) -> bool:
        """Check if dependencies are available."""
        if self._is_available is None:
            self._is_available = self._check_dependencies()
        return self._is_available
    
    @abstractmethod
    def _check_dependencies(self) -> bool:
        """Check if required dependencies are available."""
        pass
    
    @abstractmethod
    def calculate(self, predictions, references, original_texts):
        pass
    