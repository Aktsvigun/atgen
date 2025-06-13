from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import logging
from dataclasses import dataclass, field


@dataclass
class MetricConfig:
<<<<<<< HEAD
=======
    """Base configuration for metrics."""
    # General parameters
>>>>>>> a24e0f2 (removed dependencies chek)
    batch_size: int = 32
    device: str = "cuda"
    cache_dir: str = "cache"
    aggregate: bool = True
    
<<<<<<< HEAD
    checkpoint: Optional[str] = None
    model_name: Optional[str] = None
    
    # API configuration parameters  
    provider: Optional[str] = None
=======
    # Model-specific parameters (for local models)
    checkpoint: Optional[str] = None
    model_name: Optional[str] = None
    
    # API-based parameters (for LLM-based metrics)
>>>>>>> a24e0f2 (removed dependencies chek)
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model: Optional[str] = None
    
<<<<<<< HEAD
=======
    # DeepEval specific parameters
>>>>>>> a24e0f2 (removed dependencies chek)
    threshold: float = 0.5
    include_reason: bool = False
    strict_mode: bool = False
    async_mode: bool = True
    verbose_mode: bool = False
    truths_extraction_limit: Optional[int] = None
<<<<<<< HEAD
=======
    
    # BigBenchHard specific parameters
>>>>>>> a24e0f2 (removed dependencies chek)
    benchmark_params: Dict[str, Any] = field(default_factory=dict)
    generation_params: Dict[str, Any] = field(default_factory=dict)


class BaseMetric(ABC):
    def __init__(self, config: Optional[MetricConfig] = None):
        self.config = config or MetricConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
<<<<<<< HEAD
        
        self._validate_config()
            
    @property
    def name(self) -> str:
        return self.__class__.__name__.replace("Metric", "").lower()
    
    def _validate_config(self):
=======
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
>>>>>>> a24e0f2 (removed dependencies chek)
        pass
    
    @abstractmethod
    def calculate(self, predictions, references, original_texts):
        pass
<<<<<<< HEAD
    
    def is_available(self) -> bool:
        """Check if the metric is available for use."""
        return True
=======
>>>>>>> a24e0f2 (removed dependencies chek)
    