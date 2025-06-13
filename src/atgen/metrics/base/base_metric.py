from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import logging
from dataclasses import dataclass, field


@dataclass
class MetricConfig:
    batch_size: int = 32
    device: str = "cuda"
    cache_dir: str = "cache"
    aggregate: bool = True
    
    checkpoint: Optional[str] = None
    model_name: Optional[str] = None
    
    # API configuration parameters  
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
    benchmark_params: Dict[str, Any] = field(default_factory=dict)
    generation_params: Dict[str, Any] = field(default_factory=dict)


class BaseMetric(ABC):
    def __init__(self, config: Optional[MetricConfig] = None):
        self.config = config or MetricConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self._validate_config()
            
    @property
    def name(self) -> str:
        return self.__class__.__name__.replace("Metric", "").lower()
    
    def _validate_config(self):
        pass
    
    @abstractmethod
    def calculate(self, predictions, references, original_texts):
        pass
    
    def is_available(self) -> bool:
        """Check if the metric is available for use."""
        return True
    