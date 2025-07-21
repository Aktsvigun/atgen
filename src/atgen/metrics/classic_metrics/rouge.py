from atgen.metrics.base_metric import BaseMetric, MetricConfig
from typing import List, Optional
from evaluate import load
import numpy as np


class RougeConfig(MetricConfig):
    use_stemmer: bool = True
    


class Rouge(BaseMetric):
    def __init__(self, config: RougeConfig):
        super().__init__(config)
        self.rouge = load("rouge")
        
    def compute(self, predictions: List[str], references: List[str], sources: Optional[List[str]] = None, **kwargs) -> float:
        rouge_scores = self.rouge.compute(
            predictions=predictions,
            references=references,
            use_stemmer=self.config.use_stemmer,
        )

        if self.config.aggregate:
            return {k: float(np.mean(v)) for k, v in rouge_scores.items()}
        else:
            return rouge_scores