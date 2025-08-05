from atgen.metrics.base_metric import BaseMetric, MetricConfig
from typing import List, Optional
import numpy as np

class ExactMatchConfig(MetricConfig):
    aggregate: bool = True


class ExactMatch(BaseMetric):
    def __init__(self, config: ExactMatchConfig):
        super().__init__(config)
        
                
    def compute(self, predictions: List[str], references: List[str], sources: Optional[List[str]] = None, **kwargs) -> float:

        if isinstance(references[0], list):
            scores = np.array(
                [
                    any(self._preprocess_text(pred) == self._preprocess_text(one_ref) for one_ref in ref)
                    for pred, ref in zip(predictions, references)
                ]
            )
        else:
            scores = np.array(
                [self._preprocess_text(pred) == self._preprocess_text(ref) for pred, ref in zip(predictions, references)]
            )
        
        if self.config.aggregate:
            return {"exact_match": float(np.mean(scores))}
        else:
            return {"exact_match": scores}