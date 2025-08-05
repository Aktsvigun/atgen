from time import time
from typing import List, Optional
from atgen.metrics.base_metric import BaseMetric, MetricConfig
from evaluate import load
import numpy as np



sacrebleu = load("sacrebleu")


class SacrebleuConfig(MetricConfig):
    pass


class Sacrebleu(BaseMetric):
    def __init__(self, config: SacrebleuConfig):
        super().__init__(config)
        
    def compute(self, predictions: List[str], references: List[str], sources: Optional[List[str]] = None, **kwargs) -> float:
        if not isinstance(references[0], list):
            sacrebleu_references = [[ref] for ref in references]
            sacrebleu_result = sacrebleu.compute(
                predictions=predictions, references=sacrebleu_references
            )
            return float(sacrebleu_result.pop("score"))
        else:
            sacrebleu_scores = []
            for pred, ref in zip(predictions, references):
                sacrebleu_result = sacrebleu.compute(
                    predictions=[pred], references=[ref]
                )
                sacrebleu_scores.append(sacrebleu_result.pop("score"))
            if self.config.aggregate:
                return {"sacrebleu": float(np.mean(sacrebleu_scores))}
            else:
                return {"sacrebleu": sacrebleu_scores}