from multiprocessing import reduction
from typing import Literal, Optional
from omegaconf import DictConfig
import numpy as np

from atgen.metrics.classic_metrics.base_metric import BaseMetric, BaseMetricConfig


class ExactMatchMathConfig(BaseMetricConfig):
    aggregate: bool = True

class ExactMatchMath(BaseMetric):
    def __init__(self, config: DictConfig):
        super().__init__(config)

    def compute(self, generated_texts: list[str], reference_texts: list[str], original_texts: list[str], task: Literal["summarization", "open-qa", "multi-choice-qa", "translation", "math"]) -> float:
        scores = np.array(
            [
                pred.split("#### ")[-1].lower() == ref.split("#### ")[-1].lower()
                for pred, ref in zip(generated_texts, reference_texts)
            ]
        )
        if self.config.aggregate:
            return {"exact_match_math": float(np.mean(scores))}
        else:
            return {"exact_match_math": scores}
