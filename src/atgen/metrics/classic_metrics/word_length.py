from atgen.metrics.base_metric import BaseMetric, MetricConfig
from typing import List, Optional
import numpy as np


class WordLengthConfig(MetricConfig):
    pass


class WordLength(BaseMetric):
    def __init__(self, config: WordLengthConfig):
        super().__init__(config)
        
    def compute(self, predictions: List[str], references: List[str], sources: Optional[List[str]] = None, **kwargs) -> float:
        # Calculate generated text lengths
        gen_word_lengths = np.array([len(text.split()) for text in predictions])
        
        # Calculate reference text lengths
        if isinstance(references[0], list):
            ref_word_lengths = np.array(
                [
                    np.mean([len(text.split()) for text in ref])
                    for ref in references
                ]
            )
        else:
            ref_word_lengths = np.array([len(ref.split()) for ref in references])
        
        # Avoid division by zero
        ref_word_lengths_safe = np.where(ref_word_lengths > 0, ref_word_lengths, 1)
        relative_lengths = gen_word_lengths / ref_word_lengths_safe
        
        if self.config.aggregate:
            return {"word_length": float(np.mean(relative_lengths))}
        else:
            return {"word_length": relative_lengths}