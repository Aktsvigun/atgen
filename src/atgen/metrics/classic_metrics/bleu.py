from atgen.metrics.base_metric import BaseMetric, MetricConfig
from typing import List, Optional
from nltk import ngrams
from nltk.stem import porter
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.translate.bleu_score import corpus_bleu
import numpy as np


class BleuConfig(MetricConfig):
    pass


class Bleu(BaseMetric):
    def __init__(self, config: BleuConfig):
        super().__init__(config)


    def _smoothing_function(self, p_n, references, hypothesis, hyp_len):
        smoothed_p_n = []
        for i, p_i in enumerate(p_n, start=1):
            # Smoothing is not applied for unigrams
            if i > 1:
                # If hypothesis length is lower than the current order, its value equals (0 + 1) / (0 + 1) = 0
                if hyp_len < i:
                    assert p_i.denominator == 1
                    smoothed_p_n.append(1)
                # Otherwise apply smoothing
                else:
                    smoothed_p_i = (p_i.numerator + 1) / (p_i.denominator + 1)
                    smoothed_p_n.append(smoothed_p_i)
            else:
                smoothed_p_n.append(p_i)
        return smoothed_p_n

    def compute(self, predictions: List[str], references: List[str], sources: Optional[List[str]] = None, **kwargs) -> float:
        scores = []
        for pred, ref in zip(predictions, references):
            if isinstance(ref, str):
                ref_list = [ref]
            else:
                ref_list = ref
            
            tok_ref = [[word_tokenize(r) for r in ref_list]]
            tok_pred = [word_tokenize(pred)]
            
            try:
                bleu_score = corpus_bleu(tok_ref, tok_pred, smoothing_function=self._smoothing_function)
                scores.append(bleu_score)
            except (KeyError, ZeroDivisionError):
                scores.append(0.0)
        
        if self.config.aggregate:
            return {"bleu": float(np.mean(scores))}
        else:
            return {"bleu": scores}