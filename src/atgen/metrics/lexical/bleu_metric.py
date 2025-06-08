from typing import Dict, List, Optional, Union
import numpy as np
from nltk import download
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import corpus_bleu
import logging

from ..base.base_metric import BaseMetric, MetricConfig

# Ensure nltk data is downloaded
try:
    word_tokenize("test")
except LookupError:
    download('punkt')


def smoothing_function(p_n, references, hypothesis, hyp_len):
    """
    Smooth-BLEU (BLEUS) as proposed in the paper:
    Chin-Yew Lin, Franz Josef Och. ORANGE: a method for evaluating automatic
    evaluation metrics for machine translation. COLING 2004.
    """
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


class BleuMetric(BaseMetric):
    """BLEU (Bilingual Evaluation Understudy) metric implementation."""
    
    @property
    def category(self) -> str:
        return "lexical"
    
    @property
    def supports_multiple_references(self) -> bool:
        return True
    
    
    def _compute(
        self,
        predictions: List[str],
        references: Optional[List[Union[str, List[str]]]] = None,
        original_texts: Optional[List[str]] = None
    ) -> Dict[str, Union[float, np.ndarray]]:
        """
        Compute BLEU scores.
        
        Args:
            predictions: List of predicted texts
            references: List of reference texts (can be list of lists for multiple references)
            original_texts: Not used for BLEU computation
            
        Returns:
            Dictionary with BLEU scores
        """
        scores = []
        
        for pred, ref in zip(predictions, references):
            # Ensure references is always a list of lists
            if isinstance(ref, str):
                ref = [ref]
            
            # Tokenize
            tok_ref = [word_tokenize(r) for r in ref]
            tok_pred = word_tokenize(pred)
            
            try:
                score = corpus_bleu(tok_ref, [tok_pred], smoothing_function=smoothing_function)
                scores.append(score)
            except (KeyError, ZeroDivisionError):
                scores.append(0.0)
        
        return {"bleu": np.array(scores)} 