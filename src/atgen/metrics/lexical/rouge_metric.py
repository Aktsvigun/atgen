from typing import List, Dict, Union, Optional
import numpy as np
from evaluate import load

from ..base.base_metric import BaseMetric, MetricConfig


class RougeMetric(BaseMetric):
    """ROUGE (Recall-Oriented Understudy for Gisting Evaluation) metric."""
    
    def __init__(self, config: Optional[MetricConfig] = None):
        super().__init__(config)
        self.rouge = None
    
    def _initialize_rouge(self):
        """Initialize ROUGE if not already initialized."""
        if self.rouge is None:
            self.rouge = load("rouge", cache_dir=self.config.cache_dir)
<<<<<<< HEAD
            
=======
>>>>>>> a24e0f2 (removed dependencies chek)
    
    def calculate(self, predictions: List[str], references: Optional[List[Union[str, List[str]]]] = None, original_texts: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Calculate ROUGE scores.
        
        Args:
            predictions: List of predicted texts
            references: List of reference texts (can be list of lists for multiple references)
            original_texts: Not used for ROUGE computation
            
        Returns:
            Dictionary with ROUGE scores
        """
        
        if references is None:
            raise ValueError("ROUGE requires reference texts")
        
        self._initialize_rouge()
        
        # ROUGE expects different format for multiple references
        if isinstance(references[0], list):
            # Multiple references - convert to format expected by ROUGE
            rouge_references = references
        else:
            # Single references
            rouge_references = references
        
        results = self.rouge.compute(
            predictions=predictions,
            references=rouge_references,
            use_stemmer=True,
        )
        
        # Convert to float values
        scores = {}
        for key, value in results.items():
            if isinstance(value, (int, float)):
                scores[key] = float(value)
<<<<<<< HEAD
            elif hasattr(value, 'item'):  #
=======
            elif hasattr(value, 'item'):  # For numpy scalars
>>>>>>> a24e0f2 (removed dependencies chek)
                scores[key] = float(value.item())
            else:
                scores[key] = float(value)
        
<<<<<<< HEAD
        if self.config.aggregate:
            for key, value in scores.items():
                if isinstance(value, np.ndarray):
                    scores[key] = float(np.mean(value))
        
=======
>>>>>>> a24e0f2 (removed dependencies chek)
        return scores 