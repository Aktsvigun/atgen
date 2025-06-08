"""
DeepEval Summarization metric for evaluating summarization quality.
"""

from typing import List, Dict, Union, Optional
from deepeval.test_case import LLMTestCase
from deepeval.metrics import SummarizationMetric

from .base_deepeval_metric import BaseDeepEvalMetric


class DeepEvalSummarizationMetric(BaseDeepEvalMetric):
    """DeepEval Summarization metric for evaluating summarization quality."""
    
    def calculate(self, predictions: List[str], references: Optional[List[Union[str, List[str]]]] = None, original_texts: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Calculate Summarization scores.
        
        Args:
            predictions: List of predicted texts
            references: Not used for Summarization
            original_texts: List of original/source texts (required)
            
        Returns:
            Dictionary with Summarization scores
        """
        if not self.is_available():
            raise RuntimeError("DeepEval dependencies not available")
        
        if original_texts is None:
            raise ValueError("Summarization metric requires original texts")
        
        # Create DeepEval metric
        metric = self._create_deepeval_metric(SummarizationMetric)
        
        # Create test cases
        test_cases = []
        for pred, src in zip(predictions, original_texts):
            test_case = LLMTestCase(
                input=src,
                actual_output=pred,
            )
            test_cases.append(test_case)
        
        # Run evaluation
        return self._run_evaluation(test_cases, metric, "deepeval_summarization") 