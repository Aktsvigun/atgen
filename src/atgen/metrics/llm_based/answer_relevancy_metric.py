"""
DeepEval Answer Relevancy metric for evaluating how well outputs answer inputs.
"""

from typing import List, Dict, Union, Optional
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric

from .base_deepeval_metric import BaseDeepEvalMetric


class DeepEvalAnswerRelevancyMetric(BaseDeepEvalMetric):
    """DeepEval Answer Relevancy metric for evaluating how well the output answers the input."""
    
    def calculate(self, predictions: List[str], references: Optional[List[Union[str, List[str]]]] = None, original_texts: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Calculate Answer Relevancy scores.
        
        Args:
            predictions: List of predicted texts
            references: Not used for Answer Relevancy
            original_texts: List of original/source texts (required)
            
        Returns:
            Dictionary with Answer Relevancy scores
        """
        if not self.is_available():
            raise RuntimeError("DeepEval dependencies not available")
        
        if original_texts is None:
            raise ValueError("Answer Relevancy metric requires original texts")
        
        # Create DeepEval metric
        metric = self._create_deepeval_metric(AnswerRelevancyMetric)
        
        # Create test cases
        test_cases = []
        for pred, src in zip(predictions, original_texts):
            test_case = LLMTestCase(
                input=src,
                actual_output=pred,
            )
            test_cases.append(test_case)
        
        # Run evaluation
        return self._run_evaluation(test_cases, metric, "deepeval_answer_relevance") 