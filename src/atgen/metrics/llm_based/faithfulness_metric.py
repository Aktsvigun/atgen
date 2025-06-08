"""
DeepEval Faithfulness metric for evaluating factual consistency with input.
"""

from typing import List, Dict, Union, Optional
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric

from .base_deepeval_metric import BaseDeepEvalMetric


class DeepEvalFaithfulnessMetric(BaseDeepEvalMetric):
    """DeepEval Faithfulness metric for evaluating factual consistency with the input."""
    
    def calculate(self, predictions: List[str], references: Optional[List[Union[str, List[str]]]] = None, original_texts: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Calculate Faithfulness scores.
        
        Args:
            predictions: List of predicted texts
            references: Not used for Faithfulness
            original_texts: List of original/source texts (required)
            
        Returns:
            Dictionary with Faithfulness scores
        """
        if not self.is_available():
            raise RuntimeError("DeepEval dependencies not available")
        
        if original_texts is None:
            raise ValueError("Faithfulness metric requires original texts")
        
        # Create DeepEval metric with truths extraction limit if specified
        kwargs = {}
        if self.config.truths_extraction_limit is not None:
            kwargs['truths_extraction_limit'] = self.config.truths_extraction_limit
        
        metric = self._create_deepeval_metric(FaithfulnessMetric, **kwargs)
        
        # Create test cases
        test_cases = []
        for pred, src in zip(predictions, original_texts):
            test_case = LLMTestCase(
                input=src,
                actual_output=pred,
                retrieval_context=[src],  # Use source as retrieval context
            )
            test_cases.append(test_case)
        
        # Run evaluation
        return self._run_evaluation(test_cases, metric, "deepeval_faithfulness") 