"""
DeepEval Prompt Alignment metric for evaluating alignment with expected output.
"""

from typing import List, Dict, Union, Optional
from deepeval.test_case import LLMTestCase
from deepeval.metrics import PromptAlignmentMetric

from .base_deepeval_metric import BaseDeepEvalMetric


class DeepEvalPromptAlignmentMetric(BaseDeepEvalMetric):
    """DeepEval Prompt Alignment metric for evaluating alignment with the expected output."""
    
    def calculate(self, predictions: List[str], references: Optional[List[Union[str, List[str]]]] = None, original_texts: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Calculate Prompt Alignment scores.
        
        Args:
            predictions: List of predicted texts
            references: List of reference texts (required)
            original_texts: List of original/source texts (required)
            
        Returns:
            Dictionary with Prompt Alignment scores
        """

        if original_texts is None:
            raise ValueError("Prompt Alignment metric requires original texts")
        
        if references is None:
            raise ValueError("Prompt Alignment metric requires reference texts")
        
        if isinstance(references[0], list):
            self.logger.error("Prompt Alignment does not support multiple references. Skipping...")
            return {}
        
        # Create DeepEval metric with default prompt instructions
        metric = self._create_deepeval_metric(
            PromptAlignmentMetric,
            prompt_instructions=["Do what you are told to do in the prompt"]
        )
        
        # Create test cases
        test_cases = []
        for pred, ref, src in zip(predictions, references, original_texts):
            test_case = LLMTestCase(
                input=src,
                actual_output=pred,
                expected_output=ref,
            )
            test_cases.append(test_case)
        
        # Run evaluation
        return self._run_evaluation(test_cases, metric, "deepeval_prompt_alignment") 