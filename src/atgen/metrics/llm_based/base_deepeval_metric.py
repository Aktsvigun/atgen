"""
Base class for DeepEval metrics.
"""

import os
import sys
from typing import List, Dict, Union, Optional
import numpy as np
from deepeval import evaluate
from deepeval.test_case import LLMTestCase

from ..base.base_metric import BaseMetric, MetricConfig
from .evaluation_llm import EvaluationLLM


class BaseDeepEvalMetric(BaseMetric):
    """Base class for all DeepEval metrics."""
    
    def __init__(self, config: Optional[MetricConfig] = None):
        super().__init__(config)
        self.llm = None
    
    def _check_dependencies(self) -> bool:
        """Check if DeepEval dependencies are available."""
        try:
            from deepeval import evaluate
            from deepeval.test_case import LLMTestCase
            from openai import OpenAI, AsyncOpenAI
            return True
        except ImportError:
            return False
    
    def _validate_config(self):
        """Validate DeepEval configuration."""
        super()._validate_config()
        if self.config.api_key is None:
            self.logger.warning("API key is recommended for DeepEval metrics")
    
    def _initialize_llm(self):
        """Initialize the LLM for evaluation if not already initialized."""
        if self.llm is None:
            self.llm = EvaluationLLM(
                api_key=self.config.api_key,
                model=self.config.model or "openai/gpt-4o-2024-11-20",
                base_url=self.config.base_url or "https://openrouter.ai/api/v1",
            )
    
    def _create_deepeval_metric(self, metric_class, **kwargs):
        """Create a DeepEval metric instance with common parameters."""
        self._initialize_llm()
        
        return metric_class(
            threshold=self.config.threshold,
            model=self.llm,
            include_reason=self.config.include_reason,
            strict_mode=self.config.strict_mode,
            async_mode=self.config.async_mode,
            **kwargs
        )
    
    def _run_evaluation(self, test_cases: List[LLMTestCase], metric, metric_name: str) -> Dict[str, float]:
        """Run evaluation with proper output handling."""
        if not test_cases:
            return {}
        
        # Disable printing to console during evaluation if not verbose
        original_stdout = sys.stdout
        if not self.config.verbose_mode:
            sys.stdout = open(os.devnull, "w")
        
        try:
            # Run evaluation
            evaluation_results = evaluate(
                test_cases=test_cases,
                metrics=[metric],
                run_async=self.config.async_mode,
            )
            
            # Process results
            scores = []
            for result in evaluation_results.test_results:
                scores.append(1 if result.success else 0)
            
            # Calculate average score
            if scores:
                if self.config.aggregate:
                    return {metric_name: float(np.mean(scores))}
                else:
                    return {metric_name: np.array(scores)}
            else:
                return {metric_name: 0.0}
                
        finally:
            # Restore stdout
            if not self.config.verbose_mode:
                sys.stdout.close()
                sys.stdout = original_stdout 