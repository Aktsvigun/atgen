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
    
    def _validate_config(self):
        """Validate DeepEval configuration."""
        super()._validate_config()
        if self.config.api_key is None:
            self.logger.warning("API key is recommended for DeepEval metrics")
    
    def _initialize_llm(self):
        """Initialize the LLM for evaluation if not already initialized."""
        if self.llm is None:
            # Determine base_url based on provider if not explicitly set
            base_url = self.config.base_url
            if base_url is None and self.config.provider:
                provider = self.config.provider.lower()
                if provider == "openai":
                    base_url = "https://api.openai.com/v1"
                elif provider == "anthropic":
                    base_url = "https://api.anthropic.com/v1"
                elif provider == "openrouter":
                    base_url = "https://openrouter.ai/api/v1"
                else:
                    self.logger.warning(f"Unknown provider '{provider}', using default base_url")
                    base_url = "https://openrouter.ai/api/v1"
            else:
                base_url = base_url or "https://openrouter.ai/api/v1"
            
            # Determine default model based on provider if not explicitly set
            model = self.config.model
            if model is None and self.config.provider:
                provider = self.config.provider.lower()
                if provider == "openai":
                    model = "gpt-4o-mini"
                elif provider == "anthropic":
                    model = "claude-3-5-sonnet"
                elif provider == "openrouter":
                    model = "openai/gpt-4o-mini"
                else:
                    model = "openai/gpt-4o-mini"
            else:
                model = model or "openai/gpt-4o-mini"
            
            self.llm = EvaluationLLM(
                api_key=self.config.api_key,
                model=model,
                base_url=base_url,
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
        
        original_stdout = sys.stdout
        if not self.config.verbose_mode:
            sys.stdout = open(os.devnull, "w")
        
        try:
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