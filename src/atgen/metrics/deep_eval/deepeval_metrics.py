from math import lcm
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    SummarizationMetric,
    PromptAlignmentMetric,
)
from atgen.metrics.base_metric import BaseMetric, MetricConfig
from atgen.metrics.deep_eval.evaluationllm import EvaluationLLM
from typing import List, Optional
from deepeval.test_case import LLMTestCase
from deepeval import evaluate
import numpy as np
import sys
import os


class DeepevalConfig(MetricConfig):
    base_url: str = "https://openrouter.ai/api/v1"
    api_key: str = None
    model: str = "openai/gpt-4o-2024-11-20"
    threshold: float = 0.5
    include_reason: bool = False
    strict_mode: bool = False
    async_mode: bool = True
    verbose_mode: bool = False
    truths_extraction_limit: int = None
    metrics_to_calculate: list[str] = ["deepeval_answer_relevance", "deepeval_faithfulness", "deepeval_summarization", "deepeval_prompt_alignment"]
    
    
class Deepeval(BaseMetric):
    def __init__(self, config: DeepevalConfig):
        super().__init__(config)
        
        self.llm = EvaluationLLM(
            base_url=self.config.base_url,
            api_key=self.config.api_key,
            model=self.config.model,
            threshold=self.config.threshold,
            include_reason=self.config.include_reason,
            strict_mode=self.config.strict_mode,
        )
        
    def compute(self, predictions: List[str], references: List[str], original_texts: List[str], sources: Optional[List[str]] = None, **kwargs) -> float:
        results = {}

        metrics = []
        metric_name_mapping = {}  # Maps metric class name to the deepeval metric name

        # Dictionary to store test cases for each metric
        metric_test_cases = {}

        if "deepeval_answer_relevance" in self.config.metrics_to_calculate:
            metric = AnswerRelevancyMetric(
                threshold=self.config.threshold,
                model=self.llm,
                include_reason=self.config.include_reason,
                strict_mode=self.config.strict_mode,
                async_mode=self.config.async_mode,
            )
            metrics.append(metric)
            metric_name_mapping[metric.__class__.__name__] = "deepeval_answer_relevance"

            # Create specific test cases for AnswerRelevancy metric
            answer_relevance_test_cases = []
            for i, (pred, src) in enumerate(zip(predictions, original_texts)):
                test_case = LLMTestCase(
                    input=src,
                    actual_output=pred,
                )
                answer_relevance_test_cases.append(test_case)
            metric_test_cases[metric.__class__.__name__] = answer_relevance_test_cases

        if "deepeval_faithfulness" in self.config.metrics_to_calculate:
            metric = FaithfulnessMetric(
                threshold=self.config.threshold,
                model=self.llm,
                include_reason=self.config.include_reason,
                strict_mode=self.config.strict_mode,
                async_mode=self.config.async_mode,
                truths_extraction_limit=self.config.truths_extraction_limit,
            )
            metrics.append(metric)
            metric_name_mapping[metric.__class__.__name__] = "deepeval_faithfulness"

            # Create specific test cases for Faithfulness metric
            faithfulness_test_cases = []
            for i, (pred, src) in enumerate(zip(predictions, original_texts)):
                test_case = LLMTestCase(
                    input=src,
                    actual_output=pred,
                    retrieval_context=[src],
                )
                faithfulness_test_cases.append(test_case)
            metric_test_cases[metric.__class__.__name__] = faithfulness_test_cases

        if "deepeval_summarization" in self.config.metrics_to_calculate:
            metric = SummarizationMetric(
                threshold=self.config.threshold,
                model=self.llm,
                include_reason=self.config.include_reason,
                strict_mode=self.config.strict_mode,
                async_mode=self.config.async_mode,
            )
            metrics.append(metric)
            metric_name_mapping[metric.__class__.__name__] = "deepeval_summarization"

            # Create specific test cases for Summarization metric
            summarization_test_cases = []
            for i, (pred, src) in enumerate(zip(predictions, original_texts)):
                test_case = LLMTestCase(
                    input=src,
                    actual_output=pred,
                )
                summarization_test_cases.append(test_case)
            metric_test_cases[metric.__class__.__name__] = summarization_test_cases

        if "deepeval_prompt_alignment" in self.config.metrics_to_calculate:
            metric = PromptAlignmentMetric(
                threshold=self.config.threshold,
                model=self.llm,
                prompt_instructions=["Do what you are told to do in the prompt"],
                include_reason=self.config.include_reason,
                strict_mode=self.config.strict_mode,
                async_mode=self.config.async_mode,
            )
            metrics.append(metric)
            metric_name_mapping[metric.__class__.__name__] = "deepeval_prompt_alignment"

            # Create specific test cases for PromptAlignment metric
            prompt_alignment_test_cases = []
            for i, (pred, ref, src) in enumerate(
                zip(predictions, references, original_texts)
            ):
                test_case = LLMTestCase(
                    input=src,
                    actual_output=pred,
                    expected_output=ref,
                )
                prompt_alignment_test_cases.append(test_case)
            metric_test_cases[metric.__class__.__name__] = prompt_alignment_test_cases

        for metric in metrics:
            metric_class_name = metric.__class__.__name__
            test_cases = metric_test_cases.get(metric_class_name, [])

            if test_cases:
                original_stdout = sys.stdout
                if not self.config.verbose_mode:
                    sys.stdout = open(os.devnull, "w")

                try:
                    # Run evaluation for this specific metric
                    evaluation_results = evaluate(
                        test_cases=test_cases,
                        metrics=[metric],
                        run_async=self.config.async_mode,
                    )

                    deepeval_metric_name = metric_name_mapping.get(metric_class_name)
                    scores = []
                    reasons = []

                    # Process results for this metric
                    for result in evaluation_results.test_results:
                        scores.append(1 if result.success else 0)

                    # Calculate average score
                    if scores:
                        results[deepeval_metric_name] = np.mean(scores)

                finally:
                    # Restore stdout
                    if not self.config.verbose_mode:
                        sys.stdout.close()
                        sys.stdout = original_stdout
        print("================================================")
        print("Results:")
        print(results)
        print("================================================")

        return {"deepeval": results}

