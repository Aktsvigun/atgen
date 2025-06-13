"""
BigBenchHard metric for evaluating model performance on challenging reasoning tasks.
"""

from typing import List, Dict, Union, Optional
from transformers import GenerationMixin, PreTrainedTokenizerBase
from deepeval.benchmarks import BigBenchHard
from deepeval.benchmarks.big_bench_hard.template import BigBenchHardTemplate
from deepeval.models.base_model import DeepEvalBaseLLM

from ..base.base_metric import BaseMetric, MetricConfig


class BigBenchHardModel(DeepEvalBaseLLM):
    """DeepEval model wrapper for BigBenchHard evaluation."""
    
    def __init__(
        self,
        model: GenerationMixin,
        tokenizer: PreTrainedTokenizerBase,
        device: str,
        model_name: str = "Model",
        **generation_kwargs,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model_name = model_name
        self.generation_kwargs = generation_kwargs

    def load_model(self) -> GenerationMixin:
        return self.model

    def generate(self, prompt: str, **kwargs) -> str:
        model = self.load_model()
        model.to(self.device)
        model_inputs = self.tokenizer([prompt], return_tensors="pt").to(self.device)
        generated_ids = model.generate(**model_inputs, **self.generation_kwargs)
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    async def a_generate(self, prompt: str, **kwargs) -> str:
        return self.generate(prompt)

    def batch_generate(self, prompts: list[str], **kwargs) -> list[str]:
        model = self.load_model()
        model.to(self.device)
        model_inputs = self.tokenizer(prompts, return_tensors="pt").to(self.device)
        generated_ids = model.generate(**model_inputs, **self.generation_kwargs)
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    def get_model_name(self):
        return self.model_name


class BigBenchHardFixed(BigBenchHard):
    """
    Fixed version of BigBenchHard that addresses batch prediction issues.
    
    The main branch for deepeval contains broken code, as "NumberModel" is not defined:
    https://github.com/confident-ai/deepeval/blob/main/deepeval/benchmarks/big_bench_hard/big_bench_hard.py#L153
    """
    
    def batch_predict(self, model, task, goldens):
        prompts = []
        for golden in goldens:
            prompt: dict = BigBenchHardTemplate.generate_output(
                input=golden.input,
                task=task,
                n_shots=self.n_shots,
                enable_cot=self.enable_cot,
            )
            prompts.append(prompt)

        # Enforced model generation
        prompts = [
            prompt + "Make sure to output only the numerical answer."
            for prompt in prompts
        ]
        predictions = model.batch_generate(prompts)
        predictions = [str(pred) for pred in predictions]

        if len(predictions) != len(goldens):
            raise ValueError(
                "Custom `batch_generate` method did not return the same "
                "number of generations as the number of prompts."
            )

        res = []
        for i in range(len(predictions)):
            prediction = predictions[i]
            prediction = prediction.split()[-1]
            prediction = prediction[:-1] if self.enable_cot else prediction
            golden = goldens[i]

            # Define Metric
            score = self.scorer.exact_match_score(golden.expected_output, prediction)
            res.append({"prediction": prediction, "score": score})

        return res


class BigBenchHardMetric(BaseMetric):
    """BigBenchHard benchmark metric for evaluating reasoning performance."""
    
    def __init__(self, config: Optional[MetricConfig] = None):
        super().__init__(config)
        self.model = None
        self.tokenizer = None
        self.benchmark = None
        self.benchmark_model = None
        
        # Generation parameters
        self.generation_params = getattr(config, 'generation_params', {}) if config else {}
        self.benchmark_params = getattr(config, 'benchmark_params', {}) if config else {}
        self.model_name = getattr(config, 'model_name', 'Model') if config else 'Model'
    
    
    def set_model_and_tokenizer(self, model: GenerationMixin, tokenizer: PreTrainedTokenizerBase):
        """
        Set the model and tokenizer for evaluation.
        
        Args:
            model: The model to evaluate
            tokenizer: The tokenizer for the model
        """
        self.model = model
        self.tokenizer = tokenizer
        
        # Initialize the benchmark model and benchmark
        self.benchmark_model = BigBenchHardModel(
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.config.device,
            model_name=self.model_name,
            **self.generation_params
        )
        self.benchmark = BigBenchHardFixed(**self.benchmark_params)
    
    def calculate(self, predictions: List[str], references: Optional[List[Union[str, List[str]]]] = None, original_texts: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Calculate BigBenchHard score.
        
        Note: This method signature is maintained for consistency with BaseMetric,
        but BigBenchHard doesn't use predictions/references in the traditional way.
        The model and tokenizer must be set using set_model_and_tokenizer() before calling this.
        
        Args:
            predictions: Not used for BigBenchHard
            references: Not used for BigBenchHard  
            original_texts: Not used for BigBenchHard
            
        Returns:
            Dictionary with BigBenchHard score
        """
        if not self.is_available():
            raise RuntimeError("BigBenchHard dependencies not available")
        
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer must be set using set_model_and_tokenizer() before evaluation")
        
        if self.benchmark_model is None or self.benchmark is None:
            self.set_model_and_tokenizer(self.model, self.tokenizer)
        
        # Run the benchmark evaluation
        self.benchmark.evaluate(model=self.benchmark_model, batch_size=self.config.batch_size)
        
        return {"bigbench_hard_score": float(self.benchmark.overall_score)}
    
    def evaluate_benchmark(self, model: GenerationMixin, tokenizer: PreTrainedTokenizerBase, batch_size: Optional[int] = None) -> float:
        """
        Convenience method for direct benchmark evaluation.
        
        Args:
            model: The model to evaluate
            tokenizer: The tokenizer for the model
            batch_size: Batch size for evaluation (uses config default if None)
            
        Returns:
            Overall benchmark score
        """
        self.set_model_and_tokenizer(model, tokenizer)
        
        eval_batch_size = batch_size if batch_size is not None else self.config.batch_size
        self.benchmark.evaluate(model=self.benchmark_model, batch_size=eval_batch_size)
        
        return self.benchmark.overall_score 