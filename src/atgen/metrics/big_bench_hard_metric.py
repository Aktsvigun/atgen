from deepeval.benchmarks import BigBenchHard
from deepeval.benchmarks.big_bench_hard.template import BigBenchHardTemplate
from deepeval.models.base_model import DeepEvalBaseLLM
from transformers import (
    GenerationMixin,
    PreTrainedTokenizerBase,
)

"""
https://arxiv.org/abs/2210.09261v1

For benchmark_params, refer to https://docs.confident-ai.com/docs/benchmarks-big-bench-hard
"""


class BigBenchHardModel(DeepEvalBaseLLM):
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


"""
The main branch for deepeval seems to contain broken code, as "NumberModel" is not defined:
https://github.com/confident-ai/deepeval/blob/main/deepeval/benchmarks/big_bench_hard/big_bench_hard.py#L153

This child class fixes batch prediction.
"""


class BigBenchHardFixed(BigBenchHard):
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

        if len(predictions) is not len(goldens):
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


class BigBenchHardMetric:
    def __init__(
        self,
        model: GenerationMixin,
        tokenizer: PreTrainedTokenizerBase,
        device: str = "cuda",
        model_name: str = "Model",
        generation_params: dict = {},
        benchmark_params: dict = {},
    ) -> None:
        self.model = BigBenchHardModel(
            model, tokenizer, device, model_name, **generation_params
        )
        self.benchmark = BigBenchHardFixed(**benchmark_params)

    def score(self, batch_size: int | None = None) -> float:
        self.benchmark.evaluate(model=self.model, batch_size=batch_size)
        return self.benchmark.overall_score
