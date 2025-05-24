from datasets import Dataset
from pathlib import Path
from omegaconf import DictConfig
from torch import bfloat16, cuda
from transformers import PreTrainedModel, PreTrainedTokenizerFast
from vllm import LLM
import gc

from .base_labeller import BaseLabeler
from ..utils.load_model_tokenizer import load_model_tokenizer
from ..utils.generate import generate


# TODO: add support for TensorRT-LLM
class CustomLLMLabeller(BaseLabeler):
    def __init__(
        self,
        config: DictConfig,
        output_column_name: str = "output",
        budget: int = 1_000_000,
        cache_dir: str | Path = "cache",
        data_config: DictConfig = None,
        model_config: DictConfig = None,
    ):
        super().__init__(output_column_name, budget)

        self.framework = config.inference.framework
        self.checkpoint = config.model.checkpoint
        # TODO: ask is it ok to additionally store the two previous attributes?
        self.config = config
        self.cache_dir = cache_dir

        # For vLLM we need to load the model after saved with `.save_pretrained` into vLLM
        if self.framework == "vllm":
            checkpoint_str = self.checkpoint.replace("/", "__")
            self.model_path = Path(cache_dir) / f"labeller_{checkpoint_str}"
            self.model_path.mkdir(exist_ok=True)
            self._save_model()
        elif self.framework == "transformers":
            self.data_config = data_config
            self.model_config = model_config

        self.keep_in_memory = config.keep_in_memory
        if self.keep_in_memory:
            self.runner_kwargs = self._load_runner_kwargs()

    def __call__(self, dataset: Dataset) -> Dataset:
        if not self.keep_in_memory:
            runner_kwargs = self._load_runner_kwargs()
        else:
            runner_kwargs = self.runner_kwargs
        annotations = generate(self.config.inference, data=dataset, **runner_kwargs)
        if self.output_column_name in dataset.column_names:
            dataset = dataset.remove_columns([self.output_column_name])
        dataset = dataset.add_column(self.output_column_name, annotations)
        if not self.keep_in_memory:
            del runner_kwargs
            gc.collect()
            cuda.empty_cache()
        return dataset

    def _save_model(self) -> None:
        model, tokenizer = self._load_model_and_tokenizer()
        model.save_pretrained(self.model_path / "model")
        tokenizer.save_pretrained(self.model_path / "tokenizer")
        del model, tokenizer
        gc.collect()

    def _load_runner_kwargs(self):
        if self.framework == "vllm":
            return self._load_runner_vllm()
        elif self.framework == "transformers":
            model, tokenizer = self._load_model_and_tokenizer()
            return {
                "model": model,
                "tokenizer": tokenizer,
                "data_config": self.data_config,
                "model_config": self.model_config,
            }

    def _load_runner_vllm(self) -> dict[str, LLM]:
        gpu_memory_utilization = getattr(
            self.config.inference, "gpu_memory_utilization", 0.5
        )
        # Need to clean the memory in advance to avoid 'Error in memory profiling.'
        gc.collect()
        cuda.empty_cache()
        vllm = LLM(
            f"{self.model_path}/model",
            f"{self.model_path}/tokenizer",
            gpu_memory_utilization=gpu_memory_utilization,  # TODO: make arbitrary
            dtype=bfloat16,
            trust_remote_code=True,
        )
        # Make it a dict because different frameworks can have different kwargs
        return {"llm_runner": vllm}

    def _load_model_and_tokenizer(
        self,
    ) -> tuple[PreTrainedModel, PreTrainedTokenizerFast]:
        return load_model_tokenizer(
            self.config.model.checkpoint, self.config.model, self.cache_dir
        )
