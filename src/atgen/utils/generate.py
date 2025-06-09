import gc
from math import ceil
from pathlib import Path
from typing import Union, Any
import logging

from datasets import Dataset
from omegaconf import DictConfig

from torch.utils.data import DataLoader
from torch import bfloat16, float16, cuda, no_grad, LongTensor
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer
from vllm import LLM

if cuda.is_available():
    from unsloth import FastLanguageModel

from .constants import (
    DEFAULT_GPU_MEMORY_UTILIZATION,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    MESSAGES_COLUMN_NAME,
    REASONING_END_TOKEN,
)

from .training_utils import _get_data_collator
from .find_response_token_ids_in_text import find_response_token_ids_in_text


log = logging.getLogger()


def generate_vllm(
    inference_config: DictConfig,
    data: Dataset,
    data_config: DictConfig,
    model: PreTrainedModel = None,
    tokenizer: PreTrainedTokenizer = None,
    save_dir: str | Path = "tmp",
    model_tokenizer_dir: Union[str, Path, None] = None,
    llm_runner: LLM | None = None,
    **useless_kwargs,
) -> list[str]:
    """
    TODO: improve the description.
    Function for generating with the vLLM framework.
    Requires either model + tokenizer + save_dir or the path to the saved model and tokenizer.
    In the last case, they need to be stored inside "PATH/model" and "PATH/tokenizer".
    """
    from vllm import SamplingParams

    delete_vllm_after_inference = False
    if llm_runner is None:
        if model_tokenizer_dir is None:
            model.save_pretrained(f"{save_dir}/model")
            tokenizer.save_pretrained(f"{save_dir}/tokenizer")
            model_tokenizer_dir = save_dir
        gpu_memory_utilization = getattr(
            inference_config, "gpu_memory_utilization", DEFAULT_GPU_MEMORY_UTILIZATION
        )
        llm_runner = LLM(
            model=f"{model_tokenizer_dir}/model",
            tokenizer=f"{model_tokenizer_dir}/tokenizer",
            gpu_memory_utilization=gpu_memory_utilization,  # TODO: make arbitrary
            dtype=bfloat16,
            trust_remote_code=True,
        )
        delete_vllm_after_inference = True

    del model
    gc.collect()
    cuda.empty_cache()

    sampling_params = SamplingParams(
        temperature=inference_config.get("temperature", DEFAULT_TEMPERATURE),
        seed=42,  # TODO: make arbitrary
        max_tokens=inference_config.max_new_tokens,
        top_p=inference_config.get("top_p", DEFAULT_TOP_P),
    )
    if data_config.assistant_response_start:
        generation_params = {
            "add_generation_prompt": False,
            "continue_final_message": True,
        }
    else:
        generation_params = {
            "add_generation_prompt": True,
            "continue_final_message": False,
        }

    generations = []
    num_batches = ceil(len(data) / inference_config.batch_size)
    for i in tqdm(range(num_batches)):
        batch = data[
            i * inference_config.batch_size : (i + 1) * inference_config.batch_size
        ][MESSAGES_COLUMN_NAME]
        out = llm_runner.chat(
            batch, sampling_params, use_tqdm=False, **generation_params
        )
        batch_generations = [x.outputs[0].text for x in out]
        generations += batch_generations

    # Remove reasoning tokens from DeepSeek-R1
    if "deepseek-r1" in llm_runner.llm_engine.model_config.model:
        for i, generation in enumerate(generations):
            generations[i] = _remove_thinking_part(generation)

    if delete_vllm_after_inference:
        del llm_runner
        gc.collect()
        cuda.empty_cache()
    return generations


def generate_sglang() -> list[str]:
    """
    Function for generating with the SGLang framework.
    Requires either model + tokenizer or the path to the saved model and tokenizer.
    """
    pass


def generate_sglang(
    inference_config: DictConfig,
    data: Dataset,
    data_config: DictConfig,
    model: PreTrainedModel = None,
    tokenizer: PreTrainedTokenizer = None,
    save_dir: str | Path = "tmp",
    model_tokenizer_dir: Union[str, Path, None] = None,
    **useless_kwargs,
) -> list[str]:
    """
    Function for generating with the SGLang framework.
    Requires either model + tokenizer or the path to the saved model and tokenizer.
    """
    import sglang as sgl
    import os

    # Determine the model path
    if model_tokenizer_dir is None:
        if model is not None and tokenizer is not None:
            model.save_pretrained(f"{save_dir}/model")
            tokenizer.save_pretrained(f"{save_dir}/tokenizer")
            model_path = f"{save_dir}/model"
        else:
            raise ValueError(
                "Either model_tokenizer_dir or model and tokenizer must be provided"
            )
    else:
        model_path = f"{model_tokenizer_dir}/model"

    # Free up memory
    del model
    gc.collect()
    cuda.empty_cache()

    # Get generation parameters
    temperature = inference_config.get("temperature", DEFAULT_TEMPERATURE)
    top_p = inference_config.get("top_p", DEFAULT_TOP_P)
    max_tokens = inference_config.max_new_tokens

    # Initialize SGLang engine
    engine = sgl.Runtime(
        model=model_path,
        model_config={
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        },
    )

    # Define generation function using SGLang
    @sgl.function
    def generate_response(s, messages):
        # Apply chat template
        for message in messages:
            role = message.get("role", "")
            content = message.get("content", "")

            if role == "system":
                s += f"System: {content}\n"
            elif role == "user":
                s += f"User: {content}\n"
            elif role == "assistant":
                s += f"Assistant: {content}\n"

        # Generate the response
        s += "Assistant: " + sgl.gen("response")
        return s["response"]

    # Generate responses
    generations = []
    num_batches = ceil(len(data) / inference_config.batch_size)

    for i in tqdm(range(num_batches)):
        batch = data[
            i * inference_config.batch_size : (i + 1) * inference_config.batch_size
        ][MESSAGES_COLUMN_NAME]

        # Use SGLang to generate responses
        batch_results = engine.run_batch(
            [generate_response.bind(messages=messages) for messages in batch]
        )

        # Extract generated text
        batch_generations = [result.outputs["response"] for result in batch_results]
        generations += batch_generations

    # Clean up
    engine.shutdown()
    return generations


def generate_transformers(
    inference_config: DictConfig,
    data: Dataset,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    data_config: DictConfig = None,
    **useless_kwargs,
) -> list:
    # Tokenize dataset if necessary
    if "input_ids" not in data.column_names:
        data = data.map(
            tokenize_conversational_example,
            batched=False,
            fn_kwargs={"tokenizer": tokenizer},
        )

    data_collator = _get_data_collator(
        tokenizer=tokenizer, model_config=inference_config.model
    )
    dataloader = DataLoader(
        data.remove_columns(
            [x for x in data.column_names if x not in ("input_ids", "attention_mask")]
        ),
        batch_size=inference_config.batch_size,
        collate_fn=data_collator,
        shuffle=False,
    )

    if cuda.is_available() and model.device.type != "cuda":
        model = model.cuda()
    if cuda.is_available():
        FastLanguageModel.for_inference(model)

    generations = []
    with no_grad():
        for batch in tqdm(dataloader):
            try:
                out = model.generate(
                    batch["input_ids"].to(model.device),
                    attention_mask=batch["attention_mask"].to(model.device),
                    max_new_tokens=inference_config.max_new_tokens,
                    temperature=inference_config.get(
                        "temperature", DEFAULT_TEMPERATURE
                    ),
                    top_p=inference_config.get("top_p", DEFAULT_TOP_P),
                    return_dict_in_generate=True,
                    output_scores=True,
                )
                outputs_only = []
                for output in out.sequences:
                    seq_only = find_response_token_ids_in_text(
                        output.tolist(), data_collator.response_token_ids
                    )
                    outputs_only.append(tokenizer.decode(seq_only, True).strip())
                generations += outputs_only
            except Exception as e:
                print(f"Error in model.generate: {e}")
                # Add empty strings for this batch to maintain alignment with input data
                generations += [""] * len(batch["input_ids"])
    return generations


def generate(
    inference_config: DictConfig,
    data: Dataset,
    model: PreTrainedModel = None,
    tokenizer: PreTrainedTokenizer = None,
    save_dir: str | Path = "tmp",
    data_config: DictConfig = None,
    model_config: DictConfig = None,
    **kwargs,
) -> list[str]:
    framework = inference_config.framework
    if framework == "vllm":
        return generate_vllm(
            inference_config=inference_config,
            data=data,
            model=model,
            tokenizer=tokenizer,
            save_dir=save_dir,
            data_config=data_config,
            **kwargs,
        )
    elif framework == "transformers":
        return generate_transformers(
            inference_config=inference_config,
            data=data,
            model=model,
            tokenizer=tokenizer,
            save_dir=save_dir,
            data_config=data_config,
            model_config=model_config,
            **kwargs,
        )
    elif framework == "sglang":
        return generate_sglang(
            inference_config=inference_config,
            data=data,
            data_config=data_config,
            model=model,
            tokenizer=tokenizer,
            save_dir=save_dir,
            model_tokenizer_dir=kwargs.get("model_tokenizer_dir"),
            **kwargs,
        )
    else:
        raise NotImplementedError


def tokenize_conversational_example(
    example: dict[str, Any], tokenizer: PreTrainedTokenizer
) -> dict[str, list[int]]:
    input_ids = tokenizer.apply_chat_template(example["messages"])
    attention_mask = [1 for _ in range(len(input_ids))]
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def _remove_thinking_part(text: str) -> str:
    return REASONING_END_TOKEN.join(text.split(REASONING_END_TOKEN)[1:]).strip()
