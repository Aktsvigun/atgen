from pathlib import Path

import torch

if torch.cuda.is_available():
    from unsloth import FastLanguageModel
else:

    class FastLanguageModel:
        pass


from omegaconf import DictConfig
from transformers import PreTrainedTokenizerFast
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)


# def load_tokenizer(
#     model_config: DictConfig,
#     cache_dir: str,
# ) -> PreTrainedTokenizerFast:
#     tokenizer = AutoTokenizer.from_pretrained(
#         model_config.checkpoint,
#         model_max_length=model_config.model_max_length,
#         cache_dir=cache_dir,
#         padding_side="left",
#     )
#     if tokenizer.pad_token is None:
#         last_reserved_token = {v: k for k, v in tokenizer.vocab.items()}[len(tokenizer) - 1]
#         tokenizer.pad_token = last_reserved_token
#     return tokenizer


def load_model_tokenizer(
    checkpoint: str | Path,
    model_config: DictConfig,
    cache_dir: str,
) -> tuple[FastLanguageModel, PreTrainedTokenizerFast]:
    dtype = getattr(torch, model_config.dtype, torch.bfloat16)
    if torch.cuda.is_available():
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=checkpoint,
            max_seq_length=model_config.model_max_length,
            dtype=dtype,
            load_in_4bit=model_config.quantize,
            cache_dir=cache_dir,
            trust_remote_code=True,
        )
        tokenizer.model_max_length = model_config.model_max_length
    else:
        kwargs = {
            "cache_dir": cache_dir,
            "torch_dtype": model_config.dtype,
            "trust_remote_code": True,
        }
        model = AutoModelForCausalLM.from_pretrained(checkpoint, **kwargs)

        tokenizer = AutoTokenizer.from_pretrained(
            model_config.checkpoint,
            model_max_length=model_config.model_max_length,
            cache_dir=cache_dir,
            # TODO: choose automatically
            # padding_side="left",
        )

    if tokenizer.pad_token is None:
        last_reserved_token = {v: k for k, v in tokenizer.vocab.items()}[
            len(tokenizer) - 1
        ]
        tokenizer.pad_token = last_reserved_token
    return model, tokenizer
