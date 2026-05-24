from functools import partial
from pathlib import Path
import warnings
from typing import Optional, Union, List
import logging

import torch
from torch.optim import Adam
from datasets import Dataset
from omegaconf import DictConfig
from peft import PeftModel
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerFast,
    ProcessorMixin,
    EarlyStoppingCallback,
)
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM
from trl.data_utils import (
    is_conversational,
    maybe_apply_chat_template,
    maybe_convert_to_chatml,
)
from trl.trainer.utils import ConstantLengthDataset
from accelerate import PartialState

if torch.cuda.is_available():
    from unsloth import is_bfloat16_supported
else:

    def is_bfloat16_supported():
        return False


from .constants import APPROX_NUM_SYSTEM_TOKENS_PER_MESSAGE


TEXT_FIELD = "text"

logger = logging.getLogger(__name__)


def _get_training_args(
    hyperparam_config: DictConfig,
    train_data: Dataset,
    eval_data: Dataset | None,
    seed: int,
    output_dir: str,
) -> SFTConfig:
    if eval_data is None:
        evaluation = "no"
    else:
        evaluation = "epoch"
    return SFTConfig(
        num_train_epochs=hyperparam_config.num_epochs,
        per_device_train_batch_size=min(
            hyperparam_config.train_batch_size, len(train_data)
        ),
        per_device_eval_batch_size=hyperparam_config.eval_batch_size,
        gradient_accumulation_steps=hyperparam_config.gradient_accumulation_steps,
        learning_rate=hyperparam_config.lr,
        weight_decay=hyperparam_config.weight_decay,
        max_grad_norm=hyperparam_config.max_grad_norm,
        warmup_ratio=hyperparam_config.warmup_ratio,
        max_seq_length=hyperparam_config.model_max_length,
        packing=hyperparam_config.packing,
        dataset_num_proc=hyperparam_config.dataset_num_proc,
        lr_scheduler_type=hyperparam_config.lr_scheduler_type,
        gradient_checkpointing=hyperparam_config.gradient_checkpointing,
        eval_strategy=evaluation,
        dataset_text_field=TEXT_FIELD,
        logging_strategy=evaluation,
        save_strategy=evaluation,
        load_best_model_at_end=(True if eval_data is not None else False),
        seed=seed,
        output_dir=output_dir,
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),
        save_total_limit=1,
        report_to="none",
        include_num_input_tokens_seen=True,
        remove_unused_columns=True,
    )


class DataCollatorForLastCompletionOnlyLM(DataCollatorForCompletionOnlyLM):
    """
    Data collator that extends DataCollatorForCompletionOnlyLM to only train on the last assistant message.
    It ensures that only the final assistant response will contribute to the loss, while all previous messages
    (including earlier assistant responses) are masked with the ignore_index.

    """

    def torch_call(self, examples):
        batch = super(DataCollatorForCompletionOnlyLM, self).torch_call(examples)

        for i in range(batch["input_ids"].shape[0]):
            # Find all occurrences of both templates
            response_token_ids_idxs = self._find_token_ids_indices(
                batch["input_ids"][i], self.response_token_ids
            )

            if not response_token_ids_idxs:
                # Log warning and skip this example by masking everything
                logger.warning(
                    f"Could not find response key {self.response_token_ids} in example {i}. "
                    f"Masking entire example to exclude from loss."
                )
                batch["labels"][i, :] = self.ignore_index
                continue

            # If there's an instruction template, find those too
            human_token_ids_idxs = []
            if self.instruction_token_ids is not None:
                human_token_ids_idxs = self._find_token_ids_indices(
                    batch["input_ids"][i], self.instruction_token_ids
                )

            # Ensure sequence ends with instruction_template + last assistant response
            last_response_idx = response_token_ids_idxs[-1]

            # Find the user message before the last assistant response
            preceding_human_idxs = [
                idx for idx in human_token_ids_idxs if idx < last_response_idx
            ]
            if not preceding_human_idxs:
                # Log warning and skip this example by masking everything
                logger.warning(
                    f"No user message found before the last assistant response in example {i}. "
                    f"Masking entire example to exclude from loss."
                )
                batch["labels"][i, :] = self.ignore_index
                continue

            # Find any content after the last assistant response
            next_human_idxs = [
                idx for idx in human_token_ids_idxs if idx > last_response_idx
            ]

            # If there's content after the last assistant response, truncate it
            if next_human_idxs:
                end_idx = next_human_idxs[0]
                # Truncate the sequence
                batch["input_ids"][i, end_idx:] = self.tokenizer.pad_token_id
                batch["attention_mask"][i, end_idx:] = 0

            # Set all labels to ignore_index as default
            batch["labels"][i, :] = self.ignore_index

            # Only unmask the last assistant response
            content_start_idx = last_response_idx + len(self.response_token_ids)
            end_idx = batch["input_ids"].shape[1]

            # Find the actual end of the response (before padding)
            if self.tokenizer.pad_token_id is not None:
                padding_mask = batch["input_ids"][i] == self.tokenizer.pad_token_id
                if padding_mask.any():
                    end_idx = padding_mask.nonzero()[0].item()

            # Unmask only the content after the template and before the end
            batch["labels"][i, content_start_idx:end_idx] = batch["input_ids"][
                i, content_start_idx:end_idx
            ]

            # Always mask padding tokens
            if self.tokenizer.pad_token_id is not None:
                padding_mask = batch["input_ids"][i] == self.tokenizer.pad_token_id
                batch["labels"][i, padding_mask] = self.ignore_index

        return batch

    def _find_token_ids_indices(self, token_ids, pattern):
        """
        Find all starting indices of the pattern in the token_ids
        """
        indices = []
        pattern_len = len(pattern)

        # Convert to list if it's a tensor
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()

        # Scan through the sequence to find pattern matches
        for i in range(len(token_ids) - pattern_len + 1):
            if token_ids[i : i + pattern_len] == pattern:
                indices.append(i)

        return indices


def _get_response_instruction_templates(
    tokenizer: Union[PreTrainedTokenizerFast, ProcessorMixin], model_config: Optional[DictConfig] = None
) -> tuple[str, str]:
    """
    Determine the response and instruction templates based on the tokenizer and model config.

    Args:
        tokenizer: The tokenizer or processor to determine templates from
        model_config: Optional model configuration that may contain template information

    Returns:
        Tuple of (response_template, instruction_template)
    """
    if isinstance(tokenizer, ProcessorMixin):
        tokenizer = tokenizer.tokenizer
    if "gemma" in tokenizer.name_or_path.lower():
        response_template = "<start_of_turn>model\n"
        instruction_template = "<start_of_turn>user\n"
    elif "qwen3" in tokenizer.name_or_path.lower():
        response_template = "<|im_start|>assistant\n"  # don't include empty reasoning
        instruction_template = "<|im_start|>user\n"
    elif "llama" in tokenizer.name_or_path.lower():
        response_template = "<|start_header_id|>assistant<|end_header_id|>\n\n"
        instruction_template = "<|start_header_id|>user<|end_header_id|>\n\n"
    elif model_config is not None:
        response_template = model_config.get("response_template")
        instruction_template = model_config.get("instruction_template")
        if response_template is None or instruction_template is None:
            raise ValueError(
                "response_template and instruction_template must be provided for non-Gemma, Qwen, or Llama models"
            )
    else:
        raise ValueError(
            "Config not provided. response_template and instruction_template must be provided for non-Gemma, Qwen, or Llama models"
        )

    return response_template, instruction_template


def _get_data_collator(
    tokenizer: PreTrainedTokenizerFast, model_config: Optional[DictConfig] = None
) -> DataCollatorForLastCompletionOnlyLM:
    # Derive the response/instruction markers as token ids directly from the
    # tokenizer's own chat template — this matches how the trainer tokenizes the
    # data and works regardless of whether the model uses ``<start_of_turn>``,
    # ``<|turn>``, ``<|im_start|>``, ``<|start_header_id|>``, etc. Hardcoded
    # marker strings break the moment a model ships with a new role-marker
    # syntax (e.g. Gemma 4 uses ``<|turn>...<turn|>``).
    response_template = _probe_response_marker(tokenizer)
    instruction_template = _probe_instruction_marker(tokenizer)
    if response_template is None or instruction_template is None:
        response_template, instruction_template = _get_response_instruction_templates(
            tokenizer, model_config
        )
    return DataCollatorForLastCompletionOnlyLM(
        tokenizer=tokenizer,
        response_template=response_template,
        instruction_template=instruction_template,
    )


def _probe_response_marker(
    tokenizer: PreTrainedTokenizerFast,
) -> Optional[List[int]]:
    if isinstance(tokenizer, ProcessorMixin):
        tokenizer = tokenizer.tokenizer
    try:
        msgs = [{"role": "user", "content": "X"}]
        with_p = list(
            tokenizer.apply_chat_template(
                msgs, tokenize=True, add_generation_prompt=True
            )
        )
        without_p = list(
            tokenizer.apply_chat_template(
                msgs, tokenize=True, add_generation_prompt=False
            )
        )
    except Exception:
        return None
    if len(with_p) <= len(without_p) or with_p[: len(without_p)] != without_p:
        return None
    return with_p[len(without_p):]


def _probe_instruction_marker(
    tokenizer: PreTrainedTokenizerFast,
) -> Optional[List[int]]:
    if isinstance(tokenizer, ProcessorMixin):
        tokenizer = tokenizer.tokenizer
    placeholder = "USRMARKERPROBEUNIQUEZZZ"
    msgs_short = [
        {"role": "user", "content": "A"},
        {"role": "assistant", "content": "B"},
    ]
    msgs_full = msgs_short + [{"role": "user", "content": placeholder}]
    try:
        rendered_full = tokenizer.apply_chat_template(msgs_full, tokenize=False)
        rendered_short = tokenizer.apply_chat_template(msgs_short, tokenize=False)
    except Exception:
        return None
    if not rendered_full.startswith(rendered_short):
        return None
    pos_placeholder = rendered_full.find(placeholder)
    if pos_placeholder < 0:
        return None

    try:
        enc = tokenizer(
            rendered_full, add_special_tokens=False, return_offsets_mapping=True
        )
    except (TypeError, ValueError, NotImplementedError):
        return None
    ids_full = list(enc["input_ids"])
    offsets = list(enc["offset_mapping"])
    try:
        ids_short = list(
            tokenizer(rendered_short, add_special_tokens=False).input_ids
        )
    except Exception:
        return None
    if ids_full[: len(ids_short)] != ids_short:
        return None

    marker_tokens: List[int] = []
    for i in range(len(ids_short), len(ids_full)):
        s, e = offsets[i]
        # Special tokens often have (0, 0) offsets and contribute no rendered chars;
        # always include them. For anything else, stop once we cross into the
        # placeholder content.
        if (s, e) != (0, 0) and s >= pos_placeholder:
            break
        marker_tokens.append(ids_full[i])
    return marker_tokens or None


def _get_train_eval_datasets(
    train_data: Dataset,
    eval_data: Dataset | None,
    tokenizer: PreTrainedTokenizerFast,
    data_collator: DataCollatorForLastCompletionOnlyLM,
) -> tuple[Dataset, Dataset | None]:
    orig_train_data_len = len(train_data)
    train_data = train_data.map(
        _dataset_to_chat_template,
        batched=True,
        fn_kwargs={"tokenizer": tokenizer, "data_collator": data_collator},
    ).filter(lambda x: x[TEXT_FIELD] != "")
    logger.warning(
        f"Truncated {orig_train_data_len - len(train_data)} examples from train set."
    )
    if eval_data is not None:
        orig_eval_data_len = len(eval_data)
        eval_data = eval_data.map(
            _dataset_to_chat_template,
            batched=True,
            fn_kwargs={"tokenizer": tokenizer, "data_collator": data_collator},
        ).filter(lambda x: x[TEXT_FIELD] != "")
        logger.warning(
            f"Truncated {orig_eval_data_len - len(eval_data)} examples from eval set."
        )
    return train_data, eval_data


def _dataset_to_chat_template(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerFast,
    data_collator: DataCollatorForLastCompletionOnlyLM,
) -> dict[str, list[str]]:
    tokenized_texts = _formatting_fn(dataset, tokenizer)
    texts = []
    for text in tokenized_texts:
        # Check if the tokenized response template is present
        # This ensures we only keep examples that will work with the collator
        tokenized = tokenizer(text, truncation=True, return_tensors="pt")
        input_ids = tokenized["input_ids"][0].tolist()

        # Check if response_token_ids exist in the tokenized input
        response_token_ids = data_collator.response_token_ids
        found = False
        for i in range(len(input_ids) - len(response_token_ids) + 1):
            if input_ids[i : i + len(response_token_ids)] == response_token_ids:
                found = True
                break

        if not found:
            text = ""
        texts.append(text.strip())
    return {TEXT_FIELD: texts}


# TODO: move this after each query is obtained
def _formatting_fn(examples, tokenizer: PreTrainedTokenizerFast):
    return tokenizer.apply_chat_template(
        examples["messages"],
        truncation=True,
        tokenize=False,
        add_generation_prompt=False,
    )


def get_trainer(
    config: DictConfig,
    model: PreTrainedModel | PeftModel,
    tokenizer: PreTrainedTokenizerFast | ProcessorMixin,
    train_data: Dataset,
    eval_data: Dataset | None,
    output_dir: str | Path,
    seed: int = 42,
) -> SFTTrainer:
    train_args = _get_training_args(
        config.training.hyperparameters, train_data, eval_data, seed, output_dir
    )
    if isinstance(tokenizer, ProcessorMixin):
        processor = tokenizer
        tokenizer = processor.tokenizer
    else:
        processor = None
    data_collator = _get_data_collator(tokenizer, config.model)
    callbacks = (
        [
            EarlyStoppingCallback(
                early_stopping_patience=config.training.hyperparameters.early_stopping_patience
            )
        ]
        if eval_data is not None
        else []
    )
    # TODO: move this after each query is obtained
    train_dataset, eval_dataset = _get_train_eval_datasets(
        train_data=train_data,
        eval_data=eval_data,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    return SFTTrainer(
        model=model,
        processing_class=tokenizer if processor is None else processor,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=train_args,
        data_collator=data_collator,
        callbacks=callbacks,
        formatting_func=partial(_formatting_fn, tokenizer=tokenizer),
    )
