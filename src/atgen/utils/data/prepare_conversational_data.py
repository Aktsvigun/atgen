from itertools import chain
from datasets import Dataset
from omegaconf import DictConfig
from transformers import PreTrainedTokenizer

from .get_preprocess_function import get_preprocess_function


def prepare_conversational_data(
    dataset: Dataset,
    data_config: DictConfig,
    split: str,
    few_shot_examples: Dataset | None = None,
    model_name: str = "kek",
) -> Dataset:
    input_column_name = data_config.input_column_name
    output_column_name = data_config.output_column_name

    if not few_shot_examples:
        few_shot_messages = []
    else:
        few_shot_messages = list(
            chain.from_iterable(
                [
                    [
                        {"role": "user", "content": fs_input},
                        {"role": "assistant", "content": fs_output},
                    ]
                    for (fs_input, fs_output) in zip(
                        few_shot_examples[input_column_name],
                        few_shot_examples[output_column_name]
                        if isinstance(output_column_name, str)
                        else few_shot_examples[output_column_name][0],
                    )
                ]
            )
        )
    # Get appropriate preprocessing function based on all parameters
    preprocess_fn = get_preprocess_function(
        model_name=model_name,
        few_shot_messages=few_shot_messages,
        system_prompt=data_config.system_prompt,
        split=split,
        is_in_conversational_format=data_config.is_in_conversational_format,
        input_column_name=input_column_name,
        output_column_name=output_column_name,
        assistant_response_start=data_config.assistant_response_start,
    )
    dataset = dataset.map(
        preprocess_fn,
        batched=False,
        num_proc=data_config.num_proc,
    )
    return dataset
