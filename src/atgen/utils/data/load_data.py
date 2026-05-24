import os
from typing import Union

from datasets import load_dataset, load_from_disk, Dataset, DatasetDict
from omegaconf import DictConfig, ListConfig, OmegaConf, open_dict

from .get_output_column_name_for_phase import get_output_column_name_for_phase
from ..constants import OUTPUT_FIELD_PURPOSE_TRAIN, OUTPUT_FIELD_PURPOSE_TEST


def get_effective_data_config(data_config: DictConfig, split: str) -> DictConfig:
    """Return the data config to use for a given split.

    When ``data.eval_dataset`` is set, the test split uses ``eval_dataset.*``
    overlaid on top of the base ``data.*`` config (deep merge). The train split
    is unchanged. This lets you train on dataset A and evaluate on dataset B
    with different schemas / prompts / tasks (e.g. Tulu-3 → MMLU).

    Hydra puts ``config.data`` in struct mode, which would reject overlay keys
    that don't already exist in the base (e.g. ``user_prompt_template`` for
    multi-choice MMLU when the train pool is Tulu-3). We work around this by
    materialising both sides as plain containers, merging, then re-wrapping —
    so the result is an open DictConfig that accepts any field the overlay
    introduces.
    """
    if split != OUTPUT_FIELD_PURPOSE_TEST:
        return data_config
    eval_overlay = data_config.get("eval_dataset")
    if eval_overlay is None or len(eval_overlay) == 0:
        return data_config
    base_dict = OmegaConf.to_container(data_config, resolve=False)
    overlay_dict = OmegaConf.to_container(eval_overlay, resolve=False)
    merged_dict = _deep_merge(base_dict, overlay_dict)
    merged_dict.pop("eval_dataset", None)
    return OmegaConf.create(merged_dict)


def _deep_merge(base: dict, overlay: dict) -> dict:
    """Recursive dict merge: overlay wins on leaves, both sides combine on
    nested dicts. Non-dict overlay values fully replace the base value."""
    if not isinstance(base, dict) or not isinstance(overlay, dict):
        return overlay
    out = dict(base)
    for key, value in overlay.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def get_phase_input_column_name(data_config: DictConfig, split: str):
    """Effective input-column-name for a phase, accounting for the multi-choice
    hack (``processed_input_column_name``) and the eval_dataset overlay."""
    effective = get_effective_data_config(data_config, split)
    return (
        effective.get("processed_input_column_name")
        or effective.input_column_name
    )


def _fetch_dataset(
    dataset_name_or_path: Union[str, list[str]],
    subset_name: str,
    fetch_kwargs: dict | DictConfig,
) -> Dataset:
    # Load a subset of a dataset from HuggingFace
    if isinstance(dataset_name_or_path, (list, ListConfig)):
        dataset = load_dataset(*dataset_name_or_path, **fetch_kwargs)
    # Load local dataset
    elif os.path.exists(dataset_name_or_path):
        # Load a saved on disk dataset
        if os.path.isdir(dataset_name_or_path):
            # Remove `cache_dir` from fetch_kwargs
            fetch_kwargs.pop("cache_dir", None)
            dataset = load_from_disk(dataset_name_or_path, **fetch_kwargs)
        # Load csv dataset
        elif dataset_name_or_path.endswith("csv"):
            dataset = Dataset.from_csv(dataset_name_or_path, **fetch_kwargs)
        # Load json dataset
        elif dataset_name_or_path.endswith("json"):
            dataset = Dataset.from_json(dataset_name_or_path, **fetch_kwargs)
        else:
            raise NotImplementedError(
                f"Unexpected format {dataset_name_or_path.split('.')[-1]} of the dataset. Supported formats: csv, json."
            )
    # Load dataset from HuggingFace
    else:
        dataset = load_dataset(dataset_name_or_path, **fetch_kwargs)

    if isinstance(dataset, DatasetDict):
        return dataset[subset_name]
    else:
        return dataset


def _add_id_column(dataset: Dataset) -> Dataset:
    if "id" in dataset.column_names:
        dataset = dataset.remove_columns(["id"])
    dataset = dataset.add_column("id", list(range(len(dataset))))
    return dataset


def _take_subset(dataset_subset: Dataset, size: int, seed: int) -> Dataset:
    if size >= len(dataset_subset):
        return dataset_subset
    dataset_subset = dataset_subset.shuffle(seed=seed)
    dataset_subset = dataset_subset.select(range(size))
    dataset_subset = dataset_subset.remove_columns(["id"]).add_column(
        "id", list(range(len(dataset_subset)))
    )
    return dataset_subset


def _preprocess_multicolumn_labels_if_needed(
    dataset: Dataset,
    output_column_names: Union[
        DictConfig, ListConfig, dict[str, Union[str, list[str]]], list[str], str
    ],
    data_config: DictConfig,
    phase: str = OUTPUT_FIELD_PURPOSE_TRAIN,
) -> Dataset:
    if isinstance(output_column_names, (list, ListConfig)):
        # Get preprocessed column name
        if phase == OUTPUT_FIELD_PURPOSE_TRAIN:
            new_column_name = data_config.train_output_column_name
        elif phase == OUTPUT_FIELD_PURPOSE_TEST:
            new_column_name = data_config.test_output_column_name
        else:
            raise NotImplementedError(f"Unexpected phase {phase}")
        # Get preprocessed column values
        values = []
        for inst in dataset:
            for col_name in output_column_names:
                inst = inst[col_name]
            values.append(inst)
        dataset = dataset.add_column(new_column_name, values)
    elif isinstance(output_column_names, (dict, DictConfig)):
        for phase, column_name in output_column_names.items():
            dataset = _preprocess_multicolumn_labels_if_needed(
                dataset, column_name, data_config, phase
            )
    # Nothing to preprocess in this case
    elif isinstance(output_column_names, str):
        pass
    else:
        raise NotImplementedError(
            f"Unexpected type {type(output_column_names)} of the output column names."
        )
    return dataset


def load_data(
    data_config: DictConfig,
    split: str,
    cache_dir: str,
    seed: int,
) -> Dataset:
    effective = get_effective_data_config(data_config, split)
    if split == "train":
        subset_name = effective.get("train_split_name", split)
        subset_size = effective.get("train_subset_size")
    elif split == "test":
        subset_name = effective.get("test_split_name", split)
        subset_size = effective.get("test_subset_size")
    else:
        raise NotImplementedError(
            f"Unexpected split {split}; Please specify either `train` or `test`."
        )
    dataset = _fetch_dataset(
        dataset_name_or_path=effective.dataset,
        subset_name=subset_name,
        fetch_kwargs=dict(effective.fetch_kwargs, cache_dir=cache_dir),
    )
    dataset = _preprocess_multicolumn_labels_if_needed(
        dataset=dataset,
        output_column_names=effective.output_column_name,
        data_config=effective,
        phase=split,
    )
    if effective.task == "multi-choice-qa":
        dataset = _preprocess_multi_choice_qa(
            dataset=dataset, data_config=effective, split=split
        )
        # Mirror processed_input_column_name back to the eval_dataset overlay so
        # downstream callers reading data_config can still see it via the helper.
        if split == OUTPUT_FIELD_PURPOSE_TEST and effective is not data_config:
            with open_dict(data_config):
                if data_config.get("eval_dataset") is None:
                    data_config.eval_dataset = {}
                data_config.eval_dataset.processed_input_column_name = (
                    effective.processed_input_column_name
                )
                data_config.eval_dataset.is_in_conversational_format = True
    # Add `id` column to the dataset (practical use) or to train subset (benchmarking)
    dataset = _add_id_column(dataset)
    if subset_size is not None:
        dataset = _take_subset(dataset, subset_size, seed)

    return dataset


def _preprocess_multi_choice_qa(
    dataset: Dataset, data_config: DictConfig, split: str
) -> Dataset:
    alphabet_titled = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    input_column_names = data_config.input_column_name
    options_column_name = input_column_names["options"]
    user_prompt_template = data_config.user_prompt_template
    system_prompt = data_config.system_prompt
    messages = []
    for inst in dataset:
        preprocessed_options = ""
        for option, letter in zip(inst[options_column_name], alphabet_titled):
            preprocessed_options += f"- {letter}. {option}\n"
        user_prompt_kwargs = {
            key: inst[key]
            for key in input_column_names.keys()
            if key != options_column_name
        }
        user_prompt_kwargs["options"] = preprocessed_options
        inst_messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": user_prompt_template.format(**user_prompt_kwargs),
            },
        ]
        if split == "train":
            inst_messages.append({"role": "assistant", "content": inst["answer"]})
        messages.append(inst_messages)
    if "messages" in dataset.column_names:
        dataset = dataset.remove_columns(["messages"])
    dataset = dataset.add_column("messages", messages)
    # Can't directly update `processed_input_column_name` because test data is loaded separately
    OmegaConf.update(
        data_config, "processed_input_column_name", "messages", force_add=True
    )
    data_config.is_in_conversational_format = True
    return dataset
