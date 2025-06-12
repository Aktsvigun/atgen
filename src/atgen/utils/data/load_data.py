import os
from typing import Union

from datasets import load_dataset, load_from_disk, Dataset, DatasetDict
from omegaconf import DictConfig, ListConfig

from .get_output_column_name_for_phase import get_output_column_name_for_phase
from ..constants import OUTPUT_FIELD_PURPOSE_TRAIN, OUTPUT_FIELD_PURPOSE_TEST


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
    if split == "train":
        subset_name = data_config.get("train_split_name", split)
        subset_size = data_config.get("train_subset_size")
    elif split == "test":
        subset_name = data_config.get("test_split_name", split)
        subset_size = data_config.get("test_subset_size")
    else:
        raise NotImplementedError(
            f"Unexpected split {split}; Please specify either `train` or `test`."
        )
    dataset = _fetch_dataset(
        dataset_name_or_path=data_config.dataset,
        subset_name=subset_name,
        fetch_kwargs=dict(data_config.fetch_kwargs, cache_dir=cache_dir),
    )
    dataset = _preprocess_multicolumn_labels_if_needed(
        dataset=dataset,
        output_column_names=data_config.output_column_name,
        data_config=data_config,
        phase=split,
    )
    # Add `id` column to the dataset (practical use) or to train subset (benchmarking)
    dataset = _add_id_column(dataset)
    if subset_size is not None:
        dataset = _take_subset(dataset, subset_size, seed)

    return dataset
