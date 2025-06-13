from typing import Union
from omegaconf import DictConfig, ListConfig
from datasets import Dataset

from .base_labeller import BaseLabeler


# Labels rows by using their label (should be used when evaluating strategies)
class GoldenLabeler(BaseLabeler):
    def __call__(self, dataset: Dataset) -> Dataset:
        _check_output_column_in_dataset(
            column_names=self.output_column_name, dataset=dataset
        )
        return dataset


def _check_output_column_in_dataset(
    column_names: Union[
        DictConfig, ListConfig, dict[str, Union[str, list[str]]], list[str], str
    ],
    dataset: Union[Dataset, dict[str, Union[str, dict[str, str]]]],
) -> None:
    if isinstance(column_names, str):
        assert (
            column_names in dataset.column_names
        ), f"Column {column_names} with labels was not found in dataset"
    elif isinstance(column_names, (ListConfig, list)):
        if isinstance(dataset, Dataset):
            assert (
                column_names[0] in dataset.column_names
            ), f"Column {column_names[0]} with labels was not found in dataset"
            dataset = dataset[0]
        else:
            assert (
                column_names[0] in dataset.keys()
            ), f"Column {column_names[0]} with labels was not found in dataset"
        if len(column_names) > 1:
            _check_output_column_in_dataset(
                column_names=column_names[1:], dataset=dataset[column_names[0]]
            )
    elif isinstance(column_names, (DictConfig, dict)):
        for purpose, column_name in column_names.items():
            _check_output_column_in_dataset(column_names=column_name, dataset=dataset)
