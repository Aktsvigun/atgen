from typing import Union
from omegaconf import DictConfig, ListConfig

from ..constants import OUTPUT_FIELD_PURPOSE_TRAIN


def get_output_column_name_for_phase(
    output_column_name: Union[
        DictConfig, ListConfig, dict[str, Union[str, list[str]]], list[str], str
    ],
    purpose: str = OUTPUT_FIELD_PURPOSE_TRAIN,
) -> str:
    if isinstance(output_column_name, (list, ListConfig)):
        return "_".join(output_column_name)
    elif isinstance(output_column_name, (dict, DictConfig)):
        return "_".join(output_column_name[purpose])
    elif isinstance(output_column_name, str):
        return output_column_name
    else:
        raise NotImplementedError(
            f"Unexpected type {type(output_column_name)} of the output column name."
        )
