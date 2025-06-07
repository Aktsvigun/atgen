from typing import Union

from datasets import Dataset


def _get_output_field(data: Union[Dataset, dict], output_field: list[str]) -> str:
    """Access nested dictionary using a tuple path."""
    result = data
    for key in output_field:
        result = result[key]
    return result