from omegaconf import DictConfig
import warnings

from ..metrics import AVAILABLE_METRICS


def check_required_performance(
    required_performance: DictConfig | None,
) -> DictConfig | None:
    if required_performance is None:
        return None
    checked_dict = {}
    for key, value in required_performance.items():
        if key not in AVAILABLE_METRICS:
            warnings.warn(f"Metric {key} is undefined. It will be skipped.")
        elif not key.startswith("BART") and (value < 0 or value > 1):
            warnings.warn(
                f"Metric {key} can only take values in the range from 0 to 1. Value {value} is impossible. The metric will be skipped."
            )
        else:
            checked_dict[key] = value
    return DictConfig(checked_dict)
