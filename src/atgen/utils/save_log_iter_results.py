import json
from pathlib import Path
import logging
from typing import Optional

from omegaconf import DictConfig
from transformers import PreTrainedModel

from atgen.utils.combine_results import combine_results


log = logging.getLogger(__name__)


def save_log_iter_results(
    config: DictConfig,
    workdir: str,
    iter_dir: str,
    metrics: dict,
    generations: list,
    al_iter: int,
    train_result: dict,
    model: Optional[PreTrainedModel] = None,
):
    log.info(metrics)
    with open(iter_dir / "train_result.json", "w") as f:
        json.dump(train_result, f)
    with open(iter_dir / "metrics.json", "w") as f:
        json.dump(metrics, f)
    with open(iter_dir / "generations.json", "w") as f:
        json.dump(generations, f)
    combine_results(workdir, al_iter)

    log.info(f"Iteration {al_iter}: saving the trained model...")
    if config.save_model and model is not None:
        model.save_pretrained(workdir / "model.bin")
