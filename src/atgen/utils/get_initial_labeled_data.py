from omegaconf import DictConfig
from datasets import Dataset
import logging
from ..labellers import BaseLabeler
from ..utils.data.prepare_conversational_data import prepare_conversational_data

log = logging.getLogger()


log = logging.getLogger()


def get_initial_labeled_data_with_few_shot(
    config: DictConfig,
    init_query_size: int,
    unlabeled_data: Dataset,
    labeller: BaseLabeler,
    model_name: str = "",
) -> tuple[Dataset, list[int]]:

    # Fix: Ensure we don't try to select more examples than available
    available_size = len(unlabeled_data)
    if init_query_size > available_size:
        log.warning(
            f"Requested {init_query_size} examples for initial labeling, but only {available_size} are available. "
            f"Using all available examples."
        )
        init_query_size = available_size

    random_data_to_label = unlabeled_data.train_test_split(
        train_size=init_query_size, shuffle=True, seed=config.seed
    )["train"]

    random_data_to_label = prepare_conversational_data(
        dataset=random_data_to_label,
        data_config=config.data,
        split="train",
        model_name=model_name,
    )

    labeled_data = labeller(random_data_to_label)
    if labeller.is_out_of_budget:
        log.info("Labeler ran out of budget when labeling the initial query.")

    return labeled_data, labeled_data["id"]
