import json
from pathlib import Path
from datasets import Dataset


def maybe_get_few_shot_examples(
    config,
    labeled_data: Dataset,
    workdir: Path,
) -> tuple[Dataset, Dataset]:
    """
    Selects few-shot examples from labeled data and saves them to a file.

    Args:
        config: Configuration object containing few_shot settings and seed
        labeled_data: The dataset containing labeled examples
        workdir: Working directory to save few-shot IDs
    Returns:
        few_shot_examples: Dataset containing the few-shot examples
        labeled_data: Dataset with the few-shot examples removed
    """
    if config.data.few_shot.count <= 0:
        return [], labeled_data
    elif config.data.few_shot.count >= len(labeled_data):
        # If the number of few-shot examples is greater than or equal to the number of labeled data,
        # labeled data is empty, and few shot examples are the same as 'old' labeled data
        return labeled_data, labeled_data.select(range(0, 0))
    else:
        few_shot_examples = labeled_data.train_test_split(
            train_size=config.data.few_shot.count, shuffle=True, seed=config.seed
        )["train"]

    # Remove the few-shot examples from the labeled data
    few_shot_ids = set(few_shot_examples["id"])
    labeled_data = labeled_data.filter(lambda x: x["id"] not in few_shot_ids)

    # Save the few-shot examples to the workdir
    with open(workdir / "few_shot_ids.json", "w") as f:
        json.dump(few_shot_examples["id"], f)
    few_shot_examples.save_to_disk(str(workdir / "few_shot_examples"))

    return few_shot_examples, labeled_data
