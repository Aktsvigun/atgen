import json
from pathlib import Path
from datasets import Dataset


def save_labeled_data(
    labeled_data: Dataset,
    labeled_query: Dataset,
    workdir: str | Path,
    iter_dir: str | Path,
    labeled_ids: list[int],
    query_ids: list[int],
):
    Path(iter_dir).mkdir(exist_ok=True)
    with open(iter_dir / "new_labeled_ids.json", "w") as f:
        json.dump(query_ids, f)
    labeled_query.save_to_disk(str(iter_dir / "query"))
    with open(workdir / "labeled_ids.json", "w") as f:
        json.dump(labeled_ids, f)
    labeled_data.save_to_disk(str(workdir / "query"))
