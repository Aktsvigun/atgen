from datasets import Dataset, load_from_disk
from pathlib import Path
from time import sleep
from shutil import rmtree
import logging

from .base_labeller import BaseLabeler


log = logging.getLogger()


class HumanLabeler(BaseLabeler):
    def __init__(
        self,
        output_column_name: str = "output",
        budget: int = 1_000_000,
        workdir: str | Path = "tmp",
    ):
        super().__init__(output_column_name, budget)
        self.workdir = Path(workdir)

    def __call__(self, dataset: Dataset) -> Dataset:
        if self.output_column_name in dataset.column_names:
            dataset = dataset.remove_columns([self.output_column_name])
        dataset.save_to_disk(Path(self.workdir) / "dataset_to_annotate")
        annotated_dataset_path = Path(self.workdir) / "annotated_query"
        if annotated_dataset_path.exists():
            rmtree(annotated_dataset_path)
        log.info("Waiting for the texts to be annotated...")
        while True:
            if annotated_dataset_path.exists():
                break
            sleep(2)
        log.info("Annotation finished. Continuing...")
        dataset = load_from_disk(str(annotated_dataset_path))
        # TODO: fix
        self.budget -= 1 * len(dataset)
        if self.budget < 0:
            self.is_out_of_budget = True

        dataset = dataset.rename_column("annotation", self.output_column_name)
        return dataset
