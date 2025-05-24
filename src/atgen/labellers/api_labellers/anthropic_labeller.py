from datasets import Dataset
from omegaconf import DictConfig
from tqdm import tqdm
from anthropic import Anthropic
import logging

from ..base_labeller import BaseLabeler
from ...utils.constants import MESSAGES_COLUMN_NAME

log = logging.getLogger()


class AnthropicLabeller(BaseLabeler):
    def __init__(
        self,
        config: DictConfig,
        output_column_name: str = "output",
        budget: int = 1_000_000,
    ):
        super().__init__(output_column_name, budget)
        self.config = config
        # Create the Anthropic client
        self.client = Anthropic(api_key=self.config.api_key)
        self.mode = config.get("mode")

    def __call__(self, dataset: Dataset) -> Dataset:
        # TODO: add batched mode

        data = dataset[MESSAGES_COLUMN_NAME]
        base_request_kwargs = dict(self.config.parameters)

        annotations = []
        price = 0
        for text in tqdm(data):
            request_kwargs = dict(base_request_kwargs)
            request_kwargs["messages"] = [{"role": "user", "content": text}]
            output = self.client.messages.create(**request_kwargs)
            price += self._calculate_price(output)
            annotations.append(output.content[0].text.strip())
            if self.budget <= price:
                self.is_out_of_budget = True
                annotations += ["" for _ in range(len(dataset) - len(annotations))]
                break

        log.info(f"Labelling price: ${price:.2f}")
        self.budget -= price
        # TODO: make better in the future
        if self.output_column_name in dataset.column_names:
            dataset = dataset.remove_columns(self.output_column_name)
        dataset = dataset.add_column(self.output_column_name, annotations)
        return dataset

    def _calculate_price(self, output):
        return (
            output.usage.input_tokens * self.config.price.input_per_1m / 1_000_000
            + output.usage.output_tokens * self.config.price.output_per_1m / 1_000_000
        )
