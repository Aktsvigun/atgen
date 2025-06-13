from datasets import Dataset
from omegaconf import DictConfig
import json
from tqdm import tqdm
from openai import OpenAI
import time
from copy import deepcopy
import logging
from pathlib import Path
from shutil import rmtree

from ..base_labeller import BaseLabeler
from ...utils.constants import REASONING_END_TOKEN, MESSAGES_COLUMN_NAME


log = logging.getLogger()


TMP_DIR = "tmp_openai"
INPUT_FILE_PATH = f"{TMP_DIR}/batch_input.jsonl"
OUTPUT_FILE_PATH = f"{TMP_DIR}/batch_output.jsonl"
MAX_NUM_TRIES = 3
UPDATE_TIME_IN_SECONDS = 10  # update time when checking for the completion


# SYSTEM_PROMPT = """
# Act as an experienced ...
# [INSERT]
# Examples for few-shot learning are below.
# """.strip()

# EXAMPLE_1_INPUT = """
# [INSERT]
# """.lstrip()
#
# EXAMPLE_2_INPUT = """
# [INSERT]
# """.lstrip()
#
# EXAMPLE_1_OUTPUT = """
# [INSERT]
# """.strip()
#
# EXAMPLE_2_OUTPUT = """
# [INSERT]
# """.strip()

# Can remove / add examples. If you remove them completely, do not forget to remove the last line from the system prompt
messages_template = [
    # {"role": "system", "content": SYSTEM_PROMPT},
    # {"role": "system", "name": "example_user", "content": EXAMPLE_1_INPUT},
    # {"role": "system", "name": "example_assistant", "content": EXAMPLE_1_OUTPUT},
    {"role": "user", "content": ""},
]


class OpenAILabeller(BaseLabeler):
    def __init__(
        self,
        config: DictConfig,
        output_column_name: str = "output",
        budget: int = 1_000_000,
        base_url: str = None,
    ):
        super().__init__(output_column_name, budget)
        self.config = config
        # Create the OpenAI client
        kwargs = {} if base_url is None else {"base_url": base_url}
        self.client = OpenAI(api_key=self.config.api_key, **kwargs)
        self.mode = config.get("mode")

    def _sync_call(self, dataset: Dataset) -> Dataset:
        # Process each input in the dataset with individual API calls
        data = dataset[MESSAGES_COLUMN_NAME]
        annotations = []
        total_price = 0

        for text in tqdm(data, desc="Processing with OpenAI API"):
            # Prepare messages for this input
            text_messages = deepcopy(messages_template)
            text_messages[-1]["content"] = text

            # Make API call with retries
            for attempt in range(MAX_NUM_TRIES):
                try:
                    response = self.client.chat.completions.create(
                        messages=text_messages, **self.config.parameters
                    )
                    break
                except Exception as e:
                    log.warning(f"Attempt {attempt+1}/{MAX_NUM_TRIES} failed: {str(e)}")
                    if attempt == MAX_NUM_TRIES - 1:
                        log.error(
                            f"Failed to process input after {MAX_NUM_TRIES} attempts"
                        )
                        annotations.append("")
                        continue
                    time.sleep(2**attempt)  # Exponential backoff

            # Extract and store the response
            content = response.choices[0].message.content.strip()
            annotations.append(content)

            # Calculate price for this specific call
            call_price = (
                response.usage.prompt_tokens
                * self.config.price.input_per_1m
                / 1_000_000
                + response.usage.completion_tokens
                * self.config.price.output_per_1m
                / 1_000_000
            )
            total_price += call_price

            # Check if budget is depleted
            if self.budget <= total_price:
                self.is_out_of_budget = True
                # Fill remaining annotations with empty strings
                annotations += ["" for _ in range(len(data) - len(annotations))]
                break

        log.info(f"Labelling price: ${total_price:.2f}")
        self.budget -= total_price

        # Remove reasoning tokens from DeepSeek-R1
        # Remove thinking tokens from DeepSeek-R1
        if "deepseek-r1" in self.config.parameters.model:
            for i, annotation in enumerate(annotations):
                annotations[i] = REASONING_END_TOKEN.join(
                    annotation.split(REASONING_END_TOKEN)[1:]
                ).strip()

        # Add the annotations as a new column in the dataset
        if self.output_column_name in dataset.column_names:
            dataset = dataset.remove_columns(self.output_column_name)

        dataset = dataset.add_column(self.output_column_name, annotations)
        return dataset

    def _batched_call(self, dataset: Dataset) -> Dataset:

        data = dataset[MESSAGES_COLUMN_NAME]
        base_request_kwargs = dict(self.config.parameters)

        Path(TMP_DIR).mkdir(exist_ok=True)

        # Create the file with the instances to process
        with open(INPUT_FILE_PATH, "w") as f:
            for i, text in enumerate(tqdm(data)):
                # Insert it into the `USER_PROMPT`
                text_messages = deepcopy(messages_template)
                # Insert the updated `USER_PROMPT` into the messages
                text_messages[-1]["content"] = text
                # Update kwargs
                request_kwargs = dict(base_request_kwargs)
                request_kwargs["messages"] = text_messages
                request = {
                    "custom_id": f"obs-{i}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": request_kwargs,
                }
                json.dump(request, f)
                if i != len(data) - 1:
                    f.write("\n")

        log.info("Done with processing the data for generation.")

        # Upload the file to OpenAI
        batch_input_file = self.client.files.create(
            file=open(INPUT_FILE_PATH, "rb"), purpose="batch"
        )

        content = None
        unsuccessful_tries = 0
        while content is None and unsuccessful_tries < MAX_NUM_TRIES:
            content = self._make_batched_request(batch_input_file.id)
            unsuccessful_tries += 1
        if content is None:
            log.info(
                "Unable to label data with OpenAI. Please try another labeller. Returning empty outputs."
            )
            return dataset.add_column(
                self.output_column_name, ["" for _ in range(len(dataset))]
            )
        content.write_to_file(OUTPUT_FILE_PATH)

        log.info("Done with generating the data.")

        # Load the results and potentially combine them etc.
        all_outputs = []
        annotations = []
        with open(OUTPUT_FILE_PATH) as f:
            for line in f.readlines():
                all_outputs.append(json.loads(line))
                text = json.loads(line)["response"]["body"]["choices"][0]["message"][
                    "content"
                ].strip()
                annotations.append(text)
        # Calculate the price and write it to the file
        price = self._calculate_price(all_outputs)
        log.info(f"Labelling price: ${price:.2f}")
        rmtree(TMP_DIR)
        self.budget -= price
        if self.budget <= 0:
            self.is_out_of_budget = True
        # TODO: make better in the future
        if self.output_column_name in dataset.column_names:
            dataset = dataset.remove_columns(self.output_column_name)
        dataset = dataset.add_column(self.output_column_name, annotations)
        return dataset

    def __call__(self, dataset: Dataset) -> Dataset:
        if self.mode == "batched":
            return self._batched_call(dataset)
        elif self.mode == "sync":
            return self._sync_call(dataset)
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

    def _make_batched_request(self, input_file_id: str) -> list[str]:
        """
        Create the request to OpenAI. This line will start the generation.
        If at any point you decide to stop it (or you've launched it with
        errors and need to relaunch it), use `client.batches.cancel(create_id)`.
        Make sure to use the same OpenAI API key for your client
        that you used to launch the process.
        """
        create = self.client.batches.create(
            input_file_id=input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"description": "Next AL batch"},
        )
        create_id = create.id
        log.info(f"Process id (for debugging purposes): {create_id}")

        # Loop to check when the process is completed
        status = None
        while status != "completed":
            time.sleep(UPDATE_TIME_IN_SECONDS)
            batch_data = self.client.batches.retrieve(create_id)
            status = batch_data.status
            localtime = time.localtime()
            log.info(
                f"{localtime.tm_hour}:{localtime.tm_min}:{localtime.tm_sec}\t Status: {status}"
            )
            if status == "failed":
                return None

        content = self.client.files.content(batch_data.output_file_id)
        return content

    def _calculate_price(self, outputs):
        return sum(
            [
                x["response"]["body"]["usage"]["prompt_tokens"]
                * self.config.price.input_per_1m
                / 1_000_000
                + x["response"]["body"]["usage"]["completion_tokens"]
                * self.config.price.output_per_1m
                / 1_000_000
                for x in outputs
            ]
        )
