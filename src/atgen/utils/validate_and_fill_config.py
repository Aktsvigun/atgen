import os
from omegaconf import DictConfig, open_dict
import logging

log = logging.getLogger(__name__)


def validate_field(config, field):
    subfields = field.split(".")
    try:
        for subfield in subfields:
            config = config[subfield]
    except Exception:
        raise Exception(f"{field} is not defined in config")


def validate_and_fill_config(config: DictConfig) -> DictConfig:
    with open_dict(config):
        # Common fields
        config.setdefault("name", "I have no name!")
        config.setdefault("seed", 42)
        config.setdefault("cache_dir", "cache")
        config.setdefault("save_model", True)

        # Experiment
        validate_field(config, "al")
        validate_field(config, "al.init_query_size")
        validate_field(config, "al.budget")
        validate_field(config, "al.num_iterations")
        validate_field(config, "al.query_size")
        # TODO choose strategy automatically
        validate_field(config, "al.strategy")
        config.al.setdefault("required_performance", {})
        config.al.setdefault("additional_metrics", [])

        # Data
        validate_field(config, "data")
        validate_field(config, "data.dataset")
        validate_field(config, "data.input_column_name")
        validate_field(config, "data.output_column_name")

        validate_field(config, "data.system_prompt")
        config.data.system_prompt = config.data.system_prompt.strip()
        system_prompt_is_empty = (config.data.system_prompt == "") or (
            config.data.system_prompt is None
        )
        # TODO: check whether we need this
        # # Verify the system prompt is correctly formatted
        # if "{text}" not in config.data.system_prompt and not system_prompt_is_empty:
        #     config.data.system_prompt += "\n{text}"

        config.data.setdefault("train_subset_name", "train")
        config.data.setdefault("test_subset_name", "test")
        config.data.setdefault("train_subset_size", None)
        config.data.setdefault("test_subset_size", None)
        config.data.setdefault("num_proc", 16)
        config.data.setdefault("fetch_kwargs", {})
        config.data.setdefault("few_shot", {})
        config.data.few_shot.setdefault("count", 0)
        config.data.few_shot.setdefault("separator", "\n\n")

        # Model
        validate_field(config, "model")
        validate_field(config, "model.checkpoint")
        config.model.setdefault("quantize", False)
        config.model.setdefault("model_max_length", None)
        config.model.setdefault(
            "separator",
            (
                "<|assistant|>\n"
                if "stable" in config.model.checkpoint
                else "<|assistant|>"
            ),
        )

        # Labeler
        validate_field(config, "labeller")
        validate_field(config, "labeller.type")
        config.labeller.setdefault("budget", None)

        # Training
        config.setdefault("training", {})
        config.training.setdefault("num_epochs", 1)
        config.training.setdefault("train_batch_size", 64)
        config.training.setdefault("eval_batch_size", 64)
        config.training.setdefault("gradient_accumulation_steps", 2)
        config.training.setdefault("lr", 0.00003)
        config.training.setdefault("warmup_ratio", 0.03)
        config.training.setdefault("weight_decay", 0.01)
        config.training.setdefault("max_grad_norm", 1.0)
        config.training.setdefault("early_stopping_patience", 5)

        # Inference
        config.setdefault("inference", {})
        config.inference.setdefault("batch_size", 64)
        config.inference.setdefault("framework", "vllm")
        config.inference.setdefault("max_new_tokens", None)

        # Evaluation
        config = _validate_and_fill_eval_config(config)
    return config


def _validate_and_fill_eval_config(config: DictConfig) -> DictConfig:
    provider = config.evaluation.provider.lower()
    # Check if the API key is provided
    if not (api_key := config.evaluation.api_key):
        api_key = os.environ.get("EVALUATION_API_KEY")
    if not api_key:
        if provider == "openai":
            api_key = os.environ.get("OPENAI_API_KEY")
        elif provider == "anthropic":
            api_key = os.environ.get("ANTHROPIC_API_KEY")
        elif provider == "openrouter":
            api_key = os.environ.get("OPENROUTER_API_KEY")
        else:
            api_key = os.environ.get("API_KEY")
        if not api_key:
            api_key = config.labeller.get("api_key")
            provider = config.labeller.get("provider")
            if not api_key and any(
                metric.startswith("deepeval")
                for metric in config.evaluation.additional_metrics
            ):
                log.warning(
                    "API key is required for DeepEval metrics. "
                    "Set it as an environment variable `EVALUATION_API_KEY` or pass it as a parameter."
                )

    # Set default base URL if not provided
    if (base_url := config.evaluation.base_url) is None:
        if provider == "openai":
            base_url = "https://api.openai.com/v1"
        elif provider == "anthropic":
            base_url = "https://api.anthropic.com/v1"
        elif provider == "openrouter":
            base_url = "https://openrouter.ai/api/v1"
        elif provider == config.labeller.get("provider"):
            base_url = config.labeller.get("base_url")
        else:
            log.error(
                f"Base URL not provided for the provider {provider}. Deepeval metrics will not be calculated."
            )
            return config

    config["evaluation"]["provider"] = provider
    config["evaluation"]["api_key"] = api_key
    config["evaluation"]["base_url"] = base_url
    return config
