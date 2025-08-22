from omegaconf import DictConfig
from pathlib import Path
from . import (
    CustomLLMLabeller,
    HumanLabeler,
    GoldenLabeler,
    OpenAILabeller,
    AnthropicLabeller,
)


def get_labeller(
    config: DictConfig,
    output_column_name: str = "output",
    budget: int = 1_000_000,
    workdir: str | Path = "tmp",
    **kwargs,
):
    if config.type == "custom_llm":
        return CustomLLMLabeller(
            config, output_column_name=output_column_name, budget=budget, **kwargs
        )
    elif config.type == "api_llm":
        provider = config.provider.lower()
        if provider == "openai":
            return OpenAILabeller(
                config, output_column_name=output_column_name, budget=budget
            )
        elif provider == "anthropic":
            return AnthropicLabeller(
                config, output_column_name=output_column_name, budget=budget
            )
        elif provider == "custom":
            return OpenAILabeller(
                config,
                output_column_name=output_column_name,
                budget=budget,
                base_url=config.base_url,
            )
        else:
            raise NotImplementedError(f"Provider {config.provider} is not supported!")
    elif config.type == "golden":
        return GoldenLabeler(output_column_name=output_column_name, budget=budget)
    elif config.type == "human":
        return HumanLabeler(
            output_column_name=output_column_name, budget=budget, workdir=workdir
        )
    else:
        raise NotImplementedError
