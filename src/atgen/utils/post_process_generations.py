from typing import Optional
from omegaconf import DictConfig

from .constants import REASONING_END_TOKEN

def post_process_generations(generations: list[str], data_config: DictConfig, model_name: Optional[str] = None) -> list[str]:
    if data_config.assistant_response_start:
        return [_remove_thinking_part(generation) for generation in generations]
    # Remove reasoning tokens from DeepSeek-R1
    elif model_name and "deepseek-r1" in model_name:
        return [_remove_thinking_part(generation) for generation in generations]
    return generations


def _remove_thinking_part(text: str) -> str:
    return REASONING_END_TOKEN.join(text.split(REASONING_END_TOKEN)[1:]).strip()
