from typing import Optional, Literal
from omegaconf import DictConfig

from .constants import REASONING_END_TOKEN

def post_process_generations(
        generations: list[str],
        data_config: DictConfig,
        model_name: Optional[str] = None,
        framework: Literal["vllm", "sglang", "transformers"] = "vllm"
) -> list[str]:
    # Remove assistant response start from transformers generations
    if framework == "transformers" and (ass_resp_start := data_config.assistant_response_start):
        return [ass_resp_start.join(gen.split(ass_resp_start)[1:]).strip() for gen in generations]
    elif model_name and "deepseek-r1" in model_name:
        return [_remove_thinking_part(gen) for gen in generations]
    return generations


def _remove_thinking_part(text: str) -> str:
    return REASONING_END_TOKEN.join(text.split(REASONING_END_TOKEN)[1:]).strip()
