from typing import Literal

# Available DeepEval metrics
DEEPEVAL_METRICS = [
    "deepeval_answer_relevance",
    "deepeval_faithfulness",
    "deepeval_summarization",
    "deepeval_prompt_alignment",
]

# Available API models by provider
# TODO: somehow update periodically
API_MODELS = {
    "openai": ["gpt-4o", "gpt-4o-mini", "gpt-4.5", "o1-preview"],
    "anthropic": [
        "claude-3.7-sonnet:thinking",
        "claude-3.7-sonnet",
        "claude-3-5-sonnet",
        "claude-3-opus",
        "claude-3-sonnet",
        "claude-3-haiku",
    ],
    "openrouter": [
        "openai/gpt-4o-2024-11-20",
        "google/gemini-2.0-flash-001",
        "anthropic/claude-3.5-sonnet",
        "openai/gpt-4o-mini",
        "mistralai/mistral-nemo",
        "meta-llama/llama-3.1-70b-instruct",
    ],
}


def get_available_models(
    provider: Literal["openai", "anthropic", "openrouter"] = "openai",
):
    """
    Get a list of all available models for a specific provider.

    Args:
        provider: The provider to get models for Literal["openai", "anthropic", "openrouter"]

    Returns:
        List of available models for the specified provider
    """
    provider = provider.lower()
    if provider in API_MODELS:
        return API_MODELS[provider]
    return []


def get_available_metrics():
    """
    Get a list of all available metrics.

    Returns:
        List of available metrics
    """
    basic_metrics = [
        "bartscore",
        "alignscore",
    ]

    return basic_metrics + DEEPEVAL_METRICS
