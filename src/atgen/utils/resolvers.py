"""
Custom Hydra resolvers for configuration calculations
"""

import secrets

from omegaconf import OmegaConf


def random_suffix(num_bytes: int = 3) -> str:
    """
    Returns a short random hex string used to disambiguate run directories
    when multiple runs start in the same second (e.g. parallel GPU jobs).
    """
    return secrets.token_hex(int(num_bytes))


def multiply_with_few_shot(input_max_length: int, few_shot_count: int) -> int:
    """
    Calculates model max length by multiplying input_max_length by (1 + few_shot_count)

    Args:
        input_max_length: The base input max length
        few_shot_count: Number of few shot examples

    Returns:
        The calculated model max length
    """
    return input_max_length * (1 + few_shot_count)


def to_string(model_name: str):
    """
    Converts a model name to a string
    """
    return model_name.replace("/", "__")


def register_resolvers() -> None:
    """Register all custom resolvers with OmegaConf"""
    # Register resolvers only if they are not already registered
    resolvers_to_register = {
        "multiply_with_few_shot": multiply_with_few_shot,
        "to_string": to_string,
        # Cached so every reference to ${random_suffix:} in one config
        # resolves to the same value within a single run.
        "random_suffix": (random_suffix, True),
    }

    for name, resolver_fn in resolvers_to_register.items():
        use_cache = False
        if isinstance(resolver_fn, tuple):
            resolver_fn, use_cache = resolver_fn
        try:
            OmegaConf.register_new_resolver(
                name, resolver_fn, use_cache=use_cache
            )
        except ValueError:
            # Skip registration if the resolver is already registered
            pass
