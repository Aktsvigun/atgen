import logging
from time import time
from typing import List, Dict, Literal, Optional

from omegaconf import DictConfig

# The factory is now the single entry point to get a metric runner.
# We'll assume it's located in atgen/metrics/factory.py
from atgen.metrics.factory import MetricFactory

log = logging.getLogger(__name__)

# This mapping replaces the large if/elif/else block. It's declarative and easy to modify.
TASK_TO_DEFAULT_METRICS = {
    "summarization": ["exact_match", "sacrebleu", "bleu", "rouge", "word_length"],
    "open-qa": ["exact_match"],
    "multi-choice-qa": ["exact_match"],
    "translation": ["exact_match", "sacrebleu", "bleu", "word_length"],
    "math": ["exact_match_math"],
}

# Define which metrics require the 'original_texts' (sources) input.
# This avoids passing it to metrics that don't need it.
METRICS_REQUIRING_SOURCE = {"bartscore", "alignscore", "deepeval"}


def compute_metrics(
    generated_texts: List[str],
    reference_texts: Optional[List[str]],
    original_texts: Optional[List[str]],
    task: Literal["summarization", "open-qa", "multi-choice-qa", "translation", "math"],
    config: DictConfig,
    cache_dir: Optional[str] = None,
) -> Dict[str, float]:

    if task not in TASK_TO_DEFAULT_METRICS:
        raise NotImplementedError(f"Task '{task}' is not implemented in TASK_TO_DEFAULT_METRICS.")

    # 1. Determine the full list of metrics to run
    base_metrics = TASK_TO_DEFAULT_METRICS.get(task, [])
    additional_metrics = list(config.get("additional_metrics", []))
    # Use a set to handle duplicates, then sort for predictable execution order
    metrics_to_calculate = sorted(list(set(base_metrics + additional_metrics)))

    if not metrics_to_calculate:
        log.warning("No metrics specified for calculation. Returning empty results.")
        return {}

    # 2. Instantiate the factory that will build our metric runners
    metric_factory = MetricFactory(config, cache_dir=cache_dir)
    
    final_results = {}
    time_dict = {}

    log.info(f"Starting evaluation for task '{task}' with metrics: {', '.join(metrics_to_calculate)}")

    # 3. Loop through metrics, delegate calculation, and collect results
    for metric_name in metrics_to_calculate:
        try:
            log.info(f"--> Calculating metric: {metric_name}")
            start_time = time()

            # The factory creates the appropriate metric runner with its specific config
            metric_runner = metric_factory.get_metric(metric_name)

            # Prepare arguments for the metric's standardized 'compute' method
            compute_kwargs = {
                "predictions": generated_texts,
                "references": reference_texts,
            }
            
            # Conditionally add 'sources' if the metric is known to require it
            # This relies on the METRICS_REQUIRING_SOURCE set defined above.
            if any(m in metric_name for m in METRICS_REQUIRING_SOURCE):
                 compute_kwargs["sources"] = original_texts

            # Each metric's compute method is responsible for its own logic
            # and should return a dictionary of scores.
            metric_scores = metric_runner.compute(**compute_kwargs)
            
            final_results.update(metric_scores)
            time_dict[f"time_{metric_name}"] = time() - start_time
            
            log.info(f"<-- Finished {metric_name} in {time_dict[f'time_{metric_name}']:.2f}s.")

        except ImportError as e:
            log.warning(f"Could not compute metric '{metric_name}' due to a missing dependency: {e}. Please install the required package and try again.")
        except Exception as e:
            # Catch other errors, log them, and continue to the next metric
            log.error(f"Failed to compute metric '{metric_name}': {e}", exc_info=True)

    # 4. Finalize and return the results
    final_results.update(time_dict)
    
    # Sort the final dictionary by key for consistent, readable output
    return dict(sorted(final_results.items()))
