import os
import logging
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd
import json
import numpy as np
from transformers import PreTrainedModel, PreTrainedTokenizer
from shutil import rmtree
from torch import cuda
import gc
from torch import cuda

from .constants import DEFAULT_NUM_THREADS_BFCL, DEFAULT_GPU_MEMORY_UTILIZATION_BFCL, BFCL_NUM_RETRIES  

# Set up logging to output to stdout
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

NON_LIVE_COLUMNS_FOR_OVERALL = [
    "Simple AST",
    "Multiple AST",
    "Parallel AST",
    "Parallel Multiple AST",
    "Irrelevance Detection",
]

def evaluate_bfcl(
    model_name: str,
    bfcl_results_dir: str | Path,
    model: PreTrainedModel | None = None,
    tokenizer: PreTrainedTokenizer | None = None,
    test_category: str = "python",
    num_threads: int = DEFAULT_NUM_THREADS_BFCL,
) -> tuple[list[str], dict[str, float]]:
    """
    Evaluate a model on the BFCL benchmark.
    
    Args:
        model_name: The name of the model to evaluate
        bfcl_results_dir: Directory to store the results of the evaluation
        model: The model to evaluate
        tokenizer: The tokenizer to use for the model
        test_category: The test category (default: "python")
        num_threads: Number of threads to use (default: DEFAULT_NUM_THREADS_BFCL)
    
    Returns:
        Dictionary containing execution results 
    """
    cuda.empty_cache()
    gc.collect()
    
    for _ in range(BFCL_NUM_RETRIES):
        try:
            return _evaluate_bfcl(
                model_name=model_name,
                bfcl_results_dir=bfcl_results_dir,
                model=model,
                tokenizer=tokenizer,
                test_category=test_category,
                num_threads=num_threads
            )
        except Exception as e:
            logger.error(f"Error evaluating BFCL: {e}")
            continue
    raise Exception("Failed to evaluate BFCL")

def _evaluate_bfcl(
    model_name: str,
    bfcl_results_dir: str | Path,
    model: PreTrainedModel | None = None,
    tokenizer: PreTrainedTokenizer | None = None,
    test_category: str = "python",
    num_threads: int = DEFAULT_NUM_THREADS_BFCL,
) -> tuple[list[str], dict[str, float]]:
    """
    Evaluate a model on the BFCL benchmark.
    
    Args:
        model_name: The name of the model to evaluate
        bfcl_results_dir: Directory to store the results of the evaluation
        model: The model to evaluate
        tokenizer: The tokenizer to use for the model
        test_category: The test category (default: "python")
        num_threads: Number of threads to use (default: DEFAULT_NUM_THREADS_BFCL)
    
    Returns:
        Dictionary containing execution results
    """
    if not isinstance(bfcl_results_dir, Path):
        bfcl_results_dir = Path(bfcl_results_dir)
    if model is not None:
        save_dir = model_name.split("/")[-1] + "-AL"
        model.save_pretrained(bfcl_results_dir / save_dir)
        tokenizer.save_pretrained(bfcl_results_dir / save_dir)
        cwd = os.getcwd()
        os.chdir(bfcl_results_dir)
        model_name = save_dir
        # Free up memory
        del model, tokenizer
        gc.collect()
        cuda.empty_cache()
        model = None

    logger.info(f"Starting BFCL evaluation for model: {model_name}")
    logger.info(f"Test category: {test_category}")
    logger.info(f"Number of threads: {num_threads}")
    logger.info(f"BFCL project root: {bfcl_results_dir}")
    
    # Set environment variables
    env = os.environ.copy()
    env["BFCL_PROJECT_ROOT"] = bfcl_results_dir
    env["TEST_CATEGORY"] = test_category
    env["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "kek")
    env["OPENAI_API_BASE"] = os.getenv("OPENAI_API_BASE", "http://localhost:8000")
    
    logger.info("Environment variables set successfully")
    
    # Run bfcl generate
    generate_cmd = [
        "bfcl", "generate",
        "--model", model_name,
        "--test-category", test_category,
        "--num-threads", str(num_threads),  
        "--gpu-memory-utilization", str(DEFAULT_GPU_MEMORY_UTILIZATION_BFCL),
    ]
    
    logger.info(f"Running BFCL generate command: {' '.join(generate_cmd)}")
    # generate_result = subprocess.run(generate_cmd, env=env, cwd=bfcl_results_dir)
    with open(bfcl_results_dir / "generate_stdout.log", "w") as stdout_file, open(bfcl_results_dir / "generate_stderr.log", "w") as stderr_file:
        generate_result = subprocess.run(
            generate_cmd, 
            env=env, 
            cwd=bfcl_results_dir,
            stdout=stdout_file,
            stderr=stderr_file
        )
    logger.info("BFCL generate completed")
    
    # Run bfcl evaluate
    evaluate_cmd = [
        "bfcl", "evaluate",
        "--model", model_name,
        "--test-category", test_category
    ]
    
    logger.info(f"Running BFCL evaluate command: {' '.join(evaluate_cmd)}")
    evaluate_result = subprocess.run(evaluate_cmd, env=env, cwd=bfcl_results_dir)
    logger.info("BFCL evaluate completed")
    
    logger.info("BFCL evaluation finished successfully")
    
    metrics = _extract_metrics(bfcl_results_dir)
    generations = _get_generations(model_name, Path(bfcl_results_dir))

    # Remove saved model and tokenizer
    if model is not None:
        rmtree(save_dir)
        os.chdir(cwd)

    return generations, metrics


def _get_generations(model_name: str, bfcl_results_dir: Path):
    results_dir = bfcl_results_dir / "result" / model_name
    generations = []
    for file in results_dir.glob("*.json"):
        with open(file, "r") as f:
            for line in f:
                generations.append(json.loads(line)["result"])
    return generations


def main():
    """Example usage of the evaluate_bfcl function."""
    import sys
    
    model_name = sys.argv[1] if len(sys.argv) > 1 else "default_model"
    bfcl_results_dir = sys.argv[2] if len(sys.argv) > 2 else "tmp"
    test_category = sys.argv[3] if len(sys.argv) > 3 else "python"
    num_threads = int(sys.argv[4]) if len(sys.argv) > 4 else 32
    
    logger.info("Starting BFCL evaluation script")
    
    results = evaluate_bfcl(
        model_name=model_name,
        bfcl_results_dir=bfcl_results_dir,
        test_category=test_category,
        num_threads=num_threads
    )
    
    logger.info("=== BFCL Evaluation Results ===")
    logger.info(results)

def _extract_metrics(bfcl_results_dir: Path) -> dict[str, float]:
    non_live_metrics = pd.read_csv(bfcl_results_dir / "score" / "data_non_live.csv").iloc[-1, 2:].dropna().to_dict()
    non_live_metrics = {
        "Non-live " + k: float(v.replace("%", ""))
        for k, v in non_live_metrics.items()
        if 'overall' not in k.lower()
    }
    if not "Simple AST" in non_live_metrics.keys():
        non_live_metrics["Simple AST"] = np.mean(
            [v for k, v in non_live_metrics.items() if "simple" in k.lower()]
        )
    non_live_metrics["Non-live Overall"] = np.mean([
        v for k, v in non_live_metrics.items()
        if k.strip("Non-live ") in NON_LIVE_COLUMNS_FOR_OVERALL
    ])

    live_metrics = pd.read_csv(bfcl_results_dir / "score" / "data_live.csv").iloc[-1, 2:].dropna().to_dict()
    live_metrics = {
        "Live " + k: float(v.replace("%", ""))
        for k, v in live_metrics.items()
    }
    non_live_metrics.update(live_metrics)
    return non_live_metrics


if __name__ == "__main__":
    main()
