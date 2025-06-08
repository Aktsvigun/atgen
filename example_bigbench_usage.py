#!/usr/bin/env python3
"""
Example usage of BigBenchHard metric with the new architecture.
"""

import sys
sys.path.append('src')

from atgen.metrics.base import MetricConfig
from atgen.metrics.llm_based import BigBenchHardMetric


def example_bigbench_usage():
    """Example of how to use BigBenchHard metric."""
    print("🧮 BigBenchHard Metric Usage Example")
    print("=" * 60)
    
    # Create configuration for BigBenchHard
    config = MetricConfig(
        batch_size=8,
        device="cuda",  # or "cpu"
        cache_dir="cache",
        model_name="MyAwesomeModel",
        # Benchmark-specific parameters
        benchmark_params={
            "n_shots": 3,  # Number of few-shot examples
            "enable_cot": True,  # Enable chain-of-thought
        },
        # Generation parameters for the model
        generation_params={
            "max_new_tokens": 100,
            "temperature": 0.1,
            "do_sample": False,
        }
    )
    
    # Create the metric
    metric = BigBenchHardMetric(config)
    
    print(f"✅ Metric created: {metric.name}")
    print(f"📦 Dependencies available: {metric.is_available()}")
    
    if not metric.is_available():
        print("❌ BigBenchHard dependencies not available")
        print("   Install with: pip install deepeval")
        return
    
    print("\n🔧 Usage Options:")
    print("\n1. Option 1: Using the standard calculate() interface:")
    print("   ```python")
    print("   # First set the model and tokenizer")
    print("   metric.set_model_and_tokenizer(model, tokenizer)")
    print("   ")
    print("   # Then calculate (predictions/references are ignored)")
    print("   results = metric.calculate([], [], [])")
    print("   print(results)  # {'bigbench_hard_score': 0.85}")
    print("   ```")
    
    print("\n2. Option 2: Using the convenience method:")
    print("   ```python")
    print("   # Direct evaluation")
    print("   score = metric.evaluate_benchmark(model, tokenizer, batch_size=16)")
    print("   print(f'BigBenchHard score: {score}')")
    print("   ```")
    
    print("\n3. Option 3: Using the original interface style:")
    print("   ```python")
    print("   # Set model first")
    print("   metric.set_model_and_tokenizer(model, tokenizer)")
    print("   ")
    print("   # Get the overall score")
    print("   score = metric.benchmark.overall_score")
    print("   ```")
    
    print("\n📝 Configuration Details:")
    print(f"   • Batch size: {config.batch_size}")
    print(f"   • Device: {config.device}")
    print(f"   • Model name: {config.model_name}")
    print(f"   • Benchmark params: {config.benchmark_params}")
    print(f"   • Generation params: {config.generation_params}")
    
    print("\n🎯 What BigBenchHard Evaluates:")
    print("   • Challenging reasoning tasks from BIG-bench")
    print("   • Mathematical reasoning")
    print("   • Logical reasoning")
    print("   • Multi-step problem solving")
    print("   • Chain-of-thought reasoning (if enabled)")
    
    print("\n💡 Integration Notes:")
    print("   • Follows the same BaseMetric interface as other metrics")
    print("   • Supports the standard MetricConfig system")
    print("   • Can be used in metric factories and pipelines")
    print("   • Provides both direct and standard interfaces")


if __name__ == "__main__":
    example_bigbench_usage() 