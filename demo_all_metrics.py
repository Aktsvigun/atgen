#!/usr/bin/env python3
"""
Comprehensive demonstration of the refactored metrics system including DeepEval metrics.
"""

import sys
import os
sys.path.append('src')

from atgen.metrics.base import BaseMetric, MetricConfig
from atgen.metrics.lexical import BleuMetric, RougeMetric
from atgen.metrics.semantic import BartScoreMetric, AlignScoreMetric, SentBertMetric
from atgen.metrics.linguistic import ColaMetric
from atgen.metrics.llm_based import (
    DeepEvalAnswerRelevancyMetric,
    DeepEvalFaithfulnessMetric,
    DeepEvalSummarizationMetric,
    DeepEvalPromptAlignmentMetric,
    BigBenchHardMetric,
    EvaluationLLM
)


def demo_all_metrics():
    """Demonstrate all metrics in the new system."""
    print("🎯 Comprehensive Metrics System Demonstration")
    print("=" * 80)
    
    # Sample data
    predictions = [
        "The cat is sitting on the mat peacefully.",
        "Python is an excellent programming language for machine learning.",
        "Deep learning has revolutionized artificial intelligence applications."
    ]
    
    references = [
        "A cat is resting on the mat.",
        "Python is great for ML development.",
        "Deep learning has transformed AI."
    ]
    
    original_texts = [
        "There is a cat on a mat",
        "What makes Python good for ML?",
        "How has deep learning impacted AI?"
    ]
    
    # Create different configurations for different metric types
    base_config = MetricConfig(
        batch_size=2,
        device="cpu",  # Use CPU for demo
        cache_dir="cache",
        aggregate=True
    )
    
    # DeepEval config (requires API key)
    deepeval_config = MetricConfig(
        batch_size=2,
        device="cpu",
        cache_dir="cache", 
        aggregate=True,
        api_key=os.getenv("OPENROUTER_API_KEY"),  # Set this in environment
        base_url="https://openrouter.ai/api/v1",
        model="openai/gpt-4o-mini",  # Cheaper model for demo
        threshold=0.5,
        include_reason=False,
        strict_mode=False,
        async_mode=False,  # Synchronous for simpler demo
        verbose_mode=False
    )
    
    # Organize metrics by category
    metrics_by_category = {
        "📝 Lexical Metrics": [
            ("BLEU", BleuMetric(base_config)),
            ("ROUGE", RougeMetric(base_config)),
        ],
        "🧠 Semantic Metrics": [
            ("SentenceBERT", SentBertMetric(base_config)),
            # ("BARTScore", BartScoreMetric(base_config)),  # Uncomment if you have dependencies
            # ("AlignScore", AlignScoreMetric(base_config)),  # Uncomment if you have dependencies
        ],
        "🗣️ Linguistic Quality": [
            ("CoLA (Grammaticality)", ColaMetric(base_config)),
        ],
        "🤖 LLM-based (DeepEval)": [
            ("Answer Relevancy", DeepEvalAnswerRelevancyMetric(deepeval_config)),
            ("Faithfulness", DeepEvalFaithfulnessMetric(deepeval_config)),
            ("Summarization", DeepEvalSummarizationMetric(deepeval_config)),
            ("Prompt Alignment", DeepEvalPromptAlignmentMetric(deepeval_config)),
        ],
        "🧮 Benchmark Metrics": [
            ("BigBenchHard", BigBenchHardMetric(deepeval_config)),
        ]
    }
    
    print(f"📊 Testing with {len(predictions)} samples\n")
    
    # Test each category
    for category, metrics in metrics_by_category.items():
        print(f"{category}")
        print("-" * 60)
        
        for metric_name, metric in metrics:
            print(f"  🔍 {metric_name} ({metric.__class__.__name__})")
            print(f"      Available: {metric.is_available()}")
            
            if metric.is_available():
                try:
                    # Configure test based on metric requirements
                    if "DeepEval" in metric.__class__.__name__:
                        if deepeval_config.api_key is None:
                            print(f"      ⚠️  API key required (set OPENROUTER_API_KEY)")
                            print()
                            continue
                        
                        # Different DeepEval metrics need different inputs
                        if "PromptAlignment" in metric.__class__.__name__:
                            results = metric.calculate(predictions, references, original_texts)
                        else:
                            results = metric.calculate(predictions, None, original_texts)
                    
                    elif metric_name == "BigBenchHard":
                        # BigBenchHard needs a model and tokenizer, skip for demo
                        print(f"      ⚠️  Requires model and tokenizer (use set_model_and_tokenizer())")
                        print(f"      💡 Example: metric.set_model_and_tokenizer(model, tokenizer)")
                        print(f"                 score = metric.evaluate_benchmark(model, tokenizer)")
                        print()
                        continue
                    
                    elif metric_name == "CoLA (Grammaticality)":
                        # CoLA only needs predictions
                        results = metric.calculate(predictions)
                    
                    elif metric_name == "SentenceBERT":
                        # SentBERT can use both references and original texts
                        results = metric.calculate(predictions, references, original_texts)
                    
                    else:
                        # Standard metrics (BLEU, ROUGE) need references
                        results = metric.calculate(predictions, references)
                    
                    print(f"      Results: {results}")
                    
                except Exception as e:
                    print(f"      ❌ Error: {e}")
            else:
                print(f"      ⚠️  Dependencies not available")
            
            print()
        
        print()
    
    print("🌟 System Architecture Highlights:")
    print("   ✅ BaseMetric abstract class with consistent interface")
    print("   ✅ MetricConfig for centralized configuration")
    print("   ✅ Category-based organization (lexical, semantic, linguistic, llm-based)")
    print("   ✅ Automatic dependency checking")
    print("   ✅ Lazy model initialization")
    print("   ✅ Proper error handling and logging")
    print("   ✅ Support for multiple references")
    print("   ✅ API-based metrics with custom LLM implementation")
    print("   ✅ Benchmark metrics (BigBenchHard)")
    print("   ✅ Configurable aggregation")
    print("   ✅ Type safety with full type hints")
    
    print("\n🔧 Next Steps:")
    print("   • Strategy factory pattern for metric selection")
    print("   • Configuration-based metric instantiation")
    print("   • Integration with existing compute_metrics.py")
    print("   • Performance optimization and caching")


if __name__ == "__main__":
    demo_all_metrics() 