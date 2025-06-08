#!/usr/bin/env python3
"""
Demonstration of the new refactored metrics system.
"""

import sys
import os
sys.path.append('src')

from atgen.metrics.base import BaseMetric, MetricConfig
from atgen.metrics.lexical import BleuMetric, RougeMetric
from atgen.metrics.semantic import BartScoreMetric, AlignScoreMetric, SentBertMetric
from atgen.metrics.linguistic import ColaMetric


def demo_metrics():
    """Demonstrate the new metrics system."""
    print("🎯 New Metrics System Demonstration")
    print("=" * 60)
    
    # Sample data
    predictions = [
        "The cat is on the mat",
        "I love programming in Python", 
        "Machine learning is fascinating"
    ]
    
    references = [
        "A cat is sitting on the mat",
        "I enjoy coding with Python",
        "Machine learning is very interesting"
    ]
    
    original_texts = [
        "There is a cat on a mat",
        "Programming with Python is great",
        "ML technology is amazing"
    ]
    
    # Create metrics with custom config
    config = MetricConfig(
        batch_size=8,
        device="cpu",  # Use CPU for demo
        cache_dir="cache",
        aggregate=True
    )
    
    metrics = [
        ("BLEU", BleuMetric(config)),
        ("ROUGE", RougeMetric(config)),
        ("CoLA (Grammaticality)", ColaMetric(config)),
        ("SentenceBERT", SentBertMetric(config)),
        # ("BARTScore", BartScoreMetric(config)),  # Uncomment if you have the dependencies
        # ("AlignScore", AlignScoreMetric(config)),  # Uncomment if you have the dependencies
    ]
    
    print(f"📊 Testing with {len(predictions)} samples\n")
    
    for metric_name, metric in metrics:
        print(f"🔍 Testing {metric_name} ({metric.__class__.__name__})")
        print(f"   Available: {metric.is_available()}")
        
        if metric.is_available():
            try:
                # Test different scenarios based on metric requirements
                if metric_name == "CoLA (Grammaticality)":
                    # CoLA only needs predictions
                    results = metric.calculate(predictions)
                elif metric_name == "SentenceBERT":
                    # SentBERT can use both references and original texts
                    results = metric.calculate(predictions, references, original_texts)
                else:
                    # Standard metrics (BLEU, ROUGE) need references
                    results = metric.calculate(predictions, references)
                
                print(f"   Results: {results}")
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
        else:
            print(f"   ⚠️  Dependencies not available")
        
        print()
    
    print("🌟 Key Features of the New System:")
    print("   ✅ Clean inheritance from BaseMetric")
    print("   ✅ Consistent configuration via MetricConfig")
    print("   ✅ Automatic dependency checking")
    print("   ✅ Lazy model initialization")
    print("   ✅ Proper error handling")
    print("   ✅ Organized by category (lexical, semantic, linguistic)")
    print("   ✅ Type hints throughout")
    print("   ✅ Configurable aggregation")


if __name__ == "__main__":
    demo_metrics() 