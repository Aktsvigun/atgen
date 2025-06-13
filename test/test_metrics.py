import pytest
import numpy as np
from unittest.mock import Mock, patch
import os
import tempfile
import shutil

from atgen.metrics import (
    BaseMetric,
    MetricConfig,
    MetricsConfig,
    MetricsFactory,
    BleuMetric,
    RougeMetric,
    SentBertMetric,
    ColaMetric,
    BartScoreMetric,
    AlignScoreMetric,
    compute_metrics,
    compute_metrics_from_config,
    get_default_config,
    get_comprehensive_config,
    get_metric_categories,
    get_metric_requirements,
)


class TestMetricConfig:
    """Test MetricConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = MetricConfig()
        assert config.batch_size == 32
        assert config.device == "cuda"
        assert config.cache_dir == "cache"
        assert config.aggregate is True
        assert config.threshold == 0.5
        
    def test_custom_config(self):
        """Test custom configuration values."""
        config = MetricConfig(
            batch_size=16,
            device="cpu",
            cache_dir="custom_cache",
            aggregate=False,
            threshold=0.7
        )
        assert config.batch_size == 16
        assert config.device == "cpu"
        assert config.cache_dir == "custom_cache"
        assert config.aggregate is False
        assert config.threshold == 0.7


class TestMetricsConfig:
    """Test MetricsConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = MetricsConfig()
        assert config.batch_size == 32
        assert config.device == "cuda"
        assert config.cache_dir == "cache"
        assert config.aggregate is True
        assert config.metrics == []
        
    def test_additional_metrics_merge(self):
        """Test that additional_metrics are merged into metrics."""
        config = MetricsConfig(
            metrics=["bleu", "rouge"],
            additional_metrics=["sentbert", "cola"]
        )
        expected_metrics = ["bleu", "rouge", "sentbert", "cola"]
        assert config.metrics == expected_metrics
        
    def test_duplicate_metrics_removed(self):
        """Test that duplicate metrics are removed while preserving order."""
        config = MetricsConfig(
            metrics=["bleu", "rouge", "bleu"],
            additional_metrics=["sentbert", "rouge"]
        )
        expected_metrics = ["bleu", "rouge", "sentbert"]
        assert config.metrics == expected_metrics
        
    def test_deepeval_legacy_params(self):
        """Test that legacy DeepEval parameters are mapped correctly."""
        config = MetricsConfig(
            deepeval_threshold=0.8,
            deepeval_include_reason=True,
            deepeval_strict_mode=True,
            deepeval_async_mode=False,
            deepeval_verbose_mode=True,
            deepeval_truths_extraction_limit=10
        )
        assert config.threshold == 0.8
        assert config.include_reason is True
        assert config.strict_mode is True
        assert config.async_mode is False
        assert config.verbose_mode is True
        assert config.truths_extraction_limit == 10


class TestMetricsFactory:
    """Test MetricsFactory class."""
    
    def test_get_available_metrics(self):
        """Test getting available metrics."""
        metrics = MetricsFactory.get_available_metrics()
        assert isinstance(metrics, list)
        assert len(metrics) > 0
        assert "bleu" in metrics
        assert "rouge" in metrics
        assert "sentbert" in metrics
        assert "cola" in metrics
        
    def test_create_metric_success(self):
        """Test successful metric creation."""
        config = MetricConfig(device="cpu")
        metric = MetricsFactory.create_metric("bleu", config)
        assert isinstance(metric, BleuMetric)
        assert metric.config.device == "cpu"
        
    def test_create_metric_case_insensitive(self):
        """Test that metric creation is case insensitive."""
        metric_lower = MetricsFactory.create_metric("bleu")
        metric_upper = MetricsFactory.create_metric("BLEU")
        metric_mixed = MetricsFactory.create_metric("Bleu")
        
        assert type(metric_lower) == type(metric_upper) == type(metric_mixed)
        
    def test_create_metric_unknown(self):
        """Test error handling for unknown metrics."""
        with pytest.raises(ValueError, match="Unknown metric"):
            MetricsFactory.create_metric("unknown_metric")
            
    def test_create_metrics_from_config(self):
        """Test creating multiple metrics from config."""
        config = MetricsConfig(
            metrics=["bleu", "rouge", "sentbert"],
            device="cpu"
        )
        metrics = MetricsFactory.create_metrics(config)
        
        assert len(metrics) == 3
        assert "bleu" in metrics
        assert "rouge" in metrics
        assert "sentbert" in metrics
        assert all(isinstance(m, BaseMetric) for m in metrics.values())
        
    def test_create_from_config_dict(self):
        """Test creating metrics from dictionary config."""
        config_dict = {
            "metrics": ["bleu", "rouge"],
            "device": "cpu",
            "batch_size": 16
        }
        metrics = MetricsFactory.create_from_config(config_dict)
        
        assert len(metrics) == 2
        assert "bleu" in metrics
        assert "rouge" in metrics
        
    def test_register_custom_metric(self):
        """Test registering a custom metric."""
        class CustomMetric(BaseMetric):
            def calculate(self, predictions, references, original_texts):
                return {"custom": 0.5}
                
        MetricsFactory.register_metric("custom", CustomMetric)
        
        try:
            available = MetricsFactory.get_available_metrics()
            assert "custom" in available
            
            metric = MetricsFactory.create_metric("custom")
            assert isinstance(metric, CustomMetric)
        finally:
            # Clean up
            if "custom" in MetricsFactory._metric_registry:
                del MetricsFactory._metric_registry["custom"]


class TestIdenticalStringsBasic:
    """Test all metrics with identical strings to ensure basic functionality."""
    
    @pytest.fixture
    def identical_data(self):
        """Sample data with identical predictions and references."""
        return {
            "predictions": ["This is a test sentence.", "Another test sentence here."],
            "references": ["This is a test sentence.", "Another test sentence here."],
            "original_texts": ["Original text one.", "Original text two."]
        }
    
    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory for tests."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_bleu_identical_strings(self, identical_data, temp_cache_dir):
        """Test BLEU metric with identical strings."""
        config = MetricConfig(device="cpu", cache_dir=temp_cache_dir)
        metric = BleuMetric(config)
        
        results = metric.calculate(
            identical_data["predictions"],
            identical_data["references"],
            identical_data["original_texts"]
        )
        
        assert "bleu" in results
        if isinstance(results["bleu"], np.ndarray):
            # Should be perfect scores for identical strings
            assert all(score == 1.0 for score in results["bleu"])
        else:
            assert results["bleu"] == 1.0
            
    def test_rouge_identical_strings(self, identical_data, temp_cache_dir):
        """Test ROUGE metric with identical strings."""
        config = MetricConfig(device="cpu", cache_dir=temp_cache_dir)
        metric = RougeMetric(config)
        
        results = metric.calculate(
            identical_data["predictions"],
            identical_data["references"],
            identical_data["original_texts"]
        )
        
        # ROUGE should return perfect scores for identical strings
        rouge_keys = ["rouge1", "rouge2", "rougeL"]
        for key in rouge_keys:
            if key in results:
                if isinstance(results[key], np.ndarray):
                    assert all(score == 1.0 for score in results[key])
                else:
                    assert results[key] == 1.0
                    
    @patch('torch.cuda.is_available', return_value=False)
    def test_sentbert_identical_strings(self, mock_cuda, identical_data, temp_cache_dir):
        """Test SentBERT metric with identical strings."""
        config = MetricConfig(device="cpu", cache_dir=temp_cache_dir)
        metric = SentBertMetric(config)
        
        results = metric.calculate(
            identical_data["predictions"],
            identical_data["references"],
            identical_data["original_texts"]
        )
        
        # SentBERT should return high similarity scores for identical strings
        if "sentbert_pred_ref" in results:
            if isinstance(results["sentbert_pred_ref"], np.ndarray):
                assert all(score > 0.99 for score in results["sentbert_pred_ref"])
            else:
                assert results["sentbert_pred_ref"] > 0.99
                
    @patch('torch.cuda.is_available', return_value=False)
    def test_cola_identical_strings(self, mock_cuda, identical_data, temp_cache_dir):
        """Test CoLA metric with identical strings."""
        config = MetricConfig(device="cpu", cache_dir=temp_cache_dir)
        metric = ColaMetric(config)
        
        results = metric.calculate(
            identical_data["predictions"],
            identical_data["references"],
            identical_data["original_texts"]
        )
        
        # CoLA should return grammaticality scores
        assert "cola" in results
        if isinstance(results["cola"], np.ndarray):
            assert all(0 <= score <= 1 for score in results["cola"])
        else:
            assert 0 <= results["cola"] <= 1

    @patch('torch.cuda.is_available', return_value=False)
    def test_bartscore_identical_strings(self, mock_cuda, identical_data, temp_cache_dir):
        """Test BARTScore metric with identical strings."""
        config = MetricConfig(device="cpu", cache_dir=temp_cache_dir, batch_size=2)
        metric = BartScoreMetric(config)
        
        try:
            results = metric.calculate(
                identical_data["predictions"],
                identical_data["references"],
                identical_data["original_texts"]
            )
            
            # BARTScore should return high scores for identical strings
            assert isinstance(results, dict)
            assert len(results) > 0
            
            # Check that scores are reasonable (BARTScore can vary but should be positive for identical strings)
            for key, value in results.items():
                if isinstance(value, np.ndarray):
                    assert all(isinstance(score, (int, float)) for score in value)
                else:
                    assert isinstance(value, (int, float))
                    
        except Exception as e:
            # BARTScore might not be available or have dependency issues
            pytest.skip(f"BARTScore test skipped due to: {e}")

    @patch('torch.cuda.is_available', return_value=False)
    def test_alignscore_identical_strings(self, mock_cuda, identical_data, temp_cache_dir):
        """Test AlignScore metric with identical strings."""
        config = MetricConfig(device="cpu", cache_dir=temp_cache_dir, batch_size=2)
        metric = AlignScoreMetric(config)
        
        try:
            results = metric.calculate(
                identical_data["predictions"],
                identical_data["references"],
                identical_data["original_texts"]
            )
            
            # AlignScore should return high scores for identical strings
            assert isinstance(results, dict)
            assert len(results) > 0
            
            # Check that scores are reasonable
            for key, value in results.items():
                if isinstance(value, np.ndarray):
                    assert all(isinstance(score, (int, float)) for score in value)
                else:
                    assert isinstance(value, (int, float))
                    
        except Exception as e:
            # AlignScore might not be available or have dependency issues
            pytest.skip(f"AlignScore test skipped due to: {e}")


class TestComputeMetrics:
    """Test the main compute_metrics function."""
    
    @pytest.fixture
    def sample_data(self):
        """Sample data for testing."""
        return {
            "predictions": ["This is a test.", "Another test."],
            "references": ["This is a test.", "Another test."],
            "original_texts": ["Source text one.", "Source text two."]
        }
    
    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory for tests."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_compute_metrics_default_config(self, sample_data, temp_cache_dir):
        """Test compute_metrics with default configuration."""
        results = compute_metrics(
            generated_texts=sample_data["predictions"],
            reference_texts=sample_data["references"],
            original_texts=sample_data["original_texts"],
            cache_dir=temp_cache_dir
        )
        
        # Should include basic metrics
        assert isinstance(results, dict)
        assert len(results) > 0
        
        # Should include timing information
        assert any(key.startswith("time_") for key in results.keys())
        assert "time_total" in results
        
        # Should include basic statistics
        assert "word_length_gen" in results
        assert "exact_match" in results
        
    def test_compute_metrics_custom_config(self, sample_data, temp_cache_dir):
        """Test compute_metrics with custom configuration."""
        config = MetricsConfig(
            metrics=["bleu", "rouge"],
            device="cpu",
            cache_dir=temp_cache_dir,
            batch_size=16
        )
        
        results = compute_metrics(
            generated_texts=sample_data["predictions"],
            reference_texts=sample_data["references"],
            original_texts=sample_data["original_texts"],
            config=config
        )
        
        assert isinstance(results, dict)
        assert "bleu" in results
        assert any(key.startswith("rouge") for key in results.keys())
        
    def test_compute_metrics_no_references(self, sample_data, temp_cache_dir):
        """Test compute_metrics without references."""
        config = MetricsConfig(
            metrics=["sentbert", "cola"],
            device="cpu",
            cache_dir=temp_cache_dir
        )
        
        results = compute_metrics(
            generated_texts=sample_data["predictions"],
            reference_texts=None,
            original_texts=sample_data["original_texts"],
            config=config
        )
        
        assert isinstance(results, dict)
        # Should still include basic statistics
        assert "word_length_gen" in results
        
    def test_compute_metrics_empty_predictions(self):
        """Test compute_metrics with empty predictions."""
        with pytest.raises(ValueError, match="predictions cannot be empty"):
            compute_metrics(generated_texts=[])
            
    def test_compute_metrics_from_config(self, sample_data, temp_cache_dir):
        """Test compute_metrics_from_config function."""
        config_dict = {
            "metrics": ["bleu", "rouge"],
            "device": "cpu",
            "cache_dir": temp_cache_dir
        }
        
        results = compute_metrics_from_config(
            predictions=sample_data["predictions"],
            references=sample_data["references"],
            original_texts=sample_data["original_texts"],
            config_dict=config_dict
        )
        
        assert isinstance(results, dict)
        assert len(results) > 0


class TestMetricRequirements:
    """Test metric requirements and categories."""
    
    def test_get_metric_categories(self):
        """Test getting metric categories."""
        categories = get_metric_categories()
        
        assert isinstance(categories, dict)
        assert "lexical" in categories
        assert "semantic" in categories
        assert "linguistic" in categories
        assert "llm_based" in categories
        
        # Check specific metrics in categories
        assert "bleu" in categories["lexical"]
        assert "rouge" in categories["lexical"]
        assert "sentbert" in categories["semantic"]
        assert "cola" in categories["linguistic"]
        
    def test_get_metric_requirements(self):
        """Test getting metric requirements."""
        requirements = get_metric_requirements()
        
        assert isinstance(requirements, dict)
        
        # Test specific requirements
        assert requirements["bleu"]["requires_references"] is True
        assert requirements["bleu"]["requires_original_texts"] is False
        
        assert requirements["sentbert"]["requires_references"] is False
        assert requirements["sentbert"]["requires_original_texts"] is False
        
        assert requirements["cola"]["requires_references"] is False
        assert requirements["cola"]["requires_original_texts"] is False


class TestConfigurationFunctions:
    """Test configuration helper functions."""
    
    def test_get_default_config(self):
        """Test get_default_config function."""
        config = get_default_config()
        
        assert isinstance(config, MetricsConfig)
        assert "bleu" in config.metrics
        assert "rouge" in config.metrics
        assert config.batch_size == 32
        assert config.device == "cuda"
        
    def test_get_comprehensive_config(self):
        """Test get_comprehensive_config function."""
        config = get_comprehensive_config()
        
        assert isinstance(config, MetricsConfig)
        assert len(config.metrics) > 2  # Should include more metrics
        assert "bleu" in config.metrics
        assert "rouge" in config.metrics
        assert "sentbert" in config.metrics
        assert "cola" in config.metrics


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory for tests."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_empty_strings(self, temp_cache_dir):
        """Test metrics with empty strings."""
        config = MetricsConfig(
            metrics=["bleu", "rouge"],
            device="cpu",
            cache_dir=temp_cache_dir
        )
        
        results = compute_metrics(
            generated_texts=["", ""],
            reference_texts=["", ""],
            original_texts=["", ""],
            config=config
        )
        
        assert isinstance(results, dict)
        # Should handle empty strings gracefully
        
    def test_single_item_lists(self, temp_cache_dir):
        """Test metrics with single item lists."""
        config = MetricsConfig(
            metrics=["bleu", "rouge"],
            device="cpu",
            cache_dir=temp_cache_dir
        )
        
        results = compute_metrics(
            generated_texts=["Single test sentence."],
            reference_texts=["Single test sentence."],
            original_texts=["Single source text."],
            config=config
        )
        
        assert isinstance(results, dict)
        assert len(results) > 0
        
    def test_multiple_references(self, temp_cache_dir):
        """Test metrics with multiple references."""
        config = MetricsConfig(
            metrics=["bleu", "rouge"],
            device="cpu",
            cache_dir=temp_cache_dir
        )
        
        results = compute_metrics(
            generated_texts=["Test sentence."],
            reference_texts=[["Test sentence.", "Another reference."]],
            original_texts=["Source text."],
            config=config
        )
        
        assert isinstance(results, dict)
        assert len(results) > 0
        
    def test_mismatched_lengths(self, temp_cache_dir):
        """Test metrics with mismatched input lengths."""
        config = MetricsConfig(
            metrics=["bleu"],
            device="cpu",
            cache_dir=temp_cache_dir
        )
        
        # This should handle gracefully or raise appropriate error
        try:
            results = compute_metrics(
                generated_texts=["Test sentence.", "Another sentence."],
                reference_texts=["Test sentence."],  # Shorter list
                original_texts=["Source text."],
                config=config
            )
            # If it doesn't raise an error, check that results are reasonable
            assert isinstance(results, dict)
        except (ValueError, IndexError, AssertionError):
            # These are acceptable errors for mismatched lengths
            pass


class TestMetricIntegration:
    """Integration tests for the complete metrics system."""
    
    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory for tests."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_all_lexical_metrics(self, temp_cache_dir):
        """Test all lexical metrics together."""
        config = MetricsConfig(
            metrics=["bleu", "rouge"],
            device="cpu",
            cache_dir=temp_cache_dir,
            aggregate=True
        )
        
        results = compute_metrics(
            generated_texts=["This is a comprehensive test sentence.", "Another test for metrics."],
            reference_texts=["This is a comprehensive test sentence.", "Another test for metrics."],
            original_texts=["Source text one.", "Source text two."],
            config=config
        )
        
        assert "bleu" in results
        assert any(key.startswith("rouge") for key in results.keys())
        
        # All scores should be 1.0 for identical strings
        # BLEU is aggregated so should be a single value
        assert isinstance(results["bleu"], (int, float))
        assert results["bleu"] == 1.0
        
    @patch('torch.cuda.is_available', return_value=False)
    def test_comprehensive_metrics_suite(self, mock_cuda, temp_cache_dir):
        """Test a comprehensive suite of metrics."""
        config = MetricsConfig(
            metrics=["bleu", "rouge", "sentbert", "cola"],
            device="cpu",
            cache_dir=temp_cache_dir,
            batch_size=8,
            aggregate=True
        )
        
        results = compute_metrics(
            generated_texts=[
                "This is a well-formed grammatical sentence.",
                "Another properly structured sentence here."
            ],
            reference_texts=[
                "This is a well-formed grammatical sentence.",
                "Another properly structured sentence here."
            ],
            original_texts=[
                "Source text for the first sentence.",
                "Source text for the second sentence."
            ],
            config=config
        )
        
        # Should include results from all metric types
        assert "bleu" in results
        assert any(key.startswith("rouge") for key in results.keys())
        assert any(key.startswith("sentbert") for key in results.keys())
        assert "cola" in results
        
        # Should include timing and basic stats
        assert "time_total" in results
        assert "word_length_gen" in results
        assert "exact_match" in results
        
        # Exact match should be 1.0 for identical strings
        assert results["exact_match"] == 1.0

    @patch('torch.cuda.is_available', return_value=False)
    def test_semantic_metrics_suite(self, mock_cuda, temp_cache_dir):
        """Test semantic metrics including BARTScore and AlignScore."""
        config = MetricsConfig(
            metrics=["sentbert", "bartscore", "alignscore"],
            device="cpu",
            cache_dir=temp_cache_dir,
            batch_size=4,
            aggregate=True
        )
        
        try:
            results = compute_metrics(
                generated_texts=[
                    "This is a semantic similarity test.",
                    "Another sentence for testing."
                ],
                reference_texts=[
                    "This is a semantic similarity test.",
                    "Another sentence for testing."
                ],
                original_texts=[
                    "Source text for semantic testing.",
                    "Another source text here."
                ],
                config=config
            )
            
            # Should include results from semantic metrics
            assert isinstance(results, dict)
            assert len(results) > 0
            
            # Should include timing and basic stats
            assert "time_total" in results
            assert "word_length_gen" in results
            assert "exact_match" in results
            
            # Exact match should be 1.0 for identical strings
            assert results["exact_match"] == 1.0
            
        except Exception as e:
            # Some semantic metrics might not be available
            pytest.skip(f"Semantic metrics test skipped due to: {e}")


if __name__ == "__main__":
    pytest.main([__file__]) 