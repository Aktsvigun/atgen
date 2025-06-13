import pytest
import numpy as np
from unittest.mock import Mock, patch
import os
import tempfile
import shutil
import warnings

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
    AVAILABLE_METRICS,
    get_available_metrics,
    get_all_possible_metric_keys,
)
from atgen.utils.check_required_performance import check_required_performance
from atgen.utils.check_performance_metrics import check_performance_against_requirements
from omegaconf import DictConfig


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
        assert config.provider is None
        assert config.api_key is None
        
    def test_custom_config(self):
        """Test custom configuration values."""
        config = MetricConfig(
            batch_size=16,
            device="cpu",
            cache_dir="custom_cache",
            aggregate=False,
            threshold=0.7,
            provider="openrouter",
            api_key="test-key"
        )
        assert config.batch_size == 16
        assert config.device == "cpu"
        assert config.cache_dir == "custom_cache"
        assert config.aggregate is False
        assert config.threshold == 0.7
        assert config.provider == "openrouter"
        assert config.api_key == "test-key"


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
            metrics=["bleu", "rouge1"],
            additional_metrics=["sentbert", "cola"]
        )
        expected_metrics = ["bleu", "rouge1", "sentbert", "cola"]
        assert config.metrics == expected_metrics
        
    def test_duplicate_metrics_removed(self):
        """Test that duplicate metrics are removed while preserving order."""
        config = MetricsConfig(
            metrics=["bleu", "rouge1", "bleu"],
            additional_metrics=["sentbert", "rouge1"]
        )
        expected_metrics = ["bleu", "rouge1", "sentbert"]
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
        assert "rouge1" in metrics
        assert "sentbert" in metrics
        assert "cola" in metrics
        
    def test_get_all_possible_metric_keys(self):
        """Test getting all possible metric keys."""
        keys = MetricsFactory.get_all_possible_metric_keys()
        assert isinstance(keys, list)
        assert len(keys) > len(MetricsFactory.get_available_metrics())
        
        # Should include specific metric keys
        assert "sentbert_pred_ref" in keys
        assert "sentbert_pred_src" in keys
        assert "rouge1" in keys
        assert "rouge2" in keys
        assert "rougeL" in keys
        assert "exact_match" in keys
        assert "word_length_gen" in keys
        
    def test_available_metrics_global(self):
        """Test AVAILABLE_METRICS global variable."""
        assert isinstance(AVAILABLE_METRICS, list)
        assert len(AVAILABLE_METRICS) > 20  # Should be comprehensive
        assert "sentbert_pred_ref" in AVAILABLE_METRICS
        assert "deepeval_answer_relevance" in AVAILABLE_METRICS
        
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
            metrics=["bleu", "rouge1", "sentbert"],
            device="cpu"
        )
        metrics = MetricsFactory.create_metrics(config)
        
        assert len(metrics) == 3
        assert "bleu" in metrics
        assert "rouge1" in metrics
        assert "sentbert" in metrics
        assert all(isinstance(m, BaseMetric) for m in metrics.values())
        
    def test_create_from_config_dict(self):
        """Test creating metrics from dictionary config."""
        config_dict = {
            "metrics": ["bleu", "rouge1"],
            "device": "cpu",
            "batch_size": 16
        }
        metrics = MetricsFactory.create_from_config(config_dict)
        
        assert len(metrics) == 2
        assert "bleu" in metrics
        assert "rouge1" in metrics


class TestPerformanceCheckingIntegration:
    """Test integration with performance checking system."""
    
    def test_check_required_performance_valid_metrics(self):
        """Test check_required_performance with valid metrics."""
        performance_dict = DictConfig({
            'bleu': 0.3,
            'rouge1': 0.5,
            'sentbert_pred_ref': 0.7,
            'deepeval_answer_relevance': 0.8
        })
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = check_required_performance(performance_dict)
            
            # Should have no warnings for valid metrics
            assert len(w) == 0
            assert dict(result) == dict(performance_dict)
    
    def test_check_required_performance_invalid_metrics(self):
        """Test check_required_performance with invalid metrics."""
        performance_dict = DictConfig({
            'bleu': 0.3,
            'nonexistent_metric': 0.5,
            'rouge1': 1.5  # Invalid value > 1
        })
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = check_required_performance(performance_dict)
            
            # Should have warnings for invalid metrics
            assert len(w) == 2
            assert dict(result) == {'bleu': 0.3}  # Only valid metric remains
    
    def test_performance_checking_pipeline(self):
        """Test complete performance checking pipeline."""
        # Simulate computed metrics
        metrics = {
            'bleu': 0.4,
            'rouge1': 0.6,
            'sentbert_pred_ref': 0.8,
            'word_length_gen': 10.5,
            'exact_match': 0.2,
            'time_total': 5.2
        }
        
        # Set performance requirements
        required_performance = DictConfig({
            'bleu': 0.3,
            'rouge1': 0.5,
            'sentbert_pred_ref': 0.7
        })
        
        # Check requirements
        checked_requirements = check_required_performance(required_performance)
        
        # Test performance pipeline
        is_performance_reached, is_metrics_availability_checked, available_metrics = (
            check_performance_against_requirements(
                metrics=metrics,
                required_performance_dict=checked_requirements,
                is_metrics_availability_checked=False,
                available_metrics={}
            )
        )
        
        assert is_performance_reached is True
        assert is_metrics_availability_checked is True
        assert len(available_metrics) == 3


class TestBasicMetricFunctionality:
    """Test basic functionality of individual metrics."""
    
    @pytest.fixture
    def sample_data(self):
        """Sample data for testing."""
        return {
            "predictions": ["This is a test sentence.", "Another test sentence here."],
            "references": ["This is a test sentence.", "Another test sentence here."],
            "original_texts": ["What is this?", "What is that?"]
        }
    
    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory for tests."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_bleu_metric(self, sample_data, temp_cache_dir):
        """Test BLEU metric functionality."""
        config = MetricConfig(device="cpu", cache_dir=temp_cache_dir)
        metric = BleuMetric(config)
        
        results = metric.calculate(
            sample_data["predictions"],
            sample_data["references"],
            sample_data["original_texts"]
        )
        
        assert "bleu" in results
        if isinstance(results["bleu"], np.ndarray):
            assert all(score == 1.0 for score in results["bleu"])  # Identical strings
        else:
            assert results["bleu"] == 1.0
            
    def test_rouge_metric(self, sample_data, temp_cache_dir):
        """Test ROUGE metric functionality."""
        config = MetricConfig(device="cpu", cache_dir=temp_cache_dir)
        metric = RougeMetric(config)
        
        results = metric.calculate(
            sample_data["predictions"],
            sample_data["references"],
            sample_data["original_texts"]
        )
        
        # Should return multiple ROUGE variants
        rouge_keys = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
        found_keys = [key for key in rouge_keys if key in results]
        assert len(found_keys) > 0
        
        # Perfect scores for identical strings
        for key in found_keys:
            if isinstance(results[key], np.ndarray):
                assert all(score == 1.0 for score in results[key])
            else:
                assert results[key] == 1.0
                
    @patch('torch.cuda.is_available', return_value=False)
    def test_sentbert_metric(self, mock_cuda, sample_data, temp_cache_dir):
        """Test SentBERT metric functionality."""
        config = MetricConfig(device="cpu", cache_dir=temp_cache_dir)
        metric = SentBertMetric(config)
        
        results = metric.calculate(
            sample_data["predictions"],
            sample_data["references"],
            sample_data["original_texts"]
        )
        
        # Should return both pred-ref and pred-src similarities
        assert "sentbert_pred_ref" in results
        assert "sentbert_pred_src" in results
        
        # High similarity for identical strings
        if isinstance(results["sentbert_pred_ref"], np.ndarray):
            assert all(score > 0.99 for score in results["sentbert_pred_ref"])
        else:
            assert results["sentbert_pred_ref"] > 0.99
                
    @patch('torch.cuda.is_available', return_value=False)
    def test_cola_metric(self, mock_cuda, sample_data, temp_cache_dir):
        """Test CoLA metric functionality."""
        config = MetricConfig(device="cpu", cache_dir=temp_cache_dir)
        metric = ColaMetric(config)
        
        results = metric.calculate(
            sample_data["predictions"],
            sample_data["references"],
            sample_data["original_texts"]
        )
        
        assert "cola" in results or "grammaticality" in results
        
        # Should return reasonable grammaticality scores
        score_key = "cola" if "cola" in results else "grammaticality"
        if isinstance(results[score_key], np.ndarray):
            assert all(0 <= score <= 1 for score in results[score_key])
        else:
            assert 0 <= results[score_key] <= 1


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
        
        assert isinstance(results, dict)
        assert len(results) > 0
        
        # Should include timing information
        assert any(key.startswith("time_") for key in results.keys())
        assert "time_total" in results
        
        # Should include basic statistics
        assert "word_length_gen" in results
        assert "exact_match" in results
        
        # Exact match should be 1.0 for identical strings
        assert results["exact_match"] == 1.0
        
    def test_compute_metrics_comprehensive_config(self, sample_data, temp_cache_dir):
        """Test compute_metrics with comprehensive configuration."""
        config = DictConfig({
            'metrics': ['bleu', 'rouge1', 'sentbert', 'cola'],
            'device': 'cpu',
            'cache_dir': temp_cache_dir,
            'batch_size': 16,
            'aggregate': True
        })
        
        results = compute_metrics(
            generated_texts=sample_data["predictions"],
            reference_texts=sample_data["references"],
            original_texts=sample_data["original_texts"],
            config=config
        )
        
        assert isinstance(results, dict)
        assert len(results) > 10  # Should have many metrics
        
        # Check key metrics are present
        assert "bleu" in results
        assert "rouge1" in results
        assert "sentbert_pred_ref" in results
        assert any(key in results for key in ["cola", "grammaticality"])
        
        # Perfect scores for identical strings
        assert results["bleu"] == 1.0
        assert results["rouge1"] == 1.0
        assert results["exact_match"] == 1.0
        
    def test_compute_metrics_with_deepeval_config(self, sample_data, temp_cache_dir):
        """Test compute_metrics with DeepEval configuration."""
        # Skip if no API key available
        api_key = os.environ.get('OPENROUTER_API_KEY')
        if not api_key:
            pytest.skip("No OpenRouter API key available for DeepEval testing")
            
        config = DictConfig({
            'metrics': ['bleu', 'deepeval_answer_relevance'],
            'provider': 'openrouter',
            'base_url': 'https://openrouter.ai/api/v1',
            'api_key': api_key,
            'model': 'openai/gpt-4o-mini',
            'threshold': 0.5,
            'async_mode': False,
            'verbose_mode': False,
            'device': 'cpu',
            'cache_dir': temp_cache_dir
        })
        
        results = compute_metrics(
            generated_texts=sample_data["predictions"],
            reference_texts=sample_data["references"],
            original_texts=sample_data["original_texts"],
            config=config
        )
        
        assert isinstance(results, dict)
        assert "bleu" in results
        
        # Check if DeepEval worked
        deepeval_keys = [k for k in results.keys() if 'deepeval' in k and not k.startswith('time_')]
        if deepeval_keys:
            assert len(deepeval_keys) > 0
            for key in deepeval_keys:
                assert 0 <= results[key] <= 1
        
    def test_compute_metrics_no_references(self, sample_data, temp_cache_dir):
        """Test compute_metrics without references."""
        config = DictConfig({
            'metrics': ['sentbert', 'cola'],
            'device': 'cpu',
            'cache_dir': temp_cache_dir
        })
        
        results = compute_metrics(
            generated_texts=sample_data["predictions"],
            reference_texts=None,
            original_texts=sample_data["original_texts"],
            config=config
        )
        
        assert isinstance(results, dict)
        assert "word_length_gen" in results
        assert "sentbert_pred_src" in results  # Should have pred-src similarity
        
    def test_compute_metrics_empty_predictions(self):
        """Test compute_metrics with empty predictions."""
        with pytest.raises(ValueError, match="predictions cannot be empty"):
            compute_metrics(generated_texts=[])
            
    def test_compute_metrics_from_config(self, sample_data, temp_cache_dir):
        """Test compute_metrics_from_config function."""
        config_dict = {
            "metrics": ["bleu", "rouge1"],
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
        assert "bleu" in results
        assert "rouge1" in results


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
        assert "rouge1" in config.metrics or any("rouge" in m for m in config.metrics)
        assert config.batch_size == 32
        assert config.device == "cuda"
        
    def test_get_comprehensive_config(self):
        """Test get_comprehensive_config function."""
        config = get_comprehensive_config()
        
        assert isinstance(config, MetricsConfig)
        assert len(config.metrics) > 2
        assert "bleu" in config.metrics
        assert any("rouge" in metric for metric in config.metrics)
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
        config = DictConfig({
            'metrics': ['bleu', 'rouge1'],
            'device': 'cpu',
            'cache_dir': temp_cache_dir
        })
        
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
        config = DictConfig({
            'metrics': ['bleu', 'rouge1'],
            'device': 'cpu',
            'cache_dir': temp_cache_dir
        })
        
        results = compute_metrics(
            generated_texts=["Single test sentence."],
            reference_texts=["Single test sentence."],
            original_texts=["Single source text."],
            config=config
        )
        
        assert isinstance(results, dict)
        assert len(results) > 0
        assert results["exact_match"] == 1.0  # Identical strings
        
    def test_multiple_references(self, temp_cache_dir):
        """Test metrics with multiple references."""
        config = DictConfig({
            'metrics': ['bleu', 'rouge1'],
            'device': 'cpu',
            'cache_dir': temp_cache_dir
        })
        
        results = compute_metrics(
            generated_texts=["Test sentence."],
            reference_texts=[["Test sentence.", "Another reference."]],
            original_texts=["Source text."],
            config=config
        )
        
        assert isinstance(results, dict)
        assert len(results) > 0


class TestRealWorldScenarios:
    """Test realistic scenarios that would occur in active learning."""
    
    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory for tests."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_active_learning_integration(self, temp_cache_dir):
        """Test complete active learning integration scenario."""
        # Realistic generated texts
        generated_texts = [
            "Machine learning algorithms learn patterns from training data to make predictions.",
            "Deep learning uses neural networks with multiple layers to process information.",
            "Natural language processing helps computers understand human language."
        ]
        
        reference_texts = [
            "ML algorithms learn from data to make predictions on new examples.",
            "Deep learning employs multi-layered neural networks for complex data processing.",
            "NLP enables computers to work with human language effectively."
        ]
        
        original_texts = [
            "How do machine learning algorithms work?",
            "What is deep learning?",
            "Explain natural language processing."
        ]
        
        # Comprehensive config similar to active learning setup
        config = DictConfig({
            'metrics': ['bleu', 'rouge1', 'rouge2', 'rougeL', 'sentbert'],
            'additional_metrics': ['cola'],
            'device': 'cpu',
            'cache_dir': temp_cache_dir,
            'batch_size': 16,
            'aggregate': True
        })
        
        # Compute metrics
        results = compute_metrics(
            generated_texts=generated_texts,
            reference_texts=reference_texts,
            original_texts=original_texts,
            config=config
        )
        
        # Verify comprehensive results
        assert isinstance(results, dict)
        assert len(results) > 10
        
        # Check key performance metrics
        performance_keys = ['bleu', 'rouge1', 'rouge2', 'rougeL', 'sentbert_pred_ref']
        for key in performance_keys:
            if key in results:
                assert isinstance(results[key], (int, float))
                assert 0 <= results[key] <= 1
        
        # Test performance checking integration
        required_performance = DictConfig({
            'bleu': 0.15,
            'rouge1': 0.35,
            'sentbert_pred_ref': 0.55
        })
        
        checked_requirements = check_required_performance(required_performance)
        assert len(checked_requirements) == 3  # All should be valid
        
        # Test complete pipeline
        is_performance_reached, is_metrics_availability_checked, available_metrics = (
            check_performance_against_requirements(
                metrics=results,
                required_performance_dict=checked_requirements,
                is_metrics_availability_checked=False,
                available_metrics={}
            )
        )
        
        assert is_metrics_availability_checked is True
        assert len(available_metrics) == 3
        
    def test_high_quality_generation_scenario(self, temp_cache_dir):
        """Test scenario with high-quality generated text."""
        # High-quality generated texts (should score well)
        generated_texts = [
            "The capital of France is Paris, which is also its largest city.",
            "Artificial intelligence is transforming various industries worldwide."
        ]
        
        reference_texts = [
            "Paris is the capital and largest city of France.",
            "AI is revolutionizing industries across the globe."
        ]
        
        original_texts = [
            "What is the capital of France?",
            "How is AI impacting industries?"
        ]
        
        config = DictConfig({
            'metrics': ['bleu', 'rouge1', 'sentbert', 'cola'],
            'device': 'cpu',
            'cache_dir': temp_cache_dir,
            'aggregate': True
        })
        
        results = compute_metrics(
            generated_texts=generated_texts,
            reference_texts=reference_texts,
            original_texts=original_texts,
            config=config
        )
        
        # Should get reasonable scores for good quality text
        assert results['bleu'] > 0.1  # Some lexical overlap
        assert results['rouge1'] > 0.2  # Some word overlap
        assert results['sentbert_pred_ref'] > 0.5  # Good semantic similarity
        
        # CoLA should give high grammaticality scores
        cola_key = 'cola' if 'cola' in results else 'grammaticality'
        if cola_key in results:
            assert results[cola_key] > 0.7  # Well-formed sentences


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 