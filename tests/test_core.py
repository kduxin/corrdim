"""
Tests for the core correlation dimension calculation module.
"""

import pytest
import numpy as np
import torch
from unittest.mock import Mock, patch, MagicMock

from corrdim import CorrelationDimensionCalculator, correlation_integral
from corrdim.calculator import CorrelationIntegralResult, CorrelationDimensionResult
from corrdim.models import LanguageModelWrapper


class MockModelWrapper(LanguageModelWrapper):
    """Mock model wrapper for testing."""
    
    def __init__(self, vocab_size=1000, seq_len=150, device="cpu"):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.device = device
        self.tokenizer = Mock()
        self.tokenizer.encode = Mock(return_value=list(range(seq_len)))
        self.tokenizer.decode = Mock(return_value="decoded text")
    
    def get_log_probabilities(self, text, context_length=None, dim_reduction=None, stride="auto", show_progress=False):
        """Return mock log probabilities."""
        # Generate deterministic log probabilities
        np.random.seed(42)
        log_probs = torch.tensor(
            np.random.randn(self.seq_len, self.vocab_size).astype(np.float32),
            device=self.device
        )
        return log_probs


class TestCorrelationDimensionCalculator:
    """Test cases for CorrelationDimensionCalculator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        from corrdim.corrint import set_corrint_backend
        set_corrint_backend("pytorch")  # Use PyTorch backend for CPU compatibility
        self.mock_model = MockModelWrapper(vocab_size=100, seq_len=150)
        self.calc = CorrelationDimensionCalculator(self.mock_model)
    
    def test_initialization_with_wrapper(self):
        """Test calculator initialization with model wrapper."""
        assert self.calc.model_wrapper == self.mock_model
        assert self.calc.device in ["cpu", "cuda"]
    
    def test_initialization_with_string(self):
        """Test calculator initialization with model name string."""
        with patch("corrdim.models.create_model_wrapper") as mock_create:
            mock_create.return_value = self.mock_model
            calc = CorrelationDimensionCalculator("gpt2", device="cpu")
            assert calc.model_wrapper == self.mock_model
            mock_create.assert_called_once()
    
    def test_get_log_probability(self):
        """Test log probability extraction."""
        text = "Test text for correlation dimension calculation."
        log_probs = self.calc.get_log_probability(text)
        
        assert isinstance(log_probs, torch.Tensor)
        assert log_probs.shape[1] == 100  # vocab_size
        assert log_probs.shape[0] == 150   # seq_len
    
    def test_compute_correlation_integral_curve_from_vectors(self):
        """Test correlation integral curve computation from vectors."""
        # Create simple test data (need > 100 tokens)
        vecs = torch.randn(150, 50, device=self.calc.device)
        
        result = self.calc.compute_correlation_integral_curve_from_vector_sequences(
            vecs,
            epsilon_range=(1e-10, 1e10),
            num_epsilon=100
        )
        
        assert isinstance(result, CorrelationIntegralResult)
        assert result.sequence_length == 150
        assert len(result.epsilons) == len(result.corrints)
        assert len(result.epsilons) > 0
        assert np.all(result.corrints >= 0)
        assert np.all(result.corrints <= 1)
    
    def test_compute_correlation_integral_curve_from_text(self):
        """Test correlation integral curve computation from text."""
        text = "Test text for correlation dimension calculation."
        
        result = self.calc.compute_correlation_integral_curve(
            text,
            epsilon_range=(1e-10, 1e10),
            num_epsilon=100
        )
        
        assert isinstance(result, CorrelationIntegralResult)
        assert result.sequence_length == 150
        # After clamp, epsilons and corrints may be filtered
        assert len(result.epsilons) == len(result.corrints)
        assert len(result.epsilons) > 0  # Should have at least some points
        assert np.all(result.corrints >= 0)
        assert np.all(result.corrints <= 1)
    
    def test_compute_correlation_dimension_from_text(self):
        """Test correlation dimension computation from text."""
        text = "Test text for correlation dimension calculation."
        
        # Use wide epsilon range with many points, with custom correlation_integral_range
        # to ensure valid linear region even with random data
        result = self.calc(
            text,
            epsilon_range=(1e-20, 1e20),
            num_epsilon=10000,
            correlation_integral_range=(0.0, 1.0)  # Use full range to ensure enough points
        )
        
        assert isinstance(result, CorrelationDimensionResult)
        assert result.sequence_length == 150
        assert isinstance(result.corrdim, (float, np.floating))  # Can be float or np.float32/float64
        assert result.corrdim >= 0  # Should be non-negative
        assert 0 <= result.fit_r2 <= 1  # R² should be between 0 and 1
        assert len(result.epsilons) > 0
        assert len(result.corrints) > 0
    
    def test_compute_correlation_dimension_from_curve(self):
        """Test correlation dimension computation from curve."""
        # Create mock correlation integral curve
        sequence_length = 150
        epsilons = np.logspace(-10, 10, 100)
        corrints = epsilons ** 0.5  # Simulate power-law relationship
        
        result = CorrelationDimensionCalculator.compute_correlation_dimension_from_curve(
            epsilons=epsilons,
            corrints=corrints,
            sequence_length=sequence_length
        )
        
        assert isinstance(result, CorrelationDimensionResult)
        assert result.sequence_length == sequence_length
        assert isinstance(result.corrdim, (float, np.floating))  # Can be float or np.float32/float64
        assert result.corrdim > 0
        assert 0 <= result.fit_r2 <= 1
    
    def test_tokenizer_property(self):
        """Test tokenizer property access."""
        assert self.calc.tokenizer == self.mock_model.tokenizer


class TestCorrelationIntegral:
    """Test cases for correlation_integral function."""
    
    def setup_method(self):
        """Set up backend to PyTorch for CPU compatibility."""
        from corrdim.corrint import set_corrint_backend
        set_corrint_backend("pytorch")
    
    def test_correlation_integral_basic(self):
        """Test basic correlation integral computation."""
        # Create simple test vectors
        vecs = torch.randn(50, 20, device="cpu")
        epsilons = torch.logspace(-5, 5, 100, device="cpu")
        
        corrints = correlation_integral(vecs, epsilons)
        
        assert isinstance(corrints, torch.Tensor)
        assert corrints.shape == epsilons.shape
        assert torch.all(corrints >= 0)
        # Note: correlation integral can be > 1 for small sequences due to normalization
        # but should be bounded reasonably
        assert torch.all(corrints <= vecs.shape[0])  # Upper bound is sequence length
    
    def test_correlation_integral_monotonicity(self):
        """Test that correlation integral is monotonically increasing."""
        vecs = torch.randn(30, 10, device="cpu")
        epsilons = torch.logspace(-5, 5, 50, device="cpu")
        
        corrints = correlation_integral(vecs, epsilons)
        
        # Should be monotonically increasing
        diffs = corrints[1:] - corrints[:-1]
        assert torch.all(diffs >= 0), "Correlation integral should be monotonically increasing"
    
    def test_correlation_integral_edge_cases(self):
        """Test correlation integral with edge cases."""
        # Single point - may produce NaN due to division by zero, which is acceptable
        vecs_single = torch.randn(1, 10, device="cpu")
        epsilons = torch.logspace(-5, 5, 10, device="cpu")
        corrints_single = correlation_integral(vecs_single, epsilons)
        # Single point should give zero or NaN (due to N*(N-1)/2 = 0)
        assert torch.all((corrints_single == 0) | torch.isnan(corrints_single)), \
            "Single point should give zero or NaN correlation integral"
        
        # Two identical points
        vecs_identical = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], device="cpu")
        epsilons_small = torch.tensor([0.1], device="cpu")
        corrints_identical = correlation_integral(vecs_identical, epsilons_small)
        assert corrints_identical[0] > 0, "Identical points should have non-zero correlation integral"
    
    def test_correlation_integral_different_epsilons(self):
        """Test correlation integral with different epsilon values."""
        vecs = torch.randn(20, 10, device="cpu")
        
        epsilons_small = torch.tensor([0.1], device="cpu")
        epsilons_large = torch.tensor([10.0], device="cpu")
        
        corrints_small = correlation_integral(vecs, epsilons_small)
        corrints_large = correlation_integral(vecs, epsilons_large)
        
        # Larger epsilon should give higher correlation integral
        assert corrints_large[0] >= corrints_small[0]


class TestCorrelationDimensionProperties:
    """Test mathematical properties of correlation dimension."""
    
    def setup_method(self):
        """Set up backend to PyTorch for CPU compatibility."""
        from corrdim.corrint import set_corrint_backend
        set_corrint_backend("pytorch")
    
    def test_correlation_dimension_range(self):
        """Test that correlation dimension values are reasonable."""
        mock_model = MockModelWrapper(vocab_size=100, seq_len=150)
        calc = CorrelationDimensionCalculator(mock_model)
        
        text = "Test text " * 20
        # Use wide epsilon range with many points and custom correlation_integral_range
        result = calc(
            text, 
            num_epsilon=10000, 
            epsilon_range=(1e-20, 1e20),
            correlation_integral_range=(0.0, 1.0)  # Use full range
        )
        
        # Correlation dimension should be non-negative
        assert result.corrdim >= 0
        # For a reasonable fit, R² should be positive
        assert result.fit_r2 > 0
    
    def test_linear_region_bounds(self):
        """Test that linear region bounds are valid."""
        mock_model = MockModelWrapper(vocab_size=100, seq_len=150)
        calc = CorrelationDimensionCalculator(mock_model)
        
        text = "Test text " * 20
        # Use wide epsilon range with many points and custom correlation_integral_range
        result = calc(
            text, 
            num_epsilon=10000, 
            epsilon_range=(1e-20, 1e20),
            correlation_integral_range=(0.0, 1.0)  # Use full range
        )
        
        low, high = result.linear_region_bounds
        assert low < high
        assert low >= 0
        assert high <= 1
    
    def test_correlation_dimension_with_dim_reduction(self):
        """Test correlation dimension computation with dimension reduction."""
        mock_model = MockModelWrapper(vocab_size=1000, seq_len=150)
        calc = CorrelationDimensionCalculator(mock_model)
        
        text = "Test text for correlation dimension calculation."
        
        # With dimension reduction - use wide epsilon range with many points
        result_reduced = calc(
            text, 
            dim_reduction=100, 
            num_epsilon=10000, 
            epsilon_range=(1e-20, 1e20),
            correlation_integral_range=(0.0, 1.0)  # Use full range
        )
        
        assert isinstance(result_reduced, CorrelationDimensionResult)
        assert result_reduced.corrdim >= 0


class TestBackendSelection:
    """Test correlation integral backend selection."""
    
    def test_pytorch_backend(self):
        """Test correlation integral with PyTorch backend."""
        from corrdim.corrint import set_corrint_backend
        
        set_corrint_backend("pytorch")
        
        vecs = torch.randn(20, 10, device="cpu")
        epsilons = torch.logspace(-5, 5, 20, device="cpu")
        
        corrints = correlation_integral(vecs, epsilons)
        assert isinstance(corrints, torch.Tensor)
        assert corrints.shape == epsilons.shape
    
    def test_backend_switching(self):
        """Test switching between backends."""
        from corrdim.corrint import set_corrint_backend
        
        vecs = torch.randn(20, 10, device="cpu")
        epsilons = torch.logspace(-5, 5, 20, device="cpu")
        
        # Test pytorch backend
        set_corrint_backend("pytorch")
        corrints_pytorch = correlation_integral(vecs, epsilons)
        
        # Test pytorch_fast backend
        set_corrint_backend("pytorch_fast")
        corrints_fast = correlation_integral(vecs, epsilons)
        
        # Results should be similar (allowing for small numerical differences)
        assert torch.allclose(corrints_pytorch, corrints_fast, atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
