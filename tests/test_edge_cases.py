"""
Tests for edge cases and error handling in correlation dimension computation.
"""

import pytest
import numpy as np
import torch
from unittest.mock import Mock

from corrdim import CorrelationDimensionCalculator
from corrdim.calculator import CorrelationIntegralResult, CorrelationDimensionResult
from corrdim.models import LanguageModelWrapper


class MockModelWrapper(LanguageModelWrapper):
    """Mock model wrapper for testing."""
    
    def __init__(self, vocab_size=1000, seq_len=50, device="cpu"):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.device = device
        self.tokenizer = Mock()
        self.tokenizer.encode = Mock(return_value=list(range(seq_len)))
        self.tokenizer.decode = Mock(return_value="decoded text")
    
    def get_log_probabilities(self, text, context_length=None, dim_reduction=None, stride="auto", show_progress=False):
        """Return mock log probabilities."""
        np.random.seed(42)
        log_probs = torch.tensor(
            np.random.randn(self.seq_len, self.vocab_size).astype(np.float32),
            device=self.device
        )
        return log_probs


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def setup_method(self):
        """Set up backend to PyTorch for CPU compatibility."""
        from corrdim.corrint import set_corrint_backend
        set_corrint_backend("pytorch")
    
    def test_short_sequence(self):
        """Test with very short sequence."""
        mock_model = MockModelWrapper(vocab_size=100, seq_len=10)
        calc = CorrelationDimensionCalculator(mock_model)
        
        # Should raise an error for sequences that are too short (< 100 tokens)
        with pytest.raises(ValueError, match="sequence length is too short"):
            calc.compute_correlation_integral_curve_from_vector_sequences(
                torch.randn(50, 20),  # Only 50 points, needs > 100
                num_epsilon=10
            )
    
    def test_empty_text(self):
        """Test with empty text."""
        mock_model = MockModelWrapper(vocab_size=100, seq_len=0)
        calc = CorrelationDimensionCalculator(mock_model)
        
        text = ""
        
        # Empty text should be handled gracefully
        with pytest.raises((ValueError, AssertionError)):
            calc(text)
    
    def test_nan_in_vectors(self):
        """Test handling of NaN values in vectors."""
        mock_model = MockModelWrapper(vocab_size=100, seq_len=150)
        calc = CorrelationDimensionCalculator(mock_model)
        
        # Create vectors with NaN
        vecs = torch.randn(150, 20)
        vecs[0, 0] = float('nan')
        
        with pytest.raises(ValueError, match="Found nan or inf"):
            calc.compute_correlation_integral_curve_from_vector_sequences(
                vecs,
                num_epsilon=10
            )
    
    def test_inf_in_vectors(self):
        """Test handling of Inf values in vectors."""
        mock_model = MockModelWrapper(vocab_size=100, seq_len=150)
        calc = CorrelationDimensionCalculator(mock_model)
        
        # Create vectors with Inf
        vecs = torch.randn(150, 20)
        vecs[0, 0] = float('inf')
        
        with pytest.raises(ValueError, match="Found nan or inf"):
            calc.compute_correlation_integral_curve_from_vector_sequences(
                vecs,
                num_epsilon=10
            )
    
    def test_very_large_sequence(self):
        """Test with very large sequence."""
        mock_model = MockModelWrapper(vocab_size=100, seq_len=1000)
        calc = CorrelationDimensionCalculator(mock_model)
        
        text = "Test text " * 200
        
        # Should handle large sequences with wide epsilon range and many points
        result = calc(
            text, 
            num_epsilon=10000, 
            epsilon_range=(1e-20, 1e20),
            correlation_integral_range=(0.0, 1.0)  # Use full range
        )
        assert isinstance(result, CorrelationDimensionResult)
        assert result.sequence_length == 1000
    
    def test_extreme_epsilon_range(self):
        """Test with different epsilon ranges."""
        mock_model = MockModelWrapper(vocab_size=100, seq_len=150)
        calc = CorrelationDimensionCalculator(mock_model)
        
        text = "Test text " * 20
        
        # Use wide epsilon range with many points and custom correlation_integral_range
        # Test with full range like standard tests
        result = calc(
            text,
            epsilon_range=(1e-20, 1e20),
            num_epsilon=10000,
            correlation_integral_range=(0.0, 1.0)  # Use full range
        )
        assert isinstance(result, CorrelationDimensionResult)
        
        # Test with a different but still wide range
        result = calc(
            text,
            epsilon_range=(1e-15, 1e15),
            num_epsilon=10000,
            correlation_integral_range=(0.0, 1.0)  # Use full range
        )
        assert isinstance(result, CorrelationDimensionResult)
    
    def test_single_epsilon_value(self):
        """Test with single epsilon value."""
        from corrdim import correlation_integral
        from corrdim.corrint import set_corrint_backend
        
        # Use PyTorch backend for CPU compatibility
        set_corrint_backend("pytorch")
        
        vecs = torch.randn(50, 20, device="cpu")
        epsilons = torch.tensor([1.0], device="cpu")
        
        corrints = correlation_integral(vecs, epsilons)
        assert corrints.shape == (1,)
        assert 0 <= corrints[0] <= 1
    
    def test_identical_vectors(self):
        """Test with identical vectors."""
        from corrdim import correlation_integral
        from corrdim.corrint import set_corrint_backend
        
        # Use PyTorch backend for CPU compatibility
        set_corrint_backend("pytorch")
        
        vecs = torch.ones(10, 5, device="cpu")
        epsilons = torch.logspace(-5, 5, 20, device="cpu")
        
        corrints = correlation_integral(vecs, epsilons)
        
        # All vectors are identical, so correlation integral should be high for large epsilon
        assert corrints[-1] > 0.5  # Most pairs should be within threshold


class TestErrorHandling:
    """Test error handling and validation."""
    
    def test_invalid_epsilon_range(self):
        """Test with invalid epsilon range."""
        mock_model = MockModelWrapper(vocab_size=100, seq_len=100)
        calc = CorrelationDimensionCalculator(mock_model)
        
        text = "Test text " * 20
        
        # Invalid range (low > high)
        with pytest.raises((ValueError, AssertionError)):
            calc(
                text,
                epsilon_range=(1e10, 1e-10),  # Invalid: low > high
                num_epsilon=50
            )
    
    def test_zero_epsilon(self):
        """Test handling of zero epsilon."""
        from corrdim import correlation_integral
        from corrdim.corrint import set_corrint_backend
        
        # Use PyTorch backend for CPU compatibility
        set_corrint_backend("pytorch")
        
        vecs = torch.randn(20, 10, device="cpu")
        epsilons = torch.tensor([0.0, 1.0], device="cpu")
        
        # Zero epsilon should be handled
        corrints = correlation_integral(vecs, epsilons)
        assert corrints.shape == (2,)
        assert corrints[0] == 0  # Zero epsilon should give zero correlation integral


class TestPrecisionHandling:
    """Test handling of different precisions."""
    
    def setup_method(self):
        """Set up backend to PyTorch for CPU compatibility."""
        from corrdim.corrint import set_corrint_backend
        set_corrint_backend("pytorch")
    
    def test_float32_precision(self):
        """Test with float32 precision."""
        mock_model = MockModelWrapper(vocab_size=100, seq_len=150)
        calc = CorrelationDimensionCalculator(mock_model)
        
        text = "Test text " * 20
        
        result = calc(
            text,
            num_epsilon=10000,
            precision=torch.float32,
            epsilon_range=(1e-20, 1e20),
            correlation_integral_range=(0.0, 1.0)  # Use full range
        )
        
        assert isinstance(result, CorrelationDimensionResult)
        assert result.corrdim >= 0
    
    def test_float16_precision(self):
        """Test with float16 precision."""
        mock_model = MockModelWrapper(vocab_size=100, seq_len=150)
        calc = CorrelationDimensionCalculator(mock_model)
        
        text = "Test text " * 20
        
        result = calc(
            text,
            num_epsilon=10000,
            precision=torch.float16,
            epsilon_range=(1e-20, 1e20),
            correlation_integral_range=(0.0, 1.0)  # Use full range
        )
        
        assert isinstance(result, CorrelationDimensionResult)
        assert result.corrdim >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

