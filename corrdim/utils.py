"""
Utility functions for the corrdim library.
"""

import torch
import numpy as np
from typing import Optional, Union, Dict, Any
from .models import LanguageModelWrapper, create_model_wrapper


def load_model(
    model_name: str,
    tokenizer: Optional[object] = None,
    device: Optional[str] = None,
    **kwargs
) -> LanguageModelWrapper:
    """
    Load a language model wrapper.
    
    Args:
        model_name: Name or path of the model
        tokenizer: Tokenizer instance (if None, will load from model_name)
        device: Device to run on
        **kwargs: Additional arguments for model wrapper
        
    Returns:
        Language model wrapper instance
    """
    return create_model_wrapper(model_name, tokenizer=tokenizer, device=device, **kwargs)


def get_log_probabilities(
    model: Union[str, LanguageModelWrapper],
    text: str,
    context_length: int = 512,
    batch_size: int = 32,
    tokenizer: Optional[object] = None,
    device: Optional[str] = None
) -> np.ndarray:
    """
    Get log-probability vectors for a text using a language model.
    
    Args:
        model: Language model (name string or wrapper instance)
        text: Input text
        context_length: Maximum context length
        batch_size: Batch size for processing
        tokenizer: Tokenizer (if model is a string)
        device: Device to run on
        
    Returns:
        Array of log-probability vectors of shape (seq_len, vocab_size)
    """
    if isinstance(model, str):
        model_wrapper = load_model(model, tokenizer=tokenizer, device=device)
    else:
        model_wrapper = model
        
    return model_wrapper.get_log_probabilities(
        text, context_length=context_length, batch_size=batch_size
    )


def estimate_correlation_dimension_from_curve(
    epsilon_values: np.ndarray,
    correlation_integrals: np.ndarray,
    min_pairs_ratio: float = 0.01,
    max_pairs_ratio: float = 0.5,
    seq_len: Optional[int] = None
) -> float:
    """
    Estimate correlation dimension from correlation integral curve.
    
    Args:
        epsilon_values: Array of epsilon values
        correlation_integrals: Array of correlation integral values
        min_pairs_ratio: Minimum ratio of pairs to include
        max_pairs_ratio: Maximum ratio of pairs to include
        seq_len: Sequence length (if None, will be estimated)
        
    Returns:
        Estimated correlation dimension
    """
    if seq_len is None:
        # Estimate sequence length from correlation integral values
        seq_len = int(np.sqrt(2 * np.max(correlation_integrals) * len(correlation_integrals)))
    
    # Filter for linear region
    min_pairs = min_pairs_ratio * seq_len * (seq_len - 1) / 2
    max_pairs = max_pairs_ratio * seq_len * (seq_len - 1) / 2
    
    # Find indices where correlation integral is in the desired range
    valid_indices = (correlation_integrals * seq_len * (seq_len - 1) / 2 >= min_pairs) & \
                   (correlation_integrals * seq_len * (seq_len - 1) / 2 <= max_pairs)
    
    if np.sum(valid_indices) < 3:
        # Fallback: use all points
        valid_indices = np.ones_like(correlation_integrals, dtype=bool)
    
    # Fit power law: log(S(ε)) = d * log(ε) + c
    log_epsilon = np.log(epsilon_values[valid_indices])
    log_s = np.log(correlation_integrals[valid_indices] + 1e-10)
    
    # Linear regression using scikit-learn
    try:
        from sklearn.linear_model import LinearRegression
        
        reg = LinearRegression()
        reg.fit(log_epsilon.reshape(-1, 1), log_s)
        correlation_dimension = reg.coef_[0]
        
    except Exception:
        # Fallback to numpy polyfit
        correlation_dimension = np.polyfit(log_epsilon, log_s, 1)[0]
    
    return max(0.0, correlation_dimension)


def compute_pairwise_distances(log_probs: np.ndarray) -> np.ndarray:
    """
    Compute pairwise Euclidean distances between log-probability vectors.
    
    Args:
        log_probs: Log-probability vectors of shape (seq_len, vocab_size)
        
    Returns:
        Distance matrix of shape (seq_len, seq_len)
    """
    return np.linalg.norm(
        log_probs[:, np.newaxis, :] - log_probs[np.newaxis, :, :], 
        axis=2
    )


def find_recurrences(
    log_probs: np.ndarray, 
    epsilon: float
) -> np.ndarray:
    """
    Find recurrence pairs within distance threshold epsilon.
    
    Args:
        log_probs: Log-probability vectors of shape (seq_len, vocab_size)
        epsilon: Distance threshold
        
    Returns:
        Boolean matrix indicating recurrences
    """
    distances = compute_pairwise_distances(log_probs)
    return distances < epsilon


def compute_recurrence_rate(
    log_probs: np.ndarray, 
    epsilon: float
) -> float:
    """
    Compute the recurrence rate for a given distance threshold.
    
    Args:
        log_probs: Log-probability vectors of shape (seq_len, vocab_size)
        epsilon: Distance threshold
        
    Returns:
        Recurrence rate (fraction of pairs within distance threshold)
    """
    t = log_probs.shape[0]
    if t < 2:
        return 0.0
        
    recurrences = find_recurrences(log_probs, epsilon)
    
    # Count pairs within distance threshold (excluding diagonal)
    mask = np.triu(np.ones((t, t)), k=1).astype(bool)
    valid_recurrences = recurrences[mask]
    count = np.sum(valid_recurrences)
    
    # Normalize by total number of pairs
    total_pairs = t * (t - 1) / 2
    return count / total_pairs if total_pairs > 0 else 0.0


def validate_text_input(text: str, min_length: int = 10) -> str:
    """
    Validate and clean text input.
    
    Args:
        text: Input text
        min_length: Minimum required text length
        
    Returns:
        Cleaned text
        
    Raises:
        ValueError: If text is too short or invalid
    """
    if not isinstance(text, str):
        raise ValueError("Text must be a string")
    
    # Remove excessive whitespace
    text = " ".join(text.split())
    
    if len(text) < min_length:
        raise ValueError(f"Text must be at least {min_length} characters long")
    
    return text


def get_device_info() -> Dict[str, Any]:
    """
    Get information about available devices.
    
    Returns:
        Dictionary with device information
    """
    info = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "current_device": None,
        "device_name": None,
    }
    
    if torch.cuda.is_available():
        info["current_device"] = torch.cuda.current_device()
        info["device_name"] = torch.cuda.get_device_name()
    
    return info


def estimate_memory_usage(
    model_name: str,
    context_length: int = 512,
    batch_size: int = 32
) -> Dict[str, float]:
    """
    Estimate memory usage for model inference.
    
    Args:
        model_name: Name of the model
        context_length: Context length
        batch_size: Batch size
        
    Returns:
        Dictionary with memory usage estimates (in GB)
    """
    # Rough estimates based on model size and parameters
    model_sizes = {
        "gpt2": 0.5,
        "gpt2-medium": 1.5,
        "gpt2-large": 3.0,
        "gpt2-xl": 6.0,
    }
    
    # Estimate model size
    model_size_gb = 0.5  # Default
    for key, size in model_sizes.items():
        if key in model_name.lower():
            model_size_gb = size
            break
    
    # Estimate activation memory
    vocab_size = 50000  # Typical vocabulary size
    activation_memory_gb = (context_length * batch_size * vocab_size * 4) / (1024**3)  # 4 bytes per float32
    
    return {
        "model_memory_gb": model_size_gb,
        "activation_memory_gb": activation_memory_gb,
        "total_memory_gb": model_size_gb + activation_memory_gb,
    }
