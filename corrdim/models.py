"""
Language model wrapper module.

This module provides interfaces for different language models to extract
log-probability vectors needed for correlation dimension computation.
"""

import torch
import numpy as np
import tqdm.auto as tqdm
from typing import Optional, Union, List
from abc import ABC, abstractmethod
from transformers import AutoTokenizer, AutoModelForCausalLM


class LanguageModelWrapper(ABC):
    """Abstract base class for language model wrappers."""

    @abstractmethod
    def get_log_probabilities(self, text: str, context_length: int = None, dim_reduction: int = None, stride: Union[int, str] = "auto", show_progress: bool = False):
        """
        Extract log-probability vectors for each token position.

        Args:
            text: Input text
            context_length: Maximum context length
            batch_size: Batch size for processing

        Returns:
            Array of log-probability vectors of shape (seq_len, vocab_size)
        """
        pass


class TransformersModelWrapper(LanguageModelWrapper):
    """Wrapper for Hugging Face Transformers models."""

    def __init__(
        self,
        model_name: str,
        tokenizer: Optional[object] = None,
        device: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize the transformers model wrapper.

        Args:
            model_name: Name or path of the model
            tokenizer: Tokenizer instance (if None, will load from model_name)
            device: Device to run on
            torch_dtype: Data type for model weights
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load tokenizer
        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        else:
            self.tokenizer = tokenizer

        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)

        # Move model to specified device
        self.model = self.model.to(self.device)

        self.model.eval()

    def encode(self, text: str, **kwargs) -> List[int]:
        """Tokenize text."""
        return self.tokenizer.encode(text, **kwargs)

    def decode(self, tokens: List[int], **kwargs) -> str:
        """Decode tokens to text."""
        return self.tokenizer.decode(tokens, **kwargs)

    @torch.no_grad()
    def get_log_probabilities(
        self, text: str, context_length: int = None, dim_reduction: int = None, stride: Union[int, str] = "auto", show_progress: bool = False
    ) -> torch.Tensor:
        """
        Extract log-probability vectors using sliding window approach.

        Args:
            text: Input text
            context_length: Maximum context length
            stride: Stride for sliding window. Can be:
                - int: Fixed stride value
                - "auto": Automatically set stride to max(1, context_length // 10)

        Returns:
            Array of log-probability vectors of shape (seq_len, vocab_size)
        """
        # Tokenize text
        tokens = self.tokenizer.encode(text, add_special_tokens=False)

        if context_length is None:
            context_length = self.model.config.max_position_embeddings
        elif context_length > self.model.config.max_position_embeddings:
            raise ValueError(
                f"Context length exceeds the max length allowed by the model: {self.model.config.max_position_embeddings}"
            )

        # check if stride exceeds or equal to context_length
        if stride == "auto":
            stride = max(1, context_length // 10)
        elif not isinstance(stride, int) or stride < 1:
            raise ValueError("stride must be a positive integer or 'auto'")
        elif stride >= context_length:
            raise ValueError(
                f"Stride must be less than context length: {context_length}. Using a large stride will decrease the reliability of the correlation dimension estimation."
            )

        if dim_reduction is not None:
            if dim_reduction > self.model.config.vocab_size:
                raise ValueError(f"Dim reduction must be less than or equal to the vocabulary size: {self.model.config.vocab_size}")
            # print("Warning: Using dimension reduction may change the correlation dimension value.")

        if len(tokens) <= context_length:
            # Short text: process all at once
            return self._get_log_probabilities_single_pass(tokens, dim_reduction=dim_reduction, stride=stride, show_progress=show_progress)
        else:
            # Long text: use sliding window with stride
            return self._get_log_probabilities_sliding_window(tokens, context_length, dim_reduction=dim_reduction, stride=stride, show_progress=show_progress)

    def _get_log_probabilities_single_pass(self, tokens: List[int], dim_reduction: int = None, stride: int = None, show_progress: bool = False) -> torch.Tensor:
        """Get log probabilities for short texts in a single pass."""
        input_ids = torch.tensor([tokens], device=self.device)

        def log_softmax_(logits: torch.Tensor, dim) -> torch.Tensor:
            """Log softmax in place."""
            logsumexp = torch.logsumexp(logits, dim=dim, keepdim=True)
            torch.sub(logits, logsumexp, out=logits)
            return logits

        if stride is None:
            stride = len(tokens)

        cache = None
        log_probs = []
        total_seq_len = len(tokens)
        for i in tqdm.trange(0, len(tokens), stride, disable=not show_progress, desc="Computing log-probs"):
            outputs = self.model(input_ids[:, i:i+stride], past_key_values=cache, use_cache=True)
            cache = outputs.past_key_values

            # Convert to log
            logp = log_softmax_(outputs.logits[0], dim=-1)
        
            if dim_reduction is not None:
                current_seq_len, vocab_size = logp.shape
                # index = torch.randint(0, dim_reduction, (vocab_size,), device=logp.device)
                index = torch.arange(vocab_size, device=logp.device) % dim_reduction

                # Sum reduction: manually implement since index_reduce_ doesn't support "sum"
                reduced = torch.zeros(current_seq_len, dim_reduction, device=logp.device, dtype=logp.dtype)
                reduced.scatter_add_(-1, index.unsqueeze(0).expand(current_seq_len, -1), logp)
                logp = reduced

            log_probs.append(logp)
        
        log_probs = torch.cat(log_probs, dim=0)
        assert log_probs.shape[0] == total_seq_len
        return log_probs

    def _get_log_probabilities_sliding_window(
        self, tokens: List[int], context_length: int, dim_reduction: int = None, stride: int = None, show_progress: bool = False
    ) -> torch.Tensor:
        """Get log probabilities for long texts using sliding window with stride."""
        seq_len = len(tokens)
        vocab_size = self.model.config.vocab_size

        input_ids = torch.tensor([tokens], device=self.device)

        # Initialize output array
        log_probs = []
        for pos in tqdm.trange(0, seq_len, stride, disable=not show_progress, desc="Computing log-probs"):
            start = max(0, pos + stride - context_length)
            end = min(pos + stride, seq_len)
            ntokens_to_update = end - pos
            if ntokens_to_update == 0:
                continue

            outputs = self.model(input_ids[:, start:end])
            logits = outputs.logits[0, -ntokens_to_update:, :]
            logp = torch.log_softmax(logits, dim=-1)

            if dim_reduction is not None:
                stride_seq_len, vocab_size = logp.shape
                # index = torch.randint(0, dim_reduction, (vocab_size,), device=logp.device)
                index = torch.arange(vocab_size, device=logp.device) % dim_reduction

                # Sum reduction: manually implement since index_reduce_ doesn't support "sum"
                reduced = torch.zeros(stride_seq_len, dim_reduction, device=logp.device, dtype=logp.dtype)
                reduced.scatter_add_(-1, index.unsqueeze(0).expand(stride_seq_len, -1), logp)
                logp = reduced

            log_probs.append(logp)

        log_probs = torch.cat(log_probs, dim=0)
        assert log_probs.shape[0] == seq_len
        return log_probs


class GPT2Wrapper(TransformersModelWrapper):
    """Specialized wrapper for GPT-2 models."""

    def __init__(self, model_size: str = "gpt2", **kwargs):
        """
        Initialize GPT-2 wrapper.

        Args:
            model_size: GPT-2 model size ('gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl')
            **kwargs: Additional arguments for TransformersModelWrapper
        """
        super().__init__(f"gpt2-{model_size}" if model_size != "gpt2" else "gpt2", **kwargs)


class LLaMAWrapper(TransformersModelWrapper):
    """Specialized wrapper for LLaMA models."""

    def __init__(self, model_name: str = "meta-llama/Llama-2-7b-hf", **kwargs):
        """
        Initialize LLaMA wrapper.

        Args:
            model_name: LLaMA model name
            **kwargs: Additional arguments for TransformersModelWrapper
        """
        super().__init__(model_name, **kwargs)


def create_model_wrapper(
    model_name: str,
    tokenizer: Optional[object] = None,
    device: Optional[str] = None,
    **kwargs,
) -> LanguageModelWrapper:
    """
    Factory function to create appropriate model wrapper.

    Args:
        model_name: Name of the model
        tokenizer: Tokenizer instance
        device: Device to run on
        **kwargs: Additional arguments

    Returns:
        Appropriate model wrapper instance
    """
    # Try to determine model type from name
    if "gpt2" in model_name.lower():
        return GPT2Wrapper(model_name, tokenizer=tokenizer, device=device, **kwargs)
    elif "llama" in model_name.lower():
        return LLaMAWrapper(model_name, tokenizer=tokenizer, device=device, **kwargs)
    else:
        # Default to generic transformers wrapper
        return TransformersModelWrapper(model_name, tokenizer=tokenizer, device=device, **kwargs)
