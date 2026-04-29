"""
Language model wrapper module.

This module provides interfaces for different language models to extract
log-probability vectors needed for correlation dimension computation.
"""

import torch
import numpy as np
import tqdm.auto as tqdm
from typing import List, Optional
from abc import ABC, abstractmethod
from transformers import AutoTokenizer, AutoModelForCausalLM
from .utils import reduce_dimension

_DEFAULT_FORWARD_CHUNK_SIZE = 512


class LanguageModelWrapper(ABC):
    """Abstract base class for language model wrappers."""

    @abstractmethod
    def get_log_probabilities(
        self,
        text: str,
        context_length: int = None,
        dim_reduction: int = None,
        stride: int = 1,
        show_progress: bool = False,
    ):
        """
        Extract log-probability vectors for each token position.

        Args:
            text: Input text
            context_length: Maximum context length

        Returns:
            Array of sampled log-probability vectors of shape (sampled_seq_len, vocab_size)
        """
        pass


class TransformersModelWrapper(LanguageModelWrapper):
    """Wrapper for Hugging Face Transformers models."""

    def __init__(
        self,
        model_name: str,
        tokenizer: Optional[object] = None,
        device: Optional[str] = None,
        forward_chunk_size: int = _DEFAULT_FORWARD_CHUNK_SIZE,
        **kwargs,
    ):
        """
        Initialize the transformers model wrapper.

        Args:
            model_name: Name or path of the model
            tokenizer: Tokenizer instance (if None, will load from model_name)
            device: Device to run on
            forward_chunk_size: Number of tokens per forward pass chunk.
                Reduce this value (e.g. 128) on systems with limited VRAM.
            torch_dtype: Data type for model weights
        """
        self.model_name = model_name
        self.forward_chunk_size = forward_chunk_size
        if device is not None:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        # Load tokenizer
        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        else:
            self.tokenizer = tokenizer

        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Prefer fast/low-CPU-memory loading on CUDA.
        # Keep explicit user-provided kwargs as highest priority.
        load_kwargs = dict(kwargs)
        if self.device.startswith("cuda"):
            load_kwargs.setdefault("device_map", "auto")
            load_kwargs.setdefault("low_cpu_mem_usage", True)
            load_kwargs.setdefault("torch_dtype", torch.float16)
        elif self.device == "mps":
            # MPS: no device_map auto, prefer float32 for stability
            load_kwargs.setdefault("torch_dtype", torch.float32)
        else:
            load_kwargs.setdefault("low_cpu_mem_usage", True)

        # Prefer torch_dtype by default; fall back to dtype for model loaders
        # that only accept dtype.
        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        except TypeError as exc:
            can_retry_with_dtype = "torch_dtype" in load_kwargs and "dtype" not in load_kwargs
            mentions_torch_dtype = "torch_dtype" in str(exc)
            if not (can_retry_with_dtype and mentions_torch_dtype):
                raise

            retry_kwargs = dict(load_kwargs)
            retry_kwargs["dtype"] = retry_kwargs.pop("torch_dtype")
            self.model = AutoModelForCausalLM.from_pretrained(model_name, **retry_kwargs)
            load_kwargs = retry_kwargs

        # Load model. With device_map=auto, do not call model.to(...).
        if "device_map" not in load_kwargs:
            self.model = self.model.to(self.device)

        self.model.eval()
        self.input_device = self._infer_input_device()

    def _infer_input_device(self) -> torch.device:
        def _to_device(mapped) -> Optional[torch.device]:
            # Accelerate may store device map entries as int (GPU ordinal),
            # torch.device, or strings like "cuda:0"/"cpu"/"disk".
            if isinstance(mapped, int):
                return torch.device(f"cuda:{mapped}")
            if isinstance(mapped, torch.device):
                return mapped
            s = str(mapped)
            if s in {"cpu"} or s.startswith("cuda") or s == "mps":
                return torch.device(s)
            return None

        # Accelerate-dispatched models may span multiple devices and may not expose
        # a single stable `.device`. Feed inputs to the first CUDA shard if present.
        hf_device_map = getattr(self.model, "hf_device_map", None)
        if isinstance(hf_device_map, dict):
            for mapped in hf_device_map.values():
                device = _to_device(mapped)
                if device is not None and device.type in ("cuda", "mps"):
                    return device
            return torch.device("cpu")
        if hasattr(self.model, "device"):
            return self.model.device
        return torch.device(self.device)

    def encode(self, text: str, **kwargs) -> List[int]:
        """Tokenize text."""
        return self.tokenizer.encode(text, **kwargs)

    def decode(self, tokens: List[int], **kwargs) -> str:
        """Decode tokens to text."""
        return self.tokenizer.decode(tokens, **kwargs)

    @torch.no_grad()
    def get_log_probabilities(
        self,
        text: str,
        context_length: int = None,
        dim_reduction: int = None,
        stride: int = 1,
        show_progress: bool = False,
    ) -> torch.Tensor:
        """
        Extract sampled log-probability vectors.

        Args:
            text: Input text
            context_length: Maximum context length
            stride: Sampling interval over token positions.
                Keep every `stride`-th token vector.

        Returns:
            Array of sampled log-probability vectors of shape (sampled_seq_len, vocab_size)
        """
        # Re-resolve input device in case model dispatch/cache changed after init.
        self.input_device = self._infer_input_device()

        # Tokenize text
        tokens = self._tokenize_texts([text])[0]
        context_length = self._resolve_context_length(context_length)

        token_stride = self._normalize_token_stride(stride)
        self._validate_dim_reduction(dim_reduction)

        if len(tokens) <= context_length:
            # Short text: process in a single pass, then sample by token position.
            return self._get_log_probabilities_single_pass(
                [tokens],
                dim_reduction=dim_reduction,
                token_stride=token_stride,
                show_progress=show_progress,
            )[0]
        else:
            return self._get_log_probabilities_sliding_window(
                [tokens],
                context_length=context_length,
                dim_reduction=dim_reduction,
                token_stride=token_stride,
                show_progress=show_progress,
            )[0]

    @torch.no_grad()
    def get_log_probabilities_batch(
        self,
        texts: List[str],
        context_length: Optional[int] = None,
        dim_reduction: Optional[int] = None,
        stride: int = 1,
        show_progress: bool = False,
        batch_size: Optional[int] = None,
    ) -> List[torch.Tensor]:
        """Encode ``texts`` using HuggingFace batched ``forward`` (padded batch, ``attention_mask``).

        Returns one tensor per input string; rows can differ in length (``sampled_seq_len``) when
        token lengths differ. This API only supports short sequences:
        every input must satisfy ``len(tokens) <= context_length``.
        """
        if batch_size is not None and batch_size < 1:
            raise ValueError("batch_size must be a positive integer or None.")
        if not texts:
            return []

        self.input_device = self._infer_input_device()
        context_length = self._resolve_context_length(context_length)

        token_stride = self._normalize_token_stride(stride)
        self._validate_dim_reduction(dim_reduction)

        token_lists = self._tokenize_texts(texts)
        max_tokens = max(len(tokens) for tokens in token_lists)
        if max_tokens > context_length:
            raise ValueError(
                "get_log_probabilities_batch only supports short texts; "
                f"got max token length {max_tokens} > context_length {context_length}."
            )

        eff_bs = 1 if batch_size is None else batch_size
        out: List[torch.Tensor] = []
        for i in range(0, len(token_lists), eff_bs):
            chunk_tokens = token_lists[i : i + eff_bs]
            out.extend(
                self._get_log_probabilities_single_pass(
                    chunk_tokens,
                    dim_reduction=dim_reduction,
                    token_stride=token_stride,
                    show_progress=show_progress,
                )
            )
        return out

    def _normalize_token_stride(self, stride: int) -> int:
        if not isinstance(stride, int) or stride < 1:
            raise ValueError("stride must be a positive integer")
        return stride

    def _resolve_context_length(self, context_length: Optional[int]) -> int:
        if context_length is None:
            return int(self.model.config.max_position_embeddings)
        max_ctx = int(self.model.config.max_position_embeddings)
        if context_length > max_ctx:
            raise ValueError(
                f"Context length exceeds the max length allowed by the model: {self.model.config.max_position_embeddings}"
            )
        return context_length

    def _validate_dim_reduction(self, dim_reduction: Optional[int]) -> None:
        if dim_reduction is None:
            return
        if dim_reduction > self.model.config.vocab_size:
            raise ValueError(
                f"Dim reduction must be less than or equal to the vocabulary size: {self.model.config.vocab_size}"
            )

    def _tokenize_texts(self, texts: List[str]) -> List[List[int]]:
        token_lists = [self.tokenizer.encode(text, add_special_tokens=False) for text in texts]
        if any(len(tokens) == 0 for tokens in token_lists):
            raise ValueError("Input text produced an empty token sequence.")
        return token_lists

    @staticmethod
    def _log_softmax_inplace(logits: torch.Tensor, dim: int) -> torch.Tensor:
        logsumexp = torch.logsumexp(logits, dim=dim, keepdim=True)
        torch.sub(logits, logsumexp, out=logits)
        return logits

    def _get_log_probabilities_single_pass(
        self,
        token_lists: List[List[int]],
        dim_reduction: Optional[int] = None,
        token_stride: int = 1,
        show_progress: bool = False,
    ) -> List[torch.Tensor]:
        """Get log probabilities for short token sequences in a single batched pass."""
        if not token_lists:
            return []

        lengths = [len(tokens) for tokens in token_lists]
        batch_size = len(token_lists)
        max_len = max(lengths)
        pad_id = int(self.tokenizer.pad_token_id)
        input_ids = torch.full((batch_size, max_len), pad_id, dtype=torch.long, device=self.input_device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long, device=self.input_device)
        for b, tokens in enumerate(token_lists):
            L = len(tokens)
            input_ids[b, :L] = torch.tensor(tokens, dtype=torch.long, device=self.input_device)
            attention_mask[b, :L] = 1

        cache = None
        per_row_chunks: List[List[torch.Tensor]] = [[] for _ in range(batch_size)]

        for chunk_start in tqdm.trange(
            0,
            max_len,
            self.forward_chunk_size,
            disable=not show_progress,
            desc="Computing log-probs",
        ):
            chunk_end = min(chunk_start + self.forward_chunk_size, max_len)
            # With KV cache enabled, many models expect attention_mask length to include
            # both past and current tokens (prefix up to chunk_end), not only current chunk.
            outputs = self.model(
                input_ids[:, chunk_start:chunk_end],
                attention_mask=attention_mask[:, :chunk_end],
                past_key_values=cache,
                use_cache=True,
            )
            cache = outputs.past_key_values
            logp = self._log_softmax_inplace(outputs.logits, dim=-1)

            if dim_reduction is not None:
                logp = reduce_dimension(logp, method="group_add", num_groups=dim_reduction)

            for b in range(batch_size):
                L_b = lengths[b]
                if chunk_start >= L_b:
                    continue
                valid_end = min(chunk_end, L_b) - chunk_start
                if valid_end <= 0:
                    continue
                row_logp = logp[b, :valid_end]
                offset = (-chunk_start) % token_stride
                if offset < row_logp.shape[0]:
                    sampled_chunk = row_logp[offset::token_stride]
                    if sampled_chunk.shape[0] > 0:
                        per_row_chunks[b].append(sampled_chunk)

        result: List[torch.Tensor] = []
        for b in range(batch_size):
            log_probs = torch.cat(per_row_chunks[b], dim=0)
            expected_rows = (lengths[b] - 1) // token_stride + 1
            assert log_probs.shape[0] == expected_rows
            result.append(log_probs)
        return result

    def _get_log_probabilities_sliding_window(
        self,
        token_lists: List[List[int]],
        context_length: int,
        dim_reduction: Optional[int] = None,
        token_stride: int = 1,
        show_progress: bool = False,
    ) -> List[torch.Tensor]:
        """Get sampled log probabilities for long token sequences using sliding windows."""
        if not token_lists:
            return []

        result: List[torch.Tensor] = []
        iterator = tqdm.trange(len(token_lists), disable=not show_progress, desc="Computing log-probs")
        for idx in iterator:
            tokens = token_lists[idx]
            seq_len = len(tokens)
            input_ids = torch.tensor([tokens], device=self.input_device)

            # Internal step for long-sequence inference. This is separate from token_stride.
            window_step = max(1, context_length // 10)

            sampled_log_probs = []
            for pos in range(0, seq_len, window_step):
                start = max(0, pos + window_step - context_length)
                end = min(pos + window_step, seq_len)
                ntokens_to_update = end - pos
                if ntokens_to_update == 0:
                    continue

                outputs = self.model(input_ids[:, start:end])
                logits = outputs.logits[0, -ntokens_to_update:, :]
                logp = torch.log_softmax(logits, dim=-1)

                if dim_reduction is not None:
                    logp = reduce_dimension(logp, method="group_add", num_groups=dim_reduction)

                global_positions = torch.arange(pos, end, device=logp.device)
                sampled_chunk = logp[(global_positions % token_stride) == 0]
                if sampled_chunk.shape[0] > 0:
                    sampled_log_probs.append(sampled_chunk)

            log_probs = torch.cat(sampled_log_probs, dim=0)
            expected_rows = (seq_len - 1) // token_stride + 1
            assert log_probs.shape[0] == expected_rows
            result.append(log_probs)
        return result


class GPT2Wrapper(TransformersModelWrapper):
    """Specialized wrapper for GPT-2 models."""

    def __init__(self, model_size: str = "gpt2", **kwargs):
        """
        Initialize GPT-2 wrapper.

        Args:
            model_size: GPT-2 model size ('gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl')
            **kwargs: Additional arguments for TransformersModelWrapper
        """
        model_name = model_size if model_size.startswith("gpt2") else f"gpt2-{model_size}"
        super().__init__(model_name, **kwargs)


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
