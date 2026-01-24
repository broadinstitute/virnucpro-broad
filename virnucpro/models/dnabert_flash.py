"""DNABERT-S model wrapper with FlashAttention-2 integration.

This module provides a wrapper around the DNABERT-S model that automatically
uses FlashAttention-2 on compatible GPUs (Ampere+) with transparent fallback to
standard attention on older hardware.

FlashAttention-2 provides 2-4x attention speedup for transformer models without
changing model outputs or accuracy.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Any
import logging
from transformers import AutoModel, AutoTokenizer

from virnucpro.cuda.attention_utils import (
    get_attention_implementation,
    configure_flash_attention
)

logger = logging.getLogger('virnucpro.models.dnabert_flash')


class DNABERTWithFlashAttention(nn.Module):
    """
    Wrapper for DNABERT-S model with FlashAttention-2 optimization.

    This wrapper automatically detects GPU capabilities and uses FlashAttention-2
    when available (Ampere+ GPUs with PyTorch 2.2+), falling back gracefully to
    standard attention on older hardware.

    The wrapper maintains full compatibility with the original DNABERT-S model interface
    while providing 2-4x attention speedup on compatible GPUs.

    Attributes:
        model: The underlying DNABERT-S model
        attention_impl: Current attention implementation ("flash_attention_2" or "standard_attention")
        device: Device the model is running on

    Example:
        >>> model = DNABERTWithFlashAttention(base_model, device="cuda:0")
        >>> # Model automatically uses FlashAttention-2 if available
        >>> output = model(tokens)
    """

    def __init__(
        self,
        base_model: nn.Module,
        device: torch.device,
        logger_instance: Optional[logging.Logger] = None
    ):
        """
        Initialize DNABERT-S wrapper with FlashAttention-2 optimization.

        Args:
            base_model: Base DNABERT-S model from transformers
            device: Device to run model on
            logger_instance: Optional logger for configuration messages
        """
        super().__init__()

        self.model = base_model
        self.device = device
        log = logger_instance if logger_instance is not None else logger

        # Move model to device
        self.model = self.model.to(device)
        self.model.eval()

        # Detect and configure attention implementation
        self.attention_impl = get_attention_implementation()

        # Configure FlashAttention-2 if available
        self.model = configure_flash_attention(self.model, log)

        # Check BF16 support for memory efficiency
        self.use_bf16 = False
        if str(device).startswith('cuda'):
            capability = torch.cuda.get_device_capability(device)
            self.use_bf16 = capability[0] >= 8  # Ampere or newer

            if self.use_bf16:
                log.info("Using BF16 mixed precision for memory efficiency")
                self.model = self.model.bfloat16()
            else:
                log.debug("BF16 not available, using FP32")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> dict:
        """
        Forward pass through DNABERT-S model with optimal attention implementation.

        For FlashAttention-2, uses PyTorch's scaled_dot_product_attention context manager
        to ensure flash kernels are used. For standard attention, uses default behavior.

        Args:
            input_ids: Input token IDs, shape (batch_size, seq_len)
            attention_mask: Optional attention mask, shape (batch_size, seq_len)
            **kwargs: Additional arguments passed to the model

        Returns:
            Dictionary with model outputs including hidden states

        Example:
            >>> with torch.no_grad():
            ...     output = model(input_ids=batch_tokens, attention_mask=mask)
            ...     embeddings = output["last_hidden_state"]
        """
        # Use FlashAttention-2 context manager if available
        if self.attention_impl == "flash_attention_2":
            # Enable only FlashAttention kernel, disable fallbacks
            with torch.backends.cuda.sdp_kernel(
                enable_flash=True,
                enable_math=False,
                enable_mem_efficient=False
            ):
                return self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **kwargs
                )
        else:
            # Standard attention path
            return self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **kwargs
            )

    def __repr__(self) -> str:
        """String representation showing attention implementation."""
        return (
            f"DNABERTWithFlashAttention("
            f"attention={self.attention_impl}, "
            f"device={self.device}, "
            f"dtype={'bfloat16' if self.use_bf16 else 'float32'})"
        )


def load_dnabert_model(
    model_name: str = "zhihan1996/DNABERT-S",
    device: str = "cuda",
    logger_instance: Optional[logging.Logger] = None
) -> Tuple[DNABERTWithFlashAttention, Any]:
    """
    Load DNABERT-S model with FlashAttention-2 optimization.

    This function loads a DNABERT-S model and wraps it with FlashAttention-2 support,
    automatically detecting GPU capabilities and configuring optimal attention
    implementation.

    Args:
        model_name: Hugging Face model name (default: "zhihan1996/DNABERT-S")
        device: Device to load model on (e.g., "cuda", "cuda:0", "cpu")
        logger_instance: Optional logger for configuration messages

    Returns:
        Tuple of (model, tokenizer):
        - model: DNABERTWithFlashAttention wrapper
        - tokenizer: AutoTokenizer for the model

    Example:
        >>> model, tokenizer = load_dnabert_model("zhihan1996/DNABERT-S", "cuda:0")
        >>> # Model automatically uses FlashAttention-2 if available
        >>> sequences = ["ATCGATCGATCG", "GCTAGCTAGCTA"]
        >>> tokens = tokenizer(sequences, return_tensors="pt", padding=True)
        >>> with torch.no_grad():
        ...     results = model(input_ids=tokens["input_ids"].to("cuda:0"))
    """
    log = logger_instance if logger_instance is not None else logger

    log.info(f"Loading DNABERT-S model: {model_name}")

    # Load base DNABERT-S model using transformers
    base_model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    log.info(f"Model loaded: {model_name}")

    # Convert device string to torch.device
    device_obj = torch.device(device)

    # Wrap with FlashAttention-2 support
    model = DNABERTWithFlashAttention(
        base_model,
        device_obj,
        logger_instance=log
    )

    log.info(f"DNABERT-S ready: {model}")

    return model, tokenizer


def get_dnabert_embeddings(
    model: DNABERTWithFlashAttention,
    tokenizer: Any,
    sequences: list,
    max_length: int = 512
) -> Tuple[list, list]:
    """
    Extract embeddings from DNABERT-S model for a batch of sequences.

    This is a convenience function for extracting mean-pooled sequence embeddings.

    Args:
        model: DNABERTWithFlashAttention model
        tokenizer: DNABERT-S tokenizer
        sequences: List of (id, sequence) tuples
        max_length: Maximum sequence length (sequences truncated beyond this)

    Returns:
        Tuple of (ids, embeddings):
        - ids: List of sequence IDs
        - embeddings: List of embedding tensors (mean-pooled, CPU)

    Example:
        >>> sequences = [("seq1", "ATCGATCG"), ("seq2", "GCTAGCTA")]
        >>> ids, embeddings = get_dnabert_embeddings(model, tokenizer, sequences)
        >>> # embeddings[0].shape == (768,) for DNABERT-S
    """
    # Extract IDs and sequences
    ids = [seq_id for seq_id, _ in sequences]
    seq_strs = [seq_str for _, seq_str in sequences]

    # Tokenize sequences
    tokens = tokenizer(
        seq_strs,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length
    )

    # Move to device
    tokens = {k: v.to(model.device) for k, v in tokens.items()}

    # Forward pass
    with torch.no_grad():
        outputs = model(**tokens)
        hidden_states = outputs.last_hidden_state

    # Extract mean-pooled embeddings
    embeddings = []
    for i in range(len(sequences)):
        # Mean pool over sequence length (excluding padding)
        attention_mask = tokens["attention_mask"][i]
        valid_length = attention_mask.sum()
        embedding = hidden_states[i, :valid_length].mean(dim=0)

        # Convert to FP32 and move to CPU for storage
        embedding = embedding.float().cpu()
        embeddings.append(embedding)

    return ids, embeddings
