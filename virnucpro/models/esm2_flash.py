"""ESM-2 model wrapper with FlashAttention-2 integration.

This module provides a wrapper around the fair-esm ESM-2 model that automatically
uses FlashAttention-2 on compatible GPUs (Ampere+) with transparent fallback to
standard attention on older hardware.

FlashAttention-2 provides 2-4x attention speedup for transformer models without
changing model outputs or accuracy.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Any, Literal
import logging
import esm

from virnucpro.cuda.attention_utils import (
    get_attention_implementation,
    configure_flash_attention
)

logger = logging.getLogger('virnucpro.models.esm2_flash')


class ESM2WithFlashAttention(nn.Module):
    """
    Wrapper for ESM-2 model with FlashAttention-2 optimization.

    This wrapper automatically detects GPU capabilities and uses FlashAttention-2
    when available (Ampere+ GPUs with PyTorch 2.2+), falling back gracefully to
    standard attention on older hardware.

    The wrapper maintains full compatibility with the original ESM-2 model interface
    while providing 2-4x attention speedup on compatible GPUs.

    Attributes:
        model: The underlying ESM-2 model
        attention_impl: Current attention implementation ("flash_attention_2" or "standard_attention")
        device: Device the model is running on

    Example:
        >>> model = ESM2WithFlashAttention(base_model, device="cuda:0")
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
        Initialize ESM-2 wrapper with FlashAttention-2 optimization.

        Args:
            base_model: Base ESM-2 model from esm.pretrained
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

        # EXPERIMENTAL: Force FP32 to test vanilla compatibility
        # Original code checked capability[0] >= 8 for BF16
        self.use_bf16 = False
        log.info("EXPERIMENTAL: Forcing FP32 precision (BF16 disabled for vanilla compatibility test)")

    def forward(
        self,
        tokens: torch.Tensor,
        repr_layers: Optional[list] = None,
        return_contacts: bool = False
    ) -> dict:
        """
        Forward pass through ESM-2 model with optimal attention implementation.

        For FlashAttention-2, uses PyTorch's scaled_dot_product_attention context manager
        to ensure flash kernels are used. For standard attention, uses default behavior.

        Args:
            tokens: Input token IDs, shape (batch_size, seq_len)
            repr_layers: List of layer indices to return representations for
            return_contacts: Whether to return contact predictions

        Returns:
            Dictionary with model outputs including representations

        Example:
            >>> with torch.no_grad():
            ...     output = model(batch_tokens, repr_layers=[36])
            ...     embeddings = output["representations"][36]
        """
        if repr_layers is None:
            repr_layers = [36]  # Default to final layer for ESM-2

        # Use FlashAttention-2 context manager if available
        if self.attention_impl == "flash_attention_2":
            # Enable only FlashAttention kernel, disable fallbacks
            # Use new API if available (PyTorch 2.5+), fall back to deprecated API (PyTorch 2.2-2.4)
            try:
                if hasattr(torch.nn.attention, 'sdpa_kernel') and hasattr(torch.nn.attention, 'SDPBackend'):
                    from torch.nn.attention import SDPBackend
                    # New API uses SDPBackend enum
                    with torch.nn.attention.sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
                        return self.model(
                            tokens,
                            repr_layers=repr_layers,
                            return_contacts=return_contacts
                        )
                else:
                    # Old deprecated API
                    with torch.backends.cuda.sdp_kernel(
                        enable_flash=True,
                        enable_math=False,
                        enable_mem_efficient=False
                    ):
                        return self.model(
                            tokens,
                            repr_layers=repr_layers,
                            return_contacts=return_contacts
                        )
            except Exception:
                # If FlashAttention context fails, fall back to standard path
                return self.model(
                    tokens,
                    repr_layers=repr_layers,
                    return_contacts=return_contacts
                )
        else:
            # Standard attention path
            return self.model(
                tokens,
                repr_layers=repr_layers,
                return_contacts=return_contacts
            )

    def __repr__(self) -> str:
        """String representation showing attention implementation."""
        return (
            f"ESM2WithFlashAttention("
            f"attention={self.attention_impl}, "
            f"device={self.device}, "
            f"dtype={'bfloat16' if self.use_bf16 else 'float32'})"
        )


def load_esm2_model(
    model_name: Literal[
        "esm2_t33_650M_UR50D",
        "esm2_t36_3B_UR50D",
        "esm2_t48_15B_UR50D"
    ] = "esm2_t36_3B_UR50D",
    device: str = "cuda",
    logger_instance: Optional[logging.Logger] = None
) -> Tuple[ESM2WithFlashAttention, Any]:
    """
    Load ESM-2 model with FlashAttention-2 optimization.

    This function loads an ESM-2 model and wraps it with FlashAttention-2 support,
    automatically detecting GPU capabilities and configuring optimal attention
    implementation.

    Supported models:
    - esm2_t33_650M_UR50D: 650M parameters, 33 layers
    - esm2_t36_3B_UR50D: 3B parameters, 36 layers (default)
    - esm2_t48_15B_UR50D: 15B parameters, 48 layers

    Args:
        model_name: Name of ESM-2 model variant to load
        device: Device to load model on (e.g., "cuda", "cuda:0", "cpu")
        logger_instance: Optional logger for configuration messages

    Returns:
        Tuple of (model, batch_converter):
        - model: ESM2WithFlashAttention wrapper
        - batch_converter: ESM alphabet batch converter for tokenization

    Example:
        >>> model, batch_converter = load_esm2_model("esm2_t33_650M_UR50D", "cuda:0")
        >>> # Model automatically uses FlashAttention-2 if available
        >>> sequences = [("protein1", "MKTAYIAK"), ("protein2", "VLSPADKTNV")]
        >>> labels, strs, tokens = batch_converter(sequences)
        >>> with torch.no_grad():
        ...     results = model(tokens.to("cuda:0"), repr_layers=[33])
    """
    log = logger_instance if logger_instance is not None else logger

    log.info(f"Loading ESM-2 model: {model_name}")

    # Load base ESM-2 model using fair-esm
    model_loader = getattr(esm.pretrained, model_name)
    base_model, alphabet = model_loader()

    log.info(f"Model loaded: {model_name}")

    # Convert device string to torch.device
    device_obj = torch.device(device)

    # Wrap with FlashAttention-2 support
    model = ESM2WithFlashAttention(
        base_model,
        device_obj,
        logger_instance=log
    )

    # Get batch converter from alphabet
    batch_converter = alphabet.get_batch_converter()

    log.info(f"ESM-2 ready: {model}")

    return model, batch_converter


def get_esm2_embeddings(
    model: ESM2WithFlashAttention,
    batch_converter: Any,
    sequences: list,
    layer: int = 36,
    truncation_length: int = 1024
) -> Tuple[list, list]:
    """
    Extract embeddings from ESM-2 model for a batch of sequences.

    This is a convenience function for extracting mean-pooled sequence embeddings.

    Args:
        model: ESM2WithFlashAttention model
        batch_converter: ESM alphabet batch converter
        sequences: List of (id, sequence) tuples
        layer: Layer index to extract representations from (default: 36 for 3B model)
        truncation_length: Maximum sequence length (sequences truncated beyond this)

    Returns:
        Tuple of (ids, embeddings):
        - ids: List of sequence IDs
        - embeddings: List of embedding tensors (mean-pooled, CPU)

    Example:
        >>> sequences = [("seq1", "MKTAYIAK"), ("seq2", "VLSPADKTNV")]
        >>> ids, embeddings = get_esm2_embeddings(model, batch_converter, sequences)
        >>> # embeddings[0].shape == (2560,) for 3B model
    """
    # Truncate sequences if needed
    truncated_sequences = []
    for seq_id, seq_str in sequences:
        if len(seq_str) > truncation_length:
            seq_str = seq_str[:truncation_length]
        truncated_sequences.append((seq_id, seq_str))

    # Convert batch
    labels, strs, tokens = batch_converter(truncated_sequences)
    tokens = tokens.to(model.device)

    # Forward pass
    with torch.no_grad():
        results = model(tokens, repr_layers=[layer])
        representations = results["representations"][layer]

    # Extract mean-pooled embeddings
    ids = []
    embeddings = []

    for i, (seq_id, seq_str) in enumerate(truncated_sequences):
        # Mean pool positions 1 to len+1 (skip BOS token)
        seq_len = min(len(seq_str), truncation_length)
        embedding = representations[i, 1:seq_len + 1].mean(dim=0)

        # Convert to FP32 and move to CPU for storage
        embedding = embedding.float().cpu()

        ids.append(seq_id)
        embeddings.append(embedding)

    return ids, embeddings
