"""DNABERT-S model wrapper with FlashAttention-2 integration.

This module provides a wrapper around the DNABERT-S model that automatically
uses FlashAttention-2 on compatible GPUs (Ampere+) with transparent fallback to
standard attention on older hardware.

FlashAttention-2 provides 2-4x attention speedup for transformer models without
changing model outputs or accuracy.

NOTE: DNABERT-S (MosaicBERT) has a known issue where its Triton-based attention
falls back to FP32 PyTorch matmul when Triton is unavailable. This module patches
the attention to use PyTorch's native scaled_dot_product_attention (SDPA) which:
- Supports BF16 natively
- Uses Flash Attention kernels without requiring Triton
- Provides 2-4x speedup on Ampere+ GPUs
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Any
import logging
from transformers import AutoModel, AutoTokenizer
from einops import rearrange

from virnucpro.cuda.attention_utils import (
    get_attention_implementation,
    configure_flash_attention
)

logger = logging.getLogger('virnucpro.models.dnabert_flash')

# Flag to track if patch has been applied
_DNABERT_ATTENTION_PATCHED = False


def _patch_dnabert_attention():
    """
    Monkey-patch DNABERT-S attention to use PyTorch SDPA instead of broken Triton fallback.

    The original MosaicBERT attention has a fallback path that uses torch.matmul
    which doesn't handle BF16 correctly. This patch replaces it with PyTorch's
    scaled_dot_product_attention which:
    1. Handles BF16 natively
    2. Uses Flash Attention kernels automatically (no Triton needed)
    3. Provides the same speedup as Triton Flash Attention

    This patch is applied once at module load time.
    """
    global _DNABERT_ATTENTION_PATCHED
    if _DNABERT_ATTENTION_PATCHED:
        return

    try:
        # Import the DNABERT-S bert_layers module from HuggingFace cache
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        # Try to get the module - this may fail if model hasn't been loaded yet
        # We'll handle that case in load_dnabert_model
        import importlib
        import sys

        # Check if the module is already loaded (from a previous model load)
        bert_layers_module = None
        for module_name in list(sys.modules.keys()):
            if 'DNABERT-S' in module_name and 'bert_layers' in module_name:
                bert_layers_module = sys.modules[module_name]
                break

        if bert_layers_module is None:
            logger.debug("DNABERT-S bert_layers not yet loaded, patch will be applied on model load")
            return

        _apply_attention_patch(bert_layers_module)
        _DNABERT_ATTENTION_PATCHED = True

    except Exception as e:
        logger.warning(f"Could not patch DNABERT attention: {e}")


def _apply_attention_patch(bert_layers_module):
    """Apply the actual attention patch to a loaded bert_layers module."""
    global _DNABERT_ATTENTION_PATCHED

    if not hasattr(bert_layers_module, 'BertUnpadSelfAttention'):
        logger.warning("BertUnpadSelfAttention not found in bert_layers module")
        return

    BertUnpadSelfAttention = bert_layers_module.BertUnpadSelfAttention

    # Store original forward for reference
    original_forward = BertUnpadSelfAttention.forward

    def patched_forward(self, hidden_states: torch.Tensor, cu_seqlens: torch.Tensor,
                        max_seqlen_in_batch: int, indices: torch.Tensor,
                        attn_mask: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        """
        Patched attention forward using PyTorch SDPA.

        This replaces the original fallback path (torch.matmul) with PyTorch's
        scaled_dot_product_attention which properly supports BF16 and uses
        Flash Attention kernels automatically.
        """
        # Import padding utilities from the same module
        pad_input = bert_layers_module.pad_input
        unpad_input_only = bert_layers_module.unpad_input_only

        qkv = self.Wqkv(hidden_states)
        qkv = pad_input(qkv, indices, cu_seqlens.shape[0] - 1,
                        max_seqlen_in_batch)  # batch, max_seqlen_in_batch, thd
        qkv = rearrange(qkv,
                        'b s (t h d) -> b s t h d',
                        t=3,
                        h=self.num_attention_heads)

        # Extract Q, K, V and reshape for SDPA: (batch, heads, seq, dim)
        q = qkv[:, :, 0, :, :].permute(0, 2, 1, 3)  # b h s d
        k = qkv[:, :, 1, :, :].permute(0, 2, 1, 3)  # b h s d
        v = qkv[:, :, 2, :, :].permute(0, 2, 1, 3)  # b h s d

        # Use PyTorch's scaled_dot_product_attention
        # This automatically uses Flash Attention on compatible GPUs
        # and properly handles BF16 without dtype issues

        # Convert ALiBi bias to attention mask format for SDPA
        # bias shape: (batch, heads, seq, seq)
        # SDPA expects attn_mask to be additive (added to attention scores before softmax)
        # CRITICAL: Cast bias to match query dtype to avoid SDPA dtype mismatch
        attn_mask_sdpa = bias.to(dtype=q.dtype)

        # Apply dropout only during training
        dropout_p = self.p_dropout if self.training else 0.0

        # Use SDPA with Flash Attention
        attention = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask_sdpa,
            dropout_p=dropout_p,
            is_causal=False,  # BERT uses bidirectional attention
            scale=1.0 / math.sqrt(self.attention_head_size)
        )

        # Reshape back: (batch, heads, seq, dim) -> (batch, seq, heads, dim)
        attention = attention.permute(0, 2, 1, 3)  # b s h d

        # attn_mask is 1 for attend and 0 for don't
        attention = unpad_input_only(attention, torch.squeeze(attn_mask) == 1)
        return rearrange(attention, 'nnz h d -> nnz (h d)')

    # Apply the patch
    BertUnpadSelfAttention.forward = patched_forward
    _DNABERT_ATTENTION_PATCHED = True
    logger.info("DNABERT-S attention patched to use PyTorch SDPA (BF16 + Flash Attention enabled)")


def _ensure_attention_patched():
    """Ensure DNABERT attention is patched after model is loaded."""
    global _DNABERT_ATTENTION_PATCHED
    if _DNABERT_ATTENTION_PATCHED:
        return

    import sys
    for module_name in list(sys.modules.keys()):
        if 'DNABERT-S' in module_name and 'bert_layers' in module_name:
            bert_layers_module = sys.modules[module_name]
            _apply_attention_patch(bert_layers_module)
            return

    logger.warning("Could not find DNABERT-S bert_layers module to patch")


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

        # EXPERIMENTAL: Force FP32 to test vanilla compatibility
        # Original code checked capability[0] >= 8 for BF16
        self.use_bf16 = False
        log.info("EXPERIMENTAL: Forcing FP32 precision (BF16 disabled for vanilla compatibility test)")

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
            # Use new API if available (PyTorch 2.5+), fall back to deprecated API (PyTorch 2.2-2.4)
            try:
                if hasattr(torch.nn.attention, 'sdpa_kernel') and hasattr(torch.nn.attention, 'SDPBackend'):
                    from torch.nn.attention import SDPBackend
                    # New API uses SDPBackend enum
                    with torch.nn.attention.sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
                        return self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            **kwargs
                        )
                else:
                    # Old deprecated API
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
            except Exception:
                # If FlashAttention context fails, fall back to standard path
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

    # EXPERIMENTAL: Skip attention patch to test vanilla compatibility
    # Original code:
    # _ensure_attention_patched()
    log.info("EXPERIMENTAL: Skipping DNABERT-S attention patch (using stock Triton/matmul attention)")

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
