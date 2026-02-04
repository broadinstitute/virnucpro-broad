"""Position ID generation and FlashAttention varlen utilities for packed sequences.

This module provides utilities for processing packed sequences (multiple sequences
concatenated into a single tensor) with FlashAttention's varlen API. The key
functionality includes:

1. Position ID generation that resets at each sequence boundary
2. FlashAttention varlen wrapper with input validation

Packed sequences use cumulative sequence lengths (cu_seqlens) to define boundaries.
For example:
    cu_seqlens = [0, 3, 7, 10]  # 3 sequences of lengths 3, 4, 3
    position_ids = [0, 1, 2, 0, 1, 2, 3, 0, 1, 2]  # Reset at each boundary
"""

import torch
import logging
from typing import Optional

logger = logging.getLogger('virnucpro.models.packed_attention')

# FlashAttention availability check
try:
    import flash_attn
    from flash_attn import flash_attn_varlen_func
    from packaging import version

    FLASH_ATTN_AVAILABLE = True

    # Version check for compatibility (Gap 11)
    if version.parse(flash_attn.__version__) < version.parse("2.6.0"):
        logger.warning(
            f"flash-attn {flash_attn.__version__} < 2.6.0. "
            "Upgrade for best performance: pip install flash-attn>=2.6.0 --no-build-isolation"
        )
except ImportError:
    FLASH_ATTN_AVAILABLE = False
    logger.info(
        "flash-attn not available. Install for 2-3x speedup: "
        "pip install flash-attn>=2.6.0 --no-build-isolation"
    )


def create_position_ids_packed(cu_seqlens: torch.Tensor) -> torch.Tensor:
    """
    Create position IDs that reset at each sequence boundary.

    For packed sequences, position IDs must reset to 0 at each sequence boundary
    rather than being sequential across the entire batch. This is critical for
    correct positional embeddings.

    Example:
        >>> cu_seqlens = torch.tensor([0, 3, 7, 10], dtype=torch.int32)
        >>> position_ids = create_position_ids_packed(cu_seqlens)
        >>> position_ids
        tensor([0, 1, 2, 0, 1, 2, 3, 0, 1, 2])

        NOT: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (WRONG - sequential)

    Args:
        cu_seqlens: Cumulative sequence lengths [num_sequences + 1]
                    Must start with 0, be monotonically increasing.
                    Format: [0, len1, len1+len2, ...] with dtype int32

    Returns:
        Position IDs tensor [total_tokens] with dtype=torch.long

    Raises:
        AssertionError: If cu_seqlens format is invalid
    """
    # Validation: cu_seqlens must start with 0
    assert cu_seqlens[0] == 0, (
        f"cu_seqlens must start with 0, got {cu_seqlens[0]}"
    )

    # Validation: cu_seqlens must be monotonically increasing
    if len(cu_seqlens) > 1:
        diffs = cu_seqlens[1:] - cu_seqlens[:-1]
        assert torch.all(diffs > 0), (
            "cu_seqlens must be monotonically increasing"
        )

    # Get total length from last element
    total_len = cu_seqlens[-1].item()
    num_sequences = len(cu_seqlens) - 1

    # Create position IDs tensor
    position_ids = torch.zeros(total_len, dtype=torch.long, device=cu_seqlens.device)

    # Fill position IDs for each sequence
    for i in range(num_sequences):
        start = cu_seqlens[i].item()
        end = cu_seqlens[i + 1].item()
        seq_len = end - start

        # Generate position IDs [0, 1, 2, ..., seq_len-1] for this sequence
        position_ids[start:end] = torch.arange(seq_len, device=cu_seqlens.device)

    # Validation: position IDs must reset to 0 at each boundary (except last)
    for i in range(num_sequences):
        boundary_idx = cu_seqlens[i].item()
        assert position_ids[boundary_idx] == 0, (
            f"Position ID at boundary {i} (index {boundary_idx}) should be 0, "
            f"got {position_ids[boundary_idx]}"
        )

    return position_ids


def flash_attn_varlen_wrapper(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    dropout_p: float = 0.0,
    causal: bool = False,
    softmax_scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Wrapper around flash_attn_varlen_func with validation.

    This wrapper provides input validation and graceful error handling for
    FlashAttention's variable-length attention function. It validates dtypes,
    cu_seqlens format, and provides helpful error messages.

    Args:
        q, k, v: Query/Key/Value tensors [total_tokens, num_heads, head_dim]
                 Must be FP16 or BF16 (FlashAttention requirement)
        cu_seqlens: Cumulative sequence lengths [batch_size + 1], dtype=int32
                    Format: [0, len1, len1+len2, ...]
        max_seqlen: Maximum sequence length in batch
        dropout_p: Dropout probability (0.0 for inference)
        causal: Use causal attention (False for BERT/ESM bidirectional)
        softmax_scale: Scaling factor for attention scores. If None, defaults to
                       1/sqrt(head_dim). ESM-2 uses 1/sqrt(64) = 0.125.

    Returns:
        Attention output [total_tokens, num_heads, head_dim]

    Raises:
        ImportError: If flash-attn not installed
        RuntimeError: If CUDA not available or inputs invalid
        ValueError: If dtype or cu_seqlens validation fails

    Example:
        >>> q = torch.randn(10, 32, 64, dtype=torch.float16, device='cuda')
        >>> k = torch.randn(10, 32, 64, dtype=torch.float16, device='cuda')
        >>> v = torch.randn(10, 32, 64, dtype=torch.float16, device='cuda')
        >>> cu_seqlens = torch.tensor([0, 3, 7, 10], dtype=torch.int32, device='cuda')
        >>> output = flash_attn_varlen_wrapper(q, k, v, cu_seqlens, max_seqlen=4, softmax_scale=0.125)
    """
    # Check FlashAttention availability
    if not FLASH_ATTN_AVAILABLE:
        raise ImportError(
            "flash-attn not installed. Install with: "
            "pip install flash-attn>=2.6.0 --no-build-isolation"
        )

    # Check CUDA availability
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA not available. FlashAttention requires CUDA-enabled GPU."
        )

    # Validate cu_seqlens dtype (must be int32)
    if cu_seqlens.dtype != torch.int32:
        raise ValueError(
            f"cu_seqlens must have dtype int32, got {cu_seqlens.dtype}. "
            f"Convert with: cu_seqlens.to(torch.int32)"
        )

    # Validate q/k/v dtypes (must be FP16 or BF16)
    valid_dtypes = (torch.float16, torch.bfloat16)
    for name, tensor in [('q', q), ('k', k), ('v', v)]:
        if tensor.dtype not in valid_dtypes:
            raise ValueError(
                f"{name} must be FP16 or BF16, got {tensor.dtype}. "
                f"FlashAttention requires half-precision inputs. "
                f"Convert with: {name}.half() or {name}.bfloat16()"
            )

    # Validate cu_seqlens format
    if cu_seqlens[0] != 0:
        raise ValueError(
            f"cu_seqlens must start with 0, got {cu_seqlens[0]}"
        )

    # Call flash_attn_varlen_func
    # cu_seqlens_q and cu_seqlens_k are the same for self-attention
    output = flash_attn_varlen_func(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        dropout_p=dropout_p,
        causal=causal,
        softmax_scale=softmax_scale,
    )

    return output
