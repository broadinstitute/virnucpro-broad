"""Model wrappers with FlashAttention-2 optimization for ESM-2 and DNABERT-S."""

from virnucpro.models.esm2_flash import (
    ESM2WithFlashAttention,
    load_esm2_model
)
from virnucpro.models.packed_attention import (
    create_position_ids_packed,
    flash_attn_varlen_wrapper,
    FLASH_ATTN_AVAILABLE
)

__all__ = [
    'ESM2WithFlashAttention',
    'load_esm2_model',
    'create_position_ids_packed',
    'flash_attn_varlen_wrapper',
    'FLASH_ATTN_AVAILABLE'
]
