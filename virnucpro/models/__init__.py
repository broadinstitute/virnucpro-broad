"""Model wrappers with FlashAttention-2 optimization for ESM-2 and DNABERT-S."""

from virnucpro.models.esm2_flash import (
    ESM2WithFlashAttention,
    load_esm2_model
)

__all__ = [
    'ESM2WithFlashAttention',
    'load_esm2_model'
]
