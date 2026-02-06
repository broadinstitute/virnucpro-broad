"""Precision utilities for FP16/FP32 model configuration.

This module provides shared utilities for controlling model precision via
the VIRNUCPRO_DISABLE_FP16 environment variable.
"""
import os
import logging

logger = logging.getLogger('virnucpro.utils.precision')


def should_use_fp16() -> bool:
    """Check if FP16 should be enabled (default: True).

    Set VIRNUCPRO_DISABLE_FP16=1 for:
    - **Debugging NaN/Inf issues:** Run same data in FP32 to isolate precision problems
    - **Autopsy mode:** Compare FP16 vs FP32 embeddings when investigating accuracy issues
    - **Legacy compatibility:** If FP16 causes model-specific problems in production
    - **Baseline comparison:** Establish FP32 baseline for performance validation

    Note: FP32 is 2x slower and uses 2x memory. Only disable for diagnostics.

    Returns:
        bool: True if FP16 should be used, False if FP32 (diagnostic mode)

    Environment:
        VIRNUCPRO_DISABLE_FP16: Set to "1", "true", or "yes" to disable FP16

    Example:
        # Normal production (FP16)
        $ python -m virnucpro predict input.fasta

        # Diagnostic mode (FP32 for troubleshooting)
        $ VIRNUCPRO_DISABLE_FP16=1 python -m virnucpro predict input.fasta
    """
    disable = os.getenv("VIRNUCPRO_DISABLE_FP16", "").strip().lower() in ("1", "true", "yes")
    if disable:
        logger.warning(
            "FP16 precision DISABLED via VIRNUCPRO_DISABLE_FP16. "
            "Using FP32 (slower, more memory). "
            "This is a diagnostic mode for troubleshooting."
        )
    return not disable
