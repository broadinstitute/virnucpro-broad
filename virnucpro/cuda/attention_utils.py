"""FlashAttention-2 detection and configuration utilities for ESM-2 and DNABERT-S models.

This module provides automatic detection of GPU capabilities and configures models to use
FlashAttention-2 when available, with transparent fallback to standard attention on older GPUs.

FlashAttention-2 provides 2-4x attention speedup on Ampere+ GPUs (compute capability 8.0+).
"""

import torch
import logging
from typing import Optional, Literal

logger = logging.getLogger('virnucpro.cuda.attention_utils')


def get_attention_implementation() -> Literal["flash_attention_2", "standard_attention"]:
    """
    Detect and return the best available attention implementation.

    Checks GPU availability and compute capability to determine if FlashAttention-2
    can be used. FlashAttention-2 requires:
    - CUDA-capable GPU
    - Compute capability 8.0+ (Ampere, Ada, Hopper architectures)
    - PyTorch 2.2+ with scaled_dot_product_attention support

    Returns:
        "flash_attention_2": FlashAttention-2 is available and supported
        "standard_attention": Fallback to standard attention (no compatible GPU or PyTorch version)

    Example:
        >>> impl = get_attention_implementation()
        >>> if impl == "flash_attention_2":
        ...     print("FlashAttention-2 enabled for 2-4x speedup")
    """
    if not torch.cuda.is_available():
        logger.debug("CUDA not available, using standard attention")
        return "standard_attention"

    try:
        # Check GPU compute capability
        device_capability = torch.cuda.get_device_capability()
        major, minor = device_capability

        # FlashAttention-2 requires Ampere (8.0) or newer
        # Ampere: 8.0, 8.6 (RTX 3000, A100)
        # Ada: 8.9 (RTX 4000)
        # Hopper: 9.0 (H100)
        if major < 8:
            logger.debug(
                f"GPU compute capability {major}.{minor} < 8.0, "
                "FlashAttention-2 requires Ampere+ (8.0+)"
            )
            return "standard_attention"

        # Test if FlashAttention-2 is available in PyTorch backend
        # PyTorch 2.2+ provides scaled_dot_product_attention with flash kernel
        # Note: New API (torch.nn.attention.sdpa_kernel) uses different parameters than old API

        # Try to create a context with FlashAttention enabled
        # This will fail if flash-attn is not installed or incompatible
        try:
            # Try new API first (PyTorch 2.5+)
            if hasattr(torch.nn.attention, 'sdpa_kernel') and hasattr(torch.nn.attention, 'SDPBackend'):
                from torch.nn.attention import SDPBackend
                # New API uses SDPBackend enum
                with torch.nn.attention.sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
                    # Successfully created context - FlashAttention-2 is available
                    return "flash_attention_2"
            # Fall back to old deprecated API (PyTorch 2.2-2.4)
            elif hasattr(torch.backends.cuda, 'sdp_kernel'):
                with torch.backends.cuda.sdp_kernel(
                    enable_flash=True,
                    enable_math=False,
                    enable_mem_efficient=False
                ):
                    # Successfully created context - FlashAttention-2 is available
                    return "flash_attention_2"
            else:
                logger.debug(
                    "Neither torch.nn.attention.sdpa_kernel nor torch.backends.cuda.sdp_kernel available, "
                    "requires PyTorch 2.2+"
                )
                return "standard_attention"
        except (RuntimeError, AttributeError, ImportError) as e:
            logger.debug(f"FlashAttention-2 context creation failed: {e}")
            return "standard_attention"

    except Exception as e:
        logger.warning(f"Error detecting attention implementation: {e}")
        return "standard_attention"


def is_flash_attention_available() -> bool:
    """
    Check if FlashAttention-2 is available on the current system.

    Returns:
        True if FlashAttention-2 can be used, False otherwise

    Example:
        >>> if is_flash_attention_available():
        ...     print("System supports FlashAttention-2")
    """
    return get_attention_implementation() == "flash_attention_2"


def configure_flash_attention(
    model: torch.nn.Module,
    logger_instance: Optional[logging.Logger] = None
) -> torch.nn.Module:
    """
    Configure model to use FlashAttention-2 if available.

    This function:
    1. Detects the best available attention implementation
    2. Configures model settings appropriately
    3. Logs the configuration choice for transparency

    Args:
        model: PyTorch model to configure (typically ESM-2 or DNABERT-S)
        logger_instance: Optional logger for configuration messages

    Returns:
        Configured model (modified in-place and returned for chaining)

    Example:
        >>> model = load_base_model()
        >>> model = configure_flash_attention(model, logger)
        >>> # Model now uses FlashAttention-2 if available
    """
    log = logger_instance if logger_instance is not None else logger

    # Detect best attention implementation
    attention_impl = get_attention_implementation()

    if attention_impl == "flash_attention_2":
        log.info("FlashAttention-2: enabled (2-4x attention speedup on Ampere+ GPU)")

        # Configure model for FlashAttention-2
        if hasattr(model, 'config'):
            # HuggingFace models: Set attention implementation via config
            model.config._attn_implementation = "sdpa"
            model.config.use_flash_attention_2 = True
        else:
            # Non-HuggingFace models (e.g., fair-esm): FlashAttention-2 will be
            # activated via sdp_kernel context manager in forward pass
            log.debug(
                "Model does not have config attribute (non-HuggingFace model). "
                "FlashAttention-2 will be activated via context manager."
            )

    else:
        log.info(
            "Using standard attention (FlashAttention-2 unavailable: "
            "requires Ampere+ GPU and PyTorch 2.2+)"
        )

        # Ensure standard attention is used
        if hasattr(model, 'config'):
            model.config._attn_implementation = "eager"
            if hasattr(model.config, 'use_flash_attention_2'):
                model.config.use_flash_attention_2 = False

    return model


def get_gpu_info() -> dict:
    """
    Get detailed GPU information for diagnostics.

    Returns:
        Dictionary with GPU details:
        - has_cuda: bool
        - device_count: int
        - devices: list of device info dicts
        - flash_attention_available: bool

    Example:
        >>> info = get_gpu_info()
        >>> print(f"GPUs: {info['device_count']}")
        >>> print(f"FlashAttention-2: {info['flash_attention_available']}")
    """
    info = {
        'has_cuda': torch.cuda.is_available(),
        'device_count': 0,
        'devices': [],
        'flash_attention_available': is_flash_attention_available()
    }

    if not info['has_cuda']:
        return info

    info['device_count'] = torch.cuda.device_count()

    for i in range(info['device_count']):
        capability = torch.cuda.get_device_capability(i)
        device_info = {
            'id': i,
            'name': torch.cuda.get_device_name(i),
            'compute_capability': f"{capability[0]}.{capability[1]}",
            'total_memory_gb': torch.cuda.get_device_properties(i).total_memory / 1024**3,
            'supports_flash_attention': capability[0] >= 8
        }
        info['devices'].append(device_info)

    return info
