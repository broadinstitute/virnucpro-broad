"""Centralized environment variable configuration for VirNucPro.

This module provides a single source of truth for all VIRNUCPRO_* application
environment variables. It replaces scattered os.getenv() calls across the codebase
with a typed dataclass for better discoverability and validation.

Architecture Notes:
-------------------
- **Singleton pattern:** get_env_config() is cached via @lru_cache(maxsize=1)
- **Initialization timing:** Environment variables are read during __post_init__
- **Cache invalidation:** Tests and late-setting code must call cache_clear()

Environment Variable Lifecycle:
-------------------------------
Environment variables must be set BEFORE the first get_env_config() call, or the
cache must be cleared after late-setting:

    # Normal usage (env vars set before import)
    os.environ['VIRNUCPRO_V1_ATTENTION'] = 'true'
    from virnucpro.core.env_config import get_env_config
    config = get_env_config()  # Reads from os.environ

    # Late-setting pattern (e.g., CLI layer setting vars at runtime)
    os.environ['VIRNUCPRO_V1_ATTENTION'] = 'true'
    get_env_config.cache_clear()  # Invalidate cached instance
    config = get_env_config()     # New instance picks up the change

    # Test isolation pattern
    def test_something():
        os.environ['VIRNUCPRO_DISABLE_PACKING'] = '1'
        get_env_config.cache_clear()  # Clear cache for test
        config = get_env_config()
        assert config.disable_packing is True

Scope:
------
EnvConfig contains ONLY VIRNUCPRO_* application configuration variables.
It does NOT include external tool/runtime variables like:
- CUDA_VISIBLE_DEVICES (set by worker_init_fn, read directly in safety checks)
- PYTORCH_CUDA_ALLOC_CONF (set by workers for CUDA allocator control)
- TOKENIZERS_PARALLELISM (set by worker_init_fn for HuggingFace tokenizers)

These external vars are set/read directly via os.environ at their use sites
because they control third-party library behavior, not VirNucPro application logic.
"""

import os
import logging
from dataclasses import dataclass
from functools import lru_cache

logger = logging.getLogger('virnucpro.core.env_config')


def _parse_bool(value: str, var_name: str) -> bool:
    """Parse boolean environment variable with standardized values.

    Accepts case-insensitive variants of common boolean representations:
    - True: "1", "true", "yes"
    - False: "0", "false", "no", "" (empty string)

    Args:
        value: String value from os.environ.get()
        var_name: Environment variable name for error messages

    Returns:
        bool: Parsed boolean value

    Raises:
        ValueError: If value is not a recognized boolean representation
    """
    normalized = value.strip().lower()

    if normalized in ('1', 'true', 'yes'):
        return True
    elif normalized in ('0', 'false', 'no', ''):
        return False
    else:
        raise ValueError(
            f"Invalid boolean value for {var_name}: '{value}'. "
            f"Expected: 1/true/yes (True) or 0/false/no (False)"
        )


@dataclass
class EnvConfig:
    """Application environment variable configuration.

    This dataclass centralizes all VIRNUCPRO_* environment variables for the
    application. Boolean parsing is standardized across all variables.

    Attributes:
        disable_packing: Emergency rollback for sequence packing (VIRNUCPRO_DISABLE_PACKING)
        disable_fp16: FP16 diagnostic rollback to FP32 (VIRNUCPRO_DISABLE_FP16)
        v1_attention: v1.0 attention compatibility mode (VIRNUCPRO_V1_ATTENTION)
        viral_checkpoint_mode: Viral workload checkpoint tuning (VIRNUCPRO_VIRAL_CHECKPOINT_MODE)

    All boolean fields default to False when environment variables are not set.
    """

    disable_packing: bool = False
    disable_fp16: bool = False
    v1_attention: bool = False
    viral_checkpoint_mode: bool = False

    def __post_init__(self):
        """Load and validate environment variables.

        Reads from os.environ and parses boolean values. Called automatically
        after dataclass initialization.

        Raises:
            ValueError: If any environment variable has an invalid boolean value
        """
        # Load each env var with standardized boolean parsing
        packing_raw = os.environ.get('VIRNUCPRO_DISABLE_PACKING', '0')
        self.disable_packing = _parse_bool(packing_raw, 'VIRNUCPRO_DISABLE_PACKING')

        fp16_raw = os.environ.get('VIRNUCPRO_DISABLE_FP16', '0')
        self.disable_fp16 = _parse_bool(fp16_raw, 'VIRNUCPRO_DISABLE_FP16')

        v1_attn_raw = os.environ.get('VIRNUCPRO_V1_ATTENTION', '0')
        self.v1_attention = _parse_bool(v1_attn_raw, 'VIRNUCPRO_V1_ATTENTION')

        checkpoint_raw = os.environ.get('VIRNUCPRO_VIRAL_CHECKPOINT_MODE', '0')
        self.viral_checkpoint_mode = _parse_bool(checkpoint_raw, 'VIRNUCPRO_VIRAL_CHECKPOINT_MODE')


@lru_cache(maxsize=1)
def get_env_config() -> EnvConfig:
    """Get cached singleton EnvConfig instance.

    Returns the same EnvConfig instance across multiple calls for efficiency.
    Environment variables are read during the first call and cached.

    Cache Invalidation:
    -------------------
    For test isolation or late-setting scenarios, call cache_clear() before
    creating a new instance:

        os.environ['VIRNUCPRO_DISABLE_PACKING'] = '1'
        get_env_config.cache_clear()  # Clear cached instance
        config = get_env_config()     # New instance with updated env vars

    Returns:
        EnvConfig: Cached singleton configuration instance

    Raises:
        ValueError: If any environment variable has an invalid boolean value
    """
    return EnvConfig()
