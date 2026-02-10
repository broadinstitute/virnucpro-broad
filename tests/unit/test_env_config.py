"""Unit tests for environment configuration centralization."""

import os
import pytest
from virnucpro.core.env_config import EnvConfig, get_env_config


class TestEnvConfigDefaults:
    """Test default values when no environment variables are set."""

    def test_default_values_all_false(self):
        """All boolean flags should default to False when env vars not set."""
        # Clear any existing env vars
        for key in ['VIRNUCPRO_DISABLE_PACKING', 'VIRNUCPRO_DISABLE_FP16',
                    'VIRNUCPRO_V1_ATTENTION', 'VIRNUCPRO_VIRAL_CHECKPOINT_MODE']:
            os.environ.pop(key, None)
        get_env_config.cache_clear()

        config = get_env_config()

        assert config.disable_packing is False
        assert config.disable_fp16 is False
        assert config.v1_attention is False
        assert config.viral_checkpoint_mode is False


class TestEnvConfigBooleanParsing:
    """Test boolean environment variable parsing with various valid inputs."""

    def test_disable_packing_true_values(self):
        """Test all valid 'true' representations for VIRNUCPRO_DISABLE_PACKING."""
        for value in ['1', 'true', 'True', 'TRUE', 'yes', 'Yes', 'YES']:
            os.environ['VIRNUCPRO_DISABLE_PACKING'] = value
            get_env_config.cache_clear()
            config = get_env_config()
            assert config.disable_packing is True, f"Failed for value: {value}"

    def test_disable_packing_false_values(self):
        """Test all valid 'false' representations for VIRNUCPRO_DISABLE_PACKING."""
        for value in ['0', 'false', 'False', 'FALSE', 'no', 'No', 'NO', '']:
            os.environ['VIRNUCPRO_DISABLE_PACKING'] = value
            get_env_config.cache_clear()
            config = get_env_config()
            assert config.disable_packing is False, f"Failed for value: {value}"

    def test_disable_fp16_true_values(self):
        """Test all valid 'true' representations for VIRNUCPRO_DISABLE_FP16."""
        for value in ['1', 'true', 'yes']:
            os.environ['VIRNUCPRO_DISABLE_FP16'] = value
            get_env_config.cache_clear()
            config = get_env_config()
            assert config.disable_fp16 is True, f"Failed for value: {value}"

    def test_v1_attention_true_values(self):
        """Test all valid 'true' representations for VIRNUCPRO_V1_ATTENTION."""
        for value in ['1', 'true', 'yes']:
            os.environ['VIRNUCPRO_V1_ATTENTION'] = value
            get_env_config.cache_clear()
            config = get_env_config()
            assert config.v1_attention is True, f"Failed for value: {value}"

    def test_viral_checkpoint_mode_true_values(self):
        """Test all valid 'true' representations for VIRNUCPRO_VIRAL_CHECKPOINT_MODE."""
        for value in ['1', 'true', 'yes']:
            os.environ['VIRNUCPRO_VIRAL_CHECKPOINT_MODE'] = value
            get_env_config.cache_clear()
            config = get_env_config()
            assert config.viral_checkpoint_mode is True, f"Failed for value: {value}"

    def test_invalid_boolean_value_raises_error(self):
        """Invalid boolean values should raise ValueError with descriptive message."""
        os.environ['VIRNUCPRO_DISABLE_PACKING'] = 'banana'
        get_env_config.cache_clear()

        with pytest.raises(ValueError) as exc_info:
            get_env_config()

        assert 'VIRNUCPRO_DISABLE_PACKING' in str(exc_info.value)
        assert 'banana' in str(exc_info.value)

    def test_invalid_value_other_vars(self):
        """Test that invalid values are rejected for all boolean env vars."""
        test_cases = [
            ('VIRNUCPRO_DISABLE_FP16', 'invalid'),
            ('VIRNUCPRO_V1_ATTENTION', 'maybe'),
            ('VIRNUCPRO_VIRAL_CHECKPOINT_MODE', 'sometimes'),
        ]

        for env_var, invalid_value in test_cases:
            # Clean slate
            for key in ['VIRNUCPRO_DISABLE_PACKING', 'VIRNUCPRO_DISABLE_FP16',
                        'VIRNUCPRO_V1_ATTENTION', 'VIRNUCPRO_VIRAL_CHECKPOINT_MODE']:
                os.environ.pop(key, None)

            os.environ[env_var] = invalid_value
            get_env_config.cache_clear()

            with pytest.raises(ValueError) as exc_info:
                get_env_config()

            assert env_var in str(exc_info.value)


class TestEnvConfigCaching:
    """Test singleton caching behavior of get_env_config()."""

    def test_get_env_config_returns_same_instance(self):
        """get_env_config() should return cached singleton instance."""
        # Clear cache and set known state
        for key in ['VIRNUCPRO_DISABLE_PACKING', 'VIRNUCPRO_DISABLE_FP16',
                    'VIRNUCPRO_V1_ATTENTION', 'VIRNUCPRO_VIRAL_CHECKPOINT_MODE']:
            os.environ.pop(key, None)
        get_env_config.cache_clear()

        config1 = get_env_config()
        config2 = get_env_config()

        assert config1 is config2, "get_env_config() should return same instance"

    def test_cache_clear_allows_new_instance(self):
        """cache_clear() should allow creating new instance with updated values."""
        # Clear and set initial state
        for key in ['VIRNUCPRO_DISABLE_PACKING', 'VIRNUCPRO_DISABLE_FP16',
                    'VIRNUCPRO_V1_ATTENTION', 'VIRNUCPRO_VIRAL_CHECKPOINT_MODE']:
            os.environ.pop(key, None)
        get_env_config.cache_clear()

        config1 = get_env_config()
        assert config1.disable_packing is False

        # Change env var and clear cache
        os.environ['VIRNUCPRO_DISABLE_PACKING'] = '1'
        get_env_config.cache_clear()

        config2 = get_env_config()
        assert config2.disable_packing is True
        assert config1 is not config2, "New instance should be created after cache_clear()"

    def test_late_setting_without_cache_clear_uses_old_value(self):
        """Setting env var after first get_env_config() call doesn't affect cached instance."""
        # Clear and set initial state
        for key in ['VIRNUCPRO_DISABLE_PACKING', 'VIRNUCPRO_DISABLE_FP16',
                    'VIRNUCPRO_V1_ATTENTION', 'VIRNUCPRO_VIRAL_CHECKPOINT_MODE']:
            os.environ.pop(key, None)
        get_env_config.cache_clear()

        config1 = get_env_config()
        assert config1.disable_fp16 is False

        # Set env var AFTER first call
        os.environ['VIRNUCPRO_DISABLE_FP16'] = '1'

        # Without cache_clear, should still return old instance
        config2 = get_env_config()
        assert config2.disable_fp16 is False
        assert config1 is config2


class TestEnvConfigStructure:
    """Test that EnvConfig has exactly the expected fields."""

    def test_env_config_has_exactly_four_fields(self):
        """EnvConfig should have exactly 4 boolean fields (VIRNUCPRO_* app vars only)."""
        # Clear env vars
        for key in ['VIRNUCPRO_DISABLE_PACKING', 'VIRNUCPRO_DISABLE_FP16',
                    'VIRNUCPRO_V1_ATTENTION', 'VIRNUCPRO_VIRAL_CHECKPOINT_MODE']:
            os.environ.pop(key, None)
        get_env_config.cache_clear()

        config = get_env_config()

        # Check exactly these 4 fields exist
        assert hasattr(config, 'disable_packing')
        assert hasattr(config, 'disable_fp16')
        assert hasattr(config, 'v1_attention')
        assert hasattr(config, 'viral_checkpoint_mode')

        # Check no extra fields (like CUDA_VISIBLE_DEVICES, etc.)
        fields = [f for f in dir(config) if not f.startswith('_')]
        expected_fields = ['disable_fp16', 'disable_packing', 'v1_attention', 'viral_checkpoint_mode']
        actual_fields = [f for f in fields if f in expected_fields]

        assert len(actual_fields) == 4, f"Expected exactly 4 fields, got: {fields}"

    def test_env_config_does_not_include_external_tool_vars(self):
        """EnvConfig should not include CUDA_VISIBLE_DEVICES or other external tool vars."""
        config = get_env_config()

        # These should NOT be in EnvConfig (external tool vars, not app config)
        assert not hasattr(config, 'cuda_visible_devices')
        assert not hasattr(config, 'pytorch_cuda_alloc_conf')
        assert not hasattr(config, 'tokenizers_parallelism')
        assert not hasattr(config, 'disable_compile')  # Future Phase 15 var
