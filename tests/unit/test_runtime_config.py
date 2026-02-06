"""Unit tests for RuntimeConfig dataclass.

Tests serialization/deserialization, validation, edge cases,
and Path<->str conversion for checkpoint_dir.
"""

import pytest
from pathlib import Path

from virnucpro.pipeline.runtime_config import RuntimeConfig


class TestRuntimeConfigDefaults:
    """Test RuntimeConfig default values."""

    def test_default_values(self):
        """Test all default values are set correctly."""
        config = RuntimeConfig()
        assert config.enable_checkpointing is True
        assert config.checkpoint_dir is None
        assert config.force_restart is False
        assert config.checkpoint_seq_threshold == 10000
        assert config.checkpoint_time_threshold == 300.0
        assert config.timeout_per_attempt == 3600.0
        assert config.max_retries_transient == 3
        assert config.max_retries_poison == 2
        assert config.spot_retry_poll_interval == 60.0
        assert config.enable_elastic_redistribution is True


class TestRuntimeConfigSerialization:
    """Test RuntimeConfig serialization and deserialization."""

    def test_to_dict_preserves_all_fields(self):
        """Test to_dict() returns dict with all fields."""
        config = RuntimeConfig()
        d = config.to_dict()

        assert isinstance(d, dict)
        assert set(d.keys()) == {
            'enable_checkpointing',
            'checkpoint_dir',
            'force_restart',
            'checkpoint_seq_threshold',
            'checkpoint_time_threshold',
            'timeout_per_attempt',
            'max_retries_transient',
            'max_retries_poison',
            'spot_retry_poll_interval',
            'enable_elastic_redistribution',
        }

    def test_roundtrip_with_defaults(self):
        """Test round-trip to_dict() -> from_dict() preserves defaults."""
        original = RuntimeConfig()
        d = original.to_dict()
        reconstructed = RuntimeConfig.from_dict(d)

        assert reconstructed.enable_checkpointing == original.enable_checkpointing
        assert reconstructed.checkpoint_dir == original.checkpoint_dir
        assert reconstructed.force_restart == original.force_restart
        assert reconstructed.checkpoint_seq_threshold == original.checkpoint_seq_threshold
        assert reconstructed.checkpoint_time_threshold == original.checkpoint_time_threshold
        assert reconstructed.timeout_per_attempt == original.timeout_per_attempt
        assert reconstructed.max_retries_transient == original.max_retries_transient
        assert reconstructed.max_retries_poison == original.max_retries_poison
        assert reconstructed.spot_retry_poll_interval == original.spot_retry_poll_interval
        assert reconstructed.enable_elastic_redistribution == original.enable_elastic_redistribution

    def test_roundtrip_with_custom_values(self):
        """Test round-trip with non-default values."""
        config = RuntimeConfig(
            enable_checkpointing=False,
            checkpoint_dir=Path("/custom/checkpoints"),
            force_restart=True,
            checkpoint_seq_threshold=5000,
            checkpoint_time_threshold=600.0,
            timeout_per_attempt=7200.0,
            max_retries_transient=5,
            max_retries_poison=3,
            spot_retry_poll_interval=30.0,
            enable_elastic_redistribution=False,
        )

        d = config.to_dict()
        reconstructed = RuntimeConfig.from_dict(d)

        assert reconstructed.enable_checkpointing is False
        assert reconstructed.checkpoint_dir == Path("/custom/checkpoints")
        assert reconstructed.force_restart is True
        assert reconstructed.checkpoint_seq_threshold == 5000
        assert reconstructed.checkpoint_time_threshold == 600.0
        assert reconstructed.timeout_per_attempt == 7200.0
        assert reconstructed.max_retries_transient == 5
        assert reconstructed.max_retries_poison == 3
        assert reconstructed.spot_retry_poll_interval == 30.0
        assert reconstructed.enable_elastic_redistribution is False


class TestRuntimeConfigPathConversion:
    """Test Path<->str conversion for checkpoint_dir."""

    def test_checkpoint_dir_none_to_dict(self):
        """Test checkpoint_dir=None converts to None in dict."""
        config = RuntimeConfig(checkpoint_dir=None)
        d = config.to_dict()
        assert d['checkpoint_dir'] is None

    def test_checkpoint_dir_path_to_str(self):
        """Test checkpoint_dir=Path converts to str in dict."""
        config = RuntimeConfig(checkpoint_dir=Path("/some/path"))
        d = config.to_dict()
        assert d['checkpoint_dir'] == "/some/path"
        assert isinstance(d['checkpoint_dir'], str)

    def test_checkpoint_dir_str_to_path_in_from_dict(self):
        """Test checkpoint_dir=str converts to Path in from_dict."""
        d = {'checkpoint_dir': '/another/path'}
        config = RuntimeConfig.from_dict(d)
        assert config.checkpoint_dir == Path('/another/path')

    def test_checkpoint_dir_none_in_from_dict(self):
        """Test checkpoint_dir=None handled correctly in from_dict."""
        d = {'checkpoint_dir': None}
        config = RuntimeConfig.from_dict(d)
        assert config.checkpoint_dir is None


class TestRuntimeConfigExtraFields:
    """Test handling of extra/unknown fields in from_dict."""

    def test_extra_fields_ignored(self):
        """Test extra fields in dict are ignored by from_dict."""
        d = {
            'enable_checkpointing': True,
            'checkpoint_dir': None,
            'force_restart': False,
            'checkpoint_seq_threshold': 10000,
            'checkpoint_time_threshold': 300.0,
            'timeout_per_attempt': 3600.0,
            'max_retries_transient': 3,
            'max_retries_poison': 2,
            'spot_retry_poll_interval': 60.0,
            'enable_elastic_redistribution': True,
            'unknown_field': 'should be ignored',
            'another_extra': 123,
        }
        config = RuntimeConfig.from_dict(d)
        assert config.enable_checkpointing is True
        assert not hasattr(config, 'unknown_field')

    def test_partial_dict_with_defaults(self):
        """Test from_dict with only some fields uses defaults for rest."""
        d = {'enable_checkpointing': False}
        config = RuntimeConfig.from_dict(d)

        assert config.enable_checkpointing is False
        assert config.checkpoint_dir is None
        assert config.force_restart is False
        assert config.checkpoint_seq_threshold == 10000


class TestRuntimeConfigValidation:
    """Test __post_init__ validation."""

    def test_max_retries_transient_negative_rejected(self):
        """Test negative max_retries_transient raises ValueError."""
        with pytest.raises(ValueError, match="max_retries_transient must be non-negative"):
            RuntimeConfig(max_retries_transient=-1)

    def test_max_retries_poison_negative_rejected(self):
        """Test negative max_retries_poison raises ValueError."""
        with pytest.raises(ValueError, match="max_retries_poison must be non-negative"):
            RuntimeConfig(max_retries_poison=-5)

    def test_checkpoint_time_threshold_zero_rejected(self):
        """Test zero checkpoint_time_threshold raises ValueError."""
        with pytest.raises(ValueError, match="checkpoint_time_threshold must be positive"):
            RuntimeConfig(checkpoint_time_threshold=0)

    def test_checkpoint_time_threshold_negative_rejected(self):
        """Test negative checkpoint_time_threshold raises ValueError."""
        with pytest.raises(ValueError, match="checkpoint_time_threshold must be positive"):
            RuntimeConfig(checkpoint_time_threshold=-100)

    def test_checkpoint_seq_threshold_zero_rejected(self):
        """Test zero checkpoint_seq_threshold raises ValueError."""
        with pytest.raises(ValueError, match="checkpoint_seq_threshold must be positive"):
            RuntimeConfig(checkpoint_seq_threshold=0)

    def test_checkpoint_seq_threshold_negative_rejected(self):
        """Test negative checkpoint_seq_threshold raises ValueError."""
        with pytest.raises(ValueError, match="checkpoint_seq_threshold must be positive"):
            RuntimeConfig(checkpoint_seq_threshold=-1)

    def test_spot_retry_poll_interval_zero_rejected(self):
        """Test zero spot_retry_poll_interval raises ValueError."""
        with pytest.raises(ValueError, match="spot_retry_poll_interval must be positive"):
            RuntimeConfig(spot_retry_poll_interval=0)

    def test_spot_retry_poll_interval_negative_rejected(self):
        """Test negative spot_retry_poll_interval raises ValueError."""
        with pytest.raises(ValueError, match="spot_retry_poll_interval must be positive"):
            RuntimeConfig(spot_retry_poll_interval=-10)

    def test_timeout_per_attempt_zero_rejected(self):
        """Test zero timeout_per_attempt raises ValueError."""
        with pytest.raises(ValueError, match="timeout_per_attempt must be positive or None"):
            RuntimeConfig(timeout_per_attempt=0)

    def test_timeout_per_attempt_negative_rejected(self):
        """Test negative timeout_per_attempt raises ValueError."""
        with pytest.raises(ValueError, match="timeout_per_attempt must be positive or None"):
            RuntimeConfig(timeout_per_attempt=-100)

    def test_timeout_per_attempt_none_allowed(self):
        """Test timeout_per_attempt=None is allowed."""
        config = RuntimeConfig(timeout_per_attempt=None)
        assert config.timeout_per_attempt is None


class TestRuntimeConfigEdgeCases:
    """Test edge cases and boundary values."""

    def test_zero_max_retries_allowed(self):
        """Test max_retries=0 is allowed."""
        config = RuntimeConfig(max_retries_transient=0, max_retries_poison=0)
        assert config.max_retries_transient == 0
        assert config.max_retries_poison == 0

    def test_large_timeout_values(self):
        """Test large timeout values are accepted."""
        config = RuntimeConfig(
            checkpoint_time_threshold=86400.0,
            timeout_per_attempt=86400.0,
            spot_retry_poll_interval=3600.0,
        )
        assert config.checkpoint_time_threshold == 86400.0
        assert config.timeout_per_attempt == 86400.0
        assert config.spot_retry_poll_interval == 3600.0

    def test_small_positive_values(self):
        """Test small positive values are accepted."""
        config = RuntimeConfig(
            checkpoint_time_threshold=0.001,
            checkpoint_seq_threshold=1,
            spot_retry_poll_interval=0.001,
        )
        assert config.checkpoint_time_threshold == 0.001
        assert config.checkpoint_seq_threshold == 1
        assert config.spot_retry_poll_interval == 0.001

    def test_force_restart_true(self):
        """Test force_restart=True is handled correctly."""
        config = RuntimeConfig(force_restart=True)
        d = config.to_dict()
        assert d['force_restart'] is True
        reconstructed = RuntimeConfig.from_dict(d)
        assert reconstructed.force_restart is True

    def test_enable_checkpointing_false(self):
        """Test enable_checkpointing=False is handled correctly."""
        config = RuntimeConfig(enable_checkpointing=False)
        d = config.to_dict()
        assert d['enable_checkpointing'] is False
        reconstructed = RuntimeConfig.from_dict(d)
        assert reconstructed.enable_checkpointing is False

    def test_enable_elastic_redistribution_false(self):
        """Test enable_elastic_redistribution=False is handled correctly."""
        config = RuntimeConfig(enable_elastic_redistribution=False)
        d = config.to_dict()
        assert d['enable_elastic_redistribution'] is False
        reconstructed = RuntimeConfig.from_dict(d)
        assert reconstructed.enable_elastic_redistribution is False

    def test_empty_dict_uses_all_defaults(self):
        """Test from_dict with empty dict uses all defaults."""
        config = RuntimeConfig.from_dict({})
        defaults = RuntimeConfig()
        assert config.enable_checkpointing == defaults.enable_checkpointing
        assert config.checkpoint_dir == defaults.checkpoint_dir
        assert config.force_restart == defaults.force_restart
        assert config.checkpoint_seq_threshold == defaults.checkpoint_seq_threshold
        assert config.checkpoint_time_threshold == defaults.checkpoint_time_threshold
        assert config.timeout_per_attempt == defaults.timeout_per_attempt
        assert config.max_retries_transient == defaults.max_retries_transient
        assert config.max_retries_poison == defaults.max_retries_poison
        assert config.spot_retry_poll_interval == defaults.spot_retry_poll_interval
        assert config.enable_elastic_redistribution == defaults.enable_elastic_redistribution
