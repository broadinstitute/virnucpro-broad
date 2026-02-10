"""Unit tests for ESM-2 packed sequence forward pass.

These tests verify the forward_packed method signature, structure, and basic logic
without requiring GPU hardware. Integration tests with actual model execution are
in tests/integration/.
"""

import pytest
import torch
import os
from unittest.mock import MagicMock, patch, Mock
import inspect


def test_forward_packed_signature():
    """Verify forward_packed has correct signature."""
    from virnucpro.models.esm2_flash import ESM2WithFlashAttention

    sig = inspect.signature(ESM2WithFlashAttention.forward_packed)
    params = list(sig.parameters.keys())

    # Verify required parameters
    assert params == ['self', 'input_ids', 'cu_seqlens', 'max_seqlen', 'repr_layers', 'v1_compatible'], \
        f"Expected signature mismatch. Got: {params}"

    # Verify repr_layers has default value
    assert sig.parameters['repr_layers'].default is not inspect.Parameter.empty, \
        "repr_layers should have default value"

    # Verify v1_compatible has default value of False
    assert sig.parameters['v1_compatible'].default is False, \
        "v1_compatible should default to False"


def test_forward_packed_method_exists():
    """Verify forward_packed method exists on ESM2WithFlashAttention class."""
    from virnucpro.models.esm2_flash import ESM2WithFlashAttention

    assert hasattr(ESM2WithFlashAttention, 'forward_packed'), \
        "ESM2WithFlashAttention missing forward_packed method"

    assert callable(getattr(ESM2WithFlashAttention, 'forward_packed')), \
        "forward_packed is not callable"


def test_fallback_method_exists():
    """Verify _forward_packed_fallback method exists for FlashAttention unavailable case."""
    from virnucpro.models.esm2_flash import ESM2WithFlashAttention

    assert hasattr(ESM2WithFlashAttention, '_forward_packed_fallback'), \
        "ESM2WithFlashAttention missing _forward_packed_fallback method"

    assert callable(getattr(ESM2WithFlashAttention, '_forward_packed_fallback')), \
        "_forward_packed_fallback is not callable"


def test_layer_forward_packed_method_exists():
    """Verify _layer_forward_packed helper method exists."""
    from virnucpro.models.esm2_flash import ESM2WithFlashAttention

    assert hasattr(ESM2WithFlashAttention, '_layer_forward_packed'), \
        "ESM2WithFlashAttention missing _layer_forward_packed method"

    assert callable(getattr(ESM2WithFlashAttention, '_layer_forward_packed')), \
        "_layer_forward_packed is not callable"


def test_packed_attention_imports():
    """Verify packed_attention utilities are imported."""
    from virnucpro.models import esm2_flash

    # Check imports exist
    assert hasattr(esm2_flash, 'create_position_ids_packed'), \
        "create_position_ids_packed not imported"
    assert hasattr(esm2_flash, 'flash_attn_varlen_wrapper'), \
        "flash_attn_varlen_wrapper not imported"
    assert hasattr(esm2_flash, 'FLASH_ATTN_AVAILABLE'), \
        "FLASH_ATTN_AVAILABLE not imported"


def test_forward_packed_docstring():
    """Verify forward_packed has comprehensive docstring."""
    from virnucpro.models.esm2_flash import ESM2WithFlashAttention

    docstring = ESM2WithFlashAttention.forward_packed.__doc__

    assert docstring is not None, "forward_packed missing docstring"
    assert 'input_ids' in docstring, "Docstring missing input_ids parameter"
    assert 'cu_seqlens' in docstring, "Docstring missing cu_seqlens parameter"
    assert 'max_seqlen' in docstring, "Docstring missing max_seqlen parameter"
    assert 'FlashAttention' in docstring, "Docstring doesn't mention FlashAttention"
    assert 'packed' in docstring.lower(), "Docstring doesn't mention packed sequences"


@pytest.mark.parametrize("flash_available", [True, False])
def test_fallback_used_when_flash_unavailable(flash_available):
    """Verify fallback path is used when FlashAttention unavailable."""
    from virnucpro.models.esm2_flash import ESM2WithFlashAttention

    # Mock the base model
    mock_model = Mock()
    mock_model.to = Mock(return_value=mock_model)
    mock_model.eval = Mock(return_value=None)

    # Mock attention implementation
    with patch('virnucpro.models.esm2_flash.get_attention_implementation') as mock_attn:
        mock_attn.return_value = "standard_attention"

        with patch('virnucpro.models.esm2_flash.configure_flash_attention') as mock_config:
            mock_config.return_value = mock_model

            # Patch FLASH_ATTN_AVAILABLE
            with patch('virnucpro.models.esm2_flash.FLASH_ATTN_AVAILABLE', flash_available):
                # Create wrapper
                device = torch.device('cpu')
                wrapper = ESM2WithFlashAttention(mock_model, device)

                # Mock the fallback method to verify it's called
                wrapper._forward_packed_fallback = Mock(return_value={'representations': {}})

                if not flash_available:
                    # Create mock inputs
                    input_ids = torch.tensor([1, 2, 3], dtype=torch.long)
                    cu_seqlens = torch.tensor([0, 3], dtype=torch.int32)
                    max_seqlen = 3

                    # Call forward_packed - should use fallback
                    with patch('virnucpro.models.esm2_flash.create_position_ids_packed'):
                        result = wrapper.forward_packed(input_ids, cu_seqlens, max_seqlen)

                    # Verify fallback was called
                    wrapper._forward_packed_fallback.assert_called_once()


def test_forward_packed_returns_dict_structure():
    """Verify forward_packed returns expected dictionary structure."""
    # This is a structure test - actual GPU execution tested in integration
    from virnucpro.models.esm2_flash import ESM2WithFlashAttention

    # The return type should be dict with 'representations' key
    # This is verified through docstring and type hints

    sig = inspect.signature(ESM2WithFlashAttention.forward_packed)
    return_annotation = sig.return_annotation

    # Should return dict
    assert return_annotation == dict, \
        f"forward_packed should return dict, got {return_annotation}"


def test_layer_forward_packed_signature():
    """Verify _layer_forward_packed has correct signature."""
    from virnucpro.models.esm2_flash import ESM2WithFlashAttention

    sig = inspect.signature(ESM2WithFlashAttention._layer_forward_packed)
    params = list(sig.parameters.keys())

    # Verify required parameters
    expected = ['self', 'layer', 'hidden_states', 'cu_seqlens', 'max_seqlen', 'position_ids']
    assert params == expected, \
        f"Expected {expected}, got {params}"


def test_v1_compatible_param_calls_fallback():
    """Verify v1_compatible=True calls the fallback path."""
    from virnucpro.models.esm2_flash import ESM2WithFlashAttention

    mock_model = Mock()
    mock_model.to = Mock(return_value=mock_model)
    mock_model.eval = Mock(return_value=None)

    with patch('virnucpro.models.esm2_flash.get_attention_implementation') as mock_attn:
        mock_attn.return_value = "flash_attention_2"

        with patch('virnucpro.models.esm2_flash.configure_flash_attention') as mock_config:
            mock_config.return_value = mock_model

            with patch('virnucpro.models.esm2_flash.FLASH_ATTN_AVAILABLE', True):
                device = torch.device('cpu')
                wrapper = ESM2WithFlashAttention(mock_model, device)

                wrapper._forward_packed_fallback = Mock(return_value={'representations': {}})

                input_ids = torch.tensor([1, 2, 3], dtype=torch.long)
                cu_seqlens = torch.tensor([0, 3], dtype=torch.int32)
                max_seqlen = 3

                with patch('virnucpro.models.esm2_flash.create_position_ids_packed'):
                    result = wrapper.forward_packed(
                        input_ids, cu_seqlens, max_seqlen, v1_compatible=True
                    )

                wrapper._forward_packed_fallback.assert_called_once()


def test_v1_compatible_default_does_not_call_fallback():
    """Verify v1_compatible=False (default) does not call fallback when FA available."""
    from virnucpro.models.esm2_flash import ESM2WithFlashAttention

    mock_model = Mock()
    mock_model.to = Mock(return_value=mock_model)
    mock_model.eval = Mock(return_value=None)
    mock_model.embed_tokens = Mock(return_value=torch.zeros(3, 2560))
    mock_model.embed_scale = 1.0
    mock_model.token_dropout = False
    mock_model.emb_layer_norm_after = torch.nn.Identity()
    mock_model.layers = []
    mock_model.layer = []

    with patch('virnucpro.models.esm2_flash.get_attention_implementation') as mock_attn:
        mock_attn.return_value = "flash_attention_2"

        with patch('virnucpro.models.esm2_flash.configure_flash_attention') as mock_config:
            mock_config.return_value = mock_model

            with patch('virnucpro.models.esm2_flash.FLASH_ATTN_AVAILABLE', True):
                device = torch.device('cpu')
                wrapper = ESM2WithFlashAttention(mock_model, device)

                wrapper._forward_packed_fallback = Mock(return_value={'representations': {}})

                input_ids = torch.tensor([1, 2, 3], dtype=torch.long)
                cu_seqlens = torch.tensor([0, 3], dtype=torch.int32)
                max_seqlen = 3

                with patch('virnucpro.models.esm2_flash.create_position_ids_packed'):
                    result = wrapper.forward_packed(
                        input_ids, cu_seqlens, max_seqlen, v1_compatible=False
                    )

                wrapper._forward_packed_fallback.assert_not_called()


@patch.dict(os.environ, {'VIRNUCPRO_V1_ATTENTION': 'true'})
def test_env_var_enables_v1_attention():
    """Verify VIRNUCPRO_V1_ATTENTION=true enables v1.0 compatibility mode."""
    from virnucpro.core.env_config import get_env_config
    from virnucpro.models.esm2_flash import ESM2WithFlashAttention

    # Clear cache after env var patched by decorator
    get_env_config.cache_clear()

    mock_model = Mock()
    mock_model.to = Mock(return_value=mock_model)
    mock_model.eval = Mock(return_value=None)

    with patch('virnucpro.models.esm2_flash.get_attention_implementation') as mock_attn:
        mock_attn.return_value = "flash_attention_2"

        with patch('virnucpro.models.esm2_flash.configure_flash_attention') as mock_config:
            mock_config.return_value = mock_model

            with patch('virnucpro.models.esm2_flash.FLASH_ATTN_AVAILABLE', True):
                device = torch.device('cpu')
                wrapper = ESM2WithFlashAttention(mock_model, device)

                wrapper._forward_packed_fallback = Mock(return_value={'representations': {}})

                input_ids = torch.tensor([1, 2, 3], dtype=torch.long)
                cu_seqlens = torch.tensor([0, 3], dtype=torch.int32)
                max_seqlen = 3

                with patch('virnucpro.models.esm2_flash.create_position_ids_packed'):
                    result = wrapper.forward_packed(
                        input_ids, cu_seqlens, max_seqlen, v1_compatible=False
                    )

                wrapper._forward_packed_fallback.assert_called_once()


@patch.dict(os.environ, {'VIRNUCPRO_V1_ATTENTION': 'false'})
def test_env_var_false_does_not_affect_default():
    """Verify VIRNUCPRO_V1_ATTENTION=false does not force v1 mode."""
    from virnucpro.core.env_config import get_env_config
    from virnucpro.models.esm2_flash import ESM2WithFlashAttention

    # Clear cache after env var patched by decorator
    get_env_config.cache_clear()

    mock_model = Mock()
    mock_model.to = Mock(return_value=mock_model)
    mock_model.eval = Mock(return_value=None)
    mock_model.embed_tokens = Mock(return_value=torch.zeros(3, 2560))
    mock_model.embed_scale = 1.0
    mock_model.token_dropout = False
    mock_model.emb_layer_norm_after = torch.nn.Identity()
    mock_model.layers = []
    mock_model.layer = []

    with patch('virnucpro.models.esm2_flash.get_attention_implementation') as mock_attn:
        mock_attn.return_value = "flash_attention_2"

        with patch('virnucpro.models.esm2_flash.configure_flash_attention') as mock_config:
            mock_config.return_value = mock_model

            with patch('virnucpro.models.esm2_flash.FLASH_ATTN_AVAILABLE', True):
                device = torch.device('cpu')
                wrapper = ESM2WithFlashAttention(mock_model, device)

                wrapper._forward_packed_fallback = Mock(return_value={'representations': {}})

                input_ids = torch.tensor([1, 2, 3], dtype=torch.long)
                cu_seqlens = torch.tensor([0, 3], dtype=torch.int32)
                max_seqlen = 3

                with patch('virnucpro.models.esm2_flash.create_position_ids_packed'):
                    result = wrapper.forward_packed(
                        input_ids, cu_seqlens, max_seqlen, v1_compatible=False
                    )

                wrapper._forward_packed_fallback.assert_not_called()


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, '-v'])
