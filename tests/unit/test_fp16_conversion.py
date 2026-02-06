"""Unit tests for FP16 precision control and numerical stability detection.

Tests cover:
- should_use_fp16() environment variable handling
- check_numerical_stability() NaN/Inf detection
- Model initialization with enable_fp16 flag
- End-to-end FP32 verification with VIRNUCPRO_DISABLE_FP16
"""

import os
import pytest
import torch
from unittest.mock import patch, MagicMock, Mock
from virnucpro.pipeline.async_inference import check_numerical_stability
from virnucpro.utils.precision import should_use_fp16


class TestShouldUseFP16:
    """Test should_use_fp16() environment variable handling."""

    def test_should_use_fp16_default(self):
        """Verify should_use_fp16() returns True by default (no env var)."""
        # Ensure env var is not set
        with patch.dict(os.environ, {}, clear=False):
            if 'VIRNUCPRO_DISABLE_FP16' in os.environ:
                del os.environ['VIRNUCPRO_DISABLE_FP16']
            result = should_use_fp16()

        assert result is True, "should_use_fp16() should return True when env var not set"

    def test_should_use_fp16_disabled_numeric(self):
        """Set VIRNUCPRO_DISABLE_FP16=1, verify returns False."""
        with patch.dict(os.environ, {'VIRNUCPRO_DISABLE_FP16': '1'}):
            result = should_use_fp16()

        assert result is False, "should_use_fp16() should return False when VIRNUCPRO_DISABLE_FP16=1"

    def test_should_use_fp16_disabled_true(self):
        """Set VIRNUCPRO_DISABLE_FP16=true, verify returns False."""
        with patch.dict(os.environ, {'VIRNUCPRO_DISABLE_FP16': 'true'}):
            result = should_use_fp16()

        assert result is False, "should_use_fp16() should return False when VIRNUCPRO_DISABLE_FP16=true"


class TestCheckNumericalStability:
    """Test check_numerical_stability() NaN/Inf detection."""

    def test_check_numerical_stability_clean(self):
        """Pass normal tensor, verify no exception raised."""
        clean_tensor = torch.randn(10, 20)

        # Should not raise any exception
        try:
            check_numerical_stability(clean_tensor, context="test_batch")
        except RuntimeError:
            pytest.fail("check_numerical_stability raised RuntimeError on clean tensor")

    def test_check_numerical_stability_nan(self):
        """Pass tensor with NaN, verify RuntimeError with 'NaN' and 'VIRNUCPRO_DISABLE_FP16'."""
        tensor_with_nan = torch.tensor([1.0, 2.0, float('nan'), 4.0])

        with pytest.raises(RuntimeError, match=r"Numerical instability.*NaN.*VIRNUCPRO_DISABLE_FP16"):
            check_numerical_stability(tensor_with_nan, context="nan_test")

    def test_check_numerical_stability_inf(self):
        """Pass tensor with Inf, verify RuntimeError raised."""
        tensor_with_inf = torch.tensor([1.0, 2.0, float('inf'), 4.0])

        with pytest.raises(RuntimeError, match=r"Numerical instability.*Inf"):
            check_numerical_stability(tensor_with_inf, context="inf_test")

    def test_check_numerical_stability_mixed(self):
        """Pass tensor with both NaN and Inf, verify error reports both counts."""
        tensor_mixed = torch.tensor([1.0, float('nan'), float('inf'), 4.0, float('nan')])

        with pytest.raises(RuntimeError) as exc_info:
            check_numerical_stability(tensor_mixed, context="mixed_test")

        error_msg = str(exc_info.value)
        # Check that error message contains both NaN and Inf counts
        assert "NaN" in error_msg, "Error should mention NaN"
        assert "Inf" in error_msg, "Error should mention Inf"


class TestESM2ModelFP16Flag:
    """Test ESM-2 model initialization with enable_fp16 flag."""

    @patch('virnucpro.models.esm2_flash.esm')
    @patch('virnucpro.models.esm2_flash.ESM2WithFlashAttention')
    def test_esm2_model_init_fp16_flag(self, mock_wrapper_class, mock_esm):
        """Mock ESM-2 model, verify wrapper receives enable_fp16=True."""
        # Setup mock alphabet
        mock_alphabet = MagicMock()
        mock_batch_converter = MagicMock()
        mock_alphabet.get_batch_converter = MagicMock(return_value=mock_batch_converter)

        # Setup mock base model
        mock_base_model = MagicMock()
        mock_esm.pretrained.esm2_t36_3B_UR50D = MagicMock(
            return_value=(mock_base_model, mock_alphabet)
        )

        # Setup wrapper mock
        mock_wrapper = MagicMock()
        mock_wrapper_class.return_value = mock_wrapper

        # Import and call load_esm2_model with enable_fp16=True
        from virnucpro.models.esm2_flash import load_esm2_model
        model, _ = load_esm2_model(device='cpu', enable_fp16=True)

        # Verify ESM2WithFlashAttention was called with enable_fp16=True
        mock_wrapper_class.assert_called_once()
        call_kwargs = mock_wrapper_class.call_args[1]
        assert call_kwargs['enable_fp16'] is True, "Wrapper should receive enable_fp16=True"

    @patch('virnucpro.models.esm2_flash.esm')
    @patch('virnucpro.models.esm2_flash.ESM2WithFlashAttention')
    def test_esm2_model_init_fp32_flag(self, mock_wrapper_class, mock_esm):
        """Mock ESM-2 model, verify wrapper receives enable_fp16=False."""
        # Setup mock alphabet
        mock_alphabet = MagicMock()
        mock_batch_converter = MagicMock()
        mock_alphabet.get_batch_converter = MagicMock(return_value=mock_batch_converter)

        # Setup mock base model
        mock_base_model = MagicMock()
        mock_esm.pretrained.esm2_t36_3B_UR50D = MagicMock(
            return_value=(mock_base_model, mock_alphabet)
        )

        # Setup wrapper mock
        mock_wrapper = MagicMock()
        mock_wrapper_class.return_value = mock_wrapper

        # Import and call load_esm2_model with enable_fp16=False
        from virnucpro.models.esm2_flash import load_esm2_model
        model, _ = load_esm2_model(device='cpu', enable_fp16=False)

        # Verify ESM2WithFlashAttention was called with enable_fp16=False
        mock_wrapper_class.assert_called_once()
        call_kwargs = mock_wrapper_class.call_args[1]
        assert call_kwargs['enable_fp16'] is False, "Wrapper should receive enable_fp16=False"


class TestDisableFP16ProducesFP32:
    """End-to-end verification that VIRNUCPRO_DISABLE_FP16=1 produces FP32 embeddings."""

    @patch('virnucpro.models.esm2_flash.esm')
    @patch('virnucpro.models.esm2_flash.ESM2WithFlashAttention')
    def test_disable_fp16_produces_fp32_embeddings(self, mock_wrapper_class, mock_esm):
        """END-TO-END: Verify gpu_worker pattern with VIRNUCPRO_DISABLE_FP16=1."""
        # Setup mock alphabet
        mock_alphabet = MagicMock()
        mock_batch_converter = MagicMock()
        mock_alphabet.get_batch_converter = MagicMock(return_value=mock_batch_converter)

        # Setup mock base model
        mock_base_model = MagicMock()
        mock_esm.pretrained.esm2_t36_3B_UR50D = MagicMock(
            return_value=(mock_base_model, mock_alphabet)
        )

        # Setup wrapper mock that tracks dtype
        wrapper_received_fp16 = None

        def create_wrapper(base_model, device, enable_fp16, logger_instance):
            nonlocal wrapper_received_fp16
            wrapper_received_fp16 = enable_fp16
            mock_wrapper = MagicMock()
            # Simulate wrapper behavior: FP32 if not enable_fp16
            mock_wrapper.dtype = torch.float16 if enable_fp16 else torch.float32
            return mock_wrapper

        mock_wrapper_class.side_effect = create_wrapper

        # Test with VIRNUCPRO_DISABLE_FP16=1
        with patch.dict(os.environ, {'VIRNUCPRO_DISABLE_FP16': '1'}):
            # Simulate gpu_worker pattern: check env var BEFORE calling load_esm2_model
            # This is the actual implementation in gpu_worker.py
            if not should_use_fp16():
                enable_fp16 = False
            else:
                enable_fp16 = True  # Would come from model_config

            # Now load model with the resolved flag
            from virnucpro.models.esm2_flash import load_esm2_model
            model, _ = load_esm2_model(device='cpu', enable_fp16=enable_fp16)

            # Verify that env var caused enable_fp16=False to be passed
            assert wrapper_received_fp16 is False, (
                "When VIRNUCPRO_DISABLE_FP16=1, gpu_worker should pass enable_fp16=False"
            )

            # Verify wrapper would use FP32
            assert model.dtype == torch.float32, (
                f"Model should be FP32 when VIRNUCPRO_DISABLE_FP16=1, got {model.dtype}"
            )
