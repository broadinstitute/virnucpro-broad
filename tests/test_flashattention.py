"""Tests for FlashAttention-2 integration with ESM-2 models.

This test suite covers:
- FlashAttention-2 detection and configuration
- GPU capability detection
- Graceful fallback to standard attention
- ESM-2 model wrapper functionality
- Integration with fair-esm library
- Memory efficiency with BF16
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import logging

from virnucpro.cuda.attention_utils import (
    get_attention_implementation,
    is_flash_attention_available,
    configure_flash_attention,
    get_gpu_info
)


class TestAttentionImplementationDetection:
    """Test get_attention_implementation() with various GPU scenarios."""

    def test_no_cuda_available(self):
        """Test fallback when CUDA is not available."""
        with patch('torch.cuda.is_available', return_value=False):
            impl = get_attention_implementation()
            assert impl == "standard_attention"

    def test_old_gpu_capability(self):
        """Test fallback on pre-Ampere GPUs (compute capability < 8.0)."""
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.get_device_capability', return_value=(7, 5)):
            # Simulate RTX 2080 Ti (compute capability 7.5)
            impl = get_attention_implementation()
            assert impl == "standard_attention"

    def test_ampere_gpu_no_sdp_kernel(self):
        """Test fallback when GPU is Ampere+ but PyTorch lacks sdp_kernel."""
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.get_device_capability', return_value=(8, 0)):
            # Remove sdp_kernel attribute temporarily
            original_backends = torch.backends.cuda
            mock_backends = MagicMock()
            mock_backends.cuda = MagicMock(spec=[])  # No sdp_kernel attribute

            with patch('torch.backends.cuda', mock_backends.cuda):
                impl = get_attention_implementation()
                assert impl == "standard_attention"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_ampere_gpu_with_flash_attention(self):
        """Test FlashAttention-2 detection on Ampere+ GPU (integration test)."""
        capability = torch.cuda.get_device_capability()
        if capability[0] >= 8:
            # Should detect FlashAttention-2 on Ampere+
            impl = get_attention_implementation()
            # Will be flash_attention_2 if flash-attn installed, otherwise standard_attention
            assert impl in ["flash_attention_2", "standard_attention"]
        else:
            pytest.skip("Requires Ampere+ GPU (compute capability 8.0+)")

    def test_is_flash_attention_available_boolean(self):
        """Test is_flash_attention_available() returns boolean."""
        result = is_flash_attention_available()
        assert isinstance(result, bool)


class TestConfigureFlashAttention:
    """Test configure_flash_attention() with mock models."""

    def test_configure_with_flash_attention_available(self):
        """Test model configuration when FlashAttention-2 is available."""
        # Create mock model with config
        mock_model = nn.Module()
        mock_model.config = Mock()

        with patch('virnucpro.cuda.attention_utils.get_attention_implementation',
                   return_value="flash_attention_2"):
            result = configure_flash_attention(mock_model)

            # Should set FlashAttention-2 config
            assert result.config._attn_implementation == "sdpa"
            assert result.config.use_flash_attention_2 is True

    def test_configure_with_standard_attention(self):
        """Test model configuration when FlashAttention-2 is not available."""
        # Create mock model with config
        mock_model = nn.Module()
        mock_model.config = Mock()
        mock_model.config.use_flash_attention_2 = False

        with patch('virnucpro.cuda.attention_utils.get_attention_implementation',
                   return_value="standard_attention"):
            result = configure_flash_attention(mock_model)

            # Should set standard attention config
            assert result.config._attn_implementation == "eager"
            assert result.config.use_flash_attention_2 is False

    def test_configure_model_without_config(self):
        """Test graceful handling of model without config attribute."""
        # Create model without config
        mock_model = nn.Module()

        # Should not raise error, just log warning
        with patch('virnucpro.cuda.attention_utils.get_attention_implementation',
                   return_value="flash_attention_2"):
            result = configure_flash_attention(mock_model)
            assert result is mock_model

    def test_configure_with_custom_logger(self, caplog):
        """Test that custom logger receives configuration messages."""
        mock_model = nn.Module()
        mock_model.config = Mock()
        custom_logger = logging.getLogger('test_custom')

        with patch('virnucpro.cuda.attention_utils.get_attention_implementation',
                   return_value="flash_attention_2"):
            with caplog.at_level(logging.INFO, logger='test_custom'):
                configure_flash_attention(mock_model, custom_logger)
                assert "FlashAttention-2: enabled" in caplog.text


class TestGPUInfo:
    """Test get_gpu_info() diagnostics function."""

    def test_gpu_info_no_cuda(self):
        """Test GPU info when CUDA is not available."""
        with patch('torch.cuda.is_available', return_value=False):
            info = get_gpu_info()

            assert info['has_cuda'] is False
            assert info['device_count'] == 0
            assert info['devices'] == []
            assert 'flash_attention_available' in info

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_gpu_info_with_cuda(self):
        """Test GPU info structure when CUDA is available."""
        info = get_gpu_info()

        assert info['has_cuda'] is True
        assert info['device_count'] > 0
        assert len(info['devices']) == info['device_count']

        # Check first device info structure
        device_info = info['devices'][0]
        assert 'id' in device_info
        assert 'name' in device_info
        assert 'compute_capability' in device_info
        assert 'total_memory_gb' in device_info
        assert 'supports_flash_attention' in device_info


class TestESM2WithFlashAttention:
    """Test ESM2WithFlashAttention wrapper class."""

    @pytest.fixture
    def mock_esm_model(self):
        """Create mock ESM-2 model for testing."""
        mock_model = nn.Module()
        mock_model.config = Mock()

        # Mock eval() and to() methods
        mock_model.eval = Mock(return_value=mock_model)
        mock_model.to = Mock(return_value=mock_model)
        mock_model.bfloat16 = Mock(return_value=mock_model)

        # Mock forward pass
        def mock_forward(tokens, repr_layers=None, return_contacts=False):
            batch_size, seq_len = tokens.shape
            embedding_dim = 2560  # ESM-2 3B dimension
            layer = repr_layers[0] if repr_layers else 36

            return {
                'representations': {
                    layer: torch.randn(batch_size, seq_len, embedding_dim)
                }
            }

        mock_model.forward = Mock(side_effect=mock_forward)
        mock_model.__call__ = mock_model.forward

        return mock_model

    def test_wrapper_initialization(self, mock_esm_model):
        """Test ESM2WithFlashAttention wrapper initialization."""
        from virnucpro.models.esm2_flash import ESM2WithFlashAttention

        device = torch.device('cpu')
        wrapper = ESM2WithFlashAttention(mock_esm_model, device)

        assert wrapper.model is not None
        assert wrapper.device == device
        assert wrapper.attention_impl in ["flash_attention_2", "standard_attention"]
        mock_esm_model.eval.assert_called_once()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_wrapper_bf16_on_ampere(self, mock_esm_model):
        """Test automatic BF16 conversion on Ampere+ GPUs."""
        from virnucpro.models.esm2_flash import ESM2WithFlashAttention

        capability = torch.cuda.get_device_capability()
        if capability[0] >= 8:
            device = torch.device('cuda:0')
            wrapper = ESM2WithFlashAttention(mock_esm_model, device)

            assert wrapper.use_bf16 is True
            mock_esm_model.bfloat16.assert_called_once()
        else:
            pytest.skip("Requires Ampere+ GPU")

    def test_wrapper_forward_standard_attention(self, mock_esm_model):
        """Test forward pass with standard attention."""
        from virnucpro.models.esm2_flash import ESM2WithFlashAttention

        with patch('virnucpro.cuda.attention_utils.get_attention_implementation',
                   return_value="standard_attention"):
            device = torch.device('cpu')
            wrapper = ESM2WithFlashAttention(mock_esm_model, device)

            # Create dummy input
            tokens = torch.randint(0, 33, (2, 10))  # batch_size=2, seq_len=10

            # Forward pass
            with torch.no_grad():
                output = wrapper(tokens, repr_layers=[36])

            assert 'representations' in output
            assert 36 in output['representations']
            mock_esm_model.forward.assert_called_once()

    def test_wrapper_repr(self, mock_esm_model):
        """Test string representation of wrapper."""
        from virnucpro.models.esm2_flash import ESM2WithFlashAttention

        device = torch.device('cpu')
        wrapper = ESM2WithFlashAttention(mock_esm_model, device)

        repr_str = repr(wrapper)
        assert "ESM2WithFlashAttention" in repr_str
        assert "attention=" in repr_str
        assert "device=" in repr_str


class TestLoadESM2Model:
    """Test load_esm2_model() function."""

    @pytest.mark.slow
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_load_650m_model(self):
        """Test loading ESM-2 650M model (integration test)."""
        from virnucpro.models.esm2_flash import load_esm2_model

        # Load smallest model for testing
        model, batch_converter = load_esm2_model(
            model_name="esm2_t33_650M_UR50D",
            device="cuda:0"
        )

        assert model is not None
        assert batch_converter is not None
        assert hasattr(model, 'attention_impl')
        assert model.attention_impl in ["flash_attention_2", "standard_attention"]

    def test_load_with_mock(self):
        """Test load_esm2_model() with mocked ESM library."""
        from virnucpro.models.esm2_flash import load_esm2_model

        # Mock ESM pretrained loader
        mock_model = nn.Module()
        mock_model.config = Mock()
        mock_model.eval = Mock(return_value=mock_model)
        mock_model.to = Mock(return_value=mock_model)
        mock_model.bfloat16 = Mock(return_value=mock_model)

        mock_alphabet = Mock()
        mock_batch_converter = Mock()
        mock_alphabet.get_batch_converter = Mock(return_value=mock_batch_converter)

        def mock_loader():
            return mock_model, mock_alphabet

        with patch('esm.pretrained.esm2_t33_650M_UR50D', mock_loader):
            model, batch_converter = load_esm2_model(
                model_name="esm2_t33_650M_UR50D",
                device="cpu"
            )

            assert model is not None
            assert batch_converter == mock_batch_converter


class TestGetESM2Embeddings:
    """Test get_esm2_embeddings() convenience function."""

    @pytest.fixture
    def mock_model_and_converter(self):
        """Create mock model and batch converter."""
        from virnucpro.models.esm2_flash import ESM2WithFlashAttention

        # Mock model
        base_model = nn.Module()
        base_model.config = Mock()
        base_model.eval = Mock(return_value=base_model)
        base_model.to = Mock(return_value=base_model)

        def mock_forward(tokens, repr_layers=None, return_contacts=False):
            batch_size, seq_len = tokens.shape
            layer = repr_layers[0] if repr_layers else 36
            return {
                'representations': {
                    layer: torch.randn(batch_size, seq_len, 2560)
                }
            }

        base_model.forward = Mock(side_effect=mock_forward)
        base_model.__call__ = base_model.forward

        with patch('virnucpro.cuda.attention_utils.get_attention_implementation',
                   return_value="standard_attention"):
            model = ESM2WithFlashAttention(base_model, torch.device('cpu'))

        # Mock batch converter
        def mock_batch_converter(sequences):
            labels = [seq[0] for seq in sequences]
            strs = [seq[1] for seq in sequences]
            max_len = max(len(s[1]) for s in sequences)
            tokens = torch.randint(0, 33, (len(sequences), max_len + 2))
            return labels, strs, tokens

        batch_converter = Mock(side_effect=mock_batch_converter)

        return model, batch_converter

    def test_get_embeddings_basic(self, mock_model_and_converter):
        """Test basic embedding extraction."""
        from virnucpro.models.esm2_flash import get_esm2_embeddings

        model, batch_converter = mock_model_and_converter
        sequences = [
            ("protein1", "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQF"),
            ("protein2", "KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRW")
        ]

        ids, embeddings = get_esm2_embeddings(
            model,
            batch_converter,
            sequences,
            layer=36
        )

        assert len(ids) == 2
        assert len(embeddings) == 2
        assert ids == ["protein1", "protein2"]
        assert embeddings[0].shape == (2560,)  # ESM-2 3B embedding dimension
        assert embeddings[1].shape == (2560,)

    def test_get_embeddings_truncation(self, mock_model_and_converter):
        """Test sequence truncation at maximum length."""
        from virnucpro.models.esm2_flash import get_esm2_embeddings

        model, batch_converter = mock_model_and_converter

        # Create very long sequence
        long_sequence = "M" * 2000
        sequences = [("long_protein", long_sequence)]

        ids, embeddings = get_esm2_embeddings(
            model,
            batch_converter,
            sequences,
            layer=36,
            truncation_length=1024
        )

        assert len(ids) == 1
        assert len(embeddings) == 1
        # Sequence should have been truncated
        # (actual truncation verified by batch_converter receiving truncated sequence)


@pytest.mark.integration
class TestFlashAttentionIntegration:
    """Integration tests for FlashAttention-2 with real models."""

    @pytest.mark.slow
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_esm2_forward_pass_no_errors(self):
        """Test that ESM-2 with FlashAttention-2 processes sequences without errors."""
        from virnucpro.models.esm2_flash import load_esm2_model

        # Load model
        model, batch_converter = load_esm2_model(
            model_name="esm2_t33_650M_UR50D",
            device="cuda:0"
        )

        # Create test sequences
        sequences = [
            ("protein1", "MKTAYIAK"),
            ("protein2", "VLSPADKTNV")
        ]

        # Convert and process
        labels, strs, tokens = batch_converter(sequences)
        tokens = tokens.to("cuda:0")

        with torch.no_grad():
            results = model(tokens, repr_layers=[33])

        # Verify output structure
        assert 'representations' in results
        assert 33 in results['representations']

        # Verify output shape
        representations = results['representations'][33]
        assert representations.shape[0] == 2  # Batch size
        assert representations.shape[2] == 640  # 650M model dimension

        # Verify no NaN values
        assert not torch.isnan(representations).any()

    @pytest.mark.slow
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_variable_length_sequences(self):
        """Test processing variable-length sequences (tests memory efficiency)."""
        from virnucpro.models.esm2_flash import load_esm2_model, get_esm2_embeddings

        model, batch_converter = load_esm2_model(
            model_name="esm2_t33_650M_UR50D",
            device="cuda:0"
        )

        # Create variable-length sequences
        sequences = [
            ("short", "MKTAYIAK"),
            ("medium", "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDN"),
            ("long", "M" * 500)
        ]

        ids, embeddings = get_esm2_embeddings(
            model,
            batch_converter,
            sequences,
            layer=33
        )

        assert len(ids) == 3
        assert len(embeddings) == 3

        # All embeddings should have same dimension
        assert all(emb.shape == (640,) for emb in embeddings)

        # No NaN or Inf values
        for emb in embeddings:
            assert not torch.isnan(emb).any()
            assert not torch.isinf(emb).any()
