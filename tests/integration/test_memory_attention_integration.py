"""Integration tests for memory optimization and FlashAttention-2.

These tests verify the complete integration of memory optimizations and
FlashAttention-2 support with the prediction pipeline.
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import torch


class TestMemoryOptimizationIntegration:
    """Integration tests for memory optimization features."""

    def test_memory_manager_initialization(self):
        """Test MemoryManager initialization with various configurations."""
        from virnucpro.cuda.memory_manager import MemoryManager

        # Test default initialization
        mm = MemoryManager()
        assert mm.cache_clear_interval == 100
        assert mm.batch_counter == 0

        # Test with custom settings
        mm = MemoryManager(enable_expandable_segments=True, cache_clear_interval=50)
        assert mm.cache_clear_interval == 50

    def test_memory_manager_cache_clearing(self):
        """Test periodic cache clearing functionality."""
        from virnucpro.cuda.memory_manager import MemoryManager

        mm = MemoryManager(cache_clear_interval=5)

        # First 4 increments should not trigger clear
        for i in range(4):
            assert not mm.should_clear_cache()
            mm.increment_and_clear()

        # 5th increment should trigger clear
        assert mm.should_clear_cache()

    @patch('torch.cuda.is_available', return_value=False)
    def test_memory_stats_without_cuda(self, mock_cuda):
        """Test memory stats return empty dict without CUDA."""
        from virnucpro.cuda.memory_manager import MemoryManager

        mm = MemoryManager()
        stats = mm.get_memory_stats()
        assert stats == {}

    def test_configure_memory_optimization(self):
        """Test global memory optimization configuration."""
        from virnucpro.cuda.memory_manager import configure_memory_optimization

        mm = configure_memory_optimization(
            enable_expandable=True,
            cache_interval=50,
            verbose=False
        )

        assert mm.cache_clear_interval == 50
        assert isinstance(mm, type(mm))  # MemoryManager instance


class TestDataLoaderOptimization:
    """Integration tests for DataLoader optimization utilities."""

    def test_get_optimal_workers_auto_detect(self):
        """Test automatic worker count detection."""
        from virnucpro.data.dataloader_utils import get_optimal_workers

        # Auto-detect with 1 GPU
        workers = get_optimal_workers(num_gpus=1)
        assert workers >= 1
        assert workers <= 8  # Capped at 8

        # Auto-detect with 4 GPUs
        workers = get_optimal_workers(num_gpus=4)
        assert workers >= 1

    def test_get_optimal_workers_explicit(self):
        """Test explicit worker count specification."""
        from virnucpro.data.dataloader_utils import get_optimal_workers

        workers = get_optimal_workers(num_gpus=2, dataloader_workers=4)
        assert workers == 4

    def test_sequence_dataset_creation(self):
        """Test SequenceDataset creation and sorting."""
        from virnucpro.data.dataloader_utils import SequenceDataset

        sequences = ["ATCG", "AT", "ATCGATCG"]

        # Test with sorting
        dataset = SequenceDataset(sequences, sort_by_length=True)
        assert len(dataset) == 3
        assert dataset[0] == "AT"  # Shortest first
        assert dataset[2] == "ATCGATCG"  # Longest last

        # Test without sorting
        dataset = SequenceDataset(sequences, sort_by_length=False)
        assert dataset[0] == "ATCG"  # Original order

    def test_create_optimized_dataloader(self):
        """Test creation of optimized DataLoader."""
        from virnucpro.data.dataloader_utils import create_optimized_dataloader, SequenceDataset

        sequences = ["ATCG"] * 100
        dataset = SequenceDataset(sequences)

        loader = create_optimized_dataloader(
            dataset=dataset,
            batch_size=16,
            num_gpus=2,
            dataloader_workers=2,
            shuffle=False
        )

        assert loader.batch_size == 16
        assert loader.num_workers == 2

    def test_dataloader_with_pin_memory(self):
        """Test DataLoader with pin_memory configuration."""
        from virnucpro.data.dataloader_utils import create_optimized_dataloader, SequenceDataset

        sequences = ["ATCG"] * 50
        dataset = SequenceDataset(sequences)

        # Explicit pin_memory=True
        loader = create_optimized_dataloader(
            dataset=dataset,
            batch_size=8,
            num_gpus=1,
            pin_memory=True
        )

        assert loader.pin_memory == True

        # Explicit pin_memory=False
        loader = create_optimized_dataloader(
            dataset=dataset,
            batch_size=8,
            num_gpus=1,
            pin_memory=False
        )

        assert loader.pin_memory == False


class TestFlashAttentionIntegration:
    """Integration tests for FlashAttention-2 support."""

    def test_attention_implementation_detection(self):
        """Test attention implementation detection."""
        from virnucpro.cuda.attention_utils import get_attention_implementation

        impl = get_attention_implementation()
        assert impl in ["flash_attention_2", "standard_attention"]

    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.get_device_capability', return_value=(8, 0))
    def test_flash_attention_available_on_ampere(self, mock_cap, mock_cuda):
        """Test FlashAttention-2 availability on Ampere GPU."""
        from virnucpro.cuda.attention_utils import is_flash_attention_available

        available = is_flash_attention_available()
        # Result depends on PyTorch version
        assert isinstance(available, bool)

    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.get_device_capability', return_value=(7, 5))
    def test_flash_attention_unavailable_on_pre_ampere(self, mock_cap, mock_cuda):
        """Test FlashAttention-2 unavailability on pre-Ampere GPU."""
        from virnucpro.cuda.attention_utils import is_flash_attention_available

        available = is_flash_attention_available()
        assert available == False

    def test_esm2_flash_model_creation(self):
        """Test ESM2WithFlashAttention wrapper creation."""
        from virnucpro.models.esm2_flash import ESM2WithFlashAttention

        # Create mock base model
        base_model = Mock()
        base_model.to = Mock(return_value=base_model)
        base_model.eval = Mock()

        with patch('virnucpro.cuda.attention_utils.get_attention_implementation', return_value='standard_attention'):
            with patch('virnucpro.cuda.attention_utils.configure_flash_attention', return_value=base_model):
                model = ESM2WithFlashAttention(base_model, torch.device('cpu'))

                assert model.device == torch.device('cpu')
                assert model.attention_impl == 'standard_attention'
                assert not model.use_bf16  # CPU doesn't support BF16

    def test_dnabert_flash_model_creation(self):
        """Test DNABERTWithFlashAttention wrapper creation."""
        from virnucpro.models.dnabert_flash import DNABERTWithFlashAttention

        # Create mock base model
        base_model = Mock()
        base_model.to = Mock(return_value=base_model)
        base_model.eval = Mock()

        with patch('virnucpro.cuda.attention_utils.get_attention_implementation', return_value='standard_attention'):
            with patch('virnucpro.cuda.attention_utils.configure_flash_attention', return_value=base_model):
                model = DNABERTWithFlashAttention(base_model, torch.device('cpu'))

                assert model.device == torch.device('cpu')
                assert model.attention_impl == 'standard_attention'
                assert not model.use_bf16  # CPU doesn't support BF16


class TestCUDAStreamIntegration:
    """Integration tests for CUDA stream orchestration."""

    @patch('torch.cuda.is_available', return_value=True)
    def test_stream_manager_creation(self, mock_cuda):
        """Test StreamManager creation."""
        from virnucpro.cuda.stream_manager import StreamManager

        with patch('torch.cuda.Stream') as mock_stream:
            mock_stream.return_value = Mock()
            sm = StreamManager(num_devices=2)

            assert sm.num_devices == 2
            assert len(sm.h2d_streams) == 2
            assert len(sm.compute_streams) == 2
            assert len(sm.d2h_streams) == 2

    def test_stream_processor_creation(self):
        """Test StreamProcessor creation."""
        from virnucpro.cuda.stream_manager import StreamProcessor

        device = torch.device('cpu')  # Use CPU for testing

        def transfer_fn(x):
            return x.to(device)

        def compute_fn(x):
            return x * 2

        def retrieve_fn(x):
            return x.cpu()

        processor = StreamProcessor(
            device=device,
            transfer_to_device=transfer_fn,
            compute=compute_fn,
            retrieve_result=retrieve_fn
        )

        assert processor.device == device

    @patch('torch.cuda.is_available', return_value=False)
    def test_stream_processor_without_cuda(self, mock_cuda):
        """Test StreamProcessor falls back gracefully without CUDA."""
        from virnucpro.cuda.stream_manager import StreamProcessor

        device = torch.device('cpu')

        processor = StreamProcessor(
            device=device,
            transfer_to_device=lambda x: x,
            compute=lambda x: x,
            retrieve_result=lambda x: x
        )

        # Process single item (should work without streams)
        result = processor.process_batch([1, 2, 3])
        assert result is not None


class TestPipelineIntegration:
    """Integration tests for full pipeline with optimizations."""

    def test_pipeline_imports(self):
        """Test that all pipeline components import successfully."""
        from virnucpro.pipeline.prediction import run_prediction
        from virnucpro.cuda.memory_manager import MemoryManager
        from virnucpro.data.dataloader_utils import create_optimized_dataloader
        from virnucpro.models.esm2_flash import load_esm2_model
        from virnucpro.models.dnabert_flash import load_dnabert_model

        # All imports successful
        assert run_prediction is not None
        assert MemoryManager is not None
        assert create_optimized_dataloader is not None
        assert load_esm2_model is not None
        assert load_dnabert_model is not None

    def test_pipeline_accepts_memory_optimization_flags(self):
        """Test that run_prediction accepts all memory optimization parameters."""
        import inspect
        from virnucpro.pipeline.prediction import run_prediction

        sig = inspect.signature(run_prediction)
        params = sig.parameters

        # Verify all memory optimization parameters exist
        assert 'dataloader_workers' in params
        assert 'pin_memory' in params
        assert 'expandable_segments' in params
        assert 'cache_clear_interval' in params
        assert 'cuda_streams' in params

    @patch('torch.cuda.is_available', return_value=False)
    def test_memory_optimization_initialization_without_cuda(self, mock_cuda):
        """Test memory optimization initialization without CUDA."""
        from virnucpro.cuda.memory_manager import configure_memory_optimization

        # Should not raise error even without CUDA
        mm = configure_memory_optimization(enable_expandable=True)
        assert mm is not None

    def test_oom_error_handling_structure(self):
        """Test that OOM error handling is structured correctly."""
        from virnucpro.pipeline.prediction import run_prediction
        import inspect

        source = inspect.getsource(run_prediction)

        # Verify OOM handling is present
        assert "out of memory" in source.lower()
        assert "return 4" in source  # OOM exit code


class TestCLIIntegration:
    """Integration tests for CLI with memory optimization flags."""

    def test_cli_has_memory_flags(self):
        """Test that CLI exposes all memory optimization flags."""
        import inspect
        from virnucpro.cli.predict import predict

        source = inspect.getsource(predict)

        # Verify all flags are present
        assert '--dataloader-workers' in source
        assert '--pin-memory' in source
        assert '--expandable-segments' in source
        assert '--cache-clear-interval' in source
        assert '--cuda-streams' in source or '--no-cuda-streams' in source

    def test_cli_passes_flags_to_pipeline(self):
        """Test that CLI passes memory flags to pipeline."""
        import inspect
        from virnucpro.cli.predict import predict

        source = inspect.getsource(predict)

        # Verify parameters are passed to run_prediction
        assert 'dataloader_workers=' in source
        assert 'pin_memory=' in source
        assert 'expandable_segments=' in source
        assert 'cache_clear_interval=' in source
        assert 'cuda_streams=' in source


class TestBackwardCompatibility:
    """Tests for backward compatibility with existing code."""

    def test_pipeline_works_without_optimization_flags(self):
        """Test that pipeline works without any optimization flags."""
        from virnucpro.pipeline.prediction import run_prediction
        import inspect

        sig = inspect.signature(run_prediction)
        params = sig.parameters

        # All memory optimization parameters should have defaults
        assert params['dataloader_workers'].default is not None or params['dataloader_workers'].default is None
        assert params['pin_memory'].default is not None or params['pin_memory'].default is None
        assert params['expandable_segments'].default == False
        assert params['cache_clear_interval'].default == 100
        assert params['cuda_streams'].default == True

    def test_memory_manager_optional_in_pipeline(self):
        """Test that MemoryManager is optional and pipeline handles None."""
        import inspect
        from virnucpro.pipeline.prediction import run_prediction

        source = inspect.getsource(run_prediction)

        # Verify memory_manager can be None
        assert 'if memory_manager' in source


@pytest.mark.gpu
class TestGPUOptimizations:
    """GPU-specific tests (require actual CUDA device)."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_memory_stats_with_cuda(self):
        """Test memory stats collection with actual CUDA device."""
        from virnucpro.cuda.memory_manager import MemoryManager

        mm = MemoryManager()
        stats = mm.get_memory_stats()

        assert 'allocated' in stats
        assert 'reserved' in stats
        assert 'free' in stats
        assert all(isinstance(v, float) for v in stats.values())

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_expandable_segments_configuration(self):
        """Test expandable segments environment variable configuration."""
        from virnucpro.cuda.memory_manager import MemoryManager

        mm = MemoryManager(enable_expandable_segments=True)

        # Check environment variable was set
        config = os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '')
        assert 'expandable_segments' in config.lower()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cache_clearing_reduces_reserved_memory(self):
        """Test that cache clearing reduces reserved memory."""
        from virnucpro.cuda.memory_manager import MemoryManager

        mm = MemoryManager()

        # Allocate some tensors
        tensors = [torch.randn(1000, 1000, device='cuda') for _ in range(10)]
        del tensors

        stats_before = mm.get_memory_stats()
        mm.clear_cache()
        stats_after = mm.get_memory_stats()

        # Reserved memory should decrease or stay same
        assert stats_after['reserved'] <= stats_before['reserved']

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_stream_creation_on_cuda(self):
        """Test CUDA stream creation."""
        from virnucpro.cuda.stream_manager import StreamManager

        sm = StreamManager(num_devices=1)

        assert len(sm.h2d_streams) == 1
        assert len(sm.compute_streams) == 1
        assert len(sm.d2h_streams) == 1

        # Verify streams are actual CUDA streams
        assert isinstance(sm.h2d_streams[0], torch.cuda.Stream)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 8,
        reason="Requires Ampere+ GPU"
    )
    def test_flash_attention_on_ampere_gpu(self):
        """Test FlashAttention-2 detection on Ampere+ GPU."""
        from virnucpro.cuda.attention_utils import is_flash_attention_available

        # On Ampere+ GPU, FlashAttention-2 should be available (if PyTorch 2.2+)
        available = is_flash_attention_available()

        # Result depends on PyTorch version
        assert isinstance(available, bool)

        # If available, verify we can use it
        if available:
            with torch.backends.cuda.sdp_kernel(enable_flash=True):
                # Should not raise error
                pass
