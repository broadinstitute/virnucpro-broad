"""Tests for DataLoader optimization and memory management

Tests cover:
- DataLoader utilities with CPU-aware worker configuration
- MemoryManager with expandable segments and cache clearing
- Sequence sorting for memory efficiency
- Integration scenarios with mocked CUDA operations
- Error handling and edge cases
"""

import os
import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# Mock torch before importing modules that depend on it
torch_mock = MagicMock()
torch_mock.cuda.is_available.return_value = True
torch_mock.cuda.device_count.return_value = 4
torch_mock.cuda.get_device_capability.return_value = (8, 0)  # Ampere
torch_mock.cuda.get_device_properties.return_value.total_memory = 24 * 1024**3  # 24GB
torch_mock.cuda.memory_allocated.return_value = 8 * 1024**3  # 8GB
torch_mock.cuda.memory_reserved.return_value = 12 * 1024**3  # 12GB


@pytest.fixture(autouse=True)
def mock_torch():
    """Mock torch module for all tests"""
    with patch.dict('sys.modules', {'torch': torch_mock}):
        # Reset torch.utils.data mock for each test
        torch_mock.utils = MagicMock()
        torch_mock.utils.data.DataLoader = MagicMock(return_value=Mock())
        torch_mock.utils.data.Dataset = object
        yield torch_mock


@pytest.fixture
def mock_multiprocessing():
    """Mock multiprocessing.cpu_count()"""
    with patch('multiprocessing.cpu_count', return_value=32):
        yield


# DataLoader utilities tests
class TestDataLoaderUtils:
    """Tests for DataLoader configuration utilities"""

    def test_get_optimal_workers_auto_detect(self, mock_multiprocessing):
        """Test automatic worker count detection"""
        from virnucpro.data.dataloader_utils import get_optimal_workers

        # With 32 CPUs and 4 GPUs: min(32 // 4, 8) = 8
        workers = get_optimal_workers(num_gpus=4)
        assert workers == 8

        # With 32 CPUs and 1 GPU: min(32 // 1, 8) = 8 (capped)
        workers = get_optimal_workers(num_gpus=1)
        assert workers == 8

        # With 32 CPUs and 8 GPUs: min(32 // 8, 8) = 4
        workers = get_optimal_workers(num_gpus=8)
        assert workers == 4

    def test_get_optimal_workers_explicit(self, mock_multiprocessing):
        """Test explicit worker count override"""
        from virnucpro.data.dataloader_utils import get_optimal_workers

        workers = get_optimal_workers(num_gpus=4, dataloader_workers=16)
        assert workers == 16

    def test_get_optimal_workers_zero_gpus(self, mock_multiprocessing):
        """Test worker count with 0 GPUs (CPU-only)"""
        from virnucpro.data.dataloader_utils import get_optimal_workers

        # With 0 GPUs, use max(num_gpus, 1) = 1 to avoid division by zero
        workers = get_optimal_workers(num_gpus=0)
        assert workers == 8  # min(32 // 1, 8) = 8

    def test_get_optimal_workers_capping(self):
        """Test worker count capping at 8"""
        from virnucpro.data.dataloader_utils import get_optimal_workers

        with patch('multiprocessing.cpu_count', return_value=128):
            # With 128 CPUs and 1 GPU: min(128 // 1, 8) = 8
            workers = get_optimal_workers(num_gpus=1)
            assert workers == 8

    def test_create_optimized_dataloader_config(self, mock_multiprocessing):
        """Test DataLoader creation with optimized settings"""
        from virnucpro.data.dataloader_utils import create_optimized_dataloader

        # Create mock dataset
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=100)

        # Create DataLoader
        dataloader = create_optimized_dataloader(
            dataset=mock_dataset,
            batch_size=32,
            num_gpus=4,
            shuffle=True
        )

        # Verify DataLoader was called
        assert torch_mock.utils.data.DataLoader.called

        # Get call arguments
        call_kwargs = torch_mock.utils.data.DataLoader.call_args[1]

        # Verify configuration
        assert call_kwargs['batch_size'] == 32
        assert call_kwargs['shuffle'] is True
        assert call_kwargs['num_workers'] == 8  # min(32 // 4, 8) = 8
        assert call_kwargs['pin_memory'] is True  # CUDA available
        assert call_kwargs['prefetch_factor'] == 2
        assert call_kwargs['persistent_workers'] is True
        assert call_kwargs['multiprocessing_context'] == 'spawn'

    def test_create_optimized_dataloader_no_workers(self):
        """Test DataLoader with num_workers=0"""
        from virnucpro.data.dataloader_utils import create_optimized_dataloader

        mock_dataset = Mock()

        with patch('multiprocessing.cpu_count', return_value=0):
            dataloader = create_optimized_dataloader(
                dataset=mock_dataset,
                batch_size=32,
                num_gpus=1
            )

            call_kwargs = torch_mock.utils.data.DataLoader.call_args[1]
            assert call_kwargs['num_workers'] == 0
            # Should not set prefetch_factor and persistent_workers for num_workers=0
            assert 'prefetch_factor' not in call_kwargs
            assert 'persistent_workers' not in call_kwargs

    def test_create_optimized_dataloader_pin_memory_control(self, mock_multiprocessing):
        """Test pin_memory flag control"""
        from virnucpro.data.dataloader_utils import create_optimized_dataloader

        mock_dataset = Mock()

        # Test explicit pin_memory=False
        dataloader = create_optimized_dataloader(
            dataset=mock_dataset,
            batch_size=32,
            pin_memory=False
        )

        call_kwargs = torch_mock.utils.data.DataLoader.call_args[1]
        assert call_kwargs['pin_memory'] is False

    def test_create_sequence_dataloader_with_sorting(self):
        """Test sequence DataLoader with sorting"""
        from virnucpro.data.dataloader_utils import create_sequence_dataloader

        sequences = ["AAA", "AAAAA", "A", "AAAA"]

        dataloader = create_sequence_dataloader(
            sequences=sequences,
            batch_size=2,
            sort_by_length=True,
            num_gpus=1
        )

        # DataLoader should be created
        assert torch_mock.utils.data.DataLoader.called

    def test_create_sequence_dataloader_no_sorting(self):
        """Test sequence DataLoader without sorting"""
        from virnucpro.data.dataloader_utils import create_sequence_dataloader

        sequences = ["AAA", "AAAAA", "A", "AAAA"]

        dataloader = create_sequence_dataloader(
            sequences=sequences,
            batch_size=2,
            sort_by_length=False
        )

        assert torch_mock.utils.data.DataLoader.called

    def test_sequence_dataset_sorting(self):
        """Test SequenceDataset sorts sequences correctly"""
        from virnucpro.data.dataloader_utils import SequenceDataset

        sequences = ["AAA", "AAAAA", "A", "AAAA"]

        dataset = SequenceDataset(sequences, sort_by_length=True)

        # Should be sorted by length
        assert dataset.sequences == ["A", "AAA", "AAAA", "AAAAA"]

    def test_sequence_dataset_no_sorting(self):
        """Test SequenceDataset without sorting"""
        from virnucpro.data.dataloader_utils import SequenceDataset

        sequences = ["AAA", "AAAAA", "A", "AAAA"]

        dataset = SequenceDataset(sequences, sort_by_length=False)

        # Original order preserved
        assert dataset.sequences == ["AAA", "AAAAA", "A", "AAAA"]

    def test_sequence_dataset_len_and_getitem(self):
        """Test SequenceDataset __len__ and __getitem__"""
        from virnucpro.data.dataloader_utils import SequenceDataset

        sequences = ["A", "AA", "AAA"]
        dataset = SequenceDataset(sequences)

        assert len(dataset) == 3
        assert dataset[0] == "A"
        assert dataset[1] == "AA"
        assert dataset[2] == "AAA"

    def test_estimate_memory_usage(self):
        """Test memory usage estimation"""
        from virnucpro.data.dataloader_utils import estimate_memory_usage

        # Test with default parameters
        estimates = estimate_memory_usage(
            batch_size=32,
            max_seq_length=1024,
            model_size_gb=3.0
        )

        assert 'model' in estimates
        assert 'activations' in estimates
        assert 'total' in estimates

        assert estimates['model'] == 3.0
        assert estimates['activations'] > 0
        assert estimates['total'] == estimates['model'] + estimates['activations']

    def test_estimate_memory_usage_different_dtypes(self):
        """Test memory estimation with different data types"""
        from virnucpro.data.dataloader_utils import estimate_memory_usage

        # FP16 (2 bytes)
        fp16_estimate = estimate_memory_usage(
            batch_size=32,
            max_seq_length=1024,
            dtype_bytes=2
        )

        # FP32 (4 bytes)
        fp32_estimate = estimate_memory_usage(
            batch_size=32,
            max_seq_length=1024,
            dtype_bytes=4
        )

        # FP32 should use 2x more activation memory than FP16
        assert fp32_estimate['activations'] == fp16_estimate['activations'] * 2


# MemoryManager tests
class TestMemoryManager:
    """Tests for memory fragmentation prevention"""

    def test_memory_manager_init(self):
        """Test MemoryManager initialization"""
        from virnucpro.cuda.memory_manager import MemoryManager

        mm = MemoryManager(
            enable_expandable_segments=False,
            cache_clear_interval=100,
            verbose=True
        )

        assert mm.cache_clear_interval == 100
        assert mm.verbose is True
        assert mm.batch_counter == 0

    def test_configure_expandable_segments(self):
        """Test expandable segments configuration"""
        from virnucpro.cuda.memory_manager import MemoryManager

        # Clear existing env var
        if 'PYTORCH_CUDA_ALLOC_CONF' in os.environ:
            del os.environ['PYTORCH_CUDA_ALLOC_CONF']

        mm = MemoryManager(enable_expandable_segments=True)

        assert 'expandable_segments:True' in os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '')

    def test_configure_expandable_segments_preserves_existing(self):
        """Test expandable segments preserves existing config"""
        from virnucpro.cuda.memory_manager import MemoryManager

        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

        mm = MemoryManager(enable_expandable_segments=True)

        config = os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '')
        assert 'max_split_size_mb:128' in config
        assert 'expandable_segments:True' in config

    def test_get_memory_stats(self):
        """Test memory statistics retrieval"""
        from virnucpro.cuda.memory_manager import MemoryManager

        mm = MemoryManager()
        stats = mm.get_memory_stats(device=0)

        assert 'allocated' in stats
        assert 'reserved' in stats
        assert 'free' in stats

        # Values should be in GB
        assert stats['allocated'] > 0
        assert stats['reserved'] > 0
        assert stats['free'] > 0

    def test_get_memory_stats_no_cuda(self):
        """Test memory stats when CUDA unavailable"""
        from virnucpro.cuda.memory_manager import MemoryManager

        torch_mock.cuda.is_available.return_value = False

        mm = MemoryManager()
        stats = mm.get_memory_stats()

        assert stats == {'allocated': 0.0, 'reserved': 0.0, 'free': 0.0}

        # Reset mock
        torch_mock.cuda.is_available.return_value = True

    def test_clear_cache(self):
        """Test cache clearing"""
        from virnucpro.cuda.memory_manager import MemoryManager

        mm = MemoryManager()
        mm.clear_cache(device=0)

        # Should call empty_cache
        assert torch_mock.cuda.empty_cache.called

    def test_should_clear_cache(self):
        """Test cache clearing interval logic"""
        from virnucpro.cuda.memory_manager import MemoryManager

        mm = MemoryManager(cache_clear_interval=10)

        assert mm.should_clear_cache(0) is True  # 0 % 10 == 0
        assert mm.should_clear_cache(5) is False
        assert mm.should_clear_cache(10) is True
        assert mm.should_clear_cache(20) is True

    def test_should_clear_cache_disabled(self):
        """Test cache clearing when disabled (interval=0)"""
        from virnucpro.cuda.memory_manager import MemoryManager

        mm = MemoryManager(cache_clear_interval=0)

        assert mm.should_clear_cache(0) is False
        assert mm.should_clear_cache(100) is False

    def test_increment_and_clear(self):
        """Test batch counter increment and clearing"""
        from virnucpro.cuda.memory_manager import MemoryManager

        torch_mock.cuda.empty_cache.reset_mock()

        mm = MemoryManager(cache_clear_interval=10)

        # First 9 batches should not clear
        for i in range(9):
            cleared = mm.increment_and_clear()
            assert cleared is False

        # 10th batch should clear
        cleared = mm.increment_and_clear()
        assert cleared is True
        assert mm.batch_counter == 10
        assert torch_mock.cuda.empty_cache.called

    def test_sort_sequences_by_length(self):
        """Test sequence sorting"""
        from virnucpro.cuda.memory_manager import MemoryManager

        mm = MemoryManager()

        sequences = ["AAA", "AAAAA", "A", "AAAA"]
        sorted_seqs = mm.sort_sequences_by_length(sequences)

        assert sorted_seqs == ["A", "AAA", "AAAA", "AAAAA"]

    def test_sort_sequences_empty(self):
        """Test sorting empty sequence list"""
        from virnucpro.cuda.memory_manager import MemoryManager

        mm = MemoryManager()
        sorted_seqs = mm.sort_sequences_by_length([])

        assert sorted_seqs == []

    def test_check_memory_available(self):
        """Test memory availability check"""
        from virnucpro.cuda.memory_manager import MemoryManager

        mm = MemoryManager()

        # Mock shows 12GB free
        assert mm.check_memory_available(required_gb=10.0, device=0) is True
        assert mm.check_memory_available(required_gb=15.0, device=0) is False

    def test_get_safe_batch_size(self):
        """Test safe batch size calculation"""
        from virnucpro.cuda.memory_manager import MemoryManager

        mm = MemoryManager()

        batch_size = mm.get_safe_batch_size(
            model_size_gb=3.0,
            max_seq_length=1024,
            device=0,
            safety_factor=0.8
        )

        # Should return at least 1
        assert batch_size >= 1
        assert isinstance(batch_size, int)

    def test_get_safe_batch_size_insufficient_memory(self):
        """Test safe batch size with insufficient memory"""
        from virnucpro.cuda.memory_manager import MemoryManager

        mm = MemoryManager()

        # Model larger than available memory
        batch_size = mm.get_safe_batch_size(
            model_size_gb=20.0,
            max_seq_length=1024,
            available_memory_gb=10.0
        )

        # Should return minimum 1
        assert batch_size == 1

    def test_get_fragmentation_ratio(self):
        """Test fragmentation ratio calculation"""
        from virnucpro.cuda.memory_manager import MemoryManager

        mm = MemoryManager()

        # Mock shows: allocated=8GB, reserved=12GB
        # Fragmentation = (12 - 8) / 12 = 0.333...
        fragmentation = mm.get_fragmentation_ratio(device=0)

        assert 0.33 <= fragmentation <= 0.34

    def test_suggest_batch_size_adjustment_high_fragmentation(self):
        """Test batch size suggestion with high fragmentation"""
        from virnucpro.cuda.memory_manager import MemoryManager

        mm = MemoryManager()

        suggestion = mm.suggest_batch_size_adjustment(
            current_batch_size=100,
            fragmentation_ratio=0.4  # 40% fragmentation
        )

        assert suggestion['suggested_batch_size'] == 80  # 20% reduction
        assert 'High fragmentation' in suggestion['reason']

    def test_suggest_batch_size_adjustment_low_fragmentation(self):
        """Test batch size suggestion with low fragmentation"""
        from virnucpro.cuda.memory_manager import MemoryManager

        mm = MemoryManager()

        suggestion = mm.suggest_batch_size_adjustment(
            current_batch_size=100,
            fragmentation_ratio=0.05  # 5% fragmentation
        )

        # Should suggest increase due to low fragmentation and free memory
        assert suggestion['suggested_batch_size'] == 120

    def test_suggest_batch_size_adjustment_optimal(self):
        """Test batch size suggestion when optimal"""
        from virnucpro.cuda.memory_manager import MemoryManager

        mm = MemoryManager()

        suggestion = mm.suggest_batch_size_adjustment(
            current_batch_size=100,
            fragmentation_ratio=0.15  # 15% fragmentation (moderate)
        )

        assert suggestion['suggested_batch_size'] == 100
        assert 'optimal' in suggestion['reason']

    def test_memory_tracking_context_manager(self):
        """Test memory tracking context manager"""
        from virnucpro.cuda.memory_manager import MemoryManager

        mm = MemoryManager()

        # Should not raise exception
        with mm.memory_tracking("test operation"):
            pass

    def test_memory_tracking_no_cuda(self):
        """Test memory tracking when CUDA unavailable"""
        from virnucpro.cuda.memory_manager import MemoryManager

        torch_mock.cuda.is_available.return_value = False

        mm = MemoryManager()

        with mm.memory_tracking("test operation"):
            pass

        # Reset mock
        torch_mock.cuda.is_available.return_value = True


# Integration tests
class TestIntegration:
    """Integration tests for DataLoader and MemoryManager"""

    def test_dataloader_with_memory_manager(self, mock_multiprocessing):
        """Test DataLoader with MemoryManager integration"""
        from virnucpro.data.dataloader_utils import create_sequence_dataloader
        from virnucpro.cuda.memory_manager import MemoryManager

        sequences = ["A" * i for i in range(1, 100)]

        mm = MemoryManager(cache_clear_interval=10)
        dataloader = create_sequence_dataloader(
            sequences=sequences,
            batch_size=16,
            sort_by_length=True
        )

        # Simulate batch processing
        for batch_num in range(20):
            mm.increment_and_clear()

        # Should have cleared cache twice (at batch 10 and 20)
        assert mm.batch_counter == 20

    def test_configure_memory_optimization_global(self):
        """Test global memory optimization configuration"""
        from virnucpro.cuda.memory_manager import configure_memory_optimization

        # Clear env var
        if 'PYTORCH_CUDA_ALLOC_CONF' in os.environ:
            del os.environ['PYTORCH_CUDA_ALLOC_CONF']

        mm = configure_memory_optimization(
            enable_expandable=True,
            cache_interval=50,
            verbose=True
        )

        assert mm.cache_clear_interval == 50
        assert mm.verbose is True
        assert 'expandable_segments:True' in os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '')


# Performance benchmarks (marked as slow)
class TestPerformance:
    """Performance benchmark tests"""

    @pytest.mark.slow
    def test_sorted_vs_unsorted_memory_efficiency(self):
        """Benchmark: sorted sequences should have better memory efficiency"""
        from virnucpro.data.dataloader_utils import SequenceDataset

        # Create sequences with varying lengths
        import random
        random.seed(42)
        sequences = ["A" * random.randint(10, 1000) for _ in range(1000)]

        # Sorted dataset
        sorted_dataset = SequenceDataset(sequences, sort_by_length=True)

        # Unsorted dataset
        unsorted_dataset = SequenceDataset(sequences, sort_by_length=False)

        # Both should have same length
        assert len(sorted_dataset) == len(unsorted_dataset)

        # Sorted sequences should be ordered by length
        sorted_lengths = [len(seq) for seq in sorted_dataset.sequences]
        assert sorted_lengths == sorted(sorted_lengths)


# Error handling tests
class TestErrorHandling:
    """Tests for error handling and edge cases"""

    def test_sequence_dataset_invalid_sequences(self):
        """Test SequenceDataset with objects that don't support len()"""
        from virnucpro.data.dataloader_utils import SequenceDataset

        # Objects without __len__
        sequences = [1, 2, 3, 4]

        # Should not raise exception, just log warning
        dataset = SequenceDataset(sequences, sort_by_length=True)

        # Should preserve original order
        assert dataset.sequences == [1, 2, 3, 4]

    def test_memory_manager_invalid_interval(self):
        """Test MemoryManager with negative interval"""
        from virnucpro.cuda.memory_manager import MemoryManager

        mm = MemoryManager(cache_clear_interval=-10)

        # Negative interval should disable clearing
        assert mm.should_clear_cache(0) is False
        assert mm.should_clear_cache(100) is False

    def test_estimate_memory_zero_batch_size(self):
        """Test memory estimation with edge cases"""
        from virnucpro.data.dataloader_utils import estimate_memory_usage

        estimates = estimate_memory_usage(
            batch_size=0,
            max_seq_length=1024
        )

        # Should handle gracefully
        assert estimates['activations'] == 0.0
        assert estimates['total'] == estimates['model']
