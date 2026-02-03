"""Unit tests for sequence packing algorithms.

Tests GreedyPacker FFD algorithm and calculate_token_budget function.
"""

import pytest
import logging

# Mock torch for testing without CUDA
import sys
from unittest.mock import MagicMock, Mock

# Create mock torch module
torch_mock = MagicMock()
torch_mock.tensor = lambda data, dtype=None: data
torch_mock.long = 'long'
torch_mock.int32 = 'int32'
torch_mock.cuda.is_available.return_value = False

# Mock torch.cuda.get_device_properties
device_props_mock = Mock()
device_props_mock.total_memory = 16 * 1024**3  # 16GB
torch_mock.cuda.get_device_properties.return_value = device_props_mock

sys.modules['torch'] = torch_mock

from virnucpro.data.packing import GreedyPacker, calculate_token_budget


class TestGreedyPacker:
    """Test GreedyPacker FFD algorithm."""

    def test_ffd_sorting(self):
        """Verify sequences sorted by length descending (ARCH-11)."""
        packer = GreedyPacker(max_tokens_per_batch=100)
        sequences = [
            {'id': 'short1', 'sequence': 'MKTAY'},         # 5 aa
            {'id': 'long1', 'sequence': 'M' * 50},        # 50 aa
            {'id': 'medium1', 'sequence': 'MKTAYIAKQR'},  # 10 aa
        ]

        batches = packer.pack_sequences(sequences)

        # Verify sequences in first batch are sorted by length descending
        # FFD puts longest sequences first
        all_packed = [seq for batch in batches for seq in batch]
        assert len(all_packed[0]['sequence']) >= len(all_packed[1]['sequence'])
        assert len(all_packed[1]['sequence']) >= len(all_packed[2]['sequence'])

    def test_sort_by_length(self):
        """Explicit test for sort_by_length method."""
        packer = GreedyPacker(max_tokens_per_batch=100)
        sequences = [
            {'id': 'short', 'sequence': 'MKT'},
            {'id': 'long', 'sequence': 'MKTAYIAKQR'},
            {'id': 'medium', 'sequence': 'MKTAY'},
        ]

        sorted_seqs = packer.sort_by_length(sequences)

        # Verify descending length order
        assert sorted_seqs[0]['id'] == 'long'
        assert sorted_seqs[1]['id'] == 'medium'
        assert sorted_seqs[2]['id'] == 'short'

        # Verify original list not mutated
        assert sequences[0]['id'] == 'short'

    def test_deterministic_tiebreaking(self):
        """Same-length sequences sorted by ID."""
        packer = GreedyPacker(max_tokens_per_batch=100)
        sequences = [
            {'id': 'seq_c', 'sequence': 'MKTAY'},  # 5 aa
            {'id': 'seq_a', 'sequence': 'VLSPD'},  # 5 aa
            {'id': 'seq_b', 'sequence': 'AKQRG'},  # 5 aa
        ]

        sorted_seqs = packer.sort_by_length(sequences)

        # Same length, so sort by ID alphabetically
        assert sorted_seqs[0]['id'] == 'seq_a'
        assert sorted_seqs[1]['id'] == 'seq_b'
        assert sorted_seqs[2]['id'] == 'seq_c'

    def test_token_budget_respected(self):
        """Batches don't exceed max_tokens_per_batch."""
        max_tokens = 30
        packer = GreedyPacker(max_tokens_per_batch=max_tokens)
        sequences = [
            {'id': 'seq1', 'sequence': 'MKTAY'},         # 5 aa + 2 = 7 tokens
            {'id': 'seq2', 'sequence': 'MKTAYIAKQR'},    # 10 aa + 2 = 12 tokens
            {'id': 'seq3', 'sequence': 'VLSPAD'},        # 6 aa + 2 = 8 tokens
            {'id': 'seq4', 'sequence': 'AKQRG'},         # 5 aa + 2 = 7 tokens
        ]

        batches = packer.pack_sequences(sequences)

        # Verify each batch respects token budget
        for batch in batches:
            batch_tokens = sum(packer._tokenized_length(s['sequence']) for s in batch)
            assert batch_tokens <= max_tokens, f"Batch exceeds budget: {batch_tokens} > {max_tokens}"

    def test_truncation_warning(self, caplog):
        """Oversized sequences truncated with warning."""
        packer = GreedyPacker(max_tokens_per_batch=100, max_sequence_length=10)
        sequences = [
            {'id': 'oversized', 'sequence': 'M' * 50},  # 50 aa > 10 max
        ]

        with caplog.at_level(logging.WARNING):
            batches = packer.pack_sequences(sequences)

        # Verify truncation happened
        assert len(batches[0][0]['sequence']) == 10
        assert batches[0][0]['truncated'] is True

        # Verify warning was logged
        assert 'exceeds max_length' in caplog.text
        assert 'oversized' in caplog.text

    def test_efficiency_calculation(self):
        """Efficiency computed correctly."""
        packer = GreedyPacker(max_tokens_per_batch=30)
        sequences = [
            {'id': 'seq1', 'sequence': 'MKTAY'},      # 5 aa + 2 = 7 tokens
            {'id': 'seq2', 'sequence': 'MKTAYIAKQR'}, # 10 aa + 2 = 12 tokens
        ]

        batches = packer.pack_sequences(sequences)

        efficiency = packer.compute_efficiency(batches)

        # Total tokens: 7 + 12 = 19
        # Total capacity: 1 batch × 30 tokens = 30
        # Efficiency: 19 / 30 = 0.6333...
        assert 0.63 <= efficiency <= 0.64

    def test_bos_eos_accounting(self):
        """Verify +2 tokens per sequence accounted for."""
        packer = GreedyPacker(max_tokens_per_batch=100)

        # Test _tokenized_length directly
        assert packer._tokenized_length('MKTAY') == 7  # 5 + 2
        assert packer._tokenized_length('MKTAYIAKQR') == 12  # 10 + 2

    @pytest.mark.parametrize("sequences,expected_batches", [
        # Empty list
        ([], 0),
        # Single sequence
        ([{'id': 'seq1', 'sequence': 'MKTAY'}], 1),
        # Exact budget fit
        ([
            {'id': 'seq1', 'sequence': 'M' * 13},  # 13 + 2 = 15 tokens
            {'id': 'seq2', 'sequence': 'M' * 13},  # 13 + 2 = 15 tokens
        ], 1),  # 15 + 15 = 30 tokens = exact fit
    ])
    def test_edge_cases(self, sequences, expected_batches):
        """Test edge cases (empty list, single sequence, exact budget)."""
        packer = GreedyPacker(max_tokens_per_batch=30)
        batches = packer.pack_sequences(sequences)
        assert len(batches) == expected_batches

    def test_empty_efficiency(self):
        """Empty batches return 0.0 efficiency."""
        packer = GreedyPacker(max_tokens_per_batch=100)
        assert packer.compute_efficiency([]) == 0.0

    def test_packing_with_varied_lengths(self):
        """Test realistic packing scenario with varied sequence lengths."""
        packer = GreedyPacker(max_tokens_per_batch=50)
        sequences = [
            {'id': f'seq{i}', 'sequence': 'M' * length}
            for i, length in enumerate([5, 10, 15, 3, 7, 20, 8, 12])
        ]

        batches = packer.pack_sequences(sequences)

        # Verify all sequences were packed
        total_packed = sum(len(batch) for batch in batches)
        assert total_packed == len(sequences)

        # Verify FFD: first batch should have longest sequence
        first_batch_max = max(len(s['sequence']) for s in batches[0])
        assert first_batch_max == 20  # Longest sequence


class TestCalculateTokenBudget:
    """Test dynamic token budget calculation."""

    def test_calculate_token_budget_no_cuda(self, caplog):
        """Test fallback when CUDA unavailable."""
        # torch_mock.cuda.is_available already returns False
        with caplog.at_level(logging.WARNING):
            budget = calculate_token_budget(device_id=0)

        assert budget == 4096
        assert 'CUDA not available' in caplog.text

    def test_calculate_token_budget_with_cuda(self, caplog):
        """Test GPU memory-based calculation (mocked)."""
        # Enable CUDA mock
        torch_mock.cuda.is_available.return_value = True

        with caplog.at_level(logging.INFO):
            budget = calculate_token_budget(
                device_id=0,
                model_memory_gb=5.0,
                safety_margin_gb=2.0,
                bytes_per_token=4096,
            )

        # 16GB GPU - 5GB model - 2GB safety = 9GB available
        # 9GB = 9 * 1024^3 bytes
        # tokens = (9 * 1024^3) / 4096 = 2359296
        # Clamped to max_tokens=16384
        assert budget == 16384  # Maxed out

        # Verify logging
        assert 'Token budget' in caplog.text
        assert '16.0GB total' in caplog.text

        # Reset for other tests
        torch_mock.cuda.is_available.return_value = False

    def test_calculate_token_budget_min_clamp(self):
        """Test minimum token budget clamping."""
        torch_mock.cuda.is_available.return_value = True

        # Set very low available memory
        small_gpu_props = Mock()
        small_gpu_props.total_memory = 2 * 1024**3  # 2GB GPU
        torch_mock.cuda.get_device_properties.return_value = small_gpu_props

        budget = calculate_token_budget(
            device_id=0,
            model_memory_gb=1.5,  # Model uses most of GPU
            safety_margin_gb=0.4,
            min_tokens=1024,
        )

        # Should clamp to minimum
        assert budget >= 1024

        # Reset
        torch_mock.cuda.is_available.return_value = False
        torch_mock.cuda.get_device_properties.return_value = device_props_mock

    def test_calculate_token_budget_max_clamp(self):
        """Test maximum token budget clamping."""
        torch_mock.cuda.is_available.return_value = True

        # Set very high available memory
        large_gpu_props = Mock()
        large_gpu_props.total_memory = 80 * 1024**3  # 80GB A100
        torch_mock.cuda.get_device_properties.return_value = large_gpu_props

        budget = calculate_token_budget(
            device_id=0,
            model_memory_gb=5.0,
            safety_margin_gb=2.0,
            max_tokens=16384,
        )

        # Should clamp to maximum
        assert budget == 16384

        # Reset
        torch_mock.cuda.is_available.return_value = False
        torch_mock.cuda.get_device_properties.return_value = device_props_mock
