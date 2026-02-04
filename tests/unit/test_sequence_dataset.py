"""Unit tests for SequenceDataset and IndexBasedDataset modules.

Tests cover:
- IndexBasedDataset iteration and ordering
- Byte offset seeking and sequence reading
- CUDA isolation validation
- DataLoader integration
- VarlenCollator compatibility
"""

import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader

from virnucpro.data.sequence_dataset import IndexBasedDataset
from virnucpro.data.shard_index import create_sequence_index
from virnucpro.data.collators import VarlenCollator


# Test fixtures

@pytest.fixture
def temp_fasta_with_index() -> Tuple[Path, Path, List[dict]]:
    """Create temporary FASTA file and index for testing.

    Returns:
        Tuple of (fasta_path, index_path, expected_sequences)

    Expected sequences are sorted by length descending (as in index).
    """
    # Create temp FASTA file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as f:
        fasta_path = Path(f.name)

        # Write test sequences with varying lengths
        f.write(">seq1\n")
        f.write("MKTAYIAKQR\n")  # length 10

        f.write(">seq2\n")
        f.write("MKTAYIAKQRQIS\n")  # length 13
        f.write("MGST\n")  # continued, total length 17

        f.write(">seq3\n")
        f.write("MKT\n")  # length 3

        f.write(">seq4\n")
        f.write("MKTAYIAKQRQISGTG\n")  # length 16
        f.write("AEL\n")  # continued, total length 19

        f.write(">seq5\n")
        f.write("MKTAYI\n")  # length 6

    # Create index
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as idx_f:
        index_path = Path(idx_f.name)

    create_sequence_index([fasta_path], index_path)

    # Expected sequences in index order (sorted by length descending)
    expected = [
        {'id': 'seq4', 'sequence': 'MKTAYIAKQRQISGTGAEL', 'length': 19},
        {'id': 'seq2', 'sequence': 'MKTAYIAKQRQISMGST', 'length': 17},
        {'id': 'seq1', 'sequence': 'MKTAYIAKQR', 'length': 10},
        {'id': 'seq5', 'sequence': 'MKTAYI', 'length': 6},
        {'id': 'seq3', 'sequence': 'MKT', 'length': 3},
    ]

    yield fasta_path, index_path, expected

    # Cleanup
    fasta_path.unlink()
    index_path.unlink()


# TestIndexBasedDataset class

class TestIndexBasedDataset:
    """Test suite for IndexBasedDataset."""

    @patch('torch.cuda.is_available', return_value=False)
    def test_dataset_yields_all_sequences(self, mock_cuda, temp_fasta_with_index):
        """Test that dataset yields all sequences when given all indices."""
        fasta_path, index_path, expected = temp_fasta_with_index

        # Create dataset with all indices
        indices = list(range(len(expected)))
        dataset = IndexBasedDataset(index_path, indices, max_length=1024)

        # Collect all sequences
        sequences = list(dataset)

        # Verify count
        assert len(sequences) == len(expected)

        # Verify each sequence
        for seq, exp in zip(sequences, expected):
            assert seq['id'] == exp['id']
            assert seq['sequence'] == exp['sequence']
            assert 'file' in seq

    @patch('torch.cuda.is_available', return_value=False)
    def test_dataset_yields_in_index_order(self, mock_cuda, temp_fasta_with_index):
        """Test that sequences are returned in index order (length-sorted)."""
        fasta_path, index_path, expected = temp_fasta_with_index

        # Create dataset with all indices
        indices = list(range(len(expected)))
        dataset = IndexBasedDataset(index_path, indices, max_length=1024)

        # Collect sequences
        sequences = list(dataset)

        # Verify order matches expected (descending by length)
        assert sequences[0]['id'] == 'seq4'  # length 19
        assert sequences[1]['id'] == 'seq2'  # length 17
        assert sequences[2]['id'] == 'seq1'  # length 10
        assert sequences[3]['id'] == 'seq5'  # length 6
        assert sequences[4]['id'] == 'seq3'  # length 3

        # Verify lengths are descending
        lengths = [len(s['sequence']) for s in sequences]
        assert lengths == sorted(lengths, reverse=True)

    @patch('torch.cuda.is_available', return_value=False)
    def test_dataset_subset_indices(self, mock_cuda, temp_fasta_with_index):
        """Test that dataset yields only specified subset of indices."""
        fasta_path, index_path, expected = temp_fasta_with_index

        # Create dataset with subset of indices (stride pattern)
        indices = [0, 2, 4]  # seq4, seq1, seq3
        dataset = IndexBasedDataset(index_path, indices, max_length=1024)

        # Collect sequences
        sequences = list(dataset)

        # Verify count
        assert len(sequences) == 3

        # Verify correct sequences
        assert sequences[0]['id'] == 'seq4'
        assert sequences[1]['id'] == 'seq1'
        assert sequences[2]['id'] == 'seq3'

    @patch('torch.cuda.is_available', return_value=False)
    def test_dataset_byte_offset_seek(self, mock_cuda, temp_fasta_with_index):
        """Test that sequences are read correctly via byte offset."""
        fasta_path, index_path, expected = temp_fasta_with_index

        # Create dataset
        indices = list(range(len(expected)))
        dataset = IndexBasedDataset(index_path, indices, max_length=1024)

        # Collect sequences
        sequences = list(dataset)

        # Verify content matches expected (byte offset read correctly)
        for seq, exp in zip(sequences, expected):
            assert seq['sequence'] == exp['sequence'], \
                f"Sequence {exp['id']} content mismatch"

    @patch('torch.cuda.is_available', return_value=False)
    def test_dataset_max_length_truncation(self, mock_cuda, temp_fasta_with_index):
        """Test that long sequences are truncated to max_length."""
        fasta_path, index_path, expected = temp_fasta_with_index

        # Create dataset with small max_length
        max_length = 8
        indices = list(range(len(expected)))
        dataset = IndexBasedDataset(index_path, indices, max_length=max_length)

        # Collect sequences
        sequences = list(dataset)

        # Verify all sequences are truncated
        for seq in sequences:
            assert len(seq['sequence']) <= max_length, \
                f"Sequence {seq['id']} not truncated (length {len(seq['sequence'])})"

        # Verify specific truncations
        assert sequences[0]['sequence'] == 'MKTAYIAK'  # seq4 truncated to 8
        assert sequences[4]['sequence'] == 'MKT'  # seq3 is 3, not truncated

    @patch('torch.cuda.is_available', return_value=False)
    def test_dataset_single_process(self, mock_cuda, temp_fasta_with_index):
        """Test that dataset works without worker sharding (worker_info is None)."""
        fasta_path, index_path, expected = temp_fasta_with_index

        # Create dataset
        indices = list(range(len(expected)))
        dataset = IndexBasedDataset(index_path, indices, max_length=1024)

        # Iterate in single-process mode (no DataLoader workers)
        sequences = list(dataset)

        # Verify all sequences returned
        assert len(sequences) == len(expected)

    @patch('torch.cuda.is_available', return_value=False)
    def test_dataset_len(self, mock_cuda, temp_fasta_with_index):
        """Test that __len__ returns correct count."""
        fasta_path, index_path, expected = temp_fasta_with_index

        # Full dataset
        indices = list(range(len(expected)))
        dataset = IndexBasedDataset(index_path, indices, max_length=1024)
        assert len(dataset) == len(expected)

        # Subset dataset
        indices = [0, 2, 4]
        dataset = IndexBasedDataset(index_path, indices, max_length=1024)
        assert len(dataset) == 3


# TestDatasetIntegration class

class TestDatasetIntegration:
    """Integration tests for IndexBasedDataset with DataLoader and collators."""

    @patch('torch.cuda.is_available', return_value=False)
    def test_dataset_with_dataloader(self, mock_cuda, temp_fasta_with_index):
        """Test that DataLoader iterates IndexBasedDataset correctly."""
        fasta_path, index_path, expected = temp_fasta_with_index

        # Create dataset
        indices = list(range(len(expected)))
        dataset = IndexBasedDataset(index_path, indices, max_length=1024)

        # Create DataLoader (single-process for simplicity)
        dataloader = DataLoader(dataset, batch_size=None, num_workers=0)

        # Collect sequences
        sequences = []
        for item in dataloader:
            sequences.append(item)

        # Verify all sequences returned
        assert len(sequences) == len(expected)

        # Verify order preserved
        assert sequences[0]['id'] == 'seq4'
        assert sequences[-1]['id'] == 'seq3'

    @patch('torch.cuda.is_available', return_value=False)
    def test_dataset_with_collator(self, mock_cuda, temp_fasta_with_index):
        """Test that VarlenCollator processes IndexBasedDataset output correctly."""
        fasta_path, index_path, expected = temp_fasta_with_index

        # Create dataset
        indices = list(range(len(expected)))
        dataset = IndexBasedDataset(index_path, indices, max_length=1024)

        # Load ESM-2 model for batch_converter
        from virnucpro.models.esm2_flash import load_esm2_model
        import torch

        # Use smallest ESM-2 model for testing
        model, batch_converter = load_esm2_model(
            model_name='esm2_t6_8M_UR50D',
            device='cpu'
        )

        # Create collator
        collator = VarlenCollator(
            batch_converter=batch_converter,
            max_tokens_per_batch=512,
            enable_packing=False  # Disable packing for simple test
        )

        # Process dataset items through collator
        items = list(dataset)

        # Collator expects list of items
        batch = collator([items[0]])

        # Verify collator output format (packed batch format)
        assert 'input_ids' in batch
        assert 'cu_seqlens' in batch
        assert 'max_seqlen' in batch
        assert 'sequence_ids' in batch
        assert 'num_sequences' in batch

        # Verify sequence ID preserved
        assert batch['sequence_ids'][0] == 'seq4'

        # Verify num_sequences is correct
        assert batch['num_sequences'] == 1
