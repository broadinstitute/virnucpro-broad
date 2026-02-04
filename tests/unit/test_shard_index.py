"""Unit tests for sequence index creation and stride-based sharding.

Tests cover:
- Index creation from FASTA files
- Byte offset tracking and validation
- Cache validity and invalidation
- Stride distribution across workers
- Token balance and coverage verification
- Edge cases (empty files, single sequence, multi-file)
"""

import json
import tempfile
import time
from pathlib import Path

import pytest

from virnucpro.data.shard_index import (
    SequenceEntry,
    create_sequence_index,
    get_worker_indices,
    load_sequence_index,
)


@pytest.fixture
def temp_fasta_file():
    """Create temporary FASTA file with known sequences of varying lengths."""
    # Create 10 sequences with lengths: 500, 450, 400, 350, 300, 250, 200, 150, 100, 50
    fasta_content = ""
    for i in range(10):
        seq_id = f"seq_{i}"
        length = 500 - (i * 50)
        sequence = "M" * length
        fasta_content += f">{seq_id}\n{sequence}\n"

    with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as f:
        f.write(fasta_content)
        temp_path = Path(f.name)

    yield temp_path

    # Cleanup
    if temp_path.exists():
        temp_path.unlink()


@pytest.fixture
def temp_index_path():
    """Create temporary path for index file."""
    # Use NamedTemporaryFile to get unique path, then delete it
    # This avoids creating an empty file that would be invalid JSON
    with tempfile.NamedTemporaryFile(suffix='.json', delete=True) as f:
        temp_path = Path(f.name)

    yield temp_path

    # Cleanup
    if temp_path.exists():
        temp_path.unlink()


class TestSequenceIndex:
    """Tests for sequence index creation and validation."""

    def test_create_index_from_fasta(self, temp_fasta_file, temp_index_path):
        """Test creating index from FASTA file."""
        # Create index
        result_path = create_sequence_index([temp_fasta_file], temp_index_path)

        assert result_path == temp_index_path
        assert temp_index_path.exists()

        # Load and validate index
        index_data = load_sequence_index(temp_index_path)

        assert index_data['version'] == '1.0'
        assert 'created' in index_data
        assert index_data['total_sequences'] == 10
        assert index_data['total_tokens'] == sum(500 - i*50 for i in range(10))

        # Verify sequences exist
        sequences = index_data['sequences']
        assert len(sequences) == 10

    def test_index_sorted_by_length_descending(self, temp_fasta_file, temp_index_path):
        """Test that index is sorted by length descending."""
        create_sequence_index([temp_fasta_file], temp_index_path)
        index_data = load_sequence_index(temp_index_path)

        sequences = index_data['sequences']
        lengths = [s['length'] for s in sequences]

        # Verify descending order
        assert lengths == sorted(lengths, reverse=True)

        # Verify actual values
        expected_lengths = [500, 450, 400, 350, 300, 250, 200, 150, 100, 50]
        assert lengths == expected_lengths

    def test_index_contains_byte_offsets(self, temp_fasta_file, temp_index_path):
        """Test that each sequence entry has valid byte_offset."""
        create_sequence_index([temp_fasta_file], temp_index_path)
        index_data = load_sequence_index(temp_index_path)

        sequences = index_data['sequences']

        for seq in sequences:
            # Verify byte_offset exists and is non-negative
            assert 'byte_offset' in seq
            assert seq['byte_offset'] >= 0

            # Verify byte_offset actually points to header line
            with open(temp_fasta_file, 'rb') as f:
                f.seek(seq['byte_offset'])
                line = f.readline()
                # Should be a header line starting with '>'
                assert line.startswith(b'>')

    def test_index_cache_valid(self, temp_fasta_file, temp_index_path, caplog):
        """Test that valid cache is reused."""
        import logging
        caplog.set_level(logging.INFO, logger='virnucpro.data.shard_index')

        # Create index first time
        create_sequence_index([temp_fasta_file], temp_index_path)

        # Clear log
        caplog.clear()

        # Create again - should use cache
        create_sequence_index([temp_fasta_file], temp_index_path)

        # Check that cache was used
        assert "Using cached index" in caplog.text

    def test_index_cache_invalidated_on_mtime_change(
        self, temp_fasta_file, temp_index_path, caplog
    ):
        """Test that cache is rebuilt when FASTA file is modified."""
        import logging
        caplog.set_level(logging.INFO, logger='virnucpro.data.shard_index')

        # Create index
        create_sequence_index([temp_fasta_file], temp_index_path)
        original_mtime = temp_fasta_file.stat().st_mtime

        # Sleep to ensure mtime changes
        time.sleep(0.01)

        # Touch the file to update mtime
        temp_fasta_file.touch()
        new_mtime = temp_fasta_file.stat().st_mtime
        assert new_mtime > original_mtime

        # Clear log
        caplog.clear()

        # Recreate index - should rebuild
        create_sequence_index([temp_fasta_file], temp_index_path)

        # Check that cache was invalidated
        assert "Using cached index" not in caplog.text
        assert "Building sequence index" in caplog.text

    def test_total_tokens_correct(self, temp_fasta_file, temp_index_path):
        """Test that total_tokens field matches sum of sequence lengths."""
        create_sequence_index([temp_fasta_file], temp_index_path)
        index_data = load_sequence_index(temp_index_path)

        sequences = index_data['sequences']
        calculated_total = sum(s['length'] for s in sequences)

        assert index_data['total_tokens'] == calculated_total
        assert calculated_total == sum(500 - i*50 for i in range(10))


class TestStrideDistribution:
    """Tests for stride-based worker distribution."""

    def test_stride_distribution_world_size_4(self, temp_fasta_file, temp_index_path):
        """Test stride distribution with 4 workers."""
        # Create index with 100 sequences
        fasta_content = ""
        for i in range(100):
            fasta_content += f">seq_{i}\n{'M' * (100 - i)}\n"

        with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as f:
            f.write(fasta_content)
            temp_fasta = Path(f.name)

        try:
            create_sequence_index([temp_fasta], temp_index_path)

            # Get indices for each worker
            world_size = 4
            worker_indices = {}

            for rank in range(world_size):
                indices = get_worker_indices(temp_index_path, rank, world_size)
                worker_indices[rank] = indices

            # Verify each worker gets ~25 sequences
            for rank in range(world_size):
                # 100 sequences / 4 workers = 25 each
                assert len(worker_indices[rank]) == 25

            # Verify stride pattern
            assert worker_indices[0] == list(range(0, 100, 4))  # [0, 4, 8, ...]
            assert worker_indices[1] == list(range(1, 100, 4))  # [1, 5, 9, ...]
            assert worker_indices[2] == list(range(2, 100, 4))  # [2, 6, 10, ...]
            assert worker_indices[3] == list(range(3, 100, 4))  # [3, 7, 11, ...]

        finally:
            temp_fasta.unlink()

    def test_stride_distribution_coverage(self, temp_fasta_file, temp_index_path):
        """Test that all indices are covered exactly once across workers."""
        create_sequence_index([temp_fasta_file], temp_index_path)
        index_data = load_sequence_index(temp_index_path)
        total_sequences = len(index_data['sequences'])

        world_size = 4
        all_indices = []

        for rank in range(world_size):
            indices = get_worker_indices(temp_index_path, rank, world_size)
            all_indices.extend(indices)

        # Verify all indices covered exactly once
        assert len(all_indices) == total_sequences
        assert len(set(all_indices)) == total_sequences
        assert set(all_indices) == set(range(total_sequences))

    def test_stride_distribution_balanced_tokens(self, temp_index_path):
        """Test that token distribution is balanced across workers."""
        # Create FASTA with sequences sorted by length descending
        # This simulates real index (already sorted)
        fasta_content = ""
        for i in range(100):
            length = 1000 - (i * 10)  # Lengths: 1000, 990, 980, ..., 10
            fasta_content += f">seq_{i}\n{'M' * length}\n"

        with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as f:
            f.write(fasta_content)
            temp_fasta = Path(f.name)

        try:
            create_sequence_index([temp_fasta], temp_index_path)
            index_data = load_sequence_index(temp_index_path)

            world_size = 4
            token_counts = []

            for rank in range(world_size):
                indices = get_worker_indices(temp_index_path, rank, world_size)
                tokens = sum(index_data['sequences'][i]['length'] for i in indices)
                token_counts.append(tokens)

            # Calculate balance
            mean_tokens = sum(token_counts) / len(token_counts)
            max_deviation = max(abs(count - mean_tokens) for count in token_counts)
            max_deviation_pct = (max_deviation / mean_tokens) * 100

            # With stride distribution on sorted sequences,
            # deviation should be within 10% of mean
            # Each worker gets [0,4,8,...], [1,5,9,...], etc.
            # which mixes long and short sequences
            # (Actual deviation ~3% with 100 sequences)
            assert max_deviation_pct < 10.0

        finally:
            temp_fasta.unlink()

    def test_worker_indices_deterministic(self, temp_fasta_file, temp_index_path):
        """Test that same rank/world_size always returns same indices."""
        create_sequence_index([temp_fasta_file], temp_index_path)

        rank = 1
        world_size = 4

        # Call multiple times
        indices_1 = get_worker_indices(temp_index_path, rank, world_size)
        indices_2 = get_worker_indices(temp_index_path, rank, world_size)
        indices_3 = get_worker_indices(temp_index_path, rank, world_size)

        # All should be identical
        assert indices_1 == indices_2 == indices_3


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_fasta_file(self, temp_index_path):
        """Test handling of empty FASTA file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as f:
            f.write("")  # Empty file
            temp_fasta = Path(f.name)

        try:
            create_sequence_index([temp_fasta], temp_index_path)
            index_data = load_sequence_index(temp_index_path)

            # Should create valid index with 0 sequences
            assert index_data['total_sequences'] == 0
            assert index_data['total_tokens'] == 0
            assert len(index_data['sequences']) == 0

        finally:
            temp_fasta.unlink()

    def test_single_sequence(self, temp_index_path):
        """Test index with single sequence."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as f:
            f.write(">seq_only\nMKTAYIAK\n")
            temp_fasta = Path(f.name)

        try:
            create_sequence_index([temp_fasta], temp_index_path)
            index_data = load_sequence_index(temp_index_path)

            assert index_data['total_sequences'] == 1
            assert index_data['total_tokens'] == 8

            seq = index_data['sequences'][0]
            assert seq['sequence_id'] == 'seq_only'
            assert seq['length'] == 8

        finally:
            temp_fasta.unlink()

    def test_multi_file_index(self, temp_index_path):
        """Test creating index across multiple FASTA files."""
        # Create two FASTA files
        fasta1_content = ">file1_seq1\n" + "M" * 100 + "\n>file1_seq2\n" + "A" * 50 + "\n"
        fasta2_content = ">file2_seq1\n" + "K" * 150 + "\n>file2_seq2\n" + "T" * 75 + "\n"

        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.fasta', delete=False
        ) as f1, tempfile.NamedTemporaryFile(
            mode='w', suffix='.fasta', delete=False
        ) as f2:
            f1.write(fasta1_content)
            f2.write(fasta2_content)
            temp_fasta1 = Path(f1.name)
            temp_fasta2 = Path(f2.name)

        try:
            create_sequence_index([temp_fasta1, temp_fasta2], temp_index_path)
            index_data = load_sequence_index(temp_index_path)

            # Should have all 4 sequences
            assert index_data['total_sequences'] == 4
            assert index_data['total_tokens'] == 100 + 50 + 150 + 75

            # Verify both files tracked
            assert str(temp_fasta1) in index_data['fasta_mtimes']
            assert str(temp_fasta2) in index_data['fasta_mtimes']

            # Verify sequences sorted by length descending
            lengths = [s['length'] for s in index_data['sequences']]
            assert lengths == [150, 100, 75, 50]

        finally:
            temp_fasta1.unlink()
            temp_fasta2.unlink()

    def test_sequence_with_multiline_fasta(self, temp_index_path):
        """Test handling of multi-line FASTA sequences."""
        # FASTA format allows sequences to span multiple lines
        fasta_content = (
            ">multiline_seq\n"
            "MKTAYIAK\n"
            "QRKLSSDT\n"
            "GTYMLEKS\n"
        )

        with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as f:
            f.write(fasta_content)
            temp_fasta = Path(f.name)

        try:
            create_sequence_index([temp_fasta], temp_index_path)
            index_data = load_sequence_index(temp_index_path)

            seq = index_data['sequences'][0]
            # Total length should be 8 + 8 + 8 = 24
            assert seq['length'] == 24
            assert seq['sequence_id'] == 'multiline_seq'

        finally:
            temp_fasta.unlink()

    def test_corrupted_cache_rebuilds(self, temp_fasta_file, temp_index_path):
        """Test that corrupted cache file triggers rebuild."""
        # Create valid index first
        create_sequence_index([temp_fasta_file], temp_index_path)

        # Corrupt the cache
        with open(temp_index_path, 'w') as f:
            f.write("{ invalid json content")

        # Should rebuild without error
        create_sequence_index([temp_fasta_file], temp_index_path)
        index_data = load_sequence_index(temp_index_path)

        # Verify valid index was created
        assert index_data['total_sequences'] == 10
