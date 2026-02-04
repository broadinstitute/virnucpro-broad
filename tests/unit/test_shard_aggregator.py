"""Unit tests for HDF5 shard aggregation.

Tests aggregate_shards function with chunk-wise streaming, duplicate detection,
missing sequence validation, and exception safety.
"""

import pytest
import h5py
import numpy as np
import tempfile
from pathlib import Path
from typing import List, Dict

from virnucpro.pipeline.shard_aggregator import (
    aggregate_shards,
    validate_shard_completeness,
    get_shard_info,
    CHUNK_SIZE
)


@pytest.fixture
def temp_output_dir(tmp_path):
    """Temporary directory for output files."""
    return tmp_path


def create_test_shard(
    path: Path,
    sequence_ids: List[str],
    embedding_dim: int = 128
) -> Path:
    """Create HDF5 shard with embeddings and sequence_ids.

    Args:
        path: Output path for shard file
        sequence_ids: List of sequence IDs
        embedding_dim: Embedding dimension (default: 128)

    Returns:
        Path to created shard file
    """
    num_sequences = len(sequence_ids)

    with h5py.File(path, 'w') as f:
        # Create embeddings with unique values per sequence
        embeddings = np.random.randn(num_sequences, embedding_dim).astype('float32')
        # Make each sequence's embedding identifiable (first value = index)
        for i in range(num_sequences):
            embeddings[i, 0] = float(i)

        f.create_dataset('embeddings', data=embeddings)

        # Variable-length string dataset for sequence IDs
        string_dtype = h5py.special_dtype(vlen=str)
        sequence_ids_arr = np.array(sequence_ids, dtype=object)
        f.create_dataset('sequence_ids', data=sequence_ids_arr, dtype=string_dtype)

    return path


class TestAggregateShards:
    """Test aggregate_shards basic functionality."""

    def test_aggregate_two_shards(self, temp_output_dir):
        """Merge 2 shards and verify combined output."""
        shard1_path = temp_output_dir / 'shard_0.h5'
        shard2_path = temp_output_dir / 'shard_1.h5'
        output_path = temp_output_dir / 'merged.h5'

        # Create test shards
        create_test_shard(shard1_path, ['seq1', 'seq2', 'seq3'])
        create_test_shard(shard2_path, ['seq4', 'seq5'])

        # Aggregate
        result_path = aggregate_shards(
            [shard1_path, shard2_path],
            output_path
        )

        assert result_path == output_path
        assert output_path.exists()

        # Verify merged output
        with h5py.File(output_path, 'r') as f:
            assert f['embeddings'].shape[0] == 5
            assert f['sequence_ids'].shape[0] == 5

    def test_output_contains_all_embeddings(self, temp_output_dir):
        """All embeddings present in output."""
        shard1_path = temp_output_dir / 'shard_0.h5'
        shard2_path = temp_output_dir / 'shard_1.h5'
        output_path = temp_output_dir / 'merged.h5'

        create_test_shard(shard1_path, ['seq1', 'seq2'])
        create_test_shard(shard2_path, ['seq3', 'seq4'])

        aggregate_shards([shard1_path, shard2_path], output_path)

        with h5py.File(output_path, 'r') as f:
            embeddings = f['embeddings'][:]
            assert embeddings.shape[0] == 4
            assert embeddings.shape[1] == 128  # Default embedding_dim

    def test_output_contains_all_ids(self, temp_output_dir):
        """All sequence IDs present in output."""
        shard1_path = temp_output_dir / 'shard_0.h5'
        shard2_path = temp_output_dir / 'shard_1.h5'
        output_path = temp_output_dir / 'merged.h5'

        create_test_shard(shard1_path, ['seq1', 'seq2'])
        create_test_shard(shard2_path, ['seq3', 'seq4'])

        aggregate_shards([shard1_path, shard2_path], output_path)

        with h5py.File(output_path, 'r') as f:
            sequence_ids = [sid.decode('utf-8') if isinstance(sid, bytes) else sid
                           for sid in f['sequence_ids'][:]]
            assert set(sequence_ids) == {'seq1', 'seq2', 'seq3', 'seq4'}

    def test_preserves_embedding_values(self, temp_output_dir):
        """Embedding values identical after merge."""
        shard_path = temp_output_dir / 'shard_0.h5'
        output_path = temp_output_dir / 'merged.h5'

        # Create shard with known embedding values
        with h5py.File(shard_path, 'w') as f:
            embeddings = np.array([
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0]
            ], dtype='float32')
            f.create_dataset('embeddings', data=embeddings)

            string_dtype = h5py.special_dtype(vlen=str)
            f.create_dataset('sequence_ids', data=['seq1', 'seq2'], dtype=string_dtype)

        aggregate_shards([shard_path], output_path)

        # Verify values preserved
        with h5py.File(output_path, 'r') as f:
            output_embeddings = f['embeddings'][:]
            np.testing.assert_array_equal(output_embeddings[0], [1.0, 2.0, 3.0])
            np.testing.assert_array_equal(output_embeddings[1], [4.0, 5.0, 6.0])

    def test_chunk_wise_memory(self, temp_output_dir):
        """Large shard processed in chunks."""
        # Create shard larger than CHUNK_SIZE
        large_count = CHUNK_SIZE + 500
        shard_path = temp_output_dir / 'shard_large.h5'
        output_path = temp_output_dir / 'merged.h5'

        sequence_ids = [f'seq{i}' for i in range(large_count)]
        create_test_shard(shard_path, sequence_ids)

        aggregate_shards([shard_path], output_path)

        # Verify all sequences present
        with h5py.File(output_path, 'r') as f:
            assert f['embeddings'].shape[0] == large_count
            assert f['sequence_ids'].shape[0] == large_count


class TestDuplicateDetection:
    """Test duplicate sequence ID detection."""

    def test_duplicate_within_shard(self, temp_output_dir):
        """Shard with duplicate IDs raises ValueError."""
        shard_path = temp_output_dir / 'shard_dup.h5'
        output_path = temp_output_dir / 'merged.h5'

        # Create shard with duplicate IDs
        create_test_shard(shard_path, ['seq1', 'seq2', 'seq1'])  # seq1 appears twice

        with pytest.raises(ValueError, match="Duplicate sequence ID found: seq1"):
            aggregate_shards([shard_path], output_path)

        # Verify partial output cleaned up
        assert not output_path.exists()

    def test_duplicate_across_shards(self, temp_output_dir):
        """Same ID in two shards raises ValueError."""
        shard1_path = temp_output_dir / 'shard_0.h5'
        shard2_path = temp_output_dir / 'shard_1.h5'
        output_path = temp_output_dir / 'merged.h5'

        create_test_shard(shard1_path, ['seq1', 'seq2'])
        create_test_shard(shard2_path, ['seq2', 'seq3'])  # seq2 is duplicate

        with pytest.raises(ValueError, match="Duplicate sequence ID found: seq2"):
            aggregate_shards([shard1_path, shard2_path], output_path)

    def test_duplicate_error_message(self, temp_output_dir):
        """Error message includes duplicate ID and shard."""
        shard1_path = temp_output_dir / 'shard_0.h5'
        shard2_path = temp_output_dir / 'shard_1.h5'
        output_path = temp_output_dir / 'merged.h5'

        create_test_shard(shard1_path, ['seq1'])
        create_test_shard(shard2_path, ['seq1'])  # Duplicate

        with pytest.raises(ValueError) as exc_info:
            aggregate_shards([shard1_path, shard2_path], output_path)

        error_msg = str(exc_info.value)
        assert 'seq1' in error_msg
        assert 'shard_1.h5' in error_msg


class TestMissingValidation:
    """Test missing sequence validation."""

    def test_missing_sequences_raises(self, temp_output_dir):
        """Missing expected IDs raises ValueError."""
        shard_path = temp_output_dir / 'shard_0.h5'
        output_path = temp_output_dir / 'merged.h5'

        create_test_shard(shard_path, ['seq1', 'seq2'])

        expected_ids = {'seq1', 'seq2', 'seq3', 'seq4'}  # seq3, seq4 missing

        with pytest.raises(ValueError, match="Missing 2 expected sequences"):
            aggregate_shards([shard_path], output_path, expected_ids)

    def test_missing_error_lists_ids(self, temp_output_dir):
        """Error includes first 10 missing IDs."""
        shard_path = temp_output_dir / 'shard_0.h5'
        output_path = temp_output_dir / 'merged.h5'

        create_test_shard(shard_path, ['seq1'])

        # Create many missing IDs
        expected_ids = {f'seq{i}' for i in range(1, 20)}  # seq2-seq19 missing

        with pytest.raises(ValueError) as exc_info:
            aggregate_shards([shard_path], output_path, expected_ids)

        error_msg = str(exc_info.value)
        assert 'Missing 18 expected sequences' in error_msg
        assert 'First 10 missing IDs' in error_msg

    def test_extra_sequences_warns(self, temp_output_dir, caplog):
        """Extra IDs log warning (not error)."""
        import logging
        caplog.set_level(logging.WARNING)

        shard_path = temp_output_dir / 'shard_0.h5'
        output_path = temp_output_dir / 'merged.h5'

        create_test_shard(shard_path, ['seq1', 'seq2', 'seq3'])

        expected_ids = {'seq1', 'seq2'}  # seq3 is extra

        # Should succeed (extra is warning, not error)
        aggregate_shards([shard_path], output_path, expected_ids)

        # Verify warning logged
        assert any('extra sequences' in record.message.lower() for record in caplog.records)

    def test_no_expected_ids_skips_validation(self, temp_output_dir):
        """None expected_ids skips validation check."""
        shard_path = temp_output_dir / 'shard_0.h5'
        output_path = temp_output_dir / 'merged.h5'

        create_test_shard(shard_path, ['seq1', 'seq2'])

        # Should succeed without validation
        result = aggregate_shards([shard_path], output_path, expected_sequence_ids=None)
        assert result == output_path


class TestValidateShardCompleteness:
    """Test validate_shard_completeness function."""

    def test_returns_missing_and_extra(self, temp_output_dir):
        """Quick validation returns correct sets."""
        shard_path = temp_output_dir / 'shard_0.h5'

        create_test_shard(shard_path, ['seq1', 'seq2', 'seq3'])

        expected_ids = {'seq1', 'seq2', 'seq4', 'seq5'}

        missing, extra = validate_shard_completeness([shard_path], expected_ids)

        assert missing == {'seq4', 'seq5'}
        assert extra == {'seq3'}

    def test_empty_shards(self, temp_output_dir):
        """Handles empty shards gracefully."""
        shard_path = temp_output_dir / 'shard_empty.h5'

        create_test_shard(shard_path, [])  # Empty shard

        expected_ids = {'seq1', 'seq2'}

        missing, extra = validate_shard_completeness([shard_path], expected_ids)

        assert missing == {'seq1', 'seq2'}
        assert extra == set()


class TestExceptionSafety:
    """Test exception handling and cleanup."""

    def test_cleanup_on_failure(self, temp_output_dir):
        """Partial output deleted on exception."""
        shard1_path = temp_output_dir / 'shard_0.h5'
        shard2_path = temp_output_dir / 'shard_1.h5'
        output_path = temp_output_dir / 'merged.h5'

        create_test_shard(shard1_path, ['seq1', 'seq2'])
        create_test_shard(shard2_path, ['seq2', 'seq3'])  # Duplicate seq2

        # Should raise ValueError for duplicate
        with pytest.raises(ValueError):
            aggregate_shards([shard1_path, shard2_path], output_path)

        # Output file should be cleaned up
        assert not output_path.exists()

    def test_no_shard_files(self, temp_output_dir):
        """Empty shard list raises ValueError."""
        output_path = temp_output_dir / 'merged.h5'

        with pytest.raises(ValueError, match="No shard files provided"):
            aggregate_shards([], output_path)


class TestGetShardInfo:
    """Test get_shard_info metadata function."""

    def test_returns_metadata(self, temp_output_dir):
        """Get metadata from shard file."""
        shard_path = temp_output_dir / 'shard_0.h5'

        create_test_shard(shard_path, ['seq1', 'seq2', 'seq3'], embedding_dim=256)

        info = get_shard_info(shard_path)

        assert info['num_sequences'] == 3
        assert info['embedding_dim'] == 256
        assert len(info['sequence_ids_sample']) == 3
        assert 'seq1' in info['sequence_ids_sample']

    def test_sample_truncates_large_shards(self, temp_output_dir):
        """Sample limited to first 5 IDs for large shards."""
        shard_path = temp_output_dir / 'shard_large.h5'

        sequence_ids = [f'seq{i}' for i in range(100)]
        create_test_shard(shard_path, sequence_ids)

        info = get_shard_info(shard_path)

        assert info['num_sequences'] == 100
        assert len(info['sequence_ids_sample']) == 5
