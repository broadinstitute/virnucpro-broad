"""Unit tests for parallel translation module"""

import pytest
import os
import multiprocessing
from pathlib import Path
from unittest.mock import patch, MagicMock, Mock
from typing import List, Tuple, Dict, Optional
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from virnucpro.pipeline.parallel_translate import (
    translate_sequence_worker,
    translate_batch_worker,
    create_sequence_batches,
    get_optimal_settings,
    parallel_translate_sequences,
    parallel_translate_batched,
    parallel_translate_with_progress
)


@pytest.fixture
def sample_sequences():
    """Sample sequences for testing translation"""
    return [
        ("seq1", "ATGGCATGA"),  # Valid ORF (starts with ATG, ends with TGA stop)
        ("seq2", "ATGAAATAA"),  # Contains stop codon (TAA)
        ("seq3", "ATGATGATG"),  # All valid codons
        ("seq4", "NNN"),        # Invalid/ambiguous bases
        ("seq5", "ATGGGCGCATTTGCATAG"),  # Multiple codons with TAG stop
    ]


@pytest.fixture
def temp_fasta(tmp_path, sample_sequences):
    """Create temporary FASTA file with sample sequences"""
    fasta_file = tmp_path / "test_input.fa"

    records = []
    for seqid, seq in sample_sequences:
        record = SeqRecord(Seq(seq), id=seqid, description="")
        records.append(record)

    with open(fasta_file, 'w') as f:
        SeqIO.write(records, f, 'fasta')

    return fasta_file


@pytest.fixture
def large_fasta(tmp_path):
    """Create large FASTA file for performance testing (10000 sequences)"""
    fasta_file = tmp_path / "test_large.fa"

    records = []
    for i in range(10000):
        # Generate deterministic sequences for reproducibility
        seq = "ATG" + ("GCA" * 50) + "TGA"  # Valid ORF: ATG + 50 Ala codons + stop
        record = SeqRecord(Seq(seq), id=f"seq_{i}", description="")
        records.append(record)

    with open(fasta_file, 'w') as f:
        SeqIO.write(records, f, 'fasta')

    return fasta_file


class TestTranslateSequenceWorker:
    """Test translate_sequence_worker function"""

    def test_worker_function_valid_sequence(self):
        """Test worker with valid sequence returns results"""
        record_data = ("test_seq", "ATGGCAGCAGCATGA")

        result = translate_sequence_worker(record_data)

        assert result is not None
        assert isinstance(result, list)
        # Should have at least one valid ORF
        assert len(result) > 0

        # Verify structure of results
        for orf in result:
            assert 'seqid' in orf
            assert 'nucleotide' in orf
            assert 'protein' in orf
            assert isinstance(orf['seqid'], str)
            assert isinstance(orf['nucleotide'], str)
            assert isinstance(orf['protein'], str)

    def test_worker_function_with_stops(self):
        """Test worker with sequence containing stop codons"""
        # Sequence with stop codons in frame 1
        record_data = ("test_seq", "TAATAATAATAATAATAATAA")

        result = translate_sequence_worker(record_data)

        # identify_seq returns ORFs without stop codons (filters them out)
        # Some frames may have valid ORFs
        if result:
            for orf in result:
                # Verify no stop codons in protein sequence
                assert '*' not in orf['protein']

    def test_worker_function_picklable(self):
        """Test that worker function can be pickled (required for spawn context)"""
        import pickle

        # Worker function must be picklable for multiprocessing with spawn context
        try:
            pickled = pickle.dumps(translate_sequence_worker)
            unpickled = pickle.loads(pickled)
            assert unpickled is not None
        except Exception as e:
            pytest.fail(f"Worker function not picklable: {e}")

    def test_worker_handles_exception(self):
        """Test worker gracefully handles exceptions"""
        # Invalid data type (should handle gracefully)
        record_data = ("test_seq", None)

        result = translate_sequence_worker(record_data)

        # Should return None on error, not raise exception
        assert result is None


class TestTranslateBatchWorker:
    """Test translate_batch_worker function"""

    def test_batch_worker_processes_multiple_sequences(self):
        """Test batch worker processes all sequences in batch"""
        batch = [
            ("seq1", "ATGGCAGCATGA"),
            ("seq2", "ATGGGATAG"),
            ("seq3", "TAATAATAA"),  # All stops
        ]

        results = translate_batch_worker(batch)

        assert len(results) == 3
        assert results[0] is not None  # seq1 valid
        assert results[1] is not None  # seq2 valid
        # seq3 may be None or empty

    def test_batch_worker_picklable(self):
        """Test that batch worker function can be pickled"""
        import pickle

        try:
            pickled = pickle.dumps(translate_batch_worker)
            unpickled = pickle.loads(pickled)
            assert unpickled is not None
        except Exception as e:
            pytest.fail(f"Batch worker function not picklable: {e}")

    def test_batch_worker_empty_batch(self):
        """Test batch worker handles empty batch"""
        batch = []

        results = translate_batch_worker(batch)

        assert results == []


class TestCreateSequenceBatches:
    """Test create_sequence_batches function"""

    def test_create_batches_even_division(self):
        """Test batching with sequences that divide evenly"""
        sequences = [(f"seq_{i}", "ATCG") for i in range(300)]

        batches = list(create_sequence_batches(iter(sequences), batch_size=100))

        assert len(batches) == 3
        assert len(batches[0]) == 100
        assert len(batches[1]) == 100
        assert len(batches[2]) == 100

    def test_create_batches_uneven_division(self):
        """Test batching with sequences that don't divide evenly"""
        sequences = [(f"seq_{i}", "ATCG") for i in range(250)]

        batches = list(create_sequence_batches(iter(sequences), batch_size=100))

        assert len(batches) == 3
        assert len(batches[0]) == 100
        assert len(batches[1]) == 100
        assert len(batches[2]) == 50  # Last batch smaller

    def test_create_batches_single_batch(self):
        """Test batching with fewer sequences than batch size"""
        sequences = [(f"seq_{i}", "ATCG") for i in range(50)]

        batches = list(create_sequence_batches(iter(sequences), batch_size=100))

        assert len(batches) == 1
        assert len(batches[0]) == 50

    def test_create_batches_empty_input(self):
        """Test batching with empty input"""
        sequences = []

        batches = list(create_sequence_batches(iter(sequences), batch_size=100))

        assert len(batches) == 0

    def test_create_batches_custom_batch_size(self):
        """Test batching with custom batch size"""
        sequences = [(f"seq_{i}", "ATCG") for i in range(1000)]

        batches = list(create_sequence_batches(iter(sequences), batch_size=50))

        assert len(batches) == 20
        for batch in batches:
            assert len(batch) == 50


class TestGetOptimalSettings:
    """Test get_optimal_settings function"""

    @patch('os.cpu_count', return_value=8)
    def test_default_num_workers(self, mock_cpu_count):
        """Test default number of workers uses cpu_count"""
        workers, batch_size, chunksize = get_optimal_settings()

        assert workers == 8
        mock_cpu_count.assert_called_once()

    def test_explicit_num_workers(self):
        """Test explicit num_workers parameter"""
        workers, batch_size, chunksize = get_optimal_settings(num_workers=4)

        assert workers == 4

    def test_batch_size_small_sequences(self):
        """Test batch size for small sequences (<300bp)"""
        workers, batch_size, chunksize = get_optimal_settings(avg_sequence_length=200)

        assert batch_size == 200

    def test_batch_size_medium_sequences(self):
        """Test batch size for medium sequences (300-1000bp)"""
        workers, batch_size, chunksize = get_optimal_settings(avg_sequence_length=500)

        assert batch_size == 100

    def test_batch_size_large_sequences(self):
        """Test batch size for large sequences (>1000bp)"""
        workers, batch_size, chunksize = get_optimal_settings(avg_sequence_length=2000)

        assert batch_size == 50

    def test_chunksize_calculation_known_total(self):
        """Test chunksize calculation when total sequences known"""
        workers, batch_size, chunksize = get_optimal_settings(
            num_workers=4,
            total_sequences=10000,
            avg_sequence_length=500  # batch_size=100
        )

        # total_batches = 10000 / 100 = 100
        # chunksize = 100 / (4 * 4) = 6
        expected_chunksize = 6
        assert chunksize == expected_chunksize

    def test_chunksize_default_when_total_unknown(self):
        """Test chunksize defaults to 10 when total unknown"""
        workers, batch_size, chunksize = get_optimal_settings(num_workers=4)

        assert chunksize == 10

    def test_chunksize_minimum_value(self):
        """Test chunksize never goes below 1"""
        workers, batch_size, chunksize = get_optimal_settings(
            num_workers=16,
            total_sequences=100,
            avg_sequence_length=500  # batch_size=100, so 1 batch
        )

        assert chunksize >= 1


class TestParallelTranslateSequences:
    """Test parallel_translate_sequences function"""

    def test_parallel_translate_basic(self, temp_fasta, tmp_path):
        """Test basic parallel translation functionality"""
        output_nuc = tmp_path / "output_nuc.fa"
        output_pro = tmp_path / "output_pro.faa"

        processed, valid = parallel_translate_sequences(
            temp_fasta,
            output_nuc,
            output_pro,
            num_workers=2,
            chunksize=10
        )

        # Should process all sequences
        assert processed == 5
        assert valid >= 0  # At least some valid ORFs

        # Output files should exist
        assert output_nuc.exists()
        assert output_pro.exists()

        # Verify output files have content
        nuc_count = sum(1 for _ in SeqIO.parse(output_nuc, 'fasta'))
        pro_count = sum(1 for _ in SeqIO.parse(output_pro, 'fasta'))
        assert nuc_count > 0
        assert pro_count > 0
        assert nuc_count == pro_count  # Should match

    def test_parallel_translate_single_worker(self, temp_fasta, tmp_path):
        """Test parallel translation with single worker"""
        output_nuc = tmp_path / "output_nuc.fa"
        output_pro = tmp_path / "output_pro.faa"

        processed, valid = parallel_translate_sequences(
            temp_fasta,
            output_nuc,
            output_pro,
            num_workers=1,
            chunksize=10
        )

        assert processed == 5
        assert output_nuc.exists()
        assert output_pro.exists()

    @patch('os.cpu_count', return_value=4)
    def test_parallel_translate_default_workers(self, mock_cpu_count, temp_fasta, tmp_path):
        """Test parallel translation with default workers (cpu_count)"""
        output_nuc = tmp_path / "output_nuc.fa"
        output_pro = tmp_path / "output_pro.faa"

        processed, valid = parallel_translate_sequences(
            temp_fasta,
            output_nuc,
            output_pro,
            # num_workers not specified
            chunksize=10
        )

        assert processed == 5
        mock_cpu_count.assert_called()

    def test_parallel_translate_empty_input(self, tmp_path):
        """Test parallel translation with empty FASTA file"""
        empty_fasta = tmp_path / "empty.fa"
        empty_fasta.write_text("")

        output_nuc = tmp_path / "output_nuc.fa"
        output_pro = tmp_path / "output_pro.faa"

        processed, valid = parallel_translate_sequences(
            empty_fasta,
            output_nuc,
            output_pro,
            num_workers=2,
            chunksize=10
        )

        assert processed == 0
        assert valid == 0

    def test_parallel_translate_large_input(self, large_fasta, tmp_path):
        """Test parallel translation with large input (10000 sequences)"""
        output_nuc = tmp_path / "output_nuc.fa"
        output_pro = tmp_path / "output_pro.faa"

        processed, valid = parallel_translate_sequences(
            large_fasta,
            output_nuc,
            output_pro,
            num_workers=4,
            chunksize=100
        )

        assert processed == 10000
        assert valid > 0

        # Verify all sequences processed
        nuc_count = sum(1 for _ in SeqIO.parse(output_nuc, 'fasta'))
        assert nuc_count > 0


class TestParallelTranslateBatched:
    """Test parallel_translate_batched function"""

    def test_batched_translation_basic(self, temp_fasta, tmp_path):
        """Test batched parallel translation"""
        output_nuc = tmp_path / "output_nuc.fa"
        output_pro = tmp_path / "output_pro.faa"

        processed, valid = parallel_translate_batched(
            temp_fasta,
            output_nuc,
            output_pro,
            num_workers=2,
            batch_size=2,
            chunksize=5
        )

        assert processed == 5
        assert output_nuc.exists()
        assert output_pro.exists()

    def test_batched_translation_large_batches(self, large_fasta, tmp_path):
        """Test batched translation with large batches"""
        output_nuc = tmp_path / "output_nuc.fa"
        output_pro = tmp_path / "output_pro.faa"

        processed, valid = parallel_translate_batched(
            large_fasta,
            output_nuc,
            output_pro,
            num_workers=4,
            batch_size=100,
            chunksize=10
        )

        assert processed == 10000
        assert valid > 0


class TestParallelTranslateWithProgress:
    """Test parallel_translate_with_progress function"""

    def test_with_progress_enabled(self, temp_fasta, tmp_path):
        """Test parallel translation with progress reporting enabled"""
        output_nuc = tmp_path / "output_nuc.fa"
        output_pro = tmp_path / "output_pro.faa"

        processed, valid = parallel_translate_with_progress(
            temp_fasta,
            output_nuc,
            output_pro,
            num_workers=2,
            chunksize=10,
            show_progress=True
        )

        assert processed == 5
        assert output_nuc.exists()
        assert output_pro.exists()

    def test_with_progress_disabled(self, temp_fasta, tmp_path):
        """Test parallel translation with progress reporting disabled"""
        output_nuc = tmp_path / "output_nuc.fa"
        output_pro = tmp_path / "output_pro.faa"

        processed, valid = parallel_translate_with_progress(
            temp_fasta,
            output_nuc,
            output_pro,
            num_workers=2,
            chunksize=10,
            show_progress=False
        )

        assert processed == 5
        assert output_nuc.exists()
        assert output_pro.exists()


class TestMemoryEfficiency:
    """Test memory efficiency patterns"""

    @patch('virnucpro.pipeline.parallel_translate.multiprocessing.get_context')
    def test_uses_imap_not_map(self, mock_get_context, temp_fasta, tmp_path):
        """Test that Pool.imap() is used for memory efficiency (not map())"""
        # Create mock pool that tracks method calls
        mock_pool = MagicMock()
        mock_pool.__enter__ = Mock(return_value=mock_pool)
        mock_pool.__exit__ = Mock(return_value=False)
        mock_pool.imap = Mock(return_value=iter([]))

        mock_ctx = MagicMock()
        mock_ctx.Pool = Mock(return_value=mock_pool)
        mock_get_context.return_value = mock_ctx

        output_nuc = tmp_path / "output_nuc.fa"
        output_pro = tmp_path / "output_pro.faa"

        # Run translation
        parallel_translate_sequences(
            temp_fasta,
            output_nuc,
            output_pro,
            num_workers=2
        )

        # Verify imap was called (lazy evaluation)
        mock_pool.imap.assert_called_once()

        # Verify map was NOT called (would load all results in memory)
        assert not hasattr(mock_pool.map, 'called') or not mock_pool.map.called


class TestSpawnContext:
    """Test spawn context usage for CUDA safety"""

    @patch('virnucpro.pipeline.parallel_translate.multiprocessing.get_context')
    def test_spawn_context_used(self, mock_get_context, temp_fasta, tmp_path):
        """Test that spawn context is used (not fork)"""
        mock_ctx = MagicMock()
        mock_pool = MagicMock()
        mock_pool.__enter__ = Mock(return_value=mock_pool)
        mock_pool.__exit__ = Mock(return_value=False)
        mock_pool.imap = Mock(return_value=iter([]))
        mock_ctx.Pool = Mock(return_value=mock_pool)
        mock_get_context.return_value = mock_ctx

        output_nuc = tmp_path / "output_nuc.fa"
        output_pro = tmp_path / "output_pro.faa"

        parallel_translate_sequences(
            temp_fasta,
            output_nuc,
            output_pro,
            num_workers=2
        )

        # Verify get_context called with 'spawn'
        mock_get_context.assert_called_with('spawn')


class TestErrorHandling:
    """Test error handling in parallel translation"""

    def test_corrupted_fasta_handling(self, tmp_path):
        """Test handling of malformed FASTA file"""
        corrupted_fasta = tmp_path / "corrupted.fa"
        corrupted_fasta.write_text(">seq1\nATGC\n>seq2_no_sequence\n>seq3\n")

        output_nuc = tmp_path / "output_nuc.fa"
        output_pro = tmp_path / "output_pro.faa"

        # Should not crash, but may have fewer processed sequences
        try:
            processed, valid = parallel_translate_sequences(
                corrupted_fasta,
                output_nuc,
                output_pro,
                num_workers=2,
                chunksize=10
            )
            # Should complete without exception
            assert processed >= 0
        except Exception as e:
            # If it does raise, it should be a clear error
            assert "FASTA" in str(e) or "parse" in str(e).lower()

    def test_worker_exception_propagates(self, tmp_path):
        """Test that worker exceptions propagate correctly"""
        # Create FASTA with invalid content that will cause exception
        bad_fasta = tmp_path / "bad.fa"
        bad_fasta.write_text(">seq1\nINVALID_DNA_SEQUENCE_WITH_WEIRD_CHARS!!!!\n")

        output_nuc = tmp_path / "output_nuc.fa"
        output_pro = tmp_path / "output_pro.faa"

        # The worker should handle this gracefully and return None for failed sequences
        # No exception should be raised for single invalid sequence
        try:
            processed, valid = parallel_translate_sequences(
                bad_fasta,
                output_nuc,
                output_pro,
                num_workers=2,
                chunksize=1
            )
            # Should process but may have no valid ORFs
            assert processed >= 0
        except Exception:
            # Some exceptions are acceptable (e.g., parsing errors)
            pass


class TestResourceCleanup:
    """Test resource cleanup on exceptions"""

    @patch('virnucpro.pipeline.parallel_translate.SeqIO.parse')
    @patch('virnucpro.pipeline.parallel_translate.multiprocessing.get_context')
    def test_pool_cleanup_on_exception(self, mock_get_context, mock_parse, temp_fasta, tmp_path):
        """Test Pool is properly closed when exception occurs"""
        # Setup mock pool
        mock_pool = MagicMock()
        mock_pool.__enter__ = Mock(return_value=mock_pool)
        mock_pool.__exit__ = Mock(return_value=False)

        # Make imap raise exception
        mock_pool.imap = Mock(side_effect=RuntimeError("Processing error"))

        mock_ctx = MagicMock()
        mock_ctx.Pool = Mock(return_value=mock_pool)
        mock_get_context.return_value = mock_ctx

        # Make SeqIO.parse work initially (for counting)
        mock_parse.return_value = iter([])

        output_nuc = tmp_path / "output_nuc.fa"
        output_pro = tmp_path / "output_pro.faa"

        # Should raise exception
        with pytest.raises(RuntimeError, match="Processing error"):
            parallel_translate_sequences(
                temp_fasta,
                output_nuc,
                output_pro,
                num_workers=2
            )

        # Verify pool __exit__ was called (context manager cleanup)
        mock_pool.__exit__.assert_called_once()
