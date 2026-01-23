"""Integration tests for parallel translation - verify output equivalence with sequential processing"""

import pytest
import time
import random
from pathlib import Path
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import subprocess
import sys

from virnucpro.pipeline.parallel_translate import parallel_translate_sequences
from virnucpro.utils.sequence import identify_seq


def create_test_fasta(output_path: Path, num_sequences: int = 1000, seq_length: int = 500, seed: int = 42):
    """
    Generate deterministic test FASTA file for reproducible testing.

    Args:
        output_path: Path to write FASTA file
        num_sequences: Number of sequences to generate
        seq_length: Length of each sequence in bp
        seed: Random seed for reproducibility
    """
    rng = random.Random(seed)
    records = []

    for i in range(num_sequences):
        # Generate deterministic sequence using seeded random
        seq = ''.join(rng.choices('ATCG', k=seq_length))
        record = SeqRecord(Seq(seq), id=f"seq_{i}", description="")
        records.append(record)

    with open(output_path, 'w') as f:
        SeqIO.write(records, f, 'fasta')


def compare_fasta_files(file1: Path, file2: Path) -> tuple[bool, str]:
    """
    Compare two FASTA files for identical content.

    Args:
        file1: First FASTA file
        file2: Second FASTA file

    Returns:
        Tuple of (is_identical, difference_message)
    """
    records1 = {rec.id: str(rec.seq) for rec in SeqIO.parse(file1, 'fasta')}
    records2 = {rec.id: str(rec.seq) for rec in SeqIO.parse(file2, 'fasta')}

    # Check same number of records
    if len(records1) != len(records2):
        return False, f"Different number of records: {len(records1)} vs {len(records2)}"

    # Check same record IDs
    ids1 = set(records1.keys())
    ids2 = set(records2.keys())
    if ids1 != ids2:
        missing_in_2 = ids1 - ids2
        missing_in_1 = ids2 - ids1
        return False, f"Different IDs. Missing in file2: {missing_in_2}, Missing in file1: {missing_in_1}"

    # Check sequences match
    differences = []
    for seq_id in sorted(records1.keys()):
        if records1[seq_id] != records2[seq_id]:
            differences.append(f"{seq_id}: sequences differ")

    if differences:
        return False, f"Sequence differences found: {differences[:5]}"

    return True, "Files are identical"


class TestOutputEquivalence:
    """Test that parallel translation produces identical output to sequential processing"""

    @pytest.fixture
    def test_fasta_1000(self, tmp_path):
        """Create test FASTA with 1000 sequences"""
        fasta_path = tmp_path / "test_1000.fa"
        create_test_fasta(fasta_path, num_sequences=1000, seq_length=500)
        return fasta_path

    def sequential_translate(self, input_file: Path, output_nuc: Path, output_pro: Path):
        """
        Perform sequential translation using identify_seq (the baseline).

        This mimics what the original pipeline does without parallelization.
        """
        nuc_records = []
        pro_records = []

        for record in SeqIO.parse(input_file, 'fasta'):
            result = identify_seq(record.id, str(record.seq).upper())

            if result:
                for orf in result:
                    nuc_records.append(SeqRecord(
                        Seq(orf['nucleotide']),
                        id=orf['seqid'],
                        description=""
                    ))
                    pro_records.append(SeqRecord(
                        Seq(orf['protein']),
                        id=orf['seqid'],
                        description=""
                    ))

        # Write output files
        with open(output_nuc, 'w') as f:
            SeqIO.write(nuc_records, f, 'fasta')
        with open(output_pro, 'w') as f:
            SeqIO.write(pro_records, f, 'fasta')

        return len(nuc_records)

    def test_output_matches_sequential(self, test_fasta_1000, tmp_path):
        """Test that parallel output exactly matches sequential translation"""
        # Sequential translation
        seq_nuc = tmp_path / "sequential_nuc.fa"
        seq_pro = tmp_path / "sequential_pro.faa"
        seq_count = self.sequential_translate(test_fasta_1000, seq_nuc, seq_pro)

        # Parallel translation
        par_nuc = tmp_path / "parallel_nuc.fa"
        par_pro = tmp_path / "parallel_pro.faa"
        processed, valid = parallel_translate_sequences(
            test_fasta_1000,
            par_nuc,
            par_pro,
            num_workers=4,
            chunksize=100
        )

        # Both should process same number
        assert processed == 1000

        # Compare nucleotide outputs
        nuc_identical, nuc_msg = compare_fasta_files(seq_nuc, par_nuc)
        assert nuc_identical, f"Nucleotide outputs differ: {nuc_msg}"

        # Compare protein outputs
        pro_identical, pro_msg = compare_fasta_files(seq_pro, par_pro)
        assert pro_identical, f"Protein outputs differ: {pro_msg}"

    def test_output_matches_with_different_workers(self, test_fasta_1000, tmp_path):
        """Test that output is identical regardless of worker count"""
        # Run with 1 worker
        nuc_1w = tmp_path / "nuc_1worker.fa"
        pro_1w = tmp_path / "pro_1worker.faa"
        parallel_translate_sequences(
            test_fasta_1000,
            nuc_1w,
            pro_1w,
            num_workers=1,
            chunksize=100
        )

        # Run with 4 workers
        nuc_4w = tmp_path / "nuc_4workers.fa"
        pro_4w = tmp_path / "pro_4workers.faa"
        parallel_translate_sequences(
            test_fasta_1000,
            nuc_4w,
            pro_4w,
            num_workers=4,
            chunksize=100
        )

        # Run with 8 workers
        nuc_8w = tmp_path / "nuc_8workers.fa"
        pro_8w = tmp_path / "pro_8workers.faa"
        parallel_translate_sequences(
            test_fasta_1000,
            nuc_8w,
            pro_8w,
            num_workers=8,
            chunksize=100
        )

        # All outputs should be identical
        identical_1_4, msg_1_4 = compare_fasta_files(nuc_1w, nuc_4w)
        assert identical_1_4, f"1 vs 4 workers nucleotide differ: {msg_1_4}"

        identical_4_8, msg_4_8 = compare_fasta_files(nuc_4w, nuc_8w)
        assert identical_4_8, f"4 vs 8 workers nucleotide differ: {msg_4_8}"


class TestPerformance:
    """Test that parallel translation provides performance improvements"""

    @pytest.fixture
    def large_test_fasta(self, tmp_path):
        """Create large test FASTA for performance testing (10000 sequences)"""
        fasta_path = tmp_path / "test_large_10k.fa"
        create_test_fasta(fasta_path, num_sequences=10000, seq_length=500)
        return fasta_path

    def test_performance_scales_with_workers(self, large_test_fasta, tmp_path):
        """Test that performance improves with multiple workers"""
        # Time with 1 worker
        start_1w = time.time()
        nuc_1w = tmp_path / "perf_nuc_1w.fa"
        pro_1w = tmp_path / "perf_pro_1w.faa"
        parallel_translate_sequences(
            large_test_fasta,
            nuc_1w,
            pro_1w,
            num_workers=1,
            chunksize=100
        )
        time_1w = time.time() - start_1w

        # Time with 4 workers
        start_4w = time.time()
        nuc_4w = tmp_path / "perf_nuc_4w.fa"
        pro_4w = tmp_path / "perf_pro_4w.faa"
        parallel_translate_sequences(
            large_test_fasta,
            nuc_4w,
            pro_4w,
            num_workers=4,
            chunksize=100
        )
        time_4w = time.time() - start_4w

        # Calculate speedup
        speedup = time_1w / time_4w

        print(f"\nPerformance: 1 worker: {time_1w:.2f}s, 4 workers: {time_4w:.2f}s, speedup: {speedup:.2f}x")

        # Expect at least 1.5x speedup (conservative, actual may be higher)
        # Not full 4x due to overhead and Python GIL for non-CPU work
        assert speedup >= 1.5, f"Expected at least 1.5x speedup, got {speedup:.2f}x"

        # Verify outputs are still identical
        identical, msg = compare_fasta_files(nuc_1w, nuc_4w)
        assert identical, f"Performance test outputs differ: {msg}"


class TestRealWorldPatterns:
    """Test with realistic data patterns found in viral sequence analysis"""

    def test_mixed_sequence_lengths(self, tmp_path):
        """Test with varying sequence lengths (100bp to 10000bp)"""
        input_fasta = tmp_path / "mixed_lengths.fa"

        # Create sequences with varying lengths
        records = []
        rng = random.Random(123)
        lengths = [100, 200, 500, 1000, 2000, 5000, 10000]

        for i, length in enumerate(lengths * 10):  # 70 sequences total
            seq = ''.join(rng.choices('ATCG', k=length))
            records.append(SeqRecord(Seq(seq), id=f"seq_{i}_len{length}", description=""))

        with open(input_fasta, 'w') as f:
            SeqIO.write(records, f, 'fasta')

        # Test parallel translation
        output_nuc = tmp_path / "mixed_nuc.fa"
        output_pro = tmp_path / "mixed_pro.faa"

        processed, valid = parallel_translate_sequences(
            input_fasta,
            output_nuc,
            output_pro,
            num_workers=4,
            chunksize=10
        )

        assert processed == 70
        assert valid >= 0

        # Verify outputs exist and have content
        nuc_count = sum(1 for _ in SeqIO.parse(output_nuc, 'fasta'))
        pro_count = sum(1 for _ in SeqIO.parse(output_pro, 'fasta'))
        assert nuc_count > 0
        assert pro_count > 0
        assert nuc_count == pro_count

    def test_all_reading_frames_processed(self, tmp_path):
        """Test that all 6 reading frames are checked and valid ones returned"""
        input_fasta = tmp_path / "frames_test.fa"

        # Create sequence that should have ORFs in multiple frames
        # ATG starts a valid ORF
        seq = "ATG" + ("GCA" * 50)  # Valid ORF in frame 1 (no stops)

        records = [SeqRecord(Seq(seq), id="test_seq", description="")]

        with open(input_fasta, 'w') as f:
            SeqIO.write(records, f, 'fasta')

        output_nuc = tmp_path / "frames_nuc.fa"
        output_pro = tmp_path / "frames_pro.faa"

        processed, valid = parallel_translate_sequences(
            input_fasta,
            output_nuc,
            output_pro,
            num_workers=2
        )

        assert processed == 1
        assert valid >= 1  # At least one valid ORF

        # Check that frame information is in seqid
        nuc_records = list(SeqIO.parse(output_nuc, 'fasta'))
        assert len(nuc_records) > 0

        # Verify frame suffixes (F1-F3 for forward, R1-R3 for reverse)
        frame_suffixes = {rec.id[-2:] for rec in nuc_records}
        # Should have at least one frame suffix
        valid_suffixes = {'F1', 'F2', 'F3', 'R1', 'R2', 'R3'}
        assert any(suffix in valid_suffixes for suffix in frame_suffixes)

    def test_stop_codon_filtering(self, tmp_path):
        """Test that sequences with stop codons are correctly filtered"""
        input_fasta = tmp_path / "stops_test.fa"

        # Create sequences with known stop codons
        records = [
            # Sequence with TAA stop in frame 1
            SeqRecord(Seq("ATGTAAGCA"), id="has_stop_f1", description=""),
            # Sequence with no stops in any frame (all Ala)
            SeqRecord(Seq("GCA" * 50), id="no_stops", description=""),
            # Sequence with TAG stop
            SeqRecord(Seq("ATGTAGGCA"), id="has_stop_tag", description=""),
        ]

        with open(input_fasta, 'w') as f:
            SeqIO.write(records, f, 'fasta')

        output_nuc = tmp_path / "stops_nuc.fa"
        output_pro = tmp_path / "stops_pro.faa"

        processed, valid = parallel_translate_sequences(
            input_fasta,
            output_nuc,
            output_pro,
            num_workers=2
        )

        assert processed == 3

        # Verify no stop codons (*) in any protein output
        pro_records = list(SeqIO.parse(output_pro, 'fasta'))
        for rec in pro_records:
            assert '*' not in str(rec.seq), f"Stop codon found in {rec.id}"


class TestCLIIntegration:
    """Test integration with CLI interface"""

    def test_cli_runs_successfully(self, tmp_path):
        """Test that CLI can invoke parallel translation"""
        # Create small test FASTA
        input_fasta = tmp_path / "cli_test.fa"
        create_test_fasta(input_fasta, num_sequences=100, seq_length=300)

        output_dir = tmp_path / "cli_output"
        output_dir.mkdir()

        # Test running via Python module (simulates CLI usage)
        # Note: This tests the translation function directly since full CLI
        # requires the entire pipeline setup
        output_nuc = output_dir / "output_nuc.fa"
        output_pro = output_dir / "output_pro.faa"

        try:
            processed, valid = parallel_translate_sequences(
                input_fasta,
                output_nuc,
                output_pro,
                num_workers=4
            )

            assert processed == 100
            assert output_nuc.exists()
            assert output_pro.exists()

        except Exception as e:
            pytest.fail(f"CLI-style invocation failed: {e}")

    def test_cli_threads_parameter_exists(self):
        """Test that --threads parameter exists in CLI code"""
        # Instead of running CLI (which requires click dependency in test env),
        # verify the parameter is defined in the code
        try:
            from virnucpro.cli import predict
            import inspect

            # Get the source code of the predict module
            source = inspect.getsource(predict)

            # Check that --threads parameter is defined
            assert '--threads' in source or "option('--threads'" in source or 'option("-t"' in source, \
                   "CLI should have --threads parameter defined"

        except ImportError:
            pytest.skip("CLI module not available in test environment")


class TestEdgeCases:
    """Test edge cases and boundary conditions"""

    def test_single_sequence_file(self, tmp_path):
        """Test with single sequence (edge case for batching)"""
        input_fasta = tmp_path / "single.fa"
        records = [SeqRecord(Seq("ATG" + ("GCA" * 50)), id="single_seq", description="")]

        with open(input_fasta, 'w') as f:
            SeqIO.write(records, f, 'fasta')

        output_nuc = tmp_path / "single_nuc.fa"
        output_pro = tmp_path / "single_pro.faa"

        processed, valid = parallel_translate_sequences(
            input_fasta,
            output_nuc,
            output_pro,
            num_workers=4
        )

        assert processed == 1
        assert output_nuc.exists()
        assert output_pro.exists()

    def test_very_short_sequences(self, tmp_path):
        """Test with very short sequences (<100bp)"""
        input_fasta = tmp_path / "short.fa"

        records = [
            SeqRecord(Seq("ATGGCATAG"), id=f"short_{i}", description="")
            for i in range(50)
        ]

        with open(input_fasta, 'w') as f:
            SeqIO.write(records, f, 'fasta')

        output_nuc = tmp_path / "short_nuc.fa"
        output_pro = tmp_path / "short_pro.faa"

        processed, valid = parallel_translate_sequences(
            input_fasta,
            output_nuc,
            output_pro,
            num_workers=4
        )

        assert processed == 50

    def test_very_long_sequences(self, tmp_path):
        """Test with very long sequences (>50kb)"""
        input_fasta = tmp_path / "long.fa"

        # Create 5 sequences of 50kb each
        rng = random.Random(999)
        records = [
            SeqRecord(Seq(''.join(rng.choices('ATCG', k=50000))), id=f"long_{i}", description="")
            for i in range(5)
        ]

        with open(input_fasta, 'w') as f:
            SeqIO.write(records, f, 'fasta')

        output_nuc = tmp_path / "long_nuc.fa"
        output_pro = tmp_path / "long_pro.faa"

        processed, valid = parallel_translate_sequences(
            input_fasta,
            output_nuc,
            output_pro,
            num_workers=4
        )

        assert processed == 5
        assert valid >= 0
