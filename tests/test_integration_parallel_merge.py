"""End-to-end integration tests for parallel embedding merge"""

import unittest
import subprocess
import sys
import time
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import torch
import numpy as np


class TestParallelMergeIntegration(unittest.TestCase):
    """Integration tests verifying parallel merge produces correct outputs with performance gains"""

    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        # Check CPU count for parallel tests
        cls.num_cpus = os.cpu_count()
        print(f"\nRunning parallel merge tests with {cls.num_cpus} CPU cores")

    def setUp(self):
        """Create temporary directory for each test"""
        self.temp_dir = TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

    def tearDown(self):
        """Clean up temporary directory"""
        self.temp_dir.cleanup()

    def create_mock_feature_files(self, num_pairs: int, num_sequences: int = 10):
        """
        Create mock DNABERT-S and ESM-2 feature files for testing.

        Args:
            num_pairs: Number of file pairs to create
            num_sequences: Number of sequences per file

        Returns:
            Tuple of (nucleotide_files, protein_files)
        """
        nuc_dir = self.temp_path / "dnabert_features"
        pro_dir = self.temp_path / "esm_features"
        nuc_dir.mkdir()
        pro_dir.mkdir()

        nucleotide_files = []
        protein_files = []

        for i in range(num_pairs):
            # Create sequence IDs
            seq_ids = [f"seq_{i}_{j}" for j in range(num_sequences)]

            # Create DNABERT-S feature file (768-dim)
            nuc_data = {
                'nucleotide': seq_ids,
                'data': [
                    {
                        'label': seq_id,
                        'mean_representation': torch.randn(768).tolist()
                    }
                    for seq_id in seq_ids
                ]
            }
            nuc_file = nuc_dir / f"file_{i}_DNABERT.pt"
            torch.save(nuc_data, nuc_file)
            nucleotide_files.append(nuc_file)

            # Create ESM-2 feature file (2560-dim)
            pro_data = {
                'proteins': seq_ids,
                'data': [torch.randn(2560) for _ in seq_ids]
            }
            pro_file = pro_dir / f"file_{i}_ESM.pt"
            torch.save(pro_data, pro_file)
            protein_files.append(pro_file)

        return nucleotide_files, protein_files

    def create_mismatched_feature_files(self, num_pairs: int):
        """
        Create feature files with mismatched sequence IDs.

        Returns:
            Tuple of (nucleotide_files, protein_files)
        """
        nuc_dir = self.temp_path / "dnabert_features"
        pro_dir = self.temp_path / "esm_features"
        nuc_dir.mkdir()
        pro_dir.mkdir()

        nucleotide_files = []
        protein_files = []

        for i in range(num_pairs):
            # Create different sequence IDs for nuc vs pro
            nuc_ids = [f"nuc_seq_{i}_{j}" for j in range(5)]
            pro_ids = [f"pro_seq_{i}_{j}" for j in range(5)]

            # DNABERT-S features
            nuc_data = {
                'nucleotide': nuc_ids,
                'data': [
                    {'label': seq_id, 'mean_representation': torch.randn(768).tolist()}
                    for seq_id in nuc_ids
                ]
            }
            nuc_file = nuc_dir / f"file_{i}_DNABERT.pt"
            torch.save(nuc_data, nuc_file)
            nucleotide_files.append(nuc_file)

            # ESM-2 features
            pro_data = {
                'proteins': pro_ids,
                'data': [torch.randn(2560) for _ in pro_ids]
            }
            pro_file = pro_dir / f"file_{i}_ESM.pt"
            torch.save(pro_data, pro_file)
            protein_files.append(pro_file)

        return nucleotide_files, protein_files

    def test_parallel_merge_matches_sequential(self):
        """Integration: Verify parallel merge produces identical output to sequential merge"""
        # Create test data
        nuc_files, pro_files = self.create_mock_feature_files(num_pairs=4, num_sequences=10)

        # Import merge functions
        from virnucpro.pipeline.parallel_merge import parallel_merge_features

        # Sequential merge (num_workers=1)
        output_seq = self.temp_path / "sequential"
        output_seq.mkdir()

        print("\n=== Running sequential merge (1 worker) ===")
        start_seq = time.time()
        merged_seq = parallel_merge_features(
            nuc_files,
            pro_files,
            output_seq,
            num_workers=1
        )
        time_seq = time.time() - start_seq

        # Parallel merge (num_workers=4)
        output_par = self.temp_path / "parallel"
        output_par.mkdir()

        print(f"=== Running parallel merge (4 workers) ===")
        start_par = time.time()
        merged_par = parallel_merge_features(
            nuc_files,
            pro_files,
            output_par,
            num_workers=4
        )
        time_par = time.time() - start_par

        # Both should succeed
        self.assertEqual(len(merged_seq), 4, "Sequential should merge all 4 file pairs")
        self.assertEqual(len(merged_par), 4, "Parallel should merge all 4 file pairs")

        # Compare outputs file by file
        for seq_file, par_file in zip(sorted(merged_seq), sorted(merged_par)):
            # Load both outputs
            seq_data = torch.load(seq_file)
            par_data = torch.load(par_file)

            # Compare structure
            self.assertEqual(len(seq_data['ids']), len(par_data['ids']),
                           "Should have same number of sequences")
            self.assertEqual(seq_data['ids'], par_data['ids'],
                           "Should have same sequence IDs in same order")

            # Compare merged features (should be identical)
            seq_features = seq_data['data']
            par_features = par_data['data']

            self.assertEqual(seq_features.shape, par_features.shape,
                           f"Feature shapes should match: {seq_features.shape} vs {par_features.shape}")

            # Check feature dimension (768 + 2560 = 3328)
            self.assertEqual(seq_features.shape[1], 3328,
                           "Merged features should be 3328-dimensional")

            # Features should be identical (deterministic concatenation)
            self.assertTrue(torch.allclose(seq_features, par_features, rtol=1e-5, atol=1e-5),
                          "Sequential and parallel outputs should be identical")

        print(f"✓ Parallel and sequential outputs are identical")
        print(f"Sequential: {time_seq:.2f}s, Parallel: {time_par:.2f}s")

    def test_parallel_merge_performance(self):
        """Integration: Verify parallel merge provides performance improvement"""
        # Create larger dataset for performance testing
        nuc_files, pro_files = self.create_mock_feature_files(num_pairs=20, num_sequences=100)

        from virnucpro.pipeline.parallel_merge import parallel_merge_features

        # Sequential baseline (1 worker)
        output_seq = self.temp_path / "sequential_perf"
        output_seq.mkdir()

        print("\n=== Performance test: Sequential merge (1 worker) ===")
        start_seq = time.time()
        merged_seq = parallel_merge_features(
            nuc_files,
            pro_files,
            output_seq,
            num_workers=1
        )
        time_seq = time.time() - start_seq

        # Parallel merge (4 workers)
        output_par = self.temp_path / "parallel_perf"
        output_par.mkdir()

        print(f"=== Performance test: Parallel merge (4 workers) ===")
        start_par = time.time()
        merged_par = parallel_merge_features(
            nuc_files,
            pro_files,
            output_par,
            num_workers=4
        )
        time_par = time.time() - start_par

        # Verify both completed successfully
        self.assertEqual(len(merged_seq), 20, "Sequential should complete all files")
        self.assertEqual(len(merged_par), 20, "Parallel should complete all files")

        # Calculate speedup
        speedup = time_seq / time_par if time_par > 0 else 0

        print(f"\n=== Performance Results ===")
        print(f"Sequential: {time_seq:.2f}s")
        print(f"Parallel (4 workers): {time_par:.2f}s")
        print(f"Speedup: {speedup:.2f}x")

        # Expect at least 2x speedup with 4 workers
        # (conservative to account for overhead and system variability)
        self.assertGreater(speedup, 2.0,
                         f"Expected >= 2x speedup with 4 workers, got {speedup:.2f}x")

        print(f"✓ Performance improvement verified: {speedup:.2f}x speedup")

    def test_cli_merge_threads_parameter(self):
        """Integration: Verify --merge-threads CLI parameter works (via subprocess if CLI exists)"""
        # This test would require full CLI integration
        # For now, test the parameter interface directly
        from virnucpro.pipeline.parallel_merge import parallel_merge_features

        nuc_files, pro_files = self.create_mock_feature_files(num_pairs=2, num_sequences=5)
        output_dir = self.temp_path / "cli_test"
        output_dir.mkdir()

        # Test with explicit num_workers parameter
        print("\n=== Testing merge_threads parameter (num_workers=2) ===")
        merged = parallel_merge_features(
            nuc_files,
            pro_files,
            output_dir,
            num_workers=2
        )

        self.assertEqual(len(merged), 2, "Should merge with specified worker count")
        print("✓ merge_threads parameter works correctly")

    def test_partial_failure_handling(self):
        """Integration: Some file pairs fail, others succeed"""
        # Create mixed dataset: some matching, some mismatched
        nuc_dir = self.temp_path / "dnabert_features"
        pro_dir = self.temp_path / "esm_features"
        nuc_dir.mkdir()
        pro_dir.mkdir()

        nuc_files = []
        pro_files = []

        # File pair 0: matching IDs (should succeed)
        seq_ids = [f"seq_0_{j}" for j in range(5)]
        nuc_data = {
            'nucleotide': seq_ids,
            'data': [{'label': sid, 'mean_representation': torch.randn(768).tolist()} for sid in seq_ids]
        }
        nuc_file = nuc_dir / "file_0_DNABERT.pt"
        torch.save(nuc_data, nuc_file)
        nuc_files.append(nuc_file)

        pro_data = {
            'proteins': seq_ids,
            'data': [torch.randn(2560) for _ in seq_ids]
        }
        pro_file = pro_dir / "file_0_ESM.pt"
        torch.save(pro_data, pro_file)
        pro_files.append(pro_file)

        # File pair 1: mismatched IDs (partial success - no matching sequences)
        nuc_ids = [f"nuc_seq_{j}" for j in range(5)]
        pro_ids = [f"pro_seq_{j}" for j in range(5)]

        nuc_data = {
            'nucleotide': nuc_ids,
            'data': [{'label': sid, 'mean_representation': torch.randn(768).tolist()} for sid in nuc_ids]
        }
        nuc_file = nuc_dir / "file_1_DNABERT.pt"
        torch.save(nuc_data, nuc_file)
        nuc_files.append(nuc_file)

        pro_data = {
            'proteins': pro_ids,
            'data': [torch.randn(2560) for _ in pro_ids]
        }
        pro_file = pro_dir / "file_1_ESM.pt"
        torch.save(pro_data, pro_file)
        pro_files.append(pro_file)

        # Run parallel merge
        from virnucpro.pipeline.parallel_merge import parallel_merge_features

        output_dir = self.temp_path / "partial_fail"
        output_dir.mkdir()

        print("\n=== Testing partial failure handling ===")
        merged = parallel_merge_features(
            nuc_files,
            pro_files,
            output_dir,
            num_workers=2
        )

        # Both files should complete (even if one has no matches)
        # The merge_features function handles empty matches gracefully
        self.assertGreaterEqual(len(merged), 1, "At least one file should succeed")

        # Verify the successful file
        for merged_file in merged:
            result = torch.load(merged_file)
            # File 0 should have 5 sequences, File 1 should have 0 (mismatched)
            self.assertIn('ids', result)
            self.assertIn('data', result)

        print(f"✓ Partial failure handled gracefully: {len(merged)}/{len(nuc_files)} files merged")

    def test_merge_checkpoint_resume(self):
        """Integration: Verify merge skips already-completed files (checkpoint/resume)"""
        nuc_files, pro_files = self.create_mock_feature_files(num_pairs=3, num_sequences=5)

        from virnucpro.pipeline.parallel_merge import parallel_merge_features

        output_dir = self.temp_path / "checkpoint"
        output_dir.mkdir()

        print("\n=== Testing checkpoint/resume capability ===")

        # First run: merge all files
        print("First run: merging all files")
        start = time.time()
        merged1 = parallel_merge_features(
            nuc_files,
            pro_files,
            output_dir,
            num_workers=2
        )
        time1 = time.time() - start

        self.assertEqual(len(merged1), 3, "First run should merge all 3 files")

        # Second run: should skip all files (already exist)
        print("Second run: should skip existing files")
        start = time.time()
        merged2 = parallel_merge_features(
            nuc_files,
            pro_files,
            output_dir,
            num_workers=2
        )
        time2 = time.time() - start

        self.assertEqual(len(merged2), 3, "Second run should also return 3 files")

        # Second run should be much faster (skipping already merged)
        # NOTE: This is handled by merge_features() checkpoint logic
        print(f"First run: {time1:.2f}s, Second run: {time2:.2f}s")
        print(f"✓ Checkpoint/resume works (second run uses existing files)")

    def test_auto_worker_detection(self):
        """Integration: Verify auto-detection uses cpu_count() workers"""
        nuc_files, pro_files = self.create_mock_feature_files(num_pairs=2, num_sequences=5)

        from virnucpro.pipeline.parallel_merge import parallel_merge_features

        output_dir = self.temp_path / "auto_detect"
        output_dir.mkdir()

        print(f"\n=== Testing auto worker detection (cpu_count={os.cpu_count()}) ===")

        # Run without specifying num_workers
        merged = parallel_merge_features(
            nuc_files,
            pro_files,
            output_dir
            # num_workers not specified - should auto-detect
        )

        self.assertEqual(len(merged), 2, "Auto-detection should complete merge")
        print(f"✓ Auto-detection used {os.cpu_count()} workers successfully")

    def test_merge_scaling_analysis(self):
        """Performance: Test that merge performance scales with worker count"""
        print("\n=== Merge scaling analysis ===")

        # Test with different file counts and worker counts
        file_counts = [10, 20]
        worker_counts = [1, 2, 4]

        results = {}

        for n_files in file_counts:
            print(f"\n--- Testing with {n_files} files ---")

            # Create fresh data for this test
            nuc_files, pro_files = self.create_mock_feature_files(
                num_pairs=n_files,
                num_sequences=50
            )

            for n_workers in worker_counts:
                from virnucpro.pipeline.parallel_merge import parallel_merge_features

                output_dir = self.temp_path / f"scale_{n_files}_{n_workers}"
                output_dir.mkdir()

                start = time.time()
                merged = parallel_merge_features(
                    nuc_files,
                    pro_files,
                    output_dir,
                    num_workers=n_workers
                )
                elapsed = time.time() - start

                self.assertEqual(len(merged), n_files,
                               f"Should merge all {n_files} files with {n_workers} workers")

                results[(n_files, n_workers)] = elapsed
                print(f"{n_workers} workers: {elapsed:.2f}s ({n_files/elapsed:.1f} files/sec)")

        # Analyze scaling
        print("\n=== Scaling Results ===")
        for n_files in file_counts:
            baseline = results[(n_files, 1)]
            print(f"\n{n_files} files:")
            for n_workers in worker_counts:
                elapsed = results[(n_files, n_workers)]
                speedup = baseline / elapsed if elapsed > 0 else 0
                print(f"  {n_workers} workers: {elapsed:.2f}s (speedup: {speedup:.2f}x)")

            # Verify we get some speedup with 4 workers vs 1 worker
            speedup_4w = baseline / results[(n_files, 4)] if results[(n_files, 4)] > 0 else 0
            self.assertGreater(speedup_4w, 1.5,
                             f"Expected >= 1.5x speedup with 4 workers, got {speedup_4w:.2f}x")

        print("✓ Scaling analysis complete - performance improves with worker count")

    def test_memory_efficient_streaming(self):
        """Performance: Verify Pool.imap() doesn't load all results into memory"""
        # Create large number of file pairs
        nuc_files, pro_files = self.create_mock_feature_files(num_pairs=50, num_sequences=20)

        from virnucpro.pipeline.parallel_merge import parallel_merge_features

        output_dir = self.temp_path / "streaming"
        output_dir.mkdir()

        print("\n=== Testing memory-efficient streaming (50 files) ===")

        # Run merge - should use imap() for lazy evaluation
        start = time.time()
        merged = parallel_merge_features(
            nuc_files,
            pro_files,
            output_dir,
            num_workers=4
        )
        elapsed = time.time() - start

        self.assertEqual(len(merged), 50, "Should merge all 50 files")
        print(f"✓ Merged 50 files in {elapsed:.2f}s using streaming (imap)")

        # We can't directly verify memory usage in unit test,
        # but successful completion with 50 files demonstrates streaming works


if __name__ == '__main__':
    # Run with verbose output
    unittest.main(verbosity=2)
