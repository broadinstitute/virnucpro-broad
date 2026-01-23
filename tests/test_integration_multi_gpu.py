"""End-to-end integration tests for multi-GPU prediction pipeline"""

import unittest
import subprocess
import sys
import time
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
import torch


class TestMultiGPUIntegration(unittest.TestCase):
    """Integration tests verifying multi-GPU implementation produces correct outputs"""

    @classmethod
    def setUpClass(cls):
        """Set up test environment and check GPU availability"""
        # Check if multiple GPUs are available
        cls.num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

        if cls.num_gpus < 2:
            cls.skip_multi_gpu = True
            print(f"\nSkipping multi-GPU tests: only {cls.num_gpus} GPU(s) available")
        else:
            cls.skip_multi_gpu = False
            print(f"\nRunning multi-GPU tests with {cls.num_gpus} GPUs")

        # Locate test data
        cls.test_data_dir = Path(__file__).parent / "data"
        cls.test_input = cls.test_data_dir / "test_sequences_small.fa"

        if not cls.test_input.exists():
            raise FileNotFoundError(f"Test data not found: {cls.test_input}")

    def setUp(self):
        """Create temporary directory for each test"""
        self.temp_dir = TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

    def tearDown(self):
        """Clean up temporary directory"""
        self.temp_dir.cleanup()

    def test_single_vs_multi_gpu_equivalence(self):
        """Integration: Verify multi-GPU output matches single-GPU baseline"""
        if self.skip_multi_gpu:
            self.skipTest("Multiple GPUs not available")

        output_single = self.temp_path / "single_gpu_output"
        output_multi = self.temp_path / "multi_gpu_output"

        # Run with single GPU
        print("\n=== Running single-GPU baseline ===")
        start_single = time.time()
        result_single = subprocess.run([
            sys.executable, '-m', 'virnucpro', 'predict',
            str(self.test_input),
            '--output-dir', str(output_single),
            '--gpus', '0',
            '--quiet'
        ], capture_output=True, text=True, check=False)
        time_single = time.time() - start_single

        print(f"Single GPU exit code: {result_single.returncode}")
        if result_single.returncode not in (0, 2):
            print("STDOUT:", result_single.stdout)
            print("STDERR:", result_single.stderr)
            self.fail(f"Single GPU run failed with exit code {result_single.returncode}")

        # Run with multiple GPUs
        gpu_list = ','.join(str(i) for i in range(min(4, self.num_gpus)))
        print(f"\n=== Running multi-GPU with GPUs: {gpu_list} ===")
        start_multi = time.time()
        result_multi = subprocess.run([
            sys.executable, '-m', 'virnucpro', 'predict',
            str(self.test_input),
            '--output-dir', str(output_multi),
            '--gpus', gpu_list,
            '--quiet'
        ], capture_output=True, text=True, check=False)
        time_multi = time.time() - start_multi

        print(f"Multi GPU exit code: {result_multi.returncode}")
        if result_multi.returncode not in (0, 2):
            print("STDOUT:", result_multi.stdout)
            print("STDERR:", result_multi.stderr)
            self.fail(f"Multi GPU run failed with exit code {result_multi.returncode}")

        # Both should have same exit code
        self.assertEqual(
            result_single.returncode,
            result_multi.returncode,
            "Single and multi-GPU runs should have same exit code"
        )

        # Compare prediction results
        # Pipeline creates: {output_dir}/{input_stem}_merged/ with prediction_results.txt
        merged_dir_single = output_single / "test_sequences_small_merged"
        merged_dir_multi = output_multi / "test_sequences_small_merged"

        # Check if merged directories exist
        self.assertTrue(
            merged_dir_single.exists(),
            f"Single GPU merged directory not found at {merged_dir_single}"
        )
        self.assertTrue(
            merged_dir_multi.exists(),
            f"Multi GPU merged directory not found at {merged_dir_multi}"
        )

        # Check for prediction results files
        single_results = merged_dir_single / "prediction_results.txt"
        multi_results = merged_dir_multi / "prediction_results.txt"

        self.assertTrue(
            single_results.exists(),
            f"Single GPU prediction results not found at {single_results}"
        )
        self.assertTrue(
            multi_results.exists(),
            f"Multi GPU prediction results not found at {multi_results}"
        )

        # Load and compare prediction results
        import pandas as pd
        single_df = pd.read_csv(single_results, sep='\t')
        multi_df = pd.read_csv(multi_results, sep='\t')

        print(f"\nSingle GPU predictions: {len(single_df)} sequences")
        print(f"Multi GPU predictions: {len(multi_df)} sequences")

        # Verify same number of predictions
        self.assertEqual(
            len(single_df),
            len(multi_df),
            "Single and multi-GPU should predict same number of sequences"
        )

        # Verify same sequence IDs
        self.assertTrue(
            (single_df['Sequence_ID'] == multi_df['Sequence_ID']).all(),
            "Single and multi-GPU should process same sequences in same order"
        )

        # Verify predictions match
        self.assertTrue(
            (single_df['Prediction'] == multi_df['Prediction']).all(),
            "Single and multi-GPU should make same predictions"
        )

        # Verify scores match within tolerance
        import numpy as np
        score1_match = np.allclose(single_df['score1'], multi_df['score1'], rtol=1e-4, atol=1e-4)
        score2_match = np.allclose(single_df['score2'], multi_df['score2'], rtol=1e-4, atol=1e-4)

        if not score1_match or not score2_match:
            max_diff_s1 = np.max(np.abs(single_df['score1'] - multi_df['score1']))
            max_diff_s2 = np.max(np.abs(single_df['score2'] - multi_df['score2']))
            print(f"✗ Score differences - score1: {max_diff_s1:.6f}, score2: {max_diff_s2:.6f}")
            self.fail("Prediction scores differ between single and multi-GPU")

        print("✓ Predictions and scores match within tolerance")

        # Log performance
        speedup = time_single / time_multi if time_multi > 0 else 0
        print(f"\n=== Performance Metrics ===")
        print(f"Single GPU: {time_single:.1f}s")
        print(f"Multi GPU:  {time_multi:.1f}s")
        print(f"Speedup:    {speedup:.2f}x")

        # For small test data, speedup might not be ideal due to overhead
        # Just verify multi-GPU doesn't take significantly longer
        if speedup < 0.5:
            print(f"Warning: Multi-GPU slower than expected (speedup: {speedup:.2f}x)")

    def test_single_gpu_fallback(self):
        """Integration: Verify single GPU mode works correctly"""
        if not torch.cuda.is_available():
            self.skipTest("No GPU available")

        output_dir = self.temp_path / "single_gpu"

        print("\n=== Testing single GPU fallback mode ===")
        result = subprocess.run([
            sys.executable, '-m', 'virnucpro', 'predict',
            str(self.test_input),
            '--output-dir', str(output_dir),
            '--gpus', '0',
            '--quiet'
        ], capture_output=True, text=True, check=False)

        print(f"Exit code: {result.returncode}")
        if result.returncode not in (0, 2):
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)

        self.assertIn(
            result.returncode,
            (0, 2),
            "Single GPU mode should complete successfully"
        )

        # Verify output exists
        self.assertTrue(
            output_dir.exists(),
            "Output directory should be created"
        )

    def test_failed_file_recovery(self):
        """Integration: Verify failed files are logged correctly"""
        # This test would require creating corrupted/problematic sequences
        # Skip for now as it requires special test data
        self.skipTest("Requires special test data with problematic sequences")

    def test_checkpoint_compatibility(self):
        """Integration: Verify checkpoint resume works correctly"""
        # This test would require interrupting a run and resuming
        # Skip for now as it requires complex orchestration
        self.skipTest("Requires complex test orchestration")


if __name__ == '__main__':
    # Run with verbose output
    unittest.main(verbosity=2)
