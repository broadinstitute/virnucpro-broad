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

        # Compare merged feature outputs
        single_features_path = output_single / "merged" / "merged_features.pt"
        multi_features_path = output_multi / "merged" / "merged_features.pt"

        # Check if files exist
        if not single_features_path.exists():
            # Try alternative location
            single_features_path = output_single / "merged_features.pt"
        if not multi_features_path.exists():
            multi_features_path = output_multi / "merged_features.pt"

        self.assertTrue(
            single_features_path.exists(),
            f"Single GPU output not found at {single_features_path}"
        )
        self.assertTrue(
            multi_features_path.exists(),
            f"Multi GPU output not found at {multi_features_path}"
        )

        # Load and compare features
        single_features = torch.load(single_features_path)
        multi_features = torch.load(multi_features_path)

        # Check equivalence (within floating point tolerance for BF16)
        # Features might be in different keys depending on format
        if isinstance(single_features, dict) and 'data' in single_features:
            single_data = single_features['data']
            multi_data = multi_features['data']
        elif isinstance(single_features, torch.Tensor):
            single_data = single_features
            multi_data = multi_features
        else:
            # Try to find tensor data in dict
            single_data = next(v for v in single_features.values() if isinstance(v, torch.Tensor))
            multi_data = next(v for v in multi_features.values() if isinstance(v, torch.Tensor))

        print(f"\nSingle GPU tensor shape: {single_data.shape}")
        print(f"Multi GPU tensor shape: {multi_data.shape}")

        # Verify shapes match
        self.assertEqual(
            single_data.shape,
            multi_data.shape,
            "Single and multi-GPU outputs should have same shape"
        )

        # Verify values match within tolerance
        try:
            torch.testing.assert_close(
                single_data,
                multi_data,
                rtol=1e-4,  # Slightly looser for BF16
                atol=1e-4,
                msg="Multi-GPU output should match single-GPU within tolerance"
            )
            print("✓ Outputs match within tolerance")
        except AssertionError as e:
            # Calculate max difference for debugging
            max_diff = torch.max(torch.abs(single_data - multi_data)).item()
            print(f"✗ Max difference: {max_diff}")
            raise

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
