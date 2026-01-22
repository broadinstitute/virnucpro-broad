"""Tests for ESM-2 parallel processing utilities"""

import unittest
from pathlib import Path
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
import torch

from virnucpro.pipeline.parallel_esm import (
    assign_files_round_robin,
    process_esm_files_worker,
    count_sequences
)


class TestCountSequences(unittest.TestCase):
    """Test sequence counting utility"""

    def setUp(self):
        """Create temporary directory for test files"""
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        """Clean up temporary directory"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_count_sequences_empty(self):
        """Test counting sequences in empty file"""
        test_file = self.temp_dir / "empty.fa"
        test_file.write_text("")
        self.assertEqual(count_sequences(test_file), 0)

    def test_count_sequences_single(self):
        """Test counting single sequence"""
        test_file = self.temp_dir / "single.fa"
        test_file.write_text(">seq1\nACGT\n")
        self.assertEqual(count_sequences(test_file), 1)

    def test_count_sequences_multiple(self):
        """Test counting multiple sequences"""
        test_file = self.temp_dir / "multiple.fa"
        test_file.write_text(">seq1\nACGT\n>seq2\nTGCA\n>seq3\nAAAA\n")
        self.assertEqual(count_sequences(test_file), 3)


class TestAssignFilesRoundRobin(unittest.TestCase):
    """Test round-robin file assignment"""

    def test_empty_files(self):
        """Test with empty file list"""
        result = assign_files_round_robin([], 4)
        self.assertEqual(len(result), 4)
        self.assertTrue(all(len(worker_files) == 0 for worker_files in result))

    def test_single_worker(self):
        """Test all files go to one worker"""
        files = [Path(f"file_{i}.fa") for i in range(5)]
        result = assign_files_round_robin(files, 1)
        self.assertEqual(len(result), 1)
        self.assertEqual(len(result[0]), 5)
        self.assertEqual(result[0], files)

    def test_balanced_assignment(self):
        """Test files distributed evenly (round-robin pattern)"""
        files = [Path(f"file_{i}.fa") for i in range(8)]
        result = assign_files_round_robin(files, 4)

        # Should have 4 workers
        self.assertEqual(len(result), 4)

        # Each worker should get 2 files
        for worker_files in result:
            self.assertEqual(len(worker_files), 2)

        # Check round-robin pattern
        self.assertEqual(result[0], [Path("file_0.fa"), Path("file_4.fa")])
        self.assertEqual(result[1], [Path("file_1.fa"), Path("file_5.fa")])
        self.assertEqual(result[2], [Path("file_2.fa"), Path("file_6.fa")])
        self.assertEqual(result[3], [Path("file_3.fa"), Path("file_7.fa")])

    def test_uneven_assignment(self):
        """Test with files not evenly divisible by workers"""
        files = [Path(f"file_{i}.fa") for i in range(10)]
        result = assign_files_round_robin(files, 3)

        # Should have 3 workers
        self.assertEqual(len(result), 3)

        # Workers should get 4, 3, 3 files
        self.assertEqual(len(result[0]), 4)  # files 0, 3, 6, 9
        self.assertEqual(len(result[1]), 3)  # files 1, 4, 7
        self.assertEqual(len(result[2]), 3)  # files 2, 5, 8

        # Verify all files assigned
        all_assigned = []
        for worker_files in result:
            all_assigned.extend(worker_files)
        self.assertEqual(sorted(all_assigned), sorted(files))

    def test_invalid_num_workers(self):
        """Test error handling for invalid worker count"""
        files = [Path("file.fa")]
        with self.assertRaises(ValueError):
            assign_files_round_robin(files, 0)
        with self.assertRaises(ValueError):
            assign_files_round_robin(files, -1)


class TestESMWorker(unittest.TestCase):
    """Test ESM-2 worker function"""

    def setUp(self):
        """Create temp directory and mock files"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.output_dir = self.temp_dir / "output"
        self.output_dir.mkdir()

        # Create test protein files
        self.test_files = []
        for i in range(3):
            test_file = self.temp_dir / f"protein_{i}.faa"
            test_file.write_text(f">protein_{i}\nMKLLVLGLLGAALA\n")
            self.test_files.append(test_file)

    def tearDown(self):
        """Clean up temp files"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    @patch('virnucpro.pipeline.parallel_esm.extract_esm_features')
    def test_worker_processes_files(self):
        """Test worker processes files successfully"""
        # Mock extract_esm_features to avoid actual model loading
        mock_extract = MagicMock()

        with patch('virnucpro.pipeline.parallel_esm.extract_esm_features', mock_extract):
            processed, failed = process_esm_files_worker(
                file_subset=self.test_files,
                device_id=0,
                toks_per_batch=2048,
                output_dir=self.output_dir
            )

        # Should process all files successfully
        self.assertEqual(len(processed), 3)
        self.assertEqual(len(failed), 0)

        # Check extract_esm_features was called for each file
        self.assertEqual(mock_extract.call_count, 3)

        # Verify output file paths
        for i, output_file in enumerate(processed):
            self.assertEqual(output_file, self.output_dir / f"protein_{i}_ESM.pt")

    @patch('virnucpro.pipeline.parallel_esm.extract_esm_features')
    @patch('virnucpro.pipeline.parallel_esm.torch.cuda.empty_cache')
    def test_worker_handles_oom(self):
        """Test worker handles OOM errors gracefully"""
        # Mock OOM error on second file
        def mock_extract_side_effect(input_file, output_file, device, toks_per_batch):
            if "protein_1" in str(input_file):
                raise RuntimeError("CUDA out of memory")

        mock_extract = MagicMock(side_effect=mock_extract_side_effect)
        mock_cache = MagicMock()

        with patch('virnucpro.pipeline.parallel_esm.extract_esm_features', mock_extract):
            with patch('virnucpro.pipeline.parallel_esm.torch.cuda.empty_cache', mock_cache):
                processed, failed = process_esm_files_worker(
                    file_subset=self.test_files,
                    device_id=0,
                    toks_per_batch=2048,
                    output_dir=self.output_dir
                )

        # Should process 2 files, fail 1
        self.assertEqual(len(processed), 2)
        self.assertEqual(len(failed), 1)

        # Check failed file
        failed_file, error_msg = failed[0]
        self.assertTrue("protein_1" in str(failed_file))
        self.assertIn("out of memory", error_msg.lower())

        # Verify cache was cleared after OOM
        mock_cache.assert_called_once()

    @patch('virnucpro.pipeline.parallel_esm.extract_esm_features')
    def test_worker_handles_general_errors(self):
        """Test worker handles non-OOM errors"""
        # Mock general error on first file
        def mock_extract_side_effect(input_file, output_file, device, toks_per_batch):
            if "protein_0" in str(input_file):
                raise ValueError("Invalid sequence format")

        mock_extract = MagicMock(side_effect=mock_extract_side_effect)

        with patch('virnucpro.pipeline.parallel_esm.extract_esm_features', mock_extract):
            processed, failed = process_esm_files_worker(
                file_subset=self.test_files,
                device_id=0,
                toks_per_batch=2048,
                output_dir=self.output_dir
            )

        # Should process 2 files, fail 1
        self.assertEqual(len(processed), 2)
        self.assertEqual(len(failed), 1)

        # Check failed file
        failed_file, error_msg = failed[0]
        self.assertTrue("protein_0" in str(failed_file))
        self.assertIn("Invalid sequence format", error_msg)

    @patch('virnucpro.pipeline.parallel_esm.torch.device')
    @patch('virnucpro.pipeline.parallel_esm.extract_esm_features')
    def test_worker_cuda_initialization(self):
        """Test worker initializes CUDA device correctly"""
        mock_device = MagicMock()
        mock_extract = MagicMock()

        with patch('virnucpro.pipeline.parallel_esm.torch.device', mock_device):
            with patch('virnucpro.pipeline.parallel_esm.extract_esm_features', mock_extract):
                process_esm_files_worker(
                    file_subset=self.test_files[:1],
                    device_id=2,
                    toks_per_batch=2048,
                    output_dir=self.output_dir
                )

        # Verify device was created for cuda:2
        mock_device.assert_called_once_with('cuda:2')


if __name__ == '__main__':
    unittest.main()
