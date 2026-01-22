"""Tests for batch queue manager for multi-GPU work distribution"""

import unittest
from pathlib import Path
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
from typing import List, Tuple

from virnucpro.pipeline.work_queue import (
    BatchQueueManager,
    WorkerStatus
)


class TestWorkerStatus(unittest.TestCase):
    """Test WorkerStatus enum"""

    def test_enum_values(self):
        """Verify all status values exist and have correct values"""
        self.assertEqual(WorkerStatus.IDLE.value, "idle")
        self.assertEqual(WorkerStatus.PROCESSING.value, "processing")
        self.assertEqual(WorkerStatus.COMPLETED.value, "completed")
        self.assertEqual(WorkerStatus.FAILED.value, "failed")

    def test_enum_members(self):
        """Verify all expected members exist"""
        expected_members = {'IDLE', 'PROCESSING', 'COMPLETED', 'FAILED'}
        actual_members = {member.name for member in WorkerStatus}
        self.assertEqual(actual_members, expected_members)


class TestBatchQueueManager(unittest.TestCase):
    """Test BatchQueueManager"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        """Clean up temp directory"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_initialization(self):
        """Test manager initialization"""
        def valid_worker(file_subset, device_id, **kwargs):
            return ([], [])

        manager = BatchQueueManager(4, valid_worker)

        self.assertEqual(manager.num_workers, 4)
        self.assertEqual(manager.worker_function, valid_worker)
        self.assertEqual(len(manager.worker_status), 4)

        # All workers should start idle
        for worker_id in range(4):
            self.assertEqual(manager.worker_status[worker_id], WorkerStatus.IDLE)

    def test_worker_function_validation_invalid(self):
        """Test worker function signature validation rejects invalid functions"""
        # Function with too few parameters
        def invalid_worker(files):
            return ([], [])

        with self.assertRaises(ValueError) as context:
            BatchQueueManager(4, invalid_worker)

        self.assertIn("at least (file_subset, device_id", str(context.exception))

    def test_worker_function_validation_valid(self):
        """Test worker function signature validation accepts valid functions"""
        # Valid worker with required params
        def valid_worker_1(file_subset, device_id):
            return ([], [])

        # Valid worker with kwargs
        def valid_worker_2(file_subset, device_id, **kwargs):
            return ([], [])

        # Valid worker with extra params
        def valid_worker_3(file_subset, device_id, batch_size, output_dir):
            return ([], [])

        # All should initialize without error
        BatchQueueManager(2, valid_worker_1)
        BatchQueueManager(2, valid_worker_2)
        BatchQueueManager(2, valid_worker_3)

    def test_get_worker_status(self):
        """Test getting worker status"""
        def worker(file_subset, device_id, **kwargs):
            return ([], [])

        manager = BatchQueueManager(3, worker)

        # Modify status
        manager.worker_status[0] = WorkerStatus.PROCESSING
        manager.worker_status[1] = WorkerStatus.COMPLETED

        status = manager.get_worker_status()

        # Should return copy with current status
        self.assertEqual(status[0], WorkerStatus.PROCESSING)
        self.assertEqual(status[1], WorkerStatus.COMPLETED)
        self.assertEqual(status[2], WorkerStatus.IDLE)

        # Modifying returned dict shouldn't affect manager
        status[0] = WorkerStatus.FAILED
        self.assertEqual(manager.worker_status[0], WorkerStatus.PROCESSING)

    def test_is_complete(self):
        """Test completion check"""
        def worker(file_subset, device_id, **kwargs):
            return ([], [])

        manager = BatchQueueManager(3, worker)

        # Initially not complete (all idle)
        self.assertFalse(manager.is_complete())

        # Set all to processing - still not complete
        for i in range(3):
            manager.worker_status[i] = WorkerStatus.PROCESSING
        self.assertFalse(manager.is_complete())

        # Set all to completed - now complete
        for i in range(3):
            manager.worker_status[i] = WorkerStatus.COMPLETED
        self.assertTrue(manager.is_complete())

        # Mix of completed and failed - still complete
        manager.worker_status[1] = WorkerStatus.FAILED
        self.assertTrue(manager.is_complete())

    def test_process_files_success(self):
        """Test successful file processing"""
        # Create mock worker that succeeds
        def successful_worker(file_subset: List[Path], device_id: int, **kwargs) -> Tuple[List[Path], List[Tuple[Path, str]]]:
            processed = [f.parent / f"{f.stem}_output.pt" for f in file_subset]
            return (processed, [])

        manager = BatchQueueManager(2, successful_worker, spawn_context=False)

        # Create test files
        test_files = [
            [Path(self.temp_dir / "file_0.fa"), Path(self.temp_dir / "file_2.fa")],
            [Path(self.temp_dir / "file_1.fa"), Path(self.temp_dir / "file_3.fa")]
        ]

        # Process files
        processed, failed = manager.process_files(test_files)

        # Should process all files
        self.assertEqual(len(processed), 4)
        self.assertEqual(len(failed), 0)

        # Check worker status
        for worker_id in range(2):
            self.assertEqual(manager.worker_status[worker_id], WorkerStatus.COMPLETED)

    def test_process_files_with_failures(self):
        """Test file processing with some failures"""
        # Worker that fails on files containing "fail"
        def partial_worker(file_subset: List[Path], device_id: int, **kwargs) -> Tuple[List[Path], List[Tuple[Path, str]]]:
            processed = []
            failed = []

            for f in file_subset:
                if "fail" in f.name:
                    failed.append((f, "Simulated failure"))
                else:
                    processed.append(f.parent / f"{f.stem}_output.pt")

            return (processed, failed)

        manager = BatchQueueManager(2, partial_worker, spawn_context=False)

        # Create test files (some with "fail" in name)
        test_files = [
            [Path(self.temp_dir / "file_0.fa"), Path(self.temp_dir / "fail_2.fa")],
            [Path(self.temp_dir / "file_1.fa"), Path(self.temp_dir / "fail_3.fa")]
        ]

        # Process files
        processed, failed = manager.process_files(test_files)

        # Should process 2, fail 2
        self.assertEqual(len(processed), 2)
        self.assertEqual(len(failed), 2)

        # Verify failed files
        failed_names = [str(f[0].name) for f in failed]
        self.assertIn("fail_2.fa", failed_names)
        self.assertIn("fail_3.fa", failed_names)

    def test_process_files_worker_crash(self):
        """Test handling of worker crashes"""
        # Worker that crashes (raises exception)
        def crashing_worker(file_subset: List[Path], device_id: int, **kwargs) -> Tuple[List[Path], List[Tuple[Path, str]]]:
            if device_id == 1:
                raise RuntimeError("Worker crashed")
            return ([f.parent / f"{f.stem}_output.pt" for f in file_subset], [])

        manager = BatchQueueManager(2, crashing_worker, spawn_context=False)

        test_files = [
            [Path(self.temp_dir / "file_0.fa")],
            [Path(self.temp_dir / "file_1.fa")]
        ]

        # Process files - worker 1 will crash
        processed, failed = manager.process_files(test_files)

        # Worker 0 should succeed, worker 1 should fail
        self.assertEqual(len(processed), 1)  # Only worker 0's file
        self.assertEqual(manager.worker_status[0], WorkerStatus.COMPLETED)
        self.assertEqual(manager.worker_status[1], WorkerStatus.FAILED)

    def test_systemic_failure_detection(self):
        """Test detection of systemic failures (3+ workers fail)"""
        # Worker that always crashes
        def always_crash_worker(file_subset: List[Path], device_id: int, **kwargs):
            raise RuntimeError("Systemic failure")

        manager = BatchQueueManager(4, always_crash_worker, spawn_context=False)

        test_files = [
            [Path(self.temp_dir / f"file_{i}.fa")]
            for i in range(4)
        ]

        # Should raise RuntimeError about systemic failure
        with self.assertRaises(RuntimeError) as context:
            manager.process_files(test_files)

        self.assertIn("Systemic failure", str(context.exception))
        self.assertIn("4/4 workers failed", str(context.exception))

    def test_process_files_invalid_assignments(self):
        """Test error handling for mismatched file assignments"""
        def worker(file_subset, device_id, **kwargs):
            return ([], [])

        manager = BatchQueueManager(3, worker)

        # Provide wrong number of assignments
        test_files = [
            [Path("file_0.fa")],
            [Path("file_1.fa")]
        ]  # Only 2 assignments, but manager expects 3

        with self.assertRaises(ValueError) as context:
            manager.process_files(test_files)

        self.assertIn("Expected 3 file assignments", str(context.exception))


if __name__ == '__main__':
    unittest.main()
