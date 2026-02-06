"""Unit tests for CheckpointManifest with concurrency safety.

Tests manifest initialization, shard tracking, completion/failure marking,
queries, atomic writes, concurrent multi-process updates (no corruption),
file locking, and redistribution tracking.
"""

import json
import multiprocessing
import os
import time
from pathlib import Path
from unittest.mock import patch, Mock

import pytest

from virnucpro.pipeline.checkpoint_manifest import CheckpointManifest


class TestManifestBasicOperations:
    """Test basic manifest operations."""

    def test_manifest_initialize_creates_json(self, tmp_path):
        """Test initialization creates JSON with correct structure."""
        manifest_path = tmp_path / "manifest.json"
        manifest = CheckpointManifest(manifest_path)

        manifest.initialize(world_size=4, input_fingerprint="abc123", model_config_hash="def456")

        # Verify JSON file exists
        assert manifest_path.exists()

        # Load and verify structure
        with open(manifest_path) as f:
            data = json.load(f)

        assert data["version"] == "2.0"
        assert data["world_size"] == 4
        assert len(data["shards"]) == 4

        # Verify all shards are in_progress
        for rank in range(4):
            assert data["shards"][str(rank)]["status"] == "in_progress"

    def test_manifest_update_shard_checkpoint(self, tmp_path):
        """Test updating shard checkpoint progress."""
        manifest_path = tmp_path / "manifest.json"
        manifest = CheckpointManifest(manifest_path)
        manifest.initialize(world_size=2)

        # Create checkpoint directory structure
        shard_dir = tmp_path / "shard_0"
        shard_dir.mkdir()
        checkpoint_file = shard_dir / "batch_00005.pt"
        checkpoint_file.touch()

        # Update shard 0 checkpoint
        manifest.update_shard_checkpoint(
            rank=0,
            batch_idx=5,
            num_sequences=100,
            checkpoint_file="batch_00005.pt"
        )

        # Load and verify
        with open(manifest_path) as f:
            data = json.load(f)

        shard_0 = data["shards"]["0"]
        assert shard_0["last_checkpoint_batch"] == 5
        assert shard_0["total_sequences"] == 100
        assert len(shard_0["checkpoints"]) == 1
        assert shard_0["checkpoints"][0]["file"] == "batch_00005.pt"

    def test_manifest_mark_shard_complete(self, tmp_path):
        """Test marking shard as complete."""
        manifest_path = tmp_path / "manifest.json"
        manifest = CheckpointManifest(manifest_path)
        manifest.initialize(world_size=2)

        # Mark shard 0 complete
        manifest.mark_shard_complete(rank=0)

        # Load and verify
        with open(manifest_path) as f:
            data = json.load(f)

        shard_0 = data["shards"]["0"]
        assert shard_0["status"] == "complete"
        assert "completed_at" in shard_0

    def test_manifest_mark_shard_failed(self, tmp_path):
        """Test marking shard as failed."""
        manifest_path = tmp_path / "manifest.json"
        manifest = CheckpointManifest(manifest_path)
        manifest.initialize(world_size=2)

        # Mark shard 1 failed
        error_message = "CUDA out of memory"
        manifest.mark_shard_failed(rank=1, error=error_message)

        # Load and verify
        with open(manifest_path) as f:
            data = json.load(f)

        shard_1 = data["shards"]["1"]
        assert shard_1["status"] == "failed"
        assert shard_1["error"] == error_message
        assert shard_1["retry_count"] == 1

    def test_manifest_get_resumable_shards(self, tmp_path):
        """Test getting resumable (in_progress or failed) shards."""
        manifest_path = tmp_path / "manifest.json"
        manifest = CheckpointManifest(manifest_path)
        manifest.initialize(world_size=4)

        # Mark shard 0 complete
        manifest.mark_shard_complete(rank=0)

        # Leave shard 1 in_progress

        # Mark shard 2 failed
        manifest.mark_shard_failed(rank=2, error="Test error")

        # Leave shard 3 in_progress

        # Get resumable shards
        resumable = manifest.get_resumable_shards()

        # Should return shards 1, 2, 3 (in_progress and failed)
        assert set(resumable) == {1, 2, 3}

    def test_manifest_get_completed_shards(self, tmp_path):
        """Test getting completed shards."""
        manifest_path = tmp_path / "manifest.json"
        manifest = CheckpointManifest(manifest_path)
        manifest.initialize(world_size=4)

        # Mark shards 0 and 2 complete
        manifest.mark_shard_complete(rank=0)
        manifest.mark_shard_complete(rank=2)

        # Get completed shards
        completed = manifest.get_completed_shards()

        assert set(completed) == {0, 2}

    def test_manifest_get_global_progress(self, tmp_path):
        """Test getting global progress summary."""
        manifest_path = tmp_path / "manifest.json"
        manifest = CheckpointManifest(manifest_path)
        manifest.initialize(world_size=4)

        # Mark shard 0 complete
        manifest.mark_shard_complete(rank=0)

        # Mark shard 1 failed
        manifest.mark_shard_failed(rank=1, error="Test error")

        # Shards 2 and 3 remain in_progress

        # Get global progress
        progress = manifest.get_global_progress()

        assert progress["total_shards"] == 4
        assert progress["completed"] == 1
        assert progress["in_progress"] == 2
        assert progress["failed"] == 1

    def test_manifest_atomic_write_no_temp_remains(self, tmp_path):
        """Test no .tmp files remain after successful operations."""
        manifest_path = tmp_path / "manifest.json"
        manifest = CheckpointManifest(manifest_path)
        manifest.initialize(world_size=2)

        # Perform several operations
        manifest.mark_shard_complete(rank=0)
        manifest.mark_shard_failed(rank=1, error="Test")

        # Check no .tmp files
        tmp_files = list(tmp_path.glob("*.tmp"))
        assert len(tmp_files) == 0

    def test_manifest_exists_false_before_init(self, tmp_path):
        """Test exists() returns False before initialization."""
        manifest_path = tmp_path / "manifest.json"
        manifest = CheckpointManifest(manifest_path)

        assert manifest.exists() is False

        # Initialize
        manifest.initialize(world_size=2)

        assert manifest.exists() is True

    def test_manifest_cumulative_sequence_count(self, tmp_path):
        """Test total_sequences is cumulative across updates."""
        manifest_path = tmp_path / "manifest.json"
        manifest = CheckpointManifest(manifest_path)
        manifest.initialize(world_size=1)

        # Create checkpoint directory
        shard_dir = tmp_path / "shard_0"
        shard_dir.mkdir()

        # First update
        checkpoint1 = shard_dir / "batch_00000.pt"
        checkpoint1.touch()
        manifest.update_shard_checkpoint(
            rank=0,
            batch_idx=0,
            num_sequences=50,
            checkpoint_file="batch_00000.pt"
        )

        # Second update
        checkpoint2 = shard_dir / "batch_00001.pt"
        checkpoint2.touch()
        manifest.update_shard_checkpoint(
            rank=0,
            batch_idx=1,
            num_sequences=75,
            checkpoint_file="batch_00001.pt"
        )

        # Load and verify cumulative total
        with open(manifest_path) as f:
            data = json.load(f)

        assert data["shards"]["0"]["total_sequences"] == 125  # 50 + 75


class TestManifestConcurrency:
    """Test concurrent manifest access safety."""

    def test_manifest_concurrent_updates_no_corruption(self, tmp_path):
        """Test concurrent updates from multiple processes don't corrupt JSON."""
        manifest_path = tmp_path / "manifest.json"

        # Initialize manifest
        manifest = CheckpointManifest(manifest_path)
        manifest.initialize(world_size=4)

        # Create shard directories
        for rank in range(4):
            shard_dir = tmp_path / f"shard_{rank}"
            shard_dir.mkdir()

        # Worker function for each process
        def worker_update(rank, manifest_path, tmp_path):
            manifest = CheckpointManifest(manifest_path)

            # Each worker makes 50 updates
            for i in range(50):
                # Create checkpoint file
                shard_dir = tmp_path / f"shard_{rank}"
                checkpoint_file = shard_dir / f"batch_{i:05d}.pt"
                checkpoint_file.touch()

                # Update manifest
                manifest.update_shard_checkpoint(
                    rank=rank,
                    batch_idx=i,
                    num_sequences=10,
                    checkpoint_file=f"batch_{i:05d}.pt"
                )

        # Spawn 4 processes
        processes = []
        for rank in range(4):
            p = multiprocessing.Process(
                target=worker_update,
                args=(rank, manifest_path, tmp_path)
            )
            p.start()
            processes.append(p)

        # Wait for all processes to complete
        for p in processes:
            p.join(timeout=30)

        # Verify final manifest is valid JSON
        with open(manifest_path) as f:
            data = json.load(f)

        # Verify all 200 updates recorded (4 ranks × 50 updates)
        total_checkpoints = sum(
            len(data["shards"][str(rank)]["checkpoints"])
            for rank in range(4)
        )
        assert total_checkpoints == 200

        # Verify each rank has 50 updates
        for rank in range(4):
            shard = data["shards"][str(rank)]
            assert len(shard["checkpoints"]) == 50
            assert shard["total_sequences"] == 500  # 50 updates × 10 sequences

    def test_manifest_concurrent_same_shard_updates(self, tmp_path):
        """Test concurrent updates to same shard don't lose data."""
        manifest_path = tmp_path / "manifest.json"

        # Initialize manifest
        manifest = CheckpointManifest(manifest_path)
        manifest.initialize(world_size=1)

        # Create shard directory
        shard_dir = tmp_path / "shard_0"
        shard_dir.mkdir()

        # Worker function - both workers update rank 0
        def worker_update(worker_id, manifest_path, tmp_path):
            manifest = CheckpointManifest(manifest_path)

            # Each worker makes 25 updates
            for i in range(25):
                batch_idx = worker_id * 25 + i
                # Create checkpoint file
                shard_dir = tmp_path / "shard_0"
                checkpoint_file = shard_dir / f"batch_{batch_idx:05d}.pt"
                checkpoint_file.touch()

                # Update manifest
                manifest.update_shard_checkpoint(
                    rank=0,
                    batch_idx=batch_idx,
                    num_sequences=10,
                    checkpoint_file=f"batch_{batch_idx:05d}.pt"
                )

        # Spawn 2 processes both updating rank 0
        processes = []
        for worker_id in range(2):
            p = multiprocessing.Process(
                target=worker_update,
                args=(worker_id, manifest_path, tmp_path)
            )
            p.start()
            processes.append(p)

        # Wait for completion
        for p in processes:
            p.join(timeout=30)

        # Load final manifest
        with open(manifest_path) as f:
            data = json.load(f)

        # Verify all 50 updates recorded
        shard_0 = data["shards"]["0"]
        assert len(shard_0["checkpoints"]) == 50
        assert shard_0["total_sequences"] == 500  # 50 × 10

    def test_manifest_file_lock_prevents_corruption(self, tmp_path):
        """Test file locking is used for concurrent access."""
        manifest_path = tmp_path / "manifest.json"
        manifest = CheckpointManifest(manifest_path)

        # Mock fcntl.flock to track calls
        lock_calls = []
        unlock_calls = []

        import fcntl
        original_flock = fcntl.flock

        def mock_flock(fd, operation):
            if operation == fcntl.LOCK_EX:
                lock_calls.append(fd)
            elif operation == fcntl.LOCK_UN:
                unlock_calls.append(fd)
            return original_flock(fd, operation)

        with patch('fcntl.flock', side_effect=mock_flock):
            # Initialize should acquire lock
            manifest.initialize(world_size=2)

            # Verify lock was acquired and released
            assert len(lock_calls) >= 1
            assert len(unlock_calls) >= 1


class TestManifestRedistribution:
    """Test elastic shard redistribution tracking."""

    def test_manifest_record_redistribution(self, tmp_path):
        """Test recording shard redistribution."""
        manifest_path = tmp_path / "manifest.json"
        manifest = CheckpointManifest(manifest_path)
        manifest.initialize(world_size=4)

        # Record redistribution
        manifest.reassign_shard(shard_rank=2, new_assigned_rank=0)

        # Load and verify
        with open(manifest_path) as f:
            data = json.load(f)

        shard_2 = data["shards"]["2"]
        assert shard_2["original_rank"] == 2  # Immutable
        assert shard_2["assigned_rank"] == 0  # Updated

    def test_manifest_get_redistributed_shards(self, tmp_path):
        """Test getting redistributed shards mapping."""
        manifest_path = tmp_path / "manifest.json"
        manifest = CheckpointManifest(manifest_path)
        manifest.initialize(world_size=4)

        # Record 2 redistributions
        manifest.reassign_shard(shard_rank=2, new_assigned_rank=0)
        manifest.reassign_shard(shard_rank=3, new_assigned_rank=1)

        # Get redistributed mapping
        with open(manifest_path) as f:
            data = json.load(f)

        redistributed = {}
        for rank_str, shard in data["shards"].items():
            if shard["original_rank"] != shard["assigned_rank"]:
                redistributed[shard["original_rank"]] = shard["assigned_rank"]

        assert redistributed == {2: 0, 3: 1}
