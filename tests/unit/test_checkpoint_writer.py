"""Unit tests for checkpoint writer foundation.

Tests CheckpointTrigger, AsyncCheckpointWriter, validate_checkpoint_pt,
and resume_from_checkpoints with .pt format validation, GPU→CPU transfer
safety, async failure propagation, and resume data integrity.
"""

import os
import time
import threading
from pathlib import Path
from unittest.mock import patch, Mock, MagicMock, call

import pytest
import numpy as np
import torch

from virnucpro.pipeline.checkpoint_writer import (
    CheckpointTrigger,
    AsyncCheckpointWriter,
    validate_checkpoint_pt,
    resume_from_checkpoints
)


class TestCheckpointTrigger:
    """Test CheckpointTrigger threshold logic."""

    def test_trigger_sequence_threshold_fires(self):
        """Test trigger fires on sequence threshold."""
        trigger = CheckpointTrigger(seq_threshold=100, time_threshold_sec=300.0)

        # First call with 50 sequences
        should_checkpoint, reason = trigger.should_checkpoint(batch_size=50)
        assert should_checkpoint is False
        assert reason is None

        # Second call with 50 more sequences (total 100)
        should_checkpoint, reason = trigger.should_checkpoint(batch_size=50)
        assert should_checkpoint is True
        assert reason == "sequence_threshold"

    def test_trigger_time_threshold_fires(self):
        """Test trigger fires on time threshold."""
        trigger = CheckpointTrigger(seq_threshold=10000, time_threshold_sec=0.1)

        # Wait for time threshold to pass
        time.sleep(0.15)

        should_checkpoint, reason = trigger.should_checkpoint(batch_size=1)
        assert should_checkpoint is True
        assert reason == "time_threshold"

    def test_trigger_emergency_override(self):
        """Test emergency override fires even with low sequence count."""
        # Emergency must be > time_threshold, so use 0.2 > 0.1
        trigger = CheckpointTrigger(
            seq_threshold=10000,
            time_threshold_sec=0.1,
            emergency_override_sec=0.2
        )

        # Wait for emergency override
        time.sleep(0.25)

        # Call with very small batch (far below threshold)
        should_checkpoint, reason = trigger.should_checkpoint(batch_size=1)
        assert should_checkpoint is True
        assert reason == "emergency_time_override"

    def test_trigger_reset_clears_counters(self):
        """Test reset clears sequence counter and timer."""
        trigger = CheckpointTrigger(seq_threshold=100, time_threshold_sec=300.0)

        # Trigger on sequence threshold
        trigger.should_checkpoint(batch_size=100)
        should_checkpoint, _ = trigger.should_checkpoint(batch_size=1)
        assert should_checkpoint is True

        # Reset
        trigger.reset()

        # Small batch should not trigger after reset
        should_checkpoint, reason = trigger.should_checkpoint(batch_size=10)
        assert should_checkpoint is False
        assert reason is None

    def test_trigger_viral_mode_overrides_defaults(self, monkeypatch):
        """Test viral mode env var overrides default thresholds."""
        # Set env var
        monkeypatch.setenv('VIRNUCPRO_VIRAL_CHECKPOINT_MODE', 'true')

        # Create trigger with DEFAULT args (no explicit thresholds)
        trigger = CheckpointTrigger()

        # Assert viral mode thresholds applied
        assert trigger.seq_threshold == 5000
        assert trigger.time_threshold_sec == 180.0

        # Create trigger with EXPLICIT args
        trigger_explicit = CheckpointTrigger(seq_threshold=999)

        # Assert explicit args take precedence over env var
        assert trigger_explicit.seq_threshold == 999
        # time_threshold still uses default (not viral) because it's not default value
        # Actually, looking at the code, viral mode only applies when BOTH are defaults
        # So with explicit seq_threshold, viral mode doesn't apply at all
        assert trigger_explicit.time_threshold_sec == 300.0

    def test_trigger_viral_mode_disabled_uses_defaults(self, monkeypatch):
        """Test defaults used when viral mode disabled."""
        # Ensure env var is not set
        monkeypatch.delenv('VIRNUCPRO_VIRAL_CHECKPOINT_MODE', raising=False)

        trigger = CheckpointTrigger()

        # Assert standard defaults
        assert trigger.seq_threshold == 10000
        assert trigger.time_threshold_sec == 300.0


class TestAsyncCheckpointWriter:
    """Test AsyncCheckpointWriter with .pt format."""

    def test_writer_creates_pt_with_done_marker(self, tmp_path):
        """Test writer creates .pt file with .done marker."""
        writer = AsyncCheckpointWriter()
        checkpoint_path = tmp_path / "batch_00000.pt"

        # Write checkpoint
        embeddings = np.random.randn(10, 128)
        sequence_ids = [f"seq_{i}" for i in range(10)]
        metadata = {'batch_idx': 0, 'num_sequences': 10, 'timestamp': '2026-02-06T00:00:00'}

        writer.write_checkpoint_async(checkpoint_path, embeddings, sequence_ids, metadata)
        writer.wait_all()

        # Assert .pt file exists
        assert checkpoint_path.exists()

        # Assert .done marker exists
        done_marker = checkpoint_path.with_suffix('.pt.done')
        assert done_marker.exists()

        writer.shutdown()

    def test_writer_pt_contains_correct_data(self, tmp_path):
        """Test .pt file contains correct data structure."""
        writer = AsyncCheckpointWriter()
        checkpoint_path = tmp_path / "batch_00000.pt"

        embeddings = np.random.randn(10, 128)
        sequence_ids = [f"seq_{i}" for i in range(10)]
        metadata = {'batch_idx': 0, 'num_sequences': 10, 'timestamp': '2026-02-06T00:00:00'}

        writer.write_checkpoint_async(checkpoint_path, embeddings, sequence_ids, metadata)
        writer.wait_all()

        # Read back checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        # Assert structure
        assert 'embeddings' in checkpoint
        assert 'sequence_ids' in checkpoint
        assert 'metadata' in checkpoint

        # Assert data matches
        assert checkpoint['embeddings'].shape == (10, 128)
        assert checkpoint['sequence_ids'] == sequence_ids
        assert checkpoint['metadata']['batch_idx'] == 0
        assert checkpoint['metadata']['num_sequences'] == 10

        writer.shutdown()

    def test_writer_atomic_write_no_temp_file_remains(self, tmp_path):
        """Test no .tmp files remain after successful write."""
        writer = AsyncCheckpointWriter()
        checkpoint_path = tmp_path / "batch_00000.pt"

        embeddings = np.random.randn(5, 64)
        sequence_ids = [f"seq_{i}" for i in range(5)]
        metadata = {'batch_idx': 0}

        writer.write_checkpoint_async(checkpoint_path, embeddings, sequence_ids, metadata)
        writer.wait_all()

        # Assert no .tmp files in directory
        tmp_files = list(tmp_path.glob("*.tmp"))
        assert len(tmp_files) == 0

        writer.shutdown()

    def test_writer_copies_numpy_data_safely(self, tmp_path):
        """Test writer copies data before async write to prevent race conditions."""
        writer = AsyncCheckpointWriter()
        checkpoint_path = tmp_path / "batch_00000.pt"

        # Create original array
        original_array = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        sequence_ids = ['seq_0', 'seq_1']
        metadata = {'batch_idx': 0}

        # Use threading.Barrier to pause write and modify data
        barrier = threading.Barrier(2)
        original_write_sync = writer._write_checkpoint_sync

        def paused_write_sync(checkpoint_path, embeddings, sequence_ids, metadata):
            # Wait for main thread to modify original_array
            barrier.wait()
            # Now proceed with actual write
            original_write_sync(checkpoint_path, embeddings, sequence_ids, metadata)

        with patch.object(writer, '_write_checkpoint_sync', side_effect=paused_write_sync):
            # Start async write
            future = writer.write_checkpoint_async(
                checkpoint_path, original_array, sequence_ids, metadata
            )

            # Modify original array while write is pending
            original_array[0, 0] = 999.0

            # Release barrier
            barrier.wait()

            # Wait for completion
            future.result()

        # Read checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        # Verify ORIGINAL data saved (not modified)
        assert checkpoint['embeddings'][0, 0] == 1.0
        assert checkpoint['embeddings'][1, 2] == 6.0

        writer.shutdown()

    def test_writer_gpu_to_cpu_transfer(self, tmp_path):
        """Test GPU tensor is transferred to CPU before async write."""
        # Use a real tensor but track the method calls via patching
        writer = AsyncCheckpointWriter()
        checkpoint_path = tmp_path / "batch_00000.pt"

        # Create a real tensor
        gpu_tensor = torch.randn(5, 64)

        # Track calls by wrapping the actual methods
        to_cpu_call_tracker = []
        numpy_call_tracker = []

        original_to = gpu_tensor.to

        def tracked_to(device):
            if device == 'cpu':
                to_cpu_call_tracker.append(True)
                result = original_to(device)
                # Also track numpy calls on the CPU tensor
                original_numpy = result.numpy
                def tracked_numpy():
                    numpy_call_tracker.append(True)
                    return original_numpy()
                result.numpy = tracked_numpy
                return result
            return original_to(device)

        gpu_tensor.to = tracked_to

        sequence_ids = [f"seq_{i}" for i in range(5)]
        metadata = {'batch_idx': 0}

        # Write checkpoint
        writer.write_checkpoint_async(checkpoint_path, gpu_tensor, sequence_ids, metadata)
        writer.wait_all()

        # Verify .to('cpu') was called
        assert len(to_cpu_call_tracker) == 1, "to('cpu') should be called once"

        # Verify .numpy() was called on the CPU tensor
        assert len(numpy_call_tracker) == 1, "numpy() should be called once on CPU tensor"

        writer.shutdown()

    def test_writer_shutdown_waits_for_pending(self, tmp_path):
        """Test shutdown waits for all pending writes to complete."""
        writer = AsyncCheckpointWriter()

        # Queue 3 writes
        for i in range(3):
            checkpoint_path = tmp_path / f"batch_{i:05d}.pt"
            embeddings = np.random.randn(5, 32)
            sequence_ids = [f"seq_{j}" for j in range(5)]
            metadata = {'batch_idx': i}
            writer.write_checkpoint_async(checkpoint_path, embeddings, sequence_ids, metadata)

        # Shutdown (should wait for all writes)
        writer.shutdown()

        # Assert all 3 .pt files exist
        for i in range(3):
            checkpoint_path = tmp_path / f"batch_{i:05d}.pt"
            assert checkpoint_path.exists()

    def test_writer_async_failure_propagates(self, tmp_path):
        """Test async write failures propagate via wait_all()."""
        writer = AsyncCheckpointWriter()

        # Mock torch.save to fail on second call
        call_count = [0]

        def failing_save(obj, path):
            call_count[0] += 1
            if call_count[0] == 2:
                raise IOError("Simulated I/O error")
            # First call succeeds
            torch.save(obj, path)

        with patch('virnucpro.pipeline.checkpoint_writer.torch.save', side_effect=failing_save):
            # Queue 2 writes
            checkpoint_path1 = tmp_path / "batch_00000.pt"
            checkpoint_path2 = tmp_path / "batch_00001.pt"

            embeddings1 = np.random.randn(3, 16)
            embeddings2 = np.random.randn(3, 16)
            sequence_ids = ['seq_0', 'seq_1', 'seq_2']
            metadata = {'batch_idx': 0}

            writer.write_checkpoint_async(checkpoint_path1, embeddings1, sequence_ids, metadata)
            writer.write_checkpoint_async(checkpoint_path2, embeddings2, sequence_ids, metadata)

            # wait_all() should raise RuntimeError with aggregated errors
            with pytest.raises(RuntimeError) as exc_info:
                writer.wait_all()

            # Verify error message contains failure info
            error_message = str(exc_info.value)
            assert "checkpoint write(s) failed" in error_message.lower()

        writer.shutdown()

    def test_writer_has_pending_returns_true_before_completion(self, tmp_path):
        """Test has_pending() returns True before completion, False after."""
        writer = AsyncCheckpointWriter()
        checkpoint_path = tmp_path / "batch_00000.pt"

        # Before any writes
        assert writer.has_pending() is False

        # Queue write
        embeddings = np.random.randn(5, 32)
        sequence_ids = [f"seq_{i}" for i in range(5)]
        metadata = {'batch_idx': 0}

        writer.write_checkpoint_async(checkpoint_path, embeddings, sequence_ids, metadata)

        # Immediately check - should be pending
        # Note: There's a race here, but with ThreadPoolExecutor the future is queued
        # before returning, so has_pending should be True
        assert writer.has_pending() is True

        # Wait for completion
        writer.wait_all()

        # After completion
        assert writer.has_pending() is False

        writer.shutdown()


class TestValidateCheckpointPt:
    """Test validate_checkpoint_pt function."""

    def test_validate_empty_file_fails(self, tmp_path):
        """Test validation fails for 0-byte file."""
        checkpoint_path = tmp_path / "empty.pt"
        checkpoint_path.touch()  # Create 0-byte file

        is_valid, error_msg = validate_checkpoint_pt(checkpoint_path)

        assert is_valid is False
        assert "0 bytes" in error_msg

    def test_validate_no_done_marker_fails(self, tmp_path):
        """Test validation fails when .done marker missing."""
        checkpoint_path = tmp_path / "batch_00000.pt"

        # Create valid .pt file but no .done marker
        checkpoint_dict = {
            'embeddings': np.random.randn(5, 32),
            'sequence_ids': [f"seq_{i}" for i in range(5)],
            'metadata': {'batch_idx': 0}
        }
        torch.save(checkpoint_dict, checkpoint_path)

        # Don't create .done marker

        is_valid, error_msg = validate_checkpoint_pt(checkpoint_path)

        assert is_valid is False
        assert "missing .done marker" in error_msg.lower()

    def test_validate_corrupt_torch_file_fails(self, tmp_path):
        """Test validation fails for corrupted pickle file."""
        checkpoint_path = tmp_path / "corrupt.pt"

        # Write garbage bytes
        with open(checkpoint_path, 'wb') as f:
            f.write(b"this is not a valid pickle file")

        # Create .done marker
        done_marker = checkpoint_path.with_suffix('.pt.done')
        done_marker.touch()

        is_valid, error_msg = validate_checkpoint_pt(checkpoint_path)

        assert is_valid is False
        assert "failed to load" in error_msg.lower()

    def test_validate_missing_keys_fails(self, tmp_path):
        """Test validation fails when required keys missing."""
        checkpoint_path = tmp_path / "missing_keys.pt"

        # Save checkpoint with only embeddings (missing sequence_ids)
        checkpoint_dict = {
            'embeddings': np.random.randn(5, 32)
        }
        torch.save(checkpoint_dict, checkpoint_path)

        # Create .done marker
        done_marker = checkpoint_path.with_suffix('.pt.done')
        done_marker.touch()

        is_valid, error_msg = validate_checkpoint_pt(checkpoint_path)

        assert is_valid is False
        assert "missing" in error_msg.lower()
        assert "sequence_ids" in error_msg.lower()

    def test_validate_shape_mismatch_fails(self, tmp_path):
        """Test validation fails when embeddings/sequence_ids count mismatch."""
        checkpoint_path = tmp_path / "shape_mismatch.pt"

        # Save checkpoint where counts don't match
        checkpoint_dict = {
            'embeddings': np.random.randn(10, 64),  # 10 embeddings
            'sequence_ids': [f"seq_{i}" for i in range(5)],  # Only 5 IDs
            'metadata': {'batch_idx': 0}
        }
        torch.save(checkpoint_dict, checkpoint_path)

        # Create .done marker
        done_marker = checkpoint_path.with_suffix('.pt.done')
        done_marker.touch()

        is_valid, error_msg = validate_checkpoint_pt(checkpoint_path)

        assert is_valid is False
        assert "shape mismatch" in error_msg.lower()
        assert "10" in error_msg  # 10 embeddings
        assert "5" in error_msg   # 5 sequence_ids

    def test_validate_valid_checkpoint_passes(self, tmp_path):
        """Test validation passes for valid checkpoint."""
        # Use AsyncCheckpointWriter to create a valid checkpoint
        writer = AsyncCheckpointWriter()
        checkpoint_path = tmp_path / "valid.pt"

        embeddings = np.random.randn(8, 128)
        sequence_ids = [f"seq_{i}" for i in range(8)]
        metadata = {'batch_idx': 0, 'num_sequences': 8}

        writer.write_checkpoint_async(checkpoint_path, embeddings, sequence_ids, metadata)
        writer.wait_all()
        writer.shutdown()

        # Validate
        is_valid, error_msg = validate_checkpoint_pt(checkpoint_path)

        assert is_valid is True
        assert error_msg == ""


class TestResumeFromCheckpoints:
    """Test resume_from_checkpoints function."""

    def test_resume_no_checkpoints_returns_empty_4tuple(self, tmp_path):
        """Test resume returns empty 4-tuple when no checkpoints exist."""
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()

        all_ids, embeddings, resume_batch_idx, corrupted_ids = resume_from_checkpoints(
            checkpoint_dir,
            rank=0
        )

        assert all_ids == []
        assert embeddings is None
        assert resume_batch_idx == 0
        assert corrupted_ids == []

    def test_resume_force_restart_ignores_checkpoints(self, tmp_path):
        """Test force_restart=True ignores existing checkpoints."""
        checkpoint_dir = tmp_path / "checkpoints"
        shard_dir = checkpoint_dir / "shard_0"
        shard_dir.mkdir(parents=True)

        # Create a valid checkpoint
        writer = AsyncCheckpointWriter()
        checkpoint_path = shard_dir / "batch_00000.pt"
        embeddings = np.random.randn(5, 32)
        sequence_ids = [f"seq_{i}" for i in range(5)]
        metadata = {'batch_idx': 0, 'num_sequences': 5, 'timestamp': '2026-02-06T00:00:00'}
        writer.write_checkpoint_async(checkpoint_path, embeddings, sequence_ids, metadata)
        writer.wait_all()
        writer.shutdown()

        # Resume with force_restart
        all_ids, embeddings_out, resume_batch_idx, corrupted_ids = resume_from_checkpoints(
            checkpoint_dir,
            rank=0,
            force_restart=True
        )

        # Should ignore checkpoint and return empty
        assert all_ids == []
        assert embeddings_out is None
        assert resume_batch_idx == 0
        assert corrupted_ids == []

    def test_resume_loads_valid_pt_checkpoints_in_order(self, tmp_path):
        """Test resume loads multiple checkpoints in correct order."""
        checkpoint_dir = tmp_path / "checkpoints"
        shard_dir = checkpoint_dir / "shard_0"
        shard_dir.mkdir(parents=True)

        writer = AsyncCheckpointWriter()

        # Write batch_00000.pt
        checkpoint_path0 = shard_dir / "batch_00000.pt"
        embeddings0 = np.array([[1.0, 2.0], [3.0, 4.0]])
        sequence_ids0 = ['seq_0', 'seq_1']
        metadata0 = {'batch_idx': 0, 'num_sequences': 2, 'timestamp': '2026-02-06T00:00:00'}
        writer.write_checkpoint_async(checkpoint_path0, embeddings0, sequence_ids0, metadata0)

        # Write batch_00001.pt
        checkpoint_path1 = shard_dir / "batch_00001.pt"
        embeddings1 = np.array([[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]])
        sequence_ids1 = ['seq_2', 'seq_3', 'seq_4']
        metadata1 = {'batch_idx': 1, 'num_sequences': 3, 'timestamp': '2026-02-06T00:01:00'}
        writer.write_checkpoint_async(checkpoint_path1, embeddings1, sequence_ids1, metadata1)

        writer.wait_all()
        writer.shutdown()

        # Resume
        all_ids, embeddings_out, resume_batch_idx, corrupted_ids = resume_from_checkpoints(
            checkpoint_dir,
            rank=0
        )

        # Verify concatenation
        assert all_ids == ['seq_0', 'seq_1', 'seq_2', 'seq_3', 'seq_4']
        assert embeddings_out.shape == (5, 2)
        assert resume_batch_idx == 2  # last_valid_batch + 1 = 1 + 1
        assert corrupted_ids == []

        # Verify data integrity
        assert embeddings_out[0, 0] == 1.0
        assert embeddings_out[4, 1] == 10.0

    def test_resume_stops_at_corrupted_checkpoint_and_returns_corrupted_ids(self, tmp_path):
        """Test resume stops at corruption and returns IDs for requeue."""
        checkpoint_dir = tmp_path / "checkpoints"
        shard_dir = checkpoint_dir / "shard_0"
        shard_dir.mkdir(parents=True)

        writer = AsyncCheckpointWriter()

        # Write 3 valid checkpoints
        for i in range(3):
            checkpoint_path = shard_dir / f"batch_{i:05d}.pt"
            embeddings = np.random.randn(3, 16)
            sequence_ids = [f"seq_{i*3 + j}" for j in range(3)]
            metadata = {
                'batch_idx': i,
                'num_sequences': 3,
                'timestamp': f'2026-02-06T00:{i:02d}:00'
            }
            writer.write_checkpoint_async(checkpoint_path, embeddings, sequence_ids, metadata)

        writer.wait_all()
        writer.shutdown()

        # Corrupt the middle checkpoint (remove .done marker)
        corrupt_checkpoint = shard_dir / "batch_00001.pt"
        done_marker = corrupt_checkpoint.with_suffix('.pt.done')
        done_marker.unlink()

        # Resume
        all_ids, embeddings_out, resume_batch_idx, corrupted_ids = resume_from_checkpoints(
            checkpoint_dir,
            rank=0
        )

        # Should load only first checkpoint
        assert len(all_ids) == 3  # Only batch 0
        assert all_ids == ['seq_0', 'seq_1', 'seq_2']
        assert embeddings_out.shape == (3, 16)
        assert resume_batch_idx == 1  # last_valid_batch (0) + 1

        # Corrupted IDs should include batch 1 and 2
        assert len(corrupted_ids) == 6
        assert 'seq_3' in corrupted_ids  # From batch 1
        assert 'seq_8' in corrupted_ids  # From batch 2

    def test_resume_data_integrity_shapes_align(self, tmp_path):
        """Test resume concatenation maintains shape integrity."""
        checkpoint_dir = tmp_path / "checkpoints"
        shard_dir = checkpoint_dir / "shard_0"
        shard_dir.mkdir(parents=True)

        writer = AsyncCheckpointWriter()

        # Write checkpoints with different sequence counts
        checkpoint_path0 = shard_dir / "batch_00000.pt"
        embeddings0 = np.random.randn(5, 128)
        sequence_ids0 = [f"seq_{i}" for i in range(5)]
        metadata0 = {'batch_idx': 0, 'num_sequences': 5, 'timestamp': '2026-02-06T00:00:00'}
        writer.write_checkpoint_async(checkpoint_path0, embeddings0, sequence_ids0, metadata0)

        checkpoint_path1 = shard_dir / "batch_00001.pt"
        embeddings1 = np.random.randn(3, 128)
        sequence_ids1 = [f"seq_{i}" for i in range(5, 8)]
        metadata1 = {'batch_idx': 1, 'num_sequences': 3, 'timestamp': '2026-02-06T00:01:00'}
        writer.write_checkpoint_async(checkpoint_path1, embeddings1, sequence_ids1, metadata1)

        writer.wait_all()
        writer.shutdown()

        # Resume
        all_ids, embeddings_out, resume_batch_idx, corrupted_ids = resume_from_checkpoints(
            checkpoint_dir,
            rank=0
        )

        # Verify shape alignment
        assert embeddings_out.shape == (8, 128)  # 5 + 3 sequences
        assert len(all_ids) == 8
        assert corrupted_ids == []

    def test_resume_data_integrity_no_corruption(self, tmp_path):
        """Test resume preserves exact data values through save/load cycle."""
        checkpoint_dir = tmp_path / "checkpoints"
        shard_dir = checkpoint_dir / "shard_0"
        shard_dir.mkdir(parents=True)

        writer = AsyncCheckpointWriter()

        # Use known data values
        known_embeddings = np.arange(30).reshape(10, 3).astype(np.float32)
        sequence_ids = [f"seq_{i}" for i in range(10)]
        metadata = {'batch_idx': 0, 'num_sequences': 10, 'timestamp': '2026-02-06T00:00:00'}

        checkpoint_path = shard_dir / "batch_00000.pt"
        writer.write_checkpoint_async(checkpoint_path, known_embeddings, sequence_ids, metadata)
        writer.wait_all()
        writer.shutdown()

        # Resume
        all_ids, embeddings_out, resume_batch_idx, corrupted_ids = resume_from_checkpoints(
            checkpoint_dir,
            rank=0
        )

        # Verify exact data match
        np.testing.assert_array_equal(embeddings_out, known_embeddings)
        assert all_ids == sequence_ids
        assert resume_batch_idx == 1
        assert corrupted_ids == []
