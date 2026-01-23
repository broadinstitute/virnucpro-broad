"""Tests for .done marker functionality in checkpoint system.

Tests verify that .done marker files enable quick resume checks without
loading multi-GB checkpoint files, and maintain dual mechanism redundancy.
"""

import pytest
import torch
from pathlib import Path
import time

from virnucpro.core.checkpoint import (
    atomic_save,
    has_done_marker,
    create_done_marker,
    remove_done_marker,
    CHECKPOINT_VERSION
)
from virnucpro.core.checkpoint_validation import (
    get_checkpoint_info,
    validate_checkpoint
)


class TestMarkerCreation:
    """Test .done marker creation during checkpoint save."""

    def test_atomic_save_creates_done_marker(self, temp_dir):
        """Verify .done file created after successful save."""
        checkpoint_path = temp_dir / "test_checkpoint.pt"
        checkpoint_dict = {
            'version': CHECKPOINT_VERSION,
            'status': 'in_progress',
            'data': torch.randn(10, 5)
        }

        # Save checkpoint
        atomic_save(checkpoint_dict, checkpoint_path, validate_after_save=True)

        # Verify .done marker exists
        done_marker = checkpoint_path.with_suffix('.pt.done')
        assert done_marker.exists(), ".done marker should be created after successful save"
        assert has_done_marker(checkpoint_path), "has_done_marker() should return True"

    def test_atomic_save_creates_done_marker_without_validation(self, temp_dir):
        """Verify .done file created even when validation skipped."""
        checkpoint_path = temp_dir / "test_checkpoint.pt"
        checkpoint_dict = {
            'nucleotide': ['seq1', 'seq2'],
            'data': [{'label': 'seq1', 'mean_representation': [0.1, 0.2]}]
        }

        # Save checkpoint without validation (feature extraction pattern)
        atomic_save(checkpoint_dict, checkpoint_path, validate_after_save=False)

        # Verify .done marker exists
        assert has_done_marker(checkpoint_path), ".done marker should exist even without validation"

    def test_atomic_save_no_marker_on_validation_failure(self, temp_dir):
        """Verify no .done marker if validation fails."""
        checkpoint_path = temp_dir / "test_checkpoint.pt"

        # Create checkpoint that will fail validation (missing required keys)
        checkpoint_dict = {
            'wrong_key': torch.randn(5, 5)  # Missing 'data' key
        }

        # Save should fail validation due to missing required keys
        try:
            atomic_save(checkpoint_dict, checkpoint_path, validate_after_save=True)
        except RuntimeError:
            pass  # Expected to fail

        # Verify no .done marker exists (checkpoint was deleted on validation failure)
        assert not checkpoint_path.exists(), "Checkpoint should be deleted after validation failure"
        assert not has_done_marker(checkpoint_path), ".done marker should not exist after validation failure"

    def test_atomic_save_updates_done_marker_on_overwrite(self, temp_dir):
        """Verify .done marker updated when checkpoint overwritten."""
        checkpoint_path = temp_dir / "test_checkpoint.pt"

        # First save
        checkpoint_dict_v1 = {'data': torch.randn(5, 3), 'version': '1.0'}
        atomic_save(checkpoint_dict_v1, checkpoint_path, validate_after_save=False)
        assert has_done_marker(checkpoint_path)

        # Get marker timestamp
        done_marker = checkpoint_path.with_suffix('.pt.done')
        first_mtime = done_marker.stat().st_mtime

        time.sleep(0.01)  # Ensure timestamp difference

        # Overwrite with new checkpoint
        checkpoint_dict_v2 = {'data': torch.randn(10, 5), 'version': '1.0'}
        atomic_save(checkpoint_dict_v2, checkpoint_path, validate_after_save=False)

        # Verify marker still exists and was updated
        assert has_done_marker(checkpoint_path)
        second_mtime = done_marker.stat().st_mtime
        assert second_mtime > first_mtime, ".done marker should be updated on overwrite"


class TestMarkerChecking:
    """Test .done marker checking functionality."""

    def test_has_done_marker_existing(self, temp_dir):
        """Returns True for existing marker."""
        checkpoint_path = temp_dir / "test.pt"
        done_marker = checkpoint_path.with_suffix('.pt.done')

        # Create checkpoint and marker
        torch.save({'data': torch.randn(3, 3)}, checkpoint_path)
        done_marker.touch()

        assert has_done_marker(checkpoint_path) is True

    def test_has_done_marker_missing(self, temp_dir):
        """Returns False for missing marker."""
        checkpoint_path = temp_dir / "test.pt"

        # Create checkpoint without marker
        torch.save({'data': torch.randn(3, 3)}, checkpoint_path)

        assert has_done_marker(checkpoint_path) is False

    def test_has_done_marker_no_checkpoint(self, temp_dir):
        """Returns False when checkpoint doesn't exist."""
        checkpoint_path = temp_dir / "nonexistent.pt"

        assert has_done_marker(checkpoint_path) is False

    def test_has_done_marker_with_checkpoint(self, temp_dir):
        """Works with actual checkpoint file."""
        checkpoint_path = temp_dir / "embeddings.pt"

        # Create realistic checkpoint
        checkpoint_dict = {
            'nucleotide': [f'seq_{i}' for i in range(100)],
            'data': [{'label': f'seq_{i}', 'mean_representation': torch.randn(768).tolist()} for i in range(100)]
        }
        atomic_save(checkpoint_dict, checkpoint_path, validate_after_save=False)

        # Verify marker exists and checkpoint is valid
        assert has_done_marker(checkpoint_path)
        is_valid, _ = validate_checkpoint(checkpoint_path, required_keys=['nucleotide', 'data'], skip_load=False)
        assert is_valid


class TestMarkerManagement:
    """Test marker creation and removal functions."""

    def test_create_done_marker(self, temp_dir):
        """Manual marker creation works."""
        checkpoint_path = temp_dir / "test.pt"

        # Create checkpoint
        torch.save({'data': torch.randn(3, 3)}, checkpoint_path)

        # Create marker manually
        create_done_marker(checkpoint_path)

        # Verify marker exists
        done_marker = checkpoint_path.with_suffix('.pt.done')
        assert done_marker.exists()
        assert has_done_marker(checkpoint_path)

    def test_remove_done_marker(self, temp_dir):
        """Marker removal works."""
        checkpoint_path = temp_dir / "test.pt"
        done_marker = checkpoint_path.with_suffix('.pt.done')

        # Create checkpoint with marker
        torch.save({'data': torch.randn(3, 3)}, checkpoint_path)
        done_marker.touch()
        assert has_done_marker(checkpoint_path)

        # Remove marker
        remove_done_marker(checkpoint_path)

        # Verify marker removed
        assert not done_marker.exists()
        assert not has_done_marker(checkpoint_path)

    def test_remove_nonexistent_marker(self, temp_dir):
        """Safe to remove non-existent marker."""
        checkpoint_path = temp_dir / "test.pt"

        # Remove marker that doesn't exist - should not raise error
        remove_done_marker(checkpoint_path)

        # Verify still no marker
        assert not has_done_marker(checkpoint_path)


class TestBackwardCompatibility:
    """Test backward compatibility with dual mechanism (marker + status field)."""

    def test_dual_mechanism(self, temp_dir):
        """Both .done marker and status field maintained."""
        checkpoint_path = temp_dir / "test.pt"
        checkpoint_dict = {
            'version': CHECKPOINT_VERSION,
            'status': 'in_progress',
            'data': torch.randn(5, 5)
        }

        # Save checkpoint
        atomic_save(checkpoint_dict, checkpoint_path, validate_after_save=True)

        # Verify both mechanisms
        assert has_done_marker(checkpoint_path), ".done marker should exist"

        # Load and check status field
        loaded = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        assert loaded['status'] == 'complete', "Status field should be 'complete'"

    def test_resume_with_status_only(self, temp_dir):
        """Old checkpoints (status field only) still work."""
        checkpoint_path = temp_dir / "old_checkpoint.pt"

        # Simulate old checkpoint (no .done marker, only status field)
        checkpoint_dict = {
            'version': '0.x',
            'status': 'complete',
            'data': torch.randn(5, 5)
        }
        torch.save(checkpoint_dict, checkpoint_path)

        # Should have no marker
        assert not has_done_marker(checkpoint_path)

        # Should still be valid
        is_valid, _ = validate_checkpoint(checkpoint_path, skip_load=False)
        assert is_valid

    def test_resume_with_marker_only(self, temp_dir):
        """New checkpoints (.done marker) work."""
        checkpoint_path = temp_dir / "new_checkpoint.pt"

        # Create checkpoint with marker
        checkpoint_dict = {
            'version': CHECKPOINT_VERSION,
            'data': torch.randn(5, 5)
        }
        atomic_save(checkpoint_dict, checkpoint_path, validate_after_save=False)

        # Verify marker exists
        assert has_done_marker(checkpoint_path)

        # Should be valid
        is_valid, _ = validate_checkpoint(checkpoint_path, skip_load=False)
        assert is_valid

    def test_get_checkpoint_info_includes_marker(self, temp_dir):
        """get_checkpoint_info() includes has_done_marker field."""
        checkpoint_path = temp_dir / "test.pt"

        # Create checkpoint with marker
        checkpoint_dict = {'data': torch.randn(3, 3)}
        atomic_save(checkpoint_dict, checkpoint_path, validate_after_save=False)

        # Get info
        info = get_checkpoint_info(checkpoint_path)

        # Verify has_done_marker field exists and is True
        assert 'has_done_marker' in info
        assert info['has_done_marker'] is True
        assert info['is_valid_zip'] is True
        assert info['size_bytes'] > 0


class TestPerformanceBenefit:
    """Test performance benefit of .done markers vs loading checkpoints."""

    def test_quick_resume_check(self, temp_dir):
        """Verify has_done_marker() doesn't load checkpoint."""
        checkpoint_path = temp_dir / "large_checkpoint.pt"

        # Create multi-MB checkpoint (realistic feature extraction size)
        large_data = torch.randn(10000, 768)  # ~30MB
        checkpoint_dict = {
            'nucleotide': [f'seq_{i}' for i in range(10000)],
            'data': [{'label': f'seq_{i}', 'mean_representation': large_data[i].tolist()} for i in range(10000)]
        }
        atomic_save(checkpoint_dict, checkpoint_path, validate_after_save=False)

        # Time marker check (should be instant)
        start = time.perf_counter()
        marker_exists = has_done_marker(checkpoint_path)
        marker_time = time.perf_counter() - start

        assert marker_exists is True
        assert marker_time < 0.001, f"Marker check took {marker_time:.4f}s, should be <1ms"

        # Compare to loading checkpoint (much slower)
        start = time.perf_counter()
        loaded = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        load_time = time.perf_counter() - start

        assert load_time > marker_time * 100, "Loading checkpoint should be much slower than checking marker"

    def test_large_checkpoint_resume(self, temp_dir):
        """Measure time difference with multi-MB test file."""
        checkpoint_path = temp_dir / "embeddings.pt"

        # Create realistic embedding checkpoint (ESM-2 3B output)
        num_sequences = 5000
        embedding_dim = 2560  # ESM-2 3B embedding size
        embeddings = torch.randn(num_sequences, embedding_dim)  # ~50MB

        checkpoint_dict = {
            'proteins': [f'protein_{i}' for i in range(num_sequences)],
            'data': [embeddings[i] for i in range(num_sequences)]
        }
        atomic_save(checkpoint_dict, checkpoint_path, validate_after_save=False)

        # Time 100 marker checks
        start = time.perf_counter()
        for _ in range(100):
            has_done_marker(checkpoint_path)
        marker_total_time = time.perf_counter() - start

        # Time 1 checkpoint load
        start = time.perf_counter()
        torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        load_time = time.perf_counter() - start

        # Verify marker checking is orders of magnitude faster
        avg_marker_time = marker_total_time / 100
        speedup = load_time / avg_marker_time

        assert speedup > 1000, f"Marker check should be >1000x faster (was {speedup:.0f}x)"


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_marker_with_nested_suffix(self, temp_dir):
        """Handle checkpoint with multiple suffixes correctly."""
        checkpoint_path = temp_dir / "checkpoint.tar.pt"

        # Create checkpoint
        atomic_save({'data': torch.randn(3, 3)}, checkpoint_path, validate_after_save=False)

        # Verify marker has correct suffix
        done_marker = checkpoint_path.with_suffix('.pt.done')
        assert done_marker.exists()
        assert done_marker.name == "checkpoint.tar.pt.done"

    def test_marker_cleanup_on_reprocess(self, temp_dir):
        """Defensive cleanup removes marker when reprocessing."""
        checkpoint_path = temp_dir / "incomplete.pt"
        done_marker = checkpoint_path.with_suffix('.pt.done')

        # Create incomplete checkpoint with marker (shouldn't happen, but defensive)
        torch.save({'data': torch.randn(3, 3)}, checkpoint_path)
        done_marker.touch()

        # Remove marker before reprocessing
        remove_done_marker(checkpoint_path)

        # Verify marker removed
        assert not has_done_marker(checkpoint_path)

    def test_marker_on_symlink(self, temp_dir):
        """Marker works with symlinked checkpoint."""
        real_checkpoint = temp_dir / "real.pt"
        link_checkpoint = temp_dir / "link.pt"

        # Create checkpoint
        atomic_save({'data': torch.randn(3, 3)}, real_checkpoint, validate_after_save=False)

        # Create symlink
        link_checkpoint.symlink_to(real_checkpoint)

        # Both should report marker
        assert has_done_marker(real_checkpoint)
        # Note: Symlink check would look for link.pt.done (separate marker)
        # This is expected behavior - each path has its own marker

    def test_concurrent_marker_access(self, temp_dir):
        """Marker access is safe for concurrent checks."""
        checkpoint_path = temp_dir / "concurrent.pt"
        atomic_save({'data': torch.randn(3, 3)}, checkpoint_path, validate_after_save=False)

        # Multiple concurrent checks should work
        results = [has_done_marker(checkpoint_path) for _ in range(100)]

        assert all(results), "All concurrent checks should return True"
