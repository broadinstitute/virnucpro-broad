"""Tests for checkpoint validation utilities.

This test suite covers multi-level checkpoint validation:
- File size checks (0-byte detection)
- ZIP format validation
- PyTorch load validation
- Required keys validation
- Error type distinction (corrupted vs incompatible)
- Batch validation
- Failed checkpoint logging
"""

import pytest
import torch
import zipfile
from pathlib import Path
import tempfile
import shutil

from virnucpro.core.checkpoint_validation import (
    validate_checkpoint,
    validate_checkpoint_batch,
    distinguish_error_type,
    log_failed_checkpoint,
    load_failed_checkpoints,
    CheckpointError,
    CHECKPOINT_EXIT_CODE
)


@pytest.fixture
def temp_checkpoint_dir():
    """Create temporary directory for test checkpoints."""
    tmpdir = tempfile.mkdtemp()
    yield Path(tmpdir)
    shutil.rmtree(tmpdir)


@pytest.fixture
def valid_checkpoint(temp_checkpoint_dir):
    """Create a valid checkpoint file for testing."""
    checkpoint_path = temp_checkpoint_dir / "valid_checkpoint.pt"
    checkpoint_data = {
        'version': '1.0',
        'status': 'complete',
        'data': torch.randn(10, 5),
        'metadata': {'test': True}
    }
    torch.save(checkpoint_data, checkpoint_path)
    return checkpoint_path


class TestValidateCheckpoint:
    """Test validate_checkpoint() function with various corruption types."""

    def test_validate_empty_file(self, temp_checkpoint_dir):
        """Test detection of 0-byte checkpoint file."""
        empty_file = temp_checkpoint_dir / "empty.pt"
        empty_file.touch()  # Create 0-byte file

        is_valid, error_msg = validate_checkpoint(empty_file)

        assert not is_valid
        assert "corrupted: file is 0 bytes" in error_msg

    def test_validate_non_zip_file(self, temp_checkpoint_dir):
        """Test detection of non-ZIP file (invalid format)."""
        text_file = temp_checkpoint_dir / "text.pt"
        with open(text_file, 'w') as f:
            f.write("This is not a ZIP file")

        is_valid, error_msg = validate_checkpoint(text_file)

        assert not is_valid
        assert "corrupted: not a valid ZIP archive" in error_msg

    def test_validate_corrupted_torch_file(self, temp_checkpoint_dir):
        """Test detection of broken pickle/torch file (corrupted ZIP)."""
        corrupted_file = temp_checkpoint_dir / "corrupted.pt"

        # Create a ZIP file with invalid contents
        with zipfile.ZipFile(corrupted_file, 'w') as zf:
            zf.writestr('data.pkl', b'invalid pickle data')

        is_valid, error_msg = validate_checkpoint(corrupted_file, skip_load=False)

        assert not is_valid
        assert "corrupted: torch.load failed" in error_msg

    def test_validate_missing_required_keys(self, temp_checkpoint_dir):
        """Test validation fails when required keys are missing."""
        checkpoint_path = temp_checkpoint_dir / "missing_keys.pt"
        checkpoint_data = {
            'version': '1.0',
            'status': 'complete'
            # Missing 'data' key
        }
        torch.save(checkpoint_data, checkpoint_path)

        is_valid, error_msg = validate_checkpoint(
            checkpoint_path,
            required_keys=['data'],
            skip_load=False
        )

        assert not is_valid
        assert "incompatible: missing required keys" in error_msg
        assert "'data'" in error_msg

    def test_validate_valid_checkpoint(self, valid_checkpoint):
        """Test successful validation of valid checkpoint."""
        is_valid, error_msg = validate_checkpoint(
            valid_checkpoint,
            required_keys=['data'],
            skip_load=False
        )

        assert is_valid
        assert error_msg == ""

    def test_validate_skip_load(self, valid_checkpoint):
        """Test validation with skip_load=True (faster, less thorough)."""
        is_valid, error_msg = validate_checkpoint(
            valid_checkpoint,
            skip_load=True
        )

        # Should pass file size and ZIP format checks
        assert is_valid
        assert error_msg == ""

    def test_validate_nonexistent_file(self, temp_checkpoint_dir):
        """Test validation of file that doesn't exist."""
        nonexistent = temp_checkpoint_dir / "does_not_exist.pt"

        is_valid, error_msg = validate_checkpoint(nonexistent)

        assert not is_valid
        assert "corrupted: file does not exist" in error_msg


class TestErrorTypeDistinction:
    """Test error type distinction between corrupted and incompatible."""

    def test_distinguish_corrupted_vs_incompatible(self):
        """Test that error messages are categorized correctly."""
        # Corrupted errors
        assert distinguish_error_type("corrupted: file is 0 bytes") == "corrupted"
        assert distinguish_error_type("corrupted: not a valid ZIP archive") == "corrupted"
        assert distinguish_error_type("corrupted: torch.load failed") == "corrupted"

        # Incompatible errors
        assert distinguish_error_type("incompatible: missing required keys") == "incompatible"
        assert distinguish_error_type("incompatible: version mismatch") == "incompatible"

        # Unknown errors default to corrupted
        assert distinguish_error_type("unknown error") == "corrupted"


class TestBatchValidation:
    """Test batch validation of multiple checkpoints."""

    def test_validate_checkpoint_batch(self, temp_checkpoint_dir):
        """Test validation of mixed valid and invalid checkpoints."""
        # Create mix of valid and invalid checkpoints
        valid1 = temp_checkpoint_dir / "valid1.pt"
        valid2 = temp_checkpoint_dir / "valid2.pt"
        empty = temp_checkpoint_dir / "empty.pt"
        text = temp_checkpoint_dir / "text.pt"

        # Valid checkpoints
        for path in [valid1, valid2]:
            torch.save({'data': torch.randn(5, 3)}, path)

        # Invalid checkpoints
        empty.touch()  # 0-byte file
        with open(text, 'w') as f:
            f.write("not a checkpoint")

        checkpoint_paths = [valid1, valid2, empty, text]

        valid_paths, failed_items = validate_checkpoint_batch(
            checkpoint_paths,
            required_keys=['data'],
            skip_load=False
        )

        # Verify results
        assert len(valid_paths) == 2
        assert valid1 in valid_paths
        assert valid2 in valid_paths

        assert len(failed_items) == 2
        failed_paths = [path for path, _ in failed_items]
        assert empty in failed_paths
        assert text in failed_paths

    def test_validate_empty_batch(self):
        """Test batch validation with empty list."""
        valid_paths, failed_items = validate_checkpoint_batch([])

        assert valid_paths == []
        assert failed_items == []


class TestFailedCheckpointLogging:
    """Test failed checkpoint tracking."""

    def test_log_failed_checkpoint(self, temp_checkpoint_dir):
        """Test logging failed checkpoint to tracking file."""
        checkpoint_path = temp_checkpoint_dir / "failed.pt"
        checkpoint_path.touch()

        log_failed_checkpoint(
            checkpoint_path,
            "corrupted: file is 0 bytes",
            checkpoint_dir=temp_checkpoint_dir,
            timestamp="2026-01-23T12:00:00Z"
        )

        failed_log = temp_checkpoint_dir / "failed_checkpoints.txt"
        assert failed_log.exists()

        # Verify log format
        with open(failed_log, 'r') as f:
            line = f.read().strip()

        assert str(checkpoint_path) in line
        assert "corrupted: file is 0 bytes" in line
        assert "2026-01-23T12:00:00Z" in line
        assert line.count('|') == 2  # Pipe-delimited format

    def test_load_failed_checkpoints(self, temp_checkpoint_dir):
        """Test loading failed checkpoint log."""
        failed_log = temp_checkpoint_dir / "failed_checkpoints.txt"

        # Write sample failed checkpoint entries
        with open(failed_log, 'w') as f:
            f.write("checkpoint1.pt|corrupted: 0 bytes|2026-01-23T12:00:00Z\n")
            f.write("checkpoint2.pt|incompatible: missing keys|2026-01-23T12:05:00Z\n")

        failed_items = load_failed_checkpoints(temp_checkpoint_dir)

        assert len(failed_items) == 2
        assert failed_items[0] == ("checkpoint1.pt", "corrupted: 0 bytes", "2026-01-23T12:00:00Z")
        assert failed_items[1] == ("checkpoint2.pt", "incompatible: missing keys", "2026-01-23T12:05:00Z")

    def test_load_failed_checkpoints_empty(self, temp_checkpoint_dir):
        """Test loading when no failed checkpoints exist."""
        failed_items = load_failed_checkpoints(temp_checkpoint_dir)

        assert failed_items == []

    def test_log_append_multiple(self, temp_checkpoint_dir):
        """Test that logging appends to existing file."""
        checkpoint1 = temp_checkpoint_dir / "failed1.pt"
        checkpoint2 = temp_checkpoint_dir / "failed2.pt"

        log_failed_checkpoint(checkpoint1, "error1", checkpoint_dir=temp_checkpoint_dir)
        log_failed_checkpoint(checkpoint2, "error2", checkpoint_dir=temp_checkpoint_dir)

        failed_items = load_failed_checkpoints(temp_checkpoint_dir)
        assert len(failed_items) == 2


class TestCheckpointError:
    """Test CheckpointError exception."""

    def test_checkpoint_error_creation(self):
        """Test CheckpointError exception creation."""
        error = CheckpointError(
            Path("test.pt"),
            "corrupted",
            "file is 0 bytes"
        )

        assert error.checkpoint_path == Path("test.pt")
        assert error.error_type == "corrupted"
        assert error.message == "file is 0 bytes"
        assert "corrupted" in str(error)
        assert "file is 0 bytes" in str(error)

    def test_checkpoint_exit_code(self):
        """Test that CHECKPOINT_EXIT_CODE is defined correctly."""
        # Exit codes: 0=success, 1=generic, 2=partial, 3=checkpoint
        assert CHECKPOINT_EXIT_CODE == 3


class TestValidationWithLogging:
    """Test validation with automatic failure logging."""

    def test_validate_with_log_failures_enabled(self, temp_checkpoint_dir):
        """Test that validation logs failures when log_failures=True."""
        empty_file = temp_checkpoint_dir / "empty.pt"
        empty_file.touch()

        is_valid, error_msg = validate_checkpoint(
            empty_file,
            log_failures=True
        )

        assert not is_valid

        # Verify failure was logged
        failed_log = temp_checkpoint_dir / "failed_checkpoints.txt"
        assert failed_log.exists()

        failed_items = load_failed_checkpoints(temp_checkpoint_dir)
        assert len(failed_items) == 1
        assert str(empty_file) in failed_items[0][0]

    def test_validate_with_log_failures_disabled(self, temp_checkpoint_dir):
        """Test that validation doesn't log when log_failures=False."""
        empty_file = temp_checkpoint_dir / "empty.pt"
        empty_file.touch()

        is_valid, error_msg = validate_checkpoint(
            empty_file,
            log_failures=False  # Default
        )

        assert not is_valid

        # Verify no log was created
        failed_log = temp_checkpoint_dir / "failed_checkpoints.txt"
        assert not failed_log.exists()
