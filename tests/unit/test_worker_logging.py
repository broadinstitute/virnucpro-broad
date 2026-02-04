"""Unit tests for per-worker logging infrastructure."""

import logging
import tempfile
from pathlib import Path

import pytest

from virnucpro.pipeline.worker_logging import (
    get_worker_log_path,
    setup_worker_logging,
)


@pytest.fixture
def temp_log_dir():
    """Temporary directory for log files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


class TestWorkerLogging:
    """Tests for setup_worker_logging function."""

    def test_creates_log_file(self, temp_log_dir):
        """Test that setup_worker_logging creates worker_0.log."""
        log_file = setup_worker_logging(rank=0, log_dir=temp_log_dir)

        assert log_file.exists()
        assert log_file.name == "worker_0.log"
        assert log_file.parent == temp_log_dir

    def test_log_file_format(self, temp_log_dir, caplog):
        """Test that log messages include worker rank."""
        setup_worker_logging(rank=2, log_dir=temp_log_dir)

        # Log a test message
        logger = logging.getLogger("test")
        with caplog.at_level(logging.INFO):
            logger.info("Test message")

        # Read log file
        log_file = temp_log_dir / "worker_2.log"
        content = log_file.read_text()

        # Verify format includes worker rank
        assert "Worker 2" in content
        assert "Test message" in content

    def test_append_mode(self, temp_log_dir):
        """Test that multiple calls append to same file (file grows)."""
        # First call
        setup_worker_logging(rank=1, log_dir=temp_log_dir)
        logger = logging.getLogger("test1")
        logger.info("First session")

        log_file = temp_log_dir / "worker_1.log"
        first_size = log_file.stat().st_size

        # Second call (simulating resume)
        setup_worker_logging(rank=1, log_dir=temp_log_dir)
        logger = logging.getLogger("test2")
        logger.info("Second session")

        second_size = log_file.stat().st_size

        # File should have grown
        assert second_size > first_size

        # Both messages should be present
        content = log_file.read_text()
        assert "First session" in content
        assert "Second session" in content

    def test_resume_separator(self, temp_log_dir):
        """Test that resume logs separator line."""
        # First call
        setup_worker_logging(rank=3, log_dir=temp_log_dir)

        # Second call (simulating resume)
        setup_worker_logging(rank=3, log_dir=temp_log_dir)

        # Read log file
        log_file = temp_log_dir / "worker_3.log"
        content = log_file.read_text()

        # Verify separator is present
        assert "=== Resume at" in content

    def test_console_handler_level(self, temp_log_dir, caplog):
        """Test that console only receives WARNING and above."""
        setup_worker_logging(rank=0, log_dir=temp_log_dir)

        logger = logging.getLogger("console_test")

        # Clear caplog
        caplog.clear()

        # Log at different levels
        with caplog.at_level(logging.DEBUG):
            logger.debug("Debug message")
            logger.info("Info message")
            logger.warning("Warning message")
            logger.error("Error message")

        # Console handler filters to WARNING+
        # Find console handlers (StreamHandler but not FileHandler)
        root_logger = logging.getLogger()
        console_handlers = [
            h for h in root_logger.handlers
            if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        ]

        assert len(console_handlers) > 0
        for handler in console_handlers:
            assert handler.level == logging.WARNING

    def test_multiple_workers(self, temp_log_dir):
        """Test that ranks 0, 1, 2 create separate files."""
        # Create logging for multiple workers
        log_file_0 = setup_worker_logging(rank=0, log_dir=temp_log_dir)
        log_file_1 = setup_worker_logging(rank=1, log_dir=temp_log_dir)
        log_file_2 = setup_worker_logging(rank=2, log_dir=temp_log_dir)

        # Verify separate files created
        assert log_file_0.name == "worker_0.log"
        assert log_file_1.name == "worker_1.log"
        assert log_file_2.name == "worker_2.log"

        assert log_file_0.exists()
        assert log_file_1.exists()
        assert log_file_2.exists()


class TestGetWorkerLogPath:
    """Tests for get_worker_log_path helper function."""

    def test_get_path_without_setup(self, temp_log_dir):
        """Test that get_worker_log_path returns correct path without side effects."""
        # Call helper without setting up logging
        log_path = get_worker_log_path(temp_log_dir, rank=5)

        # Verify path is correct
        assert log_path == temp_log_dir / "worker_5.log"

        # Verify no file was created (no side effects)
        assert not log_path.exists()

    def test_path_matches_setup(self, temp_log_dir):
        """Test that path matches what setup_worker_logging creates."""
        # Get path from helper
        helper_path = get_worker_log_path(temp_log_dir, rank=4)

        # Create log file with setup
        setup_path = setup_worker_logging(rank=4, log_dir=temp_log_dir)

        # Verify they match
        assert helper_path == setup_path
