"""Per-worker logging infrastructure for multi-GPU coordination.

Each GPU worker writes to separate log file to avoid log interleaving
and enable debugging of individual worker failures.
"""

import logging
from datetime import datetime
from pathlib import Path


def setup_worker_logging(
    rank: int,
    log_dir: Path,
    log_level: int = logging.INFO
) -> Path:
    """
    Configure per-worker logging to separate files.

    Each worker gets: {log_dir}/worker_{rank}.log

    Args:
        rank: Worker rank (0, 1, 2, ...)
        log_dir: Directory for log files
        log_level: Logging level (default: INFO)

    Returns:
        Path to the worker's log file
    """
    # Create log directory if needed
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"worker_{rank}.log"

    # Get root logger
    logger = logging.getLogger()

    # Remove any existing handlers (prevent duplicate logging)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create file handler with APPEND mode (preserves previous run logs)
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setLevel(log_level)

    # Format includes worker rank for easy identification
    formatter = logging.Formatter(
        f'%(asctime)s - Worker {rank} - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)

    # Create console handler for immediate feedback (warnings/errors only)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)  # Only warnings/errors to console
    console_handler.setFormatter(formatter)

    # Add both handlers to root logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.setLevel(log_level)

    # Add timestamp separator for resume detection
    # If file exists (not first write), log separator to distinguish runs
    if log_file.stat().st_size > 0:
        timestamp = datetime.now().isoformat()
        logger.info(f"=== Resume at {timestamp} ===")

    # Log initialization
    logger.info(f"Worker {rank} logging initialized: {log_file}")

    return log_file


def get_worker_log_path(log_dir: Path, rank: int) -> Path:
    """
    Get log file path for worker without setting up logging.

    Used by parent process to find worker logs after completion.

    Args:
        log_dir: Directory for log files
        rank: Worker rank

    Returns:
        Path to the worker's log file
    """
    return log_dir / f"worker_{rank}.log"
