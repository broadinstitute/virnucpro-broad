"""Centralized logging configuration for VirNucPro"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


def setup_logging(
    verbose: bool = False,
    log_file: Optional[Path] = None,
    quiet: bool = False
) -> logging.Logger:
    """
    Configure logging for the application.

    Args:
        verbose: Enable DEBUG level logging
        log_file: Optional file path for log output
        quiet: Suppress console output (file logging only)

    Returns:
        Configured logger instance
    """
    # Determine log level
    if verbose:
        log_level = logging.DEBUG
    elif quiet:
        log_level = logging.WARNING
    else:
        log_level = logging.INFO

    # Create formatter
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(log_format, datefmt='%Y-%m-%d %H:%M:%S')

    # Configure root logger
    logger = logging.getLogger('virnucpro')
    logger.setLevel(log_level)
    logger.handlers = []  # Clear any existing handlers

    # Console handler (unless quiet)
    if not quiet:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)  # Always debug level to file
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        if not quiet:
            logger.info(f"Logging to file: {log_file}")

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module.

    Args:
        name: Module name (typically __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(f'virnucpro.{name}')
