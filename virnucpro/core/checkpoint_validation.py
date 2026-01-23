"""Checkpoint validation utilities for PyTorch checkpoints.

This module provides multi-level validation for PyTorch checkpoint files
to detect corruption and incompatibility before they cause pipeline failures.
"""

import zipfile
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from datetime import datetime

import torch

logger = logging.getLogger('virnucpro.checkpoint_validation')

# Checkpoint exit code for pipeline failures
# 0: success, 1: generic failure, 2: partial pipeline, 3: checkpoint issue
CHECKPOINT_EXIT_CODE = 3


class CheckpointError(Exception):
    """Exception raised for checkpoint-specific errors.

    Attributes:
        checkpoint_path: Path to the checkpoint that failed
        error_type: Type of error ('corrupted' or 'incompatible')
        message: Detailed error message
    """

    def __init__(self, checkpoint_path: Path, error_type: str, message: str):
        self.checkpoint_path = checkpoint_path
        self.error_type = error_type
        self.message = message
        super().__init__(f"{error_type}: {message}")


def log_failed_checkpoint(
    checkpoint_path: Path,
    reason: str,
    checkpoint_dir: Optional[Path] = None,
    timestamp: Optional[str] = None
) -> None:
    """Log failed checkpoint to tracking file for diagnostics.

    Appends failed checkpoint information to failed_checkpoints.txt
    in the checkpoint directory for resume operation logging.

    Args:
        checkpoint_path: Path to checkpoint that failed validation
        reason: Error message or reason for failure
        checkpoint_dir: Directory containing checkpoints (default: checkpoint_path.parent)
        timestamp: ISO timestamp (default: current UTC time)

    Example:
        >>> log_failed_checkpoint(
        ...     Path("checkpoints/model.pt"),
        ...     "corrupted: file is 0 bytes"
        ... )
    """
    if checkpoint_dir is None:
        checkpoint_dir = checkpoint_path.parent

    failed_log = checkpoint_dir / "failed_checkpoints.txt"

    if timestamp is None:
        timestamp = datetime.utcnow().isoformat()

    # Format: {checkpoint_path}|{reason}|{timestamp}
    log_entry = f"{checkpoint_path}|{reason}|{timestamp}\n"

    try:
        with open(failed_log, 'a') as f:
            f.write(log_entry)
        logger.debug(f"Logged failed checkpoint: {checkpoint_path.name}")
    except Exception as e:
        logger.warning(f"Failed to log checkpoint failure: {e}")


def load_failed_checkpoints(
    checkpoint_dir: Path
) -> List[Tuple[str, str, str]]:
    """Load list of failed checkpoints from tracking file.

    Reads failed_checkpoints.txt and returns list of failures
    for resume operation logging and diagnostics.

    Args:
        checkpoint_dir: Directory containing checkpoints

    Returns:
        List of (checkpoint_path, reason, timestamp) tuples

    Example:
        >>> failed = load_failed_checkpoints(Path("checkpoints"))
        >>> for path, reason, timestamp in failed:
        ...     print(f"{path}: {reason}")
    """
    failed_log = checkpoint_dir / "failed_checkpoints.txt"

    if not failed_log.exists():
        return []

    failed_items = []
    try:
        with open(failed_log, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split('|', 2)
                    if len(parts) == 3:
                        failed_items.append((parts[0], parts[1], parts[2]))
                    else:
                        logger.warning(f"Malformed failed checkpoint entry: {line}")
    except Exception as e:
        logger.warning(f"Failed to load failed checkpoints: {e}")

    return failed_items


def validate_checkpoint(
    checkpoint_path: Path,
    required_keys: Optional[List[str]] = None,
    skip_load: bool = False,
    logger_instance: Optional[logging.Logger] = None,
    log_failures: bool = False
) -> Tuple[bool, str]:
    """Validate PyTorch checkpoint file integrity with multi-level checks.

    Performs validation at multiple levels:
    - Level 1: File size check (>0 bytes)
    - Level 2: ZIP format validation (PyTorch checkpoints are ZIP archives)
    - Level 3: PyTorch load validation (optional, can be slow for large files)
    - Level 4: Required keys validation (if specified)

    Args:
        checkpoint_path: Path to checkpoint file to validate
        required_keys: List of required keys in checkpoint dict (default: ['data'])
        skip_load: Skip torch.load validation (faster, less thorough)
        logger_instance: Logger instance for diagnostic output
        log_failures: Log validation failures to failed_checkpoints.txt (default: False)

    Returns:
        Tuple of (is_valid, error_message)
        - is_valid: True if checkpoint passes all checks
        - error_message: Empty string if valid, detailed error otherwise

    Example:
        >>> is_valid, error = validate_checkpoint(Path("model.pt"), log_failures=True)
        >>> if not is_valid:
        ...     print(f"Validation failed: {error}")
    """
    log = logger_instance or logger

    if required_keys is None:
        required_keys = ['data']

    # Level 1: File size check (fast)
    try:
        file_size = checkpoint_path.stat().st_size
    except FileNotFoundError:
        error_msg = "corrupted: file does not exist"
        if log_failures:
            log_failed_checkpoint(checkpoint_path, error_msg)
        return False, error_msg
    except Exception as e:
        error_msg = f"corrupted: cannot access file - {str(e)}"
        if log_failures:
            log_failed_checkpoint(checkpoint_path, error_msg)
        return False, error_msg

    if file_size == 0:
        log.error(f"Checkpoint validation failed: {checkpoint_path}")
        log.error(f"  Reason: File is 0 bytes")
        error_msg = "corrupted: file is 0 bytes"
        if log_failures:
            log_failed_checkpoint(checkpoint_path, error_msg)
        return False, error_msg

    log.debug(f"Checkpoint size: {file_size:,} bytes")

    # Level 2: ZIP format check (fast)
    # PyTorch checkpoints are ZIP archives using pickle format
    if not zipfile.is_zipfile(checkpoint_path):
        log.error(f"Checkpoint validation failed: {checkpoint_path}")
        log.error(f"  Reason: Not a valid ZIP archive")
        log.error(f"  File size: {file_size} bytes")
        error_msg = "corrupted: not a valid ZIP archive"
        if log_failures:
            log_failed_checkpoint(checkpoint_path, error_msg)
        return False, error_msg

    log.debug(f"Checkpoint is valid ZIP archive")

    # Level 3: PyTorch load check (slow, optional)
    if not skip_load:
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        except Exception as e:
            log.error(f"Checkpoint load failed: {checkpoint_path}")
            log.error(f"  Error: {str(e)}")
            log.error(f"  File size: {file_size} bytes")
            error_msg = f"corrupted: torch.load failed - {str(e)}"
            if log_failures:
                log_failed_checkpoint(checkpoint_path, error_msg)
            return False, error_msg

        log.debug(f"Checkpoint loaded successfully")

        # Level 4: Required keys validation
        if not isinstance(checkpoint, dict):
            log.error(f"Checkpoint validation failed: {checkpoint_path}")
            log.error(f"  Reason: Checkpoint is not a dict (type: {type(checkpoint).__name__})")
            error_msg = f"incompatible: checkpoint is not a dict (got {type(checkpoint).__name__})"
            if log_failures:
                log_failed_checkpoint(checkpoint_path, error_msg)
            return False, error_msg

        missing_keys = set(required_keys) - set(checkpoint.keys())
        if missing_keys:
            log.error(f"Checkpoint validation failed: {checkpoint_path}")
            log.error(f"  Reason: Missing required keys: {missing_keys}")
            log.error(f"  Keys found: {list(checkpoint.keys())}")
            error_msg = f"incompatible: missing required keys {missing_keys}"
            if log_failures:
                log_failed_checkpoint(checkpoint_path, error_msg)
            return False, error_msg

        # Log tensor information for diagnostics
        for key in required_keys:
            value = checkpoint[key]
            if isinstance(value, torch.Tensor):
                log.debug(f"  Key '{key}': Tensor shape={value.shape}, dtype={value.dtype}")
            else:
                log.debug(f"  Key '{key}': {type(value).__name__}")

    return True, ""


def distinguish_error_type(error_message: str) -> str:
    """Categorize validation error as 'corrupted' or 'incompatible'.

    Args:
        error_message: Error message from validation

    Returns:
        'corrupted' or 'incompatible' based on error pattern

    Example:
        >>> distinguish_error_type("corrupted: file is 0 bytes")
        'corrupted'
        >>> distinguish_error_type("incompatible: missing keys")
        'incompatible'
    """
    if error_message.startswith('corrupted:'):
        return 'corrupted'
    elif error_message.startswith('incompatible:'):
        return 'incompatible'
    else:
        # Default to corrupted for unknown errors
        return 'corrupted'


def validate_checkpoint_batch(
    checkpoint_paths: List[Path],
    required_keys: Optional[List[str]] = None,
    skip_load: bool = False,
    logger_instance: Optional[logging.Logger] = None
) -> Tuple[List[Path], List[Tuple[Path, str]]]:
    """Validate multiple checkpoint files and categorize results.

    Args:
        checkpoint_paths: List of checkpoint file paths to validate
        required_keys: List of required keys in checkpoint dict
        skip_load: Skip torch.load validation (faster, less thorough)
        logger_instance: Logger instance for diagnostic output

    Returns:
        Tuple of (valid_paths, failed_items)
        - valid_paths: List of paths that passed validation
        - failed_items: List of (path, error_message) tuples for failures

    Example:
        >>> valid, failed = validate_checkpoint_batch([Path("a.pt"), Path("b.pt")])
        >>> print(f"Valid: {len(valid)}, Failed: {len(failed)}")
    """
    log = logger_instance or logger

    valid_paths = []
    failed_items = []

    log.info(f"Validating {len(checkpoint_paths)} checkpoints...")

    for checkpoint_path in checkpoint_paths:
        is_valid, error_msg = validate_checkpoint(
            checkpoint_path,
            required_keys=required_keys,
            skip_load=skip_load,
            logger_instance=log
        )

        if is_valid:
            valid_paths.append(checkpoint_path)
        else:
            failed_items.append((checkpoint_path, error_msg))

    log.info(f"Validation complete: {len(valid_paths)} valid, {len(failed_items)} failed")

    return valid_paths, failed_items


def load_checkpoint_with_validation(
    checkpoint_path: Path,
    required_keys: Optional[List[str]] = None,
    skip_validation: bool = False,
    logger_instance: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """Load checkpoint with comprehensive validation.

    Validates checkpoint integrity before loading. Includes version and status checks.

    Args:
        checkpoint_path: Path to checkpoint file
        required_keys: List of required keys in checkpoint dict
        skip_validation: Skip validation for trusted scenarios (--skip-checkpoint-validation)
        logger_instance: Logger instance for diagnostic output

    Returns:
        Loaded checkpoint dict

    Raises:
        CheckpointError: If checkpoint is corrupted or incompatible

    Example:
        >>> checkpoint = load_checkpoint_with_validation(Path("model.pt"))
        >>> data = checkpoint['data']
    """
    log = logger_instance or logger

    if not skip_validation:
        # Perform validation before loading
        is_valid, error_msg = validate_checkpoint(
            checkpoint_path,
            required_keys=required_keys,
            skip_load=False,
            logger_instance=log
        )

        if not is_valid:
            error_type = distinguish_error_type(error_msg)
            raise CheckpointError(checkpoint_path, error_type, error_msg)

    # Load checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    except Exception as e:
        log.error(f"Checkpoint load failed: {checkpoint_path}")
        log.error(f"  Error: {str(e)}")
        raise CheckpointError(checkpoint_path, 'corrupted', f"torch.load failed - {str(e)}")

    # Version compatibility check
    if not skip_validation:
        version = checkpoint.get('version', '0.x')
        log.debug(f"Checkpoint version: {version}")

        # Check for future versions
        if version.startswith('2.'):
            error_msg = f"version {version} requires virnucpro >= 2.0.0"
            log.error(f"Checkpoint incompatible: {checkpoint_path}")
            log.error(f"  Version: {version}")
            raise CheckpointError(checkpoint_path, 'incompatible', error_msg)

        # Check status field
        status = checkpoint.get('status', 'unknown')
        if status == 'in_progress':
            log.warning(
                f"Checkpoint marked as in-progress (may be incomplete): {checkpoint_path}"
            )

    log.info(f"Checkpoint loaded successfully: {checkpoint_path}")
    return checkpoint


def get_checkpoint_info(checkpoint_path: Path) -> Dict[str, Any]:
    """Get checkpoint metadata without full validation.

    Quick inspection of checkpoint version, status, and .done marker for diagnostics.
    Checks .done marker for quick completion status without loading checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file

    Returns:
        Dict with keys: 'version', 'status', 'size_bytes', 'is_valid_zip', 'has_done_marker'

    Example:
        >>> info = get_checkpoint_info(Path("model.pt"))
        >>> print(f"Version: {info['version']}, Status: {info['status']}")
        >>> print(f"Has .done marker: {info['has_done_marker']}")
    """
    info = {
        'version': 'unknown',
        'status': 'unknown',
        'size_bytes': 0,
        'is_valid_zip': False,
        'has_done_marker': False
    }

    # Check for .done marker (quick check without loading checkpoint)
    done_marker = checkpoint_path.with_suffix(checkpoint_path.suffix + '.done')
    info['has_done_marker'] = done_marker.exists()

    # Get file size
    try:
        info['size_bytes'] = checkpoint_path.stat().st_size
    except Exception:
        return info

    # Check ZIP format
    try:
        info['is_valid_zip'] = zipfile.is_zipfile(checkpoint_path)
    except Exception:
        pass

    # Try to load and get version/status
    if info['is_valid_zip']:
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            if isinstance(checkpoint, dict):
                info['version'] = checkpoint.get('version', '0.x')
                info['status'] = checkpoint.get('status', 'unknown')
        except Exception:
            pass

    return info
