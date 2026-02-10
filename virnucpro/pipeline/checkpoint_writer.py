"""Checkpoint foundation for Phase 9: adaptive triggers, async writes, validation, resume.

This module provides the core building blocks for incremental checkpointing in the
async DataLoader + sequence packing + multi-GPU pipeline:

- CheckpointTrigger: Determines WHEN to checkpoint (sequence count OR time threshold)
- AsyncCheckpointWriter: Handles HOW to write without blocking GPU (background thread)
- validate_checkpoint_pt: Validates .pt checkpoint integrity (size, .done, loadable, shape)
- validate_checkpoint_metadata: Verifies model config and packing compatibility
- resume_from_checkpoints: Loads prior progress, handles corruption, returns requeue list

Checkpoint format uses PyTorch .pt files (consistent with Phase 3 atomic_save pattern)
instead of HDF5 to avoid append corruption risks and leverage proven serialization.

Architecture:
- Adaptive trigger fires on sequence count OR time (whichever first)
- Emergency override (>600s without checkpoint) forces checkpoint even mid-batch
- Async writes transfer GPU tensors to CPU before background thread submission
- Atomic temp-then-rename with .done markers for corruption detection
- Resume stops at first corruption, returns corrupted sequence IDs for requeue
- Optional manifest integration for multi-GPU coordinator validation
"""

import time
import os
import re
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, Future
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import torch

from virnucpro.core.checkpoint import has_done_marker, remove_done_marker
from virnucpro.core.env_config import get_env_config

logger = logging.getLogger('virnucpro.pipeline.checkpoint_writer')

BATCH_PATTERN = re.compile(r'^batch_(\d+)\.pt$')


class CheckpointTrigger:
    """Adaptive checkpoint trigger based on sequence count OR time threshold.

    Fires when either threshold is reached (whichever comes first). Respects
    batch boundaries to preserve packed attention atomicity, except during
    emergency override (>emergency_override_sec without checkpoint).

    Env var override: VIRNUCPRO_VIRAL_CHECKPOINT_MODE=true uses 5000 seq / 180s
    thresholds when constructor args are at their default values. Env var overrides
    default thresholds but not explicit arguments. This helps tune for viral (400AA+)
    workloads without code changes.

    Attributes:
        seq_threshold: Number of sequences to trigger checkpoint
        time_threshold_sec: Seconds elapsed to trigger checkpoint
        emergency_override_sec: Force checkpoint after this delay (even mid-batch)
        sequences_since_checkpoint: Running counter reset on checkpoint
        last_checkpoint_time: perf_counter timestamp of last checkpoint
    """

    def __init__(
        self,
        seq_threshold: int = 10000,
        time_threshold_sec: float = 300.0,
        emergency_override_sec: float = 600.0
    ):
        """Initialize checkpoint trigger with configurable thresholds.

        Args:
            seq_threshold: Sequences processed before checkpoint (default: 10000)
            time_threshold_sec: Seconds elapsed before checkpoint (default: 300)
            emergency_override_sec: Force checkpoint after delay (default: 600)
        """
        if seq_threshold <= 0:
            raise ValueError(f"seq_threshold must be positive, got {seq_threshold}")
        if time_threshold_sec <= 0:
            raise ValueError(f"time_threshold_sec must be positive, got {time_threshold_sec}")
        if emergency_override_sec <= time_threshold_sec:
            raise ValueError(
                f"emergency_override_sec ({emergency_override_sec}) must be > "
                f"time_threshold_sec ({time_threshold_sec})"
            )

        # Check for viral mode override (only applies to defaults)
        env = get_env_config()
        viral_mode = env.viral_checkpoint_mode
        defaults_used = (
            seq_threshold == 10000
            and time_threshold_sec == 300.0
        )

        if viral_mode and defaults_used:
            # Viral mode: more frequent checkpoints for 400AA+ sequences
            self.seq_threshold = 5000
            self.time_threshold_sec = 180.0
            logger.info("Viral checkpoint mode enabled: 5000 seq / 180s thresholds")
        else:
            # Use explicit args (env var doesn't override non-default values)
            self.seq_threshold = seq_threshold
            self.time_threshold_sec = time_threshold_sec

        self.emergency_override_sec = emergency_override_sec
        self.sequences_since_checkpoint = 0
        self.last_checkpoint_time = time.perf_counter()

        logger.debug(
            f"CheckpointTrigger initialized: seq_threshold={self.seq_threshold}, "
            f"time_threshold={self.time_threshold_sec}s, "
            f"emergency_override={self.emergency_override_sec}s"
        )

    def should_checkpoint(self, batch_size: int) -> Tuple[bool, Optional[str]]:
        """Check if checkpoint should be triggered.

        Args:
            batch_size: Number of sequences in current batch

        Returns:
            (should_checkpoint, reason) where reason is one of:
                - "sequence_threshold": Sequence count reached
                - "time_threshold": Time elapsed reached
                - "emergency_time_override": Emergency override triggered
                - None: No checkpoint needed
        """
        self.sequences_since_checkpoint += batch_size
        elapsed = time.perf_counter() - self.last_checkpoint_time

        # Emergency override: >emergency_override_sec without checkpoint
        if elapsed > self.emergency_override_sec:
            logger.warning(
                f"Emergency checkpoint override after {elapsed:.1f}s "
                f"({self.sequences_since_checkpoint} sequences pending)"
            )
            return True, "emergency_time_override"

        # Normal triggers (batch boundary safe)
        if self.sequences_since_checkpoint >= self.seq_threshold:
            logger.debug(
                f"Sequence threshold reached: {self.sequences_since_checkpoint} >= {self.seq_threshold}"
            )
            return True, "sequence_threshold"

        if elapsed >= self.time_threshold_sec:
            logger.debug(
                f"Time threshold reached: {elapsed:.1f}s >= {self.time_threshold_sec}s"
            )
            return True, "time_threshold"

        return False, None

    def reset(self) -> None:
        """Reset counters after checkpoint write.

        Resets both sequence counter and last checkpoint time.
        """
        self.sequences_since_checkpoint = 0
        self.last_checkpoint_time = time.perf_counter()
        logger.debug("Checkpoint trigger reset")


class AsyncCheckpointWriter:
    """Async checkpoint writer using background thread for I/O.

    Transfers GPU tensors to CPU before submitting to background thread to prevent
    CUDA context issues. Uses atomic temp-then-rename with .done markers.

    Optional manifest and rank parameters enable coordinator integration (Issue 6)
    for multi-GPU tracking of checkpoint progress.

    Attributes:
        executor: ThreadPoolExecutor for background I/O
        pending_futures: List of submitted write futures
        lock: Thread lock protecting pending_futures
        manifest: Optional CheckpointManifest for coordinator integration
        rank: Optional GPU rank for manifest updates
    """

    def __init__(
        self,
        max_workers: int = 1,
        manifest: Optional[Any] = None,
        rank: Optional[int] = None
    ):
        """Initialize async checkpoint writer.

        Args:
            max_workers: Thread pool size (default: 1 for sequential writes)
            manifest: Optional CheckpointManifest for coordinator tracking
            rank: Optional GPU rank for manifest updates
        """
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.pending_futures: List[Future] = []
        # Note: Uses unbounded queue. Caller is responsible for rate-limiting
        # submissions to prevent memory exhaustion.
        self.lock = threading.Lock()
        self.manifest = manifest
        self.rank = rank

        logger.debug(f"AsyncCheckpointWriter initialized: max_workers={max_workers}")

    def __del__(self):
        self.shutdown()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()

    def write_checkpoint_async(
        self,
        checkpoint_path: Path,
        embeddings: Any,
        sequence_ids: List[str],
        metadata: Dict[str, Any]
    ) -> Future:
        """Submit checkpoint write to background thread.

        CRITICAL: Transfers GPU tensors to CPU and copies data before submitting
        to prevent CUDA context issues and race conditions.

        Args:
            checkpoint_path: Target path for .pt checkpoint file
            embeddings: torch.Tensor (GPU or CPU) or np.ndarray
            sequence_ids: List of sequence identifiers
            metadata: Checkpoint metadata dict

        Returns:
            Future for async write operation
        """
        # CRITICAL: Transfer from GPU to CPU before background thread
        if isinstance(embeddings, torch.Tensor):
            embeddings_copy = embeddings.to('cpu').numpy()
            logger.debug(f"Transferred embeddings from GPU to CPU: shape={embeddings_copy.shape}")
        elif isinstance(embeddings, np.ndarray):
            embeddings_copy = embeddings.copy()
            logger.debug(f"Copied numpy embeddings: shape={embeddings_copy.shape}")
        else:
            raise TypeError(f"Embeddings must be torch.Tensor or np.ndarray, got {type(embeddings)}")

        # Copy sequence IDs and metadata to prevent race conditions
        ids_copy = list(sequence_ids)
        metadata_copy = metadata.copy()

        # Submit to background thread
        future = self.executor.submit(
            self._write_checkpoint_sync,
            checkpoint_path,
            embeddings_copy,
            ids_copy,
            metadata_copy
        )

        with self.lock:
            self.pending_futures.append(future)

        logger.debug(f"Async checkpoint write submitted: {checkpoint_path.name}")
        return future

    def _write_checkpoint_sync(
        self,
        checkpoint_path: Path,
        embeddings: np.ndarray,
        sequence_ids: List[str],
        metadata: Dict[str, Any]
    ) -> None:
        """Internal sync write implementation (runs in background thread).

        Uses atomic temp-then-rename pattern with .done marker for corruption
        detection. Consistent with Phase 3 atomic_save pattern.

        Args:
            checkpoint_path: Target path for .pt checkpoint file
            embeddings: Numpy array of embeddings
            sequence_ids: List of sequence identifiers
            metadata: Checkpoint metadata dict

        Raises:
            RuntimeError: If write fails (cleaned up temp file)
        """
        temp_path = checkpoint_path.with_suffix('.tmp')

        try:
            # Build checkpoint dict
            checkpoint_dict = {
                'embeddings': embeddings,
                'sequence_ids': sequence_ids,
                'metadata': metadata
            }

            # Write to temp file
            torch.save(checkpoint_dict, temp_path, pickle_protocol=4)

            # Atomic rename
            try:
                temp_path.replace(checkpoint_path)
            except OSError:
                if temp_path.exists():
                    temp_path.unlink()
                raise

            # Create .done marker (indicates successful write)
            done_marker = checkpoint_path.with_suffix(checkpoint_path.suffix + '.done')
            done_marker.touch()

            logger.info(
                f"Checkpoint written: {checkpoint_path.name} "
                f"({len(sequence_ids)} sequences, {embeddings.nbytes / 1024**2:.1f} MB)"
            )

            # Optional manifest integration (Issue 6)
            if self.manifest is not None and self.rank is not None:
                batch_idx = metadata.get('batch_idx', 0)
                self.manifest.update_shard_checkpoint(
                    self.rank,
                    batch_idx,
                    len(sequence_ids),
                    checkpoint_path.name
                )
                logger.debug(f"Updated manifest: shard={self.rank}, batch={batch_idx}")

        except Exception as e:
            # Clean up temp file on failure
            if temp_path.exists():
                temp_path.unlink()
            logger.error(f"Failed to write checkpoint {checkpoint_path.name}: {e}")
            raise RuntimeError(f"Checkpoint write failed: {e}") from e

    def wait_all(self, timeout: Optional[float] = None) -> None:
        """Block until all pending writes complete.

        MUST iterate through all futures and call .result() to re-raise any
        exceptions from background threads (Issue 7). Aggregates errors.

        Args:
            timeout: Optional timeout in seconds (None = wait indefinitely)

        Raises:
            RuntimeError: If any writes failed (aggregated error messages)
        """
        with self.lock:
            futures_to_check = self.pending_futures
            self.pending_futures = []

        errors = []
        for future in futures_to_check:
            try:
                future.result(timeout=timeout)
            except Exception as e:
                errors.append((type(e).__name__, str(e)))
                logger.error(f"Async checkpoint write failed: {e}")

        # Raise aggregated errors if any writes failed
        if errors:
            error_msg = f"{len(errors)} checkpoint write(s) failed:\n" + "\n".join(
                f"{name}: {msg}" for name, msg in errors
            )
            raise RuntimeError(error_msg)

        logger.debug(f"All pending checkpoint writes completed: {len(futures_to_check)} writes")

    def has_pending(self) -> bool:
        """Check if any checkpoint writes are pending.

        Thread-safe check with lock protection (Issue 12).

        Returns:
            True if pending writes exist
        """
        with self.lock:
            return len(self.pending_futures) > 0

    def shutdown(self) -> None:
        """Shutdown executor and wait for pending writes.

        Blocks until all pending writes complete before shutting down.
        """
        logger.debug("Shutting down async checkpoint writer")
        try:
            self.wait_all()
        except RuntimeError as e:
            logger.error(f"Errors during shutdown: {e}")
        finally:
            self.executor.shutdown(wait=True)


def validate_checkpoint_pt(checkpoint_path: Path) -> Tuple[bool, str]:
    """Validate .pt checkpoint file integrity.

    Multi-level validation for PyTorch .pt checkpoint files:
    - Level 1: File exists and size > 0
    - Level 2: .done marker exists
    - Level 3: File loadable via torch.load
    - Level 4: Shape consistency (embeddings count == sequence_ids count)

    Args:
        checkpoint_path: Path to .pt checkpoint file

    Returns:
        (is_valid, error_description) where error_description is empty on success
    """
    # Level 1: File exists and non-empty
    if not checkpoint_path.exists():
        return False, f"Checkpoint file does not exist: {checkpoint_path}"

    file_size = checkpoint_path.stat().st_size
    if file_size == 0:
        return False, f"Checkpoint file is empty (0 bytes): {checkpoint_path}"

    # Level 2: .done marker exists
    if not has_done_marker(checkpoint_path):
        return False, f"Missing .done marker: {checkpoint_path}"

    # Level 3: File loadable via torch.load
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    except Exception as e:
        return False, f"Failed to load checkpoint: {e}"

    # Verify required keys
    if 'embeddings' not in checkpoint:
        return False, "Checkpoint missing 'embeddings' key"

    if 'sequence_ids' not in checkpoint:
        return False, "Checkpoint missing 'sequence_ids' key"

    # Level 4: Shape consistency
    try:
        embeddings = checkpoint['embeddings']
        sequence_ids = checkpoint['sequence_ids']

        if not hasattr(embeddings, '__len__'):
            return False, f"Embeddings do not support length: {type(embeddings)}"

        if hasattr(embeddings, 'shape') and len(embeddings.shape) != 2:
            return False, f"Embeddings must be 2D, got shape {embeddings.shape}"

        num_embeddings = len(embeddings)
        num_ids = len(sequence_ids)

        if num_ids == 0:
            return False, "Checkpoint has empty sequence_ids list"

        if num_embeddings != num_ids:
            return False, (
                f"Shape mismatch: {num_embeddings} embeddings vs {num_ids} sequence_ids"
            )
    except Exception as e:
        return False, f"Failed to validate shapes: {e}"

    # All validation levels passed
    return True, ""


def validate_checkpoint_metadata(
    metadata: Dict[str, Any],
    current_model_config_hash: Optional[str] = None,
    current_packing_enabled: Optional[bool] = None
) -> Tuple[bool, List[str]]:
    """Validate checkpoint metadata compatibility.

    Returns warnings (not errors) for metadata mismatches. Caller decides whether
    to proceed with loading. Useful for detecting model config or packing changes.

    Args:
        metadata: Metadata dict from checkpoint
        current_model_config_hash: Optional current model config hash to compare
        current_packing_enabled: Optional current packing setting to compare

    Returns:
        (is_valid, warnings) where is_valid=False if any warnings, True otherwise
    """
    warnings = []

    # Check 1: Model config hash compatibility
    if current_model_config_hash is not None:
        checkpoint_hash = metadata.get('model_config_hash')
        if checkpoint_hash is not None and checkpoint_hash != current_model_config_hash:
            warnings.append(
                f"Model config hash mismatch: checkpoint={checkpoint_hash}, "
                f"current={current_model_config_hash}"
            )

    # Check 2: Packing config compatibility
    if current_packing_enabled is not None:
        checkpoint_packing = metadata.get('packing_enabled')
        if checkpoint_packing is not None and checkpoint_packing != current_packing_enabled:
            warnings.append(
                f"Packing config mismatch: checkpoint={checkpoint_packing}, "
                f"current={current_packing_enabled}"
            )

    # Check 3: Required metadata keys
    required_keys = ['batch_idx', 'num_sequences', 'timestamp']
    for key in required_keys:
        if key not in metadata:
            warnings.append(f"Missing required metadata key: {key}")

    # Return validation result
    is_valid = len(warnings) == 0
    return is_valid, warnings


def resume_from_checkpoints(
    checkpoint_dir: Path,
    rank: int,
    force_restart: bool = False,
    manifest: Optional[Any] = None
) -> Tuple[List[str], Optional[np.ndarray], int, List[str]]:
    """Resume from checkpoint files, handling corruption and returning requeue list.

    Loads valid checkpoints in order, stops at first corruption, and returns:
    - All sequence IDs processed successfully
    - Concatenated embeddings from valid checkpoints
    - Resume batch index (last_valid_batch + 1)
    - Corrupted sequence IDs for requeue by caller

    Args:
        checkpoint_dir: Base checkpoint directory
        rank: GPU rank (shard identifier)
        force_restart: If True, ignore checkpoints and start fresh
        manifest: Optional CheckpointManifest for validation (Issue 4)

    Returns:
        (all_ids, concatenated_embeddings, resume_batch_idx, corrupted_sequence_ids)
        Returns ([], None, 0, []) if force_restart or no valid checkpoints
    """
    shard_dir = checkpoint_dir / f"shard_{rank}"

    # Force restart or shard directory doesn't exist
    if force_restart:
        logger.info(f"Force restart enabled: ignoring checkpoints for shard {rank}")
        return [], None, 0, []

    if not shard_dir.exists():
        logger.info(f"No checkpoint directory for shard {rank}: starting fresh")
        return [], None, 0, []

    # Find all batch_*.pt files
    checkpoint_files = list(shard_dir.glob("batch_*.pt"))

    if not checkpoint_files:
        logger.info(f"No checkpoints found for shard {rank}: starting fresh")
        return [], None, 0, []

    # Sort by batch number using regex (Issue 8)
    def extract_batch_num(path: Path) -> int:
        match = BATCH_PATTERN.search(path.name)
        if match:
            return int(match.group(1))
        else:
            logger.warning(f"Skipping file with unexpected name: {path.name}")
            return -1

    # Filter out files that don't match pattern and sort
    valid_files = [f for f in checkpoint_files if extract_batch_num(f) >= 0]
    sorted_files = sorted(valid_files, key=extract_batch_num)

    logger.info(f"Found {len(sorted_files)} checkpoint(s) for shard {rank}")

    # Load valid checkpoints until corruption detected
    all_ids = []
    all_embeddings = []
    last_valid_batch = -1
    corrupted_sequence_ids = []
    corruption_detected = False

    for checkpoint_file in sorted_files:
        batch_num = extract_batch_num(checkpoint_file)

        # If corruption already detected, just collect sequence IDs for requeue
        if corruption_detected:
            try:
                checkpoint = torch.load(checkpoint_file, map_location='cpu', weights_only=False)
                if 'sequence_ids' in checkpoint:
                    corrupted_sequence_ids.extend(checkpoint['sequence_ids'])
                    logger.debug(
                        f"Collected {len(checkpoint['sequence_ids'])} IDs from batch {batch_num} for requeue"
                    )
            except Exception as e:
                logger.warning(
                    f"Could not load sequence IDs from batch {batch_num}: {e}. "
                    "Sequences from this batch are lost."
                )
            continue

        # Validate checkpoint
        is_valid, error_msg = validate_checkpoint_pt(checkpoint_file)

        if not is_valid:
            logger.warning(
                f"Corruption detected at batch {batch_num}: {error_msg}. "
                f"Stopping resume at batch {last_valid_batch}"
            )

            # Invalidate corrupted checkpoint
            remove_done_marker(checkpoint_file)

            # Mark corruption detected
            corruption_detected = True

            # Try to load sequence IDs from corrupted checkpoint for requeue
            try:
                checkpoint = torch.load(checkpoint_file, map_location='cpu', weights_only=False)
                if 'sequence_ids' in checkpoint:
                    corrupted_sequence_ids.extend(checkpoint['sequence_ids'])
                    logger.debug(
                        f"Collected {len(checkpoint['sequence_ids'])} IDs from corrupted batch {batch_num}"
                    )
            except Exception as e:
                logger.error(f"Could not load sequence IDs from corrupted batch {batch_num}: {e}")
                corrupted_sequence_ids.append(f"batch_{batch_num}_LOST")

            continue

        # Load valid checkpoint
        try:
            checkpoint = torch.load(checkpoint_file, map_location='cpu', weights_only=False)
            embeddings = checkpoint['embeddings']
            sequence_ids = checkpoint['sequence_ids']

            all_embeddings.append(embeddings)
            all_ids.extend(sequence_ids)
            last_valid_batch = batch_num

            logger.debug(
                f"Loaded batch {batch_num}: {len(sequence_ids)} sequences, "
                f"embeddings shape={embeddings.shape}"
            )

        except Exception as e:
            logger.error(f"Failed to load valid checkpoint batch {batch_num}: {e}")
            # Treat as corruption
            remove_done_marker(checkpoint_file)
            corruption_detected = True
            continue

    # Concatenate all valid embeddings
    if all_embeddings:
        concatenated = np.concatenate(all_embeddings, axis=0)
    else:
        concatenated = None

    resume_batch_idx = last_valid_batch + 1

    # Log resume summary
    if concatenated is not None:
        logger.info(
            f"Resuming from {len(sorted_files)} checkpoint(s): "
            f"{len(all_ids)} sequences, last_batch={last_valid_batch}"
        )
    else:
        logger.info(f"No valid checkpoints for shard {rank}: starting fresh")

    if corruption_detected:
        logger.warning(
            f"Corruption detected at batch {last_valid_batch + 1}: "
            f"{len(corrupted_sequence_ids)} sequences need reprocessing"
        )

    # Optional manifest validation (Issue 4)
    if manifest is not None and concatenated is not None:
        _validate_manifest(manifest, rank, resume_batch_idx - 1, len(all_ids))

    return all_ids, concatenated, resume_batch_idx, corrupted_sequence_ids


def _validate_manifest(
    manifest: Any,
    rank: int,
    last_batch: int,
    total_sequences: int
) -> None:
    """Validate manifest consistency (internal helper).

    Logs warnings for mismatches but does not fail. Filesystem checkpoints
    are the source of truth.

    Args:
        manifest: CheckpointManifest instance
        rank: GPU rank (shard identifier)
        last_batch: Last valid batch index
        total_sequences: Total sequences loaded from checkpoints
    """
    # Check if shard exists in manifest
    if not hasattr(manifest, 'get_shard_state'):
        logger.debug("Manifest does not support shard state queries: skipping validation")
        return

    try:
        shard_state = manifest.get_shard_state(rank)

        if shard_state is None:
            logger.warning(f"Shard {rank} not found in manifest (expected from filesystem)")
            return

        # Validate last checkpoint batch
        manifest_last_batch = shard_state.get('last_checkpoint_batch', -1)
        if manifest_last_batch != last_batch:
            logger.warning(
                f"Manifest batch mismatch for shard {rank}: "
                f"manifest={manifest_last_batch}, filesystem={last_batch}"
            )

        # Validate total sequences
        manifest_total = shard_state.get('total_sequences', 0)
        if manifest_total != total_sequences:
            logger.warning(
                f"Manifest sequence count mismatch for shard {rank}: "
                f"manifest={manifest_total}, filesystem={total_sequences}"
            )

    except Exception as e:
        logger.warning(f"Failed to validate manifest for shard {rank}: {e}")
