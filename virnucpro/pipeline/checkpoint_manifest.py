"""Coordinator manifest for multi-GPU checkpoint tracking.

Tracks per-shard checkpoint progress to enable partial failure recovery,
elastic shard redistribution, and global completeness validation.
Uses JSON format consistent with Phase 7 sequence_index.json.
Checkpoint files referenced in manifest use .pt format (Phase 3 torch.save pattern).

## Concurrency Model

CheckpointManifest assumes a single coordinator process that initializes the
manifest and reads global state, plus multiple worker processes that write
per-shard checkpoints independently. Workers DO NOT directly update the
manifest — AsyncCheckpointWriter integration (Plan 09-01) handles this via
the writer's _write_checkpoint_sync method, which calls
manifest.update_shard_checkpoint() AFTER the checkpoint file and .done marker
are successfully written.

For cross-process safety, all manifest mutations use POSIX file locking
(fcntl.flock) to prevent concurrent writes from corrupting the JSON file.
This replaces the original threading.Lock design which was process-local and
provided no cross-process coordination.

For single-coordinator scenarios (typical), the coordinator is the only
writer and workers only write per-shard checkpoint files. The coordinator
periodically calls get_global_progress() to monitor worker health via
checkpoint timestamps.
"""

import json
import logging
import os
import fcntl
import shutil
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger('virnucpro.pipeline.checkpoint_manifest')


class ManifestCorruptedError(Exception):
    """Raised when manifest JSON cannot be loaded from primary, .tmp, or .backup files.

    This error indicates that all three manifest file versions (primary, .tmp, .backup)
    are corrupted and cannot be parsed. Recovery requires manual intervention or
    restart from scratch.

    Attributes:
        manifest_path: Path to the primary manifest file for recovery instructions
    """

    def __init__(self, message: str, manifest_path: Optional[Path] = None):
        super().__init__(message)
        self.manifest_path = manifest_path


class CheckpointManifest:
    """Multi-GPU checkpoint coordination manifest with fault tolerance.

    Provides a global view of per-shard checkpoint progress across multiple GPU workers,
    enabling partial failure recovery, elastic shard redistribution, staleness detection,
    and completion validation.

    Uses file-system coordination (POSIX file locks + atomic rename) rather than
    in-process threading locks, because GPU workers run as separate spawned processes
    with independent memory spaces.

    Attributes:
        manifest_path: Path to the JSON manifest file
        staleness_threshold_sec: Time threshold for zombie shard detection (default 600s = 10min)
    """

    MAX_ORPHANED_RETRIES = 3

    def __init__(self, manifest_path: Path, staleness_threshold_sec: int = 600):
        """Initialize checkpoint manifest.

        Args:
            manifest_path: Path to JSON manifest file
            staleness_threshold_sec: Time threshold in seconds for zombie shard detection
                (default 600 = 10 minutes)
        """
        self.manifest_path = Path(manifest_path)
        self.staleness_threshold_sec = staleness_threshold_sec
        self._lock_path = self.manifest_path.with_suffix('.lock')

    def _acquire_file_lock(self) -> int:
        """Acquire exclusive POSIX file lock for cross-process safety.

        Opens lock file and acquires exclusive lock using fcntl.flock.
        Caller must pass returned file descriptor to _release_file_lock().

        Returns:
            File descriptor for the lock file (pass to _release_file_lock)

        Raises:
            RuntimeError: If lock file cannot be created or lock cannot be acquired
        """
        try:
            fd = os.open(str(self._lock_path), os.O_CREAT | os.O_WRONLY, 0o644)
        except OSError as create_err:
            raise RuntimeError(f"Failed to create lock file: {create_err}")
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
        except OSError as lock_err:
            os.close(fd)
            os.unlink(self._lock_path)
            raise RuntimeError(f"Failed to acquire lock: {lock_err}")
        return fd

    def _release_file_lock(self, fd: int):
        """Release POSIX file lock and remove lock file.

        Args:
            fd: File descriptor returned from _acquire_file_lock()
        """
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
        try:
            os.unlink(self._lock_path)
        except OSError:
            pass  # File may not exist if never created or already removed

    def initialize(
        self,
        world_size: int,
        input_fingerprint: str = "",
        model_config_hash: str = ""
    ) -> Dict:
        """Initialize manifest with shard entries for world_size workers.

        Creates initial manifest JSON structure with global metadata and per-shard
        tracking entries. Uses atomic write to prevent corruption.

        Args:
            world_size: Number of GPU workers (shard count)
            input_fingerprint: SHA256 hash of concatenated FASTA files
            model_config_hash: Hash of model dtype/architecture/weights

        Returns:
            Initialized manifest dictionary
        """
        fd = self._acquire_file_lock()
        try:
            manifest = {
                "version": "2.0",
                "world_size": world_size,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "input_fingerprint": input_fingerprint,
                "model_config_hash": model_config_hash,
                "global_checkpoints": [],
                "shards": {}
            }

            # Initialize per-shard tracking entries
            for rank in range(world_size):
                manifest["shards"][str(rank)] = {
                    "status": "in_progress",
                    "original_rank": rank,
                    "assigned_rank": rank,
                    "last_checkpoint_batch": -1,
                    "total_sequences": 0,
                    "last_checkpoint_time": None,
                    "retry_count": 0,
                    "error": None,
                    "checkpoints": []
                }

            self._save_manifest(manifest)
            logger.info(f"Initialized manifest: {world_size} shards, version 2.0")

            return manifest
        finally:
            self._release_file_lock(fd)

    def set_global_metadata(self, input_fingerprint: str, model_config_hash: str):
        """Set global metadata fields (input fingerprint and model config hash).

        Args:
            input_fingerprint: SHA256 hash of concatenated FASTA files
            model_config_hash: Hash of model dtype/architecture/weights
        """
        fd = self._acquire_file_lock()
        try:
            manifest = self._load_manifest()
            manifest["input_fingerprint"] = input_fingerprint
            manifest["model_config_hash"] = model_config_hash
            self._save_manifest(manifest)
            logger.debug("Set global metadata in manifest")
        finally:
            self._release_file_lock(fd)

    def validate_run_compatibility(
        self,
        input_fingerprint: str,
        model_config_hash: str
    ) -> Tuple[bool, List[str]]:
        """Validate that current run is compatible with checkpointed run.

        Compares input fingerprint and model config hash against manifest values
        to detect configuration changes that would invalidate checkpoints.

        Args:
            input_fingerprint: SHA256 hash of current FASTA files
            model_config_hash: Hash of current model config

        Returns:
            Tuple of (is_compatible, list_of_warnings)
            is_compatible is True if no warnings, False otherwise
        """
        manifest = self._load_manifest()
        warnings = []

        # Check input fingerprint
        manifest_fingerprint = manifest.get("input_fingerprint", "")
        if manifest_fingerprint and manifest_fingerprint != input_fingerprint:
            warnings.append(
                f"Input fingerprint mismatch: manifest={manifest_fingerprint}, "
                f"current={input_fingerprint} — FASTA files may have changed"
            )

        # Check model config hash
        manifest_hash = manifest.get("model_config_hash", "")
        if manifest_hash and manifest_hash != model_config_hash:
            warnings.append(
                f"Model config hash mismatch: manifest={manifest_hash}, "
                f"current={model_config_hash} — model weights or config may have changed"
            )

        is_compatible = len(warnings) == 0
        return is_compatible, warnings

    def update_shard_checkpoint(
        self,
        rank: int,
        batch_idx: int,
        num_sequences: int,
        checkpoint_file: str,
        sequence_range: str = ""
    ):
        """Update shard checkpoint progress after successful checkpoint write.

        Validates that checkpoint file exists before updating manifest to ensure
        consistency between filesystem state and manifest state.

        Args:
            rank: GPU worker rank
            batch_idx: Batch index that was checkpointed
            num_sequences: Number of sequences in this checkpoint
            checkpoint_file: Checkpoint filename (e.g., "batch_00042.pt")
            sequence_range: Informational sequence ID range (e.g., "seq_4200-seq_4299")

        Raises:
            FileNotFoundError: If checkpoint file doesn't exist at expected path
        """
        fd = self._acquire_file_lock()
        try:
            manifest = self._load_manifest()

            # Validate checkpoint file path is within expected directory
            if ".." in checkpoint_file or "/" in checkpoint_file or "\\" in checkpoint_file:
                raise ValueError(f"Invalid checkpoint file path (path traversal not allowed): {checkpoint_file}")

            checkpoint_dir = (self.manifest_path.parent / f"shard_{rank}").resolve()
            expected_path = checkpoint_dir / checkpoint_file

            resolved = expected_path.resolve()
            if not resolved.is_relative_to(checkpoint_dir):
                raise ValueError(f"Invalid checkpoint file path: {checkpoint_file}")

            # Validate checkpoint file exists
            if not expected_path.exists():
                raise FileNotFoundError(
                    f"Checkpoint {checkpoint_file} not found at expected path "
                    f"{expected_path}, cannot update manifest"
                )

            # Update shard entry
            shard = manifest["shards"][str(rank)]
            shard["last_checkpoint_batch"] = batch_idx
            shard["total_sequences"] += num_sequences
            shard["last_checkpoint_time"] = datetime.now(timezone.utc).isoformat()

            # Append checkpoint record
            shard["checkpoints"].append({
                "batch_idx": batch_idx,
                "num_sequences": num_sequences,
                "file": checkpoint_file,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "sequence_range": sequence_range
            })

            self._save_manifest(manifest)
            logger.debug(
                f"Updated shard {rank} checkpoint: batch={batch_idx}, "
                f"sequences={num_sequences}, file={checkpoint_file}"
            )
        finally:
            self._release_file_lock(fd)

    def mark_shard_complete(self, rank: int):
        """Mark shard as successfully completed.

        Args:
            rank: GPU worker rank
        """
        fd = self._acquire_file_lock()
        try:
            manifest = self._load_manifest()
            shard = manifest["shards"][str(rank)]
            shard["status"] = "complete"
            shard["completed_at"] = datetime.now(timezone.utc).isoformat()
            self._save_manifest(manifest)
            logger.info(f"Marked shard {rank} as complete")
        finally:
            self._release_file_lock(fd)

    def mark_shard_failed(self, rank: int, error: str):
        """Mark shard as failed with error message.

        Args:
            rank: GPU worker rank
            error: Error message describing failure
        """
        fd = self._acquire_file_lock()
        try:
            manifest = self._load_manifest()
            shard = manifest["shards"][str(rank)]
            shard["status"] = "failed"
            shard["failed_at"] = datetime.now(timezone.utc).isoformat()
            shard["error"] = error
            shard["retry_count"] += 1
            self._save_manifest(manifest)
            logger.warning(f"Marked shard {rank} as failed: {error}")
        finally:
            self._release_file_lock(fd)

    def get_shard_status(self, rank: int) -> Optional[Dict]:
        """Get status information for a specific shard.

        Args:
            rank: GPU worker rank

        Returns:
            Shard dictionary or None if rank not found
        """
        manifest = self._load_manifest()
        return manifest["shards"].get(str(rank))

    def get_incomplete_shard_ranks(self) -> List[int]:
        """Get list of shard ranks that need reprocessing.

        Returns ranks where status is "in_progress" or "failed".

        Returns:
            List of rank integers needing reprocessing
        """
        manifest = self._load_manifest()
        incomplete = []

        for rank_str, shard in manifest["shards"].items():
            if shard["status"] in ("in_progress", "failed"):
                incomplete.append(int(rank_str))

        return sorted(incomplete)

    def get_resumable_shards(self) -> List[int]:
        """Alias for get_incomplete_shard_ranks() for backward compatibility.

        Returns:
            List of rank integers needing reprocessing
        """
        return self.get_incomplete_shard_ranks()

    def get_resume_info(self, rank: int) -> Dict:
        """Get detailed resume information for a specific shard.

        Args:
            rank: GPU worker rank

        Returns:
            Dictionary with resume details:
                - rank: Worker rank
                - resume_from_batch: Next batch to process
                - last_completed_batch: Last successfully checkpointed batch
                - total_sequences_completed: Total sequences checkpointed
                - status: Shard status
                - assigned_rank: Current assigned rank (for elastic redistribution)
                - retry_count: Number of retry attempts
                - last_checkpoint_time: ISO timestamp of last checkpoint

            Returns minimal dict with status="unknown" if rank not found
        """
        manifest = self._load_manifest()
        shard = manifest["shards"].get(str(rank))

        if shard is None:
            return {"rank": rank, "status": "unknown"}

        return {
            "rank": rank,
            "resume_from_batch": shard["last_checkpoint_batch"] + 1,
            "last_completed_batch": shard["last_checkpoint_batch"],
            "total_sequences_completed": shard["total_sequences"],
            "status": shard["status"],
            "assigned_rank": shard["assigned_rank"],
            "retry_count": shard["retry_count"],
            "last_checkpoint_time": shard["last_checkpoint_time"]
        }

    def get_completed_shards(self) -> List[int]:
        """Get list of completed shard ranks.

        Returns:
            List of rank integers with status "complete"
        """
        manifest = self._load_manifest()
        completed = []

        for rank_str, shard in manifest["shards"].items():
            if shard["status"] == "complete":
                completed.append(int(rank_str))

        return sorted(completed)

    def get_global_progress(self) -> Dict:
        """Get global progress summary across all shards.

        Includes staleness detection for zombie shard identification.

        Returns:
            Dictionary with global progress metrics:
                - total_shards: Number of shards
                - completed: Count of completed shards
                - in_progress: Count of in-progress shards
                - failed: Count of failed shards
                - total_sequences_checkpointed: Sum across all shards
                - stale_shards: List of rank ints exceeding staleness threshold
                - input_fingerprint: Input fingerprint from manifest
                - model_config_hash: Model config hash from manifest
        """
        manifest = self._load_manifest()
        now = datetime.now(timezone.utc)

        completed = 0
        in_progress = 0
        failed = 0
        total_sequences = 0
        stale_shards = []

        for rank_str, shard in manifest["shards"].items():
            status = shard["status"]

            if status == "complete":
                completed += 1
            elif status == "in_progress":
                in_progress += 1

                # Check staleness
                last_checkpoint_time = shard.get("last_checkpoint_time")
                if last_checkpoint_time is not None:
                    checkpoint_time = datetime.fromisoformat(last_checkpoint_time)
                    elapsed_sec = (now - checkpoint_time).total_seconds()
                    if elapsed_sec > self.staleness_threshold_sec:
                        stale_shards.append(int(rank_str))
            elif status == "failed":
                failed += 1

            total_sequences += shard.get("total_sequences", 0)

        return {
            "total_shards": manifest["world_size"],
            "completed": completed,
            "in_progress": in_progress,
            "failed": failed,
            "total_sequences_checkpointed": total_sequences,
            "stale_shards": sorted(stale_shards),
            "input_fingerprint": manifest.get("input_fingerprint", ""),
            "model_config_hash": manifest.get("model_config_hash", "")
        }

    def is_shard_stale(
        self,
        rank: int,
        threshold_sec: Optional[int] = None
    ) -> bool:
        """Check if a shard is stale (potential zombie worker).

        A shard is stale if it's in_progress status AND has a last_checkpoint_time
        AND elapsed time since last checkpoint exceeds threshold.

        Args:
            rank: GPU worker rank
            threshold_sec: Custom threshold in seconds (uses instance threshold if None)

        Returns:
            True if shard is stale, False otherwise
        """
        manifest = self._load_manifest()
        shard = manifest["shards"].get(str(rank))

        if shard is None:
            return False

        if shard["status"] != "in_progress":
            return False

        last_checkpoint_time = shard.get("last_checkpoint_time")
        if last_checkpoint_time is None:
            # No checkpoint yet - could be starting up, not necessarily stale
            return False

        threshold = threshold_sec if threshold_sec is not None else self.staleness_threshold_sec
        checkpoint_time = datetime.fromisoformat(last_checkpoint_time)
        elapsed_sec = (datetime.now(timezone.utc) - checkpoint_time).total_seconds()

        return elapsed_sec > threshold

    def reassign_shard(self, shard_rank: int, new_assigned_rank: int):
        """Reassign shard to a different worker for elastic redistribution.

        Updates assigned_rank while preserving original_rank. Used for elastic
        redistribution when workers fail and orphaned work needs reassignment.

        Args:
            shard_rank: Original shard rank
            new_assigned_rank: New worker rank to assign this shard to
        """
        fd = self._acquire_file_lock()
        try:
            manifest = self._load_manifest()
            shard = manifest["shards"][str(shard_rank)]
            old_assigned_rank = shard["assigned_rank"]
            shard["assigned_rank"] = new_assigned_rank
            self._save_manifest(manifest)
            logger.info(
                f"Shard {shard_rank} reassigned from rank {old_assigned_rank} "
                f"to rank {new_assigned_rank}"
            )
        finally:
            self._release_file_lock(fd)

    def get_orphaned_shards(self) -> List[int]:
        """Get list of orphaned shards that exceeded max retry attempts.

        Orphaned shards have status "failed" AND retry_count >= MAX_ORPHANED_RETRIES.
        These are candidates for redistribution to healthy GPUs.

        Returns:
            List of shard rank integers that are orphaned
        """
        manifest = self._load_manifest()
        orphaned = []

        for rank_str, shard in manifest["shards"].items():
            if shard["status"] == "failed" and shard.get("retry_count", 0) >= self.MAX_ORPHANED_RETRIES:
                orphaned.append(int(rank_str))

        return sorted(orphaned)

    def record_global_checkpoint(
        self,
        checkpoint_id: str,
        batch_boundaries: Dict[int, int]
    ):
        """Record a global checkpoint barrier across all shards.

        Global checkpoints represent consistent sync points where all shards
        reached compatible batch boundaries. Used for coordinated recovery.

        Args:
            checkpoint_id: Unique identifier for this global checkpoint
            batch_boundaries: Dict mapping rank to batch_idx for each shard
        """
        fd = self._acquire_file_lock()
        try:
            manifest = self._load_manifest()

            global_checkpoint = {
                "checkpoint_id": checkpoint_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "batch_boundaries": {str(rank): batch_idx for rank, batch_idx in batch_boundaries.items()}
            }

            manifest["global_checkpoints"].append(global_checkpoint)
            self._save_manifest(manifest)
            logger.info(f"Recorded global checkpoint: {checkpoint_id}")
        finally:
            self._release_file_lock(fd)

    def get_latest_global_checkpoint(self) -> Optional[Dict]:
        """Get the most recent global checkpoint.

        Returns:
            Latest global checkpoint dictionary or None if no global checkpoints exist
        """
        manifest = self._load_manifest()
        global_checkpoints = manifest.get("global_checkpoints", [])

        if not global_checkpoints:
            return None

        return global_checkpoints[-1]

    def archive_manifest(self, archive_dir: Path):
        """Archive final manifest after successful completion.

        Validates that all shards are complete before archiving. Copies manifest
        to archive directory as manifest_final.json.

        Args:
            archive_dir: Directory to archive manifest to

        Raises:
            ValueError: If any shards are not complete or archive_dir is invalid
        """
        archive_dir = Path(archive_dir).resolve()
        expected_base = self.manifest_path.parent.resolve()
        if not archive_dir.is_relative_to(expected_base):
            raise ValueError(f"Invalid archive directory: {archive_dir} is outside expected base {expected_base}")

        fd = self._acquire_file_lock()
        try:
            manifest = self._load_manifest()

            # Validate all shards complete
            incomplete = []
            for rank_str, shard in manifest["shards"].items():
                if shard["status"] != "complete":
                    incomplete.append(int(rank_str))

            if incomplete:
                raise ValueError(
                    f"Cannot archive: shards {incomplete} are not complete"
                )

            # Create archive directory and copy manifest
            archive_dir.mkdir(parents=True, exist_ok=True)
            archive_path = archive_dir / "manifest_final.json"

            shutil.copy2(self.manifest_path, archive_path)
            logger.info(f"Manifest archived to {archive_path}")
        finally:
            self._release_file_lock(fd)

    def cleanup_checkpoints(self, keep_final_only: bool = True):
        """Clean up per-batch checkpoint files.

        Removes intermediate checkpoint files to reclaim disk space after successful
        completion. Can optionally keep final checkpoint per shard for validation.

        Args:
            keep_final_only: If True, keep only the last checkpoint per shard.
                If False, remove all per-batch checkpoints.
        """
        fd = self._acquire_file_lock()
        try:
            manifest = self._load_manifest()

            for rank_str, shard in manifest["shards"].items():
                rank = int(rank_str)
                checkpoints = shard.get("checkpoints", [])

                if not checkpoints:
                    continue

                # Determine which checkpoints to remove
                if keep_final_only:
                    checkpoints_to_remove = checkpoints[:-1]  # All except last
                else:
                    checkpoints_to_remove = checkpoints  # All

                removed_count = 0
                for checkpoint in checkpoints_to_remove:
                    checkpoint_file = checkpoint["file"]
                    checkpoint_path = self.manifest_path.parent / f"shard_{rank}" / checkpoint_file

                    # Remove checkpoint file
                    if checkpoint_path.exists():
                        checkpoint_path.unlink()
                        removed_count += 1

                    # Remove .done marker
                    done_marker = checkpoint_path.with_suffix(checkpoint_path.suffix + '.done')
                    if done_marker.exists():
                        done_marker.unlink()

                if removed_count > 0:
                    logger.info(f"Shard {rank}: removed {removed_count} checkpoint files")
        finally:
            self._release_file_lock(fd)

    def _load_manifest(self) -> Dict:
        """Load manifest from disk with corruption recovery and version migration.

        Tries loading from primary file, then .tmp file, then .backup file.
        Performs version migration for older manifest formats.
        Raises ManifestCorruptedError if all three fail.

        Returns:
            Parsed manifest dictionary

        Raises:
            ManifestCorruptedError: If all recovery attempts fail
        """
        # Try primary file
        try:
            with open(self.manifest_path, 'r') as f:
                manifest = json.load(f)
                manifest_version = manifest.get("version", "1.0")
                if manifest_version == "1.0":
                    manifest.setdefault("input_fingerprint", "")
                    manifest.setdefault("model_config_hash", "")
                    manifest.setdefault("global_checkpoints", [])
                    manifest["version"] = "2.0"
                return manifest
        except json.JSONDecodeError as e:
            logger.error(f"Primary manifest corrupted: {e}")
        except FileNotFoundError:
            # Primary doesn't exist yet - not an error during initialization
            raise

        # Try .tmp file
        tmp_path = self.manifest_path.with_suffix('.tmp')
        if tmp_path.exists():
            try:
                with open(tmp_path, 'r') as f:
                    logger.warning("Loaded manifest from .tmp file (primary was corrupted)")
                    return json.load(f)
            except json.JSONDecodeError as e:
                logger.error(f"Temp manifest corrupted: {e}")

        # Try .backup file
        backup_path = self.manifest_path.with_suffix('.backup')
        if backup_path.exists():
            try:
                with open(backup_path, 'r') as f:
                    logger.warning("Loaded manifest from .backup file (primary and .tmp were corrupted)")
                    return json.load(f)
            except json.JSONDecodeError as e:
                logger.error(f"Backup manifest corrupted: {e}")

        # All recovery attempts failed
        raise ManifestCorruptedError(
            f"Manifest corrupted: primary, .tmp, and .backup all failed to parse. "
            f"Path: {self.manifest_path}",
            manifest_path=self.manifest_path
        )

    def _save_manifest(self, manifest: Dict):
        """Save manifest to disk with atomic write.

        Uses atomic replace with .tmp file for crash recovery. No separate backup
        is needed since .tmp file provides recovery capability if write is interrupted.

        Args:
            manifest: Manifest dictionary to save
        """
        temp_path = self.manifest_path.with_suffix('.tmp')
        with open(temp_path, 'w') as f:
            json.dump(manifest, f, indent=2)

        temp_path.replace(self.manifest_path)

    def exists(self) -> bool:
        """Check if manifest file exists.

        Returns:
            True if manifest file exists
        """
        return self.manifest_path.exists()
