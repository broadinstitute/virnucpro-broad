"""Checkpointing system for pipeline resume capability"""

import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum, IntEnum
import logging

import torch

from virnucpro.core.checkpoint_validation import (
    validate_checkpoint,
    CheckpointError
)

logger = logging.getLogger('virnucpro.checkpoint')

# Checkpoint version management
# Version 1.0: Optimized checkpoints with atomic write and validation
# Version 0.x: Pre-optimization checkpoints (backward compatible, read-only)
CHECKPOINT_VERSION = "1.0"


class StageStatus(Enum):
    """Pipeline stage status"""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class PipelineStage(IntEnum):
    """Pipeline stages for prediction (ordered)"""
    CHUNKING = 1
    TRANSLATION = 2
    NUCLEOTIDE_SPLITTING = 3
    PROTEIN_SPLITTING = 4
    NUCLEOTIDE_FEATURES = 5
    PROTEIN_FEATURES = 6
    FEATURE_MERGING = 7
    PREDICTION = 8
    CONSENSUS = 9


def atomic_save(
    checkpoint_dict: Dict[str, Any],
    output_file: Path,
    validate_after_save: bool = True,
    skip_validation: bool = False
) -> Path:
    """Save PyTorch checkpoint with atomic write to prevent corruption.

    Uses temp-then-rename pattern to ensure checkpoint is never partially written.
    Optionally validates checkpoint after save to ensure integrity.

    Args:
        checkpoint_dict: Dictionary to save as checkpoint
        output_file: Target path for checkpoint file
        validate_after_save: Validate checkpoint integrity after write (default: True)
        skip_validation: Skip validation (--skip-checkpoint-validation flag)

    Returns:
        Path to saved checkpoint file

    Raises:
        RuntimeError: If save or validation fails

    Example:
        >>> checkpoint = {'version': '1.0', 'status': 'complete', 'data': tensor}
        >>> atomic_save(checkpoint, Path("model.pt"))
        Path("model.pt")
    """
    temp_file = output_file.with_suffix('.tmp')

    try:
        # Embed version metadata if dict
        if isinstance(checkpoint_dict, dict):
            checkpoint_dict.setdefault('version', CHECKPOINT_VERSION)
            checkpoint_dict.setdefault('status', 'in_progress')

        # Save to temporary file
        torch.save(checkpoint_dict, temp_file)

        # Atomic rename (overwrites existing file)
        temp_file.replace(output_file)

        logger.debug(f"Checkpoint saved: {output_file}")

        # Validate after save if requested
        if validate_after_save and not skip_validation:
            is_valid, error_msg = validate_checkpoint(
                output_file,
                skip_load=False,  # Full validation
                logger_instance=logger
            )

            if not is_valid:
                # Remove corrupted file
                output_file.unlink()
                raise RuntimeError(f"Checkpoint validation failed: {error_msg}")

            logger.debug(f"Checkpoint validated successfully: {output_file}")

        # Mark checkpoint as complete after successful save/validation
        if isinstance(checkpoint_dict, dict) and 'status' in checkpoint_dict:
            checkpoint_dict['status'] = 'complete'
            torch.save(checkpoint_dict, temp_file)
            temp_file.replace(output_file)

    except Exception as e:
        # Clean up temp file on any failure
        if temp_file.exists():
            temp_file.unlink()
        raise RuntimeError(f"Failed to save checkpoint: {e}")

    return output_file


def load_with_compatibility(
    checkpoint_path: Path,
    skip_validation: bool = False,
    logger_instance: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """Load checkpoint with version compatibility checking.

    Handles backward compatibility with pre-optimization checkpoints (version 0.x).
    Raises error if checkpoint is from future version requiring upgrade.

    Args:
        checkpoint_path: Path to checkpoint file
        skip_validation: Skip validation (--skip-checkpoint-validation flag)
        logger_instance: Logger instance for diagnostic output

    Returns:
        Loaded checkpoint dict with version info

    Raises:
        CheckpointError: If checkpoint is corrupted or requires upgrade

    Example:
        >>> checkpoint = load_with_compatibility(Path("model.pt"))
        >>> version = checkpoint.get('version', '0.x')
        >>> print(f"Loading checkpoint version {version}")
    """
    log = logger_instance or logger

    # Validate checkpoint if not skipped
    if not skip_validation:
        is_valid, error_msg = validate_checkpoint(
            checkpoint_path,
            skip_load=False,
            logger_instance=log
        )

        if not is_valid:
            error_type = 'corrupted' if 'corrupted:' in error_msg else 'incompatible'
            raise CheckpointError(checkpoint_path, error_type, error_msg)

    # Load checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    except Exception as e:
        log.error(f"Failed to load checkpoint: {checkpoint_path}")
        log.error(f"  Error: {str(e)}")
        raise CheckpointError(checkpoint_path, 'corrupted', f"torch.load failed - {str(e)}")

    # Check version compatibility
    if isinstance(checkpoint, dict):
        version = checkpoint.get('version', '0.x')
        log.info(f"Loading checkpoint v{version}: {checkpoint_path.name}")

        # Check for future versions requiring upgrade
        if version.startswith('2.') or (version[0].isdigit() and int(version[0]) > 1):
            error_msg = f"Checkpoint version {version} requires virnucpro >= {version[0]}.0.0 (upgrade required)"
            log.error(f"Incompatible checkpoint version: {checkpoint_path}")
            log.error(f"  Version: {version}")
            log.error(f"  Current version: {CHECKPOINT_VERSION}")
            raise CheckpointError(checkpoint_path, 'incompatible', error_msg)

        # Handle version 0.x (pre-optimization) checkpoints
        if version == '0.x':
            log.warning(
                f"Loading pre-optimization checkpoint (read-only mode): {checkpoint_path.name}"
            )
            log.warning("  This checkpoint will not be modified to preserve compatibility")

    else:
        log.warning(f"Checkpoint is not a dict (type: {type(checkpoint).__name__})")

    return checkpoint


def load_checkpoint_safe(
    checkpoint_path: Path,
    skip_validation: bool = False,
    required_keys: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Load PyTorch checkpoint with validation.

    Args:
        checkpoint_path: Path to checkpoint file
        skip_validation: Skip validation (--skip-checkpoint-validation flag)
        required_keys: List of required keys in checkpoint dict

    Returns:
        Loaded checkpoint dict

    Raises:
        CheckpointError: If checkpoint is corrupted or incompatible

    Example:
        >>> checkpoint = load_checkpoint_safe(Path("model.pt"))
        >>> data = checkpoint['data']
    """
    if not skip_validation:
        # Validate before loading
        is_valid, error_msg = validate_checkpoint(
            checkpoint_path,
            required_keys=required_keys,
            skip_load=False,
            logger_instance=logger
        )

        if not is_valid:
            error_type = 'corrupted' if 'corrupted:' in error_msg else 'incompatible'
            raise CheckpointError(checkpoint_path, error_type, error_msg)

    # Load checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        logger.info(f"Checkpoint loaded: {checkpoint_path}")
        return checkpoint
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {checkpoint_path}")
        logger.error(f"  Error: {str(e)}")
        raise CheckpointError(checkpoint_path, 'corrupted', f"torch.load failed - {str(e)}")


class CheckpointManager:
    """
    Manages pipeline checkpoints for resume capability.

    Implements hash-based validation of inputs and parameters
    to detect when cached results can be reused.
    """

    def __init__(self, checkpoint_dir: Path, pipeline_config: Dict):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to store checkpoint data
            pipeline_config: Configuration dictionary for validation
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.pipeline_config = pipeline_config
        self.config_hash = self._compute_config_hash(pipeline_config)
        self.state_file = self.checkpoint_dir / "pipeline_state.json"

        logger.debug(f"Checkpoint manager initialized: {self.checkpoint_dir}")

    def _compute_config_hash(self, config: Dict) -> str:
        """Compute SHA256 hash of configuration"""
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()

    def _compute_file_checksum(self, filepath: Path) -> str:
        """Compute SHA256 checksum of a file"""
        sha256_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return f"sha256:{sha256_hash.hexdigest()}"

    def load_state(self) -> Dict:
        """
        Load pipeline state from checkpoint.

        Returns:
            State dictionary

        Raises:
            ValueError: If checkpoint config doesn't match current config
        """
        if self.state_file.exists():
            logger.info(f"Loading checkpoint from {self.state_file}")

            with open(self.state_file, 'r') as f:
                state = json.load(f)

            # Validate config compatibility
            checkpoint_hash = state.get('pipeline_config', {}).get('config_hash')
            if checkpoint_hash != self.config_hash:
                logger.warning(
                    f"Configuration changed since checkpoint was created.\n"
                    f"Checkpoint will be ignored and pipeline will run from scratch."
                )
                return self._create_initial_state()

            logger.info("Checkpoint loaded successfully")
            return state
        else:
            logger.debug("No checkpoint found, creating new state")
            return self._create_initial_state()

    def _create_initial_state(self) -> Dict:
        """Create initial pipeline state"""
        return {
            "created_at": datetime.utcnow().isoformat(),
            "pipeline_config": {
                "config_hash": self.config_hash,
                "expected_length": self.pipeline_config.get('expected_length'),
                "model_path": str(self.pipeline_config.get('model_path'))
            },
            "stages": {
                stage.name: {
                    "status": StageStatus.NOT_STARTED.value,
                    "started_at": None,
                    "completed_at": None,
                    "duration_seconds": None,
                    "inputs": {},
                    "outputs": {},
                    "error": None
                }
                for stage in PipelineStage
            }
        }

    def save_state(self, state: Dict):
        """Save pipeline state to checkpoint"""
        state['updated_at'] = datetime.utcnow().isoformat()

        # Save to temp file first, then atomic rename
        temp_file = self.state_file.with_suffix('.tmp')
        with open(temp_file, 'w') as f:
            json.dump(state, f, indent=2)

        temp_file.replace(self.state_file)
        logger.debug("Checkpoint saved")

    def find_resume_stage(self, state: Dict) -> Optional[PipelineStage]:
        """
        Determine which stage to resume from.

        Args:
            state: Current pipeline state

        Returns:
            Stage to resume from, or None if all complete
        """
        stages = list(PipelineStage)

        for stage in stages:
            stage_state = state['stages'][stage.name]
            status = StageStatus(stage_state['status'])

            # Resume from failed or in-progress stages
            if status in [StageStatus.FAILED, StageStatus.IN_PROGRESS]:
                logger.info(f"Resuming from stage: {stage.name} (status: {status.value})")
                return stage

            # Resume from first not-started stage
            if status == StageStatus.NOT_STARTED:
                logger.info(f"Starting from stage: {stage.name}")
                return stage

        # All stages completed
        logger.info("All stages already completed")
        return None

    def mark_stage_started(self, state: Dict, stage: PipelineStage):
        """Mark a stage as started"""
        stage_state = state['stages'][stage.name]
        stage_state['status'] = StageStatus.IN_PROGRESS.value
        stage_state['started_at'] = datetime.utcnow().isoformat()
        self.save_state(state)

        logger.info(f"Stage started: {stage.name}")

    def mark_stage_completed(
        self,
        state: Dict,
        stage: PipelineStage,
        outputs: Dict[str, Any]
    ):
        """
        Mark a stage as completed.

        Args:
            state: Pipeline state
            stage: Completed stage
            outputs: Output files/data from stage
        """
        stage_state = state['stages'][stage.name]
        started_at = stage_state.get('started_at')

        stage_state['status'] = StageStatus.COMPLETED.value
        stage_state['completed_at'] = datetime.utcnow().isoformat()

        # Calculate duration
        if started_at:
            start_dt = datetime.fromisoformat(started_at)
            end_dt = datetime.utcnow()
            duration = (end_dt - start_dt).total_seconds()
            stage_state['duration_seconds'] = duration

        # Save outputs
        stage_state['outputs'] = outputs

        self.save_state(state)

        logger.info(f"Stage completed: {stage.name}")
        if stage_state.get('duration_seconds'):
            logger.info(f"  Duration: {stage_state['duration_seconds']:.1f}s")

    def mark_stage_failed(self, state: Dict, stage: PipelineStage, error: str):
        """Mark a stage as failed"""
        stage_state = state['stages'][stage.name]
        stage_state['status'] = StageStatus.FAILED.value
        stage_state['failed_at'] = datetime.utcnow().isoformat()
        stage_state['error'] = error

        self.save_state(state)

        logger.error(f"Stage failed: {stage.name} - {error}")

    def can_skip_stage(self, state: Dict, stage: PipelineStage) -> bool:
        """
        Determine if a stage can be skipped based on checkpoint.

        Args:
            state: Pipeline state
            stage: Stage to check

        Returns:
            True if stage can be skipped
        """
        stage_state = state['stages'][stage.name]
        status = StageStatus(stage_state['status'])

        # Can only skip completed stages
        if status != StageStatus.COMPLETED:
            return False

        # Validate outputs still exist
        outputs = stage_state.get('outputs', {})
        if 'files' in outputs:
            for file_path in outputs['files']:
                if not Path(file_path).exists():
                    logger.warning(
                        f"Output file missing for {stage.name}: {file_path}"
                    )
                    return False

        logger.info(f"Skipping completed stage: {stage.name}")
        return True


class FileProgressTracker:
    """
    Track processing progress for individual files within a stage.

    Enables resume when only some files in a batch have been processed.
    """

    def __init__(self, stage_name: str, checkpoint_dir: Path):
        """
        Initialize file progress tracker.

        Args:
            stage_name: Name of the pipeline stage
            checkpoint_dir: Checkpoint directory
        """
        self.stage_name = stage_name
        self.progress_file = checkpoint_dir / f"{stage_name}_files.json"

    def load_progress(self, input_files: List[Path]) -> Dict:
        """
        Load or initialize file processing progress.

        Args:
            input_files: List of files to process

        Returns:
            Progress dictionary
        """
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                progress = json.load(f)
            logger.debug(f"Loaded file progress for {self.stage_name}")
        else:
            progress = {
                'stage': self.stage_name,
                'total_files': len(input_files),
                'files': {}
            }

            for filepath in input_files:
                progress['files'][str(filepath)] = {
                    'status': 'pending',
                    'started_at': None,
                    'completed_at': None,
                    'output': None,
                    'error': None
                }

            self._save_progress(progress)

        return progress

    def _save_progress(self, progress: Dict):
        """Save file progress to disk"""
        with open(self.progress_file, 'w') as f:
            json.dump(progress, f, indent=2)

    def get_pending_files(self, progress: Dict) -> List[str]:
        """Get list of files that still need processing"""
        return [
            filepath
            for filepath, info in progress['files'].items()
            if info['status'] in ['pending', 'failed']
        ]

    def mark_file_completed(self, progress: Dict, filepath: str, output: str):
        """Mark a file as successfully processed"""
        progress['files'][filepath]['status'] = 'completed'
        progress['files'][filepath]['completed_at'] = datetime.utcnow().isoformat()
        progress['files'][filepath]['output'] = output
        self._save_progress(progress)

    def mark_file_failed(self, progress: Dict, filepath: str, error: str):
        """Mark a file as failed"""
        progress['files'][filepath]['status'] = 'failed'
        progress['files'][filepath]['error'] = error
        self._save_progress(progress)

    def get_summary(self, progress: Dict) -> Dict:
        """Get processing progress summary"""
        statuses = [f['status'] for f in progress['files'].values()]
        total = len(statuses)
        completed = statuses.count('completed')

        return {
            'total': total,
            'completed': completed,
            'pending': statuses.count('pending'),
            'failed': statuses.count('failed'),
            'percentage': (completed / total * 100) if total > 0 else 0
        }
