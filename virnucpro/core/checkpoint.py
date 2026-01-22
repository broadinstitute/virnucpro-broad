"""Checkpointing system for pipeline resume capability"""

import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum, IntEnum
import logging

logger = logging.getLogger('virnucpro.checkpoint')


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
