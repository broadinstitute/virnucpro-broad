"""Runtime configuration for GPU inference operations.

Separates operational concerns (checkpointing, timeouts, retry policies) from
model architecture configuration (dtype, hidden_size, num_layers).

This separation ensures:
- Model config remains architecture-only for reproducibility
- Runtime params don't pollute checkpoint metadata
- Clear distinction between "what model" vs "how to run it"
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional


@dataclass
class RuntimeConfig:
    """Runtime operational configuration for GPU inference.

    These parameters control execution behavior, not model architecture.
    They are NOT saved to checkpoint metadata.

    Attributes:
        enable_checkpointing: Enable incremental checkpointing
        checkpoint_dir: Directory for checkpoint files (default: output_dir/checkpoints)
        force_restart: Ignore existing checkpoints and start fresh
        checkpoint_seq_threshold: Sequence count trigger for checkpoint
        checkpoint_time_threshold: Time threshold (seconds) for checkpoint
        timeout_per_attempt: Timeout per retry attempt (not global)
        max_retries_transient: Max retries for transient errors (OOM, network)
        max_retries_poison: Max retries for same batch before circuit breaker (default: 2)
        spot_retry_poll_interval: Seconds to wait between spot capacity polls (default: 60)
        enable_elastic_redistribution: Reassign failed work to healthy GPUs
    """

    # Checkpointing
    enable_checkpointing: bool = True
    checkpoint_dir: Optional[Path] = None
    force_restart: bool = False
    checkpoint_seq_threshold: int = 10000
    checkpoint_time_threshold: float = 300.0

    # Timeouts (Issue 7: per-attempt, not global)
    timeout_per_attempt: Optional[float] = 3600.0  # 1 hour per retry attempt

    # Retry policies (Issue 1: differentiated by failure type)
    max_retries_transient: int = 3  # OOM, network errors
    max_retries_poison: int = 2  # Same batch failures before circuit breaker
    spot_retry_poll_interval: float = 60.0  # Spot preemption: poll every 60s, infinite retries

    # Elastic redistribution (Issue 5)
    enable_elastic_redistribution: bool = True

    def to_dict(self) -> dict:
        """Convert to dict for passing to workers."""
        d = asdict(self)
        # Convert Path to str for serialization
        if self.checkpoint_dir:
            d['checkpoint_dir'] = str(self.checkpoint_dir)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> 'RuntimeConfig':
        """Reconstruct from dict."""
        if 'checkpoint_dir' in d and d['checkpoint_dir']:
            d['checkpoint_dir'] = Path(d['checkpoint_dir'])
        return cls(**d)
