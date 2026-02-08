"""Pipeline telemetry for per-stage wall-clock timing and summary.

Provides PipelineTelemetry class that tracks timing for each pipeline stage
and produces a formatted summary block at pipeline completion. All output
goes through the virnucpro logging system -- no separate JSON files.

Integration:
    Used by prediction.py to instrument the 9-stage pipeline.
    Each stage calls start_stage()/end_stage() bracketing its work.
    After all stages, log_summary() prints the timing breakdown.

Example:
    telemetry = PipelineTelemetry()
    telemetry.start_stage("Sequence Chunking")
    # ... do chunking ...
    telemetry.end_stage("Sequence Chunking", {'sequences': 50000})
    # ... more stages ...
    telemetry.log_summary()
"""

import logging
import time
from typing import Dict, List, Optional

logger = logging.getLogger('virnucpro.pipeline.telemetry')


class PipelineTelemetry:
    """Tracks per-stage wall-clock timing and produces a pipeline summary.

    All output goes through the logging system (logger.info). No JSON files
    are created. The summary block includes:
    - Total pipeline wall time
    - Per-stage timing breakdown with percentages
    - Top 3 bottleneck stages
    - Key metrics accumulated from extra dicts
    - Optional v1.0 baseline speedup comparison
    """

    def __init__(self) -> None:
        self._stages: Dict[str, Dict] = {}
        self._stage_order: List[str] = []
        self._pipeline_start: Optional[float] = None
        self._pipeline_end: Optional[float] = None

    def start_stage(self, name: str) -> None:
        """Record the start of a pipeline stage.

        Args:
            name: Human-readable stage name (e.g. "Sequence Chunking")
        """
        now = time.monotonic()
        if self._pipeline_start is None:
            self._pipeline_start = now

        stage_num = len(self._stage_order) + 1
        self._stages[name] = {'start': now, 'end': None, 'extra': {}}
        self._stage_order.append(name)

        logger.info(f"=== Stage {stage_num}: {name} ===")

    def end_stage(self, name: str, extra: Optional[Dict] = None) -> None:
        """Record the end of a pipeline stage and log elapsed time.

        Args:
            name: Stage name matching a previous start_stage() call
            extra: Optional dict with stage metrics (e.g. sequences, files, tokens)
        """
        now = time.monotonic()

        if name not in self._stages:
            logger.warning(f"end_stage called for unknown stage: {name}")
            return

        stage = self._stages[name]
        stage['end'] = now
        if extra:
            stage['extra'] = extra

        elapsed = now - stage['start']

        # Build extra info string
        extra_parts = []
        if extra:
            for key, value in extra.items():
                if key == 'skipped' and value:
                    extra_parts.append(f"skipped ({extra.get('reason', 'unknown')})")
                elif isinstance(value, (int, float)) and key not in ('skipped',):
                    if isinstance(value, int):
                        extra_parts.append(f"{key}: {value:,}")
                    else:
                        extra_parts.append(f"{key}: {value:.1f}")
                elif isinstance(value, str):
                    extra_parts.append(f"{key}: {value}")

        extra_str = f" [{', '.join(extra_parts)}]" if extra_parts else ""
        logger.info(f"  {name} completed in {elapsed:.1f}s{extra_str}")

    def log_summary(self, v1_baseline_seconds: Optional[float] = None) -> None:
        """Print a formatted pipeline summary block via logging.

        Args:
            v1_baseline_seconds: Optional v1.0 total pipeline time in seconds.
                When provided, computes and displays speedup ratio.
        """
        self._pipeline_end = time.monotonic()

        if self._pipeline_start is None:
            logger.warning("No stages recorded -- nothing to summarize")
            return

        total_time = self._pipeline_end - self._pipeline_start
        total_hms = _format_hms(total_time)

        # Build per-stage timing data
        stage_timings = []
        for i, name in enumerate(self._stage_order):
            stage = self._stages[name]
            if stage['end'] is not None:
                elapsed = stage['end'] - stage['start']
                skipped = stage['extra'].get('skipped', False)
            else:
                elapsed = 0.0
                skipped = False
            pct = (elapsed / total_time * 100) if total_time > 0 else 0.0
            stage_timings.append((i + 1, name, elapsed, pct, skipped))

        # Identify top 3 bottlenecks (non-skipped stages only)
        active_stages = [(name, elapsed, pct) for _, name, elapsed, pct, skipped in stage_timings if not skipped]
        bottlenecks = sorted(active_stages, key=lambda x: x[1], reverse=True)[:3]

        # Accumulate key metrics from extra dicts
        total_sequences = 0
        architecture = None
        for name in self._stage_order:
            extra = self._stages[name].get('extra', {})
            if 'sequences' in extra:
                total_sequences += extra['sequences']
            if 'architecture' in extra:
                architecture = extra['architecture']

        # Build summary block
        lines = []
        lines.append("")
        lines.append("=" * 80)
        lines.append("PIPELINE SUMMARY")
        lines.append("=" * 80)
        lines.append(f"Total wall time: {total_hms} ({total_time:.1f}s)")
        lines.append("")
        lines.append("Stages:")

        # Find the longest stage name for alignment
        max_name_len = max(len(name) for _, name, _, _, _ in stage_timings) if stage_timings else 20
        for num, name, elapsed, pct, skipped in stage_timings:
            dots = '.' * (max_name_len - len(name) + 4)
            if skipped:
                lines.append(f"  {num}. {name} {dots}  skipped")
            else:
                lines.append(f"  {num}. {name} {dots}  {elapsed:6.1f}s ({pct:5.1f}%)")

        lines.append("")
        if bottlenecks:
            bottleneck_parts = [f"{name} ({pct:.0f}%)" for name, _, pct in bottlenecks if pct > 0]
            if bottleneck_parts:
                lines.append(f"Bottlenecks: {', '.join(bottleneck_parts)}")

        lines.append("")
        lines.append("Key metrics:")
        if total_sequences > 0:
            lines.append(f"  Sequences processed: {total_sequences:,}")
        if architecture:
            arch_desc = "async DataLoader + FlashAttention" if architecture == 'v2.0' else "standard extraction"
            lines.append(f"  ESM-2 architecture: {architecture} ({arch_desc})")

        if v1_baseline_seconds is not None and v1_baseline_seconds > 0:
            speedup = v1_baseline_seconds / total_time if total_time > 0 else 0.0
            lines.append(f"  v1.0 baseline: {_format_hms(v1_baseline_seconds)} ({v1_baseline_seconds:.1f}s)")
            lines.append(f"  Speedup: {speedup:.1f}x")

        lines.append("=" * 80)

        # Log as a single block
        for line in lines:
            logger.info(line)


def _format_hms(seconds: float) -> str:
    """Format seconds as HH:MM:SS string.

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted string like "01:23:45"
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"
