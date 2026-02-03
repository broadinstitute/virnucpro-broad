"""Enhanced GPU monitoring for benchmarks with nvitop integration.

This module provides comprehensive GPU monitoring for performance validation:

1. GPUMonitor: Basic monitoring using torch.cuda (backward compatible)
2. NvitopMonitor: Enhanced monitoring with nvitop for detailed metrics
3. Benchmark-specific tracking: Per-stage metrics, idle time detection
4. Log file output: Utilization and memory logs for MON-01 compliance

The module automatically falls back to GPUMonitor if nvitop is not available.
"""

import torch
import logging
import threading
import time
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
import json

logger = logging.getLogger('virnucpro.utils.gpu_monitor')


# ==================== Backward Compatible GPUMonitor ====================

class GPUMonitor:
    """
    Basic GPU memory monitor using torch.cuda APIs.

    Provides backward compatibility with existing code. Monitors GPU memory
    usage across multiple devices in a background thread.

    This is the fallback implementation when nvitop is not available.
    """

    def __init__(self, device_ids: List[int], log_interval: float = 10.0):
        """
        Initialize GPU monitor.

        Args:
            device_ids: List of CUDA device IDs to monitor
            log_interval: Seconds between logging memory stats (default: 10.0)
        """
        self.device_ids = device_ids
        self.log_interval = log_interval
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._stats_cache: Dict[int, Dict[str, float]] = {}
        self._lock = threading.Lock()

        # Validate device IDs
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, GPU monitoring disabled")
            return

        num_gpus = torch.cuda.device_count()
        for device_id in device_ids:
            if device_id >= num_gpus:
                raise ValueError(f"Invalid device_id {device_id}, only {num_gpus} GPUs available")

    def start_monitoring(self):
        """Start background memory monitoring thread."""
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, monitoring not started")
            return

        if self._monitoring:
            logger.warning("Monitoring already started")
            return

        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info(f"Started GPU monitoring for devices {self.device_ids} (interval: {self.log_interval}s)")

    def stop_monitoring(self):
        """Stop background memory monitoring thread."""
        if not self._monitoring:
            return

        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=self.log_interval + 1.0)
        logger.info("Stopped GPU monitoring")

    def _monitor_loop(self):
        """Background monitoring loop that runs in separate thread."""
        while self._monitoring:
            try:
                self._update_stats()
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")

            # Sleep in small increments to allow quick shutdown
            for _ in range(int(self.log_interval * 10)):
                if not self._monitoring:
                    break
                time.sleep(0.1)

    def _update_stats(self):
        """Update cached statistics for all monitored GPUs."""
        with self._lock:
            for device_id in self.device_ids:
                try:
                    free_bytes, total_bytes = torch.cuda.mem_get_info(device_id)
                    used_bytes = total_bytes - free_bytes
                    percent_used = (used_bytes / total_bytes) * 100.0 if total_bytes > 0 else 0.0

                    self._stats_cache[device_id] = {
                        'free': free_bytes,
                        'total': total_bytes,
                        'used': used_bytes,
                        'percent': percent_used
                    }
                except Exception as e:
                    logger.error(f"Failed to update stats for GPU {device_id}: {e}")

    def get_current_stats(self) -> Dict[int, Dict[str, float]]:
        """
        Get current memory statistics for all monitored GPUs.

        Returns:
            Dictionary mapping device_id -> memory info dict
        """
        with self._lock:
            return self._stats_cache.copy()


# ==================== Enhanced NvitopMonitor ====================

@dataclass
class GPUMetrics:
    """GPU metrics snapshot."""
    timestamp: float
    device_id: int
    gpu_util: float  # 0-100%
    mem_used: int  # bytes
    mem_total: int  # bytes
    mem_percent: float  # 0-100%
    temperature: Optional[float] = None  # Celsius
    power_draw: Optional[float] = None  # Watts
    stage: Optional[str] = None  # Pipeline stage name


@dataclass
class DataLoaderMetrics:
    """
    DataLoader performance metrics snapshot.

    NOTE: queue_depth is a heuristic estimate, not actual DataLoader queue state.
    PyTorch DataLoader doesn't expose internal queue depth. We infer from timing:
    - wait_time_ms < 1ms → queue likely full (batch ready)
    - wait_time_ms > 50ms → queue likely empty (starved)
    """
    timestamp: float
    wait_time_ms: float        # Time spent waiting for next batch
    batch_idx: int             # Current batch index
    sequences_in_batch: int    # Sequences in this batch
    tokens_in_batch: int       # Tokens in this batch (for packed batches)

    # Batch composition (for packed batches)
    avg_sequence_length: float = 0.0  # Average tokens per sequence
    max_sequence_length: int = 0      # Longest sequence in batch

    # Heuristic queue state (not actual DataLoader queue)
    queue_state: str = 'unknown'  # 'full' | 'starved' | 'normal' | 'unknown'


def infer_queue_state(wait_time_ms: float) -> str:
    """Infer DataLoader queue state from wait time heuristic."""
    if wait_time_ms < 1.0:
        return 'full'      # Batch was ready, queue had data
    elif wait_time_ms > 50.0:
        return 'starved'   # Long wait, queue was likely empty
    else:
        return 'normal'    # Typical prefetch timing


class NvitopMonitor:
    """
    Enhanced GPU monitor with nvitop integration for detailed metrics.

    Provides:
    - Real-time GPU utilization tracking
    - Memory usage monitoring
    - Temperature and power draw tracking
    - Per-stage performance metrics
    - Idle time detection between stages
    - Log file output for MON-01 compliance

    Falls back gracefully if nvitop not available.
    """

    def __init__(self,
                 device_ids: List[int],
                 log_interval: float = 1.0,
                 log_file: Optional[Path] = None):
        """
        Initialize enhanced GPU monitor.

        Args:
            device_ids: List of CUDA device IDs to monitor
            log_interval: Seconds between metric sampling (default: 1.0)
            log_file: Path to log file (default: logs/gpu_metrics_{timestamp}.log)
        """
        self.device_ids = device_ids
        self.log_interval = log_interval
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._metrics: List[GPUMetrics] = []
        self._lock = threading.Lock()
        self._current_stage: Optional[str] = None

        # DataLoader metrics tracking
        self._dataloader_metrics: List[DataLoaderMetrics] = []
        self._batch_log_interval: int = 10  # Log every N batches
        self._total_sequences: int = 0
        self._total_tokens: int = 0
        self._inference_start_time: Optional[float] = None

        # Setup log file
        if log_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_dir = Path('logs')
            log_dir.mkdir(exist_ok=True)
            log_file = log_dir / f'gpu_metrics_{timestamp}.log'

        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        # Try to import nvitop
        self._nvitop_available = False
        self._devices = []

        try:
            from nvitop import Device
            self._nvitop_available = True

            # Initialize devices
            for device_id in device_ids:
                try:
                    device = Device(device_id)
                    self._devices.append(device)
                except Exception as e:
                    logger.warning(f"Failed to initialize nvitop for GPU {device_id}: {e}")

            if self._devices:
                logger.info(f"Nvitop monitoring initialized for {len(self._devices)} GPUs")
            else:
                logger.warning("Nvitop available but failed to initialize devices, falling back to torch.cuda")
                self._nvitop_available = False

        except ImportError:
            logger.warning("nvitop not available, using fallback monitoring (torch.cuda only)")
            self._nvitop_available = False

    def start_monitoring(self):
        """Start background monitoring thread with logging."""
        if self._monitoring:
            logger.warning("Monitoring already started")
            return

        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()

        logger.info(f"Started enhanced GPU monitoring for devices {self.device_ids}")
        logger.info(f"Logging to: {self.log_file}")

        # Write log header
        self._write_log_header()

    def stop_monitoring(self) -> Dict[int, Dict[str, Any]]:
        """
        Stop monitoring and return aggregated statistics.

        Returns:
            Dictionary mapping device_id -> statistics
        """
        if not self._monitoring:
            return {}

        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=self.log_interval + 1.0)

        logger.info("Stopped GPU monitoring")

        # Calculate and return statistics
        stats = self.get_statistics()

        # Save final summary to log
        self._write_log_footer(stats)

        return stats

    def set_stage(self, stage: str):
        """
        Set current pipeline stage for tracking.

        Args:
            stage: Stage name (e.g., 'translation', 'dnabert', 'esm2', 'merge')
        """
        with self._lock:
            self._current_stage = stage
            logger.info(f"GPU monitoring: entering stage '{stage}'")

    def record_dataloader_wait(
        self,
        wait_time_ms: float,
        batch_idx: int,
        sequences_in_batch: int,
        tokens_in_batch: int = 0,
        avg_sequence_length: float = 0.0,
        max_sequence_length: int = 0
    ) -> None:
        """
        Record DataLoader fetch timing for bottleneck detection.

        Args:
            wait_time_ms: Time spent waiting for DataLoader to return next batch
            batch_idx: Current batch index
            sequences_in_batch: Number of sequences in batch
            tokens_in_batch: Total tokens in batch (for packed batches)
            avg_sequence_length: Average tokens per sequence
            max_sequence_length: Longest sequence in batch
        """
        metrics = DataLoaderMetrics(
            timestamp=time.time(),
            wait_time_ms=wait_time_ms,
            batch_idx=batch_idx,
            sequences_in_batch=sequences_in_batch,
            tokens_in_batch=tokens_in_batch,
            avg_sequence_length=avg_sequence_length,
            max_sequence_length=max_sequence_length,
            queue_state=infer_queue_state(wait_time_ms)
        )
        with self._lock:
            self._dataloader_metrics.append(metrics)
            self._total_sequences += sequences_in_batch
            self._total_tokens += tokens_in_batch

        # Log every N batches per CONTEXT.md decision
        if batch_idx % self._batch_log_interval == 0:
            logger.info(
                f"Batch {batch_idx}: wait={wait_time_ms:.1f}ms, "
                f"seqs={sequences_in_batch}, tokens={tokens_in_batch}, "
                f"queue_state={metrics.queue_state}"
            )

    def check_bottleneck(self, recent_samples: int = 10) -> Tuple[bool, str, float]:
        """
        Check if GPU is idle too often indicating I/O bottleneck.

        Tiered thresholds to avoid false positives with short sequences:
        - <50%: Critical bottleneck (definitely I/O bound)
        - <80%: Mild bottleneck (may be batch size or I/O issue)
        - ≥80%: No bottleneck

        Returns:
            Tuple of (is_bottleneck, severity, avg_utilization)
            severity: 'critical' | 'mild' | 'none'
        """
        with self._lock:
            if len(self._metrics) < recent_samples:
                return False, 'none', 0.0

            recent = self._metrics[-recent_samples:]
            avg_util = sum(m.gpu_util for m in recent) / len(recent)

        # Tiered bottleneck detection
        if avg_util < 50:
            logger.warning(
                f"CRITICAL I/O bottleneck: GPU utilization {avg_util:.1f}% "
                f"(threshold: 50%). DataLoader is starving the GPU."
            )
            return True, 'critical', avg_util
        elif avg_util < 80:
            logger.info(
                f"Mild I/O bottleneck: GPU utilization {avg_util:.1f}% "
                f"(threshold: 80%). Consider increasing batch size or prefetch_factor."
            )
            return True, 'mild', avg_util
        else:
            return False, 'none', avg_util

    def get_dataloader_statistics(self) -> Dict[str, Any]:
        """Get aggregated DataLoader performance statistics."""
        with self._lock:
            metrics = self._dataloader_metrics.copy()
            total_sequences = self._total_sequences
            total_tokens = self._total_tokens

        if not metrics:
            return {}

        wait_times = [m.wait_time_ms for m in metrics]
        queue_states = [m.queue_state for m in metrics]

        # Calculate packing efficiency from batch composition
        seq_lengths = [m.avg_sequence_length for m in metrics if m.avg_sequence_length > 0]
        max_lengths = [m.max_sequence_length for m in metrics if m.max_sequence_length > 0]

        # Packing efficiency: actual_tokens / (num_sequences * max_length_in_batch)
        # For packed batches: high efficiency means less padding waste
        total_actual_tokens = sum(m.tokens_in_batch for m in metrics)
        total_theoretical_tokens = sum(
            m.sequences_in_batch * m.max_sequence_length
            for m in metrics if m.max_sequence_length > 0
        )
        packing_efficiency = (
            total_actual_tokens / total_theoretical_tokens
            if total_theoretical_tokens > 0 else 0.0
        )

        # Queue state distribution
        queue_state_counts = {
            'full': sum(1 for s in queue_states if s == 'full'),
            'normal': sum(1 for s in queue_states if s == 'normal'),
            'starved': sum(1 for s in queue_states if s == 'starved'),
        }

        return {
            # Wait time metrics
            'avg_wait_time_ms': sum(wait_times) / len(wait_times),
            'max_wait_time_ms': max(wait_times),
            'min_wait_time_ms': min(wait_times),
            'p95_wait_time_ms': sorted(wait_times)[int(len(wait_times) * 0.95)] if wait_times else 0,

            # Queue state heuristic (not actual queue depth)
            'queue_state_distribution': queue_state_counts,
            'pct_starved': queue_state_counts['starved'] / len(metrics) * 100 if metrics else 0,

            # Batch composition
            'avg_sequence_length': sum(seq_lengths) / len(seq_lengths) if seq_lengths else 0,
            'avg_max_length': sum(max_lengths) / len(max_lengths) if max_lengths else 0,

            # Packing efficiency
            'packing_efficiency': packing_efficiency,

            # Volume
            'total_batches': len(metrics),
            'total_sequences': total_sequences,
            'total_tokens': total_tokens,
        }

    def start_inference_timer(self) -> None:
        """Start timer for throughput calculation."""
        self._inference_start_time = time.time()

    def get_throughput(self) -> Dict[str, float]:
        """
        Get throughput metrics.

        Returns both sequences/sec and tokens/sec.
        tokens/sec is more stable for packed batches with variable sequence lengths.
        """
        if self._inference_start_time is None:
            return {
                'sequences_per_sec': 0.0,
                'tokens_per_sec': 0.0,
                'elapsed_seconds': 0.0
            }

        elapsed = time.time() - self._inference_start_time
        seqs_per_sec = self._total_sequences / elapsed if elapsed > 0 else 0.0
        tokens_per_sec = self._total_tokens / elapsed if elapsed > 0 else 0.0

        return {
            'sequences_per_sec': seqs_per_sec,
            'tokens_per_sec': tokens_per_sec,
            'elapsed_seconds': elapsed,
            'total_sequences': self._total_sequences,
            'total_tokens': self._total_tokens,
        }

    def _monitor_loop(self):
        """Background monitoring loop."""
        while self._monitoring:
            try:
                self._collect_metrics()
            except Exception as e:
                logger.error(f"Error collecting metrics: {e}")

            # Sleep in small increments for responsive shutdown
            for _ in range(int(self.log_interval * 10)):
                if not self._monitoring:
                    break
                time.sleep(0.1)

    def _collect_metrics(self):
        """Collect metrics from all monitored GPUs."""
        timestamp = time.time()
        current_stage = None

        with self._lock:
            current_stage = self._current_stage

        if self._nvitop_available and self._devices:
            # Use nvitop for detailed metrics
            for device in self._devices:
                try:
                    metrics = GPUMetrics(
                        timestamp=timestamp,
                        device_id=device.index,
                        gpu_util=device.gpu_utilization(),
                        mem_used=device.memory_used(),
                        mem_total=device.memory_total(),
                        mem_percent=(device.memory_used() / device.memory_total() * 100)
                                   if device.memory_total() > 0 else 0.0,
                        temperature=device.temperature(),
                        power_draw=device.power_usage(),
                        stage=current_stage
                    )

                    with self._lock:
                        self._metrics.append(metrics)

                    # Write to log
                    self._write_log_entry(metrics)

                except Exception as e:
                    logger.error(f"Failed to collect nvitop metrics for GPU {device.index}: {e}")

        else:
            # Fallback to torch.cuda
            for device_id in self.device_ids:
                try:
                    if not torch.cuda.is_available():
                        continue

                    free_bytes, total_bytes = torch.cuda.mem_get_info(device_id)
                    used_bytes = total_bytes - free_bytes
                    mem_percent = (used_bytes / total_bytes * 100) if total_bytes > 0 else 0.0

                    metrics = GPUMetrics(
                        timestamp=timestamp,
                        device_id=device_id,
                        gpu_util=0.0,  # Not available via torch.cuda
                        mem_used=used_bytes,
                        mem_total=total_bytes,
                        mem_percent=mem_percent,
                        stage=current_stage
                    )

                    with self._lock:
                        self._metrics.append(metrics)

                    self._write_log_entry(metrics)

                except Exception as e:
                    logger.error(f"Failed to collect torch.cuda metrics for GPU {device_id}: {e}")

    def get_statistics(self) -> Dict[int, Dict[str, Any]]:
        """
        Calculate aggregated statistics from collected metrics.

        Returns:
            Dictionary mapping device_id -> statistics:
            - gpu_util_avg: Average GPU utilization (%)
            - gpu_util_min: Minimum GPU utilization (%)
            - gpu_util_max: Maximum GPU utilization (%)
            - mem_used_avg: Average memory usage (bytes)
            - mem_used_peak: Peak memory usage (bytes)
            - mem_percent_avg: Average memory percentage
            - sample_count: Number of samples collected
            - by_stage: Per-stage statistics (if stages tracked)
        """
        with self._lock:
            metrics = self._metrics.copy()

        if not metrics:
            return {}

        # Group by device
        by_device: Dict[int, List[GPUMetrics]] = {}
        for m in metrics:
            if m.device_id not in by_device:
                by_device[m.device_id] = []
            by_device[m.device_id].append(m)

        stats = {}

        for device_id, device_metrics in by_device.items():
            gpu_utils = [m.gpu_util for m in device_metrics]
            mem_useds = [m.mem_used for m in device_metrics]
            mem_percents = [m.mem_percent for m in device_metrics]

            device_stats = {
                'gpu_util_avg': sum(gpu_utils) / len(gpu_utils) if gpu_utils else 0.0,
                'gpu_util_min': min(gpu_utils) if gpu_utils else 0.0,
                'gpu_util_max': max(gpu_utils) if gpu_utils else 0.0,
                'mem_used_avg': sum(mem_useds) / len(mem_useds) if mem_useds else 0,
                'mem_used_peak': max(mem_useds) if mem_useds else 0,
                'mem_percent_avg': sum(mem_percents) / len(mem_percents) if mem_percents else 0.0,
                'sample_count': len(device_metrics),
            }

            # Per-stage statistics
            stages = set(m.stage for m in device_metrics if m.stage)
            if stages:
                device_stats['by_stage'] = {}
                for stage in stages:
                    stage_metrics = [m for m in device_metrics if m.stage == stage]
                    stage_utils = [m.gpu_util for m in stage_metrics]
                    stage_mems = [m.mem_used for m in stage_metrics]

                    device_stats['by_stage'][stage] = {
                        'gpu_util_avg': sum(stage_utils) / len(stage_utils) if stage_utils else 0.0,
                        'mem_used_peak': max(stage_mems) if stage_mems else 0,
                        'duration_seconds': (stage_metrics[-1].timestamp - stage_metrics[0].timestamp)
                                           if len(stage_metrics) > 1 else 0.0,
                    }

            stats[device_id] = device_stats

        # Add DataLoader statistics (device-agnostic)
        dataloader_stats = self.get_dataloader_statistics()
        if dataloader_stats:
            # Return as a special entry in the stats dict
            stats['dataloader'] = dataloader_stats

        return stats

    def get_average_utilization(self, device_id: Optional[int] = None) -> float:
        """
        Get average GPU utilization.

        Args:
            device_id: Specific device ID (None = average across all)

        Returns:
            Average utilization percentage (0-100)
        """
        stats = self.get_statistics()

        if device_id is not None:
            return stats.get(device_id, {}).get('gpu_util_avg', 0.0)

        # Average across all devices
        if not stats:
            return 0.0

        utils = [s['gpu_util_avg'] for s in stats.values()]
        return sum(utils) / len(utils) if utils else 0.0

    def get_peak_memory_usage(self, device_id: Optional[int] = None) -> int:
        """
        Get peak memory usage.

        Args:
            device_id: Specific device ID (None = max across all)

        Returns:
            Peak memory usage in bytes
        """
        stats = self.get_statistics()

        if device_id is not None:
            return stats.get(device_id, {}).get('mem_used_peak', 0)

        # Max across all devices
        if not stats:
            return 0

        peaks = [s['mem_used_peak'] for s in stats.values()]
        return max(peaks) if peaks else 0

    def get_throughput_metrics(self,
                               num_sequences: int,
                               total_time: float) -> Dict[str, float]:
        """
        Calculate throughput metrics.

        Args:
            num_sequences: Total sequences processed
            total_time: Total processing time (seconds)

        Returns:
            Dictionary with:
            - sequences_per_second: Throughput
            - seconds_per_sequence: Average latency
            - gpu_util_avg: Average GPU utilization during processing
        """
        throughput = num_sequences / total_time if total_time > 0 else 0.0
        latency = total_time / num_sequences if num_sequences > 0 else 0.0

        return {
            'sequences_per_second': throughput,
            'seconds_per_sequence': latency,
            'gpu_util_avg': self.get_average_utilization(),
        }

    def export_metrics(self, output_path: Path, format: str = 'json'):
        """
        Export metrics to file.

        Args:
            output_path: Output file path
            format: 'json' or 'csv'
        """
        output_path = Path(output_path)
        stats = self.get_statistics()

        if format == 'json':
            with open(output_path, 'w') as f:
                json.dump(stats, f, indent=2)

        elif format == 'csv':
            import csv
            with open(output_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'device_id', 'gpu_util', 'mem_used', 'mem_percent', 'stage'])

                with self._lock:
                    for m in self._metrics:
                        writer.writerow([
                            m.timestamp,
                            m.device_id,
                            m.gpu_util,
                            m.mem_used,
                            m.mem_percent,
                            m.stage or ''
                        ])

        logger.info(f"Metrics exported to {output_path}")

    def save_logs(self):
        """
        Finalize and save log file.

        Ensures logs are written to disk with final summary.
        """
        stats = self.get_statistics()
        self._write_log_footer(stats)
        logger.info(f"GPU metrics log saved to {self.log_file}")

    def _write_log_header(self):
        """Write log file header."""
        with open(self.log_file, 'w') as f:
            f.write(f"# GPU Monitoring Log\n")
            f.write(f"# Started: {datetime.now().isoformat()}\n")
            f.write(f"# Devices: {self.device_ids}\n")
            f.write(f"# Nvitop: {'available' if self._nvitop_available else 'fallback'}\n")
            f.write(f"#\n")
            f.write(f"# timestamp,device_id,gpu_util,mem_used_mb,mem_percent,temperature,power_draw,stage\n")

    def _write_log_entry(self, metrics: GPUMetrics):
        """Write single log entry."""
        with open(self.log_file, 'a') as f:
            f.write(
                f"{metrics.timestamp:.3f},"
                f"{metrics.device_id},"
                f"{metrics.gpu_util:.1f},"
                f"{metrics.mem_used / (1024**2):.1f},"
                f"{metrics.mem_percent:.1f},"
                f"{metrics.temperature or ''},"
                f"{metrics.power_draw or ''},"
                f"{metrics.stage or ''}\n"
            )

    def _write_log_footer(self, stats: Dict[int, Dict[str, Any]]):
        """Write log file footer with summary."""
        with open(self.log_file, 'a') as f:
            f.write(f"#\n")
            f.write(f"# Monitoring Stopped: {datetime.now().isoformat()}\n")
            f.write(f"#\n")
            f.write(f"# Summary:\n")

            for device_id, device_stats in stats.items():
                f.write(f"# GPU {device_id}:\n")
                f.write(f"#   Avg Utilization: {device_stats['gpu_util_avg']:.1f}%\n")
                f.write(f"#   Peak Memory: {device_stats['mem_used_peak'] / (1024**3):.2f} GB\n")
                f.write(f"#   Samples: {device_stats['sample_count']}\n")

                if 'by_stage' in device_stats:
                    f.write(f"#   By Stage:\n")
                    for stage, stage_stats in device_stats['by_stage'].items():
                        f.write(f"#     {stage}: {stage_stats['gpu_util_avg']:.1f}% util, "
                               f"{stage_stats['mem_used_peak'] / (1024**3):.2f} GB peak, "
                               f"{stage_stats['duration_seconds']:.1f}s\n")
