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
