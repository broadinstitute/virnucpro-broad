"""GPU monitoring utilities for adaptive batching and memory tracking"""

import torch
import logging
from typing import Dict, List, Optional, Tuple
import threading
import time

logger = logging.getLogger('virnucpro.pipeline.gpu_monitor')


def check_bf16_support() -> bool:
    """
    Check if current GPU supports BF16 (Brain Float 16) operations.

    BF16 is supported on Ampere GPUs (compute capability >= 8.0) and newer.
    Using BF16 can reduce memory usage by ~50% compared to FP32 and allows
    larger batch sizes.

    Returns:
        True if CUDA is available and GPU supports BF16, False otherwise
    """
    if not torch.cuda.is_available():
        logger.warning("CUDA not available, BF16 not supported")
        return False

    try:
        # Get compute capability of default device
        capability = torch.cuda.get_device_capability(0)
        major, minor = capability

        # Ampere and newer (8.0+) support BF16
        supports_bf16 = major >= 8

        if supports_bf16:
            logger.info(f"GPU compute capability {major}.{minor}: BF16 supported")
        else:
            logger.warning(f"GPU compute capability {major}.{minor}: BF16 not supported (requires 8.0+)")

        return supports_bf16

    except Exception as e:
        logger.error(f"Error checking BF16 support: {e}")
        return False


def get_gpu_memory_info(device_id: int = 0) -> Dict[str, float]:
    """
    Get current GPU memory usage statistics using direct CUDA API.

    Uses torch.cuda.mem_get_info() for accurate, real-time memory information
    without the overhead of subprocess calls to nvidia-smi.

    Args:
        device_id: CUDA device ID (default: 0)

    Returns:
        Dictionary with keys:
            - 'free': Free memory in bytes
            - 'total': Total memory in bytes
            - 'used': Used memory in bytes
            - 'percent': Memory usage percentage (0-100)

    Raises:
        RuntimeError: If CUDA is not available or device_id is invalid
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")

    if device_id >= torch.cuda.device_count():
        raise RuntimeError(f"Invalid device_id {device_id}, only {torch.cuda.device_count()} GPUs available")

    try:
        # Direct CUDA API call - returns (free, total) in bytes
        free_bytes, total_bytes = torch.cuda.mem_get_info(device_id)
        used_bytes = total_bytes - free_bytes
        percent_used = (used_bytes / total_bytes) * 100.0 if total_bytes > 0 else 0.0

        return {
            'free': free_bytes,
            'total': total_bytes,
            'used': used_bytes,
            'percent': percent_used
        }

    except Exception as e:
        logger.error(f"Error getting GPU {device_id} memory info: {e}")
        raise


class GPUMonitor:
    """
    Background GPU memory monitor for adaptive batching and memory pressure detection.

    Monitors GPU memory usage across multiple devices in a background thread,
    logging periodic statistics and detecting memory pressure conditions that
    may require batch size reduction.

    Example:
        >>> monitor = GPUMonitor(device_ids=[0, 1, 2, 3], log_interval=10.0)
        >>> monitor.start_monitoring()
        >>> # ... do work ...
        >>> if monitor.detect_memory_pressure(threshold=0.9):
        ...     print("Memory pressure detected, reduce batch size")
        >>> monitor.stop_monitoring()
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
                self.log_stats()
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
                    self._stats_cache[device_id] = get_gpu_memory_info(device_id)
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

    def detect_memory_pressure(self, threshold: float = 0.9) -> List[int]:
        """
        Detect GPUs experiencing high memory pressure.

        Args:
            threshold: Memory usage percentage threshold (0.0-1.0, default: 0.9)

        Returns:
            List of device IDs with memory usage above threshold
        """
        pressure_gpus = []
        stats = self.get_current_stats()

        for device_id, info in stats.items():
            if info['percent'] >= (threshold * 100.0):
                pressure_gpus.append(device_id)

        return pressure_gpus

    def log_stats(self):
        """Log current memory usage for all monitored GPUs."""
        stats = self.get_current_stats()

        if not stats:
            return

        # Format memory values in GB for readability
        log_lines = []
        for device_id in sorted(stats.keys()):
            info = stats[device_id]
            used_gb = info['used'] / (1024**3)
            total_gb = info['total'] / (1024**3)
            percent = info['percent']
            log_lines.append(f"GPU {device_id}: {used_gb:.2f}/{total_gb:.2f} GB ({percent:.1f}%)")

        logger.info("GPU Memory: " + " | ".join(log_lines))


def find_optimal_batch_size(
    model: torch.nn.Module,
    sample_input: torch.Tensor,
    device: torch.device,
    initial_batch: int = 1,
    max_batch: int = 512
) -> int:
    """
    Find optimal batch size using binary search to maximize GPU utilization.

    Tries progressively larger batch sizes until OOM error is encountered,
    then returns the last working batch size. Useful for finding the maximum
    batch size that fits in GPU memory.

    Args:
        model: PyTorch model to test
        sample_input: Sample input tensor (single example)
        device: Device to run test on
        initial_batch: Starting batch size (default: 1)
        max_batch: Maximum batch size to try (default: 512)

    Returns:
        Optimal batch size that fits in GPU memory

    Note:
        This function temporarily moves the model to the device and runs
        test inferences. The model is left on the device after completion.
    """
    model = model.to(device)
    model.eval()

    last_working_batch = initial_batch
    current_batch = initial_batch

    logger.info(f"Finding optimal batch size (device: {device}, max: {max_batch})")

    with torch.no_grad():
        while current_batch <= max_batch:
            try:
                # Clear cache before test
                torch.cuda.empty_cache()

                # Create batch by repeating sample input
                if sample_input.dim() == 1:
                    batch = sample_input.unsqueeze(0).repeat(current_batch, 1)
                else:
                    batch = sample_input.repeat(current_batch, *([1] * (sample_input.dim() - 1)))

                batch = batch.to(device)

                # Try forward pass
                _ = model(batch)

                # Success - this batch size works
                last_working_batch = current_batch
                logger.debug(f"Batch size {current_batch}: OK")

                # Try larger batch (double it)
                current_batch *= 2

                # Clean up
                del batch
                torch.cuda.empty_cache()

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.info(f"Batch size {current_batch}: OOM, using {last_working_batch}")
                    torch.cuda.empty_cache()
                    break
                else:
                    # Non-OOM error, re-raise
                    raise

            except Exception as e:
                logger.error(f"Error testing batch size {current_batch}: {e}")
                break

    logger.info(f"Optimal batch size: {last_working_batch}")
    return last_working_batch
