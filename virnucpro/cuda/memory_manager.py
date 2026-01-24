"""Memory management system for fragmentation prevention and OOM avoidance

This module provides utilities for managing CUDA memory allocation, preventing
fragmentation, and avoiding out-of-memory errors during long-running processing.

Based on PyTorch memory management best practices:
- Expandable segments for dynamic allocation patterns
- Periodic cache clearing to reclaim fragmented memory
- Sequence sorting to minimize padding and fragmentation
- Memory stats tracking for diagnostics

Reference patterns from:
- gpu_monitor.py: Memory tracking and logging
- parallel.py: Batch processing with memory management
"""

import os
import gc
import logging
from typing import Dict, List, Any, Optional
from contextlib import contextmanager

import torch

logger = logging.getLogger('virnucpro.memory')


class MemoryManager:
    """
    Memory management system for fragmentation prevention.

    Provides utilities for:
    - Configuring expandable segments
    - Periodic cache clearing
    - Memory usage tracking and diagnostics
    - OOM prevention calculations
    - Sequence sorting for efficiency
    """

    def __init__(
        self,
        enable_expandable_segments: bool = False,
        cache_clear_interval: int = 100,
        verbose: bool = False
    ):
        """
        Initialize memory manager.

        Args:
            enable_expandable_segments: If True, configure expandable segments
            cache_clear_interval: Clear cache every N batches (0 = never)
            verbose: Enable verbose logging of memory operations
        """
        self.cache_clear_interval = cache_clear_interval
        self.verbose = verbose
        self.batch_counter = 0

        if enable_expandable_segments:
            self.configure_expandable_segments()

        logger.info(
            f"MemoryManager initialized: expandable_segments={enable_expandable_segments}, "
            f"cache_interval={cache_clear_interval}"
        )

    def configure_expandable_segments(self) -> None:
        """
        Configure PyTorch to use expandable memory segments.

        Expandable segments reduce fragmentation by allowing memory blocks
        to grow dynamically instead of allocating fixed-size segments.

        Sets PYTORCH_CUDA_ALLOC_CONF environment variable.
        """
        # Get existing config or start fresh
        existing_config = os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '')

        if 'expandable_segments' not in existing_config:
            if existing_config:
                new_config = f"{existing_config},expandable_segments:True"
            else:
                new_config = "expandable_segments:True"

            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = new_config
            logger.info("Configured expandable segments: PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'")
        else:
            logger.debug("Expandable segments already configured")

    def get_memory_stats(self, device: int = 0) -> Dict[str, float]:
        """
        Get current memory statistics for a device.

        Args:
            device: CUDA device ID

        Returns:
            Dictionary with memory stats in GB:
            - 'allocated': Currently allocated memory
            - 'reserved': Reserved by allocator (includes fragmentation)
            - 'free': Free memory on device
        """
        if not torch.cuda.is_available():
            return {'allocated': 0.0, 'reserved': 0.0, 'free': 0.0}

        device_obj = torch.device(f'cuda:{device}')

        # Get memory info
        allocated = torch.cuda.memory_allocated(device_obj) / (1024 ** 3)
        reserved = torch.cuda.memory_reserved(device_obj) / (1024 ** 3)

        # Get total memory
        total = torch.cuda.get_device_properties(device_obj).total_memory / (1024 ** 3)
        free = total - reserved

        return {
            'allocated': allocated,
            'reserved': reserved,
            'free': free,
        }

    def log_memory_status(self, prefix: str = "", device: int = 0) -> None:
        """
        Log current memory usage.

        Args:
            prefix: Prefix string for log message
            device: CUDA device ID
        """
        if not torch.cuda.is_available():
            logger.debug(f"{prefix}CUDA not available")
            return

        stats = self.get_memory_stats(device)

        log_msg = (
            f"{prefix}GPU {device} Memory - "
            f"Allocated: {stats['allocated']:.2f}GB, "
            f"Reserved: {stats['reserved']:.2f}GB, "
            f"Free: {stats['free']:.2f}GB"
        )

        if self.verbose:
            logger.info(log_msg)
        else:
            logger.debug(log_msg)

    def clear_cache(self, device: Optional[int] = None) -> None:
        """
        Clear CUDA memory cache.

        Calls torch.cuda.empty_cache() to release fragmented memory blocks
        back to the system.

        Args:
            device: Specific device to clear (None = all devices)
        """
        if not torch.cuda.is_available():
            return

        # Collect Python garbage first
        gc.collect()

        # Clear CUDA cache
        torch.cuda.empty_cache()

        if self.verbose:
            if device is not None:
                self.log_memory_status(prefix="After cache clear - ", device=device)
            else:
                logger.info("Cleared CUDA cache on all devices")

    def should_clear_cache(self, batch_num: int) -> bool:
        """
        Check if cache should be cleared based on interval.

        Args:
            batch_num: Current batch number

        Returns:
            True if cache should be cleared
        """
        if self.cache_clear_interval <= 0:
            return False

        return batch_num % self.cache_clear_interval == 0

    def increment_and_clear(self, device: Optional[int] = None) -> bool:
        """
        Increment batch counter and clear cache if interval reached.

        Convenience method for batch processing loops.

        Args:
            device: Device to clear (None = all devices)

        Returns:
            True if cache was cleared
        """
        self.batch_counter += 1

        if self.should_clear_cache(self.batch_counter):
            if self.verbose:
                logger.info(f"Clearing cache at batch {self.batch_counter} "
                           f"(interval: {self.cache_clear_interval})")
            self.clear_cache(device)
            return True

        return False

    def sort_sequences_by_length(self, sequences: List[Any]) -> List[Any]:
        """
        Sort sequences by length to minimize padding and fragmentation.

        Grouping similar-length sequences reduces wasted memory from
        padding and improves cache efficiency.

        Args:
            sequences: List of sequences (must support len())

        Returns:
            Sorted list of sequences
        """
        try:
            sorted_seqs = sorted(sequences, key=lambda x: len(x) if hasattr(x, '__len__') else 0)
            logger.debug(f"Sorted {len(sequences)} sequences by length")
            return sorted_seqs
        except (TypeError, AttributeError) as e:
            logger.warning(f"Unable to sort sequences by length: {e}")
            return sequences

    def check_memory_available(self, required_gb: float, device: int = 0) -> bool:
        """
        Check if sufficient memory is available.

        Args:
            required_gb: Required memory in GB
            device: CUDA device ID

        Returns:
            True if enough memory available
        """
        if not torch.cuda.is_available():
            return False

        stats = self.get_memory_stats(device)
        available = stats['free']

        if available >= required_gb:
            logger.debug(f"Memory check passed: {available:.2f}GB >= {required_gb:.2f}GB")
            return True
        else:
            logger.warning(
                f"Insufficient memory: {available:.2f}GB < {required_gb:.2f}GB required"
            )
            return False

    def get_safe_batch_size(
        self,
        model_size_gb: float,
        max_seq_length: int,
        available_memory_gb: Optional[float] = None,
        device: int = 0,
        safety_factor: float = 0.8
    ) -> int:
        """
        Calculate safe batch size based on available memory.

        Uses heuristic: batch_size = (available - model) / (seq_length * hidden * bytes * factor)

        Args:
            model_size_gb: Model parameter size in GB
            max_seq_length: Maximum sequence length
            available_memory_gb: Available memory (None = query device)
            device: CUDA device ID
            safety_factor: Fraction of memory to use (0.8 = 80%)

        Returns:
            Recommended batch size (minimum 1)
        """
        # Get available memory
        if available_memory_gb is None:
            if not torch.cuda.is_available():
                return 1
            stats = self.get_memory_stats(device)
            available_memory_gb = stats['free']

        # Calculate memory for activations
        # Estimate: seq_length * 768 (hidden) * 2 (bytes for FP16) * 4 (factor for attention)
        activation_per_sample = (max_seq_length * 768 * 2 * 4) / (1024 ** 3)

        # Available memory for batch (after model and safety factor)
        batch_memory = (available_memory_gb - model_size_gb) * safety_factor

        if batch_memory <= 0:
            logger.warning(
                f"Insufficient memory for model: {available_memory_gb:.2f}GB available, "
                f"{model_size_gb:.2f}GB required"
            )
            return 1

        # Calculate batch size
        batch_size = max(1, int(batch_memory / activation_per_sample))

        logger.debug(
            f"Safe batch size calculation: available={available_memory_gb:.2f}GB, "
            f"model={model_size_gb:.2f}GB, seq_len={max_seq_length}, "
            f"batch_size={batch_size}"
        )

        return batch_size

    def get_fragmentation_ratio(self, device: int = 0) -> float:
        """
        Estimate memory fragmentation ratio.

        Fragmentation ratio = (reserved - allocated) / reserved
        Higher ratio indicates more fragmentation.

        Args:
            device: CUDA device ID

        Returns:
            Fragmentation ratio (0.0 to 1.0)
        """
        if not torch.cuda.is_available():
            return 0.0

        stats = self.get_memory_stats(device)

        if stats['reserved'] == 0:
            return 0.0

        fragmentation = (stats['reserved'] - stats['allocated']) / stats['reserved']

        if self.verbose and fragmentation > 0.3:
            logger.warning(
                f"High memory fragmentation detected: {fragmentation:.1%} "
                f"(reserved: {stats['reserved']:.2f}GB, allocated: {stats['allocated']:.2f}GB)"
            )

        return fragmentation

    def suggest_batch_size_adjustment(
        self,
        current_batch_size: int,
        fragmentation_ratio: Optional[float] = None,
        device: int = 0
    ) -> Dict[str, Any]:
        """
        Suggest batch size adjustment based on fragmentation.

        Args:
            current_batch_size: Current batch size
            fragmentation_ratio: Current fragmentation (None = calculate)
            device: CUDA device ID

        Returns:
            Dictionary with:
            - 'suggested_batch_size': Recommended batch size
            - 'reason': Explanation for suggestion
            - 'fragmentation': Current fragmentation ratio
        """
        if fragmentation_ratio is None:
            fragmentation_ratio = self.get_fragmentation_ratio(device)

        # High fragmentation (>30%) suggests reducing batch size
        if fragmentation_ratio > 0.3:
            suggested = max(1, int(current_batch_size * 0.8))
            reason = f"High fragmentation ({fragmentation_ratio:.1%}), reduce batch size by 20%"
        # Low fragmentation (<10%) and free memory suggests increasing
        elif fragmentation_ratio < 0.1:
            stats = self.get_memory_stats(device)
            if stats['free'] > 2.0:  # >2GB free
                suggested = int(current_batch_size * 1.2)
                reason = f"Low fragmentation ({fragmentation_ratio:.1%}) and {stats['free']:.1f}GB free, increase batch size by 20%"
            else:
                suggested = current_batch_size
                reason = "Batch size appears optimal"
        else:
            suggested = current_batch_size
            reason = "Batch size appears optimal"

        return {
            'suggested_batch_size': suggested,
            'reason': reason,
            'fragmentation': fragmentation_ratio,
        }

    @contextmanager
    def memory_tracking(self, operation: str = "operation", device: int = 0):
        """
        Context manager for tracking memory usage of an operation.

        Usage:
            with memory_manager.memory_tracking("model forward pass"):
                output = model(input)

        Args:
            operation: Name of operation being tracked
            device: CUDA device ID

        Yields:
            None
        """
        if not torch.cuda.is_available():
            yield
            return

        # Log memory before
        before = self.get_memory_stats(device)
        logger.debug(
            f"Before {operation}: Allocated={before['allocated']:.2f}GB, "
            f"Reserved={before['reserved']:.2f}GB"
        )

        try:
            yield
        finally:
            # Log memory after
            after = self.get_memory_stats(device)
            delta_allocated = after['allocated'] - before['allocated']
            delta_reserved = after['reserved'] - before['reserved']

            logger.info(
                f"After {operation}: Allocated={after['allocated']:.2f}GB (+{delta_allocated:.2f}GB), "
                f"Reserved={after['reserved']:.2f}GB (+{delta_reserved:.2f}GB)"
            )


def configure_memory_optimization(
    enable_expandable: bool = False,
    cache_interval: int = 100,
    verbose: bool = False
) -> MemoryManager:
    """
    Global configuration function for memory optimization.

    Convenience function to configure environment variables and
    create MemoryManager instance.

    Args:
        enable_expandable: Enable expandable memory segments
        cache_interval: Clear cache every N batches (0 = never)
        verbose: Enable verbose logging

    Returns:
        Configured MemoryManager instance
    """
    logger.info(
        f"Configuring memory optimization: expandable={enable_expandable}, "
        f"interval={cache_interval}, verbose={verbose}"
    )

    return MemoryManager(
        enable_expandable_segments=enable_expandable,
        cache_clear_interval=cache_interval,
        verbose=verbose
    )
