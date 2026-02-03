"""Optimized DataLoader configuration utilities for efficient data loading

This module provides utilities for creating DataLoaders with CPU-aware worker counts
and optimized settings for high-throughput processing of sequence data.

Features:
- CPU-aware worker count configuration
- Async DataLoader for GPU inference with CUDA-safe workers
- Memory estimation utilities

Based on PyTorch DataLoader best practices and patterns from:
- parallel_translate.py: Spawn context for multiprocessing
- base_worker.py: Worker assignment and resource allocation
"""

import multiprocessing
import logging
from typing import Optional, List, Dict, Any, Callable
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset
from virnucpro.data.packing import calculate_token_budget

logger = logging.getLogger('virnucpro.dataloader')


def cuda_safe_worker_init(worker_id: int) -> None:
    """
    Initialize DataLoader worker with CUDA isolation.

    This function is called in each worker process to ensure:
    1. CUDA_VISIBLE_DEVICES is empty (no GPU access)
    2. HuggingFace tokenizer parallelism disabled (prevents deadlocks)
    3. Worker is seeded for reproducibility

    Args:
        worker_id: Worker process ID (0 to num_workers-1)

    Raises:
        RuntimeError: If CUDA is accessible in worker (should never happen with spawn)
    """
    import os
    import numpy as np

    # Hide CUDA devices from worker
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

    # Disable HuggingFace tokenizer parallelism to prevent deadlocks with DataLoader
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    # Seed for reproducibility
    np.random.seed(worker_id)

    # Validate CUDA is not accessible
    # Note: We import torch here to check, but the env var should prevent CUDA init
    import torch
    if torch.cuda.is_available():
        raise RuntimeError(
            f"Worker {worker_id}: CUDA is accessible despite CUDA_VISIBLE_DEVICES=''. "
            "This indicates a multiprocessing context issue. Use 'spawn' context."
        )

    logger.debug(f"Worker {worker_id} initialized with CUDA isolation")


class SequenceDataset(Dataset):
    """Simple Dataset wrapper for sequences"""

    def __init__(self, sequences: List[Any], sort_by_length: bool = True):
        """
        Create dataset from sequences.

        Args:
            sequences: List of sequences (can be strings, tuples, or objects)
            sort_by_length: If True, sort sequences by length for memory efficiency
        """
        self.sequences = sequences

        if sort_by_length and sequences:
            # Sort by length if sequences support len()
            try:
                self.sequences = sorted(sequences, key=lambda x: len(x) if hasattr(x, '__len__') else 0)
                logger.info(f"Sorted {len(sequences)} sequences by length for memory efficiency")
            except (TypeError, AttributeError):
                logger.debug("Sequences don't support length comparison, using original order")

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Any:
        return self.sequences[idx]


def get_optimal_workers(num_gpus: int, dataloader_workers: Optional[int] = None) -> int:
    """
    Determine optimal number of DataLoader workers based on system resources.

    Worker count is calculated as min(cpu_count // num_gpus, 8) to:
    - Balance CPU resources across multiple GPU workers
    - Cap at 8 to prevent memory explosion from too many worker processes
    - Ensure at least 1 worker even with many GPUs

    Args:
        num_gpus: Number of GPUs being used for processing
        dataloader_workers: Explicit worker count (if provided, returns this value)

    Returns:
        Optimal number of DataLoader workers
    """
    if dataloader_workers is not None:
        logger.info(f"Using explicit worker count: {dataloader_workers}")
        return dataloader_workers

    cpu_count = multiprocessing.cpu_count()
    # min(cpu_count // max(num_gpus, 1), 8) ensures:
    # - Division by at least 1 (avoids division by zero)
    # - Cap at 8 workers to prevent memory issues
    optimal = min(cpu_count // max(num_gpus, 1), 8)

    # Ensure at least 1 worker if CPUs available
    optimal = max(optimal, 1) if cpu_count > 0 else 0

    logger.info(f"Auto-detected optimal DataLoader workers: {optimal} "
                f"(CPU count: {cpu_count}, GPUs: {num_gpus})")

    return optimal


def create_optimized_dataloader(
    dataset: Dataset,
    batch_size: int,
    num_gpus: int = 1,
    dataloader_workers: Optional[int] = None,
    pin_memory: Optional[bool] = None,
    shuffle: bool = True,
    **kwargs
) -> DataLoader:
    """
    Create DataLoader with optimized settings for high-throughput processing.

    Configuration follows PyTorch best practices:
    - CPU-aware worker count via get_optimal_workers()
    - pin_memory=True when CUDA available for faster GPU transfer
    - prefetch_factor=2 for good I/O-compute overlap
    - persistent_workers=True to avoid worker restart overhead
    - spawn context for safe multiprocessing (matches GPU worker pattern)

    Args:
        dataset: PyTorch Dataset to load from
        batch_size: Batch size for DataLoader
        num_gpus: Number of GPUs being used (for worker count calculation)
        dataloader_workers: Explicit worker count (None = auto-detect)
        pin_memory: Enable pinned memory (None = auto-detect from CUDA availability)
        shuffle: Whether to shuffle data
        **kwargs: Additional DataLoader arguments

    Returns:
        Configured DataLoader with optimized settings
    """
    # Auto-detect optimal worker count
    num_workers = get_optimal_workers(num_gpus, dataloader_workers)

    # Default pin_memory to CUDA availability
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    # Configure DataLoader settings
    dataloader_config = {
        'batch_size': batch_size,
        'shuffle': shuffle,
        'num_workers': num_workers,
        'pin_memory': pin_memory,
    }

    # Add prefetch and persistence for workers > 0
    if num_workers > 0:
        dataloader_config['prefetch_factor'] = 2  # Fixed good default
        dataloader_config['persistent_workers'] = True  # Keep workers alive
        dataloader_config['multiprocessing_context'] = 'spawn'  # Safe context

    # Merge with any additional kwargs
    dataloader_config.update(kwargs)

    logger.info(
        f"Creating DataLoader: batch_size={batch_size}, num_workers={num_workers}, "
        f"pin_memory={pin_memory}, prefetch_factor={dataloader_config.get('prefetch_factor', 'N/A')}"
    )

    return DataLoader(dataset, **dataloader_config)


def create_sequence_dataloader(
    sequences: List[Any],
    batch_size: int,
    sort_by_length: bool = True,
    **kwargs
) -> DataLoader:
    """
    Create optimized DataLoader for sequence data.

    Convenience wrapper that:
    1. Optionally sorts sequences by length to minimize padding
    2. Wraps sequences in SequenceDataset
    3. Creates optimized DataLoader with best practices

    Sequence sorting reduces memory fragmentation by grouping similar-length
    sequences together, minimizing padding overhead in batches.

    Args:
        sequences: List of sequences to process
        batch_size: Batch size for DataLoader
        sort_by_length: If True, sort by length for memory efficiency
        **kwargs: Additional arguments passed to create_optimized_dataloader()

    Returns:
        Configured DataLoader with sequence dataset
    """
    logger.info(f"Creating sequence DataLoader for {len(sequences)} sequences "
                f"(sorted: {sort_by_length})")

    # Create dataset with optional sorting
    dataset = SequenceDataset(sequences, sort_by_length=sort_by_length)

    # Create optimized DataLoader
    return create_optimized_dataloader(dataset, batch_size, **kwargs)


def create_async_dataloader(
    dataset: IterableDataset,
    collate_fn: Callable,
    batch_size: Optional[int] = None,
    num_workers: int = 4,
    prefetch_factor: int = 4,
    pin_memory: bool = True,
    timeout: float = 600.0,
    token_budget: Optional[int] = None,
    device_id: int = 0,
    model_memory_gb: float = 5.0,
) -> DataLoader:
    """
    Create DataLoader with CUDA-safe configuration for async GPU inference.

    This factory configures DataLoader for the async architecture where:
    - Workers perform CPU-only I/O (FASTA parsing)
    - Main process handles tokenization (via collate_fn)
    - GPU inference receives prefetched, pinned batches

    Configuration follows PyTorch best practices for GPU workloads:
    - spawn context: Prevents CUDA context inheritance (safe multiprocessing)
    - persistent_workers: Keeps workers alive (faster, no restart overhead)
    - pin_memory: Enables fast CPU-to-GPU transfer
    - prefetch_factor: Aggressive prefetching (4 batches per worker = 16 total)

    Args:
        dataset: IterableDataset (e.g., SequenceDataset) for streaming data
        collate_fn: Callable to process batches (e.g., VarlenCollator)
        batch_size: Number of items per batch. Use None for dynamic batching where
            collate_fn determines batch size (e.g., VarlenCollator with token budget).
            For VarlenCollator (token-budget packing), MUST be None - otherwise fixed
            batch_size would override the packing logic and hurt efficiency.
        num_workers: Number of CPU workers for I/O (default: 4)
        prefetch_factor: Batches to prefetch per worker (default: 4)
        pin_memory: Enable pinned memory for fast GPU transfer (default: True)
        timeout: Timeout in seconds for worker operations (default: 600 = 10 minutes,
            increased from 5 minutes to handle large FASTA parsing)
        token_budget: Optional explicit token budget. If None and CUDA available,
            calculates dynamically using calculate_token_budget (PACK-03).
            If None and CUDA unavailable, uses collator's default.
        device_id: CUDA device for dynamic budget calculation
        model_memory_gb: Estimated model memory for budget calculation

    Returns:
        Configured DataLoader ready for async GPU inference

    Example:
        >>> from virnucpro.data import SequenceDataset, VarlenCollator
        >>> dataset = SequenceDataset(fasta_files)
        >>> collator = VarlenCollator(batch_converter, max_tokens_per_batch=4096)
        >>> # batch_size=None lets VarlenCollator control batching via token budget
        >>> # token_budget=None enables dynamic budget calculation (PACK-03)
        >>> loader = create_async_dataloader(dataset, collator, batch_size=None)
        >>> for batch in loader:
        ...     # batch has pinned tensors ready for GPU transfer
        ...     gpu_batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

    Raises:
        ValueError: If num_workers < 1 (async requires workers)

    Note:
        - Workers MUST NOT touch CUDA (enforced via cuda_safe_worker_init)
        - Tokenization happens in main process via collate_fn
        - batch_size=None is recommended for VarlenCollator (token-budget packing)
        - For single-threaded loading, use create_optimized_dataloader instead
    """
    if num_workers < 1:
        raise ValueError(
            f"Async DataLoader requires num_workers >= 1, got {num_workers}. "
            "Use create_optimized_dataloader for single-threaded loading."
        )

    # Calculate dynamic token budget if not explicitly provided (PACK-03)
    if token_budget is None:
        if torch.cuda.is_available():
            token_budget = calculate_token_budget(
                device_id=device_id,
                model_memory_gb=model_memory_gb,
            )
            # Update collator's token budget
            collate_fn.max_tokens_per_batch = token_budget
            if collate_fn.packer is not None:
                collate_fn.packer.max_tokens_per_batch = token_budget
            logger.info(f"Dynamic token budget: {token_budget}")
    elif token_budget is not None:
        # Explicit budget provided - update collator
        collate_fn.max_tokens_per_batch = token_budget
        if collate_fn.packer is not None:
            collate_fn.packer.max_tokens_per_batch = token_budget

    logger.info(
        f"Creating async DataLoader: batch_size={batch_size}, num_workers={num_workers}, "
        f"prefetch_factor={prefetch_factor}, pin_memory={pin_memory}, timeout={timeout}s"
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=pin_memory,
        persistent_workers=True,
        multiprocessing_context='spawn',
        collate_fn=collate_fn,
        worker_init_fn=cuda_safe_worker_init,
        timeout=timeout,
    )


def estimate_memory_usage(
    batch_size: int,
    max_seq_length: int,
    model_size_gb: float = 3.0,
    hidden_size: int = 768,
    dtype_bytes: int = 2
) -> Dict[str, float]:
    """
    Estimate GPU memory usage for batch processing.

    Provides rough estimates to help avoid OOM errors. Actual usage
    may vary based on model architecture and PyTorch version.

    Args:
        batch_size: Batch size for processing
        max_seq_length: Maximum sequence length in batch
        model_size_gb: Model parameter size in GB (default: 3.0 for ESM-2 650M)
        hidden_size: Model hidden dimension size (default: 768)
        dtype_bytes: Bytes per element (2 for FP16/BF16, 4 for FP32)

    Returns:
        Dictionary with memory estimates in GB:
        - 'model': Model parameter memory
        - 'activations': Forward pass activation memory
        - 'total': Total estimated memory usage
    """
    # Activation memory: batch_size * seq_length * hidden_size * dtype_bytes
    # Factor of 4 accounts for attention mechanism and intermediate layers
    activation_gb = (batch_size * max_seq_length * hidden_size * dtype_bytes * 4) / (1024 ** 3)

    total_gb = model_size_gb + activation_gb

    estimates = {
        'model': model_size_gb,
        'activations': activation_gb,
        'total': total_gb,
    }

    logger.debug(
        f"Memory estimate for batch_size={batch_size}, max_seq_len={max_seq_length}: "
        f"model={model_size_gb:.2f}GB, activations={activation_gb:.2f}GB, "
        f"total={total_gb:.2f}GB"
    )

    return estimates
