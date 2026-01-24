"""Optimized DataLoader configuration utilities for efficient data loading

This module provides utilities for creating DataLoaders with CPU-aware worker counts
and optimized settings for high-throughput processing of sequence data.

Based on PyTorch DataLoader best practices and patterns from:
- parallel_translate.py: Spawn context for multiprocessing
- base_worker.py: Worker assignment and resource allocation
"""

import multiprocessing
import logging
from typing import Optional, List, Dict, Any
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger('virnucpro.dataloader')


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
