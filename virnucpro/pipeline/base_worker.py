"""Abstract base class for embedding extraction workers"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Tuple, Optional
import torch
import logging

logger = logging.getLogger('virnucpro.base_worker')


class BaseEmbeddingWorker(ABC):
    """
    Abstract base class defining unified interface for embedding workers.

    This class provides the contract that both DNABERT-S and ESM-2 workers
    must implement, ensuring consistency in parallel processing behavior.

    Worker implementations must:
    - Support spawn context multiprocessing (no unpicklable instance state)
    - Use deferred CUDA initialization (only in worker process)
    - Report progress via multiprocessing.Queue
    - Return (processed_files, failed_files) tuple
    - Handle per-file errors gracefully
    """

    @abstractmethod
    def process_files_worker(
        self,
        file_subset: List[Path],
        device_id: int,
        batch_size: int,
        output_dir: Path,
        **kwargs
    ) -> Tuple[List[Path], List[Tuple[Path, str]]]:
        """
        Process files on specific GPU device.

        This method is called by multiprocessing Pool workers. Each worker
        loads the model on its assigned GPU and processes all files in its
        subset with automatic batching and queuing.

        Args:
            file_subset: List of input files to process
            device_id: CUDA device ID (e.g., 0 for cuda:0)
            batch_size: Batch size parameter (interpretation varies by model)
            output_dir: Directory where output files should be saved
            **kwargs: Additional arguments (log_level, log_format)

        Returns:
            Tuple of (processed_files, failed_files)
            - processed_files: List of successfully processed output .pt file paths
            - failed_files: List of (file_path, error_message) tuples for failures

        Raises:
            Exception: Critical errors that prevent worker startup
        """
        pass

    @abstractmethod
    def get_optimal_batch_size(self, device: torch.device) -> int:
        """
        Determine optimal batch size for given device.

        This method analyzes device capabilities (memory, compute capability)
        and returns recommended batch size for the specific model.

        Args:
            device: PyTorch device to analyze

        Returns:
            Recommended batch size (tokens or sequences depending on model)
        """
        pass


def count_sequences(file_path: Path) -> int:
    """
    Count number of sequences in a FASTA file.

    Shared utility used by both DNABERT-S and ESM-2 workers for
    bin-packing file assignment.

    Args:
        file_path: Path to FASTA file

    Returns:
        Number of sequences in file
    """
    count = 0
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('>'):
                count += 1
    return count


def assign_files_by_sequences(files: List[Path], num_workers: int) -> List[List[Path]]:
    """
    Distribute files across workers using balanced bin-packing by sequence count.

    Uses greedy bin-packing algorithm to balance work by sequence count,
    not just file count, ensuring even GPU utilization. This is the shared
    implementation used by both DNABERT-S and ESM-2 workers.

    Args:
        files: List of file paths to process
        num_workers: Number of worker processes

    Returns:
        List of file lists, one per worker

    Raises:
        ValueError: If num_workers is not positive

    Example:
        >>> files = [Path('a.fa'), Path('b.fa'), Path('c.fa')]
        >>> assign_files_by_sequences(files, 2)
        [[Path('a.fa'), Path('c.fa')], [Path('b.fa')]]
    """
    if num_workers <= 0:
        raise ValueError(f"num_workers must be positive, got {num_workers}")

    if not files:
        return [[] for _ in range(num_workers)]

    # Count sequences in each file
    file_sizes = []
    for file_path in files:
        seq_count = count_sequences(file_path)
        file_sizes.append((file_path, seq_count))

    # Sort by sequence count (descending) for better bin-packing
    file_sizes.sort(key=lambda x: x[1], reverse=True)

    # Initialize bins (workers) with running totals
    worker_files = [[] for _ in range(num_workers)]
    worker_totals = [0] * num_workers

    # Greedy bin-packing: assign each file to worker with lowest current total
    for file_path, seq_count in file_sizes:
        # Find worker with minimum load
        min_worker_idx = min(range(num_workers), key=lambda i: worker_totals[i])

        # Assign file to that worker
        worker_files[min_worker_idx].append(file_path)
        worker_totals[min_worker_idx] += seq_count

    logger.info(f"Assigned {len(files)} files to {num_workers} workers (bin-packing by sequences)")
    for worker_idx, file_list in enumerate(worker_files):
        total_seqs = worker_totals[worker_idx]
        logger.info(f"  Worker {worker_idx}: {len(file_list)} files, {total_seqs} sequences")

    return worker_files


def detect_bf16_support(device: torch.device) -> bool:
    """
    Detect if device supports BF16 mixed precision.

    BF16 (Brain Float 16) is supported on Ampere GPUs and newer
    (compute capability >= 8.0). This provides 50% memory savings
    with minimal accuracy impact.

    Args:
        device: PyTorch device to check

    Returns:
        True if BF16 is supported, False otherwise
    """
    if not str(device).startswith('cuda'):
        return False

    capability = torch.cuda.get_device_capability(device)
    use_bf16 = capability[0] >= 8  # Ampere or newer

    if use_bf16:
        logger.info(f"Device {device} supports BF16 (compute capability {capability[0]}.{capability[1]})")
    else:
        logger.info(f"Device {device} does not support BF16 (compute capability {capability[0]}.{capability[1]})")

    return use_bf16
