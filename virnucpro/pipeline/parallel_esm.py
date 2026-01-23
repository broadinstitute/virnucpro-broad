"""ESM-2 specific multiprocessing utilities for parallel feature extraction"""

import torch
import esm
from pathlib import Path
from typing import List, Tuple
from Bio import SeqIO
import logging

logger = logging.getLogger('virnucpro.parallel_esm')

# Import at module level for worker access
from virnucpro.pipeline.features import extract_esm_features
from virnucpro.core.logging_setup import setup_worker_logging


def _get_progress_queue():
    """
    Get the progress queue from work_queue module.

    Returns None if queue not initialized (when running without progress reporting).
    """
    try:
        from virnucpro.pipeline.work_queue import _worker_progress_queue
        return _worker_progress_queue
    except ImportError:
        return None


def count_sequences(file_path: Path) -> int:
    """
    Count number of sequences in a FASTA file.

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


def assign_files_round_robin(files: List[Path], num_workers: int) -> List[List[Path]]:
    """
    Distribute files across workers using balanced bin-packing by sequence count.

    Uses greedy bin-packing algorithm to balance work by sequence count,
    not just file count, ensuring even GPU utilization.

    Args:
        files: List of file paths to process
        num_workers: Number of worker processes

    Returns:
        List of file lists, one per worker

    Example:
        >>> files = [Path('a.fa'), Path('b.fa'), Path('c.fa'), Path('d.fa'), Path('e.fa')]
        >>> assign_files_round_robin(files, 2)
        [[Path('a.fa'), Path('c.fa')], [Path('b.fa'), Path('d.fa'), Path('e.fa')]]
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

    logger.info(f"Assigned {len(files)} files to {num_workers} workers (bin-packing)")
    for worker_idx, file_list in enumerate(worker_files):
        total_seqs = worker_totals[worker_idx]
        logger.info(f"  Worker {worker_idx}: {len(file_list)} files, {total_seqs} sequences")

    return worker_files


def process_esm_files_worker(
    file_subset: List[Path],
    device_id: int,
    toks_per_batch: int = 2048,
    output_dir: Path = None,
    **kwargs
) -> Tuple[List[Path], List[Tuple[Path, str]]]:
    """
    Worker function to process ESM-2 features on a specific GPU.

    This function is called by multiprocessing Pool workers. Each worker
    loads the ESM-2 model on its assigned GPU and processes all files
    in its subset. Deferred CUDA initialization ensures no parent process
    CUDA context issues.

    Args:
        file_subset: List of protein FASTA files to process
        device_id: CUDA device ID (e.g., 0 for cuda:0)
        toks_per_batch: Tokens per batch for ESM-2 processing (default: 2048)
        output_dir: Directory where output files should be saved
        **kwargs: Additional arguments (log_level, log_format)

    Returns:
        Tuple of (processed_files, failed_files)
        - processed_files: List of successfully processed output .pt file paths
        - failed_files: List of (file_path, error_message) tuples for failures

    Raises:
        Exception: Critical errors that prevent worker startup (logged with device context)
    """
    # Initialize logging in worker process
    log_level = kwargs.get('log_level', logging.INFO)
    log_format = kwargs.get('log_format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    setup_worker_logging(log_level, log_format)

    # Get progress queue from module global (set by Pool initializer)
    progress_queue = _get_progress_queue()

    processed_files = []
    failed_files = []

    try:
        # Deferred CUDA initialization - happens only in worker process
        device = torch.device(f'cuda:{device_id}')
        logger.info(f"Worker {device_id}: Initializing on {device}, processing {len(file_subset)} files")

        # Wrap all inference in torch.no_grad() context
        with torch.no_grad():
            for file in file_subset:
                try:
                    output_file = output_dir / f"{file.stem}_ESM.pt"

                    logger.info(f"Worker {device_id}: Processing {file.name}")
                    extract_esm_features(
                        file,
                        output_file,
                        device,
                        toks_per_batch=toks_per_batch
                    )

                    processed_files.append(output_file)
                    logger.info(f"Worker {device_id}: Completed {file.name} -> {output_file.name}")

                    # Report progress if queue available
                    if progress_queue is not None:
                        progress_queue.put({
                            'gpu_id': device_id,
                            'file': str(file),
                            'status': 'complete'
                        })

                except RuntimeError as e:
                    # Handle OOM and other CUDA errors
                    error_msg = str(e)
                    if "out of memory" in error_msg.lower():
                        logger.error(f"Worker {device_id}: OOM error on {file.name}")
                        torch.cuda.empty_cache()  # Clear cache and continue
                    else:
                        logger.error(f"Worker {device_id}: CUDA error on {file.name}: {error_msg}")

                    failed_files.append((file, error_msg))

                    # Report failure if queue available
                    if progress_queue is not None:
                        progress_queue.put({
                            'gpu_id': device_id,
                            'file': str(file),
                            'status': 'failed'
                        })

                except Exception as e:
                    # Handle other errors
                    error_msg = str(e)
                    logger.exception(f"Worker {device_id}: Error processing {file.name}")
                    failed_files.append((file, error_msg))

                    # Report failure if queue available
                    if progress_queue is not None:
                        progress_queue.put({
                            'gpu_id': device_id,
                            'file': str(file),
                            'status': 'failed'
                        })

        logger.info(f"Worker {device_id}: Completed {len(processed_files)}/{len(file_subset)} files "
                   f"({len(failed_files)} failed)")

        return (processed_files, failed_files)

    except Exception as e:
        logger.exception(f"Worker {device_id}: Critical error during initialization")
        raise
