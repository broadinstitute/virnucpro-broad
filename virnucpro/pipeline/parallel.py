"""Multiprocessing utilities for parallel feature extraction"""

import torch
from pathlib import Path
from typing import List
import logging

logger = logging.getLogger('virnucpro.parallel')


def detect_cuda_devices() -> List[int]:
    """
    Detect available CUDA devices.

    Returns:
        List of CUDA device indices (e.g., [0, 1, 2, 3] for 4 GPUs)
        Empty list if no CUDA available
    """
    if not torch.cuda.is_available():
        logger.warning("CUDA not available, parallel processing disabled")
        return []

    num_devices = torch.cuda.device_count()
    logger.info(f"Detected {num_devices} CUDA device(s)")

    return list(range(num_devices))


def assign_files_round_robin(files: List[Path], num_workers: int) -> List[List[Path]]:
    """
    Assign files to workers in round-robin fashion.

    Args:
        files: List of file paths to process
        num_workers: Number of worker processes

    Returns:
        List of file lists, one per worker

    Example:
        >>> files = [Path(f"file_{i}.fa") for i in range(8)]
        >>> assign_files_round_robin(files, 4)
        [[file_0.fa, file_4.fa], [file_1.fa, file_5.fa], ...]
    """
    if num_workers <= 0:
        raise ValueError(f"num_workers must be positive, got {num_workers}")

    if not files:
        return [[] for _ in range(num_workers)]

    worker_files = [[] for _ in range(num_workers)]

    for idx, file_path in enumerate(files):
        worker_idx = idx % num_workers
        worker_files[worker_idx].append(file_path)

    logger.debug(f"Assigned {len(files)} files to {num_workers} workers")
    for worker_idx, file_list in enumerate(worker_files):
        logger.debug(f"  Worker {worker_idx}: {len(file_list)} files")

    return worker_files


def process_dnabert_files_worker(
    file_subset: List[Path],
    device_id: int,
    batch_size: int,
    output_dir: Path
) -> List[Path]:
    """
    Worker function to process DNABERT-S features on a specific GPU.

    This function is called by multiprocessing Pool workers. Each worker
    loads the DNABERT-S model on its assigned GPU and processes all
    files in its subset.

    Args:
        file_subset: List of nucleotide FASTA files to process
        device_id: CUDA device ID (e.g., 0 for cuda:0)
        batch_size: Batch size for DNABERT-S processing
        output_dir: Directory where output files should be saved

    Returns:
        List of output .pt file paths

    Raises:
        Exception: Any error during processing (logged with device context)
    """
    output_files = []

    try:
        device = torch.device(f'cuda:{device_id}')
        logger.info(f"Worker {device_id}: Processing {len(file_subset)} files on {device}")

        from virnucpro.pipeline.features import extract_dnabert_features

        for nuc_file in file_subset:
            output_file = output_dir / f"{nuc_file.stem}_DNABERT_S.pt"

            extract_dnabert_features(
                nuc_file,
                output_file,
                device,
                batch_size=batch_size
            )
            output_files.append(output_file)

        logger.info(f"Worker {device_id}: Completed {len(output_files)} files")
        return output_files

    except Exception as e:
        logger.exception(f"Worker {device_id}: Error processing files")
        raise
