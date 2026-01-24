"""ESM-2 specific multiprocessing utilities for parallel feature extraction"""

import os
import torch
import esm
from pathlib import Path
from typing import List, Tuple
from Bio import SeqIO
import logging

logger = logging.getLogger('virnucpro.parallel_esm')

# Module-level globals for persistent worker model storage
_esm_model = None
_batch_converter = None
_device = None

# Import base worker utilities for shared functionality
from virnucpro.pipeline.base_worker import (
    BaseEmbeddingWorker,
    count_sequences as base_count_sequences,
    assign_files_by_sequences
)
from virnucpro.pipeline.features import extract_esm_features
from virnucpro.core.logging_setup import setup_worker_logging
from virnucpro.cuda import StreamProcessor


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

    This is a wrapper around the base_worker implementation for backward compatibility.

    Args:
        file_path: Path to FASTA file

    Returns:
        Number of sequences in file
    """
    return base_count_sequences(file_path)


def assign_files_round_robin(files: List[Path], num_workers: int) -> List[List[Path]]:
    """
    Distribute files across workers using balanced bin-packing by sequence count.

    This is a wrapper around assign_files_by_sequences from base_worker for backward compatibility.
    The name "round_robin" is a misnomer - this actually uses greedy bin-packing.

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
    return assign_files_by_sequences(files, num_workers)


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

    Supports optional CUDA stream-based processing for I/O-compute overlap
    via enable_streams kwarg (default: False for backward compatibility).

    Args:
        file_subset: List of protein FASTA files to process
        device_id: CUDA device ID (e.g., 0 for cuda:0)
        toks_per_batch: Tokens per batch for ESM-2 processing (default: 2048)
        output_dir: Directory where output files should be saved
        **kwargs: Additional arguments (log_level, log_format, enable_streams)

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

    # Check if streams are enabled
    enable_streams = kwargs.get('enable_streams', False)

    processed_files = []
    failed_files = []

    try:
        # Deferred CUDA initialization - happens only in worker process
        device = torch.device(f'cuda:{device_id}')
        logger.info(f"Worker {device_id}: Initializing on {device}, processing {len(file_subset)} files")

        # Initialize stream processor if enabled
        stream_processor = None
        if enable_streams:
            stream_processor = StreamProcessor(device=device, enable_streams=True)
            logger.info(f"Worker {device_id}: Stream-based processing enabled")

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
                        toks_per_batch=toks_per_batch,
                        stream_processor=stream_processor
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


# ============================================================================
# Persistent Worker Functions (for long-lived worker pools)
# ============================================================================


def init_esm_worker(
    device_id: int,
    model_name: str = "esm2_t36_3B_UR50D",
    log_level: int = logging.INFO
) -> None:
    """
    Initialize persistent ESM-2 worker with pre-loaded model.

    This function is called once during worker pool initialization to load
    the ESM-2 model into GPU memory. The model remains loaded for the worker's
    lifetime, eliminating repeated loading overhead.

    Configures CUDA memory management (expandable segments) before any CUDA
    operations to prevent fragmentation.

    Args:
        device_id: CUDA device ID (e.g., 0 for cuda:0)
        model_name: ESM-2 model variant to load (default: esm2_t36_3B_UR50D)
        log_level: Logging level for worker process (default: INFO)

    Module Globals:
        Sets _esm_model, _batch_converter, _device for worker lifetime

    Example:
        >>> # Called by Pool initializer
        >>> from multiprocessing import Pool
        >>> pool = Pool(processes=2, initializer=init_esm_worker, initargs=(0,))
    """
    global _esm_model, _batch_converter, _device

    # Configure CUDA memory BEFORE any CUDA operations
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    # Set up logging in worker process
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    setup_worker_logging(log_level, log_format)

    # Initialize CUDA context
    _device = torch.device(f'cuda:{device_id}')
    logger.info(f"Worker {device_id}: Initializing persistent ESM-2 worker on {_device}")

    # Load ESM-2 model using load_esm2_model from virnucpro.models.esm2_flash
    from virnucpro.models.esm2_flash import load_esm2_model

    _esm_model, _batch_converter = load_esm2_model(
        model_name=model_name,
        device=str(_device),
        logger_instance=logger
    )

    logger.info(f"Worker {device_id}: ESM-2 model loaded successfully - ready for processing")


def process_esm_files_persistent(
    file_subset: List[Path],
    device_id: int,
    toks_per_batch: int = 2048,
    output_dir: Path = None,
    **kwargs
) -> Tuple[List[Path], List[Tuple[Path, str]]]:
    """
    Process ESM-2 features using pre-loaded model from persistent worker.

    This function processes files using the model loaded during worker initialization
    (via init_esm_worker). No model loading overhead - uses cached model from globals.

    Implements periodic cache clearing every 10 files to prevent memory fragmentation
    in long-running workers.

    Args:
        file_subset: List of protein FASTA files to process
        device_id: CUDA device ID (for logging, model already loaded)
        toks_per_batch: Tokens per batch for ESM-2 processing (default: 2048)
        output_dir: Directory where output files should be saved
        **kwargs: Additional arguments (enable_streams)

    Returns:
        Tuple of (processed_files, failed_files)
        - processed_files: List of successfully processed output .pt file paths
        - failed_files: List of (file_path, error_message) tuples for failures

    Raises:
        RuntimeError: If called before init_esm_worker (model not loaded)

    Note:
        Uses module-level globals (_esm_model, _batch_converter, _device) set by
        init_esm_worker during pool initialization.

    Example:
        >>> # Called by pool.map or pool.imap
        >>> pool.map(process_esm_files_persistent, file_batches)
    """
    global _esm_model, _batch_converter, _device

    # Verify model was loaded by init_esm_worker
    if _esm_model is None or _batch_converter is None or _device is None:
        raise RuntimeError(
            "Persistent worker not initialized. "
            "Call init_esm_worker during pool initialization."
        )

    # Get progress queue from module global (set by Pool initializer)
    progress_queue = _get_progress_queue()

    # Check if streams are enabled
    enable_streams = kwargs.get('enable_streams', False)

    processed_files = []
    failed_files = []

    try:
        logger.info(f"Worker {device_id}: Processing {len(file_subset)} files with pre-loaded model")

        # Initialize stream processor if enabled
        stream_processor = None
        if enable_streams:
            stream_processor = StreamProcessor(device=_device, enable_streams=True)
            logger.info(f"Worker {device_id}: Stream-based processing enabled")

        # Wrap all inference in torch.no_grad() context
        with torch.no_grad():
            for idx, file in enumerate(file_subset):
                try:
                    output_file = output_dir / f"{file.stem}_ESM.pt"

                    logger.info(f"Worker {device_id}: Processing {file.name}")

                    # Process using pre-loaded model
                    extract_esm_features(
                        file,
                        output_file,
                        _device,
                        toks_per_batch=toks_per_batch,
                        stream_processor=stream_processor
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

                    # Periodic cache clearing to prevent fragmentation (every 10 files)
                    if (idx + 1) % 10 == 0:
                        torch.cuda.empty_cache()
                        logger.debug(f"Worker {device_id}: Cleared CUDA cache after {idx + 1} files")

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
        logger.exception(f"Worker {device_id}: Critical error during processing")
        raise
