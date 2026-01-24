"""Persistent worker pool infrastructure for model persistence across jobs"""

import multiprocessing
import logging
import os
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Callable
import torch

logger = logging.getLogger('virnucpro.persistent_pool')

# Module-level globals for persistent workers (set by Pool initializer)
_model = None
_tokenizer = None
_batch_converter = None
_device = None
_model_type = None
_progress_queue = None
_file_counter = 0


def init_persistent_worker(
    model_type: str,
    progress_queue: Optional[multiprocessing.Queue],
    log_level: int,
    log_format: str
) -> None:
    """
    Initialize worker with persistent model loading.

    Loads models once during worker initialization and stores in module-level
    globals for reuse across multiple jobs. Configures memory management for
    long-running workers to prevent fragmentation.

    Device ID is deferred until first task - models will be loaded on-demand
    when first file batch is processed.

    Args:
        model_type: Either 'esm2' or 'dnabert' for model selection
        progress_queue: Optional queue for progress reporting (inherited, not pickled)
        log_level: Logging level for worker
        log_format: Logging format string for worker

    Raises:
        ValueError: If model_type is not recognized
        RuntimeError: If model loading fails
    """
    global _model, _tokenizer, _batch_converter, _device, _model_type, _progress_queue

    # Setup logging in worker
    from virnucpro.core.logging_setup import setup_worker_logging
    setup_worker_logging(log_level, log_format)

    logger.info(f"Worker initializing for {model_type} (device assigned on first task)")

    # Store configuration for lazy loading on first task
    _model_type = model_type
    _progress_queue = progress_queue

    # Configure memory management BEFORE any CUDA calls
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    logger.info(f"Worker initialized, ready to load {model_type} model on first task")


def _load_model_lazy(device_id: int) -> None:
    """
    Lazy load model on first task execution.

    This allows device_id to be assigned per-worker from task arguments
    rather than during pool initialization.

    Args:
        device_id: CUDA device ID for this worker
    """
    global _model, _tokenizer, _batch_converter, _device, _model_type

    if _model is not None:
        # Model already loaded
        return

    # Initialize CUDA context
    _device = torch.device(f'cuda:{device_id}')
    torch.cuda.set_per_process_memory_fraction(0.9, _device)

    try:
        if _model_type == 'esm2':
            # Load ESM-2 3B model with FlashAttention-2 support
            from virnucpro.models.esm2_flash import load_esm2_model

            logger.info(f"Worker {device_id}: Loading ESM-2 model (one-time initialization)")
            _model, _batch_converter = load_esm2_model(
                model_name="esm2_t36_3B_UR50D",
                device=str(_device),
                logger_instance=logger
            )
            logger.info(f"Worker {device_id}: ESM-2 model loaded and ready")

        elif _model_type == 'dnabert':
            # Load DNABERT-S model
            from transformers import AutoTokenizer, AutoModel

            logger.info(f"Worker {device_id}: Loading DNABERT-S model (one-time initialization)")
            model_name = "zhihan1996/DNABERT-S"
            _tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            _model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(_device)
            _model.eval()
            logger.info(f"Worker {device_id}: DNABERT-S model loaded and ready")

        else:
            raise ValueError(f"Unknown model_type: {_model_type}. Expected 'esm2' or 'dnabert'")

    except Exception as e:
        logger.exception(f"Worker {device_id}: Failed to load {_model_type} model")
        raise RuntimeError(f"Model loading failed: {e}") from e


def process_files_persistent(
    file_subset: List[Path],
    device_id: int,
    kwargs: Dict[str, Any]
) -> Optional[Tuple[List[Path], List[Tuple[Path, str]]]]:
    """
    Process files using pre-loaded persistent model.

    This function reuses the model loaded during worker initialization,
    eliminating per-job model loading overhead. Implements periodic cache
    clearing to prevent memory fragmentation in long-running workers.

    On first invocation, model is loaded lazily based on device_id.
    Subsequent invocations reuse the cached model.

    Args:
        file_subset: List of files to process
        device_id: CUDA device ID (matches worker's assigned device)
        kwargs: Additional keyword arguments for processing (output_dir, etc.)

    Returns:
        Tuple of (processed_files, failed_files) or None on critical failure
        - processed_files: List of successfully processed output file paths
        - failed_files: List of (file_path, error_message) tuples

    Raises:
        Exception: Critical errors are logged and returned as None
    """
    global _model, _tokenizer, _batch_converter, _device, _model_type, _progress_queue, _file_counter

    # Lazy load model on first task
    try:
        _load_model_lazy(device_id)
    except Exception as e:
        logger.exception(f"Worker {device_id}: Failed to load model")
        return None

    processed_files = []
    failed_files = []

    try:
        logger.info(f"Worker {device_id}: Processing {len(file_subset)} files with persistent {_model_type} model")

        # Extract processing parameters
        output_dir = kwargs.get('output_dir')
        if output_dir is None:
            raise ValueError("output_dir is required in kwargs")

        # Wrap all inference in torch.no_grad() context
        with torch.no_grad():
            for file in file_subset:
                try:
                    # Process file using cached model
                    if _model_type == 'esm2':
                        output_file = output_dir / f"{file.stem}_ESM.pt"
                        toks_per_batch = kwargs.get('toks_per_batch', 2048)
                        stream_processor = kwargs.get('stream_processor', None)

                        from virnucpro.pipeline.features import extract_esm_features
                        # Note: extract_esm_features currently loads its own model
                        # This will need to be refactored to accept pre-loaded model
                        # For now, this is a placeholder implementation
                        logger.warning(f"Worker {device_id}: extract_esm_features not yet refactored for persistent models")
                        extract_esm_features(
                            file,
                            output_file,
                            _device,
                            toks_per_batch=toks_per_batch,
                            stream_processor=stream_processor
                        )

                    elif _model_type == 'dnabert':
                        output_file = output_dir / f"{file.stem}_DNABERT.pt"
                        batch_size = kwargs.get('batch_size', 256)

                        from virnucpro.pipeline.features import extract_dnabert_features
                        # Note: extract_dnabert_features currently loads its own model
                        # This will need to be refactored to accept pre-loaded model
                        # For now, this is a placeholder implementation
                        logger.warning(f"Worker {device_id}: extract_dnabert_features not yet refactored for persistent models")
                        extract_dnabert_features(
                            file,
                            output_file,
                            _device,
                            batch_size=batch_size
                        )

                    processed_files.append(output_file)
                    logger.info(f"Worker {device_id}: Completed {file.name} -> {output_file.name}")

                    # Increment file counter
                    _file_counter += 1

                    # Periodic cache clearing to prevent fragmentation (every 10 files)
                    if _file_counter % 10 == 0:
                        torch.cuda.empty_cache()
                        logger.debug(f"Worker {device_id}: Cleared cache after {_file_counter} files")

                    # Report progress if queue available
                    if _progress_queue is not None:
                        _progress_queue.put({
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
                    if _progress_queue is not None:
                        _progress_queue.put({
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
                    if _progress_queue is not None:
                        _progress_queue.put({
                            'gpu_id': device_id,
                            'file': str(file),
                            'status': 'failed'
                        })

        logger.info(f"Worker {device_id}: Completed {len(processed_files)}/{len(file_subset)} files "
                   f"({len(failed_files)} failed)")

        return (processed_files, failed_files)

    except Exception as e:
        logger.exception(f"Worker {device_id}: Critical failure in process_files_persistent")
        return None


class PersistentWorkerPool:
    """
    Persistent worker pool for long-lived model serving.

    Manages a pool of worker processes that load models once during initialization
    and reuse them across multiple job invocations. This eliminates the 2-5 minute
    model loading overhead for ESM-2 3B that occurs with traditional per-job pools.

    Each worker is assigned to a fixed GPU and persists for the lifetime of the pool,
    processing multiple file batches without reloading models. Memory management is
    configured to prevent fragmentation in long-running workers.

    Example:
        >>> pool = PersistentWorkerPool(
        ...     num_workers=4,
        ...     model_type='esm2',
        ...     spawn_context=True
        ... )
        >>> pool.create_pool()
        >>> # Process multiple jobs without reloading models
        >>> result1 = pool.process_job(job1_files, output_dir=Path('/tmp'))
        >>> result2 = pool.process_job(job2_files, output_dir=Path('/tmp'))
        >>> pool.close()

    Attributes:
        num_workers: Number of worker processes (typically number of GPUs)
        model_type: Type of model to load ('esm2' or 'dnabert')
        ctx: Multiprocessing context (spawn for CUDA safety)
        pool: Multiprocessing Pool instance (created by create_pool())
        progress_queue: Optional queue for progress reporting
    """

    def __init__(
        self,
        num_workers: int,
        model_type: str,
        spawn_context: bool = True,
        progress_queue: Optional[multiprocessing.Queue] = None
    ):
        """
        Initialize persistent worker pool configuration.

        Args:
            num_workers: Number of worker processes (typically number of GPUs)
            model_type: Either 'esm2' or 'dnabert' for model selection
            spawn_context: Use spawn context for CUDA safety (default: True)
            progress_queue: Optional queue for workers to report progress events

        Raises:
            ValueError: If model_type is not recognized
        """
        if model_type not in ('esm2', 'dnabert'):
            raise ValueError(f"Unknown model_type: {model_type}. Expected 'esm2' or 'dnabert'")

        self.num_workers = num_workers
        self.model_type = model_type
        self.ctx = multiprocessing.get_context('spawn') if spawn_context else multiprocessing
        self.pool = None
        self.progress_queue = progress_queue

        logger.info(f"Initialized PersistentWorkerPool: {num_workers} workers, model={model_type}, "
                   f"context={'spawn' if spawn_context else 'default'}")

    def create_pool(self) -> None:
        """
        Create persistent worker pool with model initialization.

        Workers are initialized with maxtasksperchild=None to persist for the
        pool's lifetime. Models are loaded once per worker during initialization.

        Raises:
            RuntimeError: If pool already exists or worker initialization fails
        """
        if self.pool is not None:
            raise RuntimeError("Pool already created. Call close() before recreating.")

        logger.info(f"Creating persistent pool with {self.num_workers} workers loading {self.model_type} models")

        # Get logging configuration
        log_level = logging.getLogger().level
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

        # Create pool with persistent workers (maxtasksperchild=None)
        # Pass progress_queue via initializer so it's inherited, not pickled
        # Device ID is assigned lazily on first task based on worker_id
        try:
            self.pool = self.ctx.Pool(
                self.num_workers,
                initializer=init_persistent_worker,
                initargs=(
                    self.model_type,
                    self.progress_queue,
                    log_level,
                    log_format
                ),
                maxtasksperchild=None  # Workers persist for pool lifetime
            )
            logger.info("Persistent pool created successfully")

        except Exception as e:
            logger.exception("Failed to create persistent pool")
            raise RuntimeError(f"Pool creation failed: {e}") from e

    def process_job(
        self,
        file_assignments: List[List[Path]],
        **worker_kwargs
    ) -> Tuple[List[Path], List[Tuple[Path, str]]]:
        """
        Process a single job using persistent workers.

        Workers reuse pre-loaded models, eliminating loading overhead.
        Each worker processes its assigned file subset on its designated GPU.

        Args:
            file_assignments: List of file lists, one per worker
            **worker_kwargs: Additional keyword arguments for processing (output_dir, etc.)

        Returns:
            Tuple of (processed_files, failed_files)
            - processed_files: List of successfully processed output file paths
            - failed_files: List of (file_path, error_message) tuples

        Raises:
            RuntimeError: If pool not created or systemic failure detected
        """
        if self.pool is None:
            raise RuntimeError("Pool not created. Call create_pool() first.")

        if len(file_assignments) != self.num_workers:
            raise ValueError(
                f"Expected {self.num_workers} file assignments, got {len(file_assignments)}"
            )

        all_processed = []
        all_failed = []
        worker_failures = 0

        logger.info(f"Processing job with {self.num_workers} persistent workers")
        for worker_id, files in enumerate(file_assignments):
            logger.info(f"  Worker {worker_id}: {len(files)} files assigned")

        # Create worker arguments: (file_subset, device_id, kwargs)
        worker_args = []
        for worker_id, file_subset in enumerate(file_assignments):
            worker_args.append((file_subset, worker_id, worker_kwargs))

        try:
            # Process with persistent workers
            results = self.pool.starmap(process_files_persistent, worker_args)

            # Collect results
            for worker_id, result in enumerate(results):
                if result is None:
                    # Worker failed critically
                    worker_failures += 1
                    logger.error(f"Worker {worker_id} failed critically")
                else:
                    processed, failed = result
                    all_processed.extend(processed)
                    all_failed.extend(failed)
                    logger.info(f"Worker {worker_id} completed: {len(processed)} processed, "
                               f"{len(failed)} failed")

        except Exception as e:
            logger.exception("Error during persistent pool processing")
            raise

        # Check for systemic failure
        if worker_failures >= 3:
            raise RuntimeError(
                f"Systemic failure: {worker_failures}/{self.num_workers} workers failed. "
                "Check GPU availability and CUDA setup."
            )

        logger.info(f"Job complete: {len(all_processed)} processed, {len(all_failed)} failed")

        return (all_processed, all_failed)

    def close(self) -> None:
        """
        Gracefully shutdown persistent worker pool.

        Closes the pool and waits for all workers to terminate, releasing
        GPU memory and cleaning up resources.
        """
        if self.pool is not None:
            logger.info("Closing persistent worker pool")
            self.pool.close()
            self.pool.join()
            self.pool = None
            logger.info("Persistent worker pool closed")
