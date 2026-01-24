"""Generic batch queue manager for multi-GPU work distribution"""

import multiprocessing
import inspect
import logging
from enum import Enum
from pathlib import Path
from typing import List, Callable, Tuple, Dict, Any, Optional
import time

from virnucpro.pipeline.persistent_pool import PersistentWorkerPool

logger = logging.getLogger('virnucpro.work_queue')

# Module-level globals for worker processes (set by Pool initializer)
_worker_progress_queue = None
_worker_function = None


def _init_worker(progress_queue, worker_function):
    """
    Pool initializer function to set module-level worker globals.

    This allows Queue to be inherited by child processes through multiprocessing
    context instead of being pickled as a function argument.

    Args:
        progress_queue: Multiprocessing Queue for progress reporting (can be None)
        worker_function: Function to execute in workers
    """
    global _worker_progress_queue, _worker_function
    _worker_progress_queue = progress_queue
    _worker_function = worker_function


def _worker_wrapper(
    file_subset: List[Path],
    device_id: int,
    kwargs: Dict[str, Any]
) -> Optional[Tuple[List[Path], List[Tuple[Path, str]]]]:
    """
    Wrapper around worker function to handle exceptions.

    This is a module-level function (not a method) so it can be pickled
    for multiprocessing.

    Args:
        file_subset: Files to process
        device_id: CUDA device ID
        kwargs: Additional keyword arguments

    Returns:
        Worker function result or None if critical failure
    """
    try:
        return _worker_function(file_subset, device_id, **kwargs)
    except Exception as e:
        logger.exception(f"Worker {device_id} critical failure")
        return None


class WorkerStatus(Enum):
    """Status states for worker processes"""
    IDLE = "idle"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class BatchQueueManager:
    """
    Generic batch queue manager for coordinating multi-GPU work.

    Uses spawn context for CUDA safety and SimpleQueue to avoid deadlocks.
    Validates worker function signatures and tracks worker status.

    Example:
        >>> def worker(files, device_id, output_dir):
        ...     return (processed, failed)
        >>> manager = BatchQueueManager(4, worker, spawn_context=True)
        >>> assignments = assign_files_round_robin(files, 4)
        >>> processed, failed = manager.process_files(assignments, output_dir=Path('/tmp'))
    """

    def __init__(
        self,
        num_workers: int,
        worker_function: Callable,
        spawn_context: bool = True,
        progress_queue: Optional[multiprocessing.Queue] = None,
        use_persistent_pool: bool = False,
        model_type: Optional[str] = None
    ):
        """
        Initialize batch queue manager.

        Args:
            num_workers: Number of worker processes (typically number of GPUs)
            worker_function: Function to execute in workers
            spawn_context: Use spawn context for CUDA safety (default: True)
            progress_queue: Optional queue for workers to report progress events
            use_persistent_pool: Use persistent worker pool (default: False for backward compatibility)
            model_type: Model type for persistent pool ('esm2' or 'dnabert', required if use_persistent_pool=True)

        Raises:
            ValueError: If worker_function signature is invalid or model_type missing when use_persistent_pool=True
        """
        # Validate worker function signature (skip if using persistent pool)
        if not use_persistent_pool:
            sig = inspect.signature(worker_function)
            params = list(sig.parameters.keys())
            if len(params) < 2:
                raise ValueError(
                    f"Worker function must accept at least (file_subset, device_id, **kwargs), "
                    f"got parameters: {params}"
                )

        # Validate persistent pool configuration
        if use_persistent_pool and model_type is None:
            raise ValueError("model_type is required when use_persistent_pool=True")

        self.num_workers = num_workers
        self.worker_function = worker_function
        self.ctx = multiprocessing.get_context('spawn') if spawn_context else multiprocessing
        self.worker_status = {i: WorkerStatus.IDLE for i in range(num_workers)}
        self.progress_queue = progress_queue
        self.use_persistent_pool = use_persistent_pool
        self.model_type = model_type
        self.persistent_pool = None

        logger.info(f"Initialized BatchQueueManager with {num_workers} workers "
                   f"({'spawn' if spawn_context else 'default'} context, "
                   f"persistent_pool={'enabled' if use_persistent_pool else 'disabled'})")

    def process_files(
        self,
        file_assignments: List[List[Path]],
        **worker_kwargs
    ) -> Tuple[List[Path], List[Tuple[Path, str]]]:
        """
        Process files across multiple workers in parallel.

        If persistent pool is enabled, uses pre-loaded models from persistent workers.
        Otherwise, creates a new pool for this job (traditional behavior).

        Args:
            file_assignments: List of file lists, one per worker
            **worker_kwargs: Additional keyword arguments passed to worker function

        Returns:
            Tuple of (processed_files, failed_files)
            - processed_files: List of successfully processed output file paths
            - failed_files: List of (file_path, error_message) tuples

        Raises:
            RuntimeError: If systemic failure detected (3+ workers fail)
        """
        if len(file_assignments) != self.num_workers:
            raise ValueError(
                f"Expected {self.num_workers} file assignments, got {len(file_assignments)}"
            )

        # Check if using persistent pool
        if self.use_persistent_pool and self.persistent_pool is not None:
            logger.info("Using persistent worker pool for processing")
            return self.persistent_pool.process_job(file_assignments, **worker_kwargs)

        # Check for fallback condition
        if self.use_persistent_pool:
            logger.warning("Persistent pool requested but not created, falling back to standard pool")
        else:
            logger.info("Using standard worker pool for processing")

        # Traditional behavior: create new pool for this job
        # Track results
        all_processed = []
        all_failed = []
        worker_failures = 0

        logger.info(f"Starting parallel processing with {self.num_workers} workers")
        for worker_id, files in enumerate(file_assignments):
            logger.info(f"  Worker {worker_id}: {len(files)} files assigned")

        # Add logging configuration to kwargs
        log_level = logging.getLogger().level
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        worker_kwargs['log_level'] = log_level
        worker_kwargs['log_format'] = log_format

        # NOTE: progress_queue is NOT added to kwargs - it's passed via Pool initializer
        # to avoid pickling Queue objects (which triggers RuntimeError)

        # Create worker arguments: (file_subset, device_id, **worker_kwargs)
        worker_args = []
        for worker_id, file_subset in enumerate(file_assignments):
            self.worker_status[worker_id] = WorkerStatus.PROCESSING
            worker_args.append((file_subset, worker_id))

        start_time = time.time()

        try:
            # Use spawn context pool for CUDA safety
            # Pass progress_queue and worker_function via initializer so they're inherited, not pickled
            # This avoids "Queue objects should only be shared between processes through inheritance" error
            with self.ctx.Pool(
                self.num_workers,
                initializer=_init_worker,
                initargs=(self.progress_queue, self.worker_function)
            ) as pool:
                # Launch workers with starmap using module-level wrapper function
                results = pool.starmap(
                    _worker_wrapper,
                    [(args[0], args[1], worker_kwargs) for args in worker_args]
                )

                # Collect results
                for worker_id, result in enumerate(results):
                    if result is None:
                        # Worker failed critically
                        self.worker_status[worker_id] = WorkerStatus.FAILED
                        worker_failures += 1
                        logger.error(f"Worker {worker_id} failed critically")
                    else:
                        processed, failed = result
                        all_processed.extend(processed)
                        all_failed.extend(failed)
                        self.worker_status[worker_id] = WorkerStatus.COMPLETED
                        logger.info(f"Worker {worker_id} completed: {len(processed)} processed, "
                                   f"{len(failed)} failed")

        except Exception as e:
            logger.exception("Error during parallel processing")
            raise

        elapsed = time.time() - start_time

        # Check for systemic failure
        if worker_failures >= 3:
            raise RuntimeError(
                f"Systemic failure: {worker_failures}/{self.num_workers} workers failed. "
                "Check GPU availability and CUDA setup."
            )

        logger.info(f"Parallel processing complete in {elapsed:.1f}s: "
                   f"{len(all_processed)} processed, {len(all_failed)} failed")

        return (all_processed, all_failed)

    def get_worker_status(self) -> Dict[int, WorkerStatus]:
        """
        Get current status of all workers.

        Returns:
            Dictionary mapping worker ID to WorkerStatus
        """
        return self.worker_status.copy()

    def is_complete(self) -> bool:
        """
        Check if all workers have completed or failed.

        Returns:
            True if all workers are in COMPLETED or FAILED state
        """
        return all(
            status in (WorkerStatus.COMPLETED, WorkerStatus.FAILED)
            for status in self.worker_status.values()
        )

    def create_persistent_pool(self) -> None:
        """
        Create persistent worker pool with model pre-loading.

        This method initializes a persistent pool that loads models once and
        reuses them across multiple process_files() calls. Only available when
        use_persistent_pool=True was set during initialization.

        Raises:
            RuntimeError: If persistent pool not enabled or already created
        """
        if not self.use_persistent_pool:
            raise RuntimeError(
                "Persistent pool not enabled. Set use_persistent_pool=True during initialization."
            )

        if self.persistent_pool is not None:
            raise RuntimeError("Persistent pool already created. Call close_persistent_pool() first.")

        logger.info(f"Creating persistent worker pool with {self.num_workers} workers for {self.model_type}")

        self.persistent_pool = PersistentWorkerPool(
            num_workers=self.num_workers,
            model_type=self.model_type,
            spawn_context=True,
            progress_queue=self.progress_queue
        )
        self.persistent_pool.create_pool()

        logger.info(f"Persistent worker pool created successfully (model_type={self.model_type})")

    def close_persistent_pool(self) -> None:
        """
        Close persistent worker pool and release resources.

        This method gracefully shuts down the persistent pool, terminating
        worker processes and releasing GPU memory.

        Safe to call even if pool doesn't exist (no-op).
        """
        if self.persistent_pool is not None:
            logger.info("Closing persistent pool")
            self.persistent_pool.close()
            self.persistent_pool = None
            logger.info("Persistent pool closed")
