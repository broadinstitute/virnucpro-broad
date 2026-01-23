"""Generic batch queue manager for multi-GPU work distribution"""

import multiprocessing
import inspect
import logging
from enum import Enum
from pathlib import Path
from typing import List, Callable, Tuple, Dict, Any, Optional
import time

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
        progress_queue: Optional[multiprocessing.Queue] = None
    ):
        """
        Initialize batch queue manager.

        Args:
            num_workers: Number of worker processes (typically number of GPUs)
            worker_function: Function to execute in workers
            spawn_context: Use spawn context for CUDA safety (default: True)
            progress_queue: Optional queue for workers to report progress events

        Raises:
            ValueError: If worker_function signature is invalid
        """
        # Validate worker function signature
        sig = inspect.signature(worker_function)
        params = list(sig.parameters.keys())
        if len(params) < 2:
            raise ValueError(
                f"Worker function must accept at least (file_subset, device_id, **kwargs), "
                f"got parameters: {params}"
            )

        self.num_workers = num_workers
        self.worker_function = worker_function
        self.ctx = multiprocessing.get_context('spawn') if spawn_context else multiprocessing
        self.worker_status = {i: WorkerStatus.IDLE for i in range(num_workers)}
        self.progress_queue = progress_queue

        logger.info(f"Initialized BatchQueueManager with {num_workers} workers "
                   f"({'spawn' if spawn_context else 'default'} context)")

    def process_files(
        self,
        file_assignments: List[List[Path]],
        **worker_kwargs
    ) -> Tuple[List[Path], List[Tuple[Path, str]]]:
        """
        Process files across multiple workers in parallel.

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
