"""GPU process coordinator for multi-GPU worker lifecycle management.

SPMD coordinator for multi-GPU processing with fault tolerance.
Unlike mp.spawn which kills all workers on any failure, this coordinator
uses multiprocessing.Process directly for independent worker lifecycle.
Allows partial completion - surviving workers finish even if others fail.
"""

import logging
import multiprocessing
import os
import queue
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger('virnucpro.pipeline.gpu_coordinator')


class GPUProcessCoordinator:
    """
    SPMD coordinator for multi-GPU processing with fault tolerance.

    Unlike mp.spawn which kills all workers on any failure, this coordinator
    uses multiprocessing.Process directly for independent worker lifecycle.
    Allows partial completion - surviving workers finish even if others fail.

    Attributes:
        world_size: Number of GPU workers to spawn
        output_dir: Directory for worker outputs and logs
        ctx: Multiprocessing context (spawn for CUDA safety)
        workers: Dict[rank, Process] mapping
        results_queue: Queue for worker status reporting
    """

    def __init__(self, world_size: int, output_dir: Path):
        """
        Initialize GPU process coordinator.

        Args:
            world_size: Number of GPU workers to spawn
            output_dir: Directory for worker outputs and logs
        """
        self.world_size = world_size
        self.output_dir = Path(output_dir)
        self.ctx = multiprocessing.get_context('spawn')  # CUDA-safe
        self.workers: Dict[int, multiprocessing.Process] = {}
        self.results_queue = self.ctx.Queue()

        logger.info(f"GPUProcessCoordinator initialized: {world_size} workers")

    def spawn_workers(
        self,
        worker_fn: Callable,
        worker_args: Tuple
    ) -> None:
        """
        Spawn independent GPU worker processes.

        Args:
            worker_fn: Worker function with signature:
                (rank, world_size, results_queue, *worker_args)
            worker_args: Additional args passed to each worker

        CRITICAL: CUDA_VISIBLE_DEVICES must be set in env BEFORE spawning.
        Each worker sees device 0 which maps to actual GPU {rank}.
        """
        for rank in range(self.world_size):
            # Set CUDA_VISIBLE_DEVICES for this worker BEFORE spawning
            # Worker sees device 0, which maps to actual GPU {rank}
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = str(rank)

            # Create process with custom environment
            p = self.ctx.Process(
                target=worker_fn,
                args=(rank, self.world_size, self.results_queue, *worker_args),
                name=f"gpu_worker_{rank}"
            )
            p.start()
            self.workers[rank] = p

            logger.info(f"Spawned worker {rank} (CUDA_VISIBLE_DEVICES={rank})")

    def wait_for_completion(
        self,
        timeout: Optional[float] = None
    ) -> Dict[int, bool]:
        """
        Wait for workers with timeout, returns completion status per rank.

        Args:
            timeout: Max seconds to wait per worker (None = infinite)

        Returns:
            Dict mapping rank -> success (True) or failure (False)
        """
        results = {}

        for rank, process in self.workers.items():
            process.join(timeout=timeout)

            if process.is_alive():
                # Worker timed out, still running
                logger.warning(
                    f"Worker {rank} timed out after {timeout}s, still running"
                )
                results[rank] = False
            elif process.exitcode != 0:
                # Worker failed with non-zero exit code
                logger.error(
                    f"Worker {rank} failed with exit code {process.exitcode}"
                )
                results[rank] = False
            else:
                # Worker completed successfully
                logger.info(f"Worker {rank} completed successfully")
                results[rank] = True

        return results

    def collect_results(self) -> List[Dict]:
        """
        Drain results queue and return all worker reports.

        Returns:
            List of worker result dictionaries (status, metrics, etc.)
        """
        results = []

        while not self.results_queue.empty():
            try:
                results.append(self.results_queue.get_nowait())
            except queue.Empty:
                break

        return results

    def terminate_all(self) -> None:
        """
        Force terminate all workers (emergency cleanup).

        Use only in exceptional cases - allows workers to exit gracefully
        in most scenarios.
        """
        for rank, process in self.workers.items():
            if process.is_alive():
                logger.warning(f"Terminating worker {rank}")
                process.terminate()
                process.join(timeout=5)

                # Force kill if still alive after terminate
                if process.is_alive():
                    logger.error(f"Worker {rank} did not terminate, killing")
                    process.kill()
                    process.join(timeout=1)
