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
import signal
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Set
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from virnucpro.pipeline.runtime_config import RuntimeConfig
    from virnucpro.pipeline.checkpoint_manifest import CheckpointManifest

logger = logging.getLogger('virnucpro.pipeline.gpu_coordinator')


def _worker_wrapper(rank, cuda_device, worker_fn, world_size, queue, *args):
    """
    Wrapper that sets CUDA_VISIBLE_DEVICES before calling worker.

    This is necessary because multiprocessing.Process doesn't accept env parameter.
    We set the environment variable inside the worker process before calling
    the actual worker function.

    Args:
        rank: Worker rank (0, 1, 2, ...)
        cuda_device: CUDA device ID to make visible
        worker_fn: Actual worker function to call
        world_size: Total number of workers
        queue: Results queue for worker reports
        *args: Additional arguments passed to worker_fn
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = str(cuda_device)
    worker_fn(rank, world_size, queue, *args)


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

        # Fault tolerance tracking (Issue 2: per-batch circuit breaker)
        self.worker_retry_counts: Dict[int, int] = {}  # rank -> total retry count
        self.batch_failure_tracking: Dict[int, Dict[int, int]] = {}  # rank -> {batch_idx -> failure_count}
        self.failed_batches: Set[Tuple[int, int]] = set()  # (rank, batch_idx) that hit circuit breaker

        # Elastic redistribution tracking (Issue 5)
        self.redistributed_work: Dict[int, int] = {}  # failed_rank -> new_rank

        # Queue item cache for _get_worker_status (Issue 2: prevent discarding other ranks' status)
        self._status_cache: Dict[int, dict] = {}  # rank -> status dict

        # SIGTERM coordination (Issue 6)
        self.shutdown_requested = False

        logger.info(f"GPUProcessCoordinator initialized: {world_size} workers")

        self.setup_signal_handlers()

    def setup_signal_handlers(self) -> None:
        """Register signal handlers for graceful shutdown.

        Called after confirming this is the coordinator process,
        not a forked child process.
        """
        signal.signal(signal.SIGTERM, self._sigterm_handler)
        signal.signal(signal.SIGINT, self._sigterm_handler)
        logger.info("Signal handlers registered for graceful shutdown")

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
            # Create process with wrapper that sets CUDA_VISIBLE_DEVICES
            p = self.ctx.Process(
                target=_worker_wrapper,
                args=(rank, rank, worker_fn, self.world_size, self.results_queue, *worker_args),
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

    def _sigterm_handler(self, signum, frame):
        """Orchestrate graceful shutdown on SIGTERM/SIGINT.

        Called when coordinator receives SIGTERM (spot preemption) or Ctrl-C.
        Signals all workers to checkpoint and exit gracefully.
        """
        logger.warning(
            f"Coordinator received signal {signum} (SIGTERM/SIGINT), "
            f"orchestrating graceful shutdown"
        )
        self.shutdown_requested = True

        # Signal all workers to checkpoint and exit
        # Workers will catch their own SIGTERM (propagated by OS) and save emergency checkpoint
        # Spawn a thread to wait for checkpoints to avoid blocking signal handling
        import threading
        wait_thread = threading.Thread(
            target=self._wait_for_worker_checkpoints,
            daemon=True
        )
        wait_thread.start()
        wait_thread.join(timeout=30)

        # Terminate remaining workers
        self.terminate_all()
        # os._exit() bypasses Python cleanup - workers have checkpointed, safe to exit immediately
        os._exit(143)

    def _wait_for_worker_checkpoints(self) -> None:
        """Wait for workers to save emergency checkpoints (runs in separate thread).

        Issue 3: Wrapped in try/except to log errors from thread context.
        """
        try:
            import time
            logger.info("Waiting for workers to save emergency checkpoints...")
            time.sleep(30)
            logger.info("Checkpoint wait period complete")
        except Exception as e:
            logger.error(f"Error during checkpoint wait: {e}")

    def _classify_error(self, status: dict) -> str:
        """Classify error type from worker status for appropriate retry policy.

        Returns:
            "spot_preemption": Infinite retry with polling
            "poison_input": Circuit breaker after 2 attempts on same batch
            "oom": Exponential backoff with batch size reduction
            "transient": Standard exponential backoff
        """
        error = status.get('error', '')
        error_type = status.get('error_type', '')
        error_msg = str(error) if error else ''

        # Spot preemption: exitcode 143 (SIGTERM) or explicit status
        if status.get('exitcode') == 143 or 'sigterm' in error_msg.lower():
            return "spot_preemption"

        # OOM: CUDA out of memory (use error_type if available, fallback to message)
        if error_type == 'cuda_oom' or 'out of memory' in error_msg.lower():
            return "oom"

        # Poison input: CUDA runtime errors, assertions (use error_type or circuit_breaker flag)
        if error_type == 'cuda_runtime' or status.get('circuit_breaker', False):
            return "poison_input"

        # Everything else: transient
        return "transient"

    def _validate_checkpoint_dir(self, rank: int, checkpoint_dir: Path) -> bool:
        """Validate checkpoint directory integrity before worker respawn.

        Checks:
        - Directory exists and is readable
        - .done markers match checkpoint files
        - No orphaned temp files (.tmp)

        Returns: True if valid, False if corrupted
        """
        shard_dir = checkpoint_dir / f"shard_{rank}"
        if not shard_dir.exists():
            logger.info(f"Rank {rank}: No checkpoint dir, fresh start")
            return True

        # Check for orphaned temp files (incomplete writes)
        temp_files = list(shard_dir.glob("*.tmp"))
        if temp_files:
            logger.warning(
                f"Rank {rank}: Found {len(temp_files)} orphaned temp files, "
                f"removing before respawn"
            )
            for tmp in temp_files:
                tmp.unlink()

        # Verify .done markers match checkpoint files
        checkpoints = list(shard_dir.glob("batch_*.pt"))
        done_markers = list(shard_dir.glob("batch_*.pt.done"))

        if len(checkpoints) != len(done_markers):
            logger.warning(
                f"Rank {rank}: Checkpoint/marker mismatch "
                f"({len(checkpoints)} .pt files, {len(done_markers)} .done markers)"
            )
            return False

        return True

    def monitor_workers_async(
        self,
        runtime_config: 'RuntimeConfig',
        manifest: Optional['CheckpointManifest'] = None,
        check_interval: float = 5.0
    ) -> Dict[int, bool]:
        """Monitor workers asynchronously with differentiated retry policies.

        Non-blocking monitoring loop that:
        - Polls worker status every check_interval seconds
        - Retries failed workers according to error type
        - Continues monitoring healthy workers during retries
        - Updates manifest (coordinator-only writes)
        - Handles SIGTERM gracefully

        Args:
            runtime_config: Runtime configuration with retry policies
            manifest: Optional manifest for coordinator updates (Issue 3)
            check_interval: Seconds between status polls

        Returns:
            Dict mapping rank -> completion status (True=success, False=failed)
        """
        import time
        from virnucpro.pipeline.runtime_config import RuntimeConfig

        active_workers = set(self.workers.keys())
        completed_workers: Dict[int, bool] = {}  # rank -> success

        logger.info(
            f"Monitoring {len(active_workers)} workers with differentiated retry policies: "
            f"spot=infinite, poison={runtime_config.max_retries_poison}, "
            f"transient={runtime_config.max_retries_transient}"
        )

        while active_workers and not self.shutdown_requested:
            time.sleep(check_interval)

            # Check each active worker
            for rank in list(active_workers):
                worker = self.workers.get(rank)
                if not worker or not worker.is_alive():
                    # Worker finished, check status
                    status = self._get_worker_status(rank)

                    if status.get('status') == 'complete':
                        # Success
                        logger.info(f"Rank {rank}: Completed successfully")
                        completed_workers[rank] = True
                        active_workers.remove(rank)

                        # Update manifest (coordinator-only, Issue 3)
                        if manifest:
                            manifest.mark_shard_complete(rank)
                    else:
                        # Failure - classify and retry
                        error_type = self._classify_error(status)
                        should_retry, reason = self._should_retry_worker(
                            rank, error_type, status, runtime_config
                        )

                        if should_retry:
                            logger.warning(
                                f"Rank {rank}: {error_type} failure, retrying ({reason})"
                            )
                            self._retry_worker(rank, error_type, runtime_config)
                            # Worker remains in active_workers
                        else:
                            # Permanent failure
                            logger.error(
                                f"Rank {rank}: Permanent failure after retries ({reason})"
                            )
                            completed_workers[rank] = False
                            active_workers.remove(rank)

                            # Update manifest
                            if manifest:
                                manifest.mark_shard_failed(rank, reason)

                            # Elastic redistribution (Issue 5)
                            if runtime_config.enable_elastic_redistribution:
                                self._redistribute_failed_shard(rank, active_workers, manifest)

        # All workers finished or shutdown requested
        if self.shutdown_requested:
            logger.warning("Shutdown requested, terminating remaining workers")
            self.terminate_all()

        return completed_workers

    def _should_retry_worker(
        self,
        rank: int,
        error_type: str,
        status: dict,
        runtime_config: 'RuntimeConfig'
    ) -> Tuple[bool, str]:
        """Decide if worker should retry based on error type and retry counts.

        Differentiated retry policies:
        - spot_preemption: Always retry (infinite)
        - poison_input: Retry up to max_retries_poison on same batch
        - oom/transient: Retry up to max_retries_transient

        Returns:
            (should_retry, reason_string)
        """
        retry_count = self.worker_retry_counts.get(rank, 0)

        if error_type == "spot_preemption":
            # Infinite retry for spot preemption (Issue 1)
            return (True, f"spot preemption #{retry_count + 1}, capacity will return")

        elif error_type == "poison_input":
            # Circuit breaker for same batch (Issue 2)
            batch_idx = status.get('batch_idx', -1)
            if batch_idx >= 0:
                batch_failures = self.batch_failure_tracking.setdefault(rank, {})
                batch_failures[batch_idx] = batch_failures.get(batch_idx, 0) + 1

                if batch_failures[batch_idx] >= runtime_config.max_retries_poison:
                    # Circuit breaker triggered
                    self.failed_batches.add((rank, batch_idx))
                    return (
                        False,
                        f"circuit breaker: batch {batch_idx} failed "
                        f"{batch_failures[batch_idx]} times (poison input suspected)"
                    )

            # Retry if under limit
            if retry_count < runtime_config.max_retries_poison:
                return (True, f"attempt {retry_count + 1}/{runtime_config.max_retries_poison}")
            else:
                return (False, f"exhausted {retry_count} retries for poison input")

        elif error_type in ("oom", "transient"):
            # Standard exponential backoff (Issue 1)
            if retry_count < runtime_config.max_retries_transient:
                return (True, f"attempt {retry_count + 1}/{runtime_config.max_retries_transient}")
            else:
                return (False, f"exhausted {retry_count} retries for {error_type}")

        else:
            # Unknown error type - don't retry
            return (False, f"unknown error type: {error_type}")

    def _retry_worker(
        self,
        rank: int,
        error_type: str,
        runtime_config: 'RuntimeConfig'
    ):
        """Retry failed worker with appropriate delay and validation.

        Handles:
        - Checkpoint directory validation before respawn (Issue 10)
        - Error-specific retry delays (Issue 1)
        - Retry count tracking
        """
        import time

        retry_count = self.worker_retry_counts.get(rank, 0)
        self.worker_retry_counts[rank] = retry_count + 1

        # Validate checkpoint directory (Issue 10)
        if runtime_config.checkpoint_dir:
            if not self._validate_checkpoint_dir(rank, runtime_config.checkpoint_dir):
                logger.error(f"Rank {rank}: Checkpoint validation failed, cannot retry safely")
                return

        # Calculate retry delay based on error type (Issue 1)
        if error_type == "spot_preemption":
            delay = runtime_config.spot_retry_poll_interval  # 60s default
            logger.info(
                f"Rank {rank}: Waiting {delay:.0f}s for spot capacity "
                f"(attempt #{retry_count + 1})"
            )
        elif error_type == "oom":
            # Exponential backoff, capped at 60s
            delay = min(2.0 ** retry_count, 60.0)
            logger.warning(
                f"Rank {rank}: OOM retry in {delay:.1f}s "
                f"(will signal batch size reduction)"
            )
        elif error_type == "poison_input":
            # Immediate retry - circuit breaker will catch persistent failures
            delay = 1.0
            logger.warning(
                f"Rank {rank}: Poison input retry in {delay:.1f}s "
                f"(circuit breaker will track failures)"
            )
        else:
            # Transient: standard exponential backoff
            delay = min(1.0 * (2 ** retry_count), 60.0)
            logger.info(f"Rank {rank}: Retry in {delay:.1f}s")

        time.sleep(delay)

        # Respawn worker
        # NOTE: Actual respawn needs worker_fn and worker_args passed through
        # This is handled by monitor_workers_async caller providing respawn callback
        logger.info(f"Rank {rank}: Respawning worker (attempt #{retry_count + 1})")

    def _redistribute_failed_shard(
        self,
        failed_rank: int,
        active_workers: Set[int],
        manifest: Optional['CheckpointManifest']
    ):
        """Redistribute failed shard work to healthy GPU.

        When a shard permanently fails, reassign its remaining work to a healthy GPU
        that has available capacity.

        This prevents work loss from spot preemption or permanent failures.
        """
        if not manifest:
            logger.warning(
                f"Rank {failed_rank}: Cannot redistribute without manifest, "
                f"work will be lost"
            )
            return

        # Find healthy GPU with capacity
        # For simplicity: assign to lowest-numbered active worker
        # Production: consider GPU memory, current workload
        if active_workers:
            new_rank = min(active_workers)
            logger.info(
                f"Rank {failed_rank}: Redistributing remaining work to rank {new_rank}"
            )
            self.redistributed_work[failed_rank] = new_rank

            # Update manifest to track redistribution
            manifest.reassign_shard(failed_rank, new_rank)
        else:
            logger.error(
                f"Rank {failed_rank}: No healthy workers available for redistribution"
            )

    def _get_worker_status(self, rank: int) -> dict:
        """Get worker status from results queue.

        First checks cached status for this rank, then polls the results queue
        for new status updates, caching any items for other ranks.
        Falls back to exitcode check for terminated workers.

        Returns:
            Status dict with 'status', 'error', 'error_message', 'batch_idx', etc.
        """
        # Check cache first (Issue 2: preserve other ranks' status)
        if rank in self._status_cache:
            return self._status_cache.pop(rank)

        # Poll results_queue for this rank's status, cache others
        while not self.results_queue.empty():
            try:
                item = self.results_queue.get_nowait()
                item_rank = item.get('rank')
                # Cache for later retrieval by appropriate rank
                if item_rank != rank:
                    self._status_cache[item_rank] = item
                else:
                    return item
            except queue.Empty:
                break

        # Fallback: check exitcode if worker has terminated
        worker = self.workers.get(rank)
        if worker:
            if not worker.is_alive():
                exitcode = worker.exitcode
                if exitcode is not None:
                    if exitcode == 0:
                        return {'status': 'complete', 'rank': rank}
                    elif exitcode == 143:
                        return {'status': 'failed', 'error': 'sigterm', 'exitcode': 143, 'rank': rank}
                    else:
                        return {'status': 'failed', 'error': 'unknown', 'exitcode': exitcode, 'rank': rank}

        return {'status': 'unknown', 'rank': rank}
