"""Unit tests for GPUProcessCoordinator."""

import multiprocessing
import os
import time
from pathlib import Path

import pytest

from virnucpro.pipeline.gpu_coordinator import GPUProcessCoordinator


# Mock worker functions for testing
def successful_worker(rank, world_size, results_queue):
    """Worker that completes successfully."""
    # Simulate some work
    time.sleep(0.1)

    # Report success
    results_queue.put({
        'rank': rank,
        'status': 'complete',
        'message': f'Worker {rank} completed'
    })


def failing_worker(rank, world_size, results_queue):
    """Worker that fails with exception."""
    # Report failure before raising
    results_queue.put({
        'rank': rank,
        'status': 'failed',
        'error': 'Simulated failure'
    })

    # Exit with non-zero code
    import sys
    sys.exit(1)


def slow_worker(rank, world_size, results_queue):
    """Worker that takes longer than timeout."""
    # Sleep longer than test timeout
    time.sleep(10)

    results_queue.put({
        'rank': rank,
        'status': 'complete'
    })


def cuda_env_reporter(rank, world_size, results_queue):
    """Worker that reports its CUDA_VISIBLE_DEVICES setting."""
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', 'NOT_SET')

    results_queue.put({
        'rank': rank,
        'cuda_visible_devices': cuda_visible
    })


def mixed_worker(rank, world_size, results_queue):
    """Worker where rank 0 succeeds, rank 1 fails."""
    if rank == 0:
        successful_worker(rank, world_size, results_queue)
    else:
        failing_worker(rank, world_size, results_queue)


def selective_failure(rank, world_size, results_queue):
    """Worker where rank 1 fails, others succeed."""
    if rank == 1:
        failing_worker(rank, world_size, results_queue)
    else:
        successful_worker(rank, world_size, results_queue)


def arg_reporter(rank, world_size, results_queue, arg1, arg2):
    """Worker that reports received args."""
    results_queue.put({
        'rank': rank,
        'arg1': arg1,
        'arg2': arg2
    })


def world_size_reporter(rank, world_size, results_queue):
    """Worker that reports world_size."""
    results_queue.put({
        'rank': rank,
        'world_size': world_size
    })


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create temporary output directory."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return output_dir


class TestGPUProcessCoordinator:
    """Test GPUProcessCoordinator class."""

    def test_initialization(self, temp_output_dir):
        """Test coordinator initialization."""
        coordinator = GPUProcessCoordinator(
            world_size=2,
            output_dir=temp_output_dir
        )

        assert coordinator.world_size == 2
        assert coordinator.output_dir == temp_output_dir
        assert coordinator.ctx._name == 'spawn'
        assert len(coordinator.workers) == 0
        assert coordinator.results_queue is not None

    def test_spawn_workers_creates_processes(self, temp_output_dir):
        """Test that spawn_workers creates N processes."""
        coordinator = GPUProcessCoordinator(
            world_size=2,
            output_dir=temp_output_dir
        )

        coordinator.spawn_workers(successful_worker, ())

        # Verify processes were created
        assert len(coordinator.workers) == 2
        assert 0 in coordinator.workers
        assert 1 in coordinator.workers

        # Verify processes are alive
        for rank, process in coordinator.workers.items():
            assert process.is_alive()

        # Wait for completion
        status = coordinator.wait_for_completion(timeout=5.0)

        # All workers should succeed
        assert status[0] is True
        assert status[1] is True

    def test_cuda_visible_devices_set(self, temp_output_dir):
        """Test that each worker gets correct CUDA_VISIBLE_DEVICES."""
        coordinator = GPUProcessCoordinator(
            world_size=3,
            output_dir=temp_output_dir
        )

        coordinator.spawn_workers(cuda_env_reporter, ())

        # Wait for completion
        coordinator.wait_for_completion(timeout=5.0)

        # Collect results from queue
        results = coordinator.collect_results()

        # Sort by rank for easier verification
        results.sort(key=lambda x: x['rank'])

        # Verify each worker got its rank as CUDA_VISIBLE_DEVICES
        assert results[0]['cuda_visible_devices'] == '0'
        assert results[1]['cuda_visible_devices'] == '1'
        assert results[2]['cuda_visible_devices'] == '2'

    def test_wait_for_completion_all_success(self, temp_output_dir):
        """Test wait_for_completion when all workers succeed."""
        coordinator = GPUProcessCoordinator(
            world_size=2,
            output_dir=temp_output_dir
        )

        coordinator.spawn_workers(successful_worker, ())
        status = coordinator.wait_for_completion(timeout=5.0)

        # All workers should succeed
        assert len(status) == 2
        assert all(status.values())

    def test_wait_for_completion_partial_failure(self, temp_output_dir):
        """Test wait_for_completion with partial failure."""
        coordinator = GPUProcessCoordinator(
            world_size=2,
            output_dir=temp_output_dir
        )

        # Use mixed worker function - rank 0 succeeds, rank 1 fails
        coordinator.spawn_workers(mixed_worker, ())
        status = coordinator.wait_for_completion(timeout=5.0)

        # Worker 0 should succeed, worker 1 should fail
        assert status[0] is True
        assert status[1] is False

    def test_wait_for_completion_timeout(self, temp_output_dir):
        """Test wait_for_completion with timeout."""
        coordinator = GPUProcessCoordinator(
            world_size=1,
            output_dir=temp_output_dir
        )

        coordinator.spawn_workers(slow_worker, ())
        status = coordinator.wait_for_completion(timeout=0.5)

        # Worker should timeout (still running)
        assert status[0] is False

        # Cleanup - terminate the still-running worker
        coordinator.terminate_all()

    def test_results_queue_receives_reports(self, temp_output_dir):
        """Test that results queue receives worker reports."""
        coordinator = GPUProcessCoordinator(
            world_size=2,
            output_dir=temp_output_dir
        )

        coordinator.spawn_workers(successful_worker, ())
        coordinator.wait_for_completion(timeout=5.0)

        # Collect results
        results = coordinator.collect_results()

        # Should have 2 results (one per worker)
        assert len(results) == 2

        # Verify result structure
        for result in results:
            assert 'rank' in result
            assert 'status' in result
            assert result['status'] == 'complete'

    def test_terminate_all(self, temp_output_dir):
        """Test force termination of workers."""
        coordinator = GPUProcessCoordinator(
            world_size=2,
            output_dir=temp_output_dir
        )

        # Spawn slow workers
        coordinator.spawn_workers(slow_worker, ())

        # Verify workers are alive
        time.sleep(0.1)
        for process in coordinator.workers.values():
            assert process.is_alive()

        # Terminate all
        coordinator.terminate_all()

        # Verify workers are stopped
        for process in coordinator.workers.values():
            assert not process.is_alive()


class TestWorkerIsolation:
    """Test worker independence and isolation."""

    def test_workers_independent(self, temp_output_dir):
        """Test that failure of one worker doesn't affect others."""
        coordinator = GPUProcessCoordinator(
            world_size=3,
            output_dir=temp_output_dir
        )

        # Worker 1 fails, others succeed
        coordinator.spawn_workers(selective_failure, ())
        status = coordinator.wait_for_completion(timeout=5.0)

        # Workers 0 and 2 should succeed, worker 1 should fail
        assert status[0] is True
        assert status[1] is False
        assert status[2] is True

    def test_spawn_context_is_spawn(self, temp_output_dir):
        """Test that coordinator uses spawn context (not fork)."""
        coordinator = GPUProcessCoordinator(
            world_size=1,
            output_dir=temp_output_dir
        )

        # Verify spawn context
        assert coordinator.ctx._name == 'spawn'

        # Verify spawned processes use spawn context
        coordinator.spawn_workers(successful_worker, ())

        # Process should be created and running
        assert len(coordinator.workers) == 1
        assert coordinator.workers[0].is_alive()

        coordinator.wait_for_completion(timeout=5.0)


class TestWorkerArguments:
    """Test worker function argument passing."""

    def test_worker_args_passed_correctly(self, temp_output_dir):
        """Test that additional worker args are passed correctly."""
        coordinator = GPUProcessCoordinator(
            world_size=2,
            output_dir=temp_output_dir
        )

        # Pass additional args
        coordinator.spawn_workers(arg_reporter, ('test_arg', 42))
        coordinator.wait_for_completion(timeout=5.0)

        # Collect results
        results = coordinator.collect_results()

        # Verify args received correctly
        assert len(results) == 2
        for result in results:
            assert result['arg1'] == 'test_arg'
            assert result['arg2'] == 42

    def test_world_size_passed_correctly(self, temp_output_dir):
        """Test that world_size is passed correctly to workers."""
        coordinator = GPUProcessCoordinator(
            world_size=4,
            output_dir=temp_output_dir
        )

        coordinator.spawn_workers(world_size_reporter, ())
        coordinator.wait_for_completion(timeout=5.0)

        # Collect results
        results = coordinator.collect_results()

        # All workers should receive world_size=4
        assert len(results) == 4
        for result in results:
            assert result['world_size'] == 4


class TestGPUCoordinatorFaultTolerance:
    """Test GPUProcessCoordinator fault tolerance (requires Plan 09-05 implementation).

    NOTE: These tests are placeholders for fault tolerance features to be implemented
    in Plan 09-05. They are currently skipped with pytest.skip() markers.
    """

    @pytest.mark.skip(reason="Requires Plan 09-05: fault tolerance implementation")
    def test_classify_error_spot_preemption_exitcode(self):
        """Test error classification for spot preemption via exit code."""
        # This test will verify _classify_error returns "spot_preemption"
        # when status dict has exitcode=143
        pass

    @pytest.mark.skip(reason="Requires Plan 09-05: fault tolerance implementation")
    def test_classify_error_spot_preemption_message(self):
        """Test error classification for spot preemption via error message."""
        # This test will verify _classify_error returns "spot_preemption"
        # when error_message contains "Received SIGTERM"
        pass

    @pytest.mark.skip(reason="Requires Plan 09-05: fault tolerance implementation")
    def test_classify_error_oom(self):
        """Test error classification for OOM errors."""
        # This test will verify _classify_error returns "oom"
        # when error="cuda_oom" or message contains "CUDA out of memory"
        pass

    @pytest.mark.skip(reason="Requires Plan 09-05: fault tolerance implementation")
    def test_classify_error_poison_input(self):
        """Test error classification for poison inputs."""
        # This test will verify _classify_error returns "poison_input"
        # when error="cuda_runtime" or circuit_breaker=True
        pass

    @pytest.mark.skip(reason="Requires Plan 09-05: fault tolerance implementation")
    def test_classify_error_transient_fallback(self):
        """Test error classification falls back to transient."""
        # This test will verify _classify_error returns "transient"
        # for generic/unknown errors
        pass

    @pytest.mark.skip(reason="Requires Plan 09-05: fault tolerance implementation")
    def test_should_retry_spot_preemption_infinite(self):
        """Test spot preemption allows infinite retries."""
        # This test will verify _should_retry_worker returns (True, message)
        # for error_type="spot_preemption" regardless of retry_count
        pass

    @pytest.mark.skip(reason="Requires Plan 09-05: fault tolerance implementation")
    def test_should_retry_poison_input_circuit_breaker(self):
        """Test circuit breaker triggers after 2 same-batch failures."""
        # This test will verify:
        # - First failure: returns (True, retry message)
        # - Second failure on same batch: returns (False, circuit breaker message)
        # - (batch_idx, rank) added to failed_batches set
        pass

    @pytest.mark.skip(reason="Requires Plan 09-05: fault tolerance implementation")
    def test_should_retry_oom_exponential_backoff(self):
        """Test OOM retry with exponential backoff (max 3 attempts)."""
        # This test will verify _should_retry_worker:
        # - retry_count < 3: returns (True, "attempt N/3")
        # - retry_count >= 3: returns (False, "exhausted retries")
        pass

    @pytest.mark.skip(reason="Requires Plan 09-05: fault tolerance implementation")
    def test_should_retry_transient_max_retries(self):
        """Test transient errors respect max retries (3 attempts)."""
        # Similar to OOM test but with error_type="transient"
        pass

    @pytest.mark.skip(reason="Requires Plan 09-05: fault tolerance implementation")
    def test_validate_checkpoint_dir_removes_orphaned_tmp_files(self):
        """Test checkpoint validation removes .tmp files."""
        # This test will verify _validate_checkpoint_dir:
        # - Removes .tmp files
        # - Preserves .pt files
        # - Returns True
        pass

    @pytest.mark.skip(reason="Requires Plan 09-05: fault tolerance implementation")
    def test_validate_checkpoint_dir_done_marker_mismatch_warning(self):
        """Test validation warns on .done marker mismatch."""
        # This test will verify _validate_checkpoint_dir:
        # - Logs warning when .pt files != .done markers
        # - Still returns True (allows resume)
        pass

    @pytest.mark.skip(reason="Requires Plan 09-05: fault tolerance implementation")
    def test_validate_checkpoint_dir_fresh_start(self):
        """Test validation succeeds for non-existent shard directory."""
        # This test will verify _validate_checkpoint_dir returns True
        # when shard directory doesn't exist (fresh start)
        pass

    @pytest.mark.skip(reason="Requires Plan 09-05: fault tolerance implementation")
    def test_retry_worker_spot_preemption_delay(self):
        """Test retry delay for spot preemption."""
        # This test will verify _retry_worker calls time.sleep
        # with runtime_config.spot_retry_poll_interval (60s default)
        pass

    @pytest.mark.skip(reason="Requires Plan 09-05: fault tolerance implementation")
    def test_retry_worker_oom_exponential_backoff(self):
        """Test exponential backoff for OOM retries."""
        # This test will verify _retry_worker sleep duration:
        # - retry_count=0: sleep(1.0)
        # - retry_count=2: sleep(4.0)
        # - retry_count=10: sleep(60.0) - capped at max
        pass

    @pytest.mark.skip(reason="Requires Plan 09-05: fault tolerance implementation")
    def test_retry_worker_validates_checkpoint_before_respawn(self):
        """Test worker not respawned if checkpoint validation fails."""
        # This test will verify _retry_worker:
        # - Calls _validate_checkpoint_dir
        # - If validation returns False, worker is NOT respawned
        pass

    @pytest.mark.skip(reason="Requires Plan 09-05: fault tolerance implementation")
    def test_sigterm_handler_sets_shutdown_flag(self):
        """Test SIGTERM handler sets shutdown flag."""
        # This test will verify _sigterm_handler:
        # - Sets self.shutdown_requested = True
        pass

    @pytest.mark.skip(reason="Requires Plan 09-05: fault tolerance implementation")
    def test_sigterm_handler_waits_for_checkpoints(self):
        """Test SIGTERM handler waits for worker checkpoints."""
        # This test will verify _sigterm_handler:
        # - Calls time.sleep(30) to wait for worker checkpoints
        # - Calls terminate_all()
        pass
