---
phase: 09-checkpointing-integration
plan: 05
subsystem: multi-gpu-coordination
tags: [fault-tolerance, retry-policies, checkpointing, spot-instances, elastic-redistribution, async-monitoring]

# Dependency graph
requires:
  - phase: 09-02
    provides: CheckpointManifest for multi-GPU coordination
  - phase: 09-04
    provides: GPU worker checkpoint integration with resume and SIGTERM handling
provides:
  - RuntimeConfig dataclass separating operational from model architecture config
  - Fault-tolerant coordinator with differentiated retry policies (spot=infinite, poison=2-attempt circuit breaker, transient=3-attempt exponential backoff)
  - Async worker monitoring (non-blocking, continues healthy workers during retries)
  - SIGTERM handler for graceful spot preemption shutdown (30s checkpoint wait)
  - Elastic redistribution of failed shard work to healthy GPUs
  - Checkpoint directory validation before respawn (removes orphaned .tmp files)
  - Error classification with intelligent diagnostic tiering (spot_preemption, poison_input, oom, transient)
  - Coordinator-only manifest writes (workers signal via results_queue)
affects: [09-06, phase-10, production-deployment]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - RuntimeConfig dataclass for operational parameters (checkpointing, retries, timeouts)
    - Differentiated retry policies based on error classification
    - Per-batch circuit breaker for poison input detection
    - Async monitoring loop with non-blocking worker polls
    - SIGTERM handler coordination with worker emergency checkpoints
    - Elastic redistribution using CheckpointManifest.reassign_shard()

key-files:
  created:
    - virnucpro/pipeline/runtime_config.py
  modified:
    - virnucpro/pipeline/gpu_coordinator.py
    - virnucpro/pipeline/multi_gpu_inference.py

key-decisions:
  - "RuntimeConfig separates operational params from model_config (Issue 8)"
  - "Timeout is per-attempt, not global (Issue 7)"
  - "Only coordinator writes manifest, workers signal via results_queue (Issue 3)"
  - "Spot preemption retries infinitely with 60s polling (Issue 1)"
  - "Poison input triggers 2-attempt circuit breaker per batch (Issue 2)"
  - "OOM and transient errors retry up to 3 times with exponential backoff (Issue 1)"
  - "Async monitoring non-blocking, continues healthy workers during retries (Issue 4)"
  - "Elastic redistribution reassigns failed work to lowest-numbered healthy GPU (Issue 5)"
  - "SIGTERM handler waits 30s for workers to checkpoint before terminating (Issue 6)"
  - "Checkpoint directory validation removes orphaned .tmp files before respawn (Issue 10)"
  - "Error classification explicit with spot_preemption, poison_input, oom, transient categories (Issue 9)"

patterns-established:
  - "RuntimeConfig.to_dict()/from_dict() for worker serialization"
  - "GPUProcessCoordinator.monitor_workers_async() replaces wait_for_completion() for retry-enabled workflows"
  - "Error type from gpu_worker status dict (error_type field + backward compatible error field)"
  - "Retry count tracking per rank (worker_retry_counts) and per batch (batch_failure_tracking)"
  - "SIGTERM signal.signal() registration in __init__ for coordinator lifecycle"
  - "Checkpoint validation before respawn as safety gate"

# Metrics
duration: 4.5min
completed: 2026-02-06
---

# Phase 09 Plan 05: Coordinator Integration Summary

**Fault-tolerant coordinator with differentiated retry policies (spot=infinite, poison=2-attempt circuit breaker, OOM/transient=3-attempt exponential backoff), async monitoring, elastic redistribution, and graceful SIGTERM handling**

## Performance

- **Duration:** 4.5 min
- **Started:** 2026-02-06T05:01:29Z
- **Completed:** 2026-02-06T05:06:01Z
- **Tasks:** 3/3
- **Files modified:** 3

## Accomplishments

- RuntimeConfig dataclass separates operational parameters from model architecture config
- Async worker monitoring with differentiated retry policies based on error classification
- SIGTERM handler orchestrates graceful shutdown with 30s checkpoint wait for spot instances
- Elastic redistribution reassigns failed shard work to healthy GPUs via manifest
- Checkpoint directory validation before respawn prevents corruption from orphaned .tmp files
- Error classification with explicit spot_preemption, poison_input, oom, transient categories
- Coordinator-only manifest writes eliminate cross-process JSON corruption
- Per-batch circuit breaker for poison input detection (2 failures on same batch → permanent failure)

## Task Commits

Each task was committed atomically:

1. **Task 1: Create RuntimeConfig dataclass** - `fdbdd8d` (feat)
2. **Task 2: Add fault-tolerant coordinator with async monitoring** - `251863d` (feat)
3. **Task 3: Wire RuntimeConfig into multi_gpu_inference** - `ca637e3` (feat)

## Files Created/Modified

- `virnucpro/pipeline/runtime_config.py` - RuntimeConfig dataclass for operational parameters (checkpointing, retry policies, timeouts, elastic redistribution control)
- `virnucpro/pipeline/gpu_coordinator.py` - Added monitor_workers_async(), _classify_error(), _should_retry_worker(), _retry_worker(), _redistribute_failed_shard(), _validate_checkpoint_dir(), _sigterm_handler()
- `virnucpro/pipeline/multi_gpu_inference.py` - Integrated RuntimeConfig, async monitoring, coordinator-only manifest writes, partial failure handling

## Decisions Made

**Key architectural decisions:**

1. **RuntimeConfig separation (Issue 8)**: Operational parameters (checkpointing, retries, timeouts) separated from model architecture config (dtype, hidden_size, num_layers) for clean reproducibility boundary

2. **Per-attempt timeout (Issue 7)**: timeout_per_attempt applies to each retry attempt independently, not globally, enabling spot preemption infinite retry

3. **Differentiated retry policies (Issue 1)**:
   - Spot preemption: Infinite retry with 60s polling (capacity returns eventually)
   - Poison input: 2-attempt circuit breaker per batch (same batch crashes twice → isolate toxic sequences)
   - OOM: 3-attempt exponential backoff with batch size reduction signal
   - Transient: 3-attempt exponential backoff

4. **Per-batch circuit breaker (Issue 2)**: Track failures per (rank, batch_idx) tuple, trigger circuit breaker after 2 failures on same batch to isolate poison inputs

5. **Coordinator-only manifest writes (Issue 3)**: Workers signal via results_queue, only coordinator updates manifest to eliminate POSIX lock contention and cross-process JSON corruption

6. **Async monitoring (Issue 4)**: Non-blocking monitoring loop polls workers every 5s, continues monitoring healthy workers during retries, enables partial completion

7. **Elastic redistribution (Issue 5)**: Failed shard work reassigned to lowest-numbered active worker via CheckpointManifest.reassign_shard()

8. **SIGTERM handler (Issue 6)**: Coordinator catches SIGTERM, waits 30s for workers to save emergency checkpoints, then terminates remaining workers gracefully with exit code 143

9. **Error classification (Issue 9)**: Explicit _classify_error() method checks exitcode, error_type field, error message patterns, returns spot_preemption/poison_input/oom/transient for policy dispatch

10. **Checkpoint validation (Issue 10)**: _validate_checkpoint_dir() checks for orphaned .tmp files, verifies .done markers, removes corrupted state before worker respawn

## Deviations from Plan

None - plan executed exactly as written. All 10 issues from plan review addressed:
- Issue 1: Differentiated retry policies implemented
- Issue 2: Per-batch circuit breaker for poison inputs
- Issue 3: Coordinator-only manifest writes (workers use results_queue)
- Issue 4: Async monitoring with non-blocking polls
- Issue 5: Elastic redistribution to healthy GPUs
- Issue 6: SIGTERM handler with 30s checkpoint wait
- Issue 7: timeout_per_attempt (not global)
- Issue 8: RuntimeConfig separates operational from model config
- Issue 9: Explicit error classification
- Issue 10: Checkpoint directory validation before respawn

## Issues Encountered

None - implementation followed Phase 9 architectural patterns from 09-01 (CheckpointTrigger), 09-02 (CheckpointManifest), and 09-04 (gpu_worker integration).

## User Setup Required

None - no external service configuration required. Coordinator retry policies are automatic based on error classification.

## Next Phase Readiness

**Ready for 09-06 (End-to-End Integration Tests):**
- Coordinator async monitoring tested (verification command passed)
- Error classification logic validated (all 4 error types covered)
- RuntimeConfig serialization confirmed (to_dict/from_dict roundtrip)
- Partial failure handling ready (salvages successful worker results)

**Next steps (09-06):**
- Integration test: Simulated spot preemption (SIGTERM) → emergency checkpoint → resume → completion
- Integration test: Poison input injection → 2-attempt circuit breaker → partial completion
- Integration test: OOM simulation → exponential backoff → batch size reduction signal
- Integration test: Multiple worker failures → elastic redistribution → successful completion
- Integration test: Manifest corruption → fallback to .tmp/.backup → recovery

**No blockers.**

---
*Phase: 09-checkpointing-integration*
*Completed: 2026-02-06*
