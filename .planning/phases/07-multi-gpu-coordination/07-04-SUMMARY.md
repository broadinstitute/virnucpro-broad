---
phase: 07-multi-gpu-coordination
plan: 04
subsystem: infra
tags: [multiprocessing, cuda, spawn-context, fault-tolerance]

# Dependency graph
requires:
  - phase: 07-01
    provides: "SequenceIndex for stride distribution across workers"
  - phase: 07-03
    provides: "Per-worker logging infrastructure"
provides:
  - "GPUProcessCoordinator for multi-GPU worker lifecycle management"
  - "Fault-tolerant process spawning (partial failure support)"
  - "CUDA_VISIBLE_DEVICES isolation per worker"
affects: [07-05, 07-06, 07-07, 07-08]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "multiprocessing.Process with spawn context for CUDA safety"
    - "Wrapper function for per-process environment variable injection"
    - "Results queue for worker status reporting"
    - "Independent worker lifecycle (no mp.spawn killing all on failure)"

key-files:
  created:
    - virnucpro/pipeline/gpu_coordinator.py
    - tests/unit/test_gpu_coordinator.py
  modified: []

key-decisions:
  - "Use multiprocessing.Process directly instead of mp.spawn for fault tolerance"
  - "Module-level _worker_wrapper for CUDA_VISIBLE_DEVICES injection (pickle compatibility)"
  - "wait_for_completion returns per-rank status dict (partial failure tracking)"
  - "Emergency terminate_all method for cleanup (graceful by default)"

patterns-established:
  - "Pattern: SPMD coordinator spawns independent workers, doesn't kill all on single failure"
  - "Pattern: CUDA_VISIBLE_DEVICES set via wrapper function inside worker process"
  - "Pattern: Results queue for push-based worker status reporting"

# Metrics
duration: 4min
completed: 2026-02-04
---

# Phase 7 Plan 4: GPU Process Coordinator Summary

**SPMD coordinator for multi-GPU worker lifecycle with fault tolerance using multiprocessing.Process and spawn context**

## Performance

- **Duration:** 4 min
- **Started:** 2026-02-04T19:45:52Z
- **Completed:** 2026-02-04T19:49:48Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- GPUProcessCoordinator spawns independent GPU workers using multiprocessing.Process
- CUDA_VISIBLE_DEVICES isolation via module-level wrapper function
- Partial failure support - surviving workers complete even if others crash
- Comprehensive unit tests with 12 test cases covering all fault scenarios

## Task Commits

Each task was committed atomically:

1. **Task 1: Create GPUProcessCoordinator class** - `1065c76` (feat)
2. **Task 2: Add unit tests for GPUProcessCoordinator** - `821c411` (test)

## Files Created/Modified
- `virnucpro/pipeline/gpu_coordinator.py` - SPMD coordinator for multi-GPU worker lifecycle
- `tests/unit/test_gpu_coordinator.py` - Unit tests for coordinator and worker spawning

## Decisions Made

**Use multiprocessing.Process directly instead of mp.spawn:**
- `mp.spawn` kills all workers when any single worker fails
- `multiprocessing.Process` allows independent worker lifecycle
- Surviving workers complete and produce partial results
- Enables fault tolerance required by CONTEXT.md

**Module-level _worker_wrapper for environment variable injection:**
- `multiprocessing.Process` doesn't accept env parameter
- Wrapper function sets CUDA_VISIBLE_DEVICES inside worker process
- Must be module-level for pickle compatibility with spawn context
- Each worker sees device 0, remapped to actual GPU rank

**Per-rank completion status tracking:**
- `wait_for_completion` returns `Dict[int, bool]` mapping rank to success/failure
- Enables parent to identify which workers completed vs timed out vs crashed
- Supports partial result aggregation (Phase 7 Plan 5)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Fixed pickle compatibility for spawn context**
- **Found during:** Task 2 (Test execution)
- **Issue:** Local functions inside test methods can't be pickled by spawn context
- **Fix:** Moved all worker functions to module level (successful_worker, failing_worker, etc.)
- **Files modified:** tests/unit/test_gpu_coordinator.py
- **Verification:** All 12 tests pass
- **Committed in:** 821c411 (Task 2 commit)

**2. [Rule 3 - Blocking] Fixed _worker_wrapper pickle compatibility**
- **Found during:** Task 2 (Test execution)
- **Issue:** Local function defined inside spawn_workers method can't be pickled
- **Fix:** Moved _worker_wrapper to module level in gpu_coordinator.py
- **Files modified:** virnucpro/pipeline/gpu_coordinator.py
- **Verification:** All tests pass with module-level wrapper
- **Committed in:** 821c411 (Task 2 commit)

---

**Total deviations:** 2 auto-fixed (2 blocking)
**Impact on plan:** Both auto-fixes required for spawn context compatibility. No scope creep.

## Issues Encountered

**Multiprocessing spawn context pickle requirements:**
- Problem: Spawn context requires all target functions to be picklable (module-level)
- Solution: Moved worker functions and wrapper to module level
- Impact: All tests pass, CUDA_VISIBLE_DEVICES isolation works correctly

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

**Ready for Phase 7 Plan 5 (HDF5 shard aggregation):**
- GPUProcessCoordinator provides worker spawning and completion tracking
- Per-rank status enables identification of which shards are available
- Fault tolerance allows aggregation of partial results from successful workers

**Ready for Phase 7 Plan 6-8 (End-to-end coordination):**
- Coordinator can spawn arbitrary worker functions with custom args
- Results queue supports push-based progress reporting
- Emergency terminate_all available for cleanup scenarios

**Test coverage:**
- 12 unit tests covering initialization, spawning, CUDA isolation, completion tracking
- Partial failure scenarios validated (one worker fails, others succeed)
- Worker independence verified (spawn context, argument passing)

---
*Phase: 07-multi-gpu-coordination*
*Completed: 2026-02-04*
