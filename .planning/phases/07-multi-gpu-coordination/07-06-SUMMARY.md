---
phase: 07-multi-gpu-coordination
plan: 06
subsystem: multi-gpu
tags: [gpu-worker, async-inference, hdf5, process-coordination]

requires:
  - Phase 5 (AsyncInferenceRunner)
  - Phase 6 (Packing integration)
  - 07-01 (SequenceIndex)
  - 07-02 (IndexBasedDataset)
  - 07-03 (Per-worker logging)
  - 07-04 (GPUProcessCoordinator)

provides:
  - gpu_worker function for single-GPU shard processing
  - Complete integration of all async/packing/sharding components
  - Status reporting protocol via multiprocessing Queue

affects:
  - 07-07 (Coordinator integration - will use gpu_worker as worker function)
  - 07-08 (End-to-end multi-GPU testing)

tech-stack:
  added: []
  patterns:
    - Worker process architecture with status reporting
    - HDF5 shard file format (embeddings + sequence_ids)
    - Empty shard handling for zero-sequence workers

key-files:
  created:
    - virnucpro/pipeline/gpu_worker.py
    - tests/unit/test_gpu_worker.py
  modified: []

decisions:
  - decision: "Worker sets up logging before any operations"
    rationale: "Ensures all worker actions are logged, including early failures"
    impact: "First operation in worker is setup_worker_logging() call"

  - decision: "Handle empty all_embeddings list gracefully"
    rationale: "Worker may receive zero sequences if world_size > total_sequences"
    impact: "Creates empty HDF5 dataset (0, 0) instead of crashing on torch.cat"

  - decision: "Report status via Queue.put() before sys.exit(1)"
    rationale: "Parent needs to know worker failed even after process exits"
    impact: "Exception handler puts failure status then exits with code 1"

metrics:
  duration: "3 min"
  completed: "2026-02-04"
  commits: 2
  files_created: 2
  tests_added: 9
---

# Phase 07 Plan 06: GPU Worker Function Integration Summary

**One-liner:** GPU worker function integrating AsyncInferenceRunner, packing, and index-based sharding with HDF5 output.

## What Was Built

Created `gpu_worker()` function that serves as the entry point for spawned worker processes in multi-GPU coordination. The worker:

1. **Sets up per-worker logging** (first operation before any other work)
2. **Loads sequence index** and gets assigned indices via stride distribution
3. **Loads ESM-2 model** (or DNABERT-S) on CUDA device 0
4. **Creates IndexBasedDataset and async DataLoader** with assigned indices
5. **Runs AsyncInferenceRunner** to process sequences with packing
6. **Saves HDF5 shard** containing embeddings and sequence IDs
7. **Reports status** to parent process via multiprocessing Queue

This is the first plan that integrates **all three major subsystems**:
- **Phase 5:** AsyncInferenceRunner with stream-based GPU I/O
- **Phase 6:** Sequence packing with FlashAttention varlen
- **Phase 7:** Multi-GPU sharding with index-based distribution

## Key Components

### 1. Worker Function (`gpu_worker()`)

**Location:** `virnucpro/pipeline/gpu_worker.py`

**Signature:**
```python
def gpu_worker(
    rank: int,
    world_size: int,
    results_queue: Queue,
    index_path: Path,
    output_dir: Path,
    model_config: Dict[str, Any]
) -> None
```

**Flow:**
```
1. setup_worker_logging(rank, log_dir)         # FIRST - log everything
2. torch.cuda.set_device(0)                     # CUDA_VISIBLE_DEVICES remaps this
3. indices = get_worker_indices(index, rank, N) # Stride distribution
4. model = load_esm2_model(...)                 # Load on GPU
5. dataset = IndexBasedDataset(index, indices)  # Only assigned sequences
6. dataloader = create_async_dataloader(...)    # Async + packing
7. runner = AsyncInferenceRunner(model, device)
8. for result in runner.run(dataloader):        # Process all batches
       all_embeddings.append(result.embeddings)
       all_ids.extend(result.sequence_ids)
9. Save to shard_{rank}.h5                      # HDF5 output
10. results_queue.put({'status': 'complete'})   # Report to parent
```

**CUDA device mapping:**
- Parent sets `CUDA_VISIBLE_DEVICES={rank}` before spawning worker
- Worker sees `device 0`, which maps to physical GPU {rank}
- Example: Worker 2 sees device 0, but it's actually GPU 2

### 2. HDF5 Shard Format

**File naming:** `shard_{rank}.h5`

**Contents:**
- `embeddings`: Float32 array of shape `(num_sequences, hidden_dim)`
- `sequence_ids`: Variable-length string array of sequence IDs

**Empty shard handling:**
- If `all_embeddings` is empty (zero sequences assigned to worker):
  - Create empty dataset with shape `(0, 0)`
  - Still save `sequence_ids` (empty list)
  - Report success with `num_sequences=0`

### 3. Status Reporting Protocol

**Success:**
```python
{
    'rank': 0,
    'status': 'complete',
    'shard_path': '/path/to/shard_0.h5',
    'num_sequences': 12345
}
```

**Failure:**
```python
{
    'rank': 0,
    'status': 'failed',
    'error': 'CUDA out of memory'
}
```

**Note:** Status is put to queue **before** `sys.exit(1)` in exception handler.

## Unit Tests

**Location:** `tests/unit/test_gpu_worker.py` (9 tests, all passing)

### TestGPUWorkerFlow (6 tests)
- `test_worker_sets_up_logging_first`: Verifies logging is first operation
- `test_worker_gets_assigned_indices`: Verifies correct rank/world_size passed
- `test_worker_creates_dataset_with_indices`: Verifies IndexBasedDataset creation
- `test_worker_saves_hdf5_shard`: Verifies shard file creation and format
- `test_worker_reports_success`: Verifies Queue.put() with success status
- `test_worker_reports_failure`: Verifies exception → failed status → sys.exit(1)

### TestWorkerErrorHandling (3 tests)
- `test_model_load_failure`: Model load error reported properly
- `test_inference_failure`: Inference error (e.g., CUDA OOM) reported properly
- `test_shard_save_failure`: Disk write error reported properly

**Mocking strategy:**
- Mock `torch.cuda` to avoid GPU requirement
- Mock `load_esm2_model` to return fake model
- Mock `AsyncInferenceRunner` to yield fake results
- Keep `shard_index`, `logging`, and `IndexBasedDataset` as real implementations

## Decisions Made

### 1. Logging Setup First

**Decision:** Call `setup_worker_logging()` before any other operations.

**Rationale:**
- Early failures (CUDA init, index load, model load) must be logged
- Without logging, worker failures are silent and hard to debug
- Parent process can inspect worker logs after failure

**Implementation:**
```python
# FIRST operation in worker
log_file = setup_worker_logging(rank, log_dir)
logger = logging.getLogger(f"gpu_worker_{rank}")
logger.info(f"Worker {rank}/{world_size} starting")
```

### 2. Empty Embeddings Handling

**Decision:** Handle zero-sequence case gracefully.

**Context:** If `world_size > total_sequences`, some workers get zero sequences.

**Problem:** `torch.cat([], dim=0)` raises `RuntimeError: expected a non-empty list of Tensors`

**Solution:**
```python
if all_embeddings:
    embeddings = torch.cat(all_embeddings, dim=0).numpy()
    f.create_dataset('embeddings', data=embeddings)
else:
    # Empty shard - create empty dataset
    f.create_dataset('embeddings', shape=(0, 0), dtype='float32')
```

**Impact:** Workers can succeed even with zero assigned sequences.

### 3. Status Before Exit

**Decision:** Call `results_queue.put(status)` before `sys.exit(1)`.

**Rationale:**
- Parent needs to know which workers failed
- Parent can decide to aggregate partial results or fail entirely
- Logging alone isn't sufficient (parent needs machine-readable status)

**Implementation:**
```python
except Exception as e:
    logger.exception(f"Worker {rank} failed: {e}")
    results_queue.put({
        'rank': rank,
        'status': 'failed',
        'error': str(e)
    })
    sys.exit(1)  # AFTER putting status
```

## Integration Points

### Depends On (Inputs)

1. **Phase 5 - AsyncInferenceRunner:**
   - `AsyncInferenceRunner(model, device)` for async GPU processing
   - `InferenceResult` dataclass for batch results
   - Stream-based I/O overlap and GPU monitoring

2. **Phase 6 - Packing:**
   - `VarlenCollator` for packed batch creation
   - `create_async_dataloader` with automatic token budget
   - FlashAttention varlen integration in model

3. **Phase 7 - Sharding:**
   - `get_worker_indices(index, rank, world_size)` for stride distribution
   - `IndexBasedDataset(index, indices)` for assigned sequence reading
   - `setup_worker_logging(rank, log_dir)` for per-worker logs

4. **Models:**
   - `load_esm2_model(model_name, device)` from `virnucpro.models.esm2_flash`

### Provides (Outputs)

1. **For 07-07 (Coordinator Integration):**
   - `gpu_worker` function to pass to `GPUProcessCoordinator.spawn_workers()`
   - Status reporting protocol for completion tracking
   - HDF5 shard format for aggregation

2. **For 07-08 (End-to-End Testing):**
   - Complete worker implementation for integration tests
   - Error handling examples for failure injection testing

## Deviations from Plan

### Auto-Fixed Issues

**1. [Rule 1 - Bug] Empty tensor list crashes torch.cat**

- **Found during:** Task 1 (initial implementation)
- **Issue:** When worker gets zero sequences, `all_embeddings = []` and `torch.cat([], dim=0)` raises RuntimeError
- **Fix:** Check `if all_embeddings:` before `torch.cat()`, create empty HDF5 dataset if needed
- **Files modified:** `virnucpro/pipeline/gpu_worker.py`
- **Commit:** 6efeb79

This was discovered during test development when testing edge cases.

## Testing

### Unit Tests

**Command:** `pytest tests/unit/test_gpu_worker.py -v`

**Results:** 9/9 tests passed

**Coverage:**
- Worker setup and logging
- Index loading and dataset creation
- Inference execution
- HDF5 shard saving
- Status reporting (success and failure)
- Error handling (model load, inference, disk write)

### Mocking Strategy

Tests avoid GPU requirements by mocking:
- `torch.cuda.*` → No CUDA needed
- `load_esm2_model` → Returns mock model
- `AsyncInferenceRunner` → Yields fake InferenceResults
- `create_async_dataloader` → Returns mock DataLoader

This enables CPU-only testing on CI/CD without GPU access.

## Next Phase Readiness

### Blockers

None. All components integrated successfully.

### Concerns

**1. Real GPU testing:**
- Unit tests mock everything GPU-related
- Need integration test on actual GPU to verify:
  - CUDA_VISIBLE_DEVICES remapping works
  - Model loading doesn't crash
  - AsyncInferenceRunner handles packed batches correctly
  - HDF5 writing works under GPU context

**Resolution:** Plan 07-08 (End-to-End Testing) will validate on GPU server.

**2. Worker fault tolerance:**
- If one worker crashes, what happens?
- Does GPUProcessCoordinator wait for all workers?
- How are partial results handled?

**Resolution:** Plan 07-07 (Coordinator Integration) will implement completion tracking.

### Ready For

- **07-07:** Coordinator integration (use `gpu_worker` as worker function)
- **07-08:** End-to-end multi-GPU testing on real hardware

## Files Changed

### Created

**1. `virnucpro/pipeline/gpu_worker.py` (187 lines)**
- Worker function with 8-step flow
- Empty embeddings handling
- Status reporting protocol
- Comprehensive logging

**2. `tests/unit/test_gpu_worker.py` (630 lines)**
- 9 unit tests covering all worker flows
- Extensive mocking for CPU-only testing
- Error injection for failure paths

### Modified

None (new module, no changes to existing code).

## Metrics

- **Duration:** 3 minutes
- **Commits:** 2
  - `df7c698`: feat(07-06): create gpu_worker function
  - `6efeb79`: test(07-06): add unit tests for gpu_worker
- **Tests:** 9 added, 9 passing
- **Lines of Code:** ~817 total (187 implementation + 630 tests)

## Lessons Learned

### What Worked Well

1. **Incremental mocking:** Building up mocks one layer at a time made tests easier to debug.

2. **Empty case handling:** Testing edge cases (zero sequences) early caught real bug.

3. **Status protocol design:** Simple dict format makes testing and debugging easy.

### What Could Be Better

1. **Integration testing gap:** Unit tests mock everything GPU-related. Need real GPU tests to validate assumptions about CUDA_VISIBLE_DEVICES remapping.

2. **HDF5 format validation:** Tests verify file creation but don't validate HDF5 schema deeply. Future tests should verify dtype, shape, and readability.

---

**Status:** ✅ Complete - All tasks executed, tests passing, ready for coordinator integration.
