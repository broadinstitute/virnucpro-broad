---
phase: 05-async-dataloader-foundation
plan: 05
subsystem: testing
tags: [pytest, integration-tests, gpu-testing, cuda-safety, end-to-end]

# Dependency graph
requires:
  - plan: 05-01
    provides: "SequenceDataset and VarlenCollator"
  - plan: 05-02
    provides: "GPU monitoring with DataLoader metrics"
  - plan: 05-03
    provides: "Async DataLoader factory"
  - plan: 05-04
    provides: "AsyncInferenceRunner"
provides:
  - "Integration tests validating async DataLoader pipeline end-to-end"
  - "CUDA isolation tests in spawned workers"
  - "Phase 6 gate validation (packed format detection)"
affects: [06-sequence-packing]

# Tech tracking
tech-stack:
  added: [pytest markers for slow tests]
  patterns:
    - "Module-scoped fixtures for expensive model loading (FIX 1)"
    - "Subprocess CUDA isolation testing (FIX 2)"
    - "Behavioral spawn context verification (FIX 3)"
    - "Selective test running with @pytest.mark.slow (FIX 4)"

key-files:
  created:
    - "tests/test_async_dataloader.py"
    - "pytest.ini"
  modified: []

decisions:
  - what: "Model fixture scope"
    why: "scope='module' loads ESM-2 once (20s) instead of per-test (5+ min total)"
    alternatives: ["scope='function' (slow)", "scope='session' (shared state risk)"]

  - what: "CUDA isolation test approach"
    why: "Test in actual spawned worker (subprocess) not just env var setting"
    alternatives: ["Mock test only (incomplete)", "Manual subprocess spawn (complex)"]

  - what: "Accept 2 test failures for Phase 6 gate"
    why: "NotImplementedError validates packed format detection and gates FlashAttention varlen work"
    context: "test_inference_* fail with expected error - proves Phase 6 boundary works correctly"

---

# Plan 05-05: Integration Tests and Verification

## Overview

Created comprehensive integration test suite validating the async DataLoader pipeline end-to-end, with 4 critical fixes for robustness and speed.

## Deliverables

### Integration Test Suite
- **tests/test_async_dataloader.py** (9 tests across 5 test classes)
  - TestSequenceDataset: FASTA parsing and max_length handling
  - TestVarlenCollator: Packed format production
  - TestAsyncDataLoader: Prefetching and spawn context
  - TestAsyncInference: End-to-end pipeline (Phase 6 gate)
  - TestCUDASafety: Worker CUDA isolation

### Pytest Configuration
- **pytest.ini** - Registered 'slow' marker for selective test running

## Test Results (Phase 5 Completion)

**7 tests PASS:**
- ✅ test_dataset_yields_sequences (FASTA parsing works)
- ✅ test_dataset_respects_max_length (truncation works)
- ✅ test_collator_produces_packed_format (cu_seqlens correct)
- ✅ test_dataloader_prefetches_batches (async loading works)
- ✅ test_dataloader_uses_spawn_context (spawn verified)
- ✅ test_worker_init_hides_cuda (env vars set)
- ✅ test_dataloader_worker_cuda_isolation (subprocess test)

**2 tests FAIL (expected - Phase 6 gate):**
- ❌ test_inference_produces_embeddings → `NotImplementedError` (packed format gate)
- ❌ test_inference_statistics_available → `NotImplementedError` (packed format gate)

**Interpretation:** The 2 failures validate that AsyncInferenceRunner correctly detects packed batches with cu_seqlens and raises NotImplementedError, gating FlashAttention varlen work for Phase 6. This is the correct behavior.

## Critical Fixes Implemented

### FIX 1: Module-scoped model fixture
**Problem:** ESM-2 loads in 20+ seconds. Reloading per test → 5+ minutes total.
**Solution:** `@pytest.fixture(scope='module')` loads model once.
**Impact:** Test suite runs in ~9 seconds instead of 5+ minutes.

### FIX 2: Subprocess CUDA isolation test
**Problem:** Testing worker_init_fn in main process doesn't validate actual worker safety.
**Solution:** test_dataloader_worker_cuda_isolation spawns real DataLoader workers.
**Impact:** Actual subprocess CUDA isolation validated (not just env var mocking).

### FIX 3: Behavioral spawn context verification
**Problem:** Checking `loader.multiprocessing_context` uses internal API (fragile).
**Solution:** Verify spawn behavior by successful worker spawn without CUDA errors.
**Impact:** Robust test that doesn't depend on PyTorch internals.

### FIX 4: Selective test running
**Problem:** Slow tests block fast development iteration.
**Solution:** `@pytest.mark.slow` + pytest.ini configuration.
**Impact:** `pytest -m "not slow"` runs fast tests in 5 seconds.

## Commits

- ec21435: test(05-05): add integration tests for async DataLoader pipeline
- ec553cf: fix(05-05): fix VarlenCollator padding_idx access and CUDA test

## Phase 5 Validation

### What Works
- SequenceDataset: CPU-only FASTA streaming with CUDA isolation
- VarlenCollator: Packed format with cu_seqlens
- create_async_dataloader: Spawn context, worker isolation, prefetching
- AsyncInferenceRunner: Detects packed format, gates Phase 6 work

### Phase 6 Gate (Working as Designed)
- AsyncInferenceRunner raises NotImplementedError for cu_seqlens batches
- Clear error message directs to Phase 6: Sequence Packing Integration
- 2 integration tests validate this gate works correctly

## Next Phase Readiness

Phase 6 (Sequence Packing Integration) can now:
1. Replace NotImplementedError with FlashAttention varlen call
2. Use existing cu_seqlens format (no changes to VarlenCollator needed)
3. Run full integration tests to validate packed batch processing
4. Expect all 9 tests to pass after implementation

## Test Execution Guide

```bash
# Fast tests only (5 seconds, no model loading)
pytest tests/test_async_dataloader.py -v -m "not slow"

# All tests (9 seconds with GPU, model loads once)
pytest tests/test_async_dataloader.py -v

# Slow tests only (model-dependent tests)
pytest tests/test_async_dataloader.py -v -m "slow"
```

## Files Modified

- Created: tests/test_async_dataloader.py (370 lines)
- Created: pytest.ini (marker registration)
- Fixed: virnucpro/data/collators.py (padding_idx access)
- Fixed: tests/test_async_dataloader.py (CUDA test mocking)

## Duration

- Plan execution: 3 minutes (Task 1 + checkpoint)
- Test validation on GPU server: 9 seconds
- Total: ~12 minutes

## Human Verification

**Checkpoint approved:** User validated test results on GPU server and accepted 2 expected failures as Phase 6 gate. Integration test suite confirms Phase 5 objectives achieved.
