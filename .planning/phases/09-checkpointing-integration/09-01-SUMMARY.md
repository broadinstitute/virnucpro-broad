---
phase: 09-checkpointing-integration
plan: 01
subsystem: checkpointing
tags: [checkpoint, async-io, validation, resume, corruption-handling]
requires: [virnucpro.core.checkpoint, torch, concurrent.futures]
provides:
  - CheckpointTrigger for adaptive checkpoint timing
  - AsyncCheckpointWriter for non-blocking GPU-safe writes
  - validate_checkpoint_pt for .pt file integrity validation
  - validate_checkpoint_metadata for config compatibility checks
  - resume_from_checkpoints for loading prior progress with corruption handling
affects: [09-02, 09-03, 09-04, 09-05, 09-06]
tech-stack:
  added: []
  patterns: [adaptive-trigger, async-checkpoint-write, atomic-rename-done-marker, corruption-recovery]
key-files:
  created: [virnucpro/pipeline/checkpoint_writer.py]
  modified: []
decisions:
  - id: checkpoint-format-pt-not-hdf5
    choice: Use PyTorch .pt format instead of HDF5 for incremental checkpoints
    rationale: Consistent with Phase 3 atomic_save pattern, avoids HDF5 append corruption risks, proven serialization format
    context: Plan specified .pt format explicitly after research showed HDF5 append mode can corrupt on crashes
  - id: env-var-precedence
    choice: VIRNUCPRO_VIRAL_CHECKPOINT_MODE overrides defaults but not explicit args
    rationale: Enables viral workload tuning without code changes while respecting explicit configuration
    context: Env var sets 5K seq / 180s for viral (400AA+) sequences, but explicit constructor args always win
  - id: corruption-returns-requeue-list
    choice: resume_from_checkpoints returns corrupted sequence IDs for caller requeue
    rationale: Enables idempotent recovery - caller reprocesses lost sequences, no full pipeline restart
    context: 4-tuple return (ids, embeddings, resume_batch, corrupted_ids) supports partial failure recovery
  - id: manifest-optional-filesystem-truth
    choice: Manifest validation logs warnings but doesn't fail, filesystem is source of truth
    rationale: Checkpoints are authoritative, manifest is metadata tracking - don't block resume on manifest issues
    context: Per-shard checkpoints can be scanned independently, manifest helps coordination but isn't required
metrics:
  duration: 213s
  completed: 2026-02-06
---

# Phase 09 Plan 01: Checkpoint Foundation Summary

**One-liner:** Adaptive checkpoint trigger, async .pt writer with GPU-to-CPU transfer, 4-level validation, and resume with corrupted sequence requeue.

## What Was Built

Created `virnucpro/pipeline/checkpoint_writer.py` (683 lines) with five core components for Phase 9 checkpointing:

1. **CheckpointTrigger** - Fires on sequence count OR time threshold (whichever first)
   - Default: 10K sequences OR 5 minutes
   - Emergency override (>600s) forces checkpoint even mid-batch to prevent unbounded work loss
   - VIRNUCPRO_VIRAL_CHECKPOINT_MODE env var enables 5K seq / 180s for viral (400AA+) workloads
   - Env var overrides defaults but respects explicit constructor arguments (precedence documented)

2. **AsyncCheckpointWriter** - Background thread I/O to prevent GPU stalls
   - GPU tensors transferred to CPU via `.cpu().numpy()` before async submission (prevents CUDA context issues)
   - Numpy arrays copied to prevent race conditions when buffers reused
   - Atomic temp-then-rename with .done markers (consistent with Phase 3 pattern)
   - Uses `torch.save()` for .pt format instead of HDF5 (avoids append corruption, proven serialization)
   - `wait_all()` propagates errors from background threads via `.result()` (Issue 7 - no silent swallowing)
   - Optional manifest + rank parameters for coordinator integration (Issue 6)
   - `has_pending()` with lock protection for thread safety (Issue 12)

3. **validate_checkpoint_pt** - Multi-level validation for .pt checkpoint files
   - Level 1: File exists and size > 0
   - Level 2: .done marker exists (uses `has_done_marker` from `virnucpro.core.checkpoint`)
   - Level 3: `torch.load` succeeds with required keys ('embeddings', 'sequence_ids')
   - Level 4: Shape consistency (`len(embeddings) == len(sequence_ids)`)
   - Returns `(is_valid, error_description)` for caller diagnostics

4. **validate_checkpoint_metadata** - Config compatibility checks
   - Returns warnings (not errors) for caller discretion
   - Compares `model_config_hash` and `packing_enabled` settings against current config
   - Verifies required metadata keys: `batch_idx`, `num_sequences`, `timestamp`
   - Returns `(is_valid, warnings)` where `is_valid=False` if any warnings present

5. **resume_from_checkpoints** - Load valid checkpoints, handle corruption gracefully
   - Returns 4-tuple: `(sequence_ids, embeddings, resume_batch_idx, corrupted_sequence_ids)`
   - Loads checkpoints sorted by batch number using regex `batch_(\d+)\.pt` (Issue 8 - robust parsing)
   - Stops at first corruption, calls `remove_done_marker` to invalidate corrupted checkpoint
   - Collects sequence IDs from checkpoints AFTER corruption point for requeue by caller (Issue 3)
   - Concatenates valid embeddings with `np.concatenate`
   - Optional manifest validation (filesystem checkpoints are source of truth, logs warnings on mismatch)
   - Handles force_restart flag to ignore checkpoints and start fresh

## Integration Points

**Imports from existing code:**
- `virnucpro.core.checkpoint`: `has_done_marker`, `remove_done_marker` for .done marker management
- `torch`: `torch.save` and `torch.load` for .pt checkpoint serialization (consistent with Phase 3)
- `concurrent.futures`: `ThreadPoolExecutor` and `Future` for async I/O
- `numpy`: `np.concatenate` for embedding concatenation, `np.ndarray` for type checking

**Used by downstream plans:**
- Plan 09-02 (AsyncInferenceRunner integration): Uses `CheckpointTrigger` and `AsyncCheckpointWriter`
- Plan 09-03 (GPU worker integration): Uses `resume_from_checkpoints` for worker restart
- Plan 09-04 (Coordinator manifest): Extends `AsyncCheckpointWriter.manifest` parameter
- Plan 09-05 (Restart logic): Uses `corrupted_sequence_ids` return for requeue
- Plan 09-06 (CLI flags): Exposes `force_restart` and trigger thresholds

**Architecture pattern:**
- Checkpoint trigger decoupled from write logic (single responsibility)
- Async writer handles data safety (GPU->CPU, copy) before background thread submission
- Validation is multi-level (quick checks first, expensive loading last)
- Resume is corruption-tolerant (partial success, requeue lost work)

## Decisions Made

### 1. PyTorch .pt Format Instead of HDF5
**Issue:** Plan initially referenced HDF5 in research docs, but specified .pt in must_haves
**Decision:** Use `torch.save()` with .pt format for all incremental checkpoints
**Rationale:**
- Consistent with Phase 3 `atomic_save()` pattern
- Avoids HDF5 append mode corruption risks (crash during write can corrupt entire file)
- Proven PyTorch serialization format used throughout codebase
- Simpler implementation (no h5py dependency, just torch)

### 2. Env Var Precedence (VIRNUCPRO_VIRAL_CHECKPOINT_MODE)
**Issue:** Viral workloads (400AA+ sequences) need more frequent checkpoints without code changes
**Decision:** Env var overrides default thresholds but not explicit constructor arguments
**Rationale:**
- Enables tuning for viral workloads via environment variable
- Respects explicit configuration (if user passes seq_threshold=2000, env var doesn't override)
- Documented in CheckpointTrigger docstring: "Env var overrides default thresholds but not explicit arguments"
- Precedence: explicit args > env var > defaults

### 3. Corrupted Sequence IDs Return (4-tuple)
**Issue:** How to handle sequences from checkpoints after corruption point?
**Decision:** Return 4-tuple `(ids, embeddings, resume_batch_idx, corrupted_sequence_ids)`
**Rationale:**
- Enables idempotent recovery - caller reprocesses lost sequences
- No full pipeline restart required for single corrupted checkpoint
- Treats checkpoints as completion markers, not fragile state snapshots
- Supports partial failure recovery (Issue 3)

### 4. Manifest Validation Optional (Filesystem Truth)
**Issue:** Should manifest mismatches block resume?
**Decision:** Manifest validation logs warnings but doesn't fail, filesystem checkpoints are source of truth
**Rationale:**
- Per-shard checkpoints can be scanned independently
- Manifest helps coordination but isn't required for correctness
- Warns user about mismatches (helpful diagnostics) but doesn't prevent resume
- Aligns with "checkpoints are authoritative" principle

## Testing Performed

**Verification approach:** Functional tests for all five components

### CheckpointTrigger
- Sequence threshold trigger at 100 sequences (50 + 60 = 110 total)
- Time threshold trigger after 0.5s delay
- Env var override: VIRNUCPRO_VIRAL_CHECKPOINT_MODE sets 5K seq / 180s
- Explicit args override env var (seq_threshold=2000 beats env var)
- Reset clears both sequence counter and timer

### AsyncCheckpointWriter
- torch.Tensor input: `.cpu().numpy()` transfer before async submission
- np.ndarray input: `.copy()` to prevent race conditions
- Atomic write creates checkpoint + .done marker
- `wait_all()` blocks until completion
- `has_pending()` returns False after wait_all()
- Shutdown waits for pending writes

### validate_checkpoint_pt
- Valid checkpoint with .done marker returns `(True, "")`
- Missing .done marker returns `(False, "Missing .done marker: ...")`
- Empty file (0 bytes) returns `(False, "empty (0 bytes)")`
- All 4 validation levels verified

### validate_checkpoint_metadata
- Compatible metadata (no mismatches) returns `(True, [])`
- Mismatched hash + packing returns `(False, [warning1, warning2])`
- Warnings contain descriptive messages

### resume_from_checkpoints
- Loads 2 checkpoints: 10 + 5 = 15 sequences total
- Concatenates to (15, 128) embeddings array
- Returns resume_batch_idx=2 (after batch 0, 1)
- No corruption detected: corrupted_sequence_ids=[]
- force_restart=True returns ([], None, 0, [])

**Results:** All tests passed. Components importable and functional.

## Deviations from Plan

None - plan executed exactly as written.

All must_haves implemented:
- ✓ Checkpoint trigger fires at sequence count OR time threshold (whichever first)
- ✓ Async checkpoint writes do not block GPU inference loop
- ✓ Checkpoint files use atomic temp-then-rename with .done markers
- ✓ Checkpoint validation detects corruption (size, .done marker, loadability, shape)
- ✓ GPU tensors are transferred to CPU before async write to prevent CUDA context issues
- ✓ Resume returns corrupted sequence IDs for requeue by caller
- ✓ Async write failures propagate via wait_all() instead of being silently swallowed

All key_links satisfied:
- ✓ Uses torch.save/torch.load for .pt checkpoint format (consistent with Phase 3)
- ✓ Uses ThreadPoolExecutor for background thread async I/O
- ✓ Imports has_done_marker and remove_done_marker from virnucpro.core.checkpoint

Artifact meets min_lines requirement: 683 lines (requirement: 250+)

## Files Changed

**Created:**
- `virnucpro/pipeline/checkpoint_writer.py` (683 lines)
  - CheckpointTrigger class (94 lines)
  - AsyncCheckpointWriter class (179 lines)
  - validate_checkpoint_pt function (57 lines)
  - validate_checkpoint_metadata function (44 lines)
  - resume_from_checkpoints function (165 lines)
  - _validate_manifest helper (54 lines)
  - Module docstring + imports (90 lines)

**Modified:**
- None

## Next Phase Readiness

**Blockers:** None

**Phase 9 dependencies satisfied:**
- Plan 09-02 can integrate CheckpointTrigger and AsyncCheckpointWriter into AsyncInferenceRunner
- Plan 09-03 can use resume_from_checkpoints for GPU worker restart
- Plan 09-04 can implement CheckpointManifest and integrate with AsyncCheckpointWriter.manifest
- Plan 09-05 can use corrupted_sequence_ids for requeue logic
- Plan 09-06 can expose force_restart and trigger thresholds in CLI

**Open questions:** None

**Recommendations:**
- Next plan (09-02) should integrate trigger/writer into AsyncInferenceRunner.run() loop
- Consider adding unit tests for edge cases (e.g., emergency override, regex mismatch handling)
- Validate manifest integration (Issue 6) when Plan 09-04 implements CheckpointManifest

## Performance Notes

**Execution time:** 213 seconds (3.6 minutes)
- File creation: 683 lines of production code
- Testing: Functional verification of all five components
- No performance benchmarks yet (async I/O benefits measured in Plan 09-02)

**Expected performance impact:**
- Async writes should reduce checkpoint overhead from ~1-5s (sync) to <100ms (async submit)
- GPU continues inference while background thread handles I/O
- Actual measurements in Plan 09-02 integration testing

**Memory considerations:**
- AsyncCheckpointWriter copies embeddings before async submission (2x memory briefly)
- For 10K sequences × 1280 dim × FP32 = ~50MB per checkpoint copy (acceptable overhead)
- ThreadPoolExecutor max_workers=1 limits concurrent checkpoint writes to 1 (sequential I/O)

---

**Status:** ✓ Complete - All components implemented and tested
**Commit:** 6591cfa
**Files:** virnucpro/pipeline/checkpoint_writer.py (+683)
**Tests:** All functional tests passed
