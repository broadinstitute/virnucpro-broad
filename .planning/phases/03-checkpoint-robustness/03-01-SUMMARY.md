---
phase: 03-checkpoint-robustness
plan: 01
subsystem: checkpoint
tags: [pytorch, checkpoint, validation, atomic-write, corruption-prevention]

# Dependency graph
requires:
  - phase: 02.1-parallel-embedding-merge
    provides: atomic write pattern in merge_features for checkpoints
provides:
  - Checkpoint validation utilities with multi-level checks
  - Centralized atomic_save function for all PyTorch checkpoints
  - CheckpointError exception for checkpoint-specific failures
  - Validation distinguishes 'corrupted' vs 'incompatible' checkpoints

affects: [04-flash-attention-2, future-phases-using-checkpoints]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - Multi-level checkpoint validation (file size → ZIP → load → keys)
    - Atomic write with temp-then-rename using Path.replace()
    - Checkpoint validation after save for write-time corruption detection
    - CheckpointError exception hierarchy for error categorization

key-files:
  created:
    - virnucpro/core/checkpoint_validation.py
  modified:
    - virnucpro/core/checkpoint.py
    - virnucpro/pipeline/features.py

key-decisions:
  - "validate-after-save-optional: Validation after save is optional (default: True) to allow disabling for performance-critical paths"
  - "feature-checkpoints-unvalidated: Feature extraction checkpoints skip validation (validate_after_save=False) to avoid overhead on large files"
  - "path-replace-not-rename: Use Path.replace() instead of Path.rename() for atomic overwrite on all platforms"
  - "centralized-atomic-save: Consolidate atomic write pattern in checkpoint.py instead of duplicating across modules"

patterns-established:
  - "Multi-level validation: validate_checkpoint() with 4 levels (file size, ZIP format, PyTorch load, required keys)"
  - "Atomic save pattern: atomic_save() wraps torch.save with temp-then-rename and optional validation"
  - "Validation flags: skip_validation parameter for --skip-checkpoint-validation CLI flag support"
  - "Error categorization: distinguish_error_type() separates 'corrupted' vs 'incompatible' errors"

# Metrics
duration: 3.5min
completed: 2026-01-23
---

# Phase 03 Plan 01: Checkpoint Validation & Atomic Write Summary

**Multi-level checkpoint validation with atomic write pattern centralized for all PyTorch saves, preventing corruption and detecting invalid files before pipeline failures**

## Performance

- **Duration:** 3.5 min (207 seconds)
- **Started:** 2026-01-23T22:14:03Z
- **Completed:** 2026-01-23T22:17:30Z
- **Tasks:** 3
- **Files modified:** 3

## Accomplishments

- Created checkpoint validation module with 4-level validation (file size → ZIP → load → keys)
- Centralized atomic write pattern with validation in checkpoint.py
- Updated all 4 torch.save calls in features.py to use atomic_save
- Established CheckpointError exception hierarchy for error categorization
- Validation distinguishes between corrupted (broken file) and incompatible (wrong version) errors

## Task Commits

Each task was committed atomically:

1. **Task 1: Create checkpoint validation utilities** - `05ec2fa` (feat)
   - CheckpointError exception class
   - validate_checkpoint() with multi-level checks
   - validate_checkpoint_batch() for batch validation
   - load_checkpoint_with_validation() with version/status checks
   - get_checkpoint_info() for quick metadata inspection

2. **Task 2: Extend atomic write pattern to all checkpoints** - `aae8c3e` (feat)
   - atomic_save() function with temp-then-rename pattern
   - load_checkpoint_safe() with pre-load validation
   - Import checkpoint_validation utilities
   - Support for --skip-checkpoint-validation flag

3. **Task 3: Update features.py to use centralized atomic save** - `968cc88` (feat)
   - Replaced all 4 torch.save calls with atomic_save
   - extract_dnabert_features uses atomic_save
   - extract_esm_features uses atomic_save
   - merge_features (both normal and empty edge case) uses atomic_save
   - Removed manual temp-then-rename pattern (17 lines → 1 function call)

## Files Created/Modified

- `virnucpro/core/checkpoint_validation.py` - Multi-level validation utilities for PyTorch checkpoints
  - validate_checkpoint() - 4-level validation (file size, ZIP format, PyTorch load, required keys)
  - CheckpointError exception - checkpoint-specific error with error_type categorization
  - validate_checkpoint_batch() - batch validation with valid/failed separation
  - load_checkpoint_with_validation() - comprehensive load with version/status checks
  - get_checkpoint_info() - quick metadata inspection without full validation
  - distinguish_error_type() - categorize errors as 'corrupted' vs 'incompatible'

- `virnucpro/core/checkpoint.py` - Extended with atomic write utilities
  - atomic_save() - centralized atomic write with temp-then-rename and optional validation
  - load_checkpoint_safe() - load with pre-load validation and CheckpointError handling
  - Import validation utilities from checkpoint_validation module

- `virnucpro/pipeline/features.py` - Updated to use centralized atomic save
  - All 4 torch.save calls replaced with atomic_save
  - extract_dnabert_features, extract_esm_features, merge_features use atomic write
  - Removed manual temp-then-rename pattern (cleaner code)

## Decisions Made

**1. Feature checkpoints skip validation (validate_after_save=False)**
- **Rationale:** Feature extraction creates large checkpoint files (multi-GB for ESM-2). Full validation (loading checkpoint after save) would add significant overhead. File size and ZIP format checks are sufficient for write corruption detection.
- **Impact:** Fast checkpoint saves without sacrificing atomic write protection

**2. Use Path.replace() instead of Path.rename()**
- **Rationale:** Path.rename() fails on Windows if target exists. Path.replace() guarantees atomic overwrite on all platforms. Research confirmed this is the standard (PyTorch Lightning, PyTorch-Ignite).
- **Impact:** Cross-platform compatibility for atomic writes

**3. Centralize atomic save in checkpoint.py**
- **Rationale:** Manual temp-then-rename pattern was duplicated in features.py. Centralizing in checkpoint.py provides single source of truth, easier maintenance, and consistent validation options.
- **Impact:** Reduced code duplication (17 lines → 1 function call per save)

**4. Validation is optional in atomic_save**
- **Rationale:** Different use cases need different validation levels. Feature extraction needs fast saves, pipeline state needs full validation. Making validation optional via validate_after_save parameter provides flexibility.
- **Impact:** Performance optimization for large checkpoint files while maintaining safety for critical checkpoints

## Deviations from Plan

None - plan executed exactly as written. All tasks completed as specified with no auto-fixes, blocking issues, or architectural changes required.

## Issues Encountered

None - execution proceeded smoothly with expected imports and dependencies already in place (PyTorch, zipfile, pathlib all stdlib/existing).

## User Setup Required

None - no external service configuration required. All functionality uses Python stdlib and existing PyTorch dependency.

## Next Phase Readiness

**Ready for Phase 4 (FlashAttention-2 Integration)**

Checkpoint system now provides:
- Atomic write protection against corruption
- Multi-level validation to detect invalid checkpoints before use
- Clear error categorization (corrupted vs incompatible)
- Foundation for --skip-checkpoint-validation CLI flag (to be wired in future CLI work)
- Detailed diagnostic logging showing exactly what failed

**No blockers or concerns:**
- All checkpoint saves now protected by atomic write
- Validation utilities ready for integration with checkpoint resume logic
- Error types clearly distinguished for user-facing error messages
- Pattern established for future checkpoint operations

**Checkpoint corruption prevention:**
The atomic write pattern (temp-then-rename with validation) prevents the "8+ hours into resume attempt, discovers corrupted checkpoint" failure mode that Phase 3 research identified as critical for long-running jobs.

---
*Phase: 03-checkpoint-robustness*
*Completed: 2026-01-23*
