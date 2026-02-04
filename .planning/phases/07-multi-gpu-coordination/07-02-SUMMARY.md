---
phase: 07-multi-gpu-coordination
plan: 02
subsystem: data
tags: [index-based-dataset, byte-offset-seeking, multi-gpu-sharding, fasta-parsing]

# Dependency graph
requires:
  - phase: 07-multi-gpu-coordination
    plan: 01
    provides: SequenceIndex with stride distribution and byte-offset metadata
provides:
  - IndexBasedDataset class for reading sequences by byte offset
  - Stride distribution support for balanced multi-GPU work distribution
  - CUDA isolation validation for worker processes
  - Integration with VarlenCollator and DataLoader
affects: [07-03, 07-04, 07-05, 07-06]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Index-based iteration with byte-offset seeking for random access"
    - "File grouping optimization for efficient I/O"
    - "CUDA validation pattern copied from SequenceDataset"

key-files:
  created:
    - tests/unit/test_sequence_dataset.py
  modified:
    - virnucpro/data/sequence_dataset.py

key-decisions:
  - "Group indices by file_path before reading to minimize file open/close operations"
  - "Read all sequences into memory dict then yield in index order to preserve length-sorted ordering"
  - "Copy _validate_cuda_isolation from SequenceDataset (acceptable duplication for now)"
  - "__len__ returns len(indices) to enable DataLoader progress tracking"

patterns-established:
  - "Index-based dataset pattern: load index metadata, group by file, seek to offsets, yield in index order"
  - "Byte-offset seeking: seek to header line, skip header, read until next '>' or EOF"

# Metrics
duration: 3min
completed: 2026-02-04
---

# Phase 07 Plan 02: IndexBasedDataset for Byte-Offset Sequence Reading Summary

**Index-based dataset reads sequences by byte offset with stride distribution for balanced multi-GPU workload**

## Performance

- **Duration:** 3 min
- **Started:** 2026-02-04T19:45:53Z
- **Completed:** 2026-02-04T19:49:07Z
- **Tasks:** 2
- **Files modified:** 2 (1 created, 1 modified)

## Accomplishments
- IndexBasedDataset reads sequences by byte offset from pre-built sequence index
- Sequences yielded in index order (length-sorted for optimal FFD packing)
- File grouping optimization minimizes I/O overhead
- CUDA isolation validated in worker processes
- Full integration with VarlenCollator and DataLoader

## Task Commits

Each task was committed atomically:

1. **Task 1: Add IndexBasedDataset to sequence_dataset.py** - `03fa196` (feat)
2. **Task 2: Add unit tests for IndexBasedDataset** - `55f2215` (test)

## Files Created/Modified
- `virnucpro/data/sequence_dataset.py` - Added IndexBasedDataset class with byte-offset seeking
- `tests/unit/test_sequence_dataset.py` - Created comprehensive unit tests (9 tests, all passing)

## Decisions Made

1. **File grouping optimization**: Group indices by file_path before reading to minimize file operations. Improves I/O efficiency when processing large indices across multiple FASTA files.

2. **Memory-buffered ordering**: Read all assigned sequences into memory dict, then yield in index order. Ensures length-sorted order is preserved (critical for FFD packing efficiency). Trade-off: O(N) memory for assigned sequences, acceptable for worker subsets.

3. **CUDA validation duplication**: Copied `_validate_cuda_isolation` method from SequenceDataset to IndexBasedDataset. Acceptable code duplication for now - both classes need identical safety checks. Future refactor could extract to module-level function.

4. **__len__ implementation**: Return len(indices) to enable DataLoader progress tracking and iteration estimation. Required for proper DataLoader behavior with iterable datasets.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None - implementation followed existing patterns from SequenceDataset and shard_index module.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

Ready for next phase:
- IndexBasedDataset provides index-based iteration for workers
- Integration with VarlenCollator verified via unit tests
- CUDA isolation validated for multi-process safety
- __len__ enables DataLoader progress tracking

Next steps:
- Plan 07-03: Per-worker logging infrastructure
- Plan 07-04: Single-process-per-GPU worker implementation
- Plan 07-05: HDF5 aggregation for multi-GPU embeddings

---
*Phase: 07-multi-gpu-coordination*
*Completed: 2026-02-04*
