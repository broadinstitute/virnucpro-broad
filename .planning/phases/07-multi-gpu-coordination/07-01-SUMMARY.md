---
phase: 07-multi-gpu-coordination
plan: 01
subsystem: data
tags: [multi-gpu, sharding, index, stride-distribution, fasta, caching]

# Dependency graph
requires:
  - phase: 06-sequence-packing-integration
    provides: "FFD packing algorithm requiring length-sorted sequences"
provides:
  - "SequenceIndex class with create/load/shard methods"
  - "Mtime-based cache validation for index staleness detection"
  - "Stride-based worker distribution for balanced token load"
  - "Byte-offset tracking for random FASTA access"
affects: [07-02, 07-04, 07-multi-gpu-coordination]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Index-based sharding with stride distribution [rank, rank+N, rank+2N...]"
    - "Mtime-based cache invalidation for FASTA index files"
    - "Descending length sort for FFD packing efficiency"

key-files:
  created:
    - virnucpro/data/shard_index.py
    - tests/unit/test_shard_index.py
  modified:
    - virnucpro/data/__init__.py

key-decisions:
  - "Index sorted by length descending for FFD packing efficiency across all GPUs"
  - "Stride distribution ensures balanced length distribution per worker"
  - "JSON format for human-readable, debuggable index files"
  - "Byte-offset tracking enables future random access without full FASTA scan"

patterns-established:
  - "SequenceEntry dataclass: id/length/file_path/byte_offset for metadata"
  - "create_sequence_index: cached FASTA parsing with mtime validation"
  - "get_worker_indices: stride slicing for deterministic work assignment"

# Metrics
duration: 4min
completed: 2026-02-04
---

# Phase 07 Plan 01: SequenceIndex with Stride Distribution and Caching Summary

**JSON-based sequence index with descending length sort and stride distribution for balanced multi-GPU sharding**

## Performance

- **Duration:** 4 min
- **Started:** 2026-02-04T19:36:19Z
- **Completed:** 2026-02-04T19:40:03Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments

- Sequence index with metadata (id, length, file_path, byte_offset) for all sequences
- Index sorted by length descending to maximize FFD packing efficiency across GPUs
- Mtime-based cache validation detects stale index when FASTA files modified
- Stride distribution [rank::world_size] ensures balanced token load per worker
- Comprehensive test suite (15 tests) validates index creation, caching, and sharding

## Task Commits

Each task was committed atomically:

1. **Task 1: Create SequenceIndex class** - `d946219` (feat)
2. **Task 2: Add unit tests** - `12859b7` (test)

**Module exports:** `6bb43d2` (chore)

## Files Created/Modified

- `virnucpro/data/shard_index.py` - SequenceIndex class with create/load/shard methods
- `tests/unit/test_shard_index.py` - 15 comprehensive tests for index and sharding
- `virnucpro/data/__init__.py` - Export SequenceEntry, create_sequence_index, get_worker_indices, load_sequence_index

## Decisions Made

**Index format - JSON over Pickle:**
- JSON is human-readable and debuggable for ~6M sequences (~200MB file)
- Pickle would be faster but opaque - index contains simple metadata
- JSON allows manual inspection and editing if needed

**Descending length sort:**
- Critical for FFD (First-Fit Decreasing) packing efficiency
- Global sort across all files ensures all workers get representative length distribution
- Stride distribution on sorted index mixes long and short sequences per worker

**Byte-offset tracking:**
- Enables future random access to sequences without full FASTA scan
- Recorded during index creation with minimal overhead
- Points to header line ('>') for each sequence

**Mtime-based cache validation:**
- Simple staleness detection using file modification times
- Compares max(fasta_mtime) > cached_mtime for invalidation
- Rebuilds index automatically when FASTA files change

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

**Ready for Plan 02 (IndexBasedDataset):**
- Index creation infrastructure complete
- Stride distribution validated for balanced token load
- Cache validation ensures fresh index on FASTA changes

**Ready for Plan 04 (GPUProcessCoordinator):**
- get_worker_indices provides deterministic work assignment
- Logged distribution metrics (sequences/tokens per worker)
- All workers can independently load index and get assigned indices

**Validated:**
- Index sorted by length descending (verified in tests)
- Stride distribution provides <10% token deviation across 4 workers (verified in tests)
- Cache invalidation detects FASTA modifications (verified in tests)
- All 15 unit tests pass

---
*Phase: 07-multi-gpu-coordination*
*Completed: 2026-02-04*
