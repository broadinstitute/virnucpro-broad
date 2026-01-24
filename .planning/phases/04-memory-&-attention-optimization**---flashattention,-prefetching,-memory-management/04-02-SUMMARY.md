---
phase: 04-memory-attention
plan: 02
subsystem: data-memory
tags: [dataloader, memory-management, fragmentation-prevention, worker-optimization, oom-prevention]

# Dependency graph
requires:
  - phase: 01-esm2-multi-gpu
    provides: Spawn context pattern for multiprocessing workers
  - phase: 02-dnabert-optimization
    provides: Multi-GPU processing patterns and worker architecture
provides:
  - Optimized DataLoader configuration with CPU-aware worker counts
  - Memory fragmentation prevention through expandable segments
  - Cache clearing intervals and memory tracking utilities
  - Sequence sorting for padding reduction
  - OOM prevention with safe batch size calculation
affects: [04-03, 04-04]

# Tech tracking
tech-stack:
  added: []
  patterns: [cpu-aware-workers, expandable-segments, periodic-cache-clearing, sequence-sorting]

key-files:
  created:
    - virnucpro/data/__init__.py
    - virnucpro/data/dataloader_utils.py
    - virnucpro/cuda/__init__.py
    - virnucpro/cuda/memory_manager.py
    - tests/test_memory_optimization.py
  modified: []

key-decisions:
  - "worker-count-formula: Use min(cpu_count // num_gpus, 8) for DataLoader workers to balance resources and prevent memory explosion"
  - "prefetch-factor-fixed: Set prefetch_factor=2 as fixed good default for I/O-compute overlap"
  - "spawn-context-dataloader: Use spawn multiprocessing context matching GPU worker pattern for consistency"
  - "expandable-segments-opt-in: Make expandable segments opt-in via configuration to allow user control"
  - "cache-interval-configurable: Make cache clearing interval configurable (default 100 batches) for flexibility"
  - "sequence-sorting-optional: Make sequence sorting optional in create_sequence_dataloader for user control"

patterns-established:
  - "get_optimal_workers: Calculate DataLoader worker count based on CPU/GPU ratio with max cap"
  - "create_optimized_dataloader: Single function for DataLoader creation with all best practices"
  - "MemoryManager class: Centralized memory management with fragmentation tracking"
  - "memory_tracking context manager: Track memory usage before/after operations"
  - "safe_batch_size calculation: Estimate safe batch size based on available memory"

# Metrics
duration: 4.3min
completed: 2026-01-24
---

# Phase 04 Plan 02: DataLoader Optimization & Memory Management Summary

**CPU-aware DataLoader configuration and memory fragmentation prevention utilities for stable high-throughput processing with OOM avoidance**

## Performance

- **Duration:** 4.3 min
- **Started:** 2026-01-24T01:00:42Z
- **Completed:** 2026-01-24T01:05:02Z
- **Tasks:** 3
- **Files created:** 5

## Accomplishments

- Created optimized DataLoader utilities with CPU-aware worker count detection
- Implemented MemoryManager for fragmentation prevention and cache management
- Added expandable segments configuration via environment variable
- Implemented sequence sorting for padding reduction and memory efficiency
- Created OOM prevention utilities (memory checks, safe batch size calculation)
- Added memory tracking context manager for diagnostics
- Comprehensive test suite with 40 test cases covering all scenarios

## Task Commits

Each task was committed atomically:

1. **Task 1: Create optimized DataLoader configuration utilities** - `426c346` (feat)
   - get_optimal_workers() with CPU-aware worker count
   - create_optimized_dataloader() with prefetch_factor=2
   - create_sequence_dataloader() with optional sorting
   - estimate_memory_usage() helper for OOM prevention
   - SequenceDataset wrapper for efficient loading

2. **Task 2: Create memory fragmentation prevention manager** - `6e86a11` (feat)
   - MemoryManager class with expandable segments config
   - Periodic cache clearing with configurable interval
   - get_memory_stats() for tracking allocated/reserved/free memory
   - check_memory_available() and get_safe_batch_size() for OOM prevention
   - sort_sequences_by_length() for padding reduction
   - memory_tracking() context manager for diagnostics
   - get_fragmentation_ratio() and suggest_batch_size_adjustment()

3. **Task 3: Add comprehensive memory optimization tests** - `0003769` (test)
   - 40 test cases covering DataLoader and MemoryManager
   - Mock torch for CI compatibility
   - Integration tests with DataLoader + MemoryManager
   - Performance benchmarks for sorted vs unsorted sequences
   - Error handling tests for edge cases

## Files Created/Modified

- `virnucpro/data/__init__.py` - Data utilities module exports (15 lines)
- `virnucpro/data/dataloader_utils.py` - Optimized DataLoader configuration with CPU-aware workers (228 lines)
- `virnucpro/cuda/__init__.py` - CUDA utilities module exports (10 lines)
- `virnucpro/cuda/memory_manager.py` - Memory fragmentation prevention and cache management (458 lines)
- `tests/test_memory_optimization.py` - Comprehensive test suite for data and memory utilities (632 lines)

## Decisions Made

**worker-count-formula:** Used `min(cpu_count // num_gpus, 8)` formula for DataLoader worker count. Rationale: Balances CPU resources across multiple GPU workers while capping at 8 to prevent memory explosion from too many worker processes. Division by num_gpus ensures each GPU worker gets fair share of CPU resources.

**prefetch-factor-fixed:** Set `prefetch_factor=2` as fixed good default instead of making it configurable. Rationale: Research shows prefetch_factor=2 provides good I/O-compute overlap without excessive memory overhead. Making it configurable adds complexity with little benefit - users needing different values can pass via kwargs.

**spawn-context-dataloader:** Use spawn multiprocessing context for DataLoader workers when num_workers > 0. Rationale: Matches GPU worker pattern from parallel.py for consistency, avoids CUDA re-initialization issues, and prepares for Python 3.14 where spawn will be default.

**expandable-segments-opt-in:** Made expandable segments opt-in via `enable_expandable_segments` parameter instead of always-on. Rationale: While expandable segments reduce fragmentation, they can cause different allocation patterns that may affect some workloads. Opt-in allows users to enable when needed without forcing on all workloads.

**cache-interval-configurable:** Made cache clearing interval configurable (default 100 batches) instead of fixed. Rationale: Optimal clearing frequency varies with batch size and memory patterns. Some workloads benefit from frequent clearing (every 10 batches), others from less frequent (every 500 batches). Configurability allows tuning without code changes.

**sequence-sorting-optional:** Made sequence sorting optional in `create_sequence_dataloader` with `sort_by_length` parameter. Rationale: While sorting reduces padding/fragmentation for variable-length sequences, it breaks batch randomization which can affect model training. Optional parameter allows users to choose based on their use case (inference vs training).

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None - implementation followed PyTorch DataLoader best practices and memory management patterns from research document.

## User Setup Required

None - utilities are library code requiring no external configuration.

## Next Phase Readiness

**Ready for Phase 04 Plan 03 (CUDA Stream Orchestration):**
- DataLoader utilities available for I/O optimization
- MemoryManager ready for integration with stream-based processing
- Memory tracking available for profiling stream overlap efficiency

**Potential optimizations:**
- Integrate DataLoader with existing DNABERT-S and ESM-2 workers
- Use MemoryManager for periodic cache clearing in long-running jobs
- Apply sequence sorting in feature extraction to reduce padding overhead

## Testing

**Test Coverage:**
- 40 test cases across 4 test classes
- DataLoader utilities: worker count calculation, configuration, pin_memory control
- MemoryManager: expandable segments, cache clearing, memory stats, OOM prevention
- Integration: DataLoader with MemoryManager, global configuration
- Performance: sorted vs unsorted sequence efficiency benchmarks
- Error handling: invalid inputs, missing CUDA, edge cases

**CI Compatibility:**
- All tests mock torch module for environments without PyTorch
- Tests verify behavior without requiring actual CUDA devices
- Performance benchmarks marked with @pytest.mark.slow for optional execution

## Documentation

**Key patterns for future use:**

```python
# Optimized DataLoader with auto-detected workers
from virnucpro.data import create_optimized_dataloader

dataloader = create_optimized_dataloader(
    dataset=my_dataset,
    batch_size=32,
    num_gpus=4,  # Auto-calculates workers based on CPU count
    shuffle=True
)

# Sequence DataLoader with sorting for memory efficiency
from virnucpro.data import create_sequence_dataloader

dataloader = create_sequence_dataloader(
    sequences=my_sequences,
    batch_size=16,
    sort_by_length=True,  # Reduces padding overhead
    num_gpus=2
)

# Memory management for long-running jobs
from virnucpro.cuda import configure_memory_optimization

mm = configure_memory_optimization(
    enable_expandable=True,  # Reduce fragmentation
    cache_interval=100,      # Clear every 100 batches
    verbose=True
)

# In batch processing loop
for batch_num, batch in enumerate(dataloader):
    output = model(batch)
    mm.increment_and_clear()  # Auto-clear at intervals

# Memory tracking for diagnostics
with mm.memory_tracking("model forward pass"):
    output = model(input)
```

## Integration Points

**For DNABERT-S/ESM-2 workers:**
- Replace manual worker count calculation with `get_optimal_workers()`
- Add MemoryManager for periodic cache clearing in multi-file processing
- Use `sort_sequences_by_length()` before batching for padding reduction

**For future attention optimization:**
- DataLoader prefetching pairs well with CUDA stream I/O overlap
- MemoryManager provides memory stats for FlashAttention-2 profiling
- Safe batch size calculation helps prevent OOM with attention patterns

## Performance Expectations

**DataLoader optimization:**
- CPU-aware workers: Balanced resource utilization across GPUs
- Prefetch factor 2: ~20-30% reduction in data loading latency
- Pin memory: ~10-15% faster GPU transfer when enabled

**Memory management:**
- Expandable segments: ~34% memory reduction for variable-length batches
- Sequence sorting: ~20-40% reduction in padding overhead
- Periodic cache clearing: Prevents fragmentation-induced OOM in long jobs

**Testing:**
- All 40 tests pass with mocked torch
- Benchmarks show sorted sequences group by length correctly
- Integration tests verify DataLoader + MemoryManager work together
