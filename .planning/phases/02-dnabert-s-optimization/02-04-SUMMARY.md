---
phase: 02-dnabert-s-optimization
plan: 04
subsystem: performance-tuning
tags: [profiling, batch-size-optimization, gpu-memory, cli, dnabert-s, esm2]

dependencies:
  requires:
    - phase: 02
      plan: 01
      reason: "DNABERT-S parallel worker with token-based batching"
    - phase: 02
      plan: 03
      reason: "CLI patterns and pipeline integration"
  provides:
    - "Batch size profiling utilities for hardware-specific optimization"
    - "CLI command for easy batch size profiling"
    - "Automated recommendations for optimal GPU settings"
  affects:
    - phase: 03
      reason: "Performance testing will use profiling to optimize batch sizes"

tech-stack:
  added: []
  patterns:
    - "GPU profiling with throughput and memory measurement"
    - "Binary search for maximum batch size before OOM"
    - "80% of max batch size recommendation for safety headroom"
    - "Synthetic test data generation for profiling"

key-files:
  created:
    - virnucpro/pipeline/profiler.py: "Batch size profiling utilities for DNABERT-S and ESM-2"
    - virnucpro/cli/profile.py: "CLI command for batch size profiling"
  modified:
    - virnucpro/cli/main.py: "Register profile command in CLI group"

decisions:
  - id: profiling-80-percent-headroom
    decision: "Recommend 80% of maximum batch size as optimal"
    rationale: "Leaves headroom for sequence length variation and concurrent processes"
    alternatives: "Use 100% of max (risky), or 50% (too conservative)"
    impact: "Users get safe recommendations that work consistently"

  - id: synthetic-test-sequences
    decision: "Generate synthetic test sequences if no file provided"
    rationale: "Allows profiling without requiring user data files"
    alternatives: "Require test file (worse UX), use fixed sequences (not representative)"
    impact: "Profiling works out of the box without preparation"

metrics:
  duration: 156s
  completed: 2026-01-23
---

# Phase 02 Plan 04: Batch Size Profiling Utilities Summary

**GPU batch size profiling utilities with CLI integration for automated hardware-specific optimization recommendations**

## Performance

- **Duration:** 2.6 min (156 seconds)
- **Started:** 2026-01-23T16:04:33Z
- **Completed:** 2026-01-23T16:07:09Z
- **Tasks:** 2
- **Files created:** 2
- **Files modified:** 1

## Accomplishments

- Profiler utilities measure throughput and memory across batch sizes
- CLI command provides easy access to profiling without writing Python
- Automatic recommendations (80% of max) for safe batch sizes
- Support for both DNABERT-S and ESM-2 models
- BF16 auto-detection and reporting
- Throughput visualization as ASCII chart

## Task Commits

Each task was committed atomically:

1. **Task 1: Create batch size profiling utilities** - `0c4ec51` (feat)
2. **Task 2: Add CLI command for profiling** - `99e2870` (feat)

**Total commits:** 2

## Files Created/Modified

- `virnucpro/pipeline/profiler.py` - Batch size profiling utilities
  - profile_dnabert_batch_size(): Test DNABERT-S batch sizes, measure throughput/memory
  - profile_esm_batch_size(): Test ESM-2 batch sizes with token-based batching
  - measure_gpu_memory(): Track CUDA memory usage during profiling
  - create_test_sequences(): Generate synthetic DNA/protein test data
  - binary_search_max_batch(): Find maximum batch size before OOM (not used in current implementation)
  - Safety features: OOM catching, CUDA cache clearing, torch.no_grad()
  - Returns optimal batch size (80% of max) with throughput/memory curves

- `virnucpro/cli/profile.py` - CLI command for profiling
  - Profile subcommand: `virnucpro profile --model [dnabert-s|esm2] --device cuda:0`
  - Options for min/max/step batch sizes, test file, JSON output
  - User-friendly display with recommendations and ASCII throughput chart
  - Integration with profiler functions from virnucpro.pipeline.profiler
  - Help text with clear examples and usage instructions

- `virnucpro/cli/main.py` - Register profile command
  - Import profile module
  - Add profile command to CLI group

## Decisions Made

### 1. Recommend 80% of Maximum Batch Size

**Decision:** Suggest optimal batch size at 80% of measured maximum

**Reasoning:**
- Sequence length variation can cause memory spikes
- Other GPU processes may compete for memory
- Leaves safety headroom for real-world usage
- Prevents intermittent OOM errors

**Impact:** Users get reliable recommendations that work consistently in production.

### 2. Synthetic Test Sequences

**Decision:** Generate random DNA/protein sequences when no test file provided

**Reasoning:**
- Users can profile without preparing test data
- Random sequences approximate real-world variation
- Simple to implement and deterministic
- Optional test file for more accurate profiling with user data

**Impact:** Profiling works out of the box with zero configuration.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None - profiler implementation followed established patterns from parallel workers.

## User Setup Required

None - profiling is a utility command that requires no configuration.

## Next Phase Readiness

**Ready for Phase 3 (Performance Testing):**
- ✅ Profiling utilities available for batch size optimization
- ✅ CLI command accessible: `virnucpro profile --model dnabert-s`
- ✅ Automatic recommendations for user's specific GPU hardware
- ✅ Throughput and memory measurement capabilities
- ✅ Support for both DNABERT-S and ESM-2 models

**Usage example:**
```bash
# Profile DNABERT-S on GPU 0
python -m virnucpro profile --model dnabert-s --device cuda:0

# Profile ESM-2 with custom range
python -m virnucpro profile --model esm2 --min-batch 1024 --max-batch 4096

# Save results to JSON
python -m virnucpro profile --model dnabert-s --output profile_results.json
```

**Expected profiling results:**
- Batch size recommendations tailored to specific GPU (RTX 4090, A100, etc.)
- Throughput curves showing performance scaling
- Memory usage tracking to avoid OOM
- BF16 detection and recommendations

**No blockers identified.**

## Code Quality

- Follows established patterns from parallel workers (BF16 detection, device handling)
- Consistent parameter naming with existing CLI commands
- Proper error handling and logging throughout
- Safety features prevent GPU crashes during profiling
- Clear user-facing output with actionable recommendations

---

*Phase: 02-dnabert-s-optimization*
*Plan: 04*
*Status: ✅ Complete*
*Date: 2026-01-23*
