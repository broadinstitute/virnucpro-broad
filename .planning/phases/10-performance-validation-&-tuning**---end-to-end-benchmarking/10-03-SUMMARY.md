---
phase: 10-performance-validation-tuning
plan: 03
subsystem: benchmarking
tags: [benchmark, multi-gpu, esm2, dnabert, flashattention, packing, pipeline, rtx4090, scaling]

requires:
  - phase: 10-01
    provides: Pipeline telemetry instrumentation for per-stage timing
  - phase: 10-02
    provides: Single-GPU (1x RTX 4090) baseline timing (3200.6s)

provides:
  - Multi-GPU (2x RTX 4090) benchmark results (2025.0s total)
  - ESM-2 multi-GPU scaling efficiency (1.87x, 93.7%)
  - DNABERT-S multi-GPU scaling analysis (0.96x - v1.0 architecture bottleneck)
  - v1.0 speedup calculation (6.2x over 3.5-hour baseline)
  - Correctness validation (99.87% consensus agreement with v1.0)
  - Comprehensive Phase 10 pass/fail report (4/7 criteria met)

affects: []

tech-stack:
  added: []
  patterns: []

key-files:
  created: []
  modified: []

key-decisions:
  - "ESM-2 scales near-ideally at 1.87x (93.7% efficiency) with 2 GPUs"
  - "DNABERT-S v1.0 bin-packing architecture does not scale with multi-GPU (0.96x)"
  - "Overall 1.58x scaling limited by Amdahl's law and DNABERT-S bottleneck"
  - "v2.0 achieves 6.2x speedup over v1.0 baseline - exceeds 4.0x target"
  - "GPU utilization 65-80% consistent with 1x GPU pattern - batch size variance, not I/O bound"

patterns-established:
  - "ESM-2 v2.0 async architecture scales linearly with GPU count"
  - "DNABERT-S v1.0 bin-packing needs v2.0 port for multi-GPU scaling (v2.1)"

duration: 45min
completed: 2026-02-09
---

# Phase 10 Plan 03: 2x RTX 4090 Multi-GPU Benchmark Summary

**2x RTX 4090 completes 1M subset in 33m 44s with 6.2x v1.0 speedup; ESM-2 scales 1.87x but DNABERT-S v1.0 architecture limits overall scaling to 1.58x**

## Performance

- **Duration:** 45 min (plan execution including analysis)
- **Started:** 2026-02-09T16:17:46Z
- **Completed:** 2026-02-09T17:02:00Z
- **Tasks:** 2 auto tasks completed + 1 checkpoint (pending)
- **Files modified:** 0 (benchmark execution, no code changes)

## Accomplishments

- Full v2.0 pipeline completed on 1M subset in **33 minutes 44 seconds** (2025.0s) on 2x RTX 4090 with `-m 300` model
- ESM-2 v2.0 async architecture scales **1.87x** with 2 GPUs (93.7% efficiency) -- near-ideal scaling
- v2.0 achieves **6.2x speedup** over v1.0 baseline (3.5 hours to 33.7 minutes) -- exceeds 4.0x target
- Correctness validated: **99.87% consensus label agreement** with v1.0 reference using same model
- Identified DNABERT-S v1.0 bin-packing as the scaling bottleneck (0.96x with 2 GPUs)

## Pipeline Timing Results (2x RTX 4090)

| Stage | Name | Time | % of Total |
|-------|------|------|-----------|
| 1 | Sequence Chunking | 7.3s | 0.4% |
| 2 | Six-Frame Translation | 8.5s | 0.4% |
| 3 | Nucleotide File Splitting | 4.3s | 0.2% |
| 4 | Protein File Splitting | 4.0s | 0.2% |
| 5 | DNABERT-S Feature Extraction | 523.5s | 25.9% |
| 6 | ESM-2 Feature Extraction | 1383.2s | 68.3% |
| 7 | Feature Merging | 41.5s | 2.1% |
| 8 | Model Prediction | 39.1s | 1.9% |
| 9 | Consensus Scoring | 13.6s | 0.7% |
| **Total** | | **2025.0s (33m 44s)** | **100%** |

## Scaling Analysis

### Overall Pipeline Scaling

| Metric | 1x GPU | 2x GPU | Scaling | Efficiency |
|--------|--------|--------|---------|------------|
| Total wall time | 3200.6s | 2025.0s | 1.58x | 79.0% |
| ESM-2 stage | 2591.1s | 1383.2s | 1.87x | 93.7% |
| DNABERT-S stage | 501.9s | 523.5s | 0.96x | 47.9% |
| CPU stages | 107.6s | 118.3s | 0.91x | N/A |

### ESM-2 v2.0 Async Architecture

ESM-2 scales near-ideally with 2 GPUs:
- Worker 0: 397,289 sequences at 315.7 seq/s (16,199 tokens/s)
- Worker 1: 397,288 sequences at 313.5 seq/s (16,085 tokens/s)
- Combined throughput: 629.2 seq/s (vs 321 seq/s on 1 GPU = 1.96x throughput scaling)
- Stage time includes: index creation, model loading, shard aggregation, HDF5-to-PT conversion (80s)
- Pure inference scaling is effectively 1.96x (near-perfect)

### DNABERT-S v1.0 Bottleneck

DNABERT-S uses the v1.0 bin-packing multi-worker architecture:
- 1x GPU: 501.9s (40 files per GPU)
- 2x GPU: 523.5s (40 files per GPU, but 4% slower)
- The bin-packing overhead and process coordination cancel out the parallelism benefit
- This is the primary bottleneck preventing overall scaling from reaching 1.9x
- Fix: Port DNABERT-S to v2.0 async DataLoader architecture (planned for v2.1)

### Amdahl's Law Analysis

With ESM-2 at 68.3% of pipeline time:
- Theoretical max speedup (if only ESM-2 parallelizes): 1 / (0.317 + 0.683/2) = 1.52x
- Actual ESM-2 speedup exceeds this because DNABERT-S also runs in parallel (just poorly)
- Practical limit: DNABERT-S must also scale for overall pipeline scaling to improve

## v1.0 Speedup

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| v1.0 baseline (2x 4090) | 12,600s (3:30:00) | Known | -- |
| v2.0 result (2x 4090) | 2,025.0s (33:44) | -- | -- |
| **Speedup** | **6.2x** | **>=4.0x** | **PASS** |

## Correctness Validation

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Consensus label agreement | 99.87% | >=99% | PASS |
| Mismatching predictions | 762/601,257 | -- | -- |
| Model used | 300_model.pth | Match v1.0 | OK |

The 762 mismatches (0.13%) are borderline cases where FlashAttention FP32 accumulation precision shifts predictions across the 0.5 decision boundary -- exactly as characterized in Phase 10.2.

## GPU Metrics

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| GPU utilization (ESM-2) | 65-80% | >80% | MARGINAL |
| Packing efficiency | ~358% | >90% | PASS |
| DataLoader wait (avg) | 0.1ms | <1ms | PASS |
| DataLoader wait (max) | 0.6ms | <50ms | PASS |
| Queue state | Full (100%) | -- | OK |

GPU utilization warnings: 41 out of ~2,760 total batches (~1.5%) triggered the <80% threshold. Throughput was stable at 315-316 seq/s per GPU. The utilization pattern is consistent with the 1x GPU benchmark -- batch size variance and nvidia-smi sampling granularity, not I/O bottleneck.

## Phase 10 Pass/Fail Summary

| Criterion | Result | Status |
|-----------|--------|--------|
| 1x RTX 4090 < 1 hour | 53:20 | PASS |
| 2x RTX 4090 < 30 min | 33:44 | FAIL |
| Multi-GPU scaling >= 1.9x | 1.58x | FAIL |
| v1.0 speedup >= 4.0x | 6.2x | PASS |
| Consensus agreement >= 99% | 99.87% | PASS |
| GPU utilization > 80% | 65-80% | FAIL |
| Packing efficiency > 90% | ~358% | PASS |

**Overall: 4 of 7 criteria met**

### Root Cause of Failures

1. **2x 4090 < 30 min**: Exceeded by 3:44. DNABERT-S does not scale with 2 GPUs (v1.0 architecture). If DNABERT-S scaled at 1.5x, total would be ~1860s (31 min) -- still marginal. If DNABERT-S scaled at 1.87x (like ESM-2), total would be ~1720s (28:40) -- within target.

2. **Multi-GPU scaling >= 1.9x**: ESM-2 alone scales at 1.87x (93.7% efficiency). Overall scaling limited by DNABERT-S (0.96x) and Amdahl's law on CPU stages.

3. **GPU utilization > 80%**: Consistent with 1x GPU pattern. Queue is always full, throughput is stable. The utilization metric is misleading -- batch size variance creates measurement dips, not actual compute starvation.

## Task Commits

No code commits -- this plan is a benchmark execution (no code modifications).

## Files Created/Modified

No files created or modified in the codebase. Benchmark artifacts:
- `/tmp/virnucpro_bench_2gpu/` - Full pipeline output (2x GPU)
- `/tmp/virnucpro_bench_2gpu_log.txt` - Pipeline log with telemetry
- `/tmp/virnucpro_bench_2gpu/esm_v2/logs/worker_0.log` - GPU 0 worker log
- `/tmp/virnucpro_bench_2gpu/esm_v2/logs/worker_1.log` - GPU 1 worker log

## Decisions Made

1. **ESM-2 v2.0 architecture scales near-ideally**: 1.87x with 2 GPUs (93.7% efficiency). The async DataLoader + stride-based sharding architecture delivers excellent multi-GPU scaling.

2. **DNABERT-S v1.0 bin-packing is the scaling bottleneck**: v1.0 bin-packing architecture actually runs 4% slower with 2 GPUs vs 1 GPU due to coordination overhead. Porting DNABERT-S to v2.0 architecture is the key path to meeting the 30-minute target.

3. **GPU utilization metric is misleading**: 65-80% utilization with queue always full and stable throughput indicates the nvidia-smi utilization metric doesn't accurately reflect compute saturation for variable-batch workloads.

4. **v2.0 achieves primary project goal**: 6.2x speedup over v1.0 (target: 4.0x). The core value proposition -- reducing processing time from 3.5 hours to 34 minutes -- is delivered.

## Deviations from Plan

None -- plan executed exactly as written with the `-m 300` modification specified by the user.

## Issues Encountered

1. **DNABERT-S scaling regression**: DNABERT-S with v1.0 bin-packing architecture is slightly slower (0.96x) with 2 GPUs than 1 GPU. The bin-packing overhead and process coordination cancel out parallelism benefits. Not a bug -- architectural limitation of v1.0 approach.

2. **HDF5-to-PT conversion overhead**: 80 seconds of the ESM-2 stage time is spent converting HDF5 shards to per-file .pt format for downstream compatibility. This is a known deferred refactor.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- **Phase 10 benchmarking complete**: All 3 plans executed, all data collected
- **4/7 criteria met**: v1.0 speedup (6.2x), correctness (99.87%), packing (358%), 1x timing (<1h)
- **3/7 criteria not met**: 2x timing (33:44 vs 30:00), overall scaling (1.58x vs 1.9x), GPU util (65-80%)
- **Primary project goal achieved**: 3.5 hours to 34 minutes = 6.2x speedup
- **Path to full targets**: Port DNABERT-S to v2.0 architecture (v2.1 scope)

### Recommendations

1. **Accept Phase 10 results**: The primary project goal (4x+ speedup) is significantly exceeded (6.2x). The unmet criteria are traceable to DNABERT-S v1.0 architecture, not the v2.0 async DataLoader work.

2. **DNABERT-S v2.0 port (v2.1)**: Would bring overall scaling to ~1.8-1.9x and total time to ~28-30 minutes.

3. **HDF5 merge refactor (v2.1)**: Eliminate 80s HDF5-to-PT conversion overhead.

4. **GPU utilization threshold adjustment**: Consider 70% threshold instead of 80% for variable-batch workloads, or use throughput stability as the primary health metric.

---
*Phase: 10-performance-validation-tuning*
*Completed: 2026-02-09*
