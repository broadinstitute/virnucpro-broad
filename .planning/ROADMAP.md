# Roadmap: VirNucPro GPU Optimization

## Milestones

- ✅ **v1.0 GPU Optimization Foundation** - Phases 1-4.1 (shipped 2026-02-02)
- 🚧 **v2.0 Async Architecture + Sequence Packing** - Phases 5-10 (in progress)

## Overview

v2.0 replaces the v1.0 multi-worker-per-GPU architecture with a modern async DataLoader + sequence packing pipeline to achieve 4.5× speedup (45h → <10h target). The journey starts by establishing single-GPU async DataLoader foundation with CUDA safety guarantees (Phase 5), then adds sequence packing with FlashAttention varlen for 2-3× throughput gain (Phase 6), scales to multi-GPU coordination (Phase 7), validates FP16 precision for 2× memory reduction (Phase 8), integrates robust checkpointing for 6M sequence workloads (Phase 9), and completes with end-to-end performance validation and telemetry (Phase 10).

## Phases

<details>
<summary>✅ v1.0 GPU Optimization Foundation (Phases 1-4.1) - SHIPPED 2026-02-02</summary>

**Milestone Goal:** Multi-GPU parallelization for DNABERT-S and ESM-2 with BF16 optimization, FlashAttention-2, and persistent model loading

**Key Accomplishments:**
- Multi-GPU parallelization for ESM-2 and DNABERT-S with file-level work distribution
- BF16 mixed precision on Ampere+ GPUs with automatic capability detection
- FlashAttention-2 integration for 2-4x attention speedup
- Persistent model loading eliminating re-loading overhead between pipeline stages
- Checkpoint robustness with atomic writes and .done markers
- Parallel translation and embedding merge for CPU-bound stages
- Dynamic batching optimization with token-based batching and padding efficiency

**Stats:**
- 11,431 lines of Python code
- 7 phases, 34 plans, 129 tasks (estimated)
- 84 days from start to ship
- Average plan duration: 3.4 minutes
- Total execution time: 2.2 hours

See MILESTONES.md for detailed v1.0 retrospective.

</details>

### 🚧 v2.0 Async Architecture + Sequence Packing (In Progress)

**Milestone Goal:** Replace multi-worker-per-GPU architecture with single-process-per-GPU + async DataLoader, add sequence packing, and switch to FP16 precision for maximum throughput.

**Architectural Shift:** From multiprocessing.Pool with N workers per GPU → single process per GPU with async DataLoader pattern. Addresses v1.0 limitations: N×11GB memory overhead, pickle serialization tax, GPU starvation from small batches.

**Phase Numbering:**
- Integer phases (5, 6, 7): Planned milestone work
- Decimal phases (5.1, 5.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [x] **Phase 5: Async DataLoader Foundation** - Single-GPU async pattern with CUDA safety
- [x] **Phase 6: Sequence Packing Integration** - FlashAttention varlen + greedy packing
- [ ] **Phase 7: Multi-GPU Coordination** - Index-based sharding across N GPUs
- [ ] **Phase 8: FP16 Precision Validation** - Memory and speed optimization
- [ ] **Phase 9: Checkpointing Integration** - Robustness for 6M sequence workloads
- [ ] **Phase 10: Performance Validation & Tuning** - End-to-end benchmarking

## Phase Details

### Phase 5: Async DataLoader Foundation

**Goal**: Single-GPU DataLoader safely handles I/O without CUDA corruption

**Depends on**: Phase 4.1 (v1.0 baseline)

**Requirements**: ARCH-01, ARCH-02, ARCH-03, ARCH-04, ARCH-05, SAFE-01, SAFE-02, SAFE-03, SAFE-04, SAFE-05

**Success Criteria** (what must be TRUE):
  1. DataLoader workers parse FASTA and tokenize on CPU without CUDA initialization
  2. GPU process receives prefetched batches with <5% idle time (validated via nvitop)
  3. Output embeddings match v1.0 baseline (cosine similarity >0.999)
  4. Single-GPU throughput improves 1.2-1.5x over v1.0 sequential loading

**Plans**: 5 plans in 4 waves

Plans:
- [x] 05-01-PLAN.md — CUDA-safe SequenceDataset and VarlenCollator
- [x] 05-02-PLAN.md — GPU monitor DataLoader metrics extension
- [x] 05-03-PLAN.md — Async DataLoader factory with CUDA safety
- [x] 05-04-PLAN.md — AsyncInferenceRunner with stream overlap
- [x] 05-05-PLAN.md — Integration tests and verification

### Phase 6: Sequence Packing Integration

**Goal**: Variable-length sequences pack efficiently with FlashAttention varlen

**Depends on**: Phase 5

**Requirements**: ARCH-09, ARCH-10, ARCH-11, PACK-01, PACK-02, PACK-03, PACK-04, PACK-04b, PACK-05, PACK-06, PACK-06a

**Success Criteria** (what must be TRUE):
  1. Packing density >90% (token utilization vs padding waste)
  2. Packed embeddings match unpacked baseline (cosine similarity >0.999)
  3. No cross-sequence attention contamination (validated via position ID and attention mask tests)
  4. Throughput improves 2-3x over Phase 5 unpacked baseline
  5. Sequences >max_length processed in isolated batch_size=1 with truncation warnings

**Plans**: 8 plans in 5 waves

Plans:
- [x] 06-01-PLAN.md — GreedyPacker with FFD algorithm and dynamic token budget (PACK-03, ARCH-11)
- [x] 06-02-PLAN.md — Position ID generator and FlashAttention varlen wrapper
- [x] 06-03-PLAN.md — ESM2 forward_packed method
- [x] 06-04-PLAN.md — Wire packed inference in AsyncInferenceRunner
- [x] 06-05-PLAN.md — Packed vs unpacked equivalence validation
- [x] 06-06-PLAN.md — Packing efficiency metrics and monitoring
- [x] 06-07-PLAN.md — End-to-end integration tests and verification
- [x] 06-08-PLAN.md — Integrate GreedyPacker into VarlenCollator (PACK-02)

### Phase 7: Multi-GPU Coordination

**Goal**: N GPUs process independent sequence shards without conflicts

**Depends on**: Phase 6

**Requirements**: ARCH-06, ARCH-07, ARCH-08, GPU-01, GPU-02, GPU-03, GPU-04

**Success Criteria** (what must be TRUE):
  1. Index-based stride distribution balances work across GPUs (within 10%)
  2. Each GPU process runs independently without shared state during processing
  3. Checkpoint aggregation produces complete output with no duplicates or missing sequences
  4. Multi-GPU throughput scales linearly (4 GPUs = 3.8-4x faster than Phase 6 single-GPU)

**Plans**: 8 plans in 5 waves

Plans:
- [ ] 07-01-PLAN.md — SequenceIndex with stride distribution and caching
- [ ] 07-02-PLAN.md — IndexBasedDataset for byte-offset sequence reading
- [ ] 07-03-PLAN.md — Per-worker logging infrastructure
- [ ] 07-04-PLAN.md — GPUProcessCoordinator for worker lifecycle
- [ ] 07-05-PLAN.md — HDF5 shard aggregation with validation
- [ ] 07-06-PLAN.md — GPU worker function integrating inference pipeline
- [ ] 07-07-PLAN.md — run_multi_gpu_inference orchestration entry point
- [ ] 07-08-PLAN.md — Integration tests and human verification

### Phase 8: FP16 Precision Validation

**Goal**: FP16 delivers throughput improvement while maintaining embedding accuracy

**Depends on**: Phase 7

**Requirements**: PREC-01, PREC-02, PREC-03

**Success Criteria** (what must be TRUE):
  1. FP16 embeddings match FP32 baseline (cosine similarity >0.99)
  2. GPU memory usage reduced 40-50% (11GB → 6GB per model)
  3. Throughput improves 1.5-2x over Phase 7 FP32 baseline
  4. Batch sizes double (64-128 vs 32-64) due to memory headroom

**Plans**: 5 plans in 3 waves

Plans:
- [ ] 08-01-PLAN.md — FP16 model conversion with feature flag and FlashAttention alignment
- [ ] 08-02-PLAN.md — NaN/Inf detection, gpu_worker FP16 passthrough (ESM-2 + DNABERT-S), and unit tests
- [ ] 08-03-PLAN.md — FP16 vs FP32 equivalence integration tests (stratified validation)
- [ ] 08-04-PLAN.md — FP16 vs FP32 (Phase 7 baseline) throughput benchmarking
- [ ] 08-05-PLAN.md — Selective FP32 fallback for LayerNorm (PREC-03, conditional on 08-03 failure)

### Phase 9: Checkpointing Integration

**Goal**: Pipeline resumes from partial completion after crashes

**Depends on**: Phase 8

**Requirements**: CKPT-01, CKPT-02, CKPT-03, CKPT-04, CKPT-05, CKPT-06

**Success Criteria** (what must be TRUE):
  1. Incremental checkpoints save every 10K sequences per shard
  2. Resume from last checkpoint without reprocessing completed sequences
  3. GPU process crash recovery validated (kill mid-batch, resume completes successfully)
  4. Checkpoint validation detects corruption (size check, sequence count verification)

**Plans**: TBD

Plans:
- [ ] TBD during phase planning

### Phase 10: Performance Validation & Tuning

**Goal**: Pipeline meets <10h target with >80% GPU utilization

**Depends on**: Phase 9

**Requirements**: PERF-01, PERF-02, PERF-03, PERF-04, PERF-05

**Success Criteria** (what must be TRUE):
  1. End-to-end pipeline completes one sample in <10 hours on 4 GPUs
  2. GPU utilization >80% during embedding steps (validated via nvitop)
  3. Linear GPU scaling verified (2x GPUs = 1.9-2x faster)
  4. Telemetry logs tokens/sec, packing efficiency, and I/O wait time per batch
  5. Throughput targets met: 1M-2M sequences/hour per GPU with FlashAttention

**Plans**: TBD

Plans:
- [ ] TBD during phase planning

## Progress

**Execution Order:**
Phases execute in numeric order: 5 → 6 → 7 → 8 → 9 → 10

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 1. Setup & Planning | v1.0 | 5/5 | Complete | 2025-11-19 |
| 2. ESM-2 Multi-GPU Parallelization | v1.0 | 7/7 | Complete | 2025-11-24 |
| 3. DNABERT-S Optimization | v1.0 | 7/7 | Complete | 2025-12-01 |
| 4. Performance Optimization | v1.0 | 12/12 | Complete | 2026-01-15 |
| 4.1. BFloat16 Precision Integration | v1.0 | 3/3 | Complete | 2026-02-02 |
| 5. Async DataLoader Foundation | v2.0 | 5/5 | Complete | 2026-02-03 |
| 6. Sequence Packing Integration | v2.0 | 8/8 | Complete | 2026-02-04 |
| 7. Multi-GPU Coordination | v2.0 | 0/8 | Planned | - |
| 8. FP16 Precision Validation | v2.0 | 0/5 | Planned | - |
| 9. Checkpointing Integration | v2.0 | 0/TBD | Not started | - |
| 10. Performance Validation & Tuning | v2.0 | 0/TBD | Not started | - |

---
*Last updated: 2026-02-05 for Phase 8 revision (5 plans in 3 waves)*
