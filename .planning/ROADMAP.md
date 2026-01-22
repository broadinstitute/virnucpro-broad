# Roadmap: VirNucPro GPU Optimization

## Overview

This roadmap transforms VirNucPro from a 45-hour single-GPU bottleneck into a sub-10-hour multi-GPU powerhouse. Starting with ESM-2 parallelization (the critical path), we progressively layer in DNABERT-S optimization, checkpoint robustness, memory/attention optimizations, load balancing, and finally validate the complete system achieves linear GPU scaling with >80% utilization.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [ ] **Phase 1: ESM-2 Multi-GPU Foundation** - Parallelize ESM-2 across GPUs with file-level distribution
- [ ] **Phase 2: DNABERT-S Optimization** - Optimize DNABERT-S batching and queuing to match ESM-2
- [ ] **Phase 3: Checkpoint Robustness** - Atomic writes, validation, backward compatibility
- [ ] **Phase 4: Memory & Attention Optimization** - FlashAttention, prefetching, memory management
- [ ] **Phase 5: Load Balancing & Monitoring** - Efficient work distribution and GPU utilization visibility
- [ ] **Phase 6: Performance Validation** - Verify <10 hour target and linear scaling

## Phase Details

### Phase 1: ESM-2 Multi-GPU Foundation
**Goal**: ESM-2 feature extraction parallelizes across multiple GPUs using file-level work distribution, delivering 3-4x throughput improvement with backward-compatible single-GPU fallback.
**Depends on**: Nothing (foundation phase)
**Requirements**: ESM-01, ESM-02, GPU-01, INFRA-01, INFRA-02, SCALE-02, COMPAT-03, COMPAT-01, TEST-01, TEST-02
**Success Criteria** (what must be TRUE):
  1. User runs `virnucpro predict` with unchanged CLI and ESM-2 extraction automatically uses all available GPUs
  2. Processing 10k protein sequences completes 3-4x faster with 4 GPUs compared to single GPU
  3. Pipeline runs successfully on single-GPU systems without code changes (automatic fallback)
  4. ESM-2 workers use BF16 mixed precision and torch.no_grad mode (verifiable in logs)
  5. Batch queue manager distributes files round-robin across GPU workers with spawn context
  6. Unit tests verify ESM-2 worker model loading, batching, and output format
  7. Integration test confirms multi-GPU output matches single-GPU baseline (vanilla comparison)
**Plans**: 4 plans

Plans:
- [ ] 01-01-PLAN.md — Create ESM-2 worker functions and batch queue manager
- [ ] 01-02-PLAN.md — Add BF16 optimization and GPU monitoring dashboard
- [ ] 01-03-PLAN.md — Integrate into pipeline with CLI support and tests
- [ ] 01-04-PLAN.md — End-to-end integration test and documentation

### Phase 2: DNABERT-S Optimization
**Goal**: DNABERT-S feature extraction matches ESM-2's optimization level with improved batching, automatic queuing, and unified worker infrastructure.
**Depends on**: Phase 1
**Requirements**: DNABERT-01, DNABERT-02, GPU-02, TEST-03
**Success Criteria** (what must be TRUE):
  1. DNABERT-S processes multiple batches per GPU automatically without manual file splitting
  2. Batch sizes for both DNABERT-S and ESM-2 are optimized via profiling (2-4x increase from baseline)
  3. DNABERT-S and ESM-2 use the same worker pool pattern for consistency
  4. Unit tests verify DNABERT-S optimized batching produces identical output to vanilla implementation
**Plans**: TBD

Plans:
- [ ] 02-01: TBD
- [ ] 02-02: TBD

### Phase 3: Checkpoint Robustness
**Goal**: Checkpoint system prevents corruption, validates integrity, supports resume from pre-optimization runs, and maintains backward compatibility.
**Depends on**: Phase 2
**Requirements**: INFRA-03, INFRA-04, INFRA-05, COMPAT-02, LOAD-02, TEST-04
**Success Criteria** (what must be TRUE):
  1. All checkpoint writes use atomic temp-then-rename pattern (no partial files)
  2. Pipeline validates checkpoint files are >0 bytes and optionally validates keys before marking complete
  3. Pipeline resumes from checkpoints created by pre-optimization code without errors
  4. Checkpoint files include .done markers to distinguish complete vs in-progress work
  5. Checkpoint version field supports migration functions for future format changes
  6. Unit tests verify atomic writes, corruption handling, and successful resume from pre-optimization checkpoints
**Plans**: TBD

Plans:
- [ ] 03-01: TBD
- [ ] 03-02: TBD

### Phase 4: Memory & Attention Optimization
**Goal**: Memory-efficient processing with FlashAttention-2, DataLoader prefetching, CUDA streams, and fragmentation prevention delivers additional 1.5-2x speedup.
**Depends on**: Phase 3
**Requirements**: ATT-01, ATT-02, MEM-01, MEM-02, MEM-03, MEM-04, MEM-05, GPU-02, TEST-05
**Success Criteria** (what must be TRUE):
  1. ESM-2 uses FlashAttention-2 for 2-4x attention speedup (verifiable in model config)
  2. DNABERT-S uses FlashAttention-2 for attention optimization
  3. DataLoader uses num_workers=4-8 with prefetch_factor=2 and pin_memory=True
  4. CUDA streams overlap I/O and computation (stream 1 for loading, stream 2 for inference)
  5. Memory fragmentation prevented via sequence sorting, expandable segments, and periodic cache clearing
  6. Pipeline processes sequences without OOM errors despite variable length batches
  7. Unit tests verify FlashAttention-2 integration and memory fragmentation prevention mechanisms
**Plans**: TBD

Plans:
- [ ] 04-01: TBD
- [ ] 04-02: TBD
- [ ] 04-03: TBD

### Phase 5: Load Balancing & Monitoring
**Goal**: GPU utilization monitoring shows >80% usage, load-balanced file assignment prevents idle GPUs, and heterogeneous GPU configurations are supported.
**Depends on**: Phase 4
**Requirements**: MON-01, MON-02, MON-03, LOAD-01, LOAD-03
**Success Criteria** (what must be TRUE):
  1. nvitop logs GPU compute % and memory usage every 10 seconds during embedding stages
  2. Throughput logging tracks sequences/second per GPU for performance debugging
  3. Monitoring detects stalled workers and reports imbalanced load warnings
  4. File assignment uses greedy bin packing by sequence count for balanced work distribution
  5. Pipeline handles mixed GPU types and memory sizes without manual configuration
**Plans**: TBD

Plans:
- [ ] 05-01: TBD
- [ ] 05-02: TBD

### Phase 6: Performance Validation
**Goal**: Pipeline completes one sample in under 10 hours, demonstrates >80% GPU utilization, and proves linear scaling (2x GPUs = 2x speedup).
**Depends on**: Phase 5
**Requirements**: PERF-01, PERF-02, SCALE-01, TEST-06
**Success Criteria** (what must be TRUE):
  1. Pipeline processes one full sample (thousands of sequences) in under 10 hours on 4-GPU system
  2. GPU utilization logs show >80% compute usage during ESM-2 and DNABERT-S embedding stages
  3. Benchmark results show linear scaling: 2 GPUs = ~2x speedup, 4 GPUs = ~4x speedup, 8 GPUs = ~8x speedup
  4. Performance report documents baseline (45 hours single GPU) vs optimized throughput
  5. End-to-end integration test confirms full pipeline output matches vanilla baseline (predictions identical)
**Plans**: TBD

Plans:
- [ ] 06-01: TBD
- [ ] 06-02: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3 → 4 → 5 → 6

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. ESM-2 Multi-GPU Foundation | 0/4 | Planning complete | - |
| 2. DNABERT-S Optimization | 0/TBD | Not started | - |
| 3. Checkpoint Robustness | 0/TBD | Not started | - |
| 4. Memory & Attention Optimization | 0/TBD | Not started | - |
| 5. Load Balancing & Monitoring | 0/TBD | Not started | - |
| 6. Performance Validation | 0/TBD | Not started | - |