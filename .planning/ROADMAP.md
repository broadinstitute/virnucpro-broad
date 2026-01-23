# Roadmap: VirNucPro GPU Optimization

## Overview

This roadmap transforms VirNucPro from a 45-hour single-GPU bottleneck into a sub-10-hour multi-GPU powerhouse. Starting with ESM-2 parallelization (the critical path), we progressively layer in DNABERT-S optimization, checkpoint robustness, memory/attention optimizations, load balancing, and finally validate the complete system achieves linear GPU scaling with >80% utilization.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [x] **Phase 1: ESM-2 Multi-GPU Foundation** - Parallelize ESM-2 across GPUs with file-level distribution
- [x] **Phase 1.1: Parallel Translation (INSERTED)** - Parallelize six-frame translation with CPU multiprocessing
- [x] **Phase 2: DNABERT-S Optimization** - Optimize DNABERT-S batching and queuing to match ESM-2
- [x] **Phase 2.1: Parallel Embedding Merge (INSERTED)** - Parallelize embedding merge with multi-processing/multi-threading
- [x] **Phase 3: Checkpoint Robustness** - Atomic writes, validation, backward compatibility
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
**Plans**: 7 plans

Plans:
- [x] 01-01-PLAN.md — Create ESM-2 worker functions and batch queue manager
- [x] 01-02-PLAN.md — Add BF16 optimization and GPU monitoring dashboard
- [x] 01-03-PLAN.md — Integrate into pipeline with CLI support and tests
- [x] 01-04-PLAN.md — End-to-end integration test and documentation
- [x] 01-05-PLAN.md — Fix multi-GPU auto-detection (gap closure)
- [x] 01-06-PLAN.md — Fix BF16 logging visibility (gap closure)
- [x] 01-07-PLAN.md — Add progress dashboard for GPU workers (gap closure)

### Phase 1.1: Parallel Translation (INSERTED)
**Goal**: Parallelize six-frame translation step to reduce processing time from >10 minutes to under 2 minutes for 22M sequences using CPU multiprocessing.
**Depends on**: Phase 1
**Requirements**: TBD
**Success Criteria** (what must be TRUE):
  1. Six-frame translation uses multiprocessing to parallelize across CPU cores
  2. --threads CLI parameter controls number of worker processes (default: CPU count)
  3. Processing 22M sequences completes in under 2 minutes on 8-core system
  4. Translation output remains identical to single-threaded implementation
  5. Memory usage stays reasonable (no explosive growth with worker count)
**Plans**: 3 plans

Plans:
- [x] 01.1-01-PLAN.md — Create parallel translation worker functions and orchestration
- [x] 01.1-02-PLAN.md — Integrate into pipeline with CLI support
- [x] 01.1-03-PLAN.md — Add comprehensive tests and validation

### Phase 2: DNABERT-S Optimization
**Goal**: DNABERT-S feature extraction matches ESM-2's optimization level with improved batching, automatic queuing, and unified worker infrastructure.
**Depends on**: Phase 1
**Requirements**: DNABERT-01, DNABERT-02, GPU-02, TEST-03
**Success Criteria** (what must be TRUE):
  1. DNABERT-S processes multiple batches per GPU automatically without manual file splitting
  2. Batch sizes for both DNABERT-S and ESM-2 are optimized via profiling (2-4x increase from baseline)
  3. DNABERT-S and ESM-2 use the same worker pool pattern for consistency
  4. Unit tests verify DNABERT-S optimized batching produces identical output to vanilla implementation
**Plans**: 5 plans

Plans:
- [x] 02-01-PLAN.md — Create BaseEmbeddingWorker abstraction and DNABERT-S parallel worker
- [x] 02-02-PLAN.md — Add BF16 optimization and performance validation
- [x] 02-03-PLAN.md — Integrate into pipeline with CLI support and profiling tools
- [x] 02-04-PLAN.md — Add batch size profiling utilities
- [x] 02-05-PLAN.md — Integration tests and optimization documentation

### Phase 2.1: Parallel Embedding Merge (INSERTED)
**Goal**: Parallelize embedding merge step using CPU multiprocessing to reduce merge time from sequential bottleneck to parallel throughput.
**Depends on**: Phase 2
**Requirements**: None (uses standard library multiprocessing)
**Success Criteria** (what must be TRUE):
  1. Multiple file pairs merge simultaneously using all CPU cores
  2. Merge throughput scales linearly with CPU core count
  3. Sequential fallback works for single-core systems
  4. Progress reporting shows file merge status in real-time
  5. Merged output is identical to sequential implementation
**Plans**: 5 plans

Plans:
- [x] 02.1-01-PLAN.md — Create parallel merge worker functions and orchestration
- [x] 02.1-02-PLAN.md — Integrate into pipeline with CLI support
- [x] 02.1-03-PLAN.md — Add integration tests and performance validation
- [x] 02.1-04-PLAN.md — Unify thread control parameters (gap closure)
- [x] 02.1-05-PLAN.md — Fix workload-aware merge strategy (gap closure)

### Phase 3: Checkpoint Robustness
**Goal**: Checkpoint system prevents corruption, validates integrity, supports resume from pre-optimization runs, and maintains backward compatibility.
**Depends on**: Phase 2.1
**Requirements**: INFRA-03, INFRA-04, INFRA-05, COMPAT-02, LOAD-02, TEST-04
**Success Criteria** (what must be TRUE):
  1. All checkpoint writes use atomic temp-then-rename pattern (no partial files)
  2. Pipeline validates checkpoint files are >0 bytes and optionally validates keys before marking complete
  3. Pipeline resumes from checkpoints created by pre-optimization code without errors
  4. Checkpoint files include .done markers to distinguish complete vs in-progress work
  5. Checkpoint version field supports migration functions for future format changes
  6. Unit tests verify atomic writes, corruption handling, and successful resume from pre-optimization checkpoints
**Plans**: 4 plans

Plans:
- [x] 03-01-PLAN.md — Create validation utilities and extend atomic writes
- [x] 03-02-PLAN.md — Add version management and backward compatibility
- [x] 03-03-PLAN.md — Integrate into pipeline with comprehensive tests
- [x] 03-04-PLAN.md — Add .done marker files for quick resume checks (gap closure)

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
  2. File assignment algorithm balances sequences based on estimated compute (length-aware)
  3. GPU dashboard shows real-time utilization, file progress, and per-GPU throughput
  4. Heterogeneous GPUs (e.g., 3090+4090) get work proportional to their compute capability
  5. Unit tests verify load balancing algorithm fairly distributes sequences by compute time
**Plans**: TBD

Plans:
- [ ] 05-01: TBD
- [ ] 05-02: TBD
- [ ] 05-03: TBD

### Phase 6: Performance Validation
**Goal**: System benchmarking proves <10 hour processing time for typical samples and demonstrates linear GPU scaling up to 8 GPUs.
**Depends on**: Phase 5
**Requirements**: PERF-01, PERF-02, SCALE-01, MON-03, TEST-06
**Success Criteria** (what must be TRUE):
  1. Benchmark suite runs on 1, 2, 4, 8 GPU configurations and reports speedup ratios
  2. Processing one sample (thousands of sequences) completes in <10 hours on 4 GPUs
  3. GPU utilization stays above 80% during embedding stages (measured via nvitop)
  4. Speedup is near-linear: 2 GPUs = ~1.8x, 4 GPUs = ~3.5x, 8 GPUs = ~7x faster than single GPU
  5. Performance report generated showing throughput, memory usage, and bottleneck analysis
  6. Integration tests confirm optimized pipeline produces identical predictions to vanilla
**Plans**: TBD

Plans:
- [ ] 06-01: TBD
- [ ] 06-02: TBD

## Completion Criteria

The optimization project is complete when:
- [ ] Pipeline processes one sample in <10 hours on a 4-GPU system
- [ ] GPU utilization exceeds 80% during embedding stages
- [ ] All optimizations maintain backward compatibility
- [ ] Performance scales near-linearly with GPU count
- [ ] All tests pass including vanilla comparison
- [ ] Documentation updated with optimization guide