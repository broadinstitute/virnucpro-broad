# Optimization Roadmap

## Project Goal

Embedding steps (DNABERT-S and ESM-2) efficiently utilize all available GPUs and automatically queue batches, reducing sample processing time from 45+ hours to under 10 hours.

## Key Requirements

From REQUIREMENTS.md:

| ID | Requirement | Verification |
|----|-------------|--------------|
| **SCALE-01** | "Embedding steps distribute load across all available GPUs automatically" | Multi-GPU tests |
| **SCALE-02** | "Batching system queues sequences until optimal batch size reached" | Queue monitoring |
| **SCALE-03** | "GPU memory allocation adapts to available VRAM" | Memory tracking |
| **PERF-01** | "Processing one sample (thousands of sequences) completes in <10 hours" | Benchmark suite |
| **PERF-02** | "ESM-2 embedding step shows >5x speedup on multi-GPU systems" | Performance tests |
| **MON-01** | "Real-time progress reporting shows sequences/second per GPU" | Dashboard UI |
| **MON-02** | "GPU utilization visible during processing (nvidia-smi integration)" | nvitop display |
| **TEST-05** | "Performance regression tests catch >10% slowdowns" | CI benchmarks |
| **TEST-06** | "Multi-GPU tests verify load distribution fairness" | Distribution metrics |

## Phases

### Phase 1: Multi-GPU Load Distribution ✓
**Goal**: DNABERT-S and ESM-2 distribute work across all available GPUs automatically, with each GPU processing independent file batches in parallel.
**Requirements**: SCALE-01, SCALE-02, SCALE-03, TEST-05, TEST-06
**Plans**: 7 plans

Plans:
- [x] 01-01-PLAN.md — Create GPU detection and worker pool infrastructure
- [x] 01-02-PLAN.md — Implement file-based work distribution with multi-GPU pools
- [x] 01-03-PLAN.md — Parallelize DNABERT-S feature extraction with file batching
- [x] 01-04-PLAN.md — Parallelize ESM-2 feature extraction with file batching
- [x] 01-05-PLAN.md — Create BatchQueueManager for coordinating parallel processing
- [x] 01-06-PLAN.md — Add progress tracking and worker monitoring
- [x] 01-07-PLAN.md — Create parallel processing tests and benchmarks

### Phase 1.1: Load Distribution Recovery ✓
**Goal**: Fix multi-GPU processing to properly distribute work and use all available GPUs effectively.
**Requirements**: Same as Phase 1
**Plans**: 3 plans

Plans:
- [x] 01.1-01-PLAN.md — Fix prediction module imports and add basic multi-GPU infrastructure
- [x] 01.1-02-PLAN.md — Implement worker pools with proper GPU assignment
- [x] 01.1-03-PLAN.md — Complete integration and CLI support

### Phase 2: Model Streaming & BF16 ✓
**Goal**: Heterogeneous GPU support with automatic BF16 on capable hardware (Ampere+) and comprehensive memory management.
**Requirements**: SCALE-03, HETERO-01, HETERO-02, HETERO-03
**Success Criteria** (what must be TRUE):
  1. BF16 automatically enabled on Ampere+ GPUs (A100, 4090), FP32 on older GPUs (V100, 3090)
  2. Pipeline adapts batch sizes based on available GPU memory
  3. Heterogeneous setups (e.g., 3090 + 4090) process without errors
  4. Memory limits enforced per GPU to prevent OOM
  5. Tests verify BF16/FP32 outputs are numerically close (within tolerance)
**Plans**: 5 plans

Plans:
- [x] 02-01-PLAN.md — Implement GPU capability detection and BF16 auto-configuration
- [x] 02-02-PLAN.md — Add memory-aware batch sizing per GPU
- [x] 02-03-PLAN.md — Create heterogeneous GPU support with mixed precision
- [x] 02-04-PLAN.md — Add memory monitoring and OOM prevention
- [x] 02-05-PLAN.md — Test suite for mixed precision and heterogeneous setups

### Phase 2.1: Fix Model Streaming and Memory Management ✓
**Goal**: Correctly implement model streaming with proper BF16/FP32 handling per GPU.
**Requirements**: Same as Phase 2
**Success Criteria** (what must be TRUE):
  1. BF16 used only on GPUs with compute capability >= 8.0 (Ampere+)
  2. Each GPU worker independently determines its dtype based on hardware
  3. Memory limits properly enforced to prevent OOM
  4. Model streaming reduces memory usage for large models
  5. Tests pass on both BF16-capable and older GPUs
**Plans**: 5 plans

Plans:
- [x] 02.1-01-PLAN.md — Fix compute capability detection and BF16 logic
- [x] 02.1-02-PLAN.md — Implement per-GPU dtype configuration in workers
- [x] 02.1-03-PLAN.md — Add model streaming with device_map='auto'
- [x] 02.1-04-PLAN.md — Fix memory management and batch sizing
- [x] 02.1-05-PLAN.md — Update tests for correct BF16/FP32 handling

### Phase 3: Dynamic Batching Optimization ✓
**Goal**: Smart batch assembly that groups sequences by length to maximize GPU utilization while preventing OOM.
**Depends on**: Phase 2.1
**Requirements**: SCALE-02, BATCH-01, BATCH-02, BATCH-03
**Success Criteria** (what must be TRUE):
  1. Sequences are sorted by length before batching to minimize padding
  2. Dynamic batch sizes adapt based on sequence lengths in current batch
  3. Token-based batching for ESM-2 (e.g., 4096 tokens per batch)
  4. Padding efficiency > 85% (measured as actual tokens / padded tokens)
  5. No OOM errors even with highly variable sequence lengths
  6. GPU utilization remains > 80% during processing
**Plans**: 4 plans

Plans:
- [x] 03-01-PLAN.md — Implement sequence length analysis and sorting
- [x] 03-02-PLAN.md — Create dynamic batch size calculator based on lengths
- [x] 03-03-PLAN.md — Add token-based batching for ESM-2
- [x] 03-04-PLAN.md — Optimize padding efficiency and GPU utilization

### Phase 4: Memory & Attention Optimization ✓
**Goal**: FlashAttention-2 integration for ESM-2 and optimized memory patterns for sustained high throughput.
**Depends on**: Phase 3
**Requirements**: None (memory optimizations are implementation details)
**Success Criteria** (what must be TRUE):
  1. FlashAttention-2 enabled on compatible GPUs (Ampere+) with automatic fallback on older hardware
  2. BF16 mixed precision active where supported, reducing memory usage by ~50%
  3. DataLoader uses num_workers=4-8 with prefetch_factor=2 and pin_memory=True
  4. CUDA streams overlap I/O and computation (stream 1 for loading, stream 2 for inference)
  5. Memory fragmentation prevented via sequence sorting, expandable segments, and periodic cache clearing
  6. Pipeline processes sequences without OOM errors despite variable length batches
  7. Unit tests verify FlashAttention-2 integration and memory fragmentation prevention mechanisms
**Plans**: 4 plans

Plans:
- [x] 04-01-PLAN.md — Integrate FlashAttention-2 for ESM-2 with automatic fallback
- [x] 04-02-PLAN.md — Create optimized DataLoader and memory fragmentation prevention
- [x] 04-03-PLAN.md — Implement CUDA streams for I/O-compute overlap
- [x] 04-04-PLAN.md — Complete integration with CLI flags and DNABERT-S FlashAttention

### Phase 4.1: Persistent Model Loading (INSERTED)
**Goal**: Keep models in GPU memory persistently to eliminate re-loading overhead between pipeline stages
**Depends on**: Phase 4
**Requirements**: None (uses existing PyTorch and multiprocessing)
**Success Criteria** (what must be TRUE):
  1. Workers load models once during pool initialization, not per file batch
  2. Models remain in GPU memory across multiple file processing jobs
  3. Memory management prevents fragmentation via expandable segments and periodic cache clearing
  4. CLI flag --persistent-models enables the feature (default: disabled for backward compatibility)
  5. Output with persistent models matches standard worker output exactly
  6. Integration tests verify memory management and output correctness
**Plans**: 6 plans

Plans:
- [x] 04.1-01-PLAN.md — Create PersistentWorkerPool infrastructure
- [x] 04.1-02-PLAN.md — Implement persistent worker functions
- [x] 04.1-03-PLAN.md — Pipeline & CLI Integration
- [ ] 04.1-04-PLAN.md — Refactor feature extraction to accept pre-loaded models
- [ ] 04.1-05-PLAN.md — Wire pipeline to create and use persistent pools
- [ ] 04.1-06-PLAN.md — Fix integration tests and verify model persistence

### Phase 5: Advanced Load Balancing
**Goal**: Work stealing and dynamic rebalancing ensure no GPU sits idle while others have queued work.
**Depends on**: Phase 4.1
**Requirements**: SCALE-01, HETERO-01, HETERO-02, HETERO-03, MON-03, TEST-06
**Success Criteria** (what must be TRUE):
  1. Idle GPUs steal work from busy GPUs' queues automatically
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