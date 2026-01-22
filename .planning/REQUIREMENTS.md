# Requirements: VirNucPro GPU Optimization

**Defined:** 2026-01-22
**Core Value:** Embedding steps (DNABERT-S and ESM-2) efficiently utilize all available GPUs and automatically queue batches, reducing sample processing time from 45+ hours to under 10 hours.

## v1 Requirements

Requirements for GPU optimization release. Each maps to roadmap phases.

### GPU Parallelization

- [ ] **ESM-01**: ESM-2 parallelizes across multiple GPUs using PyTorch DDP
- [ ] **ESM-02**: ESM-2 automatically queues and processes multiple batches per GPU without manual intervention
- [ ] **DNABERT-01**: DNABERT-S uses optimized batching (better than current one-file-per-GPU)
- [ ] **DNABERT-02**: DNABERT-S automatically queues and processes multiple batches per GPU
- [ ] **GPU-01**: Mixed precision (BF16) enabled for both ESM-2 and DNABERT-S (2x speedup)
- [ ] **GPU-02**: Batch sizes optimized via profiling for target GPUs (2-4x increase from current)

### Attention & Memory Optimization

- [ ] **ATT-01**: FlashAttention-2 integrated for ESM-2 embeddings (2-4x attention speedup)
- [ ] **ATT-02**: FlashAttention-2 integrated for DNABERT-S embeddings
- [ ] **MEM-01**: DataLoader prefetching with optimized worker count (num_workers=4-8)
- [ ] **MEM-02**: CUDA streams for async I/O overlap (hide 20-40% latency)
- [ ] **MEM-03**: Memory fragmentation prevention via sequence sorting
- [ ] **MEM-04**: Memory fragmentation prevention via expandable segments (PYTORCH_CUDA_ALLOC_CONF)
- [ ] **MEM-05**: Periodic CUDA cache clearing between file batches

### Infrastructure & Checkpointing

- [ ] **INFRA-01**: Batch queue manager coordinates work distribution across GPUs
- [ ] **INFRA-02**: GPU worker pool with spawn context (CUDA-safe multiprocessing)
- [ ] **INFRA-03**: Checkpoint integration with .done markers (distinguish complete vs in-progress)
- [ ] **INFRA-04**: Atomic file writes via temp-then-rename pattern (prevent corruption)
- [ ] **INFRA-05**: Checkpoint validation checks file size >0 and optionally validates keys

### Monitoring & Load Balancing

- [ ] **MON-01**: GPU utilization monitoring via nvitop logs compute % and memory usage
- [ ] **MON-02**: Throughput logging tracks sequences/sec per GPU
- [ ] **MON-03**: Monitoring detects stalled workers or imbalanced load
- [ ] **LOAD-01**: Load-balanced file assignment using greedy bin packing by sequence count
- [ ] **LOAD-02**: Checkpoint versioning with migration functions for backward compatibility
- [ ] **LOAD-03**: Heterogeneous GPU support (handle mixed GPU types/memory sizes)

### Performance & Scalability

- [ ] **PERF-01**: Pipeline completes one sample (thousands of sequences) in under 10 hours
- [ ] **PERF-02**: GPU utilization metrics show >80% GPU usage during embedding steps
- [ ] **SCALE-01**: Adding GPUs provides linear speedup (2x GPUs = ~2x faster, measured)
- [ ] **SCALE-02**: Works with variable GPU counts (1, 4, 8, or any number of GPUs)

### Compatibility

- [ ] **COMPAT-01**: CLI interface unchanged (users run same `virnucpro predict` command)
- [ ] **COMPAT-02**: Can resume checkpoints from pre-optimization runs (backward compatible)
- [ ] **COMPAT-03**: Single-GPU fallback mode works when only one GPU available

## v2 Requirements

Deferred to future optimization cycles. Tracked but not in current roadmap.

### Advanced Optimizations

- **QUANT-01**: INT8 quantization for ESM-2 with embedding quality validation
- **QUANT-02**: 4-bit quantization for DNABERT-S with accuracy benchmarking
- **ZERO-01**: DeepSpeed ZeRO-Inference integration for massive batch sizes via CPU offload
- **BATCH-01**: Continuous batching (vLLM-style) for variable-length sequence scheduling
- **BATCH-02**: Work-stealing queue for perfect load balancing (vs greedy bin packing)

### Extended Monitoring

- **MON-04**: Memory leak detection in long-lived workers
- **MON-05**: Real-time performance dashboard
- **MON-06**: Automated batch size tuning based on GPU memory profiling

## Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| Multi-node distributed processing | Single-machine 4-8 GPUs sufficient, high complexity |
| Custom CUDA kernels | Extreme maintenance burden, marginal benefit over FlashAttention |
| Tensor parallelism (split layers) | ESM-2 3B fits on single GPU, not needed |
| Pipeline parallelism | Training-focused, not applicable to inference |
| Optimizing non-embedding stages | Benchmark first, current focus is embeddings only |
| CPU-only optimization | GPUs are the target environment |
| Gradient accumulation | Training technique, irrelevant for inference |
| DataParallel (vs DDP) | Deprecated, single-process bottlenecks |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| ESM-01 | TBD | Pending |
| ESM-02 | TBD | Pending |
| DNABERT-01 | TBD | Pending |
| DNABERT-02 | TBD | Pending |
| GPU-01 | TBD | Pending |
| GPU-02 | TBD | Pending |
| ATT-01 | TBD | Pending |
| ATT-02 | TBD | Pending |
| MEM-01 | TBD | Pending |
| MEM-02 | TBD | Pending |
| MEM-03 | TBD | Pending |
| MEM-04 | TBD | Pending |
| MEM-05 | TBD | Pending |
| INFRA-01 | TBD | Pending |
| INFRA-02 | TBD | Pending |
| INFRA-03 | TBD | Pending |
| INFRA-04 | TBD | Pending |
| INFRA-05 | TBD | Pending |
| MON-01 | TBD | Pending |
| MON-02 | TBD | Pending |
| MON-03 | TBD | Pending |
| LOAD-01 | TBD | Pending |
| LOAD-02 | TBD | Pending |
| LOAD-03 | TBD | Pending |
| PERF-01 | TBD | Pending |
| PERF-02 | TBD | Pending |
| SCALE-01 | TBD | Pending |
| SCALE-02 | TBD | Pending |
| COMPAT-01 | TBD | Pending |
| COMPAT-02 | TBD | Pending |
| COMPAT-03 | TBD | Pending |

**Coverage:**
- v1 requirements: 32 total
- Mapped to phases: 0 (roadmap not yet created)
- Unmapped: 32 ⚠️

---
*Requirements defined: 2026-01-22*
*Last updated: 2026-01-22 after initial definition*
