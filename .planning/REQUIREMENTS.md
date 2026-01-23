# Requirements: VirNucPro GPU Optimization

**Defined:** 2026-01-22
**Core Value:** Embedding steps (DNABERT-S and ESM-2) efficiently utilize all available GPUs and automatically queue batches, reducing sample processing time from 45+ hours to under 10 hours.

## v1 Requirements

Requirements for GPU optimization release. Each maps to roadmap phases.

### GPU Parallelization

- [x] **ESM-01**: ESM-2 parallelizes across multiple GPUs using PyTorch DDP
- [x] **ESM-02**: ESM-2 automatically queues and processes multiple batches per GPU without manual intervention
- [x] **DNABERT-01**: DNABERT-S uses optimized batching (better than current one-file-per-GPU)
- [x] **DNABERT-02**: DNABERT-S automatically queues and processes multiple batches per GPU
- [x] **GPU-01**: Mixed precision (BF16) enabled for both ESM-2 and DNABERT-S (2x speedup)
- [x] **GPU-02**: Batch sizes optimized via profiling for target GPUs (2-4x increase from current)

### Attention & Memory Optimization

- [ ] **ATT-01**: FlashAttention-2 integrated for ESM-2 embeddings (2-4x attention speedup)
- [ ] **ATT-02**: FlashAttention-2 integrated for DNABERT-S embeddings
- [ ] **MEM-01**: DataLoader prefetching with optimized worker count (num_workers=4-8)
- [ ] **MEM-02**: CUDA streams for async I/O overlap (hide 20-40% latency)
- [ ] **MEM-03**: Memory fragmentation prevention via sequence sorting
- [ ] **MEM-04**: Memory fragmentation prevention via expandable segments (PYTORCH_CUDA_ALLOC_CONF)
- [ ] **MEM-05**: Periodic CUDA cache clearing between file batches

### Infrastructure & Checkpointing

- [x] **INFRA-01**: Batch queue manager coordinates work distribution across GPUs
- [x] **INFRA-02**: GPU worker pool with spawn context (CUDA-safe multiprocessing)
- [x] **INFRA-03**: Checkpoint integration with .done markers (distinguish complete vs in-progress)
- [x] **INFRA-04**: Atomic file writes via temp-then-rename pattern (prevent corruption)
- [x] **INFRA-05**: Checkpoint validation checks file size >0 and optionally validates keys

### Monitoring & Load Balancing

- [ ] **MON-01**: GPU utilization monitoring via nvitop logs compute % and memory usage
- [ ] **MON-02**: Throughput logging tracks sequences/sec per GPU
- [ ] **MON-03**: Monitoring detects stalled workers or imbalanced load
- [x] **LOAD-01**: Load-balanced file assignment using greedy bin packing by sequence count
- [x] **LOAD-02**: Checkpoint versioning with migration functions for backward compatibility
- [ ] **LOAD-03**: Heterogeneous GPU support (handle mixed GPU types/memory sizes)

### Performance & Scalability

- [ ] **PERF-01**: Pipeline completes one sample (thousands of sequences) in under 10 hours
- [ ] **PERF-02**: GPU utilization metrics show >80% GPU usage during embedding steps
- [ ] **SCALE-01**: Adding GPUs provides linear speedup (2x GPUs = ~2x faster, measured)
- [x] **SCALE-02**: Works with variable GPU counts (1, 4, 8, or any number of GPUs)

### Compatibility

- [x] **COMPAT-01**: CLI interface unchanged (users run same `virnucpro predict` command)
- [x] **COMPAT-02**: Can resume checkpoints from pre-optimization runs (backward compatible)
- [x] **COMPAT-03**: Single-GPU fallback mode works when only one GPU available

### Testing

- [x] **TEST-01**: ESM-2 worker unit tests verify model loading, batching, and output format
- [x] **TEST-02**: ESM-2 multi-GPU integration test compares output against single-GPU baseline
- [x] **TEST-03**: DNABERT-S worker unit tests verify optimized batching matches vanilla output
- [x] **TEST-04**: Checkpoint unit tests verify atomic writes, corruption handling, and resume capability
- [ ] **TEST-05**: Memory/attention unit tests verify FlashAttention integration and fragmentation prevention
- [ ] **TEST-06**: End-to-end integration test compares full pipeline output (optimized vs vanilla)

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
| ESM-01 | Phase 1 | Complete |
| ESM-02 | Phase 1 | Complete |
| DNABERT-01 | Phase 2 | Complete |
| DNABERT-02 | Phase 2 | Complete |
| GPU-01 | Phase 1 | Complete |
| GPU-02 | Phase 2 | Complete |
| ATT-01 | Phase 4 | Pending |
| ATT-02 | Phase 4 | Pending |
| MEM-01 | Phase 4 | Pending |
| MEM-02 | Phase 4 | Pending |
| MEM-03 | Phase 4 | Pending |
| MEM-04 | Phase 4 | Pending |
| MEM-05 | Phase 4 | Pending |
| INFRA-01 | Phase 1 | Complete |
| INFRA-02 | Phase 1 | Complete |
| INFRA-03 | Phase 3 | Complete |
| INFRA-04 | Phase 3 | Complete |
| INFRA-05 | Phase 3 | Complete |
| MON-01 | Phase 5 | Pending |
| MON-02 | Phase 5 | Pending |
| MON-03 | Phase 5 | Pending |
| LOAD-01 | Phase 5 | Pending |
| LOAD-02 | Phase 3 | Complete |
| LOAD-03 | Phase 5 | Pending |
| PERF-01 | Phase 6 | Pending |
| PERF-02 | Phase 6 | Pending |
| SCALE-01 | Phase 6 | Pending |
| SCALE-02 | Phase 1 | Complete |
| COMPAT-01 | Phase 1 | Complete |
| COMPAT-02 | Phase 3 | Complete |
| COMPAT-03 | Phase 1 | Complete |

| TEST-01 | Phase 1 | Complete |
| TEST-02 | Phase 1 | Complete |
| TEST-03 | Phase 2 | Complete |
| TEST-04 | Phase 3 | Complete |
| TEST-05 | Phase 4 | Pending |
| TEST-06 | Phase 6 | Pending |

**Coverage:**
- v1 requirements: 37 total
- Mapped to phases: 37 (100% coverage)
- Unmapped: 0

---
*Requirements defined: 2026-01-22*
*Last updated: 2026-01-22 after adding testing requirements*
