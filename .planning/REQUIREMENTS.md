# Requirements: VirNucPro GPU Optimization

**Defined:** 2026-02-02
**Core Value:** Embedding steps (DNABERT-S and ESM-2) efficiently utilize all available GPUs and automatically queue batches, reducing sample processing time from 45+ hours to under 10 hours.

## v2.0 Requirements

Requirements for v2.0 async architecture and sequence packing milestone. Each maps to roadmap phases.

### Architecture (Async DataLoader)

- [x] **ARCH-01**: Single GPU process per GPU (replace multi-worker-per-GPU pattern)
- [x] **ARCH-02**: Async DataLoader with CPU workers (num_workers=4-8 for I/O parallelism)
- [x] **ARCH-03**: Batch prefetching (prefetch_factor=2 to overlap data loading with compute)
- [x] **ARCH-04**: GPU memory pinning in main process only (pin_memory=True in DataLoader)
- [x] **ARCH-05**: CUDA stream processing for async I/O overlap
- [ ] **ARCH-06**: Triple-buffered async pipeline (Buffer 1: load batch N+1, Buffer 2: compute batch N, Buffer 3: write results N-1)
- [ ] **ARCH-07**: Buffer size matching GPU batch output (e.g., 256 seqs × 2560 dims × 2 bytes = 1.25MB per buffer)
- [ ] **ARCH-08**: Overflow handling - backpressure if I/O thread falls behind 3 buffers (pause DataLoader)
- [ ] **ARCH-09**: FlashAttention variable-length support (flash_attn_varlen_func for variable batch sizes)
- [ ] **ARCH-10**: SequenceDataset (IterableDataset) streams FASTA sequences with file-level sharding
- [ ] **ARCH-11**: Length-aware sharding - sort sequences by length before packing to maximize packing density (greedy bin packing)

### CUDA Safety (Critical Correctness Requirements)

- [x] **SAFE-01**: CPU workers must NOT initialize CUDA (no torch.cuda calls in worker_init_fn or Dataset.__init__). Implementation: Set CUDA_VISIBLE_DEVICES="" in worker env; validate torch.cuda.is_available() returns False in Dataset.__init__
- [x] **SAFE-02**: Use spawn method for multiprocessing (not fork - fork copies CUDA context causing errors)
- [x] **SAFE-03**: Deferred CUDA initialization in GPU processes (only after spawn, never in parent)
- [x] **SAFE-04**: Worker process validation (assert workers only do CPU work: FASTA parsing, tokenization)
- [x] **SAFE-05**: Memory pinning safety (only main process pins memory, workers return unpinned tensors)

### Sequence Packing

- [ ] **PACK-01**: FlashAttention varlen integration with cu_seqlens (prevent cross-sequence attention)
- [ ] **PACK-02**: PackingCollator (custom collate_fn) packs sequences into dense batches
- [ ] **PACK-03**: Dynamic token budget calculation based on available GPU memory and model size
- [ ] **PACK-04**: Unpacking validation - packed output matches non-packed output within tolerance (cosine similarity >0.999 or MSE <0.02)
- [ ] **PACK-04b**: Validation mode - optional FP32 ground-truth comparison for regression testing
- [ ] **PACK-05**: Cross-contamination test (position IDs, attention masks, dtype correctness)
- [ ] **PACK-06**: Outlier strategy for sequences exceeding token budget (truncate | isolate | split)
- [ ] **PACK-06a**: Sequences >max_length processed in isolated batch_size=1 with truncation warning

### Precision

- [ ] **PREC-01**: FP16 mixed precision via torch.amp.autocast
- [ ] **PREC-02**: Accuracy validation - cosine similarity >0.99 between FP16 and FP32 embeddings
- [ ] **PREC-03**: Selective FP32 for numerically unstable layers if needed (LayerNorm, softmax)

### Multi-GPU Coordination

- [ ] **GPU-01**: GPUProcessCoordinator spawns independent processes (1 per GPU)
- [ ] **GPU-02**: File-level sharding across GPUs (deterministic rank % world_size assignment)
- [ ] **GPU-03**: Checkpoint aggregation from multiple GPU outputs
- [ ] **GPU-04**: Per-GPU checkpoint markers (gpu{rank}_ESM.pt + .done markers)

### Checkpointing & Resume

- [ ] **CKPT-01**: Incremental checkpointing every 10K sequences per shard (granular resume points)
- [ ] **CKPT-02**: Resume from last checkpoint (don't restart full 6M sequences on OOM or crash)
- [ ] **CKPT-03**: Atomic shard completion markers (prevent partial file corruption during writes)
- [ ] **CKPT-04**: Atomic writes for checkpoint files (temp + rename pattern)
- [ ] **CKPT-05**: Checkpoint validation (size check, sequence count verification, optional load test)
- [ ] **CKPT-06**: Per-GPU checkpoint isolation (each GPU writes independent checkpoints without conflicts)

### Performance & Validation

- [ ] **PERF-01**: Pipeline completes one sample in under 10 hours on 4 GPUs
- [ ] **PERF-02**: GPU utilization >80% during embedding steps (validated with nvitop)
- [ ] **PERF-03**: Linear GPU scaling verified (2x GPUs ≈ 2x faster within 90-95% efficiency)
- [ ] **PERF-04**: Throughput targets: (a) With FlashAttention: 1M-2M sequences/hour per GPU; (b) Fallback (fixed-shape): 200K-400K sequences/hour per GPU
- [ ] **PERF-05**: Telemetry - Log tokens/sec, packing efficiency (% padding waste), and I/O wait time per batch

## v3.0+ Future Requirements

Deferred to future releases. Tracked but not in current roadmap.

### Advanced Optimizations

- **OPT-01**: Advanced packing algorithms (LPFHP for better bin-packing efficiency)
- **OPT-02**: Dynamic batch sizing (adaptive to sequence length distribution)
- **OPT-03**: Multi-stage pipeline overlap (translation → DNABERT → ESM async streaming)
- **OPT-04**: torch.compile integration for kernel fusion
- **OPT-05**: Continuous batching (vLLM-style iteration-level scheduling)

### Security & Maintenance

- **SEC-01**: Upgrade transformers to 4.53.0+ (address 12 CVEs including 4 RCE vulnerabilities)
- **SEC-02**: Dependency audit and version pinning

### Load Balancing

- **LOAD-01**: Work-stealing queue for dynamic load balancing
- **LOAD-02**: Heterogeneous GPU weighting (handle mixed GPU types)

## Out of Scope

| Feature | Reason |
|---------|--------|
| Backward compatibility with v1.0 CLI | v2.0 is major refactor - breaking changes acceptable |
| Optimizing non-embedding pipeline stages | Embeddings are bottleneck, other stages fast enough |
| Distributed multi-node processing | Single-machine multi-GPU sufficient for target workloads |
| CPU-only optimization | GPUs are the target environment |
| BF16 precision | Research shows FP16 is optimal for ESM-2 (model trained in FP16) |
| Maintaining v1.0 multi-worker architecture | v2.0 replaces it entirely (v1.x stable branch if needed) |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| ARCH-01 | Phase 5 | Pending |
| ARCH-02 | Phase 5 | Pending |
| ARCH-03 | Phase 5 | Pending |
| ARCH-04 | Phase 5 | Pending |
| ARCH-05 | Phase 5 | Pending |
| ARCH-06 | Phase 7 | Pending |
| ARCH-07 | Phase 7 | Pending |
| ARCH-08 | Phase 7 | Pending |
| ARCH-09 | Phase 6 | Pending |
| ARCH-10 | Phase 6 | Pending |
| ARCH-11 | Phase 6 | Pending |
| SAFE-01 | Phase 5 | Pending |
| SAFE-02 | Phase 5 | Pending |
| SAFE-03 | Phase 5 | Pending |
| SAFE-04 | Phase 5 | Pending |
| SAFE-05 | Phase 5 | Pending |
| PACK-01 | Phase 6 | Pending |
| PACK-02 | Phase 6 | Pending |
| PACK-03 | Phase 6 | Pending |
| PACK-04 | Phase 6 | Pending |
| PACK-04b | Phase 6 | Pending |
| PACK-05 | Phase 6 | Pending |
| PACK-06 | Phase 6 | Pending |
| PACK-06a | Phase 6 | Pending |
| PREC-01 | Phase 8 | Pending |
| PREC-02 | Phase 8 | Pending |
| PREC-03 | Phase 8 | Pending |
| GPU-01 | Phase 7 | Pending |
| GPU-02 | Phase 7 | Pending |
| GPU-03 | Phase 7 | Pending |
| GPU-04 | Phase 7 | Pending |
| CKPT-01 | Phase 9 | Pending |
| CKPT-02 | Phase 9 | Pending |
| CKPT-03 | Phase 9 | Pending |
| CKPT-04 | Phase 9 | Pending |
| CKPT-05 | Phase 9 | Pending |
| CKPT-06 | Phase 9 | Pending |
| PERF-01 | Phase 10 | Pending |
| PERF-02 | Phase 10 | Pending |
| PERF-03 | Phase 10 | Pending |
| PERF-04 | Phase 10 | Pending |
| PERF-05 | Phase 10 | Pending |

**Coverage:**
- v2.0 requirements: 42 total
- Mapped to phases: 42/42 (100%)
- Unmapped: 0

**By Phase:**
- Phase 5: 10 requirements (ARCH-01 to ARCH-05, SAFE-01 to SAFE-05)
- Phase 6: 11 requirements (ARCH-09 to ARCH-11, PACK-01 to PACK-06a)
- Phase 7: 7 requirements (ARCH-06 to ARCH-08, GPU-01 to GPU-04)
- Phase 8: 3 requirements (PREC-01 to PREC-03)
- Phase 9: 6 requirements (CKPT-01 to CKPT-06)
- Phase 10: 5 requirements (PERF-01 to PERF-05)

---
*Requirements defined: 2026-02-02*
*Last updated: 2026-02-03 with phase mappings for v2.0 roadmap*
