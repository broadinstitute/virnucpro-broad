# Requirements: VirNucPro v2.5 Model Optimizations Round 2

**Defined:** 2026-02-09
**Core Value:** Embedding steps (DNABERT-S and ESM-2) efficiently utilize all available GPUs with async DataLoader and sequence packing, delivering 6.2x speedup over v1.0 baseline.

## v2.5 Requirements

### DNABERT-S v2.0 Port

- [ ] **DNABERT-11**: DNABERT-S uses async DataLoader with CPU workers for I/O (replacing v1.0 bin-packing)
- [ ] **DNABERT-12**: DNABERT-S uses sequence packing via VarlenCollator with k-mer tokenization
- [ ] **DNABERT-13**: DNABERT-S uses FlashAttention varlen for packed attention (MosaicBERT integration)
- [ ] **DNABERT-14**: DNABERT-S scales near-linearly on multi-GPU (>1.5x on 2 GPUs, up from 0.96x)
- [ ] **DNABERT-15**: DNABERT-S v2.0 embeddings match v1.0 within cosine similarity >0.99
- [ ] **DNABERT-16**: DNABERT-S v2.0 integrated into CLI and full pipeline (replaces v1.0 path)

### ESM-2 Model Flexibility

- [ ] **ESM-10**: User can select ESM-2 model via `--esm-model` CLI flag (650M, 3B, or custom path)
- [ ] **ESM-11**: repr_layers auto-detected from model architecture (no hardcoding)
- [ ] **ESM-12**: Hidden dimension auto-detected for downstream pipeline compatibility
- [ ] **ESM-13**: Checkpoint metadata includes model name; resume validates model compatibility
- [ ] **ESM-14**: Default model remains esm2_t36_3B_UR50D for backward compatibility

### Performance Optimizations

- [ ] **PERF-10**: Environment variables cached at initialization (not read per-batch)
- [ ] **PERF-11**: packed_queue uses collections.deque instead of list (O(1) popleft)
- [ ] **PERF-12**: Model loading combines .to(device) and .half() in single transfer
- [ ] **PERF-13**: torch.compile available via `--compile` CLI flag (default OFF, 10-20% speedup)
- [ ] **PERF-14**: Position ID generation vectorized (no Python loop with .item() calls)
- [ ] **PERF-15**: Embedding extraction optimized (minimize CPU-GPU sync points)

### Code Quality

- [ ] **QUAL-01**: Environment variables centralized in EnvConfig dataclass
- [ ] **QUAL-02**: Duplicate code extracted to shared utilities (_validate_cuda_isolation, _get_progress_queue)
- [ ] **QUAL-03**: async_inference.run() refactored into focused methods (<100 lines each)
- [ ] **QUAL-04**: gpu_worker() refactored into focused helper functions
- [ ] **QUAL-05**: prediction.run_prediction() refactored into focused methods

## Future Requirements

### v3.0 (Deferred)

- **LOAD-01**: Work-stealing queue for dynamic load balancing
- **STREAM-01**: True pipeline parallelism with triple buffering (15-25% improvement)
- **SEC-01**: Upgrade transformers to 4.53.0+ (address 12 CVEs including 4 RCE)
- **FA3-01**: FlashAttention-3 support for Hopper GPUs (1.5-2x attention speedup)

## Out of Scope

| Feature | Reason |
|---------|--------|
| Dynamic work-stealing queue | Multi-GPU scaling adequate with stride sharding; defer to v3.0 |
| BFD packing algorithm | FFD achieves 92-94% already; 2-3% gain not worth complexity |
| True pipeline parallelism (triple buffering) | High effort stream optimization; defer to v3.0 |
| FlashAttention-3 support | Requires Hopper GPUs not in current hardware |
| Tensor pooling | Memory fragmentation not a measured bottleneck |
| Security upgrade (transformers 4.53+) | Orthogonal concern, separate milestone |
| Model weight sharing across GPUs | Complex, marginal benefit with current process model |
| Process pooling | Sequential spawn overhead <1s per worker; negligible |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| DNABERT-11 | TBD | Pending |
| DNABERT-12 | TBD | Pending |
| DNABERT-13 | TBD | Pending |
| DNABERT-14 | TBD | Pending |
| DNABERT-15 | TBD | Pending |
| DNABERT-16 | TBD | Pending |
| ESM-10 | TBD | Pending |
| ESM-11 | TBD | Pending |
| ESM-12 | TBD | Pending |
| ESM-13 | TBD | Pending |
| ESM-14 | TBD | Pending |
| PERF-10 | TBD | Pending |
| PERF-11 | TBD | Pending |
| PERF-12 | TBD | Pending |
| PERF-13 | TBD | Pending |
| PERF-14 | TBD | Pending |
| PERF-15 | TBD | Pending |
| QUAL-01 | TBD | Pending |
| QUAL-02 | TBD | Pending |
| QUAL-03 | TBD | Pending |
| QUAL-04 | TBD | Pending |
| QUAL-05 | TBD | Pending |

**Coverage:**
- v2.5 requirements: 22 total
- Mapped to phases: 0
- Unmapped: 22 (awaiting roadmap)

---
*Requirements defined: 2026-02-09*
*Last updated: 2026-02-09 after initial definition*
