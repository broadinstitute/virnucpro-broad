# Roadmap: VirNucPro GPU Optimization

## Milestones

- ✅ **v1.0 GPU Optimization Foundation** - Phases 1-4.1 (shipped 2026-02-02)
- ✅ **v2.0 Async Architecture + Sequence Packing** - Phases 5-10 + 10.1, 10.2 (shipped 2026-02-09)
- 📋 **v2.5 Model Optimizations Round 2** - Phases 11-17 (planned)

## Phases

<details>
<summary>✅ v1.0 GPU Optimization Foundation (Phases 1-4.1) - SHIPPED 2026-02-02</summary>

### Phase 1: Multi-GPU ESM-2
**Goal**: Parallelize ESM-2 embedding extraction across multiple GPUs
**Plans**: 5 plans

### Phase 2: DNABERT-S Optimization
**Goal**: Optimize DNABERT-S with BF16 and dynamic batching
**Plans**: 7 plans

### Phase 3: FlashAttention Integration
**Goal**: Integrate FlashAttention-2 for 2-4x attention speedup
**Plans**: 7 plans

### Phase 4: Integration & Validation
**Goal**: End-to-end validation and production readiness
**Plans**: 12 plans

### Phase 4.1: Checkpoint Robustness
**Goal**: Atomic writes and .done markers for checkpoint reliability
**Plans**: 3 plans

</details>

<details>
<summary>✅ v2.0 Async Architecture + Sequence Packing (Phases 5-10) - SHIPPED 2026-02-09</summary>

### Phase 5: Async DataLoader Foundation
**Goal**: Single-process-per-GPU with async DataLoader and CUDA safety
**Plans**: 5 plans

### Phase 6: Sequence Packing Integration
**Goal**: FlashAttention varlen with FFD packing (~92-94% efficiency)
**Plans**: 8 plans

### Phase 7: Multi-GPU Coordination
**Goal**: Stride-based sharding and worker orchestration
**Plans**: 8 plans

### Phase 8: FP16 Precision Validation
**Goal**: FP16 precision with NaN/Inf detection and >0.99 cosine similarity
**Plans**: 4 plans

### Phase 9: Checkpointing Integration
**Goal**: Fault-tolerant checkpointing with elastic redistribution
**Plans**: 7 plans

### Phase 10.1: CLI Integration for v2.0 Architecture (INSERTED)
**Goal**: Wire v2.0 async architecture into CLI (hybrid ESM-2 v2.0 + DNABERT-S v1.0)
**Plans**: 4 plans

### Phase 10.2: FlashAttention Scoring Divergence Resolution (INSERTED)
**Goal**: Validate v2.0 FlashAttention correctness and provide v1.0-compatible fallback
**Plans**: 2 plans

### Phase 10: Performance Validation & Tuning
**Goal**: End-to-end benchmarks validating 6.2x speedup over v1.0
**Plans**: 3 plans

</details>

### 📋 v2.5 Model Optimizations Round 2 (In Planning)

**Milestone Goal:** Port DNABERT-S to v2.0 async architecture, add configurable ESM-2 model selection, and apply targeted performance optimizations from the v2.0 review.

#### Phase 11: Code Quality Foundations
**Goal**: Environment variable centralization and function extraction for maintainability
**Depends on**: Nothing (first phase)
**Requirements**: QUAL-01, QUAL-02, QUAL-03, QUAL-04, QUAL-05
**Success Criteria** (what must be TRUE):
  1. Environment variables accessed via centralized EnvConfig dataclass (not scattered os.getenv)
  2. async_inference.run() refactored into focused methods under 100 lines each
  3. gpu_worker() refactored into component helper functions
  4. Queue operations use collections.deque for O(1) popleft performance
  5. All existing tests pass with refactored code (1:1 behavior equivalence)
**Plans**: 5 plans

Plans:
- [ ] 11-01-PLAN.md -- Create EnvConfig dataclass with cached factory (TDD)
- [ ] 11-02-PLAN.md -- Extract duplicate CUDA validation + deque migration
- [ ] 11-03-PLAN.md -- Refactor async_inference.run() into focused methods
- [ ] 11-04-PLAN.md -- Refactor gpu_worker() into component functions
- [ ] 11-05-PLAN.md -- Migrate remaining env var sites + full test validation

#### Phase 12: ESM-2 Model Flexibility
**Goal**: Configurable ESM-2 model selection (650M, 3B, custom) with auto-detected configuration
**Depends on**: Phase 11 (env config for feature flags)
**Requirements**: ESM-10, ESM-11, ESM-12, ESM-13, ESM-14
**Success Criteria** (what must be TRUE):
  1. User can select ESM-2 model via --esm-model CLI flag (650M, 3B, or custom path)
  2. repr_layers automatically detected from model architecture (no hardcoding)
  3. Hidden dimension automatically detected for downstream compatibility
  4. Checkpoint metadata includes model name and validates compatibility on resume
  5. Default model remains esm2_t36_3B_UR50D (backward compatibility)
**Plans**: 3-4 plans

Plans:
- [ ] 12-01: Create ESM2Config dataclass with auto-detection logic
- [ ] 12-02: Update load_esm2_model to return (model, batch_converter, config)
- [ ] 12-03: Add --esm-model CLI flag with model variant choices
- [ ] 12-04: Add checkpoint metadata validation for model compatibility

#### Phase 13: Performance Quick Wins
**Goal**: Low-hanging fruit optimizations (env var caching, deque, combined tensor ops)
**Depends on**: Phase 11 (env config enables caching)
**Requirements**: PERF-10, PERF-11, PERF-12
**Success Criteria** (what must be TRUE):
  1. Environment variables cached at initialization (not read per-batch)
  2. packed_queue uses collections.deque for O(1) popleft
  3. Model loading combines .to(device) and .half() in single transfer
**Plans**: 2-3 plans

Plans:
- [ ] 13-01: Implement env var caching via EnvConfig
- [ ] 13-02: Replace packed_queue list with deque
- [ ] 13-03: Combine model .to(device) and .half() calls

#### Phase 14: Vectorized Operations
**Goal**: Vectorize position ID generation and embedding extraction hot paths
**Depends on**: Phase 11 (clean code for integration)
**Requirements**: PERF-14, PERF-15
**Success Criteria** (what must be TRUE):
  1. Position ID generation uses torch.cumsum (no Python loops with .item())
  2. Embedding extraction optimized with cu_seqlens_cpu caching (minimize CUDA syncs)
  3. Hybrid implementation (loop for small batches, vectorized for large)
  4. Edge cases handled (empty sequences, single-token, boundaries)
**Plans**: 2-3 plans

Plans:
- [ ] 14-01: Vectorize create_position_ids_packed with torch.cumsum
- [ ] 14-02: Optimize embedding extraction with cu_seqlens_cpu caching
- [ ] 14-03: Add property-based tests for edge cases

#### Phase 15: torch.compile Integration (Experimental)
**Goal**: Optional torch.compile for 10-20% speedup with FlashAttention compatibility validation
**Depends on**: Phase 12 (test with multiple models)
**Requirements**: PERF-13
**Success Criteria** (what must be TRUE):
  1. torch.compile available via --torch-compile CLI flag (default OFF)
  2. Compilation mode 'reduce-overhead' with dynamic=True for variable shapes
  3. FlashAttention remains active under compilation (integration test verified)
  4. Recompilation monitoring and cache size tracking
  5. VIRNUCPRO_DISABLE_COMPILE rollback mechanism
**Plans**: 3-4 plans

Plans:
- [ ] 15-01: Add torch.compile wrapper in load_esm2_model
- [ ] 15-02: Add --torch-compile CLI flag and warmup phase
- [ ] 15-03: Create integration test verifying FlashAttention active
- [ ] 15-04: Add recompilation monitoring and cache tracking

#### Phase 16: DNABERT-S v2.0 Port (HIGH RISK)
**Goal**: Port DNABERT-S from v1.0 bin-packing to v2.0 async architecture for >1.5x scaling
**Depends on**: Phases 11-14 (clean code, model flexibility, vectorization)
**Requirements**: DNABERT-11, DNABERT-12, DNABERT-13, DNABERT-14, DNABERT-15, DNABERT-16
**Success Criteria** (what must be TRUE):
  1. DNABERT-S uses async DataLoader with CPU workers for I/O
  2. DNABERT-S uses sequence packing via VarlenCollator with k-mer tokenization
  3. DNABERT-S uses FlashAttention varlen for packed attention
  4. DNABERT-S scales >1.5x on 2 GPUs (up from 0.96x v1.0 baseline)
  5. DNABERT-S v2.0 embeddings match v1.0 within cosine similarity >0.99
  6. DNABERT-S v2.0 integrated into CLI and full pipeline
**Plans**: 6-9 plans

Plans:
- [ ] 16-01: Extract v1.0 reference embeddings for validation
- [ ] 16-02: Create DnabertVarlenCollator with transformers tokenizer
- [ ] 16-03: Implement DNABERTWithFlashAttention wrapper
- [ ] 16-04: Test unpacked equivalence (v2.0 standard forward vs v1.0)
- [ ] 16-05: Test packed equivalence (forward_packed vs standard forward)
- [ ] 16-06: Integrate DNABERT-S into gpu_worker and cli
- [ ] 16-07: End-to-end correctness test (v2.0 vs v1.0 cosine >0.99)
- [ ] 16-08: Multi-GPU scaling validation (>1.5x on 2 GPUs)
- [ ] 16-09: Add --dnabert-v1-fallback CLI flag

#### Phase 17: Final Integration & Validation
**Goal**: Full pipeline benchmarks with all v2.5 features enabled
**Depends on**: Phase 16 (DNABERT-S port complete)
**Requirements**: All 22 v2.5 requirements (integration validation)
**Success Criteria** (what must be TRUE):
  1. Full pipeline runs on 1M subset with all features enabled
  2. Multi-GPU scaling validation on 2x RTX 4090
  3. Performance regression suite (compare v2.0 baseline)
  4. No slowdown vs v2.0 baseline (v2.5 >= 1.0x)
  5. DNABERT-S achieves >1.5x scaling (up from 0.96x)
**Plans**: 2-3 plans

Plans:
- [ ] 17-01: Full pipeline benchmark on 1M subset (1x GPU)
- [ ] 17-02: Multi-GPU scaling validation (2x RTX 4090)
- [ ] 17-03: Performance regression suite and final validation

## Progress

**Execution Order:**
Phases execute in numeric order: 11 → 12 → 13 → 14 → 15 → 16 → 17

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 1. Multi-GPU ESM-2 | v1.0 | 5/5 | Complete | 2026-02-02 |
| 2. DNABERT-S Optimization | v1.0 | 7/7 | Complete | 2026-02-02 |
| 3. FlashAttention Integration | v1.0 | 7/7 | Complete | 2026-02-02 |
| 4. Integration & Validation | v1.0 | 12/12 | Complete | 2026-02-02 |
| 4.1. Checkpoint Robustness | v1.0 | 3/3 | Complete | 2026-02-02 |
| 5. Async DataLoader Foundation | v2.0 | 5/5 | Complete | 2026-02-09 |
| 6. Sequence Packing Integration | v2.0 | 8/8 | Complete | 2026-02-09 |
| 7. Multi-GPU Coordination | v2.0 | 8/8 | Complete | 2026-02-09 |
| 8. FP16 Precision Validation | v2.0 | 4/4 | Complete | 2026-02-09 |
| 9. Checkpointing Integration | v2.0 | 7/7 | Complete | 2026-02-09 |
| 10.1. CLI Integration | v2.0 | 4/4 | Complete | 2026-02-09 |
| 10.2. FlashAttention Divergence | v2.0 | 2/2 | Complete | 2026-02-09 |
| 10. Performance Validation | v2.0 | 3/3 | Complete | 2026-02-09 |
| 11. Code Quality Foundations | v2.5 | 5/5 | Complete | 2026-02-10 |
| 12. ESM-2 Model Flexibility | v2.5 | 0/4 | Not started | - |
| 13. Performance Quick Wins | v2.5 | 0/3 | Not started | - |
| 14. Vectorized Operations | v2.5 | 0/3 | Not started | - |
| 15. torch.compile Integration | v2.5 | 0/4 | Not started | - |
| 16. DNABERT-S v2.0 Port | v2.5 | 0/9 | Not started | - |
| 17. Final Integration | v2.5 | 0/3 | Not started | - |
