# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-02)

**Core value:** Embedding steps (DNABERT-S and ESM-2) efficiently utilize all available GPUs and automatically queue batches, reducing sample processing time from 45+ hours to under 10 hours.

**Current focus:** Phase 5 - Async DataLoader Foundation

## Current Position

Phase: 5 of 10 (Async DataLoader Foundation)
Plan: 3 of TBD in current phase
Status: In progress
Last activity: 2026-02-03 — Completed 05-03-PLAN.md

Progress: [████░░░░░░] 37/TBD plans (v1.0: 34/34 complete, v2.0: 3/TBD)

## Performance Metrics

**Velocity:**
- Total plans completed: 37 (v1.0: 34, v2.0: 3)
- Average duration: 3.2 min
- Total execution time: 2.4 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 1 | 5 | 17 min | 3.4 min |
| 2 | 7 | 24 min | 3.4 min |
| 3 | 7 | 24 min | 3.4 min |
| 4 | 12 | 41 min | 3.4 min |
| 4.1 | 3 | 10 min | 3.3 min |
| 5 | 3 | 7.1 min | 2.4 min |

**Recent Trend:**
- Last 5 plans: ~2.7 min average
- Trend: Improving (faster execution in phase 5)

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- **v2.0 Async architecture**: Multi-worker-per-GPU causes N×11GB memory overhead, serialization tax, GPU starvation - replacing with single-process-per-GPU + async DataLoader (major refactor, breaking changes acceptable)
- **FP16 over BF16**: Research shows ESM-2 trained in FP16 (norm difference <1e-3 vs FP32) - FP16 is optimal for this model
- **FlashAttention varlen**: Required for sequence packing to prevent cross-sequence attention contamination
- **File-level sharding**: Multi-GPU coordination uses deterministic file assignment (rank % world_size) instead of dynamic work-stealing
- **Queue state heuristic (05-02)**: PyTorch DataLoader doesn't expose queue depth - infer from wait_time_ms (<1ms=full, >50ms=starved)
- **Tiered bottleneck thresholds (05-02)**: <50% critical, <80% mild, ≥80% none - avoids false positives with short sequences
- **Dual throughput metrics (05-02)**: Track both sequences/sec and tokens/sec - tokens/sec more stable for packed batches
- **CUDA validation timing (05-01)**: Validate in __iter__ not __init__ because workers spawned during iteration, not Dataset creation
- **ESM padding stripping (05-01)**: Must strip padding_idx tokens before packing to prevent contamination in packed format
- **cu_seqlens format (05-01)**: Cumulative boundaries [0, len1, len1+len2, ...] with N+1 elements for N sequences
- **batch_size=None for VarlenCollator (05-03)**: Allows VarlenCollator to control packing via token budget instead of fixed batch size
- **timeout=600s for FASTA parsing (05-03)**: Increased from default 5 min to handle large FASTA files without timeout errors
- **prefetch_factor=4 (05-03)**: Aggressive prefetching (4×4=16 batches) saturates GPU even with occasional slow I/O

### Pending Todos

None yet.

### Blockers/Concerns

**Phase 5 (Async DataLoader Foundation):**
- ✅ CUDA worker safety: RESOLVED in 05-01 - SequenceDataset validates CUDA isolation in __iter__ (workers spawn during iteration)
- ✅ DataLoader CUDA isolation: RESOLVED in 05-03 - cuda_safe_worker_init() sets CUDA_VISIBLE_DEVICES='' and validates no CUDA
- HuggingFace cache race: Not applicable for 05-01 (workers yield raw strings, no model loading) - relevant for future tokenizer integration
- ⚠️ Persistent worker memory leaks: prefetch_factor=4 chosen for GPU saturation (05-03 decision overrides 05-02 concern - accepted trade-off for inference performance)
- **ONGOING**: Test environment setup needed - torch/esm not available in current Python environment (verification tests skipped in 05-01, 05-03)

**Phase 6 (Sequence Packing Integration):**
- FlashAttention varlen API: fair-esm 2.0.0 may require manual integration of flash_attn_varlen_func - investigate esm2_flash.py wrapper layer
- Position ID off-by-one: Must reset position IDs to 0 at each cu_seqlens boundary (not sequential [0,1,2,3,4,5])
- Packing correctness: Need extensive validation (packed output == unpacked output for same sequences)

**Phase 8 (FP16 Precision Validation):**
- Numerical precision: LayerNorm may have limited dynamic range in FP16 - may need selective FP32 for specific layers while keeping rest in FP16

## Session Continuity

Last session: 2026-02-03T15:52:54Z
Stopped at: Completed 05-03-PLAN.md (Async DataLoader Factory)
Resume file: None

**Next step:** Continue with next plan in phase 5 (likely 05-04: Async Inference Runner)
