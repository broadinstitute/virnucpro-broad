# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-02)

**Core value:** Embedding steps (DNABERT-S and ESM-2) efficiently utilize all available GPUs and automatically queue batches, reducing sample processing time from 45+ hours to under 10 hours.

**Current focus:** Phase 6 - Sequence Packing Integration

## Current Position

Phase: 6 of 10 (Sequence Packing Integration)
Plan: 0 of TBD in current phase
Status: Ready to plan
Last activity: 2026-02-03 — Phase 5 complete

Progress: [█████░░░░░] 39/TBD plans (v1.0: 34/34 complete, v2.0: 5/TBD)

## Performance Metrics

**Velocity:**
- Total plans completed: 39 (v1.0: 34, v2.0: 5)
- Average duration: 3.1 min
- Total execution time: 2.5 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 1 | 5 | 17 min | 3.4 min |
| 2 | 7 | 24 min | 3.4 min |
| 3 | 7 | 24 min | 3.4 min |
| 4 | 12 | 41 min | 3.4 min |
| 4.1 | 3 | 10 min | 3.3 min |
| 5 | 5 | 13 min | 2.6 min |

**Recent Trend:**
- Last 5 plans (Phase 5): ~2.6 min average
- Trend: Faster (parallel execution, streamlined testing)

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
- **Inter-batch arrival timing (05-04)**: Measure time BEFORE fetching batch (not processing time) for queue state heuristic
- **Single GPU transfer (05-04)**: Use gpu_batch_ref to prevent double transfer of cu_seqlens and other tensors
- **Pinned memory validation (05-04)**: Check tensors on first batch to detect misconfiguration early
- **FP16→FP32 embeddings (05-04)**: Model may compute in FP16 but embeddings always stored in FP32 for stability
- **sequence_ids required (05-04)**: ValueError if batch missing sequence_ids, prevents synthetic ID generation bugs

### Pending Todos

None yet.

### Blockers/Concerns

**Phase 5 (Async DataLoader Foundation):** ✅ COMPLETE
- All CUDA safety mechanisms implemented and tested
- Integration tests validated on GPU server (7/9 pass, 2 expected Phase 6 failures)
- Phase 6 gate working correctly (packed format raises NotImplementedError)

**Phase 6 (Sequence Packing Integration):**
- FlashAttention varlen API: fair-esm 2.0.0 may require manual integration of flash_attn_varlen_func - investigate esm2_flash.py wrapper layer
- Position ID off-by-one: Must reset position IDs to 0 at each cu_seqlens boundary (not sequential [0,1,2,3,4,5])
- Packing correctness: Need extensive validation (packed output == unpacked output for same sequences)

**Phase 8 (FP16 Precision Validation):**
- Numerical precision: LayerNorm may have limited dynamic range in FP16 - may need selective FP32 for specific layers while keeping rest in FP16

## Session Continuity

Last session: 2026-02-03
Stopped at: Phase 5 complete - async DataLoader foundation with CUDA safety verified
Resume file: None

**Next step:** `/gsd:discuss-phase 6` to gather context for Sequence Packing Integration, or `/gsd:plan-phase 6` to plan directly
