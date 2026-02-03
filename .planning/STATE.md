# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-02)

**Core value:** Embedding steps (DNABERT-S and ESM-2) efficiently utilize all available GPUs and automatically queue batches, reducing sample processing time from 45+ hours to under 10 hours.

**Current focus:** Phase 6 - Sequence Packing Integration

## Current Position

Phase: 6 of 10 (Sequence Packing Integration)
Plan: 5 of 8 in current phase
Status: In progress
Last activity: 2026-02-03 — Completed 06-04-PLAN.md

Progress: [█████░░░░░] 44/TBD plans (v1.0: 34/34 complete, v2.0: 10/TBD)

## Performance Metrics

**Velocity:**
- Total plans completed: 44 (v1.0: 34, v2.0: 10)
- Average duration: 2.9 min
- Total execution time: 2.9 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 1 | 5 | 17 min | 3.4 min |
| 2 | 7 | 24 min | 3.4 min |
| 3 | 7 | 24 min | 3.4 min |
| 4 | 12 | 41 min | 3.4 min |
| 4.1 | 3 | 10 min | 3.3 min |
| 5 | 5 | 13 min | 2.6 min |
| 6 | 5 | 19 min | 3.8 min |

**Recent Trend:**
- Last 5 plans (Phase 6): ~2.8 min average
- Trend: Faster (streamlined implementation, focused scope)

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
- **FFD packing algorithm (06-01)**: First-Fit Decreasing sorts sequences by length descending for ~92-94% efficiency vs ~70% unsorted - critical for greedy bin packing (ARCH-11)
- **Buffer-based packing (06-01)**: Optimized for 1000-5000 sequence buffers, not DataLoader micro-batches - efficiency scales with buffer size
- **N-terminal truncation (06-01)**: Oversized sequences truncated preserving N-terminal region where ESM-2 biological signal is strongest
- **Dynamic token budget (06-01)**: Use torch.cuda.get_device_properties with safety margins for GPU-specific batch sizing (PACK-03)
- **Position ID reset at boundaries (06-02)**: Position IDs must reset to 0 at each cu_seqlens boundary for correct positional embeddings in packed format
- **FlashAttention dtype validation (06-02)**: Validate FP16/BF16 before calling flash_attn_varlen_func - provides clear error messages vs cryptic kernel errors
- **flash-attn version check (06-02)**: Warn if <2.6.0 but don't block - allows testing while encouraging upgrade for bug fixes
- **Layer-level FlashAttention integration (06-03)**: Extract Q/K/V from in_proj_weight and wrap attention with flash_attn_varlen_wrapper - avoids modifying ESM layer classes
- **Unpack/repack fallback (06-03)**: When FlashAttention unavailable, unpack to 2D padded, run standard forward, repack to 1D - ensures universal compatibility
- **Stateful collator with buffer (06-08)**: VarlenCollator maintains buffer and packed_queue for streaming architecture - accumulates 2000 sequences before packing (PACK-02)
- **flush() for completeness (06-08)**: Collator.flush() ensures no data loss at end-of-dataset by packing remaining buffer and queue
- **Dataloader dynamic budget (06-08)**: create_async_dataloader calculates token budget from GPU memory when token_budget=None (PACK-03 integration)
- **VIRNUCPRO_DISABLE_PACKING env var (06-04)**: Emergency rollback mechanism to disable packing in production without code changes
- **1D packed embedding extraction (06-04)**: Packed format is [total_tokens, hidden_dim] not [batch, seq, hidden] - no batch dimension indexing
- **Buffer flush protocol (06-04)**: After DataLoader exhaustion, flush collator buffer to prevent data loss for remaining sequences

### Pending Todos

None yet.

### Blockers/Concerns

**Phase 5 (Async DataLoader Foundation):** ✅ COMPLETE
- All CUDA safety mechanisms implemented and tested
- Integration tests validated on GPU server (7/9 pass, 2 expected Phase 6 failures)
- Phase 6 gate working correctly (packed format raises NotImplementedError)

**Phase 6 (Sequence Packing Integration):**
- ✅ Position ID generator: Implemented with boundary reset validation (06-02)
- ✅ FlashAttention wrapper: Implemented with dtype/format validation (06-02)
- ✅ ESM-2 forward pass: forward_packed method with layer-level FlashAttention varlen integration (06-03)
- ✅ Packed inference path: Wired in AsyncInferenceRunner with forward_packed call (06-04)
- ✅ Embedding extraction: Fixed 1D packed format handling (06-04)
- ⏳ Packing correctness: Need validation tests (packed == unpacked output for same sequences) (06-05)

**Phase 8 (FP16 Precision Validation):**
- Numerical precision: LayerNorm may have limited dynamic range in FP16 - may need selective FP32 for specific layers while keeping rest in FP16

## Session Continuity

Last session: 2026-02-03
Stopped at: Completed 06-04-PLAN.md - Wired packed inference path with forward_packed and buffer flush
Resume file: None

**Next step:** Continue with remaining Phase 6 plans (06-05 through 06-07) to complete sequence packing integration
