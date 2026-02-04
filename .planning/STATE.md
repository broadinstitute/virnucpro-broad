# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-02)

**Core value:** Embedding steps (DNABERT-S and ESM-2) efficiently utilize all available GPUs and automatically queue batches, reducing sample processing time from 45+ hours to under 10 hours.

**Current focus:** Phase 7 - Multi-GPU Coordination

## Current Position

Phase: 7 of 10 (Multi-GPU Coordination)
Plan: 2 of 8 in current phase
Status: In progress
Last activity: 2026-02-04 — Completed 07-02-PLAN.md (IndexBasedDataset for byte-offset sequence reading)

Progress: [█████░░░░░] 56/TBD plans (v1.0: 34/34 complete, v2.0: 22/TBD)

## Performance Metrics

**Velocity:**
- Total plans completed: 56 (v1.0: 34, v2.0: 22)
- Average duration: 3.0 min
- Total execution time: 3.35 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 1 | 5 | 17 min | 3.4 min |
| 2 | 7 | 24 min | 3.4 min |
| 3 | 7 | 24 min | 3.4 min |
| 4 | 12 | 41 min | 3.4 min |
| 4.1 | 3 | 10 min | 3.3 min |
| 5 | 5 | 13 min | 2.6 min |
| 6 | 8 | 28 min | 3.5 min |
| 7 | 2 (in progress) | 7 min | 3.5 min |

**Recent Trend:**
- Last 5 plans: ~3.0 min average
- Trend: Steady (Phase 7 infrastructure modules)

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
- **Two-tier similarity thresholds (06-05)**: Strict 0.999 for 99% sequences, lenient 0.995 for 1% - catches bugs while allowing FP16 precision variations
- **Two-tier efficiency thresholds (06-06)**: <80% critical error (packing broken), <85% warning (buffer too small) - distinguishes broken packing from suboptimal buffer sizing
- **Token utilization metric (06-06)**: Use num_tokens / max_tokens_per_batch as primary efficiency metric - simpler than padding waste calculation
- **Periodic logging interval (06-06)**: Log efficiency summary every 100 batches to avoid log spam while providing regular progress updates
- **Index-based sharding (07-01)**: Stride distribution [rank::world_size] on length-sorted index ensures balanced token load across GPUs - superior to file-level sharding
- **Descending length sort (07-01)**: Global sort across all FASTA files maximizes FFD packing efficiency - all workers get representative length distribution
- **JSON index format (07-01)**: Human-readable metadata index (~200MB for 6M sequences) enables manual inspection and debugging
- **Mtime-based cache validation (07-01)**: Compare max(fasta_mtime) > cached_mtime to detect stale index - automatic rebuild on FASTA changes
- **Byte-offset tracking (07-01)**: Record sequence byte positions in FASTA files during index creation for future random access capability
- **File grouping optimization (07-02)**: Group indices by file_path before reading to minimize file operations - improves I/O efficiency with large indices
- **Memory-buffered ordering (07-02)**: Read sequences into memory dict then yield in index order - preserves length-sorted order for FFD packing
- **CUDA validation duplication (07-02)**: Copy _validate_cuda_isolation to IndexBasedDataset - acceptable duplication for safety-critical checks

### Pending Todos

None yet.

### Blockers/Concerns

**Phase 5 (Async DataLoader Foundation):** ✅ COMPLETE
- All CUDA safety mechanisms implemented and tested
- Integration tests validated on GPU server (7/9 pass, 2 expected Phase 6 failures)
- Phase 6 gate working correctly (packed format raises NotImplementedError)

**Phase 6 (Sequence Packing Integration):** ✅ COMPLETE
- ✅ GreedyPacker with FFD algorithm and dynamic token budget (06-01)
- ✅ Position ID generator with boundary reset validation (06-02)
- ✅ FlashAttention wrapper with dtype/format validation (06-02)
- ✅ ESM-2 forward_packed method with layer-level FlashAttention (06-03)
- ✅ Packed inference path wired in AsyncInferenceRunner (06-04)
- ✅ Embedding extraction for 1D packed format (06-04)
- ✅ Packing correctness validated via cosine similarity tests (06-05)
- ✅ Packing efficiency monitoring with two-tier thresholds (06-06)
- ✅ End-to-end integration tests and human verification (06-07)
- ✅ Buffer-based packing integration in VarlenCollator (06-08)

**Phase 7 (Multi-GPU Coordination):** IN PROGRESS
- ✅ SequenceIndex with stride distribution and caching (07-01)
- ✅ IndexBasedDataset for byte-offset sequence reading (07-02)
- Pending: Per-worker logging (07-03), GPU worker process (07-04), HDF5 aggregation (07-05), coordinator integration (07-06+)

**Phase 8 (FP16 Precision Validation):**
- Numerical precision: LayerNorm may have limited dynamic range in FP16 - may need selective FP32 for specific layers while keeping rest in FP16

## Session Continuity

Last session: 2026-02-04
Stopped at: Completed 07-02-PLAN.md (IndexBasedDataset for byte-offset sequence reading)
Resume file: None

**Next step:** Continue Phase 7 - Plan 07-03 (Per-worker logging infrastructure)
