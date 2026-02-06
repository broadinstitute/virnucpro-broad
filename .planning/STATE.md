# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-02)

**Core value:** Embedding steps (DNABERT-S and ESM-2) efficiently utilize all available GPUs and automatically queue batches, reducing sample processing time from 45+ hours to under 10 hours.

**Current focus:** Phase 9 - Checkpointing Integration

## Current Position

Phase: 9 of 10 (Checkpointing Integration) - IN PROGRESS
Plan: 6 of 6 in phase - COMPLETE (Wave 3: 1/1 complete)
Status: Checkpoint integration tests complete - all components verified
Last activity: 2026-02-06 — Completed 09-07-PLAN.md (Integration tests)

Progress: [███████░░░] 77/TBD plans (v1.0: 34/34 complete, v2.0: 43/TBD)

## Performance Metrics

**Velocity:**
- Total plans completed: 75 (v1.0: 34, v2.0: 41)
- Average duration: 3.2 min
- Total execution time: 4.1 hours

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
| 7 | 8 | 29 min | 3.6 min |
| 8 | 4 (complete) | 13 min | 3.25 min |
| 9 | 6 (complete) | 32.5 min | 5.4 min |

**Recent Trend:**
- Last 5 plans: ~4.1 min average
- Trend: Steady (Phase 9 complete)

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
- **multiprocessing.Process over mp.spawn (07-04)**: Use Process directly instead of mp.spawn for fault tolerance - allows partial completion when one worker fails
- **CPU-only integration tests (09-07)**: All checkpoint integration tests run on CPU with MockModel and mocked CUDA - verifies checkpoint mechanics without GPU requirements
- **Module-level wrapper for CUDA_VISIBLE_DEVICES (07-04)**: Wrapper function must be module-level for pickle compatibility with spawn context
- **Per-rank completion tracking (07-04)**: wait_for_completion returns Dict[int, bool] to identify which workers succeeded/failed for partial aggregation
- **Worker logging first (07-06)**: setup_worker_logging() called before any other operations to ensure all worker actions are logged
- **Empty embeddings handling (07-06)**: Handle zero-sequence case gracefully with empty HDF5 dataset instead of crashing on torch.cat
- **Status before exit (07-06)**: results_queue.put(status) before sys.exit(1) ensures parent knows which workers failed
- **Partial failure handling (07-07)**: Return (output_path, failed_ranks) tuple instead of raising - salvages results from successful workers
- **Auto-detect world_size (07-07)**: Use torch.cuda.device_count() when world_size not specified for simplified common case
- **Partial expected IDs validation (07-07)**: Calculate expected IDs only from successful workers when failures occur to match actual shard content
- **Stream sync before extraction (07-08)**: Must synchronize compute stream before _extract_embeddings when retrieve_fn=None - prevents race condition where extraction runs on default stream before compute completes
- **Single CUDA sync for NaN/Inf detection (08-02)**: Batch all GPU ops before .item() calls reduces overhead from 5-10ms to <1ms - critical for per-batch stability checks
- **Env var precedence pattern (08-02)**: Caller (gpu_worker) checks VIRNUCPRO_DISABLE_FP16 and overrides model_config before calling load_esm2_model - separation of policy (worker) vs implementation (loader)
- **NaN/Inf detection on both inference paths (08-02)**: check_numerical_stability runs after packed AND unpacked inference - universal FP16 overflow protection regardless of packing state
- **Expected FP16 speedup 1.5-1.8x (08-04)**: FP16 tensor cores ~1.3-1.5x + larger batches ~1.2x over Phase 7 FP32+FlashAttention baseline - adjusted from 1.8-2x because baseline already includes FlashAttention
- **Stratified length testing (08-04)**: Separate measurements for short/medium/long sequences prevent padding skew - homogeneous lengths show true per-length performance variance
- **FlashAttention verification before benchmarking (08-04)**: verify_flashattention_active() checks forward_packed exists and runs test inference to catch fallback warnings - ensures measuring FlashAttention+FP16 not just FP16
- **No specific speedup assertion (08-04)**: Assert FP16 ≥ 1.0x (not slower) but don't assert ≥1.5x - environment-dependent, print comprehensive table for user evaluation
- **Warmup 10 iterations (08-04)**: Increased from typical 5 for 3B model + FlashAttention kernel compilation - ensures stable timing measurements
- **Function-scoped fixtures for memory (08-03)**: Sequential loading (FP32 → cleanup → FP16) reduces peak memory from ~40GB to ~22GB - enables reliable testing on A100-40GB GPUs
- **Realistic FP16 validation thresholds (08-03)**: Mean abs diff <0.01 (not 0.1), std rel diff <5% (not absolute), cosine >0.99 - reflects ESM-2 FP16 mantissa precision
- **Per-token similarity for packed format (08-03)**: Validates production forward_packed() code path separately due to RoPE timing and FlashAttention kernel differences
- **Statistical validation beyond similarity (08-03)**: Mean/std/L2 norm/outlier distributions confirm FP16 preserves statistical properties, not just cosine similarity
- **Checkpoint format .pt not HDF5 (09-01)**: PyTorch .pt format for incremental checkpoints instead of HDF5 - consistent with Phase 3 atomic_save, avoids append corruption risks
- **Env var precedence for checkpoints (09-01)**: VIRNUCPRO_VIRAL_CHECKPOINT_MODE overrides default thresholds (5K seq / 180s) but not explicit constructor args - enables viral workload tuning
- **Corrupted sequence requeue (09-01)**: resume_from_checkpoints returns 4-tuple with corrupted_sequence_ids for caller requeue - enables idempotent recovery without full restart
- **Manifest optional for resume (09-01)**: Manifest validation logs warnings but doesn't fail - filesystem checkpoints are source of truth for per-shard resume
- **POSIX file locking for manifest (09-02)**: Use fcntl.flock for cross-process coordination instead of threading.Lock - GPU workers are spawned processes with independent memory spaces
- **Checkpoint file validation before manifest update (09-02)**: update_shard_checkpoint validates file exists to ensure filesystem and manifest state consistency
- **Elastic shard redistribution (09-02)**: Separate original_rank (immutable) and assigned_rank (mutable) fields enable reassigning failed work to healthy workers
- **Triple-redundancy manifest recovery (09-02)**: Primary -> .tmp -> .backup fallback chain provides high fault tolerance for JSON corruption
- **Staleness threshold 600s default (09-02)**: Conservative 10-minute threshold for zombie detection avoids false positives from slow I/O or large batches
- **Orphaned shards retry_count >= 3 (09-02)**: Max 3 retries per shard before marking as orphaned for redistribution - prevents infinite retry loops
- **Batch boundary checkpointing (09-03)**: Checkpoint trigger fires AFTER yield, at batch boundaries only - respects packed attention atomicity by never checkpointing mid-batch
- **CPU transfer before accumulation (09-03)**: Transfer embeddings to CPU via .cpu().numpy() before accumulation - prevents CUDA memory growth from accumulating GPU tensors
- **Resumed data as InferenceResult (09-03)**: Yield resumed data as normal InferenceResult with batch_idx=-1 marker - seamless pipeline integration without special downstream handling
- **Packing stats in checkpoint metadata (09-03)**: Capture packing efficiency from batch result and include in checkpoint metadata - enables post-mortem debugging
- **Resume with index filtering (09-04)**: gpu_worker resumes from checkpoints BEFORE creating dataset, filters index to skip processed sequences - prevents duplicates, clean separation (resume happens once at start, not intermixed)
- **Per-shard checkpoint isolation (09-04)**: checkpoint_dir = checkpoint_base_dir / f"shard_{rank}" prevents cross-GPU conflicts when multiple workers checkpoint simultaneously
- **SIGTERM handler for spot preemption (09-04)**: signal.signal(SIGTERM, sigterm_handler) saves emergency checkpoint when spot instance receives termination signal (30s timeout, exit code 143)
- **Differentiated error handling (09-04)**: Categorize errors (cuda_oom, cuda_runtime, generic) via error_type field while maintaining backward compatibility (error field contains message) - enables targeted retry strategies
- **RuntimeConfig separation (09-05)**: Operational parameters (checkpointing, retries, timeouts) separated from model architecture config - clean serialization boundary for worker passing
- **Per-attempt timeout (09-05)**: timeout_per_attempt applies per retry attempt, not globally - enables infinite spot retry while preventing individual attempt hangs
- **Spot preemption infinite retry (09-05)**: Spot preemption (SIGTERM exitcode 143) retries infinitely with 60s polling - capacity returns eventually
- **Poison input circuit breaker (09-05)**: Track failures per (rank, batch_idx), trigger circuit breaker after 2 failures on same batch to isolate toxic sequences
- **Coordinator-only manifest writes (09-05)**: Workers signal via results_queue, only coordinator updates manifest - eliminates POSIX lock contention
- **Async monitoring non-blocking (09-05)**: Coordinator polls workers every 5s, continues monitoring healthy workers during retries - enables partial completion
- **Elastic redistribution (09-05)**: Failed shard work reassigned to lowest-numbered active worker via CheckpointManifest.reassign_shard()
- **SIGTERM handler coordination (09-05)**: Coordinator waits 30s for workers to save emergency checkpoints before terminating - graceful spot instance handling
- **Checkpoint validation before respawn (09-05)**: Remove orphaned .tmp files and verify .done markers before worker restart - prevents corruption propagation

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

**Phase 7 (Multi-GPU Coordination):** ✅ COMPLETE
- ✅ SequenceIndex with stride distribution and caching (07-01)
- ✅ IndexBasedDataset for byte-offset sequence reading (07-02)
- ✅ Per-worker logging infrastructure (07-03)
- ✅ GPUProcessCoordinator for worker lifecycle (07-04)
- ✅ HDF5 shard aggregation with validation (07-05)
- ✅ GPU worker function integration (07-06)
- ✅ run_multi_gpu_inference orchestration entry point (07-07)
- ✅ Integration tests + stream sync race condition fix (07-08)

**Phase 8 (FP16 Precision Validation):** ✅ COMPLETE
- ✅ FP16 precision validation (mean abs diff <0.01, cosine >0.99)
- ✅ Numerical stability checks (NaN/Inf detection with minimal sync overhead)
- ✅ FP16 performance benchmarks (454K seq/hour, 6.06GB memory)
- Note: LayerNorm may have limited dynamic range in FP16 - selective FP32 for specific layers if needed

**Phase 9 (Checkpointing Integration):** IN PROGRESS - Wave 2 COMPLETE
- ✅ Checkpoint foundation (CheckpointTrigger, AsyncCheckpointWriter, validation, resume) - 09-01
- ✅ CheckpointManifest for multi-GPU coordination - 09-02
- ✅ AsyncInferenceRunner checkpoint integration (batch boundaries, resume, metadata) - 09-03
- ✅ GPU worker integration (resume, index filtering, SIGTERM, error tiers) - 09-04
- ✅ Coordinator integration (differentiated retry policies, async monitoring, elastic redistribution) - 09-05
- Next: End-to-end integration tests (09-06)

## Session Continuity

Last session: 2026-02-06
Stopped at: Completed 09-05-PLAN.md (Coordinator integration with fault-tolerant retry policies)
Resume file: None

**Next step:** Continue Phase 9 Wave 3 - End-to-end integration tests (09-06)
