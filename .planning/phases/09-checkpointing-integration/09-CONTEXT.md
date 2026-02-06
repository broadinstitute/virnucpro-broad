# Phase 9: Checkpointing Integration - Context

**Gathered:** 2026-02-05
**Status:** Ready for planning

<domain>
## Phase Boundary

Pipeline resumes from partial completion after crashes for multi-hour GPU workloads processing 6M sequences. This phase adds incremental checkpointing to the existing async DataLoader + sequence packing + multi-GPU pipeline (Phases 5-8), enabling recovery from crashes without reprocessing completed work.

</domain>

<decisions>
## Implementation Decisions

### Checkpoint Granularity
- **Adaptive trigger**: Checkpoint when either N sequences OR M minutes threshold reached (whichever comes first)
- **Default interval**: 10K sequences OR 5 minutes (balanced I/O overhead vs recovery precision)
- **Batch boundary constraint**: Never checkpoint mid-packed-batch unless emergency (>10 min overdue) - respects packed attention atomicity
- **Async I/O**: Use background thread for serialization to avoid GPU stalls during checkpoint writes
- **Viral mode override**: Environment variable to tune for 400AA+ workloads (5K sequences / 3 min for faster checkpointing)
- **Tracking strategy**: Both per-shard checkpoints + coordinator manifest
  - Shards write local checkpoints independently
  - Coordinator writes manifest tracking batch boundaries + sequence counts per shard
  - Manifest enables global validation and partial failure recovery

### Resume Behavior
- **Automatic resume with override**: Pipeline resumes from checkpoints by default, `--force-restart` flag to force fresh start
- **Validation**: Basic validation via file size + `.done` marker (matches Phase 4 checkpoint patterns)
- **Corruption handling**: Invalidate the specific batch, requeue its sequences, and let pipeline continue
  - Treats checkpoints as idempotent completion markers, not fragile state snapshots
  - No full pipeline restart for single corrupted checkpoint
- **Logging strategy**: Layered reporting
  - Summary header: "Resuming from X/Y sequences"
  - Progress bar initialized at resume position for continuity
  - Per-shard diagnostics available via `--verbose` flag

### Checkpoint Format
- **File structure**: Atomic checkpoint snapshots (write temp, rename on completion - matches Phase 4 `.done` marker pattern)
- **Content**: Embeddings + metadata (true incremental progress, not just tracking IDs)
- **Metadata fields**: Verbose diagnostics
  - Sequence count + last IDs + timestamp + batch_idx
  - Model config (dtype, packing enabled, FlashAttention active)
  - Packing stats (efficiency, buffer size, token budget)
  - GPU memory usage snapshot
  - Helps debug crashes and verify consistency on resume
- **File organization**: Hierarchical directories by shard_id, files named by batch_id, with `.done` markers
  - Example: `checkpoints/shard_0/batch_00042.h5` + `batch_00042.done`
  - Supports manifest-based resume logic while allowing parallel scanning per GPU
  - Phase 4 compatibility for validation patterns

### Failure Recovery
- **Crash response**: Restart only the failed GPU's shard (other GPUs continue processing)
- **Restart attempts**: 3 retries with exponential backoff (handles transient OOM, CUDA errors)
- **Coordination mode**: Async restart - other GPUs keep processing while failed shard restarts
- **Diagnostic logging**: Intelligent tiering by failure type
  - Full diagnostics for CUDA errors and Python exceptions (stack trace, GPU memory snapshot, model state)
  - Minimal logging for spot preemption (expected transient failures)
  - Always include sequence IDs and lengths in batch metadata for poison-input detection
  - Helps identify if specific sequences cause crashes

### Claude's Discretion
- Specific exponential backoff timing (e.g., 1s, 2s, 4s delays)
- Exact async I/O thread pool sizing
- Checkpoint manifest file format (JSON vs YAML)
- Emergency checkpoint trigger threshold details (>10 min overdue implementation)

</decisions>

<specifics>
## Specific Ideas

- Batch atomicity is critical - checkpoints must not break packed attention sequences mid-batch
- Async I/O to prevent GPU stalls during checkpoint writes (learned from Phase 5 async DataLoader patterns)
- Hierarchical directory structure allows parallel shard scanning without coordination overhead
- Poison-input detection via sequence ID/length logging helps debug specific sequence failures
- Intelligent diagnostic tiering reduces log spam from expected failures (spot preemption) while capturing actionable info for bugs

</specifics>

<deferred>
## Deferred Ideas

None - discussion stayed within phase scope

</deferred>

---

*Phase: 09-checkpointing-integration*
*Context gathered: 2026-02-05*
