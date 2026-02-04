# Phase 7: Multi-GPU Coordination - Context

**Gathered:** 2026-02-04
**Status:** Ready for planning

<domain>
## Phase Boundary

Multi-GPU coordination for independent shard processing across N GPUs (typically 4). Each GPU processes assigned sequences deterministically, then parent process aggregates outputs into complete checkpoint. Focuses on work distribution, process lifecycle, and output aggregation - NOT on checkpointing strategy itself (Phase 9).

</domain>

<decisions>
## Implementation Decisions

### Work Distribution Strategy

**Index-based sharding (NOT file-level):**
- Create sequence index file containing metadata: `sequence_id`, `length`, `file_path`, `byte_offset`
- Index creation on first run, cached for reuse
- Cache invalidation: `max(fasta_mtime) > index_mtime` triggers rebuild
- Global sort by length descending (for optimal packing efficiency across all GPUs)
- Stride distribution: GPU rank N gets indices [N, N+world_size, N+2*world_size, ...]
- Log distribution metrics (total tokens per GPU, sequence count per GPU) to detect systematic imbalance

**Rationale:** Index-based approach with global sort ensures all GPUs get representative length distribution, maximizing packing efficiency. File-level sharding could create skewed distributions if files have biased lengths.

### Process Lifecycle Management

**Architecture (SPMD with parent orchestrator):**
```
Parent/Orchestrator Process (non-GPU):
├── Creates/validates index file
├── Spawns GPU workers (torch.multiprocessing.spawn)
├── Monitors health (via results queue or PID polling)
├── Aggregates outputs when workers finish
└── Reports final status

Worker 0 (GPU 0)    Worker 1 (GPU 1)    Worker N (GPU N)
├── Load index      ├── Load index      ├── Load index
├── Take indices    ├── Take indices    ├── Take indices
│   0,N,2N,3N...       1,N+1,2N+1...       N-1,2N-1,3N-1...
├── Run inference   ├── Run inference   ├── Run inference
├── Save shard_0    ├── Save shard_1    ├── Save shard_N
└── Report "done"   └── Report "done"   └── Report "done"
```

**Key decisions:**
- Launch mechanism: `torch.multiprocessing.spawn` (standard PyTorch pattern)
- GPU assignment: `CUDA_VISIBLE_DEVICES` per process (each worker sees device 0)
- Process model: SPMD - all workers are symmetric, no special rank 0 logic
- Parent coordination: Orchestration happens outside GPU context (fault-tolerant)
- Error handling: If one GPU crashes, let other GPUs finish and warn about partial results
  - Depends on Phase 9 checkpointing to handle failed shard recovery
- Logging: Per-worker log files (`worker_0.log`, `worker_1.log`, etc.) - no interleaving

### Checkpoint Aggregation Protocol

**Shard format:**
- HDF5 (current format) - consistent with v1.0 codebase
- Each worker writes `shard_N.h5` with its assigned sequences

**Aggregation strategy:**
- Sequential chunk-wise merge (hybrid approach)
- Process shards one by one (simple logic)
- Read each shard in chunks (e.g., 10K sequences at a time) to control memory usage
- Write to final merged output

**Validation:**
- Sequence count: verify total matches expected count from index
- No duplicates: build set of all sequence IDs across shards
- No missing IDs: verify all expected sequence IDs are present
- **On validation failure (missing sequences):** Fail with clear error message listing missing IDs

**Rationale:** Strict validation ensures data integrity. Missing sequences indicate worker crash or stride distribution bug - must be investigated, not silently ignored.

### Progress Monitoring & Logging

**Worker logging:**
- Each worker writes to `worker_N.log` (no shared log file)
- Metrics logged: sequences/sec, tokens/sec, packing efficiency (from Phase 6)

**Parent logging:**
- Periodic progress summary (not real-time streaming)
- Aggregate stats across all workers
- Simple text format: "Total: 1.2M seq/s, 45M tok/s across 4 GPUs"

**Rationale:** Per-worker files avoid race conditions and interleaving. Parent summary provides high-level visibility without log spam. Detailed debugging available via individual worker logs.

### Claude's Discretion

- Exact format of sequence index file (JSON, pickle, etc.)
- Health monitoring mechanism (results queue vs PID polling)
- Chunk size for aggregation (10K sequences is suggested but not required)
- Periodic summary interval for parent logging
- Exact error message format for validation failures

</decisions>

<specifics>
## Specific Ideas

**Generalization:**
- Support N GPUs, not just 4 - user has 4 GPUs but implementation should generalize
- Architecture must work for 1, 2, 8, or any number of GPUs

**Fault tolerance:**
- Workers are completely symmetric - no worker has special responsibilities
- Parent orchestrator outside GPU context can't be crashed by GPU failures
- Partial results with clear warnings allow investigation and selective re-runs

**Memory efficiency:**
- Index file is metadata-only (~100-200MB for 6M sequences) - easily fits in memory
- Chunk-wise aggregation prevents memory blowup with large shard files

</specifics>

<deferred>
## Deferred Ideas

None - discussion stayed within phase scope.

</deferred>

---

*Phase: 07-multi-gpu-coordination*
*Context gathered: 2026-02-04*
