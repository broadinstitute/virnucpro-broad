# Phase 1: ESM-2 Multi-GPU Foundation - Context

**Gathered:** 2026-01-22
**Status:** Ready for planning

<domain>
## Phase Boundary

Parallelize ESM-2 protein feature extraction across multiple GPUs using file-level work distribution, delivering 3-4x throughput improvement while maintaining backward compatibility with single-GPU systems. This phase establishes the foundation for multi-GPU parallelization that will be extended to DNABERT-S in Phase 2.

</domain>

<decisions>
## Implementation Decisions

### Work Distribution Strategy
- **File assignment:** Size-aware assignment based on sequence counts to balance total sequences per GPU
- **Rebalancing:** Claude's discretion on dynamic work queue vs pre-assignment trade-off
- **Batching:** Adaptive batching based on sequence length to maximize GPU memory usage
- **Granularity:** Whole files only (no file splitting) — simpler bookkeeping

### Failure Handling
- **OOM errors:** Skip problematic file, log to failures file, continue with other files
- **Worker crashes:** Reassign failed work to other GPUs with crash detection
- **Failure tracking:** Write failed files to structured log file (failed_files.txt) with paths and error messages
- **Systemic failure detection:** Abort if 3+ consecutive files fail across workers (likely systemic issue like bad model)
- **Pre-flight validation:** Check CUDA and GPU availability upfront, fail fast with clear message
- **CUDA context:** Claude's discretion following STATE.md blocker about deferring CUDA to workers
- **Partial success checkpoints:** Write checkpoints for successful files even if some failed
- **Exit codes:** Distinguish complete success (0), partial success (2), total failure (1)

### Progress Visibility
- **Log verbosity:** Configurable with flags (--verbose, --quiet, default=moderate)
- **ETA display:** No ETA calculation — just show progress percentage (X/Y files, Z%)
- **Per-GPU status:** Live updating dashboard using rich/tqdm for real-time visibility
- **Throughput logging:** Summary only at end (sequences/second final stats, not periodic)

### Resource Constraints
- **GPU selection:** Support both CUDA_VISIBLE_DEVICES and --gpus flag (flag overrides env var)
- **Batch size:** User-configurable via --batch-size flag for OOM troubleshooting
- **Heterogeneous GPUs:** Treat all GPUs equally in Phase 1 (no weighting by capability)
- **Memory limits:** No explicit memory budget limit — trust adaptive batching

### Claude's Discretion
- Dynamic work queue vs pre-assignment trade-off (complexity vs utilization)
- CUDA context initialization approach (following STATE.md critical blocker)
- Terminal compatibility for live dashboard (fallback if rich unavailable)
- Exact adaptive batching algorithm (balance memory utilization with OOM safety)

</decisions>

<specifics>
## Specific Ideas

- Live dashboard should show per-GPU file assignment and progress
- Failed files log format: one line per failure with `path|error_type|error_message`
- Exit code 2 distinguishes partial success for scripting/CI integration
- Systemic failure threshold: 3 consecutive failures triggers abort with diagnostic message
- Summary stats at end should include: total time, files processed, sequences/sec per GPU, total failures

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 01-esm-2-multi-gpu-foundation*
*Context gathered: 2026-01-22*
