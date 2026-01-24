# Phase 4: Memory & Attention Optimization - Context

**Gathered:** 2026-01-23
**Status:** Ready for planning

<domain>
## Phase Boundary

Memory-efficient processing infrastructure that integrates FlashAttention-2, DataLoader prefetching, CUDA streams, and memory fragmentation prevention to achieve 1.5-2x additional speedup beyond current multi-GPU parallelization.

This phase optimizes how the system manages GPU memory and computes attention operations. It does not add new pipeline capabilities or change the user-facing prediction workflow.

</domain>

<decisions>
## Implementation Decisions

### FlashAttention-2 Integration
- ESM-2 only initially — integrate FlashAttention-2 for ESM-2 first, add DNABERT-S in follow-up if feasible
- Automatic fallback — use standard attention with warning log if FlashAttention-2 unavailable (missing library/incompatible GPU)
- Log verification at startup — check model config and log 'FlashAttention-2: enabled' or 'using standard attention'
- Automatic on compatible GPUs — enable FlashAttention-2 by default when GPU supports it (Ampere+, flash-attn installed)

### DataLoader Configuration
- CLI configurable — add `--dataloader-workers` flag for user control
- Auto-detect (CPU-aware) default — `min(cpu_count // num_gpus, 8)` when flag not specified
- Fixed prefetch_factor=2 — keep as internal implementation detail (good default)
- CLI flag control for pin_memory — add `--pin-memory` flag, default based on system RAM or conservative off

### CUDA Streams Orchestration
- Auto-scale with GPUs — use more streams when more GPUs available (e.g., 2 streams per GPU)
- Fail entire worker — any stream error terminates the GPU worker immediately and reports failure
- Synchronize after every batch — sync streams after each batch completes for safety (simpler, slight overhead)
- Log at debug level — log stream operations with `--verbose` or debug mode for troubleshooting

### Memory Fragmentation Prevention
- Sequence sorting strategy: Claude's discretion — determine optimal sorting stage based on performance and memory impact
- Cache clearing frequency: Claude's discretion — determine clearing strategy based on memory patterns and overhead
- Expandable segments CLI control — add `--expandable-segments` flag so users can toggle based on their hardware
- OOM recovery: Fail with diagnostics — exit immediately with memory usage report and recommendations for `--batch-size`

### Claude's Discretion
- Sequence sorting stage (during file loading vs. batch creation vs. hybrid)
- CUDA cache clearing frequency (periodic, reactive, checkpoint-based)
- FlashAttention-2 validation implementation details
- DataLoader internal buffer management
- Stream allocation strategy (exact count per GPU)

</decisions>

<specifics>
## Specific Ideas

**FlashAttention-2:**
- Compatibility check should happen early (at model load) and log clearly
- Fallback to standard attention should be transparent to users (they see warning but pipeline continues)

**DataLoader:**
- Auto-detection formula: `min(cpu_count // num_gpus, 8)` balances I/O workers per GPU
- Pin memory flag should show RAM impact in help text

**CUDA Streams:**
- Debug logging should show stream ID and operation type for troubleshooting OOM issues
- Stream count scales with GPU count to maintain I/O/compute overlap ratio

**Memory Management:**
- OOM diagnostics should include: current batch size, peak memory usage, recommended batch size
- Expandable segments flag should document which GPU architectures benefit most

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 04-memory-&-attention-optimization*
*Context gathered: 2026-01-23*
