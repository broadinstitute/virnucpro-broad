# Phase 2: DNABERT-S Optimization - Context

**Gathered:** 2026-01-23
**Status:** Ready for planning

<domain>
## Phase Boundary

Optimize DNABERT-S feature extraction to match ESM-2's optimization level with improved batching, automatic queuing, and unified worker infrastructure. This phase applies Phase 1's multi-GPU patterns to DNABERT-S, delivering 3-4x throughput improvement with consistent monitoring and user experience.

</domain>

<decisions>
## Implementation Decisions

### Batching Strategy
- File-level distribution matching ESM-2 approach (workers get complete files)
- Dynamic batching by tokens/k-mers to handle variable DNA sequence lengths
- Automatic queuing: workers process files in multiple batches automatically
- Token-based batch sizing abstracts k-mer details (treat like ESM-2 tokens)

### Worker Infrastructure
- Unified worker abstraction: create shared base class (BaseEmbeddingWorker) that both DNABERT-S and ESM-2 inherit from
- Separate worker pools: DNABERT-S and ESM-2 run sequentially with their own pools (simpler isolation, predictable memory)
- Unified dashboard: both models report progress to same monitoring infrastructure
- Same optimizations as ESM-2: spawn context, BF16 auto-detection, torch.no_grad mode

### Performance Tuning
- Manual batch size configuration via CLI flags
- Independent flags: --dnabert-batch-size and --esm-batch-size (separate per-model tuning)
- Target 3-4x throughput improvement matching ESM-2's multi-GPU scaling
- Full monitoring: memory tracking, GPU utilization logging (same infrastructure as ESM-2)

### Integration Approach
- Shared base class at code level: BaseEmbeddingWorker enforces consistency and reduces duplication
- Hybrid CLI flags: shared --gpus for both models, separate batch size flags per model
- Sequential execution: DNABERT-S completes fully before ESM-2 starts (simpler, predictable memory)
- Auto-detect GPUs: zero-config multi-GPU like ESM-2 (automatically uses available GPUs)

### Claude's Discretion
- K-mer tokenization specifics (whether special handling needed beyond token abstraction)
- Exact base class interface design
- Profiling methodology for determining optimal batch size defaults
- Error handling patterns specific to DNABERT-S vs ESM-2

</decisions>

<specifics>
## Specific Ideas

- "Match ESM-2's pattern" - consistency is key, users should see same behavior across both embedding models
- Success criteria already specifies 2-4x batch size increase - this guides profiling targets
- CLI should feel unified (--gpus shared) but allow model-specific tuning where needed (batch sizes)

</specifics>

<deferred>
## Deferred Ideas

None - discussion stayed within phase scope.

</deferred>

---

*Phase: 02-dnabert-s-optimization*
*Context gathered: 2026-01-23*
