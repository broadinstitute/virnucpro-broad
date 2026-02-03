# Phase 5: Async DataLoader Foundation - Context

**Gathered:** 2026-02-03
**Status:** Ready for planning

<domain>
## Phase Boundary

This phase delivers single-GPU async DataLoader architecture where CPU workers handle pure I/O (FASTA parsing) while the main process performs tokenization and GPU inference. Workers yield raw sequence strings and are guaranteed CUDA-safe via spawn context and explicit validation. The GPU receives prefetched batches to minimize idle time.

**Out of scope:** Multi-GPU coordination (Phase 7), sequence packing (Phase 6), FP16 precision (Phase 8), checkpointing (Phase 9).

</domain>

<decisions>
## Implementation Decisions

### Worker Process Safety
- **Process context:** Spawn (not fork) - creates fresh processes with no CUDA inheritance
- **CUDA validation:** Explicit assert check in worker `__init__` - validate `torch.cuda.is_available() == False`
- **Worker architecture:** Pure I/O only - workers read FASTA, parse sequences, yield strings (no tokenization, no model loading)
- **Tokenizer location:** Main process only - workers never touch HuggingFace models or tokenizers
- **Error handling:** Fail fast with assertion - if worker somehow initializes CUDA, crash immediately with clear error

### Batch Prefetching Strategy
- **Worker count:** 4 workers - balance between prefetching and memory overhead
- **Prefetch factor:** 4+ batches per worker - aggressive prefetching for deeper queue
- **Worker lifecycle:** Persistent workers - keep workers alive across batches (faster, acceptable memory risk with monitoring)
- **Memory pinning:** `pin_memory=True` - enable pinned memory for faster CPU-to-GPU transfer

### Tokenization Location
- **Tokenization stage:** Workers yield strings, main process tokenizes in `collate_fn`
- **Tokenizer loading:** Load once in main process - single tokenizer instance shared via collate_fn
- **Padding strategy:** No traditional padding - use FlashAttention varlen pattern (concatenated tensor + `cu_seqlens`)
- **Collate output:** Concatenated 1D tensor with cumulative sequence lengths array for varlen attention

### Performance Monitoring
- **Core metrics (all required):**
  - GPU utilization % (compute utilization via nvitop or nvidia-smi)
  - DataLoader wait time (time GPU spends waiting for next batch)
  - Worker queue depth (number of prefetched batches ready)
  - Throughput in sequences/sec (end-to-end processing rate)
  - CPU and GPU RAM (detect leaks or OOM risks)
- **Logging frequency:** Every 10 batches - balance detail and noise
- **Bottleneck threshold:** Warn when GPU idle >10% - moderate threshold for I/O bottleneck detection

### Claude's Discretion
- Exact implementation of worker validation checks
- DataLoader timeout and error recovery mechanisms
- Specific nvitop vs nvidia-smi API usage for utilization tracking
- Log message format and verbosity levels
- Memory leak detection heuristics for persistent workers

</decisions>

<specifics>
## Specific Ideas

**Architecture pattern referenced:**
```
DataLoader Workers (CPU-only, spawn):
  ├── Read FASTA (I/O)
  ├── Parse sequences (CPU)
  └── Yield strings to main process

Main Process (GPU):
  ├── Receive CPU tensors via pin_memory
  ├── Move to GPU (non_blocking=True)
  ├── ESM2 inference (FP16)
  └── Triple-buffered async write
```

**Key constraints:**
- "Even with spawn, your DataLoader workers should never touch the GPU"
- "FlashAttention varlen is the approach here so we should be utilizing this functionality"
- Workers are pure I/O - no HuggingFace calls, no CUDA, no tokenization

</specifics>

<deferred>
## Deferred Ideas

None - discussion stayed within phase scope

</deferred>

---

*Phase: 05-async-dataloader-foundation*
*Context gathered: 2026-02-03*
