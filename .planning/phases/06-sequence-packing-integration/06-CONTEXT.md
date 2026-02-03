# Phase 6: Sequence Packing Integration - Context

**Gathered:** 2026-02-03
**Status:** Ready for planning

<domain>
## Phase Boundary

Packing variable-length sequences into fixed-size batches with FlashAttention varlen to maximize GPU utilization while preventing cross-sequence attention contamination. Builds on Phase 5's async DataLoader foundation. Multi-GPU coordination (Phase 7) and FP16 validation (Phase 8) are separate phases.

</domain>

<decisions>
## Implementation Decisions

### Packing Strategy

**Algorithm:**
- Greedy first-fit with descending length sort (First Fit Decreasing)
- Sort sequences within each GPU shard (not globally) for ~90-95% packing efficiency
- Deterministic tie-breaking: sequences with identical length sorted by sequence ID
- No file-based grouping - pure greedy packing across all sequences

**Oversized sequences (>max_length):**
- Truncate to max_length with warning (default behavior)
- Intelligent truncation preserving biological signal (N-terminal region for ESM-2)
- Track truncation statistics per batch
- Log warnings with sequence IDs for sequences truncated

**Memory considerations:**
- Loading 1.5M sequences per GPU shard for sorting requires <1GB RAM (acceptable)
- Pre-sort within SequenceDataset or use sorted index for IterableDataset compatibility

### Correctness Validation

**Primary validation:**
- Cosine similarity >0.999 between packed and unpacked embeddings
- Lenient threshold: 99% of sequences must pass, allow 1% in range 0.995-0.999
- Tolerates FP16/FP32 precision outliers while catching systematic errors

**Cross-sequence contamination testing:**
- Use known test sequences with validated outputs
- Pack with random sequences, verify outputs unchanged (functional test)
- Verify cu_seqlens boundaries separate sequences correctly (unit test)
- Gold standard: if packed embeddings match unpacked embeddings (within tolerance), no contamination exists

**Production validation:**
- Claude's discretion: choose between testing-only, first-N-batches, or random sampling
- Balance safety vs performance overhead based on statistical rigor

### FlashAttention Integration

**Position IDs:**
- Reset to 0 at each sequence boundary (not sequential across batch)
- cu_seqlens tensor defines boundaries where position IDs reset
- Natural for language models and compatible with FlashAttention varlen

**cu_seqlens construction:**
- Compute in VarlenCollator on CPU before GPU transfer
- Format: cumulative boundaries [0, len1, len1+len2, ...] with N+1 elements for N sequences
- Always validate format: verify N+1 elements, starts with 0, monotonically increasing
- Validation runs in collator (not GPU hot path) for efficiency

**Fallback behavior:**
- If FlashAttention unavailable (older GPU, missing package): fall back to standard attention with padding
- Clear performance warnings when fallback triggered
- Preserves correctness at cost of efficiency

### Efficiency Metrics

**Packing density:**
- Report both token utilization % and padding waste %
- Token utilization = (actual_tokens / max_tokens_per_batch) × 100
- Padding waste = (padding_tokens / total_tokens) × 100

**Logging granularity:**
- Periodic summary: log stats every N batches (e.g., every 100)
- Reduces noise while catching trends and issues

**Warning thresholds:**
- Claude's discretion: set appropriate threshold (likely 80-90% utilization)
- Based on practical packing limits for greedy first-fit algorithm

**Additional metrics:**
- Claude's discretion: choose from sequences/batch, tokens/sec throughput, sequence length distribution
- Include metrics most useful for optimization and debugging

### Claude's Discretion

- Exact validation mode (testing-only vs first-N-batches vs random sampling)
- Inefficient packing warning threshold (80% vs 90% utilization)
- Additional metrics beyond packing density to track
- Specific implementation of intelligent truncation for oversized sequences
- Periodic logging interval (every N batches)

</decisions>

<specifics>
## Specific Ideas

**Sorting rationale:**
- First Fit Decreasing is optimal greedy algorithm for bin packing
- Packing efficiency: random order ~70%, ascending ~60%, descending ~90-95%
- Big items anchor bins, small items fill gaps

**Truncation rationale for viral sequences:**
- ESM-2 uses N-terminal signal heavily (classifier token + early sequence features)
- Long ORFs (>512aa) are often polyproteins where N-terminal contains sufficient signal
- Losing C-terminal 10% rarely affects biological conclusions for coding potential prediction

**Validation approach:**
- Functional test (packed vs unpacked outputs) is gold standard for FlashAttention varlen
- If packing produces identical embeddings (within FP16 tolerance), no contamination exists
- Unit tests verify cu_seqlens construction correctness

**cu_seqlens validation:**
- Must validate in collator (CPU) not GPU hot path for efficiency
- Format check: N+1 elements for N sequences, starts with 0, monotonically increasing
- Catches configuration bugs early without performance penalty

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 06-sequence-packing-integration*
*Context gathered: 2026-02-03*
