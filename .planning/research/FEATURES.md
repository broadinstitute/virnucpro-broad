# Feature Research

**Domain:** Async DataLoader and Sequence Packing for Transformer Inference
**Researched:** 2026-02-02
**Confidence:** HIGH

## Feature Landscape

### Table Stakes (Users Expect These)

Features that are essential for async DataLoader and sequence packing to work correctly. Missing these means the feature is broken.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Async data loading with CPU workers | Overlap I/O (FASTA parsing, tokenization) with GPU inference | MEDIUM | PyTorch DataLoader with num_workers>0 provides this. Requires pinned memory and non_blocking transfers. Dependencies: existing batch processing infrastructure. |
| Prefetch buffer management | Keep GPU fed with ready batches to eliminate idle time | LOW | DataLoader prefetch_factor controls queue depth. Rule of thumb: prefetch_factor=2-4 per worker. Dependencies: async loading. |
| Sequence concatenation for packing | Pack multiple variable-length sequences into single tensor | MEDIUM | Concatenate sequences and track boundaries with cu_seqlens (cumulative sequence lengths). Dependencies: tokenization pipeline. |
| Attention masking for packed sequences | Prevent cross-contamination between packed sequences | HIGH | Document masking via flash_attn_varlen_func ensures sequences don't attend to each other. Critical for correctness. Dependencies: FlashAttention-2 (already integrated). |
| GPU memory pinning | Enable asynchronous CPU→GPU transfers | LOW | DataLoader pin_memory=True. Required for non_blocking transfers to overlap with compute. Dependencies: none. |
| Non-blocking GPU transfers | Overlap data transfer with kernel execution | LOW | tensor.to(device, non_blocking=True). Works only with pinned memory. Dependencies: pin_memory. |
| Token budget enforcement | Respect GPU memory limits when packing sequences | MEDIUM | Track total tokens in batch (sum of sequence lengths). Reject sequences that exceed toks_per_batch budget. Dependencies: token counting logic. |
| FP16 precision support | 2x memory reduction and 2x speedup on tensor cores | MEDIUM | torch.autocast('cuda', dtype=torch.float16). Requires accuracy validation for embeddings. Currently forced FP32 due to dtype mismatch issues. Dependencies: model compatibility testing. |

### Differentiators (Competitive Advantage)

Features that set VirNucPro v2.0 apart from standard inference pipelines. Not required for correctness, but provide significant performance gains.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Intelligent packing algorithm | Maximize GPU utilization by minimizing padding waste | HIGH | Implement First-Fit Decreasing (FFD) or Longest-Pack-First-Highest-Priority (LPFHP). Research shows 1.5-2x throughput gains by eliminating 50-70% padding. Dependencies: sorted sequence length tracking. |
| Dynamic batch sizing | Adapt batch size based on sequence lengths to maintain constant token budget | MEDIUM | Small sequences → larger batches. Long sequences → smaller batches. Keeps GPU memory utilization constant. Dependencies: token budget tracking. |
| Multi-stream GPU execution | Overlap data transfer and kernel execution using CUDA streams | HIGH | Separate streams for H2D transfer and inference. Requires careful synchronization. Provides 20-30% speedup. Dependencies: CUDA stream management, pinned memory. |
| Continuous batching | Group sequences at iteration level, not batch level | HIGH | Don't wait for all sequences in batch to finish. Start new sequences as old ones complete. Achieves 10-20x better throughput than static batching. Dependencies: dynamic scheduling, inflight batch management. |
| Adaptive prefetching | Dynamically adjust num_workers and prefetch_factor based on GPU utilization | HIGH | Monitor GPU idle time and adjust CPU worker count accordingly. Prevents over/under-provisioning. Dependencies: GPU monitoring, dynamic worker spawn. |
| Zero-copy data path | Minimize memory copies between components | MEDIUM | Direct tensor passing from DataLoader → GPU → inference. Avoid intermediate CPU copies. Dependencies: careful memory lifecycle management. |

### Anti-Features (Commonly Requested, Often Problematic)

Features that seem beneficial but create more problems than they solve for this use case.

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|-----------------|-------------|
| Multi-GPU sequence distribution | Desire to use all available GPUs | Adds communication overhead between GPUs. Model (3B params) fits on single GPU. Adds complexity without benefit for inference. | Use single GPU per process, multiple processes for multiple GPUs (current v1.0 architecture). |
| Distributed coordination layer | Want centralized batch scheduling across GPUs | Requires network/IPC coordination. Latency kills throughput gains. Inference is embarrassingly parallel. | File-based work distribution (current approach) or simple queue-based task assignment. |
| KV cache optimization | Training literature emphasizes KV cache for generation | VirNucPro does embedding extraction, not autoregressive generation. No repeated decoding steps. KV cache provides zero benefit. | Focus on batch throughput, not per-token latency. |
| Complex packing schedulers | Research papers show sophisticated scheduling algorithms | Adds significant implementation complexity. Diminishing returns beyond simple FFD or LPFHP algorithms. Debugging becomes nightmare. | Start with First-Fit Decreasing (FFD). Profile before adding complexity. |
| Model sharding | Desire to handle larger models | Model fits on single GPU. Sharding adds memory transfer overhead. Only useful when model > GPU memory. | Stick to single-GPU inference. Use larger GPU if needed. |
| Asynchronous tokenization | Want to tokenize on-the-fly | Tokenization is CPU-bound and irregular (sequence length dependent). Creates backpressure. Better to batch tokenize. | Pre-tokenize in DataLoader workers before packing. Token IDs transfer faster than text. |

## Feature Dependencies

```
Async DataLoader Architecture:
    GPU Memory Pinning (pin_memory=True)
        └──enables──> Non-blocking GPU Transfers (non_blocking=True)
            └──enables──> Multi-stream GPU Execution (overlap transfer & compute)

Sequence Packing Pipeline:
    Sequence Concatenation
        └──requires──> Token Budget Enforcement
        └──requires──> Attention Masking (flash_attn_varlen_func)
            └──enhances──> Intelligent Packing Algorithm (minimize padding)
                └──enables──> Dynamic Batch Sizing (constant token budget)

FP16 Precision:
    Model Compatibility (no dtype mismatch)
        └──enables──> FP16 Autocast
            └──requires──> Accuracy Validation (embedding quality check)

Conflicts:
    Multi-GPU Sequence Distribution ──conflicts──> Single-process-per-GPU Architecture
    Continuous Batching ──conflicts──> Static File-based Work Distribution
    Asynchronous Tokenization ──conflicts──> Pre-tokenize in DataLoader Workers
```

### Dependency Notes

- **GPU Memory Pinning enables Non-blocking Transfers:** Pinned memory allows DMA access, which is required for asynchronous CPU→GPU transfers. Without pinning, non_blocking=True has no effect.
- **Attention Masking requires FlashAttention-2:** Standard PyTorch attention doesn't support efficient packed sequences. flash_attn_varlen_func with cu_seqlens is the canonical approach (FlashAttention-2 already integrated in v1.0).
- **Intelligent Packing enhances Dynamic Batch Sizing:** Sorting sequences by length before packing reduces padding within batches. Dynamic sizing ensures batches stay within token budget.
- **FP16 requires Accuracy Validation:** ESM-2 embeddings may degrade in FP16 (LayerNorm has limited dynamic range). Must validate cosine similarity between FP32 and FP16 embeddings >0.99 before enabling.
- **Multi-stream GPU Execution conflicts with simple architecture:** Requires separate CUDA streams, careful synchronization, and pinned memory. Only beneficial if CPU→GPU transfer is bottleneck (profile first).

## MVP Definition (VirNucPro v2.0)

### Launch With (v2.0 Milestone)

Minimum viable async DataLoader + sequence packing. What's needed to validate 2-3x throughput gain hypothesis.

- [ ] **Async data loading with CPU workers** — Essential for I/O/compute overlap. Start with num_workers=4, prefetch_factor=2.
- [ ] **GPU memory pinning** — Required for non-blocking transfers. DataLoader pin_memory=True.
- [ ] **Non-blocking GPU transfers** — Overlap data movement with compute. tensor.to(device, non_blocking=True).
- [ ] **Sequence concatenation for packing** — Pack variable-length sequences into single tensor. Track boundaries.
- [ ] **Attention masking for packed sequences** — Use flash_attn_varlen_func with cu_seqlens. Critical for correctness.
- [ ] **Token budget enforcement** — Respect toks_per_batch=2048 limit when packing. Reject sequences that overflow.
- [ ] **First-Fit Decreasing (FFD) packing** — Simple greedy algorithm. Sort sequences by length descending, pack into batches. Good enough for 1.5-2x gains.
- [ ] **FP16 precision (validated)** — Enable torch.autocast after validating embedding accuracy. Blocks 2x memory/speed gains until validated.

### Add After Validation (v2.1+)

Features to add once core async + packing is working and benchmarked.

- [ ] **Dynamic batch sizing** — Trigger: If batches have high padding waste (>30% padding tokens). Adapt batch size to maintain constant token budget.
- [ ] **Multi-stream GPU execution** — Trigger: Profile shows GPU idle time >20% waiting for data. Separate streams for transfer and compute.
- [ ] **Adaptive prefetching** — Trigger: Workload has highly variable sequence lengths. Dynamically tune num_workers based on GPU starvation metrics.
- [ ] **Advanced packing algorithm (LPFHP)** — Trigger: FFD leaves >20% padding waste. Longest-Pack-First-Highest-Priority can squeeze out extra 10-15% throughput.
- [ ] **Zero-copy data path** — Trigger: Profiling shows significant time in tensor copying. Optimize memory lifecycle to avoid intermediate copies.

### Future Consideration (v3.0+)

Features to defer until v2.0 performance is proven and bottlenecks are understood.

- [ ] **Continuous batching** — Defer: Requires major architecture change from file-based work distribution. Not compatible with current checkpoint/resume model. Only valuable if batch completion time variance is high.
- [ ] **BF16 precision** — Defer: BF16 has better dynamic range than FP16 (safer for embeddings). But requires Ampere+ GPUs (A100, H100). Check deployment hardware first.
- [ ] **Custom CUDA kernels for packing** — Defer: PyTorch native operations are fast enough initially. Only optimize if packing overhead >10% of total time.
- [ ] **Distributed inference coordination** — Defer: File-based distribution works fine for embarrassingly parallel workload. Only needed if central scheduling becomes bottleneck.

## Feature Prioritization Matrix

| Feature | User Value | Implementation Cost | Priority |
|---------|------------|---------------------|----------|
| Async data loading with CPU workers | HIGH (eliminates I/O gaps) | LOW (DataLoader built-in) | P1 |
| Sequence concatenation for packing | HIGH (enables packing) | MEDIUM (requires cu_seqlens) | P1 |
| Attention masking for packed sequences | HIGH (correctness critical) | MEDIUM (flash_attn_varlen_func) | P1 |
| Token budget enforcement | HIGH (prevents OOM) | LOW (simple token counting) | P1 |
| GPU memory pinning | HIGH (enables async transfer) | LOW (DataLoader flag) | P1 |
| Non-blocking GPU transfers | HIGH (overlap transfer/compute) | LOW (single flag) | P1 |
| First-Fit Decreasing packing | HIGH (1.5-2x throughput gain) | MEDIUM (sort + greedy pack) | P1 |
| FP16 precision | HIGH (2x memory/speed) | MEDIUM (validation required) | P1 |
| Dynamic batch sizing | MEDIUM (extra 10-20% gain) | MEDIUM (adaptive logic) | P2 |
| Multi-stream GPU execution | MEDIUM (20-30% speedup) | HIGH (CUDA stream mgmt) | P2 |
| Adaptive prefetching | MEDIUM (robustness) | HIGH (monitoring + tuning) | P2 |
| Advanced packing (LPFHP) | LOW (marginal gains) | HIGH (complex algorithm) | P3 |
| Continuous batching | LOW (incompatible architecture) | HIGH (major refactor) | P3 |
| Zero-copy data path | LOW (micro-optimization) | HIGH (careful memory mgmt) | P3 |

**Priority key:**
- P1: Must have for v2.0 launch — core async + packing functionality
- P2: Should have in v2.1+ — optimization after validation
- P3: Nice to have in v3.0+ — diminishing returns or architectural mismatch

## Implementation Complexity Breakdown

### Async DataLoader (MEDIUM complexity)

**What it involves:**
- Configure PyTorch DataLoader with num_workers, prefetch_factor, pin_memory
- Implement custom Dataset that loads FASTA files and tokenizes sequences
- Handle worker process lifecycle and cleanup
- Ensure thread-safety for model loading (spawn context already used in v1.0)

**Existing foundation:**
- v1.0 already uses multiprocessing.Pool for multi-GPU
- Token-based batching logic exists (toks_per_batch=2048)
- FASTA loading in sequence_utils.py

**New code required:**
- Custom torch.utils.data.Dataset wrapper
- Collate function for variable-length sequences
- Worker initialization function

**Estimated effort:** 2-3 days

### Sequence Packing with FlashAttention (HIGH complexity)

**What it involves:**
- Pack multiple sequences into single tensor (concatenate)
- Compute cu_seqlens (cumulative sequence length array) for each batch
- Replace standard attention with flash_attn_varlen_func
- Validate packed sequences produce identical embeddings to unpacked

**Existing foundation:**
- FlashAttention-2 already integrated (v1.0 Phase 4)
- Models already support FlashAttention
- Token counting logic exists

**New code required:**
- Sequence packing algorithm (FFD or LPFHP)
- cu_seqlens computation
- Attention call replacement in forward pass
- Unpacking logic to restore original sequence boundaries
- Validation tests (packed vs unpacked equivalence)

**Critical gotcha:**
- Must update positional embeddings and attention masks correctly
- Cross-contamination between sequences is silent correctness bug

**Estimated effort:** 5-7 days

### FP16 Precision Validation (MEDIUM complexity)

**What it involves:**
- Enable torch.autocast('cuda', dtype=torch.float16) for inference
- Extract embeddings in both FP32 and FP16
- Compute cosine similarity between FP32/FP16 embeddings
- Validate similarity >0.99 for random sample of sequences
- Debug dtype mismatches if they occur

**Existing foundation:**
- v1.0 currently forces FP32 due to previous dtype issues
- Model loading infrastructure exists
- Embedding extraction logic exists

**Known issues:**
- Previous attempt had dtype mismatch between BF16 model and FP32 input
- LayerNorm in transformers can have limited dynamic range in FP16
- ESM-2 3B model may be sensitive to precision

**Blockers resolved:**
- BF16/FP32 mismatch fixed in Phase 2 (commit 27a4ee9)
- Models now load in FP32 consistently

**New code required:**
- Autocast context manager integration
- Embedding comparison logic
- Validation test suite

**Estimated effort:** 3-4 days (including debugging)

### Intelligent Packing Algorithm (MEDIUM-HIGH complexity)

**What it involves:**
- Sort sequences by length (descending for FFD)
- Greedily pack sequences into batches respecting token budget
- Handle edge cases (sequence longer than budget, last batch padding)
- Optimize for minimal padding waste

**Algorithm choices:**

1. **First-Fit Decreasing (FFD)** — LOW complexity
   - Sort sequences by length descending
   - For each sequence, add to first batch with enough space
   - Create new batch if no fit
   - Expected padding waste: 20-30%

2. **Longest-Pack-First-Highest-Priority (LPFHP)** — HIGH complexity
   - Priority queue of batches by remaining capacity
   - Pack longest sequences first into highest-priority batch
   - Recompute priorities after each pack
   - Expected padding waste: 10-20%

**Recommendation:** Start with FFD. Profile padding waste. Only implement LPFHP if waste >30%.

**Estimated effort:**
- FFD: 2-3 days
- LPFHP: 5-7 days

### Multi-stream GPU Execution (HIGH complexity)

**What it involves:**
- Create separate CUDA streams for data transfer and compute
- Synchronize streams at batch boundaries
- Manage pinned memory buffers for async transfers
- Profile to verify overlap (nvprof or PyTorch Profiler)

**Requirements:**
- GPU must support concurrent copy and execution (all modern GPUs do)
- Transfers on non-default stream
- Source memory must be pinned

**Existing foundation:**
- Pinned memory already enabled (DataLoader)
- Non-blocking transfers implemented

**New code required:**
- CUDA stream creation and management
- Stream synchronization logic
- Stream-aware tensor lifecycle

**Risk:**
- Easy to introduce race conditions
- Debugging requires CUDA profiling tools
- Benefit depends on whether transfer is bottleneck

**Estimated effort:** 4-6 days (including profiling and debugging)

## Expected Behavior: Async DataLoader for Inference

### Workflow

1. **Main process spawns N CPU workers** (num_workers=4 recommended)
2. **Each worker independently:**
   - Loads FASTA file chunk
   - Parses sequences
   - Tokenizes sequences
   - Packs sequences into batch (respecting token budget)
   - Pins memory and queues batch
3. **Main process (GPU):**
   - Pops batch from prefetch queue (non-blocking)
   - Transfers batch to GPU (async, non-blocking)
   - Runs inference on current batch while next batch transfers
   - Writes results to output
   - Repeats

### Expected Performance Characteristics

**Without async DataLoader (v1.0 baseline):**
```
[Load FASTA] → [Tokenize] → [GPU Inference] → [Load FASTA] → [Tokenize] → [GPU Inference]
     ↑ CPU idle         ↑ GPU idle      ↑ CPU idle         ↑ GPU idle
```

**With async DataLoader (v2.0 target):**
```
Worker 1: [Load] → [Tokenize] → [Pack] → [Load] → [Tokenize] → [Pack] → ...
Worker 2:    [Load] → [Tokenize] → [Pack] → [Load] → [Tokenize] → ...
Worker 3:       [Load] → [Tokenize] → [Pack] → [Load] → [Tokenize] → ...
Worker 4:          [Load] → [Tokenize] → [Pack] → [Load] → [Tokenize] → ...
                              ↓ prefetch queue (depth = workers × prefetch_factor)
Main:                    [GPU] → [GPU] → [GPU] → [GPU] → [GPU] → [GPU] → ...
                              ↑ continuous inference, no gaps
```

### Tuning Parameters

| Parameter | Range | Impact | Recommendation |
|-----------|-------|--------|----------------|
| num_workers | 0-8 | CPU parallelism for I/O | Start with 4. Increase if GPU is starved (idle). Decrease if CPU memory is constrained. |
| prefetch_factor | 2-8 | Queue depth per worker | Start with 2. Increase if GPU has intermittent gaps. Each unit adds workers × batch_size × sequence_length memory. |
| pin_memory | True/False | Enable async transfer | Always True for GPU inference. |
| toks_per_batch | 1024-4096 | GPU memory usage | Current: 2048. Increase if GPU memory <80% utilized. Decrease if OOM. |

### Memory Footprint

**CPU memory:**
- Base: Main process memory
- Workers: num_workers × (model memory + batch memory)
- Queue: num_workers × prefetch_factor × batch_size × sequence_length × sizeof(token_id)

**Example:**
- num_workers=4, prefetch_factor=2, batch_size=16, sequence_length=512, token_id=int32
- Queue memory: 4 × 2 × 16 × 512 × 4 bytes = 256 KB (negligible)
- Worker memory: 4 × (0 if no model in worker) = 0 (if using GPU-side models)

**GPU memory:**
- Model: 3B params × 4 bytes (FP32) = 12 GB or 6 GB (FP16)
- Batch: batch_size × sequence_length × hidden_size × 4 bytes
- Activations: ~2-3x batch memory during forward pass

### Bottleneck Identification

**Symptoms of CPU bottleneck:**
- GPU utilization <80% (use nvidia-smi dmon)
- Increasing num_workers improves throughput
- Profiler shows GPU idle time

**Solution:** Increase num_workers or prefetch_factor

**Symptoms of GPU bottleneck:**
- GPU utilization >95%
- Increasing num_workers doesn't improve throughput
- CPU workers are idle (queue is full)

**Solution:** This is optimal. GPU is saturated.

**Symptoms of memory bottleneck:**
- OOM errors
- High CPU memory usage
- System swapping

**Solution:** Decrease num_workers, prefetch_factor, or batch size

## Expected Behavior: Sequence Packing for Inference

### What Sequence Packing Does

**Without packing (v1.0 baseline):**
```
Batch 1: [seq1: 200 tokens] [seq2: 180 tokens] [seq3: 220 tokens] → pad to max(220) = 220
         Total: 660 tokens + 120 padding = 780 tokens (15% waste)

Batch 2: [seq4: 400 tokens] [seq5: 100 tokens] → pad to max(400) = 400
         Total: 500 tokens + 300 padding = 800 tokens (37% waste)
```

**With packing (v2.0 target):**
```
Pack 1: [seq1: 200][seq2: 180][seq3: 220][seq5: 100] → concatenate to 700 tokens
        cu_seqlens = [0, 200, 380, 600, 700]
        Attention mask ensures seq1 doesn't attend to seq2, seq2 doesn't attend to seq3, etc.
        Total: 700 tokens, 0 padding, 0% waste

Pack 2: [seq4: 400][pad: 48] → single sequence, pad to budget
        Total: 400 tokens + 48 padding = 448 tokens (10% waste)
```

**Typical gains:**
- Datasets with 50-70% padding waste → 1.5-2x throughput with packing
- Datasets with uniform lengths → minimal benefit (already low waste)

### flash_attn_varlen_func Integration

**Standard FlashAttention call (v1.0):**
```python
# Input: [batch_size, seq_len, num_heads, head_dim]
output = flash_attn_func(
    q=query,
    k=key,
    v=value,
    causal=False  # bidirectional attention for BERT-like models
)
```

**Variable-length FlashAttention call (v2.0):**
```python
# Input: [total_tokens, num_heads, head_dim] — concatenated sequences
# cu_seqlens: [batch_size + 1] — cumulative sequence lengths
output = flash_attn_varlen_func(
    q=query_packed,
    k=key_packed,
    v=value_packed,
    cu_seqlens_q=cu_seqlens,
    cu_seqlens_k=cu_seqlens,
    max_seqlen_q=max_seq_len_in_pack,
    max_seqlen_k=max_seq_len_in_pack,
    causal=False
)
```

**cu_seqlens computation:**
```python
# Example: 3 sequences of lengths [200, 180, 220]
sequence_lengths = [200, 180, 220]
cu_seqlens = torch.tensor([0] + list(np.cumsum(sequence_lengths)), dtype=torch.int32)
# Result: [0, 200, 380, 600]
```

**Critical:** cu_seqlens must be int32, on CPU, and include leading 0.

### Correctness Validation

**Test:** Packed sequences must produce identical embeddings to unpacked sequences.

```python
# Unpacked (baseline)
emb1 = model(seq1)  # [1, 200, 768]
emb2 = model(seq2)  # [1, 180, 768]

# Packed
packed_input = torch.cat([seq1, seq2], dim=1)  # [1, 380, 768]
cu_seqlens = torch.tensor([0, 200, 380], dtype=torch.int32)
packed_emb = model_with_varlen(packed_input, cu_seqlens)  # [380, 768]

# Unpack
emb1_from_pack = packed_emb[0:200]    # [200, 768]
emb2_from_pack = packed_emb[200:380]  # [180, 768]

# Validate
assert torch.allclose(emb1.squeeze(0), emb1_from_pack, atol=1e-5)
assert torch.allclose(emb2.squeeze(0), emb2_from_pack, atol=1e-5)
```

### Edge Cases

| Case | Handling |
|------|----------|
| Single sequence in pack | Works fine. cu_seqlens = [0, seq_len]. No cross-attention to prevent. |
| Sequence longer than token budget | Reject or split into chunks (depends on use case). VirNucPro already chunks to 500bp. |
| Empty sequence | Skip. Don't pack zero-length sequences. |
| Token budget not fully utilized | Accept. Last batch may have padding. Optimize packing algorithm to minimize this. |
| Variable sequence lengths (50-500 tokens) | Ideal for packing. Sort by length, use FFD to minimize waste. |

## Performance Expectations

### Throughput Gain Estimates

Based on research literature and typical genomic sequence datasets:

| Optimization | Expected Gain | Confidence | Dependency |
|--------------|---------------|------------|------------|
| Async DataLoader (I/O overlap) | 1.2-1.5x | HIGH | CPU I/O is currently bottleneck (verify with profiling) |
| Sequence Packing (padding elimination) | 1.5-2x | HIGH | Dataset has 50-70% padding waste (typical for genomic data) |
| FP16 Precision | 2x memory, 1.5-2x speed | MEDIUM | Accuracy validation passes (LayerNorm may limit FP16 gains) |
| Combined (async + packing + FP16) | 3-5x | MEDIUM | All three optimizations stack multiplicatively |

**Target for v2.0:** 2-3x throughput over v1.0 baseline.

**Stretch goal:** 4-5x if FP16 works without accuracy degradation.

### Latency Impact

**Per-sequence latency may increase:**
- Packing adds sequences to same batch → each sequence waits for entire batch to complete
- Acceptable tradeoff for throughput-oriented inference (processing millions of sequences)

**Batch latency remains similar:**
- Batch size may decrease (more tokens per sequence due to packing)
- GPU compute time per batch stays constant (same total tokens)

**Use case fit:**
- VirNucPro is throughput-oriented (process large FASTA files, not real-time)
- Latency increase is acceptable tradeoff for 2-3x throughput gain

## Integration with Existing v1.0 Features

### Multi-GPU Parallelization (multiprocessing.Pool)

**Current:** Multiple worker processes, each with own GPU, process separate file chunks.

**Change:** Replace per-GPU multiprocessing.Pool with single process per GPU + async DataLoader.

**Compatibility:** Maintains process-per-GPU isolation. DataLoader workers handle I/O parallelism within process.

**Migration path:**
1. Keep file-based work distribution (input_dir with chunks)
2. Replace Pool.map with sequential file processing
3. Add DataLoader for async I/O within each file
4. Packing happens in DataLoader collate function

### Batch Processing (token-based batching)

**Current:** Batches formed by toks_per_batch=2048 budget.

**Change:** Packing still respects token budget, but sequences are concatenated instead of padded.

**Compatibility:** Token counting logic is same. Packing is additional optimization layer.

### FlashAttention-2

**Current:** flash_attn_func for standard batched attention.

**Change:** flash_attn_varlen_func for packed sequences with cu_seqlens.

**Compatibility:** Both functions from same flash-attn library. API is similar. Replace call sites.

**Validation:** Verify packed and unpacked produce same embeddings.

### Checkpoint Resume

**Current:** Checkpoints after each pipeline stage.

**Change:** Checkpoints remain at same granularity. DataLoader state is ephemeral (restarted from scratch on resume).

**Compatibility:** No change to checkpoint logic. DataLoader doesn't persist state.

### BF16 Mixed Precision (currently disabled)

**Current:** Models forced to FP32 due to dtype mismatch.

**Change:** Enable FP16 (not BF16) after validation. torch.autocast handles dtype conversion.

**Compatibility:** FP16 is more widely supported than BF16. Better precision for embeddings (more mantissa bits).

**Blocker:** Must validate embedding accuracy in FP16 vs FP32.

## Sources

### Async DataLoader and Prefetching
- [PyTorch DataLoader Official Documentation](https://docs.pytorch.org/docs/stable/data.html)
- [8 PyTorch DataLoader Tactics to Max Out Your GPU](https://medium.com/@Modexa/8-pytorch-dataloader-tactics-to-max-out-your-gpu-22270f6f3fa8)
- [Unveiling the Magic of PyTorch prefetch_factor](https://www.codegenes.net/blog/pytorch-prefetch_factor/)
- [PyTorch Data API: Worker, Pinned Memory, Prefetch, Non-blocking](https://oongjoon.github.io/pytorch/Data-loading/)
- [Data Prefetching in Deep Learning](https://www.jpatrickpark.com/post/prefetcher/)

### Sequence Packing for Transformers
- [PackedBERT: How to Accelerate NLP Tasks for Transformers with Packing](https://www.graphcore.ai/posts/packedbert-how-to-accelerate-nlp-tasks-for-transformers-with-packing)
- [Enhancing Training Efficiency Using Packing with Flash Attention](https://arxiv.org/html/2407.09105v4)
- [Packing Data for Efficient Training and Inference](https://lweitkamp.github.io/posts/packing/index.html)
- [Dynamic Batching vs. Sequence Packing](https://medium.com/better-ml/dynamic-batching-vs-sequence-packing-0ef4a3894dad)
- [Efficient LLM Pretraining: Packed Sequences and Masked Attention](https://huggingface.co/blog/sirluk/llm-sequence-packing)
- [Efficient Sequence Packing Without Cross-contamination](https://ar5iv.labs.arxiv.org/html/2107.02027)

### FlashAttention Variable-Length Sequences
- [GitHub - Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)
- [Hacking Vanilla FlashAttention for Variable-Length Inputs](https://gdewael.github.io/blog/flashattnvarlen/)
- [Improving Hugging Face Training Efficiency Through Packing with Flash Attention 2](https://huggingface.co/blog/packing-with-FA2)
- [FlashMask: Efficient and Rich Mask Extension of FlashAttention](https://arxiv.org/html/2410.01359v1)
- [FlexAttention: The Flexibility of PyTorch with the Performance of FlashAttention](https://pytorch.org/blog/flexattention/)

### GPU Memory and Transfer Optimization
- [PyTorch Guide on pin_memory() and non_blocking](https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html)
- [When to Set pin_memory to True in PyTorch](https://medium.com/data-scientists-diary/when-to-set-pin-memory-to-true-in-pytorch-75141c0f598d)
- [Memory Pinning to Accelerate Model Training](https://blog.dailydoseofds.com/p/memory-pinning-to-accelerate-model)

### CUDA Streams and Async Execution
- [How to Overlap Data Transfers in CUDA C/C++](https://developer.nvidia.com/blog/how-overlap-data-transfers-cuda-cc/)
- [CUDA Series: Streams and Synchronization](https://medium.com/@dmitrijtichonov/cuda-series-streams-and-synchronization-873a3d6c22f4)
- [CUDA Stream - Lei Mao's Log Book](https://leimao.github.io/blog/CUDA-Stream/)
- [PyTorch CUDA Streams Introduction](https://wentao.site/cuda_streams/)

### Dynamic Batching and Packing Algorithms
- [LLM Inference Optimization Techniques](https://www.clarifai.com/blog/llm-inference-optimization/)
- [LLM Inference Performance Engineering: Best Practices](https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices)
- [FairBatching: Fairness-Aware Batch Formation for LLM Inference](https://arxiv.org/html/2510.14392)

### FP16 Mixed Precision for Inference
- [Defeating the Training-Inference Mismatch via FP16](https://arxiv.org/html/2510.26788v1)
- [Mixed-Precision Quantization for Language Models](https://arxiv.org/html/2510.16805v1)
- [What is Half-Precision (FP16) in AI?](https://www.ultralytics.com/glossary/half-precision)
- [Mixed Precision Training in LLMs: FP16, BF16, FP8, and Beyond](https://medium.com/@dpratishraj7991/mixed-precision-training-in-llms-fp16-bf16-fp8-and-beyond-b4af13ca846f)

---
*Feature research for: VirNucPro v2.0 Async DataLoader and Sequence Packing*
*Researched: 2026-02-02*
*Confidence: HIGH (all major findings verified with official documentation and recent research)*
