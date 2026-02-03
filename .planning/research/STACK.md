# Stack Research: Async DataLoader & Sequence Packing

**Domain:** High-throughput protein/DNA sequence processing with async I/O and sequence packing
**Researched:** 2026-02-02
**Confidence:** HIGH

## Executive Summary

This document outlines the recommended technology stack for VirNucPro v2.0's async DataLoader architecture and sequence packing optimization. The v1.0 multi-GPU implementation using multiprocessing.Pool suffers from N×11GB memory overhead (multiple model copies), pickle serialization tax, and GPU starvation from small batches. The v2.0 architecture replaces this with single-process-per-GPU + async DataLoader pattern using native PyTorch capabilities.

**Key Finding:** All required technologies are already in PyTorch >=2.8.0. No external dependencies needed for async DataLoader, FP16 mixed precision, or CUDA streams. Sequence packing requires custom implementation (no suitable off-the-shelf library for inference-only ESM-2/DNABERT workloads).

---

## Recommended Stack

### Core Technologies (Native PyTorch)

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| torch.utils.data.DataLoader | PyTorch >=2.8.0 | Async data loading with CPU workers | Native PyTorch API with num_workers, pin_memory, prefetch_factor for overlapping I/O with GPU compute. Well-tested, zero external dependencies, integrates seamlessly with existing ESM/transformers code. Best practices: num_workers=4-8 (2x CPU cores per GPU), prefetch_factor=2, persistent_workers=True. |
| torch.amp.autocast | PyTorch >=2.8.0 | Automatic mixed precision (FP16) | Replaces deprecated torch.cuda.amp.autocast. ESM-2 was trained in FP16 making it ideal for FP16 inference. Provides 2x memory reduction and 20-30% speedup with minimal accuracy loss (norm difference <1e-3 for embeddings). Use torch.float16 (not bfloat16) for closest match to training. |
| torch.cuda.Stream | PyTorch >=2.8.0 | CUDA stream management for I/O-compute overlap | Native PyTorch API for creating async execution pipelines (H2D transfer, compute, D2H transfer). Enables 20-40% latency reduction by overlapping data movement with inference. Existing stream_manager.py already implements this pattern. |
| torch.compile | PyTorch >=2.8.0 | JIT compilation for kernel optimization | Provides up to 2x inference speedup (1.35x-2x geomean on benchmarks) with single-line code addition. Particularly effective with FlashAttention and reduces Python overhead in async loops. Use mode="default" or "reduce-overhead" for inference. |

### Sequence Packing Approach

| Approach | Implementation | Purpose | Why This Strategy |
|----------|---------------|---------|-------------------|
| **Manual Greedy Packing** | Custom algorithm in Python | Pack variable-length sequences (100-3000nt, 30-1000aa) into fixed-size batches to minimize padding waste | No suitable off-the-shelf library exists for inference-only ESM-2/DNABERT workloads. torchtune's PackedDataset is training-focused (requires labels, RoPE encoding). Manual packing gives full control over batching strategy for inference. Implementation: sort sequences by length, greedily pack into max_tokens budget per batch. |
| torch.nn.attention.flex_attention | PyTorch >=2.5.0 | Document masking for packed sequences | Optional advanced optimization. FlexAttention with document masking prevents cross-sequence attention in packed batches. Requires Ampere+ GPUs (Turing fallback to SDPA). torchtune achieved 71% throughput boost and 2.4x training speedup. Inference gains depend on packing density. Start without, add if profiling shows attention overhead. |
| PyTorch NestedTensor (NJT) | PyTorch >=2.10.0 | Alternative to padding for variable-length batches | NOT RECOMMENDED for this use case. NestedTensor avoids padding but has eager mode overhead (more visible on smaller inputs), limited operator support, and requires model modifications. Manual packing + padding is simpler and better tested with ESM-2/DNABERT. Consider only if FlexAttention insufficient. |

### Supporting Libraries (Existing Stack)

| Library | Version | Purpose | Integration Notes |
|---------|---------|---------|-------------------|
| transformers | 4.30.0 (existing) | DNABERT-S model loading | Already validated. Compatible with torch.amp.autocast for FP16 inference. No changes needed. |
| fair-esm | 2.0.0 (existing) | ESM-2 3B model loading | Already validated. ESM-2 trained in FP16, excellent compatibility with torch.amp. Use with torch.no_grad() for inference-only workloads. No changes needed. |
| biopython | Latest (existing) | FASTA file parsing | Already used. Compatible with async DataLoader pattern (parse in CPU workers, transfer to GPU in main process). No changes needed. |

### Development Tools

| Tool | Purpose | Notes |
|------|---------|-------|
| torch.profiler | Performance profiling for async pipeline | Use to measure DataLoader worker efficiency, stream overlap, and identify bottlenecks. Critical for tuning num_workers and prefetch_factor. Records CUDA kernel launches, memory transfers, CPU operations. |
| nvidia-smi / torch.cuda.memory_summary() | GPU memory monitoring | Track memory savings from FP16 (expect ~11GB -> ~6GB per model with FP16 weights). Validate no OOM with larger batch sizes. Monitor utilization during async operations. |

---

## Installation

No new dependencies required. All features are native to PyTorch >=2.8.0.

```bash
# Verify PyTorch version (should be >=2.8.0)
python -c "import torch; print(f'PyTorch {torch.__version__}')"

# Existing dependencies remain unchanged
# transformers==4.30.0
# fair-esm==2.0.0
# biopython
```

---

## Alternatives Considered

| Recommended | Alternative | When to Use Alternative |
|-------------|-------------|-------------------------|
| **Manual greedy packing** | torchtune.datasets.PackedDataset | Only if training (not inference). torchtune's packing handles RoPE position IDs, document masking, and label packing for fine-tuning workloads. Not suitable for inference-only ESM-2 embedding extraction. Custom packing simpler and more flexible for inference. |
| **torch.amp.autocast (FP16)** | BF16 (bfloat16) | If Ampere+ GPUs (A100, H100) and willing to sacrifice some precision. BF16 has wider dynamic range than FP16 but was disabled in current code due to divergence issues (esm2_flash.py:79). FP16 is safer: ESM-2 trained in FP16, closer outputs to FP32 (norm diff <1e-3), works on older GPUs. |
| **Native DataLoader** | Ray Data / Dask | Only if processing exceeds single-node capacity (>8 GPUs) or distributed training needed. For 4-8 GPU single-node inference, native PyTorch DataLoader is simpler, lower overhead, better tested. Avoid premature distributed complexity. Ray adds 300KB per worker overhead. |
| **torch.cuda.Stream** | CUDA graphs (torch.cuda.CUDAGraph) | Only if batch sizes are 100% static. CUDA graphs require fixed input shapes and control flow. VirNucPro has variable sequence lengths (100-3000nt), making streams more flexible. Use "reduce-overhead" mode in torch.compile instead for Python overhead reduction with dynamic shapes. |
| **torch.compile** | TorchScript | Never for new projects. torch.compile supersedes TorchScript with better performance (2x vs 1.3x typical speedups), easier debugging, active development. TorchScript is in maintenance-only mode as of PyTorch 2.0+. |

---

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| torch.cuda.amp.autocast | Deprecated in PyTorch 2.5+ | torch.amp.autocast("cuda", ...) - New API uses device_type="cuda" argument instead of torch.cuda namespace. Old API still works but will be removed in future versions. |
| torch.cuda.amp.GradScaler | Deprecated and not needed for inference | Omit GradScaler entirely for inference (only needed for training with gradient scaling). Just use torch.amp.autocast wrapper around forward pass. |
| Third-party packing libraries (e.g., HuggingFace DataCollatorForSeq2SeqWithPacking) | Does not exist in official HuggingFace transformers library (verified 2026-02-02) | Implement custom greedy packing algorithm. Simple to implement (sort by length, bin-pack into token budget), full control over batching strategy, no unnecessary dependencies. |
| model.half() for FP16 conversion | Less flexible than autocast, requires manual dtype management | torch.amp.autocast context manager - handles FP32/FP16 ops automatically, better numerical stability, works with existing FP32 checkpoints without conversion, easier debugging. |
| Aggressive prefetch_factor (>4) | Increases memory pressure with minimal benefit | prefetch_factor=2 (PyTorch default for num_workers>0). Values >4 can cause OOM on CPU RAM, especially with large FASTA files. Start at 2, increase only if profiling shows worker starvation. Diminishing returns beyond 2-4. |
| num_workers > 8 | Memory explosion from too many worker processes | Cap at 8 workers per GPU process. Each worker duplicates dataset in memory. For typical systems, num_workers=4-8 is optimal (min(cpu_count // num_gpus, 8)). More workers rarely improve throughput. |

---

## Stack Patterns by Variant

**If GPU Memory Limited (<40GB per GPU):**
- Use torch.amp.autocast for FP16 to reduce model footprint from ~11GB to ~6GB
- Enables larger batch sizes (64-128 instead of 32-64)
- Accept small embedding differences (norm <1e-3) for 2x memory gain
- Critical for running ESM-2 3B on 24GB GPUs (RTX 3090, RTX 4090)

**If Variable Sequence Lengths (100-3000nt for DNA, 30-1000aa for protein):**
- Implement greedy sequence packing to reduce padding waste
- Sort sequences by length before packing to maximize packing efficiency
- Use FlexAttention document masking (optional) if Ampere+ GPU and profiling shows attention overhead
- Expected padding reduction: 30-50% depending on length distribution

**If Targeting Maximum Throughput (50K-200K seqs/hour):**
- Use torch.compile with mode="default" or "reduce-overhead"
- Configure DataLoader: num_workers=4-8, pin_memory=True, prefetch_factor=2
- Use torch.cuda.Stream for H2D/compute/D2H overlap (reuse existing stream_manager.py)
- Enable persistent_workers=True to avoid worker restart overhead
- Combine all optimizations for 5-8x speedup over v1.0

**If Debugging or Development:**
- Start with num_workers=0 (synchronous, easier debugging)
- Disable torch.compile (first run compilation overhead confuses profiling)
- Use FP32 initially, add FP16 after validating correctness
- Add optimizations incrementally, benchmark each change

---

## Version Compatibility

| Package | Compatible With | Notes |
|---------|-----------------|-------|
| PyTorch >=2.8.0 | transformers 4.30.0 | Verified compatible. transformers uses torch.nn.functional APIs that are stable across PyTorch 2.x. No breaking changes. |
| PyTorch >=2.8.0 | fair-esm 2.0.0 | Verified compatible. ESM models use standard nn.Module patterns. torch.amp.autocast works with ESM-2 (trained in FP16). No modifications needed. |
| torch.amp.autocast | ESM-2 3B model | HIGH compatibility. ESM-2 trained in FP16, making FP16 inference ideal. Expect norm differences <1e-3 vs FP32. Use torch.float16 (not bfloat16) for closest match to training dtype. |
| torch.compile | FlashAttention (SDPA) | Full compatibility. torch.compile optimizes SDPA kernels further via TorchInductor. Use together for maximum performance (2x-4x speedup over eager mode). No conflicts. |
| FlexAttention | Ampere+ GPUs (A100, H100) | Requires Turing+ (compute capability 7.5+) for FlashAttention backend. Falls back to memory-efficient SDPA on older GPUs, but performance gains reduced (2.4x -> 1.2x). Check GPU before enabling. |
| DataLoader persistent_workers | num_workers > 0 | persistent_workers=True requires num_workers > 0. Keeps workers alive between epochs/batches. Saves worker initialization overhead (~1-2s per restart). Critical for multi-batch workloads. |

---

## Implementation Strategy

### Phase 1: Async DataLoader (Highest Impact, MUST DO)

**What to implement:**
1. Replace multiprocessing.Pool with single process per GPU
2. Add DataLoader with num_workers=4, pin_memory=True, prefetch_factor=2, persistent_workers=True
3. Load model once per GPU process (not per DataLoader worker)

**Why first:**
- Eliminates N×11GB memory overhead (biggest current bottleneck per test.txt)
- Removes pickle serialization tax (50-100ms per file batch)
- Enables larger batch sizes (GPU no longer starved from small batches)
- Estimated impact: 2-3x throughput improvement

**Implementation notes:**
- Reuse existing dataloader_utils.py (create_optimized_dataloader already implements best practices)
- Workers parse FASTA files, return (id, sequence) tuples
- Main process tokenizes and moves to GPU with pin_memory for fast transfer
- Use spawn context (already used in v1.0) for CUDA compatibility

**Code pattern:**
```python
# Single process per GPU (not multiprocessing.Pool)
for gpu_id in range(num_gpus):
    # One model per GPU, kept loaded
    model = load_esm2_model(device=f'cuda:{gpu_id}')

    # Async DataLoader with CPU workers for I/O
    dataloader = DataLoader(
        dataset,
        batch_size=512,  # Much larger than v1.0
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )

    # Stream processing
    for batch in dataloader:
        embeddings = model(batch)
```

### Phase 2: FP16 Mixed Precision (Medium Impact, HIGH ROI)

**What to implement:**
1. Wrap model inference with torch.amp.autocast("cuda", dtype=torch.float16)
2. Keep embeddings in FP32 for storage (convert after mean pooling)
3. Validate embedding divergence <1e-3 norm difference vs FP32

**Why second:**
- Requires Phase 1 async architecture for memory headroom
- Provides 2x memory reduction -> enables larger batches
- Estimated impact: 1.5-2x throughput improvement (combined with larger batches)

**Implementation notes:**
- Update esm2_flash.py to remove "EXPERIMENTAL: Forcing FP32" (line 79-80)
- Use torch.float16 (not bfloat16) - ESM-2 trained in FP16, better compatibility
- Add validation tests comparing FP16 vs FP32 embeddings (reuse test_vanilla_equivalence.py pattern)

**Code pattern:**
```python
model.eval()
with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
    outputs = model(tokens)
    representations = outputs["representations"][36]

# Convert to FP32 for storage (after mean pooling)
embeddings = representations.mean(dim=1).float().cpu()
```

### Phase 3: Sequence Packing (Low-Medium Impact, OPTIONAL)

**What to implement:**
1. Custom greedy packing algorithm: sort sequences by length, bin-pack into token budget
2. Dynamic batching: variable batch_size based on sequence lengths to maintain constant token count
3. Optional: FlexAttention document masking if Ampere+ GPU

**Why third:**
- Lower priority than async + FP16
- Benefit depends on sequence length distribution (high variance = more benefit)
- Estimated impact: 1.2-1.5x throughput improvement (depends on dataset)

**Implementation notes:**
- Start simple: sort + greedy pack without FlexAttention
- Measure padding waste with current approach (log batch statistics)
- Add FlexAttention only if profiling shows attention is bottleneck (unlikely for ESM-2 inference)

**Code pattern:**
```python
# Simple greedy packing
def pack_sequences(sequences, max_tokens=4096):
    sequences = sorted(sequences, key=lambda x: len(x[1]))
    batches = []
    current_batch = []
    current_tokens = 0

    for seq_id, seq_str in sequences:
        seq_len = len(seq_str)
        if current_tokens + seq_len > max_tokens and current_batch:
            batches.append(current_batch)
            current_batch = []
            current_tokens = 0
        current_batch.append((seq_id, seq_str))
        current_tokens += seq_len

    if current_batch:
        batches.append(current_batch)

    return batches
```

### Phase 4: CUDA Streams + torch.compile (Optional, LOW Priority)

**What to implement:**
1. torch.cuda.Stream for H2D/compute/D2H overlap (reuse existing stream_manager.py)
2. torch.compile wrapper on model with mode="default"

**Why last:**
- Smallest incremental gains (10-20% each)
- Phase 1-3 provide 4-6x combined speedup
- Adds complexity (compilation cache, stream synchronization)

**Implementation notes:**
- stream_manager.py already exists, just needs integration with DataLoader loop
- torch.compile: wrap model after loading, before first inference (first batch will be slow)
- Use mode="reduce-overhead" for reduced Python overhead (good for small batches)

**Code pattern:**
```python
# torch.compile
model = torch.compile(model, mode="reduce-overhead")

# CUDA streams (existing stream_manager.py)
stream_processor = StreamProcessor(device=device)
for batch in dataloader:
    embeddings = stream_processor.process_batch_async(
        batch,
        transfer_fn=lambda b: b.to(device, non_blocking=True),
        compute_fn=lambda b: model(b)
    )
```

---

## Performance Expectations

Based on research and current v1.0 baseline from test.txt:

| Optimization | Expected Speedup | Memory Impact | Implementation Effort |
|--------------|------------------|---------------|----------------------|
| Async DataLoader (Phase 1) | 2-3x | -75% GPU memory (N×11GB -> 1×11GB per GPU) | High (architectural change) |
| FP16 Mixed Precision (Phase 2) | 1.5-2x | -45% model memory (11GB -> 6GB) | Medium |
| Sequence Packing (Phase 3) | 1.2-1.5x | Neutral (reduces padding waste) | Medium |
| CUDA Streams (Phase 4) | 1.1-1.2x | Neutral | Low (reuse existing code) |
| torch.compile (Phase 4) | 1.1-1.2x | +10% (compilation cache) | Low (one-line change) |
| **Combined (Phases 1-3)** | **4-6x** | **-80% total** | **High** |
| **Combined (All phases)** | **5-8x** | **-75% total** | **High** |

**Target validation:**
- v1.0 baseline: ~10K-30K seqs/hour (1-6M seqs in 1-9 hours per test.txt)
- v2.0 target: 50K-200K seqs/hour
- Expected: Phases 1-2 sufficient to hit 50K/hour, Phases 3-4 to reach 100K-200K/hour

**Confidence levels:**
- Phase 1 (Async DataLoader): HIGH confidence (2-3x) - addresses documented bottlenecks
- Phase 2 (FP16): HIGH confidence (1.5-2x) - ESM-2 trained in FP16, proven compatible
- Phase 3 (Packing): MEDIUM confidence (1.2-1.5x) - depends on sequence length variance
- Phase 4 (Streams/compile): MEDIUM confidence (1.1-1.2x each) - smaller incremental gains

---

## Sources

### Official Documentation (HIGH confidence)
- [PyTorch DataLoader Documentation](https://pytorch.org/docs/stable/data.html) - num_workers, pin_memory, prefetch_factor parameters
- [PyTorch AMP Documentation](https://pytorch.org/docs/stable/amp.html) - torch.amp.autocast API, deprecated torch.cuda.amp namespace
- [PyTorch CUDA Streams Documentation](https://pytorch.org/docs/stable/notes/cuda.html) - torch.cuda.Stream for async execution
- [PyTorch FlexAttention Documentation](https://pytorch.org/docs/stable/nn.attention.flex_attention.html) - document masking for packed sequences
- [PyTorch NestedTensor Documentation](https://pytorch.org/docs/stable/nested.html) - variable-length sequence handling
- [PyTorch pinmem_nonblock Tutorial](https://pytorch.org/tutorials/intermediate/pinmem_nonblock.html) - pin_memory best practices

### Performance Benchmarks (MEDIUM-HIGH confidence)
- [torch.compile Tutorial (PyTorch official, updated 2026-01-26)](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html) - 1.35x-2x inference speedup claims
- [torchtune Sample Packing (PyTorch official)](https://pytorch.org/torchtune/stable/basics/packing.html) - FlexAttention 71% throughput boost, 2.4x training speedup
- [PyTorch FlexAttention Blog](https://pytorch.org/blog/flexattention/) - 2.04x inference speedup on LLaMa3
- [DRAMA NJT Performance](https://pytorch.org/blog/drama-model-inference-efficiency-boosted/) - 1.7x-2.3x inference speedup with NestedTensor

### Community Best Practices (MEDIUM confidence)
- [8 PyTorch DataLoader Tactics (Medium, 2025+)](https://medium.com/@Modexa/8-pytorch-dataloader-tactics-to-max-out-your-gpu-22270f6f3fa8) - num_workers best practices (2x CPU cores per GPU)
- [PyTorch DataLoader Deep Dive (oongjoon.github.io)](https://oongjoon.github.io/pytorch/Data-loading/) - pin_memory and prefetch_factor interaction
- [Optimizing Data Transfer in Batched Inference (Medium, Jan 2026)](https://chaimrand.medium.com/optimizing-data-transfer-in-batched-ai-ml-inference-workloads-a9f4165208b8) - DataPrefetcher pattern with CUDA streams
- [Variable Length Sequences Tutorial (PyTorch official)](https://pytorch.org/tutorials/intermediate/variable_length_attention_tutorial.html) - varlen_attn API for packed sequences
- [Efficient LLM Pretraining with Packed Sequences (HuggingFace)](https://huggingface.co/blog/sirluk/llm-sequence-packing) - sequence packing patterns

### ESM-2 Specific (MEDIUM-HIGH confidence)
- [ESM-2 FP16 Discussion (GitHub facebookresearch/esm #684)](https://github.com/facebookresearch/esm/discussions/684) - ESM-2 trained in FP16, norm difference <1e-3 for embeddings
- [FastESM2_650 (HuggingFace)](https://huggingface.co/Synthyra/FastESM2_650) - "FP16 has closer outputs to FP32 than BF16, loading in FP16 recommended"
- [Efficient ESM-2 Inference PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12481099/) - FlashAttention + sequence packing achieves 4-9x faster inference

### Negative Findings (LOW confidence, flagged for validation)
- HuggingFace DataCollatorForSeq2SeqWithPacking: NOT FOUND in official transformers library (searched 2026-02-02). Custom implementation required.

---

*Stack research for: VirNucPro v2.0 async DataLoader and sequence packing*
*Researched: 2026-02-02*
*Confidence: HIGH*
