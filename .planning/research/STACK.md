# Stack Research: v2.5 Model Optimizations Round 2

**Project:** VirNucPro GPU Optimization v2.5
**Researched:** 2026-02-09
**Confidence:** HIGH (verified with official PyTorch 2.10 docs, fair-esm GitHub, Hugging Face model cards, and current WebSearch results)

## Executive Summary

v2.5 requires **minimal stack changes** to existing dependencies. All new capabilities are achievable with current PyTorch >=2.8.0, fair-esm 2.0.0, and transformers 4.30.0. The primary additions are:

1. **torch.compile integration** - Available in PyTorch >=2.8.0, requires understanding mode parameters and FlashAttention compatibility
2. **ESM-2 model variants** - fair-esm 2.0.0 already supports 8M/35M/150M/650M/3B/15B models via different model names
3. **DNABERT-S architecture details** - Already using zhihan1996/DNABERT-S with transformers 4.30.0, no new dependencies
4. **Vectorized operations** - Native PyTorch operations (torch.scatter_reduce, torch.cumsum) available in PyTorch >=2.8.0

**No new dependencies required.** All work involves using existing library features more effectively.

## Recommended Stack Changes

**None.** All v2.5 features achievable with current stack:

| Current | Remains | Purpose |
|---------|---------|---------|
| torch >=2.8.0 | torch >=2.8.0 | torch.compile, scatter_reduce, cumsum available |
| fair-esm 2.0.0 | fair-esm 2.0.0 | All ESM-2 model variants (8M to 15B) supported |
| transformers 4.30.0 | transformers 4.30.0 | DNABERT-S zhihan1996/DNABERT-S compatible |
| flash-attn >=2.6.0 | flash-attn >=2.6.0 | FlashAttention-2 for packed attention (no torch.compile support needed) |
| biopython | biopython | FASTA parsing |
| click >=8.0.0 | click >=8.0.0 | CLI interface |
| h5py | h5py | Shard storage |

**Optional for future (out of v2.5 scope):**
- torch-scatter 2.1.1 - Optimized segment_coo/segment_csr operations if native PyTorch scatter_reduce proves insufficient (LOW priority - native ops likely sufficient)

## torch.compile Integration

### Availability

torch.compile available in PyTorch >=2.0, fully supported in PyTorch 2.8.0+. **No version upgrade required.**

### Mode Parameters

| Mode | Compilation Time | Inference Speed | When to Use |
|------|-----------------|-----------------|-------------|
| "default" | Fast (~seconds) | Good baseline | General use, fast iteration |
| "reduce-overhead" | Moderate (~10-30s) | 10-15% faster | Small batches, CUDA graphs |
| "max-autotune" | Slow (~1-5 min) | 15-25% faster | Production, batch benchmarking |

**Recommendation for VirNucPro:** Start with "reduce-overhead" for production (targets small batches with CUDA graphs). Provide CLI flag `--compile-mode` with options "none", "reduce-overhead", "max-autotune".

**Expected speedup:** 10-20% inference speedup based on reduce-overhead mode with packed attention. max-autotune may provide 15-25% but requires longer first-run compilation.

### Compatibility with FlashAttention

**CRITICAL LIMITATION:** FlashAttention-2 custom CUDA kernels (flash-attn library) **cannot be compiled into single CUDA graphs** via torch.compile as of PyTorch 2.10.

**What this means:**
- torch.compile will compile the surrounding PyTorch operations (embedding lookups, layer norms, linear layers)
- FlashAttention kernel calls will remain as-is (already optimized, no compilation needed)
- Overall speedup will be 10-20% from non-attention operations, not the full model

**Fallback behavior:** When unsupported operations encountered, torch.compile falls back to eager mode for those operations only. Model execution continues normally.

### Dynamic Shapes

**Challenge:** VirNucPro uses variable-length sequences (cu_seqlens changes per batch).

**Solutions:**

1. **Default behavior (simplest):** torch.compile recompiles on shape change. With length-sorted packing, shapes remain relatively stable (few recompilations per run).

2. **Mark dynamic dimensions:** Use `torch._dynamo.mark_dynamic(tensor, dim)` to indicate sequence length dimension is dynamic. Generates more flexible kernels at cost of slightly lower performance.

3. **Pad to fixed sizes:** Not recommended - defeats packing efficiency gains.

**Recommendation:** Use default behavior initially. Recompilation overhead amortized across long inference runs (1M sequences). Monitor recompilation frequency with `torch._dynamo.explain()` if overhead becomes issue.

### Integration Points

**Where to apply torch.compile:**

```python
# In virnucpro/models/esm2_flash.py
class ESM2WithFlashAttention(nn.Module):
    def __init__(self, base_model, device, enable_fp16=True, compile_mode=None):
        super().__init__()
        self.model = base_model.to(device).eval()

        if enable_fp16:
            self.model = self.model.half()

        # Compile entire model if requested
        if compile_mode and compile_mode != "none":
            logger.info(f"Compiling ESM-2 model with mode={compile_mode}")
            self.model = torch.compile(self.model, mode=compile_mode)
```

**What gets compiled:**
- Embedding layers (token, position)
- All transformer blocks (linear layers, layer norms, GELU, residual connections)
- FlashAttention kernel calls (pass-through, not compiled)
- Output projection layers

**What remains eager:**
- Data loading (DataLoader workers)
- Tokenization (CPU operations)
- Packing logic (greedy bin packing)
- HDF5 writes (h5py)

### Known Limitations

1. **First-run compilation time:** 10s-5min depending on mode. Amortized over long runs.
2. **Memory overhead:** Compiled kernels cached in memory (~500MB-1GB). Acceptable on 24GB GPUs.
3. **FlashAttention pass-through:** Custom CUDA kernels not compiled, but this is fine (already optimal).
4. **Dynamic shape recompilation:** May recompile 5-10 times per run with variable sequence lengths. Monitor with torch._dynamo logging.

### Verification Strategy

```python
# Check if model is compiled
assert hasattr(model, '_orig_mod'), "Model not compiled"

# Monitor recompilation frequency
import torch._dynamo
torch._dynamo.config.verbose = True  # Log recompilations
```

## ESM-2 Model Variants

### Available Models (fair-esm 2.0.0)

| Model Name | Layers | Hidden Dim | Attn Heads | Params | VRAM (FP16) | repr_layers |
|------------|--------|-----------|------------|--------|-------------|-------------|
| esm2_t6_8M_UR50D | 6 | 320 | - | 8M | ~1GB | 0-6 |
| esm2_t12_35M_UR50D | 12 | 480 | - | 35M | ~1.5GB | 0-12 |
| esm2_t30_150M_UR50D | 30 | 640 | - | 150M | ~2.5GB | 0-30 |
| esm2_t33_650M_UR50D | 33 | 1280 | 20 | 650M | ~3-4GB | 0-33 |
| esm2_t36_3B_UR50D | 36 | 2560 | 40 | 3B | ~12-16GB | 0-36 |
| esm2_t48_15B_UR50D | 48 | - | - | 15B | ~60GB | 0-48 |

**Current implementation:** Hardcoded `esm2_t36_3B_UR50D` with `repr_layers=[36]` in 6 locations (PROJECT.md technical debt).

### Architecture Differences

**Key insight:** Models differ in **layer count**, **hidden dimension**, and **attention heads**. All use same ESM-2 architecture (transformer encoder), same tokenizer, same training objective.

**What changes between models:**

1. **repr_layers argument:** Must match layer count (e.g., `repr_layers=[33]` for 650M, `repr_layers=[36]` for 3B).
2. **Embedding dimension:** 650M produces 1280-dim embeddings, 3B produces 2560-dim embeddings.
3. **Memory requirements:** 650M fits 3-4x larger batches than 3B on same GPU.
4. **Inference speed:** 650M ~3-4x faster than 3B (fewer layers, smaller hidden dim).

**What remains the same:**

- Tokenizer (same 33-token alphabet for all models)
- Input format (same BOS/EOS tokens, padding behavior)
- Flash Attention integration (works with all models)
- FP16 precision (all models support FP16)
- Sequence packing (cu_seqlens format model-agnostic)

### Loading Different Models

**fair-esm 2.0.0 API:**

```python
import esm

# Load specific model variant
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()  # 650M model
model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()     # 3B model (current)

# Or use torch.hub
model, alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t33_650M_UR50D")
```

**Custom model paths:** fair-esm does not support loading from custom checkpoint paths via simple API. Models loaded by name only (downloaded from hub to `~/.cache/torch/hub/checkpoints/`).

**Recommendation:** Support model selection via CLI flag `--esm-model` with choices: "esm2_t33_650M_UR50D", "esm2_t36_3B_UR50D", "esm2_t48_15B_UR50D". Default to "esm2_t36_3B_UR50D" for backward compatibility.

### repr_layers Configuration

**Current problem:** `repr_layers=[36]` hardcoded in 6 locations, blocks model swaps.

**Solution:** Dynamic repr_layers based on model:

```python
MODEL_REPR_LAYERS = {
    "esm2_t6_8M_UR50D": [6],
    "esm2_t12_35M_UR50D": [12],
    "esm2_t30_150M_UR50D": [30],
    "esm2_t33_650M_UR50D": [33],
    "esm2_t36_3B_UR50D": [36],
    "esm2_t48_15B_UR50D": [48],
}

# Usage
model_name = "esm2_t33_650M_UR50D"
repr_layers = MODEL_REPR_LAYERS[model_name]
results = model(tokens, repr_layers=repr_layers)
```

**Extract embeddings:** Same pattern for all models:

```python
# All models return same dict structure
embeddings = results["representations"][repr_layers[-1]]  # [batch, seq, hidden_dim]
```

### Integration with Existing Code

**Changes required:**

1. **virnucpro/models/esm2_flash.py:** Add `model_name` parameter to constructor, load specified model
2. **virnucpro/pipeline/async_inference.py:** Pass model_name from config
3. **virnucpro/pipeline/multi_gpu_inference.py:** Propagate model_name to workers
4. **virnucpro/cli/predict.py:** Add `--esm-model` CLI flag
5. **virnucpro/core/config.py:** Add `esm.model_name` config key (default: "esm2_t36_3B_UR50D")

**Backward compatibility:** Default to "esm2_t36_3B_UR50D" when model_name not specified. Existing configs and CLIs continue to work.

### Performance Tradeoffs

**650M vs 3B:**

| Metric | 650M | 3B |
|--------|------|-----|
| Inference speed | ~3-4x faster | Baseline |
| Memory usage | ~3-4GB FP16 | ~12-16GB FP16 |
| Batch size (24GB GPU) | ~3-4x larger | Baseline |
| Embedding quality | Good for most tasks | Best (state-of-art) |
| Overall throughput | Potentially 5-10x higher | Baseline |

**Use case guidance:**
- 3B: Maximum accuracy, current production baseline
- 650M: Faster iteration, higher throughput, sufficient for many tasks
- 150M/35M/8M: Rapid prototyping, CPU inference (out of scope)

## DNABERT-S Tokenization & Architecture

### Model Details

**Hugging Face:** zhihan1996/DNABERT-S
**Base architecture:** DNABERT-2 (MosaicBERT variant for DNA sequences)
**Training:** Species-aware DNA embeddings (ISMB 2025)

### Architecture Specifications

| Property | Value |
|----------|-------|
| Layers | 12 transformer layers |
| Hidden size | 768 |
| Attention heads | 12 |
| Vocab size | ~4100 (DNA k-mers) |
| Max sequence length | 512 tokens |
| Precision support | FP16, BF16, FP32 |

**Current implementation:** v1.0 bin-packing with BF16 optimization (virnucpro/pipeline/parallel_dnabert.py).

### Tokenization

**Tokenizer API:**

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-S", trust_remote_code=True)

# Tokenize DNA sequence
seq = "ATCGATCGATCG"
tokens = tokenizer(seq, return_tensors="pt", padding=True, truncation=True)
# Returns: {'input_ids': tensor, 'attention_mask': tensor}
```

**Key differences from ESM-2:**

1. **Tokenization:** K-mer based (not single nucleotide). Tokenizer handles k-mer sliding window internally.
2. **Special tokens:** Uses BERT-style [CLS] and [SEP] tokens, not ESM-2 style BOS/EOS.
3. **Padding:** Uses attention_mask for padding (not padding_idx like ESM-2).
4. **Max length:** 512 tokens (vs ESM-2's 1024).

### v2.0 Async Port Requirements

**Current v1.0 architecture (parallel_dnabert.py):**
- Bin-packing file assignment by sequence count
- Multi-worker-per-GPU with BF16
- Token-based batching (toks_per_batch=2048)
- BertUnpadSelfAttention with MosaicBERT optimization

**v2.0 architecture (needed for parity with ESM-2):**
- Single-process-per-GPU with async DataLoader
- Sequence packing via VarlenCollator + GreedyPacker FFD
- FlashAttention varlen for packed format
- FP16 precision (or BF16 - both supported)

**Integration challenges:**

1. **Tokenization in collator:** DNABERT-S tokenization more complex than ESM-2 (k-mer sliding). Must tokenize in VarlenCollator after length-based sorting but before packing.

2. **Attention mask handling:** DNABERT-S uses attention_mask, ESM-2 uses padding_idx stripping. VarlenCollator must adapt for both models.

3. **Position IDs:** DNABERT-S uses BERT-style position embeddings (0 to seq_len-1). Same cu_seqlens boundary reset logic as ESM-2.

4. **Precision:** Current v1.0 uses BF16. v2.0 ESM-2 uses FP16. DNABERT-S supports both - choose based on compatibility testing (likely FP16 for consistency).

**Tokenization strategy for async DataLoader:**

```python
# In VarlenCollator.__call__():

if self.model_type == "esm2":
    # ESM-2: Tokenize protein sequences
    tokens = [self.esm_alphabet.encode(seq) for seq in sequences]
elif self.model_type == "dnabert":
    # DNABERT-S: Use transformers tokenizer
    tokenized = self.dnabert_tokenizer(
        sequences,
        return_tensors="pt",
        padding=False,  # We'll pack ourselves
        truncation=True,
        max_length=512
    )
    tokens = tokenized['input_ids']
```

### FlashAttention Integration

**Current DNABERT-S implementation (dnabert_flash.py):**
- Patches BertUnpadSelfAttention to use PyTorch SDPA
- Replaces broken Triton fallback with torch.nn.functional.scaled_dot_product_attention
- Supports BF16 natively

**v2.0 packed format requirements:**
- Must use flash_attn_varlen_func (same as ESM-2)
- Requires cu_seqlens, max_seqlen, position_ids
- DNABERT-S layer structure compatible (Q/K/V extraction same as BERT)

**Recommendation:** Reuse ESM-2 FlashAttention integration pattern (virnucpro/models/esm2_flash.py layer-level wrapper). DNABERT-S is BERT-based - same attention mechanism, different tokenization.

### Model Loading

**Current v1.0:**

```python
from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-S", trust_remote_code=True)
model = AutoModel.from_pretrained("zhihan1996/DNABERT-S", trust_remote_code=True)
```

**v2.0 requirements:**
- Load in gpu_worker (spawn context, deferred CUDA init)
- Convert to FP16 or BF16
- Wrap with FlashAttention integration (new DNABERTWithFlashAttention class)
- Compile with torch.compile if enabled

**No new dependencies required.** transformers 4.30.0 supports trust_remote_code and dynamic module loading.

### Precision Recommendations

**DNABERT-S training:** No public information about training precision found in search results. DNABERT-2 documentation mentions "Low Precision Layer Normalization" for efficiency.

**Options:**

1. **FP16 (recommended for consistency):** Matches ESM-2 v2.0 implementation. Proven stable with >0.99 cosine similarity. Unified precision across both models.

2. **BF16 (current v1.0):** Matches current DNABERT-S implementation. Wider dynamic range but lower precision. May be better for DNABERT-S if training used BF16.

**Testing protocol:** Port to FP16 initially (consistency with ESM-2). If numerical instability detected, fall back to BF16. Add `--dnabert-precision` CLI flag for override.

## Vectorized Operations

### Native PyTorch Operations (Available in >=2.8.0)

VirNucPro can replace Python loops with native PyTorch vectorized operations. **No new dependencies required.**

| Operation | PyTorch Function | Use Case in VirNucPro |
|-----------|-----------------|----------------------|
| Cumulative sum | torch.cumsum(tensor, dim=0) | Computing cu_seqlens from sequence lengths |
| Scatter reduce | torch.scatter_reduce(tensor, dim, index, src, reduce='sum') | Aggregating per-sequence embeddings from packed format |
| Gather | torch.gather(tensor, dim, index) | Extracting specific layer embeddings |
| Segment reduce | Native loop or torch-scatter | Per-sequence statistics (mean, max) |

### Current Opportunities

**1. Position IDs generation (virnucpro/models/packed_attention.py):**

```python
# Current: Python loop
def create_position_ids_packed(cu_seqlens, device):
    position_ids = []
    for i in range(len(cu_seqlens) - 1):
        seq_len = cu_seqlens[i+1] - cu_seqlens[i]
        position_ids.append(torch.arange(seq_len, device=device))
    return torch.cat(position_ids)

# Vectorized: Use cumsum + scatter
def create_position_ids_packed_vectorized(cu_seqlens, device):
    total_tokens = cu_seqlens[-1].item()
    # Create reset mask at sequence boundaries
    reset_mask = torch.zeros(total_tokens, dtype=torch.long, device=device)
    reset_mask[cu_seqlens[1:-1]] = 1  # Mark boundaries
    # Cumsum with resets
    position_ids = torch.arange(total_tokens, device=device) - torch.cumsum(reset_mask, dim=0)
    return position_ids
```

**Expected speedup:** ~2-10x for large cu_seqlens (1000+ sequences). Minimal for small batches (<100 sequences).

**2. Embedding extraction from packed format:**

```python
# Current: Python loop or list comprehension
embeddings_per_seq = []
for i in range(len(cu_seqlens) - 1):
    start, end = cu_seqlens[i], cu_seqlens[i+1]
    seq_embedding = packed_embeddings[start:end]  # [seq_len, hidden_dim]
    embeddings_per_seq.append(seq_embedding.mean(dim=0))  # Mean pooling

# Vectorized: Use segment_reduce (torch-scatter) or native loop with preallocated tensor
# Native PyTorch doesn't have segment_reduce, but can use scatter_reduce:
num_seqs = len(cu_seqlens) - 1
hidden_dim = packed_embeddings.shape[1]
seq_embeddings = torch.zeros(num_seqs, hidden_dim, device=device)

# Create index tensor mapping each token to its sequence
seq_indices = torch.zeros(total_tokens, dtype=torch.long, device=device)
for i in range(num_seqs):
    seq_indices[cu_seqlens[i]:cu_seqlens[i+1]] = i

# Scatter reduce with mean
# Note: PyTorch scatter_reduce doesn't support 'mean' directly, use sum + count
seq_embeddings = torch.scatter_reduce(
    seq_embeddings,
    dim=0,
    index=seq_indices.unsqueeze(1).expand(-1, hidden_dim),
    src=packed_embeddings,
    reduce='sum'
)
seq_counts = torch.bincount(seq_indices, minlength=num_seqs).float().unsqueeze(1)
seq_embeddings = seq_embeddings / seq_counts
```

**Complexity:** Native PyTorch scatter_reduce doesn't support 'mean' reduction. Requires sum + count division.

**Alternative:** Use torch-scatter library (segment_coo) for cleaner API, but adds dependency.

**Recommendation:** Keep current Python loop for embedding extraction (clear, debuggable). Vectorize position IDs generation only (simpler, clear benefit).

### torch-scatter Library (Optional)

**If native PyTorch operations prove insufficient:**

- **Library:** torch-scatter 2.1.1 (PyPI)
- **Installation:** `pip install torch-scatter -f https://data.pyg.org/whl/torch-2.8.0+cu128.html`
- **Operations:** segment_coo, segment_csr, scatter with 'mean' reduction
- **Pros:** Optimized GPU kernels, cleaner API for segment operations
- **Cons:** Additional dependency, installation complexity (CUDA version matching)

**Recommendation:** Defer torch-scatter to future optimization if native PyTorch operations insufficient. Not required for v2.5.

### Integration Strategy

1. **Phase 1 (v2.5):** Vectorize position IDs generation with torch.cumsum. Low risk, clear benefit.
2. **Phase 2 (future):** Benchmark embedding extraction vectorization. Only adopt if >2x speedup observed.
3. **torch-scatter:** Only add if native operations prove insufficient AND profiling shows segment operations are bottleneck (unlikely - packing and attention dominate).

## What NOT to Add

### torch-scatter (LOW priority)

**Why defer:**
- Native PyTorch operations (torch.scatter_reduce, torch.cumsum) sufficient for position IDs
- Embedding extraction loop not a bottleneck (packing and attention dominate)
- Additional dependency complexity (CUDA version matching)
- Installation friction (requires matching torch/CUDA versions)

**When to reconsider:**
- Profiling shows segment operations >10% of runtime (unlikely)
- Native PyTorch scatter_reduce API proves too verbose

### transformers Upgrade

**Current:** 4.30.0
**Latest:** 4.53.0+ (addresses 12 CVEs including 4 RCE)

**Why defer to future milestone:**
- DNABERT-S compatibility risk (trust_remote_code, dynamic modules)
- Scope creep for v2.5 (security vs performance optimization focus)
- Requires regression testing across all model loading paths

**Note:** PROJECT.md lists SEC-01 as future enhancement. Address in dedicated security milestone.

### Triton

**Why not add:**
- FlashAttention-2 already uses Triton kernels internally (flash-attn library dependency)
- torch.compile uses Triton for kernel generation automatically
- No need for manual Triton kernel development

### apex

**Why not add:**
- Original DNABERT documentation mentions apex for FP16 training
- VirNucPro does inference only (not training)
- PyTorch native FP16 (torch.cuda.amp) sufficient
- apex deprecated in favor of native PyTorch mixed precision

### New model libraries (esm3, DNABERT-3, etc.)

**Why defer:**
- v2.5 focused on optimizing existing models (ESM-2, DNABERT-S)
- fair-esm 2.0.0 and transformers 4.30.0 provide needed capabilities
- Model upgrades belong in separate feature milestone (model compatibility testing required)

## Sources

**torch.compile and PyTorch operations:**
- [torch.compile — PyTorch 2.10 documentation](https://docs.pytorch.org/docs/stable/generated/torch.compile.html)
- [Dynamic Shapes — PyTorch 2.10 documentation](https://docs.pytorch.org/docs/stable/torch.compiler_dynamic_shapes.html)
- [Everything You Need to Know About PyTorch Compile | Medium](https://medium.com/@lambdafluxofficial/everything-you-need-to-know-about-pytorch-compile-3d7fd94ce701)
- [Using Max-Autotune Compilation on CPU for Better Performance — PyTorch Tutorials](https://docs.pytorch.org/tutorials/unstable/max_autotune_on_CPU_tutorial.html)
- [torch.Tensor.scatter_reduce_ — PyTorch 2.10 documentation](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.scatter_reduce_.html)
- [torch.cumsum — PyTorch 2.10 documentation](https://docs.pytorch.org/docs/stable/generated/torch.cumsum.html)

**FlashAttention and torch.compile compatibility:**
- [FlashAttention with PyTorch Compile | Mixed Precision](https://benjaminwarner.dev/2023/08/16/flash-attention-compile)
- [GitHub - Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)
- [Definitive Guide to PyTorch, CUDA, Flash Attention Compatibility | Medium](https://medium.com/@vici0549/the-definitive-guide-to-pytorch-cuda-and-flash-attention-compatibility-ebec1161ec10)

**ESM-2 model variants:**
- [ESM-2 - BioNeMo Framework](https://docs.nvidia.com/bionemo-framework/2.0/models/esm2/)
- [facebook/esm2_t33_650M_UR50D · Hugging Face](https://huggingface.co/facebook/esm2_t33_650M_UR50D)
- [GitHub - facebookresearch/esm](https://github.com/facebookresearch/esm)
- [fair-esm · PyPI](https://pypi.org/project/fair-esm/)
- [Medium-sized protein language models perform well at transfer learning | Scientific Reports](https://www.nature.com/articles/s41598-025-05674-x)

**DNABERT-S:**
- [zhihan1996/DNABERT-S · Hugging Face](https://huggingface.co/zhihan1996/DNABERT-S)
- [GitHub - MAGICS-LAB/DNABERT_S](https://github.com/MAGICS-LAB/DNABERT_S)
- [GitHub - MAGICS-LAB/DNABERT_2](https://github.com/MAGICS-LAB/DNABERT_2)
- [DNABERT-2: EFFICIENT FOUNDATION MODEL | ICLR 2024](https://openreview.net/pdf?id=oMLQB4EZE1)

**Vectorized operations:**
- [PyTorch Scatter Documentation](https://pytorch-scatter.readthedocs.io/)
- [GitHub - rusty1s/pytorch_scatter](https://github.com/rusty1s/pytorch_scatter)
- [torch-scatter · PyPI](https://pypi.org/project/torch-scatter/)
- [Segment COO — pytorch_scatter 2.1.1 documentation](https://pytorch-scatter.readthedocs.io/en/latest/functions/segment_coo.html)

**Precision training (FP16/BF16):**
- [Defeating the Training-Inference Mismatch via FP16 | arXiv](https://arxiv.org/abs/2510.26788)
- [BF16 vs FP16: Understanding the Battle of Precision in AI Training](https://www.oreateai.com/blog/bf16-vs-fp16-understanding-the-battle-of-precision-in-ai-training/50de9eb1f1b903ad0b95b97181fdcc21)
