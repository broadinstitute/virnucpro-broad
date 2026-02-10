# Features Research: v2.5 Model Optimizations Round 2

**Project:** VirNucPro GPU Optimization
**Milestone:** v2.5 Model Optimizations Round 2
**Research Mode:** Ecosystem - Feature Landscape
**Date:** 2026-02-09
**Overall Confidence:** HIGH

---

## Executive Summary

This research covers the feature landscape for v2.5 model optimizations, which builds on the successful v2.0 async architecture. The milestone targets six categories of improvements:

1. **DNABERT-S v2.0 port** - Async DataLoader + sequence packing for DNA embedding extraction
2. **ESM-2 model flexibility** - Configurable model selection (650M, 3B, custom paths)
3. **torch.compile integration** - Optional model compilation for 10-20% speedup
4. **Vectorized operations** - PyTorch-native position ID and embedding extraction
5. **Quick wins** - Environment variable caching, deque, combined tensor operations
6. **Code quality** - Environment variable centralization, duplicate code removal

**Key findings:**
- torch.compile is production-ready (PyTorch 2.2+) with 10-20% speedup at minimal risk
- Vectorized operations provide 5-15% improvement over Python loops with .item() calls
- Model-agnostic configuration patterns are well-established in Transformers ecosystem
- DNABERT-S v2.0 port removes 0.96x scaling bottleneck (current 4% slowdown on 2 GPUs)
- Environment variable sprawl is documented technical debt requiring centralization

---

## DNABERT-S v2.0 Port

### Current State (v1.0 Architecture)

DNABERT-S uses the legacy v1.0 bin-packing architecture while ESM-2 uses v2.0 async. This hybrid creates bottleneck:
- **2 GPU scaling:** 0.96x (4% slower than 1 GPU)
- **Root cause:** Greedy bin-packing file assignment with process coordination overhead

### v2.0 Architecture (From ESM-2)

Delivers 1.87x scaling on 2 GPUs (93.7% efficiency):
- Single-process-per-GPU with async DataLoader
- Sequence packing via GreedyPacker FFD (~92-94% efficiency)
- FlashAttention varlen for packed sequences
- Stride-based multi-GPU `[rank::world_size]`

### Feature Requirements

**DNA Tokenization:** K-mer overlapping windows (k=6), token budget accounts for expansion
**Multi-GPU:** Reuse SequenceIndex, GPUProcessCoordinator, module-level worker function
**Complexity:** MEDIUM (adapter for k-mer tokenization, memory footprint validation)

---

## ESM-2 Model Flexibility

### Current Hardcoding

Model name: `esm2_t36_3B_UR50D`, repr_layers: `[36]` (hardcoded in 6 locations)

### ESM-2 Family

| Model | Layers | repr_layers |
|-------|--------|-------------|
| esm2_t33_650M_UR50D | 33 | [33] |
| esm2_t36_3B_UR50D | 36 | [36] |
| esm2_t48_15B_UR50D | 48 | [48] |

### Expected CLI Pattern

```bash
python -m virnucpro predict input.fasta --esm-model 650M
python -m virnucpro predict input.fasta --esm-model /path/to/checkpoint
```

**Complexity:** LOW (configuration change only)

---

## torch.compile Integration

### Production Readiness (2026)

PyTorch 2.2+ with 10-20% speedup, 30-120s compilation overhead (amortized)

Best practices:
- `mode="reduce-overhead"` for inference
- Warmup with representative batch
- Triton cache reuse across runs

### Feature Requirements

```python
if ENV.compile_model:
    self.model = torch.compile(model, mode='reduce-overhead')
```

**Complexity:** LOW (env flag + numerical equivalence test)

---

## Vectorized Operations

### Current Bottlenecks

Position ID generation: `.item()` calls force 2N CPU-GPU syncs
Embedding extraction: List append + torch.stack

### Expected Speedup

Position IDs: 5x (0.5ms → 0.1ms for 100 seqs)
Embeddings: 5x (2ms → 0.4ms for 100 seqs)

**Complexity:** MEDIUM (complex indexing, edge cases)

---

## Quick Wins

1. **Env var caching:** Cache at init, not per-batch
2. **Deque:** Replace `list.pop(0)` O(n) with `deque.popleft()` O(1)
3. **Combined .to():** Single transfer `to(device=device, dtype=torch.float16)`

**Complexity:** LOW for all three

---

## Code Quality

### Environment Variable Centralization

```python
# virnucpro/core/env_config.py
@dataclass(frozen=True)
class EnvConfig:
    disable_packing: bool
    disable_fp16: bool
    v1_attention: bool
    compile_model: bool
```

**Complexity:** LOW (new module + import updates)

### Duplicate Code Extraction

Extract `_get_progress_queue()` (3 copies), `_validate_cuda_isolation()` (2 copies)

**Complexity:** LOW

---

## Feature Classification

### Table Stakes (Must Have)

| Feature | Why | Complexity |
|---------|-----|------------|
| DNABERT-S v2.0 port | Primary value: fix scaling bottleneck | MEDIUM |
| ESM-2 model selection | Required for 16GB GPUs (650M) | LOW |
| Env var centralization | Quality gate | LOW |
| Deque for packed_queue | Performance anti-pattern | LOW |

### Differentiators (Nice to Have)

| Feature | Value | Complexity |
|---------|-------|------------|
| torch.compile | 10-20% speedup, minimal risk | LOW |
| Vectorized position IDs | 5x speedup | MEDIUM |
| Vectorized embeddings | 5x speedup | MEDIUM |
| Combined .to() | Faster init, less memory | LOW |

### Anti-Features (Deliberately Excluded)

| Feature | Why Avoid |
|---------|-----------|
| BFD packing | 2-3% gain, high complexity |
| FlashAttention-3 | Requires H100, not RTX 4090 |
| Dynamic work distribution | Overkill (93.7% efficiency) |
| Tensor pooling | Not bottleneck in profiling |

---

## MVP Recommendation

**Phase 1 (Week 1):** DNABERT-S v2.0, ESM-2 model flexibility, env centralization
**Phase 2 (Week 2):** torch.compile, deque, combined .to()
**Phase 3 (Week 3):** Vectorized ops (if benchmarks show benefit)
**Deferred:** Function refactoring, BFD packing, FA3, work distribution

---

## Sources

### torch.compile
- [torch.compile and Diffusers: A Hands-On Guide](https://pytorch.org/blog/torch-compile-and-diffusers-a-hands-on-guide-to-peak-performance/)
- [Introduction to torch.compile — PyTorch Tutorials](https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html)
- [HuggingFace: Optimize inference using torch.compile()](https://huggingface.co/docs/transformers/main/perf_torch_compile)

### Vectorization
- [Vectorization and Broadcasting with PyTorch](https://blog.paperspace.com/pytorch-vectorization-and-broadcasting/)
- [What happens when you vectorize wide PyTorch expressions?](https://probablymarcus.com/blocks/2023/10/19/vectorizing-wide-pytorch-expressions.html)

### Configuration
- [Hugging Face Auto Classes](https://huggingface.co/docs/transformers/en/model_doc/auto)
- [Running Large Transformer Models on Mobile and Edge Devices](https://huggingface.co/blog/tugrulkaya/running-large-transformer-models-on-mobile)

### Internal
- `.planning/PROJECT.md`, `.planning/STATE.md`, `OPTIMIZATION_REVIEW.md`
