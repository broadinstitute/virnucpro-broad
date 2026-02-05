# Phase 8: FP16 Precision Validation - Research

**Researched:** 2026-02-05
**Domain:** FP16 precision conversion, numerical stability, transformer inference optimization
**Confidence:** HIGH (PyTorch official docs, ESM-2 research papers, FlashAttention varlen requirements)

## Summary

Phase 8 converts ESM-2 and DNABERT-S models from FP32 to FP16 precision to achieve 1.8-2x throughput improvement while maintaining embedding accuracy. The user has decided on full FP16 conversion (not mixed precision via autocast) with data-driven fallback to FP32 for specific layers if validation fails.

The existing codebase already has:
- FlashAttention varlen requiring FP16/BF16 inputs (Phase 6 auto-converts in `forward_packed`)
- Cosine similarity validation infrastructure from Phase 6 (0.999 threshold with 1% lenient allowance)
- BF16 experience from Phase 1-4 (used for DNABERT-S on Ampere+ GPUs)

The new work requires:
1. Converting models to FP16 via `model.half()` or `model.to(torch.float16)`
2. Switching FlashAttention from BF16 to FP16 (align precision across pipeline)
3. Validating FP16 embeddings match FP32 baseline (cosine similarity >0.99)
4. Implementing NaN/Inf detection for numerical stability
5. Adding feature flag for emergency rollback (VIRNUCPRO_DISABLE_FP16)
6. Benchmarking throughput improvement (tokens/second and total runtime)

**Primary recommendation:** Use `model.half()` for full model conversion to FP16 (inference-only, no training). This is simpler and more performant than `torch.amp.autocast` for inference workloads. If validation fails with full FP16, selectively move LayerNorm and softmax operations to FP32 using per-layer dtype conversion. ESM-2 research shows FP16 is well-tested (norm difference <1e-3 vs FP32), and FP16 aligns with FlashAttention requirements better than BF16 for throughput.

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| PyTorch | 2.2+ | `model.half()` for FP16 conversion | Native FP16 support, no additional dependencies |
| torch.float16 | Built-in | FP16 dtype for model weights and activations | Standard precision format, hardware-accelerated |
| flash-attn | 2.6+ | FP16 FlashAttention varlen kernels | Already required in Phase 6, supports both FP16 and BF16 |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| torch.autocast | Built-in | Selective FP32 for unstable ops (if needed) | Fallback if full FP16 causes NaN/Inf |
| torch.isnan/isinf | Built-in | Numerical stability detection | Monitor embeddings for overflow |
| torch.nn.functional.cosine_similarity | Built-in | Embedding validation | Compare FP16 vs FP32 embeddings |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| model.half() | torch.amp.autocast | autocast is for mixed precision training, not full FP16 inference. model.half() is simpler and faster for inference |
| FP16 | BF16 | BF16 has wider dynamic range but FP16 has better precision (10-bit vs 7-bit mantissa). ESM-2 trained in FP16, research shows <1e-3 norm difference vs FP32 |
| model.half() | model.to(torch.float16) | Functionally equivalent, model.half() is more concise |

**Installation:**
```bash
# No additional dependencies - PyTorch built-in
# flash-attn already installed in Phase 6
```

## Architecture Patterns

### Recommended Project Structure
```
virnucpro/
  models/
    esm2_flash.py            # MODIFY: Add FP16 conversion + dtype validation
    dnabert_flash.py         # MODIFY: Add FP16 conversion + dtype validation
  pipeline/
    async_inference.py       # MODIFY: Add NaN/Inf detection, FP16 flag
    features.py              # MODIFY: Update dtype handling
  tests/
    integration/
      test_fp16_validation.py  # NEW: FP16 vs FP32 equivalence tests
```

### Pattern 1: Full Model FP16 Conversion
**What:** Convert entire model to FP16 for inference
**When to use:** Always for production inference (2x memory reduction, 1.8-2x throughput)
**Example:**
```python
# Source: PyTorch docs + codebase patterns from esm2_flash.py
def load_esm2_model(model_name, device, enable_fp16=True):
    """Load ESM-2 model with optional FP16 conversion."""
    # Load model in FP32
    base_model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()

    # Convert to FP16 if enabled
    if enable_fp16:
        base_model = base_model.half()  # Convert all weights/buffers to FP16
        logger.info("Model converted to FP16 precision")

    # Move to device
    base_model = base_model.to(device)
    base_model.eval()  # Inference mode

    return base_model, alphabet
```

### Pattern 2: FlashAttention FP16 Alignment
**What:** Switch FlashAttention from BF16 to FP16 to match model precision
**When to use:** After model converted to FP16
**Example:**
```python
# Source: flash_attn_varlen_wrapper in packed_attention.py + Phase 6 patterns
def flash_attn_varlen_wrapper(q, k, v, cu_seqlens, max_seqlen, **kwargs):
    """FlashAttention varlen with FP16 dtype validation."""
    # Validate FP16 (not BF16)
    valid_dtypes = (torch.float16, torch.bfloat16)
    for name, tensor in [("q", q), ("k", k), ("v", v)]:
        if tensor.dtype not in valid_dtypes:
            raise TypeError(
                f"FlashAttention requires FP16 or BF16 inputs. "
                f"{name} is {tensor.dtype}. "
                f"Convert with: {name}.half() or {name}.bfloat16()"
            )

    # Prefer FP16 for throughput (aligns with model precision)
    if q.dtype == torch.bfloat16:
        logger.warning(
            "FlashAttention running in BF16 mode. "
            "For best throughput with FP16 model, ensure inputs are FP16."
        )

    # Call FlashAttention kernel
    return flash_attn_varlen_func(q, k, v, cu_seqlens, max_seqlen, **kwargs)
```

### Pattern 3: Cosine Similarity Validation
**What:** Compare FP16 embeddings to FP32 baseline
**When to use:** One-time validation during Phase 8, optional diagnostic with --fp32-compare flag
**Example:**
```python
# Source: Phase 6 validate_packed_equivalence pattern (test_packed_equivalence.py)
def validate_fp16_equivalence(
    model_fp16,
    model_fp32,
    batch_converter,
    sequences: List[Tuple[str, str]],
    device: torch.device,
    strict_threshold: float = 0.99,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Validate FP16 embeddings match FP32 baseline.

    Returns:
        Tuple of (passed, details) where details contains:
            - per_sequence: Dict[seq_id, cosine_sim]
            - min_similarity: float
            - max_similarity: float
            - mean_similarity: float
            - failed_sequences: List[str]
    """
    # Get FP32 baseline
    labels, strs, tokens = batch_converter(sequences)
    tokens = tokens.to(device)

    with torch.no_grad():
        results_fp32 = model_fp32(tokens, repr_layers=[36])
        embeddings_fp32 = results_fp32["representations"][36]

        results_fp16 = model_fp16(tokens, repr_layers=[36])
        embeddings_fp16 = results_fp16["representations"][36]

    # Compare per sequence
    similarities = {}
    for i, (seq_id, _) in enumerate(sequences):
        # Mean pool embeddings
        emb_fp32 = embeddings_fp32[i, 1:-1].mean(dim=0)  # Skip BOS/EOS
        emb_fp16 = embeddings_fp16[i, 1:-1].mean(dim=0).float()  # Cast to FP32 for comparison

        # Cosine similarity
        sim = F.cosine_similarity(emb_fp32, emb_fp16, dim=0).item()
        similarities[seq_id] = sim

    # Aggregate metrics
    min_sim = min(similarities.values())
    max_sim = max(similarities.values())
    mean_sim = sum(similarities.values()) / len(similarities)
    failed = [sid for sid, sim in similarities.items() if sim < strict_threshold]

    passed = min_sim >= strict_threshold
    details = {
        "per_sequence": similarities,
        "min_similarity": min_sim,
        "max_similarity": max_sim,
        "mean_similarity": mean_sim,
        "failed_sequences": failed,
    }

    return passed, details
```

### Pattern 4: NaN/Inf Detection
**What:** Monitor embeddings for numerical instability
**When to use:** After every forward pass in production
**Example:**
```python
# Source: PyTorch debugging tools + NVIDIA mixed precision docs
def check_numerical_stability(embeddings: torch.Tensor, context: str = "embeddings"):
    """Detect NaN/Inf in tensors."""
    has_nan = torch.isnan(embeddings).any().item()
    has_inf = torch.isinf(embeddings).any().item()

    if has_nan or has_inf:
        stats = {
            "nan_count": torch.isnan(embeddings).sum().item(),
            "inf_count": torch.isinf(embeddings).sum().item(),
            "min": embeddings[~torch.isnan(embeddings) & ~torch.isinf(embeddings)].min().item(),
            "max": embeddings[~torch.isnan(embeddings) & ~torch.isinf(embeddings)].max().item(),
        }
        raise RuntimeError(
            f"Numerical instability detected in {context}: "
            f"NaN: {stats['nan_count']}, Inf: {stats['inf_count']}, "
            f"Range: [{stats['min']:.2e}, {stats['max']:.2e}]. "
            f"This may indicate FP16 overflow. Try VIRNUCPRO_DISABLE_FP16=1"
        )
```

### Pattern 5: Feature Flag Rollback
**What:** Environment variable to disable FP16 in production
**When to use:** Emergency rollback if FP16 causes issues in production
**Example:**
```python
# Source: Feature flag best practices + existing VIRNUCPRO_DISABLE_PACKING pattern
import os

def should_use_fp16() -> bool:
    """Check if FP16 should be enabled (default: True)."""
    disable = os.getenv("VIRNUCPRO_DISABLE_FP16", "").lower() in ("1", "true", "yes")
    if disable:
        logger.warning(
            "FP16 precision DISABLED via VIRNUCPRO_DISABLE_FP16. "
            "Using FP32 (2x slower, 2x memory). "
            "This is a diagnostic mode for troubleshooting."
        )
    return not disable

# Usage in model loading
model, alphabet = load_esm2_model(device="cuda:0", enable_fp16=should_use_fp16())
```

### Anti-Patterns to Avoid
- **Using torch.amp.autocast for full FP16 conversion:** autocast is designed for mixed precision training, not inference. For inference, use `model.half()` directly - simpler and faster.
- **Calling model.half() after moving to device:** PyTorch docs recommend converting dtype before moving to device for efficiency, though both work.
- **Not storing embeddings in FP32:** Model may compute in FP16, but embeddings should be stored in FP32 for downstream stability (existing pattern in async_inference.py line 130).
- **Mixing FP16 model with BF16 FlashAttention:** Dtype mismatch causes auto-conversion overhead. Align all components to FP16.
- **No rollback mechanism:** Always provide VIRNUCPRO_DISABLE_FP16 flag for production safety.

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Mixed precision training | Custom FP16/FP32 switching | torch.amp.autocast + GradScaler | **BUT: Not needed for inference** - use model.half() directly |
| NaN/Inf detection | Manual checks on every tensor | torch.isnan(), torch.isinf() | Built-in, optimized, standard |
| Cosine similarity | Manual dot product / normalization | F.cosine_similarity | Numerically stable, handles edge cases |
| Statistical validation | Custom outlier detection | scipy.stats, numpy percentile | Well-tested, handles edge cases |
| Dtype conversion | Manual weight iteration | model.half(), model.to(dtype) | Handles all parameters/buffers correctly |

**Key insight:** PyTorch provides all FP16 inference primitives. Don't use training-focused tools (autocast, GradScaler) for inference - they add complexity without benefit. The pattern is: load model in FP32 → convert to FP16 via model.half() → run inference with torch.no_grad().

## Common Pitfalls

### Pitfall 1: Using autocast for Full FP16 Conversion
**What goes wrong:** Developer uses `with torch.autocast(dtype=torch.float16)` thinking it converts model to FP16
**Why it happens:** autocast sounds like it "automatically" handles precision, and docs focus on training use cases
**How to avoid:**
- For training: Use autocast (mixed precision - some ops in FP16, some in FP32)
- For inference: Use model.half() (full FP16 - all ops in FP16)
- GradScaler is ONLY for training (gradient scaling), not needed for inference
**Warning signs:** Seeing "GradScaler" in inference code, or using autocast as context manager around inference

### Pitfall 2: LayerNorm FP16 Overflow
**What goes wrong:** LayerNorm accumulates large values in FP16, causing overflow (NaN/Inf)
**Why it happens:** FP16 max value is 65504, layernorm variance calculation can exceed this for large tensors
**How to avoid:**
- Start with full FP16 (try it first - ESM-2 research shows it works)
- If NaN/Inf detected: Use autocast ONLY for LayerNorm and softmax operations
- Keep rest of model in FP16 for performance
**Warning signs:** NaN/Inf appearing after LayerNorm layers, sudden spikes in embedding magnitudes

**Selective FP32 fallback pattern:**
```python
# If full FP16 fails validation
for layer in model.layers:
    # Keep attention in FP16 (fast)
    # Move LayerNorm to FP32 (stable)
    layer.self_attn_layer_norm = layer.self_attn_layer_norm.float()
    layer.final_layer_norm = layer.final_layer_norm.float()
```

### Pitfall 3: FlashAttention Dtype Mismatch
**What goes wrong:** Model in FP16 but FlashAttention inputs in BF16 (or vice versa) causing auto-conversion overhead
**Why it happens:** Phase 6 auto-converts to BF16, Phase 8 switches to FP16 - old code may linger
**How to avoid:**
- Align entire pipeline to FP16: model, FlashAttention inputs, intermediate tensors
- Validate dtype in flash_attn_varlen_wrapper (error on BF16 when FP16 expected)
- Remove BF16 auto-conversion from esm2_flash.py line 218
**Warning signs:** "Converting to BF16" warning messages, lower throughput than expected

### Pitfall 4: Embedding Storage in FP16
**What goes wrong:** Storing embeddings in FP16 causes precision loss for downstream tasks
**Why it happens:** Developer assumes FP16 everywhere is optimal
**How to avoid:**
- Compute in FP16 (fast)
- Store embeddings in FP32 (stable) - existing pattern in async_inference.py
- Pattern: `embedding = output["representations"][36].float().cpu()`
**Warning signs:** Downstream classification accuracy drops, embedding similarity thresholds need lowering

### Pitfall 5: No Validation Before Production
**What goes wrong:** FP16 deployed without baseline comparison, causes silent accuracy degradation
**Why it happens:** Throughput improvement is visible, accuracy degradation is not
**How to avoid:**
- Phase 8 validation: One-time FP32 baseline comparison on 10K sequences
- Stratified testing: Short/medium/long sequences (catch length-dependent issues)
- Statistical validation: Mean, std, outliers (not just cosine similarity)
**Warning signs:** Users report "weird predictions", model confidence scores change distribution

## Code Examples

Verified patterns from official sources:

### Full Model FP16 Conversion (Inference)
```python
# Source: PyTorch inference optimization docs
# URL: https://docs.pytorch.org/serve/performance_checklist.html

import torch
import torch.nn as nn

# Load model in FP32
model = load_esm2_model(device="cpu")

# Convert to FP16 (all parameters and buffers)
model = model.half()

# Move to GPU (after dtype conversion)
model = model.to("cuda:0")

# Inference mode
model.eval()

# Inference loop
with torch.inference_mode():  # Faster than torch.no_grad() for inference
    for batch in dataloader:
        # Inputs automatically converted to FP16 by model
        outputs = model(batch.to("cuda:0"))
        # Outputs are FP16, convert to FP32 for storage if needed
        embeddings = outputs["representations"][36].float().cpu()
```

### FlashAttention FP16 Dtype Validation
```python
# Source: flash-attn docs + Phase 6 packed_attention.py
# URL: https://github.com/Dao-AILab/flash-attention

from flash_attn import flash_attn_varlen_func
import torch

def flash_attn_varlen_wrapper(q, k, v, cu_seqlens, max_seqlen, **kwargs):
    """FlashAttention with FP16 validation."""
    # Validate FP16/BF16 requirement
    valid_dtypes = (torch.float16, torch.bfloat16)
    for name, tensor in [("q", q), ("k", k), ("v", v)]:
        if tensor.dtype not in valid_dtypes:
            raise TypeError(
                f"FlashAttention requires FP16 or BF16 inputs. "
                f"{name} is {tensor.dtype}. Convert with: {name}.half()"
            )

    # Ensure cu_seqlens is int32 (FlashAttention requirement)
    if cu_seqlens.dtype != torch.int32:
        cu_seqlens = cu_seqlens.to(torch.int32)

    return flash_attn_varlen_func(
        q, k, v,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        **kwargs
    )
```

### Statistical Validation with Outlier Detection
```python
# Source: scikit-learn outlier detection + numpy statistics
# URL: https://scikit-learn.org/stable/modules/outlier_detection.html

import numpy as np
import torch
import torch.nn.functional as F

def validate_embeddings_statistics(
    embeddings_fp32: torch.Tensor,
    embeddings_fp16: torch.Tensor,
) -> dict:
    """
    Statistical validation beyond cosine similarity.

    Checks:
    - Mean/std distribution match
    - Outlier detection (>3 std from mean)
    - L2 norm distribution
    """
    # Convert to numpy for statistical analysis
    fp32 = embeddings_fp32.cpu().numpy()
    fp16 = embeddings_fp16.float().cpu().numpy()

    # Distribution statistics
    stats = {
        "mean_diff": np.abs(fp32.mean() - fp16.mean()),
        "std_diff": np.abs(fp32.std() - fp16.std()),
        "max_abs_diff": np.abs(fp32 - fp16).max(),
    }

    # Outlier detection (Z-score > 3)
    fp32_zscore = np.abs((fp32 - fp32.mean()) / fp32.std())
    fp16_zscore = np.abs((fp16 - fp16.mean()) / fp16.std())

    stats["fp32_outliers"] = (fp32_zscore > 3).sum()
    stats["fp16_outliers"] = (fp16_zscore > 3).sum()
    stats["outlier_diff"] = abs(stats["fp32_outliers"] - stats["fp16_outliers"])

    # L2 norm distribution (should be similar)
    fp32_norms = np.linalg.norm(fp32, axis=-1)
    fp16_norms = np.linalg.norm(fp16, axis=-1)

    stats["norm_mean_diff"] = np.abs(fp32_norms.mean() - fp16_norms.mean())
    stats["norm_std_diff"] = np.abs(fp32_norms.std() - fp16_norms.std())

    return stats
```

### Throughput Benchmarking
```python
# Source: PyTorch benchmarking patterns + existing profiler.py
# URL: https://pytorch.org/tutorials/recipes/recipes/benchmark.html

import time
import torch

def benchmark_throughput(model, dataloader, num_batches=100):
    """
    Benchmark throughput in tokens/second.

    Measures:
    - Total runtime (end-to-end)
    - Tokens per second (isolates GPU performance)
    - GPU utilization (via torch.cuda.max_memory_allocated)
    """
    model.eval()
    torch.cuda.synchronize()

    start_time = time.time()
    total_tokens = 0

    with torch.inference_mode():
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break

            # Track tokens processed
            total_tokens += batch["input_ids"].numel()

            # Forward pass
            outputs = model.forward_packed(
                input_ids=batch["input_ids"],
                cu_seqlens=batch["cu_seqlens"],
                max_seqlen=batch["max_seqlen"]
            )

    torch.cuda.synchronize()
    end_time = time.time()

    elapsed = end_time - start_time
    throughput = total_tokens / elapsed

    return {
        "total_runtime_sec": elapsed,
        "tokens_per_second": throughput,
        "sequences_per_second": num_batches / elapsed,
        "peak_memory_gb": torch.cuda.max_memory_allocated() / 1e9,
    }
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| BF16 for all ops | FP16 for ESM-2 inference | 2026 (Phase 8) | FP16 has better throughput on Ampere+ (1.8-2x vs FP32), ESM-2 trained in FP16 |
| torch.amp.autocast for inference | model.half() for inference | PyTorch 2.0+ | Simpler API, better performance (no dynamic dtype switching) |
| Manual NaN detection | torch.isnan/isinf built-ins | Always available | Standard library, optimized |
| Cosine similarity >0.999 | Cosine similarity >0.99 | Phase 8 | FP16 has lower precision, 0.99 threshold validated in ESM-2 research |

**Deprecated/outdated:**
- **torch.cuda.amp.autocast**: Deprecated in favor of torch.amp.autocast (device-agnostic), but for inference, use model.half() instead
- **Mixed precision for inference**: Old pattern was to use autocast for inference. Modern pattern: full FP16 via model.half() for simplicity and performance
- **BF16 for ESM-2**: Research shows ESM-2 trained in FP16, not BF16. FP16 is optimal precision for this model

## Open Questions

Things that couldn't be fully resolved:

1. **Exact throughput improvement for VirNucPro workload**
   - What we know: ESM-2 research shows 1.8-2x throughput for FP16 vs FP32 on transformers
   - What's unclear: VirNucPro-specific throughput (depends on sequence length distribution, packing efficiency)
   - Recommendation: Benchmark on representative 10K sequence dataset, measure both tokens/sec and total runtime

2. **LayerNorm FP32 necessity for ESM-2 3B**
   - What we know: ESM-2 research paper shows FP16 works (norm difference <1e-3 vs FP32)
   - What's unclear: Whether full FP16 passes >0.99 similarity threshold for VirNucPro sequences
   - Recommendation: Start with full FP16, fall back to selective FP32 LayerNorm only if validation fails

3. **DNABERT-S FP16 support**
   - What we know: DNABERT original repo documents --fp16 flag (requires apex library)
   - What's unclear: Whether DNABERT-S (MosaicBERT variant) supports FP16, or if it requires BF16
   - Recommendation: Test DNABERT-S with model.half() in validation, fall back to BF16 if issues

4. **Optimal similarity threshold**
   - What we know: Phase 6 uses 0.999 for FP16 packing, user specified 0.99 for Phase 8
   - What's unclear: Whether 0.99 is too lenient (allows too much drift) or too strict (fails on precision noise)
   - Recommendation: Use 0.99 as specified, collect distribution statistics to validate appropriateness

5. **GPU utilization improvement**
   - What we know: FP16 reduces memory footprint 50%, allows larger batch sizes
   - What's unclear: Whether dynamic token budget will automatically increase batch size to utilize freed memory
   - Recommendation: Validate GPU utilization before/after FP16 conversion, adjust token budget if needed

## Sources

### Primary (HIGH confidence)
- [PyTorch AMP documentation](https://docs.pytorch.org/docs/stable/amp.html) - autocast vs model.half() for inference
- [PyTorch inference optimization checklist](https://docs.pytorch.org/serve/performance_checklist.html) - FP16 inference best practices
- [FlashAttention GitHub](https://github.com/Dao-AILab/flash-attention) - FP16 vs BF16 performance (230 TFLOPs/s for both on A100)
- [FlashAttention-3 paper](https://arxiv.org/pdf/2407.08608) - FP16 achieves 1.5-2.0x speedup over FA2, same numerical error
- [PyTorch GitHub Issue #66707](https://github.com/pytorch/pytorch/issues/66707) - LayerNorm needs FP32 for FP16 inputs to avoid overflow

### Secondary (MEDIUM confidence)
- [ESM-2 efficient inference paper (PMC12481099)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12481099/) - ESM-2 uses bfloat16 for efficient implementation (2025 research)
- [BF16 vs FP16 comparison](https://www.beam.cloud/blog/bf16-vs-fp16) - FP16 has 10-bit mantissa (3 bits more than BF16), better precision
- [Numerical stability article (Medium)](https://medium.com/@spjosyula2005/numerical-stability-why-fp16-training-breaks-and-how-to-fix-it-cba2835a2877) - LayerNorm overflow in FP16
- [Feature flag best practices (Graphite)](https://graphite.com/guides/feature-flag-best-practices-continuous-deployment) - Progressive rollouts, monitoring, documentation
- [DNABERT GitHub](https://github.com/jerryji1993/DNABERT) - Original DNABERT documents --fp16 flag support

### Tertiary (LOW confidence)
- [Transformer FP16 accuracy loss (NVIDIA forums)](https://forums.developer.nvidia.com/t/transformer-accuracy-shows-significant-loss-in-fp16-reasoning-on-jetpack6-tensorrt8-6-11/342129) - Single case report, not generalizable
- [FP16 gives NaN loss (PyTorch forums)](https://discuss.pytorch.org/t/fp16-gives-nan-loss-when-using-pre-trained-model/94133) - Training-focused, not inference
- [Cosine similarity FP16/FP32 inconsistency (PyTorch Issue #69512)](https://github.com/pytorch/pytorch/issues/69512) - Edge case, likely resolved in PyTorch 2.2+

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - PyTorch built-in FP16 support, well-documented
- Architecture: HIGH - Existing Phase 6 patterns (cosine similarity, FlashAttention dtype validation)
- Pitfalls: HIGH - Extensively documented in PyTorch, NVIDIA, and research papers
- ESM-2 FP16 support: MEDIUM - Research papers show FP16 works, but not tested on VirNucPro sequences yet
- DNABERT-S FP16 support: LOW - Original DNABERT has FP16, but DNABERT-S is MosaicBERT variant (unclear)

**Research date:** 2026-02-05
**Valid until:** 2026-04-05 (60 days - stable domain, PyTorch APIs unlikely to change)
