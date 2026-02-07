# Stack Research: FastESM2_650 Migration

**Domain:** Protein Language Model Integration for Viral Classification
**Researched:** 2026-02-07
**Confidence:** MEDIUM

## Recommended Stack

### Core Technologies for FastESM2_650 Integration

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| PyTorch | ≥2.5.0 | Deep learning framework | Required for SDPA (Scaled Dot-Product Attention) optimization in FastESM2. PyTorch 2.5+ provides up to 2x speedup on longer sequences through native SDPA support. |
| transformers | ≥4.30.0, <5.0.0 | HuggingFace model loading | Provides AutoModel API for loading FastESM2_650. Stay on 4.x series to maintain TensorFlow/Flax compatibility and avoid v5 breaking changes. Current system uses 4.30.0 which is compatible. |
| fair-esm | Remove | Legacy ESM2 loader | **DO NOT USE with FastESM2**. The fair-esm 2.0.0 package is deprecated (last updated Nov 2022) and incompatible with HuggingFace-based FastESM2. Must migrate away from pretrained.load_model_and_alphabet(). |

### Supporting Libraries

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| torch | ≥2.5.0 | GPU acceleration | Always - core dependency for both model types |
| biopython | Current (existing) | FASTA parsing, sequence handling | Always - unchanged from current implementation |
| numpy | Current (existing) | Array operations | Always - unchanged from current implementation |
| tqdm | Current (existing) | Progress bars | Always - unchanged from current implementation |

### Development Tools

| Tool | Purpose | Notes |
|------|---------|-------|
| HuggingFace Hub | Model repository access | FastESM2_650 downloaded directly from Synthyra/FastESM2_650 |
| CUDA toolkit | GPU support | Ensure CUDA version compatible with PyTorch 2.5+ |

## Installation

```bash
# Remove deprecated fair-esm
pip uninstall fair-esm

# Upgrade PyTorch to 2.5+ (critical for SDPA performance)
pip install torch>=2.5.0

# Keep transformers in 4.x series (avoid v5 for now)
pip install "transformers>=4.30.0,<5.0.0"

# Existing dependencies (no change)
# biopython, numpy, tqdm, etc. remain as-is
```

## Critical API Migration Changes

### Model Loading

**OLD (fair-esm 2.0.0):**
```python
from esm import pretrained

model, alphabet = pretrained.load_model_and_alphabet('esm2_t36_3B_UR50D')
```

**NEW (FastESM2_650):**
```python
from transformers import AutoModel, AutoTokenizer
import torch

model = AutoModel.from_pretrained(
    'Synthyra/FastESM2_650',
    trust_remote_code=True,  # REQUIRED - uses custom SDPA implementation
    torch_dtype=torch.float16  # Recommended for speed/memory
)
tokenizer = model.tokenizer  # Tokenizer bundled with model
```

### Embedding Extraction

**OLD (fair-esm with esm2_t36_3B_UR50D):**
```python
# Layer 36, dimension 2560
out = model(toks, repr_layers=[36], return_contacts=False)
representations = out["representations"][36]
# Shape: [batch, seq_len, 2560]
```

**NEW (FastESM2_650):**
```python
# Layer 33, dimension 1280
outputs = model(input_ids)
hidden_states = outputs.last_hidden_state
# Shape: [batch, seq_len, 1280]
# Or access specific layer via outputs.hidden_states if output_hidden_states=True
```

### Key Differences

| Aspect | ESM2-3B (fair-esm) | FastESM2-650 (HuggingFace) |
|--------|-------------------|---------------------------|
| **Embedding dimension** | 2560 | 1280 |
| **Number of layers** | 36 | 33 |
| **Attention heads** | 40 | 20 |
| **Model size** | 3B parameters | 650M parameters |
| **Loading API** | `pretrained.load_model_and_alphabet()` | `AutoModel.from_pretrained()` |
| **Tokenization** | `alphabet.get_batch_converter()` | `tokenizer()` from transformers |
| **Output format** | `out["representations"][layer]` | `outputs.last_hidden_state` |
| **trust_remote_code** | Not required | **REQUIRED** |

## CRITICAL: Embedding Dimension Change

**This is a breaking change that requires downstream model retraining.**

The current VirNucPro system extracts:
- DNABERT-S embeddings: 768 dimensions (unchanged)
- ESM2-3B embeddings: 2560 dimensions

After migration:
- DNABERT-S embeddings: 768 dimensions (unchanged)
- FastESM2-650 embeddings: 1280 dimensions

**Merged feature vector changes:**
- OLD: 768 + 2560 = 3328 dimensions
- NEW: 768 + 1280 = 2048 dimensions

**Impact:** The MLP classifier trained on 3328-dimensional input MUST be retrained on 2048-dimensional input. Cannot load existing model.pth checkpoints.

## Version Compatibility Matrix

| Package | Current | Required | Notes |
|---------|---------|----------|-------|
| PyTorch | Unknown | ≥2.5.0 | CRITICAL - older versions don't have SDPA optimization |
| transformers | 4.30.0 | ≥4.30.0, <5.0.0 | Current version OK; avoid v5 (PyTorch-only, breaking changes) |
| fair-esm | 2.0.0 | REMOVE | Incompatible with HuggingFace API |
| Python | 3.9 | 3.9+ | Current version OK |
| CUDA | Unknown | Compatible with PyTorch 2.5 | Verify CUDA version |

## Alternatives Considered

| Recommended | Alternative | When to Use Alternative |
|-------------|-------------|-------------------------|
| FastESM2-650 | ESM2-3B (fair-esm) | If you MUST maintain exact embedding dimensions (2560) and can't retrain classifier. However, fair-esm is deprecated. |
| FastESM2-650 | ESM2-650M (HuggingFace) | If you want standard ESM2-650M without additional training. FastESM2-650 is preferred due to longer context training (up to 2048 tokens) and fp16 optimization. |
| FastESM2-650 | ESM2-3B (HuggingFace) | If model size isn't a constraint and you want maximum embedding capacity. But FastESM2-650 offers 2x speed with strong performance. |
| transformers 4.x | transformers 5.x | Never for this project. v5 drops TensorFlow/Flax support and has breaking changes with position embeddings in ESM models. |

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| fair-esm 2.0.0 | Deprecated (last update Nov 2022), incompatible with HuggingFace API, no SDPA optimization | transformers with AutoModel |
| PyTorch <2.5 | Missing SDPA support = no 2x speedup for FastESM2 | PyTorch ≥2.5.0 |
| transformers 5.x | Breaking changes in ESM position embeddings, PyTorch-only (drops multi-framework support) | transformers 4.30.0-4.47.x |
| `trust_remote_code=False` | FastESM2 REQUIRES trust_remote_code=True due to custom SDPA implementation | Always use trust_remote_code=True for FastESM2 |
| Mixed ESM2-3B + FastESM2 | Incompatible embedding dimensions (2560 vs 1280) break downstream classifier | Choose one model and retrain classifier |

## Performance Expectations

Based on Synthyra benchmarks:

- **Speed:** FastESM2-650 is 2x faster than ESM2-650 on longer sequences (with PyTorch 2.5+)
- **Memory:** fp16 training reduces memory footprint vs fp32
- **Context length:** Trained up to 2048 tokens (vs 1024 for original ESM2-650)
- **Accuracy:** No performance degradation vs ESM2-650 (additional 50K training steps on OMGprot50)

## Migration Checklist

- [ ] Verify PyTorch version ≥2.5.0 (critical for performance)
- [ ] Uninstall fair-esm 2.0.0
- [ ] Update units.py::extract_esm() to use AutoModel API
- [ ] Change tokenization from alphabet.get_batch_converter() to tokenizer()
- [ ] Update layer extraction from repr_layers=[36] to outputs.last_hidden_state
- [ ] Handle embedding dimension change: 2560 → 1280
- [ ] Update merge_data() to expect 2048-dimensional merged features (768+1280)
- [ ] **RETRAIN classifier** - existing model.pth incompatible due to dimension change
- [ ] Test with small dataset to verify embeddings match expected shape
- [ ] Update requirements.txt to remove fair-esm, specify PyTorch ≥2.5

## Sources

**HIGH Confidence:**
- [Synthyra/FastESM2_650 - HuggingFace](https://huggingface.co/Synthyra/FastESM2_650) - Official model card
- [ESM-2 650M Architecture - BioLM](https://biolm.ai/models/esm2-650m/) - Embedding dimensions (1280)
- [ESM-2 3B Architecture - BioNeMo](https://docs.nvidia.com/bionemo-framework/2.1/models/esm2/) - Embedding dimensions (2560)

**MEDIUM Confidence:**
- [GitHub - Synthyra/FastPLMs](https://github.com/Synthyra/FastPLMs) - FastESM implementation details
- [HuggingFace ESM Documentation](https://huggingface.co/docs/transformers/en/model_doc/esm) - Transformers API
- [transformers PyPI](https://pypi.org/project/transformers/) - Version compatibility

**LOW Confidence (needs validation):**
- PyTorch 2.5 SDPA performance claims (Synthyra documentation, not independently verified)
- Exact speedup metrics (depends on hardware, sequence length, batch size)

---
*Stack research for: FastESM2_650 Migration*
*Researched: 2026-02-07*
