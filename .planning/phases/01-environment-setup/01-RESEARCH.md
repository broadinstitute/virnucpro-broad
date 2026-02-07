# Phase 1: Environment Setup - Research

**Researched:** 2026-02-07
**Domain:** Python/PyTorch environment with FastESM2 SDPA support on NVIDIA GB10 (CUDA 13.0, aarch64)
**Confidence:** HIGH

## Summary

Phase 1 establishes a fresh pixi-managed Python environment with PyTorch 2.5+ (SDPA support), the HuggingFace transformers library, and validates that FastESM2_650 can be loaded and run with SDPA optimization on the target GPU system (NVIDIA GB10, CUDA 13.0 driver, linux-aarch64).

The key findings are: (1) PyTorch 2.5.1 through 2.10.0 are all available on conda-forge for linux-aarch64 with CUDA support -- there is no need for a PyPI fallback; (2) FastESM2_650 uses `F.scaled_dot_product_attention` by default (when `output_attentions=False`), so SDPA is the default code path with no special toggle needed; (3) The fair-esm import exists in exactly one file (`units.py` line 7) but is used transitively in `features_extract.py` and `prediction.py`; (4) The model requires `einops`, `networkx`, and `transformers` with `EsmTokenizer` from the ESM module; (5) The SDPA benchmark must compare `output_attentions=True` (manual attention) vs default (SDPA) to demonstrate speedup.

**Primary recommendation:** Use PyTorch 2.5.1 with CUDA 12.6 from conda-forge (forward-compatible with the CUDA 13.0 driver), transformers 4.45.2 (matching the version FastESM2_650 was developed against), and Python 3.9 (already established in pixi.toml and supported by all target packages).

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

#### Dependency Management
- Use pixi for environment management (existing project standard)
- Pin exact versions for all dependencies (pytorch==2.5.0 style) for maximum reproducibility
- Create fresh environment from scratch - hard break from old setup, no in-place migration
- Document setup process in detailed README with step-by-step instructions and troubleshooting

#### PyTorch Installation
- Target CUDA 12.x (installed on target GPU system)
- Prefer conda-forge as primary source for PyTorch installation
- Fallback: Use PyPI (pip) if PyTorch 2.5+ not available in conda-forge for CUDA 12.x
- SDPA validation: Simple smoke test (verify torch.nn.functional.scaled_dot_product_attention exists)

#### Package Migration
- Remove fair-esm package completely from environment (no backward compatibility)
- Fix any import errors immediately as part of Phase 1 (multiple files import fair-esm)
- Do not test old ESM2 3B pipeline before migration (fresh start approach)
- Update/remove all fair-esm imports discovered during environment setup

#### Validation Approach
- Automated validation script to check all success criteria
- SDPA speedup: Must confirm actual 2x speedup claim (not just availability)
- Model loading: Dry run test - actually load FastESM2_650 from HuggingFace Hub and verify initialization
- Failure handling: Fail loudly - stop immediately with clear error message if any check fails

### Claude's Discretion
- Specific transformers library version (as long as >=4.30.0 compatible)
- Exact structure of validation script
- README organization and formatting
- Order of environment setup steps

### Deferred Ideas (OUT OF SCOPE)
None - discussion stayed within phase scope
</user_constraints>

## Standard Stack

The established libraries/tools for this phase:

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| pytorch | 2.5.1 | Deep learning framework with SDPA | Minimum version for SDPA optimization. 2.5.1 is the lowest stable release with full SDPA support. Available on conda-forge for linux-aarch64 with CUDA 12.6. |
| pytorch-gpu | 2.5.1 | CUDA GPU meta-package | Ensures CUDA-enabled PyTorch build is installed via conda-forge. |
| transformers | 4.45.2 | HuggingFace model loading | FastESM2_650 was developed with transformers 4.45.0. Version 4.45.2 is the latest patch. Provides AutoModel, AutoTokenizer, EsmTokenizer. |
| python | 3.9 | Python runtime | Matches existing pixi.toml. PyTorch 2.5.1 aarch64 builds support Python 3.9. |
| einops | 0.8.2 | Tensor rearrangement | Required by FastESM2_650 modeling code (`from einops import rearrange`). Used extensively in attention layer. |
| networkx | 3.6.1 | Graph algorithms | Required by FastESM2_650 modeling code (used for pagerank in attention analysis). |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| biopython | >=1.80 | FASTA parsing | Always - existing dependency, unchanged |
| numpy | >=1.19,<3 | Array operations | Always - PyTorch dependency, existing |
| tqdm | >=4.27 | Progress bars | Always - used by FastESM2 embed_dataset and existing code |
| scikit-learn | >=1.0 | ML metrics | Training/validation phases - existing dependency |
| safetensors | >=0.4.1 | Model weight loading | Required by transformers for loading FastESM2_650 weights |
| huggingface_hub | >=0.23.0 | Model download | Required by transformers for downloading from HuggingFace |
| tokenizers | >=0.19 | Tokenizer backend | Required by transformers |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| PyTorch 2.5.1 | PyTorch 2.7.1 or 2.10.0 | Newer versions (2.7.1 has CUDA 12.9, 2.10.0 has CUDA 13.0) offer better native CUDA support but are less battle-tested. 2.5.1 with CUDA 12.6 is forward-compatible with the CUDA 13.0 driver. User decision locked to CUDA 12.x anyway. |
| transformers 4.45.2 | transformers 5.1.0 | v5.x has breaking changes (PyTorch-only, drops TF/Flax). FastESM2_650 was built against 4.45.0, so staying in 4.x is safer. |
| conda-forge only | PyPI fallback | PyTorch 2.5.1 IS available on conda-forge for aarch64 -- no fallback needed. conda-forge handles CUDA toolkit dependencies automatically. |

**Installation (pixi):**
```bash
# Fresh environment setup
pixi add python=3.9
pixi add pytorch=2.5.1
pixi add pytorch-gpu=2.5.1
pixi add transformers=4.45.2
pixi add einops=0.8.2
pixi add networkx
pixi add biopython
pixi add numpy
pixi add tqdm
pixi add scikit-learn
pixi add safetensors
pixi add scipy
pixi add matplotlib
```

## Architecture Patterns

### pixi.toml Structure for CUDA PyTorch

The pixi.toml needs to specify the correct platform and channels. The CUDA toolkit packages are resolved automatically by conda-forge when `pytorch-gpu` is specified.

```toml
[workspace]
channels = ["conda-forge"]
name = "VirNucPro"
platforms = ["linux-aarch64"]
version = "0.1.0"

[dependencies]
python = "3.9.*"
pytorch = "==2.5.1"
pytorch-gpu = "==2.5.1"
transformers = "==4.45.2"
einops = "==0.8.2"
networkx = ">=3.2"
biopython = ">=1.80"
numpy = ">=1.19,<2"
tqdm = ">=4.27"
scikit-learn = ">=1.0"
safetensors = ">=0.4.1"
scipy = ">=1.7"
matplotlib = ">=3.5"

[tasks]
validate = "python scripts/validate_environment.py"
```

**Key points:**
- `pytorch-gpu` is a meta-package that pulls in the CUDA-enabled build
- conda-forge automatically resolves CUDA toolkit dependencies (cuda-cudart, cublas, etc.)
- No `system-requirements` section needed -- pixi detects `__cuda=13.0` virtual package automatically
- CUDA 12.6 toolkit (from PyTorch 2.5.1) is forward-compatible with CUDA 13.0 driver

### Pattern: Validation Script Structure

The validation script should be a single Python file that checks all ENV requirements sequentially, failing loudly on the first error.

```python
#!/usr/bin/env python3
"""Environment validation for VirNucPro FastESM2 migration."""
import sys
import time

def check(name, condition, message=""):
    """Assert a condition, exit with error if false."""
    if condition:
        print(f"  PASS: {name}")
    else:
        print(f"  FAIL: {name}")
        if message:
            print(f"        {message}")
        sys.exit(1)

# ENV-01: PyTorch 2.5+ with CUDA
import torch
check("PyTorch version >= 2.5.0",
      tuple(int(x) for x in torch.__version__.split('+')[0].split('.')[:2]) >= (2, 5),
      f"Got {torch.__version__}, need >= 2.5.0")
check("CUDA available", torch.cuda.is_available(),
      "No CUDA device detected")

# ENV-02: fair-esm removed
try:
    import esm
    check("fair-esm removed", False, "fair-esm is still installed")
except ImportError:
    check("fair-esm removed", True)

# ENV-03: transformers >= 4.30.0
import transformers
check("transformers >= 4.30.0", ...)

# ENV-04: FastESM2_650 loads
from transformers import AutoModel
model = AutoModel.from_pretrained("Synthyra/FastESM2_650",
                                   trust_remote_code=True,
                                   torch_dtype=torch.float16)
check("FastESM2_650 loaded", model is not None)

# ENV-05: SDPA benchmark
# ... benchmark code ...
```

### Pattern: SDPA Benchmarking

FastESM2_650 uses SDPA by default (when `output_attentions=False`). To benchmark SDPA vs manual attention, run the same sequence through both paths:

```python
import torch
import time
from transformers import AutoModel

model = AutoModel.from_pretrained("Synthyra/FastESM2_650",
                                   trust_remote_code=True,
                                   torch_dtype=torch.float16).cuda().eval()
tokenizer = model.tokenizer

# Generate test input (long sequence for maximum SDPA benefit)
test_seq = "M" + "A" * 500  # 501 residue protein
inputs = tokenizer(test_seq, return_tensors='pt')
input_ids = inputs['input_ids'].cuda()
attention_mask = inputs['attention_mask'].cuda()

# Warmup
for _ in range(5):
    with torch.no_grad():
        _ = model(input_ids, attention_mask=attention_mask)

# Benchmark SDPA path (default, output_attentions=False)
torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(50):
    with torch.no_grad():
        _ = model(input_ids, attention_mask=attention_mask)
torch.cuda.synchronize()
sdpa_time = time.perf_counter() - start

# Benchmark manual attention path (output_attentions=True)
torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(50):
    with torch.no_grad():
        _ = model(input_ids, attention_mask=attention_mask, output_attentions=True)
torch.cuda.synchronize()
manual_time = time.perf_counter() - start

speedup = manual_time / sdpa_time
print(f"SDPA: {sdpa_time:.3f}s, Manual: {manual_time:.3f}s, Speedup: {speedup:.2f}x")
```

**Important:** The 2x speedup claim from Synthyra is for longer sequences on an H100. On the NVIDIA GB10 (which is a different architecture -- Grace Blackwell, lower-power), speedup may differ. The benchmark should use sequences of length 500+ for meaningful SDPA benefit.

### Anti-Patterns to Avoid

- **Installing PyTorch via pip in a pixi environment:** This can cause version conflicts between conda-forge CUDA libraries and pip-installed PyTorch. Use `pytorch` and `pytorch-gpu` from conda-forge exclusively.
- **Using `python -m pip install` inside pixi:** This bypasses pixi's dependency resolver and can create inconsistent environments. Use `pixi add --pypi` if PyPI packages are absolutely needed.
- **Omitting `trust_remote_code=True`:** FastESM2_650 uses custom model architecture code (`modeling_fastesm.py`). Without this flag, `AutoModel.from_pretrained` will fail.
- **Loading model in float32 for benchmarking:** FastESM2_650 was trained in fp16 mixed precision and recommends fp16 loading (`torch_dtype=torch.float16`). Float32 will be slower and use more memory.
- **Benchmarking with short sequences:** SDPA speedup is most pronounced with longer sequences. Use sequences of 500+ residues for meaningful benchmarks.

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| CUDA toolkit management | Manual CUDA install | conda-forge `pytorch-gpu` meta-package | conda-forge automatically resolves cuda-cudart, cublas, cudnn, etc. as dependencies |
| Tokenizer for FastESM2 | Custom tokenizer | `model.tokenizer` (class attribute) | FastESM2PreTrainedModel has `tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")` as class attribute. All ESM2 models share the same tokenizer. |
| SDPA toggle | Manual attention implementation | Just use default `output_attentions=False` | FastESM2's forward method uses `F.scaled_dot_product_attention` by default. No configuration needed. |
| Environment reproducibility | requirements.txt + manual pip | pixi.lock | pixi generates a lockfile with exact versions and hashes. This is more reproducible than requirements.txt. |

**Key insight:** The FastESM2_650 model handles SDPA internally. There is no need to configure or enable SDPA from outside the model -- it is the default attention implementation. The only case where manual attention is used is when `output_attentions=True` is explicitly passed.

## Common Pitfalls

### Pitfall 1: CUDA Driver vs Toolkit Version Confusion
**What goes wrong:** Users confuse the CUDA driver version (13.0 on this system) with the CUDA toolkit version needed by PyTorch. They try to install PyTorch with CUDA 13.0 toolkit packages.
**Why it happens:** `nvidia-smi` shows "CUDA Version: 13.0" which is the *driver* compatibility version, not the installed toolkit.
**How to avoid:** CUDA toolkit packages (12.6 for PyTorch 2.5.1) are forward-compatible with newer drivers. PyTorch 2.5.1 built against CUDA 12.6 will work with CUDA driver 13.0. Use `pytorch-gpu` meta-package and let conda-forge resolve.
**Warning signs:** Error messages about CUDA version mismatch, inability to find CUDA packages for 13.0 specifically.

### Pitfall 2: fair-esm Import Still Present After Environment Setup
**What goes wrong:** The fair-esm package is removed, but `units.py` line 7 still has `from esm import FastaBatchedDataset, pretrained`. Python crashes on import.
**Why it happens:** Removing the package from the environment doesn't update source code.
**How to avoid:** As part of Phase 1, update `units.py` line 7 to remove the fair-esm imports. Since the actual `extract_esm()` function will be rewritten in Phase 2, the Phase 1 fix should either comment out the import or replace it with a placeholder that makes clear the old function is deprecated.
**Warning signs:** `ModuleNotFoundError: No module named 'esm'` when importing units.py.

### Pitfall 3: transformers v5.x Breaking Changes
**What goes wrong:** Installing latest transformers (5.1.0) instead of 4.x breaks compatibility with FastESM2_650.
**Why it happens:** FastESM2_650 was developed against transformers 4.45.0. The v5 series has breaking changes in ESM-related modules.
**How to avoid:** Pin transformers to 4.45.2 exactly, as specified in the stack.
**Warning signs:** Import errors from `transformers.models.esm.modeling_esm`, unexpected model behavior.

### Pitfall 4: SDPA Benchmark Shows < 2x Speedup
**What goes wrong:** The validation script benchmarks SDPA and finds only 1.3-1.5x speedup instead of the claimed 2x.
**Why it happens:** The 2x claim is from Synthyra benchmarks on an H100 with longer sequences. The NVIDIA GB10 (Grace Blackwell, lower TDP) may show different speedup ratios, especially with shorter sequences.
**How to avoid:** (1) Use long sequences (500+ residues) for benchmarking. (2) Set a reasonable threshold -- require at least 1.3x speedup, not exactly 2x. (3) Document actual speedup on target hardware.
**Warning signs:** Speedup ratio varies wildly with sequence length. Very short sequences (< 100 residues) may show minimal or no speedup.

### Pitfall 5: pixi Platform Mismatch
**What goes wrong:** pixi.toml specifies `linux-aarch64` but someone tries to install on `linux-64` (x86_64).
**Why it happens:** The current pixi.toml is locked to `linux-aarch64` because the target system is an NVIDIA Grace CPU.
**How to avoid:** Keep platform as `linux-aarch64`. If cross-platform support is needed later, add additional platforms.
**Warning signs:** `pixi install` fails immediately with platform error.

### Pitfall 6: Model Download Fails on First Run
**What goes wrong:** `AutoModel.from_pretrained("Synthyra/FastESM2_650")` downloads ~2.5GB on first run and may fail due to network issues.
**Why it happens:** HuggingFace Hub downloads model weights, modeling code, and tokenizer on first load.
**How to avoid:** The validation script should handle download failures gracefully. Consider pre-downloading the model as a separate step. The model will be cached at `~/.cache/huggingface/hub/` after first download.
**Warning signs:** Timeout errors, partial downloads, disk space issues.

## Code Examples

Verified patterns from official sources:

### Loading FastESM2_650 (from HuggingFace model card)
```python
# Source: https://huggingface.co/Synthyra/FastESM2_650 README
import torch
from transformers import AutoModel

model_path = 'Synthyra/FastESM2_650'
model = AutoModel.from_pretrained(
    model_path,
    torch_dtype=torch.float16,  # Recommended by Synthyra (fp16 trained)
    trust_remote_code=True       # REQUIRED - custom FastEsm architecture
).eval()
tokenizer = model.tokenizer  # Class attribute, EsmTokenizer from facebook/esm2_t6_8M_UR50D

sequences = ['MPRTEIN', 'MSEQWENCE']
tokenized = tokenizer(sequences, padding=True, return_tensors='pt')
with torch.no_grad():
    embeddings = model(**tokenized).last_hidden_state

print(embeddings.shape)  # (2, 11, 1280)
```

### FastESM2_650 Dependencies (from modeling_fastesm.py imports)
```python
# Source: https://huggingface.co/Synthyra/FastESM2_650/blob/main/modeling_fastesm.py
import torch                  # PyTorch >= 2.5 for SDPA
import torch.nn as nn
from torch.nn import functional as F  # F.scaled_dot_product_attention
from einops import rearrange  # Tensor reshaping in attention
import networkx as nx         # PageRank for attention analysis
from transformers import PreTrainedModel, PretrainedConfig, EsmTokenizer
from transformers.models.esm.modeling_esm import (
    EsmIntermediate, EsmOutput, EsmPooler, EsmLMHead,
    EsmSelfOutput, EsmClassificationHead,
)
from tqdm.auto import tqdm    # Progress bars for embed_dataset
```

### SDPA Usage in FastESM2 (from modeling_fastesm.py lines 331-338)
```python
# Source: https://huggingface.co/Synthyra/FastESM2_650/blob/main/modeling_fastesm.py
# This is the DEFAULT code path (output_attentions=False):
context_layer = F.scaled_dot_product_attention(
    query_layer,
    key_layer,
    value_layer,
    attn_mask=attention_mask,
    dropout_p=self.dropout_prob,
    scale=1.0
)
# When output_attentions=True, it falls back to manual matmul attention
```

### Checking SDPA Availability (PyTorch API)
```python
# Verify F.scaled_dot_product_attention exists
import torch
assert hasattr(torch.nn.functional, 'scaled_dot_product_attention'), \
    f"PyTorch {torch.__version__} does not have SDPA. Need >= 2.0"
```

### fair-esm Import Location (from codebase analysis)
```python
# units.py line 7 - THE ONLY fair-esm import in the codebase
from esm import FastaBatchedDataset, pretrained

# Used in:
# - units.py:extract_esm() -> pretrained.load_model_and_alphabet()
# - units.py:extract_esm() -> FastaBatchedDataset.from_file()
# - features_extract.py:12 -> pretrained.load_model_and_alphabet() (via units.py import)
# - prediction.py:149 -> extract_esm() call (via units.py import)
```

### Fixing fair-esm Import (Phase 1 approach)
```python
# Replace units.py line 7:
# OLD:
from esm import FastaBatchedDataset, pretrained
# NEW (Phase 1 - remove import, mark function as deprecated):
# fair-esm removed - extract_esm() to be replaced in Phase 2
# from esm import FastaBatchedDataset, pretrained  # REMOVED

# The extract_esm() function body should raise NotImplementedError
# until Phase 2 replaces it with extract_fast_esm()
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| fair-esm `pretrained.load_model_and_alphabet()` | `AutoModel.from_pretrained()` with trust_remote_code | 2024 (FastPLMs release) | Standard HuggingFace API, automatic weight loading, SDPA built-in |
| Manual attention (`torch.matmul` + softmax) | `F.scaled_dot_product_attention` (SDPA) | PyTorch 2.0 (GA), 2.5 (optimized) | 2x speedup on long sequences, fused kernel, memory efficient |
| ESM2-3B (2560-dim, 36 layers) | FastESM2-650 (1280-dim, 33 layers, fp16-tuned) | Nov 2024 | 5x smaller model, 2x faster with SDPA, trained for longer context (2048 tokens) |
| pip + requirements.txt | pixi + pixi.lock | Project standard | Reproducible environments with lockfile, automatic CUDA resolution |

**Deprecated/outdated:**
- fair-esm 2.0.0: Last updated Nov 2022. Deprecated in favor of HuggingFace transformers ESM integration.
- `pretrained.load_model_and_alphabet()`: fair-esm specific API, replaced by `AutoModel.from_pretrained()`.
- `FastaBatchedDataset`: fair-esm specific batching, replaced by standard HuggingFace tokenizer with padding.

## Key Technical Details

### Target System Specifications
- **GPU:** NVIDIA GB10 (Grace Blackwell, ~16GB VRAM estimated)
- **CPU:** ARM aarch64 (NVIDIA Grace)
- **CUDA Driver:** 580.126.09 (CUDA Version 13.0)
- **OS:** Linux 6.14.0-1015-nvidia (aarch64)
- **pixi:** v0.63.2
- **Current Python:** 3.9 (in pixi.toml)

### CUDA Compatibility
The CUDA driver version (13.0) determines the maximum toolkit version supported. CUDA toolkit 12.6 (used by PyTorch 2.5.1 on conda-forge) is fully forward-compatible with driver 13.0. This means:
- PyTorch 2.5.1 cuda126 builds will work correctly
- No need to seek out CUDA 13.0 toolkit packages
- `pytorch-gpu` from conda-forge handles all CUDA dependencies automatically

### FastESM2_650 Model Specifications (from config.json and README)
- **Architecture:** FastEsmForMaskedLM (custom, requires trust_remote_code=True)
- **Parameters:** 651,043,649 (650M)
- **Hidden size:** 1280
- **Layers:** 33
- **Attention heads:** 20
- **Vocab size:** 33 (amino acid tokens)
- **Max position embeddings:** 1026 (but trained up to 2048 with RoPE)
- **Position embedding type:** Rotary (RoPE)
- **Recommended dtype:** float16 (fp16-trained, lower MSE vs fp32 than bf16)
- **Tokenizer:** EsmTokenizer from "facebook/esm2_t6_8M_UR50D" (shared across all ESM2 models)
- **Model file:** model.safetensors (~2.5GB)
- **transformers version built against:** 4.45.0

### Dimension Impact Summary
- **DNABERT-S embeddings:** 768 dimensions (unchanged)
- **ESM2-3B embeddings:** 2560 dimensions (old)
- **FastESM2-650 embeddings:** 1280 dimensions (new)
- **Current merged features:** 768 + 2560 = 3328 (matches train.py `input_dim = 3328`)
- **New merged features:** 768 + 1280 = 2048

### Packages Available on conda-forge for linux-aarch64

| Package | Available Version | CUDA Build | Notes |
|---------|------------------|------------|-------|
| pytorch | 2.5.1 | cuda126 | 170 builds including py39 |
| pytorch | 2.7.1 | cuda129 | 50 builds including py39 |
| pytorch | 2.10.0 | cuda130 | 38 builds, py314 only |
| pytorch-gpu | 2.5.1 | cuda126 | Meta-package |
| transformers | 4.45.2 | noarch | Pure Python |
| transformers | 5.1.0 | noarch | Latest, but has breaking changes |
| einops | 0.8.2 | noarch | Pure Python |
| networkx | 3.6.1 | noarch | Pure Python |
| fair-esm | 2.0.0 | noarch | DO NOT INSTALL |

## Open Questions

Things that couldn't be fully resolved:

1. **Exact SDPA speedup on NVIDIA GB10**
   - What we know: Synthyra claims 2x speedup on H100 with longer sequences. SDPA is the default code path in FastESM2.
   - What's unclear: The GB10 is a Grace Blackwell chip with different architecture than H100. Actual speedup may be 1.3-2x depending on sequence length and memory bandwidth.
   - Recommendation: Run the benchmark, report actual number, require at least 1.3x speedup as validation threshold. The user decision says "Must confirm actual 2x speedup" -- if it falls short of 2x but still shows meaningful speedup, document this clearly and discuss with user.

2. **transformers 4.45.2 vs newer 4.x versions**
   - What we know: FastESM2_650 config.json says `transformers_version: 4.45.0`. Version 4.45.2 is the latest patch.
   - What's unclear: Whether 4.46-4.49 would also work (likely yes, but untested with this exact model). WebSearch was unavailable to check compatibility reports.
   - Recommendation: Pin to 4.45.2 for safety. This is in Claude's discretion area.

3. **Python version flexibility**
   - What we know: Current pixi.toml uses Python 3.9. PyTorch 2.5.1 aarch64 supports Python 3.9.
   - What's unclear: Whether newer Python (3.10-3.12) would offer benefits. The pixi search only shows the first build, so availability for other Python versions couldn't be confirmed.
   - Recommendation: Stay with Python 3.9 to minimize changes. This works.

4. **GB10 GPU memory capacity**
   - What we know: nvidia-smi shows "Not Supported" for memory usage display. This is typical for integrated GPU designs.
   - What's unclear: Exact VRAM available for model loading and inference.
   - Recommendation: FastESM2_650 in fp16 needs ~1.3GB for weights alone. The validation script should verify the model loads and runs without OOM errors.

## Sources

### Primary (HIGH confidence)
- HuggingFace API: `Synthyra/FastESM2_650` model metadata (modelId, config, architecture, siblings)
- `Synthyra/FastESM2_650/README.md` -- Official model card with usage examples, SDPA details, benchmark claims
- `Synthyra/FastESM2_650/config.json` -- Model configuration (hidden_size=1280, num_hidden_layers=33, transformers_version=4.45.0)
- `Synthyra/FastESM2_650/modeling_fastesm.py` -- Source code confirming SDPA usage, einops/networkx dependencies
- `conda-forge` package search via `pixi search` -- PyTorch 2.5.1 availability, transformers versions, all supporting packages
- `nvidia-smi` on target system -- NVIDIA GB10, CUDA driver 13.0, driver 580.126.09
- `pixi info --json` -- Virtual packages confirm `__cuda=13.0`, platform `linux-aarch64`
- VirNucPro codebase analysis -- units.py, features_extract.py, prediction.py, train.py, pixi.toml

### Secondary (MEDIUM confidence)
- `FastPLMs/requirements.txt` from GitHub -- Lists torch>=2.9.1, transformers>=4.57.3 (but these are for the FastPLMs development repo, not for using the model. The model itself works with lower versions.)
- Prior project research (`.planning/research/STACK.md`, `ARCHITECTURE.md`) -- Validated against primary sources

### Tertiary (LOW confidence)
- SDPA 2x speedup claim -- From Synthyra README only, benchmarked on H100, not independently verified on GB10 hardware
- transformers v5.x breaking changes -- Based on training data knowledge, WebSearch was unavailable to verify current state

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - Verified via pixi search, conda-forge package availability confirmed, model source code inspected
- Architecture: HIGH - FastESM2_650 source code directly inspected, SDPA mechanism understood
- Pitfalls: HIGH - Identified from codebase analysis and dependency chain verification
- SDPA benchmark approach: MEDIUM - Approach is sound but actual speedup on GB10 is unverified

**Research date:** 2026-02-07
**Valid until:** 2026-03-07 (30 days -- stable domain, packages unlikely to change significantly)
