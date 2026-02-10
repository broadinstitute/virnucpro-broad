# Architecture Research: v2.5 Model Optimizations Round 2

**Domain:** GPU-accelerated viral nucleotide prediction pipeline
**Researched:** 2026-02-09
**Confidence:** HIGH (production codebase analysis)

## Executive Summary

The v2.5 Model Optimizations build on an existing v2.0 async architecture that has been battle-tested for ESM-2 only. Integration points are well-defined: DNABERT-S needs AsyncInferenceRunner port (v1.0→v2.0), model flexibility requires repr_layers parameterization, torch.compile fits in load_*_model functions, and vectorized ops replace Python loops in hot paths. Code quality improvements span env var centralization and function extraction in oversized files (async_inference.py: 526-line run() method, gpu_worker.py: 414-line function).

**Architecture compatibility:** All v2.5 features integrate with existing v2.0 components without requiring new infrastructure. DNABERT-S reuses 90% of ESM-2's async stack, differing only in tokenization (transformers.AutoTokenizer vs ESM batch_converter).

**Risk assessment:** LOW for code quality/vectorization, MEDIUM for torch.compile (dynamic shape interaction with FlashAttention varlen unknown), HIGH for DNABERT-S v2.0 port (replaces working v1.0 architecture, requires separate validation phase).

## Current Architecture (v2.0)

### Single-Process-Per-GPU Foundation

ESM-2 uses async DataLoader architecture (DNABERT-S remains on v1.0 bin-packing):

```
Pipeline Entry (prediction.py)
  ↓
Multi-GPU Orchestrator (run_multi_gpu_inference)
  ↓
GPUProcessCoordinator (spawns N processes, CUDA_VISIBLE_DEVICES isolation)
  ├→ gpu_worker(rank=0) on GPU 0
  │   ↓ load_esm2_model (one copy per GPU)
  │   ↓ IndexBasedDataset (stride [rank::world_size])
  │   ↓ create_async_dataloader (4-8 CPU workers)
  │     ├→ Workers: FASTA read (CPU-only, no CUDA)
  │     └→ Main: VarlenCollator (tokenize + pack)
  │   ↓ AsyncInferenceRunner.run()
  │     ├→ GreedyPacker (FFD, 92-94% efficiency)
  │     ├→ StreamProcessor (H2D/compute/D2H overlap)
  │     ├→ forward_packed() with FlashAttention varlen
  │     └→ CheckpointTrigger + AsyncCheckpointWriter
  │   ↓ HDF5 shard output (shard_0.h5)
  ├→ gpu_worker(rank=1) → shard_1.h5
  └→ ...
  ↓
ShardAggregator (merge → final HDF5)
```

**Key data structures:**
- `input_ids`: 1D packed `[seq1_tok1, ..., seq2_tok1, ...]`
- `cu_seqlens`: `[0, len1, len1+len2, ...]` (cumulative boundaries)
- `position_ids`: Reset to 0 at each cu_seqlens boundary
- `InferenceResult`: (sequence_ids, embeddings, batch_idx)

**Performance (ESM-2 on RTX 4090):**
- Multi-GPU scaling: 1.87x with 2 GPUs (93.7% efficiency)
- Packing efficiency: ~358% token utilization vs padded
- Throughput: 321 seq/s, 16.5K tokens/s

### DNABERT-S v1.0 Current Architecture

File: `virnucpro/pipeline/parallel_dnabert.py`

```
parallel_feature_extraction()
  ↓ assign_files_by_sequences() (greedy bin-pack)
  ↓ multiprocessing.Pool (1 worker per GPU)
  ↓ process_dnabert_files_worker()
    ├→ load_dnabert_model() (per-worker load)
    ├→ SeqIO.parse() (FASTA parsing)
    ├→ Token batching (6-mer estimate)
    ├→ tokenizer(batch, padding=True)
    ├→ model(input_ids, attention_mask) → [0]
    ├→ Mean pooling with attention_mask
    └→ atomic_save() to .pt files
```

**Critical differences from ESM-2:**
- File-level parallelism (not sequence-level)
- Synchronous processing (no async DataLoader)
- Padded batches (no packing)
- BF16 autocast (not FP16)
- transformers tokenizer (not ESM batch_converter)

**Performance issues:**
- 0.96x scaling with 2 GPUs (4% slowdown)
- Bin-packing coordination overhead
- No checkpoint support
- 100% padding waste

## DNABERT-S v2.0 Integration

### Tokenization Integration Point

**ESM-2 tokenization (current):**
```python
# VarlenCollator.__call__
batch_converter = alphabet.get_batch_converter()
labels, strs, tokens = batch_converter(sequences)  # 2D padded
# Strip padding → 1D packed
```

**DNABERT-S tokenization (needed):**
```python
# VarlenCollator.__call__ with DNABERT-S
tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-S")
inputs = tokenizer(sequences, return_tensors='pt', padding=True)
input_ids = inputs["input_ids"]  # 2D padded
attention_mask = inputs["attention_mask"]
# Strip padding → 1D packed (same as ESM-2)
```

**Key difference:** DNABERT-S tokenizer returns dict `{input_ids, attention_mask}`, not tuple. VarlenCollator needs conditional:

```python
if hasattr(self.batch_converter, 'tokenize'):  # transformers
    inputs = self.batch_converter(sequences, return_tensors='pt', padding=True)
    tokens = inputs["input_ids"]
else:  # ESM batch_converter
    labels, strs, tokens = self.batch_converter(sequences)
```

### Reusable Components (90% overlap)

| Component | ESM-2 v2.0 | DNABERT-S v2.0 | Modification |
|-----------|------------|----------------|--------------|
| IndexBasedDataset | ✓ Reuse | ✓ Reuse | None - model-agnostic FASTA reader |
| SequenceIndex | ✓ Reuse | ✓ Reuse | None - stride sharding works for all |
| create_async_dataloader | ✓ Reuse | ✓ Reuse | None - abstracts tokenization |
| VarlenCollator | ✓ Reuse | ⚠ Minor mod | Add tokenizer type detection (3 lines) |
| GreedyPacker | ✓ Reuse | ✓ Reuse | None - length-based, model-agnostic |
| AsyncInferenceRunner | ✓ Reuse | ✓ Reuse | None - calls model.forward() generically |
| gpu_worker | ✓ Reuse | ⚠ Minor mod | Add 'dnabert' to model_type switch (5 lines) |
| GPUProcessCoordinator | ✓ Reuse | ✓ Reuse | None - spawns workers generically |
| run_multi_gpu_inference | ✓ Reuse | ✓ Reuse | None - orchestration only |
| CheckpointTrigger | ✓ Reuse | ✓ Reuse | None - sequence-count-based |
| AsyncCheckpointWriter | ✓ Reuse | ✓ Reuse | None - numpy array agnostic |

**Code reuse: 90%** - Only VarlenCollator and gpu_worker need model-specific logic.

### New/Modified Components

#### 1. VarlenCollator Tokenization Detection

**File:** `virnucpro/data/collators.py`
**Lines:** ~150 (in `_tokenize_and_pack`)
**Change:** Add tokenizer type detection

```python
def _tokenize_and_pack(self, batch: List[Dict[str, str]]) -> Dict[str, Any]:
    sequences = [(item['id'], item['sequence']) for item in batch]

    # Detect tokenizer type
    if isinstance(self.batch_converter, AutoTokenizer):  # DNABERT-S
        inputs = self.batch_converter(
            [seq for _, seq in sequences],
            return_tensors='pt',
            padding=True,
            add_special_tokens=True
        )
        labels = [id for id, _ in sequences]
        tokens = inputs["input_ids"]
    else:  # ESM-2
        labels, strs, tokens = self.batch_converter(sequences)

    # Rest of packing logic unchanged
    ...
```

**Risk:** LOW - isolated conditional, both paths tested separately in v1.0.

#### 2. DNABERT-S Forward Packed Implementation

**File:** `virnucpro/models/dnabert_flash.py`
**Status:** NOT IMPLEMENTED
**Requirement:** Add `forward_packed()` method to `DNABERTWithFlashAttention`

**Challenge:** DNABERT-S (MosaicBERT) already has `BertUnpadSelfAttention` that accepts cu_seqlens. Need to:
1. Bypass HuggingFace's standard BERT forward (uses 2D)
2. Call layers directly with 1D packed format
3. Handle position embeddings for packed sequences

**Risk:** MEDIUM - DNABERT-S architecture less documented than ESM-2, requires reverse-engineering MosaicBERT's unpad path.

### Suggested Build Order for DNABERT-S v2.0

**Phase 1: Validation prep (1-2 plans)**
1. Extract DNABERT-S v1.0 reference embeddings (100 sequences)
2. Document expected shapes

**Phase 2: Tokenization (1 plan)**
3. Update VarlenCollator with tokenizer type detection
4. Test tokenization outputs match v1.0 (unpacked path)

**Phase 3: Forward packed (2-3 plans)**
5. Implement DNABERTWithFlashAttention.forward_packed()
6. Test packed == unpacked equivalence (cosine >0.999)

**Phase 4: Integration (1-2 plans)**
7. Wire DNABERT-S into gpu_worker model_type='dnabert'
8. End-to-end integration test (1K sequences)

**Phase 5: Validation (1 plan)**
9. Correctness test: v2.0 vs v1.0 embeddings (cosine >0.99)
10. Performance test: 1.8x+ scaling with 2 GPUs

**Total: 6-9 plans**

### Integration Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| DNABERT-S forward_packed breaks existing v1.0 | HIGH | Keep v1.0 path intact, add v2.0 as separate method |
| Tokenizer detection fails edge cases | MEDIUM | Comprehensive tests with ESM + DNABERT batches |
| Position embeddings wrong in packed format | HIGH | Validate against unpacked baseline (cosine >0.999) |
| FlashAttention patch conflicts with packing | MEDIUM | Test patched attention with cu_seqlens input |
| Packing efficiency lower for DNA (different length distribution) | LOW | Monitor efficiency, adjust buffer_size if needed |

## ESM-2 Model Flexibility

### Current Hardcoded repr_layers

**Locations (6 total):**
- `async_inference.py:315` - Packed inference: `repr_layers=[36]`
- `async_inference.py:340` - Unpacked inference: `repr_layers=[36]`
- `esm2_flash.py:122` - forward() default: `repr_layers = [36]`
- `esm2_flash.py:207` - forward_packed() default: `repr_layers = [36]`
- `esm2_flash.py:278` - Final layer check: `if 36 in repr_layers`
- `features.py` - 2 locations: `repr_layers=[36]`

**Problem:** ESM-2 3B has 36 layers, but:
- ESM-2 650M has 33 layers
- ESM-2 15B has 48 layers

Hardcoding `[36]` breaks model swapping.

### Proposed Architecture

**1. Model Configuration Dataclass**

**File:** `virnucpro/core/model_config.py` (NEW)

```python
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class ESM2Config:
    """ESM-2 model configuration."""
    model_name: str = "esm2_t36_3B_UR50D"
    repr_layers: List[int] = None
    enable_fp16: bool = True

    def __post_init__(self):
        if self.repr_layers is None:
            self.repr_layers = [self._get_num_layers()]

    def _get_num_layers(self) -> int:
        """Detect layer count from model name."""
        if "t33" in self.model_name:
            return 33  # 650M
        elif "t36" in self.model_name:
            return 36  # 3B
        elif "t48" in self.model_name:
            return 48  # 15B
        else:
            return None  # Detect at runtime

    @staticmethod
    def from_model_instance(model) -> 'ESM2Config':
        """Detect config from loaded model."""
        num_layers = len(model.model.layers)
        return ESM2Config(model_name="custom", repr_layers=[num_layers])
```

**2. Integration Points**

- `load_esm2_model()`: Return `(model, batch_converter, config)`
- `AsyncInferenceRunner.__init__()`: Accept `model_config: ESM2Config`
- `gpu_worker()`: Pass config through pipeline
- CLI: Add `--esm-model` and `--repr-layers` options

**Hidden dimension:** Auto-detected from `model.model.layers[0].embed_dim` at runtime. No changes needed.

### Build Order for Model Flexibility

**Phase 1: Config infrastructure (1 plan)**
1. Create ESM2Config dataclass with auto-detection
2. Update load_esm2_model to return config

**Phase 2: Pipeline integration (1-2 plans)**
3. Update AsyncInferenceRunner to use model_config.repr_layers
4. Update gpu_worker to pass config through
5. Test with 650M, 3B, 15B models

**Phase 3: CLI (1 plan)**
6. Add --esm-model and --repr-layers CLI options
7. Integration test with different model variants

**Total: 3-4 plans**

**Risk:** LOW - backward compatible (defaults to current behavior), purely additive changes.

## torch.compile Integration

### Placement Options

**Option 1: Compile entire model wrapper (RECOMMENDED)**

```python
def load_esm2_model(..., compile_model: bool = False):
    model = ESM2WithFlashAttention(base_model, device, ...)

    if compile_model:
        model = torch.compile(
            model,
            mode='reduce-overhead',
            fullgraph=False,  # Allow graph breaks
            dynamic=True      # Handle dynamic shapes
        )

    return model, batch_converter, config
```

**Option 2: Compile layers only** - Target FFN (avoid FlashAttention conflict)

**Option 3: Compile embeddings/projections only** - Safest but minimal speedup

### Dynamic Shape Considerations

**Problem:** torch.compile optimizes for specific tensor shapes. Variable-length sequences cause recompilations.

**Current shape variation:**
- `input_ids`: `[total_tokens]` - changes every batch
- `cu_seqlens`: `[num_sequences+1]` - changes every batch

**Mitigation:** Enable `dynamic=True` in torch.compile, measure recompilation overhead. If >10% slowdown, disable compile for packed path.

### FlashAttention Compatibility

**Unknown interaction** - requires testing:

**Hypothesis 1:** torch.compile treats FlashAttention as opaque (graph break)
**Hypothesis 2:** torch.compile recompiles for every cu_seqlens shape
**Hypothesis 3:** torch.compile fuses embeddings + position IDs (~10-15% speedup)

**Testing order:**
1. Compile with `fullgraph=False, dynamic=True`
2. Run packing benchmark (1K sequences)
3. Check `torch._dynamo.explain()` for graph breaks
4. If >5 graph breaks per batch → disable compile for packed path
5. If recompilation >5% overhead → disable compile entirely

### Build Order for torch.compile

**Phase 1: Basic integration (1 plan)**
1. Add compile_model parameter to load_esm2_model
2. Test with simple forward() path (no packing)
3. Measure speedup baseline

**Phase 2: Packed path testing (1-2 plans)**
4. Enable compile for forward_packed()
5. Test for graph breaks and recompilation
6. Measure speedup vs overhead

**Phase 3: Conditional compilation (1 plan)**
7. Add logic to disable compile if overhead >10%
8. Add CLI flag --torch-compile
9. Document when to use

**Total: 3-4 plans**

**Risk:** MEDIUM - Unknown interaction with FlashAttention varlen, may provide no benefit.

## Vectorized Operations

### Position ID Generation (Current Implementation)

**File:** `virnucpro/models/packed_attention.py:44-107`
**Current:** Python for-loop over cu_seqlens

```python
for i in range(num_sequences):
    start = cu_seqlens[i].item()
    end = cu_seqlens[i + 1].item()
    seq_len = end - start
    position_ids[start:end] = torch.arange(seq_len, device=cu_seqlens.device)
```

**Vectorized version:**

```python
# Compute sequence lengths: cu_seqlens[1:] - cu_seqlens[:-1]
seq_lengths = cu_seqlens[1:] - cu_seqlens[:-1]

# Create mask of sequence boundaries
mask = torch.zeros(total_len, dtype=torch.long, device=cu_seqlens.device)
mask[cu_seqlens[:-1]] = 1

# Cumsum to get sequence indices
seq_indices = mask.cumsum(0)

# Subtract starting position
position_ids = torch.arange(total_len, device=cu_seqlens.device) - cu_seqlens[seq_indices]
```

**Performance:** ~10x speedup (negligible overall <0.1% of batch time)

### Embedding Extraction (Current Implementation)

**File:** `virnucpro/pipeline/async_inference.py:348-410`
**Current:** Python for-loop with .item() CUDA syncs

```python
for i in range(len(sequence_ids)):
    start = cu_seqlens[i].item()  # CUDA sync
    end = cu_seqlens[i + 1].item()
    seq_repr = representations[start + 1:end - 1].mean(dim=0)
    embeddings.append(seq_repr)
```

**Optimized version (keep loop, remove syncs):**

```python
# Pre-convert cu_seqlens to CPU once
cu_seqlens_cpu = cu_seqlens.cpu()

for i in range(len(sequence_ids)):
    start = cu_seqlens_cpu[i].item()  # CPU tensor, no CUDA sync
    end = cu_seqlens_cpu[i + 1].item()
    seq_repr = representations[start + 1:end - 1].mean(dim=0)
    embeddings.append(seq_repr)
```

**Performance:** ~4-20x speedup (still <1% of batch time)

**Recommendation:** Optimize loop first (simpler), then consider full vectorization if profiling justifies.

### Build Order for Vectorized Ops

**Phase 1: Position ID vectorization (1 plan)**
1. Implement vectorized create_position_ids_packed
2. Validate with existing tests
3. Benchmark speedup

**Phase 2: Embedding extraction optimization (1-2 plans)**
4. Option A: Optimize loop (remove .item() syncs)
5. Option B: Full vectorization (if profiling justifies)
6. Validate correctness

**Total: 2-3 plans**

**Risk:** LOW - Pure optimization, existing tests validate correctness.

## Code Quality Architecture

### Env Var Centralization

**Current state:** 6 env vars scattered across 6 files:
- `VIRNUCPRO_DISABLE_FP16` (precision.py)
- `VIRNUCPRO_DISABLE_PACKING` (async_inference.py, 2 locations)
- `VIRNUCPRO_VIRAL_CHECKPOINT_MODE` (checkpoint_writer.py)
- `VIRNUCPRO_V1_ATTENTION` (predict.py, esm2_flash.py)
- `CUDA_VISIBLE_DEVICES` (multiple files)
- `TOKENIZERS_PARALLELISM` (dataloader_utils.py)
- `PYTORCH_CUDA_ALLOC_CONF` (5 files)

**Proposed:** `virnucpro/core/env_config.py` (NEW)

```python
class EnvConfig:
    """Centralized environment variable configuration with caching."""

    @staticmethod
    @lru_cache(maxsize=1)
    def disable_fp16() -> bool:
        """VIRNUCPRO_DISABLE_FP16: Disable FP16 precision (use FP32)."""
        return os.getenv("VIRNUCPRO_DISABLE_FP16", "").lower() in ("1", "true", "yes")

    # ... (similar for other vars)

# Global instance
env = EnvConfig()
```

**Benefits:**
- Caching: `@lru_cache` ensures env var parsed once
- Documentation: Centralized docstrings
- Testing: Easy to mock

**Migration:** Replace `os.getenv("VIRNUCPRO_DISABLE_FP16", "")...` with `env.disable_fp16()`

### Function Extraction Targets

**async_inference.py: run() method (526 lines)**

**Proposed extraction:**

```python
def run(self, dataloader, progress_callback=None, force_restart=False):
    self._setup_inference()
    resumed_result = self._resume_from_checkpoints(force_restart)

    if resumed_result:
        yield resumed_result

    for batch_result in self._process_batches(dataloader, progress_callback):
        yield batch_result

    for flushed_result in self._flush_remaining_sequences(dataloader):
        yield flushed_result

    self._finalize_inference()

def _setup_inference(self):
    """Lines 1-50: Model setup and monitoring."""
    ...

def _process_batches(self, dataloader, progress_callback) -> Iterator[InferenceResult]:
    """Lines 151-450: Main batch processing loop."""
    ...
```

**gpu_worker.py: gpu_worker function (414 lines)**

**Proposed extraction:**

```python
def gpu_worker(rank, world_size, results_queue, index_path, output_dir, model_config):
    """Main worker entry point (now ~50 lines)."""
    setup_worker_logging(rank, output_dir / "logs")

    try:
        runner, dataset, dataloader = _setup_worker_components(
            rank, world_size, index_path, output_dir, model_config
        )

        all_embeddings, all_ids = _run_worker_inference(runner, dataloader, rank)

        shard_path = _save_worker_shard(all_embeddings, all_ids, output_dir, rank)

        results_queue.put({'rank': rank, 'status': 'complete', ...})

    except RuntimeError as e:
        _handle_worker_error(e, rank, results_queue)
```

**Benefits:**
- Each function <100 lines
- Clear responsibility separation
- Easier testing

### Build Order for Code Quality

**Phase 1: Env var centralization (1 plan)**
1. Create EnvConfig class
2. Replace all os.getenv calls
3. Add tests

**Phase 2: Function extraction - async_inference (1-2 plans)**
4. Extract run() into smaller methods
5. Update tests
6. Verify no behavior change

**Phase 3: Function extraction - gpu_worker (1-2 plans)**
7. Extract gpu_worker into components
8. Update integration tests
9. Verify no behavior change

**Total: 3-5 plans**

**Risk:** LOW - Pure refactor, existing tests validate behavior unchanged.

## Suggested Build Order (All v2.5 Features)

### Recommended Sequencing

**Parallel tracks:**
1. Code Quality (independent, can start immediately)
2. Model Flexibility (independent, low risk)
3. Vectorized Ops (independent, pure optimization)
4. torch.compile (depends on Model Flexibility for testing)
5. DNABERT-S v2.0 (HIGH RISK, requires separate validation phase)

**Sequential plan:**

| Phase | Feature | Plans | Dependencies | Risk |
|-------|---------|-------|--------------|------|
| 1 | Code Quality - Env Vars | 1 | None | LOW |
| 2 | Code Quality - Function Extraction | 2-4 | None | LOW |
| 3 | Model Flexibility | 3-4 | None | LOW |
| 4 | Vectorized Ops | 2-3 | None | LOW |
| 5 | torch.compile (experimental) | 3-4 | Phase 3 | MEDIUM |
| 6 | DNABERT-S v2.0 Port | 6-9 | Phases 1-4 | HIGH |

**Total: 17-25 plans** (v2.0 took 59 plans across 10 phases, v2.5 is smaller scope)

**Critical path:** DNABERT-S v2.0 port is highest risk and should be last. All other features can proceed in parallel.

### Validation Strategy

**Per-feature validation:**
- Code Quality: Existing tests should pass unchanged
- Model Flexibility: Test with 650M, 3B, 15B models
- Vectorized Ops: Correctness tests (cosine similarity), performance benchmarks
- torch.compile: Performance benchmark, recompilation overhead check
- DNABERT-S v2.0: Dedicated validation phase with v1.0 reference

**Integration validation:**
- All features together on 1K sequence subset
- Full pipeline benchmark on 1M subset (compare to v2.0 baseline)
- Multi-GPU scaling test (2x RTX 4090)

### Rollback Plan

Each feature should have env var disable:
- Code Quality: No rollback needed (pure refactor)
- Model Flexibility: Falls back to current hardcoded [36] if config not provided
- Vectorized Ops: No rollback needed (pure optimization)
- torch.compile: `--no-torch-compile` CLI flag (default OFF for initial release)
- DNABERT-S v2.0: Keep v1.0 parallel_dnabert.py intact, add `--dnabert-v1-fallback` flag

## Sources

All sources are production codebase files analyzed directly:

**HIGH confidence (production code):**
- `.planning/PROJECT.md` - v2.0 architecture documentation
- `.planning/STATE.md` - accumulated decisions and context
- `virnucpro/pipeline/async_inference.py` - 976 lines, core v2.0 inference loop
- `virnucpro/pipeline/gpu_worker.py` - 462 lines, per-GPU worker implementation
- `virnucpro/pipeline/parallel_dnabert.py` - 608 lines, v1.0 DNABERT-S architecture
- `virnucpro/data/collators.py` - 367 lines, VarlenCollator tokenization + packing
- `virnucpro/data/packing.py` - 532 lines, GreedyPacker FFD algorithm
- `virnucpro/models/esm2_flash.py` - 668 lines, ESM-2 with FlashAttention varlen
- `virnucpro/models/dnabert_flash.py` - 438 lines, DNABERT-S with patched attention
- `virnucpro/models/packed_attention.py` - 205 lines, position ID generation utilities

**Line counts verified via `wc -l` on 2026-02-09**
