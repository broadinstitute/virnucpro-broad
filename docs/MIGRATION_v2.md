# Migration Guide: v1.0 to v2.0

## Overview

VirNucPro v2.0 replaces the multi-worker-per-GPU architecture with a single-process-per-GPU
async DataLoader pattern for ESM-2 protein embedding. This delivers 4-5x throughput improvement
through sequence packing, FlashAttention varlen, and FP16 precision.

**v2.0 is the default when using `--parallel`.** v1.0 behavior is available via `--v1-fallback`.

## Hybrid Architecture (v2.0)

v2.0 uses a **hybrid approach** where only ESM-2 embedding uses the new architecture:

| Pipeline Stage | v1.0 | v2.0 (current) | Future (v2.1) |
|----------------|------|----------------|---------------|
| Chunking | Sequential | Sequential (unchanged) | - |
| Translation | Parallel CPU | Parallel CPU (unchanged) | - |
| File Splitting | Sequential | Sequential (unchanged) | - |
| DNABERT-S embedding | Multi-worker-per-GPU | **v1.0 (unchanged)** | v2.0 planned |
| ESM-2 embedding | Multi-worker-per-GPU | **v2.0 async DataLoader** | - |
| Feature Merging | Parallel CPU | Parallel CPU (unchanged) | - |
| Prediction | Sequential | Sequential (unchanged) | - |
| Consensus | Sequential | Sequential (unchanged) | - |

### Why DNABERT-S Stays v1.0

1. **Performance:** DNABERT-S processes 1M sequences in 3-4 minutes. ESM-2 (3B params) is the
   bottleneck at 45+ hours. Optimizing DNABERT-S provides minimal overall speedup.
2. **Validation:** Phases 5-9 only validated ESM-2 protein sequences through v2.0. DNABERT-S
   requires different tokenizer (DNA k-mers), different max_length (512), and different input
   format (nucleotide chunks vs protein sequences).
3. **Risk:** Running untested DNABERT-S through v2.0 could produce garbage embeddings.

DNABERT-S v2.0 support is planned for Phase 10.2 or the v2.1 milestone.

## What Changed

### Architecture (ESM-2 Only)

| Aspect | v1.0 | v2.0 |
|--------|------|------|
| GPU process model | N workers per GPU (multiprocessing.Pool) | 1 process per GPU (async DataLoader) |
| Memory per GPU | N x 11GB (model copies) | 1 x 6GB (single model, FP16) |
| Batching | Fixed batch size, padded | Dynamic token budget, packed sequences |
| Precision | BF16 (Ampere+) / FP32 | FP16 with FlashAttention |
| Attention | Standard / PyTorch SDPA | FlashAttention-2 varlen (packed) |
| Checkpointing | Per-file .pt with .done markers | Per-shard HDF5 with manifest |
| Output format | Per-file .pt tensors | Merged embeddings.h5 (adapter converts to .pt for pipeline) |

### CLI Changes

**No breaking changes to command-line interface.**

The `--parallel` flag now routes ESM-2 to v2.0 architecture by default. All existing flags
continue to work. New flags added:

| Flag | Purpose |
|------|---------|
| `--v1-fallback` | Force v1.0 multi-worker architecture for all stages (including ESM-2) |

### Deprecated Options

These options still work but have no effect in v2.0 mode (ESM-2 only):

| Option | v1.0 Behavior | v2.0 Behavior |
|--------|---------------|---------------|
| `--esm-batch-size` | Sets tokens per batch | Ignored (dynamic token budget from GPU memory) |
| `--dnabert-batch-size` | Sets tokens per batch | Still used (DNABERT-S is v1.0) |
| `--dataloader-workers` | Sets DataLoader worker count | Used by v2.0 DataLoader (default: auto-detect) |
| `--persistent-models` | Keep models loaded between stages | No effect on ESM-2 (single model per process) |

### New Capabilities (v2.0, ESM-2 only)

- **Sequence packing**: Multiple short sequences packed into single batches via FFD algorithm
- **FlashAttention varlen**: Packed attention without cross-sequence contamination
- **Incremental checkpointing**: Resume from sequence-level checkpoints (not file-level)
- **Spot instance support**: SIGTERM handler saves emergency checkpoint, automatic resume
- **Elastic redistribution**: Failed GPU work reassigned to healthy GPUs

## How to Migrate

### Basic Usage (No Changes Required)

```bash
# This just works -- v2.0 is the default for ESM-2 with --parallel
python -m virnucpro predict input.fasta --parallel
```

### If You Need Full v1.0 Behavior

```bash
# Explicitly request v1.0 architecture for all stages
python -m virnucpro predict input.fasta --parallel --v1-fallback
```

### Checking Which Architecture Is Active

Look for architecture log line in output:
```
Architecture: v2.0 hybrid
  ESM-2 embedding: v2.0 (async DataLoader + sequence packing)
  DNABERT-S embedding: v1.0 (fast enough, v2.0 planned for v2.1)
```
or:
```
Architecture: v1.0 (--v1-fallback, all stages use legacy multi-worker)
```

### Running Performance Benchmarks

```bash
# Full v2.0 validation suite
virnucpro benchmark --suite v2-validation

# v1.0 vs v2.0 comparison (requires v1.0 git tag)
virnucpro benchmark --suite v1-comparison

# Multi-GPU scaling test
virnucpro benchmark --suite v2-scaling

# Parameter optimization sweep
virnucpro benchmark --suite v2-throughput
```

### Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `VIRNUCPRO_DISABLE_FP16` | Force FP32 precision in v2.0 ESM-2 | unset (FP16 enabled) |
| `VIRNUCPRO_DISABLE_PACKING` | Disable sequence packing in v2.0 ESM-2 | unset (packing enabled) |

## Deprecation Timeline

- **v2.0 (current)**: `--v1-fallback` available, v1.0 ESM-2 code remains. DNABERT-S uses v1.0.
- **v2.1 (planned)**: DNABERT-S v2.0 support added. `--v1-fallback` emits deprecation warning.
- **v3.0 (future)**: v1.0 code removed, `--v1-fallback` removed.

## Error Handling (v2.0 ESM-2)

v2.0 uses **fail-fast** error handling for ESM-2:
- If any GPU worker fails during ESM-2 embedding, the entire stage fails with RuntimeError
- This is different from v1.0 which tracked individual file failures
- Rationale: v2.0 distributes sequences across GPUs by stride, so a failed worker means
  missing every Nth sequence, which would corrupt downstream predictions

## Troubleshooting

### v2.0 crashes but v1.0 works
```bash
# Temporary: use v1.0 while investigating
python -m virnucpro predict input.fasta --parallel --v1-fallback

# Check if FP16 issue:
VIRNUCPRO_DISABLE_FP16=true python -m virnucpro predict input.fasta --parallel

# Check if packing issue:
VIRNUCPRO_DISABLE_PACKING=true python -m virnucpro predict input.fasta --parallel
```

### GPU Out of Memory in v2.0
v2.0 uses dynamic token budgets for ESM-2. If OOM occurs:
1. The system automatically adjusts batch size based on GPU memory
2. Emergency: set `VIRNUCPRO_DISABLE_PACKING=true` to reduce memory usage
3. Emergency: set `VIRNUCPRO_DISABLE_FP16=true` (counter-intuitive but may help with memory fragmentation)
