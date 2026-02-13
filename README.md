# VirNucPro - Viral Nucleotide and Protein Identifier

A production-ready refactoring of the [original VirNucPro tool](https://github.com/Li-Jing-1997/VirNucPro) for identifying viral sequences using six-frame translation and deep learning models (DNABERT-S and ESM-2).

## About This Project

This is a comprehensive refactoring and GPU optimization of the original VirNucPro bioinformatics tool. The v2.0 async architecture delivers a **6.2x speedup** over v1.0 (3.5 hours to 34 minutes on 2x RTX 4090) through single-process-per-GPU design, async DataLoader, sequence packing with FlashAttention, and FP16 precision.

### Key Capabilities

- **6.2x speedup** over v1.0 baseline with async DataLoader and sequence packing
- **Multi-GPU scaling** with 93.7% efficiency on ESM-2 (1.87x on 2 GPUs)
- **Sequence packing** via FFD algorithm (~358% token utilization) with FlashAttention varlen
- **FP16 precision** with >0.99 cosine similarity to FP32
- **Fault-tolerant checkpointing** with SIGTERM handling and elastic redistribution
- **Click-based CLI** with YAML configuration and resume capability
- **99.87% prediction accuracy** compared to v1.0

### Original Tool

The original VirNucPro was developed by Li Jing and is available at:
**https://github.com/Li-Jing-1997/VirNucPro**

This refactoring maintains full compatibility with the original tool's prediction methodology while adding GPU optimization and production-grade features.

## Installation

### Requirements

- Python 3.9+
- PyTorch >= 2.8.0 (with CUDA support)
- BioPython
- transformers == 4.30.0 (DNABERT-S)
- fair-esm == 2.0.0 (ESM-2 3B)
- flash-attn >= 2.6.0 (FlashAttention-2 for packed attention with FP32 accumulation)
- Click, PyYAML, tqdm, rich, h5py

### Setup

1. Clone this repository:
```bash
git clone https://github.com/broadinstitute/virnucpro-broad.git
cd virnucpro-broad
```

2. Install dependencies:
```bash
pixi install          # Uses pixi (Python 3.9, conda-forge)
pip install -r requirements.txt
```

> **Note:** `flash-attn` ships pre-built wheels for common CUDA/torch/Python combinations.
> If installation fails (e.g., uncommon CUDA version), install it separately with:
> ```bash
> pip install flash-attn>=2.6.0 --no-build-isolation
> ```

3. Verify installation:
```bash
python -c "import virnucpro; print(virnucpro.__version__)"
```

## Usage

### Quick Start

```bash
# Basic prediction (single GPU)
python -m virnucpro predict input.fasta

# Multi-GPU with v2.0 async architecture (recommended)
python -m virnucpro predict input.fasta --parallel

# Use specific model type
python -m virnucpro predict input.fasta --parallel --model-type 500

# Resume interrupted prediction
python -m virnucpro predict input.fasta --parallel --resume
```

### Multi-GPU Processing

The `--parallel` flag enables the v2.0 async architecture for ESM-2 (auto-detects all available GPUs):

```bash
# Auto-detect GPUs, use v2.0 architecture
python -m virnucpro predict input.fasta --parallel

# Fall back to v1.0 architecture if needed
python -m virnucpro predict input.fasta --parallel --v1-fallback

# Use v1.0-compatible attention (exact v1.0 embedding reproduction)
python -m virnucpro predict input.fasta --parallel --v1-attention
```

**Performance** (1M sequence subset):

| Configuration | Time | Speedup |
|--------------|------|---------|
| v1.0 baseline | 3.5 hours | 1.0x |
| v2.0, 1x RTX 4090 | 53 min | 4.0x |
| v2.0, 2x RTX 4090 | 34 min | 6.2x |

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VIRNUCPRO_DISABLE_PACKING` | `false` | Disable sequence packing (emergency rollback) |
| `VIRNUCPRO_DISABLE_FP16` | `false` | Disable FP16 precision |
| `VIRNUCPRO_V1_ATTENTION` | `false` | Use v1.0-compatible attention path |
| `VIRNUCPRO_VIRAL_CHECKPOINT_MODE` | `false` | Tune checkpointing for viral workloads |

### Utilities

```bash
# List available GPU devices
python -m virnucpro utils list-devices

# Validate input file
python -m virnucpro utils validate input.fasta

# Generate default config
python -m virnucpro utils generate-config -o my_config.yaml
```

## Architecture

### v2.0 Pipeline (ESM-2)

```
FASTA Files
    |
    v
SequenceIndex (length-sorted, stride-based sharding)
    |
    v
[GPU 0]                    [GPU 1]                    [GPU N]
IndexBasedDataset          IndexBasedDataset          IndexBasedDataset
(byte-offset reading)      (byte-offset reading)      (byte-offset reading)
    |                          |                          |
DataLoader Workers         DataLoader Workers         DataLoader Workers
(4-8 CPU workers, I/O)     (4-8 CPU workers, I/O)     (4-8 CPU workers, I/O)
    |                          |                          |
VarlenCollator             VarlenCollator             VarlenCollator
(tokenize + FFD packing)   (tokenize + FFD packing)   (tokenize + FFD packing)
    |                          |                          |
AsyncInferenceRunner       AsyncInferenceRunner       AsyncInferenceRunner
(FP16 + FlashAttention)    (FP16 + FlashAttention)    (FP16 + FlashAttention)
    |                          |                          |
shard_0.h5                 shard_1.h5                 shard_N.h5
    \                         |                         /
     \________________________|________________________/
                              |
                    Shard Aggregator
                    (embeddings.h5)
```

Each GPU runs its own `AsyncInferenceRunner` in a spawned process coordinated by `GPUProcessCoordinator`.

### Project Structure

```
virnucpro/
  cli/                        # Click-based CLI
    main.py, predict.py, profile.py, benchmark.py
  core/                       # Config, device validation, checkpointing
  cuda/                       # Stream manager, attention utils, memory manager
  data/                       # Async DataLoader components (v2.0)
    collators.py              #   VarlenCollator with buffer-based packing
    dataloader_utils.py       #   create_async_dataloader() factory
    packing.py                #   GreedyPacker FFD algorithm (~92-94% efficiency)
    sequence_dataset.py       #   IndexBasedDataset for byte-offset reading
    shard_index.py            #   Multi-GPU stride-based index distribution
  models/                     # ESM-2 FlashAttention, DNABERT flash, packed attention
  pipeline/                   # Inference orchestration
    async_inference.py        #   AsyncInferenceRunner (single-GPU)
    multi_gpu_inference.py    #   run_multi_gpu_inference() entry point
    gpu_coordinator.py        #   GPUProcessCoordinator lifecycle management
    gpu_worker.py             #   Per-GPU worker function
    shard_aggregator.py       #   HDF5 shard merging with validation
    checkpoint_writer.py      #   Async checkpointing with crash recovery
    checkpoint_manifest.py    #   Multi-GPU checkpoint coordination
    prediction.py             #   Full 9-stage pipeline orchestration
  utils/                      # Sequence processing, validation, GPU monitor
tests/
  unit/                       # Component tests (62 test files)
  integration/                # Multi-component tests
  benchmarks/                 # Performance and scaling tests
```

## Testing

```bash
# All tests
pytest tests/ -v

# Unit tests only
pytest tests/unit/ -v

# Skip slow tests
pytest tests/ -v -m "not slow"

# GPU-only tests
pytest tests/ -v -m "gpu"

# Pattern match
pytest tests/ -v -k "test_packing"
```

**Test coverage**: 18,846 lines of production Python (61 files), 27,951 lines of test Python (62 files).

## Performance Validation

v2.0 was validated on a 1M sequence subset with 2x RTX 4090:

| Metric | Result | Target |
|--------|--------|--------|
| v2.0 vs v1.0 speedup | 6.2x | >= 4.0x |
| 1x GPU time | 53 min | < 1 hour |
| 2x GPU time | 34 min | — |
| ESM-2 scaling (2 GPUs) | 1.87x (93.7%) | > 1.8x |
| Prediction accuracy | 99.87% | > 99% |
| Packing efficiency | ~358% | > 200% |
| ESM-2 throughput | 321 seq/s, 16.5K tok/s | — |

## Comparison with Original

| Feature | Original VirNucPro | This Refactoring |
|---------|-------------------|------------------|
| CLI Interface | Basic `sys.argv` | Click framework with help |
| Configuration | Hardcoded values | YAML config + CLI overrides |
| GPU Processing | Single GPU | Multi-GPU with async DataLoader |
| Sequence Batching | Sequential (1 seq/call) | Packed batches with FlashAttention |
| Precision | FP32 | FP16 with >0.99 cosine similarity |
| Performance | Baseline | 6.2x speedup (2x RTX 4090) |
| Checkpointing | Not available | Fault-tolerant with crash recovery |
| Resume | Not available | Automatic with elastic redistribution |
| Error Handling | Minimal | Comprehensive with SIGTERM handling |
| Logging | Print statements | Structured logging (levels) |
| Testing | None | 62 test files, unit + integration |

## Project Timeline

- **2025-11-10**: Core infrastructure (config, logging, device management)
- **2025-11-15**: Pipeline refactoring (modular architecture)
- **2025-11-18**: CLI implementation with Click
- **2025-12-15**: Checkpointing system
- **2026-01-22**: Testing framework and validation
- **2026-02-02**: **v1.0 GPU Optimization** shipped (multi-GPU, BF16, FlashAttention-2)
- **2026-02-09**: **v2.0 Async Architecture** shipped (6.2x speedup, sequence packing, FP16)

## Citation

If you use VirNucPro in your research, please cite the original tool:

```
Repository: https://github.com/Li-Jing-1997/VirNucPro
```

## Acknowledgments

- **Original VirNucPro**: [Li Jing](https://github.com/Li-Jing-1997) and contributors
- **DNABERT-S**: Zhihan Zhou et al.
- **ESM-2**: Meta AI Research (Facebook)
- **FlashAttention-2**: Tri Dao et al.
- **BioPython**: The BioPython Project
- **PyTorch**: Meta AI Research
