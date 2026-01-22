# VirNucPro - Viral Nucleotide and Protein Identifier

A production-ready refactoring of the [original VirNucPro tool](https://github.com/Li-Jing-1997/VirNucPro) for identifying viral sequences using six-frame translation and deep learning models (DNABERT-S and ESM-2).

## About This Project

This is a comprehensive refactoring of the original VirNucPro bioinformatics tool, transforming it from a collection of standalone scripts into a production-ready Python package with:

- 🎯 **Modern CLI interface** with Click framework
- 🔧 **Modular architecture** with proper package structure
- 🎮 **GPU device selection** with validation and auto-detection
- 💾 **Checkpointing/resume capability** with hash-based validation
- 🚀 **Multi-GPU parallelization** for 150-380x speedup
- ⚡ **Batched processing** with proper attention masking (50-100x faster)
- ⚙️ **YAML configuration** support with CLI overrides
- 📊 **Progress reporting** with tqdm integration
- 📝 **Comprehensive logging** with configurable levels
- ✅ **Input validation** and error handling
- 🧹 **Automatic cleanup** of intermediate files
- ✅ **Comprehensive testing** with vanilla comparison validation

### Original Tool

The original VirNucPro was developed by Li Jing and is available at:
**https://github.com/Li-Jing-1997/VirNucPro**

This refactoring maintains full compatibility with the original tool's prediction methodology while adding enterprise-grade features for production use.

## Project Status

✅ **Production Ready** - All core features implemented and tested.

### Completed
- ✅ **Phase 1**: Core infrastructure (config, logging, device management, progress reporting)
- ✅ **Phase 2**: Core pipeline refactoring (extracting models and utilities)
- ✅ **Phase 3**: CLI implementation with Click
- ✅ **Phase 4**: Checkpointing system with hash-based validation and resume capability
- ✅ **Phase 5**: Testing framework with vanilla comparison validation

### Validated
- ✅ **30/30 unit tests passing** (features, parallel processing, predictions)
- ✅ **Vanilla comparison tests passing** - Predictions match 100% with batching optimizations
- ✅ **Multi-GPU parallelization** - 150-380x speedup with 4 GPUs
- ✅ **Batched processing** - 50-100x faster than sequential with proper attention masking

See [STATUS.md](STATUS.md) for detailed progress tracking.

## Features

### Original VirNucPro Capabilities

- Six-frame translation of DNA sequences
- DNABERT-S feature extraction for nucleotide sequences
- ESM-2 (3B) feature extraction for protein sequences
- MLP-based viral sequence classification
- Support for 300bp and 500bp sequence models
- Consensus scoring across reading frames

### New Refactored Features

- **Click-based CLI**: Intuitive command-line interface
  ```bash
  python -m virnucpro predict input.fasta --model-type 500 --device cuda:0
  ```

- **GPU Selection**: Flexible device management
  ```bash
  python -m virnucpro utils list-devices
  python -m virnucpro predict input.fasta --device cuda:1
  ```

- **Resume Capability**: Automatic checkpointing
  ```bash
  python -m virnucpro predict input.fasta --resume
  ```

- **Configuration Management**: YAML-based settings
  ```bash
  python -m virnucpro utils generate-config -o my_config.yaml
  python -m virnucpro predict input.fasta --config my_config.yaml
  ```

- **Input Validation**: Pre-flight checks
  ```bash
  python -m virnucpro utils validate input.fasta
  ```

## Installation

### Requirements

- Python 3.9+
- PyTorch (with optional CUDA support)
- BioPython
- transformers (HuggingFace)
- ESM (Facebook Research)
- Click, PyYAML, tqdm

### Setup

1. Clone this repository:
```bash
git clone https://github.com/YOUR-USERNAME/virnucpro-broad.git
cd virnucpro-broad
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Verify installation:
```bash
python -c "import virnucpro; print(virnucpro.__version__)"
```

## Usage

### Quick Start

Basic prediction workflow:

```bash
# Basic prediction with default settings
python -m virnucpro predict input.fasta

# Use specific model and GPU
python -m virnucpro predict input.fasta --model-type 300 --device cuda:0

# Resume interrupted prediction
python -m virnucpro predict input.fasta --resume

# Custom configuration
python -m virnucpro predict input.fasta --config my_config.yaml
```

### Multi-GPU Parallel Processing

Enable parallel DNABERT-S feature extraction across multiple GPUs for significant speedup:

```bash
# Enable parallel processing (auto-detects all GPUs)
python -m virnucpro predict input.fasta --parallel

# Combine with custom batch size for memory management
python -m virnucpro predict input.fasta --parallel --dnabert-batch-size 128

# Full example with all options
python -m virnucpro predict input.fasta \
  --model-type 500 \
  --parallel \
  --dnabert-batch-size 256 \
  --resume
```

**Performance**: With 4 GPUs, expect 150-380x speedup compared to sequential processing.

**Memory considerations**:
- Default `--dnabert-batch-size 256` requires ~2GB VRAM per GPU
- For GPUs with less VRAM: use `--dnabert-batch-size 128` (2-4GB) or `--dnabert-batch-size 64` (<2GB)
- Each GPU loads its own DNABERT-S model instance (1.5GB)

**When to use**:
- Large input files (>100k sequences)
- Multiple GPUs available
- Not in shared GPU environments (use `--parallel` opt-in to avoid monopolizing resources)

### Current Status (Phase 1)

Phase 1 infrastructure is complete and can be tested:

```bash
# Test package import
python -c "import virnucpro; print(virnucpro.__version__)"

# Test configuration loading
python -c "from virnucpro.core.config import Config; c = Config.load(); print(c.get('prediction.batch_size'))"

# Test device management
python -c "from virnucpro.core.device import list_available_devices; list_available_devices()"
```

## Architecture

```
virnucpro-broad/
├── virnucpro/                  # Main package
│   ├── __init__.py            # Package initialization
│   ├── __main__.py            # CLI entry point
│   ├── cli/                   # Command-line interface
│   │   ├── main.py           # Main Click group
│   │   ├── predict.py        # Predict command
│   │   └── utils.py          # Utility commands
│   ├── core/                  # Core infrastructure
│   │   ├── config.py         # Configuration management
│   │   ├── device.py         # GPU/device handling
│   │   ├── logging_setup.py  # Logging configuration
│   │   └── checkpoint.py     # Checkpointing system
│   ├── pipeline/              # Prediction pipeline
│   │   ├── models.py         # PyTorch models
│   │   ├── prediction.py     # Main pipeline
│   │   ├── chunking.py       # Sequence chunking
│   │   ├── translation.py    # Six-frame translation
│   │   └── features.py       # Feature extraction
│   └── utils/                 # Utilities
│       ├── sequence.py       # Sequence processing
│       ├── validation.py     # Input validation
│       └── progress.py       # Progress reporting
├── config/                    # Configuration files
│   └── default_config.yaml   # Default settings
├── tests/                     # Test suite
├── thoughts/                  # Planning documents
│   └── shared/
│       ├── plans/            # Implementation plans
│       └── research/         # Research notes
├── prediction.py             # Original script (reference)
├── units.py                  # Original utilities (reference)
├── 300_model.pth            # Pre-trained model (300bp)
├── 500_model.pth            # Pre-trained model (500bp)
└── README.md                # This file
```

## Development

### Refactoring Plan

The refactoring follows a phased approach documented in:
`thoughts/shared/plans/2025-11-10-virnucpro-cli-refactoring.md`

**Phase 1: Project Structure & Infrastructure** ✅
- Package structure with modular design
- YAML configuration system with CLI overrides
- Structured logging framework with levels
- GPU device management and validation
- Integrated progress reporting with tqdm

**Phase 2: Core Pipeline Refactoring** ✅
- Extracted and modularized pipeline components
- Comprehensive docstrings
- Type hints throughout
- Full backward compatibility maintained

**Phase 3: CLI Implementation** ✅
- Click-based command interface
- Input validation
- Comprehensive error handling

**Phase 4: Checkpointing System** ✅
- Hash-based state tracking with config validation
- Resume capability for interrupted runs
- Stage-level and file-level checkpoints
- Atomic state saves

**Phase 5: Testing & Documentation** ✅
- 30 unit tests for features, parallel processing, predictions
- Vanilla comparison validation (100% prediction match)
- Comprehensive documentation with performance benchmarks
- Empirically-validated tolerances for batching differences

### Contributing

This is an active refactoring project. If you'd like to contribute:

1. Check the current status in [STATUS.md](STATUS.md)
2. Review the implementation plan in `thoughts/shared/plans/`
3. Open an issue to discuss proposed changes
4. Submit a pull request

## Comparison with Original

| Feature | Original VirNucPro | This Refactoring |
|---------|-------------------|------------------|
| **CLI Interface** | Basic `sys.argv` | Click framework with help |
| **Configuration** | Hardcoded values | YAML config + CLI overrides |
| **GPU Selection** | Auto-detect only | Manual selection + validation |
| **Multi-GPU Support** | Not available | Parallel processing with `--parallel` |
| **Batching** | Sequential (1 seq/GPU call) | Batched processing (256 seqs/batch) |
| **Performance** | Baseline | 150-380x speedup with 4 GPUs |
| **Error Handling** | Minimal | Comprehensive validation |
| **Logging** | Print statements | Structured logging (levels) |
| **Progress** | Basic tqdm | Integrated progress bars |
| **Resume** | Not available | Checkpoint-based resume ✅ |
| **Package Structure** | Flat scripts | Modular package |
| **Testing** | None | 30 unit tests + vanilla validation ✅ |
| **Documentation** | Basic README | Comprehensive docs + types |
| **Input Validation** | None | Pre-flight checks |
| **Cleanup** | Manual | Automatic (configurable) |

## Citation

If you use VirNucPro in your research, please cite the original tool:

```
[Citation information for original VirNucPro - to be added]
Repository: https://github.com/Li-Jing-1997/VirNucPro
```

## License

[License information to be determined - should match or be compatible with original]

See [LICENSE](LICENSE) for details.

## Acknowledgments

- **Original VirNucPro**: [Li Jing](https://github.com/Li-Jing-1997) and contributors
- **DNABERT-S**: Zhihan Zhou et al.
- **ESM-2**: Meta AI Research (Facebook)
- **BioPython**: The BioPython Project
- **PyTorch**: Meta AI Research

## Contact

For questions about this refactoring project:
- Open an issue on GitHub
- See [STATUS.md](STATUS.md) for project status

For questions about the original VirNucPro methodology:
- See the [original repository](https://github.com/Li-Jing-1997/VirNucPro)

## Testing & Validation

### Test Suite

The refactored implementation includes comprehensive testing:

```bash
# Run all tests
pixi run pytest tests/ -v

# Run specific test suites
pixi run pytest tests/test_features.py -v          # Feature extraction tests
pixi run pytest tests/test_parallel.py -v          # Multi-GPU parallelization tests
pixi run pytest tests/test_vanilla_comparison.py -v # Vanilla equivalence validation
```

### Vanilla Comparison

Extensive testing confirms the refactored implementation produces **scientifically equivalent** results to the original:

- ✅ **100% prediction match**: All virus/non-virus classifications identical
- ✅ **Negligible score differences**: <0.001% variance in prediction scores
- ✅ **Embedding differences**: ~1-2% from batching optimizations (scientifically irrelevant)

**Root cause of embedding differences**:
- Batched processing (4 sequences/batch vs. 1 sequence/call)
- Proper attention masking for padding tokens (more mathematically correct)

**Impact**: The small embedding differences are absorbed by the MLP classifier and do not affect final predictions.

See `tests/VANILLA_COMPARISON_RESULTS.md` for detailed analysis.

## Project Timeline

- **2025-11-10**: Phase 1 infrastructure complete
- **2025-11-15**: Phase 2 pipeline refactoring complete
- **2025-11-18**: Phase 3 CLI implementation complete
- **2025-12-15**: Phase 4 checkpointing system complete
- **2026-01-22**: Phase 5 testing & validation complete

---

**Status**: ✅ **Production ready** - All phases complete and validated against vanilla implementation.
