# Directory Structure

## Repository Layout

```
virnucpro-broad/
├── virnucpro/               # Main package
│   ├── __init__.py
│   ├── __main__.py          # Entry point for `python -m virnucpro`
│   ├── cli/                 # Command-line interface
│   ├── core/                # Core infrastructure
│   ├── pipeline/            # Prediction pipeline
│   └── utils/               # Shared utilities
├── tests/                   # Test suite
│   ├── data/                # Test fixtures (FASTA files)
│   └── conftest.py          # Pytest fixtures
├── config/                  # Configuration files
│   └── default_config.yaml  # Default pipeline config
├── 300_model.pth            # Pre-trained model (300bp)
├── 500_model.pth            # Pre-trained model (500bp)
├── prediction.py            # Legacy vanilla script
├── units.py                 # Legacy vanilla utilities
├── pyproject.toml           # Python package metadata
├── pixi.toml                # Pixi environment spec
├── pixi.lock                # Locked dependencies
├── README.md                # Project documentation
└── USA-MA-Broad_MGH-*/      # Example output directory
```

## Key Directories

### `virnucpro/` - Main Package

**Purpose:** Core application code organized by layer

**Subdirectories:**

#### `virnucpro/cli/` - Command-Line Interface
```
cli/
├── __init__.py
├── main.py          # CLI entry point (Click group)
└── predict.py       # Predict command (183 lines)
```

**Naming Pattern:** Verb-based command files (`predict.py`)

#### `virnucpro/core/` - Core Infrastructure
```
core/
├── __init__.py
├── config.py        # Configuration management (143 lines)
├── checkpointing.py # Checkpoint management (347 lines)
├── device.py        # GPU/CPU device management (149 lines)
└── logging.py       # Logging setup (76 lines)
```

**Naming Pattern:** Noun-based infrastructure modules

#### `virnucpro/pipeline/` - Prediction Pipeline
```
pipeline/
├── __init__.py
├── prediction.py                    # Main orchestrator (462 lines)
├── feature_extraction.py            # Feature extraction (244 lines)
├── parallel_feature_extraction.py   # Multi-GPU extraction (113 lines)
├── models.py                        # Model loading (110 lines)
└── predictor.py                     # Prediction logic (165 lines)
```

**Naming Pattern:** Process/action-based modules

#### `virnucpro/utils/` - Shared Utilities
```
utils/
├── __init__.py
├── sequence_utils.py  # FASTA I/O, chunking, translation (294 lines)
├── validation.py      # Input validation (94 lines)
├── progress.py        # Progress bar utilities (185 lines)
└── file_utils.py      # File operations (85 lines)
```

**Naming Pattern:** `*_utils.py` for utility modules

### `tests/` - Test Suite

```
tests/
├── __init__.py
├── conftest.py                           # Pytest fixtures
├── test_prediction_vanilla_comparison.py # Main test suite
└── data/                                 # Test fixtures
    ├── test_fixed_500bp_chunked500.fa
    ├── test_fixed_500bp_identified_nucleotide.fa
    ├── test_fixed_500bp_identified_protein.faa
    ├── test_fixed_500bp_merged/
    ├── test_fixed_500bp_nucleotide/
    ├── test_fixed_500bp_protein/
    ├── test_fixed_500bp_refactored/
    ├── test_sequences.fa
    ├── test_sequences_small.fa
    ├── test_sequences_small_chunked500.fa
    ├── test_sequences_small_identified_nucleotide.fa
    ├── test_sequences_small_identified_protein.faa
    ├── test_with_orfs_chunked500.fa
    ├── test_with_orfs_identified_nucleotide.fa
    ├── test_with_orfs_identified_protein.faa
    ├── test_with_orfs_merged/
    ├── test_with_orfs_nucleotide/
    ├── test_with_orfs_protein/
    └── test_with_orfs_refactored/
```

**Organization:** Test data includes intermediate outputs for vanilla comparison

**Naming Convention:**
- Test files: `test_*.py`
- Test data: `test_*.fa` or `test_*_<stage>.fa`

### `config/` - Configuration

```
config/
└── default_config.yaml  # Default pipeline configuration
```

**Contents:** Chunk sizes, model paths, feature extraction settings

## Key File Locations

### Entry Points

| Purpose | File Path |
|---------|-----------|
| Main entry | `virnucpro/__main__.py` |
| CLI setup | `virnucpro/cli/main.py` |
| Predict command | `virnucpro/cli/predict.py` |

### Pipeline Core

| Purpose | File Path |
|---------|-----------|
| Pipeline orchestration | `virnucpro/pipeline/prediction.py` |
| Feature extraction | `virnucpro/pipeline/feature_extraction.py` |
| Model loading | `virnucpro/pipeline/models.py` |

### Infrastructure

| Purpose | File Path |
|---------|-----------|
| Configuration | `virnucpro/core/config.py` |
| Checkpointing | `virnucpro/core/checkpointing.py` |
| Device management | `virnucpro/core/device.py` |

### Utilities

| Purpose | File Path |
|---------|-----------|
| Sequence operations | `virnucpro/utils/sequence_utils.py` |
| Validation | `virnucpro/utils/validation.py` |
| Progress tracking | `virnucpro/utils/progress.py` |

### Configuration & Models

| Purpose | File Path |
|---------|-----------|
| Default config | `config/default_config.yaml` |
| 300bp model | `300_model.pth` |
| 500bp model | `500_model.pth` |

### Legacy/Reference

| Purpose | File Path |
|---------|-----------|
| Original prediction | `prediction.py` |
| Original utilities | `units.py` |

## Naming Conventions

### Modules

**Pattern: Snake case**
- `sequence_utils.py`
- `parallel_feature_extraction.py`
- `checkpointing.py`

### Packages

**Pattern: Lowercase, short, descriptive**
- `cli/`
- `core/`
- `pipeline/`
- `utils/`

### Python Files

**All packages have `__init__.py`:**
- Marks directories as Python packages
- Mostly empty (no re-exports)

### Test Files

**Pattern: `test_<functionality>.py`**
- `test_prediction_vanilla_comparison.py`

### Test Data

**Pattern: `test_<variant>_<stage>.<ext>`**
- `test_sequences.fa` - Input sequences
- `test_fixed_500bp_chunked500.fa` - Chunked output
- `test_with_orfs_identified_protein.faa` - Protein sequences

### Configuration Files

**Pattern: `<purpose>_config.yaml`**
- `default_config.yaml`

### Model Files

**Pattern: `<variant>_model.pth`**
- `300_model.pth`
- `500_model.pth`

## Output Directory Structure

**Created at runtime (user-specified via `--output-dir`):**

```
output_dir/
├── chunked_sequences.pt                # Stage 1 checkpoint
├── translated_sequences/               # Stage 2 checkpoint
│   └── *.fa
├── nucleotide/                         # Stage 3 checkpoint (nucleotide)
│   └── *.fa
├── protein/                            # Stage 3 checkpoint (protein)
│   └── *.faa
├── nucleotide_features.pt              # Stage 4 checkpoint
├── protein_features.pt                 # Stage 5 checkpoint
├── merged_features.pt                  # Stage 6 checkpoint
├── predictions.pt                      # Stage 7 checkpoint
├── consensus.pt                        # Stage 8 checkpoint
├── predictions.csv                     # Final output
└── consensus.txt                       # Final output
```

**Pattern:** Stage checkpoints use `.pt` (PyTorch) or FASTA format

## Import Path Patterns

**Absolute imports from package root:**
```python
from virnucpro.core.config import load_config
from virnucpro.pipeline.prediction import run_prediction
from virnucpro.utils.sequence_utils import chunk_sequences
```

**No relative imports detected**

## Package Organization

**Clean layered structure:**
- CLI depends on Pipeline
- Pipeline depends on Core + Utils
- Core and Utils are independent
- No circular dependencies

**All `__init__.py` files present for proper package structure**
