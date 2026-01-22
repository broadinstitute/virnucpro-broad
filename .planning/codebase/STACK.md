# Technology Stack

## Languages

**Primary Language: Python 3.9**
- Locked via pixi package manager
- Minimum version enforced in `pyproject.toml`

**Build Dependencies:**
- Rust (required for certain package builds)

## Runtime Environment

**Package Manager:** pixi
- Configuration: `pixi.toml`, `pixi.lock`
- Platform-specific locks for linux-64, osx-arm64, osx-64

**Python Environment:**
- Python 3.9.x (locked)
- Virtual environment managed by pixi

## Core Frameworks & Libraries

### Deep Learning

**PyTorch >=2.8.0**
- Primary deep learning framework
- CUDA support for GPU acceleration
- Model loading and inference

**Transformers 4.30.0 (HuggingFace)**
- DNABERT-S model integration
- DNA sequence tokenization and embedding
- Model: `zhihan1996/DNABERT-S`

**ESM 2.0.0 (Facebook Research/fair-esm)**
- ESM-2 3B protein language model
- Protein sequence embedding
- Downloaded via fair-esm package

### Bioinformatics

**BioPython**
- FASTA file parsing and writing
- Sequence manipulation
- SeqIO for sequence I/O operations

### Scientific Computing

**NumPy**
- Array operations
- Numerical computations

**SciPy**
- Scientific algorithms
- Statistical functions

**scikit-learn**
- Machine learning utilities
- Data preprocessing

### CLI & User Interface

**Click >=8.0.0**
- Command-line interface framework
- Main entry point: `virnucpro/cli/main.py`
- Predict command: `virnucpro/cli/predict.py` (183 lines)

**tqdm**
- Progress bars for long-running operations
- File processing feedback

### Configuration Management

**OmegaConf**
- YAML configuration parsing
- Config file: `config/default_config.yaml`
- Nested configuration structure

**PyYAML**
- YAML file I/O
- Configuration serialization

### Testing

**pytest >=8.4.1**
- Primary testing framework
- Custom markers: `@pytest.mark.gpu`, `@pytest.mark.slow`
- Test fixtures in `tests/conftest.py`

**pytest-mock**
- Mocking support for tests

**unittest.mock**
- Standard library mocking

## Pre-trained Models

**Model Files (at repository root):**
- `300_model.pth` - Model for 300bp chunks
- `500_model.pth` - Model for 500bp chunks

**Model Architecture:**
- Custom PyTorch models saved as `.pth` files
- Loaded via `torch.load()`
- No explicit model versioning

## Configuration Files

**Application Config:**
- `config/default_config.yaml` - Default pipeline configuration
  - Chunk sizes
  - Model paths
  - Feature extraction settings
  - Output formats

**Package Config:**
- `pyproject.toml` - Python package metadata and dependencies
- `pixi.toml` - Pixi environment specification
- `pixi.lock` - Locked dependency versions

**No Environment Variables Required:**
- All configuration via YAML files
- Model paths relative to repository root

## Development Tools

**Code Quality:**
- No explicit linter configuration detected
- No formatter configuration (Black, isort, etc.)

**Version Control:**
- Git
- `.gitignore` present

**Documentation:**
- README.md
- Markdown-based documentation

## Key Dependencies Summary

```
Production Dependencies:
- torch>=2.8.0
- transformers==4.30.0
- fair-esm==2.0.0
- biopython
- click>=8.0.0
- omegaconf
- pyyaml
- numpy
- scipy
- scikit-learn
- tqdm

Development Dependencies:
- pytest>=8.4.1
- pytest-mock
```

## Build System

**No Build Step Required:**
- Pure Python package
- No compilation needed (except for dependencies like PyTorch)

**Installation:**
- Via pixi: `pixi install`
- Package mode: editable install via pip
