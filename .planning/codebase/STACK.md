# Technology Stack

**Analysis Date:** 2026-02-06

## Languages

**Primary:**
- Python 3.9 - Primary language for all scripts and core functionality

**Secondary:**
- None

## Runtime

**Environment:**
- Python 3.9.23 (CPython via Pixi/conda-forge)

**Package Manager:**
- pip 25.2+ - Primary package manager
- Pixi 0.1.0 - Environment management tool
- Lockfile: `pixi.lock` (present)

## Frameworks

**Core ML/Deep Learning:**
- PyTorch - Deep learning framework for model training and inference
- transformers 4.30.0 - Hugging Face transformers for DNABERT-S and model loading
- ESM (Fair ESM 2.0.0) - Facebook's Evolutionary Scale Modeling for protein sequence embeddings

**Feature Extraction:**
- DNABERT-S - DNA sequence tokenization and embedding via transformers
- ESM2_3B - Protein structure model for feature extraction

**Data Processing:**
- Biopython - Bioinformatics utilities for sequence parsing and manipulation
- scikit-learn - Machine learning utilities (metrics, data handling)
- pandas - Data manipulation and CSV/TSV handling
- NumPy - Numerical computing

**Text/Sequence Processing:**
- TextAugment - Text augmentation utilities
- TextBlob 0.17.1 - NLP processing

**Utilities:**
- einops - Tensor dimension manipulation
- peft - Parameter-efficient fine-tuning
- omegaconf - Configuration management
- accelerate - Distributed training utilities
- evaluate - Model evaluation tools
- Pillow - Image processing
- SciPy - Scientific computing
- gdown - Google Drive file downloads
- matplotlib - Plotting and visualization
- requests - HTTP library for API calls
- BeautifulSoup - Web scraping

## Key Dependencies

**Critical:**
- torch - Core deep learning framework; required for model training and inference
- transformers 4.30.0 - Required for DNABERT-S tokenizer and AutoModel loading
- fair-esm 2.0.0 - Required for protein sequence feature extraction (ESM2_3B model)
- Biopython - Required for FASTA file parsing and sequence handling

**Infrastructure:**
- scikit-learn - Evaluation metrics (accuracy, precision, recall, F1, ROC-AUC)
- pandas - Output result handling and CSV writing
- NumPy - Tensor/array operations

## Configuration

**Environment:**
- Managed through `pixi.toml` and `pixi.lock`
- Platform constraint: linux-aarch64
- Uses conda-forge channel for dependencies

**Build:**
- No build configuration files (setup.py, pyproject.toml)
- Direct script execution model

**Conda/Pixi Setup:**
```
channels = ["conda-forge"]
name = "VirNucPro"
platforms = ["linux-aarch64"]
version = "0.1.0"

dependencies:
- python = "3.9.*"
- pip = ">=25.2,<26"
```

## Platform Requirements

**Development:**
- Linux aarch64 architecture (required by pixi.lock)
- CUDA-capable GPU highly recommended (checked at runtime with `torch.cuda.is_available()`)
- 8+ GB RAM for training (uses batch_size=32 for training)
- 4+ CPU cores for multiprocessing (8 workers used in feature extraction)

**Production:**
- Local filesystem execution
- GPU optional but significantly recommended for inference speed
- Model files: `300_model.pth` (6.8 MB) and `500_model.pth` (6.8 MB) pre-trained models included

---

*Stack analysis: 2026-02-06*
