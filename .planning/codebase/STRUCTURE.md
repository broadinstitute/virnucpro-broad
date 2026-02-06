# Codebase Structure

**Analysis Date:** 2026-02-06

## Directory Layout

```
VirNucPro/
├── data/                              # Training/prediction data (not committed)
│   ├── viral/                         # Viral genome files
│   ├── vertebrate/                    # Vertebrate host organism data
│   ├── bacteria/                      # Bacterial host organism data
│   ├── fungi/                         # Fungal host organism data
│   ├── archaea/                       # Archaeal host organism data
│   ├── protozoa/                      # Protozoan host organism data
│   ├── plant/                         # Plant host organism data
│   ├── invertebrate/                  # Invertebrate host organism data
│   └── data_merge/                    # Merged feature tensors
│       └── viral.1.1_merged/          # Viral merged features
├── 300_model.pth                      # Pre-trained model for 300bp sequences
├── 500_model.pth                      # Pre-trained model for 500bp sequences
├── units.py                           # Core utility functions (shared module)
├── download_data.py                   # Data download and staging
├── make_train_dataset_300.py          # Training data prep for 300bp
├── make_train_dataset_500.py          # Training data prep for 500bp
├── features_extract.py                # Feature extraction pipeline
├── train.py                           # Model training pipeline
├── prediction.py                      # Inference pipeline
├── drew_fig.py                        # Visualization utility
├── requirements.txt                   # Python dependencies
├── pixi.toml                          # Pixi environment configuration
├── pixi.lock                          # Pixi lock file
├── README.md                          # Project documentation
├── LICENSE                            # License file
└── .planning/codebase/                # GSD codebase documentation
```

## Directory Purposes

**data/:**
- Purpose: Stores all biological sequence data and intermediate processing artifacts
- Contains: Compressed genome files (*.gz), chunked FASTA sequences, feature tensors (*.pt), identified sequences
- Key files: `viral/viral.1.1.genomic.fna.gz`, `data_merge/*/` directories with merged tensors
- Generated: Yes (created by download_data.py, make_train_dataset_*.py, features_extract.py)
- Committed: No (in .gitignore)

**Project Root:**
- Purpose: Contains all executable pipeline scripts and configuration
- Contains: Python modules, model checkpoints, configuration files
- Key files: `units.py` (shared functions), all pipeline entry points

## Key File Locations

**Entry Points:**
- `download_data.py`: Download viral and host genomes from NCBI RefSeq FTP
- `make_train_dataset_300.py`: Prepare training data with 300bp chunks
- `make_train_dataset_500.py`: Prepare training data with 500bp chunks
- `features_extract.py`: Extract DNABERT-S and ESM2 embeddings from sequences
- `train.py`: Train MLPClassifier on merged feature tensors
- `prediction.py`: Predict viral status from user-provided sequences
- `drew_fig.py`: Visualize prediction results

**Configuration:**
- `pixi.toml`: Pixi environment declaration (Python 3.9, pip 25.2)
- `requirements.txt`: Python package dependencies
- `pixi.lock`: Locked dependency versions

**Core Logic:**
- `units.py`: Shared utilities for biological sequence processing
  - Sequence translation and frame manipulation
  - FASTA file chunking and splitting
  - Feature extraction wrappers
  - Feature merging and tensor management

**Models:**
- `300_model.pth`: Trained PyTorch model for 300bp sequence classification
- `500_model.pth`: Trained PyTorch model for 500bp sequence classification

**Documentation:**
- `README.md`: Installation and usage instructions
- `LICENSE`: Project license

## Naming Conventions

**Files:**

- **Pipeline Scripts**: `<action>.py` format
  - `download_data.py`, `features_extract.py`, `train.py`, `prediction.py`
  - Executable directly with `python <script>.py`

- **Dataset Preparation**: `make_train_dataset_<size>.py` format
  - Size parameter indicates chunk length (300bp, 500bp)
  - Can be extended with additional sizes (e.g., `make_train_dataset_250.py`)

- **Visualization**: `drew_fig.py` (contains typo - "drew" instead of "draw")
  - Single visualization utility, named for legacy consistency

- **Generated Data Files**:
  - Nucleotide sequences: `*_identified_nucleotide.fa` (FASTA format)
  - Protein sequences: `*_identified_protein.fa` (FASTA format)
  - DNA embeddings: `*_DNABERT_S.pt` (PyTorch tensor)
  - Protein embeddings: `*_ESM.pt` (PyTorch tensor)
  - Merged embeddings: `*_merged.pt` (PyTorch tensor with label)

- **Output Results**:
  - Raw predictions: `prediction_results.txt` (TSV format)
  - Aggregated predictions: `prediction_results_highestscore.csv` (CSV format)

**Directories:**

- **Data Categories**: `<organism_type>/` format
  - Examples: `viral/`, `bacteria/`, `fungi/`, `vertebrate/`, `plant/`
  - Represents biological classification

- **Processing Intermediate**: `<base>_<stage>/` format
  - Examples: `*_nucleotide/`, `*_protein/`, `*_chunked/`
  - Indicates processing step or data type

- **Merged Features**: `*.1.1_merged/` format (viral specific)
  - Represents merged feature tensor directory for virus category
  - Other organisms use same pattern: `<organism>_merged/`

## Where to Add New Code

**New Feature (e.g., additional preprocessing step):**
- Primary code: Create `<feature_name>.py` in project root
- Utilities: Add functions to `units.py` if reusable across pipelines
- Tests: Create `test_<feature_name>.py` (if testing framework added)
- Update: Modify `README.md` to document new step

**New Model/Classifier:**
- Implementation: Define class in `train.py` or create `models.py`
- Integration: Update `train.py` instantiation and `prediction.py` inference code
- Checkpoint format: Follow existing `.pth` PyTorch model format

**New Data Source:**
- Download integration: Add URLs and logic to `download_data.py`
- Parsing: Add organism-specific parsing to `make_train_dataset_*.py`
- Feature extraction: Extend `features_extract.py` to handle new data layout

**New Visualization:**
- Implementation: Create `<visualization_name>.py` in project root
- Input: Accept similar TSV/CSV format as `drew_fig.py`
- Output: Save to `<input>_<type>.png` format

**Utilities/Helpers:**
- Shared functions: Add to `units.py`
- Sequence operations: Extend `translate_dna()`, `reverse_complement()`
- Feature operations: Extend `merge_data()`, `extract_*()` functions
- File operations: Extend `split_fasta_file()`, `split_fasta_chunk()`

## Special Directories

**data/ (Generated Data):**
- Purpose: Stores all intermediate and final data artifacts
- Generated: Yes (by download_data.py, data prep scripts, feature extraction)
- Committed: No (added to .gitignore)
- Cleanup: Can safely delete and regenerate with pipeline scripts
- Subdirectories contain organism-type-specific sequences and features

**data/data_merge/ (Feature Tensors):**
- Purpose: Stores final merged DNABERT-S + ESM2 feature tensors
- Generated: Yes (by features_extract.py)
- Committed: No (tensor files too large)
- Format: PyTorch .pt files with structure: `{'ids': [...], 'data': tensor, 'labels': [0/1]}`
- Critical: Input to `train.py`, must exist before training

**.planning/codebase/ (GSD Documentation):**
- Purpose: Codebase analysis documents for Claude Code
- Generated: Yes (by /gsd:map-codebase command)
- Committed: Yes (part of project planning)
- Contains: ARCHITECTURE.md, STRUCTURE.md, CONVENTIONS.md, TESTING.md, CONCERNS.md

## File Organization Patterns

**By Processing Stage:**
1. Data ingestion: `download_data.py`
2. Data preparation: `make_train_dataset_*.py`
3. Feature extraction: `features_extract.py` + `units.py` helpers
4. Model training: `train.py`
5. Inference: `prediction.py`
6. Visualization: `drew_fig.py`

**By Responsibility:**
- **Executable Scripts** (entry points): Top-level `*.py` files
- **Shared Library**: `units.py` (imported by multiple scripts)
- **Configuration**: `pixi.toml`, `requirements.txt`
- **Data**: `data/` directory (organized by organism type and processing stage)
- **Models**: `*.pth` checkpoints at root level

**By Temporal Lifecycle:**
- **One-time Setup**: `download_data.py`, initial `make_train_dataset_*.py` run
- **Repeated Processing**: `features_extract.py` (can be resumed, checks file existence)
- **Model Development**: `train.py` (creates/overwrites `model.pth`)
- **User-Facing**: `prediction.py`, `drew_fig.py` (uses pre-trained models)

## Data Flow Through Directory Structure

```
Raw Data (NCBI FTP)
    ↓
download_data.py → data/*.gz
    ↓
make_train_dataset_*.py → data/*/identified_*.fa
    ↓
features_extract.py
    ├→ units.py:extract_DNABERT_S() → data/*/_DNABERT_S.pt
    ├→ units.py:extract_esm() → data/*/_ESM.pt
    └→ units.py:merge_data() → data/data_merge/*_merged.pt
    ↓
train.py loads from data/data_merge/ → trains → writes model.pth
    ↓
[User provides new FASTA]
    ↓
prediction.py
    ├→ identifies sequences → temporary nucleotide/protein files
    ├→ features_extract → temporary _DNABERT_S.pt, _ESM.pt
    ├→ merge_data → temporary _merged.pt
    ├→ loads model.pth → inference → prediction_results.txt
    └→ aggregates → prediction_results_highestscore.csv
    ↓
drew_fig.py reads prediction_results.txt → writes prediction.png
```

---

*Structure analysis: 2026-02-06*
