---
date: 2025-11-10T14:01:46-05:00
researcher: Claude
git_commit: 0f93f2755a7ea1acd3ff1bcf182892f960ef3326
branch: main
repository: VirNucPro
topic: "VirNucPro Codebase Structure and Implementation"
tags: [research, codebase, viral-prediction, machine-learning, bioinformatics]
status: complete
last_updated: 2025-11-10
last_updated_by: Claude
---

# Research: VirNucPro Codebase Structure and Implementation

**Date**: 2025-11-10T14:01:46-05:00
**Researcher**: Claude
**Git Commit**: 0f93f2755a7ea1acd3ff1bcf182892f960ef3326
**Branch**: main
**Repository**: VirNucPro (https://github.com/Li-Jing-1997/VirNucPro.git)

## Research Question

Document the complete structure and implementation of the VirNucPro codebase, including the prediction pipeline, training workflow, data preparation, and current CLI interfaces, to support refactoring into a production-ready tool with proper CLI using the Click library.

## Summary

VirNucPro is a viral nucleotide sequence identifier that uses six-frame translation combined with deep learning models (DNABERT-S for DNA and ESM-2 for proteins) to classify sequences as viral or non-viral. The codebase consists of 8 Python scripts organized in a flat directory structure, using basic `sys.argv` for CLI arguments in only 2 scripts, with the remaining 6 scripts using hardcoded configuration. The complete pipeline involves downloading data from NCBI, chunking sequences to fixed lengths (300bp or 500bp), performing six-frame translation, extracting dual-modality features, and training/using an MLP classifier for binary classification.

## Detailed Findings

### Project Structure

The VirNucPro project uses a flat directory structure with all source files in the root:

- **Location**: `/home/unix/carze/.local/bin/VirNucPro/`
- **Total Python files**: 8 source files
- **Configuration**: No package structure (`__init__.py` files)
- **Dependencies**: Managed via `requirements.txt` (20 packages) and `pixi.toml`
- **Pre-trained models**: Two model files provided (`300_model.pth`, `500_model.pth` - 6.8MB each)

**Core Scripts**:
1. `prediction.py` - Main prediction pipeline (7906 bytes)
2. `units.py` - Shared utility functions (13328 bytes)
3. `train.py` - Model training (7787 bytes)
4. `download_data.py` - Data acquisition from NCBI (3926 bytes)
5. `make_train_dataset_300.py` - 300bp dataset preparation (3627 bytes)
6. `make_train_dataset_500.py` - 500bp dataset preparation (3627 bytes)
7. `features_extract.py` - Feature extraction using transformers (13877 bytes)
8. `drew_fig.py` - Visualization of results (2828 bytes)

### Prediction Pipeline (`prediction.py`)

The prediction pipeline orchestrates a multi-stage process for viral sequence identification:

#### Entry Point and CLI
- **Location**: `prediction.py:193-197`
- **CLI Method**: Basic `sys.argv` parsing (3 required arguments)
- **Arguments**:
  1. `sys.argv[1]`: Input FASTA file path
  2. `sys.argv[2]`: Expected sequence length (e.g., "300" or "500")
  3. `sys.argv[3]`: Path to trained model file
- **Usage**: `python prediction.py input.fasta 500 500_model.pth`

#### Core Classes

**1. MLPClassifier** (`prediction.py:48-71`)
- Neural network for binary classification (virus vs. non-virus)
- **Architecture**:
  - Input → Linear(input_dim, hidden_dim) → BatchNorm → ReLU → Dropout(0.5) → Linear(hidden_dim, 2) → Output
  - Xavier uniform weight initialization (`prediction.py:60-62`)
- **Input**: Concatenated DNABERT-S + ESM-2 embeddings (3328 dimensions)
- **Output**: Logits for 2 classes (converted to probabilities via softmax)

**2. PredictDataBatchDataset** (`prediction.py:20-45`)
- PyTorch Dataset class for loading merged feature tensors
- Loads multiple `.pt` files containing feature dictionaries
- Supports index-based access across multiple files
- Data format: `{'ids': [...], 'data': tensor(N, 3328)}`

**3. predict() Function** (`prediction.py:74-95`)
- Performs batch prediction using trained model
- Applies softmax to convert logits to probabilities (`prediction.py:85`)
- Maps predictions: `0 → 'others'`, `1 → 'virus'` (`prediction.py:88-89`)
- Returns sequence IDs, predictions, and probability scores

#### Main Workflow (`make_predictdata()` - `prediction.py:98-197`)

**Step 1: Setup** (`prediction.py:99-107`)
- Creates output directories for nucleotide and protein sequences
- File naming patterns: `{base}_chunked{length}.fa`, `{base}_identified_nucleotide.fa`, `{base}_identified_protein.faa`

**Step 2: Sequence Chunking** (`prediction.py:110`)
- Calls `split_fasta_chunk()` from `units.py:9-36`
- Splits sequences into chunks of specified length with overlapping regions
- Overlap distributed evenly across chunks to ensure exact chunk size

**Step 3: Six-Frame Translation** (`prediction.py:112-127`)
- Translates each chunk in all 6 reading frames (3 forward + 3 reverse)
- Filters frames without stop codons (`*`)
- Generates paired nucleotide and protein FASTA files
- Frame indicators appended to sequence IDs: F1-F3 (forward), R1-R3 (reverse)

**Step 4: File Splitting** (`prediction.py:129-151`)
- Splits into chunks of 10,000 sequences per file for parallel processing
- Creates separate directories for nucleotide and protein sequences
- Output pattern: `output_{N}.fa` in respective folders

**Step 5: Feature Extraction** (`prediction.py:137, 149`)
- **Nucleotide features** (`prediction.py:137`): Calls `extract_DNABERT_S()` → `*_DNABERT_S.pt` (768-dim embeddings)
- **Protein features** (`prediction.py:149`): Calls `extract_esm()` → `*_ESM.pt` (2560-dim embeddings)

**Step 6: Feature Merging** (`prediction.py:160-164`)
- Concatenates DNABERT-S and ESM-2 features using `merge_data()`
- Output: `*_merged.pt` files containing 3328-dimensional feature vectors

**Step 7: Model Prediction** (`prediction.py:166-181`)
- Creates DataLoader with batch_size=256, num_workers=4
- Loads model: `torch.load(model_path, weights_only=False)` (`prediction.py:168`)
- Runs prediction and saves results to `prediction_results.txt`
- Format: `Sequence_ID\tPrediction\tscore1\tscore2`

**Step 8: Consensus Scoring** (`prediction.py:183-191`)
- Groups predictions by original sequence ID (removes frame indicators)
- Applies `determine_virus()` function: `Is_Virus = (max_score2 >= max_score1)`
- Saves consensus to `prediction_results_highestscore.csv`

### Utility Functions (`units.py`)

The `units.py` module provides 11 core functions used across the pipeline:

#### Sequence Processing Functions

**1. split_fasta_chunk()** (`units.py:9-36`)
- Splits sequences into overlapping chunks of specified size
- Algorithm:
  - Calculates chunks needed: `num_chunks = -(-seq_length // chunk_size)` (ceiling division)
  - Computes total overlap required and distributes across chunks
  - Creates chunks with variable overlap using `move_step` accumulator
- Chunk IDs: `{original_id}_chunk_{N}`

**2. translate_dna()** (`units.py:54-73`)
- Translates DNA in all 6 reading frames using standard genetic code
- Codon table maps 64 codons to amino acids (`units.py:55-64`)
- Returns list of 6 protein sequences (3 forward + 3 reverse complement)

**3. identify_seq()** (`units.py:81-146`)
- Identifies valid protein-coding regions from six-frame translation
- **Training mode** (`istraindata=True`): Filters ambiguous bases, validates against reference proteins
- **Prediction mode** (`istraindata=False`): Only filters stop codons
- Output: List of dictionaries with `{'seqid': ..., 'nucleotide': ..., 'protein': ...}`

**4. reverse_complement()** (`units.py:50-52`)
- Generates reverse complement using translation table and string reversal
- Maps: `A↔T`, `C↔G` (both cases)

#### Feature Extraction Functions

**5. extract_DNABERT_S()** (`units.py:160-201`)
- Extracts DNA sequence embeddings using DNABERT-S transformer
- **Model**: `"zhihan1996/DNABERT-S"` from Hugging Face
- **Process**:
  - Tokenizes sequences
  - Extracts hidden states from model forward pass
  - Mean-pools across sequence dimension (`units.py:191`)
- **Output**: 768-dimensional embeddings per sequence
- **Format**: `{'nucleotide': [ids], 'data': [embeddings]}`

**6. extract_esm()** (`units.py:204-265`)
- Extracts protein sequence embeddings using ESM-2 model
- **Model**: `'esm2_t36_3B_UR50D'` (3 billion parameters, 36 layers)
- **Configuration**:
  - Truncation length: 1024 residues
  - Tokens per batch: 2048
  - Representation layer: 36 (final layer)
- **Process**:
  - Batches sequences by token count
  - Extracts layer 36 representations
  - Mean-pools over sequence positions [1:truncate_len+1]
- **Output**: 2560-dimensional embeddings per sequence
- **Format**: `{'proteins': [ids], 'data': [tensors]}`

**7. merge_data()** (`units.py:290-324`)
- Merges DNABERT-S (768-dim) and ESM-2 (2560-dim) features
- Concatenates features using `torch.cat()` along feature dimension (`units.py:311`)
- **Total dimension**: 3328 (768 + 2560)
- **Labels**: `0` for 'host', `1` for 'viral', `None` for prediction
- **Format**: `{'ids': [...], 'data': tensor(N, 3328), 'labels': [...]}`

#### File Management Functions

**8. split_fasta_file()** (`units.py:267-288`)
- Splits FASTA into multiple files with fixed sequence count
- Output pattern: `output_{N}.fa`
- Used for parallel processing (default: 10,000 sequences per file)

**9. create_refseq_pro_list()** (`units.py:38-48`)
- Extracts protein sequences from GenBank format files
- Identifies CDS features and extracts "translation" qualifier
- Returns dictionary mapping record IDs to protein lists

**10. seq_in_reflist()** (`units.py:75-79`)
- Checks if protein sequence exists as substring in reference list
- Used for training data validation

### Training Pipeline

#### Data Download (`download_data.py`)

**Sources** (`download_data.py:7-11`):
- 8 NCBI RefSeq organism categories: bacteria, archaea, fungi, protozoa, plant, invertebrate, vertebrate_mammalian, vertebrate_other
- Viral data: Fixed URLs for `viral.1.1.genomic.fna.gz` and `viral.1.genomic.gbff.gz`

**Sampling Strategy** (`download_data.py:13-51`):
- **High-sample categories** (plant, invertebrate, vertebrates): 10 GBFF files per category
- **Low-sample categories** (bacteria, archaea, fungi, protozoa): 1 GBFF file per category
- Random seed: 42 for reproducibility

**File Types Downloaded**:
- `.genomic.gbff.gz`: GenBank format with annotations
- `.genomic.fna.gz`: FASTA nucleotide sequences

#### Dataset Preparation (`make_train_dataset_300.py`, `make_train_dataset_500.py`)

**Common Workflow**:
1. Discover `.fna.gz` and `.gbff.gz` files in `./data/`
2. Decompress paired files using gzip
3. Chunk sequences: `split_fasta_chunk()` with size 300 or 500
4. Extract reference proteins from GBFF: `create_refseq_pro_list()`
5. Identify valid ORFs: `identify_seq()` with training mode enabled
   - Filters sequences with ambiguous bases
   - Validates proteins against reference list
   - Six-frame translation filtering
6. Output paired FASTA files: `identified_nucleotide.fa` and `identified_protein.fa`
7. Cleanup intermediate files

**Key Difference**:
- `make_train_dataset_300.py`: `chunk_size = 300` (line 49)
- `make_train_dataset_500.py`: `chunk_size = 500` (line 49)

#### Feature Extraction (`features_extract.py`)

**Pre-loaded Models** (`features_extract.py:9-12`):
- DNABERT-S tokenizer and model
- ESM2-3B model and alphabet

**Viral Data Processing** (`features_extract.py:41-84`):
1. Split nucleotide/protein files into 10,000-sequence chunks
2. Filter for complete files (exactly 10,000 sequences)
3. Extract features in parallel:
   - DNABERT-S: 8 processes
   - ESM-2: 2 processes
4. Merge features with label `'viral'` (encoded as 1)
5. Output to `./data/data_merge/viral.1.1_merged/`

**Host Data Processing** (`features_extract.py:86-241`):
1. Categorize files into 7 organism types
2. Sample balanced data: each category contributes `ceil(len(viral_files)/7)` files
3. Resample to match exact viral count (`features_extract.py:212`)
4. Extract features in parallel (same settings as viral)
5. Merge features with label `'host'` (encoded as 0)
6. Output to category-specific directories

**Parallel Processing**:
- Nucleotide feature extraction: 8 worker processes
- Protein feature extraction: 2 worker processes (GPU memory intensive)

#### Model Training (`train.py`)

**Dataset Class** (`train.py:15-41`):
- `FileBatchDataset`: Loads all `.pt` files from `./data/data_merge/`
- Separates viral and host files based on path containing `'viral.1.1_merged'`
- Stores all data in memory for fast access

**Data Split** (`train.py:64-74`):
- 90% train / 10% test split
- Random shuffle with seed 42
- Batch size: 32
- DataLoader workers: 12

**Model Architecture** (`train.py:80-102`):
- Input: 3328 dimensions (DNABERT-S 768 + ESM-2 2560)
- Hidden: 512 dimensions
- Output: 2 classes (virus vs. host)
- Layers: Linear → BatchNorm → ReLU → Dropout(0.5) → Linear
- Weight initialization: Xavier uniform

**Training Configuration** (`train.py:113-119`):
- Loss: CrossEntropyLoss
- Optimizer: SGD (lr=0.0002, momentum=0.9)
- Scheduler: StepLR (multiply by 0.85 every 10 epochs)
- Max epochs: 200
- Early stopping patience: 5 epochs

**Metrics Tracked** (`train.py:200-205`):
- Accuracy, Precision, Recall, F1 Score, AUROC
- Logged to `MLP_log.txt` per epoch

**Output**: Trained model saved to `model.pth` when early stopping triggers

### Current CLI Interfaces

#### Scripts with CLI Arguments

**1. prediction.py** (`prediction.py:193-195`)
- **Method**: Direct `sys.argv` parsing
- **Arguments**:
  - `sys.argv[1]`: Input FASTA file
  - `sys.argv[2]`: Expected length ("300" or "500")
  - `sys.argv[3]`: Model path (e.g., "500_model.pth")
- **Usage**: `python prediction.py input.fasta 500 500_model.pth`
- **Output**: `{input}_merged/prediction_results.txt` and `prediction_results_highestscore.csv`

**2. drew_fig.py** (`drew_fig.py:8`)
- **Method**: Direct `sys.argv` parsing
- **Arguments**:
  - `sys.argv[1]`: Results file path
- **Usage**: `python drew_fig.py input_merged/prediction_results.txt`
- **Output**: `prediction.png` in same directory

#### Scripts with No CLI Arguments (Hardcoded Configuration)

**1. train.py**
- All configuration hardcoded
- Data dir: `'./data/data_merge/'`
- Model output: `'model.pth'`
- Log file: `'MLP_log.txt'`

**2. download_data.py**
- NCBI URLs hardcoded
- Sampling rates hardcoded
- Output dir: `'data/'`

**3. make_train_dataset_300.py / make_train_dataset_500.py**
- Chunk size hardcoded (300 or 500)
- Data dir: `'./data'`
- File patterns hardcoded

**4. features_extract.py**
- Model names hardcoded
- Sampling logic hardcoded
- Processing pools hardcoded (8 nucleotide, 2 protein)
- Output dirs hardcoded

### No External Configuration
- No JSON, YAML, INI, or other config files
- All behavior controlled through hardcoded values or positional CLI arguments
- No help messages, parameter validation, or error handling for CLI

## Code References

### Key Files and Line Numbers

**Prediction Pipeline**:
- `prediction.py:98-197` - Main prediction workflow (`make_predictdata()`)
- `prediction.py:48-71` - MLP classifier definition
- `prediction.py:74-95` - Prediction function with softmax
- `prediction.py:193-195` - CLI argument parsing

**Utility Functions**:
- `units.py:9-36` - Sequence chunking with overlap
- `units.py:54-73` - Six-frame translation
- `units.py:81-146` - Sequence identification and ORF finding
- `units.py:160-201` - DNABERT-S feature extraction (768-dim)
- `units.py:204-265` - ESM-2 feature extraction (2560-dim)
- `units.py:290-324` - Feature merging (3328-dim output)

**Training Pipeline**:
- `train.py:80-102` - MLP architecture definition
- `train.py:144-173` - Training loop with early stopping
- `train.py:176-218` - Evaluation with metrics
- `download_data.py:7-11` - NCBI source URLs
- `download_data.py:53-87` - Download function with progress
- `features_extract.py:41-84` - Viral data processing
- `features_extract.py:86-241` - Balanced host data sampling

**Dataset Preparation**:
- `make_train_dataset_300.py:49` - Chunk size configuration (300)
- `make_train_dataset_500.py:49` - Chunk size configuration (500)
- `features_extract.py:9-12` - Pre-loaded transformer models

## Architecture Documentation

### Data Flow

```
1. Data Acquisition
   NCBI RefSeq → download_data.py → ./data/*.genomic.{fna,gbff}.gz

2. Dataset Preparation
   Compressed files → make_train_dataset_{300,500}.py →
   Decompress → Chunk → Six-frame translate → Validate →
   identified_{nucleotide,protein}.fa

3. Feature Extraction
   FASTA files → features_extract.py →
   Split (10k seqs) → DNABERT-S (768) + ESM-2 (2560) →
   Merge (3328) → ./data/data_merge/*_merged/*.pt

4. Training
   Merged .pt files → train.py →
   FileBatchDataset → MLPClassifier (3328→512→2) →
   model.pth

5. Prediction
   Input FASTA → prediction.py →
   Chunk → Translate → Extract features → Merge →
   MLP prediction → prediction_results.txt + _highestscore.csv
```

### Model Architecture

```
Input: Concatenated DNABERT-S + ESM-2 embeddings (3328 dimensions)
  ↓
Linear(3328 → 512)
  ↓
BatchNorm1d(512)
  ↓
ReLU
  ↓
Dropout(0.5)
  ↓
Linear(512 → 2)
  ↓
Output: Logits for binary classification (virus / non-virus)
```

### Feature Extraction Architecture

```
DNA Sequence
  ↓
DNABERT-S Tokenizer
  ↓
DNABERT-S Transformer (zhihan1996/DNABERT-S)
  ↓
Hidden States → Mean Pool → 768-dim embedding

Protein Sequence
  ↓
ESM-2 Tokenizer
  ↓
ESM2-3B Transformer (esm2_t36_3B_UR50D)
  ↓
Layer 36 Representations → Mean Pool → 2560-dim embedding

Concatenate → 3328-dim feature vector
```

### Current Limitations for Production Use

1. **CLI Interface**:
   - Only 2 of 8 scripts accept CLI arguments
   - No parameter validation or error messages
   - No help documentation
   - Hardcoded paths and configuration throughout

2. **Error Handling**:
   - Minimal exception handling
   - No graceful failure modes
   - No progress reporting for long operations

3. **Configuration**:
   - No configuration file support
   - Hardcoded model names, paths, batch sizes
   - No way to customize without editing source

4. **Code Organization**:
   - Flat directory structure
   - No package organization
   - All imports assume same directory
   - Utility functions mixed with main logic

5. **Logging**:
   - Minimal logging
   - Training logs to file, but inconsistent across scripts
   - No log levels or structured logging

6. **Flexibility**:
   - Fixed batch sizes, worker counts
   - No option to skip intermediate files
   - Cannot specify output directories
   - Model paths hardcoded

## Open Questions

1. What is the optimal strategy for organizing the code into a Python package structure?
2. Should intermediate files be kept by default or cleaned up (needs CLI flag)?
3. What validation should be performed on input FASTA files before processing?
4. How should the Click CLI handle backwards compatibility with existing usage patterns?
5. Should configuration be JSON, YAML, or TOML format?
6. What logging framework and log levels are most appropriate?
7. Should the tool support streaming large files or does it require full file loading?
8. How should GPU/CPU device selection be exposed to users?
