# Architecture

**Analysis Date:** 2026-02-06

## Pattern Overview

**Overall:** Sequential pipeline architecture with distinct processing stages

**Key Characteristics:**
- Multi-stage data transformation pipeline (sequence preprocessing → feature extraction → model training → prediction)
- Model-centric design with PyTorch deep learning framework
- Heavy reliance on pre-trained transformer models (DNABERT-S for DNA, ESM2 for proteins)
- Batch processing with file-based data management using PyTorch tensors
- Command-line interface for end-user workflows

## Layers

**Data Input/Sequencing Layer:**
- Purpose: Parse biological sequence files (FASTA format) and prepare raw genomic/proteomic data
- Location: `units.py` (functions: `split_fasta_chunk`, `split_fasta_file`), `download_data.py`
- Contains: FASTA parsing via BioPython, sequence chunking utilities
- Depends on: BioPython library
- Used by: Feature extraction and training pipeline scripts

**Biological Translation Layer:**
- Purpose: Convert DNA sequences to protein sequences using six-frame translation
- Location: `units.py` (functions: `translate_dna`, `reverse_complement`, `identify_seq`)
- Contains: Codon table-based translation, ambiguous base filtering, frame selection
- Depends on: BioPython for sequence handling
- Used by: Training data preparation (`make_train_dataset_300.py`, `make_train_dataset_500.py`) and prediction pipeline

**Feature Extraction Layer:**
- Purpose: Convert biological sequences into high-dimensional numerical embeddings using pre-trained models
- Location: `units.py` (functions: `extract_DNABERT_S`, `extract_esm`, `merge_data`), `features_extract.py`
- Contains:
  - DNABERT-S model integration for DNA sequence embeddings (DNA → 384-dim vectors)
  - ESM2 3B model integration for protein sequence embeddings (protein → 2560-dim vectors)
  - Feature merging that concatenates DNA and protein embeddings (combined: 2944-dim vectors)
- Depends on: Transformers library, fair-esm library, PyTorch
- Used by: Training pipeline and prediction pipeline

**Data Management Layer:**
- Purpose: Handle batched loading and management of large tensor datasets
- Location: `train.py` (class: `FileBatchDataset`), `prediction.py` (class: `PredictDataBatchDataset`)
- Contains: PyTorch Dataset classes for loading pre-computed feature tensors, multi-file batch management
- Depends on: PyTorch DataLoader
- Used by: Training and prediction models

**Model Training Layer:**
- Purpose: Train MLP classifier to distinguish viral from non-viral sequences
- Location: `train.py` (classes: `MLPClassifier`, `EarlyStopping`; functions: `train_model`, `test_model`)
- Contains:
  - MLPClassifier: 2-layer neural network (3328 → 512 → 2 outputs) with batch normalization and dropout
  - Training loop with epoch-based iteration
  - Evaluation metrics (accuracy, precision, recall, F1, AUC-ROC)
  - Early stopping mechanism with model checkpointing
- Depends on: PyTorch, scikit-learn metrics
- Used by: Training pipeline entry point

**Inference/Prediction Layer:**
- Purpose: Apply trained model to predict viral status of new sequences
- Location: `prediction.py` (functions: `predict`, `make_predictdata`; class: `MLPClassifier`)
- Contains:
  - Model loading from checkpoint
  - Batch prediction with probability scores
  - Sequence ID tracking and result aggregation
- Depends on: PyTorch, pandas for results aggregation
- Used by: End-user prediction workflow

**Visualization Layer:**
- Purpose: Generate plots of prediction scores across sequence chunks
- Location: `drew_fig.py`
- Contains: Matplotlib-based visualization of per-chunk predictions
- Depends on: matplotlib
- Used by: Optional visualization of prediction results

**Training Data Preparation Layer:**
- Purpose: Process reference genome files to create training datasets
- Location: `make_train_dataset_300.py`, `make_train_dataset_500.py`
- Contains:
  - Reference sequence parsing from GenBank format
  - ORF (open reading frame) identification and filtering
  - Nucleotide/protein pair extraction for specific chunk sizes
- Depends on: BioPython, `units.py`
- Used by: Training pipeline setup

## Data Flow

**Training Workflow:**

1. **Data Download** (`download_data.py`)
   - Fetches viral and host genome files from NCBI RefSeq FTP
   - Stores compressed gzip files to `data/` directory

2. **Training Data Preparation** (`make_train_dataset_300.py` or `make_train_dataset_500.py`)
   - Decompresses gzip files
   - Parses GenBank (GBFF) files to extract protein translations (refseq_pro_list)
   - Chunks genomic DNA sequences to fixed sizes (300bp or 500bp)
   - Performs six-frame translation on chunks
   - Identifies ORFs (sequences without stop codons) that match reference proteins
   - Outputs paired nucleotide and protein FASTA files

3. **Feature Extraction** (`features_extract.py`)
   - Splits FASTA files into manageable chunks (10,000 sequences per file)
   - Processes chunks in parallel using multiprocessing (8 processes for DNA, 2 for protein)
   - Extracts DNA embeddings using DNABERT-S tokenizer and model
   - Extracts protein embeddings using ESM2 model
   - Merges paired embeddings into combined feature vectors
   - Organizes by category (viral vs. host organism type)
   - Saves merged feature tensors to PyTorch format (.pt files)

4. **Training** (`train.py`)
   - Loads merged feature tensors from all files
   - Splits dataset: 90% training, 10% validation
   - Creates MLP classifier with Xavier weight initialization
   - Trains using SGD optimizer (lr=0.0002, momentum=0.9)
   - Learning rate scheduler (StepLR: decay by 0.85 every 10 epochs)
   - Early stopping based on validation loss (patience=5)
   - Saves final model to `model.pth`

**Prediction Workflow:**

1. **Input Processing** (`prediction.py` → `make_predictdata`)
   - Reads user-provided FASTA file
   - Chunks sequences to expected length (user-specified: 300 or 500)
   - Performs six-frame translation on all chunks

2. **Feature Extraction (Prediction)**
   - Splits sequences into batches (10,000 per file)
   - Extracts DNABERT-S embeddings for nucleotides
   - Extracts ESM2 embeddings for proteins
   - Merges embeddings same as training pipeline
   - Creates PredictDataBatchDataset for batch loading

3. **Model Inference**
   - Loads trained MLPClassifier from checkpoint (model.pth)
   - Runs batches through model (batch_size=256)
   - Generates softmax probabilities for both classes

4. **Result Aggregation**
   - Groups predictions by original sequence ID (truncates chunk identifiers)
   - Takes maximum score across all six-frame translations
   - Determines final classification: viral if score2 >= score1
   - Exports detailed results and highest-score summary

**State Management:**

- **Persistent State**: Trained model weights stored in `model.pth`
- **Intermediate State**: Feature tensors cached in `.pt` files to avoid recomputation
- **Transient State**: In-memory batches during feature extraction and training
- **Output State**: Prediction results written to text files and CSV summaries

## Key Abstractions

**Sequence Translation Unit:**
- Purpose: Abstracts biological sequence translation across six reading frames
- Examples: `units.py:translate_dna()`, `units.py:reverse_complement()`
- Pattern: Pure functions without side effects, handles both forward and reverse strand translation

**Feature Extraction Unit:**
- Purpose: Abstracts transformer model inference for sequence-to-embedding conversion
- Examples: `units.py:extract_DNABERT_S()`, `units.py:extract_esm()`
- Pattern: Model-agnostic extraction with optional pre-loaded models, returns serialized embeddings

**Dataset Abstraction:**
- Purpose: Abstracts multi-file tensor batch loading
- Examples: `train.py:FileBatchDataset`, `prediction.py:PredictDataBatchDataset`
- Pattern: PyTorch Dataset interface wrapping multiple pre-computed tensor files

**Model Classifier:**
- Purpose: Abstracts binary classification logic
- Examples: `train.py:MLPClassifier`, `prediction.py:MLPClassifier`
- Pattern: PyTorch nn.Module with consistent architecture across training and inference

**Early Stopping Callback:**
- Purpose: Abstracts training control and checkpointing logic
- Examples: `train.py:EarlyStopping`
- Pattern: Callable class maintaining state across epochs

## Entry Points

**Training Entry Point:**
- Location: `train.py` (lines 43-220: module-level execution)
- Triggers: `python train.py`
- Responsibilities:
  - Loads all merged feature files from `./data/data_merge/`
  - Organizes files by category (viral vs. host)
  - Splits dataset, creates DataLoader
  - Trains MLPClassifier
  - Saves model checkpoint

**Prediction Entry Point:**
- Location: `prediction.py` (lines 193-197: module-level execution with sys.argv)
- Triggers: `python prediction.py <fasta_file> <expected_length> <model_path>`
- Responsibilities:
  - Parses command-line arguments (input file, sequence length, model checkpoint)
  - Invokes `make_predictdata()` for complete pipeline
  - Generates per-sequence predictions and aggregated results

**Data Preparation Entry Points:**
- Location: `make_train_dataset_300.py` (main function, lines 46-77) and `make_train_dataset_500.py`
- Triggers: `python make_train_dataset_300.py` or `python make_train_dataset_500.py`
- Responsibilities: Process compressed genome files, extract ORFs, write training data

**Feature Extraction Entry Point:**
- Location: `features_extract.py` (module-level execution, lines 35-241)
- Triggers: `python features_extract.py`
- Responsibilities: Batch extract features from prepared sequence files, merge embeddings

**Download Entry Point:**
- Location: `download_data.py` (module-level execution, lines 89-93)
- Triggers: `python download_data.py`
- Responsibilities: Download viral and host genomes from NCBI RefSeq

**Visualization Entry Point:**
- Location: `drew_fig.py` (lines 8-91: module-level execution with sys.argv)
- Triggers: `python drew_fig.py <prediction_results.txt>`
- Responsibilities: Parse prediction output, generate matplotlib visualization

## Error Handling

**Strategy:** Minimal explicit error handling; relies on exceptions propagating up

**Patterns:**
- **File I/O**: No try-catch blocks; missing files cause FileNotFoundError
- **Data Validation**:
  - Ambiguous bases in training data filtered via conditional check (lines 84-86 in `units.py`)
  - Empty sequences skipped silently
  - Mismatched nucleotide/protein pairs skipped with warning (line 314 in `units.py`)
- **Model Training**: Early stopping prevents over-training without explicit exception handling
- **Missing Data**: Feature extraction checks file existence before processing, skips if already computed

## Cross-Cutting Concerns

**Logging:**
- Uses `tqdm` progress bars for batch processing (biological relevance: sequence count visibility)
- Print statements for informational messages (extraction status, timing info)
- Manual logging to file in training (`write_log()` function, train.py line 113-115)

**Validation:**
- Biological sequence validation:
  - Stop codon detection ("*" character filtering)
  - Ambiguous base detection (IUPAC codes N, R, Y, etc.)
- Data shape validation: Tensor dimension matching in feature merge (line 311 in `units.py`)

**Multiprocessing:**
- DNA embedding extraction: 8 parallel processes (features_extract.py line 55)
- Protein embedding extraction: 2 parallel processes (features_extract.py line 69)
- No process-level error handling; failures cause complete pipeline abort

**GPU/Device Management:**
- Auto-detection of CUDA availability (lines in `units.py`, `train.py`, `prediction.py`)
- Model/tensor movement to device before computation
- Batch-level device management in DataLoader

---

*Architecture analysis: 2026-02-06*
