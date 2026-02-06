# External Integrations

**Analysis Date:** 2026-02-06

## APIs & External Services

**NCBI RefSeq Data:**
- NCBI FTP servers (ftp.ncbi.nlm.nih.gov)
- What it's used for: Downloading genomic data for viral and host organisms (bacteria, archaea, fungi, protozoa, plant, invertebrate, vertebrate)
  - SDK/Client: `requests` library + `BeautifulSoup` for HTML parsing
  - URLs: Multiple FTP endpoints at `https://ftp.ncbi.nlm.nih.gov/refseq/release/` for different organism categories
  - Location: `download_data.py` (lines 7-11, 18-51)

**Hugging Face Model Hub:**
- DNABERT-S model repository
- What it's used for: Loading pre-trained DNA sequence tokenizer and embedding model
  - SDK/Client: `transformers.AutoTokenizer` and `transformers.AutoModel`
  - Model ID: `"zhihan1996/DNABERT-S"`
  - Trust remote code: Required (`trust_remote_code=True`)
  - Location: `features_extract.py` (lines 9-10), `units.py` (lines 164-165), `prediction.py` (not directly loaded)

**Facebook ESM (Evolutionary Scale Modeling):**
- ESM2_3B protein structure model
- What it's used for: Extracting protein sequence embeddings from translated DNA sequences
  - SDK/Client: `esm.pretrained.load_model_and_alphabet()`
  - Model location: `'esm2_t36_3B_UR50D'` (hardcoded)
  - Loaded via: `esm` package (fair-esm 2.0.0)
  - Location: `features_extract.py` (line 12), `units.py` (lines 213-214)

## Data Storage

**Databases:**
- None used - project uses local file system only

**File Storage:**
- Local filesystem only
  - Training data: `./data/data_merge/` directory structure with subdirectories like `viral.1.1_merged/`
  - Model checkpoints: `*.pth` files (PyTorch saved models)
  - Intermediate features: `*.pt` files (PyTorch tensors for DNA-BERT and ESM embeddings)
  - FASTA sequences: `*.fa` and `*.faa` files for nucleotide and protein sequences

**Caching:**
- None implemented
- Models loaded fresh on each run (not cached)
- Feature extraction can skip existing `*.pt` files based on path existence checks (`os.path.exists()`)

## Authentication & Identity

**Auth Provider:**
- None - NCBI FTP accessed without authentication
- Hugging Face model downloads use public endpoints (no token required for DNABERT-S)
- ESM2 model downloads from public repository (no token required)

## Monitoring & Observability

**Error Tracking:**
- None detected

**Logs:**
- Custom file logging implemented in `train.py`
  - Log file: `MLP_log.txt` (line 144)
  - Format: Training loss per epoch, validation metrics (loss, accuracy, precision, recall, F1, AUROC)
  - Location: `train.py` (lines 113-116, write_log function)
- Console output via `print()` statements and `tqdm` progress bars

**Warnings:**
- Warning messages for data merging mismatches in `units.py` line 314
- Warning messages for unexpected chunk formats in `drew_fig.py` line 63

## CI/CD & Deployment

**Hosting:**
- Local execution only - no cloud platform integration
- Desktop/server-based analysis tool

**CI Pipeline:**
- None detected

**Version Control:**
- Git repository present (`.git/` directory)
- No GitHub Actions or CI service configuration files

## Environment Configuration

**Required env vars:**
- None explicitly required
- All configuration is through command-line arguments or hardcoded paths

**Configuration via Command-Line Arguments:**

From `prediction.py` (lines 193-195):
```python
predict_fasta_file = sys.argv[1]      # Input FASTA file path
expect_length = sys.argv[2]            # Expected sequence length (300 or 500)
model_path = sys.argv[3]               # Path to trained model (e.g., '500_model.pth')
```

From `drew_fig.py` (line 8):
```python
infile = sys.argv[1]                   # Path to prediction results file
```

**Hardcoded Configuration:**
- NCBI FTP URLs for data download
- Model repository IDs (DNABERT-S, ESM2_3B)
- Data paths: `./data/` structure
- Model parameters: Input dim=3328, hidden dim=512, num_class=2 (in `train.py` and `prediction.py`)
- Batch sizes: 32 (training), 256 (prediction)
- Sequence chunking size: 300bp or 500bp (configurable via script variant or argument)
- Sequences per file: 10,000 (hardcoded in multiple scripts)

**Secrets location:**
- No secrets management implemented
- No `.env` file detected
- All credentials/tokens: None required

## Webhooks & Callbacks

**Incoming:**
- None

**Outgoing:**
- None

## Data Pipeline

**Sequence Download & Processing:**
1. `download_data.py` - Downloads viral and host genomic sequences from NCBI RefSeq FTP
   - Dependencies: requests, BeautifulSoup
   - Output: `.fna.gz` and `.gbff.gz` files in `./data/`

2. `make_train_dataset_300.py` or `make_train_dataset_500.py` - Chunks sequences and identifies coding regions
   - Dependencies: Biopython, units.py
   - Input: Compressed FASTA and GenBank files
   - Output: `identified_nucleotide.fa` and `identified_protein.fa` files

3. `features_extract.py` - Extracts embeddings using DNABERT-S and ESM2_3B
   - Dependencies: transformers, esm, torch
   - External models: DNABERT-S (Hugging Face), ESM2_3B (Fair ESM)
   - Output: `*.pt` PyTorch tensor files with embeddings
   - Multiprocessing: Uses 8 processes for DNA-BERT, 2 processes for ESM

4. `train.py` - Trains MLP classifier
   - Dependencies: torch, scikit-learn
   - Output: `model.pth` (or terminates on early stopping)
   - Metrics: Logged to `MLP_log.txt`

**Inference Pipeline:**
1. `prediction.py` - Predicts on new sequences
   - Input: FASTA file, expected length, model path
   - Processing: Chunks, features extraction (DNABERT-S + ESM2_3B), classification
   - Output: `prediction_results.txt` and `prediction_results_highestscore.csv`

2. `drew_fig.py` - Visualization of predictions
   - Input: `prediction_results.txt`
   - Output: `prediction.png`

## Model Loading & Inference

**Pre-trained Models:**
- `300_model.pth` - MLP classifier trained for 300bp sequences (included)
- `500_model.pth` - MLP classifier trained for 500bp sequences (included)
- Location: `./` (project root)
- Loaded via: `torch.load(model_path, weights_only=False)` in `prediction.py` (line 168)

**Feature Models (External):**
- DNABERT-S: Loaded from Hugging Face on first use
- ESM2_3B: Loaded from Fair ESM on first use
- Both support pre-loading for performance optimization in `features_extract.py` (lines 9-12)

---

*Integration audit: 2026-02-06*
