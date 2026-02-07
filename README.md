# VirNucPro: An Identifier for the Identification of Viral Short Sequences Using Six-Frame Translation and Large Language Models

## FastESM2 Migration Setup

**Note:** This repository is undergoing migration from ESM2 3B to FastESM2_650 embeddings. The setup process has been updated to use pixi for dependency management.

### Prerequisites

- NVIDIA GPU with CUDA driver 12.x or later (verify with `nvidia-smi`)
- Linux aarch64 platform
- [pixi](https://prefix.dev/docs/pixi/overview) package manager

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Li-Jing-1997/VirNucPro.git
   cd VirNucPro
   ```

2. Install pixi if not already installed:
   ```bash
   curl -fsSL https://pixi.sh/install.sh | bash
   ```

3. Set up the environment with all dependencies:
   ```bash
   pixi install
   ```
   This installs:
   - PyTorch 2.5.1 with CUDA 12.6 support
   - transformers 4.45.2 (FastESM2 compatible)
   - All required dependencies (einops, networkx, biopython, etc.)

4. Validate the environment:
   ```bash
   pixi run validate
   ```
   **Note:** First run downloads the FastESM2_650 model (~2.5GB) from HuggingFace Hub. The model is cached at `~/.cache/huggingface/hub/` after the first download.

5. Verify all 5 environment checks pass:
   - ENV-01: PyTorch 2.5+ with CUDA
   - ENV-02: fair-esm removed
   - ENV-03: transformers >= 4.30.0
   - ENV-04: FastESM2_650 model loads successfully
   - ENV-05: SDPA benchmark (may show warnings on certain GPU architectures)

### Troubleshooting

**`pixi install` fails with platform error**
- Check that you're on linux-aarch64 platform with `uname -m`
- The pixi.toml is configured for linux-aarch64 only

**CUDA not available (`torch.cuda.is_available()` returns False)**
- Verify NVIDIA driver is installed: `nvidia-smi`
- The CUDA toolkit is provided by conda-forge through pixi, but the NVIDIA driver must be installed at the system level
- Ensure CUDA driver version is 12.x or later

**Model download fails during validation**
- Check internet connection
- The FastESM2_650 model (~2.5GB) is downloaded from HuggingFace Hub on first run
- If download is interrupted, cached files at `~/.cache/huggingface/hub/` may be incomplete - delete the cache directory and retry

**SDPA speedup below threshold or shows slowdown**
- SDPA performance varies by GPU architecture and sequence length
- The validation benchmark uses 500+ residue sequences for maximum benefit
- **Known issue:** NVIDIA GB10 GPU (sm_121 compute capability) is not officially supported by PyTorch 2.5.1, causing SDPA to be slower than manual attention
- If using GB10, the FastESM2 migration will rely on smaller model size (650M vs 3B parameters) for speed improvements, not SDPA optimizations
- Consider using H100, A100, or other officially supported GPUs for full SDPA benefits

**`ModuleNotFoundError: No module named 'esm'` when importing units.py**
- This is expected - fair-esm has been intentionally removed as part of the FastESM2 migration
- The old `extract_esm()` function has been deprecated and will be replaced with `extract_fast_esm()` in Phase 2

**Warnings about "Some weights of FastEsmModel were not initialized"**
- This is expected - the contact prediction head and pooler layers are not used for embedding extraction
- These warnings can be safely ignored for the VirNucPro use case

---

## Legacy Installation (Pre-Migration)

The following conda/pip installation method is deprecated. Use the pixi setup above for FastESM2 migration.

<details>
<summary>Show legacy installation steps</summary>

    git clone https://github.com/Li-Jing-1997/VirNucPro.git

    conda create -n VirNucPro python=3.9
    conda activate VirNucPro

    pip install -r requirements.txt
    pip uninstall triton # this can lead to errors in GPUs other than A100

</details>

---

## Run VirNucPro for prediction
Prepare the sequence file you want to predict, and run:
```
python prediction.py input_sequence.fasta Expected_length model.pth
```
For example:
```
python prediction.py input_sequence.fasta 500 500_model.pth
```
The prediction results will be saved in a folder prefixed with input_sequence and suffixed with _merged. e.g., `input_sequence_merged/prediction_results.txt` and `input_sequence_merged/prediction_results_highestscore.csv`

You can also visualize the prediction results using:
```
python drew_fig.py input_sequence_merged/prediction_results.txt
```

## Model training
You can also train your own model according to your task.
### 1.Download data
You can download the training data from [NCBI](https://ftp.ncbi.nlm.nih.gov/refseq/release/) using download_data.py
```
python download_data.py
```
This will download the data for viral and randomly select enough data from other species for downloading.

### 2.Prepare train dataset
Next, prepare the training dataset according to the expected prediction length, for 300bp:
```
python make_train_dataset_300.py
```
and for 500bp:
```
python make_train_dataset_500.py
```
Nucleic acid and corresponding amino acid sequences will stored with the suffixes identified_nucleotide.fa and identified_protein.fa, respectively.

### 3.Extract features
Extract features unsing DNABERT_S and ESM2_3B by running:
```
python features_extract.py
```
This will identify all files in the data folder with the names identified_nucleotide.fa and identified_protein.fa, extract all sequences that start with "viral," and randomly select an equal number of sequences from other species in proportionate amounts.

### 4.Train Model
Use train.py for training:
```
python train.py
```
The trained model will be saved to `model.pth`.
