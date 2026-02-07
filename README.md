# VirNucPro: An Identifier for the Identification of Viral Short Sequences Using Six-Frame Translation and Large Language Models

## FastESM2 Migration Setup

**Note:** This repository is undergoing migration from ESM2 3B to FastESM2_650 embeddings. The setup uses Docker with NVIDIA PyTorch container for optimal GPU support, including native GB10 (sm_121) compatibility.

### Prerequisites

- **Docker** and **Docker Compose** installed
- **NVIDIA Container Toolkit** ([installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html))
- **NVIDIA GPU** with driver installed (verify with `nvidia-smi`)
- **Internet connection** for model download (~2.5GB on first run)

### Quick Start with Docker Compose

1. Clone the repository:
   ```bash
   git clone https://github.com/Li-Jing-1997/VirNucPro.git
   cd VirNucPro
   ```

2. Build and run environment validation:
   ```bash
   docker-compose build
   docker-compose run --rm virnucpro
   ```

   This will:
   - Build the Docker image with PyTorch 2.9.0a0+50eac811a6.nv25.09 and CUDA 13.0
   - Download FastESM2_650 model (~2.5GB) from HuggingFace Hub on first run
   - Run validation script to verify all 5 environment requirements
   - Cache the model in `.cache/huggingface/` for future use

3. Verify all 5 environment checks pass:
   - ENV-01: PyTorch 2.5+ with CUDA
   - ENV-02: fair-esm removed
   - ENV-03: transformers >= 4.30.0
   - ENV-04: FastESM2_650 model loads successfully
   - ENV-05: SDPA benchmark (1.29x speedup on GB10 with NVIDIA container)

### Alternative: Direct Docker Usage

If you prefer direct Docker commands without docker-compose:

```bash
# Build the image
docker build -t virnucpro:latest .

# Run validation
docker run --gpus all --rm -v $(pwd):/workspace virnucpro:latest python scripts/validate_environment.py

# Interactive shell for development
docker run --gpus all --rm -it -v $(pwd):/workspace virnucpro:latest bash
```

### Running VirNucPro Commands in Docker

All VirNucPro commands should be run inside the Docker container:

```bash
# Using docker-compose
docker-compose run --rm virnucpro python prediction.py input.fasta 500 model.pth

# Using direct Docker
docker run --gpus all --rm -v $(pwd):/workspace virnucpro:latest python prediction.py input.fasta 500 model.pth
```

### Why Docker for GB10 GPUs?

The NVIDIA GB10 GPU uses sm_121 compute capability, which is **not officially supported** by standard PyTorch 2.5.1 builds. This causes SDPA optimizations to perform poorly (0.55x slowdown instead of expected 2x speedup).

**Solution:** NVIDIA's PyTorch container (25.09-py3) includes:
- PyTorch 2.9.0a0 with native GB10 support
- CUDA 13.0 with sm_121 kernels
- Optimized cuDNN and NCCL libraries
- Result: **1.29x SDPA speedup** on GB10 hardware

This Docker-based setup ensures consistent performance across all NVIDIA GPU architectures.

### Troubleshooting

**`docker-compose` command not found**
- Install Docker Compose: `sudo apt-get install docker-compose-plugin` (Ubuntu/Debian)
- Or use Docker Compose V2: `docker compose` (built into Docker 20.10+)

**`docker: Error response from daemon: could not select device driver "" with capabilities: [[gpu]]`**
- NVIDIA Container Toolkit is not installed or not configured properly
- Follow the [installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- After installation, restart Docker: `sudo systemctl restart docker`

**CUDA not available inside container**
- Verify GPU is accessible on host: `nvidia-smi`
- Check Docker can see GPU: `docker run --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi`
- Ensure `--gpus all` flag is present in docker run command

**Model download fails during validation**
- Check internet connection
- The FastESM2_650 model (~2.5GB) is downloaded from HuggingFace Hub on first run
- Model is cached in `.cache/huggingface/` directory (mounted as volume)
- If download is interrupted, delete `.cache/huggingface/` and retry

**Container build is slow (>10 minutes)**
- First build downloads NVIDIA PyTorch base image (~8GB) - this is cached for future builds
- Subsequent builds are much faster due to Docker layer caching
- Use `docker-compose build --parallel` to speed up multi-service builds

**`ModuleNotFoundError: No module named 'esm'` when importing units.py**
- This is expected - fair-esm has been intentionally removed as part of the FastESM2 migration
- The old `extract_esm()` function has been deprecated and will be replaced with `extract_fast_esm()` in Phase 2

**Warnings about "Some weights of FastEsmModel were not initialized"**
- This is expected - the contact prediction head and pooler layers are not used for embedding extraction
- These warnings can be safely ignored for the VirNucPro use case

---

## Alternative: Pixi Setup (Deprecated for GB10)

**Note:** The pixi-based setup is deprecated in favor of Docker due to GB10 GPU compatibility issues. PyTorch 2.5.1 from conda-forge does not support GB10's sm_121 compute capability, resulting in poor SDPA performance.

If you're using a different GPU (H100, A100, etc.) and prefer pixi, the configuration is still available in `pixi.toml`. However, Docker is the recommended approach for consistency across all GPU architectures.

<details>
<summary>Show pixi setup (not recommended for GB10)</summary>

Prerequisites:
- NVIDIA GPU with CUDA driver 12.x or later
- Linux aarch64 platform
- [pixi](https://prefix.dev/docs/pixi/overview) package manager

Installation:
```bash
pixi install
pixi run validate
```

Known limitations:
- PyTorch 2.5.1 does not have GB10 (sm_121) support
- SDPA shows 0.55x slowdown instead of speedup on GB10
- Docker with NVIDIA container achieves 1.29x speedup

</details>

---

## Legacy Installation (Original VirNucPro)

The following conda/pip installation method is for the original VirNucPro with ESM2 3B. This is deprecated for the FastESM2 migration.

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
