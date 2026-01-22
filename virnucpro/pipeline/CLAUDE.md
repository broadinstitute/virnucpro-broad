# Pipeline Module

Navigation index for viral sequence prediction pipeline components.

| File | What | When |
|------|------|------|
| `models.py` | PyTorch MLP classifier architecture | Loading trained models for prediction |
| `prediction.py` | Main pipeline orchestration for viral sequence identification | Running end-to-end prediction workflow |
| `features.py` | DNABERT-S and ESM-2 feature extraction with batching | Extracting embeddings from nucleotide/protein sequences |
| `parallel.py` | Multiprocessing utilities for multi-GPU feature extraction | Parallelizing DNABERT-S processing across multiple GPUs |
| `predictor.py` | Prediction inference and consensus scoring | Generating predictions from extracted features |
| `README.md` | Parallel processing architecture and design decisions | Understanding multi-GPU parallelization strategy |
