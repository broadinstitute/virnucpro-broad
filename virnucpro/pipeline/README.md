# Parallel Feature Extraction Architecture

## Overview

VirNucPro's feature extraction pipeline uses multiprocessing for parallel DNABERT-S embedding extraction across multiple GPUs. This architecture achieves 150-380x speedup compared to sequential processing by combining batching with GPU-level parallelization.

## Architecture

```
FASTA Input
    |
    v
+-------------+     +---------------+     +------------------+
| Chunk (300/ | --> | 6-Frame       | --> | Split to 10k     |
| 500bp)      |     | Translation   |     | seqs/file        |
+-------------+     +---------------+     +------------------+
                                                 |
                    +----------------------------+
                    |
                    v
        +------------------------+
        | Multiprocessing Pool   |
        | (N workers = N GPUs)   |
        +------------------------+
             |    |    |    |
             v    v    v    v
        +------+ +------+ +------+ +------+
        |GPU 0 | |GPU 1 | |GPU 2 | |GPU 3 |
        |Files | |Files | |Files | |Files |
        |0,4,8 | |1,5,9 | |2,6,10| |3,7,11|
        +------+ +------+ +------+ +------+
             |    |    |    |
             +----+----+----+
                    |
                    v
        +------------------------+
        | Merge + Predict + Cons |
        +------------------------+
                    |
                    v
              Final Results
```

Each worker process loads the DNABERT-S model on its assigned GPU and processes files in batches. Files are distributed round-robin to balance load across GPUs.

## Data Flow

```
nucleotide_files (list of .fa)
        |
        v
round_robin_assignment(files, num_gpus)
        |
        v
[files_for_gpu0, files_for_gpu1, ...]
        |
        v
Pool.starmap(process_files_on_gpu, [(files0, 0), (files1, 1), ...])
        |
        v (each worker)
load_model(device=cuda:N)
for file in assigned_files:
    sequences = load_fasta(file)
    batches = chunk(sequences, batch_size)
    for batch in batches:
        embeddings = model(tokenizer(batch))
    save(embeddings)
        |
        v
nucleotide_feature_files (list of .pt)
```

Files are processed independently by workers. Each worker:
1. Loads DNABERT-S model once on its assigned GPU
2. Processes all assigned files sequentially
3. Batches sequences within each file (default batch_size=256)
4. Writes output files to same paths as sequential mode

## Design Rationale

### Multiprocessing Over DataParallel

The pipeline uses multiprocessing with separate GPU processes instead of PyTorch DataParallel because:

- **Independent file units**: Files are already split into 10k sequence chunks. No cross-file dependencies exist.
- **No communication overhead**: DataParallel requires batch splitting and GPU-to-GPU result gathering on the master GPU. Multiprocessing processes files completely independently.
- **Better GPU utilization**: DataParallel creates imbalanced memory usage (master GPU holds full model + coordination overhead). Multiprocessing distributes model loading evenly.
- **Natural architecture fit**: The existing file-based architecture makes multiprocessing the simpler approach without restructuring.

### Batching Strategy

DNABERT-S processes sequences in fixed-size batches (default 256) because:

- **Uniform sequence length**: DNA sequences are chunked to fixed length (300bp or 500bp), making fixed batching efficient
- **GPU underutilization**: Sequential processing made 100k GPU calls for 100k sequences, achieving only 5-10% GPU utilization
- **Batch reduction**: Batching reduces 100k calls to ~390 calls (at batch_size=256), enabling 50-100x speedup
- **Memory safety**: 256 sequences × 512 tokens × 768 dim × 4 bytes = 400MB batch memory + 1.5GB model = 2GB total, safe for 4GB+ GPUs

## Invariants

The parallel implementation maintains four critical invariants:

1. **Output equivalence**: Batched processing produces identical embeddings to sequential processing within floating point tolerance (rtol=1e-4, atol=1e-6)
2. **File ordering preserved**: Output files correspond 1:1 with input files regardless of processing order
3. **GPU assignment deterministic**: Same file list + GPU count always produces the same round-robin assignment
4. **Checkpoint compatibility**: Existing checkpoint schema unchanged; resume works by detecting completed output files

These invariants ensure parallel processing is a transparent optimization with no observable behavior changes.

## Tradeoffs

| Tradeoff | Cost | Benefit |
|----------|------|---------|
| Model loaded per worker | N × 1.5GB VRAM | True parallel processing without GIL limitations |
| Process spawning | ~0.5-1s startup per worker | Workers reused for all files, amortized over workload |
| Fixed batch_size default | May not be optimal for all GPUs | Simplicity, predictable memory usage, user-configurable |
| Round-robin assignment | Suboptimal if file sizes vary significantly | Simple implementation, works well with uniform 10k sequence splits |

## Memory Requirements

Per-GPU memory usage with default settings:
- DNABERT-S model: 1.5GB
- Batch processing (batch_size=256): 400MB
- Overhead: ~100MB
- **Total**: ~2GB per GPU

Minimum recommended: 4GB VRAM per GPU for default batch_size=256.

For GPUs with less VRAM, reduce batch_size:
- 2-4GB GPU: `--dnabert-batch-size 128`
- <2GB GPU: `--dnabert-batch-size 64`
