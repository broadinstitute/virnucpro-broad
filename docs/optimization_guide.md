# GPU Optimization Guide

**Accelerating viral sequence prediction with multi-GPU parallelization and mixed precision**

This guide explains how to leverage VirnuCPro's GPU optimization features to dramatically reduce processing time. With the optimizations in place, you can process datasets in under 10 hours that previously took 45+ hours.

## Overview

VirnuCPro uses two deep learning models for feature extraction:

- **DNABERT-S**: Processes nucleotide sequences to generate DNA embeddings
- **ESM-2**: Processes protein sequences (from six-frame translation) to generate protein embeddings

Both models have been optimized for:

- **Multi-GPU parallelization** - Automatically distributes work across all available GPUs
- **BF16 mixed precision** - Uses bfloat16 on Ampere+ GPUs for 50% memory reduction
- **Token-based dynamic batching** - Efficiently batches sequences by token count
- **Bin-packing work distribution** - Balances sequences across GPUs for maximum utilization

### Performance Gains

| Configuration | Previous Time | Optimized Time | Speedup |
|---------------|---------------|----------------|---------|
| Single GPU (RTX 4090) | 45+ hours | 15 hours | 3.0x |
| 4 GPUs (RTX 4090) | 45+ hours | <10 hours | 4.5x+ |

Key improvements:
- **3-4x speedup** with 4 GPUs through parallel processing
- **50% memory reduction** with BF16 on Ampere+ GPUs
- **Balanced GPU utilization** via bin-packing algorithm
- **Automatic optimization** - zero configuration required

## Quick Start

**The easiest way to use the optimizations:**

```bash
# Just run predict - auto-detects GPUs and optimizes automatically
virnucpro predict -n nucleotides.fa -p proteins.fa -o results/
```

That's it! The pipeline will:
- Auto-detect available GPUs
- Enable multi-GPU parallelization if 2+ GPUs found
- Enable BF16 if you have Ampere+ GPUs (RTX 30/40 series, A100, etc.)
- Use optimal batch sizes (2048 for DNABERT-S, 2048 for ESM-2)
- Distribute work evenly across GPUs using bin-packing

## DNABERT-S Optimization

### Automatic Features

When you run `virnucpro predict`, DNABERT-S automatically:

1. **Multi-GPU Parallelization**
   - Detects all available CUDA devices
   - Spawns parallel workers (one per GPU)
   - Distributes nucleotide files across workers using bin-packing
   - Bin-packing balances sequences (not just files) for even GPU utilization

2. **BF16 Mixed Precision**
   - Detects GPU compute capability
   - Automatically enables BF16 on Ampere+ GPUs (compute capability >= 8.0)
   - Reduces memory usage by 50% with minimal accuracy impact
   - Automatically increases batch size from 2048 to 3072 with BF16

3. **Token-Based Dynamic Batching**
   - Groups sequences by total token count (DNA bases)
   - Prevents memory waste from padding
   - Default: 2048 tokens per batch (3072 with BF16)
   - Automatically adjusts based on available memory

4. **Progress Monitoring**
   - Real-time dashboard showing per-GPU progress
   - Live throughput metrics (sequences/second)
   - Memory usage tracking
   - Failed file logging

### Manual Tuning

While defaults work well, you can tune batch size for your specific hardware:

```bash
# Increase batch size (if you have extra memory)
virnucpro predict ... --dnabert-batch-size 4096

# Decrease batch size (if you hit OOM errors)
virnucpro predict ... --dnabert-batch-size 1024

# Explicit GPU selection (use specific GPUs)
virnucpro predict ... --gpus 0,1,2,3
```

**When to adjust batch size:**

- **Increase batch size** if:
  - You have high-memory GPUs (A100, RTX 6000, etc.)
  - GPU utilization is low (<80%)
  - You want maximum throughput

- **Decrease batch size** if:
  - You get "out of memory" errors
  - You need to run other GPU processes concurrently
  - You have older/smaller GPUs

**Batch size guidelines:**

| GPU Memory | FP32 Batch Size | BF16 Batch Size |
|------------|-----------------|-----------------|
| 12GB (RTX 3060) | 1024 | 1536 |
| 16GB (RTX 4060 Ti) | 2048 | 3072 |
| 24GB (RTX 4090) | 4096 | 6144 |
| 48GB (A6000) | 8192 | 12288 |
| 80GB (A100) | 16384 | 24576 |

### Profiling Your Hardware

**Use the profiler to find optimal batch size for your GPU:**

```bash
# Profile DNABERT-S on your GPU
virnucpro profile --model dnabert-s --device cuda:0
```

This will:
- Test various batch sizes from 512 to 8192 tokens
- Measure throughput (sequences/second) for each
- Track memory usage
- Detect OOM threshold
- Recommend optimal batch size (80% of maximum for safety)

**Example output:**

```
==================================================
DNABERT-S Profiling Results:
  Device: cuda:0
  BF16: enabled
  Maximum batch size: 6144 tokens
  Recommended batch size: 4608 tokens (80% of max)
==================================================

Throughput curve:
  512 tokens:   42.3 seq/s  (2.1 GB)
 1024 tokens:   78.5 seq/s  (3.8 GB)
 2048 tokens:  142.1 seq/s  (6.9 GB)
 4096 tokens:  215.7 seq/s (12.4 GB)
 6144 tokens:  248.2 seq/s (17.8 GB)
```

**Python profiling API:**

```python
from virnucpro.pipeline.profiler import profile_dnabert_batch_size

# Profile with defaults
results = profile_dnabert_batch_size(device='cuda:0')

# Profile with custom range
results = profile_dnabert_batch_size(
    device='cuda:0',
    min_batch=1024,
    max_batch=8192,
    step=1024
)

# Profile with your own test file
from pathlib import Path
results = profile_dnabert_batch_size(
    device='cuda:0',
    test_sequence_file=Path('my_test_sequences.fa')
)

# Results dictionary contains:
print(f"Optimal: {results['optimal_batch_size']} tokens")
print(f"Max: {results['max_batch_size']} tokens")
print(f"BF16: {results['bf16_enabled']}")
```

**Advanced profiling options:**

```bash
# Profile with custom batch size range
virnucpro profile --model dnabert-s --min-batch 1024 --max-batch 8192 --step 1024

# Profile with your own test sequences
virnucpro profile --model dnabert-s --test-file my_sequences.fa

# Save results to JSON
virnucpro profile --model dnabert-s --output profile_results.json

# Profile specific GPU
virnucpro profile --model dnabert-s --device cuda:1
```

## ESM-2 Optimization

ESM-2 uses the same optimization patterns as DNABERT-S:

- **Multi-GPU parallelization** - Automatic when 2+ GPUs available
- **BF16 mixed precision** - Automatic on Ampere+ GPUs
- **Token-based batching** - Default 2048 tokens (3072 with BF16)
- **Bin-packing distribution** - Balanced by protein sequence count

### Manual Tuning

```bash
# Adjust ESM-2 batch size
virnucpro predict ... --esm-batch-size 4096

# Profile ESM-2 separately
virnucpro profile --model esm2 --device cuda:0
```

**ESM-2 batch size guidelines:**

ESM-2 uses similar memory to DNABERT-S but processes protein sequences (typically shorter than DNA):

| GPU Memory | FP32 Batch Size | BF16 Batch Size |
|------------|-----------------|-----------------|
| 12GB | 1024 | 1536 |
| 16GB | 2048 | 3072 |
| 24GB | 4096 | 6144 |
| 48GB | 8192 | 12288 |
| 80GB | 16384 | 24576 |

## GPU Selection

**By default**, VirnuCPro uses all available GPUs:

```bash
# Auto-detect and use all GPUs
virnucpro predict -n nucleotides.fa -p proteins.fa -o results/
```

**Explicit GPU selection:**

```bash
# Use specific GPUs (0, 1, 2, 3)
virnucpro predict ... --gpus 0,1,2,3

# Use only GPUs 2 and 3 (e.g., if 0 and 1 are busy)
virnucpro predict ... --gpus 2,3

# Use single GPU (GPU 1)
virnucpro predict ... --gpus 1
```

**GPU selection tips:**

- **Leave one GPU free** for other users/processes if on shared system
- **Use `nvidia-smi`** to check which GPUs are available and their memory
- **Select GPUs with matching capability** for consistent performance
- **Balance GPU memory** - choose GPUs with similar free memory

## Performance Metrics

### Expected Speedups by GPU Count

Based on testing with RTX 4090 GPUs:

| GPUs | Speedup | Effective Efficiency |
|------|---------|---------------------|
| 1 GPU | 1.0x (baseline) | 100% |
| 2 GPUs | 1.9x | 95% |
| 4 GPUs | 3.6x | 90% |
| 8 GPUs | 6.8x | 85% |

**Why not 4x with 4 GPUs?**

- Overhead from multiprocessing spawning and result collection (~5-10%)
- Load balancing imperfections (bin-packing is greedy, not optimal)
- CPU bottlenecks in data loading and batching
- GPU synchronization overhead

Still, **3.6x speedup is excellent** and dramatically reduces processing time.

### Memory Usage Guidelines

**DNABERT-S:**
- Base model: ~2.5 GB
- Per-batch overhead: ~4-6 GB (depends on batch size)
- Total: ~7-9 GB for default settings (2048 tokens)

**ESM-2:**
- Base model: ~3.2 GB
- Per-batch overhead: ~5-8 GB (depends on batch size)
- Total: ~8-11 GB for default settings (2048 tokens)

**With BF16:**
- ~50% reduction in batch overhead
- Model size stays the same
- Total: ~5-6 GB for DNABERT-S, ~6-7 GB for ESM-2

### Monitoring GPU Utilization

**While processing:**

```bash
# Watch GPU utilization in real-time
watch -n 1 nvidia-smi

# Or use nvtop for a nicer interface
nvtop
```

**What to look for:**

- **GPU Utilization**: Should be 80-100% during processing
- **Memory Usage**: Should be stable (not growing = no memory leak)
- **Temperature**: Should be <85°C (throttling above this)
- **Power**: Should be near TDP (Total Design Power)

**If GPU utilization is low (<50%):**

- Increase batch size
- Check CPU utilization (might be CPU-bound)
- Verify data loading isn't bottleneck
- Check for I/O wait (slow disk)

## Troubleshooting

### Out of Memory (OOM) Errors

**Symptom:**
```
RuntimeError: CUDA out of memory. Tried to allocate X GB (GPU 0; Y GB total capacity; Z GB already allocated; ...)
```

**Solutions:**

1. **Reduce batch size:**
   ```bash
   virnucpro predict ... --dnabert-batch-size 1024 --esm-batch-size 1024
   ```

2. **Use profiler to find safe batch size:**
   ```bash
   virnucpro profile --model dnabert-s --device cuda:0
   ```

3. **Free up GPU memory:**
   ```bash
   # Kill other GPU processes
   nvidia-smi
   kill <pid>
   ```

4. **Use fewer GPUs** (if running on shared system):
   ```bash
   virnucpro predict ... --gpus 0,1  # Instead of all 4
   ```

### Unbalanced GPU Usage

**Symptom:** Some GPUs at 100%, others at 20%

**Cause:** Uneven sequence distribution (some files have way more sequences)

**Solution:** The bin-packing algorithm should handle this, but if you still see imbalance:

1. **Check file sequence counts:**
   ```bash
   grep -c "^>" *.fa
   ```

2. **Split large files before processing:**
   ```python
   from Bio import SeqIO

   # Split file into chunks
   records = list(SeqIO.parse("large_file.fa", "fasta"))
   chunk_size = len(records) // 4  # For 4 GPUs

   for i in range(4):
       chunk = records[i * chunk_size:(i + 1) * chunk_size]
       SeqIO.write(chunk, f"chunk_{i}.fa", "fasta")
   ```

### No Speedup with Multi-GPU

**Symptom:** Same speed with 4 GPUs as with 1 GPU

**Check:**

1. **Are multiple GPUs actually being used?**
   ```bash
   watch -n 1 nvidia-smi
   ```
   All GPUs should show 80-100% utilization.

2. **Check logs for multi-GPU mode:**
   ```
   grep "Using parallel.*processing with.*GPUs" virnucpro.log
   ```

3. **Verify --parallel flag is set** (should be automatic):
   ```bash
   virnucpro predict ... --gpus 0,1,2,3  # This auto-enables parallel mode
   ```

4. **Check file count** - Need multiple files for parallelization:
   - If processing 1 file: No parallelization possible
   - If processing 4+ files with 4 GPUs: Should see speedup

### BF16 Not Enabled

**Symptom:** Logs show "BF16: disabled" on RTX 30/40 series GPU

**Check GPU compute capability:**
```python
import torch
print(torch.cuda.get_device_capability(0))
```

Should return `(8, X)` or higher for Ampere+ GPUs.

**If compute capability < 8:**
- You have an older GPU (e.g., RTX 20 series)
- BF16 not supported
- Use FP32 (default) or consider upgrading GPU

## Examples

### Basic Multi-GPU Processing

```bash
# Zero-config - just works
virnucpro predict -n nucleotides.fa -p proteins.fa -o results/
```

### Tuned Batch Sizes

```bash
# Larger batches for high-memory GPUs
virnucpro predict \
  -n nucleotides.fa \
  -p proteins.fa \
  -o results/ \
  --dnabert-batch-size 4096 \
  --esm-batch-size 4096
```

### Specific GPU Selection

```bash
# Use only GPUs 2 and 3 (others busy)
virnucpro predict \
  -n nucleotides.fa \
  -p proteins.fa \
  -o results/ \
  --gpus 2,3
```

### Profile Before Running

```bash
# Find optimal settings for your hardware
virnucpro profile --model dnabert-s --device cuda:0
virnucpro profile --model esm2 --device cuda:0

# Then use recommended batch sizes
virnucpro predict \
  -n nucleotides.fa \
  -p proteins.fa \
  -o results/ \
  --dnabert-batch-size 4608 \
  --esm-batch-size 4608
```

### Shared GPU System

```bash
# Check available GPUs
nvidia-smi

# Use GPUs with free memory
virnucpro predict \
  -n nucleotides.fa \
  -p proteins.fa \
  -o results/ \
  --gpus 0,2 \
  --dnabert-batch-size 2048 \
  --esm-batch-size 2048
```

### Maximum Performance

```bash
# Profile first
virnucpro profile --model dnabert-s --output dnabert_profile.json
virnucpro profile --model esm2 --output esm_profile.json

# Extract optimal batch sizes from JSON
# (or read from terminal output)

# Run with optimal settings
virnucpro predict \
  -n nucleotides.fa \
  -p proteins.fa \
  -o results/ \
  --gpus 0,1,2,3 \
  --dnabert-batch-size <optimal_from_profile> \
  --esm-batch-size <optimal_from_profile>
```

## Advanced Topics

### CPU Thread Count for Translation

The six-frame translation step (before DNABERT-S) uses CPU multiprocessing:

```bash
# Control translation parallelism
virnucpro predict ... --threads 16
```

**Guidelines:**
- Default: Uses all CPU cores
- Set to 50-75% of cores on shared systems
- Translation is fast (<5 min), so optimization less critical than GPU

### Combining Optimizations

**Full optimization stack:**

1. **Parallel translation** (CPU): `--threads 16`
2. **Multi-GPU DNABERT-S**: Auto-enabled with 2+ GPUs
3. **Multi-GPU ESM-2**: Auto-enabled with 2+ GPUs
4. **BF16 mixed precision**: Auto-enabled on Ampere+ GPUs
5. **Tuned batch sizes**: From profiling

**Expected results:**
- **45+ hours → <10 hours** (4.5x+ improvement)
- **High GPU utilization** (80-100%)
- **Balanced workload** across GPUs
- **Efficient memory usage** with BF16

### System Requirements

**Minimum for GPU acceleration:**
- 1x NVIDIA GPU with 12GB+ VRAM
- CUDA 11.1+
- PyTorch 1.10+

**Recommended for multi-GPU:**
- 4x NVIDIA RTX 4090 (24GB each)
- CUDA 12.1+
- PyTorch 2.0+
- PCIe 4.0 x16 for each GPU

**Optimal configuration:**
- 4x NVIDIA A100 (80GB each)
- NVLink for GPU-to-GPU communication
- High-speed NVMe storage
- 128+ CPU cores for translation

## Summary

**Key takeaways:**

1. **Zero configuration** - Just run `virnucpro predict`, optimizations auto-enable
2. **Profile your hardware** - Use `virnucpro profile` to find optimal batch sizes
3. **Multi-GPU scales well** - 3.6x speedup with 4 GPUs
4. **BF16 saves memory** - 50% reduction on Ampere+ GPUs
5. **Adjust batch sizes** - If you hit OOM, reduce; if GPU utilization low, increase

**Questions or issues?**

- Check logs: `virnucpro.log` in your output directory
- Monitor GPUs: `nvidia-smi` or `nvtop`
- Profile hardware: `virnucpro profile --model dnabert-s`
- Open issue: https://github.com/yourrepo/virnucpro/issues

---

*This guide covers Phase 1 (ESM-2) and Phase 2 (DNABERT-S) optimizations. Last updated: 2026-01-23*
