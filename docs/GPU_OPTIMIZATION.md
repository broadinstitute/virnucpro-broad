# GPU Optimization Guide

## Overview

VirNucPro now supports automatic multi-GPU parallelization for ESM-2 protein feature extraction, reducing processing time from 45+ hours to under 10 hours for a typical sample.

This guide explains how to use the multi-GPU features, understand performance expectations, and troubleshoot common issues.

## Features

- **Automatic GPU detection**: Uses all available GPUs by default
- **BF16 mixed precision**: 2x memory reduction on compatible GPUs (Ampere and newer: RTX 30xx/40xx, A100, H100)
- **Load balancing**: Files distributed round-robin across GPUs for even workload
- **Progress monitoring**: Live dashboard shows per-GPU status and completion
- **Failure recovery**: Failed files logged to `failed_files.txt`, partial results preserved
- **Backward compatibility**: Existing checkpoints and output formats unchanged

## Usage

### Basic Usage (Auto-detect GPUs)

By default, VirNucPro automatically detects and uses all available CUDA GPUs:

```bash
virnucpro predict --input sequences.fa --output-dir results/
```

### Specify GPUs

Control which GPUs to use with the `--gpus` flag:

```bash
# Use specific GPUs (0, 1, and 2)
virnucpro predict --input sequences.fa --output-dir results/ --gpus 0,1,2

# Use single GPU (fallback mode - useful for debugging)
virnucpro predict --input sequences.fa --output-dir results/ --gpus 0

# Use GPUs 2 and 3 only (if you have other processes on GPUs 0 and 1)
virnucpro predict --input sequences.fa --output-dir results/ --gpus 2,3
```

### Adjust Batch Size

If you encounter out-of-memory (OOM) errors, reduce the batch size:

```bash
# Reduce ESM-2 batch size (default is 2048 tokens per batch)
virnucpro predict --input sequences.fa --output-dir results/ --esm-batch-size 1024

# For severe memory constraints
virnucpro predict --input sequences.fa --output-dir results/ --esm-batch-size 512
```

### Quiet Mode

Disable the progress dashboard for cleaner log output:

```bash
virnucpro predict --input sequences.fa --output-dir results/ --quiet
```

### Resume from Checkpoint

If a run is interrupted, resume from the last checkpoint:

```bash
virnucpro predict --input sequences.fa --output-dir results/ --resume
```

## Performance

### Expected Speedup

Multi-GPU parallelization provides near-linear speedup for ESM-2 feature extraction:

| GPUs | Expected Speedup | Example Time (45hr → ?) |
|------|------------------|-------------------------|
| 1    | 1.0x (baseline)  | 45 hours                |
| 2    | ~1.8-2.0x        | ~23 hours               |
| 4    | ~3.5-4.0x        | ~11 hours               |
| 8    | ~7.0-8.0x        | ~6 hours                |

### Factors Affecting Performance

Actual speedup depends on several factors:

1. **File size distribution**: More uniform file sizes → better load balancing
2. **GPU memory**: Larger memory allows bigger batches → better throughput
3. **GPU compute capability**: Newer GPUs (Ampere/Hopper) have faster BF16 operations
4. **I/O bandwidth**: Faster storage (NVMe SSD) reduces data loading overhead
5. **Number of files**: Very few large files may limit parallelization benefit

### Performance Tips

- **Sort sequences by length**: Helps batch packing and memory efficiency
- **Use BF16-capable GPUs**: Ampere (RTX 30xx, A100) or newer for 2x memory savings
- **Monitor GPU utilization**: Use `nvidia-smi dmon` to check GPU usage
- **Balance file count**: Split very large files into multiple smaller ones for better distribution

## Hardware Requirements

### Minimum Requirements

- Single NVIDIA GPU with 8GB+ VRAM
- CUDA 11.0 or newer
- PyTorch 2.0 or newer
- Python 3.9 or newer (uses `spawn` multiprocessing by default)

### Recommended Configuration

- 4+ NVIDIA GPUs with 16GB+ VRAM each
- Ampere architecture or newer (RTX 30xx, A100, H100) for BF16 support
- NVMe SSD storage for faster I/O
- 64GB+ system RAM for large datasets

## Troubleshooting

### Out of Memory Errors

**Symptoms:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**

1. **Reduce batch size**: Use `--esm-batch-size 1024` or lower
2. **Use fewer GPUs**: More GPUs = more memory per GPU available
3. **Check GPU memory usage**: Ensure no other processes are using GPU memory
4. **Clear CUDA cache**: Add this to your Python scripts:
   ```python
   import torch
   torch.cuda.empty_cache()
   ```

### CUDA Initialization Errors

**Symptoms:**
```
RuntimeError: Cannot re-initialize CUDA in forked subprocess
```

**Cause:** Python 3.8 or older using `fork` multiprocessing.

**Solutions:**

1. **Upgrade Python**: Use Python 3.9+ which defaults to `spawn` multiprocessing
2. **Set multiprocessing method**: Add to the top of your script:
   ```python
   import multiprocessing
   multiprocessing.set_start_method('spawn', force=True)
   ```

### Uneven GPU Usage

**Symptoms:** Some GPUs show 100% utilization while others are idle or underutilized.

**Causes:**
- Files distributed round-robin, but some files much larger than others
- Very large files may cause temporary imbalance

**Solutions:**

1. **Split large files**: Break extremely large FASTA files into smaller chunks
2. **Pre-sort by size**: Group similar-sized files together for better balancing
3. **Monitor over time**: Temporary imbalance is normal and usually evens out

### Failed Files

**Symptoms:**
```
Warning: Prediction completed with some failures - check failed_files.txt
```

**What to check:**

1. Open `results/failed_files.txt` to see which files failed
2. Common causes:
   - Corrupted sequences or malformed FASTA format
   - Extremely long sequences causing memory overflow
   - Special characters in sequence headers or data

**Solutions:**

- Fix malformed FASTA entries
- Split very long sequences
- Rerun with just the failed files:
  ```bash
  virnucpro predict --input failed_subset.fa --output-dir results/ --resume
  ```

### Progress Dashboard Not Showing

**Symptoms:** No visual progress dashboard appears.

**Causes:**
- Running in non-interactive environment (batch job, CI/CD)
- `--quiet` flag enabled
- Terminal doesn't support ANSI codes

**Solutions:**

- Remove `--quiet` flag if set
- Ensure running in interactive terminal
- Check logs for progress information (logged every 10 seconds)

## Backward Compatibility

The multi-GPU implementation maintains full backward compatibility:

- **Existing checkpoints**: Can resume single-GPU runs with multi-GPU code (and vice versa)
- **Output format**: All output files use the same format as before
- **CLI interface**: All existing flags and options work unchanged
- **Downstream analysis**: No changes required to tools consuming VirNucPro outputs

## Monitoring

### Progress Dashboard

During execution, the progress dashboard shows real-time status:

```
=== GPU Progress Dashboard ===
GPU 0: ████████████████████ 100% [250/250 files] | Mem: 14.2GB/16.0GB
GPU 1: ████████████████░░░░  85% [212/250 files] | Mem: 13.8GB/16.0GB
GPU 2: ██████████████████░░  90% [225/250 files] | Mem: 14.1GB/16.0GB
GPU 3: ████████████████████ 100% [250/250 files] | Mem: 14.3GB/16.0GB

Total: 887/1000 files complete (88.7%)
Estimated time remaining: 1h 23m
```

### Detailed Monitoring

For detailed GPU monitoring during execution:

```bash
# In a separate terminal, monitor GPU usage
watch -n 1 nvidia-smi

# Or use nvidia-smi daemon for continuous monitoring
nvidia-smi dmon -s um -c 100
```

### Log Files

Enable verbose logging for detailed execution information:

```bash
virnucpro predict --input sequences.fa --output-dir results/ -v --log-file virnucpro.log
```

The log file includes:
- GPU memory usage (every 10 seconds)
- File assignments to each GPU
- Processing time per file
- Error details for failed files

## Implementation Details

### Architecture

The multi-GPU implementation uses:

1. **Work queue pattern**: Files distributed to workers via shared queue
2. **Process-based parallelism**: Each GPU runs in a separate process (avoids GIL)
3. **BF16 mixed precision**: Reduces memory footprint by 50% on compatible hardware
4. **Checkpoint compatibility**: Atomic file writes ensure resumability

### Technical Decisions

- **Round-robin distribution**: Simple, predictable, works well for most datasets
- **Process spawn**: Avoids CUDA fork issues, clean per-GPU state
- **Per-file granularity**: Allows fine-grained progress tracking and recovery
- **Exit code 2**: Indicates partial success (some files failed but others completed)

## Best Practices

1. **Start with defaults**: Let VirNucPro auto-detect GPUs and use default settings
2. **Monitor first run**: Watch GPU memory and utilization to optimize batch size
3. **Use consistent hardware**: Mixed GPU models can cause load imbalance
4. **Enable resume**: Always use `--resume` for long-running jobs
5. **Keep intermediate files**: Use `--keep-intermediate` until confident in results
6. **Log to file**: Use `--log-file` for production runs to capture detailed diagnostics

## Additional Resources

- [CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/)
- [PyTorch CUDA Semantics](https://pytorch.org/docs/stable/notes/cuda.html)
- [ESM-2 Model Documentation](https://github.com/facebookresearch/esm)

## Getting Help

If you encounter issues not covered in this guide:

1. Check the log file for detailed error messages
2. Enable verbose mode (`-v`) for more diagnostic output
3. Try single-GPU mode (`--gpus 0`) to isolate multi-GPU issues
4. Open an issue on the VirNucPro GitHub repository with:
   - Error message and full traceback
   - Output from `nvidia-smi`
   - VirNucPro version and command used
