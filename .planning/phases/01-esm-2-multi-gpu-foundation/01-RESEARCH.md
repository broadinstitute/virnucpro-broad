# Phase 1: ESM-2 Multi-GPU Foundation - Research

**Researched:** 2026-01-22
**Domain:** Multi-GPU parallelization for ESM-2 protein feature extraction
**Confidence:** HIGH

## Summary

ESM-2 multi-GPU parallelization requires careful orchestration around CUDA context initialization, memory management, and batch processing. The current implementation uses file-level work distribution with multiprocessing spawn context for DNABERT-S, which provides a solid foundation to extend to ESM-2. The key challenge is managing ESM-2's larger memory footprint (3B model at 12GB) while maintaining backward compatibility with single-GPU systems.

The standard approach uses PyTorch multiprocessing with spawn context to avoid CUDA re-initialization errors, implements size-aware file assignment based on sequence counts, and leverages BF16 mixed precision with torch.no_grad for inference optimization. Work distribution follows a batch queue manager pattern with round-robin assignment, while progress visibility uses rich for multi-task dashboards.

**Primary recommendation:** Use multiprocessing spawn with deferred CUDA initialization, BF16 autocast for memory efficiency, and rich-based live dashboard for per-GPU progress tracking.

## Standard Stack

The established libraries/tools for this domain:

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| torch.multiprocessing | 2.9+ | Multi-GPU worker orchestration | Spawn context avoids CUDA re-init errors |
| esm | 2.0+ | ESM-2 model and utilities | Official Meta implementation |
| torch.cuda.amp | 2.9+ | BF16 mixed precision | 2x memory reduction, no loss scaling needed |
| rich | 14.1+ | Live progress dashboard | Multiple concurrent progress bars for GPU tracking |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| queue.SimpleQueue | 3.9+ | Work distribution | Avoids multiprocessing.Queue threading issues |
| psutil | 5.9+ | GPU memory monitoring | Pre-flight validation and adaptive batching |
| filelock | 3.12+ | Checkpoint atomicity | Prevents corruption during concurrent writes |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| multiprocessing | Ray/Dask | More complex but better for distributed clusters |
| rich | tqdm | Simpler but poor multi-bar support |
| BF16 | FP16 | FP16 needs loss scaling, BF16 has FP32 range |

**Installation:**
```bash
pip install esm torch rich psutil filelock
```

## Architecture Patterns

### Recommended Project Structure
```
virnucpro/pipeline/
├── parallel_esm.py      # ESM-2 specific parallel processing
├── work_queue.py        # Generic batch queue manager
├── gpu_monitor.py       # GPU memory and utilization tracking
└── dashboard.py         # Rich-based progress dashboard
```

### Pattern 1: Deferred CUDA Initialization
**What:** Initialize CUDA only within worker processes, never in parent
**When to use:** Always when using multiprocessing with GPUs
**Example:**
```python
# Source: PyTorch multiprocessing best practices
def worker_function(file_list, device_id):
    # CUDA initialization happens here, not in parent
    device = torch.device(f'cuda:{device_id}')
    model = load_model().to(device)
    # Process files...

# Parent process - NO CUDA operations
ctx = multiprocessing.get_context('spawn')
with ctx.Pool(num_gpus) as pool:
    pool.starmap(worker_function, worker_args)
```

### Pattern 2: Size-Aware Work Assignment
**What:** Assign files based on total sequence count, not just file count
**When to use:** When files have varying numbers of sequences
**Example:**
```python
def assign_files_by_size(files, num_workers):
    # Count sequences per file
    file_sizes = [(f, count_sequences(f)) for f in files]
    file_sizes.sort(key=lambda x: x[1], reverse=True)

    # Assign to minimize max load
    worker_loads = [0] * num_workers
    worker_files = [[] for _ in range(num_workers)]

    for file, size in file_sizes:
        min_worker = worker_loads.index(min(worker_loads))
        worker_files[min_worker].append(file)
        worker_loads[min_worker] += size

    return worker_files
```

### Anti-Patterns to Avoid
- **Fork with CUDA:** Using fork() start method causes "Cannot re-initialize CUDA" errors
- **Parent CUDA ops:** Any torch.cuda calls in parent process before spawning
- **Shared CUDA tensors:** Passing CUDA tensors through queues (move to CPU first)

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Work queue | Custom queue + threading | queue.SimpleQueue | Thread-safe without deadlock risk |
| Progress bars | Print statements | rich.progress | Handles concurrent updates, terminal compatibility |
| Batch size finding | Manual binary search | Adaptive batching algorithms | Considers sequence length variance |
| Checkpoint atomicity | Direct file writes | filelock + temp files | Prevents corruption on concurrent access |
| GPU memory check | Parse nvidia-smi | torch.cuda.mem_get_info() | Direct API, no subprocess overhead |

**Key insight:** Multi-GPU coordination has many edge cases (worker crashes, OOM, heterogeneous GPUs) that mature libraries handle correctly.

## Common Pitfalls

### Pitfall 1: CUDA Context in Parent Process
**What goes wrong:** "RuntimeError: Cannot re-initialize CUDA in forked subprocess"
**Why it happens:** Parent process touches CUDA before spawning workers
**How to avoid:** Defer ALL CUDA operations to worker functions, use spawn context
**Warning signs:** Error occurs immediately when workers start

### Pitfall 2: Memory Fragmentation from Variable Batches
**What goes wrong:** OOM errors despite available VRAM
**Why it happens:** ESM-2 allocates different amounts per sequence length
**How to avoid:** Sort sequences by length, use adaptive batching
**Warning signs:** Sporadic OOM on similar-sized batches

### Pitfall 3: Queue Deadlocks with Large Data
**What goes wrong:** Workers hang when passing results through queues
**Why it happens:** multiprocessing.Queue uses threads that can deadlock
**How to avoid:** Use SimpleQueue or write results to files
**Warning signs:** Process hangs after processing N items

### Pitfall 4: BF16 on Older GPUs
**What goes wrong:** No speedup or errors with BF16
**Why it happens:** Pre-Ampere GPUs don't support BF16
**How to avoid:** Check compute capability, fallback to FP32
**Warning signs:** torch.cuda.get_device_capability() < (8, 0)

## Code Examples

Verified patterns from official sources:

### ESM-2 BF16 Inference
```python
# Source: PyTorch AMP documentation
import torch
from torch.cuda.amp import autocast

model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
model = model.to(device)
model.eval()

with torch.no_grad():
    with autocast(dtype=torch.bfloat16):
        # BF16 computation - 2x memory savings
        results = model(tokens, repr_layers=[36])
        embeddings = results["representations"][36]
```

### Rich Multi-GPU Dashboard
```python
# Source: Rich documentation
from rich.progress import Progress, BarColumn, TextColumn
from rich.table import Table
from rich.live import Live

with Progress() as progress:
    gpu_tasks = {}
    for gpu_id in range(num_gpus):
        task = progress.add_task(
            f"[cyan]GPU {gpu_id}",
            total=file_counts[gpu_id]
        )
        gpu_tasks[gpu_id] = task

    # Update from workers
    while not done:
        gpu_id, files_done = result_queue.get()
        progress.update(gpu_tasks[gpu_id], advance=1)
```

### Adaptive Batch Size Discovery
```python
# Source: PyTorch forums / memory management
def find_optimal_batch_size(model, sample_input, device):
    batch_size = 1
    while True:
        try:
            # Test with current batch size
            test_batch = sample_input.repeat(batch_size, 1)
            with torch.no_grad():
                _ = model(test_batch.to(device))

            # Clear cache and try larger
            torch.cuda.empty_cache()
            batch_size *= 2
        except torch.cuda.OutOfMemoryError:
            # Back off to last working size
            torch.cuda.empty_cache()
            return max(1, batch_size // 2)
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| DataParallel | DistributedDataParallel | PyTorch 1.5+ | 2-3x faster multi-GPU |
| FP16 with loss scaling | BF16 without scaling | Ampere GPUs (2020) | Simpler, more stable |
| nvidia-smi parsing | torch.cuda APIs | PyTorch 1.10+ | Direct memory queries |
| multiprocessing.Queue | SimpleQueue | Python 3.9+ | No thread deadlocks |
| tqdm multiple bars | rich.progress | 2023+ | Clean concurrent progress |

**Deprecated/outdated:**
- nn.DataParallel: GIL-bound, use DDP or multiprocessing
- Manual FP16: Use autocast for automatic mixed precision
- Fork method with CUDA: Always use spawn or forkserver

## Open Questions

Things that couldn't be fully resolved:

1. **Dynamic work queue vs pre-assignment**
   - What we know: Both patterns work, trade-offs exist
   - What's unclear: Which performs better for ESM-2 specifically
   - Recommendation: Start with pre-assignment for simplicity, measure

2. **Optimal toks_per_batch with BF16**
   - What we know: Default is 2048 for FP32
   - What's unclear: How much to increase for BF16
   - Recommendation: Start conservative (3072), tune based on GPU

3. **Heterogeneous GPU handling**
   - What we know: Phase 1 treats all GPUs equally
   - What's unclear: Performance impact of mixing GPU generations
   - Recommendation: Log warnings if detected, defer weighting to Phase 5

## Sources

### Primary (HIGH confidence)
- [PyTorch Multiprocessing Documentation](https://docs.pytorch.org/docs/stable/notes/multiprocessing.html) - Spawn context, CUDA requirements
- [Rich Progress Documentation](https://rich.readthedocs.io/en/latest/progress.html) - Multi-task progress bars
- [PyTorch AMP Documentation](https://docs.pytorch.org/docs/stable/amp.html) - BF16 autocast usage

### Secondary (MEDIUM confidence)
- [ESM GitHub Repository](https://github.com/facebookresearch/esm) - Model loading and batch processing
- [BF16 Training Without Loss Scaling](https://pytorch.org/blog/what-every-user-should-know-about-mixed-precision-training-in-pytorch/) - BF16 advantages

### Tertiary (LOW confidence)
- Community discussions on optimal batch sizes (needs empirical validation)
- Adaptive batching algorithms from 2025 papers (not tested with ESM-2)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - Based on PyTorch official docs and current codebase
- Architecture: HIGH - Proven patterns from existing parallel.py implementation
- Pitfalls: HIGH - Well-documented CUDA multiprocessing issues

**Research date:** 2026-01-22
**Valid until:** 2026-04-22 (3 months for stable PyTorch/ESM ecosystem)