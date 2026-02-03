# Phase 5: Async DataLoader Foundation - Research

**Researched:** 2026-02-03
**Domain:** PyTorch DataLoader async I/O, CUDA worker safety, GPU performance monitoring
**Confidence:** HIGH (PyTorch official docs, existing codebase, prior v2.0 research)

## Summary

This phase implements the foundational async DataLoader architecture for single-GPU processing. The key architectural shift is from v1.0's multi-worker-per-GPU pattern (each worker loads ESM-2 11GB model) to a single-process-per-GPU pattern where DataLoader workers perform pure CPU I/O (FASTA parsing) and the main process handles tokenization and GPU inference.

The critical safety constraint is that DataLoader workers MUST NEVER initialize CUDA. This requires:
1. Using `spawn` multiprocessing context (not `fork`) to prevent CUDA context inheritance
2. Explicit validation in worker `__init__` that `torch.cuda.is_available()` returns False
3. Workers yield raw sequence strings only (no tokenization, no model loading)
4. Tokenization happens in main process via `collate_fn`

The architecture prepares for Phase 6 (sequence packing) by having `collate_fn` output concatenated 1D tensors with `cu_seqlens` arrays for FlashAttention varlen attention.

**Primary recommendation:** Implement a CPU-only `SequenceDataset(IterableDataset)` that parses FASTA files and yields sequence strings, with tokenization in a `VarlenCollator` that produces packed batch format ready for FlashAttention.

## Standard Stack

The established libraries/tools for this domain:

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| PyTorch DataLoader | 2.0+ | Async batch loading with prefetching | Built-in, no alternatives needed |
| Biopython SeqIO | 1.76+ | FASTA file parsing | Standard for bioinformatics I/O |
| nvitop | 1.3+ | GPU utilization monitoring | Cleaner API than raw pynvml, PyTorch callback integration |
| multiprocessing (spawn) | stdlib | CUDA-safe process creation | PyTorch-recommended for CUDA compatibility |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| torch.cuda.Stream | 2.0+ | Async GPU operations | Already implemented in `virnucpro/cuda/stream_manager.py` |
| pynvml | 12.0+ | Low-level NVML bindings | Alternative if nvitop unavailable |
| psutil | 5.9+ | CPU/RAM monitoring | Memory leak detection |
| flash-attn | 2.6+ | FlashAttention varlen support | Phase 6 integration prep |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| nvitop | nvidia-smi subprocess | nvitop has Python API, nvidia-smi requires parsing stdout |
| SeqIO.parse() | SimpleFastaParser | SimpleFastaParser 5x faster but returns tuples not SeqRecord |
| IterableDataset | MapStyleDataset | IterableDataset better for streaming, no index building |

**Installation:**
```bash
pip install nvitop biopython torch
```

## Architecture Patterns

### Recommended Project Structure
```
virnucpro/
  data/
    sequence_dataset.py      # NEW: IterableDataset for FASTA streaming
    collators.py             # NEW: VarlenCollator for packed batch format
  cuda/
    stream_manager.py        # EXISTING: Reuse for async GPU operations
  pipeline/
    async_inference.py       # NEW: Single-GPU async inference loop
    gpu_coordinator.py       # Phase 7: Multi-GPU coordinator
  utils/
    gpu_monitor.py           # EXISTING: Extend with nvitop integration
```

### Pattern 1: CPU-Only Worker Dataset

**What:** DataLoader workers parse FASTA files and yield sequence strings (no CUDA, no tokenization)
**When to use:** Always for async DataLoader with GPU inference
**Example:**
```python
# Source: PyTorch IterableDataset docs + CONTEXT.md decisions
from torch.utils.data import IterableDataset, get_worker_info
from Bio import SeqIO
import os

class SequenceDataset(IterableDataset):
    """CPU-only dataset for FASTA parsing. Workers NEVER touch CUDA."""

    def __init__(self, fasta_files: List[Path], max_length: int = 1024):
        super().__init__()
        self.files = fasta_files
        self.max_length = max_length

        # CRITICAL: Validate no CUDA in worker process
        # This check runs when Dataset is created in each worker
        if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
            # Only validate in actual worker (not main process)
            worker_info = get_worker_info()
            if worker_info is not None:
                assert not torch.cuda.is_available(), (
                    f"FATAL: Worker {worker_info.id} has CUDA access. "
                    "Workers must be CPU-only. Check spawn context."
                )

    def __iter__(self):
        worker_info = get_worker_info()

        if worker_info is None:
            # Single-process loading
            files_to_process = self.files
        else:
            # Multi-worker: shard files across workers
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            files_to_process = [
                f for i, f in enumerate(self.files)
                if i % num_workers == worker_id
            ]

        for file_path in files_to_process:
            for record in SeqIO.parse(file_path, 'fasta'):
                sequence = str(record.seq)[:self.max_length]
                yield {
                    'id': record.id,
                    'sequence': sequence,
                    'file': file_path.name
                }
```

### Pattern 2: Varlen Collator for FlashAttention

**What:** Collate function that concatenates sequences into 1D tensor with cu_seqlens
**When to use:** When preparing for FlashAttention varlen pattern (Phase 6)
**Example:**
```python
# Source: flash-attn docs + CONTEXT.md packing decision
import torch
from typing import List, Dict, Any

class VarlenCollator:
    """Collate sequences into packed format for FlashAttention varlen."""

    def __init__(self, tokenizer, max_tokens: int = 4096):
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens

    def __call__(self, batch: List[Dict[str, str]]) -> Dict[str, Any]:
        sequences = [item['sequence'] for item in batch]
        sequence_ids = [item['id'] for item in batch]

        # Tokenize in main process (not worker)
        all_tokens = []
        cu_seqlens = [0]

        for seq in sequences:
            # ESM batch_converter or tokenizer.encode()
            tokens = self.tokenizer(seq)  # Returns list of token IDs

            # Check if adding exceeds max_tokens
            if cu_seqlens[-1] + len(tokens) > self.max_tokens:
                break

            all_tokens.extend(tokens)
            cu_seqlens.append(len(all_tokens))

        return {
            'input_ids': torch.tensor(all_tokens, dtype=torch.long),
            'cu_seqlens': torch.tensor(cu_seqlens, dtype=torch.int32),
            'max_seqlen': max(cu_seqlens[i+1] - cu_seqlens[i]
                             for i in range(len(cu_seqlens)-1)),
            'sequence_ids': sequence_ids[:len(cu_seqlens)-1],
            'num_sequences': len(cu_seqlens) - 1
        }
```

### Pattern 3: DataLoader with Spawn Context

**What:** Create DataLoader with spawn multiprocessing context and CUDA-safe configuration
**When to use:** Always when using DataLoader with GPU inference
**Example:**
```python
# Source: PyTorch multiprocessing docs
from torch.utils.data import DataLoader

def create_async_dataloader(
    dataset: SequenceDataset,
    collate_fn: VarlenCollator,
    num_workers: int = 4,
    prefetch_factor: int = 4
) -> DataLoader:
    """Create DataLoader with CUDA-safe configuration."""

    def worker_init_fn(worker_id: int):
        """Ensure worker has no CUDA access."""
        import os
        # Redundant safety: hide CUDA from workers
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # HuggingFace safety

    return DataLoader(
        dataset,
        batch_size=32,  # Sequences per batch (collate_fn may pack fewer)
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=True,             # Fast CPU->GPU transfer
        persistent_workers=True,     # Keep workers alive
        multiprocessing_context='spawn',  # CUDA safety
        collate_fn=collate_fn,
        worker_init_fn=worker_init_fn,
        timeout=300  # 5 min timeout per batch
    )
```

### Pattern 4: GPU Utilization Monitoring with nvitop

**What:** Programmatic GPU utilization tracking using nvitop Python API
**When to use:** Performance monitoring during inference loop
**Example:**
```python
# Source: nvitop documentation
from nvitop import Device
import time

class GPUMonitor:
    """Monitor GPU utilization and detect bottlenecks."""

    def __init__(self, device_id: int = 0, idle_threshold: float = 0.10):
        self.device = Device(device_id)
        self.idle_threshold = idle_threshold
        self.samples = []
        self.dataloader_wait_times = []

    def sample(self):
        """Take utilization sample."""
        with self.device.oneshot():
            util = self.device.gpu_utilization()
            mem_used = self.device.memory_used()
            mem_total = self.device.memory_total()

        self.samples.append({
            'timestamp': time.time(),
            'gpu_util': util,
            'mem_used_gb': mem_used / (1024**3),
            'mem_total_gb': mem_total / (1024**3)
        })
        return util

    def record_dataloader_wait(self, wait_time_ms: float):
        """Record time spent waiting for DataLoader."""
        self.dataloader_wait_times.append(wait_time_ms)

    def check_bottleneck(self) -> bool:
        """Check if GPU is idle too often (I/O bottleneck)."""
        if len(self.samples) < 10:
            return False
        recent = self.samples[-10:]
        avg_util = sum(s['gpu_util'] for s in recent) / len(recent)
        return avg_util < (1 - self.idle_threshold) * 100

    def get_summary(self) -> dict:
        """Get performance summary."""
        if not self.samples:
            return {}
        return {
            'avg_gpu_util': sum(s['gpu_util'] for s in self.samples) / len(self.samples),
            'max_gpu_util': max(s['gpu_util'] for s in self.samples),
            'min_gpu_util': min(s['gpu_util'] for s in self.samples),
            'avg_dataloader_wait_ms': (
                sum(self.dataloader_wait_times) / len(self.dataloader_wait_times)
                if self.dataloader_wait_times else 0
            ),
            'total_samples': len(self.samples)
        }
```

### Anti-Patterns to Avoid

- **Tokenizing in worker process:** Workers should yield raw strings, tokenization happens in `collate_fn` (main process)
- **Loading model in worker:** Model loads once in main process, workers never import model code
- **Using `fork` context:** Fork copies parent's CUDA context, causing corruption. Always use `spawn`
- **Calling `torch.cuda` in Dataset:** Even `torch.cuda.is_available()` initializes CUDA on some systems. Guard with env check
- **High prefetch_factor:** Values >4 cause memory bloat. Start with 4, reduce if RAM grows
- **Ignoring TOKENIZERS_PARALLELISM:** HuggingFace tokenizers spawn threads that deadlock with DataLoader fork

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| GPU utilization tracking | Custom nvidia-smi parser | nvitop Python API | Clean API, handles edge cases, thread-safe |
| CUDA stream management | Manual stream/event sync | Existing `StreamProcessor` | Already implemented in codebase |
| Atomic file saves | Custom temp-file logic | Existing `atomic_save()` | Already handles validation, .done markers |
| Worker CUDA isolation | Complex env manipulation | spawn context + worker_init_fn | PyTorch standard pattern |
| FASTA parsing | Custom parser | SeqIO.parse() or SimpleFastaParser | Handles edge cases, tested extensively |

**Key insight:** The existing codebase already has stream management (`virnucpro/cuda/stream_manager.py`) and atomic saves (`virnucpro/core/checkpoint.py`). Reuse these rather than rebuilding.

## Common Pitfalls

### Pitfall 1: CUDA Initialization in Workers

**What goes wrong:** Worker process calls any `torch.cuda.*` function, initializing CUDA context in forked process
**Why it happens:**
- Model code imported at module level initializes CUDA
- `torch.cuda.is_available()` can initialize CUDA on some PyTorch versions
- Tokenizer with GPU support auto-detects and initializes CUDA
**How to avoid:**
- Use `spawn` context (creates fresh process, no CUDA inheritance)
- Set `CUDA_VISIBLE_DEVICES=''` in `worker_init_fn`
- Assert `torch.cuda.is_available() == False` in Dataset `__init__` (only in workers)
- Import model code lazily in main process only
**Warning signs:** `RuntimeError: Cannot re-initialize CUDA in forked subprocess`

### Pitfall 2: HuggingFace Cache Race Condition

**What goes wrong:** Multiple workers call `AutoModel.from_pretrained()` simultaneously, corrupting model cache
**Why it happens:** Workers share filesystem, concurrent writes to same cache files
**How to avoid:**
- NEVER load models in workers (this phase: workers only parse FASTA)
- If tokenizer needed in workers, stagger with worker_id delay
- Use `TRANSFORMERS_OFFLINE=1` if model already cached
- Use filelock for multi-worker model loading (but better: don't load in workers)
**Warning signs:** Corrupted cache errors, partial downloads, SIGKILL on workers

### Pitfall 3: Persistent Worker Memory Leaks

**What goes wrong:** With `persistent_workers=True`, worker memory grows slowly over time
**Why it happens:** Workers accumulate references across batches, especially with high prefetch_factor
**How to avoid:**
- Use conservative `prefetch_factor=4` (not higher)
- Monitor worker RSS memory periodically
- Consider worker restart every N batches if leak detected
- Profile memory with `tracemalloc` if growth observed
**Warning signs:** Gradual CPU RAM increase, eventual OOM

### Pitfall 4: Uneven File Distribution

**What goes wrong:** Some workers finish early, sit idle while others process large files
**Why it happens:** Round-robin sharding doesn't account for file size variance
**How to avoid:**
- Sort files by size descending before sharding
- Use sequence-count-based bin-packing (existing `assign_files_by_sequences`)
- Monitor worker queue depths
**Warning signs:** GPU utilization drops mid-processing, uneven worker completion times

### Pitfall 5: pin_memory with High prefetch_factor

**What goes wrong:** Pinned memory exhaustion or excessive host RAM usage
**Why it happens:** `prefetch_factor * num_workers` batches pinned simultaneously
**How to avoid:**
- Start with `prefetch_factor=4`, `num_workers=4` (16 batches max pinned)
- Monitor host RAM during first runs
- Reduce prefetch_factor if RAM exceeds 50% of total
**Warning signs:** Host RAM spike, pinned memory allocation failure

## Code Examples

Verified patterns from official sources:

### DataLoader Batch Timing

```python
# Source: PyTorch best practices
import time

def inference_loop_with_monitoring(dataloader, model, device, monitor):
    """Inference loop with DataLoader wait time tracking."""

    iterator = iter(dataloader)
    model.eval()

    with torch.no_grad():
        batch_idx = 0
        while True:
            try:
                # Time DataLoader fetch
                t0 = time.perf_counter()
                batch = next(iterator)
                fetch_time_ms = (time.perf_counter() - t0) * 1000
                monitor.record_dataloader_wait(fetch_time_ms)

                # Move to GPU (pinned memory -> fast transfer)
                input_ids = batch['input_ids'].to(device, non_blocking=True)
                cu_seqlens = batch['cu_seqlens'].to(device, non_blocking=True)

                # Inference
                outputs = model(input_ids, cu_seqlens=cu_seqlens)

                # Sample GPU utilization every 10 batches
                if batch_idx % 10 == 0:
                    util = monitor.sample()
                    if monitor.check_bottleneck():
                        logger.warning(f"I/O bottleneck detected: GPU util {util}%")

                batch_idx += 1

            except StopIteration:
                break

    return monitor.get_summary()
```

### Worker CUDA Validation

```python
# Source: PyTorch multiprocessing docs + project safety requirements
import os
import torch

def worker_init_fn(worker_id: int):
    """Initialize worker with CUDA isolation."""
    # Hide CUDA devices from worker
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

    # Disable HuggingFace tokenizer parallelism (prevents deadlocks)
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    # Seed for reproducibility
    import numpy as np
    np.random.seed(worker_id)

class CUDASafeDataset(IterableDataset):
    """Dataset that validates CUDA isolation in workers."""

    def __init__(self, files: List[Path]):
        super().__init__()
        self.files = files
        self._validated = False

    def _validate_cuda_isolation(self):
        """Assert worker has no CUDA access."""
        if self._validated:
            return

        worker_info = get_worker_info()
        if worker_info is not None:
            # Only validate in actual worker process
            cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
            if cuda_visible != '':
                raise RuntimeError(
                    f"Worker {worker_info.id}: CUDA_VISIBLE_DEVICES not empty: {cuda_visible}"
                )

            # Check torch doesn't see CUDA
            if torch.cuda.is_available():
                raise RuntimeError(
                    f"Worker {worker_info.id}: torch.cuda.is_available() is True. "
                    "Worker must not have CUDA access."
                )

        self._validated = True

    def __iter__(self):
        self._validate_cuda_isolation()
        # ... yield sequences ...
```

### nvitop Integration Pattern

```python
# Source: nvitop documentation
from nvitop import Device, ResourceMetricCollector
import logging

logger = logging.getLogger(__name__)

def create_gpu_monitor_callback(device_id: int = 0):
    """Create callback for periodic GPU monitoring."""
    device = Device(device_id)

    def log_gpu_stats(batch_idx: int):
        """Log GPU stats every N batches."""
        with device.oneshot():
            gpu_util = device.gpu_utilization()
            mem_used = device.memory_used() / (1024**3)
            mem_total = device.memory_total() / (1024**3)

        logger.info(
            f"Batch {batch_idx}: GPU {gpu_util}%, "
            f"Memory {mem_used:.1f}/{mem_total:.1f} GB"
        )

        return gpu_util

    return log_gpu_stats
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Multi-worker-per-GPU | Single-process-per-GPU + DataLoader workers | v2.0 redesign | 63% memory reduction (44GB -> 16GB) |
| Fork multiprocessing | Spawn multiprocessing | PyTorch 2.0+ default | CUDA stability, no context corruption |
| Sequential file loading | Prefetched async loading | Standard since PyTorch 1.8 | GPU never waits for I/O |
| nvidia-smi parsing | nvitop Python API | nvitop stable since 2023 | Clean API, no subprocess overhead |
| Padded batches | Packed sequences with cu_seqlens | FlashAttention 2.0+ | 2-3x throughput for variable-length |

**Deprecated/outdated:**
- `torch.backends.cuda.sdp_kernel()`: Deprecated in PyTorch 2.5+, use `torch.nn.attention.sdpa_kernel()` (already handled in esm2_flash.py)
- `fork` start method for CUDA: Never safe, always use `spawn`
- `DataLoader(num_workers>0)` without spawn context: Deprecated pattern for GPU workloads

## Open Questions

Things that couldn't be fully resolved:

1. **SimpleFastaParser vs SeqIO.parse Performance**
   - What we know: SimpleFastaParser is 5x faster for raw parsing
   - What's unclear: Whether the 5x speedup matters when I/O is async (workers prefetch)
   - Recommendation: Start with SeqIO.parse (safer, more features), profile later

2. **Optimal prefetch_factor Value**
   - What we know: CONTEXT.md says 4+, but research shows memory issues at high values
   - What's unclear: Exact sweet spot for this workload
   - Recommendation: Start with 4, monitor RAM, adjust if needed

3. **ESM batch_converter Thread Safety**
   - What we know: batch_converter is used in main process (safe)
   - What's unclear: Whether it's safe to call from collate_fn (still main process, should be fine)
   - Recommendation: Load batch_converter once, pass to collate_fn via closure

## Sources

### Primary (HIGH confidence)
- [PyTorch DataLoader documentation](https://docs.pytorch.org/docs/stable/data.html) - Official API reference
- [PyTorch Multiprocessing Best Practices](https://docs.pytorch.org/docs/stable/notes/multiprocessing.html) - Spawn vs fork, CUDA safety
- [CUDA Semantics - PyTorch](https://docs.pytorch.org/docs/stable/notes/cuda.html) - Non-blocking transfers, streams
- Existing v2.0 architecture research (`.planning/research/ARCHITECTURE.md`) - Comprehensive prior research
- Existing codebase (`virnucpro/cuda/stream_manager.py`, `virnucpro/data/dataloader_utils.py`) - Current patterns

### Secondary (MEDIUM confidence)
- [nvitop documentation](https://nvitop.readthedocs.io/en/latest/api/device.html) - GPU monitoring API
- [HuggingFace Tokenizer Thread Safety Issue](https://github.com/huggingface/tokenizers/issues/1726) - TOKENIZERS_PARALLELISM rationale
- [PyTorch prefetch_factor memory issue](https://github.com/pytorch/pytorch/issues/97432) - Memory leak with high prefetch
- [flash_attn_varlen_func usage](https://github.com/Dao-AILab/flash-attention/issues/880) - cu_seqlens format

### Tertiary (LOW confidence)
- [Biopython FASTA parsing performance](https://www.biostars.org/p/495333/) - SimpleFastaParser speedup claim
- [8 PyTorch DataLoader Tactics](https://medium.com/@Modexa/8-pytorch-dataloader-tactics-to-max-out-your-gpu-22270f6f3fa8) - General guidance

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - All libraries are standard PyTorch/Python ecosystem
- Architecture patterns: HIGH - Based on prior v2.0 research and CONTEXT.md decisions
- CUDA safety: HIGH - PyTorch official docs + extensive community validation
- Pitfalls: HIGH - Known issues from PyTorch issues tracker and prior research
- nvitop API: MEDIUM - Docs-verified but not tested in this codebase yet

**Research date:** 2026-02-03
**Valid until:** 60 days (stable technologies, unlikely to change)
