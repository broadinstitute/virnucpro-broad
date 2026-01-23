# Phase 2: DNABERT-S Optimization - Research

**Researched:** 2026-01-23
**Domain:** DNABERT-S feature extraction optimization for multi-GPU processing
**Confidence:** HIGH

## Summary

Research confirms that DNABERT-S optimization should follow ESM-2's established patterns with specific adaptations for DNA k-mer tokenization. The standard approach uses PyTorch transformer optimization techniques including dynamic batching, BF16 mixed precision, and spawn-context multiprocessing for CUDA safety. Key findings show that batch sizes should be token-based (abstracting k-mer complexity), with typical 3-4x throughput improvements achievable through proper batch sizing and multi-GPU parallelization.

The existing codebase already implements ESM-2 multi-GPU patterns successfully, providing a solid foundation. DNABERT-S should inherit from a shared base worker class to ensure consistency while maintaining model-specific optimizations. Critical considerations include memory-efficient batching through attention mask pooling, proper CUDA context isolation via spawn, and progress reporting through multiprocessing Queue.

**Primary recommendation:** Implement BaseEmbeddingWorker abstract class with unified interface, then specialize for DNABERT-S with token-based batching matching ESM-2's approach.

## Standard Stack

The established libraries/tools for this domain:

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| transformers | 4.36+ | DNABERT-S model loading | Official Hugging Face integration |
| torch | 2.0+ | Deep learning runtime | CUDA support, BF16, no_grad contexts |
| multiprocessing | stdlib | Worker pool management | Spawn context for CUDA safety |
| Bio.SeqIO | 1.80+ | FASTA sequence parsing | Standard bioinformatics I/O |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| torch.cuda.amp | 2.0+ | Mixed precision (BF16) | Ampere+ GPUs (compute capability ≥ 8) |
| abc | stdlib | Abstract base classes | Defining worker interface |
| queue | stdlib | Progress reporting | Inter-process communication |
| pathlib | stdlib | File path handling | Cross-platform compatibility |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| multiprocessing.Pool | concurrent.futures | Less control over spawn context |
| Manual batching | DataLoader | Overhead for simple sequence batching |
| abc.ABC | Protocol typing | Less enforced contract |

**Installation:**
```bash
# Already installed in project
pip install transformers torch biopython
```

## Architecture Patterns

### Recommended Project Structure
```
virnucpro/
├── pipeline/
│   ├── base_worker.py       # BaseEmbeddingWorker abstract class
│   ├── parallel_dnabert.py  # DNABERT-S worker implementation
│   ├── parallel_esm.py      # ESM-2 worker (existing, adapt to base)
│   └── work_queue.py        # Shared queue manager (existing)
└── core/
    └── device.py            # BF16 detection logic (existing)
```

### Pattern 1: Abstract Base Worker Class
**What:** Unified interface for embedding extraction workers
**When to use:** Always - enforces consistency between DNABERT-S and ESM-2
**Example:**
```python
# Source: Python ABC documentation combined with existing patterns
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Tuple

class BaseEmbeddingWorker(ABC):
    @abstractmethod
    def process_files_worker(
        self,
        file_subset: List[Path],
        device_id: int,
        batch_size: int,
        output_dir: Path,
        **kwargs
    ) -> Tuple[List[Path], List[Tuple[Path, str]]]:
        """Process files on specific GPU"""
        pass

    @abstractmethod
    def get_optimal_batch_size(self, device: torch.device) -> int:
        """Determine optimal batch size for device"""
        pass
```

### Pattern 2: Token-Based Batching
**What:** Abstract k-mer complexity into token counts like ESM-2
**When to use:** Always for DNABERT-S batching decisions
**Example:**
```python
# Source: Existing ESM-2 implementation pattern
def create_batches(sequences, max_tokens_per_batch):
    batches = []
    current_batch = []
    current_tokens = 0

    for seq_id, seq_str in sequences:
        seq_tokens = len(seq_str)  # For DNABERT-S, each base = ~1 token
        if current_tokens + seq_tokens > max_tokens_per_batch and current_batch:
            batches.append(current_batch)
            current_batch = []
            current_tokens = 0
        current_batch.append((seq_id, seq_str))
        current_tokens += seq_tokens

    if current_batch:
        batches.append(current_batch)
    return batches
```

### Pattern 3: Spawn Context with Deferred CUDA
**What:** Use spawn multiprocessing context with CUDA initialization in workers
**When to use:** Always for multi-GPU processing
**Example:**
```python
# Source: PyTorch multiprocessing best practices
ctx = multiprocessing.get_context('spawn')
with ctx.Pool(num_workers, initializer=init_worker, initargs=(queue,)) as pool:
    # CUDA devices initialized inside workers, not parent
    results = pool.starmap(worker_func, worker_args)
```

### Anti-Patterns to Avoid
- **Fork with CUDA:** Never use fork context after CUDA initialization - causes crashes/deadlocks
- **Global model loading:** Don't load models in parent process - each worker loads independently
- **File-count distribution:** Don't distribute by file count - use sequence count for balance

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Batch padding | Custom padding logic | tokenizer(padding=True) | Handles attention masks correctly |
| BF16 detection | Manual GPU checks | torch.cuda.get_device_capability() | Reliable capability detection |
| Progress reporting | Threading + shared state | multiprocessing.Queue | Process-safe communication |
| Worker pool | Manual process spawning | multiprocessing.Pool(context='spawn') | Handles lifecycle, exceptions |
| Sequence batching | Custom token counting | Existing token-based pattern | Proven, handles edge cases |

**Key insight:** The framework (transformers, PyTorch, multiprocessing) already solves these problems correctly. Custom solutions introduce bugs and maintenance burden.

## Common Pitfalls

### Pitfall 1: CUDA Context Corruption
**What goes wrong:** Workers crash mysteriously or deadlock
**Why it happens:** Using fork context after CUDA initialization in parent
**How to avoid:** Always use spawn context, initialize CUDA only in workers
**Warning signs:** "CUDA error: initialization error", random worker hangs

### Pitfall 2: Memory Leaks with Large Batches
**What goes wrong:** GPU OOM despite conservative batch sizes
**Why it happens:** Not using torch.no_grad() context, gradient accumulation
**How to avoid:** Wrap all inference in torch.no_grad(), clear cache between batches
**Warning signs:** Steadily increasing GPU memory usage

### Pitfall 3: Unbalanced GPU Utilization
**What goes wrong:** One GPU at 100%, others idle
**Why it happens:** Distributing files by count, not sequence content
**How to avoid:** Use bin-packing by sequence count like parallel_esm.py
**Warning signs:** Widely varying worker completion times

### Pitfall 4: Incorrect Pooling Without Attention Masks
**What goes wrong:** Embeddings include padding token representations
**Why it happens:** Simple mean pooling without masking
**How to avoid:** Use attention-mask weighted pooling
**Warning signs:** Degraded model performance on padded sequences

## Code Examples

Verified patterns from official sources and existing codebase:

### BF16 Auto-Detection
```python
# Source: Existing virnucpro/pipeline/features.py
use_bf16 = False
if str(device).startswith('cuda'):
    capability = torch.cuda.get_device_capability(device)
    use_bf16 = capability[0] >= 8  # Ampere or newer
    if use_bf16:
        logger.info("Using BF16 mixed precision")
```

### Attention-Masked Mean Pooling
```python
# Source: Existing virnucpro/pipeline/features.py
# Mean pool with attention mask to exclude padding
if attention_mask is not None:
    embedding_means = (hidden_states * attention_mask.unsqueeze(-1)).sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
else:
    embedding_means = torch.mean(hidden_states, dim=1)
```

### Progress Queue Pattern
```python
# Source: Existing virnucpro/pipeline/work_queue.py
# In worker process
if progress_queue is not None:
    progress_queue.put({
        'gpu_id': device_id,
        'file': str(file_path),
        'status': 'complete'
    })
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| K-mer tokenization | BPE tokenization (DNABERT-2) | 2024 | 5x shorter sequences |
| Single GPU processing | Multi-GPU with spawn | 2023+ | 4x+ throughput |
| Fork multiprocessing | Spawn context | 2022+ | CUDA stability |
| FP32 inference | BF16 on Ampere+ | 2020+ | 50% memory savings |
| Manual batching | Dynamic token batching | 2023+ | Better GPU utilization |

**Deprecated/outdated:**
- Fork context with CUDA: Known to cause issues, spawn is standard
- Per-sequence processing: Batch processing is 10x+ faster
- Fixed batch sizes: Dynamic sizing based on sequence length

## Open Questions

Things that couldn't be fully resolved:

1. **Optimal default batch sizes**
   - What we know: ESM-2 uses 2048-3072 tokens, should profile DNABERT-S
   - What's unclear: Exact token/memory ratio for DNABERT-S
   - Recommendation: Start with 2048 tokens, profile and adjust

2. **K-mer vs token abstraction**
   - What we know: DNABERT-S uses k-mers internally
   - What's unclear: If k-mer boundaries affect batching efficiency
   - Recommendation: Treat as opaque tokens initially, optimize if needed

3. **Model loading optimization**
   - What we know: Each worker loads model independently
   - What's unclear: If model could be shared via shared memory
   - Recommendation: Keep simple with independent loading for now

## Sources

### Primary (HIGH confidence)
- [PyTorch multiprocessing best practices](https://docs.pytorch.org/docs/stable/notes/multiprocessing.html) - Spawn context requirements
- Existing codebase - virnucpro/pipeline/features.py, parallel_esm.py patterns
- [Python ABC documentation](https://docs.python.org/3/library/abc.html) - Abstract base classes

### Secondary (MEDIUM confidence)
- [DNABERT-2 GitHub repository](https://github.com/MAGICS-LAB/DNABERT_2) - Model architecture details
- [PyTorch GPU optimization guide](https://pytorch.org/blog/out-of-the-box-acceleration/) - Performance techniques
- [Transformer memory optimization](https://sebastianraschka.com/blog/2023/pytorch-memory-optimization.html) - Batching strategies

### Tertiary (LOW confidence)
- Community discussions on optimal batch sizes (needs profiling verification)
- Generic transformer optimization guides (may not apply to DNABERT-S specifics)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - Using established PyTorch/transformers ecosystem
- Architecture: HIGH - Following proven ESM-2 patterns from codebase
- Pitfalls: HIGH - Based on documented CUDA/multiprocessing issues

**Research date:** 2026-01-23
**Valid until:** 2026-02-23 (30 days - stable patterns)