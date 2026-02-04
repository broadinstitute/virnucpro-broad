# Phase 7: Multi-GPU Coordination - Research

**Researched:** 2026-02-04
**Domain:** PyTorch multiprocessing, SPMD multi-GPU coordination, HDF5 aggregation
**Confidence:** HIGH (PyTorch documentation verified, codebase patterns analyzed, existing infrastructure leveraged)

## Summary

Phase 7 implements multi-GPU coordination for independent shard processing across N GPUs. The architecture uses a parent orchestrator (non-GPU) that spawns GPU workers, monitors health, and aggregates outputs. Each worker processes assigned sequences deterministically using stride-based index distribution, then writes shard files that are merged into a complete checkpoint.

The existing codebase already has foundational patterns for multi-GPU processing:
- `PersistentWorkerPool` in `persistent_pool.py` - Worker pool with lazy model loading
- `multiprocessing.Pool` with spawn context used throughout
- `AsyncInferenceRunner` from Phase 5/6 for single-GPU processing
- Atomic save patterns in `checkpoint.py`

The critical new components are:
1. **Sequence index file** with metadata (id, length, file, offset) for stride-based sharding
2. **GPUProcessCoordinator** using `multiprocessing.Process` (not `mp.spawn`) for fault tolerance
3. **HDF5 shard aggregation** with chunk-wise streaming to control memory
4. **Per-worker log files** to avoid interleaving

**Primary recommendation:** Use `multiprocessing.Process` with spawn context directly (not `mp.spawn` or `Pool`) to allow partial failure handling where surviving workers complete and produce partial results. This aligns with CONTEXT.md's fault tolerance requirement and provides more control than `torch.multiprocessing.spawn` which kills all workers on any failure.

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| torch.multiprocessing | PyTorch 2.0+ | CUDA-safe process spawning | Native PyTorch multiprocessing with spawn context |
| h5py | 3.9+ | HDF5 shard writing and chunk-wise aggregation | Standard for large tensor storage, existing v1.0 format |
| multiprocessing.Queue | stdlib | Inter-process communication for health/progress | Thread-safe, picklable, integrates with spawn |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| json | stdlib | Sequence index file format | Human-readable, debuggable index files |
| logging.handlers.QueueHandler | stdlib | Per-worker logging without interleaving | Route logs through queue to parent |
| os.stat/mtime | stdlib | Cache invalidation for index files | Detect FASTA file changes |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `multiprocessing.Process` | `torch.multiprocessing.spawn` | spawn kills ALL workers on any failure; Process allows partial results |
| `multiprocessing.Process` | `multiprocessing.Pool` | Pool hangs if worker dies; Process gives direct control |
| JSON index | Pickle index | Pickle is faster for large objects but not human-readable; index is simple metadata |
| HDF5 | PyTorch .pt files | HDF5 supports chunk-wise access; .pt requires loading full file into memory |

**Installation:**
```bash
pip install h5py  # HDF5 support (may already be installed)
```

## Architecture Patterns

### Recommended Project Structure
```
virnucpro/
  pipeline/
    gpu_coordinator.py       # NEW: GPUProcessCoordinator, worker lifecycle
    shard_index.py           # NEW: SequenceIndex creation/caching/sharding
    shard_aggregator.py      # NEW: HDF5 chunk-wise merge, validation
  data/
    sequence_dataset.py      # EXISTING: Update for index-based iteration
```

### Pattern 1: GPUProcessCoordinator with Fault Tolerance

**What:** Parent orchestrator spawns independent GPU workers, monitors health, allows partial completion
**When to use:** Multi-GPU inference where worker failures shouldn't abort entire job
**Example:**
```python
# Source: Python multiprocessing.Process docs + CONTEXT.md decisions
import multiprocessing as mp
import os
import signal
from typing import Dict, List, Optional
from pathlib import Path

class GPUProcessCoordinator:
    """
    SPMD coordinator for multi-GPU processing with fault tolerance.

    Unlike mp.spawn which kills all workers on any failure, this coordinator
    allows workers to complete independently and reports partial results.
    """

    def __init__(self, world_size: int, output_dir: Path):
        self.world_size = world_size
        self.output_dir = output_dir
        self.ctx = mp.get_context('spawn')  # CUDA-safe
        self.workers: Dict[int, mp.Process] = {}
        self.results_queue = self.ctx.Queue()

    def spawn_workers(self, index_path: Path, worker_fn):
        """Spawn independent GPU worker processes."""
        for rank in range(self.world_size):
            # Each worker sees only its assigned GPU
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = str(rank)

            p = self.ctx.Process(
                target=worker_fn,
                args=(rank, self.world_size, index_path,
                      self.output_dir, self.results_queue),
                name=f"gpu_worker_{rank}"
            )
            p.start()
            self.workers[rank] = p

    def wait_for_completion(self, timeout: Optional[float] = None) -> Dict[int, bool]:
        """
        Wait for workers with timeout, returns completion status per rank.

        Allows partial failures - other workers continue running.
        """
        results = {}

        for rank, process in self.workers.items():
            process.join(timeout=timeout)

            if process.is_alive():
                # Timed out - log warning but don't kill
                logger.warning(f"Worker {rank} timed out, still running")
                results[rank] = False
            elif process.exitcode != 0:
                # Worker failed
                logger.error(f"Worker {rank} failed with exit code {process.exitcode}")
                results[rank] = False
            else:
                # Worker completed successfully
                results[rank] = True

        return results
```

### Pattern 2: Sequence Index with Stride Distribution

**What:** Create metadata index of all sequences, distribute by stride to workers
**When to use:** Index-based sharding ensures balanced length distribution across GPUs
**Example:**
```python
# Source: CONTEXT.md decisions + pyfaidx patterns
import json
from pathlib import Path
from typing import List, Dict, Iterator
from dataclasses import dataclass, asdict

@dataclass
class SequenceEntry:
    """Metadata for a single sequence."""
    sequence_id: str
    length: int
    file_path: str
    byte_offset: int  # Start position in FASTA file

def create_sequence_index(fasta_files: List[Path], index_path: Path) -> Path:
    """
    Create or load cached sequence index.

    Index format (JSON for human readability):
    {
        "version": "1.0",
        "created": "2026-02-04T12:00:00",
        "fasta_mtimes": {"file1.fa": 1234567890, ...},
        "sequences": [
            {"sequence_id": "seq1", "length": 500, "file_path": "file1.fa", "byte_offset": 0},
            ...
        ]
    }
    """
    # Check cache validity
    if index_path.exists():
        with open(index_path) as f:
            cached = json.load(f)

        # Validate mtimes match
        valid = True
        for fasta_file in fasta_files:
            cached_mtime = cached.get('fasta_mtimes', {}).get(str(fasta_file))
            if cached_mtime != fasta_file.stat().st_mtime:
                valid = False
                break

        if valid:
            logger.info(f"Using cached index: {index_path}")
            return index_path

    # Build new index
    logger.info(f"Building sequence index for {len(fasta_files)} files")
    entries = []
    mtimes = {}

    for fasta_file in fasta_files:
        mtimes[str(fasta_file)] = fasta_file.stat().st_mtime

        with open(fasta_file, 'rb') as f:
            byte_offset = 0
            current_id = None
            current_len = 0

            for line in f:
                if line.startswith(b'>'):
                    # Save previous entry
                    if current_id is not None:
                        entries.append(SequenceEntry(
                            sequence_id=current_id,
                            length=current_len,
                            file_path=str(fasta_file),
                            byte_offset=byte_offset
                        ))
                    # Start new sequence
                    current_id = line[1:].decode().split()[0]
                    byte_offset = f.tell() - len(line)
                    current_len = 0
                else:
                    current_len += len(line.strip())

            # Save last entry
            if current_id is not None:
                entries.append(SequenceEntry(
                    sequence_id=current_id,
                    length=current_len,
                    file_path=str(fasta_file),
                    byte_offset=byte_offset
                ))

    # Sort by length descending for optimal packing
    entries.sort(key=lambda e: e.length, reverse=True)

    # Write index
    index_data = {
        "version": "1.0",
        "created": datetime.now().isoformat(),
        "fasta_mtimes": mtimes,
        "total_sequences": len(entries),
        "total_tokens": sum(e.length for e in entries),
        "sequences": [asdict(e) for e in entries]
    }

    with open(index_path, 'w') as f:
        json.dump(index_data, f, indent=2)

    logger.info(f"Created index: {len(entries)} sequences, {sum(e.length for e in entries)} total tokens")
    return index_path

def get_worker_indices(index_path: Path, rank: int, world_size: int) -> List[int]:
    """
    Get sequence indices for worker using stride distribution.

    GPU rank N gets indices [N, N+world_size, N+2*world_size, ...]
    This ensures balanced length distribution since index is sorted by length.
    """
    with open(index_path) as f:
        index_data = json.load(f)

    total = len(index_data['sequences'])
    indices = list(range(rank, total, world_size))

    # Log distribution metrics
    worker_seqs = [index_data['sequences'][i] for i in indices]
    worker_tokens = sum(s['length'] for s in worker_seqs)

    logger.info(f"Worker {rank}/{world_size}: {len(indices)} sequences, {worker_tokens} tokens")

    return indices
```

### Pattern 3: HDF5 Chunk-Wise Aggregation

**What:** Merge shard files into single output using chunk-wise streaming
**When to use:** Aggregating GPU shard outputs without loading all into memory
**Example:**
```python
# Source: h5py documentation + CONTEXT.md decisions
import h5py
import numpy as np
from pathlib import Path
from typing import List, Set

CHUNK_SIZE = 10000  # Sequences per chunk (Claude's discretion)

def aggregate_shards(
    shard_files: List[Path],
    output_path: Path,
    expected_sequence_ids: Set[str]
) -> Path:
    """
    Aggregate HDF5 shards into single output with validation.

    Uses chunk-wise streaming to control memory usage.
    """
    # Count total sequences
    total_sequences = 0
    embedding_dim = None

    for shard in shard_files:
        with h5py.File(shard, 'r') as f:
            total_sequences += len(f['sequence_ids'])
            if embedding_dim is None:
                embedding_dim = f['embeddings'].shape[1]

    logger.info(f"Aggregating {len(shard_files)} shards: {total_sequences} sequences")

    # Create output file with resizable datasets
    with h5py.File(output_path, 'w') as out_f:
        # Pre-allocate datasets
        embeddings_ds = out_f.create_dataset(
            'embeddings',
            shape=(total_sequences, embedding_dim),
            dtype='float32',
            chunks=(min(CHUNK_SIZE, total_sequences), embedding_dim)
        )

        # Variable-length string dataset for sequence IDs
        dt = h5py.special_dtype(vlen=str)
        ids_ds = out_f.create_dataset(
            'sequence_ids',
            shape=(total_sequences,),
            dtype=dt
        )

        # Process shards sequentially, read in chunks
        write_offset = 0
        seen_ids: Set[str] = set()

        for shard in shard_files:
            with h5py.File(shard, 'r') as shard_f:
                shard_len = len(shard_f['sequence_ids'])

                # Read and write in chunks
                for chunk_start in range(0, shard_len, CHUNK_SIZE):
                    chunk_end = min(chunk_start + CHUNK_SIZE, shard_len)

                    # Read chunk from shard
                    emb_chunk = shard_f['embeddings'][chunk_start:chunk_end]
                    ids_chunk = shard_f['sequence_ids'][chunk_start:chunk_end]

                    # Check for duplicates
                    for seq_id in ids_chunk:
                        if seq_id in seen_ids:
                            raise ValueError(f"Duplicate sequence ID: {seq_id}")
                        seen_ids.add(seq_id)

                    # Write to output
                    embeddings_ds[write_offset:write_offset + len(emb_chunk)] = emb_chunk
                    ids_ds[write_offset:write_offset + len(ids_chunk)] = ids_chunk
                    write_offset += len(emb_chunk)

        # Validate completeness
        missing = expected_sequence_ids - seen_ids
        if missing:
            # Log first 10 missing IDs
            sample = list(missing)[:10]
            raise ValueError(
                f"Missing {len(missing)} sequences. "
                f"First 10: {sample}"
            )

        extra = seen_ids - expected_sequence_ids
        if extra:
            sample = list(extra)[:10]
            logger.warning(f"Found {len(extra)} unexpected sequences: {sample}")

    logger.info(f"Aggregation complete: {output_path}")
    return output_path
```

### Pattern 4: Per-Worker Logging

**What:** Each worker writes to separate log file to avoid interleaving
**When to use:** Always for multi-GPU workers to enable debugging
**Example:**
```python
# Source: Python logging cookbook + CONTEXT.md decisions
import logging
from pathlib import Path

def setup_worker_logging(rank: int, log_dir: Path, log_level: int = logging.INFO):
    """
    Configure per-worker logging to separate files.

    Each worker gets: worker_{rank}.log
    """
    log_file = log_dir / f"worker_{rank}.log"

    # Remove any existing handlers
    logger = logging.getLogger()
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create file handler for this worker
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)

    formatter = logging.Formatter(
        f'%(asctime)s - Worker {rank} - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.setLevel(log_level)

    # Also add console handler for immediate feedback
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)  # Only warnings/errors to console
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    logger.info(f"Worker {rank} logging initialized: {log_file}")
```

### Anti-Patterns to Avoid

- **Using `torch.multiprocessing.spawn` for fault tolerance:** spawn kills ALL workers when any worker fails. Use `multiprocessing.Process` directly for partial failure handling
- **Using `multiprocessing.Pool` with GPU workers:** Pool can hang indefinitely if a worker dies. Process gives direct control over lifecycle
- **Shared log files across workers:** Log interleaving makes debugging impossible. Use per-worker log files
- **Loading entire shards into memory for aggregation:** Use chunk-wise reading to control memory usage
- **Sequential position IDs in index file:** Index is already sorted by length; stride distribution ensures balanced distribution
- **Pickle for index files:** JSON is human-readable and debuggable; index contains simple metadata

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| CUDA-safe process creation | Custom fork logic | `multiprocessing.get_context('spawn')` | Spawn context prevents CUDA context inheritance |
| Atomic file writes | Manual temp-then-rename | Existing `checkpoint.atomic_save()` | Already handles validation, .done markers |
| GPU monitoring | Custom nvidia-smi parsing | Existing `NvitopMonitor` | Already integrated in async_inference.py |
| Stream-based GPU inference | New implementation | Existing `AsyncInferenceRunner` | Phase 5/6 provides complete single-GPU pipeline |
| HDF5 chunk reading | Manual offset calculations | `h5py.Dataset[start:end]` slicing | HDF5 handles chunk boundaries automatically |

**Key insight:** Phase 7 orchestrates existing single-GPU infrastructure (Phase 5/6). The new code is coordination layer, not inference reimplementation.

## Common Pitfalls

### Pitfall 1: mp.spawn Kills All Workers on Any Failure

**What goes wrong:** One GPU OOM/crash terminates entire job, losing hours of progress
**Why it happens:** `torch.multiprocessing.spawn` is designed for DDP where all workers must sync - any failure means restart
**How to avoid:**
- Use `multiprocessing.Process` directly with spawn context
- Monitor each process independently via `is_alive()` and `exitcode`
- Collect partial results from successful workers
- Log clear warnings about which shards are missing
**Warning signs:** "Entire job failed" when only one GPU had issues

### Pitfall 2: CUDA Device Assignment Race Condition

**What goes wrong:** Multiple workers see same GPU, causing contention
**Why it happens:** Setting `CUDA_VISIBLE_DEVICES` after CUDA initialization has no effect
**How to avoid:**
- Set `CUDA_VISIBLE_DEVICES` in parent BEFORE spawning (via Process env arg)
- Each worker sees device 0 (remapped by CUDA_VISIBLE_DEVICES)
- Never call `torch.cuda.*` in parent before spawning workers
**Warning signs:** Multiple workers on same GPU; GPU memory usage 2x expected

### Pitfall 3: HDF5 File Left Open During Aggregation

**What goes wrong:** Aggregation crashes leave partially written files; subsequent runs fail
**Why it happens:** h5py File objects not properly closed on exception
**How to avoid:**
- Always use context managers: `with h5py.File(...) as f:`
- Never close and reopen same file within aggregation
- Delete partial output file in exception handler before re-raising
**Warning signs:** "Unable to open file" errors; truncated output files

### Pitfall 4: Index Mtime Race Condition

**What goes wrong:** FASTA file modified during indexing; stale index used
**Why it happens:** Mtime checked before indexing, file modified during, cached mtime is wrong
**How to avoid:**
- Record mtimes AFTER indexing completes
- Or check mtime both before AND after, fail if changed
- Consider file locking for production (out of scope for v2.0)
**Warning signs:** Sequence count mismatch between index and actual files

### Pitfall 5: Worker Log Files Overwritten on Resume

**What goes wrong:** Previous run's logs lost when resuming after partial failure
**Why it happens:** Log files opened in write mode, not append
**How to avoid:**
- Use append mode for log files: `FileHandler(log_file, mode='a')`
- Add timestamp separator when resuming
- Or use unique log filenames per run
**Warning signs:** Debugging failed runs impossible; logs show only successful portion

## Code Examples

### Complete GPU Worker Function

```python
# Source: Combining existing codebase patterns with CONTEXT.md decisions
import os
import sys
import logging
import torch
from pathlib import Path
from typing import Optional
from multiprocessing import Queue

def gpu_worker(
    rank: int,
    world_size: int,
    index_path: Path,
    output_dir: Path,
    results_queue: Queue
):
    """
    Independent GPU worker for shard processing.

    This is the worker function spawned by GPUProcessCoordinator.
    Each worker:
    1. Sets up per-worker logging
    2. Loads sequence index and gets assigned indices
    3. Runs async inference pipeline (Phase 5/6)
    4. Saves shard HDF5 file
    5. Reports completion to parent
    """
    # CRITICAL: CUDA_VISIBLE_DEVICES already set by parent
    # Worker sees device 0 which maps to actual GPU {rank}

    try:
        # Setup per-worker logging
        log_dir = output_dir / "logs"
        log_dir.mkdir(exist_ok=True)
        setup_worker_logging(rank, log_dir)

        logger = logging.getLogger(f"worker_{rank}")
        logger.info(f"Worker {rank}/{world_size} starting on GPU (visible as device 0)")

        # Initialize CUDA
        device = torch.device('cuda:0')  # Always 0 - remapped by CUDA_VISIBLE_DEVICES
        torch.cuda.set_device(device)

        # Load index and get assigned sequences
        indices = get_worker_indices(index_path, rank, world_size)
        logger.info(f"Assigned {len(indices)} sequences")

        # Load model using existing infrastructure
        from virnucpro.models.esm2_flash import load_esm2_model
        model, batch_converter = load_esm2_model(
            model_name="esm2_t36_3B_UR50D",
            device=str(device),
            logger_instance=logger
        )

        # Create index-based dataset (new capability)
        from virnucpro.data.sequence_dataset import IndexBasedDataset
        dataset = IndexBasedDataset(index_path, indices)

        # Create dataloader with existing infrastructure
        from virnucpro.data.collators import VarlenCollator
        from virnucpro.data.dataloader_utils import create_async_dataloader

        collator = VarlenCollator(batch_converter)
        dataloader = create_async_dataloader(
            dataset, collator,
            device_id=0  # Always 0 due to CUDA_VISIBLE_DEVICES
        )

        # Run inference using existing async runner
        from virnucpro.pipeline.async_inference import AsyncInferenceRunner
        runner = AsyncInferenceRunner(model, device)

        # Collect results
        all_embeddings = []
        all_ids = []

        for result in runner.run(dataloader):
            all_embeddings.append(result.embeddings)
            all_ids.extend(result.sequence_ids)

        # Save shard
        shard_path = output_dir / f"shard_{rank}.h5"
        with h5py.File(shard_path, 'w') as f:
            embeddings = torch.cat(all_embeddings, dim=0).numpy()
            f.create_dataset('embeddings', data=embeddings)

            dt = h5py.special_dtype(vlen=str)
            f.create_dataset('sequence_ids', data=all_ids, dtype=dt)

        logger.info(f"Worker {rank} complete: {len(all_ids)} sequences saved to {shard_path}")

        # Report success
        results_queue.put({
            'rank': rank,
            'status': 'complete',
            'shard_path': str(shard_path),
            'num_sequences': len(all_ids)
        })

    except Exception as e:
        logger.exception(f"Worker {rank} failed: {e}")
        results_queue.put({
            'rank': rank,
            'status': 'failed',
            'error': str(e)
        })
        sys.exit(1)
```

### Parent Orchestration with Partial Failure Handling

```python
# Source: CONTEXT.md architecture + Python multiprocessing patterns
def run_multi_gpu_inference(
    fasta_files: List[Path],
    output_dir: Path,
    world_size: Optional[int] = None
) -> Tuple[Path, List[int]]:
    """
    Run multi-GPU inference with fault tolerance.

    Returns:
        Tuple of (merged_output_path, failed_ranks)
    """
    if world_size is None:
        world_size = torch.cuda.device_count()

    logger.info(f"Starting multi-GPU inference: {world_size} GPUs")

    # Create/validate sequence index
    index_path = output_dir / "sequence_index.json"
    create_sequence_index(fasta_files, index_path)

    # Load expected sequence IDs for validation
    with open(index_path) as f:
        index_data = json.load(f)
    expected_ids = {s['sequence_id'] for s in index_data['sequences']}

    # Spawn workers
    coordinator = GPUProcessCoordinator(world_size, output_dir)
    coordinator.spawn_workers(index_path, gpu_worker)

    # Wait for completion with timeout (e.g., 24 hours max)
    completion_status = coordinator.wait_for_completion(timeout=86400)

    # Identify successful and failed workers
    successful_ranks = [r for r, ok in completion_status.items() if ok]
    failed_ranks = [r for r, ok in completion_status.items() if not ok]

    if failed_ranks:
        logger.warning(
            f"Partial failure: {len(failed_ranks)}/{world_size} workers failed: {failed_ranks}\n"
            f"Successful workers: {successful_ranks}\n"
            f"Aggregating partial results from successful shards."
        )

    # Collect successful shards
    shard_files = [
        output_dir / f"shard_{rank}.h5"
        for rank in successful_ranks
        if (output_dir / f"shard_{rank}.h5").exists()
    ]

    if not shard_files:
        raise RuntimeError("No successful shards to aggregate")

    # Aggregate with partial validation
    # Expected IDs for successful workers only
    successful_expected = set()
    for rank in successful_ranks:
        indices = get_worker_indices(index_path, rank, world_size)
        for i in indices:
            successful_expected.add(index_data['sequences'][i]['sequence_id'])

    output_path = output_dir / "embeddings.h5"
    aggregate_shards(shard_files, output_path, successful_expected)

    # Report missing due to failures
    if failed_ranks:
        missing_count = sum(
            len(get_worker_indices(index_path, r, world_size))
            for r in failed_ranks
        )
        logger.warning(
            f"Missing {missing_count} sequences due to worker failures.\n"
            f"Failed ranks: {failed_ranks}\n"
            f"Investigate worker logs: logs/worker_{{rank}}.log"
        )

    return output_path, failed_ranks
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| File-level sharding | Index-based stride sharding | v2.0 Phase 7 | Balanced length distribution across GPUs |
| `multiprocessing.Pool` for GPU workers | `multiprocessing.Process` direct | v2.0 Phase 7 | Fault tolerance with partial results |
| `.pt` file aggregation | HDF5 chunk-wise merge | v2.0 Phase 7 | Memory-efficient for large outputs |
| Single shared log file | Per-worker log files | v2.0 Phase 7 | Debuggable multi-GPU logs |

**Deprecated/outdated:**
- `mp.spawn` for independent workers: Use Process directly for fault tolerance
- File-level GPU sharding: Creates skewed length distributions; use stride-based index sharding
- Loading entire shards for merge: Memory explosion; use chunk-wise streaming

## Open Questions

1. **HDF5 vs PyTorch .pt for shard format**
   - What we know: CONTEXT.md specifies HDF5 for v1.0 consistency
   - What's unclear: Current codebase uses .pt exclusively (grep found no h5py imports)
   - Recommendation: Implement HDF5 as specified; enables chunk-wise access. May need to add h5py dependency.

2. **Results queue vs PID polling for health monitoring**
   - What we know: Both are viable (Claude's discretion)
   - Queue advantage: Push-based, immediate notification
   - PID polling advantage: Simpler, no queue management
   - Recommendation: Use Queue for structured progress reporting; PID polling as backup for crash detection

3. **Exact parent progress logging interval**
   - What we know: Periodic summary, not real-time (Claude's discretion)
   - Recommendation: Every 60 seconds or 10% progress, whichever is less frequent

## Sources

### Primary (HIGH confidence)
- [PyTorch Multiprocessing Documentation](https://docs.pytorch.org/docs/stable/multiprocessing.html) - spawn context, Process usage
- [PyTorch Multiprocessing Best Practices](https://docs.pytorch.org/docs/stable/notes/multiprocessing.html) - CUDA safety patterns
- [h5py Datasets Documentation](https://docs.h5py.org/en/stable/high/dataset.html) - chunk-wise reading, resizable datasets
- [Python Logging Cookbook](https://docs.python.org/3/howto/logging-cookbook.html) - multiprocessing logging patterns
- Existing codebase: `virnucpro/pipeline/persistent_pool.py` - PersistentWorkerPool patterns
- Existing codebase: `virnucpro/pipeline/async_inference.py` - AsyncInferenceRunner for single-GPU

### Secondary (MEDIUM confidence)
- [Medium: How to Use PyTorch Multiprocessing](https://medium.com/@heyamit10/how-to-use-pytorch-multiprocessing-0ddd2014f4fd) - spawn patterns
- [SigNoz: How to Log Effectively When Using Multiprocessing](https://signoz.io/guides/how-should-i-log-while-using-multiprocessing-in-python/) - per-worker logging
- [Pickle vs JSON Performance](https://medium.com/@mishraarvind2222/pickle-vs-json-which-is-faster-in-python3-6b39b9010a99) - serialization comparison

### Tertiary (LOW confidence)
- [PyPI: pyfaidx](https://pypi.org/project/pyfaidx/) - FASTA indexing patterns (not directly used)
- [GitHub: Python Issue 22393](https://github.com/python/cpython/issues/66587) - Pool hanging on worker death

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - PyTorch multiprocessing, h5py are well-documented
- Architecture patterns: HIGH - Based on existing codebase + CONTEXT.md decisions
- Fault tolerance approach: HIGH - mp.spawn limitation verified in PyTorch docs
- HDF5 aggregation: MEDIUM - Pattern is standard, but HDF5 is new to codebase
- Pitfalls: HIGH - Common issues documented in PyTorch + Python issue trackers

**Research date:** 2026-02-04
**Valid until:** 60 days (stable technologies, multiprocessing patterns well-established)
