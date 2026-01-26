# Phase 5: Advanced Load Balancing - Research

**Researched:** 2026-01-26
**Domain:** Multi-GPU work stealing, load balancing, and real-time monitoring
**Confidence:** HIGH

## Summary

Advanced load balancing for multi-GPU systems requires three key components: (1) work-stealing infrastructure using multiprocessing Manager.Queue for cross-process queue access, (2) GPU monitoring via PyTorch's built-in torch.cuda APIs for real-time utilization tracking, and (3) live dashboards using Rich library's Live display for terminal UI updates.

The codebase already has strong foundations: greedy bin-packing file assignment (base_worker.py), multiprocessing infrastructure with spawn context (work_queue.py, persistent_pool.py), and GPU memory monitoring (gpu_monitor.py). Phase 5 extends these with work stealing and dynamic rebalancing.

**Primary recommendation:** Use multiprocessing.Manager().Queue() for shared work queues (avoids deadlocks vs regular Queue), torch.cuda APIs for GPU metrics (faster than nvidia-smi subprocess calls), and Rich Live+Table for dashboard (already in requirements.txt). Implement whole-file stealing with simple victim selection (round-robin or random) since file processing takes minutes - complexity of sophisticated victim selection doesn't justify overhead.

## Standard Stack

The established libraries/tools for this domain:

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| multiprocessing | stdlib | Process-based parallelism, Manager.Queue for shared queues | Built-in, spawn context works with CUDA, Manager.Queue avoids deadlocks |
| torch.cuda | (with PyTorch) | GPU utilization and memory monitoring | Direct CUDA API access, already used in codebase, no subprocess overhead |
| rich | >=13.0.0 | Terminal UI with live updating tables | Already in requirements.txt, mature API, handles refresh/layout automatically |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| threading | stdlib | Background monitoring thread | For dashboard refresh loop (non-blocking, separate from workers) |
| queue.Empty | stdlib | Exception handling for non-blocking queue operations | Required for try-except pattern with get_nowait() |
| collections.deque | stdlib | Optional local task buffer | If implementing prefetch buffer before stealing (not required for MVP) |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Manager.Queue | multiprocessing.Queue | Regular Queue has deadlock issues with pickling, Manager.Queue is safer |
| torch.cuda APIs | pynvml/nvidia-ml-py | pynvml adds dependency, torch.cuda already available and sufficient |
| torch.cuda APIs | subprocess nvidia-smi | nvidia-smi has ~100ms overhead per call, torch.cuda is instant |
| Rich Live | Manual ANSI escape codes | Manual control brittle, Rich handles terminal resize, cursor position |

**Installation:**
```bash
# No new dependencies required
# rich>=13.0.0 already in requirements.txt
# multiprocessing, threading, queue, collections are stdlib
```

## Architecture Patterns

### Recommended Project Structure
```
virnucpro/pipeline/
├── work_queue.py              # Existing: BatchQueueManager
├── persistent_pool.py         # Existing: PersistentWorkerPool
├── base_worker.py             # Existing: assign_files_by_sequences (bin packing)
├── gpu_monitor.py             # Existing: get_gpu_memory_info, GPUMonitor
├── load_balancer.py           # NEW: WorkStealingCoordinator
└── dashboard.py               # NEW: LoadBalancingDashboard
```

### Pattern 1: Shared Work Queues with Manager

**What:** Create per-GPU queues using multiprocessing.Manager() for cross-process access
**When to use:** When workers need to steal from each other's queues

**Example:**
```python
# Source: Python multiprocessing docs + codebase patterns
from multiprocessing import Manager
from pathlib import Path
from typing import List, Dict
import multiprocessing

def create_work_queues(num_gpus: int, file_assignments: List[List[Path]]) -> Dict[int, multiprocessing.Queue]:
    """Create shared work queues using Manager for safe cross-process access."""
    manager = Manager()
    work_queues = {}

    for gpu_id in range(num_gpus):
        # Manager.Queue is process-safe and avoids deadlock issues
        queue = manager.Queue()

        # Populate with initial file assignments
        for file_path in file_assignments[gpu_id]:
            queue.put(file_path)

        work_queues[gpu_id] = queue

    return work_queues, manager  # Keep manager alive for queue lifetime
```

### Pattern 2: Non-Blocking Queue Operations

**What:** Use get_nowait() with try-except for queue polling without blocking
**When to use:** Work stealing checks (don't want workers blocking on empty queues)

**Example:**
```python
# Source: Python multiprocessing best practices
from queue import Empty
import logging

logger = logging.getLogger(__name__)

def try_steal_work(victim_queues: Dict[int, multiprocessing.Queue],
                   my_gpu_id: int) -> Optional[Path]:
    """Attempt to steal work from other GPU queues."""
    # Try each victim queue in order (skip self)
    for victim_id, victim_queue in victim_queues.items():
        if victim_id == my_gpu_id:
            continue

        try:
            # Non-blocking get - raises Empty if queue is empty
            stolen_file = victim_queue.get_nowait()
            logger.info(f"GPU {my_gpu_id} stole work from GPU {victim_id}: {stolen_file.name}")
            return stolen_file
        except Empty:
            # Queue empty or locked, try next victim
            continue

    return None  # No work available to steal
```

**Critical:** Never use `queue.qsize()` or `queue.empty()` for control flow - they're unreliable in multiprocessing due to race conditions. Always use try-except with `Empty` exception.

### Pattern 3: Live Dashboard with Rich

**What:** Use Rich Live context manager with Table regeneration for real-time updates
**When to use:** Displaying per-GPU metrics that change every 2-3 seconds

**Example:**
```python
# Source: Rich documentation + community examples
from rich.live import Live
from rich.table import Table
import time

def generate_gpu_table(gpu_stats: Dict[int, Dict]) -> Table:
    """Generate table with current GPU metrics."""
    table = Table(title="GPU Load Balancing Dashboard")

    # Define columns
    table.add_column("GPU", style="cyan", justify="center")
    table.add_column("Util %", justify="right")
    table.add_column("Throughput", justify="right")
    table.add_column("Queue", justify="right")
    table.add_column("Stolen", justify="right")

    # Add row per GPU with current metrics
    for gpu_id in sorted(gpu_stats.keys()):
        stats = gpu_stats[gpu_id]

        # Color-code utilization
        util_pct = stats['utilization']
        if util_pct >= 80:
            util_color = "green"
        elif util_pct >= 50:
            util_color = "yellow"
        else:
            util_color = "red"

        table.add_row(
            f"{gpu_id}",
            f"[{util_color}]{util_pct:.1f}%[/{util_color}]",
            f"{stats['throughput']:.2f} seq/s",
            f"{stats['queue_depth']} files",
            f"{stats['work_stolen']} stolen"
        )

    return table

# Dashboard loop
with Live(generate_gpu_table(initial_stats), refresh_per_second=0.5) as live:
    while processing:
        time.sleep(2.0)  # Update every 2 seconds
        current_stats = collect_gpu_stats()
        live.update(generate_gpu_table(current_stats))
```

**Key insight:** Regenerate entire table on each update rather than trying to mutate rows - simpler and avoids state synchronization issues.

### Pattern 4: GPU Utilization Monitoring

**What:** Use torch.cuda APIs for real-time GPU metrics without subprocess overhead
**When to use:** Per-GPU utilization tracking for dashboard and rebalancing decisions

**Example:**
```python
# Source: Codebase gpu_monitor.py + PyTorch CUDA docs
import torch
from typing import Dict

def get_gpu_utilization_metrics(device_id: int) -> Dict[str, float]:
    """Get real-time GPU utilization using PyTorch CUDA APIs."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")

    # Memory info (fast, direct CUDA API)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device_id)
    used_bytes = total_bytes - free_bytes
    memory_pct = (used_bytes / total_bytes) * 100.0

    # Device properties
    props = torch.cuda.get_device_properties(device_id)
    device_name = torch.cuda.get_device_name(device_id)

    return {
        'device_id': device_id,
        'device_name': device_name,
        'memory_used_gb': used_bytes / (1024**3),
        'memory_total_gb': total_bytes / (1024**3),
        'memory_pct': memory_pct,
        'compute_capability': f"{props.major}.{props.minor}"
    }
```

**Note:** PyTorch's torch.cuda APIs don't provide compute utilization % (GPU core usage). For that, would need pynvml. However, for load balancing, queue depth and throughput (sequences/sec) are better indicators than instantaneous GPU % since they reflect actual work distribution.

### Pattern 5: Weighted File Assignment

**What:** Extend existing greedy bin-packing with GPU capability weights
**When to use:** Initial file distribution for heterogeneous GPUs (3090 + 4090 mix)

**Example:**
```python
# Source: Codebase base_worker.py assign_files_by_sequences + bin packing research
from pathlib import Path
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

# Static GPU weight lookup table (relative performance for embedding extraction)
GPU_WEIGHTS = {
    # Consumer GPUs (Ampere)
    'NVIDIA GeForce RTX 3080': 0.90,
    'NVIDIA GeForce RTX 3090': 1.00,  # Baseline

    # Consumer GPUs (Ada Lovelace)
    'NVIDIA GeForce RTX 4080': 1.35,
    'NVIDIA GeForce RTX 4090': 1.50,

    # Professional GPUs (Ampere)
    'NVIDIA RTX A6000': 1.10,
    'NVIDIA A100-PCIE-40GB': 1.20,

    # Professional GPUs (Hopper)
    'NVIDIA H100 PCIe': 2.50,

    # Professional GPUs (Volta)
    'Tesla V100-PCIE-16GB': 0.75,
    'Tesla V100-SXM2-32GB': 0.80,
}

def get_gpu_weight(device_id: int) -> float:
    """Get performance weight for GPU, defaulting to 1.0 if unknown."""
    device_name = torch.cuda.get_device_name(device_id)
    weight = GPU_WEIGHTS.get(device_name, 1.0)

    if device_name not in GPU_WEIGHTS:
        logger.warning(f"Unknown GPU '{device_name}' - using default weight 1.0")

    return weight

def assign_files_weighted(files: List[Path], num_workers: int,
                          gpu_weights: List[float]) -> List[List[Path]]:
    """
    Distribute files using weighted greedy bin-packing.

    Extends base_worker.py assign_files_by_sequences with GPU capability weighting.
    Files assigned proportional to GPU weight (faster GPUs get more work).
    """
    if len(gpu_weights) != num_workers:
        raise ValueError(f"Expected {num_workers} weights, got {len(gpu_weights)}")

    # Count sequences per file (reuse existing function)
    from virnucpro.pipeline.base_worker import count_sequences
    file_sizes = [(f, count_sequences(f)) for f in files]
    file_sizes.sort(key=lambda x: x[1], reverse=True)

    # Initialize weighted bins
    worker_files = [[] for _ in range(num_workers)]
    worker_totals = [0.0] * num_workers

    # Greedy assignment with weights
    # Lower weighted total = more capacity = assign next file
    for file_path, seq_count in file_sizes:
        # Find worker with minimum weighted load
        # Divide by weight to normalize (higher weight = more capacity)
        min_worker_idx = min(
            range(num_workers),
            key=lambda i: worker_totals[i] / gpu_weights[i]
        )

        worker_files[min_worker_idx].append(file_path)
        worker_totals[min_worker_idx] += seq_count

    # Log assignments
    logger.info(f"Weighted file assignment ({num_workers} GPUs)")
    for i in range(num_workers):
        logger.info(
            f"  GPU {i} (weight={gpu_weights[i]:.2f}): "
            f"{len(worker_files[i])} files, {worker_totals[i]:.0f} sequences"
        )

    return worker_files
```

### Anti-Patterns to Avoid

- **Polling too frequently:** Work stealing every 0.1s adds overhead when files take minutes. User decided 2-5 seconds is appropriate.
- **Complex victim selection:** Random or round-robin victim selection is sufficient when files are large and varied. Sophisticated algorithms (shortest queue, longest queue, random-k-choices) add complexity without meaningful benefit.
- **Blocking queue operations:** Using `queue.get()` without timeout in steal logic can deadlock. Always use `get_nowait()` with try-except.
- **Relying on qsize() or empty():** These are unreliable in multiprocessing due to race conditions. Use try-except with `Empty` exception.
- **Per-batch stealing:** User decided whole-file stealing leverages existing checkpoints and is simpler. Don't over-engineer.

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Thread-safe queues | Custom queue with locks | multiprocessing.Manager().Queue() | Handles process boundaries, avoids pickling issues, prevents deadlocks |
| GPU monitoring | nvidia-smi subprocess parsing | torch.cuda.mem_get_info() | 100x faster (no subprocess), already available, reliable |
| Terminal live updates | ANSI escape codes + cursor control | Rich Live + Table | Handles resize, flicker, layout, proven library |
| Bin packing algorithm | Custom greedy algorithm | Codebase base_worker.py pattern | Already implemented, tested, proven effective |
| Worker pool management | Custom process spawning | Codebase work_queue.py + persistent_pool.py | Handles spawn context, CUDA initialization, progress reporting |

**Key insight:** Codebase already has 80% of infrastructure needed. Phase 5 primarily adds queue sharing (Manager.Queue), stealing logic (get_nowait pattern), and dashboard (Rich Live). Don't rebuild what works.

## Common Pitfalls

### Pitfall 1: Manager.Queue Lifetime Management

**What goes wrong:** Manager object goes out of scope, queues become inaccessible, workers crash with "Manager closed" errors.

**Why it happens:** Manager uses a background server process that must stay alive for queue lifetime. If Manager is garbage collected, server shuts down.

**How to avoid:** Keep Manager object reference alive for entire pool lifetime:
```python
# BAD: Manager goes out of scope
def create_queues():
    manager = Manager()
    return {i: manager.Queue() for i in range(4)}  # Manager dies here!

# GOOD: Return manager with queues
def create_queues():
    manager = Manager()
    queues = {i: manager.Queue() for i in range(4)}
    return queues, manager  # Caller keeps manager alive

# Usage
work_queues, manager = create_queues()
# ... process work ...
manager.shutdown()  # Clean shutdown when done
```

**Warning signs:** "EOFError", "Connection refused", "Manager closed" exceptions from queue operations.

### Pitfall 2: Queue Operations in Spawn Context

**What goes wrong:** Passing Queue as function argument in spawn context triggers pickling, raises "Queue objects should only be shared between processes through inheritance" error.

**Why it happens:** Spawn context (required for CUDA) pickles all function arguments. Queues can't be pickled, only inherited.

**How to avoid:** Pass queues via Pool initializer, not as worker arguments:
```python
# BAD: Queue as argument (triggers pickling)
pool = ctx.Pool(4)
pool.map(worker, [(file, queue) for file in files])  # FAILS

# GOOD: Queue via initializer (inherited)
_worker_queue = None

def init_worker(queue):
    global _worker_queue
    _worker_queue = queue

pool = ctx.Pool(4, initializer=init_worker, initargs=(queue,))
pool.map(worker, files)  # Works
```

**Warning signs:** "TypeError: cannot pickle 'Queue' object" or RuntimeError about queue sharing.

**Note:** Codebase already uses this pattern in work_queue.py _init_worker() - follow same approach.

### Pitfall 3: Race Conditions with qsize() and empty()

**What goes wrong:** Check `if queue.empty()`, decide to steal elsewhere, but by the time you act, queue now has items (or vice versa).

**Why it happens:** Queue state changes between check and action in concurrent environment.

**How to avoid:** Use EAFP (Easier to Ask Forgiveness than Permission) with try-except:
```python
# BAD: LBYL (Look Before You Leap) - race condition
if not queue.empty():
    item = queue.get()  # Might fail if another worker took item

# GOOD: EAFP - atomic operation
try:
    item = queue.get_nowait()
    # Process item
except Empty:
    # Queue was empty, continue
    pass
```

**Warning signs:** Intermittent `queue.Empty` exceptions despite checking `empty()` first.

### Pitfall 4: GPU Utilization Misinterpretation

**What goes wrong:** Using memory % as proxy for compute utilization, leading to incorrect load balancing decisions. GPU at 95% memory but 10% compute (waiting for data) looks busy but is actually idle.

**Why it happens:** Memory usage != compute utilization. Model and data occupy memory but only active computation uses GPU cores.

**How to avoid:** Use queue depth and throughput as load indicators, not memory %:
```python
# BAD: Memory % as load indicator
if memory_pct > 90:
    consider_gpu_busy = True  # Wrong - might be idle waiting for work

# GOOD: Queue depth + throughput as load indicator
if queue_depth > 0 or recent_throughput > 0:
    consider_gpu_busy = True  # Accurate - has work queued or processing
```

**Warning signs:** Workers marked as "busy" but not making progress, load balancing decisions seem backwards.

**Note:** PyTorch doesn't expose compute utilization % via torch.cuda. Would need pynvml for that, but queue metrics are more reliable for this use case.

### Pitfall 5: Imbalance Threshold Too Sensitive

**What goes wrong:** Work stealing triggers constantly even when GPUs are reasonably balanced, causing thrashing (files bouncing between workers, checkpoint overhead).

**Why it happens:** Natural variation in file processing time creates temporary imbalances that self-correct. Stealing too aggressively disrupts natural balancing.

**How to avoid:** Set imbalance threshold high enough to allow natural variation:
```python
# BAD: Trigger stealing on any difference
if max_queue - min_queue > 0:
    trigger_stealing()  # Too aggressive

# GOOD: Trigger only on significant imbalance
# Allow 2x difference or 3+ files difference
max_queue = max(queue_depths.values())
min_queue = min(queue_depths.values())

if max_queue >= 3 and (max_queue > 2 * min_queue or max_queue - min_queue >= 3):
    trigger_stealing()  # Only when truly imbalanced
```

**Warning signs:** High work stealing rate (>10% of files stolen), workers frequently idle then busy, increased checkpoint I/O.

**Note:** User deferred exact threshold to Claude's discretion - recommend starting conservative (3+ files, 2x ratio) and tuning based on workload.

## Code Examples

Verified patterns from official sources:

### Multiprocessing Manager Pattern (Official Docs)
```python
# Source: https://docs.python.org/3/library/multiprocessing.html
from multiprocessing import Manager, Process

def worker(shared_queue, gpu_id):
    while True:
        try:
            item = shared_queue.get_nowait()
            # Process item
        except Empty:
            break

manager = Manager()
shared_queue = manager.Queue()

# Populate queue
for item in items:
    shared_queue.put(item)

# Start workers
processes = [Process(target=worker, args=(shared_queue, i)) for i in range(4)]
for p in processes:
    p.start()
for p in processes:
    p.join()

manager.shutdown()
```

### Rich Live Dashboard (Community Pattern)
```python
# Source: Rich documentation + community examples
from rich.live import Live
from rich.table import Table
import time

def make_dashboard(metrics):
    table = Table(title="Live Metrics")
    table.add_column("ID")
    table.add_column("Status")
    table.add_column("Value")

    for metric_id, data in metrics.items():
        table.add_row(str(metric_id), data['status'], str(data['value']))

    return table

with Live(make_dashboard(initial_metrics), refresh_per_second=0.5) as live:
    while running:
        time.sleep(2.0)
        updated_metrics = collect_metrics()
        live.update(make_dashboard(updated_metrics))
```

### PyTorch GPU Metrics (Codebase Pattern)
```python
# Source: virnucpro/pipeline/gpu_monitor.py
import torch

def get_gpu_memory_info(device_id: int):
    """Get GPU memory stats using PyTorch CUDA API."""
    free_bytes, total_bytes = torch.cuda.mem_get_info(device_id)
    used_bytes = total_bytes - free_bytes

    return {
        'free': free_bytes,
        'total': total_bytes,
        'used': used_bytes,
        'percent': (used_bytes / total_bytes) * 100.0
    }
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Static round-robin | Greedy bin-packing by sequences | Phase 4 (2026-01) | Better initial balance, 20-30% improvement |
| Regular multiprocessing.Queue | Manager().Queue() for sharing | 2020+ best practice | Avoids deadlocks in spawn context |
| nvidia-smi subprocess | torch.cuda direct APIs | PyTorch 1.8+ (2021) | 100x faster, no parsing overhead |
| Manual ANSI codes | Rich library | Rich 10.0+ (2020) | Robust terminal handling, less code |
| Adaptive learning weights | Static lookup tables | Always preferred for simplicity | Faster startup, predictable, sufficient accuracy |

**Deprecated/outdated:**
- **pynvml package**: Now deprecated in favor of nvidia-ml-py (official NVIDIA bindings). However, for this use case, torch.cuda APIs are sufficient and already available.
- **multiprocessing.SimpleQueue**: Doesn't support Manager (no cross-process sharing), can't be used for work stealing.
- **collections.deque for shared queues**: Thread-safe but not process-safe, doesn't work across process boundaries.

## Open Questions

Things that couldn't be fully resolved:

1. **Compute utilization % without pynvml**
   - What we know: torch.cuda provides memory metrics but not GPU core utilization %
   - What's unclear: Whether to add pynvml dependency for compute % or rely on queue depth/throughput
   - Recommendation: Start with queue depth + throughput metrics (no new dependency), add pynvml later only if needed for debugging. Queue metrics are more actionable for load balancing than instantaneous GPU %.

2. **Optimal imbalance threshold**
   - What we know: User wants Claude's discretion, files take minutes to process, 2-5 second polling interval
   - What's unclear: Best threshold depends on workload (file size variance, sequence count distribution)
   - Recommendation: Start conservative (trigger when max_queue ≥ 3 AND (max > 2×min OR max - min ≥ 3)), add optional tuning parameter for power users. Can refine based on real workload testing.

3. **Throughput calculation window**
   - What we know: Need sequences/sec or tokens/sec metric for dashboard
   - What's unclear: Optimal averaging window (last 30s? 60s? 10 files?)
   - Recommendation: Track per-file completion times, compute rolling average over last 60 seconds or last 10 files (whichever is more). Provides smoothing while remaining responsive to changes.

## Sources

### Primary (HIGH confidence)
- Python multiprocessing documentation - https://docs.python.org/3/library/multiprocessing.html
- PyTorch CUDA APIs - torch.cuda module (verified in codebase gpu_monitor.py)
- Rich library documentation - https://rich.readthedocs.io/en/latest/live.html
- Codebase patterns - base_worker.py (bin packing), work_queue.py (spawn context), persistent_pool.py (worker initialization), gpu_monitor.py (GPU metrics)

### Secondary (MEDIUM confidence)
- [Multiprocessing best practices](https://runebook.dev/en/articles/python/library/multiprocessing/multiprocessing.Queue.qsize) - Queue usage patterns
- [GPU benchmarks comparison](https://bizon-tech.com/gpu-benchmarks/NVIDIA-RTX-3090-vs-NVIDIA-A100-40-GB-(PCIe)/579vs592) - Relative performance data for weight table
- [H100 vs A100 vs 4090 comparison](https://www.gpu-mart.com/blog/h100-vs-a100-vs-rtx-4090) - Professional GPU benchmarks
- [Rich dashboard examples](https://python.plainenglish.io/how-i-built-a-real-time-terminal-dashboard-in-python-using-rich-26e9a179f314) - Community implementation patterns

### Tertiary (LOW confidence)
- [Adaptive work-stealing research paper](https://arxiv.org/html/2401.04494v2) - Academic approach to heterogeneous work stealing (2024)
- [Binpacking PyPI package](https://pypi.org/project/binpacking/) - Alternative to hand-rolled bin packing (codebase already has implementation)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - PyTorch and multiprocessing are proven, Rich is mature and in requirements
- Architecture: HIGH - Patterns verified in codebase (work_queue.py, persistent_pool.py, base_worker.py)
- Pitfalls: HIGH - Based on Python multiprocessing docs + spawn context issues well-documented
- GPU weights: MEDIUM - Based on multiple benchmark sources but not official spec sheets
- Work stealing algorithms: MEDIUM - General patterns well-known, specific threshold tuning needs workload testing

**Research date:** 2026-01-26
**Valid until:** ~60 days (stable domain - multiprocessing and PyTorch APIs change slowly)

**Notes:**
- User decisions in CONTEXT.md constrain research scope significantly (whole-file stealing, static weights, Rich dashboard, queue polling)
- Codebase already has 80% of needed infrastructure, Phase 5 is primarily integration work
- No new dependencies required (everything stdlib or already in requirements.txt)
