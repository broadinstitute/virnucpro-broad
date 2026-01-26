# Phase 6: Performance Validation - Research

**Researched:** 2026-01-26
**Domain:** GPU deep learning benchmarking, multi-GPU scaling validation, performance profiling
**Confidence:** HIGH

## Summary

Performance validation for multi-GPU deep learning pipelines requires a comprehensive benchmarking infrastructure that combines automated testing, detailed performance metrics collection, and correctness validation. The standard approach uses PyTorch's built-in profiling tools (`torch.profiler`, `torch.utils.benchmark`) paired with GPU monitoring utilities (nvitop preferred over nvidia-smi for programmatic access) to measure throughput, GPU utilization, and memory usage across different GPU configurations.

The codebase already has foundational components: `gpu_monitor.py` for memory tracking, `profiler.py` for batch size optimization, and `test_integration_multi_gpu.py` for equivalence testing. This phase extends these with comprehensive benchmarking suites, scaling validation (1/2/4/8 GPUs), and dual-format reporting (markdown for humans, JSON for CI).

Key insight: Near-linear scaling (2 GPUs ≈ 1.8x, 4 GPUs ≈ 3.5x, 8 GPUs ≈ 7x) is achievable for embedding workloads when communication overhead is minimized. The main bottlenecks to monitor are data loading (CPU starvation), memory fragmentation, and GPU utilization gaps during stage transitions.

**Primary recommendation:** Build pytest-based benchmark suite using `torch.utils.benchmark.Timer` for accurate CUDA timing, nvitop's programmatic API for real-time monitoring, and numpy.allclose with rtol=1e-3 for BF16/FP32 numerical comparison. Generate both markdown reports and JSON output for CI integration.

## Standard Stack

The established tools for GPU benchmarking and performance validation in PyTorch:

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| torch.profiler | PyTorch 2.9+ | GPU/CPU profiling, memory tracking, bottleneck detection | Official PyTorch tool, zero-overhead profiling with CUDA events |
| torch.utils.benchmark | PyTorch 2.9+ | Accurate timing with automatic CUDA sync | Handles warmup, synchronization, statistical analysis automatically |
| nvitop | Latest (pip) | GPU monitoring and metrics collection | Python API for programmatic access, superior to nvidia-smi parsing |
| pytest-benchmark | 5.2.3+ | Benchmark fixture for automated testing | CI integration, JSON output, regression tracking |
| numpy | Already in deps | Numerical comparison with tolerance | Standard for allclose comparisons with rtol/atol |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| BioPython | Already in deps | Synthetic FASTA generation | Creating controlled test data with varying sequence counts |
| pandas | Already in deps | Results aggregation and comparison | Loading/comparing prediction outputs, generating reports |
| rich | Already in deps | Human-readable console output | Progress bars, formatted tables during benchmark runs |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| nvitop | nvidia-smi subprocess | nvidia-smi requires parsing text output, higher overhead, no Python API |
| torch.profiler | PyTorch Lightning Profiler | Lightning adds framework dependency, overkill for inference-only pipeline |
| pytest-benchmark | Custom timing code | Benchmark handles warmup, sync, statistics automatically |

**Installation:**
```bash
pip install nvitop pytest-benchmark
# torch, numpy, pandas, BioPython already in requirements.txt
```

## Architecture Patterns

### Recommended Project Structure
```
tests/
├── benchmarks/              # New: Benchmark suite
│   ├── __init__.py
│   ├── conftest.py         # pytest-benchmark configuration
│   ├── test_scaling.py     # Multi-GPU scaling benchmarks
│   ├── test_throughput.py  # Throughput benchmarks per stage
│   └── test_end_to_end.py  # Full pipeline timing
├── data/
│   ├── synthetic/          # Generated test data (gitignored)
│   ├── small_real/         # Committed small samples
│   └── reference/          # Pre-computed vanilla baselines
└── reports/                # Generated reports (gitignored)
    ├── benchmark_*.md      # Human-readable markdown
    └── benchmark_*.json    # Machine-readable for CI
```

### Pattern 1: PyTorch Benchmark Timer with CUDA Synchronization
**What:** Accurate GPU kernel timing that handles async execution
**When to use:** Any performance measurement of GPU operations
**Example:**
```python
# Source: https://docs.pytorch.org/docs/stable/benchmark_utils.html
import torch.utils.benchmark as benchmark

# Automatic CUDA synchronization and warmup
timer = benchmark.Timer(
    stmt='model(input_batch)',
    setup='torch.cuda.synchronize()',
    globals={'model': model, 'input_batch': batch}
)

# blocked_autorange: automatic warmup, multiple measurements
measurements = timer.blocked_autorange(min_run_time=10)
print(f"Median: {measurements.median:.4f}s")
```

**Why this matters:** CUDA operations are asynchronous. Without synchronization, timing only measures kernel launch time (~microseconds) not actual execution time (seconds). PyTorch's Timer handles this automatically.

### Pattern 2: PyTorch Profiler for Bottleneck Detection
**What:** Comprehensive profiling with GPU utilization, memory, and kernel traces
**When to use:** Identifying bottlenecks (data loading, memory, compute)
**Example:**
```python
# Source: https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html
from torch.profiler import profile, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    profile_memory=True,
    record_shapes=True,
    with_stack=True
) as prof:
    # Run one complete batch
    embeddings = model(sequences)

# Export for analysis
prof.export_chrome_trace("trace.json")

# Programmatic analysis
for event in prof.key_averages():
    if event.device_type == 'CUDA':
        print(f"{event.key}: {event.cuda_time_total/1000:.2f}ms "
              f"GPU mem: {event.cuda_memory_usage/1024**2:.1f}MB")
```

### Pattern 3: nvitop Programmatic Monitoring
**What:** Real-time GPU metrics collection via Python API
**When to use:** Continuous monitoring during long-running benchmarks
**Example:**
```python
# Source: https://nvitop.readthedocs.io/
from nvitop import Device, ResourceMetricCollector

# Monitor specific GPUs
devices = [Device(i) for i in [0, 1, 2, 3]]

# Collect metrics during benchmark
collector = ResourceMetricCollector(devices)
collector.start()

# ... run benchmark ...

collector.stop()
metrics = collector.get_metrics()

# Extract utilization and memory
for device_id, stats in metrics.items():
    print(f"GPU {device_id}: {stats['gpu_utilization_avg']:.1f}% util, "
          f"{stats['memory_used_avg']/1024**3:.2f}GB mem")
```

### Pattern 4: pytest-benchmark with JSON Output
**What:** Automated benchmark testing with CI integration
**When to use:** Regression tracking, automated performance validation
**Example:**
```python
# Source: https://pytest-benchmark.readthedocs.io/
def test_dnabert_throughput(benchmark):
    """Benchmark DNABERT-S embedding throughput"""

    def run_dnabert_batch():
        with torch.no_grad():
            embeddings = dnabert_model(input_tokens)
        torch.cuda.synchronize()
        return embeddings

    # benchmark fixture handles warmup, timing, statistics
    result = benchmark(run_dnabert_batch)

    # Assertions on performance
    assert result.stats['mean'] < 2.0  # < 2s per batch

# Run with JSON output for CI
# pytest tests/benchmarks/ --benchmark-json=results.json
```

### Pattern 5: Memory Profiling with Peak Tracking
**What:** Track peak and average GPU memory usage
**When to use:** Validating memory optimizations, sizing batches
**Example:**
```python
# Source: https://docs.pytorch.org/docs/stable/torch_cuda_memory.html
import torch

# Reset peak memory stats before test
torch.cuda.reset_peak_memory_stats(device)
torch.cuda.synchronize(device)

start_mem = torch.cuda.memory_allocated(device)

# Run operation
with torch.no_grad():
    output = model(input_batch)
torch.cuda.synchronize(device)

peak_mem = torch.cuda.max_memory_allocated(device)
end_mem = torch.cuda.memory_allocated(device)

print(f"Memory: {start_mem/1024**3:.2f}GB baseline")
print(f"Memory: {peak_mem/1024**3:.2f}GB peak (+{(peak_mem-start_mem)/1024**3:.2f}GB)")
print(f"Memory: {end_mem/1024**3:.2f}GB final")
```

### Pattern 6: Synthetic Data Generation for Scaling Tests
**What:** Generate FASTA files with controlled sequence counts for reproducible benchmarks
**When to use:** Testing scaling behavior, avoiding dependency on large external data
**Example:**
```python
# Source: Existing virnucpro/pipeline/profiler.py pattern
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
from Bio import SeqIO
import random

def generate_synthetic_fasta(num_sequences, seq_length, output_path):
    """Generate synthetic viral sequences for benchmarking"""
    bases = ['A', 'T', 'G', 'C']
    records = []

    for i in range(num_sequences):
        # Random DNA sequence
        seq = ''.join(random.choice(bases) for _ in range(seq_length))
        record = SeqRecord(
            Seq(seq),
            id=f"synthetic_{i:06d}",
            description=f"Synthetic test sequence {i}"
        )
        records.append(record)

    SeqIO.write(records, output_path, "fasta")
    return output_path

# Generate test sets with different sizes
generate_synthetic_fasta(10, 300, "tests/data/synthetic/tiny_10.fa")
generate_synthetic_fasta(100, 300, "tests/data/synthetic/small_100.fa")
generate_synthetic_fasta(1000, 300, "tests/data/synthetic/typical_1k.fa")
```

### Anti-Patterns to Avoid
- **Using time.time() for GPU ops:** CUDA is async, measures launch time not execution time. Use torch.cuda.synchronize() + Timer.
- **Single-run measurements:** GPU performance varies. Use multiple runs with statistical analysis (median, mean, std).
- **Ignoring warmup:** First run includes lazy initialization overhead. Always warmup before timing.
- **Parsing nvidia-smi output:** Brittle, high overhead. Use nvitop Python API or torch.cuda APIs.
- **Hard-coded absolute tolerances:** BF16/FP32 differences require relative tolerances (rtol). Use 1e-3 for mixed precision.

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| GPU timing | time.time() wrapper | torch.utils.benchmark.Timer | Handles CUDA sync, warmup, statistical analysis automatically |
| GPU monitoring | nvidia-smi parsing | nvitop Python API | Native Python bindings, no subprocess overhead, real-time metrics |
| Benchmark infrastructure | Custom test harness | pytest-benchmark | JSON output, CI integration, regression tracking built-in |
| Memory tracking | Manual tracking with globals | torch.cuda.max_memory_allocated() + profiler | Accurate PyTorch-internal tracking, no overhead |
| Numerical comparison | Manual threshold checks | numpy.allclose(rtol=1e-3) | Handles relative tolerance for mixed precision correctly |
| Performance reporting | Custom formatters | pandas DataFrame + markdown/JSON export | Standard formats, easy analysis and automation |

**Key insight:** GPU performance measurement has many pitfalls (async execution, warmup, statistical variance, synchronization). PyTorch's built-in tools handle these correctly. Don't reimplement timing logic.

## Common Pitfalls

### Pitfall 1: Measuring Launch Time Instead of Execution Time
**What goes wrong:** Using `time.time()` around GPU operations measures only how long it takes to launch the kernel (~10 microseconds), not how long the kernel actually runs (seconds).
**Why it happens:** CUDA operations are asynchronous. Python execution continues immediately after launching a kernel.
**How to avoid:** Always call `torch.cuda.synchronize(device)` before stopping timer, or use `torch.utils.benchmark.Timer` which does this automatically.
**Warning signs:**
- GPU-heavy operations reporting microsecond timing
- Adding more work doesn't increase reported time
- Timing shows 10-100x faster than expected

**Example:**
```python
# WRONG - measures launch time
start = time.time()
output = model(input_batch)  # Returns immediately, kernel still running
end = time.time()
print(f"Time: {end-start:.4f}s")  # Will show ~0.0001s

# CORRECT - measures execution time
start = time.time()
output = model(input_batch)
torch.cuda.synchronize()  # Wait for kernel to finish
end = time.time()
print(f"Time: {end-start:.4f}s")  # Will show actual time like 2.5s
```

### Pitfall 2: Ignoring Warmup Runs
**What goes wrong:** First GPU operation includes lazy initialization (model loading, CUDA context setup, cuDNN autotune), making first-run timings 10-100x slower than steady-state.
**Why it happens:** PyTorch and CUDA defer initialization until first use to avoid overhead in non-GPU code paths.
**How to avoid:** Always run 1-3 warmup iterations before starting timing. pytorch.utils.benchmark.Timer does this via `blocked_autorange()`.
**Warning signs:**
- First iteration much slower than subsequent ones
- Benchmark results vary wildly between runs
- Cold-start timing doesn't match production performance

### Pitfall 3: Data Loading Bottlenecks (CPU Starvation)
**What goes wrong:** GPU sits idle waiting for next batch because CPU preprocessing can't keep up. GPU utilization drops to 20-40% despite model compute capacity.
**Why it happens:** Multi-GPU setups increase data consumption rate. Single-threaded data loading becomes bottleneck.
**How to avoid:** Use DataLoader with num_workers=4-8, pin_memory=True, prefetch_factor=2. Monitor CPU utilization alongside GPU.
**Warning signs:**
- GPU utilization drops during loading phases
- Adding more GPUs doesn't improve throughput proportionally
- CPU at 100% while GPUs idle

**Detection:**
```python
# Use profiler to see CPU/GPU overlap
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    for batch in dataloader:
        output = model(batch)

# Look for gaps where GPU is idle waiting for data
prof.export_chrome_trace("trace.json")  # View in chrome://tracing
```

### Pitfall 4: Linear Scaling Expectation Without Considering Overhead
**What goes wrong:** Expecting perfect 2x speedup with 2 GPUs, getting 1.6x, and thinking optimization failed.
**Why it happens:** Communication overhead (scatter/gather), synchronization barriers, and Python GIL create inherent overhead that scales with GPU count.
**How to avoid:** Target 85-95% scaling efficiency (2 GPUs = 1.7-1.9x, 4 GPUs = 3.4-3.8x, 8 GPUs = 6.8-7.6x). Near-linear is excellent.
**Warning signs:**
- Declaring success only at perfect 2.0x speedup
- Not accounting for communication time in analysis
- Comparing small batch sizes where overhead dominates

**Realistic expectations from 2026 benchmarks:**
- 2 GPUs: 1.8-1.95x (90-97% efficiency)
- 4 GPUs: 3.5-3.8x (87-95% efficiency)
- 8 GPUs: 6.8-7.4x (85-92% efficiency)

### Pitfall 5: Incorrect Tolerance for BF16/FP32 Comparison
**What goes wrong:** Using default numpy.allclose() (rtol=1e-5, atol=1e-8) to compare BF16 and FP32 results. All tests fail despite correct implementation.
**Why it happens:** BF16 has ~3 decimal digits precision vs FP32's 7 digits. Default tolerances assume high precision.
**How to avoid:** Use rtol=1e-3 (0.1% relative error) for mixed precision comparison. BF16 machine epsilon is ~0.008.
**Warning signs:**
- Predictions match (same argmax) but scores fail comparison
- Differences are ~0.001-0.01 magnitude
- Only low-magnitude values pass tolerance

**Correct tolerance:**
```python
# BF16 machine epsilon ~= 2^-7 ~= 0.008
# Use rtol=1e-3 to allow ~0.1% relative difference
np.allclose(optimized_scores, vanilla_scores, rtol=1e-3, atol=1e-3)
```

### Pitfall 6: Benchmark Dataset Too Small
**What goes wrong:** Testing with 10 sequences shows worse multi-GPU performance than single GPU due to overhead exceeding compute time.
**Why it happens:** Multi-GPU overhead (process spawn, model replication, scatter/gather) is fixed cost. Small datasets don't amortize this.
**How to avoid:** Use representative dataset sizes (1000+ sequences for scaling tests). Test multiple scales: small (overhead-dominated), typical (target), large (memory-limited).
**Warning signs:**
- Multi-GPU slower than single GPU
- Adding GPUs makes total time worse
- Profiler shows majority of time in communication/setup

## Code Examples

Verified patterns from official sources:

### Multi-GPU Scaling Benchmark
```python
# Source: Pattern from existing test_integration_multi_gpu.py + torch.utils.benchmark
import torch.utils.benchmark as benchmark
from pathlib import Path
import subprocess
import json

def benchmark_scaling(test_data_path, gpu_configs):
    """
    Benchmark pipeline across different GPU configurations.

    Args:
        test_data_path: Path to test FASTA file
        gpu_configs: List of GPU configurations, e.g., ['0', '0,1', '0,1,2,3']

    Returns:
        Dict mapping config -> timing results
    """
    results = {}

    for gpu_config in gpu_configs:
        num_gpus = len(gpu_config.split(','))

        # Run pipeline with specific GPU config
        timer = benchmark.Timer(
            stmt='subprocess.run(cmd, check=True, capture_output=True)',
            setup='import subprocess',
            globals={
                'cmd': [
                    'virnucpro', 'predict',
                    str(test_data_path),
                    '--gpus', gpu_config,
                    '--output-dir', f'output_{num_gpus}gpu'
                ]
            }
        )

        # Single measurement (full pipeline is expensive)
        measurement = timer.timeit(1)

        results[gpu_config] = {
            'num_gpus': num_gpus,
            'time_seconds': measurement.mean,
            'speedup': None  # Calculated later vs baseline
        }

    # Calculate speedups relative to single GPU
    baseline_time = results[gpu_configs[0]]['time_seconds']
    for config in gpu_configs:
        time = results[config]['time_seconds']
        results[config]['speedup'] = baseline_time / time

    return results
```

### GPU Utilization Monitoring
```python
# Source: nvitop documentation + existing gpu_monitor.py pattern
from nvitop import Device
import threading
import time
from collections import defaultdict

class GPUUtilizationMonitor:
    """Monitor GPU utilization during benchmark execution"""

    def __init__(self, device_ids):
        self.device_ids = device_ids
        self.devices = [Device(i) for i in device_ids]
        self.running = False
        self.samples = defaultdict(list)

    def start(self):
        """Start monitoring in background thread"""
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop monitoring and return statistics"""
        self.running = False
        self.thread.join(timeout=2.0)
        return self.get_statistics()

    def _monitor_loop(self):
        """Background monitoring loop"""
        while self.running:
            for device in self.devices:
                # Query current utilization and memory
                gpu_util = device.gpu_utilization()  # Percent 0-100
                mem_used = device.memory_used()  # Bytes
                mem_total = device.memory_total()  # Bytes

                self.samples[device.index].append({
                    'gpu_util': gpu_util,
                    'mem_used': mem_used,
                    'mem_total': mem_total,
                    'timestamp': time.time()
                })

            time.sleep(0.5)  # Sample every 500ms

    def get_statistics(self):
        """Calculate average utilization and memory"""
        stats = {}
        for device_id, samples in self.samples.items():
            if not samples:
                continue

            gpu_utils = [s['gpu_util'] for s in samples]
            mem_useds = [s['mem_used'] for s in samples]

            stats[device_id] = {
                'gpu_util_avg': sum(gpu_utils) / len(gpu_utils),
                'gpu_util_min': min(gpu_utils),
                'gpu_util_max': max(gpu_utils),
                'mem_used_avg': sum(mem_useds) / len(mem_useds),
                'mem_used_peak': max(mem_useds),
                'sample_count': len(samples)
            }

        return stats

# Usage in benchmark
monitor = GPUUtilizationMonitor(device_ids=[0, 1, 2, 3])
monitor.start()

# ... run benchmark ...

stats = monitor.stop()
for device_id, metrics in stats.items():
    print(f"GPU {device_id}: {metrics['gpu_util_avg']:.1f}% avg util, "
          f"{metrics['mem_used_peak']/1024**3:.2f}GB peak mem")
```

### Throughput Measurement
```python
# Source: Existing profiler.py pattern + torch.utils.benchmark
import torch
import time

def measure_stage_throughput(model, dataloader, device, stage_name):
    """
    Measure throughput (sequences/second) for a pipeline stage.

    Args:
        model: PyTorch model
        dataloader: DataLoader with test sequences
        device: CUDA device
        stage_name: Name for reporting

    Returns:
        Dict with throughput metrics
    """
    model.eval()
    total_sequences = 0
    total_time = 0.0

    # Warmup
    warmup_batch = next(iter(dataloader))
    with torch.no_grad():
        _ = model(warmup_batch.to(device))
    torch.cuda.synchronize(device)

    # Measure
    torch.cuda.synchronize(device)
    start_time = time.time()

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            _ = model(batch)
            total_sequences += len(batch)

    torch.cuda.synchronize(device)
    total_time = time.time() - start_time

    throughput = total_sequences / total_time if total_time > 0 else 0

    return {
        'stage': stage_name,
        'total_sequences': total_sequences,
        'total_time_seconds': total_time,
        'throughput_seq_per_sec': throughput,
        'avg_time_per_sequence_ms': (total_time / total_sequences * 1000) if total_sequences > 0 else 0
    }
```

### Numerical Comparison with Appropriate Tolerance
```python
# Source: Existing test_integration_multi_gpu.py + numpy documentation
import numpy as np
import pandas as pd

def compare_predictions(optimized_results, vanilla_results, rtol=1e-3, atol=1e-3):
    """
    Compare prediction results between optimized and vanilla pipelines.

    Args:
        optimized_results: Path to optimized prediction_results.txt
        vanilla_results: Path to vanilla prediction_results.txt
        rtol: Relative tolerance (1e-3 for BF16/FP32)
        atol: Absolute tolerance

    Returns:
        Dict with comparison results
    """
    # Load results
    opt_df = pd.read_csv(optimized_results, sep='\t')
    van_df = pd.read_csv(vanilla_results, sep='\t')

    # Check sequence IDs match
    if not (opt_df['Sequence_ID'] == van_df['Sequence_ID']).all():
        return {'match': False, 'reason': 'Sequence IDs differ'}

    # Check predictions match
    predictions_match = (opt_df['Prediction'] == van_df['Prediction']).all()

    # Compare scores with tolerance
    score_cols = ['score1', 'score2']
    score_matches = {}
    max_diffs = {}

    for col in score_cols:
        opt_scores = pd.to_numeric(opt_df[col])
        van_scores = pd.to_numeric(van_df[col])

        # Use numpy.allclose for tolerance-based comparison
        matches = np.allclose(opt_scores, van_scores, rtol=rtol, atol=atol)
        score_matches[col] = matches

        # Calculate max difference for reporting
        diffs = np.abs(opt_scores - van_scores)
        max_diffs[col] = float(np.max(diffs))

    all_match = predictions_match and all(score_matches.values())

    return {
        'match': all_match,
        'predictions_match': predictions_match,
        'score1_match': score_matches['score1'],
        'score2_match': score_matches['score2'],
        'max_diff_score1': max_diffs['score1'],
        'max_diff_score2': max_diffs['score2'],
        'num_sequences': len(opt_df)
    }
```

### Dual-Format Report Generation
```python
# Source: pytest-benchmark pattern + standard library
import json
from pathlib import Path
from datetime import datetime

def generate_benchmark_report(results, output_dir):
    """
    Generate both markdown (human) and JSON (machine) benchmark reports.

    Args:
        results: Dict with benchmark results
        output_dir: Directory to write reports
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Markdown report (human-readable)
    md_path = Path(output_dir) / f"benchmark_{timestamp}.md"
    with open(md_path, 'w') as f:
        f.write(f"# Benchmark Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Scaling results
        f.write("## GPU Scaling\n\n")
        f.write("| GPUs | Time (s) | Speedup | Efficiency |\n")
        f.write("|------|----------|---------|------------|\n")
        for config, data in results['scaling'].items():
            num_gpus = data['num_gpus']
            speedup = data['speedup']
            efficiency = (speedup / num_gpus * 100) if num_gpus > 0 else 0
            f.write(f"| {num_gpus} | {data['time_seconds']:.1f} | "
                   f"{speedup:.2f}x | {efficiency:.1f}% |\n")

        # GPU utilization
        f.write("\n## GPU Utilization\n\n")
        f.write("| GPU | Avg Util % | Peak Mem (GB) |\n")
        f.write("|-----|------------|---------------|\n")
        for gpu_id, stats in results['gpu_stats'].items():
            f.write(f"| {gpu_id} | {stats['gpu_util_avg']:.1f}% | "
                   f"{stats['mem_used_peak']/1024**3:.2f} |\n")

        # Throughput per stage
        f.write("\n## Throughput by Stage\n\n")
        f.write("| Stage | Seq/sec | Time (s) |\n")
        f.write("|-------|---------|----------|\n")
        for stage, metrics in results['throughput'].items():
            f.write(f"| {stage} | {metrics['throughput_seq_per_sec']:.1f} | "
                   f"{metrics['total_time_seconds']:.1f} |\n")

    # JSON report (machine-readable for CI)
    json_path = Path(output_dir) / f"benchmark_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'results': results,
            'metadata': {
                'pytorch_version': torch.__version__,
                'cuda_version': torch.version.cuda,
                'gpu_count': torch.cuda.device_count()
            }
        }, f, indent=2)

    return {'markdown': md_path, 'json': json_path}
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| nvidia-smi subprocess parsing | nvitop Python API | 2023-2024 | Direct NVML bindings, 10x faster, no parsing fragility |
| time.time() + manual sync | torch.utils.benchmark.Timer | PyTorch 1.8+ (2021) | Automatic warmup, CUDA sync, statistical analysis |
| Custom profilers | torch.profiler with Chrome trace | PyTorch 1.8+ (2021) | GPU memory timeline, kernel-level visibility, standardized format |
| Manual tolerance checks | numpy.allclose with rtol | Always standard, but BF16 adoption (2020+) requires understanding machine epsilon | Correct comparison for mixed precision (BF16/FP32) |
| Text-only reports | Markdown + JSON dual output | CI/CD era (2020+) | Human and machine consumption, automation-friendly |

**Deprecated/outdated:**
- DataParallel: Use DDP (DistributedDataParallel) instead. DataParallel has GIL contention and single-process bottleneck.
- nvidia-smi for programmatic use: Use nvitop or PyTorch CUDA APIs. Parsing text output is brittle.
- prof.export_memory_timeline: Deprecated in PyTorch 2.x. Use torch.cuda.memory._record_memory_history instead.

## Open Questions

Things that couldn't be fully resolved:

1. **Optimal monitoring sampling frequency**
   - What we know: nvitop can sample at 100ms-1s intervals. Too frequent adds overhead, too slow misses spikes.
   - What's unclear: Best balance for 1-10 hour benchmarks. Does sampling frequency affect GPU performance?
   - Recommendation: Start with 500ms sampling (2 samples/sec). Profile overhead with/without monitoring to verify <1% impact.

2. **Synthetic vs real data for scaling validation**
   - What we know: Synthetic data (uniform length, random bases) is reproducible and portable. Real viral data has length variability and realistic patterns.
   - What's unclear: Whether length variability in real data significantly affects scaling measurements vs synthetic uniform data.
   - Recommendation: Use both. Synthetic for controlled scaling tests (isolate GPU count variable), real for end-to-end validation.

3. **Baseline generation strategy for CI**
   - What we know: Running full vanilla pipeline (45+ hours) is infeasible in CI. Pre-computed baselines are fast but may drift.
   - What's unclear: How to keep pre-computed baselines fresh as model/data changes. How to detect when re-generation needed.
   - Recommendation: Dual approach as specified in CONTEXT.md - pre-computed for CI, manual fresh runs for validation. Document baseline provenance and generation date.

## Sources

### Primary (HIGH confidence)
- [PyTorch Profiler Tutorial](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html) - Profiling API and memory tracking
- [torch.utils.benchmark Documentation](https://docs.pytorch.org/docs/stable/benchmark_utils.html) - Timer usage, CUDA synchronization, warmup
- [torch.cuda Memory Management](https://docs.pytorch.org/docs/stable/torch_cuda_memory.html) - Memory tracking APIs
- [nvitop GitHub](https://github.com/XuehaiPan/nvitop) - Python API documentation
- [nvitop Documentation](https://nvitop.readthedocs.io/) - ResourceMetricCollector, Device API
- [pytest-benchmark Documentation](https://pytest-benchmark.readthedocs.io/) - Benchmark fixture, JSON output
- [numpy.allclose Documentation](https://numpy.org/doc/stable/reference/generated/numpy.allclose.html) - Tolerance parameters

### Secondary (MEDIUM confidence)
- [Multi-GPU Benchmark: B200 vs H200 vs H100 vs MI300X](https://research.aimultiple.com/multi-gpu/) - 2026 scaling benchmarks showing near-linear performance
- [Linear Multi-GPU Scaling in Oxford Nanopore's Dorado Basecaller](https://researchbox.ai/blog/linear-multi-gpu-scaling-in-oxford-nanopores-dorado-basecaller/index.html) - Real-world example of 6.2x on 8 GPUs
- [Keeping an eye on your GPUs - GPU monitoring tools compared](https://lambda.ai/blog/keeping-an-eye-on-your-gpus-2) - nvitop vs nvidia-smi comparison
- [Fix GPU Bottlenecks: PyTorch Profiler + Nsight](https://acecloud.ai/blog/gpu-bottlenecks-pytorch-profiler-nsight/) - Bottleneck detection patterns
- [Precision Comparison: FP64 FP32 FP16 TF32 BF16 INT8](https://www.allpcb.com/allelectrohub/precision-comparison-fp64-fp32-fp16-tf32-bf16-int8) - BF16 machine epsilon ~0.008

### Tertiary (LOW confidence - needs validation)
- WebSearch results on synthetic DNA generation - general approaches found but not bioinformatics-specific best practices for benchmark data generation
- Specific rtol/atol values for BF16/FP32 comparison - inferred from machine epsilon (1e-3 commonly used) but not authoritative PyTorch guidance found

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - PyTorch official tools, nvitop well-documented with examples
- Architecture: HIGH - Patterns verified in PyTorch docs and existing codebase (gpu_monitor.py, profiler.py)
- Pitfalls: HIGH - Well-documented issues in PyTorch forums and profiling guides
- Code examples: HIGH - Adapted from official documentation and existing test_integration_multi_gpu.py

**Research date:** 2026-01-26
**Valid until:** ~30 days (stable domain - GPU benchmarking practices don't change rapidly, but PyTorch minor releases may update APIs)

**Key decisions from CONTEXT.md applied:**
- Using both real and synthetic test data (per "Benchmark Test Data" decision)
- Dual markdown + JSON output format (per "Performance Metrics & Reporting" decision)
- Tolerance of 1e-3 for BF16/FP32 comparison (per "Vanilla Comparison Testing" decision)
- nvitop chosen over nvidia-smi (per "Claude's Discretion" on monitoring tool choice)
