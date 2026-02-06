# Phase 10: Performance Validation & Tuning - Research

**Researched:** 2026-02-06
**Domain:** End-to-end performance benchmarking, GPU utilization validation, parameter tuning
**Confidence:** HIGH

## Summary

Phase 10 validates that the complete v2.0 async architecture (Phases 5-9: async DataLoader, sequence packing, multi-GPU, FP16, checkpointing) meets the <10 hour target with >70% GPU utilization. This is a measurement, validation, and tuning phase - not a feature development phase. The goal is to identify bottlenecks through profiling, tune existing parameters to hit performance targets, and document the achieved speedup vs. v1.0 baseline.

The existing codebase already has extensive infrastructure:
- **GPU monitoring**: NvitopMonitor in `virnucpro/utils/gpu_monitor.py` with utilization, memory, DataLoader metrics
- **AsyncInferenceRunner telemetry**: Per-batch metrics (tokens/sec, sequences/sec, packing efficiency, wait times)
- **Benchmark suite**: `tests/benchmarks/` with throughput, scaling, equivalence tests
- **CLI tools**: `virnucpro benchmark` and `virnucpro profile` for performance validation

The new work requires:
1. Full production workload benchmarking (6M sequences) to validate <10h target
2. v1.0 baseline comparison to measure actual speedup (target: 4.5×)
3. Bottleneck identification via profiling (DataLoader, packing, GPU compute, I/O)
4. Parameter tuning (num_workers, prefetch_factor, token_budget, checkpoint frequency)
5. Telemetry capture (real-time progress bars + JSON logs + summary reports)
6. Multi-GPU scaling validation (2 GPUs = 1.9× speedup minimum)

**Primary recommendation:** Use the existing NvitopMonitor and AsyncInferenceRunner telemetry infrastructure as the foundation. Add production-scale benchmark harness, v1.0 comparison tooling, and real-time progress bars with live GPU metrics. Focus measurement and tuning effort on the four critical areas: (1) DataLoader prefetching, (2) packing efficiency, (3) GPU utilization, (4) checkpoint overhead.

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| nvitop | Latest | GPU utilization, memory, power monitoring | Already integrated in Phase 5+, superior to nvidia-smi for programmatic access |
| torch.profiler | Built-in PyTorch 2.2+ | CPU/GPU profiling, bottleneck detection | Official PyTorch tool, TensorBoard integration |
| rich | >=13.0.0 | Real-time progress bars with live metrics | Already in requirements.txt, better than tqdm for multi-metric displays |
| pytest | Existing | Benchmark test harness | Already used for all testing, benchmark markers already defined |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| h5py | Existing | HDF5 metrics persistence | Already required for embeddings, use for detailed telemetry logs |
| json | Built-in | Summary report format | Machine-readable results for CI/regression tracking |
| pandas | Optional | Telemetry analysis | Post-run analysis of per-batch metrics (optional, not required) |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| nvitop | nvidia-smi | nvidia-smi is CLI-only, harder to parse programmatically. nvitop has Python API |
| torch.profiler | cProfile | cProfile CPU-only, doesn't capture GPU kernels or CUDA events |
| rich.progress | tqdm | tqdm simpler but rich.progress supports live multi-column metrics (GPU util, tokens/sec, etc.) |
| JSON | YAML/CSV | JSON better for CI integration, widely supported parsers |

**Installation:**
```bash
# All dependencies already installed except optional pandas
pip install pandas  # Only if doing advanced post-run analysis
```

## Architecture Patterns

### Recommended Project Structure
```
tests/
  benchmarks/
    test_production_validation.py     # NEW: Full 6M sequence production benchmark
    test_v1_comparison.py              # NEW: v1.0 vs v2.0 speedup validation
    test_parameter_tuning.py           # NEW: DataLoader/packing parameter sweep
    test_scaling_verification.py       # MODIFY: Add 2-GPU 1.9× requirement
virnucpro/
  pipeline/
    async_inference.py                 # MODIFY: Add real-time progress bars
    progress_reporter.py               # NEW: Rich progress bar integration
  utils/
    gpu_monitor.py                     # EXISTING: NvitopMonitor (no changes needed)
    telemetry.py                       # NEW: Metrics aggregation and reporting
```

### Pattern 1: Production Benchmark Harness
**What:** Full workload benchmark with real data to validate <10h target
**When to use:** Final validation that all phases (5-9) deliver target performance
**Example:**
```python
# Source: Existing test_end_to_end.py + production requirements
import pytest
import time
from pathlib import Path
from virnucpro.pipeline.multi_gpu_inference import run_multi_gpu_inference

@pytest.mark.slow
@pytest.mark.gpu
def test_production_workload_10hr_requirement(production_fasta_path):
    """
    Validate <10 hour requirement on full production workload.

    Test conditions:
    - 6M sequences (actual viral nucleotide data)
    - 2 GPUs (typical hardware config per CONTEXT.md)
    - All optimizations enabled (packing, FP16, checkpointing)

    Success criteria:
    - Total runtime < 10 hours
    - GPU utilization >70% average
    - Packing efficiency >90% average
    """
    output_dir = Path("test_outputs/production_validation")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run inference with telemetry
    start_time = time.time()

    output_path, failed_ranks = run_multi_gpu_inference(
        fasta_files=[production_fasta_path],
        output_path=output_dir / "embeddings.h5",
        model_name="esm2_t36_3B_UR50D",
        world_size=2,  # Typical 2-GPU config
        enable_checkpointing=True,
        checkpoint_base_dir=output_dir / "checkpoints",
    )

    elapsed_hours = (time.time() - start_time) / 3600

    # Load telemetry from checkpoint manifest
    from virnucpro.pipeline.checkpoint_manifest import CheckpointManifest
    manifest = CheckpointManifest(output_dir / "checkpoints" / "manifest.json")

    # Extract metrics
    gpu_util_avg = manifest.get_average_gpu_utilization()
    packing_efficiency_avg = manifest.get_average_packing_efficiency()

    # Validate requirements (PERF-01, PERF-02, PERF-05)
    assert elapsed_hours < 10.0, (
        f"10-hour requirement FAILED: {elapsed_hours:.2f}h elapsed. "
        f"Target: <10h. Check bottlenecks: DataLoader, packing, I/O"
    )

    assert gpu_util_avg >= 70.0, (
        f"GPU utilization requirement FAILED: {gpu_util_avg:.1f}% average. "
        f"Target: >70%. Check: DataLoader starvation, small batches"
    )

    assert packing_efficiency_avg >= 90.0, (
        f"Packing efficiency requirement FAILED: {packing_efficiency_avg:.1f}% average. "
        f"Target: >90%. Check: buffer size, token budget, length distribution"
    )

    print(f"✓ Production validation PASSED")
    print(f"  Runtime: {elapsed_hours:.2f}h / 10h")
    print(f"  GPU util: {gpu_util_avg:.1f}% (target: >70%)")
    print(f"  Packing: {packing_efficiency_avg:.1f}% (target: >90%)")
```

### Pattern 2: v1.0 Baseline Comparison
**What:** Measure speedup vs. v1.0 multi-worker-per-GPU architecture
**When to use:** One-time validation to confirm 4.5× speedup claim
**Example:**
```python
# Source: User decision from CONTEXT.md - validate 4.5× speedup claim
import subprocess
import time

@pytest.mark.slow
@pytest.mark.gpu
def test_v1_vs_v2_speedup_validation(test_fasta_subset):
    """
    Measure actual speedup: v2.0 (async) vs v1.0 (multi-worker).

    Test conditions:
    - Same workload (1K sequence subset for speed)
    - Same hardware (2 GPUs)
    - v1.0: git checkout v1.0 tag, run with --parallel
    - v2.0: current implementation

    Success criteria:
    - v2.0 >= 4.0× faster than v1.0 (allowing margin below 4.5× target)
    """
    # Run v1.0 baseline
    v1_start = time.time()
    subprocess.run([
        "git", "checkout", "v1.0",  # Checkout stable v1.0 tag
    ], check=True)

    subprocess.run([
        "python", "-m", "virnucpro", "predict",
        str(test_fasta_subset),
        "--parallel",
        "--output", "v1_output.csv",
    ], check=True)

    v1_elapsed = time.time() - v1_start

    # Restore v2.0
    subprocess.run(["git", "checkout", "-"], check=True)

    # Run v2.0
    v2_start = time.time()
    output_path, _ = run_multi_gpu_inference(
        fasta_files=[test_fasta_subset],
        output_path=Path("v2_output.h5"),
        model_name="esm2_t36_3B_UR50D",
        world_size=2,
    )
    v2_elapsed = time.time() - v2_start

    speedup = v1_elapsed / v2_elapsed

    # Validate speedup
    assert speedup >= 4.0, (
        f"Speedup target FAILED: {speedup:.2f}× (v1: {v1_elapsed:.1f}s, v2: {v2_elapsed:.1f}s). "
        f"Target: ≥4.5×. Check: async DataLoader, packing, FP16 all enabled"
    )

    print(f"✓ Speedup validation PASSED: {speedup:.2f}×")
```

### Pattern 3: Real-Time Progress Bars with Live Metrics
**What:** Rich progress bars showing throughput, GPU util, packing efficiency during inference
**When to use:** Production runs to provide live feedback on performance
**Example:**
```python
# Source: rich.progress documentation + AsyncInferenceRunner integration
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
from rich.live import Live
from rich.table import Table

class InferenceProgressReporter:
    """Real-time progress reporter with live GPU metrics."""

    def __init__(self, total_sequences: int, monitor):
        self.total_sequences = total_sequences
        self.monitor = monitor
        self.progress = Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("•"),
            TextColumn("[yellow]{task.fields[tokens_per_sec]:>10,.0f} tok/s"),
            TextColumn("[green]{task.fields[gpu_util]:>5.1f}% GPU"),
            TextColumn("[cyan]{task.fields[packing_eff]:>5.1f}% pack"),
            TimeRemainingColumn(),
        )
        self.task_id = None

    def __enter__(self):
        self.progress.start()
        self.task_id = self.progress.add_task(
            "Processing sequences",
            total=self.total_sequences,
            tokens_per_sec=0.0,
            gpu_util=0.0,
            packing_eff=0.0,
        )
        return self

    def __exit__(self, *args):
        self.progress.stop()

    def update(self, sequences_processed: int):
        """Update progress with current metrics."""
        throughput = self.monitor.get_throughput()
        dl_stats = self.monitor.get_dataloader_statistics()
        gpu_stats = self.monitor.get_statistics()

        # Average GPU utilization across all devices
        gpu_utils = [s.get('gpu_util_avg', 0) for s in gpu_stats.values()
                     if isinstance(s, dict) and 'gpu_util_avg' in s]
        avg_gpu_util = sum(gpu_utils) / len(gpu_utils) if gpu_utils else 0.0

        self.progress.update(
            self.task_id,
            completed=sequences_processed,
            tokens_per_sec=throughput.get('tokens_per_sec', 0),
            gpu_util=avg_gpu_util,
            packing_eff=dl_stats.get('avg_packing_efficiency', 0) * 100,
        )

# Usage in AsyncInferenceRunner.run()
with InferenceProgressReporter(total_sequences=len(dataset), monitor=self.monitor) as progress:
    for batch_idx, batch in enumerate(dataloader):
        # ... inference code ...
        progress.update(self._total_sequences)
```

### Pattern 4: Parameter Tuning Suite
**What:** Automated sweep of DataLoader and packing parameters to find optimal settings
**When to use:** When <10h target not met, identify which parameters to adjust
**Example:**
```python
# Source: Hyperparameter tuning best practices + profiler.py patterns
import itertools
from typing import Dict, List

def tune_dataloader_parameters(
    fasta_path: Path,
    parameter_grid: Dict[str, List],
) -> Dict:
    """
    Sweep DataLoader parameters to find optimal configuration.

    Args:
        fasta_path: Test workload
        parameter_grid: {
            'num_workers': [4, 8, 12, 16],
            'prefetch_factor': [2, 4, 8],
            'token_budget': [4096, 8192, 12288],
        }

    Returns:
        Best configuration with throughput metrics
    """
    results = []

    # Grid search
    for params in itertools.product(*parameter_grid.values()):
        config = dict(zip(parameter_grid.keys(), params))

        # Run inference with this config
        start = time.time()
        output_path, _ = run_multi_gpu_inference(
            fasta_files=[fasta_path],
            output_path=Path(f"tuning_output_{hash(str(config))}.h5"),
            model_name="esm2_t36_3B_UR50D",
            world_size=1,  # Single GPU for tuning speed
            dataloader_config=config,
        )
        elapsed = time.time() - start

        # Extract metrics
        manifest = CheckpointManifest(...)
        throughput = manifest.get_throughput()
        gpu_util = manifest.get_average_gpu_utilization()

        results.append({
            'config': config,
            'elapsed_sec': elapsed,
            'throughput_tokens_per_sec': throughput['tokens_per_sec'],
            'gpu_utilization': gpu_util,
        })

    # Find best config (highest throughput)
    best = max(results, key=lambda r: r['throughput_tokens_per_sec'])

    print(f"Best configuration: {best['config']}")
    print(f"  Throughput: {best['throughput_tokens_per_sec']:,.0f} tokens/sec")
    print(f"  GPU util: {best['gpu_utilization']:.1f}%")

    return best
```

### Pattern 5: Telemetry Persistence (JSON + HDF5)
**What:** Capture per-batch metrics in detailed logs for post-run analysis
**When to use:** Every production run to enable regression detection and debugging
**Example:**
```python
# Source: Existing gpu_monitor.py export_metrics + user requirements
import json
import h5py
from pathlib import Path
from datetime import datetime

class TelemetryLogger:
    """Persistent telemetry for production runs."""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.json_path = self.output_dir / f"telemetry_{timestamp}.json"
        self.hdf5_path = self.output_dir / f"telemetry_{timestamp}.h5"

        self.batch_metrics = []

    def record_batch(self, batch_idx: int, metrics: Dict):
        """Record per-batch metrics."""
        self.batch_metrics.append({
            'batch_idx': batch_idx,
            'timestamp': time.time(),
            **metrics
        })

    def finalize(self, summary: Dict):
        """Write telemetry to disk."""
        # JSON summary (machine-readable)
        with open(self.json_path, 'w') as f:
            json.dump({
                'summary': summary,
                'per_batch_metrics': self.batch_metrics,
            }, f, indent=2)

        # HDF5 detailed metrics (efficient for large arrays)
        with h5py.File(self.hdf5_path, 'w') as f:
            # Create datasets for each metric
            batch_indices = [m['batch_idx'] for m in self.batch_metrics]
            f.create_dataset('batch_idx', data=batch_indices)

            for key in ['tokens_per_sec', 'gpu_util', 'packing_efficiency']:
                values = [m.get(key, 0) for m in self.batch_metrics]
                f.create_dataset(key, data=values)

        print(f"Telemetry saved:")
        print(f"  JSON: {self.json_path}")
        print(f"  HDF5: {self.hdf5_path}")
```

### Anti-Patterns to Avoid
- **Testing on small datasets**: Small workloads (<1K sequences) don't reveal bottlenecks. Use representative 6M sequence production data or minimum 10K subset
- **Ignoring warmup**: First few batches slower due to CUDA kernel compilation. Always warm up before timing (10 iterations minimum per Phase 8 patterns)
- **Measuring without sync**: torch.cuda.synchronize() required before timing to ensure GPU work completes
- **Tuning one parameter at a time**: DataLoader parameters interact (num_workers × prefetch_factor = queue depth). Use grid search
- **No baseline comparison**: Without v1.0 baseline, can't validate 4.5× speedup claim
- **Missing GPU utilization validation**: Can hit <10h but only using 30% GPU (indicates bottleneck opportunity)

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| GPU utilization monitoring | Parse nvidia-smi output | nvitop Python API | Already integrated, structured data, handles multi-GPU |
| Progress bars | Print statements with % | rich.progress | Real-time updates, multi-metric display, time remaining |
| Parameter tuning | Manual trial-and-error | Grid search with metrics | Systematic, reproducible, finds interactions |
| Profiling bottlenecks | Manual timing | torch.profiler | CPU+GPU correlation, TensorBoard viz, automatic bottleneck detection |
| Telemetry persistence | CSV files | JSON + HDF5 | JSON for summary (CI-friendly), HDF5 for large arrays (efficient) |
| Throughput calculation | Manual tokens/elapsed | AsyncInferenceRunner.get_throughput() | Already implemented, handles edge cases |

**Key insight:** The infrastructure already exists from Phases 5-8. This phase is about connecting the pieces (production benchmark harness, v1.0 comparison, real-time reporting) and tuning parameters, not building new monitoring systems.

## Common Pitfalls

### Pitfall 1: Testing on Non-Representative Data
**What goes wrong:** Benchmark passes on synthetic data but fails on production workload
**Why it happens:** Synthetic sequences have uniform length, production has wide distribution
**How to avoid:**
- Always use actual viral nucleotide data (6M sequences from production)
- If using subset, ensure length distribution matches production (stratified sampling)
- Validate packing efficiency on production data (synthetic may over-pack)
**Warning signs:** Packing efficiency >95% on test but <85% on production, throughput drops 2× on real data

### Pitfall 2: Ignoring Multi-GPU Scaling Efficiency
**What goes wrong:** Hit <10h on 4 GPUs but scaling is poor (1.5× instead of 1.9× for 2 GPUs)
**Why it happens:** Stride-based sharding may cause imbalance, one GPU finishes early
**How to avoid:**
- Validate PERF-03: 2 GPUs = 1.9× speedup (95% efficiency)
- Check per-GPU runtime in manifest (should be within 10%)
- Monitor GPU utilization per device (imbalance shows as low util on some GPUs)
**Warning signs:** Manifest shows GPU 0: 3.2h, GPU 1: 2.1h (1.5× imbalance)

### Pitfall 3: DataLoader Starvation Not Detected
**What goes wrong:** GPU utilization <50% but no warning, bottleneck not identified
**Why it happens:** Monitoring interval too long, misses transient starvation
**How to avoid:**
- NvitopMonitor already checks every 10 batches (existing code)
- Log queue_state distribution in summary (should be <10% 'starved')
- If >20% starved batches, increase prefetch_factor or num_workers
**Warning signs:** GPU util <50%, queue_state: {'starved': 35%, 'normal': 65%}

### Pitfall 4: Checkpoint Overhead Not Measured
**What goes wrong:** Checkpointing adds 20% overhead but not detected in profiling
**Why it happens:** Checkpointing is async but still blocks DataLoader if queue full
**How to avoid:**
- Compare throughput with/without checkpointing (VIRNUCPRO_DISABLE_CHECKPOINTING=1)
- If >10% overhead, reduce checkpoint frequency (increase sequence threshold)
- Async checkpoint writer should prevent blocking (Phase 9 design)
**Warning signs:** tokens/sec drops 20% when checkpointing enabled

### Pitfall 5: No Regression Tracking
**What goes wrong:** Performance degrades over time, not caught until production
**Why it happens:** No automated comparison against baseline
**How to avoid:**
- Save JSON telemetry from each run with git commit hash
- CI pipeline compares against previous run (fail if >10% regression)
- Track metrics: tokens/sec, GPU util, packing efficiency, total runtime
**Warning signs:** No historical data, manual inspection required every time

## Code Examples

Verified patterns from official sources:

### PyTorch Profiler Integration
```python
# Source: PyTorch profiler documentation
# URL: https://pytorch.org/docs/tutorials/intermediate/tensorboard_profiler_tutorial.html

import torch
from torch.profiler import profile, record_function, ProfilerActivity

def profile_inference_bottlenecks(
    model,
    dataloader,
    num_batches=10,
    output_dir="./profiler_traces",
):
    """
    Profile inference to identify bottlenecks.

    Captures:
    - CPU time (DataLoader, collation, Python overhead)
    - GPU time (kernel execution, memory transfers)
    - Wait times (GPU idle, DataLoader starvation)
    """
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(output_dir),
    ) as prof:
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= num_batches:
                break

            with record_function("dataloader_fetch"):
                # Fetch timing already captured by profiler
                pass

            with record_function("model_forward"):
                outputs = model.forward_packed(
                    input_ids=batch["input_ids"],
                    cu_seqlens=batch["cu_seqlens"],
                    max_seqlen=batch["max_seqlen"],
                )

            prof.step()  # Mark step boundary for TensorBoard

    # Print top bottlenecks
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    # Export to TensorBoard
    print(f"TensorBoard traces saved to {output_dir}")
    print(f"View with: tensorboard --logdir={output_dir}")
```

### Multi-GPU Scaling Validation
```python
# Source: PyTorch distributed docs + Phase 7 multi_gpu_inference.py
# URL: https://pytorch.org/docs/stable/distributed.html

@pytest.mark.gpu
@pytest.mark.parametrize("world_size", [1, 2])
def test_multi_gpu_scaling_efficiency(test_fasta_path, world_size):
    """
    Validate PERF-03: 2 GPUs = 1.9× speedup (95% efficiency).

    Scaling efficiency = (speedup / num_gpus) × 100%
    """
    output_dir = Path(f"scaling_test_{world_size}gpu")

    start = time.time()
    output_path, _ = run_multi_gpu_inference(
        fasta_files=[test_fasta_path],
        output_path=output_dir / "embeddings.h5",
        model_name="esm2_t36_3B_UR50D",
        world_size=world_size,
    )
    elapsed = time.time() - start

    if world_size == 1:
        pytest.baseline_elapsed = elapsed
        return  # No comparison yet

    # Calculate scaling metrics
    speedup = pytest.baseline_elapsed / elapsed
    efficiency = (speedup / world_size) * 100

    print(f"\n{world_size}-GPU Scaling:")
    print(f"  1 GPU: {pytest.baseline_elapsed:.1f}s")
    print(f"  {world_size} GPUs: {elapsed:.1f}s")
    print(f"  Speedup: {speedup:.2f}×")
    print(f"  Efficiency: {efficiency:.1f}%")

    # PERF-03 requirement: 95% efficiency for 2 GPUs
    min_efficiency = 95.0 if world_size == 2 else 90.0

    assert efficiency >= min_efficiency, (
        f"Scaling efficiency too low: {efficiency:.1f}% "
        f"(target: ≥{min_efficiency}%). "
        f"Check: work distribution balance, inter-GPU communication overhead"
    )
```

### Telemetry Summary Report
```python
# Source: User requirements + existing report_generator.py patterns
# URL: Project requirements PERF-05

def generate_telemetry_summary(
    monitor,
    total_sequences: int,
    elapsed_seconds: float,
    output_path: Path,
):
    """
    Generate human-readable performance summary.

    Captures PERF-05 requirements:
    - Throughput (tokens/sec, sequences/sec)
    - Packing efficiency (% token utilization)
    - I/O wait time (DataLoader metrics)
    - GPU utilization (per-device average)
    """
    throughput = monitor.get_throughput()
    dl_stats = monitor.get_dataloader_statistics()
    gpu_stats = monitor.get_statistics()

    # Calculate aggregate metrics
    sequences_per_hour = (total_sequences / elapsed_seconds) * 3600
    hours_for_6M = (6_000_000 / total_sequences) * (elapsed_seconds / 3600)

    report = f"""
Performance Validation Summary
{'=' * 70}

Workload:
  Total sequences:    {total_sequences:>12,}
  Total runtime:      {elapsed_seconds / 3600:>12.2f} hours
  Sequences/hour:     {sequences_per_hour:>12,.0f}
  Projected 6M time:  {hours_for_6M:>12.2f} hours

Throughput (PERF-05):
  Sequences/sec:      {throughput['sequences_per_sec']:>12.1f}
  Tokens/sec:         {throughput['tokens_per_sec']:>12,.0f}

Packing Efficiency (PERF-05):
  Average:            {dl_stats.get('avg_packing_efficiency', 0) * 100:>12.1f}%
  Minimum:            {dl_stats.get('min_packing_efficiency', 0) * 100:>12.1f}%
  Batches <80%:       {dl_stats.get('batches_below_threshold', 0):>12}

DataLoader I/O (PERF-05):
  Avg wait time:      {dl_stats.get('avg_wait_time_ms', 0):>12.1f} ms
  P95 wait time:      {dl_stats.get('p95_wait_time_ms', 0):>12.1f} ms
  Queue starved:      {dl_stats.get('pct_starved', 0):>12.1f}%

GPU Utilization (PERF-02):
"""

    # Per-GPU metrics
    for device_id, stats in gpu_stats.items():
        if not isinstance(device_id, int):
            continue  # Skip non-device entries
        report += f"  GPU {device_id}:            {stats['gpu_util_avg']:>12.1f}% (peak mem: {stats['mem_used_peak'] / 1e9:.1f}GB)\n"

    report += f"""
{'=' * 70}

Requirements Validation:
"""

    # Check requirements
    meets_10h = hours_for_6M < 10.0
    meets_gpu_util = all(
        s['gpu_util_avg'] >= 70.0
        for s in gpu_stats.values()
        if isinstance(s, dict) and 'gpu_util_avg' in s
    )
    meets_packing = dl_stats.get('avg_packing_efficiency', 0) >= 0.90

    report += f"  PERF-01 (<10h):         {'✓ PASS' if meets_10h else '✗ FAIL'}\n"
    report += f"  PERF-02 (>70% GPU):     {'✓ PASS' if meets_gpu_util else '✗ FAIL'}\n"
    report += f"  PERF-05 (>90% pack):    {'✓ PASS' if meets_packing else '✗ FAIL'}\n"

    report += f"\n{'=' * 70}\n"

    # Write to file
    with open(output_path, 'w') as f:
        f.write(report)

    print(report)
    return report
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| nvidia-smi polling | nvitop Python API | Phase 5 | Structured GPU metrics, programmatic access |
| Manual timing with time.time() | torch.profiler with CUDA events | PyTorch 2.0+ | CPU/GPU correlation, automatic bottleneck detection |
| Print statements for progress | rich.progress with live metrics | Phase 10 | Real-time throughput, GPU util, packing efficiency display |
| CSV telemetry logs | JSON summary + HDF5 details | Phase 10 | CI-friendly JSON, efficient HDF5 for large metric arrays |
| Ad-hoc parameter tuning | Systematic grid search | Phase 10 | Reproducible, finds optimal config, documents tradeoffs |

**Deprecated/outdated:**
- **nvidia-smi**: Still works but inferior to nvitop for programmatic access
- **tqdm**: Replaced by rich.progress for multi-metric live displays
- **Manual GPU utilization checks**: torch.profiler automatically detects idle GPU time

## Open Questions

Things that couldn't be fully resolved:

1. **Optimal DataLoader parameters for 6M sequences**
   - What we know: Phase 5 uses num_workers=4, prefetch_factor=4 for test workloads
   - What's unclear: Whether these parameters are optimal for full 6M sequence production data
   - Recommendation: Run parameter sweep (4/8/12 workers, 2/4/8 prefetch) and measure throughput

2. **v1.0 availability for comparison**
   - What we know: User wants to validate 4.5× speedup claim against v1.0 baseline
   - What's unclear: Whether v1.0 tag exists in git history, or needs separate branch
   - Recommendation: Check `git tag` for v1.0, if missing create from commit before Phase 5

3. **Checkpoint overhead impact**
   - What we know: Phase 9 uses async checkpoint writer to prevent blocking
   - What's unclear: Actual overhead on production workload (may be <1% or >10%)
   - Recommendation: Measure with/without checkpointing, document in telemetry

4. **Multi-GPU scaling beyond 2 GPUs**
   - What we know: User has 2 GPUs typically, PERF-03 requires 95% efficiency for 2 GPUs
   - What's unclear: Scaling efficiency for 4 or 8 GPUs (may have communication overhead)
   - Recommendation: If >2 GPUs available, validate scaling curve (log efficiency vs num_gpus)

5. **Production data access for benchmarking**
   - What we know: Need 6M sequence real viral nucleotide data for validation
   - What's unclear: Data location, access permissions, whether subset acceptable
   - Recommendation: Coordinate with user to get production sample or representative subset

## Sources

### Primary (HIGH confidence)
- [PyTorch Profiler documentation](https://pytorch.org/blog/introducing-pytorch-profiler-the-new-and-improved-performance-tool/) - GPU performance profiling, bottleneck detection
- [PyTorch Benchmark tutorial](https://docs.pytorch.org/tutorials/recipes/recipes/benchmark.html) - CUDA synchronization, warmup, timing best practices
- [nvitop GitHub repository](https://github.com/XuehaiPan/nvitop) - GPU monitoring Python API, metrics structure
- [nvitop documentation](https://nvitop.readthedocs.io/) - API reference, utilization metrics, ResourceMetricCollector
- [rich.progress documentation](https://rich.readthedocs.io/en/stable/progress.html) - Live progress bars, custom columns
- Existing codebase:
  - `virnucpro/utils/gpu_monitor.py` - NvitopMonitor implementation
  - `virnucpro/pipeline/async_inference.py` - Throughput and DataLoader metrics
  - `tests/benchmarks/test_fp16_throughput.py` - Benchmark patterns

### Secondary (MEDIUM confidence)
- [PyTorch DataLoader profiling (Towards Data Science)](https://towardsdatascience.com/solving-bottlenecks-on-the-data-input-pipeline-with-pytorch-profiler-and-tensorboard-5dced134dbe9/) - DataLoader bottleneck detection patterns
- [PyTorch DataLoader for high-latency storage (arXiv 2211.04908)](https://arxiv.org/pdf/2211.04908) - num_workers tuning recommendations
- [PyTorch Lightning profiler](https://lightning.ai/docs/pytorch/stable/tuning/profiler_basic.html) - Profiling patterns for production

### Tertiary (LOW confidence)
- None - all sources verified against PyTorch official docs or existing codebase

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - nvitop, torch.profiler, rich all documented and integrated
- Architecture: HIGH - Patterns verified against existing Phase 5-8 infrastructure
- Pitfalls: HIGH - Based on PyTorch profiling docs, distributed scaling research, existing benchmark failures
- Production workload: MEDIUM - Need user confirmation on data access and v1.0 availability

**Research date:** 2026-02-06
**Valid until:** 2026-03-06 (30 days - fast-moving benchmark tooling, verify latest PyTorch/nvitop versions)
