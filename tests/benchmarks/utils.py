"""Benchmark utilities for timing, result formatting, and multi-GPU helpers.

Provides:
- BenchmarkTimer: Wrapper around torch.utils.benchmark.Timer with CUDA sync
- GPU memory tracking utilities
- Result formatting for markdown and JSON output
- Scaling efficiency calculations
- Multi-GPU test configuration helpers
"""

import torch
import torch.utils.benchmark as benchmark
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json
import time
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

logger = logging.getLogger('virnucpro.benchmarks.utils')


# ==================== Benchmark Timer ====================

class BenchmarkTimer:
    """
    Wrapper around torch.utils.benchmark.Timer with automatic CUDA synchronization.

    Handles:
    - Warmup iterations
    - CUDA synchronization for accurate GPU timing
    - Multiple measurements with statistics
    - Memory tracking

    Example:
        >>> timer = BenchmarkTimer()
        >>> def run_model():
        ...     output = model(input_batch)
        ...     return output
        >>> results = timer.measure(run_model, warmup=2, iterations=5)
        >>> print(f"Mean: {results['mean']:.4f}s, Std: {results['std']:.4f}s")
    """

    def __init__(self, device: Optional[torch.device] = None):
        """
        Initialize timer.

        Args:
            device: CUDA device to synchronize (None = auto-detect from cuda:0)
        """
        self.device = device
        if self.device is None and torch.cuda.is_available():
            self.device = torch.device('cuda:0')

    def measure(self,
                func,
                warmup: int = 2,
                iterations: int = 5,
                track_memory: bool = True) -> Dict[str, float]:
        """
        Measure function execution time with warmup and statistics.

        Args:
            func: Function to benchmark (no arguments)
            warmup: Number of warmup iterations
            iterations: Number of timed iterations
            track_memory: Whether to track GPU memory usage

        Returns:
            Dictionary with:
            - mean: Mean execution time (seconds)
            - std: Standard deviation (seconds)
            - min: Minimum time (seconds)
            - max: Maximum time (seconds)
            - median: Median time (seconds)
            - memory_peak_mb: Peak memory usage (MB, if track_memory=True)
            - measurements: List of individual measurements (seconds)
        """
        # Warmup
        for _ in range(warmup):
            func()
            if self.device is not None:
                torch.cuda.synchronize(self.device)

        # Clear CUDA cache before measurement
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if track_memory and self.device is not None:
                torch.cuda.reset_peak_memory_stats(self.device)

        # Measure
        measurements = []
        for _ in range(iterations):
            if self.device is not None:
                torch.cuda.synchronize(self.device)

            start = time.time()
            func()

            if self.device is not None:
                torch.cuda.synchronize(self.device)

            elapsed = time.time() - start
            measurements.append(elapsed)

        # Calculate statistics
        import statistics
        results = {
            'mean': statistics.mean(measurements),
            'std': statistics.stdev(measurements) if len(measurements) > 1 else 0.0,
            'min': min(measurements),
            'max': max(measurements),
            'median': statistics.median(measurements),
            'measurements': measurements,
        }

        # Memory tracking
        if track_memory and self.device is not None and torch.cuda.is_available():
            peak_bytes = torch.cuda.max_memory_allocated(self.device)
            results['memory_peak_mb'] = peak_bytes / (1024 ** 2)

        return results

    def measure_throughput(self,
                          func,
                          num_items: int,
                          warmup: int = 2,
                          iterations: int = 5) -> Dict[str, float]:
        """
        Measure throughput (items per second).

        Args:
            func: Function to benchmark
            num_items: Number of items processed per call
            warmup: Warmup iterations
            iterations: Timed iterations

        Returns:
            Dictionary with timing results plus:
            - throughput: Items per second
            - latency_per_item_ms: Milliseconds per item
        """
        results = self.measure(func, warmup=warmup, iterations=iterations)

        # Calculate throughput
        mean_time = results['mean']
        throughput = num_items / mean_time if mean_time > 0 else 0.0
        latency_per_item = (mean_time / num_items * 1000) if num_items > 0 else 0.0

        results['throughput'] = throughput
        results['latency_per_item_ms'] = latency_per_item

        return results


# ==================== GPU Memory Tracking ====================

def track_peak_memory(device: torch.device, reset: bool = True) -> float:
    """
    Get peak GPU memory usage.

    Args:
        device: CUDA device
        reset: Whether to reset peak memory stats after reading

    Returns:
        Peak memory usage in GB
    """
    if not torch.cuda.is_available():
        return 0.0

    peak_bytes = torch.cuda.max_memory_allocated(device)
    peak_gb = peak_bytes / (1024 ** 3)

    if reset:
        torch.cuda.reset_peak_memory_stats(device)

    return peak_gb


def get_current_memory(device: torch.device) -> Tuple[float, float]:
    """
    Get current GPU memory usage.

    Args:
        device: CUDA device

    Returns:
        Tuple of (allocated_gb, total_gb)
    """
    if not torch.cuda.is_available():
        return 0.0, 0.0

    allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)
    total = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)

    return allocated, total


# ==================== Scaling Calculations ====================

@dataclass
class ScalingResult:
    """Results from multi-GPU scaling benchmark."""
    num_gpus: int
    time_seconds: float
    speedup: float  # Relative to single GPU
    efficiency: float  # Speedup / num_gpus (0.0-1.0)
    throughput: float  # Items per second
    memory_peak_gb: Optional[float] = None


def calculate_scaling_metrics(baseline_time: float,
                              results: List[Tuple[int, float]]) -> List[ScalingResult]:
    """
    Calculate scaling metrics from benchmark results.

    Args:
        baseline_time: Single GPU baseline time (seconds)
        results: List of (num_gpus, time_seconds) tuples

    Returns:
        List of ScalingResult objects with speedup and efficiency
    """
    scaling_results = []

    for num_gpus, time_seconds in results:
        speedup = baseline_time / time_seconds if time_seconds > 0 else 0.0
        efficiency = speedup / num_gpus if num_gpus > 0 else 0.0
        throughput = 1.0 / time_seconds if time_seconds > 0 else 0.0

        scaling_results.append(ScalingResult(
            num_gpus=num_gpus,
            time_seconds=time_seconds,
            speedup=speedup,
            efficiency=efficiency,
            throughput=throughput,
        ))

    return scaling_results


def check_linear_scaling(results: List[ScalingResult],
                        min_efficiency: float = 0.85) -> bool:
    """
    Check if scaling is approximately linear.

    Args:
        results: List of ScalingResult objects
        min_efficiency: Minimum acceptable efficiency (0.85 = 85%)

    Returns:
        True if all results meet minimum efficiency threshold
    """
    for result in results:
        if result.efficiency < min_efficiency:
            logger.warning(
                f"{result.num_gpus} GPUs: {result.efficiency*100:.1f}% efficiency "
                f"(below {min_efficiency*100:.1f}% threshold)"
            )
            return False

    return True


# ==================== Result Formatting ====================

def format_results(results: Dict[str, Any],
                  output_format: str = 'markdown') -> str:
    """
    Format benchmark results for display.

    Args:
        results: Benchmark results dictionary
        output_format: 'markdown' or 'json'

    Returns:
        Formatted string
    """
    if output_format == 'json':
        return json.dumps(results, indent=2)

    # Markdown format
    lines = []
    lines.append("# Benchmark Results\n")

    if 'timestamp' in results:
        lines.append(f"**Timestamp:** {results['timestamp']}\n")

    if 'scaling' in results:
        lines.append("## GPU Scaling\n")
        lines.append("| GPUs | Time (s) | Speedup | Efficiency |")
        lines.append("|------|----------|---------|------------|")

        for entry in results['scaling']:
            lines.append(
                f"| {entry['num_gpus']} | {entry['time_seconds']:.2f} | "
                f"{entry['speedup']:.2f}x | {entry['efficiency']*100:.1f}% |"
            )
        lines.append("")

    if 'throughput' in results:
        lines.append("## Throughput\n")
        lines.append("| Stage | Seq/sec | Time (s) |")
        lines.append("|-------|---------|----------|")

        for stage, metrics in results['throughput'].items():
            lines.append(
                f"| {stage} | {metrics['throughput']:.1f} | "
                f"{metrics['total_time']:.2f} |"
            )
        lines.append("")

    if 'memory' in results:
        lines.append("## Memory Usage\n")
        lines.append("| GPU | Peak (GB) | Avg (GB) |")
        lines.append("|-----|-----------|----------|")

        for gpu_id, metrics in results['memory'].items():
            lines.append(
                f"| {gpu_id} | {metrics['peak']:.2f} | "
                f"{metrics['avg']:.2f} |"
            )
        lines.append("")

    return '\n'.join(lines)


def save_results(results: Dict[str, Any],
                output_dir: Path,
                name: str = 'benchmark') -> Tuple[Path, Path]:
    """
    Save benchmark results in both markdown and JSON formats.

    Args:
        results: Benchmark results dictionary
        output_dir: Directory to save results
        name: Base filename (without extension)

    Returns:
        Tuple of (markdown_path, json_path)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results['timestamp'] = datetime.now().isoformat()

    # Save markdown
    md_path = output_dir / f"{name}_{timestamp}.md"
    with open(md_path, 'w') as f:
        f.write(format_results(results, output_format='markdown'))

    # Save JSON
    json_path = output_dir / f"{name}_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {md_path} and {json_path}")

    return md_path, json_path


# ==================== Multi-GPU Test Configuration ====================

def get_available_gpu_configs() -> List[Dict[str, Any]]:
    """
    Get list of available GPU configurations for testing.

    Returns:
        List of configurations matching available hardware:
        [{'num_gpus': 1, 'gpu_ids': [0], 'gpu_str': '0'}, ...]
    """
    if not torch.cuda.is_available():
        return []

    num_gpus = torch.cuda.device_count()
    configs = []

    # Standard configurations: 1, 2, 4, 8
    for n in [1, 2, 4, 8]:
        if n <= num_gpus:
            gpu_ids = list(range(n))
            configs.append({
                'num_gpus': n,
                'gpu_ids': gpu_ids,
                'gpu_str': ','.join(map(str, gpu_ids)),
                'config_name': f'{n}gpu'
            })

    return configs


def create_test_config(num_gpus: int = 1) -> Dict[str, Any]:
    """
    Create test configuration for specific GPU count.

    Args:
        num_gpus: Number of GPUs to use

    Returns:
        Configuration dictionary

    Raises:
        RuntimeError: If requested GPUs not available
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")

    available = torch.cuda.device_count()
    if num_gpus > available:
        raise RuntimeError(
            f"Requested {num_gpus} GPUs but only {available} available"
        )

    gpu_ids = list(range(num_gpus))
    return {
        'num_gpus': num_gpus,
        'gpu_ids': gpu_ids,
        'gpu_str': ','.join(map(str, gpu_ids)),
        'config_name': f'{num_gpus}gpu'
    }


# ==================== Comparison Utilities ====================

def compare_outputs(reference: Path,
                   test: Path,
                   rtol: float = 1e-3,
                   atol: float = 1e-3) -> Dict[str, Any]:
    """
    Compare benchmark outputs for correctness validation.

    Uses appropriate tolerance for BF16/FP32 comparison.

    Args:
        reference: Path to reference output (e.g., vanilla pipeline)
        test: Path to test output (e.g., optimized pipeline)
        rtol: Relative tolerance (1e-3 for BF16/FP32)
        atol: Absolute tolerance

    Returns:
        Dictionary with:
        - match: Boolean indicating if outputs match within tolerance
        - max_diff: Maximum absolute difference
        - details: Additional comparison details
    """
    import numpy as np
    import pandas as pd

    # Load results (assumes tab-separated prediction files)
    try:
        ref_df = pd.read_csv(reference, sep='\t')
        test_df = pd.read_csv(test, sep='\t')
    except Exception as e:
        return {
            'match': False,
            'error': f"Failed to load files: {e}",
        }

    # Check structure
    if ref_df.shape != test_df.shape:
        return {
            'match': False,
            'error': f"Shape mismatch: {ref_df.shape} vs {test_df.shape}",
        }

    # Compare predictions
    predictions_match = (ref_df['Prediction'] == test_df['Prediction']).all()

    # Compare scores with tolerance
    score_cols = [col for col in ref_df.columns if 'score' in col.lower()]
    max_diffs = {}
    all_match = predictions_match

    for col in score_cols:
        ref_scores = pd.to_numeric(ref_df[col])
        test_scores = pd.to_numeric(test_df[col])

        matches = np.allclose(ref_scores, test_scores, rtol=rtol, atol=atol)
        all_match = all_match and matches

        diffs = np.abs(ref_scores - test_scores)
        max_diffs[col] = float(np.max(diffs))

    return {
        'match': all_match,
        'predictions_match': predictions_match,
        'max_diffs': max_diffs,
        'num_sequences': len(ref_df),
    }
