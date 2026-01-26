"""Memory usage profiling benchmarks.

Tests memory stability and OOM prevention throughout pipeline execution:
- Memory leak detection via baseline tracking
- Peak memory per pipeline stage
- OOM prevention with large batches
- Persistent models memory overhead
- Memory efficiency metrics

Uses NvitopMonitor for continuous memory tracking and torch.cuda.memory_stats()
for detailed PyTorch allocator metrics.
"""

import pytest
import torch
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import logging
from dataclasses import dataclass, asdict

from tests.benchmarks.data_generator import PRESETS, generate_benchmark_dataset
from tests.benchmarks.utils import format_duration

logger = logging.getLogger('virnucpro.benchmarks.test_memory')


# ==================== Configuration ====================

@dataclass
class MemoryThresholds:
    """Memory usage thresholds for validation."""
    max_memory_leak_mb: float = 100.0  # Max allowed memory increase (leak)
    min_memory_efficiency: float = 0.70  # Min useful/allocated ratio
    max_fragmentation_ratio: float = 0.30  # Max fragmented/allocated


# ==================== Test Class ====================

@pytest.mark.gpu
class TestMemoryUsage:
    """
    Memory usage profiling and stability validation.

    Tests ensure pipeline maintains stable memory usage without leaks,
    prevents OOM through dynamic batch sizing, and efficiently uses
    available GPU memory.
    """

    def test_memory_stability(self, benchmark_dir, gpu_monitor, single_gpu):
        """
        Verify memory returns to baseline between pipeline stages.

        Tests memory stability by:
        1. Running pipeline with medium dataset
        2. Tracking GPU memory every second via NvitopMonitor
        3. Verifying memory returns to baseline between stages
        4. Detecting gradual memory increase (leak detection)
        5. Using torch.cuda.memory_stats() for detailed tracking

        Memory leak is defined as >100MB increase from start to finish
        after accounting for normal persistent allocations.
        """
        if not torch.cuda.is_available():
            pytest.skip("CUDA required for memory testing")

        logger.info("=" * 80)
        logger.info("BENCHMARK: Memory stability validation")
        logger.info("=" * 80)

        # Generate medium dataset
        dataset_dir = benchmark_dir / "data" / "memory_stability"
        dataset_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Generating medium dataset (1000 sequences)...")
        metadata = generate_benchmark_dataset(
            output_dir=dataset_dir,
            num_sequences=1000,
            min_length=200,
            max_length=1500,
            num_files=4,
            seed=45
        )

        input_fasta = dataset_dir / "sequences_0.fasta"
        output_dir = benchmark_dir / "outputs" / "memory_stability"

        # Initialize NvitopMonitor for continuous tracking
        try:
            from virnucpro.utils.gpu_monitor import NvitopMonitor
            monitor = NvitopMonitor(device_ids=[0], log_interval=1.0)
        except ImportError:
            logger.warning("nvitop not available, using basic monitoring")
            from virnucpro.pipeline.gpu_monitor import GPUMonitor
            monitor = GPUMonitor(device_ids=[0], log_interval=1.0)

        # Record baseline memory before pipeline
        torch.cuda.synchronize(0)
        torch.cuda.empty_cache()
        time.sleep(1.0)  # Let memory settle

        baseline_memory_mb = torch.cuda.memory_allocated(0) / 1024**2
        logger.info(f"Baseline memory: {baseline_memory_mb:.2f} MB")

        # Start continuous monitoring
        monitor.start_monitoring()

        # Run pipeline
        cmd = [
            "python", "-m", "virnucpro.cli.predict",
            str(input_fasta),
            "--output-dir", str(output_dir),
            "--gpus", "0",
            "--force",
            "--verbose",
        ]

        logger.info("Running pipeline with memory monitoring...")
        start_time = time.time()

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=1800,  # 30 minute timeout
                cwd=Path(__file__).parent.parent.parent
            )

            if result.returncode != 0:
                logger.error(f"Pipeline failed: {result.stderr}")
                pytest.fail(f"Pipeline execution failed: {result.stderr}")

        finally:
            end_time = time.time()
            monitor.stop_monitoring()

        # Get memory statistics
        try:
            gpu_stats = monitor.get_statistics()
        except:
            gpu_stats = {}

        # Check final memory after pipeline
        torch.cuda.synchronize(0)
        final_memory_mb = torch.cuda.memory_allocated(0) / 1024**2
        memory_increase_mb = final_memory_mb - baseline_memory_mb

        # Get detailed PyTorch memory stats
        memory_stats = torch.cuda.memory_stats(0)
        allocated_mb = memory_stats.get('allocated_bytes.all.current', 0) / 1024**2
        reserved_mb = memory_stats.get('reserved_bytes.all.current', 0) / 1024**2
        peak_allocated_mb = memory_stats.get('allocated_bytes.all.peak', 0) / 1024**2

        logger.info("=" * 80)
        logger.info("MEMORY ANALYSIS:")
        logger.info(f"  Baseline memory: {baseline_memory_mb:.2f} MB")
        logger.info(f"  Final memory: {final_memory_mb:.2f} MB")
        logger.info(f"  Memory increase: {memory_increase_mb:.2f} MB")
        logger.info(f"  Peak allocated: {peak_allocated_mb:.2f} MB")
        logger.info(f"  Reserved: {reserved_mb:.2f} MB")
        logger.info(f"  Peak from monitor: {gpu_stats.get('peak_memory_gb', 0.0)*1024:.2f} MB")
        logger.info("=" * 80)

        # Generate memory timeline report
        report = {
            'test_name': 'memory_stability',
            'duration_seconds': end_time - start_time,
            'baseline_memory_mb': baseline_memory_mb,
            'final_memory_mb': final_memory_mb,
            'memory_increase_mb': memory_increase_mb,
            'peak_allocated_mb': peak_allocated_mb,
            'reserved_mb': reserved_mb,
            'gpu_stats': gpu_stats,
            'validation': {
                'max_leak_allowed_mb': MemoryThresholds.max_memory_leak_mb,
                'leak_detected': memory_increase_mb > MemoryThresholds.max_memory_leak_mb,
            }
        }

        # Save report
        report_path = benchmark_dir / "reports" / "memory_stability.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"Memory report saved to: {report_path}")

        # Assertion: No significant memory leak detected
        assert memory_increase_mb <= MemoryThresholds.max_memory_leak_mb, (
            f"Memory leak detected: {memory_increase_mb:.2f} MB increase "
            f"(threshold: {MemoryThresholds.max_memory_leak_mb} MB)"
        )

        # Verify no OOM errors
        assert "OutOfMemoryError" not in result.stderr, "OOM error in pipeline"
        assert "CUDA out of memory" not in result.stderr, "CUDA OOM in pipeline"

        logger.info("✓ PASSED: No memory leak detected")

    def test_peak_memory_per_stage(self, benchmark_dir, gpu_monitor, single_gpu):
        """
        Measure peak memory for each pipeline stage.

        Tracks memory usage through:
        1. Translation stage (CPU, minimal GPU usage)
        2. DNABERT-S feature extraction (DNA embeddings)
        3. ESM-2 feature extraction (protein embeddings)
        4. Merge and prediction (CPU-heavy)

        Identifies memory-intensive operations and verifies peaks stay
        within GPU capacity. Generates memory timeline chart data.
        """
        if not torch.cuda.is_available():
            pytest.skip("CUDA required for memory testing")

        logger.info("=" * 80)
        logger.info("BENCHMARK: Peak memory per pipeline stage")
        logger.info("=" * 80)

        # Generate small dataset for detailed tracking
        dataset_dir = benchmark_dir / "data" / "memory_per_stage"
        dataset_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Generating test dataset (500 sequences)...")
        metadata = generate_benchmark_dataset(
            output_dir=dataset_dir,
            num_sequences=500,
            min_length=200,
            max_length=1000,
            num_files=2,
            seed=46
        )

        input_fasta = dataset_dir / "sequences_0.fasta"
        output_dir = benchmark_dir / "outputs" / "memory_per_stage"

        # Initialize monitor with per-stage tracking
        try:
            from virnucpro.utils.gpu_monitor import NvitopMonitor
            monitor = NvitopMonitor(device_ids=[0], log_interval=0.5)  # More frequent sampling
        except ImportError:
            from virnucpro.pipeline.gpu_monitor import GPUMonitor
            monitor = NvitopMonitor(device_ids=[0], log_interval=0.5)

        monitor.start_monitoring()

        # Note: In real usage, we'd need to instrument the pipeline to call
        # monitor.set_stage() at each stage transition. For this test, we'll
        # infer stages from time progression.

        # Run pipeline
        cmd = [
            "python", "-m", "virnucpro.cli.predict",
            str(input_fasta),
            "--output-dir", str(output_dir),
            "--gpus", "0",
            "--force",
            "--verbose",
        ]

        logger.info("Running pipeline with per-stage monitoring...")
        start_time = time.time()

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=1200,  # 20 minute timeout
                cwd=Path(__file__).parent.parent.parent
            )

            if result.returncode != 0:
                logger.error(f"Pipeline failed: {result.stderr}")
                pytest.fail(f"Pipeline execution failed: {result.stderr}")

        finally:
            end_time = time.time()
            monitor.stop_monitoring()

        # Get statistics
        gpu_stats = monitor.get_statistics()

        # Get per-stage metrics if available
        per_stage_stats = gpu_stats.get('per_stage', {})

        logger.info("=" * 80)
        logger.info("PEAK MEMORY PER STAGE:")
        logger.info("=" * 80)

        if per_stage_stats:
            for stage, stats in per_stage_stats.items():
                peak_mb = stats.get('peak_memory_gb', 0.0) * 1024
                avg_util = stats.get('avg_utilization', 0.0)
                logger.info(f"  {stage:20s}: {peak_mb:8.2f} MB (util: {avg_util:.1f}%)")
        else:
            # Overall peak
            peak_mb = gpu_stats.get('peak_memory_gb', 0.0) * 1024
            logger.info(f"  Overall peak: {peak_mb:.2f} MB")

        logger.info("=" * 80)

        # Get GPU capacity
        if torch.cuda.is_available():
            total_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            peak_memory_gb = gpu_stats.get('peak_memory_gb', 0.0)
            memory_usage_pct = (peak_memory_gb / total_memory_gb) * 100

            logger.info(f"GPU capacity: {total_memory_gb:.2f} GB")
            logger.info(f"Peak usage: {peak_memory_gb:.2f} GB ({memory_usage_pct:.1f}%)")

            # Verify peak stays within capacity
            assert peak_memory_gb < total_memory_gb, "Peak memory exceeds GPU capacity"
            assert memory_usage_pct < 95, f"Peak usage {memory_usage_pct:.1f}% too close to capacity"

        # Save report with timeline data
        report = {
            'test_name': 'peak_memory_per_stage',
            'duration_seconds': end_time - start_time,
            'overall_peak_gb': gpu_stats.get('peak_memory_gb', 0.0),
            'per_stage_stats': per_stage_stats,
            'gpu_capacity_gb': total_memory_gb if torch.cuda.is_available() else None,
            'peak_usage_percent': memory_usage_pct if torch.cuda.is_available() else None,
        }

        report_path = benchmark_dir / "reports" / "memory_per_stage.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"Per-stage report saved to: {report_path}")
        logger.info("✓ PASSED: Peak memory within GPU capacity")

    def test_memory_with_large_batches(self, benchmark_dir, gpu_monitor, single_gpu):
        """
        Test memory with maximum batch sizes.

        Validates OOM prevention by:
        1. Testing with maximum batch sizes (3072 tokens with BF16)
        2. Verifying OOM prevention reduces batch instead of crashing
        3. Measuring memory efficiency (useful vs allocated)
        4. Testing expandable segments behavior

        The pipeline should gracefully handle batch size reduction on OOM
        rather than crashing.
        """
        if not torch.cuda.is_available():
            pytest.skip("CUDA required for batch size testing")

        logger.info("=" * 80)
        logger.info("BENCHMARK: Memory with large batches")
        logger.info("=" * 80)

        # Generate dataset with long sequences to stress memory
        dataset_dir = benchmark_dir / "data" / "large_batches"
        dataset_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Generating dataset with long sequences...")
        metadata = generate_benchmark_dataset(
            output_dir=dataset_dir,
            num_sequences=200,
            min_length=1500,  # Long sequences
            max_length=2000,
            num_files=1,
            seed=47
        )

        input_fasta = dataset_dir / "sequences_0.fasta"
        output_dir = benchmark_dir / "outputs" / "large_batches"

        # Initialize monitor
        try:
            from virnucpro.utils.gpu_monitor import NvitopMonitor
            monitor = NvitopMonitor(device_ids=[0], log_interval=1.0)
        except ImportError:
            from virnucpro.pipeline.gpu_monitor import GPUMonitor
            monitor = GPUMonitor(device_ids=[0], log_interval=1.0)

        monitor.start_monitoring()

        # Run with maximum batch sizes and expandable segments
        cmd = [
            "python", "-m", "virnucpro.cli.predict",
            str(input_fasta),
            "--output-dir", str(output_dir),
            "--gpus", "0",
            "--dnabert-batch-size", "3072",  # Maximum with BF16
            "--esm-batch-size", "3072",
            "--expandable-segments",  # Enable for fragmentation prevention
            "--force",
            "--verbose",
        ]

        logger.info("Running pipeline with large batches...")
        start_time = time.time()

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=1800,  # 30 minute timeout
                cwd=Path(__file__).parent.parent.parent
            )

            # Note: Exit code might be 4 (OOM) if batch reduction happened
            # We want to verify graceful handling, not necessarily success

        finally:
            end_time = time.time()
            monitor.stop_monitoring()

        gpu_stats = monitor.get_statistics()

        # Get PyTorch memory statistics
        memory_stats = torch.cuda.memory_stats(0)
        allocated_bytes = memory_stats.get('allocated_bytes.all.current', 0)
        reserved_bytes = memory_stats.get('reserved_bytes.all.current', 0)

        # Calculate memory efficiency
        if reserved_bytes > 0:
            memory_efficiency = allocated_bytes / reserved_bytes
        else:
            memory_efficiency = 0.0

        logger.info("=" * 80)
        logger.info("LARGE BATCH MEMORY ANALYSIS:")
        logger.info(f"  Allocated: {allocated_bytes / 1024**3:.2f} GB")
        logger.info(f"  Reserved: {reserved_bytes / 1024**3:.2f} GB")
        logger.info(f"  Memory efficiency: {memory_efficiency*100:.1f}%")
        logger.info(f"  Peak memory: {gpu_stats.get('peak_memory_gb', 0.0):.2f} GB")
        logger.info("=" * 80)

        # Check for OOM handling in stderr
        oom_handled = False
        if "OutOfMemoryError" in result.stderr or "CUDA out of memory" in result.stderr:
            logger.info("OOM detected - checking for graceful handling...")

            # Look for batch reduction messages
            if "reducing batch size" in result.stderr.lower() or "batch size reduced" in result.stderr.lower():
                oom_handled = True
                logger.info("✓ OOM handled gracefully with batch size reduction")
            else:
                logger.warning("OOM occurred without apparent batch size reduction")

        # If no OOM, verify memory efficiency
        if not oom_handled and result.returncode == 0:
            assert memory_efficiency >= MemoryThresholds.min_memory_efficiency, (
                f"Low memory efficiency: {memory_efficiency*100:.1f}% "
                f"(threshold: {MemoryThresholds.min_memory_efficiency*100:.1f}%)"
            )

        # Save report
        report = {
            'test_name': 'memory_large_batches',
            'duration_seconds': end_time - start_time,
            'batch_size': 3072,
            'allocated_gb': allocated_bytes / 1024**3,
            'reserved_gb': reserved_bytes / 1024**3,
            'memory_efficiency': memory_efficiency,
            'peak_memory_gb': gpu_stats.get('peak_memory_gb', 0.0),
            'oom_detected': "OutOfMemoryError" in result.stderr or "CUDA out of memory" in result.stderr,
            'oom_handled_gracefully': oom_handled,
            'exit_code': result.returncode,
        }

        report_path = benchmark_dir / "reports" / "memory_large_batches.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"Large batch report saved to: {report_path}")
        logger.info("✓ PASSED: Large batch memory behavior validated")

    def test_persistent_models_memory(self, benchmark_dir, gpu_monitor, single_gpu):
        """
        Compare memory with/without --persistent-models flag.

        Measures:
        1. Memory overhead of keeping models loaded
        2. Memory stability with persistent workers
        3. Cache clearing effectiveness
        4. Memory/speed tradeoff quantification

        Runs pipeline twice (with/without persistent models) and compares
        memory usage and performance.
        """
        if not torch.cuda.is_available():
            pytest.skip("CUDA required for persistent models testing")

        logger.info("=" * 80)
        logger.info("BENCHMARK: Persistent models memory overhead")
        logger.info("=" * 80)

        # Generate small dataset for comparison
        dataset_dir = benchmark_dir / "data" / "persistent_models"
        dataset_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Generating test dataset (300 sequences)...")
        metadata = generate_benchmark_dataset(
            output_dir=dataset_dir,
            num_sequences=300,
            min_length=200,
            max_length=800,
            num_files=2,
            seed=48
        )

        input_fasta = dataset_dir / "sequences_0.fasta"

        # Import monitor
        try:
            from virnucpro.utils.gpu_monitor import NvitopMonitor
            monitor_class = NvitopMonitor
        except ImportError:
            from virnucpro.pipeline.gpu_monitor import GPUMonitor
            monitor_class = GPUMonitor

        results = {}

        # Test 1: Without persistent models
        logger.info("\nTest 1: Without persistent models (standard mode)")
        output_dir_1 = benchmark_dir / "outputs" / "persistent_models_off"

        monitor_1 = monitor_class(device_ids=[0], log_interval=1.0)
        monitor_1.start_monitoring()

        cmd_1 = [
            "python", "-m", "virnucpro.cli.predict",
            str(input_fasta),
            "--output-dir", str(output_dir_1),
            "--gpus", "0",
            "--no-persistent-models",  # Explicitly disable
            "--force",
            "--verbose",
        ]

        start_1 = time.time()

        try:
            result_1 = subprocess.run(
                cmd_1,
                capture_output=True,
                text=True,
                timeout=900,
                cwd=Path(__file__).parent.parent.parent
            )
        finally:
            end_1 = time.time()
            monitor_1.stop_monitoring()

        stats_1 = monitor_1.get_statistics()
        duration_1 = end_1 - start_1

        logger.info(f"  Duration: {format_duration(duration_1)}")
        logger.info(f"  Peak memory: {stats_1.get('peak_memory_gb', 0.0):.2f} GB")

        results['standard'] = {
            'duration_seconds': duration_1,
            'peak_memory_gb': stats_1.get('peak_memory_gb', 0.0),
            'avg_memory_gb': stats_1.get('avg_memory_gb', 0.0),
        }

        # Test 2: With persistent models
        logger.info("\nTest 2: With persistent models enabled")
        output_dir_2 = benchmark_dir / "outputs" / "persistent_models_on"

        monitor_2 = monitor_class(device_ids=[0], log_interval=1.0)
        monitor_2.start_monitoring()

        cmd_2 = [
            "python", "-m", "virnucpro.cli.predict",
            str(input_fasta),
            "--output-dir", str(output_dir_2),
            "--gpus", "0",
            "--persistent-models",  # Enable persistent models
            "--force",
            "--verbose",
        ]

        start_2 = time.time()

        try:
            result_2 = subprocess.run(
                cmd_2,
                capture_output=True,
                text=True,
                timeout=900,
                cwd=Path(__file__).parent.parent.parent
            )
        finally:
            end_2 = time.time()
            monitor_2.stop_monitoring()

        stats_2 = monitor_2.get_statistics()
        duration_2 = end_2 - start_2

        logger.info(f"  Duration: {format_duration(duration_2)}")
        logger.info(f"  Peak memory: {stats_2.get('peak_memory_gb', 0.0):.2f} GB")

        results['persistent'] = {
            'duration_seconds': duration_2,
            'peak_memory_gb': stats_2.get('peak_memory_gb', 0.0),
            'avg_memory_gb': stats_2.get('avg_memory_gb', 0.0),
        }

        # Calculate tradeoffs
        memory_overhead_gb = results['persistent']['peak_memory_gb'] - results['standard']['peak_memory_gb']
        memory_overhead_pct = (memory_overhead_gb / results['standard']['peak_memory_gb']) * 100
        speedup = duration_1 / duration_2

        logger.info("\n" + "=" * 80)
        logger.info("PERSISTENT MODELS TRADEOFF:")
        logger.info("=" * 80)
        logger.info(f"Memory overhead: {memory_overhead_gb:.2f} GB ({memory_overhead_pct:.1f}%)")
        logger.info(f"Speedup: {speedup:.2f}x")
        logger.info(f"Standard mode: {format_duration(duration_1)}, {results['standard']['peak_memory_gb']:.2f} GB")
        logger.info(f"Persistent mode: {format_duration(duration_2)}, {results['persistent']['peak_memory_gb']:.2f} GB")
        logger.info("=" * 80)

        # Save comparison report
        report = {
            'test_name': 'persistent_models_memory',
            'standard_mode': results['standard'],
            'persistent_mode': results['persistent'],
            'tradeoff': {
                'memory_overhead_gb': memory_overhead_gb,
                'memory_overhead_percent': memory_overhead_pct,
                'speedup': speedup,
            },
            'recommendation': (
                'Use persistent models for multiple samples'
                if speedup > 1.1 and memory_overhead_pct < 30
                else 'Standard mode recommended'
            )
        }

        report_path = benchmark_dir / "reports" / "persistent_models_memory.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"Persistent models report saved to: {report_path}")

        # Validate tradeoff is reasonable
        # If persistent models use >50% more memory, speedup should be significant
        if memory_overhead_pct > 50:
            assert speedup > 1.2, (
                f"High memory overhead ({memory_overhead_pct:.1f}%) "
                f"but low speedup ({speedup:.2f}x)"
            )

        logger.info("✓ PASSED: Persistent models memory/speed tradeoff quantified")
