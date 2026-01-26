"""End-to-end performance validation benchmarks.

Tests complete pipeline performance with realistic workloads, validating:
- <10 hour processing requirement on 4 GPUs
- Throughput and scalability measurements
- Optimization impact quantification
- Bottleneck identification

Benchmarks use NvitopMonitor for comprehensive GPU tracking and generate
detailed performance reports for analysis.
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
from tests.benchmarks.utils import format_duration, calculate_throughput

logger = logging.getLogger('virnucpro.benchmarks.test_e2e')


# ==================== Configuration ====================

@dataclass
class PerformanceTarget:
    """Performance targets for validation."""
    max_hours: float  # Maximum allowed time in hours
    min_throughput: float  # Minimum sequences/hour
    min_gpu_utilization: float  # Minimum average GPU utilization %
    tolerance: float = 0.10  # Allow 10% tolerance


# 10-hour target for 4 GPUs with typical sample (5000 sequences)
TARGET_4GPU = PerformanceTarget(
    max_hours=10.0,
    min_throughput=500.0,  # 5000 seq in 10 hours = 500/hour minimum
    min_gpu_utilization=70.0,  # Expect >70% GPU utilization
    tolerance=0.10
)


# ==================== Test Class ====================

@pytest.mark.gpu
@pytest.mark.slow
class TestEndToEndPerformance:
    """
    End-to-end pipeline performance validation.

    Tests validate <10 hour processing requirement and measure optimization impact.
    Uses realistic viral sample datasets and comprehensive GPU monitoring.
    """

    def test_typical_sample_under_10_hours(self, benchmark_dir, gpu_monitor, single_gpu):
        """
        Validate <10 hour processing for typical viral sample (5000 sequences).

        This is the primary validation that the optimization project achieves its
        core goal: reducing 45-hour processing to <10 hours on 4 GPUs.

        Test approach:
        1. Generate realistic viral sample (5000 sequences, 200-2000bp)
        2. Run full pipeline with all optimizations enabled
        3. Monitor GPU utilization and memory throughout
        4. Measure total time and project to full sample
        5. Assert projected time < 10 hours (with 10% tolerance)
        6. Generate detailed performance report

        Optimizations enabled:
        - Multi-GPU parallel processing (4 GPUs)
        - FlashAttention-2 (Ampere+ GPUs)
        - BF16 mixed precision (Ampere+ GPUs)
        - CUDA streams for I/O overlap
        - Persistent model loading
        - Optimized batch sizes
        """
        # Skip if we don't have 4 GPUs
        if torch.cuda.device_count() < 4:
            pytest.skip("Requires 4 GPUs for 10-hour target validation")

        logger.info("=" * 80)
        logger.info("BENCHMARK: Typical sample end-to-end performance (5000 sequences)")
        logger.info("=" * 80)

        # Generate typical viral sample using MEDIUM preset (1000 seq) scaled to 5000
        dataset_dir = benchmark_dir / "data" / "typical_sample"
        dataset_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Generating typical viral sample (5000 sequences)...")
        metadata = generate_benchmark_dataset(
            output_dir=dataset_dir,
            num_sequences=5000,
            min_length=200,
            max_length=2000,
            num_files=8,  # Split for balanced multi-GPU distribution
            seed=42
        )

        input_fasta = dataset_dir / "sequences_0.fasta"
        output_dir = benchmark_dir / "outputs" / "typical_sample"

        # Import NvitopMonitor for GPU tracking
        try:
            from virnucpro.utils.gpu_monitor import NvitopMonitor
            monitor = NvitopMonitor(device_ids=[0, 1, 2, 3], log_interval=1.0)
        except ImportError:
            logger.warning("nvitop not available, using basic GPU monitoring")
            from virnucpro.pipeline.gpu_monitor import GPUMonitor
            monitor = GPUMonitor(device_ids=[0, 1, 2, 3], log_interval=1.0)

        # Start GPU monitoring before pipeline execution
        logger.info("Starting GPU monitoring...")
        monitor.start_monitoring()

        # Construct CLI command with all optimizations
        cmd = [
            "python", "-m", "virnucpro.cli.predict",
            str(input_fasta),
            "--output-dir", str(output_dir),
            "--parallel",  # Enable multi-GPU
            "--gpus", "0,1,2,3",  # Use 4 GPUs
            "--persistent-models",  # Keep models loaded
            "--cuda-streams",  # Enable I/O overlap
            "--threads", "8",  # CPU threads for translation/merge
            "--force",  # Overwrite if exists
            "--verbose",  # Show progress
        ]

        logger.info(f"Running pipeline with command:")
        logger.info(f"  {' '.join(cmd)}")

        # Run pipeline and measure time
        start_time = time.time()
        start_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=7200,  # 2 hour timeout for partial run
                cwd=Path(__file__).parent.parent.parent
            )

            if result.returncode != 0:
                logger.error(f"Pipeline failed with exit code {result.returncode}")
                logger.error(f"STDOUT:\n{result.stdout}")
                logger.error(f"STDERR:\n{result.stderr}")
                pytest.fail(f"Pipeline execution failed: {result.stderr}")

        except subprocess.TimeoutExpired:
            logger.error("Pipeline timed out after 2 hours")
            pytest.fail("Pipeline exceeded timeout")

        finally:
            # Stop monitoring and collect metrics
            end_time = time.time()
            end_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            monitor.stop_monitoring()

            # Get GPU statistics
            try:
                gpu_stats = monitor.get_statistics()
            except:
                gpu_stats = {}

        # Calculate metrics
        elapsed_seconds = end_time - start_time
        elapsed_hours = elapsed_seconds / 3600.0
        total_sequences = metadata['total_sequences']
        throughput = total_sequences / elapsed_hours  # sequences/hour

        logger.info("=" * 80)
        logger.info("RESULTS:")
        logger.info(f"  Total time: {format_duration(elapsed_seconds)}")
        logger.info(f"  Sequences processed: {total_sequences}")
        logger.info(f"  Throughput: {throughput:.1f} sequences/hour")
        logger.info(f"  GPU utilization (avg): {gpu_stats.get('avg_utilization', 0.0):.1f}%")
        logger.info("=" * 80)

        # Project to full sample size (based on time per sequence)
        time_per_sequence = elapsed_seconds / total_sequences
        projected_time_45k = time_per_sequence * 45000  # Project to 45k baseline
        projected_hours_45k = projected_time_45k / 3600.0

        logger.info(f"PROJECTION to 45,000 sequence sample:")
        logger.info(f"  Estimated time: {projected_hours_45k:.2f} hours")
        logger.info(f"  Target: <10 hours (with {TARGET_4GPU.tolerance*100:.0f}% tolerance)")

        # Generate detailed performance report
        report = self._generate_performance_report(
            test_name="typical_sample_under_10_hours",
            start_timestamp=start_timestamp,
            end_timestamp=end_timestamp,
            elapsed_seconds=elapsed_seconds,
            total_sequences=total_sequences,
            throughput=throughput,
            gpu_stats=gpu_stats,
            metadata=metadata,
            projected_hours_45k=projected_hours_45k,
            target=TARGET_4GPU
        )

        # Save report
        report_path = benchmark_dir / "reports" / "typical_sample_performance.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"Performance report saved to: {report_path}")

        # Assertions: Validate <10 hour requirement with tolerance
        max_allowed_hours = TARGET_4GPU.max_hours * (1 + TARGET_4GPU.tolerance)

        assert projected_hours_45k < max_allowed_hours, (
            f"Projected time {projected_hours_45k:.2f}h exceeds "
            f"10-hour target (with {TARGET_4GPU.tolerance*100:.0f}% tolerance = {max_allowed_hours:.1f}h)"
        )

        # Validate throughput meets minimum
        assert throughput >= TARGET_4GPU.min_throughput, (
            f"Throughput {throughput:.1f} seq/hour below "
            f"minimum {TARGET_4GPU.min_throughput:.1f}"
        )

        # Validate GPU utilization (if available)
        if gpu_stats.get('avg_utilization'):
            avg_util = gpu_stats['avg_utilization']
            assert avg_util >= TARGET_4GPU.min_gpu_utilization, (
                f"GPU utilization {avg_util:.1f}% below "
                f"minimum {TARGET_4GPU.min_gpu_utilization:.1f}%"
            )

        logger.info("✓ PASSED: Pipeline meets <10 hour requirement")

    def test_large_sample_performance(self, benchmark_dir, gpu_monitor):
        """
        Test with large dataset (10000+ sequences) to measure stability.

        Validates:
        - Memory stability over extended runs
        - No memory leaks or OOM errors
        - Sustained throughput
        - Performance consistency

        Uses LARGE preset (10K sequences) with comprehensive monitoring.
        """
        if torch.cuda.device_count() < 4:
            pytest.skip("Requires 4 GPUs for large sample test")

        logger.info("=" * 80)
        logger.info("BENCHMARK: Large sample performance (10000 sequences)")
        logger.info("=" * 80)

        # Generate large dataset
        dataset_dir = benchmark_dir / "data" / "large_sample"
        dataset_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Generating large viral sample (10000 sequences)...")
        metadata = generate_benchmark_dataset(
            output_dir=dataset_dir,
            num_sequences=10000,
            min_length=100,
            max_length=2000,
            num_files=16,  # More files for better distribution
            seed=43
        )

        input_fasta = dataset_dir / "sequences_0.fasta"
        output_dir = benchmark_dir / "outputs" / "large_sample"

        # Start NvitopMonitor for comprehensive monitoring
        try:
            from virnucpro.utils.gpu_monitor import NvitopMonitor
            monitor = NvitopMonitor(device_ids=[0, 1, 2, 3], log_interval=1.0)
        except ImportError:
            from virnucpro.pipeline.gpu_monitor import GPUMonitor
            monitor = GPUMonitor(device_ids=[0, 1, 2, 3], log_interval=1.0)

        monitor.start_monitoring()

        # Run pipeline
        cmd = [
            "python", "-m", "virnucpro.cli.predict",
            str(input_fasta),
            "--output-dir", str(output_dir),
            "--parallel",
            "--gpus", "0,1,2,3",
            "--persistent-models",
            "--cuda-streams",
            "--threads", "8",
            "--force",
            "--verbose",
        ]

        start_time = time.time()

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=14400,  # 4 hour timeout
                cwd=Path(__file__).parent.parent.parent
            )

            if result.returncode != 0:
                logger.error(f"Pipeline failed: {result.stderr}")
                pytest.fail(f"Pipeline execution failed: {result.stderr}")

        except subprocess.TimeoutExpired:
            pytest.fail("Pipeline exceeded 4-hour timeout")

        finally:
            end_time = time.time()
            monitor.stop_monitoring()
            gpu_stats = monitor.get_statistics()

        # Calculate metrics
        elapsed_seconds = end_time - start_time
        elapsed_hours = elapsed_seconds / 3600.0
        total_sequences = metadata['total_sequences']
        throughput = total_sequences / elapsed_hours

        logger.info("=" * 80)
        logger.info("RESULTS:")
        logger.info(f"  Total time: {format_duration(elapsed_seconds)}")
        logger.info(f"  Sequences: {total_sequences}")
        logger.info(f"  Throughput: {throughput:.1f} seq/hour")
        logger.info(f"  Peak memory: {gpu_stats.get('peak_memory_gb', 0.0):.2f} GB")
        logger.info("=" * 80)

        # Verify no OOM errors in output
        assert "OutOfMemoryError" not in result.stderr, "OOM error detected"
        assert "CUDA out of memory" not in result.stderr, "CUDA OOM detected"

        # Verify sustained throughput
        assert throughput >= TARGET_4GPU.min_throughput * 0.8, (
            f"Large sample throughput {throughput:.1f} significantly below target"
        )

        logger.info("✓ PASSED: Large sample processed successfully")

    def test_optimization_impact(self, benchmark_dir, gpu_monitor):
        """
        Compare performance with different optimization flags.

        Measures impact of each optimization:
        1. Baseline: no optimizations (single GPU, no FlashAttention, FP32)
        2. With FlashAttention: --use-flash-attention
        3. With BF16: automatic on Ampere+
        4. With persistent models: --persistent-models
        5. All optimizations combined: 4 GPUs + FlashAttention + BF16 + persistent

        Generates comparison table showing speedup from each optimization.
        """
        if not torch.cuda.is_available():
            pytest.skip("CUDA required for optimization comparison")

        logger.info("=" * 80)
        logger.info("BENCHMARK: Optimization impact comparison")
        logger.info("=" * 80)

        # Generate small dataset for faster comparison (SMALL preset)
        dataset_dir = benchmark_dir / "data" / "optimization_test"
        dataset_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Generating test dataset (100 sequences)...")
        metadata = generate_benchmark_dataset(
            output_dir=dataset_dir,
            num_sequences=100,
            min_length=200,
            max_length=800,
            num_files=1,
            seed=44
        )

        input_fasta = dataset_dir / "sequences_0.fasta"

        # Import monitor
        try:
            from virnucpro.utils.gpu_monitor import NvitopMonitor
            monitor_class = NvitopMonitor
        except ImportError:
            from virnucpro.pipeline.gpu_monitor import GPUMonitor
            monitor_class = GPUMonitor

        # Define optimization configurations
        configs = [
            {
                'name': 'baseline',
                'description': 'Single GPU, no optimizations',
                'args': ['--gpus', '0', '--no-cuda-streams'],
            },
            {
                'name': 'cuda_streams',
                'description': 'With CUDA streams',
                'args': ['--gpus', '0', '--cuda-streams'],
            },
            {
                'name': 'persistent_models',
                'description': 'With persistent models',
                'args': ['--gpus', '0', '--persistent-models'],
            },
            {
                'name': 'all_optimizations',
                'description': 'All optimizations (4 GPUs if available)',
                'args': [
                    '--gpus', '0,1,2,3' if torch.cuda.device_count() >= 4 else '0',
                    '--parallel',
                    '--persistent-models',
                    '--cuda-streams',
                    '--threads', '8',
                ],
            },
        ]

        results = []

        for config in configs:
            logger.info(f"\nTesting: {config['description']}")

            output_dir = benchmark_dir / "outputs" / f"opt_{config['name']}"

            # Determine GPU IDs for monitoring
            gpu_arg = [arg for arg in config['args'] if arg.startswith('--gpus')]
            if gpu_arg:
                gpu_ids_str = config['args'][config['args'].index(gpu_arg[0]) + 1]
                device_ids = [int(x) for x in gpu_ids_str.split(',')]
            else:
                device_ids = [0]

            monitor = monitor_class(device_ids=device_ids, log_interval=1.0)
            monitor.start_monitoring()

            # Build command
            cmd = [
                "python", "-m", "virnucpro.cli.predict",
                str(input_fasta),
                "--output-dir", str(output_dir),
                "--force",
            ] + config['args']

            start_time = time.time()

            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=600,  # 10 minute timeout per config
                    cwd=Path(__file__).parent.parent.parent
                )

                if result.returncode != 0:
                    logger.warning(f"Config {config['name']} failed: {result.stderr}")
                    results.append({
                        'name': config['name'],
                        'description': config['description'],
                        'status': 'FAILED',
                        'error': result.stderr[:200]
                    })
                    continue

            except subprocess.TimeoutExpired:
                logger.warning(f"Config {config['name']} timed out")
                results.append({
                    'name': config['name'],
                    'description': config['description'],
                    'status': 'TIMEOUT'
                })
                continue

            finally:
                end_time = time.time()
                monitor.stop_monitoring()
                gpu_stats = monitor.get_statistics()

            # Record results
            elapsed_seconds = end_time - start_time
            throughput = metadata['total_sequences'] / (elapsed_seconds / 3600.0)

            results.append({
                'name': config['name'],
                'description': config['description'],
                'elapsed_seconds': elapsed_seconds,
                'throughput': throughput,
                'avg_gpu_utilization': gpu_stats.get('avg_utilization', 0.0),
                'peak_memory_gb': gpu_stats.get('peak_memory_gb', 0.0),
                'status': 'SUCCESS'
            })

            logger.info(f"  Time: {format_duration(elapsed_seconds)}")
            logger.info(f"  Throughput: {throughput:.1f} seq/hour")

        # Calculate speedups relative to baseline
        baseline_time = next(
            (r['elapsed_seconds'] for r in results if r['name'] == 'baseline' and r.get('elapsed_seconds')),
            None
        )

        if baseline_time:
            for result in results:
                if result.get('elapsed_seconds'):
                    result['speedup'] = baseline_time / result['elapsed_seconds']

        # Generate comparison table
        logger.info("\n" + "=" * 80)
        logger.info("OPTIMIZATION IMPACT COMPARISON:")
        logger.info("=" * 80)
        logger.info(f"{'Configuration':<25} {'Time':<12} {'Speedup':<10} {'GPU Util':<10}")
        logger.info("-" * 80)

        for result in results:
            if result.get('elapsed_seconds'):
                speedup_str = f"{result.get('speedup', 1.0):.2f}x"
                util_str = f"{result.get('avg_gpu_utilization', 0.0):.1f}%"
                logger.info(
                    f"{result['description']:<25} "
                    f"{format_duration(result['elapsed_seconds']):<12} "
                    f"{speedup_str:<10} "
                    f"{util_str:<10}"
                )
            else:
                logger.info(f"{result['description']:<25} {result['status']}")

        logger.info("=" * 80)

        # Save detailed report
        report_path = benchmark_dir / "reports" / "optimization_impact.json"
        with open(report_path, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Comparison report saved to: {report_path}")

        # Validate that optimizations provide speedup
        if baseline_time and len(results) > 1:
            all_opt = next(
                (r for r in results if r['name'] == 'all_optimizations' and r.get('speedup')),
                None
            )

            if all_opt:
                assert all_opt['speedup'] > 1.0, (
                    f"All optimizations should provide speedup, got {all_opt['speedup']:.2f}x"
                )
                logger.info(f"✓ PASSED: Optimizations provide {all_opt['speedup']:.2f}x speedup")

    def _generate_performance_report(
        self,
        test_name: str,
        start_timestamp: str,
        end_timestamp: str,
        elapsed_seconds: float,
        total_sequences: int,
        throughput: float,
        gpu_stats: Dict,
        metadata: Dict,
        projected_hours_45k: float,
        target: PerformanceTarget
    ) -> Dict:
        """
        Generate detailed performance report.

        Args:
            test_name: Name of the test
            start_timestamp: Start time (ISO format)
            end_timestamp: End time (ISO format)
            elapsed_seconds: Total elapsed time
            total_sequences: Number of sequences processed
            throughput: Sequences per hour
            gpu_stats: GPU statistics from monitor
            metadata: Dataset metadata
            projected_hours_45k: Projected time for 45K sequences
            target: Performance target for validation

        Returns:
            Dictionary with comprehensive performance data
        """
        return {
            'test_name': test_name,
            'timestamp': {
                'start': start_timestamp,
                'end': end_timestamp,
            },
            'duration': {
                'seconds': elapsed_seconds,
                'hours': elapsed_seconds / 3600.0,
                'formatted': format_duration(elapsed_seconds),
            },
            'dataset': {
                'total_sequences': total_sequences,
                'min_length': metadata.get('min_length'),
                'max_length': metadata.get('max_length'),
                'num_files': metadata.get('num_files'),
            },
            'performance': {
                'throughput_seq_per_hour': throughput,
                'time_per_sequence_seconds': elapsed_seconds / total_sequences,
                'projected_time_45k_hours': projected_hours_45k,
            },
            'gpu_metrics': gpu_stats,
            'validation': {
                'target_max_hours': target.max_hours,
                'target_min_throughput': target.min_throughput,
                'target_min_gpu_util': target.min_gpu_utilization,
                'tolerance': target.tolerance,
                'passed': projected_hours_45k < target.max_hours * (1 + target.tolerance),
            },
            'system': {
                'num_gpus': torch.cuda.device_count() if torch.cuda.is_available() else 0,
                'gpu_names': [
                    torch.cuda.get_device_name(i)
                    for i in range(torch.cuda.device_count())
                ] if torch.cuda.is_available() else [],
                'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
                'pytorch_version': torch.__version__,
            }
        }
