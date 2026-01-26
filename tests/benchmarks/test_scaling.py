"""GPU scaling validation benchmarks for multi-GPU performance testing.

This module tests:
1. Linear speedup across 1, 2, 4, 8 GPU configurations
2. Persistent model loading impact on scaling
3. End-to-end pipeline throughput scaling
4. GPU utilization during scaling tests

Scaling expectations (PERF-01):
- 2 GPUs: 1.6-2.0x speedup
- 4 GPUs: 3.0-4.0x speedup
- 8 GPUs: 6.0-8.0x speedup
"""

import pytest
import subprocess
import time
import torch
from pathlib import Path
from typing import Dict, List, Tuple
import json
import logging

from tests.benchmarks.data_generator import generate_benchmark_dataset, PRESETS
from tests.benchmarks.utils import (
    calculate_scaling_metrics,
    check_linear_scaling,
    save_results,
    ScalingResult
)
from virnucpro.utils.gpu_monitor import NvitopMonitor

logger = logging.getLogger('virnucpro.benchmarks.scaling')


@pytest.mark.gpu
class TestGPUScaling:
    """
    Multi-GPU scaling validation tests.

    Verifies near-linear speedup as GPU count increases.
    Tests with/without persistent models to quantify optimization impact.
    """

    @pytest.fixture(scope="class")
    def medium_dataset(self, tmp_path_factory):
        """
        Generate medium-sized synthetic dataset for scaling tests.

        Uses MEDIUM preset (1000 sequences) as good balance between:
        - Long enough to show GPU scaling effects
        - Short enough for reasonable test time

        Returns:
            Path to dataset directory
        """
        output_dir = tmp_path_factory.mktemp("scaling_data")
        dataset_dir = generate_benchmark_dataset(
            preset='MEDIUM',
            output_dir=output_dir,
            num_files=1,
            seed=42
        )
        return dataset_dir

    @pytest.fixture(scope="class")
    def model_path(self):
        """
        Path to model file for predictions.

        Uses default 500bp model from models/ directory.

        Returns:
            Path to model checkpoint
        """
        # Assume models are in standard location
        model_dir = Path("models")
        model_file = model_dir / "VirusBinaryClassifier_500bp.pth"

        if not model_file.exists():
            pytest.skip(f"Model not found at {model_file}")

        return model_file

    def run_pipeline(self,
                     input_fasta: Path,
                     output_dir: Path,
                     gpu_ids: List[int],
                     persistent: bool = False,
                     timeout: int = 600) -> Tuple[float, Dict]:
        """
        Run pipeline with specified GPU configuration.

        Args:
            input_fasta: Path to input FASTA file
            output_dir: Output directory for results
            gpu_ids: List of GPU device IDs to use
            persistent: Whether to use --persistent-models flag
            timeout: Maximum execution time in seconds

        Returns:
            Tuple of (execution_time, stats_dict)
        """
        # Prepare command
        cmd = [
            "virnucpro", "predict",
            str(input_fasta),
            "--output-dir", str(output_dir),
            "--model-type", "500",
            "--gpus", ",".join(map(str, gpu_ids)),
            "--parallel",
            "--no-progress",  # Disable progress bars for clean timing
        ]

        if persistent:
            cmd.append("--persistent-models")

        logger.info(f"Running pipeline with {len(gpu_ids)} GPUs: {' '.join(cmd)}")

        # Start GPU monitoring
        monitor = NvitopMonitor(device_ids=gpu_ids, log_interval=1.0)
        monitor.start_monitoring()

        # Run pipeline and measure time
        start_time = time.time()

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=True
            )
            elapsed = time.time() - start_time

        except subprocess.TimeoutExpired:
            monitor.stop_monitoring()
            raise RuntimeError(f"Pipeline timed out after {timeout}s")

        except subprocess.CalledProcessError as e:
            monitor.stop_monitoring()
            raise RuntimeError(f"Pipeline failed: {e.stderr}")

        # Stop monitoring and collect stats
        stats = monitor.stop_monitoring()

        logger.info(f"Pipeline completed in {elapsed:.2f}s with {len(gpu_ids)} GPUs")

        return elapsed, stats

    @pytest.mark.parametrize("num_gpus", [1, 2, 4, 8])
    def test_linear_scaling_synthetic(self,
                                      num_gpus: int,
                                      medium_dataset: Path,
                                      tmp_path: Path):
        """
        Test linear scaling across GPU configurations.

        Verifies that adding GPUs provides near-linear speedup:
        - 2 GPUs: 1.6-2.0x speedup
        - 4 GPUs: 3.0-4.0x speedup
        - 8 GPUs: 6.0-8.0x speedup

        Args:
            num_gpus: Number of GPUs to use (parametrized)
            medium_dataset: Path to test dataset
            tmp_path: Temporary directory for outputs
        """
        # Skip if insufficient GPUs
        available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if num_gpus > available_gpus:
            pytest.skip(f"Requires {num_gpus} GPUs, only {available_gpus} available")

        # Get input FASTA file
        metadata_path = medium_dataset / "metadata.json"
        with open(metadata_path) as f:
            metadata = json.load(f)

        input_fasta = medium_dataset / metadata['files'][0]['filename']

        # Create output directory
        output_dir = tmp_path / f"output_{num_gpus}gpu"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Select GPUs
        gpu_ids = list(range(num_gpus))

        # Run pipeline
        elapsed, stats = self.run_pipeline(
            input_fasta=input_fasta,
            output_dir=output_dir,
            gpu_ids=gpu_ids,
            persistent=False
        )

        # Store result for cross-test analysis
        result_file = tmp_path.parent / "scaling_results.json"

        # Load existing results or create new
        if result_file.exists():
            with open(result_file) as f:
                all_results = json.load(f)
        else:
            all_results = {}

        # Add this result
        all_results[f"{num_gpus}gpu"] = {
            'num_gpus': num_gpus,
            'time_seconds': elapsed,
            'gpu_stats': stats,
        }

        # Save results
        with open(result_file, 'w') as f:
            json.dump(all_results, f, indent=2)

        logger.info(f"{num_gpus} GPU(s): {elapsed:.2f}s")

        # Verify basic performance (not too slow)
        # MEDIUM dataset (1000 sequences) should complete in reasonable time
        max_time_per_gpu = {1: 600, 2: 400, 4: 300, 8: 200}  # Conservative limits
        assert elapsed < max_time_per_gpu[num_gpus], \
            f"{num_gpus} GPU(s) took {elapsed:.2f}s, expected < {max_time_per_gpu[num_gpus]}s"

    def test_validate_scaling_ratios(self, tmp_path: Path):
        """
        Validate scaling ratios after all GPU configs have run.

        This test runs AFTER the parametrized tests to analyze aggregate results.
        Checks that speedup ratios meet thresholds.
        """
        result_file = tmp_path.parent / "scaling_results.json"

        if not result_file.exists():
            pytest.skip("No scaling results found - run parametrized tests first")

        with open(result_file) as f:
            all_results = json.load(f)

        if '1gpu' not in all_results:
            pytest.skip("No baseline (1 GPU) result found")

        baseline_time = all_results['1gpu']['time_seconds']

        # Calculate speedup ratios
        scaling_data = []
        for config_name, result in sorted(all_results.items()):
            num_gpus = result['num_gpus']
            time_seconds = result['time_seconds']

            speedup = baseline_time / time_seconds
            efficiency = speedup / num_gpus

            scaling_data.append({
                'num_gpus': num_gpus,
                'time_seconds': time_seconds,
                'speedup': speedup,
                'efficiency': efficiency
            })

            logger.info(f"{num_gpus} GPU(s): {time_seconds:.2f}s, "
                       f"speedup={speedup:.2f}x, efficiency={efficiency*100:.1f}%")

        # Verify scaling thresholds
        thresholds = {
            2: (1.6, 2.0),  # 1.6-2.0x speedup for 2 GPUs
            4: (3.0, 4.0),  # 3.0-4.0x speedup for 4 GPUs
            8: (6.0, 8.0),  # 6.0-8.0x speedup for 8 GPUs
        }

        for entry in scaling_data:
            num_gpus = entry['num_gpus']
            speedup = entry['speedup']

            if num_gpus in thresholds:
                min_speedup, max_speedup = thresholds[num_gpus]
                assert min_speedup <= speedup <= max_speedup, \
                    f"{num_gpus} GPUs: speedup {speedup:.2f}x outside range [{min_speedup}, {max_speedup}]"

        # Save formatted report
        report_path = tmp_path / "scaling_report.md"
        self._generate_scaling_report(scaling_data, report_path)

        logger.info(f"Scaling report saved to {report_path}")

    def test_scaling_with_persistent_models(self,
                                           medium_dataset: Path,
                                           tmp_path: Path):
        """
        Test scaling with --persistent-models flag.

        Verifies that persistent models provide additional speedup
        by eliminating model reload overhead between stages.

        Compares 2 GPU performance with/without persistent models.
        """
        available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if available_gpus < 2:
            pytest.skip("Requires at least 2 GPUs for meaningful comparison")

        # Get input FASTA file
        metadata_path = medium_dataset / "metadata.json"
        with open(metadata_path) as f:
            metadata = json.load(f)

        input_fasta = medium_dataset / metadata['files'][0]['filename']

        # Test without persistent models
        output_dir_standard = tmp_path / "output_2gpu_standard"
        output_dir_standard.mkdir(parents=True, exist_ok=True)

        time_standard, stats_standard = self.run_pipeline(
            input_fasta=input_fasta,
            output_dir=output_dir_standard,
            gpu_ids=[0, 1],
            persistent=False
        )

        # Test with persistent models
        output_dir_persistent = tmp_path / "output_2gpu_persistent"
        output_dir_persistent.mkdir(parents=True, exist_ok=True)

        time_persistent, stats_persistent = self.run_pipeline(
            input_fasta=input_fasta,
            output_dir=output_dir_persistent,
            gpu_ids=[0, 1],
            persistent=True
        )

        # Calculate speedup from persistent models
        speedup = time_standard / time_persistent

        logger.info(f"Standard (2 GPUs): {time_standard:.2f}s")
        logger.info(f"Persistent (2 GPUs): {time_persistent:.2f}s")
        logger.info(f"Persistent models speedup: {speedup:.2f}x")

        # Persistent models should provide at least 10% speedup
        # (Conservative threshold - may be higher in practice)
        min_speedup = 1.1
        assert speedup >= min_speedup, \
            f"Persistent models speedup {speedup:.2f}x < {min_speedup}x minimum"

        # Save comparison report
        comparison = {
            'standard': {
                'time_seconds': time_standard,
                'gpu_stats': stats_standard,
            },
            'persistent': {
                'time_seconds': time_persistent,
                'gpu_stats': stats_persistent,
            },
            'speedup': speedup,
        }

        report_path = tmp_path / "persistent_comparison.json"
        with open(report_path, 'w') as f:
            json.dump(comparison, f, indent=2)

    def _generate_scaling_report(self,
                                 scaling_data: List[Dict],
                                 output_path: Path):
        """
        Generate markdown scaling report.

        Args:
            scaling_data: List of scaling result dictionaries
            output_path: Path to save markdown report
        """
        lines = []
        lines.append("# GPU Scaling Benchmark Report\n")
        lines.append(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        lines.append("## Results\n")
        lines.append("| GPUs | Time (s) | Speedup | Efficiency | Status |")
        lines.append("|------|----------|---------|------------|--------|")

        thresholds = {
            2: (1.6, 2.0),
            4: (3.0, 4.0),
            8: (6.0, 8.0),
        }

        for entry in scaling_data:
            num_gpus = entry['num_gpus']
            time_s = entry['time_seconds']
            speedup = entry['speedup']
            efficiency = entry['efficiency']

            # Check status
            if num_gpus == 1:
                status = "✓ Baseline"
            elif num_gpus in thresholds:
                min_s, max_s = thresholds[num_gpus]
                if min_s <= speedup <= max_s:
                    status = "✓ Pass"
                else:
                    status = f"✗ Fail (expected {min_s:.1f}-{max_s:.1f}x)"
            else:
                status = "- Not tested"

            lines.append(
                f"| {num_gpus} | {time_s:.2f} | "
                f"{speedup:.2f}x | {efficiency*100:.1f}% | {status} |"
            )

        lines.append("\n## Thresholds\n")
        lines.append("- 2 GPUs: 1.6-2.0x speedup")
        lines.append("- 4 GPUs: 3.0-4.0x speedup")
        lines.append("- 8 GPUs: 6.0-8.0x speedup")
        lines.append("\n## Notes\n")
        lines.append("- Efficiency = Speedup / Number of GPUs")
        lines.append("- Linear scaling would show 100% efficiency")
        lines.append("- 80-90% efficiency is excellent for multi-GPU workloads")

        report_text = '\n'.join(lines)

        with open(output_path, 'w') as f:
            f.write(report_text)

        logger.info(f"Scaling report:\n{report_text}")
