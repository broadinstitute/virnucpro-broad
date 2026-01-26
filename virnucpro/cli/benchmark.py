"""Benchmark CLI command for running performance validation suite."""

import click
import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime
import logging

logger = logging.getLogger('virnucpro.cli.benchmark')


@click.command()
@click.option('--suite',
              type=click.Choice(['scaling', 'throughput', 'memory', 'equivalence', 'all']),
              default='all',
              help='Which benchmark suite to run (default: all)')
@click.option('--data-size',
              type=click.Choice(['tiny', 'small', 'medium', 'large']),
              default='small',
              help='Test data size (tiny: 10, small: 100, medium: 1K, large: 10K sequences)')
@click.option('--gpus',
              type=str,
              default=None,
              help='GPU IDs to test (e.g., "0,1,2,3"). Auto-detect if not specified.')
@click.option('--output-dir', '-o',
              type=click.Path(),
              default='tests/reports',
              help='Directory to save benchmark reports (default: tests/reports/)')
@click.option('--compare-to',
              type=click.Path(exists=True),
              default=None,
              help='Previous JSON report to compare against')
@click.option('--vanilla-baseline',
              is_flag=True,
              help='Run vanilla comparison (slower, validates correctness)')
@click.option('--quick',
              is_flag=True,
              help='Fast subset for CI (small data, fewer configs)')
@click.pass_context
def benchmark(ctx, suite, data_size, gpus, output_dir, compare_to, vanilla_baseline, quick):
    """
    Run performance benchmark suite.

    This command executes the VirNucPro benchmark suite to validate
    performance requirements:

    \b
    - GPU scaling: ≥1.6x speedup with 2 GPUs
    - 10-hour requirement: <10 hours for 10K sequences with 4 GPUs
    - GPU utilization: ≥80% for embedding stages
    - Equivalence: Optimized results match vanilla (rtol ≤ 1e-3)

    The benchmark suite includes:

    \b
    - Scaling tests: Multi-GPU speedup validation
    - Throughput tests: Per-stage sequences/second
    - Memory tests: Peak usage and efficiency
    - Equivalence tests: Accuracy validation vs vanilla

    \b
    Examples:

      # Quick CI validation (default small dataset)
      virnucpro benchmark --quick

      # Full benchmark suite on 4 GPUs
      virnucpro benchmark --suite all --gpus 0,1,2,3

      # Test GPU scaling only with medium dataset
      virnucpro benchmark --suite scaling --data-size medium

      # Validate equivalence with vanilla baseline
      virnucpro benchmark --vanilla-baseline --data-size small

      # Compare against previous run
      virnucpro benchmark --compare-to tests/reports/benchmark_20260126.json

    \b
    Output:

      - Markdown report: Human-readable performance summary
      - JSON report: Machine-readable results for CI
      - Console summary: Pass/fail status with key metrics

    \b
    Exit codes:

      0: All tests passed
      1: Performance requirements not met
      2: Benchmark execution failed
    """
    logger = ctx.obj['logger']

    logger.info("VirNucPro Performance Benchmark Suite")
    logger.info("=" * 70)

    # Validate CUDA availability
    try:
        import torch
        if not torch.cuda.is_available():
            logger.error("CUDA not available. Benchmarks require GPU.")
            sys.exit(2)

        num_gpus = torch.cuda.device_count()
        logger.info(f"Detected {num_gpus} GPU(s)")

        # Parse GPU list if provided
        if gpus:
            gpu_list = [int(x.strip()) for x in gpus.split(',')]
            # Validate GPU IDs
            for gpu_id in gpu_list:
                if gpu_id >= num_gpus:
                    logger.error(f"GPU {gpu_id} not available (only {num_gpus} GPU(s) detected)")
                    sys.exit(2)
            logger.info(f"Testing GPUs: {gpu_list}")
        else:
            gpu_list = list(range(num_gpus))
            logger.info(f"Auto-detected GPUs: {gpu_list}")

    except Exception as e:
        logger.error(f"GPU detection failed: {e}")
        sys.exit(2)

    # Quick mode adjustments
    if quick:
        data_size = 'small'
        suite = 'scaling'  # Only test scaling in quick mode
        logger.info("Quick mode: Using small dataset, scaling tests only")

    # Setup output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Reports will be saved to: {output_path}")

    # Build pytest command
    pytest_cmd = _build_pytest_command(
        suite=suite,
        data_size=data_size,
        gpu_list=gpu_list,
        vanilla_baseline=vanilla_baseline,
        quick=quick,
        output_dir=output_path
    )

    logger.info("\nRunning benchmarks...")
    logger.info(f"Command: {' '.join(pytest_cmd)}")
    logger.info("")

    # Run pytest
    try:
        result = subprocess.run(
            pytest_cmd,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )

        # Display pytest output
        if result.stdout:
            print(result.stdout)

        if result.stderr:
            print(result.stderr, file=sys.stderr)

        # Check for generated reports
        reports = _find_latest_reports(output_path)

        if reports['json']:
            logger.info(f"\n{'='*70}")
            logger.info("Benchmark Results")
            logger.info(f"{'='*70}\n")

            # Load and display summary
            _display_summary(reports['json'], logger)

            # Generate comparison if requested
            if compare_to:
                _generate_comparison(reports['json'], Path(compare_to), output_path, logger)

            # Display report locations
            logger.info(f"\nReports generated:")
            if reports['markdown']:
                logger.info(f"  Markdown: {reports['markdown']}")
            logger.info(f"  JSON: {reports['json']}")

            # Return appropriate exit code
            exit_code = _get_exit_code_from_report(reports['json'])
            logger.info(f"\nBenchmark {'PASSED' if exit_code == 0 else 'FAILED'}")
            sys.exit(exit_code)

        else:
            # No reports found - pytest may have failed
            logger.error("No benchmark reports generated")
            sys.exit(result.returncode if result.returncode != 0 else 2)

    except subprocess.TimeoutExpired:
        logger.error("Benchmark timed out after 1 hour")
        sys.exit(2)

    except Exception as e:
        logger.error(f"Benchmark execution failed: {e}")
        if ctx.obj['logger'].level == logging.DEBUG:
            logger.exception("Detailed error traceback:")
        sys.exit(2)


def _build_pytest_command(
    suite: str,
    data_size: str,
    gpu_list: list,
    vanilla_baseline: bool,
    quick: bool,
    output_dir: Path
) -> list:
    """
    Build pytest command with appropriate markers and options.

    Args:
        suite: Benchmark suite to run
        data_size: Size of test data
        gpu_list: List of GPU IDs to test
        vanilla_baseline: Whether to run vanilla comparison
        quick: Whether to run quick subset
        output_dir: Output directory for reports

    Returns:
        List of command arguments for subprocess.run
    """
    cmd = [
        'pytest',
        'tests/benchmarks/',
        '-v',  # Verbose output
        '-s',  # Show print statements
        '--tb=short',  # Short traceback
    ]

    # Add markers based on suite
    if suite == 'all':
        cmd.extend(['-m', 'gpu'])
    elif suite == 'scaling':
        cmd.extend(['-m', 'scaling'])
    elif suite == 'throughput':
        cmd.extend(['-m', 'throughput'])
    elif suite == 'memory':
        cmd.extend(['-m', 'memory'])
    elif suite == 'equivalence':
        cmd.extend(['-m', 'equivalence'])

    # Environment variables for data size and GPU configuration
    # These will be passed as env vars to pytest
    env_vars = {
        'BENCHMARK_DATA_SIZE': data_size,
        'BENCHMARK_GPUS': ','.join(map(str, gpu_list)),
        'BENCHMARK_OUTPUT_DIR': str(output_dir),
    }

    if vanilla_baseline:
        env_vars['BENCHMARK_VANILLA'] = '1'

    if quick:
        env_vars['BENCHMARK_QUICK'] = '1'

    # Add environment variables as pytest options
    for key, value in env_vars.items():
        cmd.extend(['--override-ini', f'env={key}={value}'])

    return cmd


def _find_latest_reports(output_dir: Path) -> dict:
    """
    Find latest generated benchmark reports.

    Args:
        output_dir: Directory containing reports

    Returns:
        Dictionary with 'json' and 'markdown' paths (or None)
    """
    # Find most recent benchmark reports
    json_reports = sorted(
        output_dir.glob('benchmark_*.json'),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )

    md_reports = sorted(
        output_dir.glob('benchmark_*.md'),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )

    return {
        'json': json_reports[0] if json_reports else None,
        'markdown': md_reports[0] if md_reports else None,
    }


def _display_summary(json_report_path: Path, logger):
    """
    Display benchmark summary from JSON report.

    Args:
        json_report_path: Path to JSON report
        logger: Logger instance
    """
    try:
        with open(json_report_path) as f:
            report = json.load(f)

        status = report['status']['all_tests_passed']
        status_str = "✅ PASS" if status else "❌ FAIL"

        logger.info(f"Overall Status: {status_str}\n")

        # Display key metrics
        logger.info("Key Performance Indicators:")
        logger.info("-" * 70)

        # GPU Scaling
        speedups = report['performance'].get('speedups', {})
        if '2' in speedups or 2 in speedups:
            speedup_2gpu = speedups.get('2', speedups.get(2, 0))
            scaling_status = "✅" if speedup_2gpu >= 1.6 else "❌"
            logger.info(f"  GPU Scaling (2 GPUs):     {speedup_2gpu:.2f}x {scaling_status}")

        # 10-hour requirement
        projected_time = report['performance'].get('projected_10k_time', 0)
        meets_req = report['validation'].get('meets_10hr_requirement', False)
        time_status = "✅" if meets_req else "❌"
        logger.info(f"  10K Sequences Time:       {projected_time:.1f} hours {time_status}")

        # Equivalence
        equiv = report['validation'].get('equivalence', {})
        max_diff = equiv.get('max_difference', 0)
        within_tol = equiv.get('within_tolerance', False)
        equiv_status = "✅" if within_tol else "❌"
        logger.info(f"  Equivalence (max diff):   {max_diff:.6f} {equiv_status}")

        logger.info("-" * 70)

        # Display failures if any
        failures = report['status'].get('failures', [])
        if failures:
            logger.info(f"\nFailures ({len(failures)}):")
            for i, failure in enumerate(failures, 1):
                logger.info(f"  {i}. {failure}")

    except Exception as e:
        logger.warning(f"Could not parse JSON report: {e}")


def _generate_comparison(
    current_report_path: Path,
    baseline_report_path: Path,
    output_dir: Path,
    logger
):
    """
    Generate comparison report between current and baseline.

    Args:
        current_report_path: Path to current JSON report
        baseline_report_path: Path to baseline JSON report
        output_dir: Directory to save comparison
        logger: Logger instance
    """
    try:
        from tests.benchmarks.report_generator import generate_comparison_report

        comparison_path = generate_comparison_report(
            baseline_report_path,
            current_report_path,
            output_dir
        )

        logger.info(f"\nComparison Report: {comparison_path}")

        # Display key changes
        from tests.benchmarks.report_generator import compare_runs, identify_improvements, identify_regressions

        comparison = compare_runs(baseline_report_path, current_report_path)
        improvements = identify_improvements(comparison)
        regressions = identify_regressions(comparison)

        if improvements:
            logger.info("\nImprovements:")
            for imp in improvements:
                logger.info(f"  ✅ {imp}")

        if regressions:
            logger.info("\nRegressions:")
            for reg in regressions:
                logger.info(f"  ❌ {reg}")

        if not improvements and not regressions:
            logger.info("\nNo significant performance changes")

    except Exception as e:
        logger.warning(f"Could not generate comparison: {e}")


def _get_exit_code_from_report(json_report_path: Path) -> int:
    """
    Extract exit code from JSON report.

    Args:
        json_report_path: Path to JSON report

    Returns:
        Exit code (0 = pass, 1 = fail)
    """
    try:
        with open(json_report_path) as f:
            report = json.load(f)

        return report['summary'].get('exit_code', 1)

    except Exception:
        return 1  # Default to failure if can't read report
