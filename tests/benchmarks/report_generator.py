"""Report generation utilities for benchmark results.

Provides comprehensive reporting in markdown and JSON formats for performance validation.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict

logger = logging.getLogger('virnucpro.benchmarks.report')


@dataclass
class BenchmarkResults:
    """Container for all benchmark results."""

    # Metadata
    timestamp: str
    data_size: str  # "tiny", "small", "medium", "large"
    num_sequences: int
    gpus_tested: List[int]

    # GPU Scaling Results
    gpu_scaling: Dict[int, float]  # {num_gpus: total_time}
    speedups: Dict[int, float]  # {num_gpus: speedup_vs_1gpu}

    # Stage Throughput
    stage_throughput: Dict[str, Dict[str, float]]  # {stage: {time, seq_per_sec, gpu_util}}

    # Memory Usage
    memory_usage: Dict[str, float]  # {peak, average, efficiency}

    # Equivalence Validation
    equivalence: Dict[str, Any]  # {predictions_match, max_difference, within_tolerance}

    # End-to-End Performance
    total_time: float
    projected_10k_time: float  # Hours for 10K sequences
    meets_10hr_requirement: bool

    # Pass/Fail Status
    all_tests_passed: bool
    failures: List[str]


class ReportGenerator:
    """Generate comprehensive performance reports from benchmark results."""

    def __init__(self, output_dir: Path):
        """
        Initialize report generator.

        Args:
            output_dir: Directory to save reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_reports(self, results: BenchmarkResults) -> Tuple[Path, Path]:
        """
        Generate both markdown and JSON reports.

        Args:
            results: Aggregated benchmark results

        Returns:
            Tuple of (markdown_path, json_path)
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        md_path = self.output_dir / f"benchmark_{timestamp}.md"
        json_path = self.output_dir / f"benchmark_{timestamp}.json"

        # Generate reports
        markdown_report = self._generate_markdown(results)
        json_report = self._generate_json(results)

        # Save to files
        with open(md_path, 'w') as f:
            f.write(markdown_report)

        with open(json_path, 'w') as f:
            json.dump(json_report, f, indent=2)

        logger.info(f"Reports generated: {md_path}, {json_path}")
        return md_path, json_path

    def _generate_markdown(self, results: BenchmarkResults) -> str:
        """Generate human-readable markdown report."""

        status_emoji = "✅" if results.all_tests_passed else "❌"

        report = [
            f"# VirNucPro Performance Benchmark Report",
            f"",
            f"**Status:** {status_emoji} {'PASS' if results.all_tests_passed else 'FAIL'}",
            f"**Date:** {results.timestamp}",
            f"**Data Size:** {results.data_size} ({results.num_sequences} sequences)",
            f"**GPUs Tested:** {', '.join(map(str, results.gpus_tested))}",
            f"",
            f"---",
            f"",
        ]

        # Executive Summary
        report.extend(self._executive_summary(results))

        # GPU Scaling Performance
        report.extend(self._scaling_section(results))

        # End-to-End Results
        report.extend(self._end_to_end_section(results))

        # Per-Stage Breakdown
        report.extend(self._stage_breakdown_section(results))

        # Memory Usage Profile
        report.extend(self._memory_section(results))

        # Equivalence Validation
        report.extend(self._equivalence_section(results))

        # Failures (if any)
        if results.failures:
            report.extend(self._failures_section(results))

        # Recommendations
        report.extend(self._recommendations_section(results))

        return "\n".join(report)

    def _executive_summary(self, results: BenchmarkResults) -> List[str]:
        """Generate executive summary section."""

        # Key metrics
        best_gpu_config = max(results.speedups.keys()) if results.speedups else 1
        best_speedup = results.speedups.get(best_gpu_config, 1.0)

        avg_gpu_util = sum(
            stage['gpu_util']
            for stage in results.stage_throughput.values()
            if 'gpu_util' in stage
        ) / len(results.stage_throughput) if results.stage_throughput else 0

        lines = [
            f"## Executive Summary",
            f"",
            f"### Key Performance Indicators",
            f"",
            f"| Metric | Value | Requirement | Status |",
            f"|--------|-------|-------------|--------|",
            f"| **GPU Scaling** | {best_speedup:.2f}x with {best_gpu_config} GPUs | ≥ 1.6x for 2 GPUs | {'✅ Pass' if best_speedup >= 1.6 else '❌ Fail'} |",
            f"| **10K Seq Time** | {results.projected_10k_time:.1f} hours | < 10 hours | {'✅ Pass' if results.meets_10hr_requirement else '❌ Fail'} |",
            f"| **Avg GPU Util** | {avg_gpu_util:.1f}% | ≥ 80% | {'✅ Pass' if avg_gpu_util >= 80 else '❌ Fail'} |",
            f"| **Equivalence** | Max diff: {results.equivalence.get('max_difference', 0):.6f} | rtol ≤ 1e-3 | {'✅ Pass' if results.equivalence.get('within_tolerance', False) else '❌ Fail'} |",
            f"",
        ]

        return lines

    def _scaling_section(self, results: BenchmarkResults) -> List[str]:
        """Generate GPU scaling performance section."""

        lines = [
            f"## GPU Scaling Performance",
            f"",
            f"### Speedup vs Number of GPUs",
            f"",
            f"| GPUs | Total Time (s) | Speedup | Efficiency | Status |",
            f"|------|----------------|---------|------------|--------|",
        ]

        baseline_time = results.gpu_scaling.get(1, 0)

        for num_gpus in sorted(results.gpu_scaling.keys()):
            time = results.gpu_scaling[num_gpus]
            speedup = results.speedups.get(num_gpus, 1.0)
            efficiency = (speedup / num_gpus * 100) if num_gpus > 0 else 0

            # Check if meets scaling requirement (≥1.6x for 2 GPUs)
            status = ""
            if num_gpus == 2:
                status = "✅" if speedup >= 1.6 else "❌"

            lines.append(
                f"| {num_gpus} | {time:.2f} | {speedup:.2f}x | {efficiency:.1f}% | {status} |"
            )

        lines.extend([
            f"",
            f"**Analysis:**",
            f"",
        ])

        # Scaling analysis
        if 2 in results.speedups:
            speedup_2gpu = results.speedups[2]
            if speedup_2gpu >= 1.8:
                lines.append(f"- 🎯 **Excellent** scaling: {speedup_2gpu:.2f}x with 2 GPUs (target: ≥1.6x)")
            elif speedup_2gpu >= 1.6:
                lines.append(f"- ✅ **Good** scaling: {speedup_2gpu:.2f}x with 2 GPUs (meets requirement)")
            else:
                lines.append(f"- ⚠️ **Poor** scaling: {speedup_2gpu:.2f}x with 2 GPUs (below 1.6x target)")

        if 4 in results.speedups:
            speedup_4gpu = results.speedups[4]
            efficiency_4gpu = (speedup_4gpu / 4) * 100
            lines.append(f"- 4 GPU efficiency: {efficiency_4gpu:.1f}% (speedup: {speedup_4gpu:.2f}x)")

        lines.append(f"")
        return lines

    def _end_to_end_section(self, results: BenchmarkResults) -> List[str]:
        """Generate end-to-end performance section."""

        lines = [
            f"## End-to-End Performance",
            f"",
            f"### 10-Hour Requirement Validation",
            f"",
            f"| Metric | Value | Requirement | Status |",
            f"|--------|-------|-------------|--------|",
            f"| **Test Dataset** | {results.num_sequences} sequences | - | - |",
            f"| **Measured Time** | {results.total_time:.2f} seconds | - | - |",
            f"| **Projected 10K Time** | {results.projected_10k_time:.2f} hours | < 10 hours | {'✅ Pass' if results.meets_10hr_requirement else '❌ Fail'} |",
            f"",
        ]

        # Calculate throughput
        if results.total_time > 0:
            throughput = results.num_sequences / results.total_time
            lines.extend([
                f"**Throughput:** {throughput:.2f} sequences/second",
                f"",
            ])

        return lines

    def _stage_breakdown_section(self, results: BenchmarkResults) -> List[str]:
        """Generate per-stage performance breakdown."""

        lines = [
            f"## Per-Stage Performance Breakdown",
            f"",
            f"### Stage Timing and GPU Utilization",
            f"",
            f"| Stage | Time (s) | Throughput (seq/s) | GPU Util (%) | Status |",
            f"|-------|----------|-------------------|--------------|--------|",
        ]

        for stage_name, metrics in results.stage_throughput.items():
            time = metrics.get('time', 0)
            throughput = metrics.get('seq_per_sec', 0)
            gpu_util = metrics.get('gpu_util', 0)

            # Check GPU utilization requirement (≥80%)
            status = "✅" if gpu_util >= 80 else "⚠️"

            lines.append(
                f"| {stage_name} | {time:.2f} | {throughput:.2f} | {gpu_util:.1f}% | {status} |"
            )

        lines.extend([
            f"",
            f"**Bottleneck Analysis:**",
            f"",
        ])

        # Identify bottleneck (longest stage)
        if results.stage_throughput:
            bottleneck_stage = max(
                results.stage_throughput.items(),
                key=lambda x: x[1].get('time', 0)
            )
            stage_name, metrics = bottleneck_stage
            time_pct = (metrics.get('time', 0) / results.total_time * 100) if results.total_time > 0 else 0

            lines.append(f"- **Slowest stage:** {stage_name} ({metrics.get('time', 0):.2f}s, {time_pct:.1f}% of total)")

            # Check GPU utilization for embedding stages
            if 'DNABERT' in stage_name or 'ESM-2' in stage_name:
                gpu_util = metrics.get('gpu_util', 0)
                if gpu_util < 80:
                    lines.append(f"- ⚠️ Low GPU utilization in {stage_name}: {gpu_util:.1f}% (target: ≥80%)")

        lines.append(f"")
        return lines

    def _memory_section(self, results: BenchmarkResults) -> List[str]:
        """Generate memory usage profile section."""

        lines = [
            f"## Memory Usage Profile",
            f"",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| **Peak Memory** | {results.memory_usage.get('peak', 0):.2f} GB |",
            f"| **Average Memory** | {results.memory_usage.get('average', 0):.2f} GB |",
            f"| **Memory Efficiency** | {results.memory_usage.get('efficiency', 0):.1f}% |",
            f"",
        ]

        return lines

    def _equivalence_section(self, results: BenchmarkResults) -> List[str]:
        """Generate equivalence validation section."""

        equiv = results.equivalence

        lines = [
            f"## Equivalence Validation",
            f"",
            f"### Accuracy Confirmation",
            f"",
            f"| Check | Result | Tolerance | Status |",
            f"|-------|--------|-----------|--------|",
            f"| **Predictions Match** | {'Yes' if equiv.get('predictions_match', False) else 'No'} | Exact | {'✅' if equiv.get('predictions_match', False) else '❌'} |",
            f"| **Max Difference** | {equiv.get('max_difference', 0):.6f} | rtol ≤ 1e-3 | {'✅' if equiv.get('within_tolerance', False) else '❌'} |",
            f"| **Mean Difference** | {equiv.get('mean_difference', 0):.6f} | - | - |",
            f"",
        ]

        if equiv.get('within_tolerance', False):
            lines.append("✅ **Optimized pipeline produces equivalent results to vanilla baseline.**")
        else:
            lines.append("❌ **Warning:** Numerical differences exceed acceptable tolerance.")

        lines.append("")
        return lines

    def _failures_section(self, results: BenchmarkResults) -> List[str]:
        """Generate failures section if any tests failed."""

        lines = [
            f"## Test Failures",
            f"",
            f"The following tests did not meet requirements:",
            f"",
        ]

        for i, failure in enumerate(results.failures, 1):
            lines.append(f"{i}. {failure}")

        lines.append("")
        return lines

    def _recommendations_section(self, results: BenchmarkResults) -> List[str]:
        """Generate recommendations based on results."""

        lines = [
            f"## Recommendations",
            f"",
        ]

        if results.all_tests_passed:
            lines.extend([
                f"✅ **All performance requirements met.** Pipeline is ready for production deployment.",
                f"",
                f"### Suggested Configuration",
                f"",
            ])

            # Recommend best GPU configuration
            best_gpu_config = max(results.speedups.keys()) if results.speedups else 1
            lines.append(f"- Use **{best_gpu_config} GPUs** for optimal throughput")

            # Recommend based on GPU utilization
            low_util_stages = [
                stage for stage, metrics in results.stage_throughput.items()
                if metrics.get('gpu_util', 0) < 80
            ]

            if low_util_stages:
                lines.extend([
                    f"- Consider increasing batch size for: {', '.join(low_util_stages)}",
                    f"- Run profiler to optimize batch sizes: `virnucpro profile --model dnabert-s`",
                ])
        else:
            lines.extend([
                f"❌ **Performance requirements not met.** Further optimization required.",
                f"",
                f"### Action Items",
                f"",
            ])

            for failure in results.failures:
                if "scaling" in failure.lower():
                    lines.append("- Investigate multi-GPU data loading and batch distribution")
                elif "utilization" in failure.lower():
                    lines.append("- Profile and increase batch sizes to improve GPU utilization")
                elif "10 hour" in failure.lower():
                    lines.append("- Optimize bottleneck stages or add more GPUs")
                elif "equivalence" in failure.lower():
                    lines.append("- Review numerical precision settings and BF16 tolerance")

        lines.append("")
        return lines

    def _generate_json(self, results: BenchmarkResults) -> Dict[str, Any]:
        """Generate machine-readable JSON report."""

        return {
            "metadata": {
                "timestamp": results.timestamp,
                "data_size": results.data_size,
                "num_sequences": results.num_sequences,
                "gpus_tested": results.gpus_tested,
            },
            "performance": {
                "gpu_scaling": results.gpu_scaling,
                "speedups": results.speedups,
                "stage_throughput": results.stage_throughput,
                "memory_usage": results.memory_usage,
                "total_time": results.total_time,
                "projected_10k_time": results.projected_10k_time,
            },
            "validation": {
                "equivalence": results.equivalence,
                "meets_10hr_requirement": results.meets_10hr_requirement,
            },
            "status": {
                "all_tests_passed": results.all_tests_passed,
                "failures": results.failures,
            },
            "summary": {
                "pass": results.all_tests_passed,
                "exit_code": 0 if results.all_tests_passed else 1,
            }
        }


def generate_markdown_report(results: BenchmarkResults, output_dir: Path) -> Path:
    """
    Generate markdown performance report.

    Args:
        results: Aggregated benchmark results
        output_dir: Directory to save report

    Returns:
        Path to generated markdown report
    """
    generator = ReportGenerator(output_dir)
    md_path, _ = generator.generate_reports(results)
    return md_path


def generate_json_report(results: BenchmarkResults, output_dir: Path) -> Path:
    """
    Generate JSON performance report.

    Args:
        results: Aggregated benchmark results
        output_dir: Directory to save report

    Returns:
        Path to generated JSON report
    """
    generator = ReportGenerator(output_dir)
    _, json_path = generator.generate_reports(results)
    return json_path


def compare_runs(run1_path: Path, run2_path: Path) -> Dict[str, Any]:
    """
    Compare two benchmark runs.

    Args:
        run1_path: Path to first JSON report
        run2_path: Path to second JSON report

    Returns:
        Comparison results dictionary
    """
    with open(run1_path) as f:
        run1 = json.load(f)

    with open(run2_path) as f:
        run2 = json.load(f)

    comparison = {
        "run1": {
            "timestamp": run1["metadata"]["timestamp"],
            "total_time": run1["performance"]["total_time"],
        },
        "run2": {
            "timestamp": run2["metadata"]["timestamp"],
            "total_time": run2["performance"]["total_time"],
        },
        "differences": {},
    }

    # Compare total time
    time_diff = run2["performance"]["total_time"] - run1["performance"]["total_time"]
    time_pct = (time_diff / run1["performance"]["total_time"] * 100) if run1["performance"]["total_time"] > 0 else 0

    comparison["differences"]["total_time"] = {
        "absolute": time_diff,
        "percent": time_pct,
    }

    # Compare speedups
    for num_gpus in run1["performance"]["speedups"]:
        if num_gpus in run2["performance"]["speedups"]:
            speedup1 = run1["performance"]["speedups"][num_gpus]
            speedup2 = run2["performance"]["speedups"][num_gpus]

            if num_gpus not in comparison["differences"]:
                comparison["differences"][f"{num_gpus}_gpu_speedup"] = {}

            comparison["differences"][f"{num_gpus}_gpu_speedup"] = {
                "run1": speedup1,
                "run2": speedup2,
                "delta": speedup2 - speedup1,
            }

    return comparison


def identify_improvements(comparison: Dict[str, Any]) -> List[str]:
    """
    Identify performance improvements from comparison.

    Args:
        comparison: Comparison results from compare_runs()

    Returns:
        List of improvement descriptions
    """
    improvements = []

    # Check total time improvement
    time_pct = comparison["differences"]["total_time"]["percent"]
    if time_pct < -5:  # At least 5% faster
        improvements.append(f"Total time improved by {abs(time_pct):.1f}%")

    # Check speedup improvements
    for key, value in comparison["differences"].items():
        if "speedup" in key and isinstance(value, dict):
            delta = value.get("delta", 0)
            if delta > 0.1:  # Speedup improved by at least 0.1x
                improvements.append(f"{key} improved by {delta:.2f}x")

    return improvements


def identify_regressions(comparison: Dict[str, Any]) -> List[str]:
    """
    Identify performance regressions from comparison.

    Args:
        comparison: Comparison results from compare_runs()

    Returns:
        List of regression descriptions
    """
    regressions = []

    # Check total time regression
    time_pct = comparison["differences"]["total_time"]["percent"]
    if time_pct > 10:  # More than 10% slower
        regressions.append(f"Total time regressed by {time_pct:.1f}%")

    # Check speedup regressions
    for key, value in comparison["differences"].items():
        if "speedup" in key and isinstance(value, dict):
            delta = value.get("delta", 0)
            if delta < -0.1:  # Speedup decreased by at least 0.1x
                regressions.append(f"{key} regressed by {abs(delta):.2f}x")

    return regressions


def generate_comparison_report(
    run1_path: Path,
    run2_path: Path,
    output_dir: Path
) -> Path:
    """
    Generate comparison report for two benchmark runs.

    Args:
        run1_path: Path to first JSON report (baseline)
        run2_path: Path to second JSON report (current)
        output_dir: Directory to save comparison report

    Returns:
        Path to generated comparison markdown report
    """
    comparison = compare_runs(run1_path, run2_path)
    improvements = identify_improvements(comparison)
    regressions = identify_regressions(comparison)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"comparison_{timestamp}.md"

    lines = [
        "# Benchmark Comparison Report",
        "",
        f"**Baseline:** {comparison['run1']['timestamp']}",
        f"**Current:** {comparison['run2']['timestamp']}",
        "",
        "## Summary",
        "",
    ]

    if regressions:
        lines.extend([
            "❌ **Performance Regressions Detected**",
            "",
            "### Regressions",
            "",
        ])
        for reg in regressions:
            lines.append(f"- {reg}")
        lines.append("")

    if improvements:
        lines.extend([
            "✅ **Performance Improvements**",
            "",
            "### Improvements",
            "",
        ])
        for imp in improvements:
            lines.append(f"- {imp}")
        lines.append("")

    if not regressions and not improvements:
        lines.append("➡️ **No significant performance changes**")
        lines.append("")

    lines.extend([
        "## Detailed Metrics",
        "",
        "| Metric | Baseline | Current | Delta |",
        "|--------|----------|---------|-------|",
    ])

    # Total time
    time_diff = comparison["differences"]["total_time"]["absolute"]
    time_pct = comparison["differences"]["total_time"]["percent"]
    lines.append(
        f"| Total Time | {comparison['run1']['total_time']:.2f}s | "
        f"{comparison['run2']['total_time']:.2f}s | "
        f"{time_diff:+.2f}s ({time_pct:+.1f}%) |"
    )

    # Speedups
    for key, value in comparison["differences"].items():
        if "speedup" in key and isinstance(value, dict):
            lines.append(
                f"| {key.replace('_', ' ').title()} | {value['run1']:.2f}x | "
                f"{value['run2']:.2f}x | {value['delta']:+.2f}x |"
            )

    lines.append("")

    with open(report_path, 'w') as f:
        f.write("\n".join(lines))

    logger.info(f"Comparison report generated: {report_path}")
    return report_path
