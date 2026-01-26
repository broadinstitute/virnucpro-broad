"""Performance regression tracking for CI integration.

Detects performance degradations and maintains baseline performance data.
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict

logger = logging.getLogger('virnucpro.benchmarks.regression')


@dataclass
class RegressionThresholds:
    """Thresholds for detecting performance regressions."""

    # Critical thresholds (failures)
    max_total_time_increase_pct: float = 10.0  # >10% slowdown is failure
    max_gpu_utilization_decrease_pct: float = 5.0  # >5% reduction is warning
    max_memory_increase_pct: float = 20.0  # >20% increase is warning

    # Speedup thresholds
    min_speedup_2gpu: float = 1.6  # Minimum acceptable 2-GPU speedup
    min_speedup_decrease: float = 0.1  # >0.1x speedup decrease is concerning


@dataclass
class RegressionResult:
    """Result of regression check."""

    has_regression: bool
    failures: List[str]
    warnings: List[str]
    improvements: List[str]
    metrics_comparison: Dict[str, Any]


class RegressionTracker:
    """Track performance over time and detect regressions."""

    def __init__(
        self,
        baseline_file: Optional[Path] = None,
        thresholds: Optional[RegressionThresholds] = None
    ):
        """
        Initialize regression tracker.

        Args:
            baseline_file: Path to baseline JSON file (default: tests/benchmarks/baseline.json)
            thresholds: Custom regression thresholds (default: RegressionThresholds())
        """
        if baseline_file is None:
            baseline_file = Path(__file__).parent / "baseline.json"

        self.baseline_file = Path(baseline_file)
        self.thresholds = thresholds or RegressionThresholds()
        self.baseline = self._load_baseline()

    def _load_baseline(self) -> Optional[Dict[str, Any]]:
        """
        Load baseline performance data from JSON file.

        Returns:
            Baseline data dictionary or None if not found
        """
        if not self.baseline_file.exists():
            logger.warning(f"Baseline file not found: {self.baseline_file}")
            return None

        try:
            with open(self.baseline_file) as f:
                baseline = json.load(f)

            logger.info(f"Loaded baseline from {self.baseline_file}")
            logger.info(f"Baseline timestamp: {baseline.get('metadata', {}).get('timestamp', 'unknown')}")

            return baseline

        except Exception as e:
            logger.error(f"Failed to load baseline: {e}")
            return None

    def update_baseline(self, new_results: Dict[str, Any], force: bool = False):
        """
        Update baseline with new results.

        Args:
            new_results: New benchmark results dictionary (from JSON report)
            force: Force update even if new results are worse
        """
        # Check if new results are better or comparable
        if not force and self.baseline is not None:
            regression = self.check_regression(new_results)

            if regression.has_regression:
                logger.warning("New results show regression. Use force=True to update anyway.")
                logger.warning(f"Failures: {regression.failures}")
                return

        # Create baseline directory if needed
        self.baseline_file.parent.mkdir(parents=True, exist_ok=True)

        # Add metadata
        baseline_data = {
            **new_results,
            "baseline_metadata": {
                "created": datetime.now().isoformat(),
                "source": "update_baseline",
            }
        }

        # Save baseline
        with open(self.baseline_file, 'w') as f:
            json.dump(baseline_data, f, indent=2)

        logger.info(f"Updated baseline: {self.baseline_file}")
        self.baseline = baseline_data

    def check_regression(self, current_results: Dict[str, Any]) -> RegressionResult:
        """
        Compare current results to baseline and detect regressions.

        Args:
            current_results: Current benchmark results (from JSON report)

        Returns:
            RegressionResult with detected issues
        """
        failures = []
        warnings = []
        improvements = []
        metrics = {}

        if self.baseline is None:
            logger.warning("No baseline available for comparison")
            return RegressionResult(
                has_regression=False,
                failures=[],
                warnings=["No baseline available for comparison"],
                improvements=[],
                metrics_comparison={}
            )

        # Extract performance data
        baseline_perf = self.baseline.get('performance', {})
        current_perf = current_results.get('performance', {})

        # Check total time regression
        baseline_time = baseline_perf.get('total_time', 0)
        current_time = current_perf.get('total_time', 0)

        if baseline_time > 0:
            time_increase_pct = ((current_time - baseline_time) / baseline_time) * 100

            metrics['total_time'] = {
                'baseline': baseline_time,
                'current': current_time,
                'change_pct': time_increase_pct,
            }

            if time_increase_pct > self.thresholds.max_total_time_increase_pct:
                failures.append(
                    f"Total time regressed by {time_increase_pct:.1f}% "
                    f"(threshold: {self.thresholds.max_total_time_increase_pct}%)"
                )
            elif time_increase_pct < -5:  # 5% improvement
                improvements.append(f"Total time improved by {abs(time_increase_pct):.1f}%")

        # Check GPU speedup regression
        baseline_speedups = baseline_perf.get('speedups', {})
        current_speedups = current_perf.get('speedups', {})

        for num_gpus_str in ['2', '4', '8']:
            num_gpus = int(num_gpus_str) if isinstance(num_gpus_str, str) else num_gpus_str

            baseline_speedup = baseline_speedups.get(num_gpus_str, baseline_speedups.get(num_gpus, 0))
            current_speedup = current_speedups.get(num_gpus_str, current_speedups.get(num_gpus, 0))

            if baseline_speedup > 0:
                speedup_delta = current_speedup - baseline_speedup

                metrics[f'{num_gpus}gpu_speedup'] = {
                    'baseline': baseline_speedup,
                    'current': current_speedup,
                    'delta': speedup_delta,
                }

                # Check if 2-GPU speedup meets minimum requirement
                if num_gpus == 2:
                    if current_speedup < self.thresholds.min_speedup_2gpu:
                        failures.append(
                            f"2-GPU speedup {current_speedup:.2f}x below requirement "
                            f"({self.thresholds.min_speedup_2gpu}x)"
                        )

                # Check for speedup regression
                if speedup_delta < -self.thresholds.min_speedup_decrease:
                    warnings.append(
                        f"{num_gpus}-GPU speedup decreased by {abs(speedup_delta):.2f}x "
                        f"({baseline_speedup:.2f}x → {current_speedup:.2f}x)"
                    )
                elif speedup_delta > 0.1:  # Improvement
                    improvements.append(
                        f"{num_gpus}-GPU speedup improved by {speedup_delta:.2f}x "
                        f"({baseline_speedup:.2f}x → {current_speedup:.2f}x)"
                    )

        # Check GPU utilization
        baseline_throughput = baseline_perf.get('stage_throughput', {})
        current_throughput = current_perf.get('stage_throughput', {})

        for stage_name in current_throughput:
            if stage_name in baseline_throughput:
                baseline_util = baseline_throughput[stage_name].get('gpu_util', 0)
                current_util = current_throughput[stage_name].get('gpu_util', 0)

                if baseline_util > 0:
                    util_decrease_pct = ((baseline_util - current_util) / baseline_util) * 100

                    metrics[f'{stage_name}_gpu_util'] = {
                        'baseline': baseline_util,
                        'current': current_util,
                        'change_pct': -util_decrease_pct,  # Negative = decrease
                    }

                    if util_decrease_pct > self.thresholds.max_gpu_utilization_decrease_pct:
                        warnings.append(
                            f"{stage_name} GPU utilization decreased by {util_decrease_pct:.1f}% "
                            f"({baseline_util:.1f}% → {current_util:.1f}%)"
                        )

        # Check memory usage
        baseline_memory = baseline_perf.get('memory_usage', {})
        current_memory = current_perf.get('memory_usage', {})

        baseline_peak = baseline_memory.get('peak', 0)
        current_peak = current_memory.get('peak', 0)

        if baseline_peak > 0:
            memory_increase_pct = ((current_peak - baseline_peak) / baseline_peak) * 100

            metrics['peak_memory'] = {
                'baseline': baseline_peak,
                'current': current_peak,
                'change_pct': memory_increase_pct,
            }

            if memory_increase_pct > self.thresholds.max_memory_increase_pct:
                warnings.append(
                    f"Peak memory increased by {memory_increase_pct:.1f}% "
                    f"({baseline_peak:.2f}GB → {current_peak:.2f}GB)"
                )
            elif memory_increase_pct < -5:  # 5% improvement
                improvements.append(
                    f"Peak memory reduced by {abs(memory_increase_pct):.1f}% "
                    f"({baseline_peak:.2f}GB → {current_peak:.2f}GB)"
                )

        # Determine if has regression (failures present)
        has_regression = len(failures) > 0

        return RegressionResult(
            has_regression=has_regression,
            failures=failures,
            warnings=warnings,
            improvements=improvements,
            metrics_comparison=metrics
        )

    def track_trend(
        self,
        history_file: Optional[Path] = None,
        max_history: int = 100
    ) -> Dict[str, List[Any]]:
        """
        Track performance metrics over multiple runs.

        Args:
            history_file: Path to history JSON file (default: tests/benchmarks/history.json)
            max_history: Maximum number of historical entries to keep

        Returns:
            Dictionary of metric trends
        """
        if history_file is None:
            history_file = Path(__file__).parent / "history.json"

        history_file = Path(history_file)

        # Load existing history
        if history_file.exists():
            try:
                with open(history_file) as f:
                    history = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load history: {e}")
                history = {'runs': []}
        else:
            history = {'runs': []}

        # Ensure runs list exists and limit size
        runs = history.get('runs', [])
        if len(runs) >= max_history:
            runs = runs[-(max_history - 1):]  # Keep most recent max_history-1

        # Extract trends
        trends = {
            'timestamps': [],
            'total_time': [],
            'speedup_2gpu': [],
            'speedup_4gpu': [],
            'peak_memory': [],
        }

        for run in runs:
            perf = run.get('performance', {})
            meta = run.get('metadata', {})

            trends['timestamps'].append(meta.get('timestamp', ''))
            trends['total_time'].append(perf.get('total_time', 0))

            speedups = perf.get('speedups', {})
            trends['speedup_2gpu'].append(speedups.get('2', speedups.get(2, 0)))
            trends['speedup_4gpu'].append(speedups.get('4', speedups.get(4, 0)))

            memory = perf.get('memory_usage', {})
            trends['peak_memory'].append(memory.get('peak', 0))

        return trends

    def predict_threshold_breach(
        self,
        trends: Dict[str, List[Any]],
        metric: str,
        threshold: float,
        direction: str = 'above'
    ) -> Optional[int]:
        """
        Predict when a metric will breach a threshold based on trend.

        Args:
            trends: Trends dictionary from track_trend()
            metric: Metric name to analyze
            threshold: Threshold value
            direction: 'above' or 'below'

        Returns:
            Number of runs until threshold breach, or None if not trending toward breach
        """
        if metric not in trends:
            return None

        values = trends[metric]
        if len(values) < 3:  # Need at least 3 data points
            return None

        # Calculate linear regression slope
        import statistics
        n = len(values)
        x = list(range(n))
        y = values

        # Calculate slope using least squares
        x_mean = statistics.mean(x)
        y_mean = statistics.mean(y)

        numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))

        if denominator == 0:
            return None

        slope = numerator / denominator
        intercept = y_mean - slope * x_mean

        # Predict future value
        current_value = values[-1]

        if direction == 'above':
            if slope <= 0:  # Not trending upward
                return None
            runs_to_breach = int((threshold - current_value) / slope)
        else:  # below
            if slope >= 0:  # Not trending downward
                return None
            runs_to_breach = int((threshold - current_value) / slope)

        return runs_to_breach if runs_to_breach > 0 else None


def check_regression(
    current_results: Dict[str, Any],
    baseline_file: Optional[Path] = None,
    thresholds: Optional[RegressionThresholds] = None
) -> RegressionResult:
    """
    Check for performance regressions against baseline.

    Args:
        current_results: Current benchmark results (from JSON report)
        baseline_file: Path to baseline JSON file
        thresholds: Custom regression thresholds

    Returns:
        RegressionResult with detected issues
    """
    tracker = RegressionTracker(baseline_file=baseline_file, thresholds=thresholds)
    return tracker.check_regression(current_results)


def update_baseline(
    new_results: Dict[str, Any],
    baseline_file: Optional[Path] = None,
    force: bool = False
):
    """
    Update baseline performance data.

    Args:
        new_results: New benchmark results (from JSON report)
        baseline_file: Path to baseline JSON file
        force: Force update even if results show regression
    """
    tracker = RegressionTracker(baseline_file=baseline_file)
    tracker.update_baseline(new_results, force=force)


def github_comment_format(regression_result: RegressionResult) -> str:
    """
    Format regression results as GitHub PR comment.

    Args:
        regression_result: Result from check_regression()

    Returns:
        Markdown-formatted comment string
    """
    lines = [
        "## Benchmark Results",
        "",
    ]

    if regression_result.has_regression:
        lines.extend([
            "### ❌ Performance Regressions Detected",
            "",
        ])

        if regression_result.failures:
            lines.append("**Failures:**")
            for failure in regression_result.failures:
                lines.append(f"- ❌ {failure}")
            lines.append("")

    else:
        lines.extend([
            "### ✅ No Performance Regressions",
            "",
        ])

    if regression_result.warnings:
        lines.append("**Warnings:**")
        for warning in regression_result.warnings:
            lines.append(f"- ⚠️ {warning}")
        lines.append("")

    if regression_result.improvements:
        lines.append("**Improvements:**")
        for improvement in regression_result.improvements:
            lines.append(f"- ✅ {improvement}")
        lines.append("")

    if regression_result.metrics_comparison:
        lines.extend([
            "<details>",
            "<summary>Detailed Metrics Comparison</summary>",
            "",
            "| Metric | Baseline | Current | Change |",
            "|--------|----------|---------|--------|",
        ])

        for metric_name, values in regression_result.metrics_comparison.items():
            baseline = values.get('baseline', 0)
            current = values.get('current', 0)
            change = values.get('change_pct', values.get('delta', 0))

            # Format based on metric type
            if 'speedup' in metric_name:
                lines.append(f"| {metric_name} | {baseline:.2f}x | {current:.2f}x | {change:+.2f}x |")
            elif 'memory' in metric_name:
                lines.append(f"| {metric_name} | {baseline:.2f}GB | {current:.2f}GB | {change:+.1f}% |")
            elif 'time' in metric_name:
                lines.append(f"| {metric_name} | {baseline:.2f}s | {current:.2f}s | {change:+.1f}% |")
            else:
                lines.append(f"| {metric_name} | {baseline:.2f} | {current:.2f} | {change:+.1f}% |")

        lines.extend([
            "",
            "</details>",
        ])

    return "\n".join(lines)


def exit_code_for_regressions(regression_result: RegressionResult) -> int:
    """
    Get exit code based on regression results.

    Args:
        regression_result: Result from check_regression()

    Returns:
        Exit code (0 = pass, 1 = regression detected)
    """
    return 1 if regression_result.has_regression else 0


def regression_badge_data(regression_result: RegressionResult) -> Dict[str, str]:
    """
    Generate badge data for README shields.io integration.

    Args:
        regression_result: Result from check_regression()

    Returns:
        Dictionary with badge parameters
    """
    if regression_result.has_regression:
        return {
            "schemaVersion": 1,
            "label": "performance",
            "message": "regression",
            "color": "red"
        }
    elif regression_result.warnings:
        return {
            "schemaVersion": 1,
            "label": "performance",
            "message": "warning",
            "color": "yellow"
        }
    else:
        return {
            "schemaVersion": 1,
            "label": "performance",
            "message": "passing",
            "color": "green"
        }
