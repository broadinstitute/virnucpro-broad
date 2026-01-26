"""Benchmark suite for VirNucPro performance validation.

This package contains the benchmark infrastructure for validating GPU optimizations
and multi-GPU scaling performance. It provides:

1. Synthetic data generation for controlled scaling tests
2. GPU monitoring and metrics collection (nvitop-based)
3. Pytest-benchmark integration for automated testing
4. Multi-GPU configuration testing (1, 2, 4, 8 GPUs)
5. Throughput and latency measurements per pipeline stage
6. Performance regression detection

Structure:
- conftest.py: pytest fixtures for benchmark configuration and GPU monitoring
- data_generator.py: Synthetic FASTA generation for controlled tests
- utils.py: Timing utilities, result formatting, multi-GPU helpers
- test_*.py: Actual benchmark test suites

Usage:
    # Run all benchmarks with default settings
    pytest tests/benchmarks/

    # Run with JSON output for CI
    pytest tests/benchmarks/ --benchmark-json=results.json

    # Run specific GPU configuration
    pytest tests/benchmarks/ -k "test_scaling[2gpu]"

    # Generate synthetic test data
    python -c "from tests.benchmarks.data_generator import generate_benchmark_dataset; \
               generate_benchmark_dataset('medium', output_dir='tests/data/synthetic')"

See Also:
- tests/data/synthetic/: Generated test datasets (gitignored)
- tests/reports/: Benchmark reports in markdown and JSON (gitignored)
- virnucpro/pipeline/profiler.py: Batch size profiling utilities
- virnucpro/pipeline/gpu_monitor.py: Base GPU monitoring (to be enhanced)
"""

__all__ = [
    'generate_synthetic_fasta',
    'generate_benchmark_dataset',
    'BenchmarkTimer',
    'format_results',
]
