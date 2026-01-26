"""Pytest fixtures for benchmark suite configuration and GPU monitoring.

Provides:
- GPU availability detection (skip tests if no GPU)
- GPU monitor fixture using enhanced NvitopMonitor
- Benchmark configuration (warmup, iterations, output directories)
- Test data paths and synthetic data generation
- Multi-GPU configuration fixtures (1, 2, 4, 8 GPUs)
"""

import pytest
import torch
from pathlib import Path
from typing import List, Optional
import logging

logger = logging.getLogger('virnucpro.benchmarks')


# ==================== GPU Availability ====================

def pytest_configure(config):
    """Register custom markers for GPU tests."""
    config.addinivalue_line(
        "markers", "gpu: mark test as requiring GPU (skip if no GPU available)"
    )
    config.addinivalue_line(
        "markers", "multi_gpu: mark test as requiring multiple GPUs"
    )


def pytest_collection_modifyitems(config, items):
    """Skip GPU tests if CUDA not available."""
    if not torch.cuda.is_available():
        skip_gpu = pytest.mark.skip(reason="CUDA not available")
        for item in items:
            if "gpu" in item.keywords or "multi_gpu" in item.keywords:
                item.add_marker(skip_gpu)

    # Skip multi-GPU tests if only 1 GPU available
    elif torch.cuda.device_count() < 2:
        skip_multi = pytest.mark.skip(reason="Multiple GPUs not available")
        for item in items:
            if "multi_gpu" in item.keywords:
                item.add_marker(skip_multi)


# ==================== Directory Fixtures ====================

@pytest.fixture(scope="session")
def benchmark_dir(tmp_path_factory) -> Path:
    """
    Create temporary directory for benchmark outputs.

    Structure:
        benchmark_dir/
        ├── data/          # Temporary test data
        ├── outputs/       # Pipeline outputs
        └── reports/       # Benchmark reports

    Returns:
        Path to benchmark directory
    """
    base_dir = tmp_path_factory.mktemp("benchmarks")

    # Create subdirectories
    (base_dir / "data").mkdir(exist_ok=True)
    (base_dir / "outputs").mkdir(exist_ok=True)
    (base_dir / "reports").mkdir(exist_ok=True)

    logger.info(f"Benchmark directory: {base_dir}")
    return base_dir


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """
    Path to test data directory (committed small samples and synthetic data).

    Returns:
        Path to tests/data/ directory
    """
    return Path(__file__).parent.parent / "data"


# ==================== GPU Monitoring Fixtures ====================

@pytest.fixture
def gpu_monitor():
    """
    GPU monitoring fixture for tracking utilization and memory during benchmarks.

    Uses enhanced NvitopMonitor (falls back to GPUMonitor if nvitop unavailable).
    Monitor is started before test and stopped after, returning statistics.

    Usage:
        def test_throughput(gpu_monitor):
            gpu_monitor.start()
            # ... run benchmark ...
            stats = gpu_monitor.stop()
            assert stats[0]['gpu_util_avg'] > 80.0  # >80% utilization

    Yields:
        Monitor instance with start() and stop() methods
    """
    # Import here to allow fallback if nvitop not available
    try:
        from virnucpro.utils.gpu_monitor import NvitopMonitor
        monitor_class = NvitopMonitor
    except ImportError:
        from virnucpro.pipeline.gpu_monitor import GPUMonitor
        monitor_class = GPUMonitor
        logger.warning("nvitop not available, using basic GPUMonitor")

    # Detect available GPUs
    if not torch.cuda.is_available():
        # Return mock monitor for CPU-only environments
        class MockMonitor:
            def start(self): pass
            def stop(self): return {}
            def get_statistics(self): return {}

        yield MockMonitor()
        return

    device_ids = list(range(torch.cuda.device_count()))
    monitor = monitor_class(device_ids=device_ids, log_interval=1.0)

    yield monitor

    # Cleanup: ensure monitoring stopped
    try:
        monitor.stop_monitoring()
    except:
        pass


# ==================== Benchmark Configuration ====================

@pytest.fixture(scope="session")
def benchmark_config():
    """
    Benchmark configuration settings.

    Returns:
        Dictionary with:
        - warmup_iterations: Number of warmup runs before timing (default: 2)
        - min_iterations: Minimum timed iterations (default: 3)
        - min_run_time: Minimum total benchmark time in seconds (default: 5.0)
        - timeout: Maximum benchmark time in seconds (default: 300)
    """
    return {
        'warmup_iterations': 2,
        'min_iterations': 3,
        'min_run_time': 5.0,
        'timeout': 300,
    }


# ==================== GPU Configuration Fixtures ====================

@pytest.fixture(params=['1gpu', '2gpu', '4gpu', '8gpu'])
def gpu_config(request):
    """
    Parametrized fixture for multi-GPU configurations.

    Tests using this fixture will run 4 times with different GPU counts.
    Skips configurations with more GPUs than available.

    Usage:
        @pytest.mark.multi_gpu
        def test_scaling(gpu_config):
            num_gpus = gpu_config['num_gpus']
            gpu_ids = gpu_config['gpu_ids']
            # ... run benchmark with specified GPUs ...

    Yields:
        Dictionary with:
        - config_name: '1gpu', '2gpu', '4gpu', or '8gpu'
        - num_gpus: Number of GPUs (1, 2, 4, or 8)
        - gpu_ids: List of GPU device IDs
        - gpu_str: Comma-separated GPU IDs for CLI (e.g., '0,1,2,3')
    """
    config_name = request.param
    num_gpus = int(config_name[0])

    # Skip if requested GPUs not available
    available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if num_gpus > available_gpus:
        pytest.skip(f"Requires {num_gpus} GPUs, only {available_gpus} available")

    gpu_ids = list(range(num_gpus))
    gpu_str = ','.join(map(str, gpu_ids))

    return {
        'config_name': config_name,
        'num_gpus': num_gpus,
        'gpu_ids': gpu_ids,
        'gpu_str': gpu_str,
    }


@pytest.fixture
def single_gpu():
    """
    Fixture for single GPU configuration.

    Returns:
        Dictionary with GPU 0 configuration
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    return {
        'config_name': '1gpu',
        'num_gpus': 1,
        'gpu_ids': [0],
        'gpu_str': '0',
    }


# ==================== pytest-benchmark Configuration ====================

def pytest_benchmark_update_json(config, benchmarks, output_json):
    """
    Customize pytest-benchmark JSON output with additional metadata.

    Adds:
    - PyTorch version
    - CUDA version
    - GPU count and models
    - BF16 support
    """
    if torch.cuda.is_available():
        gpu_info = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gpu_info.append({
                'id': i,
                'name': props.name,
                'memory_gb': props.total_memory / 1024**3,
                'compute_capability': f"{props.major}.{props.minor}",
            })

        # Check BF16 support on GPU 0
        compute_capability = torch.cuda.get_device_capability(0)
        bf16_support = compute_capability[0] >= 8

        output_json['machine_info']['gpu'] = {
            'count': torch.cuda.device_count(),
            'devices': gpu_info,
            'bf16_support': bf16_support,
            'cuda_version': torch.version.cuda,
        }

    output_json['machine_info']['pytorch_version'] = torch.__version__


# ==================== Data Generation Helpers ====================

@pytest.fixture
def synthetic_data_generator(benchmark_dir):
    """
    Fixture providing synthetic data generation function.

    Usage:
        def test_with_data(synthetic_data_generator):
            fasta_path = synthetic_data_generator(num_sequences=100, output_name='test.fa')
            # ... use fasta_path in benchmark ...

    Returns:
        Function: generate(num_sequences, min_length, max_length, output_name)
    """
    from tests.benchmarks.data_generator import generate_synthetic_fasta

    def generate(num_sequences: int = 100,
                 min_length: int = 100,
                 max_length: int = 500,
                 output_name: str = "test.fasta") -> Path:
        """
        Generate synthetic FASTA file in benchmark data directory.

        Args:
            num_sequences: Number of sequences to generate
            min_length: Minimum sequence length
            max_length: Maximum sequence length
            output_name: Output filename

        Returns:
            Path to generated FASTA file
        """
        output_path = benchmark_dir / "data" / output_name
        generate_synthetic_fasta(
            num_sequences=num_sequences,
            min_length=min_length,
            max_length=max_length,
            output_path=output_path
        )
        return output_path

    return generate
