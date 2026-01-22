"""Pytest fixtures for VirNucPro tests"""

import pytest
import torch
from pathlib import Path
import tempfile
import shutil
from typing import List


@pytest.fixture
def temp_dir():
    """Create temporary directory for test outputs"""
    tmpdir = tempfile.mkdtemp()
    yield Path(tmpdir)
    shutil.rmtree(tmpdir)


@pytest.fixture
def temp_fasta(temp_dir):
    """Generate temporary FASTA file with test sequences"""
    def _generate_fasta(num_sequences: int = 10, seq_length: int = 500):
        fasta_file = temp_dir / "test_sequences.fa"

        import random
        random.seed(42)

        bases = ['A', 'T', 'G', 'C']
        with open(fasta_file, 'w') as f:
            for i in range(num_sequences):
                seq = ''.join(random.choice(bases) for _ in range(seq_length))
                f.write(f">test_seq_{i}\n")
                f.write(f"{seq}\n")

        return fasta_file

    return _generate_fasta


@pytest.fixture
def mock_gpu_devices(monkeypatch):
    """Mock GPU detection for testing without actual GPUs"""
    def _mock_detection(num_gpus: int = 2):
        def mock_is_available():
            return num_gpus > 0

        def mock_device_count():
            return num_gpus

        monkeypatch.setattr(torch.cuda, "is_available", mock_is_available)
        monkeypatch.setattr(torch.cuda, "device_count", mock_device_count)

    return _mock_detection


def pytest_configure(config):
    """Register custom markers"""
    config.addinivalue_line(
        "markers", "gpu: mark test as requiring GPU (skip if no GPU available)"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )


def pytest_collection_modifyitems(config, items):
    """Skip GPU tests if no GPU available"""
    skip_gpu = pytest.mark.skip(reason="GPU not available")
    for item in items:
        if "gpu" in item.keywords:
            if not torch.cuda.is_available():
                item.add_marker(skip_gpu)
