"""CLI integration tests for predict command v2.0 routing.

Verifies that --parallel routes ESM-2 to v2.0 architecture (run_multi_gpu_inference)
while DNABERT-S stays v1.0, and --v1-fallback routes both to v1.0.
Uses mocking to avoid GPU/model dependencies.
"""

import sys
import pytest
from pathlib import Path
from click.testing import CliRunner
from unittest.mock import patch, MagicMock

from virnucpro.cli.main import cli

try:
    from virnucpro.pipeline.runtime_config import RuntimeConfig
    RUNTIME_CONFIG_IMPORTED = True
except ImportError as e:
    RUNTIME_CONFIG_IMPORTED = False
    RUNTIME_CONFIG_IMPORT_ERROR = str(e)

if 'esm' not in sys.modules:
    if not RUNTIME_CONFIG_IMPORTED:
        sys.modules['esm'] = MagicMock()
    else:
        raise ImportError("esm module required but not available")


@pytest.fixture
def tmp_fasta(tmp_path):
    """Create a minimal FASTA file for CLI invocation."""
    fasta_path = tmp_path / "test.fasta"
    fasta_path.write_text(">seq1\nATGC\n>seq2\nCGTA\n")
    return fasta_path


@pytest.fixture
def cli_mocks():
    """Common mocking setup for CLI tests. Creates fresh mocks per test."""
    import torch
    from unittest.mock import MagicMock, patch

    mock_device = MagicMock(return_value=torch.device('cpu'))
    mock_detect = MagicMock(return_value=[0, 1])  # Default: 2 GPUs
    mock_run = MagicMock(return_value=0)  # Success exit code

    def config_get(key, default=None):
        defaults = {
            'prediction.models.500': '/fake/model.pth',
            'prediction.batch_size': 256,
            'prediction.num_workers': 4,
            'features.dnabert.batch_size': 2048,
            'device.fallback_to_cpu': True,
            'files.auto_cleanup': True,
        }
        return defaults.get(key, default)

    mock_config = MagicMock()
    mock_config.return_value.get = config_get

    patchers = [
        patch('virnucpro.cli.predict.validate_and_get_device', mock_device),
        patch('virnucpro.cli.predict.detect_cuda_devices', mock_detect),
        patch('virnucpro.cli.predict.Config', mock_config),
        patch('virnucpro.pipeline.prediction.run_prediction', mock_run),
    ]

    for p in patchers:
        p.start()

    yield {
        'device': mock_device,
        'detect': mock_detect,
        'config': mock_config,
        'run_prediction': mock_run,
    }

    for p in reversed(patchers):
        p.stop()


def test_parallel_routes_esm2_to_v2(tmp_fasta, tmp_path, cli_mocks):
    """Test that --parallel routes ESM-2 to v2.0 architecture by default."""
    if not RUNTIME_CONFIG_IMPORTED:
        pytest.skip(f"RuntimeConfig import failed: {RUNTIME_CONFIG_IMPORT_ERROR}")

    runner = CliRunner()
    result = runner.invoke(cli, [
        'predict',
        str(tmp_fasta),
        '--parallel',
        '--output-dir', str(tmp_path / 'output')
    ])

    # Check CLI succeeded
    assert result.exit_code == 0, f"CLI failed: {result.output}"

    # Verify run_prediction was called
    assert cli_mocks['run_prediction'].called

    # Get the call arguments
    call_kwargs = cli_mocks['run_prediction'].call_args.kwargs

    # Assert v2.0 routing
    assert call_kwargs['use_v2_architecture'] is True, \
        "Expected use_v2_architecture=True for --parallel"

    # Assert RuntimeConfig was passed with correct fields
    assert call_kwargs['runtime_config'] is not None, \
        "Expected runtime_config to be passed"
    runtime_config = call_kwargs['runtime_config']
    assert isinstance(runtime_config, RuntimeConfig), \
        "Expected RuntimeConfig instance"
    assert runtime_config.enable_checkpointing is False, \
        "Expected enable_checkpointing=False for --parallel (no --resume)"
    assert runtime_config.force_restart is False, \
        "Expected force_restart=False for --parallel (no --force-resume)"


def test_v1_fallback_routes_all_to_v1(tmp_fasta, tmp_path, cli_mocks):
    """Test that --v1-fallback routes all stages to v1.0 architecture."""
    if not RUNTIME_CONFIG_IMPORTED:
        pytest.skip(f"RuntimeConfig import failed: {RUNTIME_CONFIG_IMPORT_ERROR}")

    runner = CliRunner()
    result = runner.invoke(cli, [
        'predict',
        str(tmp_fasta),
        '--parallel',
        '--v1-fallback',
        '--output-dir', str(tmp_path / 'output')
    ])

    assert result.exit_code == 0, f"CLI failed: {result.output}"
    assert cli_mocks['run_prediction'].called

    call_kwargs = cli_mocks['run_prediction'].call_args.kwargs

    # Assert v1.0 routing despite --parallel
    assert call_kwargs['use_v2_architecture'] is False, \
        "Expected use_v2_architecture=False for --v1-fallback"

    # Assert no RuntimeConfig for v1.0
    assert call_kwargs['runtime_config'] is None, \
        "Expected runtime_config=None for v1.0 fallback"


def test_single_gpu_routes_to_v1(tmp_fasta, tmp_path, cli_mocks):
    """Test that single-GPU mode routes to v1.0 (no parallel)."""
    with patch.object(cli_mocks['detect'], 'return_value', [0]):
        runner = CliRunner()
        result = runner.invoke(cli, [
            'predict',
            str(tmp_fasta),
            '--output-dir', str(tmp_path / 'output')
        ])

        assert result.exit_code == 0, f"CLI failed: {result.output}"
        assert cli_mocks['run_prediction'].called

        call_kwargs = cli_mocks['run_prediction'].call_args.kwargs

        # Single GPU should not trigger v2.0 (parallel is False)
        assert call_kwargs['use_v2_architecture'] is False, \
            "Expected use_v2_architecture=False for single GPU"


def test_parallel_constructs_runtime_config_with_resume(tmp_fasta, tmp_path, cli_mocks):
    """Test that --resume constructs RuntimeConfig with checkpointing enabled."""
    if not RUNTIME_CONFIG_IMPORTED:
        pytest.skip(f"RuntimeConfig import failed: {RUNTIME_CONFIG_IMPORT_ERROR}")

    runner = CliRunner()
    result = runner.invoke(cli, [
        'predict',
        str(tmp_fasta),
        '--parallel',
        '--resume',
        '--output-dir', str(tmp_path / 'output')
    ])

    assert result.exit_code == 0, f"CLI failed: {result.output}"
    assert cli_mocks['run_prediction'].called

    call_kwargs = cli_mocks['run_prediction'].call_args.kwargs
    runtime_config = call_kwargs['runtime_config']

    assert runtime_config is not None
    assert runtime_config.enable_checkpointing is True, \
        "Expected enable_checkpointing=True for --resume"


def test_parallel_force_resume_sets_config(tmp_fasta, tmp_path, cli_mocks):
    """Test that --force-resume sets force_restart in RuntimeConfig."""
    if not RUNTIME_CONFIG_IMPORTED:
        pytest.skip(f"RuntimeConfig import failed: {RUNTIME_CONFIG_IMPORT_ERROR}")

    runner = CliRunner()
    result = runner.invoke(cli, [
        'predict',
        str(tmp_fasta),
        '--parallel',
        '--force-resume',
        '--output-dir', str(tmp_path / 'output')
    ])

    assert result.exit_code == 0, f"CLI failed: {result.output}"
    assert cli_mocks['run_prediction'].called

    call_kwargs = cli_mocks['run_prediction'].call_args.kwargs
    runtime_config = call_kwargs['runtime_config']

    assert runtime_config is not None
    assert runtime_config.force_restart is True, \
        "Expected force_restart=True for --force-resume"
    assert runtime_config.enable_checkpointing is True, \
        "Expected enable_checkpointing=True for --force-resume"


def test_hybrid_architecture_logged(tmp_fasta, tmp_path, cli_mocks):
    """Test that hybrid v2.0 architecture is logged correctly."""
    runner = CliRunner()
    result = runner.invoke(cli, [
        'predict',
        str(tmp_fasta),
        '--parallel',
        '--output-dir', str(tmp_path / 'output')
    ])

    assert result.exit_code == 0, f"CLI failed: {result.output}"

    # Check that output contains hybrid architecture log
    assert "v2.0 hybrid" in result.output, \
        "Expected 'v2.0 hybrid' in log output"

    # Verify DNABERT-S v1.0 is mentioned
    assert "DNABERT-S" in result.output, \
        "Expected 'DNABERT-S' in log output"
    assert "v1.0" in result.output, \
        "Expected 'v1.0' (for DNABERT-S) in log output"


def test_v1_fallback_architecture_logged(tmp_fasta, tmp_path, cli_mocks):
    """Test that v1.0 fallback architecture is logged correctly."""
    runner = CliRunner()
    result = runner.invoke(cli, [
        'predict',
        str(tmp_fasta),
        '--parallel',
        '--v1-fallback',
        '--output-dir', str(tmp_path / 'output')
    ])

    assert result.exit_code == 0, f"CLI failed: {result.output}"

    # Check for v1.0 fallback log
    assert "v1.0" in result.output, \
        "Expected 'v1.0' in log output for fallback"
    assert "--v1-fallback" in result.output, \
        "Expected '--v1-fallback' in log output"


def test_failure_exit_code_propagated(tmp_fasta, tmp_path, cli_mocks):
    """Test that failure exit codes from run_prediction are properly propagated."""
    cli_mocks['run_prediction'].return_value = 2  # Partial failure exit code

    runner = CliRunner()
    result = runner.invoke(cli, [
        'predict',
        str(tmp_fasta),
        '--parallel',
        '--output-dir', str(tmp_path / 'output')
    ])

    assert result.exit_code == 2, f"Expected exit code 2, got {result.exit_code}: {result.output}"
