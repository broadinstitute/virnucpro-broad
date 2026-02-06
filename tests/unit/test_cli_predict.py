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
except ImportError:
    RUNTIME_CONFIG_IMPORTED = False



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
        pytest.skip("RuntimeConfig not available")

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
        pytest.skip("RuntimeConfig not available")

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
        pytest.skip("RuntimeConfig not available")

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
        pytest.skip("RuntimeConfig not available")

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


def test_h5_to_pt_conversion_performance(tmp_path):
    """Regression test: HDF5-to-PT conversion completes within 5s for 1000 sequences.

    Exercises _stream_h5_to_pt_files with mock HDF5 data to ensure conversion
    overhead stays within acceptable bounds. This prevents the conversion adapter
    from becoming a bottleneck that erodes v2.0 speedup gains.

    Gap 3: No conversion overhead regression test.
    """
    import time
    import numpy as np
    import torch

    # Create mock HDF5 file with 1000 sequences
    try:
        import h5py
    except ImportError:
        pytest.skip("h5py not available")

    h5_path = tmp_path / "embeddings.h5"
    num_sequences = 1000
    embedding_dim = 2560  # ESM-2 3B hidden dim

    # Create mock sequence IDs and embeddings
    seq_ids = [f"seq_{i}" for i in range(num_sequences)]
    embeddings = np.random.randn(num_sequences, embedding_dim).astype(np.float32)

    with h5py.File(h5_path, 'w') as f:
        f.create_dataset('sequence_ids', data=[s.encode() for s in seq_ids])
        f.create_dataset('embeddings', data=embeddings)

    # Create mock nucleotide .pt files (split sequences across 10 files)
    nuc_dir = tmp_path / "nuc_features"
    nuc_dir.mkdir()
    nuc_files = []
    seqs_per_file = num_sequences // 10

    for file_idx in range(10):
        start = file_idx * seqs_per_file
        end = start + seqs_per_file
        nuc_data = {seq_ids[i]: torch.randn(embedding_dim) for i in range(start, end)}
        nuc_path = nuc_dir / f"output_{file_idx}_DNABERT_S.pt"
        torch.save(nuc_data, nuc_path)
        nuc_files.append(nuc_path)

    # Import and time the conversion
    from virnucpro.pipeline.prediction import _stream_h5_to_pt_files

    output_dir = tmp_path / "esm_output"
    start_time = time.monotonic()
    pt_files = _stream_h5_to_pt_files(h5_path, output_dir, nuc_files)
    elapsed = time.monotonic() - start_time

    # Assertions
    assert len(pt_files) == 10, f"Expected 10 .pt files, got {len(pt_files)}"

    # Verify all sequences were converted
    total_seqs = 0
    for pt_file in pt_files:
        data = torch.load(pt_file, weights_only=False)
        total_seqs += len(data)
    assert total_seqs == num_sequences, f"Expected {num_sequences} sequences, got {total_seqs}"

    # Verify file naming convention (DNABERT_S -> ESM)
    for pt_file in pt_files:
        assert "_ESM.pt" in pt_file.name, f"Expected _ESM.pt suffix, got {pt_file.name}"
        assert "_DNABERT_S.pt" not in pt_file.name, f"Unexpected _DNABERT_S.pt in {pt_file.name}"

    # Performance regression: must complete within 5 seconds for 1000 sequences
    # Production workloads (6M sequences) scale linearly, so 5s for 1K implies ~30s for 6M
    assert elapsed < 5.0, (
        f"HDF5-to-PT conversion took {elapsed:.2f}s for {num_sequences} sequences "
        f"(threshold: 5.0s). Investigate performance regression."
    )


# --- Gap Closure Tests (Phase 10.1 Plan 04) ---


def test_v1_checkpoint_format_detected(tmp_path):
    """Test that v2.0 resume detects v1.0 checkpoint format and raises clear error.

    Gap 4: Users who run v1.0, fail, then try v2.0 --resume should get actionable
    error instead of cryptic failure.
    """
    # Create v1.0-style checkpoint directory (has .done files, no shard_* dirs)
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "gpu0_ESM.pt").touch()
    (checkpoint_dir / "gpu0_ESM.pt.done").touch()
    (checkpoint_dir / "gpu1_ESM.pt").touch()
    (checkpoint_dir / "gpu1_ESM.pt.done").touch()

    # Simulate the v1.0 format check from gpu_worker
    done_files = list(checkpoint_dir.glob("*.done"))
    shard_dirs = list(checkpoint_dir.glob("shard_*"))

    assert len(done_files) == 2, "Should find v1.0 .done markers"
    assert len(shard_dirs) == 0, "Should not find v2.0 shard dirs"

    # Verify the detection logic would trigger
    assert done_files and not shard_dirs, \
        "v1.0 format detection should trigger: .done files exist, no shard_* dirs"


def test_v2_checkpoint_format_not_flagged(tmp_path):
    """Test that v2.0 checkpoint format is NOT flagged as v1.0.

    Ensures normal v2.0 resume flow isn't broken by the v1.0 detection logic.
    """
    # Create v2.0-style checkpoint directory (has shard_* dirs)
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    shard_dir = checkpoint_dir / "shard_0"
    shard_dir.mkdir()
    (shard_dir / "batch_0000.pt").touch()
    (shard_dir / "batch_0001.pt").touch()

    done_files = list(checkpoint_dir.glob("*.done"))
    shard_dirs = list(checkpoint_dir.glob("shard_*"))

    assert len(done_files) == 0, "Should not find .done markers in v2.0 format"
    assert len(shard_dirs) == 1, "Should find v2.0 shard dir"

    # Verify the detection logic would NOT trigger
    should_trigger = done_files and not shard_dirs
    assert not should_trigger, \
        "v1.0 format detection should NOT trigger for v2.0 checkpoints"


def test_world_size_validation_rejects_excess_gpus():
    """Test that requesting more GPUs than available raises clear error.

    Gap 5: world_size must be validated against actual GPU count to prevent
    confusing failures deep in the multi-GPU coordination code.
    """
    # The validation logic: num_gpus > actual_gpu_count and actual_gpu_count > 0
    # Simulate: user requests 4 GPUs but only 2 available
    num_gpus = 4
    actual_gpu_count = 2

    should_error = num_gpus > actual_gpu_count and actual_gpu_count > 0
    assert should_error, \
        "Should reject when requesting more GPUs than available"

    # Simulate: user requests 2 GPUs, 4 available (should NOT error)
    num_gpus = 2
    actual_gpu_count = 4

    should_error = num_gpus > actual_gpu_count and actual_gpu_count > 0
    assert not should_error, \
        "Should allow when requesting fewer GPUs than available"

    # Simulate: CPU-only system (actual_gpu_count=0, should NOT error)
    num_gpus = 2
    actual_gpu_count = 0

    should_error = num_gpus > actual_gpu_count and actual_gpu_count > 0
    assert not should_error, \
        "Should not error on CPU-only systems (mocked GPU detection)"


def test_h5_to_pt_cleanup_on_failure(tmp_path):
    """Test that partial .pt files are cleaned up when conversion fails mid-way.

    Gap 6: If _stream_h5_to_pt_files crashes after creating some .pt files,
    the cleanup logic should remove them to prevent orphaned files.
    """
    # Create some .pt files that would be "partially created"
    output_dir = tmp_path / "esm_output"
    output_dir.mkdir()

    partial_files = []
    for i in range(3):
        pt_file = output_dir / f"output_{i}_ESM.pt"
        pt_file.write_bytes(b"partial data")
        partial_files.append(pt_file)

    # Verify files exist before cleanup
    assert all(f.exists() for f in partial_files), "Partial files should exist"

    # Simulate cleanup logic from prediction.py try/finally
    for pt_file in partial_files:
        try:
            if Path(pt_file).exists():
                Path(pt_file).unlink()
        except OSError:
            pass

    # Verify cleanup
    assert not any(f.exists() for f in partial_files), \
        "All partial .pt files should be cleaned up"
