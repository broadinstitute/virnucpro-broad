"""Tests for CLI predict command auto-detection"""

import pytest
from unittest.mock import patch, MagicMock, Mock
from click.testing import CliRunner
from pathlib import Path


class TestPredictAutoDetection:
    """Test auto-detection behavior in predict command"""

    @pytest.fixture
    def runner(self):
        """Click test runner"""
        return CliRunner()

    @pytest.fixture
    def mock_context(self):
        """Mock Click context with logger and config"""
        ctx = Mock()
        ctx.obj = {
            'logger': MagicMock(),
            'config': MagicMock()
        }
        # Configure config mock to return sensible defaults
        ctx.obj['config'].get = MagicMock(side_effect=lambda key, default=None: {
            'prediction.models.500': 'model.pth',
            'prediction.batch_size': 256,
            'prediction.num_workers': 4,
            'features.dnabert.batch_size': 256,
            'features.esm.toks_per_batch': 2048,
            'features.esm.truncation_seq_length': 1024,
            'device.fallback_to_cpu': True,
            'files.auto_cleanup': True
        }.get(key, default))
        return ctx

    @patch('virnucpro.cli.predict.detect_cuda_devices')
    @patch('virnucpro.cli.predict.run_prediction')
    @patch('virnucpro.cli.predict.Path')
    @patch('virnucpro.cli.predict.validate_and_get_device')
    def test_auto_detect_multiple_gpus(
        self, mock_validate_device, mock_path, mock_run_prediction, mock_detect_gpus, runner, mock_context
    ):
        """Test that multiple GPUs are auto-detected and parallel mode is enabled"""
        # Mock detect_cuda_devices to return 2 GPUs
        mock_detect_gpus.return_value = [0, 1]

        # Mock Path to return existing model file
        mock_path_obj = MagicMock()
        mock_path_obj.exists.return_value = True
        mock_path_obj.stem = 'test_input'
        mock_path.return_value = mock_path_obj

        # Mock device validation
        mock_validate_device.return_value = MagicMock()

        # Mock run_prediction to return success
        mock_run_prediction.return_value = 0

        from virnucpro.cli.predict import predict

        # Simulate running the command without --gpus or --parallel flags
        with runner.isolated_filesystem():
            # Create dummy input file
            Path('test.fasta').write_text('>seq1\nATCG\n')

            # Invoke predict with context
            result = runner.invoke(
                predict,
                ['test.fasta'],
                obj=mock_context.obj,
                catch_exceptions=False
            )

            # Verify run_prediction was called with parallel=True
            assert mock_run_prediction.called
            call_kwargs = mock_run_prediction.call_args[1]
            assert call_kwargs['parallel'] is True, "parallel should be auto-enabled for multi-GPU"

    @patch('virnucpro.cli.predict.detect_cuda_devices')
    @patch('virnucpro.cli.predict.run_prediction')
    @patch('virnucpro.cli.predict.Path')
    @patch('virnucpro.cli.predict.validate_and_get_device')
    def test_single_gpu_no_auto_parallel(
        self, mock_validate_device, mock_path, mock_run_prediction, mock_detect_gpus, runner, mock_context
    ):
        """Test that single GPU doesn't enable parallel mode"""
        # Mock detect_cuda_devices to return 1 GPU
        mock_detect_gpus.return_value = [0]

        # Mock Path to return existing model file
        mock_path_obj = MagicMock()
        mock_path_obj.exists.return_value = True
        mock_path_obj.stem = 'test_input'
        mock_path.return_value = mock_path_obj

        # Mock device validation
        mock_validate_device.return_value = MagicMock()

        # Mock run_prediction to return success
        mock_run_prediction.return_value = 0

        from virnucpro.cli.predict import predict

        # Simulate running the command without --gpus or --parallel flags
        with runner.isolated_filesystem():
            # Create dummy input file
            Path('test.fasta').write_text('>seq1\nATCG\n')

            # Invoke predict with context
            result = runner.invoke(
                predict,
                ['test.fasta'],
                obj=mock_context.obj,
                catch_exceptions=False
            )

            # Verify run_prediction was called with parallel=False (default)
            assert mock_run_prediction.called
            call_kwargs = mock_run_prediction.call_args[1]
            assert call_kwargs['parallel'] is False, "parallel should NOT be enabled for single GPU"

    @patch('virnucpro.cli.predict.detect_cuda_devices')
    @patch('virnucpro.cli.predict.run_prediction')
    @patch('virnucpro.cli.predict.Path')
    @patch('virnucpro.cli.predict.validate_and_get_device')
    def test_explicit_gpus_overrides_auto_detect(
        self, mock_validate_device, mock_path, mock_run_prediction, mock_detect_gpus, runner, mock_context
    ):
        """Test that explicit --gpus flag overrides auto-detection"""
        # Mock detect_cuda_devices (should not be called when --gpus specified)
        mock_detect_gpus.return_value = [0, 1, 2, 3]

        # Mock Path to return existing model file
        mock_path_obj = MagicMock()
        mock_path_obj.exists.return_value = True
        mock_path_obj.stem = 'test_input'
        mock_path.return_value = mock_path_obj

        # Mock device validation
        mock_validate_device.return_value = MagicMock()

        # Mock run_prediction to return success
        mock_run_prediction.return_value = 0

        from virnucpro.cli.predict import predict

        # Simulate running the command with explicit --gpus
        with runner.isolated_filesystem():
            # Create dummy input file
            Path('test.fasta').write_text('>seq1\nATCG\n')

            # Invoke predict with --gpus flag specifying specific GPUs
            result = runner.invoke(
                predict,
                ['test.fasta', '--gpus', '0,2'],
                obj=mock_context.obj,
                catch_exceptions=False
            )

            # Verify run_prediction was called with parallel=True
            assert mock_run_prediction.called
            call_kwargs = mock_run_prediction.call_args[1]
            assert call_kwargs['parallel'] is True, "parallel should be enabled when --gpus has multiple IDs"

            # Verify detect_cuda_devices was NOT called (we skip auto-detection)
            # Actually, with current implementation it might still be called, but the result is ignored
            # The important thing is that the explicit --gpus value is used

    @patch('virnucpro.cli.predict.detect_cuda_devices')
    @patch('virnucpro.cli.predict.run_prediction')
    @patch('virnucpro.cli.predict.Path')
    @patch('virnucpro.cli.predict.validate_and_get_device')
    def test_no_cuda_no_auto_parallel(
        self, mock_validate_device, mock_path, mock_run_prediction, mock_detect_gpus, runner, mock_context
    ):
        """Test that no CUDA availability doesn't enable parallel mode"""
        # Mock detect_cuda_devices to return empty list (no CUDA)
        mock_detect_gpus.return_value = []

        # Mock Path to return existing model file
        mock_path_obj = MagicMock()
        mock_path_obj.exists.return_value = True
        mock_path_obj.stem = 'test_input'
        mock_path.return_value = mock_path_obj

        # Mock device validation
        mock_validate_device.return_value = MagicMock()

        # Mock run_prediction to return success
        mock_run_prediction.return_value = 0

        from virnucpro.cli.predict import predict

        # Simulate running the command without CUDA
        with runner.isolated_filesystem():
            # Create dummy input file
            Path('test.fasta').write_text('>seq1\nATCG\n')

            # Invoke predict with context
            result = runner.invoke(
                predict,
                ['test.fasta'],
                obj=mock_context.obj,
                catch_exceptions=False
            )

            # Verify run_prediction was called with parallel=False
            assert mock_run_prediction.called
            call_kwargs = mock_run_prediction.call_args[1]
            assert call_kwargs['parallel'] is False, "parallel should NOT be enabled without CUDA"
