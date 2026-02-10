"""Unit tests for GPU worker function.

These tests mock CUDA and model loading to enable CPU-only testing.
The tests verify worker flow, error handling, and status reporting.
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from multiprocessing import Queue

import pytest
import torch
import h5py

from virnucpro.pipeline.gpu_worker import gpu_worker


# Test Fixtures

@pytest.fixture
def mock_model():
    """Mock model with forward_packed that returns fake embeddings."""
    model = Mock()
    model.eval = Mock()
    # Mock forward_packed for packed batches
    model.forward_packed = Mock(return_value={
        'representations': {
            36: torch.randn(100, 768)  # [total_tokens, hidden_dim]
        }
    })
    # Mock standard forward for unpacked batches
    model.forward = Mock(return_value={
        'representations': {
            36: torch.randn(1, 10, 768)  # [batch, seq_len, hidden_dim]
        }
    })
    return model


@pytest.fixture
def mock_batch_converter():
    """Mock batch converter."""
    converter = Mock()
    return converter


@pytest.fixture
def temp_index(tmp_path):
    """Create temporary index with test sequences."""
    index_path = tmp_path / "index.json"

    # Create fake FASTA file
    fasta_path = tmp_path / "sequences.fasta"
    fasta_path.write_text(
        ">seq1\nMKTAYIAKQRQISFVKSHFSRQLE\n"
        ">seq2\nALKVFGRCELAAMKRHGLDNYRGYSLGN\n"
        ">seq3\nVLSPADKTNVKAAWGKVGAHAGEY\n"
        ">seq4\nGSHMASLGSFPWQAKMVSHHNLTTGATLINEQWLLT\n"
    )

    # Create index
    index_data = {
        "version": "1.0",
        "created": "2026-02-04T00:00:00",
        "fasta_mtimes": {str(fasta_path): fasta_path.stat().st_mtime},
        "total_sequences": 4,
        "total_tokens": 100,
        "sequences": [
            {
                "sequence_id": "seq4",
                "length": 37,
                "file_path": str(fasta_path),
                "byte_offset": 76
            },
            {
                "sequence_id": "seq2",
                "length": 28,
                "file_path": str(fasta_path),
                "byte_offset": 31
            },
            {
                "sequence_id": "seq1",
                "length": 24,
                "file_path": str(fasta_path),
                "byte_offset": 0
            },
            {
                "sequence_id": "seq3",
                "length": 24,
                "file_path": str(fasta_path),
                "byte_offset": 125
            },
        ]
    }

    with open(index_path, 'w') as f:
        json.dump(index_data, f)

    return index_path


@pytest.fixture
def temp_output_dir(tmp_path):
    """Temp directory for worker outputs."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return output_dir


@pytest.fixture
def mock_results_queue():
    """Mock queue for testing (avoids multiprocessing complexity in unit tests)."""
    queue = Mock()
    queue.put = Mock()
    queue.get = Mock()
    queue.empty = Mock(return_value=False)
    return queue


# Test Classes

class TestGPUWorkerFlow:
    """Test GPU worker execution flow with extensive mocking."""

    @patch('virnucpro.pipeline.gpu_worker.torch.cuda')
    @patch('virnucpro.pipeline.gpu_worker.AsyncInferenceRunner')
    @patch('virnucpro.pipeline.gpu_worker.create_async_dataloader')
    @patch('virnucpro.models.esm2_flash.load_esm2_model')
    @patch('virnucpro.pipeline.gpu_worker.setup_worker_logging')
    def test_worker_sets_up_logging_first(
        self,
        mock_setup_logging,
        mock_load_model,
        mock_create_loader,
        mock_runner_class,
        mock_cuda,
        temp_index,
        temp_output_dir,
        mock_results_queue,
        mock_model,
        mock_batch_converter
    ):
        """Verify setup_worker_logging called first before any other operations."""
        # Configure mocks
        mock_setup_logging.return_value = temp_output_dir / "logs" / "worker_0.log"
        mock_load_model.return_value = (mock_model, mock_batch_converter)
        mock_cuda.get_device_name.return_value = "Mock GPU"

        # Mock dataloader to return empty iterator
        mock_loader = Mock()
        mock_loader.__iter__ = Mock(return_value=iter([]))
        mock_create_loader.return_value = mock_loader

        # Mock runner
        mock_runner = Mock()
        mock_runner.run = Mock(return_value=iter([]))
        mock_runner_class.return_value = mock_runner

        # Configure model_config
        model_config = {'model_type': 'esm2'}

        # Run worker
        gpu_worker(
            rank=0,
            world_size=4,
            results_queue=mock_results_queue,
            index_path=temp_index,
            output_dir=temp_output_dir,
            model_config=model_config
        )

        # Verify logging setup called first (before model load)
        assert mock_setup_logging.called
        mock_setup_logging.assert_called_once_with(0, temp_output_dir / "logs")

    @patch('virnucpro.pipeline.gpu_worker.torch.cuda')
    @patch('virnucpro.pipeline.gpu_worker.AsyncInferenceRunner')
    @patch('virnucpro.pipeline.gpu_worker.create_async_dataloader')
    @patch('virnucpro.models.esm2_flash.load_esm2_model')
    @patch('virnucpro.pipeline.gpu_worker.setup_worker_logging')
    @patch('virnucpro.pipeline.gpu_worker.get_worker_indices')
    def test_worker_gets_assigned_indices(
        self,
        mock_get_indices,
        mock_setup_logging,
        mock_load_model,
        mock_create_loader,
        mock_runner_class,
        mock_cuda,
        temp_index,
        temp_output_dir,
        mock_results_queue,
        mock_model,
        mock_batch_converter
    ):
        """Verify get_worker_indices called with correct rank."""
        # Configure mocks
        mock_setup_logging.return_value = temp_output_dir / "logs" / "worker_1.log"
        mock_get_indices.return_value = [1, 3]  # Worker 1 of 2
        mock_load_model.return_value = (mock_model, mock_batch_converter)
        mock_cuda.get_device_name.return_value = "Mock GPU"

        # Mock dataloader
        mock_loader = Mock()
        mock_loader.__iter__ = Mock(return_value=iter([]))
        mock_create_loader.return_value = mock_loader

        # Mock runner
        mock_runner = Mock()
        mock_runner.run = Mock(return_value=iter([]))
        mock_runner_class.return_value = mock_runner

        model_config = {'model_type': 'esm2'}

        # Run worker with rank=1, world_size=2
        gpu_worker(
            rank=1,
            world_size=2,
            results_queue=mock_results_queue,
            index_path=temp_index,
            output_dir=temp_output_dir,
            model_config=model_config
        )

        # Verify get_worker_indices called with correct parameters
        mock_get_indices.assert_called_once_with(temp_index, 1, 2)

    @patch('virnucpro.pipeline.gpu_worker.torch.cuda')
    @patch('virnucpro.pipeline.gpu_worker.AsyncInferenceRunner')
    @patch('virnucpro.pipeline.gpu_worker.create_async_dataloader')
    @patch('virnucpro.pipeline.gpu_worker.IndexBasedDataset')
    @patch('virnucpro.models.esm2_flash.load_esm2_model')
    @patch('virnucpro.pipeline.gpu_worker.setup_worker_logging')
    def test_worker_creates_dataset_with_indices(
        self,
        mock_setup_logging,
        mock_load_model,
        mock_dataset_class,
        mock_create_loader,
        mock_runner_class,
        mock_cuda,
        temp_index,
        temp_output_dir,
        mock_results_queue,
        mock_model,
        mock_batch_converter
    ):
        """Verify IndexBasedDataset created with correct indices."""
        # Configure mocks
        mock_setup_logging.return_value = temp_output_dir / "logs" / "worker_0.log"
        mock_load_model.return_value = (mock_model, mock_batch_converter)
        mock_cuda.get_device_name.return_value = "Mock GPU"

        # Mock dataset
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=2)
        mock_dataset_class.return_value = mock_dataset

        # Mock dataloader
        mock_loader = Mock()
        mock_loader.__iter__ = Mock(return_value=iter([]))
        mock_create_loader.return_value = mock_loader

        # Mock runner
        mock_runner = Mock()
        mock_runner.run = Mock(return_value=iter([]))
        mock_runner_class.return_value = mock_runner

        model_config = {'model_type': 'esm2'}

        # Run worker
        gpu_worker(
            rank=0,
            world_size=4,
            results_queue=mock_results_queue,
            index_path=temp_index,
            output_dir=temp_output_dir,
            model_config=model_config
        )

        # Verify IndexBasedDataset created with index_path and indices
        # (indices will be actual from get_worker_indices)
        assert mock_dataset_class.called
        call_args = mock_dataset_class.call_args
        assert call_args[0][0] == temp_index  # First arg is index_path

    @patch('virnucpro.pipeline.gpu_worker.torch.cuda')
    @patch('virnucpro.pipeline.gpu_worker.AsyncInferenceRunner')
    @patch('virnucpro.pipeline.gpu_worker.create_async_dataloader')
    @patch('virnucpro.models.esm2_flash.load_esm2_model')
    @patch('virnucpro.pipeline.gpu_worker.setup_worker_logging')
    def test_worker_saves_hdf5_shard(
        self,
        mock_setup_logging,
        mock_load_model,
        mock_create_loader,
        mock_runner_class,
        mock_cuda,
        temp_index,
        temp_output_dir,
        mock_results_queue,
        mock_model,
        mock_batch_converter
    ):
        """Verify shard HDF5 file created with correct name."""
        # Configure mocks
        mock_setup_logging.return_value = temp_output_dir / "logs" / "worker_2.log"
        mock_load_model.return_value = (mock_model, mock_batch_converter)
        mock_cuda.get_device_name.return_value = "Mock GPU"

        # Mock dataloader
        mock_loader = Mock()
        mock_loader.__iter__ = Mock(return_value=iter([]))
        mock_create_loader.return_value = mock_loader

        # Mock runner to return fake results
        from virnucpro.pipeline.async_inference import InferenceResult
        fake_result = InferenceResult(
            sequence_ids=['seq1', 'seq2'],
            embeddings=torch.randn(2, 768),
            batch_idx=0
        )
        mock_runner = Mock()
        mock_runner.run = Mock(return_value=iter([fake_result]))
        mock_runner_class.return_value = mock_runner

        model_config = {'model_type': 'esm2'}

        # Run worker with rank=2
        gpu_worker(
            rank=2,
            world_size=4,
            results_queue=mock_results_queue,
            index_path=temp_index,
            output_dir=temp_output_dir,
            model_config=model_config
        )

        # Verify shard file created
        shard_path = temp_output_dir / "shard_2.h5"
        assert shard_path.exists()

        # Verify HDF5 contents
        with h5py.File(shard_path, 'r') as f:
            assert 'embeddings' in f
            assert 'sequence_ids' in f
            assert f['embeddings'].shape == (2, 768)
            assert len(f['sequence_ids']) == 2

    @patch('virnucpro.pipeline.gpu_worker.torch.cuda')
    @patch('virnucpro.pipeline.gpu_worker.AsyncInferenceRunner')
    @patch('virnucpro.pipeline.gpu_worker.create_async_dataloader')
    @patch('virnucpro.models.esm2_flash.load_esm2_model')
    @patch('virnucpro.pipeline.gpu_worker.setup_worker_logging')
    def test_worker_reports_success(
        self,
        mock_setup_logging,
        mock_load_model,
        mock_create_loader,
        mock_runner_class,
        mock_cuda,
        temp_index,
        temp_output_dir,
        mock_results_queue,
        mock_model,
        mock_batch_converter
    ):
        """Verify results_queue receives complete status on success."""
        # Configure mocks
        mock_setup_logging.return_value = temp_output_dir / "logs" / "worker_0.log"
        mock_load_model.return_value = (mock_model, mock_batch_converter)
        mock_cuda.get_device_name.return_value = "Mock GPU"

        # Mock dataloader
        mock_loader = Mock()
        mock_loader.__iter__ = Mock(return_value=iter([]))
        mock_create_loader.return_value = mock_loader

        # Mock runner
        from virnucpro.pipeline.async_inference import InferenceResult
        fake_result = InferenceResult(
            sequence_ids=['seq1', 'seq2', 'seq3'],
            embeddings=torch.randn(3, 768),
            batch_idx=0
        )
        mock_runner = Mock()
        mock_runner.run = Mock(return_value=iter([fake_result]))
        mock_runner_class.return_value = mock_runner

        model_config = {'model_type': 'esm2'}

        # Run worker
        gpu_worker(
            rank=0,
            world_size=4,
            results_queue=mock_results_queue,
            index_path=temp_index,
            output_dir=temp_output_dir,
            model_config=model_config
        )

        # Verify status reported to queue
        mock_results_queue.put.assert_called_once()
        call_args = mock_results_queue.put.call_args[0][0]

        assert call_args['rank'] == 0
        assert call_args['status'] == 'complete'
        assert call_args['num_sequences'] == 3
        assert 'shard_0.h5' in call_args['shard_path']

    @patch('virnucpro.pipeline.gpu_worker.torch.cuda')
    @patch('virnucpro.pipeline.gpu_worker.AsyncInferenceRunner')
    @patch('virnucpro.pipeline.gpu_worker.create_async_dataloader')
    @patch('virnucpro.models.esm2_flash.load_esm2_model')
    @patch('virnucpro.pipeline.gpu_worker.setup_worker_logging')
    def test_worker_reports_failure(
        self,
        mock_setup_logging,
        mock_load_model,
        mock_create_loader,
        mock_runner_class,
        mock_cuda,
        temp_index,
        temp_output_dir,
        mock_results_queue,
        mock_model,
        mock_batch_converter
    ):
        """Verify exception leads to failed status in queue."""
        # Configure mocks
        mock_setup_logging.return_value = temp_output_dir / "logs" / "worker_0.log"
        mock_load_model.return_value = (mock_model, mock_batch_converter)
        mock_cuda.get_device_name.return_value = "Mock GPU"

        # Mock dataloader
        mock_loader = Mock()
        mock_loader.__iter__ = Mock(return_value=iter([]))
        mock_create_loader.return_value = mock_loader

        # Mock runner to raise exception
        mock_runner = Mock()
        mock_runner.run = Mock(side_effect=RuntimeError("Inference failed"))
        mock_runner_class.return_value = mock_runner

        model_config = {'model_type': 'esm2'}

        # Run worker (should catch exception and report failure)
        with pytest.raises(SystemExit):
            gpu_worker(
                rank=0,
                world_size=4,
                results_queue=mock_results_queue,
                index_path=temp_index,
                output_dir=temp_output_dir,
                model_config=model_config
            )

        # Verify failure reported to queue
        mock_results_queue.put.assert_called_once()
        call_args = mock_results_queue.put.call_args[0][0]

        assert call_args['rank'] == 0
        assert call_args['status'] == 'failed'
        assert 'Inference failed' in call_args['error_message']


class TestWorkerErrorHandling:
    """Test worker error handling for various failure modes."""

    @patch('virnucpro.pipeline.gpu_worker.torch.cuda')
    @patch('virnucpro.models.esm2_flash.load_esm2_model')
    @patch('virnucpro.pipeline.gpu_worker.setup_worker_logging')
    def test_model_load_failure(
        self,
        mock_setup_logging,
        mock_load_model,
        mock_cuda,
        temp_index,
        temp_output_dir,
        mock_results_queue
    ):
        """Verify model load error reported properly."""
        # Configure mocks
        mock_setup_logging.return_value = temp_output_dir / "logs" / "worker_0.log"
        mock_cuda.get_device_name.return_value = "Mock GPU"

        # Mock model load to fail
        mock_load_model.side_effect = RuntimeError("Model load failed")

        model_config = {'model_type': 'esm2'}

        # Run worker (should exit with error)
        with pytest.raises(SystemExit):
            gpu_worker(
                rank=0,
                world_size=4,
                results_queue=mock_results_queue,
                index_path=temp_index,
                output_dir=temp_output_dir,
                model_config=model_config
            )

        # Verify failure reported
        mock_results_queue.put.assert_called_once()
        call_args = mock_results_queue.put.call_args[0][0]
        assert call_args['status'] == 'failed'
        assert 'Model load failed' in call_args['error_message']

    @patch('virnucpro.pipeline.gpu_worker.torch.cuda')
    @patch('virnucpro.pipeline.gpu_worker.AsyncInferenceRunner')
    @patch('virnucpro.pipeline.gpu_worker.create_async_dataloader')
    @patch('virnucpro.models.esm2_flash.load_esm2_model')
    @patch('virnucpro.pipeline.gpu_worker.setup_worker_logging')
    def test_inference_failure(
        self,
        mock_setup_logging,
        mock_load_model,
        mock_create_loader,
        mock_runner_class,
        mock_cuda,
        temp_index,
        temp_output_dir,
        mock_results_queue,
        mock_model,
        mock_batch_converter
    ):
        """Verify inference error reported properly."""
        # Configure mocks
        mock_setup_logging.return_value = temp_output_dir / "logs" / "worker_0.log"
        mock_load_model.return_value = (mock_model, mock_batch_converter)
        mock_cuda.get_device_name.return_value = "Mock GPU"

        mock_loader = Mock()
        mock_loader.__iter__ = Mock(return_value=iter([]))
        mock_create_loader.return_value = mock_loader

        # Mock runner to fail during run
        mock_runner = Mock()
        mock_runner.run = Mock(side_effect=RuntimeError("CUDA out of memory"))
        mock_runner_class.return_value = mock_runner

        model_config = {'model_type': 'esm2'}

        # Run worker
        with pytest.raises(SystemExit):
            gpu_worker(
                rank=0,
                world_size=4,
                results_queue=mock_results_queue,
                index_path=temp_index,
                output_dir=temp_output_dir,
                model_config=model_config
            )

        # Verify error reported
        mock_results_queue.put.assert_called_once()
        call_args = mock_results_queue.put.call_args[0][0]
        assert call_args['status'] == 'failed'
        assert 'CUDA out of memory' in call_args['error_message']

    @patch('virnucpro.pipeline.gpu_worker.torch.cuda')
    @patch('virnucpro.pipeline.gpu_worker.AsyncInferenceRunner')
    @patch('virnucpro.pipeline.gpu_worker.create_async_dataloader')
    @patch('virnucpro.pipeline.gpu_worker.h5py.File')
    @patch('virnucpro.models.esm2_flash.load_esm2_model')
    @patch('virnucpro.pipeline.gpu_worker.setup_worker_logging')
    def test_shard_save_failure(
        self,
        mock_setup_logging,
        mock_load_model,
        mock_h5py,
        mock_create_loader,
        mock_runner_class,
        mock_cuda,
        temp_index,
        temp_output_dir,
        mock_results_queue,
        mock_model,
        mock_batch_converter
    ):
        """Verify disk write error reported properly."""
        # Configure mocks
        mock_setup_logging.return_value = temp_output_dir / "logs" / "worker_0.log"
        mock_load_model.return_value = (mock_model, mock_batch_converter)
        mock_cuda.get_device_name.return_value = "Mock GPU"

        mock_loader = Mock()
        mock_loader.__iter__ = Mock(return_value=iter([]))
        mock_create_loader.return_value = mock_loader

        # Mock runner
        from virnucpro.pipeline.async_inference import InferenceResult
        fake_result = InferenceResult(
            sequence_ids=['seq1'],
            embeddings=torch.randn(1, 768),
            batch_idx=0
        )
        mock_runner = Mock()
        mock_runner.run = Mock(return_value=iter([fake_result]))
        mock_runner_class.return_value = mock_runner

        # Mock h5py to fail
        mock_h5py.side_effect = OSError("Disk full")

        model_config = {'model_type': 'esm2'}

        # Run worker
        with pytest.raises(SystemExit):
            gpu_worker(
                rank=0,
                world_size=4,
                results_queue=mock_results_queue,
                index_path=temp_index,
                output_dir=temp_output_dir,
                model_config=model_config
            )

        # Verify error reported
        mock_results_queue.put.assert_called_once()
        call_args = mock_results_queue.put.call_args[0][0]
        assert call_args['status'] == 'failed'
        assert 'Disk full' in call_args['error_message']


class TestFP16Wiring:
    """Test FP16 precision wiring, env var override, and numerical instability handling in gpu_worker."""

    @patch.dict(os.environ, {'VIRNUCPRO_DISABLE_FP16': ''}, clear=False)
    @patch('virnucpro.pipeline.gpu_worker.torch.cuda')
    @patch('virnucpro.pipeline.gpu_worker.AsyncInferenceRunner')
    @patch('virnucpro.pipeline.gpu_worker.create_async_dataloader')
    @patch('virnucpro.models.esm2_flash.load_esm2_model')
    @patch('virnucpro.pipeline.gpu_worker.setup_worker_logging')
    def test_worker_fp16_default_enabled(
        self,
        mock_setup_logging,
        mock_load_model,
        mock_create_loader,
        mock_runner_class,
        mock_cuda,
        temp_index,
        temp_output_dir,
        mock_results_queue,
        mock_model,
        mock_batch_converter
    ):
        """Verify FP16 enabled by default when VIRNUCPRO_DISABLE_FP16 not set."""
        mock_setup_logging.return_value = temp_output_dir / "logs" / "worker_0.log"
        mock_load_model.return_value = (mock_model, mock_batch_converter)
        mock_cuda.get_device_name.return_value = "Mock GPU"

        mock_loader = Mock()
        mock_loader.__iter__ = Mock(return_value=iter([]))
        mock_create_loader.return_value = mock_loader

        mock_runner = Mock()
        mock_runner.run = Mock(return_value=iter([]))
        mock_runner_class.return_value = mock_runner

        model_config = {'model_type': 'esm2', 'enable_fp16': True}

        gpu_worker(
            rank=0,
            world_size=4,
            results_queue=mock_results_queue,
            index_path=temp_index,
            output_dir=temp_output_dir,
            model_config=model_config
        )

        mock_load_model.assert_called_once()
        call_kwargs = mock_load_model.call_args[1]
        assert call_kwargs['enable_fp16'] is True

    @patch.dict(os.environ, {'VIRNUCPRO_DISABLE_FP16': '1'}, clear=False)
    @patch('virnucpro.pipeline.gpu_worker.torch.cuda')
    @patch('virnucpro.pipeline.gpu_worker.AsyncInferenceRunner')
    @patch('virnucpro.pipeline.gpu_worker.create_async_dataloader')
    @patch('virnucpro.models.esm2_flash.load_esm2_model')
    @patch('virnucpro.pipeline.gpu_worker.setup_worker_logging')
    def test_worker_fp16_disabled_by_env_var(
        self,
        mock_setup_logging,
        mock_load_model,
        mock_create_loader,
        mock_runner_class,
        mock_cuda,
        temp_index,
        temp_output_dir,
        mock_results_queue,
        mock_model,
        mock_batch_converter
    ):
        """Verify VIRNUCPRO_DISABLE_FP16=1 overrides config and disables FP16."""
        mock_setup_logging.return_value = temp_output_dir / "logs" / "worker_0.log"
        mock_load_model.return_value = (mock_model, mock_batch_converter)
        mock_cuda.get_device_name.return_value = "Mock GPU"

        mock_loader = Mock()
        mock_loader.__iter__ = Mock(return_value=iter([]))
        mock_create_loader.return_value = mock_loader

        mock_runner = Mock()
        mock_runner.run = Mock(return_value=iter([]))
        mock_runner_class.return_value = mock_runner

        model_config = {'model_type': 'esm2', 'enable_fp16': True}

        gpu_worker(
            rank=0,
            world_size=4,
            results_queue=mock_results_queue,
            index_path=temp_index,
            output_dir=temp_output_dir,
            model_config=model_config
        )

        mock_load_model.assert_called_once()
        call_kwargs = mock_load_model.call_args[1]
        assert call_kwargs['enable_fp16'] is False

    @patch.dict(os.environ, {'VIRNUCPRO_DISABLE_FP16': 'true'}, clear=False)
    @patch('virnucpro.pipeline.gpu_worker.torch.cuda')
    @patch('virnucpro.pipeline.gpu_worker.AsyncInferenceRunner')
    @patch('virnucpro.pipeline.gpu_worker.create_async_dataloader')
    @patch('virnucpro.models.dnabert_flash.load_dnabert_model')
    @patch('virnucpro.pipeline.gpu_worker.setup_worker_logging')
    def test_worker_dnabert_fp16_respects_env_var(
        self,
        mock_setup_logging,
        mock_load_model,
        mock_create_loader,
        mock_runner_class,
        mock_cuda,
        temp_index,
        temp_output_dir,
        mock_results_queue,
        mock_model
    ):
        """Verify DNABERT model respects VIRNUCPRO_DISABLE_FP16 env var."""
        mock_setup_logging.return_value = temp_output_dir / "logs" / "worker_0.log"
        mock_load_model.return_value = (mock_model, Mock())
        mock_cuda.get_device_name.return_value = "Mock GPU"

        mock_loader = Mock()
        mock_loader.__iter__ = Mock(return_value=iter([]))
        mock_create_loader.return_value = mock_loader

        mock_runner = Mock()
        mock_runner.run = Mock(return_value=iter([]))
        mock_runner_class.return_value = mock_runner

        model_config = {'model_type': 'dnabert', 'enable_fp16': True}

        gpu_worker(
            rank=0,
            world_size=4,
            results_queue=mock_results_queue,
            index_path=temp_index,
            output_dir=temp_output_dir,
            model_config=model_config
        )

        mock_load_model.assert_called_once()
        call_kwargs = mock_load_model.call_args[1]
        assert call_kwargs['enable_fp16'] is False

    @patch('virnucpro.pipeline.gpu_worker.torch.cuda')
    @patch('virnucpro.pipeline.gpu_worker.AsyncInferenceRunner')
    @patch('virnucpro.pipeline.gpu_worker.create_async_dataloader')
    @patch('virnucpro.models.esm2_flash.load_esm2_model')
    @patch('virnucpro.pipeline.gpu_worker.setup_worker_logging')
    def test_worker_numerical_instability_error_handling(
        self,
        mock_setup_logging,
        mock_load_model,
        mock_create_loader,
        mock_runner_class,
        mock_cuda,
        temp_index,
        temp_output_dir,
        mock_results_queue,
        mock_model,
        mock_batch_converter
    ):
        """Verify numerical instability error reported with status='numerical_instability'."""
        mock_setup_logging.return_value = temp_output_dir / "logs" / "worker_0.log"
        mock_load_model.return_value = (mock_model, mock_batch_converter)
        mock_cuda.get_device_name.return_value = "Mock GPU"

        mock_loader = Mock()
        mock_loader.__iter__ = Mock(return_value=iter([]))
        mock_create_loader.return_value = mock_loader

        mock_runner = Mock()
        mock_runner.run = Mock(side_effect=RuntimeError("Numerical instability in batch_2_seqs_1: NaN=5, Inf=0"))
        mock_runner_class.return_value = mock_runner

        model_config = {'model_type': 'esm2'}

        with pytest.raises(SystemExit):
            gpu_worker(
                rank=0,
                world_size=4,
                results_queue=mock_results_queue,
                index_path=temp_index,
                output_dir=temp_output_dir,
                model_config=model_config
            )

        mock_results_queue.put.assert_called_once()
        call_args = mock_results_queue.put.call_args[0][0]
        assert call_args['status'] == 'numerical_instability'
        assert call_args['error'].startswith('Numerical instability')
        assert 'NaN' in call_args['error']

    @patch('virnucpro.pipeline.gpu_worker.torch.cuda')
    @patch('virnucpro.pipeline.gpu_worker.AsyncInferenceRunner')
    @patch('virnucpro.pipeline.gpu_worker.create_async_dataloader')
    @patch('virnucpro.models.esm2_flash.load_esm2_model')
    @patch('virnucpro.pipeline.gpu_worker.setup_worker_logging')
    def test_worker_batch_idx_correct_on_error(
        self,
        mock_setup_logging,
        mock_load_model,
        mock_create_loader,
        mock_runner_class,
        mock_cuda,
        temp_index,
        temp_output_dir,
        mock_results_queue,
        mock_model,
        mock_batch_converter
    ):
        """Verify failed_batch reports the correct batch index (increment before processing)."""
        mock_setup_logging.return_value = temp_output_dir / "logs" / "worker_0.log"
        mock_load_model.return_value = (mock_model, mock_batch_converter)
        mock_cuda.get_device_name.return_value = "Mock GPU"

        mock_loader = Mock()
        mock_loader.__iter__ = Mock(return_value=iter([]))
        mock_create_loader.return_value = mock_loader

        from virnucpro.pipeline.async_inference import InferenceResult
        fake_result1 = InferenceResult(
            sequence_ids=['seq1'],
            embeddings=torch.randn(1, 768),
            batch_idx=0
        )

        def results_iterator():
            yield fake_result1
            raise RuntimeError("Numerical instability in batch_1_seqs_1: NaN=1, Inf=0")

        mock_runner = Mock()
        mock_runner.run = Mock(return_value=results_iterator())
        mock_runner_class.return_value = mock_runner

        mock_loader = Mock()
        mock_loader.__iter__ = Mock(return_value=iter([Mock()]))
        mock_create_loader.return_value = mock_loader

        model_config = {'model_type': 'esm2'}

        with pytest.raises(SystemExit):
            gpu_worker(
                rank=0,
                world_size=4,
                results_queue=mock_results_queue,
                index_path=temp_index,
                output_dir=temp_output_dir,
                model_config=model_config
            )

        mock_results_queue.put.assert_called_once()
        call_args = mock_results_queue.put.call_args[0][0]
        assert call_args['status'] == 'numerical_instability'
        assert call_args['error'].startswith('Numerical instability')
        assert call_args['failed_batch'] == 1

    @patch.dict(os.environ, {'VIRNUCPRO_DISABLE_FP16': ' 1 '}, clear=False)
    @patch('virnucpro.pipeline.gpu_worker.torch.cuda')
    @patch('virnucpro.pipeline.gpu_worker.AsyncInferenceRunner')
    @patch('virnucpro.pipeline.gpu_worker.create_async_dataloader')
    @patch('virnucpro.models.esm2_flash.load_esm2_model')
    @patch('virnucpro.pipeline.gpu_worker.setup_worker_logging')
    def test_worker_fp16_disabled_by_env_var_with_whitespace(
        self,
        mock_setup_logging,
        mock_load_model,
        mock_create_loader,
        mock_runner_class,
        mock_cuda,
        temp_index,
        temp_output_dir,
        mock_results_queue,
        mock_model,
        mock_batch_converter
    ):
        """Verify VIRNUCPRO_DISABLE_FP16=' 1 ' with whitespace is correctly parsed."""
        mock_setup_logging.return_value = temp_output_dir / "logs" / "worker_0.log"
        mock_load_model.return_value = (mock_model, mock_batch_converter)
        mock_cuda.get_device_name.return_value = "Mock GPU"

        mock_loader = Mock()
        mock_loader.__iter__ = Mock(return_value=iter([]))
        mock_create_loader.return_value = mock_loader

        mock_runner = Mock()
        mock_runner.run = Mock(return_value=iter([]))
        mock_runner_class.return_value = mock_runner

        model_config = {'model_type': 'esm2', 'enable_fp16': True}

        gpu_worker(
            rank=0,
            world_size=4,
            results_queue=mock_results_queue,
            index_path=temp_index,
            output_dir=temp_output_dir,
            model_config=model_config
        )

        mock_load_model.assert_called_once()
        call_kwargs = mock_load_model.call_args[1]
        assert call_kwargs['enable_fp16'] is False

    @patch.dict(os.environ, {'VIRNUCPRO_DISABLE_FP16': ' true '}, clear=False)
    @patch('virnucpro.pipeline.gpu_worker.torch.cuda')
    @patch('virnucpro.pipeline.gpu_worker.AsyncInferenceRunner')
    @patch('virnucpro.pipeline.gpu_worker.create_async_dataloader')
    @patch('virnucpro.models.esm2_flash.load_esm2_model')
    @patch('virnucpro.pipeline.gpu_worker.setup_worker_logging')
    def test_worker_fp16_disabled_by_env_var_true_with_whitespace(
        self,
        mock_setup_logging,
        mock_load_model,
        mock_create_loader,
        mock_runner_class,
        mock_cuda,
        temp_index,
        temp_output_dir,
        mock_results_queue,
        mock_model,
        mock_batch_converter
    ):
        """Verify VIRNUCPRO_DISABLE_FP16=' true ' with whitespace is correctly parsed."""
        mock_setup_logging.return_value = temp_output_dir / "logs" / "worker_0.log"
        mock_load_model.return_value = (mock_model, mock_batch_converter)
        mock_cuda.get_device_name.return_value = "Mock GPU"

        mock_loader = Mock()
        mock_loader.__iter__ = Mock(return_value=iter([]))
        mock_create_loader.return_value = mock_loader

        mock_runner = Mock()
        mock_runner.run = Mock(return_value=iter([]))
        mock_runner_class.return_value = mock_runner

        model_config = {'model_type': 'esm2', 'enable_fp16': True}

        gpu_worker(
            rank=0,
            world_size=4,
            results_queue=mock_results_queue,
            index_path=temp_index,
            output_dir=temp_output_dir,
            model_config=model_config
        )

        mock_load_model.assert_called_once()
        call_kwargs = mock_load_model.call_args[1]
        assert call_kwargs['enable_fp16'] is False

    @patch('virnucpro.pipeline.gpu_worker.torch.cuda')
    @patch('virnucpro.pipeline.gpu_worker.AsyncInferenceRunner')
    @patch('virnucpro.pipeline.gpu_worker.create_async_dataloader')
    @patch('virnucpro.models.esm2_flash.load_esm2_model')
    @patch('virnucpro.pipeline.gpu_worker.setup_worker_logging')
    def test_worker_numerical_instability_packed_path(
        self,
        mock_setup_logging,
        mock_load_model,
        mock_create_loader,
        mock_runner_class,
        mock_cuda,
        temp_index,
        temp_output_dir,
        mock_results_queue,
        mock_model,
        mock_batch_converter
    ):
        """Verify numerical instability error in packed inference path (use_packed=True)."""
        mock_setup_logging.return_value = temp_output_dir / "logs" / "worker_0.log"
        mock_load_model.return_value = (mock_model, mock_batch_converter)
        mock_cuda.get_device_name.return_value = "Mock GPU"

        mock_loader = Mock()
        mock_loader.__iter__ = Mock(return_value=iter([]))
        mock_create_loader.return_value = mock_loader

        mock_runner = Mock()
        mock_runner.run = Mock(side_effect=RuntimeError("Numerical instability in batch_0_seqs_3: NaN=2, Inf=0"))
        mock_runner_class.return_value = mock_runner

        model_config = {'model_type': 'esm2', 'use_packed': True}

        with pytest.raises(SystemExit):
            gpu_worker(
                rank=0,
                world_size=4,
                results_queue=mock_results_queue,
                index_path=temp_index,
                output_dir=temp_output_dir,
                model_config=model_config
            )

        mock_results_queue.put.assert_called_once()
        call_args = mock_results_queue.put.call_args[0][0]
        assert call_args['status'] == 'numerical_instability'
        assert call_args['error'].startswith('Numerical instability')
        assert 'NaN' in call_args['error']


class TestErrorFormatConsistency:
    """Test error reporting format consistency: error field as category code, error_message for diagnostics."""

    @patch('virnucpro.pipeline.gpu_worker.torch.cuda')
    @patch('virnucpro.pipeline.gpu_worker.AsyncInferenceRunner')
    @patch('virnucpro.pipeline.gpu_worker.create_async_dataloader')
    @patch('virnucpro.models.esm2_flash.load_esm2_model')
    @patch('virnucpro.pipeline.gpu_worker.setup_worker_logging')
    def test_oom_error_format_category_code(
        self,
        mock_setup_logging,
        mock_load_model,
        mock_create_loader,
        mock_runner_class,
        mock_cuda,
        temp_index,
        temp_output_dir,
        mock_results_queue,
        mock_model,
        mock_batch_converter
    ):
        """Verify OOM error uses 'cuda_oom' category code in 'error' field and full message in 'error_message'."""
        mock_setup_logging.return_value = temp_output_dir / "logs" / "worker_0.log"
        mock_load_model.return_value = (mock_model, mock_batch_converter)
        mock_cuda.get_device_name.return_value = "Mock GPU"
        mock_cuda.memory_allocated.return_value = 8e9
        mock_cuda.max_memory_allocated.return_value = 10e9

        mock_loader = Mock()
        mock_loader.__iter__ = Mock(return_value=iter([]))
        mock_create_loader.return_value = mock_loader

        mock_runner = Mock()
        mock_runner.run = Mock(side_effect=RuntimeError("CUDA out of memory: out of memory"))
        mock_runner_class.return_value = mock_runner

        model_config = {'model_type': 'esm2'}

        with pytest.raises(SystemExit):
            gpu_worker(
                rank=0,
                world_size=4,
                results_queue=mock_results_queue,
                index_path=temp_index,
                output_dir=temp_output_dir,
                model_config=model_config
            )

        mock_results_queue.put.assert_called_once()
        call_args = mock_results_queue.put.call_args[0][0]

        assert call_args['status'] == 'failed'
        assert call_args['error'] == 'cuda_oom'
        assert 'out of memory' in call_args['error_message'].lower()
        assert 'error_type' not in call_args
        assert call_args['reduce_batch_size'] is True
        assert call_args['retry_recommended'] is True

    @patch('virnucpro.pipeline.gpu_worker.torch.cuda')
    @patch('virnucpro.pipeline.gpu_worker.AsyncInferenceRunner')
    @patch('virnucpro.pipeline.gpu_worker.create_async_dataloader')
    @patch('virnucpro.models.esm2_flash.load_esm2_model')
    @patch('virnucpro.pipeline.gpu_worker.setup_worker_logging')
    def test_cuda_runtime_error_format_category_code(
        self,
        mock_setup_logging,
        mock_load_model,
        mock_create_loader,
        mock_runner_class,
        mock_cuda,
        temp_index,
        temp_output_dir,
        mock_results_queue,
        mock_model,
        mock_batch_converter
    ):
        """Verify CUDA runtime error uses 'cuda_runtime' category code in 'error' field."""
        mock_setup_logging.return_value = temp_output_dir / "logs" / "worker_0.log"
        mock_load_model.return_value = (mock_model, mock_batch_converter)
        mock_cuda.get_device_name.return_value = "Mock GPU"

        mock_loader = Mock()
        mock_loader.__iter__ = Mock(return_value=iter([]))
        mock_create_loader.return_value = mock_loader

        mock_runner = Mock()
        mock_runner.run = Mock(side_effect=RuntimeError("CUDA error: an illegal memory access was encountered"))
        mock_runner_class.return_value = mock_runner

        model_config = {'model_type': 'esm2'}

        with pytest.raises(SystemExit):
            gpu_worker(
                rank=0,
                world_size=4,
                results_queue=mock_results_queue,
                index_path=temp_index,
                output_dir=temp_output_dir,
                model_config=model_config
            )

        mock_results_queue.put.assert_called_once()
        call_args = mock_results_queue.put.call_args[0][0]

        assert call_args['status'] == 'failed'
        assert call_args['error'] == 'cuda_runtime'
        assert 'illegal memory access' in call_args['error_message']
        assert 'error_type' not in call_args
        assert call_args['circuit_breaker'] is True
        assert call_args['retry_recommended'] is True

    @patch('virnucpro.pipeline.gpu_worker.torch.cuda')
    @patch('virnucpro.pipeline.gpu_worker.AsyncInferenceRunner')
    @patch('virnucpro.pipeline.gpu_worker.create_async_dataloader')
    @patch('virnucpro.models.esm2_flash.load_esm2_model')
    @patch('virnucpro.pipeline.gpu_worker.setup_worker_logging')
    def test_generic_error_format_category_code(
        self,
        mock_setup_logging,
        mock_load_model,
        mock_create_loader,
        mock_runner_class,
        mock_cuda,
        temp_index,
        temp_output_dir,
        mock_results_queue,
        mock_model,
        mock_batch_converter
    ):
        """Verify generic error uses 'generic_error' category code in 'error' field."""
        mock_setup_logging.return_value = temp_output_dir / "logs" / "worker_0.log"
        mock_load_model.return_value = (mock_model, mock_batch_converter)
        mock_cuda.get_device_name.return_value = "Mock GPU"

        mock_loader = Mock()
        mock_loader.__iter__ = Mock(return_value=iter([]))
        mock_create_loader.return_value = mock_loader

        mock_runner = Mock()
        mock_runner.run = Mock(side_effect=RuntimeError("Unexpected tensor dimension mismatch"))
        mock_runner_class.return_value = mock_runner

        model_config = {'model_type': 'esm2'}

        with pytest.raises(SystemExit):
            gpu_worker(
                rank=0,
                world_size=4,
                results_queue=mock_results_queue,
                index_path=temp_index,
                output_dir=temp_output_dir,
                model_config=model_config
            )

        mock_results_queue.put.assert_called_once()
        call_args = mock_results_queue.put.call_args[0][0]

        assert call_args['status'] == 'failed'
        assert call_args['error'] == 'generic_error'
        assert 'tensor dimension mismatch' in call_args['error_message']
        assert 'error_type' not in call_args

    @patch('virnucpro.pipeline.gpu_worker.torch.cuda')
    @patch('virnucpro.models.esm2_flash.load_esm2_model')
    @patch('virnucpro.pipeline.gpu_worker.setup_worker_logging')
    def test_generic_exception_error_format(
        self,
        mock_setup_logging,
        mock_load_model,
        mock_cuda,
        temp_index,
        temp_output_dir,
        mock_results_queue
    ):
        """Verify generic Exception (not RuntimeError) uses 'generic_error' category code."""
        mock_setup_logging.return_value = temp_output_dir / "logs" / "worker_0.log"
        mock_cuda.get_device_name.return_value = "Mock GPU"

        mock_load_model.side_effect = ValueError("Invalid model configuration: missing required parameter")

        model_config = {'model_type': 'esm2'}

        with pytest.raises(SystemExit):
            gpu_worker(
                rank=0,
                world_size=4,
                results_queue=mock_results_queue,
                index_path=temp_index,
                output_dir=temp_output_dir,
                model_config=model_config
            )

        mock_results_queue.put.assert_called_once()
        call_args = mock_results_queue.put.call_args[0][0]

        assert call_args['status'] == 'failed'
        assert call_args['error'] == 'generic_error'
        assert 'Invalid model configuration' in call_args['error_message']
        assert 'error_type' not in call_args
