"""Unit tests for multi-GPU inference orchestration.

Tests the run_multi_gpu_inference entry point with heavy mocking to test
orchestration flow without actual GPU workers.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, call
import tempfile
import json

from virnucpro.pipeline.multi_gpu_inference import (
    run_multi_gpu_inference,
    run_esm2_multi_gpu
)


@pytest.fixture
def mock_fasta_files(tmp_path):
    """Create temporary FASTA files."""
    fasta1 = tmp_path / "sequences1.fasta"
    fasta1.write_text(">seq1\nACGT\n>seq2\nTGCA\n")

    fasta2 = tmp_path / "sequences2.fasta"
    fasta2.write_text(">seq3\nGGCC\n>seq4\nCCGG\n")

    return [fasta1, fasta2]


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create temporary output directory."""
    return tmp_path / "output"


@pytest.fixture
def mock_index_data():
    """Mock sequence index data."""
    return {
        'version': '1.0',
        'created': '2026-02-04T12:00:00',
        'fasta_mtimes': {},
        'total_sequences': 4,
        'total_tokens': 16,
        'sequences': [
            {'sequence_id': 'seq1', 'length': 4, 'file_path': 'sequences1.fasta', 'byte_offset': 0},
            {'sequence_id': 'seq2', 'length': 4, 'file_path': 'sequences1.fasta', 'byte_offset': 10},
            {'sequence_id': 'seq3', 'length': 4, 'file_path': 'sequences2.fasta', 'byte_offset': 0},
            {'sequence_id': 'seq4', 'length': 4, 'file_path': 'sequences2.fasta', 'byte_offset': 10},
        ]
    }


class TestOrchestrationFlow:
    """Test correct orchestration sequence."""

    @patch('virnucpro.pipeline.multi_gpu_inference.aggregate_shards')
    @patch('virnucpro.pipeline.multi_gpu_inference.GPUProcessCoordinator')
    @patch('virnucpro.pipeline.multi_gpu_inference.load_sequence_index')
    @patch('virnucpro.pipeline.multi_gpu_inference.create_sequence_index')
    @patch('virnucpro.pipeline.multi_gpu_inference.torch.cuda.device_count')
    def test_creates_index_first(
        self,
        mock_device_count,
        mock_create_index,
        mock_load_index,
        mock_coordinator_cls,
        mock_aggregate,
        mock_fasta_files,
        temp_output_dir,
        mock_index_data
    ):
        """Verify create_sequence_index called before spawning workers."""
        mock_device_count.return_value = 2
        mock_load_index.return_value = mock_index_data

        # Mock coordinator behavior
        mock_coordinator = MagicMock()
        mock_coordinator.wait_for_completion.return_value = {0: True, 1: True}
        mock_coordinator_cls.return_value = mock_coordinator

        # Create dummy shard files
        temp_output_dir.mkdir(parents=True, exist_ok=True)
        (temp_output_dir / "shard_0.h5").touch()
        (temp_output_dir / "shard_1.h5").touch()

        # Mock aggregate_shards
        output_path = temp_output_dir / "embeddings.h5"
        mock_aggregate.return_value = output_path

        model_config = {'model_type': 'esm2'}

        result_path, failed = run_multi_gpu_inference(
            mock_fasta_files,
            temp_output_dir,
            model_config
        )

        # Verify index created before coordinator instantiated
        assert mock_create_index.called
        assert mock_coordinator_cls.called
        # Both should have been called exactly once
        assert mock_create_index.call_count == 1
        assert mock_coordinator_cls.call_count == 1

    @patch('virnucpro.pipeline.multi_gpu_inference.aggregate_shards')
    @patch('virnucpro.pipeline.multi_gpu_inference.get_worker_indices')
    @patch('virnucpro.pipeline.multi_gpu_inference.GPUProcessCoordinator')
    @patch('virnucpro.pipeline.multi_gpu_inference.load_sequence_index')
    @patch('virnucpro.pipeline.multi_gpu_inference.create_sequence_index')
    @patch('virnucpro.pipeline.multi_gpu_inference.torch.cuda.device_count')
    @patch('virnucpro.pipeline.multi_gpu_inference.gpu_worker')
    def test_spawns_correct_workers(
        self,
        mock_worker_fn,
        mock_device_count,
        mock_create_index,
        mock_load_index,
        mock_coordinator_cls,
        mock_get_worker_indices,
        mock_aggregate,
        mock_fasta_files,
        temp_output_dir,
        mock_index_data
    ):
        """Verify spawn_workers called with correct arguments."""
        mock_device_count.return_value = 2
        mock_load_index.return_value = mock_index_data
        mock_get_worker_indices.return_value = [0, 2]  # Dummy indices

        # Mock coordinator behavior
        mock_coordinator = MagicMock()
        mock_coordinator.wait_for_completion.return_value = {0: True, 1: True}
        mock_coordinator_cls.return_value = mock_coordinator

        # Create dummy shard files
        temp_output_dir.mkdir(parents=True, exist_ok=True)
        (temp_output_dir / "shard_0.h5").touch()
        (temp_output_dir / "shard_1.h5").touch()

        # Mock aggregate_shards
        output_path = temp_output_dir / "embeddings.h5"
        mock_aggregate.return_value = output_path

        model_config = {'model_type': 'esm2'}

        result_path, failed = run_multi_gpu_inference(
            mock_fasta_files,
            temp_output_dir,
            model_config
        )

        # Verify spawn_workers called with gpu_worker function
        mock_coordinator.spawn_workers.assert_called_once()
        args = mock_coordinator.spawn_workers.call_args[0]
        # First arg should be a callable (the mock replaces the actual function)
        assert callable(args[0])
        # Check worker args: (index_path, output_dir, model_config)
        worker_args = args[1]
        assert worker_args[0] == temp_output_dir / "sequence_index.json"
        assert worker_args[1] == temp_output_dir
        assert worker_args[2] == model_config

    @patch('virnucpro.pipeline.multi_gpu_inference.aggregate_shards')
    @patch('virnucpro.pipeline.multi_gpu_inference.GPUProcessCoordinator')
    @patch('virnucpro.pipeline.multi_gpu_inference.load_sequence_index')
    @patch('virnucpro.pipeline.multi_gpu_inference.create_sequence_index')
    @patch('virnucpro.pipeline.multi_gpu_inference.torch.cuda.device_count')
    def test_waits_for_completion(
        self,
        mock_device_count,
        mock_create_index,
        mock_load_index,
        mock_coordinator_cls,
        mock_aggregate,
        mock_fasta_files,
        temp_output_dir,
        mock_index_data
    ):
        """Verify wait_for_completion called."""
        mock_device_count.return_value = 2
        mock_load_index.return_value = mock_index_data

        # Mock coordinator behavior
        mock_coordinator = MagicMock()
        mock_coordinator.wait_for_completion.return_value = {0: True, 1: True}
        mock_coordinator_cls.return_value = mock_coordinator

        # Create dummy shard files
        temp_output_dir.mkdir(parents=True, exist_ok=True)
        (temp_output_dir / "shard_0.h5").touch()
        (temp_output_dir / "shard_1.h5").touch()

        # Mock aggregate_shards
        output_path = temp_output_dir / "embeddings.h5"
        mock_aggregate.return_value = output_path

        model_config = {'model_type': 'esm2'}
        timeout = 600.0

        result_path, failed = run_multi_gpu_inference(
            mock_fasta_files,
            temp_output_dir,
            model_config,
            timeout=timeout
        )

        # Verify wait_for_completion called with timeout
        mock_coordinator.wait_for_completion.assert_called_once_with(timeout=timeout)

    @patch('virnucpro.pipeline.multi_gpu_inference.aggregate_shards')
    @patch('virnucpro.pipeline.multi_gpu_inference.get_worker_indices')
    @patch('virnucpro.pipeline.multi_gpu_inference.GPUProcessCoordinator')
    @patch('virnucpro.pipeline.multi_gpu_inference.load_sequence_index')
    @patch('virnucpro.pipeline.multi_gpu_inference.create_sequence_index')
    @patch('virnucpro.pipeline.multi_gpu_inference.torch.cuda.device_count')
    def test_aggregates_successful_shards(
        self,
        mock_device_count,
        mock_create_index,
        mock_load_index,
        mock_coordinator_cls,
        mock_get_worker_indices,
        mock_aggregate,
        mock_fasta_files,
        temp_output_dir,
        mock_index_data
    ):
        """Verify aggregate_shards called with successful shards only."""
        mock_device_count.return_value = 4
        mock_load_index.return_value = mock_index_data

        # Mock get_worker_indices to return different indices per rank
        def get_indices_side_effect(index_path, rank, world_size):
            return list(range(rank, 4, world_size))
        mock_get_worker_indices.side_effect = get_indices_side_effect

        # Mock coordinator behavior - workers 0,2 succeed, 1,3 fail
        mock_coordinator = MagicMock()
        mock_coordinator.wait_for_completion.return_value = {
            0: True, 1: False, 2: True, 3: False
        }
        mock_coordinator_cls.return_value = mock_coordinator

        # Create shard files for successful workers only
        temp_output_dir.mkdir(parents=True, exist_ok=True)
        (temp_output_dir / "shard_0.h5").touch()
        (temp_output_dir / "shard_2.h5").touch()

        # Mock aggregate_shards
        output_path = temp_output_dir / "embeddings.h5"
        mock_aggregate.return_value = output_path

        model_config = {'model_type': 'esm2'}

        result_path, failed = run_multi_gpu_inference(
            mock_fasta_files,
            temp_output_dir,
            model_config,
            world_size=4
        )

        # Verify aggregate_shards called with only successful shards
        mock_aggregate.assert_called_once()
        shard_files_arg = mock_aggregate.call_args[0][0]
        assert len(shard_files_arg) == 2
        assert temp_output_dir / "shard_0.h5" in shard_files_arg
        assert temp_output_dir / "shard_2.h5" in shard_files_arg

    @patch('virnucpro.pipeline.multi_gpu_inference.aggregate_shards')
    @patch('virnucpro.pipeline.multi_gpu_inference.GPUProcessCoordinator')
    @patch('virnucpro.pipeline.multi_gpu_inference.load_sequence_index')
    @patch('virnucpro.pipeline.multi_gpu_inference.create_sequence_index')
    @patch('virnucpro.pipeline.multi_gpu_inference.torch.cuda.device_count')
    def test_returns_output_path(
        self,
        mock_device_count,
        mock_create_index,
        mock_load_index,
        mock_coordinator_cls,
        mock_aggregate,
        mock_fasta_files,
        temp_output_dir,
        mock_index_data
    ):
        """Returns correct output path."""
        mock_device_count.return_value = 2
        mock_load_index.return_value = mock_index_data

        # Mock coordinator behavior
        mock_coordinator = MagicMock()
        mock_coordinator.wait_for_completion.return_value = {0: True, 1: True}
        mock_coordinator_cls.return_value = mock_coordinator

        # Create dummy shard files
        temp_output_dir.mkdir(parents=True, exist_ok=True)
        (temp_output_dir / "shard_0.h5").touch()
        (temp_output_dir / "shard_1.h5").touch()

        # Mock aggregate_shards
        expected_path = temp_output_dir / "embeddings.h5"
        mock_aggregate.return_value = expected_path

        model_config = {'model_type': 'esm2'}

        result_path, failed = run_multi_gpu_inference(
            mock_fasta_files,
            temp_output_dir,
            model_config
        )

        # Verify return value
        assert result_path == expected_path
        assert failed == []


class TestPartialFailureHandling:
    """Test partial failure scenarios."""

    @patch('virnucpro.pipeline.multi_gpu_inference.aggregate_shards')
    @patch('virnucpro.pipeline.multi_gpu_inference.get_worker_indices')
    @patch('virnucpro.pipeline.multi_gpu_inference.GPUProcessCoordinator')
    @patch('virnucpro.pipeline.multi_gpu_inference.load_sequence_index')
    @patch('virnucpro.pipeline.multi_gpu_inference.create_sequence_index')
    @patch('virnucpro.pipeline.multi_gpu_inference.torch.cuda.device_count')
    def test_partial_failure_still_aggregates(
        self,
        mock_device_count,
        mock_create_index,
        mock_load_index,
        mock_coordinator_cls,
        mock_get_worker_indices,
        mock_aggregate,
        mock_fasta_files,
        temp_output_dir,
        mock_index_data
    ):
        """Some workers fail but still get aggregated results."""
        mock_device_count.return_value = 3
        mock_load_index.return_value = mock_index_data

        # Mock get_worker_indices
        def get_indices_side_effect(index_path, rank, world_size):
            return list(range(rank, 4, world_size))
        mock_get_worker_indices.side_effect = get_indices_side_effect

        # Mock coordinator - worker 1 fails
        mock_coordinator = MagicMock()
        mock_coordinator.wait_for_completion.return_value = {
            0: True, 1: False, 2: True
        }
        mock_coordinator_cls.return_value = mock_coordinator

        # Create shard files for successful workers
        temp_output_dir.mkdir(parents=True, exist_ok=True)
        (temp_output_dir / "shard_0.h5").touch()
        (temp_output_dir / "shard_2.h5").touch()

        # Mock aggregate_shards
        output_path = temp_output_dir / "embeddings.h5"
        mock_aggregate.return_value = output_path

        model_config = {'model_type': 'esm2'}

        result_path, failed = run_multi_gpu_inference(
            mock_fasta_files,
            temp_output_dir,
            model_config,
            world_size=3
        )

        # Verify aggregation still happens
        assert mock_aggregate.called
        assert result_path == output_path

    @patch('virnucpro.pipeline.multi_gpu_inference.aggregate_shards')
    @patch('virnucpro.pipeline.multi_gpu_inference.get_worker_indices')
    @patch('virnucpro.pipeline.multi_gpu_inference.GPUProcessCoordinator')
    @patch('virnucpro.pipeline.multi_gpu_inference.load_sequence_index')
    @patch('virnucpro.pipeline.multi_gpu_inference.create_sequence_index')
    @patch('virnucpro.pipeline.multi_gpu_inference.torch.cuda.device_count')
    def test_partial_failure_returns_failed_ranks(
        self,
        mock_device_count,
        mock_create_index,
        mock_load_index,
        mock_coordinator_cls,
        mock_get_worker_indices,
        mock_aggregate,
        mock_fasta_files,
        temp_output_dir,
        mock_index_data
    ):
        """failed_ranks list populated correctly."""
        mock_device_count.return_value = 4
        mock_load_index.return_value = mock_index_data

        # Mock get_worker_indices
        def get_indices_side_effect(index_path, rank, world_size):
            return list(range(rank, 4, world_size))
        mock_get_worker_indices.side_effect = get_indices_side_effect

        # Mock coordinator - workers 1,3 fail
        mock_coordinator = MagicMock()
        mock_coordinator.wait_for_completion.return_value = {
            0: True, 1: False, 2: True, 3: False
        }
        mock_coordinator_cls.return_value = mock_coordinator

        # Create shard files for successful workers
        temp_output_dir.mkdir(parents=True, exist_ok=True)
        (temp_output_dir / "shard_0.h5").touch()
        (temp_output_dir / "shard_2.h5").touch()

        # Mock aggregate_shards
        output_path = temp_output_dir / "embeddings.h5"
        mock_aggregate.return_value = output_path

        model_config = {'model_type': 'esm2'}

        result_path, failed = run_multi_gpu_inference(
            mock_fasta_files,
            temp_output_dir,
            model_config,
            world_size=4
        )

        # Verify failed ranks returned
        assert failed == [1, 3]

    @patch('virnucpro.pipeline.multi_gpu_inference.aggregate_shards')
    @patch('virnucpro.pipeline.multi_gpu_inference.get_worker_indices')
    @patch('virnucpro.pipeline.multi_gpu_inference.GPUProcessCoordinator')
    @patch('virnucpro.pipeline.multi_gpu_inference.load_sequence_index')
    @patch('virnucpro.pipeline.multi_gpu_inference.create_sequence_index')
    @patch('virnucpro.pipeline.multi_gpu_inference.torch.cuda.device_count')
    def test_partial_failure_validates_partial(
        self,
        mock_device_count,
        mock_create_index,
        mock_load_index,
        mock_coordinator_cls,
        mock_get_worker_indices,
        mock_aggregate,
        mock_fasta_files,
        temp_output_dir,
        mock_index_data
    ):
        """Only successful worker IDs validated."""
        mock_device_count.return_value = 2
        mock_load_index.return_value = mock_index_data

        # Mock get_worker_indices - worker 0 gets seq1,seq3, worker 1 gets seq2,seq4
        def get_indices_side_effect(index_path, rank, world_size):
            if rank == 0:
                return [0, 2]  # seq1, seq3
            else:
                return [1, 3]  # seq2, seq4
        mock_get_worker_indices.side_effect = get_indices_side_effect

        # Mock coordinator - worker 1 fails
        mock_coordinator = MagicMock()
        mock_coordinator.wait_for_completion.return_value = {
            0: True, 1: False
        }
        mock_coordinator_cls.return_value = mock_coordinator

        # Create shard files for successful worker only
        temp_output_dir.mkdir(parents=True, exist_ok=True)
        (temp_output_dir / "shard_0.h5").touch()

        # Mock aggregate_shards
        output_path = temp_output_dir / "embeddings.h5"
        mock_aggregate.return_value = output_path

        model_config = {'model_type': 'esm2'}

        result_path, failed = run_multi_gpu_inference(
            mock_fasta_files,
            temp_output_dir,
            model_config,
            world_size=2
        )

        # Verify aggregate_shards called with only successful worker IDs
        mock_aggregate.assert_called_once()
        expected_ids_arg = mock_aggregate.call_args[0][2]
        # Should only validate seq1, seq3 (worker 0), not seq2, seq4 (worker 1 failed)
        assert expected_ids_arg == {'seq1', 'seq3'}

    @patch('virnucpro.pipeline.multi_gpu_inference.GPUProcessCoordinator')
    @patch('virnucpro.pipeline.multi_gpu_inference.load_sequence_index')
    @patch('virnucpro.pipeline.multi_gpu_inference.create_sequence_index')
    @patch('virnucpro.pipeline.multi_gpu_inference.torch.cuda.device_count')
    def test_all_workers_fail_raises(
        self,
        mock_device_count,
        mock_create_index,
        mock_load_index,
        mock_coordinator_cls,
        mock_fasta_files,
        temp_output_dir,
        mock_index_data
    ):
        """RuntimeError when all workers fail."""
        mock_device_count.return_value = 2
        mock_load_index.return_value = mock_index_data

        # Mock coordinator - all workers fail
        mock_coordinator = MagicMock()
        mock_coordinator.wait_for_completion.return_value = {
            0: False, 1: False
        }
        mock_coordinator_cls.return_value = mock_coordinator

        model_config = {'model_type': 'esm2'}

        with pytest.raises(RuntimeError, match="No workers completed successfully"):
            run_multi_gpu_inference(
                mock_fasta_files,
                temp_output_dir,
                model_config,
                world_size=2
            )


class TestWorldSizeDetection:
    """Test world_size auto-detection."""

    @patch('virnucpro.pipeline.multi_gpu_inference.aggregate_shards')
    @patch('virnucpro.pipeline.multi_gpu_inference.GPUProcessCoordinator')
    @patch('virnucpro.pipeline.multi_gpu_inference.load_sequence_index')
    @patch('virnucpro.pipeline.multi_gpu_inference.create_sequence_index')
    @patch('virnucpro.pipeline.multi_gpu_inference.torch.cuda.device_count')
    def test_auto_detects_gpu_count(
        self,
        mock_device_count,
        mock_create_index,
        mock_load_index,
        mock_coordinator_cls,
        mock_aggregate,
        mock_fasta_files,
        temp_output_dir,
        mock_index_data
    ):
        """Uses torch.cuda.device_count() when not specified."""
        mock_device_count.return_value = 4
        mock_load_index.return_value = mock_index_data

        # Mock coordinator
        mock_coordinator = MagicMock()
        mock_coordinator.wait_for_completion.return_value = {
            0: True, 1: True, 2: True, 3: True
        }
        mock_coordinator_cls.return_value = mock_coordinator

        # Create dummy shards
        temp_output_dir.mkdir(parents=True, exist_ok=True)
        for i in range(4):
            (temp_output_dir / f"shard_{i}.h5").touch()

        # Mock aggregate_shards
        output_path = temp_output_dir / "embeddings.h5"
        mock_aggregate.return_value = output_path

        model_config = {'model_type': 'esm2'}

        # Don't specify world_size
        result_path, failed = run_multi_gpu_inference(
            mock_fasta_files,
            temp_output_dir,
            model_config
        )

        # Verify coordinator created with detected device count
        mock_coordinator_cls.assert_called_once_with(4, temp_output_dir)

    @patch('virnucpro.pipeline.multi_gpu_inference.aggregate_shards')
    @patch('virnucpro.pipeline.multi_gpu_inference.GPUProcessCoordinator')
    @patch('virnucpro.pipeline.multi_gpu_inference.load_sequence_index')
    @patch('virnucpro.pipeline.multi_gpu_inference.create_sequence_index')
    @patch('virnucpro.pipeline.multi_gpu_inference.torch.cuda.device_count')
    def test_respects_explicit_world_size(
        self,
        mock_device_count,
        mock_create_index,
        mock_load_index,
        mock_coordinator_cls,
        mock_aggregate,
        mock_fasta_files,
        temp_output_dir,
        mock_index_data
    ):
        """Uses provided world_size value."""
        mock_device_count.return_value = 4  # Available GPUs
        mock_load_index.return_value = mock_index_data

        # Mock coordinator
        mock_coordinator = MagicMock()
        mock_coordinator.wait_for_completion.return_value = {0: True, 1: True}
        mock_coordinator_cls.return_value = mock_coordinator

        # Create dummy shards
        temp_output_dir.mkdir(parents=True, exist_ok=True)
        (temp_output_dir / "shard_0.h5").touch()
        (temp_output_dir / "shard_1.h5").touch()

        # Mock aggregate_shards
        output_path = temp_output_dir / "embeddings.h5"
        mock_aggregate.return_value = output_path

        model_config = {'model_type': 'esm2'}

        # Explicitly specify world_size=2 (less than available 4)
        result_path, failed = run_multi_gpu_inference(
            mock_fasta_files,
            temp_output_dir,
            model_config,
            world_size=2
        )

        # Verify coordinator created with explicit world_size
        mock_coordinator_cls.assert_called_once_with(2, temp_output_dir)


class TestConvenienceFunction:
    """Test run_esm2_multi_gpu convenience wrapper."""

    @patch('virnucpro.pipeline.multi_gpu_inference.run_multi_gpu_inference')
    def test_run_esm2_multi_gpu_defaults(
        self,
        mock_run_inference,
        mock_fasta_files,
        temp_output_dir
    ):
        """Correct model config passed to run_multi_gpu_inference."""
        # Mock successful inference
        output_path = temp_output_dir / "embeddings.h5"
        mock_run_inference.return_value = (output_path, [])

        result = run_esm2_multi_gpu(
            mock_fasta_files,
            temp_output_dir
        )

        # Verify run_multi_gpu_inference called with ESM-2 config
        mock_run_inference.assert_called_once()
        args, kwargs = mock_run_inference.call_args

        assert args[0] == mock_fasta_files
        assert args[1] == temp_output_dir

        model_config = args[2]
        assert model_config['model_type'] == 'esm2'
        assert model_config['model_name'] == 'esm2_t36_3B_UR50D'
        assert model_config['enable_fp16'] is True

        assert result == output_path

    @patch('virnucpro.pipeline.multi_gpu_inference.run_multi_gpu_inference')
    def test_run_esm2_multi_gpu_raises_on_failure(
        self,
        mock_run_inference,
        mock_fasta_files,
        temp_output_dir
    ):
        """Raises RuntimeError if any workers fail."""
        # Mock partial failure
        output_path = temp_output_dir / "embeddings.h5"
        mock_run_inference.return_value = (output_path, [1, 3])  # Workers 1,3 failed

        with pytest.raises(RuntimeError, match="failed on 2 workers"):
            run_esm2_multi_gpu(
                mock_fasta_files,
                temp_output_dir
            )

    @patch('virnucpro.pipeline.multi_gpu_inference.run_multi_gpu_inference')
    def test_run_esm2_multi_gpu_custom_model(
        self,
        mock_run_inference,
        mock_fasta_files,
        temp_output_dir
    ):
        """Custom model_name passed correctly."""
        # Mock successful inference
        output_path = temp_output_dir / "embeddings.h5"
        mock_run_inference.return_value = (output_path, [])

        custom_model = 'esm2_t12_35M_UR50D'
        result = run_esm2_multi_gpu(
            mock_fasta_files,
            temp_output_dir,
            model_name=custom_model
        )

        # Verify custom model name used
        model_config = mock_run_inference.call_args[0][2]
        assert model_config['model_name'] == custom_model
