"""Tests for parallel processing utilities"""

import pytest
import torch
from pathlib import Path
from unittest.mock import patch, MagicMock
from virnucpro.pipeline.parallel import (
    detect_cuda_devices,
    assign_files_round_robin,
    process_dnabert_files_worker
)


class TestDetectCudaDevices:
    """Test CUDA device detection"""

    @patch('torch.cuda.is_available')
    @patch('torch.cuda.device_count')
    def test_multiple_gpus(self, mock_count, mock_available):
        mock_available.return_value = True
        mock_count.return_value = 4

        devices = detect_cuda_devices()

        assert devices == [0, 1, 2, 3]

    @patch('torch.cuda.is_available')
    def test_no_cuda_available(self, mock_available):
        mock_available.return_value = False

        devices = detect_cuda_devices()

        assert devices == []

    @patch('torch.cuda.is_available')
    @patch('torch.cuda.device_count')
    def test_single_gpu(self, mock_count, mock_available):
        mock_available.return_value = True
        mock_count.return_value = 1

        devices = detect_cuda_devices()

        assert devices == [0]


class TestAssignFilesRoundRobin:
    """Test file assignment to workers"""

    def test_normal_8_files_4_gpus(self):
        files = [Path(f"file_{i}.fa") for i in range(8)]

        assignments = assign_files_round_robin(files, 4)

        assert len(assignments) == 4
        assert len(assignments[0]) == 2
        assert len(assignments[1]) == 2
        assert len(assignments[2]) == 2
        assert len(assignments[3]) == 2
        assert assignments[0][0] == Path("file_0.fa")
        assert assignments[0][1] == Path("file_4.fa")
        assert assignments[1][0] == Path("file_1.fa")
        assert assignments[1][1] == Path("file_5.fa")

    def test_edge_3_files_4_gpus(self):
        files = [Path(f"file_{i}.fa") for i in range(3)]

        assignments = assign_files_round_robin(files, 4)

        assert len(assignments) == 4
        assert len(assignments[0]) == 1
        assert len(assignments[1]) == 1
        assert len(assignments[2]) == 1
        assert len(assignments[3]) == 0

    def test_edge_empty_files(self):
        files = []

        assignments = assign_files_round_robin(files, 4)

        assert len(assignments) == 4
        assert all(len(a) == 0 for a in assignments)

    def test_error_negative_workers(self):
        files = [Path("file.fa")]

        with pytest.raises(ValueError, match="num_workers must be positive"):
            assign_files_round_robin(files, -1)

    def test_error_zero_workers(self):
        files = [Path("file.fa")]

        with pytest.raises(ValueError, match="num_workers must be positive"):
            assign_files_round_robin(files, 0)

    def test_uneven_distribution(self):
        files = [Path(f"file_{i}.fa") for i in range(10)]

        assignments = assign_files_round_robin(files, 3)

        assert len(assignments) == 3
        assert len(assignments[0]) == 4
        assert len(assignments[1]) == 3
        assert len(assignments[2]) == 3


class TestProcessDnabertFilesWorker:
    """Test worker function for DNABERT-S processing"""

    @patch('virnucpro.pipeline.parallel.torch.device')
    @patch('virnucpro.pipeline.features.extract_dnabert_features')
    def test_normal_processing(self, mock_extract, mock_device, tmp_path):
        files = [tmp_path / "file_0.fa", tmp_path / "file_1.fa"]
        for f in files:
            f.touch()

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        mock_extract.return_value = tmp_path / "output_0_DNABERT_S.pt"

        result = process_dnabert_files_worker(files, 0, 256, output_dir)

        assert len(result) == 2
        assert mock_extract.call_count == 2
        mock_extract.assert_any_call(
            files[0],
            output_dir / "file_0_DNABERT_S.pt",
            mock_device.return_value,
            batch_size=256
        )

    @patch('virnucpro.pipeline.parallel.torch.device')
    @patch('virnucpro.pipeline.features.extract_dnabert_features')
    def test_error_during_extraction(self, mock_extract, mock_device, tmp_path):
        files = [tmp_path / "file_0.fa"]
        files[0].touch()

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        mock_extract.side_effect = RuntimeError("GPU out of memory")

        with pytest.raises(RuntimeError, match="GPU out of memory"):
            process_dnabert_files_worker(files, 0, 256, output_dir)

    @patch('virnucpro.pipeline.parallel.torch.device')
    @patch('virnucpro.pipeline.features.extract_dnabert_features')
    def test_empty_file_list(self, mock_extract, mock_device, tmp_path):
        files = []
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        result = process_dnabert_files_worker(files, 0, 256, output_dir)

        assert result == []
        assert mock_extract.call_count == 0

    @patch('virnucpro.pipeline.parallel.torch.device')
    @patch('virnucpro.pipeline.features.extract_dnabert_features')
    def test_device_assignment(self, mock_extract, mock_device, tmp_path):
        files = [tmp_path / "file.fa"]
        files[0].touch()

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        process_dnabert_files_worker(files, 2, 256, output_dir)

        mock_device.assert_called_with('cuda:2')
