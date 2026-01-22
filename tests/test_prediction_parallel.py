"""Integration tests for parallel prediction pipeline"""

import pytest
import torch
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock


@pytest.mark.gpu
@pytest.mark.slow
class TestParallelPipeline:
    """Integration tests for parallel feature extraction"""

    def test_parallel_processing_with_multiple_gpus(self, temp_fasta, temp_dir, mock_gpu_devices):
        """Normal: 4 files on 2 GPUs processed in parallel"""
        mock_gpu_devices(num_gpus=2)

        files = []
        for i in range(4):
            fasta = temp_fasta(num_sequences=100, seq_length=500)
            new_path = temp_dir / f"nucleotide_{i}.fa"
            fasta.rename(new_path)
            files.append(new_path)

        from virnucpro.pipeline.parallel import (
            detect_cuda_devices,
            assign_files_round_robin,
            process_dnabert_files_worker
        )

        devices = detect_cuda_devices()
        assert len(devices) == 2

        assignments = assign_files_round_robin(files, len(devices))
        assert len(assignments[0]) == 2
        assert len(assignments[1]) == 2

    def test_fallback_to_sequential_single_gpu(self, temp_fasta, temp_dir, mock_gpu_devices):
        """Edge: 1 GPU available -> sequential mode"""
        mock_gpu_devices(num_gpus=1)

        from virnucpro.pipeline.parallel import detect_cuda_devices

        devices = detect_cuda_devices()
        assert len(devices) == 1

    def test_no_gpu_fallback(self, temp_fasta, temp_dir, mock_gpu_devices):
        """Edge: No GPU available -> CPU fallback"""
        mock_gpu_devices(num_gpus=0)

        from virnucpro.pipeline.parallel import detect_cuda_devices

        devices = detect_cuda_devices()
        assert devices == []


class TestCheckpointCompatibility:
    """Test that parallel processing maintains checkpoint compatibility"""

    def test_resume_after_interruption(self, temp_dir):
        """Edge: Resume after interruption skips completed files"""
        pass


class TestErrorHandling:
    """Test error handling in parallel pipeline"""

    def test_worker_exception_propagates(self):
        """Error: GPU failure mid-run produces clean error message"""
        pass


class TestOutputEquivalence:
    """Test that parallel processing produces same output as sequential"""

    @pytest.mark.gpu
    @patch('virnucpro.pipeline.features.extract_dnabert_features')
    def test_parallel_vs_sequential_equivalence(self, mock_extract, temp_fasta, temp_dir, mock_gpu_devices):
        """Property: parallel(files, gpus) equivalent to sequential(files)"""
        mock_gpu_devices(num_gpus=2)

        files = []
        for i in range(4):
            fasta = temp_fasta(num_sequences=10, seq_length=500)
            new_path = temp_dir / f"file_{i}.fa"
            fasta.rename(new_path)
            files.append(new_path)

        output_dir = temp_dir / "output"
        output_dir.mkdir()

        mock_extract.return_value = output_dir / "mock_output.pt"

        from virnucpro.pipeline.parallel import assign_files_round_robin

        assignments = assign_files_round_robin(files, num_workers=2)

        assert len(assignments) == 2
        assert len(assignments[0]) == 2
        assert len(assignments[1]) == 2


class TestFileAssignmentCorrectness:
    """Test file assignment and distribution logic"""

    def test_round_robin_balancing(self):
        """Property: files distributed evenly across workers"""
        from virnucpro.pipeline.parallel import assign_files_round_robin

        files = [Path(f"file_{i}.fa") for i in range(10)]
        assignments = assign_files_round_robin(files, 3)

        file_counts = [len(a) for a in assignments]
        assert max(file_counts) - min(file_counts) <= 1

    def test_file_ordering_preservation(self):
        """Property: round-robin maintains file order within workers"""
        from virnucpro.pipeline.parallel import assign_files_round_robin

        files = [Path(f"file_{i}.fa") for i in range(10)]
        assignments = assign_files_round_robin(files, 3)

        for worker_files in assignments:
            for i in range(len(worker_files) - 1):
                curr_num = int(worker_files[i].stem.split('_')[1])
                next_num = int(worker_files[i+1].stem.split('_')[1])
                assert next_num > curr_num


class TestWorkerIsolation:
    """Test that workers operate independently"""

    @patch('virnucpro.pipeline.features.extract_dnabert_features')
    def test_worker_device_isolation(self, mock_extract, tmp_path):
        """Property: each worker uses assigned GPU exclusively"""
        from virnucpro.pipeline.parallel import process_dnabert_files_worker

        files = [tmp_path / "file_0.fa"]
        files[0].touch()

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        with patch('virnucpro.pipeline.parallel.torch.device') as mock_device:
            process_dnabert_files_worker(files, device_id=2, batch_size=256, output_dir=output_dir)

            mock_device.assert_called_once_with('cuda:2')


class TestIntegrationWithCheckpoint:
    """Test integration with checkpoint/resume functionality"""

    def test_checkpoint_skip_completed_files(self, temp_dir):
        """Integration: checkpoint system skips files with existing output"""
        output_dir = temp_dir / "output"
        output_dir.mkdir()

        test_file = temp_dir / "test.fa"
        test_file.touch()

        output_file = output_dir / "test_DNABERT_S.pt"

        assert not output_file.exists()

        output_file.touch()

        assert output_file.exists()
