"""Tests for CUDA stream orchestration and integration"""

import pytest
import torch
from pathlib import Path
import tempfile
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from virnucpro.cuda import StreamManager, StreamProcessor


@pytest.fixture
def cuda_device():
    """Get CUDA device if available, otherwise skip tests."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device('cuda:0')


@pytest.fixture
def sample_protein_file(tmp_path):
    """Create sample protein FASTA file for testing."""
    fasta_file = tmp_path / "test_proteins.fasta"

    records = [
        SeqRecord(Seq("MKFLKFSLLTAVLLSVVFAFSSCGDDDDTGYLPPSQAIQDLLKRMKV"), id="prot_1"),
        SeqRecord(Seq("MDSKGSSQKGSRLLLLLVVSNLLLCQGVVSTPVCPNGPGNCQVSLRDL"), id="prot_2"),
        SeqRecord(Seq("MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNL"), id="prot_3"),
    ]

    SeqIO.write(records, fasta_file, "fasta")
    return fasta_file


@pytest.fixture
def sample_dna_file(tmp_path):
    """Create sample DNA FASTA file for testing."""
    fasta_file = tmp_path / "test_dna.fasta"

    records = [
        SeqRecord(Seq("ATGAAATTCCTGAAATTCTCCCTGCTGACCGCAGTGCTGCTGTCCGTGGTGTTTGCATTCTCCTCCTGTGGCGATGATGATGAT"), id="dna_1"),
        SeqRecord(Seq("ATGGACTCCAAAGGCTCCTCCCAGAAAGGCTCCAGACTGCTGCTGCTGCTGGTGGTGTCCAACCTGCTGCTGTGTCAGGGCGTG"), id="dna_2"),
        SeqRecord(Seq("ATGAAAACCGCCTACATCGCCAAACAGCGTCAGATCTCCTTCGTGAAATCCCACTTCTCCCGTCAGCTGGAAGAACGTCTGGGC"), id="dna_3"),
    ]

    SeqIO.write(records, fasta_file, "fasta")
    return fasta_file


class TestStreamManager:
    """Test StreamManager functionality."""

    def test_stream_manager_initialization_with_cuda(self, cuda_device):
        """Test StreamManager initializes correctly with CUDA."""
        manager = StreamManager(device=cuda_device, enable_streams=True)

        assert manager.device == cuda_device
        assert manager.enable_streams is True
        assert manager.h2d_stream is not None
        assert manager.compute_stream is not None
        assert manager.d2h_stream is not None

    def test_stream_manager_initialization_disabled(self, cuda_device):
        """Test StreamManager with streams disabled."""
        manager = StreamManager(device=cuda_device, enable_streams=False)

        assert manager.device == cuda_device
        assert manager.enable_streams is False
        assert manager.h2d_stream is None
        assert manager.compute_stream is None
        assert manager.d2h_stream is None

    def test_get_stream(self, cuda_device):
        """Test getting streams by type."""
        manager = StreamManager(device=cuda_device, enable_streams=True)

        h2d = manager.get_stream('h2d')
        compute = manager.get_stream('compute')
        d2h = manager.get_stream('d2h')

        assert h2d is not None
        assert compute is not None
        assert d2h is not None
        assert h2d != compute
        assert compute != d2h

    def test_get_stream_invalid_type(self, cuda_device):
        """Test getting stream with invalid type raises error."""
        manager = StreamManager(device=cuda_device, enable_streams=True)

        with pytest.raises(ValueError, match="Invalid stream type"):
            manager.get_stream('invalid')

    def test_stream_context_manager(self, cuda_device):
        """Test stream context manager."""
        manager = StreamManager(device=cuda_device, enable_streams=True)

        # Test that operations execute on correct stream
        with manager.stream_context('h2d') as stream:
            assert stream is not None
            # Create tensor on stream
            x = torch.randn(10, 10, device=cuda_device)
            assert x.device == cuda_device

    def test_synchronize_all_streams(self, cuda_device):
        """Test synchronizing all streams."""
        manager = StreamManager(device=cuda_device, enable_streams=True)

        # Launch operations on different streams
        with manager.stream_context('h2d'):
            x = torch.randn(100, 100, device=cuda_device)

        with manager.stream_context('compute'):
            y = torch.matmul(x, x)

        # Synchronize should not raise
        manager.synchronize()

        # Verify operations completed
        assert x.shape == (100, 100)
        assert y.shape == (100, 100)

    def test_synchronize_specific_stream(self, cuda_device):
        """Test synchronizing a specific stream."""
        manager = StreamManager(device=cuda_device, enable_streams=True)

        with manager.stream_context('compute'):
            x = torch.randn(50, 50, device=cuda_device)

        # Should not raise
        manager.synchronize('compute')
        assert x.shape == (50, 50)

    def test_wait_for_stream(self, cuda_device):
        """Test making one stream wait for another."""
        manager = StreamManager(device=cuda_device, enable_streams=True)

        # Launch on h2d stream
        with manager.stream_context('h2d'):
            x = torch.randn(10, 10, device=cuda_device)

        # Make compute wait for h2d
        manager.wait_for_stream('compute', 'h2d')

        # Should not raise
        with manager.stream_context('compute'):
            y = x + 1

        manager.synchronize()
        assert torch.all(y == x + 1)

    def test_record_event(self, cuda_device):
        """Test recording events on streams."""
        manager = StreamManager(device=cuda_device, enable_streams=True)

        with manager.stream_context('compute'):
            x = torch.randn(10, 10, device=cuda_device)

        event = manager.record_event('compute')
        assert event is not None

        # Event should be recordable
        event.synchronize()

    def test_check_error_no_error(self, cuda_device):
        """Test error checking when no errors present."""
        manager = StreamManager(device=cuda_device, enable_streams=True)

        with manager.stream_context('compute'):
            x = torch.randn(10, 10, device=cuda_device)

        assert manager.check_error() is True

    def test_reset_streams(self, cuda_device):
        """Test resetting streams."""
        manager = StreamManager(device=cuda_device, enable_streams=True)

        with manager.stream_context('h2d'):
            x = torch.randn(10, 10, device=cuda_device)

        # Should synchronize and not raise
        manager.reset_streams()
        assert x.shape == (10, 10)


class TestStreamProcessor:
    """Test StreamProcessor functionality."""

    def test_stream_processor_initialization(self, cuda_device):
        """Test StreamProcessor initializes correctly."""
        processor = StreamProcessor(device=cuda_device, enable_streams=True)

        assert processor.device == cuda_device
        assert processor.stream_manager is not None
        assert processor.stream_manager.enable_streams is True

    def test_stream_processor_disabled(self, cuda_device):
        """Test StreamProcessor with streams disabled."""
        processor = StreamProcessor(device=cuda_device, enable_streams=False)

        assert processor.stream_manager.enable_streams is False

    def test_process_batch_async_simple(self, cuda_device):
        """Test async batch processing with simple operations."""
        processor = StreamProcessor(device=cuda_device, enable_streams=True)

        # Create CPU data
        cpu_data = torch.randn(10, 10)

        # Define pipeline functions
        def transfer_fn(data):
            return data.to(cuda_device, non_blocking=True)

        def compute_fn(gpu_data):
            return gpu_data * 2

        def retrieve_fn(result):
            return result.cpu()

        # Process
        result = processor.process_batch_async(
            cpu_data, transfer_fn, compute_fn, retrieve_fn
        )

        # Verify result
        processor.synchronize()
        assert torch.allclose(result, cpu_data * 2)

    def test_process_batch_async_without_retrieve(self, cuda_device):
        """Test async batch processing without D2H transfer."""
        processor = StreamProcessor(device=cuda_device, enable_streams=True)

        cpu_data = torch.randn(5, 5)

        def transfer_fn(data):
            return data.to(cuda_device, non_blocking=True)

        def compute_fn(gpu_data):
            return gpu_data + 10

        # Process without retrieve_fn
        result = processor.process_batch_async(
            cpu_data, transfer_fn, compute_fn, retrieve_fn=None
        )

        processor.synchronize()
        assert result.device == cuda_device
        assert torch.allclose(result.cpu(), cpu_data + 10)

    def test_process_batch_async_fallback_disabled(self, cuda_device):
        """Test fallback to synchronous when streams disabled."""
        processor = StreamProcessor(device=cuda_device, enable_streams=False)

        cpu_data = torch.randn(8, 8)

        def transfer_fn(data):
            return data.to(cuda_device)

        def compute_fn(gpu_data):
            return gpu_data * 3

        def retrieve_fn(result):
            return result.cpu()

        result = processor.process_batch_async(
            cpu_data, transfer_fn, compute_fn, retrieve_fn
        )

        assert torch.allclose(result, cpu_data * 3)

    def test_process_batches_pipelined(self, cuda_device):
        """Test pipelined processing of multiple batches."""
        processor = StreamProcessor(device=cuda_device, enable_streams=True)

        # Create multiple batches
        batches = [torch.randn(4, 4) for _ in range(5)]

        def transfer_fn(data):
            return data.to(cuda_device, non_blocking=True)

        def compute_fn(gpu_data):
            return gpu_data @ gpu_data.T

        def retrieve_fn(result):
            return result.cpu()

        # Process all batches
        results = processor.process_batches_pipelined(
            batches, transfer_fn, compute_fn, retrieve_fn
        )

        # Verify results
        assert len(results) == len(batches)
        for i, result in enumerate(results):
            expected = batches[i] @ batches[i].T
            assert torch.allclose(result, expected, atol=1e-5)

    def test_process_batches_pipelined_large(self, cuda_device):
        """Test pipelined processing with many batches for error checking."""
        processor = StreamProcessor(device=cuda_device, enable_streams=True)

        # Create 15 batches to trigger error checking (every 10 batches)
        batches = [torch.randn(3, 3) for _ in range(15)]

        def transfer_fn(data):
            return data.to(cuda_device, non_blocking=True)

        def compute_fn(gpu_data):
            return gpu_data + 1

        def retrieve_fn(result):
            return result.cpu()

        results = processor.process_batches_pipelined(
            batches, transfer_fn, compute_fn, retrieve_fn
        )

        assert len(results) == 15
        for i, result in enumerate(results):
            assert torch.allclose(result, batches[i] + 1)

    def test_process_batches_fallback_disabled(self, cuda_device):
        """Test batches fallback when streams disabled."""
        processor = StreamProcessor(device=cuda_device, enable_streams=False)

        batches = [torch.randn(2, 2) for _ in range(3)]

        def transfer_fn(data):
            return data.to(cuda_device)

        def compute_fn(gpu_data):
            return gpu_data * 2

        def retrieve_fn(result):
            return result.cpu()

        results = processor.process_batches_pipelined(
            batches, transfer_fn, compute_fn, retrieve_fn
        )

        assert len(results) == 3
        for i, result in enumerate(results):
            assert torch.allclose(result, batches[i] * 2)


class TestStreamIntegration:
    """Test stream integration with actual models."""

    @pytest.mark.slow
    def test_esm_with_streams(self, cuda_device, sample_protein_file, tmp_path):
        """Test ESM-2 feature extraction with stream processor."""
        # This test requires ESM model - mark as slow
        pytest.importorskip("esm")

        from virnucpro.pipeline.features import extract_esm_features

        output_file = tmp_path / "esm_features.pt"

        # Create stream processor
        stream_processor = StreamProcessor(device=cuda_device, enable_streams=True)

        # Extract features with streams
        result_file = extract_esm_features(
            sample_protein_file,
            output_file,
            cuda_device,
            toks_per_batch=1024,
            stream_processor=stream_processor
        )

        # Verify output
        assert result_file.exists()
        data = torch.load(result_file)
        assert 'proteins' in data
        assert 'data' in data
        assert len(data['proteins']) == 3
        assert len(data['data']) == 3

    @pytest.mark.slow
    def test_esm_without_streams(self, cuda_device, sample_protein_file, tmp_path):
        """Test ESM-2 feature extraction without streams (baseline)."""
        pytest.importorskip("esm")

        from virnucpro.pipeline.features import extract_esm_features

        output_file = tmp_path / "esm_features_no_streams.pt"

        # Extract features without streams
        result_file = extract_esm_features(
            sample_protein_file,
            output_file,
            cuda_device,
            toks_per_batch=1024,
            stream_processor=None
        )

        # Verify output
        assert result_file.exists()
        data = torch.load(result_file)
        assert len(data['proteins']) == 3
        assert len(data['data']) == 3

    @pytest.mark.slow
    def test_dnabert_worker_with_streams(self, cuda_device, sample_dna_file, tmp_path):
        """Test DNABERT-S worker with stream processor."""
        pytest.importorskip("transformers")

        from virnucpro.pipeline.parallel_dnabert import process_dnabert_files_worker

        # Process with streams enabled
        processed, failed = process_dnabert_files_worker(
            file_subset=[sample_dna_file],
            device_id=0,
            toks_per_batch=512,
            output_dir=tmp_path,
            enable_streams=True
        )

        # Verify results
        assert len(processed) == 1
        assert len(failed) == 0
        assert processed[0].exists()

        # Check output format
        data = torch.load(processed[0])
        assert 'nucleotide' in data
        assert 'data' in data
        assert len(data['nucleotide']) == 3

    @pytest.mark.slow
    def test_esm_worker_with_streams(self, cuda_device, sample_protein_file, tmp_path):
        """Test ESM-2 worker with stream processor."""
        pytest.importorskip("esm")

        from virnucpro.pipeline.parallel_esm import process_esm_files_worker

        # Process with streams enabled
        processed, failed = process_esm_files_worker(
            file_subset=[sample_protein_file],
            device_id=0,
            toks_per_batch=1024,
            output_dir=tmp_path,
            enable_streams=True
        )

        # Verify results
        assert len(processed) == 1
        assert len(failed) == 0
        assert processed[0].exists()

        data = torch.load(processed[0])
        assert 'proteins' in data
        assert len(data['proteins']) == 3


class TestStreamErrorHandling:
    """Test stream error detection and propagation."""

    def test_stream_error_detection(self, cuda_device):
        """Test that stream errors are detected."""
        manager = StreamManager(device=cuda_device, enable_streams=True)

        # Normal operations should pass
        with manager.stream_context('compute'):
            x = torch.randn(10, 10, device=cuda_device)

        assert manager.check_error() is True

    def test_stream_processor_error_checking(self, cuda_device):
        """Test StreamProcessor error checking during batch processing."""
        processor = StreamProcessor(device=cuda_device, enable_streams=True)

        # Valid operations
        batches = [torch.randn(5, 5) for _ in range(5)]

        def transfer_fn(data):
            return data.to(cuda_device, non_blocking=True)

        def compute_fn(gpu_data):
            return gpu_data * 2

        results = processor.process_batches_pipelined(
            batches, transfer_fn, compute_fn, retrieve_fn=None
        )

        # Should complete successfully
        assert len(results) == 5
        assert processor.check_error() is True


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
