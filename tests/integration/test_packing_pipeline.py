"""End-to-end integration tests for sequence packing pipeline.

Tests the complete flow: FASTA -> DataLoader -> packed inference -> embeddings.
Requires GPU and ESM-2 model.
"""
import pytest
import torch
import tempfile
import time
from pathlib import Path

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required for pipeline tests"
)


@pytest.fixture(scope="module")
def test_fasta_file():
    """Create temporary FASTA file with test sequences."""
    sequences = [
        (f"seq_{i}", "MKTAYIAK" * (5 + i % 20))
        for i in range(100)  # 100 sequences of varying lengths
    ]
    with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as f:
        for seq_id, seq in sequences:
            f.write(f">{seq_id}\n{seq}\n")
        return Path(f.name)


@pytest.fixture(scope="module")
def pipeline_components():
    """Load model and create pipeline components."""
    from virnucpro.models.esm2_flash import load_esm2_model
    from virnucpro.data import SequenceDataset, VarlenCollator, create_async_dataloader

    model, batch_converter = load_esm2_model(
        model_name="esm2_t33_650M_UR50D",
        device="cuda:0"
    )
    return model, batch_converter


class TestPackingPipeline:
    """End-to-end pipeline tests."""

    def test_full_pipeline_flow(self, test_fasta_file, pipeline_components):
        """Complete FASTA -> embeddings flow."""
        model, batch_converter = pipeline_components

        from virnucpro.data import SequenceDataset, VarlenCollator, create_async_dataloader
        from virnucpro.pipeline import AsyncInferenceRunner

        dataset = SequenceDataset(fasta_files=[str(test_fasta_file)])
        collator = VarlenCollator(batch_converter, max_tokens_per_batch=4096)
        dataloader = create_async_dataloader(dataset, collator, num_workers=2)

        runner = AsyncInferenceRunner(model, device=torch.device("cuda:0"))
        results = list(runner.run(dataloader))

        # Verify we got embeddings for all sequences
        total_sequences = sum(len(r.sequence_ids) for r in results)
        assert total_sequences == 100, f"Expected 100, got {total_sequences}"

        # Verify embeddings have correct shape
        for result in results:
            assert result.embeddings.dim() == 2
            assert result.embeddings.shape[1] == 2560  # ESM-2 650M hidden dim

    def test_packing_efficiency(self, test_fasta_file, pipeline_components):
        """Verify packing efficiency >90%."""
        model, batch_converter = pipeline_components

        from virnucpro.data import SequenceDataset, VarlenCollator, create_async_dataloader
        from virnucpro.pipeline import AsyncInferenceRunner

        dataset = SequenceDataset(fasta_files=[str(test_fasta_file)])
        collator = VarlenCollator(batch_converter, max_tokens_per_batch=4096)
        dataloader = create_async_dataloader(dataset, collator, num_workers=2)

        runner = AsyncInferenceRunner(model, device=torch.device("cuda:0"))
        list(runner.run(dataloader))

        stats = runner.get_statistics()
        dl_stats = stats.get('dataloader_stats', {})
        efficiency = dl_stats.get('avg_packing_efficiency', 0)

        assert efficiency > 0.9, f"Packing efficiency {efficiency:.1%} < 90%"

    def test_throughput_baseline(self, test_fasta_file, pipeline_components):
        """Measure throughput for baseline comparison."""
        model, batch_converter = pipeline_components

        from virnucpro.data import SequenceDataset, VarlenCollator, create_async_dataloader
        from virnucpro.pipeline import AsyncInferenceRunner

        dataset = SequenceDataset(fasta_files=[str(test_fasta_file)])
        collator = VarlenCollator(batch_converter, max_tokens_per_batch=4096)
        dataloader = create_async_dataloader(dataset, collator, num_workers=2)

        runner = AsyncInferenceRunner(model, device=torch.device("cuda:0"))

        start_time = time.perf_counter()
        results = list(runner.run(dataloader))
        elapsed = time.perf_counter() - start_time

        total_sequences = sum(len(r.sequence_ids) for r in results)
        seqs_per_sec = total_sequences / elapsed

        print(f"\nThroughput: {seqs_per_sec:.1f} sequences/sec")
        print(f"Total time: {elapsed:.2f}s for {total_sequences} sequences")

        # Basic sanity check - should process at least 10 seq/sec on any GPU
        assert seqs_per_sec > 10, f"Throughput too low: {seqs_per_sec:.1f} seq/s"

    def test_throughput_comparison_packed_vs_unpacked(self, test_fasta_file, pipeline_components):
        """Verify packed throughput is 2-3x faster than unpacked (Gap 9)."""
        model, batch_converter = pipeline_components

        from virnucpro.data import SequenceDataset, VarlenCollator, create_async_dataloader
        from virnucpro.pipeline import AsyncInferenceRunner
        import time

        # Baseline: unpacked (enable_packing=False)
        dataset_unpacked = SequenceDataset(fasta_files=[str(test_fasta_file)])
        collator_unpacked = VarlenCollator(batch_converter, enable_packing=False)
        dataloader_unpacked = create_async_dataloader(dataset_unpacked, collator_unpacked, num_workers=2)
        runner = AsyncInferenceRunner(model, device=torch.device("cuda:0"))

        start = time.perf_counter()
        results_unpacked = list(runner.run(dataloader_unpacked))
        unpacked_time = time.perf_counter() - start
        unpacked_seqs = sum(len(r.sequence_ids) for r in results_unpacked)
        unpacked_throughput = unpacked_seqs / unpacked_time

        # Packed (enable_packing=True, buffer_size=2000)
        dataset_packed = SequenceDataset(fasta_files=[str(test_fasta_file)])
        collator_packed = VarlenCollator(batch_converter, enable_packing=True, buffer_size=2000)
        dataloader_packed = create_async_dataloader(dataset_packed, collator_packed, num_workers=2)

        start = time.perf_counter()
        results_packed = list(runner.run(dataloader_packed))
        packed_time = time.perf_counter() - start
        packed_seqs = sum(len(r.sequence_ids) for r in results_packed)
        packed_throughput = packed_seqs / packed_time

        # Calculate speedup
        speedup = packed_throughput / unpacked_throughput

        print(f"\n--- Throughput Comparison ---")
        print(f"Unpacked: {unpacked_throughput:.1f} seq/s ({unpacked_time:.2f}s total)")
        print(f"Packed:   {packed_throughput:.1f} seq/s ({packed_time:.2f}s total)")
        print(f"Speedup:  {speedup:.1f}x")

        # Verify 2-3x target (Gap 9)
        assert speedup >= 2.0, \
            f"Packed speedup {speedup:.1f}x < 2.0x target. " \
            f"Buffer may not be filling (check warmup batches in logs)"

        # Note: May not reach 3x if test dataset is small (buffer doesn't fill)
        # Gap 8 clarification: Trust buffer design, measure after warmup
        if packed_seqs < 2000:
            print(f"  Note: Dataset has only {packed_seqs} sequences. "
                  "Full speedup requires >2000 seqs to fill buffer.")


class TestOversizedSequences:
    """Test handling of sequences exceeding limits."""

    def test_truncation_warning(self, pipeline_components, caplog):
        """Verify oversized sequences are truncated with warning."""
        model, batch_converter = pipeline_components

        from virnucpro.data import GreedyPacker

        # Create sequence exceeding max length
        packer = GreedyPacker(max_tokens_per_batch=4096, max_sequence_length=100)
        sequences = [
            {'id': 'oversized', 'sequence': 'M' * 200},  # 200 aa > 100 limit
        ]

        batches = packer.pack_sequences(sequences)
        assert batches[0][0].get('truncated', False), "Truncation flag not set"
        assert len(batches[0][0]['sequence']) == 100, "Sequence not truncated"
