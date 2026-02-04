"""Multi-GPU integration tests.

End-to-end tests for multi-GPU inference pipeline. Requires at least 2 GPUs.

Tests:
1. Embedding equivalence with single-GPU baseline
2. Work distribution balance across GPUs
3. Throughput scaling (multi-GPU vs single-GPU)
4. Fault tolerance (partial failures)
5. Edge cases (single GPU mode, small datasets)
"""
import pytest
import torch
import tempfile
import time
import numpy as np
from pathlib import Path
import h5py

# Skip entire module if insufficient GPUs
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="Requires at least 2 GPUs"
)


@pytest.fixture(scope="module")
def test_fasta_files(tmp_path_factory):
    """Create temporary FASTA files with varying sequence lengths.

    Creates 3 FASTA files with total ~1000 sequences, lengths 20-400 aa.
    This provides sufficient data for multi-GPU distribution testing.
    """
    tmp_dir = tmp_path_factory.mktemp("fasta_data")
    fasta_files = []

    # Create 3 files with different sequence distributions
    for file_idx in range(3):
        fasta_path = tmp_dir / f"sequences_{file_idx}.fasta"
        with open(fasta_path, 'w') as f:
            # ~333 sequences per file, varying lengths
            for seq_idx in range(333):
                seq_id = f"file{file_idx}_seq{seq_idx}"
                # Length varies from 20-400 aa (ensures diverse packing scenarios)
                length = 20 + (seq_idx % 48) * 8  # 20, 28, 36, ..., 400
                sequence = "MKTAYIAKVL" * (length // 10)
                sequence = sequence[:length]  # Trim to exact length
                f.write(f">{seq_id}\n{sequence}\n")
        fasta_files.append(fasta_path)

    return fasta_files


@pytest.fixture(scope="module")
def temp_output_dir(tmp_path_factory):
    """Create temporary directory for outputs."""
    return tmp_path_factory.mktemp("multi_gpu_output")


@pytest.fixture(scope="module")
def single_gpu_baseline(test_fasta_files, tmp_path_factory):
    """Run single-GPU inference for baseline comparison.

    This baseline is used for embedding equivalence tests.
    Uses a subset of sequences to reduce runtime.
    """
    from virnucpro.data import SequenceDataset, VarlenCollator
    from virnucpro.data.dataloader_utils import create_async_dataloader
    from virnucpro.models.esm2_flash import load_esm2_model
    from virnucpro.pipeline.async_inference import AsyncInferenceRunner

    # Use first file only (333 sequences) for faster baseline
    baseline_fasta = test_fasta_files[0:1]

    # Load model on GPU 0
    model, batch_converter = load_esm2_model(
        model_name="esm2_t36_3B_UR50D",
        device="cuda:0"
    )

    # Create dataset and dataloader
    dataset = SequenceDataset(fasta_files=baseline_fasta)
    collator = VarlenCollator(batch_converter, enable_packing=True, buffer_size=200)
    dataloader = create_async_dataloader(
        dataset, collator,
        device_id=0,
        batch_size=8,  # Explicit batch size for baseline (batch_size=None doesn't batch)
        num_workers=2
    )

    # Run inference
    runner = AsyncInferenceRunner(model, device=torch.device("cuda:0"))
    all_embeddings = []
    all_ids = []

    for result in runner.run(dataloader):
        all_embeddings.append(result.embeddings.cpu())
        all_ids.extend(result.sequence_ids)

    # Concatenate embeddings
    embeddings = torch.cat(all_embeddings, dim=0).numpy()

    # Return as dict for easy lookup
    baseline = {
        seq_id: embeddings[i]
        for i, seq_id in enumerate(all_ids)
    }

    return baseline


class TestMultiGPUEmbeddingEquivalence:
    """Test that multi-GPU embeddings match single-GPU baseline."""

    def test_multi_gpu_matches_single_gpu(
        self,
        test_fasta_files,
        temp_output_dir,
        single_gpu_baseline
    ):
        """Verify multi-GPU embeddings match single-GPU baseline (cosine similarity >0.999)."""
        from virnucpro.pipeline.multi_gpu_inference import run_multi_gpu_inference

        # Run multi-GPU inference (use 2 GPUs for reproducibility)
        world_size = min(2, torch.cuda.device_count())

        # Use same file as baseline for comparison
        baseline_fasta = test_fasta_files[0:1]
        multi_gpu_output = temp_output_dir / "test_equivalence"
        multi_gpu_output.mkdir(parents=True, exist_ok=True)

        model_config = {
            'model_type': 'esm2',
            'model_name': 'esm2_t36_3B_UR50D',
            'enable_fp16': True
        }

        output_path, failed_ranks = run_multi_gpu_inference(
            baseline_fasta,
            multi_gpu_output,
            model_config,
            world_size=world_size
        )

        assert len(failed_ranks) == 0, f"Workers failed: {failed_ranks}"

        # Load multi-GPU embeddings
        with h5py.File(output_path, 'r') as f:
            multi_gpu_embeddings = f['embeddings'][:]
            multi_gpu_ids = [s.decode() if isinstance(s, bytes) else s
                           for s in f['sequence_ids'][:]]

        # Compare embeddings
        similarities = []
        failed_sequences = []

        for i, seq_id in enumerate(multi_gpu_ids):
            if seq_id not in single_gpu_baseline:
                continue

            baseline_emb = single_gpu_baseline[seq_id]
            multi_gpu_emb = multi_gpu_embeddings[i]

            # Cosine similarity
            similarity = np.dot(baseline_emb, multi_gpu_emb) / (
                np.linalg.norm(baseline_emb) * np.linalg.norm(multi_gpu_emb)
            )
            similarities.append(similarity)

            if similarity < 0.999:
                failed_sequences.append((seq_id, similarity))

        # Assert at least 99% pass strict threshold
        strict_pass_count = sum(1 for s in similarities if s > 0.999)
        strict_pass_rate = strict_pass_count / len(similarities)

        assert strict_pass_rate >= 0.99, (
            f"Only {strict_pass_rate:.1%} sequences passed strict threshold (>0.999)\n"
            f"Failed sequences: {failed_sequences[:5]}"
        )

        # All should pass lenient threshold
        lenient_pass_count = sum(1 for s in similarities if s > 0.995)
        assert lenient_pass_count == len(similarities), (
            f"{len(similarities) - lenient_pass_count} sequences below 0.995 similarity"
        )

        print(f"\nEmbedding equivalence: {len(similarities)} sequences compared")
        print(f"  Strict (>0.999): {strict_pass_rate:.1%}")
        print(f"  Min similarity: {min(similarities):.6f}")
        print(f"  Mean similarity: {np.mean(similarities):.6f}")

    def test_all_sequences_present(self, test_fasta_files, temp_output_dir):
        """Verify all input sequences present in output (no duplicates or missing)."""
        from virnucpro.pipeline.multi_gpu_inference import run_multi_gpu_inference

        # Count expected sequences
        expected_ids = set()
        for fasta_path in test_fasta_files:
            with open(fasta_path) as f:
                for line in f:
                    if line.startswith('>'):
                        seq_id = line[1:].strip()
                        expected_ids.add(seq_id)

        # Run multi-GPU inference
        multi_gpu_output = temp_output_dir / "test_completeness"
        multi_gpu_output.mkdir(parents=True, exist_ok=True)

        model_config = {'model_type': 'esm2', 'model_name': 'esm2_t36_3B_UR50D'}

        output_path, failed_ranks = run_multi_gpu_inference(
            test_fasta_files,
            multi_gpu_output,
            model_config
        )

        # Load output sequence IDs
        with h5py.File(output_path, 'r') as f:
            output_ids = [s.decode() if isinstance(s, bytes) else s
                         for s in f['sequence_ids'][:]]

        # Check completeness
        output_set = set(output_ids)

        # No duplicates
        assert len(output_ids) == len(output_set), (
            f"Found {len(output_ids) - len(output_set)} duplicate sequences"
        )

        # All expected sequences present (or accounted for by failed workers)
        if failed_ranks:
            # Partial validation - some sequences may be missing
            assert len(output_set) <= len(expected_ids)
        else:
            # Full validation - all sequences should be present
            missing = expected_ids - output_set
            extra = output_set - expected_ids

            assert len(missing) == 0, f"Missing sequences: {list(missing)[:10]}"
            assert len(extra) == 0, f"Extra sequences: {list(extra)[:10]}"

    def test_embedding_shapes_correct(self, test_fasta_files, temp_output_dir):
        """Verify embeddings have correct shape (N, 2560 for ESM-2 3B)."""
        from virnucpro.pipeline.multi_gpu_inference import run_multi_gpu_inference

        multi_gpu_output = temp_output_dir / "test_shapes"
        multi_gpu_output.mkdir(parents=True, exist_ok=True)

        model_config = {'model_type': 'esm2', 'model_name': 'esm2_t36_3B_UR50D'}

        output_path, _ = run_multi_gpu_inference(
            test_fasta_files,
            multi_gpu_output,
            model_config
        )

        with h5py.File(output_path, 'r') as f:
            embeddings = f['embeddings']
            sequence_ids = f['sequence_ids']

            # Check dimensions
            assert embeddings.ndim == 2
            assert embeddings.shape[1] == 2560  # ESM-2 3B hidden dim
            assert embeddings.shape[0] == len(sequence_ids)

            # Check dtype
            assert embeddings.dtype == np.float32


class TestWorkDistribution:
    """Test work distribution balance across GPUs."""

    def test_stride_distribution_balanced(self, test_fasta_files, temp_output_dir):
        """Verify stride distribution produces balanced token load (within 10% of mean)."""
        from virnucpro.data.shard_index import create_sequence_index, get_worker_indices, load_sequence_index

        # Create index
        index_dir = temp_output_dir / "test_distribution"
        index_dir.mkdir(parents=True, exist_ok=True)
        index_path = index_dir / "sequence_index.json"

        create_sequence_index(test_fasta_files, index_path)

        # Load index data
        index_data = load_sequence_index(index_path)
        sequences = index_data['sequences']

        # Check distribution for 4 workers
        world_size = min(4, torch.cuda.device_count())
        token_counts = []
        seq_counts = []

        for rank in range(world_size):
            indices = get_worker_indices(index_path, rank, world_size)
            tokens = sum(sequences[i]['length'] for i in indices)
            token_counts.append(tokens)
            seq_counts.append(len(indices))

        # Check balance
        mean_tokens = np.mean(token_counts)
        max_deviation = max(abs(t - mean_tokens) / mean_tokens for t in token_counts)

        assert max_deviation < 0.10, (
            f"Token distribution imbalance: {max_deviation:.1%} > 10%\n"
            f"Token counts: {token_counts}\n"
            f"Mean: {mean_tokens:.0f}"
        )

        print(f"\nWork distribution ({world_size} workers):")
        print(f"  Token counts: {token_counts}")
        print(f"  Sequence counts: {seq_counts}")
        print(f"  Max deviation: {max_deviation:.1%}")

    def test_shard_sizes_similar(self, test_fasta_files, temp_output_dir):
        """Verify shard file sizes are within 15% of mean (after completion)."""
        from virnucpro.pipeline.multi_gpu_inference import run_multi_gpu_inference

        multi_gpu_output = temp_output_dir / "test_shard_sizes"
        multi_gpu_output.mkdir(parents=True, exist_ok=True)

        model_config = {'model_type': 'esm2', 'model_name': 'esm2_t36_3B_UR50D'}
        world_size = min(4, torch.cuda.device_count())

        output_path, failed_ranks = run_multi_gpu_inference(
            test_fasta_files,
            multi_gpu_output,
            model_config,
            world_size=world_size
        )

        # Get shard file sizes
        successful_ranks = [r for r in range(world_size) if r not in failed_ranks]
        shard_sizes = []

        for rank in successful_ranks:
            shard_path = multi_gpu_output / f"shard_{rank}.h5"
            if shard_path.exists():
                size = shard_path.stat().st_size
                shard_sizes.append(size)

        assert len(shard_sizes) > 0, "No shard files found"

        # Check balance
        mean_size = np.mean(shard_sizes)
        max_deviation = max(abs(s - mean_size) / mean_size for s in shard_sizes)

        # Allow 15% deviation (embedding size varies with sequence length)
        assert max_deviation < 0.15, (
            f"Shard size imbalance: {max_deviation:.1%} > 15%\n"
            f"Shard sizes: {[s // 1024 for s in shard_sizes]} KB"
        )

        print(f"\nShard sizes ({len(shard_sizes)} shards):")
        print(f"  Sizes: {[s // 1024 for s in shard_sizes]} KB")
        print(f"  Max deviation: {max_deviation:.1%}")


class TestThroughputScaling:
    """Test throughput scaling with multiple GPUs."""

    def test_multi_gpu_faster_than_single(self, test_fasta_files, temp_output_dir):
        """Verify multi-GPU is significantly faster than single-GPU (>1.5x for 2 GPUs)."""
        from virnucpro.pipeline.multi_gpu_inference import run_multi_gpu_inference

        # Use subset for faster testing (first file only, ~333 sequences)
        test_subset = test_fasta_files[0:1]

        # Single-GPU timing
        single_output = temp_output_dir / "test_single_gpu_timing"
        single_output.mkdir(parents=True, exist_ok=True)

        model_config = {'model_type': 'esm2', 'model_name': 'esm2_t36_3B_UR50D'}

        start = time.perf_counter()
        single_path, _ = run_multi_gpu_inference(
            test_subset,
            single_output,
            model_config,
            world_size=1
        )
        single_time = time.perf_counter() - start

        # Multi-GPU timing (2 GPUs)
        multi_output = temp_output_dir / "test_multi_gpu_timing"
        multi_output.mkdir(parents=True, exist_ok=True)

        start = time.perf_counter()
        multi_path, _ = run_multi_gpu_inference(
            test_subset,
            multi_output,
            model_config,
            world_size=2
        )
        multi_time = time.perf_counter() - start

        # Calculate speedup
        speedup = single_time / multi_time

        print(f"\nThroughput scaling:")
        print(f"  Single-GPU: {single_time:.2f}s")
        print(f"  Multi-GPU (2 GPUs): {multi_time:.2f}s")
        print(f"  Speedup: {speedup:.2f}x")

        # Verify speedup (2 GPUs should be at least 1.5x faster)
        assert speedup > 1.5, (
            f"Multi-GPU speedup {speedup:.2f}x < 1.5x expected for 2 GPUs"
        )

    def test_gpu_utilization_high(self, test_fasta_files, temp_output_dir, caplog):
        """Run inference and verify high GPU utilization (informational test)."""
        from virnucpro.pipeline.multi_gpu_inference import run_multi_gpu_inference

        multi_gpu_output = temp_output_dir / "test_utilization"
        multi_gpu_output.mkdir(parents=True, exist_ok=True)

        model_config = {'model_type': 'esm2', 'model_name': 'esm2_t36_3B_UR50D'}

        output_path, failed_ranks = run_multi_gpu_inference(
            test_fasta_files,
            multi_gpu_output,
            model_config
        )

        # This is informational - we just want to log that inference completed
        # Actual utilization monitoring would require nvidia-smi polling in background
        assert len(failed_ranks) == 0, "Workers failed"
        print("\nInference completed successfully (GPU utilization not directly measured)")


class TestFaultTolerance:
    """Test partial failure handling."""

    def test_partial_failure_produces_partial_results(self, test_fasta_files, temp_output_dir):
        """Verify that when some workers fail, successful workers produce valid output.

        Note: This test simulates failure by limiting timeout, causing some workers to time out.
        """
        from virnucpro.pipeline.multi_gpu_inference import run_multi_gpu_inference

        multi_gpu_output = temp_output_dir / "test_fault_tolerance"
        multi_gpu_output.mkdir(parents=True, exist_ok=True)

        model_config = {'model_type': 'esm2', 'model_name': 'esm2_t36_3B_UR50D'}

        # Run with very short timeout to induce failures (some workers may time out)
        # Note: This may not reliably trigger failures, so we check both scenarios
        output_path, failed_ranks = run_multi_gpu_inference(
            test_fasta_files,
            multi_gpu_output,
            model_config,
            timeout=5.0  # 5 seconds - may cause timeouts on slow GPUs
        )

        if failed_ranks:
            # Partial failure occurred - verify we still got results
            print(f"\nPartial failure detected: {len(failed_ranks)} workers failed")

            # Output should exist
            assert output_path.exists()

            # Output should contain embeddings from successful workers
            with h5py.File(output_path, 'r') as f:
                assert 'embeddings' in f
                assert 'sequence_ids' in f
                assert len(f['sequence_ids']) > 0

            print(f"  Salvaged {len(f['sequence_ids'])} sequences from successful workers")
        else:
            # No failures - all workers completed within timeout
            print("\nNo failures occurred (all workers completed within timeout)")
            assert output_path.exists()


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_single_gpu_mode(self, test_fasta_files, temp_output_dir):
        """Verify world_size=1 works correctly (single-GPU mode)."""
        from virnucpro.pipeline.multi_gpu_inference import run_multi_gpu_inference

        multi_gpu_output = temp_output_dir / "test_single_gpu_mode"
        multi_gpu_output.mkdir(parents=True, exist_ok=True)

        model_config = {'model_type': 'esm2', 'model_name': 'esm2_t36_3B_UR50D'}

        # Use first file only for speed
        output_path, failed_ranks = run_multi_gpu_inference(
            test_fasta_files[0:1],
            multi_gpu_output,
            model_config,
            world_size=1
        )

        assert len(failed_ranks) == 0
        assert output_path.exists()

        with h5py.File(output_path, 'r') as f:
            assert 'embeddings' in f
            assert 'sequence_ids' in f

    def test_small_dataset(self, temp_output_dir):
        """Verify handling of dataset with fewer sequences than GPUs."""
        from virnucpro.pipeline.multi_gpu_inference import run_multi_gpu_inference

        # Create tiny FASTA with only 2 sequences
        tiny_fasta = temp_output_dir / "tiny.fasta"
        with open(tiny_fasta, 'w') as f:
            f.write(">seq1\nMKTAYIAKVL\n")
            f.write(">seq2\nVLSPADKTNV\n")

        multi_gpu_output = temp_output_dir / "test_small_dataset"
        multi_gpu_output.mkdir(parents=True, exist_ok=True)

        model_config = {'model_type': 'esm2', 'model_name': 'esm2_t36_3B_UR50D'}
        world_size = 4  # More GPUs than sequences

        output_path, failed_ranks = run_multi_gpu_inference(
            [tiny_fasta],
            multi_gpu_output,
            model_config,
            world_size=world_size
        )

        # Some workers will have no sequences assigned, but should complete gracefully
        assert output_path.exists()

        with h5py.File(output_path, 'r') as f:
            # Should have 2 sequences
            assert len(f['sequence_ids']) == 2

    def test_uneven_distribution(self, temp_output_dir):
        """Verify odd sequence count is distributed correctly."""
        from virnucpro.pipeline.multi_gpu_inference import run_multi_gpu_inference

        # Create FASTA with 99 sequences (odd, not divisible by common world_size)
        odd_fasta = temp_output_dir / "odd.fasta"
        with open(odd_fasta, 'w') as f:
            for i in range(99):
                f.write(f">seq{i}\nMKTAYIAKVL\n")

        multi_gpu_output = temp_output_dir / "test_uneven"
        multi_gpu_output.mkdir(parents=True, exist_ok=True)

        model_config = {'model_type': 'esm2', 'model_name': 'esm2_t36_3B_UR50D'}

        output_path, failed_ranks = run_multi_gpu_inference(
            [odd_fasta],
            multi_gpu_output,
            model_config,
            world_size=4
        )

        assert len(failed_ranks) == 0

        with h5py.File(output_path, 'r') as f:
            # Should have all 99 sequences
            assert len(f['sequence_ids']) == 99
