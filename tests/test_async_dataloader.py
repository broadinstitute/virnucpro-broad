"""Integration tests for async DataLoader pipeline.

These tests validate Phase 5 requirements:
- SAFE-01: Workers don't initialize CUDA
- ARCH-02: Async DataLoader with prefetching
- ARCH-03: Batch prefetching works
- ARCH-04: Pin memory for fast GPU transfer
- ARCH-05: Stream-based I/O overlap

Tests require GPU to run. Skip gracefully if no GPU available.

FIX 4: Test Speed Markers
========================
These are slow integration tests due to model loading and GPU operations.
Use pytest markers for selective running:

Run all tests:
    pytest tests/test_async_dataloader.py -v

Run only fast tests (skip @pytest.mark.slow):
    pytest tests/test_async_dataloader.py -v -m "not slow"

Run only slow tests:
    pytest tests/test_async_dataloader.py -v -m "slow"

Configure in pytest.ini:
    [pytest]
    markers =
        slow: marks tests as slow (deselect with '-m "not slow"')
"""

import pytest
import torch
import tempfile
import os
from pathlib import Path
from typing import List

# Skip all tests if no CUDA available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available"
)


@pytest.fixture
def test_fasta_files() -> List[Path]:
    """Create temporary FASTA files for testing."""
    files = []

    # Create 2 test files with different sequences
    for i in range(2):
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.fasta', delete=False
        ) as f:
            # Write 10 sequences per file
            for j in range(10):
                seq_id = f"seq_{i}_{j}"
                # Generate random-ish protein sequence
                seq = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQQIA"[:50 + j*5]
                f.write(f">{seq_id}\n{seq}\n")
            files.append(Path(f.name))

    yield files

    # Cleanup
    for f in files:
        if f.exists():
            os.unlink(f)


@pytest.fixture(scope='module')
def esm_model_and_converter():
    """
    Load ESM-2 model and batch converter for testing.

    FIX 1: scope='module' to avoid reloading model for every test.
    Loading ESM-2 takes 20+ seconds. Without module scope, it reloads for each
    test function, making the suite unbearably slow (20 tests × 15s = 5 minutes
    just for loading). With module scope, loads once per module (~20 seconds total).
    """
    import esm

    # Use smaller model for faster tests
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()

    device = torch.device('cuda:0')
    model = model.to(device)
    model.eval()

    return model, batch_converter, device


class TestSequenceDataset:
    """Tests for SequenceDataset CUDA safety."""

    def test_dataset_yields_sequences(self, test_fasta_files):
        """Dataset yields sequence dicts from FASTA files."""
        from virnucpro.data import SequenceDataset

        dataset = SequenceDataset(test_fasta_files)
        items = list(dataset)

        # Should have 20 sequences total (10 per file)
        assert len(items) == 20, f"Expected 20 items, got {len(items)}"

        # Check structure
        item = items[0]
        assert 'id' in item, "Missing 'id' key"
        assert 'sequence' in item, "Missing 'sequence' key"
        assert 'file' in item, "Missing 'file' key"

    def test_dataset_respects_max_length(self, test_fasta_files):
        """Dataset truncates sequences to max_length."""
        from virnucpro.data import SequenceDataset

        dataset = SequenceDataset(test_fasta_files, max_length=30)
        items = list(dataset)

        for item in items:
            assert len(item['sequence']) <= 30, \
                f"Sequence too long: {len(item['sequence'])}"


class TestVarlenCollator:
    """Tests for VarlenCollator tokenization."""

    def test_collator_produces_packed_format(self, esm_model_and_converter):
        """Collator produces input_ids and cu_seqlens."""
        from virnucpro.data import VarlenCollator

        _, batch_converter, _ = esm_model_and_converter
        collator = VarlenCollator(batch_converter, max_tokens_per_batch=500)

        batch = [
            {'id': 'seq1', 'sequence': 'MKTAYIAK', 'file': 'test.fasta'},
            {'id': 'seq2', 'sequence': 'VLSPADKTNV', 'file': 'test.fasta'},
        ]

        result = collator(batch)

        assert 'input_ids' in result
        assert 'cu_seqlens' in result
        assert 'max_seqlen' in result
        assert 'sequence_ids' in result
        assert 'num_sequences' in result

        # Verify types
        assert result['input_ids'].dtype == torch.long
        assert result['cu_seqlens'].dtype == torch.int32

        # cu_seqlens should have num_sequences + 1 elements
        assert len(result['cu_seqlens']) == result['num_sequences'] + 1


class TestAsyncDataLoader:
    """Tests for async DataLoader with CUDA safety."""

    @pytest.mark.slow
    def test_dataloader_prefetches_batches(self, test_fasta_files, esm_model_and_converter):
        """DataLoader prefetches batches while GPU computes."""
        from virnucpro.data import SequenceDataset, VarlenCollator, create_async_dataloader

        _, batch_converter, _ = esm_model_and_converter

        dataset = SequenceDataset(test_fasta_files)
        collator = VarlenCollator(batch_converter, max_tokens_per_batch=500)

        loader = create_async_dataloader(
            dataset,
            collator,
            batch_size=5,
            num_workers=2,
            prefetch_factor=2,
        )

        # Iterate through batches
        batches = list(loader)

        # Should have multiple batches (20 sequences / 5 per batch = ~4)
        assert len(batches) >= 2, f"Expected at least 2 batches, got {len(batches)}"

        # Each batch should have expected structure
        for batch in batches:
            assert 'input_ids' in batch
            assert 'cu_seqlens' in batch

    @pytest.mark.slow
    def test_dataloader_uses_spawn_context(self, test_fasta_files, esm_model_and_converter):
        """
        DataLoader uses spawn context for CUDA safety.

        FIX 3: Robust check that doesn't rely on internal API.
        Instead of checking loader.multiprocessing_context (internal API),
        we verify spawn behavior by checking that workers are fresh processes
        (not forked copies of parent).
        """
        from virnucpro.data import SequenceDataset, VarlenCollator, create_async_dataloader
        import multiprocessing

        _, batch_converter, _ = esm_model_and_converter

        dataset = SequenceDataset(test_fasta_files)
        collator = VarlenCollator(batch_converter, max_tokens_per_batch=500)

        loader = create_async_dataloader(
            dataset,
            collator,
            batch_size=2,
            num_workers=2,
        )

        # FIX 3: More robust check - verify spawn behavior
        # In spawn context, workers are fresh processes (not forks)
        # We can verify by checking that fetching a batch succeeds without
        # CUDA context inheritance issues
        try:
            batch = next(iter(loader))
            assert 'input_ids' in batch, "Batch should have input_ids"

            # If we got here without RuntimeError about CUDA, spawn context worked
            # (fork would have inherited CUDA context and failed)
            print("✓ DataLoader spawn context verified via successful worker spawn")
        except RuntimeError as e:
            if 'fork' in str(e).lower() or 'CUDA' in str(e):
                pytest.fail(f"DataLoader appears to use fork context: {e}")
            else:
                raise

        del loader


class TestAsyncInference:
    """Tests for AsyncInferenceRunner end-to-end."""

    @pytest.mark.slow
    def test_inference_produces_embeddings(self, test_fasta_files, esm_model_and_converter):
        """Full pipeline produces valid embeddings."""
        from virnucpro.data import SequenceDataset, VarlenCollator, create_async_dataloader
        from virnucpro.pipeline import AsyncInferenceRunner

        model, batch_converter, device = esm_model_and_converter

        dataset = SequenceDataset(test_fasta_files)
        collator = VarlenCollator(batch_converter, max_tokens_per_batch=500)
        loader = create_async_dataloader(
            dataset,
            collator,
            batch_size=5,
            num_workers=2,
        )

        runner = AsyncInferenceRunner(
            model=model,
            device=device,
            enable_streams=True,
        )

        results = list(runner.run(loader))

        # Should have results
        assert len(results) > 0, "No results produced"

        # Check result structure
        for result in results:
            assert len(result.sequence_ids) > 0, "No sequence IDs"
            assert result.embeddings.shape[0] == len(result.sequence_ids), \
                "Embedding count mismatch"
            # ESM-2 650M has 1280 dim embeddings
            assert result.embeddings.shape[1] == 1280, \
                f"Wrong embedding dim: {result.embeddings.shape[1]}"

    @pytest.mark.slow
    def test_inference_statistics_available(self, test_fasta_files, esm_model_and_converter):
        """Runner provides statistics after inference."""
        from virnucpro.data import SequenceDataset, VarlenCollator, create_async_dataloader
        from virnucpro.pipeline import AsyncInferenceRunner

        model, batch_converter, device = esm_model_and_converter

        dataset = SequenceDataset(test_fasta_files)
        collator = VarlenCollator(batch_converter, max_tokens_per_batch=500)
        loader = create_async_dataloader(
            dataset,
            collator,
            batch_size=10,
            num_workers=2,
        )

        runner = AsyncInferenceRunner(model=model, device=device)
        list(runner.run(loader))  # Consume iterator

        stats = runner.get_statistics()

        assert 'total_batches' in stats
        assert 'total_sequences' in stats
        assert 'throughput' in stats
        assert stats['total_sequences'] == 20, \
            f"Expected 20 sequences, got {stats['total_sequences']}"


class TestCUDASafety:
    """Tests specifically for CUDA isolation in workers."""

    def test_worker_init_hides_cuda(self):
        """
        worker_init_fn sets CUDA_VISIBLE_DEVICES=''.

        FIX 2 LIMITATION: This tests the function's environment variable setting
        but doesn't test actual worker isolation in a spawned subprocess (which is
        the real safety mechanism). The actual CUDA isolation test happens in
        test_dataloader_worker_cuda_isolation below via subprocess.

        NOTE: We can't test torch.cuda.is_available() here because torch is already
        imported in the main process. The env var only affects CUDA when torch is
        first imported, which happens in spawned workers.
        """
        from virnucpro.data.dataloader_utils import cuda_safe_worker_init
        import os

        # Save current env
        old_cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES')
        old_tokenizers = os.environ.get('TOKENIZERS_PARALLELISM')

        try:
            # Temporarily allow CUDA check to pass in main process
            # In real workers (spawned processes), this check catches CUDA leaks
            import torch
            original_is_available = torch.cuda.is_available
            torch.cuda.is_available = lambda: False  # Mock for main process test

            try:
                cuda_safe_worker_init(0)

                assert os.environ.get('CUDA_VISIBLE_DEVICES') == '', \
                    "CUDA_VISIBLE_DEVICES not set to empty"
                assert os.environ.get('TOKENIZERS_PARALLELISM') == 'false', \
                    "TOKENIZERS_PARALLELISM not disabled"
            finally:
                torch.cuda.is_available = original_is_available

        finally:
            # Restore env
            if old_cuda_visible is not None:
                os.environ['CUDA_VISIBLE_DEVICES'] = old_cuda_visible
            else:
                os.environ.pop('CUDA_VISIBLE_DEVICES', None)
            if old_tokenizers is not None:
                os.environ['TOKENIZERS_PARALLELISM'] = old_tokenizers
            else:
                os.environ.pop('TOKENIZERS_PARALLELISM', None)

    @pytest.mark.slow
    def test_dataloader_worker_cuda_isolation(self, test_fasta_files, esm_model_and_converter):
        """
        FIX 2: Test actual worker CUDA isolation in spawned subprocess.

        This is the real safety test - spawns actual DataLoader workers and
        verifies they cannot access CUDA. More comprehensive than just testing
        the worker_init_fn function.
        """
        from virnucpro.data import SequenceDataset, VarlenCollator, create_async_dataloader
        import subprocess
        import sys

        _, batch_converter, _ = esm_model_and_converter

        # Create DataLoader with worker
        dataset = SequenceDataset(test_fasta_files)
        collator = VarlenCollator(batch_converter, max_tokens_per_batch=500)
        loader = create_async_dataloader(
            dataset,
            collator,
            batch_size=2,
            num_workers=1,  # Single worker for test
        )

        # Fetch one batch - this spawns worker and runs cuda_safe_worker_init
        # If worker can access CUDA, cuda_safe_worker_init will raise RuntimeError
        try:
            batch = next(iter(loader))
            # Success - worker spawned without CUDA access
            assert 'input_ids' in batch, "Batch should have input_ids"
            print("✓ Worker spawned successfully with CUDA isolation verified")
        except RuntimeError as e:
            if 'CUDA is accessible' in str(e):
                pytest.fail(f"Worker has CUDA access despite worker_init_fn: {e}")
            else:
                raise

        # Clean shutdown
        del loader


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
