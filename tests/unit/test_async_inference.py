"""Unit tests for AsyncInferenceRunner.

Tests cover:
- Stream synchronization (FIX 8 regression test)
- Embedding extraction correctness
- Batch processing consistency
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, Any, List


# Skip entire module if CUDA not available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required for AsyncInferenceRunner tests"
)


class TestStreamSynchronization:
    """Regression tests for stream synchronization (FIX 8).

    Background: A race condition existed where _extract_embeddings ran on
    the default stream before the compute stream finished. This caused:
    - First batch: correct embeddings
    - Subsequent batches: corrupted embeddings (100x higher norm)

    These tests verify the fix prevents regression.
    """

    @pytest.fixture
    def mock_model(self):
        """Create a mock ESM2 model that returns deterministic output."""
        model = Mock()
        model.eval = Mock(return_value=model)

        def mock_forward_packed(input_ids, cu_seqlens, max_seqlen, repr_layers):
            """Return deterministic representations based on input shape."""
            total_tokens = input_ids.shape[0]
            hidden_dim = 2560  # ESM-2 3B hidden dim

            # Create deterministic output - same input = same output
            # Use input_ids sum as seed for reproducibility
            torch.manual_seed(input_ids.sum().item())
            representations = torch.randn(
                total_tokens, hidden_dim,
                device=input_ids.device,
                dtype=torch.float32
            )

            return {'representations': {36: representations}}

        model.forward_packed = mock_forward_packed
        return model

    @pytest.fixture
    def sample_batches(self) -> List[Dict[str, Any]]:
        """Create 5 identical batches to test consistency."""
        # Same input_ids for all batches (simulates identical sequences)
        input_ids = torch.tensor([0, 20, 15, 11, 5, 5, 15, 7, 4, 2], dtype=torch.long)

        batches = []
        for i in range(5):
            batches.append({
                'input_ids': input_ids.clone(),
                'cu_seqlens': torch.tensor([0, 10], dtype=torch.int32),
                'max_seqlen': 10,
                'sequence_ids': [f'seq_{i}'],
                'num_sequences': 1,
            })
        return batches

    def test_embeddings_consistent_across_batches(self, mock_model, sample_batches):
        """Verify all batches produce identical embeddings for identical input.

        This is the core regression test for FIX 8. Before the fix:
        - Batch 0: norm ~13.89 (correct)
        - Batch 1-4: norm ~1685 (corrupted due to race condition)

        After the fix, all batches should have identical embeddings.
        """
        from virnucpro.pipeline.async_inference import AsyncInferenceRunner

        device = torch.device("cuda:0")
        runner = AsyncInferenceRunner(mock_model, device=device)

        embeddings = []
        for batch in sample_batches:
            result = runner.process_batch(batch)
            embeddings.append(result.embeddings.numpy())

        # All embeddings should be identical (same input = same output)
        reference = embeddings[0]
        for i, emb in enumerate(embeddings[1:], start=1):
            # Use allclose for numerical comparison
            assert np.allclose(reference, emb, rtol=1e-5, atol=1e-5), (
                f"Batch {i} embedding differs from batch 0. "
                f"Norms: batch0={np.linalg.norm(reference):.4f}, "
                f"batch{i}={np.linalg.norm(emb):.4f}. "
                "This may indicate a stream synchronization regression (FIX 8)."
            )

    def test_embedding_norms_stable(self, mock_model, sample_batches):
        """Verify embedding norms don't explode across batches.

        Before FIX 8, embedding norms grew ~100x after first batch due to
        reading uninitialized/in-progress GPU memory.
        """
        from virnucpro.pipeline.async_inference import AsyncInferenceRunner

        device = torch.device("cuda:0")
        runner = AsyncInferenceRunner(mock_model, device=device)

        norms = []
        for batch in sample_batches:
            result = runner.process_batch(batch)
            norm = np.linalg.norm(result.embeddings.numpy())
            norms.append(norm)

        # All norms should be within 2x of each other (generous tolerance)
        # Before fix: norms were 100x+ different
        min_norm, max_norm = min(norms), max(norms)
        ratio = max_norm / min_norm if min_norm > 0 else float('inf')

        assert ratio < 2.0, (
            f"Embedding norms vary too much across batches: "
            f"min={min_norm:.4f}, max={max_norm:.4f}, ratio={ratio:.2f}. "
            "Expected ratio < 2.0. This may indicate stream sync regression."
        )

    def test_process_batch_calls_synchronize(self, mock_model):
        """Verify process_batch synchronizes streams before extraction.

        This directly tests that the FIX 8 synchronization call is made.
        """
        from virnucpro.pipeline.async_inference import AsyncInferenceRunner

        device = torch.device("cuda:0")
        runner = AsyncInferenceRunner(mock_model, device=device)

        # Spy on the synchronize method
        original_sync = runner.stream_processor.synchronize
        sync_call_count = [0]

        def counting_sync():
            sync_call_count[0] += 1
            return original_sync()

        runner.stream_processor.synchronize = counting_sync

        # Process a batch
        batch = {
            'input_ids': torch.tensor([0, 20, 15, 11, 5, 2], dtype=torch.long),
            'cu_seqlens': torch.tensor([0, 6], dtype=torch.int32),
            'max_seqlen': 6,
            'sequence_ids': ['test_seq'],
            'num_sequences': 1,
        }
        runner.process_batch(batch)

        # Verify synchronize was called at least once
        assert sync_call_count[0] >= 1, (
            "stream_processor.synchronize() was not called during process_batch. "
            "FIX 8 requires synchronization before _extract_embeddings."
        )


class TestEmbeddingExtraction:
    """Tests for _extract_embeddings method."""

    @pytest.fixture
    def runner(self):
        """Create AsyncInferenceRunner with mock model."""
        from virnucpro.pipeline.async_inference import AsyncInferenceRunner

        mock_model = Mock()
        device = torch.device("cuda:0")
        return AsyncInferenceRunner(mock_model, device=device)

    def test_extract_embeddings_packed_format(self, runner):
        """Verify embedding extraction from packed format."""
        # Simulated packed representations: 2 sequences, 5 and 3 tokens each
        # Shape: [8, hidden_dim] for total_tokens=8
        hidden_dim = 128
        representations = torch.randn(8, hidden_dim, device=runner.device)

        gpu_batch = {
            'cu_seqlens': torch.tensor([0, 5, 8], dtype=torch.int32, device=runner.device),
            'sequence_ids': ['seq_a', 'seq_b'],
        }

        embeddings = runner._extract_embeddings(representations, gpu_batch)

        # Should return 2 embeddings (one per sequence)
        assert embeddings.shape == (2, hidden_dim)
        # Should be FP32 (FIX 7)
        assert embeddings.dtype == torch.float32

    def test_extract_embeddings_skips_bos(self, runner):
        """Verify BOS token (position 0) is skipped in mean pooling."""
        hidden_dim = 128

        # Create representations where BOS has distinct value
        representations = torch.zeros(10, hidden_dim, device=runner.device)
        representations[0] = 999.0  # BOS - should be skipped
        representations[1:5] = 1.0  # Actual sequence tokens

        gpu_batch = {
            'cu_seqlens': torch.tensor([0, 5], dtype=torch.int32, device=runner.device),
            'sequence_ids': ['seq_a'],
        }

        embeddings = runner._extract_embeddings(representations, gpu_batch)

        # Mean of positions 1-4 (value 1.0) should be 1.0, not influenced by BOS
        expected_mean = 1.0
        actual_mean = embeddings[0].mean().item()
        assert abs(actual_mean - expected_mean) < 0.01, (
            f"Expected mean ~{expected_mean}, got {actual_mean}. "
            "BOS token may not be properly skipped."
        )

    def test_extract_embeddings_single_sequence_fallback(self, runner):
        """Verify fallback path when cu_seqlens not present."""
        hidden_dim = 128

        # Standard 2D format: [batch=1, seq_len, hidden_dim]
        representations = torch.randn(1, 10, hidden_dim, device=runner.device)

        gpu_batch = {
            # No cu_seqlens - triggers fallback
            'sequence_ids': [],
        }

        embeddings = runner._extract_embeddings(representations, gpu_batch)

        # Should return single embedding
        assert embeddings.shape == (1, hidden_dim)
        assert embeddings.dtype == torch.float32
