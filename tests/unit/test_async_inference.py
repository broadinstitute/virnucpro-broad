"""Unit tests for AsyncInferenceRunner.

Tests cover:
- Stream synchronization (FIX 8 regression test)
- Embedding extraction correctness
- Batch processing consistency
"""

import pytest
import torch
import numpy as np
import logging
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path
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

        mock_param = Mock()
        mock_param.dtype = torch.float32
        mock_param.numel.return_value = 1000000
        model.parameters.return_value = [mock_param]

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
        mock_param = Mock()
        mock_param.dtype = torch.float32
        mock_param.numel.return_value = 1000000
        mock_model.parameters.return_value = [mock_param]
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


class TestRankValidation:
    """Tests for rank parameter validation."""

    def test_negative_rank_raises_value_error(self):
        """Verify ValueError is raised for negative rank values."""
        from virnucpro.pipeline.async_inference import AsyncInferenceRunner

        mock_model = Mock()
        device = torch.device("cuda:0")

        with pytest.raises(ValueError, match="rank must be non-negative"):
            AsyncInferenceRunner(mock_model, device=device, rank=-1)

    def test_negative_rank_with_checkpoint_dir_raises_value_error(self):
        """Verify ValueError is raised for negative rank even with checkpointing enabled."""
        from virnucpro.pipeline.async_inference import AsyncInferenceRunner
        import tempfile
        from pathlib import Path

        mock_model = Mock()
        device = torch.device("cuda:0")

        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="rank must be non-negative"):
                AsyncInferenceRunner(
                    mock_model,
                    device=device,
                    rank=-1,
                    checkpoint_dir=Path(tmpdir)
                )

    def test_valid_rank_zero(self):
        """Verify rank=0 is accepted (valid default)."""
        from virnucpro.pipeline.async_inference import AsyncInferenceRunner

        mock_model = Mock()
        mock_param = Mock()
        mock_param.dtype = torch.float32
        mock_param.numel.return_value = 1000000
        mock_model.parameters.return_value = iter([mock_param])
        device = torch.device("cuda:0")

        runner = AsyncInferenceRunner(mock_model, device=device, rank=0)
        assert runner.rank == 0

    def test_valid_positive_rank(self):
        """Verify positive rank values are accepted."""
        from virnucpro.pipeline.async_inference import AsyncInferenceRunner

        mock_model = Mock()
        mock_param = Mock()
        mock_param.dtype = torch.float32
        mock_param.numel.return_value = 1000000
        mock_model.parameters.return_value = iter([mock_param])
        device = torch.device("cuda:0")

        runner = AsyncInferenceRunner(mock_model, device=device, rank=2)
        assert runner.rank == 2


class TestModelConfigHash:
    """Tests for _compute_model_config_hash validation."""

    def test_empty_model_raises_value_error(self):
        """Verify ValueError is raised when model has no parameters."""
        from virnucpro.pipeline.async_inference import AsyncInferenceRunner

        mock_model = Mock()
        mock_model.parameters.return_value = iter([])
        device = torch.device("cuda:0")

        with pytest.raises(ValueError, match="Model has no parameters"):
            AsyncInferenceRunner(mock_model, device=device)

    def test_model_with_parameters_returns_hash(self):
        """Verify hash is computed correctly for models with parameters."""
        from virnucpro.pipeline.async_inference import AsyncInferenceRunner

        mock_param = Mock()
        mock_param.dtype = torch.float32
        mock_param.numel.return_value = 1000000

        mock_model = Mock()
        mock_model.parameters.return_value = [mock_param]

        device = torch.device("cuda:0")
        runner = AsyncInferenceRunner(mock_model, device=device)

        hash_value = runner._compute_model_config_hash()

        assert isinstance(hash_value, str)
        assert len(hash_value) == 16


class TestCheckpointDirectoryStructure:
    """Tests for checkpoint directory structure and path correctness.

    Regression tests for double-nesting bug where checkpoints were written
    to checkpoints/shard_0/shard_0/ instead of checkpoints/shard_0/.
    """

    @pytest.fixture
    def mock_checkpoint_base_dir(self, tmp_path):
        """Create temporary checkpoint base directory."""
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()
        return checkpoint_dir

    def test_shard_subdirectory_created_correctly(self, mock_checkpoint_base_dir):
        """Verify shard subdirectory is created at checkpoint_dir/shard_{rank}/, not nested."""
        from virnucpro.pipeline.async_inference import AsyncInferenceRunner

        mock_model = Mock()
        mock_param = Mock()
        mock_param.dtype = torch.float32
        mock_param.numel.return_value = 1000000
        mock_model.parameters.return_value = [mock_param]

        device = torch.device("cuda:0")
        rank = 0

        runner = AsyncInferenceRunner(
            mock_model,
            device=device,
            checkpoint_dir=mock_checkpoint_base_dir,
            rank=rank
        )

        expected_shard_dir = mock_checkpoint_base_dir / f"shard_{rank}"
        double_nested_dir = mock_checkpoint_base_dir / f"shard_{rank}" / f"shard_{rank}"

        assert runner.shard_checkpoint_dir == expected_shard_dir
        assert expected_shard_dir.exists(), f"Shard directory should exist at {expected_shard_dir}"
        assert not double_nested_dir.exists(), (
            f"Double-nested directory found at {double_nested_dir}. "
            "Bug: checkpoints/shard_N/shard_N/ instead of checkpoints/shard_N/"
        )

    def test_checkpoint_path_no_double_nesting_rank_0(self, tmp_path):
        """Verify rank=0 checkpoint path has no double nesting."""
        from virnucpro.pipeline.async_inference import AsyncInferenceRunner

        checkpoint_base = tmp_path / "checkpoints"
        checkpoint_base.mkdir()

        mock_model = Mock()
        mock_param = Mock()
        mock_param.dtype = torch.float32
        mock_param.numel.return_value = 1000000
        mock_model.parameters.return_value = [mock_param]

        device = torch.device("cuda:0")
        runner = AsyncInferenceRunner(mock_model, device, checkpoint_dir=checkpoint_base, rank=0)

        expected = checkpoint_base / "shard_0"
        actual = runner.shard_checkpoint_dir

        assert actual == expected, (
            f"Checkpoint dir should be {expected}, got {actual}"
        )
        assert (checkpoint_base / "shard_0").exists()
        assert not (checkpoint_base / "shard_0" / "shard_0").exists()

    def test_checkpoint_path_no_double_nesting_rank_2(self, tmp_path):
        """Verify rank=2 checkpoint path has no double nesting."""
        from virnucpro.pipeline.async_inference import AsyncInferenceRunner

        checkpoint_base = tmp_path / "checkpoints"
        checkpoint_base.mkdir()

        mock_model = Mock()
        mock_param = Mock()
        mock_param.dtype = torch.float32
        mock_param.numel.return_value = 1000000
        mock_model.parameters.return_value = [mock_param]

        device = torch.device("cuda:0")
        runner = AsyncInferenceRunner(mock_model, device, checkpoint_dir=checkpoint_base, rank=2)

        expected = checkpoint_base / "shard_2"
        actual = runner.shard_checkpoint_dir

        assert actual == expected, (
            f"Checkpoint dir should be {expected}, got {actual}"
        )
        assert (checkpoint_base / "shard_2").exists()
        assert not (checkpoint_base / "shard_2" / "shard_2").exists()

    def test_shard_subdirectory_created_once_per_runner(self, mock_checkpoint_base_dir):
        """Verify only one shard directory is created per runner instance."""
        from virnucpro.pipeline.async_inference import AsyncInferenceRunner

        mock_model = Mock()
        mock_param = Mock()
        mock_param.dtype = torch.float32
        mock_param.numel.return_value = 1000000
        mock_model.parameters.return_value = [mock_param]

        device = torch.device("cuda:0")
        rank = 1

        runner = AsyncInferenceRunner(
            mock_model,
            device=device,
            checkpoint_dir=mock_checkpoint_base_dir,
            rank=rank
        )

        shard_dirs = list(mock_checkpoint_base_dir.iterdir())
        shard_subdir_count = sum(1 for d in shard_dirs if d.is_dir() and d.name.startswith("shard_"))

        assert shard_subdir_count == 1, (
            f"Expected 1 shard directory, found {shard_subdir_count}. "
            f"Directories: {[d.name for d in shard_dirs]}"
        )
        assert runner.shard_checkpoint_dir is not None
        assert runner.shard_checkpoint_dir.name == f"shard_{rank}"

    def test_checkpoint_dir_none_disables_checkpointing(self):
        """Verify checkpoint_dir=None disables checkpointing correctly."""
        from virnucpro.pipeline.async_inference import AsyncInferenceRunner

        mock_model = Mock()
        mock_param = Mock()
        mock_param.dtype = torch.float32
        mock_param.numel.return_value = 1000000
        mock_model.parameters.return_value = [mock_param]

        device = torch.device("cuda:0")
        runner = AsyncInferenceRunner(mock_model, device, checkpoint_dir=None, rank=0)

        assert runner.checkpoint_dir is None
        assert runner.shard_checkpoint_dir is None
        assert not runner._checkpointing_enabled

    def test_multiple_ranks_create_separate_directories(self, tmp_path):
        """Verify ranks 0, 1, 2 each get their own separate directory."""
        from virnucpro.pipeline.async_inference import AsyncInferenceRunner

        checkpoint_base = tmp_path / "checkpoints"
        checkpoint_base.mkdir()

        mock_model = Mock()
        mock_param = Mock()
        mock_param.dtype = torch.float32
        mock_param.numel.return_value = 1000000
        mock_model.parameters.return_value = [mock_param]

        device = torch.device("cuda:0")

        runners = []
        for rank in range(3):
            runner = AsyncInferenceRunner(
                mock_model,
                device,
                checkpoint_dir=checkpoint_base,
                rank=rank
            )
            runners.append(runner)

        expected_dirs = {
            0: checkpoint_base / "shard_0",
            1: checkpoint_base / "shard_1",
            2: checkpoint_base / "shard_2",
        }

        for rank, expected in expected_dirs.items():
            actual = runners[rank].shard_checkpoint_dir
            assert actual == expected, (
                f"Rank {rank}: expected {expected}, got {actual}"
            )
            assert expected.exists(), f"Rank {rank} directory should exist at {expected}"

        for rank in range(3):
            for other_rank in range(3):
                if rank != other_rank:
                    assert runners[rank].shard_checkpoint_dir != runners[other_rank].shard_checkpoint_dir, (
                        f"Ranks {rank} and {other_rank} should have different directories"
                    )

    def test_checkpoint_writes_to_correct_shard_directory(self, tmp_path):
        """Verify actual checkpoint writes use shard_checkpoint_dir path."""
        from virnucpro.pipeline.async_inference import AsyncInferenceRunner

        checkpoint_base = tmp_path / "checkpoints"
        checkpoint_base.mkdir()

        mock_model = Mock()
        mock_param = Mock()
        mock_param.dtype = torch.float32
        mock_param.numel.return_value = 1000000
        mock_param.device = torch.device("cuda:0")
        mock_model.parameters.side_effect = lambda: iter([mock_param])

        device = torch.device("cuda:0")
        runner = AsyncInferenceRunner(mock_model, device, checkpoint_dir=checkpoint_base, rank=0)

        runner._ckpt_embeddings = [torch.randn(10, 768)]
        runner._ckpt_ids = ["seq_0", "seq_1", "seq_2"]
        runner._ckpt_batch_idx = 0

        runner._write_checkpoint("test_trigger")
        assert runner.writer is not None
        if runner.writer.executor is not None:
            try:
                runner.writer.executor.shutdown(wait=True)
            except Exception as e:
                logging.debug(f"Executor shutdown raised: {e}")

        assert runner.shard_checkpoint_dir is not None
        expected_ckpt = runner.shard_checkpoint_dir / "batch_00000.pt"
        assert expected_ckpt.exists(), f"Checkpoint should exist at {expected_ckpt}"

        checkpoint_data = torch.load(expected_ckpt, weights_only=False, map_location='cpu')
        assert len(checkpoint_data["embeddings"]) == 10
        assert str(checkpoint_data["embeddings"].dtype) == 'float32'
        assert checkpoint_data["sequence_ids"] == ["seq_0", "seq_1", "seq_2"]

        double_nested = checkpoint_base / "shard_0" / "shard_0" / "batch_00000.pt"
        assert not double_nested.exists(), f"Double-nested checkpoint found at {double_nested}"

    def test_model_parameter_access_pattern(self, tmp_path):
        """Verify correct parameter access when multiple parameters exist."""
        from virnucpro.pipeline.async_inference import AsyncInferenceRunner

        checkpoint_base = tmp_path / "checkpoints"
        checkpoint_base.mkdir()

        mock_model = Mock()
        mock_param1 = Mock()
        mock_param1.dtype = torch.float32
        mock_param1.numel.return_value = 1000
        mock_param2 = Mock()
        mock_param2.dtype = torch.float16
        mock_param2.numel.return_value = 2000
        mock_model.parameters.return_value = [mock_param1, mock_param2]

        device = torch.device("cuda:0")
        runner = AsyncInferenceRunner(mock_model, device, checkpoint_dir=checkpoint_base, rank=0)

        mock_model.parameters.assert_called_once()
        assert mock_param1.dtype == torch.float32
        assert mock_param1.numel() == 1000
        assert mock_param2.dtype == torch.float16
        assert mock_param2.numel() == 2000

    def test_checkpoint_dir_relative_path(self, tmp_path, monkeypatch):
        """Verify checkpoint_dir works with relative paths converted to absolute."""
        from virnucpro.pipeline.async_inference import AsyncInferenceRunner

        mock_model = Mock()
        mock_param = Mock()
        mock_param.dtype = torch.float32
        mock_param.numel.return_value = 1000000
        mock_model.parameters.return_value = [mock_param]

        monkeypatch.chdir(tmp_path)
        rel_path = Path("relative_checkpoints")
        runner = AsyncInferenceRunner(mock_model, torch.device("cuda:0"), checkpoint_dir=rel_path, rank=0)
        assert runner.shard_checkpoint_dir is not None
        assert runner.shard_checkpoint_dir.name == "shard_0"
        assert (tmp_path / "relative_checkpoints" / "shard_0").exists()
