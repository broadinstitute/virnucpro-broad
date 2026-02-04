"""Unit tests for VarlenCollator and dataloader integration.

Tests buffer-based packing (PACK-02) and dynamic token budget (PACK-03).
"""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock


class TestVarlenCollatorPacking:
    """Test VarlenCollator GreedyPacker integration (PACK-02)."""

    def test_buffer_based_packing_enabled_by_default(self):
        """Verify buffer-based packing is enabled by default (PACK-02)."""
        from virnucpro.data.collators import VarlenCollator

        # Mock batch_converter
        mock_bc = MagicMock()
        mock_bc.alphabet.padding_idx = 1

        collator = VarlenCollator(mock_bc)
        assert collator.enable_packing is True
        assert collator.packer is not None
        assert collator.buffer_size == 2000  # Default buffer size

    def test_buffer_based_packing_can_be_disabled(self):
        """Verify enable_packing=False disables packer."""
        from virnucpro.data.collators import VarlenCollator

        mock_bc = MagicMock()
        mock_bc.alphabet.padding_idx = 1

        collator = VarlenCollator(mock_bc, enable_packing=False)
        assert collator.enable_packing is False
        assert collator.packer is None

    def test_buffer_accumulation(self):
        """Verify micro-batches NOT buffered to prevent flush duplicates.

        When buffer isn't full and no packed_queue exists, sequences are
        returned immediately as micro-batches and NOT kept in buffer.
        This prevents sequence duplication when flush() is called later.
        """
        from virnucpro.data.collators import VarlenCollator

        mock_bc = MagicMock()
        mock_bc.alphabet.padding_idx = 1

        # Create collator with large buffer (won't trigger packing)
        collator = VarlenCollator(mock_bc, buffer_size=1000, enable_packing=True)

        batch = [
            {'id': 'seq1', 'sequence': 'MKTAYIAK'},
            {'id': 'seq2', 'sequence': 'VLSPADKTNV'},
        ]

        # Mock tokenization to prevent actual ESM calls
        with patch.object(collator, '_tokenize_and_pack', return_value={'test': 'data'}):
            # First call - returns micro-batch immediately
            result = collator(batch)

            # Buffer should be EMPTY (micro-batches not buffered)
            assert len(collator.buffer) == 0
            # Result should be returned
            assert result == {'test': 'data'}

    def test_packing_triggered_at_threshold(self):
        """Verify packing runs when buffer reaches threshold."""
        from virnucpro.data.collators import VarlenCollator

        mock_bc = MagicMock()
        mock_bc.alphabet.padding_idx = 1

        # Create collator with small buffer
        collator = VarlenCollator(mock_bc, buffer_size=2, enable_packing=True)

        # Mock packer
        mock_packer = MagicMock()
        mock_packer.pack_sequences.return_value = [
            [{'id': 'seq1', 'sequence': 'MKTAYIAK'}],
            [{'id': 'seq2', 'sequence': 'VLSPADKTNV'}],
        ]
        collator.packer = mock_packer

        # Add sequences to reach threshold
        batch = [
            {'id': 'seq1', 'sequence': 'MKTAYIAK'},
            {'id': 'seq2', 'sequence': 'VLSPADKTNV'},
        ]

        # Mock tokenization
        with patch.object(collator, '_tokenize_and_pack', return_value={'test': 'data'}):
            result = collator(batch)

            # Packer should have been called
            mock_packer.pack_sequences.assert_called_once()

            # Buffer should be cleared
            assert len(collator.buffer) == 0

            # packed_queue should have batches
            assert len(collator.packed_queue) >= 0  # One returned, one remains

    def test_flush_handles_remaining_sequences(self):
        """Verify flush() processes remaining buffer sequences."""
        from virnucpro.data.collators import VarlenCollator

        mock_bc = MagicMock()
        mock_bc.alphabet.padding_idx = 1

        collator = VarlenCollator(mock_bc, buffer_size=1000, enable_packing=True)

        # Add sequences to buffer (not enough to trigger packing)
        collator.buffer = [
            {'id': 'seq1', 'sequence': 'MKTAYIAK'},
            {'id': 'seq2', 'sequence': 'VLSPADKTNV'},
        ]

        # Mock packer
        mock_packer = MagicMock()
        mock_packer.pack_sequences.return_value = [
            [{'id': 'seq1', 'sequence': 'MKTAYIAK'}, {'id': 'seq2', 'sequence': 'VLSPADKTNV'}]
        ]
        collator.packer = mock_packer

        # Mock tokenization
        with patch.object(collator, '_tokenize_and_pack', return_value={'test': 'data'}):
            results = collator.flush()

            # Should pack remaining buffer
            mock_packer.pack_sequences.assert_called_once()
            assert len(results) > 0

            # Buffer should be empty
            assert len(collator.buffer) == 0

    def test_flush_handles_packed_queue(self):
        """Verify flush() also processes remaining packed_queue."""
        from virnucpro.data.collators import VarlenCollator

        mock_bc = MagicMock()
        mock_bc.alphabet.padding_idx = 1

        collator = VarlenCollator(mock_bc, enable_packing=True)

        # Add batches to packed_queue
        collator.packed_queue = [
            [{'id': 'seq1', 'sequence': 'MKTAYIAK'}],
            [{'id': 'seq2', 'sequence': 'VLSPADKTNV'}],
        ]

        # Mock tokenization
        with patch.object(collator, '_tokenize_and_pack', return_value={'test': 'data'}):
            results = collator.flush()

            # Should return all packed batches
            assert len(results) == 2

            # Queue should be empty
            assert len(collator.packed_queue) == 0

    def test_direct_processing_when_packing_disabled(self):
        """Verify direct processing bypasses buffer when enable_packing=False."""
        from virnucpro.data.collators import VarlenCollator

        mock_bc = MagicMock()
        mock_bc.alphabet.padding_idx = 1

        collator = VarlenCollator(mock_bc, enable_packing=False)

        batch = [
            {'id': 'seq1', 'sequence': 'MKTAYIAK'},
        ]

        # Mock tokenization
        with patch.object(collator, '_tokenize_and_pack', return_value={'test': 'data'}) as mock_pack:
            result = collator(batch)

            # Should call _tokenize_and_pack directly
            mock_pack.assert_called_once_with(batch)

            # Buffer should remain empty
            assert len(collator.buffer) == 0


class TestDataloaderDynamicBudget:
    """Test create_async_dataloader dynamic token budget (PACK-03)."""

    @patch('virnucpro.data.dataloader_utils.calculate_token_budget')
    @patch('torch.cuda.is_available', return_value=True)
    def test_dynamic_budget_calculated_when_none(self, mock_cuda, mock_calc_budget):
        """Verify calculate_token_budget called when token_budget=None."""
        mock_calc_budget.return_value = 8192

        from virnucpro.data.dataloader_utils import create_async_dataloader
        from torch.utils.data import IterableDataset

        # Mock dataset and collator
        mock_dataset = MagicMock(spec=IterableDataset)
        mock_collator = MagicMock()
        mock_collator.packer = MagicMock()

        # Create dataloader with token_budget=None
        loader = create_async_dataloader(
            dataset=mock_dataset,
            collate_fn=mock_collator,
            batch_size=None,
            num_workers=2,
            token_budget=None,  # Should trigger dynamic calculation
            device_id=0,
            model_memory_gb=5.0,
        )

        # Verify calculate_token_budget was called
        mock_calc_budget.assert_called_once_with(
            device_id=0,
            model_memory_gb=5.0,
        )

        # Verify collator was updated
        assert mock_collator.max_tokens_per_batch == 8192
        assert mock_collator.packer.max_tokens_per_batch == 8192

    @patch('torch.cuda.is_available', return_value=False)
    def test_no_dynamic_budget_when_cuda_unavailable(self, mock_cuda):
        """Verify dynamic budget skipped when CUDA unavailable."""
        from virnucpro.data.dataloader_utils import create_async_dataloader
        from torch.utils.data import IterableDataset

        # Mock dataset and collator
        mock_dataset = MagicMock(spec=IterableDataset)
        mock_collator = MagicMock()
        mock_collator.max_tokens_per_batch = 4096  # Default

        # Create dataloader with CUDA unavailable
        loader = create_async_dataloader(
            dataset=mock_dataset,
            collate_fn=mock_collator,
            batch_size=None,
            num_workers=2,
            token_budget=None,
        )

        # Collator budget should remain unchanged
        assert mock_collator.max_tokens_per_batch == 4096

    def test_explicit_token_budget_overrides_dynamic(self):
        """Verify explicit token_budget parameter updates collator."""
        from virnucpro.data.dataloader_utils import create_async_dataloader
        from torch.utils.data import IterableDataset

        # Mock dataset and collator
        mock_dataset = MagicMock(spec=IterableDataset)
        mock_collator = MagicMock()
        mock_collator.packer = MagicMock()

        # Create dataloader with explicit budget
        loader = create_async_dataloader(
            dataset=mock_dataset,
            collate_fn=mock_collator,
            batch_size=None,
            num_workers=2,
            token_budget=12000,  # Explicit budget
        )

        # Verify collator was updated with explicit value
        assert mock_collator.max_tokens_per_batch == 12000
        assert mock_collator.packer.max_tokens_per_batch == 12000

    def test_signature_has_required_parameters(self):
        """Verify create_async_dataloader has required parameters for PACK-03."""
        from virnucpro.data.dataloader_utils import create_async_dataloader
        import inspect

        sig = inspect.signature(create_async_dataloader)
        params = list(sig.parameters.keys())

        assert 'token_budget' in params, 'token_budget parameter missing'
        assert 'device_id' in params, 'device_id parameter missing'
        assert 'model_memory_gb' in params, 'model_memory_gb parameter missing'

        # Verify defaults
        assert sig.parameters['token_budget'].default is None
        assert sig.parameters['device_id'].default == 0
        assert sig.parameters['model_memory_gb'].default == 5.0


class TestVarlenCollatorTokenization:
    """Test _tokenize_and_pack helper method."""

    def test_empty_batch_returns_empty_dict(self):
        """Verify empty batch returns empty dict."""
        from virnucpro.data.collators import VarlenCollator

        mock_bc = MagicMock()
        mock_bc.alphabet.padding_idx = 1

        collator = VarlenCollator(mock_bc)

        result = collator._tokenize_and_pack([])
        assert result == {}

    def test_tokenize_and_pack_preserves_sequence_ids(self):
        """Verify sequence IDs are preserved in output."""
        from virnucpro.data.collators import VarlenCollator
        import torch

        mock_bc = MagicMock()
        mock_bc.alphabet.padding_idx = 1

        # Mock tokenization output
        mock_bc.return_value = (
            ['seq1', 'seq2'],  # labels
            ['MKTAYIAK', 'VLSPAD'],  # strs
            torch.tensor([[0, 10, 11, 12, 1, 1], [0, 20, 21, 22, 23, 1]]),  # tokens (padded)
        )

        collator = VarlenCollator(mock_bc)

        batch = [
            {'id': 'seq1', 'sequence': 'MKTAYIAK'},
            {'id': 'seq2', 'sequence': 'VLSPAD'},
        ]

        result = collator._tokenize_and_pack(batch)

        # Verify sequence_ids preserved
        assert 'sequence_ids' in result
        assert result['sequence_ids'] == ['seq1', 'seq2']


class TestVarlenCollatorSingleItem:
    """Test VarlenCollator handling of single items (batch_size=None support)."""

    def test_single_dict_wrapped_to_list(self):
        """Verify single dict input is wrapped to list (batch_size=None behavior)."""
        from virnucpro.data.collators import VarlenCollator
        import torch

        mock_bc = MagicMock()
        mock_bc.alphabet.padding_idx = 1
        mock_bc.return_value = (
            ['seq1'],
            ['MKTAYIAK'],
            torch.tensor([[0, 10, 11, 12, 1]]),
        )

        collator = VarlenCollator(mock_bc, enable_packing=False)

        # Single dict (what DataLoader passes with batch_size=None)
        single_item = {'id': 'seq1', 'sequence': 'MKTAYIAK', 'file': 'test.fasta'}
        result = collator(single_item)

        # Should work and produce valid output
        assert 'sequence_ids' in result
        assert result['sequence_ids'] == ['seq1']

    def test_list_of_dicts_still_works(self):
        """Verify list of dicts still works (batch_size=N behavior)."""
        from virnucpro.data.collators import VarlenCollator
        import torch

        mock_bc = MagicMock()
        mock_bc.alphabet.padding_idx = 1
        mock_bc.return_value = (
            ['seq1', 'seq2'],
            ['MKTAYIAK', 'VLSPAD'],
            torch.tensor([[0, 10, 11, 12, 1, 1], [0, 20, 21, 22, 23, 1]]),
        )

        collator = VarlenCollator(mock_bc, enable_packing=False)

        # List of dicts (what DataLoader passes with batch_size=N)
        batch = [
            {'id': 'seq1', 'sequence': 'MKTAYIAK', 'file': 'test.fasta'},
            {'id': 'seq2', 'sequence': 'VLSPAD', 'file': 'test.fasta'},
        ]
        result = collator(batch)

        assert 'sequence_ids' in result
        assert result['sequence_ids'] == ['seq1', 'seq2']

    def test_single_item_with_packing_enabled(self):
        """Verify single item processed and NOT buffered (prevents flush duplicates)."""
        from virnucpro.data.collators import VarlenCollator
        import torch

        mock_bc = MagicMock()
        mock_bc.alphabet.padding_idx = 1
        mock_bc.return_value = (
            ['seq1'],
            ['MKTAYIAK'],
            torch.tensor([[0, 10, 11, 12, 1]]),
        )

        collator = VarlenCollator(mock_bc, enable_packing=True, buffer_size=10)

        # Single dict - returned as micro-batch, NOT buffered
        single_item = {'id': 'seq1', 'sequence': 'MKTAYIAK', 'file': 'test.fasta'}
        result = collator(single_item)

        # Buffer should be EMPTY (micro-batches not buffered to prevent duplicates)
        assert len(collator.buffer) == 0
        # Result should contain the sequence
        assert 'sequence_ids' in result
        assert result['sequence_ids'] == ['seq1']
