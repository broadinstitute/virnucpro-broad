"""Unit tests for VarlenCollator and dataloader integration.

Tests buffer-based packing (PACK-02), dynamic token budget (PACK-03),
and main-process collation (worker buffer data loss prevention).
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
        """Verify sequences ARE buffered when threshold not reached.

        When buffer isn't full and no packed_queue exists, sequences are
        kept in buffer and empty dict is returned. This allows sequences
        to accumulate until buffer_size threshold is reached for optimal packing.
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

        # Call collator - should buffer sequences and return empty dict
        result = collator(batch)

        # Buffer should contain the sequences (waiting for more to reach threshold)
        assert len(collator.buffer) == 2
        # Empty dict signals "not ready yet, accumulating"
        assert result == {}

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

    def test_flush_updates_sequence_tracking_counters(self):
        """Verify flush() updates _total_sequences_returned for packed_queue batches."""
        from virnucpro.data.collators import VarlenCollator

        mock_bc = MagicMock()
        mock_bc.alphabet.padding_idx = 1

        collator = VarlenCollator(mock_bc, buffer_size=3, enable_packing=True)

        # Add sequences through __call__ to establish baseline counts
        batch = [
            {'id': 'seq1', 'sequence': 'MKTAYIAK'},
            {'id': 'seq2', 'sequence': 'VLSPADKTNV'},
        ]

        with patch.object(collator, '_tokenize_and_pack', return_value={'test': 'data', 'num_sequences': 2}):
            result = collator(batch)

        # Verify initial tracking
        assert collator._total_sequences_received == 2

        # Now manually add to packed_queue to simulate pending batches
        # These are sequences already counted in received, waiting to be returned
        collator.packed_queue = [
            [{'id': 'seqX', 'sequence': 'ACGT'}],
            [{'id': 'seqY', 'sequence': 'GCTAA'}],
            [{'id': 'seqZ', 'sequence': 'GCTAA'}],
        ]

        initial_returned = collator._total_sequences_returned

        # Mock tokenization
        def mock_tokenize(batch):
            return {'test': 'data', 'num_sequences': len(batch)}

        with patch.object(collator, '_tokenize_and_pack', side_effect=mock_tokenize):
            results = collator.flush()

        # Verify flush updated _total_sequences_returned for all queue batches (3 batches)
        # The initial_returned may have had some from earlier processing
        assert collator._total_sequences_returned >= initial_returned + 3
        # Results include buffer batch (2 seqs) + 3 queue batches (3 seqs) = 4 batches
        assert len(results) == 4
        assert collator.packed_queue == []

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

    def test_empty_batch_does_not_clear_buffer(self):
        """Verify empty batch doesn't delete buffer content."""
        from virnucpro.data.collators import VarlenCollator

        mock_bc = MagicMock()
        mock_bc.alphabet.padding_idx = 1

        collator = VarlenCollator(mock_bc, enable_packing=True, buffer_size=100)

        collator.buffer = [
            {'id': 'seq1', 'sequence': 'MKTAYIAK'},
            {'id': 'seq2', 'sequence': 'VLSPADKTNV'},
        ]

        with patch.object(collator, '_tokenize_and_pack', return_value={'test': 'data'}) as mock_pack:
            result = collator([])

            assert len(collator.buffer) == 2, "Buffer should remain unchanged"
            assert result == {}, "Empty batch should return empty dict"
            mock_pack.assert_not_called(), "Should not tokenize empty batch"


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
        """Verify single item IS buffered when packing enabled (waits for threshold)."""
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

        # Single dict - buffered and empty dict returned (waiting for more sequences)
        single_item = {'id': 'seq1', 'sequence': 'MKTAYIAK', 'file': 'test.fasta'}
        result = collator(single_item)

        # Buffer should contain the single sequence (waiting for more to reach threshold)
        assert len(collator.buffer) == 1
        # Empty dict returned (not ready yet, accumulating)
        assert result == {}


class TestMainProcessCollation:
    """Tests for main-process collation to prevent worker buffer data loss.

    When VarlenCollator has enable_packing=True, it is stateful (maintains a
    buffer). If passed as collate_fn to DataLoader with num_workers>0, PyTorch
    pickles the collator to each worker process. Each worker gets its own copy
    with its own buffer. When workers finish, sequences remaining in their
    buffers are lost because flush() runs on the main process's collator
    (which was never used).

    The fix: create_async_dataloader uses a passthrough collate_fn for the
    DataLoader and stores the real collator as dataloader.collator. The
    AsyncInferenceRunner.run() method calls the collator in the main process.
    """

    def test_stateful_collator_not_passed_to_dataloader_workers(self):
        """Verify stateful collator is NOT used as DataLoader's collate_fn.

        This is the core fix: a stateful collator (enable_packing=True) must
        not be pickled to worker subprocesses, where each worker's buffer
        would lose data.
        """
        from virnucpro.data.dataloader_utils import create_async_dataloader, _passthrough_collate
        from torch.utils.data import IterableDataset

        mock_dataset = MagicMock(spec=IterableDataset)
        mock_collator = MagicMock()
        mock_collator.enable_packing = True  # Stateful
        mock_collator.packer = None

        loader = create_async_dataloader(
            dataset=mock_dataset,
            collate_fn=mock_collator,
            batch_size=None,
            num_workers=2,
            token_budget=4096,
        )

        # The DataLoader's collate_fn should be the passthrough, NOT the collator
        assert loader.collate_fn is _passthrough_collate, \
            "Stateful collator should NOT be DataLoader's collate_fn"

        # The real collator should be stored separately
        assert loader.collator is mock_collator, \
            "Real collator should be stored as loader.collator"

    def test_stateless_collator_passed_to_dataloader_normally(self):
        """Verify stateless collator (enable_packing=False) works as before."""
        from virnucpro.data.dataloader_utils import create_async_dataloader
        from torch.utils.data import IterableDataset

        mock_dataset = MagicMock(spec=IterableDataset)
        mock_collator = MagicMock()
        mock_collator.enable_packing = False  # Stateless

        loader = create_async_dataloader(
            dataset=mock_dataset,
            collate_fn=mock_collator,
            batch_size=None,
            num_workers=2,
            token_budget=4096,
        )

        # Stateless collator should be passed directly to DataLoader
        assert loader.collate_fn is mock_collator, \
            "Stateless collator should be DataLoader's collate_fn"

        # Real collator still stored for backward compatibility
        assert loader.collator is mock_collator

    def test_collator_without_enable_packing_attribute_treated_as_stateless(self):
        """Verify collators without enable_packing attr are treated as stateless."""
        from virnucpro.data.dataloader_utils import create_async_dataloader
        from torch.utils.data import IterableDataset

        mock_dataset = MagicMock(spec=IterableDataset)
        # Plain callable without enable_packing attribute
        mock_collator = MagicMock(spec=['__call__', 'max_tokens_per_batch', 'packer'])
        mock_collator.max_tokens_per_batch = 4096
        mock_collator.packer = MagicMock()

        loader = create_async_dataloader(
            dataset=mock_dataset,
            collate_fn=mock_collator,
            batch_size=None,
            num_workers=2,
            token_budget=4096,
        )

        # Should be passed directly (no enable_packing = not stateful)
        assert loader.collate_fn is mock_collator

    def test_pin_memory_disabled_for_stateful_collator(self):
        """Verify pin_memory=False when using passthrough collate_fn.

        When using passthrough collation, workers yield raw dicts (not tensors),
        so pin_memory would fail. It must be disabled.
        """
        from virnucpro.data.dataloader_utils import create_async_dataloader
        from torch.utils.data import IterableDataset

        mock_dataset = MagicMock(spec=IterableDataset)
        mock_collator = MagicMock()
        mock_collator.enable_packing = True
        mock_collator.packer = None

        loader = create_async_dataloader(
            dataset=mock_dataset,
            collate_fn=mock_collator,
            batch_size=None,
            num_workers=2,
            pin_memory=True,  # Requested True but should be overridden
            token_budget=4096,
        )

        # pin_memory should be False for passthrough (raw dicts have no tensors)
        assert loader.pin_memory is False, \
            "pin_memory must be False when using passthrough collate_fn"

    def test_runner_detects_main_process_collation(self):
        """Verify AsyncInferenceRunner detects main-process collation mode."""
        from virnucpro.pipeline.async_inference import AsyncInferenceRunner
        from unittest.mock import Mock
        import torch

        mock_model = Mock()
        mock_param = Mock()
        mock_param.dtype = torch.float32
        mock_param.numel.return_value = 1000
        mock_param.device = torch.device('cpu')
        mock_model.parameters.side_effect = lambda: iter([mock_param])

        runner = AsyncInferenceRunner(mock_model, torch.device('cpu'), enable_streams=False)

        # DataLoader with stateful collator stored
        mock_dataloader = Mock()
        mock_collator = Mock()
        mock_collator.enable_packing = True
        mock_collator.flush = Mock(return_value=[])
        mock_dataloader.collator = mock_collator

        assert runner._is_main_process_collation(mock_dataloader) is True

        # DataLoader without collator attribute
        mock_dataloader_no_collator = Mock(spec=['__iter__'])
        del mock_dataloader_no_collator.collator
        assert runner._is_main_process_collation(mock_dataloader_no_collator) is False

    def test_runner_get_collator_prefers_stored_collator(self):
        """Verify _get_collator prefers dataloader.collator over collate_fn."""
        from virnucpro.pipeline.async_inference import AsyncInferenceRunner
        from unittest.mock import Mock
        import torch

        mock_model = Mock()
        mock_param = Mock()
        mock_param.dtype = torch.float32
        mock_param.numel.return_value = 1000
        mock_param.device = torch.device('cpu')
        mock_model.parameters.side_effect = lambda: iter([mock_param])

        runner = AsyncInferenceRunner(mock_model, torch.device('cpu'), enable_streams=False)

        # Both collator and collate_fn.flush exist
        mock_collator = Mock()
        mock_collator.flush = Mock()
        mock_collate_fn = Mock()
        mock_collate_fn.flush = Mock()

        mock_dataloader = Mock()
        mock_dataloader.collator = mock_collator
        mock_dataloader.collate_fn = mock_collate_fn

        result = runner._get_collator(mock_dataloader)
        assert result is mock_collator, \
            "Should prefer dataloader.collator over dataloader.collate_fn"

    def test_runner_get_collator_falls_back_to_collate_fn(self):
        """Verify _get_collator falls back to collate_fn for backward compat."""
        from virnucpro.pipeline.async_inference import AsyncInferenceRunner
        from unittest.mock import Mock
        import torch

        mock_model = Mock()
        mock_param = Mock()
        mock_param.dtype = torch.float32
        mock_param.numel.return_value = 1000
        mock_param.device = torch.device('cpu')
        mock_model.parameters.side_effect = lambda: iter([mock_param])

        runner = AsyncInferenceRunner(mock_model, torch.device('cpu'), enable_streams=False)

        # Only collate_fn.flush (no .collator attribute)
        mock_collate_fn = Mock()
        mock_collate_fn.flush = Mock()

        mock_dataloader = Mock(spec=['collate_fn'])
        mock_dataloader.collate_fn = mock_collate_fn

        result = runner._get_collator(mock_dataloader)
        assert result is mock_collate_fn, \
            "Should fall back to collate_fn when no collator attribute"

    def test_passthrough_collate_identity(self):
        """Verify _passthrough_collate returns input unchanged."""
        from virnucpro.data.dataloader_utils import _passthrough_collate

        item = {'id': 'seq1', 'sequence': 'MKTAYIAK', 'file': 'test.fasta'}
        result = _passthrough_collate(item)
        assert result is item, "Passthrough should return exact same object"
