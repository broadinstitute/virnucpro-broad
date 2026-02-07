"""Collator for producing FlashAttention varlen packed batch format.

This module provides VarlenCollator which tokenizes sequences in the main
process and produces packed 1D tensor format with cumulative sequence lengths
(cu_seqlens) for FlashAttention variable-length attention.

Architecture:
    Workers yield raw strings → VarlenCollator tokenizes → Packed batch

    Packed format:
        input_ids: [seq1_tok1, seq1_tok2, ..., seq2_tok1, seq2_tok2, ...]
        cu_seqlens: [0, len(seq1), len(seq1)+len(seq2), ...]

Critical:
    This collator runs in the MAIN PROCESS (not workers). Workers yield
    raw sequence strings, and this collator handles ESM tokenization.
    This ensures workers remain CUDA-safe.

Integration:
    Use as collate_fn parameter in DataLoader:
        DataLoader(dataset, collate_fn=VarlenCollator(batch_converter))
"""

import logging
import threading
from typing import List, Dict, Any, Union

import torch
from virnucpro.data.packing import GreedyPacker

logger = logging.getLogger('virnucpro.data.collators')


class VarlenCollator:
    """Collate sequences into packed format for FlashAttention varlen attention.

    This collator:
    1. Receives batch of sequence dicts from workers (raw strings)
    2. Tokenizes sequences using ESM batch_converter
    3. Strips ESM padding tokens
    4. Packs sequences into 1D concatenated tensor
    5. Produces cu_seqlens (cumulative sequence boundaries) for varlen attention

    The packed format allows FlashAttention to process variable-length sequences
    efficiently without traditional padding, using cu_seqlens to identify
    sequence boundaries.

    Attributes:
        batch_converter: ESM alphabet.get_batch_converter() for tokenization
        max_tokens_per_batch: Maximum total tokens in a packed batch
        padding_idx: ESM padding token ID (typically 1)

    Example:
        >>> import esm
        >>> _, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        >>> batch_converter = alphabet.get_batch_converter()
        >>> collator = VarlenCollator(batch_converter, max_tokens_per_batch=4096)
        >>>
        >>> batch = [
        ...     {'id': 'seq1', 'sequence': 'MKTAYIAK', 'file': 'test.fasta'},
        ...     {'id': 'seq2', 'sequence': 'VLSPADKTNV', 'file': 'test.fasta'},
        ... ]
        >>> result = collator(batch)
        >>> # result['input_ids']: 1D tensor [seq1_tokens..., seq2_tokens...]
        >>> # result['cu_seqlens']: [0, len(seq1), len(seq1)+len(seq2)]
    """

    def __init__(
        self,
        batch_converter,
        max_tokens_per_batch: int = 4096,
        max_sequence_length: int = 1022,
        buffer_size: int = 2000,
        enable_packing: bool = True,
    ):
        """Initialize collator with ESM batch_converter.

        Args:
            batch_converter: ESM alphabet.get_batch_converter() instance
            max_tokens_per_batch: Maximum total tokens in a packed batch
            max_sequence_length: Max individual sequence length (ESM-2 limit: 1022)
            buffer_size: Number of sequences to accumulate before packing.
                Default 2000 achieves 92-94% efficiency. Range: 1000-5000.
                Larger buffers improve efficiency but use more memory.
            enable_packing: If True, use buffer-based packing. If False, process
                batches directly (for testing/debugging).
        """
        self.batch_converter = batch_converter
        self.max_tokens_per_batch = max_tokens_per_batch
        self.padding_idx = batch_converter.alphabet.padding_idx

        # Buffer-based packing (PACK-02, ARCH-11)
        self.buffer = []  # Accumulates sequences before packing
        self.packed_queue = []  # Pre-packed batches ready to return
        self.buffer_size = buffer_size
        self.enable_packing = enable_packing

        # Initialize GreedyPacker for FFD algorithm
        if enable_packing:
            self.packer = GreedyPacker(
                max_tokens_per_batch=max_tokens_per_batch,
                max_sequence_length=max_sequence_length,
            )
        else:
            self.packer = None

        # Sequence tracking counters (thread-safe)
        self._total_sequences_received = 0
        self._total_sequences_returned = 0
        self._counter_lock = threading.Lock()

        logger.debug(
            f"VarlenCollator initialized: max_tokens={max_tokens_per_batch}, "
            f"padding_idx={self.padding_idx}, buffer_size={buffer_size}, "
            f"enable_packing={enable_packing}"
        )

    def _tokenize_and_pack(self, batch: List[Dict[str, str]]) -> Dict[str, Any]:
        """Tokenize and pack a batch of sequences.

        This is the core packing logic extracted from __call__.
        Used by both buffer-based and direct packing modes.

        When actual tokenized lengths exceed the token budget (due to
        differences between GreedyPacker's length estimates and actual
        ESM tokenization), overflow sequences are returned to self.buffer
        for processing in a subsequent batch.

        Args:
            batch: List of sequence dicts with 'id' and 'sequence' keys

        Returns:
            Dictionary with packed batch (input_ids, cu_seqlens, etc.)
        """
        if not batch:
            return {}

        # Extract (id, sequence) tuples for ESM batch_converter
        sequences = [(item['id'], item['sequence']) for item in batch]

        # Tokenize using ESM batch_converter
        # Returns:
        #   labels: List of sequence IDs
        #   strs: List of sequence strings
        #   tokens: 2D tensor (batch_size × max_seq_len) with PADDING
        labels, strs, tokens = self.batch_converter(sequences)

        # Build packed format: concatenate unpadded tokens
        all_tokens = []
        cu_seqlens = [0]
        sequence_ids = []
        max_seqlen = 0

        packed_count = 0  # Track how many sequences actually fit

        for i in range(len(sequences)):
            # Find actual sequence length by locating padding
            # ESM adds special tokens (BOS, EOS), so we need to find
            # where padding starts to get the true length
            seq_mask = tokens[i] != self.padding_idx
            seq_len = seq_mask.sum().item()

            # Extract unpadded tokens (everything before first padding token)
            seq_tokens = tokens[i, :seq_len].tolist()

            # Edge case: First sequence exceeds max_tokens_per_batch
            # Still include it (partial batch with 1 sequence)
            if i == 0 and len(seq_tokens) > self.max_tokens_per_batch:
                logger.warning(
                    f"Sequence {labels[i]} exceeds max_tokens_per_batch "
                    f"({len(seq_tokens)} > {self.max_tokens_per_batch}). "
                    f"Including as single-sequence batch."
                )

            # Normal case: Check if adding this sequence exceeds budget
            # (skip check for first sequence - always include it)
            if i > 0 and cu_seqlens[-1] + len(seq_tokens) > self.max_tokens_per_batch:
                # Stop packing - return overflow to buffer for later processing
                break

            # Add this sequence to packed batch
            all_tokens.extend(seq_tokens)
            cu_seqlens.append(len(all_tokens))  # Cumulative length
            sequence_ids.append(labels[i])
            max_seqlen = max(max_seqlen, len(seq_tokens))
            packed_count += 1

        # Handle overflow: return unprocessed sequences to buffer
        if packed_count < len(batch):
            overflow = batch[packed_count:]
            self.buffer.extend(overflow)
            logger.info(
                f"Token budget overflow: {len(overflow)} sequences returned to buffer "
                f"(packed {packed_count}/{len(batch)}, budget {self.max_tokens_per_batch})"
            )

        # Convert to tensors
        input_ids = torch.tensor(all_tokens, dtype=torch.long)
        cu_seqlens_tensor = torch.tensor(cu_seqlens, dtype=torch.int32)

        logger.debug(
            f"Packed {len(sequence_ids)} sequences: "
            f"{len(all_tokens)} tokens, max_seqlen={max_seqlen}"
        )

        return {
            'input_ids': input_ids,
            'cu_seqlens': cu_seqlens_tensor,
            'max_seqlen': max_seqlen,
            'sequence_ids': sequence_ids,
            'num_sequences': len(cu_seqlens) - 1,  # cu_seqlens has N+1 elements
        }

    def __call__(
        self, batch: Union[List[Dict[str, str]], Dict[str, str]]
    ) -> Dict[str, Any]:
        """Stateful collator with buffer-based packing.

        This method:
        1. Accumulates sequences into buffer
        2. When buffer reaches threshold, runs GreedyPacker.pack_sequences()
        3. Returns packed batches from queue

        Args:
            batch: Single dict or list of dicts from SequenceDataset with keys:
                - 'id': Sequence ID
                - 'sequence': Sequence string
                - 'file': Source filename
                When batch_size=None in DataLoader, receives single dicts.
                When batch_size=N, receives lists of N dicts.

        Returns:
            Dictionary with packed batch or empty dict if buffer not ready
        """
        # Handle single item when batch_size=None (DataLoader passes individual items)
        if isinstance(batch, dict):
            batch = [batch]

        # Guard: empty batch returns empty dict without processing
        if batch is None or not batch:
            return {}

        if not self.enable_packing:
            # Direct processing (no buffering)
            return self._tokenize_and_pack(batch)

        # Track total sequences received for flush verification
        with self._counter_lock:
            self._total_sequences_received += len(batch)

        # If we have pre-packed batches ready, return one and buffer the new sequences
        if self.packed_queue:
            self.buffer.extend(batch)  # Save for later packing
            batch_to_return = self.packed_queue.pop(0)
            with self._counter_lock:
                self._total_sequences_returned += len(batch_to_return)
            logger.debug(f"Returning from packed_queue ({len(self.packed_queue)} batches remaining), buffer now has {len(self.buffer)} sequences")
            return self._tokenize_and_pack(batch_to_return)

        # Add sequences to buffer
        self.buffer.extend(batch)

        # When buffer reaches threshold, pack it
        if len(self.buffer) >= self.buffer_size:
            # Run FFD on full buffer
            logger.debug(f"Buffer reached threshold ({len(self.buffer)}/{self.buffer_size}), packing...")
            packed_batches = self.packer.pack_sequences(self.buffer)

            # Count sequences in packed batches for verification
            total_packed = sum(len(b) for b in packed_batches)
            logger.debug(f"Packed {len(self.buffer)} sequences → {len(packed_batches)} batches containing {total_packed} sequences")

            # Store packed batches in queue and clear buffer
            self.packed_queue.extend(packed_batches)
            self.buffer = []  # Clear - all sequences now in packed_queue

            # Return first packed batch
            if self.packed_queue:
                batch_to_return = self.packed_queue.pop(0)
                with self._counter_lock:
                    self._total_sequences_returned += len(batch_to_return)
                logger.debug(f"Returning first packed batch ({len(batch_to_return)} sequences), {len(self.packed_queue)} batches remaining in queue")
                return self._tokenize_and_pack(batch_to_return)

        # Buffer not full yet - wait for more sequences
        # Return empty dict to signal DataLoader to keep accumulating
        # Sequences stay in buffer until threshold reached
        logger.debug(f"Buffer accumulating: {len(self.buffer)}/{self.buffer_size} sequences")
        return {}

    def flush(self) -> List[Dict[str, Any]]:
        """Flush remaining buffer at end of dataset.

        Returns list of packed batches for any sequences remaining in buffer.
        Called by AsyncInferenceRunner after dataloader exhausted.

        Loops until buffer is fully drained, since _tokenize_and_pack may
        return overflow sequences back to the buffer when actual tokenized
        lengths exceed the GreedyPacker's estimates.
        """
        # Log current state before flushing
        buffer_count = len(self.buffer)
        queue_count = len(self.packed_queue)

        logger.info(
            f"Flush called: buffer has {buffer_count} sequences, "
            f"packed_queue has {queue_count} batches, "
            f"total received: {self._total_sequences_received}, "
            f"total returned so far: {self._total_sequences_returned}"
        )

        expected_remaining = self._total_sequences_received - self._total_sequences_returned
        logger.info(f"Expected sequences remaining to flush: {expected_remaining}")

        results = []

        # Loop until buffer AND packed_queue are both empty.
        # _tokenize_and_pack may put overflow back into buffer, so we
        # must keep draining until nothing remains.
        max_iterations = 50  # Safety limit to prevent infinite loops
        iteration = 0

        while (self.buffer or self.packed_queue) and iteration < max_iterations:
            iteration += 1

            # Pack remaining buffer contents
            if self.buffer:
                logger.info(f"Flush iteration {iteration}: packing {len(self.buffer)} buffer sequences")
                packed_batches = self.packer.pack_sequences(self.buffer) if self.packer else [self.buffer]
                self.buffer = []

                # Tokenize and pack each batch (may produce overflow back to buffer)
                for packed_batch in packed_batches:
                    result = self._tokenize_and_pack(packed_batch)
                    if result:
                        with self._counter_lock:
                            self._total_sequences_returned += result.get('num_sequences', 0)
                        results.append(result)

            # Drain packed_queue (may also produce overflow back to buffer)
            while self.packed_queue:
                batch = self.packed_queue.pop(0)
                result = self._tokenize_and_pack(batch)
                if result:
                    with self._counter_lock:
                        self._total_sequences_returned += result.get('num_sequences', 0)
                    results.append(result)

        if self.buffer:
            logger.error(
                f"Flush safety limit reached after {max_iterations} iterations! "
                f"{len(self.buffer)} sequences still in buffer - these will be lost."
            )

        # Calculate total sequences in flushed results
        total_flushed = sum(r.get('num_sequences', 0) for r in results if r)
        logger.info(
            f"Flush complete: {len(results)} batches, {total_flushed} sequences flushed "
            f"(started with {buffer_count} buffer + {queue_count} queue batches, "
            f"{iteration} iteration(s))"
        )

        return results
