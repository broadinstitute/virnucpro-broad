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

        logger.debug(
            f"VarlenCollator initialized: max_tokens={max_tokens_per_batch}, "
            f"padding_idx={self.padding_idx}, buffer_size={buffer_size}, "
            f"enable_packing={enable_packing}"
        )

    def _tokenize_and_pack(self, batch: List[Dict[str, str]]) -> Dict[str, Any]:
        """Tokenize and pack a batch of sequences.

        This is the core packing logic extracted from __call__.
        Used by both buffer-based and direct packing modes.

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
                # Stop packing - would exceed token budget
                logger.debug(
                    f"Stopping at {i}/{len(sequences)} sequences: "
                    f"token budget reached ({cu_seqlens[-1]} + {len(seq_tokens)} > {self.max_tokens_per_batch})"
                )
                break

            # Add this sequence to packed batch
            all_tokens.extend(seq_tokens)
            cu_seqlens.append(len(all_tokens))  # Cumulative length
            sequence_ids.append(labels[i])
            max_seqlen = max(max_seqlen, len(seq_tokens))

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

        if not self.enable_packing:
            # Direct processing (no buffering)
            return self._tokenize_and_pack(batch)

        # If we have pre-packed batches ready, return one and buffer the new sequences
        if self.packed_queue:
            self.buffer.extend(batch)  # Save for later packing
            packed_batch = self.packed_queue.pop(0)
            return self._tokenize_and_pack(packed_batch)

        # Add sequences to buffer
        self.buffer.extend(batch)

        # When buffer reaches threshold, pack it
        if len(self.buffer) >= self.buffer_size:
            # Run FFD on full buffer
            packed_batches = self.packer.pack_sequences(self.buffer)
            logger.debug(f"Packed {len(self.buffer)} sequences → {len(packed_batches)} batches")

            # Store packed batches in queue and clear buffer
            self.packed_queue.extend(packed_batches)
            self.buffer = []  # Clear - all sequences now in packed_queue

            # Return first packed batch
            if self.packed_queue:
                packed_batch = self.packed_queue.pop(0)
                return self._tokenize_and_pack(packed_batch)

        # Buffer not full yet - return micro-batch directly
        # IMPORTANT: Remove from buffer to prevent duplication during flush()
        # We're returning these sequences now, so they shouldn't be in buffer
        del self.buffer[-len(batch):]
        return self._tokenize_and_pack(batch)

    def flush(self) -> List[Dict[str, Any]]:
        """Flush remaining buffer at end of dataset.

        Returns list of packed batches for any sequences remaining in buffer.
        Called by AsyncInferenceRunner after dataloader exhausted.
        """
        results = []

        # Pack remaining buffer contents
        if self.buffer:
            logger.debug(f"Flushing buffer with {len(self.buffer)} sequences")
            packed_batches = self.packer.pack_sequences(self.buffer) if self.packer else [self.buffer]
            self.buffer = []

            # Tokenize and pack each batch
            for packed_batch in packed_batches:
                results.append(self._tokenize_and_pack(packed_batch))

        # Also flush any remaining packed_queue
        while self.packed_queue:
            results.append(self._tokenize_and_pack(self.packed_queue.pop(0)))

        return results
