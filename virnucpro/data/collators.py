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
from typing import List, Dict, Any

import torch

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

    def __init__(self, batch_converter, max_tokens_per_batch: int = 4096):
        """Initialize collator with ESM batch_converter.

        Args:
            batch_converter: ESM alphabet.get_batch_converter() instance
            max_tokens_per_batch: Maximum total tokens in a packed batch
                Sequences are packed until this limit is reached. If a single
                sequence exceeds this limit, it's still included (partial batch).

        Note:
            The batch_converter is a BatchConverter instance with an alphabet attribute.
            We extract padding_idx from batch_converter.alphabet.
        """
        self.batch_converter = batch_converter
        self.max_tokens_per_batch = max_tokens_per_batch

        # Extract padding_idx from alphabet
        # ESM uses <pad> token with index 1 by default
        self.padding_idx = batch_converter.alphabet.padding_idx

        logger.debug(
            f"VarlenCollator initialized: max_tokens={max_tokens_per_batch}, "
            f"padding_idx={self.padding_idx}"
        )

    def __call__(self, batch: List[Dict[str, str]]) -> Dict[str, Any]:
        """Collate batch of sequences into packed format.

        This method:
        1. Extracts (id, sequence) tuples from batch dicts
        2. Tokenizes using ESM batch_converter (returns PADDED 2D tensor)
        3. Strips padding tokens from each sequence
        4. Packs sequences into 1D tensor until max_tokens_per_batch reached
        5. Builds cu_seqlens array (cumulative sequence boundaries)

        Args:
            batch: List of dicts from SequenceDataset with keys:
                - 'id': Sequence ID
                - 'sequence': Sequence string
                - 'file': Source filename

        Returns:
            Dictionary with packed batch:
                - 'input_ids': 1D long tensor of concatenated tokens
                - 'cu_seqlens': 1D int32 tensor of cumulative sequence lengths
                    Format: [0, len1, len1+len2, len1+len2+len3, ...]
                    Length: num_sequences + 1
                - 'max_seqlen': Maximum individual sequence length in batch
                - 'sequence_ids': List of sequence IDs that were packed
                - 'num_sequences': Number of sequences packed (len(cu_seqlens) - 1)

        Note:
            If the FIRST sequence exceeds max_tokens_per_batch, it's still
            included (partial batch with 1 sequence). Subsequent sequences
            are skipped if they would exceed the token budget.
        """
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
