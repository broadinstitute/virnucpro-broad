"""Sequence processing utilities"""

from Bio import SeqIO
from Bio.Seq import Seq
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import logging
import tempfile
import shutil

logger = logging.getLogger('virnucpro.sequence')


# Genetic code codon table
CODON_TABLE = {
    'TCA': 'S', 'TCC': 'S', 'TCG': 'S', 'TCT': 'S',
    'TTC': 'F', 'TTT': 'F', 'TTA': 'L', 'TTG': 'L',
    'TAC': 'Y', 'TAT': 'Y', 'TAA': '*', 'TAG': '*',
    'TGC': 'C', 'TGT': 'C', 'TGA': '*', 'TGG': 'W',
    'CTA': 'L', 'CTC': 'L', 'CTG': 'L', 'CTT': 'L',
    'CCA': 'P', 'CCC': 'P', 'CCG': 'P', 'CCT': 'P',
    'CAC': 'H', 'CAT': 'H', 'CAA': 'Q', 'CAG': 'Q',
    'CGA': 'R', 'CGC': 'R', 'CGG': 'R', 'CGT': 'R',
    'ATA': 'I', 'ATC': 'I', 'ATT': 'I', 'ATG': 'M',
    'ACA': 'T', 'ACC': 'T', 'ACG': 'T', 'ACT': 'T',
    'AAC': 'N', 'AAT': 'N', 'AAA': 'K', 'AAG': 'K',
    'AGC': 'S', 'AGT': 'S', 'AGA': 'R', 'AGG': 'R',
    'GTA': 'V', 'GTC': 'V', 'GTG': 'V', 'GTT': 'V',
    'GCA': 'A', 'GCC': 'A', 'GCG': 'A', 'GCT': 'A',
    'GAC': 'D', 'GAT': 'D', 'GAA': 'E', 'GAG': 'E',
    'GGA': 'G', 'GGC': 'G', 'GGG': 'G', 'GGT': 'G',
}


def reverse_complement(sequence: str) -> str:
    """
    Generate reverse complement of DNA sequence.

    Based on units.py:50-52

    Args:
        sequence: DNA sequence string

    Returns:
        Reverse complement sequence
    """
    complement = str.maketrans('ACGTacgt', 'TGCAtgca')
    return sequence.translate(complement)[::-1]


def translate_dna(sequence: str) -> List[str]:
    """
    Translate DNA sequence in all 6 reading frames.

    Returns 6 protein sequences: 3 forward frames + 3 reverse complement frames.

    Based on units.py:54-73

    Args:
        sequence: DNA sequence string

    Returns:
        List of 6 protein sequences
    """
    def translate_frame(seq: str, frame: int) -> str:
        """Translate a single reading frame"""
        return ''.join(
            CODON_TABLE.get(seq[i:i+3], '')
            for i in range(frame, len(seq) - 2, 3)
        )

    # Translate forward strand (3 frames)
    frames = [translate_frame(sequence, frame) for frame in range(3)]

    # Translate reverse complement (3 frames)
    rev_comp_sequence = reverse_complement(sequence)
    frames += [translate_frame(rev_comp_sequence, frame) for frame in range(3)]

    return frames


def identify_seq(seqid: str, sequence: str, refseq_pro_list: Optional[List[str]] = None,
                 istraindata: bool = False) -> Optional[List[Dict[str, str]]]:
    """
    Identify valid protein-coding regions from six-frame translation.

    Translates sequence in all 6 frames and returns valid ORFs
    (those without stop codons).

    Based on units.py:81-146

    Args:
        seqid: Sequence identifier
        sequence: DNA sequence string
        refseq_pro_list: Reference protein list for training mode validation
        istraindata: If True, use training mode with additional filtering

    Returns:
        List of dictionaries with keys: 'seqid', 'nucleotide', 'protein'
        Each dict represents a valid ORF with frame indicator (F1-F3, R1-R3)
        Returns None if training mode and ambiguous bases found
    """
    final_list = []

    # Training mode - check for ambiguous bases
    if istraindata:
        ambiguous_bases = {'N', 'R', 'Y', 'M', 'K', 'S', 'W', 'H', 'B', 'V', 'D'}
        if any(base in sequence for base in ambiguous_bases):
            return None

    proteins_list = translate_dna(sequence)

    for num, protein in enumerate(proteins_list, start=1):
        # Skip frames with stop codons
        if "*" in protein:
            continue

        # Training mode - validate against reference proteins
        if istraindata and refseq_pro_list:
            # Check if protein is in reference list
            if not any(protein in refpro for refpro in refseq_pro_list):
                continue

        # Calculate sequence lengths
        pro_len = len(protein)
        seq_len = pro_len * 3

        # Determine strand
        strand = "+" if num <= 3 else "-"

        if strand == "+":
            # Forward strand
            selected_seq = sequence[num-1:num + seq_len]
            seqid_final = seqid + 'F' + str(num)
        else:
            # Reverse strand
            selected_seq = reverse_complement(sequence)[num-3-1:num-3 + seq_len]
            seqid_final = seqid + 'R' + str(num - 3)

        final_list.append({
            'seqid': seqid_final,
            'nucleotide': selected_seq,
            'protein': protein
        })

    return final_list


def _determine_max_sequence_length(input_file: Path) -> Tuple[int, int]:
    """
    Pre-scan input sequences to determine maximum length for skip-chunking decision.

    Args:
        input_file: Input FASTA file path

    Returns:
        Tuple of (max_length, sequence_count)
    """
    max_length = 0
    sequence_count = 0

    try:
        for record in SeqIO.parse(input_file, 'fasta'):
            max_length = max(max_length, len(record.seq))
            sequence_count += 1
    except Exception as error:
        logger.error(f"Pre-scan failed: {error}")
        raise

    return (max_length, sequence_count)


def _copy_with_chunk_suffix(input_file: Path, output_file: Path, sequence_count: int):
    """
    Copy sequences with chunk_1 suffix when chunking is skipped.

    Preserves pipeline contract: downstream stages (6-frame translation, consensus scoring)
    expect chunk naming pattern {original_id}_chunk_{N} for frame suffix grouping.

    Args:
        input_file: Input FASTA file path
        output_file: Output FASTA file path
        sequence_count: Number of sequences (for logging)
    """
    try:
        # Atomic write pattern: write to temp file, then move to avoid partial writes on failure
        with tempfile.NamedTemporaryFile(mode='w', delete=False, dir=output_file.parent) as tmp_handle:
            for record in SeqIO.parse(input_file, 'fasta'):
                record.id = f"{record.id}_chunk_1"
                # Strip description to match chunk format expected by downstream stages (translation/consensus)
                record.description = ""
                SeqIO.write(record, tmp_handle, 'fasta')
        shutil.move(tmp_handle.name, output_file)
        logger.info(f"Rewrote {sequence_count} sequences with chunk_1 suffix")
    except Exception as error:
        logger.error(f"Skip failed during file rewrite: {error}")
        raise


def _chunk_with_overlaps(input_file: Path, output_file: Path, chunk_size: int):
    """
    Chunk sequences with overlapping strategy to capture all sequence information.

    Overlap distribution algorithm ensures uniform chunk sizes while minimizing
    information loss at chunk boundaries.

    Args:
        input_file: Input FASTA file path
        output_file: Output FASTA file path
        chunk_size: Target size for each chunk (300 or 500bp matching model)
    """
    total_sequences = 0
    total_chunks = 0

    try:
        with open(output_file, 'w') as out_handle:
            for record in SeqIO.parse(input_file, 'fasta'):
                sequence = record.seq
                seq_length = len(sequence)

                # Calculate number of chunks needed
                num_chunks = -(-seq_length // chunk_size)  # Ceiling division

                # Calculate overlap distribution
                total_chunk_length = num_chunks * chunk_size
                repeat_length = total_chunk_length - seq_length
                repeat_region = repeat_length / num_chunks
                lower_int = int(repeat_region)
                upper_int = lower_int + 1

                # Distribute overlap across chunks
                low_up_numbers = [lower_int] * (num_chunks - 1)
                total_low_up_numbers = sum(low_up_numbers)
                need_to_add_up_nums = repeat_length - total_low_up_numbers
                final_low_up_numbers = (
                    [upper_int] * need_to_add_up_nums +
                    [lower_int] * (num_chunks - 1 - need_to_add_up_nums) +
                    [0]
                )

                # Generate chunks
                move_step = 0
                for a, b in zip(range(0, seq_length, chunk_size), final_low_up_numbers):
                    if a > 1:
                        chunk = record[a-move_step:a-move_step + chunk_size]
                    else:
                        chunk = record[a:a + chunk_size]

                    new_record = chunk
                    new_record.id = f"{record.id}_chunk_{a // chunk_size + 1}"
                    # Strip description to match chunk format expected by downstream stages (translation/consensus)
                    new_record.description = ""

                    SeqIO.write(new_record, out_handle, 'fasta')
                    move_step += b
                    total_chunks += 1

                total_sequences += 1

        logger.info(f"Created {total_chunks} chunks from {total_sequences} sequences")
    except Exception as error:
        logger.error(f"Chunking failed during file write: {error}")
        raise


def split_fasta_chunk(input_file: Path, output_file: Path, chunk_size: int):
    """
    Split FASTA sequences into overlapping chunks of specified size.

    Optimization: skips chunking when ALL sequences are <= chunk_size. Conservative
    all-or-nothing decision maintains output format uniformity required by downstream
    stages (6-frame translation, consensus scoring).

    Uses overlapping strategy to ensure all sequence information is captured
    in fixed-size chunks.

    Based on units.py:9-36

    Args:
        input_file: Input FASTA file path
        output_file: Output FASTA file path
        chunk_size: Target size for each chunk (300 or 500bp matching model)
    """
    max_length, sequence_count = _determine_max_sequence_length(input_file)

    # Conservative decision: if ANY sequence exceeds chunk_size, chunk ALL to maintain
    # uniform output format. Downstream stages expect consistent chunk structure.
    if max_length <= chunk_size:
        logger.info(f"Skipping chunking: {sequence_count} sequences, max length {max_length}bp <= {chunk_size}bp threshold")
        _copy_with_chunk_suffix(input_file, output_file, sequence_count)
        return

    logger.info(f"Chunking sequences to {chunk_size}bp: {input_file} -> {output_file}")
    _chunk_with_overlaps(input_file, output_file, chunk_size)
