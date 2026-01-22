"""File management utilities for FASTA processing"""

from Bio import SeqIO
from pathlib import Path
from typing import List
import logging

logger = logging.getLogger('virnucpro.file_utils')


def split_fasta_file(
    input_file: Path,
    output_dir: Path,
    sequences_per_file: int = 10000,
    prefix: str = "output"
) -> List[Path]:
    """
    Split FASTA file into multiple files with fixed sequence count.

    Used for parallel processing of large FASTA files by creating
    manageable batches.

    Based on units.py:267-288

    Args:
        input_file: Input FASTA file path
        output_dir: Output directory for split files
        sequences_per_file: Number of sequences per output file
        prefix: Prefix for output filenames

    Returns:
        List of created output file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_files = []
    current_file_idx = 0  # Start at 0, will increment to 1 before first use (matches vanilla)
    current_count = 0
    current_handle = None
    current_path = None

    logger.info(f"Splitting {input_file} into files of {sequences_per_file} sequences")

    try:
        for record in SeqIO.parse(input_file, 'fasta'):
            # Open new file if needed
            if current_count == 0:
                if current_handle:
                    current_handle.close()

                # Increment before use to match vanilla (1-based indexing)
                current_file_idx += 1
                current_path = output_dir / f"{prefix}_{current_file_idx}.fa"
                current_handle = open(current_path, 'w')
                output_files.append(current_path)
                logger.debug(f"Creating {current_path}")

            # Write sequence to current file
            SeqIO.write(record, current_handle, 'fasta')
            current_count += 1

            # Check if file is complete
            if current_count >= sequences_per_file:
                current_count = 0

    finally:
        if current_handle:
            current_handle.close()

    logger.info(f"Created {len(output_files)} files in {output_dir}")
    return output_files


def count_sequences(fasta_file: Path) -> int:
    """
    Count number of sequences in a FASTA file.

    Args:
        fasta_file: Path to FASTA file

    Returns:
        Number of sequences
    """
    return sum(1 for _ in SeqIO.parse(fasta_file, 'fasta'))
