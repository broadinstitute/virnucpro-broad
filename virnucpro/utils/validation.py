"""Input validation utilities"""

from Bio import SeqIO
from pathlib import Path
from typing import Dict, List, Tuple, Set
import re
import logging

logger = logging.getLogger('virnucpro.validation')


def validate_fasta_file(
    fasta_file: Path,
    max_errors: int = 10
) -> Tuple[bool, List[str], List[str], Dict]:
    """
    Validate FASTA file for common issues.

    Args:
        fasta_file: Path to FASTA file
        max_errors: Maximum errors to collect

    Returns:
        Tuple of (is_valid, errors, warnings, statistics)
    """
    errors = []
    warnings = []

    stats = {
        'total_sequences': 0,
        'min_length': float('inf'),
        'max_length': 0,
        'avg_length': 0,
        'ambiguous_count': 0,
        'duplicate_ids': set(),
        'empty_sequences': 0
    }

    seen_ids = set()
    total_length = 0
    ambiguous_bases = {'N', 'R', 'Y', 'M', 'K', 'S', 'W', 'H', 'B', 'V', 'D'}
    valid_bases = re.compile(r'^[ATGCNRYKMSWBDHV]+$', re.IGNORECASE)

    try:
        for record in SeqIO.parse(fasta_file, 'fasta'):
            stats['total_sequences'] += 1

            # Check for duplicate IDs
            if record.id in seen_ids:
                stats['duplicate_ids'].add(record.id)
                if len(errors) < max_errors:
                    errors.append(f"Duplicate ID: {record.id}")
            seen_ids.add(record.id)

            # Check sequence length
            seq_len = len(record.seq)
            if seq_len == 0:
                stats['empty_sequences'] += 1
                if len(errors) < max_errors:
                    errors.append(f"Empty sequence: {record.id}")
                continue

            stats['min_length'] = min(stats['min_length'], seq_len)
            stats['max_length'] = max(stats['max_length'], seq_len)
            total_length += seq_len

            # Check for ambiguous bases
            seq_str = str(record.seq).upper()
            if any(base in seq_str for base in ambiguous_bases):
                stats['ambiguous_count'] += 1
                if len(warnings) < max_errors:
                    warnings.append(f"Ambiguous bases in {record.id}")

            # Check for invalid characters
            if not valid_bases.match(seq_str):
                if len(errors) < max_errors:
                    invalid_chars = set(seq_str) - set('ATGCNRYKMSWBDHV')
                    errors.append(
                        f"Invalid characters in {record.id}: {invalid_chars}"
                    )

        # Calculate average length
        if stats['total_sequences'] > 0:
            stats['avg_length'] = total_length / stats['total_sequences']

        # Determine if valid
        is_valid = len(errors) == 0 and stats['total_sequences'] > 0

        return is_valid, errors, warnings, stats

    except Exception as e:
        logger.error(f"Failed to parse FASTA file: {e}")
        errors.append(f"File parsing error: {e}")
        return False, errors, warnings, stats
