"""Sequence indexing for multi-GPU sharding with stride-based distribution.

This module provides index-based sharding infrastructure for deterministic
work distribution across multiple GPUs. The index contains metadata for all
sequences (id, length, file_path, byte_offset) sorted by length descending
for optimal packing efficiency.

Architecture:
    1. create_sequence_index: Scans FASTA files and builds sorted index
    2. Index caching: Validates FASTA mtimes for staleness detection
    3. get_worker_indices: Stride distribution [rank, rank+N, rank+2N...]

Key Features:
    - Descending length sort maximizes FFD packing efficiency
    - Byte-offset tracking enables random access to sequences
    - Mtime-based cache invalidation detects FASTA modifications
    - Stride distribution ensures balanced token load per worker

Example:
    >>> from pathlib import Path
    >>> fasta_files = [Path("sequences.fasta")]
    >>> index_path = Path("index.json")
    >>>
    >>> # Create index (or load from cache if valid)
    >>> create_sequence_index(fasta_files, index_path)
    >>>
    >>> # Get indices for worker 0 of 4 GPUs
    >>> indices = get_worker_indices(index_path, rank=0, world_size=4)
    >>> # Returns [0, 4, 8, 12, ...] for stride distribution
"""

import json
import logging
from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass, asdict
from datetime import datetime

logger = logging.getLogger('virnucpro.data.shard_index')


@dataclass
class SequenceEntry:
    """Metadata for a single sequence in the index.

    Attributes:
        sequence_id: Unique sequence identifier from FASTA header
        length: Sequence length in amino acids/nucleotides
        file_path: Path to source FASTA file
        byte_offset: Byte position of sequence start in file
    """
    sequence_id: str
    length: int
    file_path: str
    byte_offset: int


def create_sequence_index(fasta_files: List[Path], index_path: Path) -> Path:
    """Create or load cached sequence index from FASTA files.

    This function builds a metadata index containing sequence IDs, lengths,
    file paths, and byte offsets for all sequences across provided FASTA files.
    The index is sorted by length descending to maximize packing efficiency when
    used with FFD (First-Fit Decreasing) algorithm.

    Cache invalidation uses FASTA file modification times. If any FASTA file
    has been modified since the index was created, the index is rebuilt.

    Args:
        fasta_files: List of FASTA file paths to index
        index_path: Path where index JSON will be written/read

    Returns:
        Path to the index file (same as index_path)

    Index Format:
        JSON structure containing:
        - version: Index format version (currently "1.0")
        - created: ISO timestamp of index creation
        - fasta_mtimes: Dict mapping file paths to modification times
        - total_sequences: Total number of sequences indexed
        - total_tokens: Sum of all sequence lengths
        - sequences: List of SequenceEntry dictionaries

    Example:
        >>> fasta_files = [Path("data/sequences.fasta")]
        >>> index_path = Path("index.json")
        >>> create_sequence_index(fasta_files, index_path)
        # Created index: 10000 sequences, 5000000 total tokens
    """
    # Check cache validity
    if index_path.exists():
        try:
            with open(index_path, 'r') as f:
                cached = json.load(f)

            # Validate mtimes match
            valid = True
            cached_mtimes = cached.get('fasta_mtimes', {})

            for fasta_file in fasta_files:
                current_mtime = fasta_file.stat().st_mtime
                cached_mtime = cached_mtimes.get(str(fasta_file))

                if cached_mtime is None or current_mtime > cached_mtime:
                    valid = False
                    logger.info(
                        f"Index cache invalid: {fasta_file} "
                        f"(mtime {current_mtime} > cached {cached_mtime})"
                    )
                    break

            if valid:
                logger.info(f"Using cached index: {index_path}")
                return index_path
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Cache corrupted, rebuilding: {e}")

    # Build new index
    logger.info(f"Building sequence index for {len(fasta_files)} files")
    entries = []
    mtimes = {}

    for fasta_file in fasta_files:
        # Record mtime for cache validation
        mtimes[str(fasta_file)] = fasta_file.stat().st_mtime

        # Parse FASTA file tracking byte offsets
        with open(fasta_file, 'rb') as f:
            current_id = None
            current_len = 0
            header_offset = 0

            while True:
                line_start = f.tell()
                line = f.readline()

                if not line:
                    # End of file - save last entry
                    if current_id is not None:
                        entries.append(SequenceEntry(
                            sequence_id=current_id,
                            length=current_len,
                            file_path=str(fasta_file),
                            byte_offset=header_offset
                        ))
                    break

                if line.startswith(b'>'):
                    # Save previous entry if exists
                    if current_id is not None:
                        entries.append(SequenceEntry(
                            sequence_id=current_id,
                            length=current_len,
                            file_path=str(fasta_file),
                            byte_offset=header_offset
                        ))

                    # Start new sequence
                    # Extract ID from header (first word after '>')
                    header = line[1:].decode('utf-8', errors='ignore').strip()
                    current_id = header.split()[0] if header else str(line_start)
                    header_offset = line_start
                    current_len = 0
                else:
                    # Accumulate sequence length (strip whitespace)
                    current_len += len(line.strip())

    # Sort entries by length descending for optimal FFD packing
    entries.sort(key=lambda e: e.length, reverse=True)

    total_tokens = sum(e.length for e in entries)

    # Write index
    index_data = {
        "version": "1.0",
        "created": datetime.now().isoformat(),
        "fasta_mtimes": mtimes,
        "total_sequences": len(entries),
        "total_tokens": total_tokens,
        "sequences": [asdict(e) for e in entries]
    }

    with open(index_path, 'w') as f:
        json.dump(index_data, f, indent=2)

    logger.info(
        f"Created index: {len(entries)} sequences, {total_tokens} total tokens"
    )

    return index_path


def get_worker_indices(index_path: Path, rank: int, world_size: int) -> List[int]:
    """Get sequence indices for worker using stride distribution.

    Stride distribution assigns indices to workers in a round-robin fashion:
    - Worker 0 gets: [0, N, 2N, 3N, ...]
    - Worker 1 gets: [1, N+1, 2N+1, 3N+1, ...]
    - Worker N-1 gets: [N-1, 2N-1, 3N-1, ...]

    Since the index is sorted by length descending, this ensures each worker
    receives a balanced distribution of short and long sequences, maximizing
    packing efficiency across all workers.

    Args:
        index_path: Path to sequence index JSON
        rank: Worker rank (0-indexed)
        world_size: Total number of workers

    Returns:
        List of sequence indices assigned to this worker

    Example:
        >>> # For 12 sequences distributed across 4 workers:
        >>> get_worker_indices(index_path, rank=0, world_size=4)
        [0, 4, 8]  # Worker 0
        >>> get_worker_indices(index_path, rank=1, world_size=4)
        [1, 5, 9]  # Worker 1
    """
    index_data = load_sequence_index(index_path)

    total_sequences = len(index_data['sequences'])
    indices = list(range(rank, total_sequences, world_size))

    # Calculate distribution metrics
    worker_sequences = [index_data['sequences'][i] for i in indices]
    worker_tokens = sum(s['length'] for s in worker_sequences)

    logger.info(
        f"Worker {rank}/{world_size}: {len(indices)} sequences, "
        f"{worker_tokens:,} tokens"
    )

    return indices


def load_sequence_index(index_path: Path) -> Dict:
    """Load and return parsed sequence index.

    This is a low-level utility for loading the index JSON. Most users
    should use create_sequence_index() which handles caching, or
    get_worker_indices() for stride distribution.

    Args:
        index_path: Path to index JSON file

    Returns:
        Parsed index dictionary containing version, metadata, and sequences

    Raises:
        FileNotFoundError: If index_path does not exist
        json.JSONDecodeError: If index file is corrupted
    """
    with open(index_path, 'r') as f:
        return json.load(f)
