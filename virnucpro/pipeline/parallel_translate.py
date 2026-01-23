"""Parallel translation orchestration for CPU multiprocessing

This module provides parallel six-frame DNA translation using multiprocessing.Pool
with spawn context for safe process creation. Designed for memory-efficient processing
of large datasets (22M sequences) using Pool.imap() for lazy evaluation.

Based on patterns from:
- work_queue.py: Spawn context and worker orchestration
- parallel.py: Worker function structure and progress reporting
- sequence.py: identify_seq for six-frame translation
"""

import multiprocessing
import os
import logging
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any, Iterator
from Bio import SeqIO

logger = logging.getLogger('virnucpro.parallel_translate')


def translate_sequence_worker(record_data: Tuple[str, str]) -> Optional[List[Dict[str, str]]]:
    """
    Worker function for parallel translation of a single sequence.

    Top-level function (not nested) required for picklability with spawn context.

    Args:
        record_data: Tuple of (seqid, sequence_string)

    Returns:
        List of dictionaries with keys: 'seqid', 'nucleotide', 'protein'
        None if sequence is invalid or has no valid ORFs
    """
    seqid, sequence = record_data

    # Import inside worker to avoid pickling heavy objects
    from virnucpro.utils.sequence import identify_seq

    try:
        result = identify_seq(seqid, sequence)
        return result
    except Exception as e:
        logger.debug(f"Translation failed for {seqid}: {e}")
        return None


def translate_batch_worker(batch: List[Tuple[str, str]]) -> List[Optional[List[Dict[str, str]]]]:
    """
    Worker function for parallel translation of a batch of sequences.

    Processes multiple sequences in a single worker call, reducing serialization
    overhead from 22M to ~220K operations (100x reduction with batch_size=100).

    Top-level function (not nested) required for picklability with spawn context.

    Args:
        batch: List of (seqid, sequence_string) tuples

    Returns:
        List of results, one per sequence in batch
        Each result is either a list of ORF dicts or None if invalid
    """
    # Import inside worker to avoid pickling heavy objects
    from virnucpro.utils.sequence import identify_seq

    results = []
    for seqid, sequence in batch:
        try:
            result = identify_seq(seqid, sequence)
            results.append(result)
        except Exception as e:
            logger.debug(f"Translation failed for {seqid}: {e}")
            results.append(None)

    return results


def create_sequence_batches(
    sequence_iterator: Iterator[Tuple[str, str]],
    batch_size: int = 100
) -> Iterator[List[Tuple[str, str]]]:
    """
    Create batches of sequences from an iterator.

    Memory-efficient batching that yields batches of specified size.
    Last batch may be smaller if sequences don't divide evenly.

    Args:
        sequence_iterator: Iterator yielding (seqid, sequence) tuples
        batch_size: Number of sequences per batch (default: 100)

    Yields:
        Lists of (seqid, sequence) tuples, each of length batch_size
        (except possibly the last batch)

    Example:
        >>> records = ((f"seq{i}", "ATCG") for i in range(250))
        >>> batches = list(create_sequence_batches(records, batch_size=100))
        >>> len(batches)  # 3 batches: [100, 100, 50]
        3
    """
    batch = []
    for record in sequence_iterator:
        batch.append(record)
        if len(batch) >= batch_size:
            yield batch
            batch = []

    # Yield final batch if non-empty
    if batch:
        yield batch


def get_optimal_settings(
    num_workers: Optional[int] = None,
    avg_sequence_length: int = 500,
    total_sequences: Optional[int] = None
) -> Tuple[int, int, int]:
    """
    Calculate optimal settings for parallel translation.

    Determines number of workers, batch size, and chunksize for Pool.imap()
    based on system resources and data characteristics.

    Args:
        num_workers: Number of worker processes (default: cpu_count())
        avg_sequence_length: Average sequence length in bp (default: 500)
        total_sequences: Total number of sequences if known (optional)

    Returns:
        Tuple of (num_workers, batch_size, chunksize)

    Example:
        >>> workers, batch_size, chunksize = get_optimal_settings()
        >>> workers == os.cpu_count()
        True
    """
    # Determine number of workers
    if num_workers is None:
        num_workers = os.cpu_count()

    # Calculate batch_size based on sequence length
    # Larger sequences benefit from smaller batches to balance work
    # Smaller sequences can use larger batches to reduce overhead
    if avg_sequence_length < 300:
        batch_size = 200
    elif avg_sequence_length < 1000:
        batch_size = 100
    else:
        batch_size = 50

    # Calculate chunksize for Pool.imap()
    # Default formula: len(iterable) / (num_workers * 4)
    # For batched approach: total_batches / (num_workers * 4)
    if total_sequences is not None:
        total_batches = (total_sequences + batch_size - 1) // batch_size
        chunksize = max(1, total_batches // (num_workers * 4))
    else:
        # Conservative default when total unknown
        chunksize = 10

    return (num_workers, batch_size, chunksize)


def parallel_translate_batched(
    input_file: Path,
    output_nuc: Path,
    output_pro: Path,
    num_workers: Optional[int] = None,
    batch_size: int = 100,
    chunksize: Optional[int] = None
) -> Tuple[int, int]:
    """
    Parallel translation with batch processing optimization.

    Uses translate_batch_worker to process sequences in batches, reducing
    serialization overhead from 22M to ~220K operations (100x reduction).

    Args:
        input_file: Input FASTA file path
        output_nuc: Output nucleotide FASTA file path
        output_pro: Output protein FASTA file path
        num_workers: Number of worker processes (default: cpu_count())
        batch_size: Sequences per batch (default: 100)
        chunksize: Batches per Pool.imap chunk (default: calculated)

    Returns:
        Tuple of (sequences_processed, sequences_with_valid_orfs)

    Example:
        >>> processed, valid = parallel_translate_batched(
        ...     Path('input.fa'),
        ...     Path('output_nuc.fa'),
        ...     Path('output_pro.faa'),
        ...     num_workers=8,
        ...     batch_size=100
        ... )
    """
    if num_workers is None:
        num_workers = os.cpu_count()

    # Calculate optimal chunksize if not provided
    if chunksize is None:
        _, _, chunksize = get_optimal_settings(num_workers=num_workers)

    logger.info(f"Starting batched parallel translation with {num_workers} workers "
               f"(batch_size={batch_size}, chunksize={chunksize})")

    # Use spawn context for safety (matches work_queue.py pattern)
    ctx = multiprocessing.get_context('spawn')

    # Create generator of (seqid, sequence) tuples
    def sequence_iterator() -> Iterator[Tuple[str, str]]:
        """Generator to avoid loading all sequences into memory"""
        for record in SeqIO.parse(input_file, 'fasta'):
            yield (record.id, str(record.seq).upper())

    # Create batches from sequences
    batches = create_sequence_batches(sequence_iterator(), batch_size=batch_size)

    sequences_processed = 0
    sequences_with_orfs = 0

    try:
        with ctx.Pool(num_workers) as pool:
            # Use imap() for lazy evaluation with batches
            batch_results = pool.imap(
                translate_batch_worker,
                batches,
                chunksize=chunksize
            )

            # Write results as they arrive (memory-efficient streaming)
            with open(output_nuc, 'w') as nuc_out, open(output_pro, 'w') as pro_out:
                for batch_result in batch_results:
                    # Flatten batch results
                    for result in batch_result:
                        sequences_processed += 1

                        if result:  # None if invalid sequence or no valid ORFs
                            sequences_with_orfs += 1

                            # Write all ORFs for this sequence
                            for orf in result:
                                nuc_out.write(f">{orf['seqid']}\n{orf['nucleotide']}\n")
                                pro_out.write(f">{orf['seqid']}\n{orf['protein']}\n")

        logger.info(f"Batched translation complete: {sequences_processed} sequences processed, "
                   f"{sequences_with_orfs} with valid ORFs")

    except Exception as e:
        logger.exception("Error during batched parallel translation")
        raise

    return (sequences_processed, sequences_with_orfs)


def parallel_translate_sequences(
    input_file: Path,
    output_nuc: Path,
    output_pro: Path,
    num_workers: Optional[int] = None,
    chunksize: int = 1000
) -> Tuple[int, int]:
    """
    Parallelize six-frame translation across CPU cores.

    Uses spawn context for safety (matches GPU worker pattern) and Pool.imap()
    for memory-efficient processing of large datasets (22M sequences).

    Args:
        input_file: Input FASTA file path
        output_nuc: Output nucleotide FASTA file path
        output_pro: Output protein FASTA file path
        num_workers: Number of worker processes (default: cpu_count())
        chunksize: Records per chunk for Pool.imap() (default: 1000)

    Returns:
        Tuple of (sequences_processed, sequences_with_valid_orfs)

    Example:
        >>> processed, valid = parallel_translate_sequences(
        ...     Path('input.fa'),
        ...     Path('output_nuc.fa'),
        ...     Path('output_pro.faa'),
        ...     num_workers=8
        ... )
    """
    if num_workers is None:
        num_workers = os.cpu_count()

    logger.info(f"Starting parallel translation with {num_workers} workers (chunksize={chunksize})")

    # Use spawn context for safety (matches work_queue.py pattern)
    ctx = multiprocessing.get_context('spawn')

    # Create generator of (seqid, sequence) tuples for workers
    def sequence_iterator() -> Iterator[Tuple[str, str]]:
        """Generator to avoid loading all sequences into memory"""
        for record in SeqIO.parse(input_file, 'fasta'):
            yield (record.id, str(record.seq).upper())

    sequences_processed = 0
    sequences_with_orfs = 0

    try:
        with ctx.Pool(num_workers) as pool:
            # Use imap() for lazy evaluation - critical for 22M sequences
            results = pool.imap(
                translate_sequence_worker,
                sequence_iterator(),
                chunksize=chunksize
            )

            # Write results as they arrive (memory-efficient streaming)
            with open(output_nuc, 'w') as nuc_out, open(output_pro, 'w') as pro_out:
                for result in results:
                    sequences_processed += 1

                    if result:  # None if invalid sequence or no valid ORFs
                        sequences_with_orfs += 1

                        # Write all ORFs for this sequence
                        for orf in result:
                            nuc_out.write(f">{orf['seqid']}\n{orf['nucleotide']}\n")
                            pro_out.write(f">{orf['seqid']}\n{orf['protein']}\n")

        logger.info(f"Translation complete: {sequences_processed} sequences processed, "
                   f"{sequences_with_orfs} with valid ORFs")

    except Exception as e:
        logger.exception("Error during parallel translation")
        raise

    return (sequences_processed, sequences_with_orfs)


def parallel_translate_with_progress(
    input_file: Path,
    output_nuc: Path,
    output_pro: Path,
    num_workers: Optional[int] = None,
    chunksize: int = 1000,
    show_progress: bool = True
) -> Tuple[int, int]:
    """
    Parallel translation with progress reporting.

    Counts sequences first to enable accurate progress bar, then processes
    with same logic as parallel_translate_sequences().

    Args:
        input_file: Input FASTA file path
        output_nuc: Output nucleotide FASTA file path
        output_pro: Output protein FASTA file path
        num_workers: Number of worker processes (default: cpu_count())
        chunksize: Records per chunk for Pool.imap() (default: 1000)
        show_progress: Enable progress bar display (default: True)

    Returns:
        Tuple of (sequences_processed, sequences_with_valid_orfs)
    """
    if num_workers is None:
        num_workers = os.cpu_count()

    # Pre-count sequences for progress bar
    logger.info(f"Counting sequences in {input_file.name}...")
    num_sequences = sum(1 for _ in SeqIO.parse(input_file, 'fasta'))
    logger.info(f"Found {num_sequences} sequences")

    logger.info(f"Starting parallel translation with {num_workers} workers (chunksize={chunksize})")

    ctx = multiprocessing.get_context('spawn')

    def sequence_iterator() -> Iterator[Tuple[str, str]]:
        """Generator to avoid loading all sequences into memory"""
        for record in SeqIO.parse(input_file, 'fasta'):
            yield (record.id, str(record.seq).upper())

    sequences_processed = 0
    sequences_with_orfs = 0

    try:
        # Import progress reporting utilities
        try:
            from virnucpro.utils.progress import ProgressReporter
            progress = ProgressReporter(disable=not show_progress)
        except ImportError:
            # Fallback if progress module not available
            logger.warning("Progress reporting unavailable")
            show_progress = False

        with ctx.Pool(num_workers) as pool:
            results = pool.imap(
                translate_sequence_worker,
                sequence_iterator(),
                chunksize=chunksize
            )

            # Create progress context if available
            if show_progress:
                pbar = progress.create_sequence_bar(num_sequences, desc="Translating sequences")
                pbar.__enter__()

            try:
                with open(output_nuc, 'w') as nuc_out, open(output_pro, 'w') as pro_out:
                    for result in results:
                        sequences_processed += 1

                        if result:
                            sequences_with_orfs += 1
                            for orf in result:
                                nuc_out.write(f">{orf['seqid']}\n{orf['nucleotide']}\n")
                                pro_out.write(f">{orf['seqid']}\n{orf['protein']}\n")

                        if show_progress:
                            pbar.update(1)
            finally:
                if show_progress:
                    pbar.__exit__(None, None, None)

        logger.info(f"Translation complete: {sequences_processed} sequences processed, "
                   f"{sequences_with_orfs} with valid ORFs")

    except Exception as e:
        logger.exception("Error during parallel translation with progress")
        raise

    return (sequences_processed, sequences_with_orfs)
