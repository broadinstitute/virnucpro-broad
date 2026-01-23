"""Parallel merge orchestration for CPU multiprocessing

This module provides parallel feature merging using multiprocessing.Pool
with spawn context for safe process creation. Designed for parallelizing
the concatenation of DNABERT-S and ESM-2 embeddings across CPU cores.

Based on patterns from:
- parallel_translate.py: Spawn context and worker orchestration (Phase 1.1)
- features.py: merge_features() for single file pair merging
- work_queue.py: Spawn context for multiprocessing safety
"""

import multiprocessing
import os
import logging
from pathlib import Path
from typing import Optional, Tuple, List

logger = logging.getLogger('virnucpro.parallel_merge')


def merge_file_pair_worker(file_pair: Tuple[Path, Path, Path]) -> Optional[Path]:
    """
    Worker function for parallel merge of a single file pair.

    Top-level function (not nested) required for picklability with spawn context.

    Args:
        file_pair: Tuple of (nucleotide_feature_file, protein_feature_file, output_file)

    Returns:
        output_file Path if successful, None if failed
    """
    nuc_file, pro_file, output_file = file_pair

    # Import inside worker to avoid pickling heavy objects
    from virnucpro.pipeline.features import merge_features

    try:
        merge_features(nuc_file, pro_file, output_file)
        return output_file
    except Exception as e:
        logger.error(f"Merge failed for {nuc_file.name}: {e}")
        return None


def merge_batch_worker(file_pair_batch: List[Tuple[Path, Path, Path]]) -> List[Optional[Path]]:
    """
    Worker function for parallel merge of a batch of file pairs.

    Processes multiple file pairs in a single worker call, reducing serialization
    overhead when processing large numbers of files (100+).

    Top-level function (not nested) required for picklability with spawn context.

    Args:
        file_pair_batch: List of (nuc_file, pro_file, output_file) tuples

    Returns:
        List of output files (None for failures)
    """
    # Import inside worker to avoid pickling heavy objects
    from virnucpro.pipeline.features import merge_features

    results = []
    for nuc_file, pro_file, output_file in file_pair_batch:
        try:
            merge_features(nuc_file, pro_file, output_file)
            results.append(output_file)
        except Exception as e:
            logger.error(f"Merge failed for {nuc_file.name}: {e}")
            results.append(None)

    return results


def get_optimal_settings(
    num_workers: Optional[int] = None,
    num_file_pairs: Optional[int] = None
) -> Tuple[int, int, int]:
    """
    Calculate optimal settings for parallel merge.

    Determines number of workers, batch size, and chunksize for Pool.imap()
    based on system resources and number of files to merge.

    Args:
        num_workers: Number of worker processes (default: cpu_count())
        num_file_pairs: Total number of file pairs to merge (optional)

    Returns:
        Tuple of (num_workers, batch_size, chunksize)

    Example:
        >>> workers, batch_size, chunksize = get_optimal_settings(num_file_pairs=50)
        >>> workers == os.cpu_count()
        True
    """
    # Determine number of workers
    if num_workers is None:
        num_workers = os.cpu_count()

    # For merge, batch_size is typically 1 (each file pair is substantial work)
    # Unlike translation where we batch sequences, here each file contains many sequences
    batch_size = 1

    # Calculate chunksize for Pool.imap()
    # Default formula: len(iterable) / (num_workers * 4)
    if num_file_pairs is not None:
        chunksize = max(1, num_file_pairs // (num_workers * 4))
    else:
        # Conservative default when total unknown
        chunksize = 1

    return (num_workers, batch_size, chunksize)


def parallel_merge_features(
    nucleotide_files: List[Path],
    protein_files: List[Path],
    output_dir: Path,
    num_workers: Optional[int] = None
) -> List[Path]:
    """
    Parallelize feature merging across CPU cores.

    Uses spawn context for safety and consistency with GPU workers.
    Processes file pairs independently with Pool.imap() for memory efficiency.

    Args:
        nucleotide_files: List of DNABERT-S feature files (.pt)
        protein_files: List of ESM-2 feature files (.pt)
        output_dir: Directory for merged output files
        num_workers: Number of worker processes (default: cpu_count())

    Returns:
        List of successfully merged output files

    Example:
        >>> merged = parallel_merge_features(
        ...     nuc_files,
        ...     pro_files,
        ...     Path('output/merged'),
        ...     num_workers=8
        ... )
    """
    if num_workers is None:
        num_workers = os.cpu_count()

    # Get optimal chunksize
    _, _, chunksize = get_optimal_settings(
        num_workers=num_workers,
        num_file_pairs=len(nucleotide_files)
    )

    logger.info(f"Starting parallel merge with {num_workers} workers (chunksize={chunksize})")

    # Create file pair tuples for workers
    file_pairs = []
    for nuc_file, pro_file in zip(nucleotide_files, protein_files):
        # Generate output filename from nucleotide filename
        base_name = nuc_file.stem.replace('_DNABERT_S', '')
        output_file = output_dir / f"{base_name}_merged.pt"
        file_pairs.append((nuc_file, pro_file, output_file))

    # Use spawn context for safety (matches Phase 1.1 pattern)
    ctx = multiprocessing.get_context('spawn')

    merged_files = []
    try:
        with ctx.Pool(num_workers) as pool:
            # Use imap() for lazy evaluation and streaming results
            results = pool.imap(
                merge_file_pair_worker,
                file_pairs,
                chunksize=chunksize
            )

            # Collect results as they arrive
            for result in results:
                if result is not None:
                    merged_files.append(result)

        logger.info(f"Merge complete: {len(merged_files)}/{len(file_pairs)} files successful")

    except Exception as e:
        logger.exception("Error during parallel merge")
        raise

    return merged_files


def parallel_merge_with_progress(
    nucleotide_files: List[Path],
    protein_files: List[Path],
    output_dir: Path,
    num_workers: Optional[int] = None,
    show_progress: bool = True
) -> List[Path]:
    """
    Parallel merge with progress reporting.

    Same as parallel_merge_features but integrates with ProgressReporter
    for user-facing progress bar.

    Args:
        nucleotide_files: List of DNABERT-S feature files (.pt)
        protein_files: List of ESM-2 feature files (.pt)
        output_dir: Directory for merged output files
        num_workers: Number of worker processes (default: cpu_count())
        show_progress: Enable progress bar display (default: True)

    Returns:
        List of successfully merged output files
    """
    if num_workers is None:
        num_workers = os.cpu_count()

    # Get optimal chunksize
    _, _, chunksize = get_optimal_settings(
        num_workers=num_workers,
        num_file_pairs=len(nucleotide_files)
    )

    logger.info(f"Starting parallel merge with {num_workers} workers (chunksize={chunksize})")

    # Create file pair tuples for workers
    file_pairs = []
    for nuc_file, pro_file in zip(nucleotide_files, protein_files):
        base_name = nuc_file.stem.replace('_DNABERT_S', '')
        output_file = output_dir / f"{base_name}_merged.pt"
        file_pairs.append((nuc_file, pro_file, output_file))

    ctx = multiprocessing.get_context('spawn')
    merged_files = []

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
                merge_file_pair_worker,
                file_pairs,
                chunksize=chunksize
            )

            # Create progress context if available
            if show_progress:
                with progress.create_file_bar(len(file_pairs), desc="Merging features") as pbar:
                    for result in results:
                        if result is not None:
                            merged_files.append(result)
                        pbar.update(1)
            else:
                # No progress bar - just collect results
                for result in results:
                    if result is not None:
                        merged_files.append(result)

        logger.info(f"Merge complete: {len(merged_files)}/{len(file_pairs)} files successful")

    except Exception as e:
        logger.exception("Error during parallel merge with progress")
        raise

    return merged_files
