"""HDF5 shard aggregation with chunk-wise streaming and validation.

This module aggregates HDF5 shard files produced by multi-GPU workers into
a single output file. Uses chunk-wise streaming to control memory usage and
validates completeness (no duplicates, all expected sequences present).

Architecture:
- Workers write shard_N.h5 files independently
- After workers complete, aggregate_shards merges into embeddings.h5
- Chunk-wise reading prevents memory overflow on large datasets
- Validation ensures data integrity (no duplicates, no missing IDs)
"""

import h5py
import logging
from typing import List, Set, Optional, Dict, Tuple
from pathlib import Path

logger = logging.getLogger('virnucpro.pipeline.shard_aggregator')

# Chunk size for memory-controlled streaming
CHUNK_SIZE = 10000  # Sequences per chunk


def get_shard_info(shard_path: Path) -> Dict:
    """Get metadata from shard file.

    Args:
        shard_path: Path to shard HDF5 file

    Returns:
        Dict with num_sequences, embedding_dim, sequence_ids_sample
    """
    with h5py.File(shard_path, 'r') as f:
        num_sequences = f['embeddings'].shape[0]
        embedding_dim = f['embeddings'].shape[1]

        # Get sample of sequence IDs (first 5)
        sequence_ids = f['sequence_ids'][:min(5, num_sequences)]
        # Decode bytes to strings
        sequence_ids_sample = [sid.decode('utf-8') if isinstance(sid, bytes) else sid
                              for sid in sequence_ids]

    return {
        'num_sequences': num_sequences,
        'embedding_dim': embedding_dim,
        'sequence_ids_sample': sequence_ids_sample
    }


def validate_shard_completeness(
    shard_files: List[Path],
    expected_ids: Set[str]
) -> Tuple[Set[str], Set[str]]:
    """Check shard files for missing and extra sequence IDs.

    Quick validation without full aggregation - useful for pre-check.

    Args:
        shard_files: List of shard HDF5 file paths
        expected_ids: Set of expected sequence IDs

    Returns:
        Tuple of (missing_ids, extra_ids)
    """
    seen_ids = set()

    for shard_path in shard_files:
        with h5py.File(shard_path, 'r') as f:
            sequence_ids = f['sequence_ids'][:]
            # Decode bytes to strings if necessary
            for sid in sequence_ids:
                if isinstance(sid, bytes):
                    seen_ids.add(sid.decode('utf-8'))
                else:
                    seen_ids.add(sid)

    missing_ids = expected_ids - seen_ids
    extra_ids = seen_ids - expected_ids

    return (missing_ids, extra_ids)


def aggregate_shards(
    shard_files: List[Path],
    output_path: Path,
    expected_sequence_ids: Optional[Set[str]] = None
) -> Path:
    """Aggregate HDF5 shards into single output with validation.

    Uses chunk-wise streaming to control memory usage. Validates:
    - No duplicate sequence IDs across shards
    - All expected sequence IDs present (if provided)

    Args:
        shard_files: List of shard HDF5 file paths
        output_path: Path for merged output HDF5
        expected_sequence_ids: Optional set of expected IDs for validation

    Returns:
        Path to merged output file

    Raises:
        ValueError: If duplicates found or sequences missing
    """
    if not shard_files:
        raise ValueError("No shard files provided")

    # Count total sequences and get embedding dimension
    total_sequences = 0
    embedding_dim = None
    seen_ids: Set[str] = set()

    logger.info(f"Analyzing {len(shard_files)} shard files...")

    for shard_path in shard_files:
        info = get_shard_info(shard_path)
        total_sequences += info['num_sequences']

        if embedding_dim is None:
            embedding_dim = info['embedding_dim']
        elif embedding_dim != info['embedding_dim']:
            raise ValueError(
                f"Inconsistent embedding dimensions: {embedding_dim} vs {info['embedding_dim']} "
                f"in {shard_path}"
            )

    logger.info(
        f"Total sequences: {total_sequences}, embedding_dim: {embedding_dim}"
    )

    # Create output HDF5 with pre-allocated datasets
    try:
        with h5py.File(output_path, 'w') as out_f:
            # Create chunked datasets for efficient partial I/O
            out_f.create_dataset(
                'embeddings',
                shape=(total_sequences, embedding_dim),
                dtype='float32',
                chunks=True
            )

            # Variable-length string dataset for sequence IDs
            string_dtype = h5py.special_dtype(vlen=str)
            out_f.create_dataset(
                'sequence_ids',
                shape=(total_sequences,),
                dtype=string_dtype
            )

            write_offset = 0

            # Process each shard sequentially
            for shard_idx, shard_path in enumerate(shard_files):
                logger.info(f"Processing shard {shard_idx + 1}/{len(shard_files)}: {shard_path}")

                with h5py.File(shard_path, 'r') as shard_f:
                    shard_size = shard_f['embeddings'].shape[0]

                    # Process shard in chunks to control memory
                    for chunk_start in range(0, shard_size, CHUNK_SIZE):
                        chunk_end = min(chunk_start + CHUNK_SIZE, shard_size)
                        chunk_size = chunk_end - chunk_start

                        # Read chunk
                        embeddings_chunk = shard_f['embeddings'][chunk_start:chunk_end]
                        sequence_ids_chunk = shard_f['sequence_ids'][chunk_start:chunk_end]

                        # Decode and check for duplicates
                        for sid in sequence_ids_chunk:
                            if isinstance(sid, bytes):
                                sid_str = sid.decode('utf-8')
                            else:
                                sid_str = sid

                            if sid_str in seen_ids:
                                raise ValueError(
                                    f"Duplicate sequence ID found: {sid_str} "
                                    f"in shard {shard_path}"
                                )
                            seen_ids.add(sid_str)

                        # Write chunk to output
                        out_f['embeddings'][write_offset:write_offset + chunk_size] = embeddings_chunk
                        out_f['sequence_ids'][write_offset:write_offset + chunk_size] = sequence_ids_chunk

                        write_offset += chunk_size

            # Validate completeness if expected IDs provided
            if expected_sequence_ids is not None:
                missing_ids = expected_sequence_ids - seen_ids
                extra_ids = seen_ids - expected_sequence_ids

                if missing_ids:
                    # Show first 10 missing IDs for debugging
                    missing_sample = sorted(missing_ids)[:10]
                    raise ValueError(
                        f"Missing {len(missing_ids)} expected sequences. "
                        f"First 10 missing IDs: {missing_sample}"
                    )

                if extra_ids:
                    # Extra IDs are a warning, not an error (allows debugging sequences)
                    logger.warning(
                        f"Found {len(extra_ids)} extra sequences not in expected set. "
                        f"This may indicate test/debug sequences."
                    )

        logger.info(
            f"Aggregation complete: {output_path}, {total_sequences} sequences"
        )
        return output_path

    except Exception as e:
        # Clean up partial output on failure
        if output_path.exists():
            logger.warning(f"Removing partial output file: {output_path}")
            output_path.unlink()
        raise
