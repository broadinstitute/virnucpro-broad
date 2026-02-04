"""Multi-GPU inference orchestration entry point.

This module provides the high-level run_multi_gpu_inference function that
orchestrates the full workflow from FASTA files to merged embeddings.

Workflow:
1. Create/validate sequence index
2. Spawn GPU workers via GPUProcessCoordinator
3. Wait for worker completion
4. Aggregate shards from successful workers
5. Validate output completeness

The function supports partial failure - if some workers fail but others succeed,
it returns results from successful workers with warnings about failures.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Set

import torch

from virnucpro.data.shard_index import create_sequence_index, load_sequence_index, get_worker_indices
from virnucpro.pipeline.gpu_coordinator import GPUProcessCoordinator
from virnucpro.pipeline.gpu_worker import gpu_worker
from virnucpro.pipeline.shard_aggregator import aggregate_shards

logger = logging.getLogger('virnucpro.pipeline.multi_gpu_inference')


def run_multi_gpu_inference(
    fasta_files: List[Path],
    output_dir: Path,
    model_config: Dict[str, Any],
    world_size: Optional[int] = None,
    timeout: Optional[float] = None
) -> Tuple[Path, List[int]]:
    """
    Run multi-GPU inference with fault tolerance.

    Orchestrates:
    1. Create/validate sequence index
    2. Spawn GPU workers via GPUProcessCoordinator
    3. Wait for worker completion
    4. Aggregate shards from successful workers
    5. Validate output completeness

    Args:
        fasta_files: List of FASTA file paths to process
        output_dir: Directory for outputs (index, shards, logs, merged result)
        model_config: Dict with 'model_type', 'model_name', 'token_budget' etc.
        world_size: Number of GPUs (default: torch.cuda.device_count())
        timeout: Max seconds to wait per worker (default: None = infinite)

    Returns:
        Tuple of (merged_output_path, failed_ranks)
        - merged_output_path: Path to merged embeddings.h5
        - failed_ranks: List of worker ranks that failed (empty if all succeeded)

    Raises:
        RuntimeError: If no workers completed successfully
        ValueError: If output validation fails (missing/duplicate sequences)

    Example:
        >>> from pathlib import Path
        >>> fasta_files = [Path("sequences.fasta")]
        >>> output_dir = Path("output")
        >>> model_config = {'model_type': 'esm2', 'model_name': 'esm2_t36_3B_UR50D'}
        >>> output_path, failed = run_multi_gpu_inference(fasta_files, output_dir, model_config)
        >>> if failed:
        ...     print(f"Warning: {len(failed)} workers failed")
        >>> print(f"Embeddings saved to: {output_path}")
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Auto-detect world_size if not specified
    if world_size is None:
        world_size = torch.cuda.device_count()
    logger.info(f"Starting multi-GPU inference: {world_size} GPUs")

    # Step 1: Create/validate sequence index
    logger.info("Creating sequence index...")
    index_path = output_dir / "sequence_index.json"
    create_sequence_index(fasta_files, index_path)

    # Load expected sequence IDs for validation
    index_data = load_sequence_index(index_path)
    expected_ids = {s['sequence_id'] for s in index_data['sequences']}
    logger.info(f"Index: {len(expected_ids)} sequences, {index_data['total_tokens']} tokens")

    # Step 2: Spawn workers
    logger.info("Spawning GPU workers...")
    coordinator = GPUProcessCoordinator(world_size, output_dir)
    coordinator.spawn_workers(
        gpu_worker,
        (index_path, output_dir, model_config)
    )

    # Step 3: Wait for completion
    logger.info(f"Waiting for workers to complete (timeout={timeout})...")
    completion_status = coordinator.wait_for_completion(timeout=timeout)

    # Identify successful and failed workers
    successful_ranks = [r for r, ok in completion_status.items() if ok]
    failed_ranks = [r for r, ok in completion_status.items() if not ok]

    if failed_ranks:
        logger.warning(
            f"Partial failure: {len(failed_ranks)}/{world_size} workers failed: {failed_ranks}\n"
            f"Successful workers: {successful_ranks}"
        )

    if not successful_ranks:
        raise RuntimeError(
            f"No workers completed successfully. Check logs: {output_dir / 'logs' / 'worker_*.log'}"
        )

    # Step 4: Collect successful shards
    shard_files = [
        output_dir / f"shard_{rank}.h5"
        for rank in successful_ranks
        if (output_dir / f"shard_{rank}.h5").exists()
    ]

    if not shard_files:
        raise RuntimeError("No shard files found from successful workers")

    logger.info(f"Found {len(shard_files)} shard files from successful workers")

    # Step 5: Calculate expected IDs for successful workers only
    if failed_ranks:
        # Partial validation - only check IDs from successful workers
        successful_expected: Set[str] = set()
        for rank in successful_ranks:
            indices = get_worker_indices(index_path, rank, world_size)
            for i in indices:
                successful_expected.add(index_data['sequences'][i]['sequence_id'])
        expected_for_validation = successful_expected
        logger.warning(
            f"Validating partial results: {len(successful_expected)} sequences "
            f"(missing {len(expected_ids) - len(successful_expected)} due to worker failures)"
        )
    else:
        expected_for_validation = expected_ids

    # Step 6: Aggregate shards
    logger.info("Aggregating shards...")
    output_path = output_dir / "embeddings.h5"
    aggregate_shards(shard_files, output_path, expected_for_validation)

    # Step 7: Report results
    if failed_ranks:
        missing_count = len(expected_ids) - len(expected_for_validation)
        logger.warning(
            f"Completed with partial results: {len(expected_for_validation)} sequences embedded\n"
            f"Missing {missing_count} sequences from failed workers: {failed_ranks}\n"
            f"Check logs: {output_dir / 'logs' / 'worker_*.log'}"
        )
    else:
        logger.info(f"Multi-GPU inference complete: {output_path}")

    return output_path, failed_ranks


def run_esm2_multi_gpu(
    fasta_files: List[Path],
    output_dir: Path,
    world_size: Optional[int] = None,
    model_name: str = 'esm2_t36_3B_UR50D'
) -> Path:
    """
    Convenience wrapper for ESM-2 multi-GPU inference.

    This is a simplified interface for the common case of running ESM-2 inference
    with default settings. Raises an error if any workers fail.

    Args:
        fasta_files: List of FASTA file paths to process
        output_dir: Directory for outputs
        world_size: Number of GPUs (default: all available)
        model_name: ESM-2 model variant (default: esm2_t36_3B_UR50D)

    Returns:
        Path to merged embeddings.h5 file

    Raises:
        RuntimeError: If any workers fail

    Example:
        >>> from pathlib import Path
        >>> fasta_files = [Path("proteins.fasta")]
        >>> output_path = run_esm2_multi_gpu(fasta_files, Path("output"))
        >>> print(f"Embeddings: {output_path}")
    """
    model_config = {
        'model_type': 'esm2',
        'model_name': model_name,
        'enable_fp16': True
    }

    output_path, failed_ranks = run_multi_gpu_inference(
        fasta_files, output_dir, model_config, world_size=world_size
    )

    if failed_ranks:
        raise RuntimeError(
            f"ESM-2 inference failed on {len(failed_ranks)} workers: {failed_ranks}\n"
            f"Check logs: {output_dir / 'logs' / 'worker_*.log'}"
        )

    return output_path
