"""GPU worker function for single-GPU shard processing.

This module provides the worker function spawned by GPUProcessCoordinator
for distributed multi-GPU inference. Each worker:
1. Sets up per-worker logging
2. Loads sequence index and gets assigned indices
3. Loads model (ESM-2 or DNABERT-S)
4. Creates IndexBasedDataset and DataLoader
5. Runs AsyncInferenceRunner
6. Saves shard HDF5 file
7. Reports completion to parent

The worker runs independently on a single GPU (mapped to cuda:0 via
CUDA_VISIBLE_DEVICES) and saves results to shard_{rank}.h5 for later
aggregation by the parent process.
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Any
from multiprocessing import Queue

import torch
import h5py

from virnucpro.pipeline.worker_logging import setup_worker_logging
from virnucpro.data.shard_index import get_worker_indices
from virnucpro.data.sequence_dataset import IndexBasedDataset
from virnucpro.data.collators import VarlenCollator
from virnucpro.data.dataloader_utils import create_async_dataloader
from virnucpro.pipeline.async_inference import AsyncInferenceRunner
from virnucpro.utils.precision import should_use_fp16


def gpu_worker(
    rank: int,
    world_size: int,
    results_queue: Queue,
    index_path: Path,
    output_dir: Path,
    model_config: Dict[str, Any]
) -> None:
    """
    Independent GPU worker for shard processing.

    Spawned by GPUProcessCoordinator. Each worker runs on a dedicated GPU
    (via CUDA_VISIBLE_DEVICES set by parent) and processes an assigned
    subset of sequences using stride distribution [rank, rank+N, rank+2N, ...].

    The worker:
    1. Sets up per-worker logging (FIRST - before any other operations)
    2. Initializes CUDA on device 0 (remapped by CUDA_VISIBLE_DEVICES)
    3. Loads sequence index and gets assigned indices
    4. Loads model (ESM-2 or DNABERT-S)
    5. Creates IndexBasedDataset and async DataLoader
    6. Runs AsyncInferenceRunner to process all sequences
    7. Saves embeddings to shard_{rank}.h5
    8. Reports completion status to parent via results_queue

    Args:
        rank: Worker rank (0 to world_size-1)
        world_size: Total number of workers
        results_queue: Queue for reporting status to parent
        index_path: Path to sequence index JSON
        output_dir: Directory for shard output and logs
        model_config: Dict with model configuration:
            - 'model_type': 'esm2' or 'dnabert' (required)
            - 'model_name': ESM-2 model variant (default: 'esm2_t36_3B_UR50D')
            - 'token_budget': Max tokens per batch (default: None for auto)
            - 'enable_fp16': Use FP16 precision (default: True)
                Note: VIRNUCPRO_DISABLE_FP16 env var overrides this for safety-critical rollback

    Note:
        CUDA_VISIBLE_DEVICES already set by parent process.
        Worker always sees device 0, which maps to actual GPU {rank}.

    Side Effects:
        - Writes log to {output_dir}/logs/worker_{rank}.log
        - Writes shard to {output_dir}/shard_{rank}.h5
        - Puts status dict to results_queue on completion/failure

    Status Dict Format:
        Success: {
            'rank': int,
            'status': 'complete',
            'shard_path': str,
            'num_sequences': int
        }
        Failure: {
            'rank': int,
            'status': 'failed',
            'error': str
        }
    """
    # Step 1: Setup logging FIRST (before any other operations)
    log_dir = output_dir / "logs"
    log_file = setup_worker_logging(rank, log_dir)
    logger = logging.getLogger(f"gpu_worker_{rank}")
    logger.info(f"Worker {rank}/{world_size} starting")
    logger.info(f"Log file: {log_file}")

    try:
        # Step 2: Initialize CUDA
        device = torch.device('cuda:0')  # Always 0 due to CUDA_VISIBLE_DEVICES
        torch.cuda.set_device(device)
        logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")

        # Step 3: Load sequence index and get assigned indices
        logger.info(f"Loading index: {index_path}")
        indices = get_worker_indices(index_path, rank, world_size)
        logger.info(f"Assigned {len(indices)} sequences")

        # Step 4: Load model
        model_type = model_config['model_type']
        logger.info(f"Loading model: {model_type}")

        if model_type == 'esm2':
            from virnucpro.models.esm2_flash import load_esm2_model

            # Environment variable takes precedence for safety-critical rollback
            # If VIRNUCPRO_DISABLE_FP16=1 is set, it overrides config
            if not should_use_fp16():
                enable_fp16 = False
            else:
                enable_fp16 = model_config.get('enable_fp16', True)

            model, batch_converter = load_esm2_model(
                model_name=model_config.get('model_name', 'esm2_t36_3B_UR50D'),
                device=str(device),
                enable_fp16=enable_fp16
            )
            logger.info(
                f"ESM-2 model loaded: {model_config.get('model_name', 'esm2_t36_3B_UR50D')}"
            )
        elif model_type == 'dnabert':
            from virnucpro.models.dnabert_flash import load_dnabert_model

            # Environment variable takes precedence (same as ESM-2)
            if not should_use_fp16():
                enable_fp16 = False
            else:
                enable_fp16 = model_config.get('enable_fp16', True)

            model, tokenizer = load_dnabert_model(
                device=str(device),
                enable_fp16=enable_fp16
            )
            # DNABERT uses tokenizer directly, not batch_converter
            batch_converter = tokenizer
            logger.info("DNABERT-S model loaded")
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        # Step 5: Create dataset and dataloader
        logger.info("Creating dataset and dataloader")
        dataset = IndexBasedDataset(index_path, indices)
        collator = VarlenCollator(batch_converter)
        dataloader = create_async_dataloader(
            dataset, collator,
            device_id=0,  # Always 0 due to CUDA_VISIBLE_DEVICES remapping
            token_budget=model_config.get('token_budget', None)
        )
        logger.info(
            f"DataLoader created: {len(dataset)} sequences, "
            f"token_budget={model_config.get('token_budget', 'auto')}"
        )

        # Step 6: Run async inference
        logger.info("Starting inference")
        runner = AsyncInferenceRunner(model, device)
        all_embeddings = []
        all_ids = []

        batch_idx = 0
        try:
            for result in runner.run(dataloader):
                all_embeddings.append(result.embeddings)
                all_ids.extend(result.sequence_ids)
                batch_idx += 1
        except RuntimeError as e:
            if "Numerical instability" in str(e):
                # Log which sequences failed for debugging
                logger.error(
                    f"Rank {rank}: NaN/Inf detected in batch {batch_idx}. "
                    f"Error: {e}"
                )
                # Report failure to orchestrator
                results_queue.put({
                    "rank": rank,
                    "status": "numerical_instability",
                    "failed_batch": batch_idx,
                    "error": str(e)
                })
                # Fail this worker - orchestrator handles partial results (Phase 7 pattern)
                sys.exit(1)
            else:
                # Re-raise unexpected errors
                raise

        logger.info(f"Inference complete: {len(all_ids)} sequences")

        # Step 7: Save shard HDF5
        shard_path = output_dir / f"shard_{rank}.h5"
        logger.info(f"Saving shard: {shard_path}")

        with h5py.File(shard_path, 'w') as f:
            # Stack embeddings and convert to numpy
            if all_embeddings:
                embeddings = torch.cat(all_embeddings, dim=0).numpy()
                f.create_dataset('embeddings', data=embeddings)
            else:
                # Empty shard - create empty dataset
                f.create_dataset('embeddings', shape=(0, 0), dtype='float32')

            # Save sequence IDs with variable-length string dtype
            dt = h5py.special_dtype(vlen=str)
            f.create_dataset('sequence_ids', data=all_ids, dtype=dt)

        logger.info(f"Shard saved: {shard_path} ({len(all_ids)} sequences)")

        # Step 8: Report success
        result_status = {
            'rank': rank,
            'status': 'complete',
            'shard_path': str(shard_path),
            'num_sequences': len(all_ids)
        }
        results_queue.put(result_status)
        logger.info(f"Worker {rank} completed successfully")

    except Exception as e:
        logger.exception(f"Worker {rank} failed: {e}")
        result_status = {
            'rank': rank,
            'status': 'failed',
            'error': str(e)
        }
        results_queue.put(result_status)
        sys.exit(1)
