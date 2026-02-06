"""GPU worker function for single-GPU shard processing.

This module provides the worker function spawned by GPUProcessCoordinator
for distributed multi-GPU inference. Each worker:
1. Sets up per-worker logging
2. Loads checkpoint manifest and resumes from existing checkpoints
3. Filters sequence index to skip already-processed sequences
4. Loads model (ESM-2 or DNABERT-S)
5. Creates IndexBasedDataset and DataLoader
6. Runs AsyncInferenceRunner with checkpointing enabled
7. Assembles final shard from resumed + new embeddings
8. Saves shard HDF5 file
9. Reports completion to parent

The worker runs independently on a single GPU (mapped to cuda:0 via
CUDA_VISIBLE_DEVICES) and saves results to shard_{rank}.h5 for later
aggregation by the parent process.

Checkpointing support:
- Per-shard isolation via checkpoint_dir/shard_{rank}/ subdirectories
- Resume from existing .pt checkpoints before inference
- Index filtering prevents duplicate processing of resumed sequences
- SIGTERM handler saves emergency checkpoint on spot preemption
- Differentiated error handling: OOM, CUDA runtime, generic errors
"""

import sys
import signal
import logging
import hashlib
from pathlib import Path
from typing import Dict, Any, Set
from multiprocessing import Queue

import torch
import h5py

from virnucpro.pipeline.worker_logging import setup_worker_logging
from virnucpro.data.shard_index import get_worker_indices, load_sequence_index
from virnucpro.data.sequence_dataset import IndexBasedDataset
from virnucpro.data.collators import VarlenCollator
from virnucpro.data.dataloader_utils import create_async_dataloader
from virnucpro.pipeline.async_inference import AsyncInferenceRunner
from virnucpro.utils.precision import should_use_fp16
from virnucpro.pipeline.checkpoint_writer import resume_from_checkpoints


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
            - 'enable_checkpointing': Enable checkpoint support (default: True)
            - 'force_restart': Ignore existing checkpoints (default: False)
            - 'checkpoint_dir': Base checkpoint directory (default: output_dir/checkpoints)
            - 'checkpoint_seq_threshold': Sequences before checkpoint (default: 10000)
            - 'checkpoint_time_threshold': Seconds before checkpoint (default: 300.0)

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
            'num_sequences': int,
            'checkpointing_enabled': bool,
            'resumed_sequences': int
        }
        Failure (OOM): {
            'rank': int,
            'status': 'failed',
            'error': 'cuda_oom',
            'error_message': str,
            'retry_recommended': True,
            'reduce_batch_size': True
        }
        Failure (CUDA runtime): {
            'rank': int,
            'status': 'failed',
            'error': 'cuda_runtime',
            'error_message': str,
            'retry_recommended': True,
            'circuit_breaker': True
        }
        Failure (Generic): {
            'rank': int,
            'status': 'failed',
            'error': str,
            'error_message': str,
            'retry_recommended': bool
        }
    """
    # Step 1: Setup logging FIRST (before any other operations)
    log_dir = output_dir / "logs"
    log_file = setup_worker_logging(rank, log_dir)
    logger = logging.getLogger(f"gpu_worker_{rank}")
    logger.info(f"Worker {rank}/{world_size} starting")
    logger.info(f"Log file: {log_file}")

    # Reference to runner for SIGTERM handler (set later)
    runner = None

    try:
        # Step 2: Initialize CUDA
        device = torch.device('cuda:0')  # Always 0 due to CUDA_VISIBLE_DEVICES
        torch.cuda.set_device(device)
        logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")

        # Step 3: Extract checkpoint configuration
        enable_checkpointing = model_config.get('enable_checkpointing', True)
        force_restart = model_config.get('force_restart', False)
        checkpoint_base_dir = Path(model_config.get('checkpoint_dir', output_dir / "checkpoints"))
        checkpoint_seq_threshold = model_config.get('checkpoint_seq_threshold', 10000)
        checkpoint_time_threshold = model_config.get('checkpoint_time_threshold', 300.0)

        # Per-shard checkpoint isolation (prevents cross-GPU conflicts)
        checkpoint_dir = checkpoint_base_dir / f"shard_{rank}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"Rank {rank}: Checkpointing enabled={enable_checkpointing}, "
            f"dir={checkpoint_dir}, force_restart={force_restart}"
        )

        # Step 4: Load manifest (if checkpointing enabled)
        manifest = None
        if enable_checkpointing:
            from virnucpro.pipeline.checkpoint_manifest import CheckpointManifest
            manifest_path = checkpoint_base_dir / "manifest.json"
            if manifest_path.exists():
                manifest = CheckpointManifest(manifest_path)
                logger.info(f"Rank {rank}: Using checkpoint manifest at {manifest_path}")
            else:
                # First run or fresh start - coordinator will create manifest
                logger.info(f"Rank {rank}: No existing manifest, checkpointing without manifest coordination")

        # Step 5: Resume from checkpoints (before loading index)
        resumed_ids: Set[str] = set()
        resumed_embeddings = None
        if enable_checkpointing and not force_restart:
            logger.info(f"Rank {rank}: Checking for existing checkpoints in {checkpoint_dir}")
            resumed_ids_list, resumed_embs, resume_batch_idx, corrupted_ids = resume_from_checkpoints(
                checkpoint_dir, rank, force_restart
            )

            if corrupted_ids:
                logger.warning(
                    f"Rank {rank}: Checkpoint corruption detected - {len(corrupted_ids)} sequences "
                    f"need reprocessing (from batches after corruption point)"
                )
                # Corrupted sequences will be reprocessed (not in resumed_ids)

            if resumed_ids_list:
                resumed_ids = set(resumed_ids_list)
                resumed_embeddings = resumed_embs  # numpy array
                logger.info(
                    f"Rank {rank}: Resuming from {resume_batch_idx} checkpoints, "
                    f"{len(resumed_ids)} sequences already processed"
                )

        # Step 6: Load sequence index and filter out already-processed sequences
        logger.info(f"Loading index: {index_path}")
        indices = get_worker_indices(index_path, rank, world_size)

        # Filter out already-processed sequences (prevents duplicates)
        if resumed_ids:
            original_count = len(indices)
            # Load full index to access sequence metadata
            index_data = load_sequence_index(index_path)
            sequences = index_data['sequences']

            # Create filtered indices containing only unprocessed sequences
            filtered_indices = [
                i for i in indices
                if sequences[i]['sequence_id'] not in resumed_ids
            ]

            logger.info(
                f"Rank {rank}: Filtered index from {original_count} to {len(filtered_indices)} sequences "
                f"({len(resumed_ids)} already processed)"
            )
            indices = filtered_indices
        else:
            logger.info(f"Assigned {len(indices)} sequences")

        # Step 7: Load model
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

        # Step 8: Create dataset and dataloader
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

        # Step 9: Compute input fingerprint for cross-run validation
        input_fingerprint = hashlib.sha256(str(index_path).encode()).hexdigest()[:16]

        # Step 10: Create AsyncInferenceRunner with checkpoint configuration
        logger.info("Starting inference")
        runner = AsyncInferenceRunner(
            model, device,
            checkpoint_dir=checkpoint_dir if enable_checkpointing else None,
            rank=rank,
            checkpoint_seq_threshold=checkpoint_seq_threshold,
            checkpoint_time_threshold=checkpoint_time_threshold,
            manifest=manifest,
            input_fingerprint=input_fingerprint,
        )

        # Step 11: Register SIGTERM handler for spot preemption
        if enable_checkpointing:
            def sigterm_handler(signum, frame):
                logger.warning(f"Rank {rank}: SIGTERM received (spot preemption), saving emergency checkpoint")
                if runner and runner._checkpointing_enabled:
                    # Trigger emergency checkpoint
                    runner._write_checkpoint(reason="emergency_sigterm")
                    runner.writer.wait_all(timeout=30)
                sys.exit(143)  # Standard SIGTERM exit code

            signal.signal(signal.SIGTERM, sigterm_handler)
            logger.info(f"Rank {rank}: SIGTERM handler registered for spot instance support")

        # Step 12: Run async inference
        all_embeddings = []
        all_ids = []

        batch_idx = 0
        for result in runner.run(dataloader):
            # Skip resumed data marker (batch_idx == -1)
            if result.batch_idx == -1:
                logger.info(f"Rank {rank}: Yielded resumed data ({len(result.sequence_ids)} sequences)")
                continue

            batch_idx += 1
            all_embeddings.append(result.embeddings)
            all_ids.extend(result.sequence_ids)

        logger.info(f"Inference complete: {len(all_ids)} new sequences")

        # Step 13: Assemble final shard from resumed + new embeddings
        if resumed_embeddings is not None:
            logger.info(f"Rank {rank}: Merging {len(resumed_ids)} resumed + {len(all_ids)} new sequences")
            # Resumed embeddings are numpy, new are torch tensors
            all_embeddings.insert(0, torch.from_numpy(resumed_embeddings))
            all_ids = list(resumed_ids) + all_ids

        logger.info(f"Final shard: {len(all_ids)} total sequences")

        # Step 14: Save shard HDF5
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

        # Step 15: Report success
        result_status = {
            'rank': rank,
            'status': 'complete',
            'shard_path': str(shard_path),
            'num_sequences': len(all_ids),
            'checkpointing_enabled': enable_checkpointing,
            'resumed_sequences': len(resumed_ids) if resumed_ids else 0,
        }
        results_queue.put(result_status)
        logger.info(f"Worker {rank} completed successfully")

    except RuntimeError as e:
        error_str = str(e)

        # Check for numerical instability first (Phase 8 FP16 validation)
        if "Numerical instability" in error_str:
            # Log which sequences failed for debugging
            logger.error(
                f"Rank {rank}: NaN/Inf detected in batch {batch_idx}. "
                f"Error: {e}"
            )
            # Report failure to orchestrator
            result_status = {
                "rank": rank,
                "status": "numerical_instability",
                "failed_batch": batch_idx,
                "error": error_str
            }
        # Check for OOM (OutOfMemoryError is a RuntimeError subclass)
        elif 'out of memory' in error_str.lower() or 'oom' in error_str.lower():
            # OOM: Reduce batch size and retry
            try:
                mem_allocated = torch.cuda.memory_allocated(device) / 1e9
                mem_peak = torch.cuda.max_memory_allocated(device) / 1e9
                mem_info = f"GPU memory: {mem_allocated:.2f}GB allocated, {mem_peak:.2f}GB peak"
            except (AttributeError, TypeError):
                # CUDA mocked or unavailable
                mem_info = "GPU memory info unavailable"

            logger.error(f"Rank {rank}: CUDA OOM error - {e}\n{mem_info}")
            result_status = {
                'rank': rank,
                'status': 'failed',
                'error': error_str,  # Backward compatibility: error contains message
                'error_type': 'cuda_oom',  # New: categorized error type
                'retry_recommended': True,
                'reduce_batch_size': True,
            }
        elif 'CUDA' in error_str or 'assert' in error_str.lower():
            # CUDA error or assertion - likely poison input
            logger.error(
                f"Rank {rank}: CUDA runtime error (possible poison input) - {e}\n"
                f"Last batch info: {len(all_ids)} sequences processed so far"
            )
            result_status = {
                'rank': rank,
                'status': 'failed',
                'error': error_str,  # Backward compatibility: error contains message
                'error_type': 'cuda_runtime',  # New: categorized error type
                'retry_recommended': True,
                'circuit_breaker': True,  # Trigger circuit breaker after 2 attempts
            }
        else:
            # Generic runtime error
            logger.error(f"Rank {rank}: Runtime error - {e}", exc_info=True)
            result_status = {
                'rank': rank,
                'status': 'failed',
                'error': error_str,  # Backward compatibility: error contains message
                'retry_recommended': True,
            }
        results_queue.put(result_status)
        sys.exit(1)

    except Exception as e:
        # Unexpected error - full diagnostics
        logger.error(
            f"Rank {rank}: Unexpected error during inference",
            exc_info=True
        )
        result_status = {
            'rank': rank,
            'status': 'failed',
            'error': str(e),  # Backward compatibility: error contains message
        }
        results_queue.put(result_status)
        sys.exit(1)
