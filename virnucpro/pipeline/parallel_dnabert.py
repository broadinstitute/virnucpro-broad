"""DNABERT-S specific multiprocessing utilities for parallel feature extraction

This module implements parallel DNABERT-S processing following the same patterns as ESM-2.
Key features:
- Token-based batching (DNA bases treated as tokens)
- BF16 mixed precision on Ampere+ GPUs
- Spawn context multiprocessing for CUDA safety
- Greedy bin-packing file assignment by sequence count for balanced GPU utilization
"""

import torch
from torch.cuda.amp import autocast
from pathlib import Path
from typing import List, Tuple
from Bio import SeqIO
import logging

logger = logging.getLogger('virnucpro.parallel_dnabert')

# Import base worker utilities
# assign_files_by_sequences uses greedy bin-packing algorithm to distribute
# files across workers based on sequence count (not file count) for balanced work
from virnucpro.pipeline.base_worker import (
    BaseEmbeddingWorker,
    count_sequences,
    assign_files_by_sequences,  # Bin-packing file assignment
    detect_bf16_support
)
from virnucpro.core.logging_setup import setup_worker_logging


def _get_progress_queue():
    """
    Get the progress queue from work_queue module.

    Returns None if queue not initialized (when running without progress reporting).
    """
    try:
        from virnucpro.pipeline.work_queue import _worker_progress_queue
        return _worker_progress_queue
    except ImportError:
        return None


def process_dnabert_files_worker(
    file_subset: List[Path],
    device_id: int,
    toks_per_batch: int = 2048,
    output_dir: Path = None,
    **kwargs
) -> Tuple[List[Path], List[Tuple[Path, str]]]:
    """
    Worker function to process DNABERT-S features on a specific GPU.

    This function is called by multiprocessing Pool workers. Each worker
    loads the DNABERT-S model on its assigned GPU and processes all files
    in its subset. Deferred CUDA initialization ensures no parent process
    CUDA context issues.

    Implements token-based batching where DNA sequence length is treated
    as token count (abstracting k-mer complexity). Automatically enables
    BF16 mixed precision on Ampere+ GPUs for memory efficiency.

    Args:
        file_subset: List of DNA FASTA files to process
        device_id: CUDA device ID (e.g., 0 for cuda:0)
        toks_per_batch: Tokens per batch for DNABERT-S processing (default: 2048)
        output_dir: Directory where output files should be saved
        **kwargs: Additional arguments (log_level, log_format)

    Returns:
        Tuple of (processed_files, failed_files)
        - processed_files: List of successfully processed output .pt file paths
        - failed_files: List of (file_path, error_message) tuples for failures

    Raises:
        Exception: Critical errors that prevent worker startup (logged with device context)
    """
    # Initialize logging in worker process
    log_level = kwargs.get('log_level', logging.INFO)
    log_format = kwargs.get('log_format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    setup_worker_logging(log_level, log_format)

    # Get progress queue from module global (set by Pool initializer)
    progress_queue = _get_progress_queue()

    processed_files = []
    failed_files = []

    try:
        # Deferred CUDA initialization - happens only in worker process
        device = torch.device(f'cuda:{device_id}')
        logger.info(f"Worker {device_id}: Initializing on {device}, processing {len(file_subset)} files")

        # Load DNABERT-S model (defer to worker to avoid parent CUDA context)
        from transformers import AutoTokenizer, AutoModel

        model_name = "zhihan1996/DNABERT-S"
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
        model.eval()

        # Check for BF16 support and adjust batch size
        use_bf16 = detect_bf16_support(device)
        if use_bf16 and toks_per_batch == 2048:
            toks_per_batch = 3072  # Increase batch size with BF16
            logger.info(f"Worker {device_id}: Increased toks_per_batch to {toks_per_batch} with BF16")

        # Wrap all inference in torch.no_grad() context
        with torch.no_grad():
            for file in file_subset:
                try:
                    output_file = output_dir / f"{file.stem}_DNABERT.pt"

                    logger.info(f"Worker {device_id}: Processing {file.name}")

                    # Load all sequences from file
                    records = list(SeqIO.parse(file, 'fasta'))

                    # Create batches by token count (sequence length)
                    batches = []
                    current_batch = []
                    current_tokens = 0

                    for record in records:
                        seq_str = str(record.seq)
                        seq_tokens = len(seq_str)  # For DNA, each base = ~1 token

                        if current_tokens + seq_tokens > toks_per_batch and current_batch:
                            batches.append(current_batch)
                            current_batch = []
                            current_tokens = 0

                        current_batch.append(record)
                        current_tokens += seq_tokens

                    if current_batch:
                        batches.append(current_batch)

                    logger.debug(f"Worker {device_id}: Created {len(batches)} batches for {file.name}")

                    # Process batches with BF16 if supported
                    nucleotide = []
                    data = []

                    with autocast(dtype=torch.bfloat16, enabled=use_bf16):
                        for batch_idx, batch_records in enumerate(batches):
                            batch_seqs = [str(record.seq) for record in batch_records]
                            batch_labels = [record.id for record in batch_records]

                            # Tokenize batch with padding
                            inputs = tokenizer(batch_seqs, return_tensors='pt', padding=True)
                            input_ids = inputs["input_ids"].to(device)
                            attention_mask = inputs.get("attention_mask", None)
                            if attention_mask is not None:
                                attention_mask = attention_mask.to(device)

                            # Forward pass - model returns tuple, take first element
                            if attention_mask is not None:
                                hidden_states = model(input_ids, attention_mask=attention_mask)[0]
                            else:
                                hidden_states = model(input_ids)[0]

                            # Mean pool with attention mask to exclude padding
                            if attention_mask is not None:
                                embedding_means = (hidden_states * attention_mask.unsqueeze(-1)).sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
                            else:
                                embedding_means = torch.mean(hidden_states, dim=1)

                            # Store results (convert BF16 to FP32 for compatibility)
                            for label, embedding_mean in zip(batch_labels, embedding_means):
                                result = {"label": label, "mean_representation": embedding_mean.float().cpu().tolist()}
                                nucleotide.append(label)
                                data.append(result)

                            logger.debug(f"Worker {device_id}: Processed batch {batch_idx + 1}/{len(batches)}")

                    # Save to file in original format
                    torch.save({'nucleotide': nucleotide, 'data': data}, output_file)

                    processed_files.append(output_file)
                    logger.info(f"Worker {device_id}: Completed {file.name} -> {output_file.name} ({len(data)} sequences)")

                    # Report progress if queue available
                    if progress_queue is not None:
                        progress_queue.put({
                            'gpu_id': device_id,
                            'file': str(file),
                            'status': 'complete'
                        })

                except RuntimeError as e:
                    # Handle OOM and other CUDA errors
                    error_msg = str(e)
                    if "out of memory" in error_msg.lower():
                        logger.error(f"Worker {device_id}: OOM error on {file.name}")
                        torch.cuda.empty_cache()  # Clear cache and continue
                    else:
                        logger.error(f"Worker {device_id}: CUDA error on {file.name}: {error_msg}")

                    failed_files.append((file, error_msg))

                    # Report failure if queue available
                    if progress_queue is not None:
                        progress_queue.put({
                            'gpu_id': device_id,
                            'file': str(file),
                            'status': 'failed'
                        })

                except Exception as e:
                    # Handle other errors
                    error_msg = str(e)
                    logger.exception(f"Worker {device_id}: Error processing {file.name}")
                    failed_files.append((file, error_msg))

                    # Report failure if queue available
                    if progress_queue is not None:
                        progress_queue.put({
                            'gpu_id': device_id,
                            'file': str(file),
                            'status': 'failed'
                        })

        logger.info(f"Worker {device_id}: Completed {len(processed_files)}/{len(file_subset)} files "
                   f"({len(failed_files)} failed)")

        return (processed_files, failed_files)

    except Exception as e:
        logger.exception(f"Worker {device_id}: Critical error during initialization")
        raise


# Export count_sequences and assign_files_by_sequences for backward compatibility
# These are now defined in base_worker.py
__all__ = [
    'process_dnabert_files_worker',
    'count_sequences',
    'assign_files_by_sequences'
]
