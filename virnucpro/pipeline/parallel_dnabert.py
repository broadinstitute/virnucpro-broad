"""DNABERT-S specific multiprocessing utilities for parallel feature extraction

This module implements parallel DNABERT-S processing following the same patterns as ESM-2.
Key features:
- Token-based batching (DNA bases treated as tokens)
- BF16 mixed precision on Ampere+ GPUs
- Spawn context multiprocessing for CUDA safety
- Greedy bin-packing file assignment by sequence count for balanced GPU utilization
"""

import os
import torch
from torch.cuda.amp import autocast
from pathlib import Path
from typing import List, Tuple
from Bio import SeqIO
import logging

from virnucpro.core.checkpoint import atomic_save

logger = logging.getLogger('virnucpro.parallel_dnabert')

# Module-level globals for persistent worker model storage
_dnabert_model = None
_tokenizer = None
_device = None

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
from virnucpro.cuda import StreamProcessor


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

    Supports optional CUDA stream-based processing for I/O-compute overlap
    via enable_streams kwarg (default: False for backward compatibility).

    Args:
        file_subset: List of DNA FASTA files to process
        device_id: CUDA device ID (e.g., 0 for cuda:0)
        toks_per_batch: Tokens per batch for DNABERT-S processing (default: 2048)
        output_dir: Directory where output files should be saved
        **kwargs: Additional arguments (log_level, log_format, enable_streams)

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

    # Check if streams are enabled
    enable_streams = kwargs.get('enable_streams', False)

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

        # Check for BF16 support (batch size adjustment handled by pipeline)
        use_bf16 = detect_bf16_support(device)
        logger.info(f"Worker {device_id}: Using batch size {toks_per_batch}, BF16: {use_bf16}")

        # Initialize stream processor if enabled
        stream_processor = None
        if enable_streams:
            stream_processor = StreamProcessor(device=device, enable_streams=True)
            logger.info(f"Worker {device_id}: Stream-based processing enabled")

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

                            if stream_processor is not None:
                                # Stream-based processing with I/O-compute overlap
                                def transfer_fn(seqs):
                                    inputs = tokenizer(seqs, return_tensors='pt', padding=True)
                                    input_ids = inputs["input_ids"].to(device, non_blocking=True)
                                    attn_mask = inputs.get("attention_mask", None)
                                    if attn_mask is not None:
                                        attn_mask = attn_mask.to(device, non_blocking=True)
                                    return (input_ids, attn_mask)

                                def compute_fn(inputs_tuple):
                                    input_ids, attn_mask = inputs_tuple
                                    if attn_mask is not None:
                                        return model(input_ids, attention_mask=attn_mask)[0]
                                    else:
                                        return model(input_ids)[0]

                                def retrieve_fn(hidden_states):
                                    return hidden_states.cpu()

                                # Process with streams
                                hidden_states = stream_processor.process_batch_async(
                                    batch_seqs, transfer_fn, compute_fn, retrieve_fn
                                )

                                # Mean pool with attention mask (already on CPU)
                                inputs = tokenizer(batch_seqs, return_tensors='pt', padding=True)
                                attention_mask = inputs.get("attention_mask", None)
                                if attention_mask is not None:
                                    embedding_means = (hidden_states * attention_mask.unsqueeze(-1)).sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
                                else:
                                    embedding_means = torch.mean(hidden_states, dim=1)
                            else:
                                # Standard synchronous processing
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
                                if stream_processor is not None:
                                    # Already on CPU
                                    result = {"label": label, "mean_representation": embedding_mean.float().tolist()}
                                else:
                                    result = {"label": label, "mean_representation": embedding_mean.float().cpu().tolist()}
                                nucleotide.append(label)
                                data.append(result)

                            logger.debug(f"Worker {device_id}: Processed batch {batch_idx + 1}/{len(batches)}")

                    # Save to file in original format using atomic write
                    # Feature extraction checkpoints skip validation for performance (large files)
                    atomic_save(
                        {'nucleotide': nucleotide, 'data': data},
                        output_file,
                        validate_after_save=False  # Skip validation for feature extraction (performance)
                    )

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


# ============================================================================
# Persistent Worker Functions (for long-lived worker pools)
# ============================================================================


def init_dnabert_worker(
    device_id: int,
    model_name: str = "zhihan1996/DNABERT-S",
    log_level: int = logging.INFO
) -> None:
    """
    Initialize persistent DNABERT-S worker with pre-loaded model.

    This function is called once during worker pool initialization to load
    the DNABERT-S model into GPU memory. The model remains loaded for the worker's
    lifetime, eliminating repeated loading overhead.

    Configures CUDA memory management (expandable segments) before any CUDA
    operations to prevent fragmentation.

    Args:
        device_id: CUDA device ID (e.g., 0 for cuda:0)
        model_name: DNABERT-S model name (default: zhihan1996/DNABERT-S)
        log_level: Logging level for worker process (default: INFO)

    Module Globals:
        Sets _dnabert_model, _tokenizer, _device for worker lifetime

    Example:
        >>> # Called by Pool initializer
        >>> from multiprocessing import Pool
        >>> pool = Pool(processes=2, initializer=init_dnabert_worker, initargs=(0,))
    """
    global _dnabert_model, _tokenizer, _device

    # Configure CUDA memory BEFORE any CUDA operations
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    # Set up logging in worker process
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    setup_worker_logging(log_level, log_format)

    # Initialize CUDA context
    _device = torch.device(f'cuda:{device_id}')
    logger.info(f"Worker {device_id}: Initializing persistent DNABERT-S worker on {_device}")

    # Load DNABERT-S model using load_dnabert_model from virnucpro.models.dnabert_flash
    from virnucpro.models.dnabert_flash import load_dnabert_model

    _dnabert_model, _tokenizer = load_dnabert_model(
        model_name=model_name,
        device=str(_device),
        logger_instance=logger
    )

    logger.info(f"Worker {device_id}: DNABERT-S model loaded successfully - ready for processing")


def process_dnabert_files_persistent(
    file_subset: List[Path],
    device_id: int,
    toks_per_batch: int = 2048,
    output_dir: Path = None,
    **kwargs
) -> Tuple[List[Path], List[Tuple[Path, str]]]:
    """
    Process DNABERT-S features using pre-loaded model from persistent worker.

    This function processes files using the model loaded during worker initialization
    (via init_dnabert_worker). No model loading overhead - uses cached model from globals.

    Implements periodic cache clearing every 10 files to prevent memory fragmentation
    in long-running workers.

    Args:
        file_subset: List of DNA FASTA files to process
        device_id: CUDA device ID (for logging, model already loaded)
        toks_per_batch: Tokens per batch for DNABERT-S processing (default: 2048)
        output_dir: Directory where output files should be saved
        **kwargs: Additional arguments (enable_streams)

    Returns:
        Tuple of (processed_files, failed_files)
        - processed_files: List of successfully processed output .pt file paths
        - failed_files: List of (file_path, error_message) tuples for failures

    Raises:
        RuntimeError: If called before init_dnabert_worker (model not loaded)

    Note:
        Uses module-level globals (_dnabert_model, _tokenizer, _device) set by
        init_dnabert_worker during pool initialization.

    Example:
        >>> # Called by pool.map or pool.imap
        >>> pool.map(process_dnabert_files_persistent, file_batches)
    """
    global _dnabert_model, _tokenizer, _device

    # Verify model was loaded by init_dnabert_worker
    if _dnabert_model is None or _tokenizer is None or _device is None:
        raise RuntimeError(
            "Persistent worker not initialized. "
            "Call init_dnabert_worker during pool initialization."
        )

    # Get progress queue from module global (set by Pool initializer)
    progress_queue = _get_progress_queue()

    # Check if streams are enabled
    enable_streams = kwargs.get('enable_streams', False)

    # Check for BF16 support
    use_bf16 = detect_bf16_support(_device)

    processed_files = []
    failed_files = []

    try:
        logger.info(f"Worker {device_id}: Processing {len(file_subset)} files with pre-loaded model (BF16: {use_bf16})")

        # Initialize stream processor if enabled
        stream_processor = None
        if enable_streams:
            stream_processor = StreamProcessor(device=_device, enable_streams=True)
            logger.info(f"Worker {device_id}: Stream-based processing enabled")

        # Wrap all inference in torch.no_grad() context
        with torch.no_grad():
            for idx, file in enumerate(file_subset):
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

                            if stream_processor is not None:
                                # Stream-based processing with I/O-compute overlap
                                def transfer_fn(seqs):
                                    inputs = _tokenizer(seqs, return_tensors='pt', padding=True)
                                    input_ids = inputs["input_ids"].to(_device, non_blocking=True)
                                    attn_mask = inputs.get("attention_mask", None)
                                    if attn_mask is not None:
                                        attn_mask = attn_mask.to(_device, non_blocking=True)
                                    return (input_ids, attn_mask)

                                def compute_fn(inputs_tuple):
                                    input_ids, attn_mask = inputs_tuple
                                    if attn_mask is not None:
                                        return _dnabert_model(input_ids, attention_mask=attn_mask)[0]
                                    else:
                                        return _dnabert_model(input_ids)[0]

                                def retrieve_fn(hidden_states):
                                    return hidden_states.cpu()

                                # Process with streams
                                hidden_states = stream_processor.process_batch_async(
                                    batch_seqs, transfer_fn, compute_fn, retrieve_fn
                                )

                                # Mean pool with attention mask (already on CPU)
                                inputs = _tokenizer(batch_seqs, return_tensors='pt', padding=True)
                                attention_mask = inputs.get("attention_mask", None)
                                if attention_mask is not None:
                                    embedding_means = (hidden_states * attention_mask.unsqueeze(-1)).sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
                                else:
                                    embedding_means = torch.mean(hidden_states, dim=1)
                            else:
                                # Standard synchronous processing using pre-loaded model
                                # Tokenize batch with padding
                                inputs = _tokenizer(batch_seqs, return_tensors='pt', padding=True)
                                input_ids = inputs["input_ids"].to(_device)
                                attention_mask = inputs.get("attention_mask", None)
                                if attention_mask is not None:
                                    attention_mask = attention_mask.to(_device)

                                # Forward pass - model returns tuple, take first element
                                if attention_mask is not None:
                                    hidden_states = _dnabert_model(input_ids, attention_mask=attention_mask)[0]
                                else:
                                    hidden_states = _dnabert_model(input_ids)[0]

                                # Mean pool with attention mask to exclude padding
                                if attention_mask is not None:
                                    embedding_means = (hidden_states * attention_mask.unsqueeze(-1)).sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
                                else:
                                    embedding_means = torch.mean(hidden_states, dim=1)

                            # Store results (convert BF16 to FP32 for compatibility)
                            for label, embedding_mean in zip(batch_labels, embedding_means):
                                if stream_processor is not None:
                                    # Already on CPU
                                    result = {"label": label, "mean_representation": embedding_mean.float().tolist()}
                                else:
                                    result = {"label": label, "mean_representation": embedding_mean.float().cpu().tolist()}
                                nucleotide.append(label)
                                data.append(result)

                            logger.debug(f"Worker {device_id}: Processed batch {batch_idx + 1}/{len(batches)}")

                    # Save to file in original format using atomic write
                    # Feature extraction checkpoints skip validation for performance (large files)
                    atomic_save(
                        {'nucleotide': nucleotide, 'data': data},
                        output_file,
                        validate_after_save=False  # Skip validation for feature extraction (performance)
                    )

                    processed_files.append(output_file)
                    logger.info(f"Worker {device_id}: Completed {file.name} -> {output_file.name} ({len(data)} sequences)")

                    # Report progress if queue available
                    if progress_queue is not None:
                        progress_queue.put({
                            'gpu_id': device_id,
                            'file': str(file),
                            'status': 'complete'
                        })

                    # Periodic cache clearing to prevent fragmentation (every 10 files)
                    if (idx + 1) % 10 == 0:
                        torch.cuda.empty_cache()
                        logger.debug(f"Worker {device_id}: Cleared CUDA cache after {idx + 1} files")

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
        logger.exception(f"Worker {device_id}: Critical error during processing")
        raise


# Export count_sequences and assign_files_by_sequences for backward compatibility
# These are now defined in base_worker.py
__all__ = [
    'process_dnabert_files_worker',
    'count_sequences',
    'assign_files_by_sequences'
]
