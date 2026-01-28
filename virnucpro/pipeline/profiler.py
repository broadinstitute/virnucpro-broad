"""Batch size profiling utilities for GPU optimization

This module provides utilities to help users find optimal batch sizes for their
specific GPU hardware. It measures throughput and memory usage across different
batch sizes to recommend the best configuration.

Features:
- DNABERT-S batch size profiling (token-based)
- ESM-2 batch size profiling (token-based)
- Memory usage tracking
- Throughput measurement (sequences/second)
- Binary search for maximum batch size
- Safety features (OOM handling, CUDA cache clearing)
"""

import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from Bio import SeqIO
import logging
import time

logger = logging.getLogger('virnucpro.profiler')


def measure_gpu_memory(device: torch.device) -> Tuple[float, float]:
    """
    Get current GPU memory usage.

    Args:
        device: CUDA device to measure

    Returns:
        Tuple of (allocated_gb, total_gb)
    """
    if not torch.cuda.is_available():
        return 0.0, 0.0

    allocated = torch.cuda.memory_allocated(device) / 1024**3
    total = torch.cuda.get_device_properties(device).total_memory / 1024**3
    return allocated, total


def create_test_sequences(num_sequences: int = 100, seq_length: int = 300) -> List[Tuple[str, str]]:
    """
    Generate synthetic DNA test sequences.

    Args:
        num_sequences: Number of sequences to generate
        seq_length: Length of each sequence

    Returns:
        List of (id, sequence) tuples
    """
    import random
    bases = ['A', 'T', 'G', 'C']
    sequences = []

    for i in range(num_sequences):
        seq = ''.join(random.choice(bases) for _ in range(seq_length))
        sequences.append((f'test_seq_{i}', seq))

    return sequences


def load_test_sequences_from_file(file_path: Path, max_sequences: int = 100) -> List[Tuple[str, str]]:
    """
    Load test sequences from FASTA file.

    Args:
        file_path: Path to FASTA file
        max_sequences: Maximum number of sequences to load

    Returns:
        List of (id, sequence) tuples
    """
    sequences = []
    for i, record in enumerate(SeqIO.parse(file_path, 'fasta')):
        if i >= max_sequences:
            break
        sequences.append((record.id, str(record.seq)))

    return sequences


def binary_search_max_batch(
    test_fn,
    min_batch: int,
    max_batch: int,
    device: torch.device
) -> int:
    """
    Binary search to find maximum batch size before OOM.

    Args:
        test_fn: Function that takes batch_size and returns True if successful
        min_batch: Minimum batch size to try
        max_batch: Maximum batch size to try
        device: CUDA device

    Returns:
        Maximum batch size that doesn't OOM
    """
    left, right = min_batch, max_batch
    max_working = min_batch

    while left <= right:
        mid = (left + right) // 2

        # Clear CUDA cache before test
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        try:
            logger.info(f"Testing batch size: {mid}")
            if test_fn(mid):
                max_working = mid
                left = mid + 1
            else:
                right = mid - 1
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                logger.warning(f"OOM at batch size {mid}")
                right = mid - 1
            else:
                raise

    return max_working


def profile_dnabert_batch_size(
    device: str = 'cuda:0',
    test_sequence_file: Optional[Path] = None,
    min_batch: int = 512,
    max_batch: int = 8192,
    step: int = 512,
    num_test_sequences: int = 500,  # Increased for meaningful batch testing
    sequence_length: int = 300
) -> Dict:
    """
    Profile DNABERT-S to find optimal batch size.

    This function tests various batch sizes and measures:
    - Throughput (sequences/second)
    - Memory usage
    - Success/OOM status

    Args:
        device: CUDA device string (e.g., 'cuda:0')
        test_sequence_file: Optional FASTA file with test sequences
        min_batch: Minimum batch size to test (tokens)
        max_batch: Maximum batch size to test (tokens)
        step: Step size for batch size increments
        num_test_sequences: Number of sequences to generate if no file provided
        sequence_length: Length of synthetic sequences

    Returns:
        Dictionary with:
        - optimal_batch_size: Recommended batch size (80% of max)
        - max_batch_size: Maximum batch size before OOM
        - throughput_curve: List of (batch_size, sequences_per_sec) tuples
        - memory_curve: List of (batch_size, memory_gb) tuples
        - bf16_enabled: Whether BF16 was used
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available. Profiling requires GPU.")

    device_obj = torch.device(device)
    logger.info(f"Profiling DNABERT-S on {device}")

    # Load or generate test sequences
    if test_sequence_file and test_sequence_file.exists():
        logger.info(f"Loading test sequences from {test_sequence_file}")
        test_sequences = load_test_sequences_from_file(test_sequence_file, num_test_sequences)
    else:
        logger.info(f"Generating {num_test_sequences} synthetic sequences of length {sequence_length}")
        test_sequences = create_test_sequences(num_test_sequences, sequence_length)

    # Load DNABERT-S model
    logger.info("Loading DNABERT-S model...")
    from transformers import AutoTokenizer, AutoModel

    model_name = "zhihan1996/DNABERT-S"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device_obj)
    model.eval()

    # Check BF16 support
    bf16_enabled = False
    if torch.cuda.is_available():
        compute_capability = torch.cuda.get_device_capability(device_obj)
        bf16_enabled = compute_capability[0] >= 8  # Ampere or newer
        logger.info(f"BF16 support: {bf16_enabled} (compute capability: {compute_capability})")

    throughput_curve = []
    memory_curve = []
    max_successful_batch = min_batch

    # Test batch sizes
    batch_sizes = list(range(min_batch, max_batch + 1, step))

    for batch_size in batch_sizes:
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        try:
            # Calculate number of sequences that fit in this token batch size
            # DNABERT tokenization produces roughly 1 token per 6 nucleotides (k-mer)
            # Plus padding tokens. Estimate ~sequence_length/6 + 2 tokens per sequence
            tokens_per_seq = max(sequence_length // 6 + 2, 50)
            num_seqs_in_batch = max(1, batch_size // tokens_per_seq)
            num_seqs_in_batch = min(num_seqs_in_batch, len(test_sequences))

            logger.info(f"\nTesting batch size: {batch_size} tokens ({num_seqs_in_batch} sequences)")

            # Prepare batch with calculated number of sequences
            sequences = [seq for _, seq in test_sequences[:num_seqs_in_batch]]

            # Tokenize
            inputs = tokenizer(
                sequences,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=sequence_length
            )

            # Move to device
            inputs = {k: v.to(device_obj) for k, v in inputs.items()}

            # Measure memory before
            mem_before, total_mem = measure_gpu_memory(device_obj)

            # Run inference with timing
            start_time = time.time()

            with torch.no_grad():
                if bf16_enabled:
                    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                        _ = model(**inputs)
                else:
                    _ = model(**inputs)

            # Synchronize GPU
            if torch.cuda.is_available():
                torch.cuda.synchronize(device_obj)

            elapsed = time.time() - start_time

            # Measure memory after
            mem_after, _ = measure_gpu_memory(device_obj)

            # Calculate metrics
            sequences_per_sec = len(sequences) / elapsed if elapsed > 0 else 0
            memory_used = mem_after

            throughput_curve.append((batch_size, sequences_per_sec))
            memory_curve.append((batch_size, memory_used))
            max_successful_batch = batch_size

            logger.info(f"  ✓ Success: {sequences_per_sec:.1f} seq/s, {memory_used:.2f}GB memory")

        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                logger.warning(f"  ✗ OOM at batch size {batch_size}")
                break
            else:
                logger.error(f"  ✗ Error: {e}")
                raise

    # Calculate optimal batch size (80% of max to leave headroom)
    optimal_batch = int(max_successful_batch * 0.8)
    # Round to nearest step
    optimal_batch = (optimal_batch // step) * step

    results = {
        'optimal_batch_size': optimal_batch,
        'max_batch_size': max_successful_batch,
        'throughput_curve': throughput_curve,
        'memory_curve': memory_curve,
        'bf16_enabled': bf16_enabled,
        'device': str(device_obj)
    }

    logger.info(f"\n{'='*60}")
    logger.info(f"DNABERT-S Profiling Results:")
    logger.info(f"  Device: {device_obj}")
    logger.info(f"  BF16: {'enabled' if bf16_enabled else 'disabled'}")
    logger.info(f"  Maximum batch size: {max_successful_batch} tokens")
    logger.info(f"  Recommended batch size: {optimal_batch} tokens (80% of max)")
    logger.info(f"{'='*60}\n")

    return results


def profile_esm_batch_size(
    device: str = 'cuda:0',
    test_sequence_file: Optional[Path] = None,
    min_batch: int = 512,
    max_batch: int = 8192,
    step: int = 512,
    num_test_sequences: int = 500,  # Increased for meaningful batch testing
    sequence_length: int = 200  # Longer sequences for better profiling
) -> Dict:
    """
    Profile ESM-2 to find optimal batch size.

    This function tests various batch sizes and measures:
    - Throughput (sequences/second)
    - Memory usage
    - Success/OOM status

    Args:
        device: CUDA device string (e.g., 'cuda:0')
        test_sequence_file: Optional FASTA file with test protein sequences
        min_batch: Minimum batch size to test (tokens)
        max_batch: Maximum batch size to test (tokens)
        step: Step size for batch size increments
        num_test_sequences: Number of sequences to generate if no file provided
        sequence_length: Length of synthetic protein sequences

    Returns:
        Dictionary with:
        - optimal_batch_size: Recommended batch size (80% of max)
        - max_batch_size: Maximum batch size before OOM
        - throughput_curve: List of (batch_size, sequences_per_sec) tuples
        - memory_curve: List of (batch_size, memory_gb) tuples
        - bf16_enabled: Whether BF16 was used
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available. Profiling requires GPU.")

    device_obj = torch.device(device)
    logger.info(f"Profiling ESM-2 on {device}")

    # Load or generate test protein sequences
    if test_sequence_file and test_sequence_file.exists():
        logger.info(f"Loading test sequences from {test_sequence_file}")
        test_sequences = load_test_sequences_from_file(test_sequence_file, num_test_sequences)
    else:
        logger.info(f"Generating {num_test_sequences} synthetic protein sequences of length {sequence_length}")
        # Generate random protein sequences
        import random
        aa = 'ACDEFGHIKLMNPQRSTVWY'
        test_sequences = []
        for i in range(num_test_sequences):
            seq = ''.join(random.choice(aa) for _ in range(sequence_length))
            test_sequences.append((f'test_prot_{i}', seq))

    # Load ESM-2 model
    logger.info("Loading ESM-2 model...")
    import esm

    esm_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    esm_model = esm_model.to(device_obj)
    esm_model.eval()
    batch_converter = alphabet.get_batch_converter()

    # Check BF16 support
    bf16_enabled = False
    if torch.cuda.is_available():
        compute_capability = torch.cuda.get_device_capability(device_obj)
        bf16_enabled = compute_capability[0] >= 8  # Ampere or newer
        logger.info(f"BF16 support: {bf16_enabled} (compute capability: {compute_capability})")

    throughput_curve = []
    memory_curve = []
    max_successful_batch = min_batch

    # Test batch sizes
    batch_sizes = list(range(min_batch, max_batch + 1, step))

    for batch_size in batch_sizes:
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        try:
            # Calculate number of sequences that fit in this token batch size
            # ESM-2 adds 2 tokens (BOS/EOS) per sequence
            tokens_per_seq = sequence_length + 2
            num_seqs_in_batch = max(1, batch_size // tokens_per_seq)
            num_seqs_in_batch = min(num_seqs_in_batch, len(test_sequences))

            logger.info(f"\nTesting batch size: {batch_size} tokens ({num_seqs_in_batch} sequences)")

            # Prepare batch with calculated number of sequences
            sequences = test_sequences[:num_seqs_in_batch]

            # Convert batch
            batch_labels, batch_strs, batch_tokens = batch_converter(sequences)
            batch_tokens = batch_tokens.to(device_obj)

            # Measure memory before
            mem_before, total_mem = measure_gpu_memory(device_obj)

            # Run inference with timing
            start_time = time.time()

            with torch.no_grad():
                if bf16_enabled:
                    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                        _ = esm_model(batch_tokens, repr_layers=[33])
                else:
                    _ = esm_model(batch_tokens, repr_layers=[33])

            # Synchronize GPU
            if torch.cuda.is_available():
                torch.cuda.synchronize(device_obj)

            elapsed = time.time() - start_time

            # Measure memory after
            mem_after, _ = measure_gpu_memory(device_obj)

            # Calculate metrics
            sequences_per_sec = len(sequences) / elapsed if elapsed > 0 else 0
            memory_used = mem_after

            throughput_curve.append((batch_size, sequences_per_sec))
            memory_curve.append((batch_size, memory_used))
            max_successful_batch = batch_size

            logger.info(f"  ✓ Success: {sequences_per_sec:.1f} seq/s, {memory_used:.2f}GB memory")

        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                logger.warning(f"  ✗ OOM at batch size {batch_size}")
                break
            else:
                logger.error(f"  ✗ Error: {e}")
                raise

    # Calculate optimal batch size (80% of max to leave headroom)
    optimal_batch = int(max_successful_batch * 0.8)
    # Round to nearest step
    optimal_batch = (optimal_batch // step) * step

    results = {
        'optimal_batch_size': optimal_batch,
        'max_batch_size': max_successful_batch,
        'throughput_curve': throughput_curve,
        'memory_curve': memory_curve,
        'bf16_enabled': bf16_enabled,
        'device': str(device_obj)
    }

    logger.info(f"\n{'='*60}")
    logger.info(f"ESM-2 Profiling Results:")
    logger.info(f"  Device: {device_obj}")
    logger.info(f"  BF16: {'enabled' if bf16_enabled else 'disabled'}")
    logger.info(f"  Maximum batch size: {max_successful_batch} tokens")
    logger.info(f"  Recommended batch size: {optimal_batch} tokens (80% of max)")
    logger.info(f"{'='*60}\n")

    return results
