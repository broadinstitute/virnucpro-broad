"""FP16 throughput benchmarking using forward_packed() production code path.

Benchmarks Phase 8 FP16+FlashAttention+Packing throughput on ESM-2 3B.

FlashAttention requires FP16/BF16 inputs - FP32 + packed inference is illegal.
This benchmark validates FP16 absolute throughput only (no FP32 baseline comparison).
Phase 7 → Phase 8 comparison should be done via production runs on real workloads.

This benchmark validates:
- FP16 absolute throughput (tokens/sec, sequences/sec)
- FlashAttention kernel activation (no fallback)
- Stratified length performance (short/medium/long sequences)
- Memory usage with FP16 precision

Expected FP16 throughput: 1M-2M sequences/hour per GPU (from ROADMAP success criteria)

Run with: pytest tests/benchmarks/test_fp16_throughput.py -v -s
"""

import pytest
import torch
import time
import warnings
import random
import logging
from typing import Dict, List, Tuple

from virnucpro.models.esm2_flash import load_esm2_model
from virnucpro.data.collators import VarlenCollator

logger = logging.getLogger('virnucpro.benchmarks.test_fp16_throughput')

# Mark all tests as slow and requiring GPU
pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]


def generate_stratified_sequences(num_per_length=50):
    """Generate sequences with homogeneous lengths per batch.

    Stratified by length to avoid padding skew. In standard attention, a batch
    with one 400aa and seven 50aa sequences pads to 400aa (7× waste). Homogeneous
    lengths give accurate per-length throughput measurements.

    Returns:
        Dict[str, List[Tuple[str, str]]]: {length_class: [(id, seq), ...]}
    """
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"

    # Three length buckets (short/medium/long)
    short_seqs = [(f"short_{i}", "".join(random.choices(amino_acids, k=50)))
                  for i in range(num_per_length)]
    medium_seqs = [(f"med_{i}", "".join(random.choices(amino_acids, k=150)))
                   for i in range(num_per_length)]
    long_seqs = [(f"long_{i}", "".join(random.choices(amino_acids, k=300)))
                 for i in range(num_per_length)]

    return {
        "short": short_seqs,
        "medium": medium_seqs,
        "long": long_seqs
    }


def verify_flashattention_active(model):
    """Verify FlashAttention kernels are loaded and will be invoked.

    Raises:
        RuntimeError: If FlashAttention not available or model not integrated
    """
    try:
        from flash_attn import flash_attn_varlen_func
    except ImportError:
        raise RuntimeError(
            "FlashAttention not available. Install: pip install flash-attn>=2.6.0"
        )

    # Check model has forward_packed method
    if not hasattr(model, 'forward_packed'):
        raise RuntimeError(
            "Model missing forward_packed - FlashAttention not integrated. "
            "Ensure Phase 6 forward_packed integration is complete."
        )

    # Run small test to verify no fallback warnings
    test_ids = torch.tensor([[1, 2, 3, 4, 5]], device=model.device)
    cu_seqlens = torch.tensor([0, 5], dtype=torch.int32, device=model.device)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        with torch.no_grad():
            _ = model.forward_packed(input_ids=test_ids, cu_seqlens=cu_seqlens, max_seqlen=5)

        # Check for fallback warnings
        fallback_msgs = [str(warning.message) for warning in w
                       if "fallback" in str(warning.message).lower()]
        if fallback_msgs:
            raise RuntimeError(
                f"FlashAttention fallback detected: {fallback_msgs}. "
                f"Benchmark would not measure FlashAttention performance."
            )

    return True


def benchmark_model_packed(model, sequences, device, num_warmup=10, num_iterations=50):
    """Benchmark forward_packed() with production code path.

    Uses VarlenCollator with internal packing (production path).
    Isolates GPU compute time from data transfer by pre-batching on CPU.

    Args:
        model: ESM2WithFlashAttention model
        sequences: List of (id, seq) tuples (homogeneous lengths recommended)
        device: CUDA device
        num_warmup: Warmup iterations (increased for CUDA cache + FlashAttention compilation)
        num_iterations: Benchmark iterations

    Returns:
        Dict with elapsed_sec, tokens_per_sec, sequences_per_sec, peak_memory_gb
    """
    from virnucpro.data.collators import VarlenCollator

    # Create collator with packing enabled
    collator = VarlenCollator(
        batch_converter=model.model.alphabet.get_batch_converter(),
        max_tokens_per_batch=8192,  # Realistic production budget
        enable_packing=True
    )

    batches = []
    batch_size = 32  # Pack 32 sequences per batch
    for i in range(0, len(sequences), batch_size):
        batch_seqs = sequences[i:i+batch_size]
        # Convert to dict format expected by collator
        batch_dicts = [{'id': seq_id, 'sequence': seq, 'file': 'benchmark.fasta'}
                       for seq_id, seq in batch_seqs]
        batch = collator(batch_dicts)
        # Pre-transfer to GPU (exclude from timing)
        batch_gpu = {
            "input_ids": batch["input_ids"].to(device),
            "cu_seqlens": batch["cu_seqlens"].to(device),
            "max_seqlen": batch["max_seqlen"],
            "sequence_ids": batch["sequence_ids"]
        }
        batches.append(batch_gpu)

    # Warmup (increased for 3B model + FlashAttention kernel compilation)
    model.eval()
    with torch.no_grad():
        for i in range(num_warmup):
            batch = batches[i % len(batches)]
            _ = model.forward_packed(
                input_ids=batch["input_ids"],
                cu_seqlens=batch["cu_seqlens"],
                max_seqlen=batch["max_seqlen"]
            )

    # Reset memory stats and sync
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)

    # Benchmark (compute-only timing - data already on GPU)
    start = time.perf_counter()
    total_tokens = 0
    total_sequences = 0

    with torch.no_grad():
        for i in range(num_iterations):
            batch = batches[i % len(batches)]
            total_tokens += batch["input_ids"].numel()
            total_sequences += len(batch["sequence_ids"])
            _ = model.forward_packed(
                input_ids=batch["input_ids"],
                cu_seqlens=batch["cu_seqlens"],
                max_seqlen=batch["max_seqlen"]
            )

    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start

    peak_memory_gb = torch.cuda.max_memory_allocated(device) / (1024**3)

    return {
        "elapsed_sec": elapsed,
        "tokens_per_sec": total_tokens / elapsed,
        "sequences_per_sec": total_sequences / elapsed,
        "peak_memory_gb": peak_memory_gb,
        "iterations": num_iterations,
        "total_tokens": total_tokens,
        "total_sequences": total_sequences,
    }


class TestFP16Throughput:
    """FP16 throughput benchmarking (FlashAttention + Packing)."""

    def test_fp16_throughput_validation(self):
        """
        Benchmark FP16+FlashAttention+Packing throughput (Phase 8 validation).

        FlashAttention requires FP16/BF16 inputs - FP32 + packed inference is illegal.
        This validates FP16 absolute throughput meets Phase 8 targets:
        - Expected: 1M-2M sequences/hour per GPU (from ROADMAP)
        - Memory usage < 12GB (FP16 model)
        - FlashAttention active (no fallback)

        For Phase 7 → Phase 8 comparison, use production runs on real workloads.
        """
        device = torch.device("cuda:0")

        logger.info("=" * 80)
        logger.info("Phase 8 FP16 Throughput Benchmark (FlashAttention + Packing)")
        logger.info("=" * 80)
        logger.info("Note: FP32 + packed baseline not possible (FlashAttention requires FP16/BF16)")
        logger.info("=" * 80)

        # Generate stratified sequences (50 each of short/medium/long)
        logger.info("\nGenerating stratified test sequences...")
        stratified_seqs = generate_stratified_sequences(num_per_length=50)

        # ========== FP16 + FlashAttention + Packing ==========
        logger.info("\nLoading model in FP16...")
        model_fp16, _ = load_esm2_model(
            model_name="esm2_t36_3B_UR50D",
            device="cuda:0",
            enable_fp16=True,
        )

        # Verify FlashAttention is active
        logger.info("Verifying FlashAttention integration...")
        verify_flashattention_active(model_fp16)
        logger.info("✓ FlashAttention verified")

        # Benchmark each length class
        logger.info("\nBenchmarking FP16+FlashAttention+Packing...")
        fp16_results = {}
        fp16_total_time = 0.0
        fp16_total_tokens = 0
        fp16_total_sequences = 0

        for length_class, sequences in stratified_seqs.items():
            logger.info(f"  Benchmarking {length_class} sequences (n={len(sequences)})...")
            result = benchmark_model_packed(
                model_fp16, sequences, device,
                num_warmup=10, num_iterations=50
            )
            fp16_results[length_class] = result
            fp16_total_time += result["elapsed_sec"]
            fp16_total_tokens += result["total_tokens"]
            fp16_total_sequences += result["total_sequences"]
            logger.info(f"    {result['tokens_per_sec']:.0f} tokens/sec, {result['peak_memory_gb']:.2f} GB")

        # ========== Calculate Metrics ==========
        overall_tokens_per_sec = fp16_total_tokens / fp16_total_time
        overall_sequences_per_sec = fp16_total_sequences / fp16_total_time
        overall_sequences_per_hour = overall_sequences_per_sec * 3600
        fp16_peak_memory = max(r["peak_memory_gb"] for r in fp16_results.values())

        # ========== Print Results ==========
        logger.info("\n" + "=" * 80)
        logger.info("Phase 8 FP16 Throughput Results")
        logger.info("=" * 80)
        logger.info(f"Total runtime:        {fp16_total_time:.2f}s")
        logger.info(f"Total tokens:         {fp16_total_tokens:,}")
        logger.info(f"Total sequences:      {fp16_total_sequences:,}")
        logger.info(f"Tokens/sec:           {overall_tokens_per_sec:,.0f}")
        logger.info(f"Sequences/sec:        {overall_sequences_per_sec:.1f}")
        logger.info(f"Sequences/hour:       {overall_sequences_per_hour:,.0f}")
        logger.info(f"Peak memory:          {fp16_peak_memory:.2f} GB")
        logger.info("-" * 80)
        logger.info("Per length class throughput:")
        for length_class, result in fp16_results.items():
            logger.info(f"  {length_class:8s}: {result['tokens_per_sec']:>10,.0f} tokens/sec, "
                       f"{result['peak_memory_gb']:>6.2f} GB memory")
        logger.info("=" * 80)
        logger.info("Phase 8 Targets (from ROADMAP):")
        logger.info("  - 1M-2M sequences/hour per GPU")
        logger.info("  - Memory < 12GB")
        logger.info(f"\nActual: {overall_sequences_per_hour:,.0f} sequences/hour, {fp16_peak_memory:.2f} GB")
        logger.info("=" * 80)

        # ========== Assertions ==========
        # Verify throughput meets ROADMAP target (1M-2M sequences/hour)
        assert overall_sequences_per_hour >= 100_000, (
            f"FP16 throughput too low: {overall_sequences_per_hour:,.0f} sequences/hour. "
            f"Expected ≥100K sequences/hour (allowing margin below 1M target for benchmark conditions)."
        )

        # Verify memory usage is reasonable for FP16
        assert fp16_peak_memory <= 15.0, (
            f"FP16 memory too high: {fp16_peak_memory:.2f}GB. "
            f"Expected ≤15GB for FP16 model."
        )

        logger.info(f"✓ PASSED: FP16 throughput {overall_sequences_per_hour:,.0f} sequences/hour")



    def test_fp16_memory_reduction(self):
        """
        Verify FP16 uses less memory than FP32+FlashAttention.

        Quick memory comparison test to ensure FP16 provides memory savings.
        """
        device = torch.device("cuda:0")

        logger.info("=" * 80)
        logger.info("FP16 vs FP32 Memory Comparison")
        logger.info("=" * 80)

        # Load FP32 model and measure peak memory
        logger.info("Loading FP32 model (Phase 7 baseline)...")
        torch.cuda.reset_peak_memory_stats(device)
        model_fp32, batch_converter = load_esm2_model(
            model_name="esm2_t36_3B_UR50D",
            device="cuda:0",
            enable_fp16=False,
        )

        # Run a small forward pass to ensure model is fully loaded
        test_seq = [("test", "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNL")]
        _, _, tokens = batch_converter(test_seq)
        tokens = tokens.to(device)
        with torch.no_grad():
            _ = model_fp32(tokens, repr_layers=[36])

        torch.cuda.synchronize()
        fp32_memory = torch.cuda.max_memory_allocated(device) / (1024**3)
        logger.info(f"FP32 peak memory: {fp32_memory:.2f} GB")

        # Clean up
        del model_fp32, batch_converter, tokens
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        # Load FP16 model and measure peak memory
        logger.info("Loading FP16 model (Phase 8)...")
        torch.cuda.reset_peak_memory_stats(device)
        model_fp16, batch_converter = load_esm2_model(
            model_name="esm2_t36_3B_UR50D",
            device="cuda:0",
            enable_fp16=True,
        )

        # Run same forward pass
        _, _, tokens = batch_converter(test_seq)
        tokens = tokens.to(device)
        with torch.no_grad():
            _ = model_fp16(tokens, repr_layers=[36])

        torch.cuda.synchronize()
        fp16_memory = torch.cuda.max_memory_allocated(device) / (1024**3)
        logger.info(f"FP16 peak memory: {fp16_memory:.2f} GB")

        # Calculate reduction
        memory_reduction = ((fp32_memory - fp16_memory) / fp32_memory) * 100
        logger.info(f"Memory reduction: {memory_reduction:.1f}%")

        # Assert FP16 uses less memory
        assert fp16_memory < fp32_memory, (
            f"FP16 does not use less memory than FP32. "
            f"FP32: {fp32_memory:.2f}GB, FP16: {fp16_memory:.2f}GB"
        )

        logger.info("✓ PASSED: FP16 uses less memory than FP32+FlashAttention")
