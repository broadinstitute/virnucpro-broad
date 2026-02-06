"""FP16 vs FP32+FlashAttention throughput benchmarking using forward_packed().

Benchmarks the production code path (forward_packed with FlashAttention varlen)
to measure FP16 precision speedup isolated from other optimizations.

Baseline: Phase 7 FP32 with FlashAttention enabled (current production)
Target: Phase 8 FP16 with FlashAttention (FP16 tensor cores + larger batches)
Expected: 1.5-1.8x speedup (FP16 tensor cores ~1.3-1.5x + larger batches ~1.2x)

The benchmark:
- Uses forward_packed() production code path (not standard forward)
- Verifies FlashAttention kernels are active (fails if fallback detected)
- Uses stratified length batches (short/medium/long) to prevent padding skew
- Isolates compute time from data transfer (batches pre-transferred to GPU)
- Measures total runtime reduction (primary metric) and tokens/second

This is the one-time Phase 8 validation benchmark. For production diagnostics,
use --fp32-compare flag instead.

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
from virnucpro.data.packing import GreedyPacker

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

    Uses VarlenCollator + GreedyPacker to create packed batches (production path).
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
    from virnucpro.data.packing import GreedyPacker

    # Create packed batches on CPU (isolate from compute timing)
    packer = GreedyPacker(max_tokens_per_batch=8192)  # Realistic production budget
    collator = VarlenCollator(packer=packer, batch_converter=model.model.alphabet.get_batch_converter())

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
    """FP16 vs FP32+FlashAttention throughput benchmarking."""

    def test_fp16_vs_fp32_throughput(self):
        """
        Benchmark FP16 vs FP32+FlashAttention throughput using forward_packed().

        This is the main Phase 8 validation benchmark. It establishes the FP32+FlashAttention
        baseline (Phase 7) and FP16+FlashAttention (Phase 8) performance in the same run.

        Expected: 1.5-1.8x throughput improvement from FP16 precision
        (FP16 tensor cores ~1.3-1.5x + larger batches from memory savings ~1.2x)

        Baseline: Phase 7 FP32 with FlashAttention enabled
        Code path: forward_packed() (production)
        """
        device = torch.device("cuda:0")

        logger.info("=" * 80)
        logger.info("Phase 8 FP16 vs Phase 7 FP32+FlashAttention Baseline Benchmark")
        logger.info("=" * 80)

        # Generate stratified sequences (50 each of short/medium/long)
        logger.info("Generating stratified test sequences...")
        stratified_seqs = generate_stratified_sequences(num_per_length=50)

        # Results storage
        results = {}

        # ========== Phase 7 Baseline: FP32 + FlashAttention ==========
        logger.info("\n[1/2] Loading model in FP32 (Phase 7 baseline)...")
        model_fp32, _ = load_esm2_model(
            model_name="esm2_t36_3B_UR50D",
            device="cuda:0",
            enable_fp16=False,  # FP32 baseline
        )

        # Verify FlashAttention is active
        logger.info("Verifying FlashAttention integration (FP32)...")
        verify_flashattention_active(model_fp32)
        logger.info("✓ FlashAttention verified for FP32 baseline")

        # Benchmark each length class
        logger.info("Benchmarking FP32+FlashAttention (Phase 7 baseline)...")
        fp32_results = {}
        fp32_total_time = 0.0
        fp32_total_tokens = 0
        fp32_total_sequences = 0

        for length_class, sequences in stratified_seqs.items():
            logger.info(f"  Benchmarking {length_class} sequences (n={len(sequences)})...")
            result = benchmark_model_packed(
                model_fp32, sequences, device,
                num_warmup=10, num_iterations=50
            )
            fp32_results[length_class] = result
            fp32_total_time += result["elapsed_sec"]
            fp32_total_tokens += result["total_tokens"]
            fp32_total_sequences += result["total_sequences"]
            logger.info(f"    {result['tokens_per_sec']:.0f} tokens/sec, {result['peak_memory_gb']:.2f} GB")

        # Clean up FP32 model
        logger.info("Clearing FP32 model from GPU...")
        del model_fp32
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        # ========== Phase 8: FP16 + FlashAttention ==========
        logger.info("\n[2/2] Loading model in FP16 (Phase 8)...")
        model_fp16, _ = load_esm2_model(
            model_name="esm2_t36_3B_UR50D",
            device="cuda:0",
            enable_fp16=True,  # FP16
        )

        # Verify FlashAttention is active
        logger.info("Verifying FlashAttention integration (FP16)...")
        verify_flashattention_active(model_fp16)
        logger.info("✓ FlashAttention verified for FP16")

        # Benchmark each length class
        logger.info("Benchmarking FP16+FlashAttention (Phase 8)...")
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
        # Overall metrics
        overall_runtime_reduction = 1 - (fp16_total_time / fp32_total_time)
        overall_speedup = fp32_total_time / fp16_total_time
        overall_tokens_fp32 = fp32_total_tokens / fp32_total_time
        overall_tokens_fp16 = fp16_total_tokens / fp16_total_time
        tokens_speedup = overall_tokens_fp16 / overall_tokens_fp32

        overall_sequences_fp32 = fp32_total_sequences / fp32_total_time
        overall_sequences_fp16 = fp16_total_sequences / fp16_total_time
        sequences_speedup = overall_sequences_fp16 / overall_sequences_fp32

        # Memory comparison
        fp32_peak = max(r["peak_memory_gb"] for r in fp32_results.values())
        fp16_peak = max(r["peak_memory_gb"] for r in fp16_results.values())
        memory_reduction = 1 - (fp16_peak / fp32_peak)

        # Per-length-class speedups
        length_speedups = {}
        for length_class in stratified_seqs.keys():
            fp32_tokens_per_sec = fp32_results[length_class]["tokens_per_sec"]
            fp16_tokens_per_sec = fp16_results[length_class]["tokens_per_sec"]
            length_speedups[length_class] = fp16_tokens_per_sec / fp32_tokens_per_sec

        # ========== Print Results ==========
        logger.info("\n" + "=" * 80)
        logger.info("Phase 8 FP16 vs Phase 7 FP32+FlashAttention Baseline")
        logger.info("=" * 80)
        logger.info(f"{'Metric':<25} | {'FP32 (P7)':<15} | {'FP16 (P8)':<15} | {'Improvement':<15}")
        logger.info("-" * 80)
        logger.info(f"{'Total runtime':<25} | {fp32_total_time:<15.2f}s | {fp16_total_time:<15.2f}s | {overall_speedup:<15.2f}x (primary)")
        logger.info(f"{'Tokens/sec':<25} | {overall_tokens_fp32:<15.0f} | {overall_tokens_fp16:<15.0f} | {tokens_speedup:<15.2f}x")
        logger.info(f"{'Sequences/sec':<25} | {overall_sequences_fp32:<15.1f} | {overall_sequences_fp16:<15.1f} | {sequences_speedup:<15.2f}x")
        logger.info(f"{'Peak memory':<25} | {fp32_peak:<15.2f}GB | {fp16_peak:<15.2f}GB | {memory_reduction*100:<14.1f}% reduction")
        logger.info("-" * 80)
        logger.info("Per length class:")
        logger.info(f"  Short (50aa):    {length_speedups['short']:.2f}x speedup")
        logger.info(f"  Medium (150aa):  {length_speedups['medium']:.2f}x speedup")
        logger.info(f"  Long (300aa):    {length_speedups['long']:.2f}x speedup")
        logger.info("-" * 80)
        logger.info("Expected: 1.5-1.8x throughput improvement")
        logger.info("  (FP16 tensor cores ~1.3-1.5x + larger batches ~1.2x)")
        logger.info("Baseline: Phase 7 FP32 with FlashAttention enabled")
        logger.info("Code path: forward_packed() (production)")
        logger.info("=" * 80)

        # ========== Assertions ==========
        # Verify FP16 is at least as fast (not slower)
        assert overall_speedup >= 1.0, (
            f"FP16 slower than FP32: {overall_speedup:.2f}x speedup. "
            f"Expected ≥1.0x (FP16 should be at least as fast as FP32)."
        )

        # Verify FP16 uses less or equal memory
        assert fp16_peak <= fp32_peak, (
            f"FP16 uses more memory than FP32: {fp16_peak:.2f}GB vs {fp32_peak:.2f}GB. "
            f"Expected FP16 ≤ FP32."
        )

        logger.info(f"✓ PASSED: FP16 provides {overall_speedup:.2f}x speedup over FP32+FlashAttention baseline")

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
