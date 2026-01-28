#!/usr/bin/env python3
"""Benchmark script to verify BF16 + SDPA fix performance.

This script compares DNABERT-S performance before/after the SDPA patch.
Run on A100 to see the expected 10-16x speedup vs FP32.
"""

import torch
import time
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))


def benchmark_dnabert(num_sequences=100, seq_length=512, warmup=3, iterations=10):
    """Benchmark DNABERT-S with the SDPA patch."""
    from virnucpro.models.dnabert_flash import load_dnabert_model

    print("="*70)
    print("DNABERT-S BF16 + SDPA Benchmark")
    print("="*70)

    # Load model
    print("\nLoading model...")
    model, tokenizer = load_dnabert_model(device="cuda:0")

    # Print model config
    print(f"\nModel: {model}")
    print(f"Device: {model.device}")
    print(f"Using BF16: {model.use_bf16}")
    print(f"Attention: {model.attention_impl}")

    # Generate synthetic sequences
    print(f"\nGenerating {num_sequences} sequences of length {seq_length}...")
    sequences = ["".join(["ATCG"[i % 4] for i in range(seq_length)]) for _ in range(num_sequences)]

    # Tokenize
    tokens = tokenizer(
        sequences,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=seq_length
    )
    tokens = {k: v.to(model.device) for k, v in tokens.items()}

    print(f"Input shape: {tokens['input_ids'].shape}")
    print(f"Input dtype: {tokens['input_ids'].dtype}")

    # Warmup
    print(f"\nWarmup ({warmup} iterations)...")
    torch.cuda.synchronize()
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(**tokens)
    torch.cuda.synchronize()

    # Benchmark
    print(f"\nBenchmarking ({iterations} iterations)...")
    torch.cuda.synchronize()
    start = time.perf_counter()

    with torch.no_grad():
        for _ in range(iterations):
            output = model(**tokens)

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    # Results
    avg_time = elapsed / iterations
    throughput = num_sequences / avg_time

    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"Total time: {elapsed:.3f}s for {iterations} iterations")
    print(f"Average time per batch: {avg_time*1000:.1f}ms")
    print(f"Throughput: {throughput:.1f} sequences/second")
    print(f"Output dtype: {output.last_hidden_state.dtype if hasattr(output, 'last_hidden_state') else output[0].dtype}")

    # GPU info
    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    print(f"Memory cached: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")

    return throughput


def benchmark_esm2(num_sequences=50, seq_length=256, warmup=3, iterations=10):
    """Benchmark ESM-2 with BF16."""
    from virnucpro.models.esm2_flash import load_esm2_model

    print("\n" + "="*70)
    print("ESM-2 BF16 Benchmark")
    print("="*70)

    # Load model (using smaller 650M model for speed)
    print("\nLoading model (ESM-2 650M)...")
    model, batch_converter = load_esm2_model("esm2_t33_650M_UR50D", device="cuda:0")

    print(f"\nModel: {model}")
    print(f"Device: {model.device}")
    print(f"Using BF16: {model.use_bf16}")
    print(f"Attention: {model.attention_impl}")

    # Generate synthetic protein sequences
    print(f"\nGenerating {num_sequences} sequences of length {seq_length}...")
    aa = "ACDEFGHIKLMNPQRSTVWY"
    sequences = [
        (f"seq_{i}", "".join([aa[j % len(aa)] for j in range(seq_length)]))
        for i in range(num_sequences)
    ]

    # Convert batch
    labels, strs, tokens = batch_converter(sequences)
    tokens = tokens.to(model.device)

    print(f"Input shape: {tokens.shape}")
    print(f"Input dtype: {tokens.dtype}")

    # Warmup
    print(f"\nWarmup ({warmup} iterations)...")
    torch.cuda.synchronize()
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(tokens, repr_layers=[33])
    torch.cuda.synchronize()

    # Benchmark
    print(f"\nBenchmarking ({iterations} iterations)...")
    torch.cuda.synchronize()
    start = time.perf_counter()

    with torch.no_grad():
        for _ in range(iterations):
            output = model(tokens, repr_layers=[33])

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    # Results
    avg_time = elapsed / iterations
    throughput = num_sequences / avg_time

    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"Total time: {elapsed:.3f}s for {iterations} iterations")
    print(f"Average time per batch: {avg_time*1000:.1f}ms")
    print(f"Throughput: {throughput:.1f} sequences/second")
    print(f"Output dtype: {output['representations'][33].dtype}")

    # GPU info
    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    print(f"Memory cached: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")

    return throughput


if __name__ == "__main__":
    print("\n" + "="*70)
    print("BF16 + SDPA Performance Verification")
    print("="*70)
    print(f"\nPyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    capability = torch.cuda.get_device_capability(0)
    print(f"Compute Capability: {capability[0]}.{capability[1]}")
    print(f"BF16 Support: {capability[0] >= 8}")

    # Run benchmarks
    dnabert_throughput = benchmark_dnabert()

    # Clear GPU memory
    torch.cuda.empty_cache()

    esm2_throughput = benchmark_esm2()

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\nDNABERT-S: {dnabert_throughput:.1f} sequences/second")
    print(f"ESM-2 650M: {esm2_throughput:.1f} sequences/second")
    print("\nIf you see BF16 outputs and reasonable throughput, the fix is working!")
    print("\nOn A100, you should see ~10-16x speedup vs FP32 for the same workload.")
    print("Compare these numbers before/after the fix to verify improvement.")
