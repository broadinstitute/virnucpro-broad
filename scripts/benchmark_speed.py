#!/usr/bin/env python3
"""
GPU-synchronized speed benchmark for FastESM2_650 vs ESM2 3B protein embedding extraction.

Measures embedding extraction time with proper GPU synchronization, warmup iterations,
and multiple timed runs for statistical significance. Validates 2x speedup claim.

Benchmark scope: Protein embedding extraction only (not full pipeline).
"""

import argparse
import os
import sys
import time
from pathlib import Path
from datetime import datetime

import torch
from transformers import AutoModel, AutoTokenizer
from Bio import SeqIO
from tqdm import tqdm


def load_sample_sequences(num_sequences, data_dir='./data'):
    """
    Load sample protein sequences from FASTA files or generate synthetic sequences.

    Args:
        num_sequences: number of sequences to sample
        data_dir: directory to search for FASTA files

    Returns:
        list of protein sequence strings
    """
    sequences = []

    # Try to load from FASTA files first
    fasta_files = list(Path(data_dir).rglob('*protein*.fa*'))

    if fasta_files:
        print(f"Found {len(fasta_files)} protein FASTA files")
        for fasta_file in fasta_files:
            try:
                for record in SeqIO.parse(str(fasta_file), 'fasta'):
                    sequences.append(str(record.seq))
                    if len(sequences) >= num_sequences:
                        break
            except Exception as e:
                print(f"Warning: Could not read {fasta_file}: {e}")

            if len(sequences) >= num_sequences:
                break

    # If not enough sequences from files, generate synthetic sequences
    if len(sequences) < num_sequences:
        print(f"Only found {len(sequences)} sequences in files, generating synthetic sequences")
        amino_acids = 'ACDEFGHIKLMNPQRSTVWY'

        # Generate mix of short (100aa), medium (300aa), and long (500aa) sequences
        lengths = [100, 300, 500] * (num_sequences // 3 + 1)
        import random
        random.seed(42)

        while len(sequences) < num_sequences:
            seq_len = lengths[len(sequences) % len(lengths)]
            seq = ''.join(random.choice(amino_acids) for _ in range(seq_len))
            sequences.append(seq)

    # Return exactly num_sequences
    return sequences[:num_sequences]


def benchmark_fastesm(sequences, num_runs=3, warmup=10):
    """
    Benchmark FastESM2_650 embedding extraction with GPU synchronization.

    Args:
        sequences: list of protein sequence strings
        num_runs: number of timed runs to average
        warmup: number of warmup iterations

    Returns:
        tuple of (average_time, per_sequence_time, sequences_per_second)
    """
    print("\n" + "="*70)
    print("FASTESM2_650 BENCHMARK")
    print("="*70)

    # Load model
    print("Loading FastESM2_650 model...")
    try:
        model = AutoModel.from_pretrained(
            "Synthyra/FastESM2_650",
            trust_remote_code=True,
            torch_dtype=torch.float16
        ).eval().cuda()
        tokenizer = model.tokenizer
    except Exception as e:
        print(f"ERROR: Failed to load FastESM2_650: {e}")
        return None, None, None

    print(f"Model loaded on device: {model.device}")

    # Warmup runs
    print(f"\nRunning {warmup} warmup iterations...")
    for _ in tqdm(range(warmup), desc="Warmup"):
        batch = sequences[:10]  # Use small batch for warmup
        inputs = tokenizer(batch, return_tensors='pt', padding='longest', truncation=True, max_length=1026)
        input_ids = inputs['input_ids'].cuda()
        attention_mask = inputs['attention_mask'].cuda()

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        torch.cuda.synchronize()

    # Clear cache before timed runs
    torch.cuda.empty_cache()

    # Timed runs
    print(f"\nRunning {num_runs} timed iterations...")
    run_times = []

    for run_idx in range(num_runs):
        # Synchronize before starting timer
        torch.cuda.synchronize()
        start_time = time.perf_counter()

        # Process all sequences
        batch_size = 16  # Process in small batches to avoid OOM
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i:i+batch_size]

            inputs = tokenizer(
                batch,
                return_tensors='pt',
                padding='longest',
                truncation=True,
                max_length=1026
            )

            input_ids = inputs['input_ids'].cuda()
            attention_mask = inputs['attention_mask'].cuda()

            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)

                # Mean pool embeddings (matching extract_fast_esm logic)
                for j in range(len(batch)):
                    seq_len = attention_mask[j].sum().item() - 2  # Exclude BOS/EOS
                    embedding = outputs.last_hidden_state[j, 1:seq_len+1].mean(0)
                    embedding = embedding.float().cpu()

        # Synchronize after all GPU work completes
        torch.cuda.synchronize()
        end_time = time.perf_counter()

        run_time = end_time - start_time
        run_times.append(run_time)
        print(f"  Run {run_idx+1}: {run_time:.4f}s")

    # Calculate statistics
    avg_time = sum(run_times) / len(run_times)
    per_seq_time = avg_time / len(sequences)
    seqs_per_sec = len(sequences) / avg_time

    print(f"\nFastESM2_650 Results:")
    print(f"  Average time: {avg_time:.4f}s")
    print(f"  Per-sequence: {per_seq_time*1000:.2f}ms")
    print(f"  Throughput: {seqs_per_sec:.2f} sequences/second")

    return avg_time, per_seq_time, seqs_per_sec


def benchmark_esm2_3b(sequences, num_runs=3, warmup=10):
    """
    Benchmark ESM2 3B embedding extraction with GPU synchronization.

    Args:
        sequences: list of protein sequence strings
        num_runs: number of timed runs to average
        warmup: number of warmup iterations

    Returns:
        tuple of (average_time, per_sequence_time, sequences_per_second) or (None, None, None) if failed
    """
    print("\n" + "="*70)
    print("ESM2 3B BENCHMARK")
    print("="*70)

    # Load model
    print("Loading ESM2 3B model (facebook/esm2_t36_3B_UR50D)...")
    print("WARNING: This requires ~12GB GPU memory")

    try:
        model = AutoModel.from_pretrained("facebook/esm2_t36_3B_UR50D").eval().cuda()
        tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t36_3B_UR50D")
    except torch.cuda.OutOfMemoryError:
        print("ERROR: GPU out of memory - ESM2 3B requires ~12GB VRAM")
        print("Run with --fastesm-only flag to skip ESM2 3B benchmark")
        return None, None, None
    except Exception as e:
        print(f"ERROR: Failed to load ESM2 3B: {e}")
        print("ESM2 3B benchmark skipped")
        return None, None, None

    print(f"Model loaded on device: {model.device}")

    # Warmup runs
    print(f"\nRunning {warmup} warmup iterations...")
    for _ in tqdm(range(warmup), desc="Warmup"):
        batch = sequences[:10]  # Use small batch for warmup
        inputs = tokenizer(batch, return_tensors='pt', padding='longest', truncation=True, max_length=1026)
        input_ids = inputs['input_ids'].cuda()
        attention_mask = inputs['attention_mask'].cuda()

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        torch.cuda.synchronize()

    # Clear cache before timed runs
    torch.cuda.empty_cache()

    # Timed runs
    print(f"\nRunning {num_runs} timed iterations...")
    run_times = []

    for run_idx in range(num_runs):
        # Synchronize before starting timer
        torch.cuda.synchronize()
        start_time = time.perf_counter()

        # Process all sequences
        batch_size = 16  # Process in small batches to avoid OOM
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i:i+batch_size]

            inputs = tokenizer(
                batch,
                return_tensors='pt',
                padding='longest',
                truncation=True,
                max_length=1026
            )

            input_ids = inputs['input_ids'].cuda()
            attention_mask = inputs['attention_mask'].cuda()

            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)

                # Mean pool embeddings (matching ESM2 extraction logic)
                for j in range(len(batch)):
                    seq_len = attention_mask[j].sum().item() - 2  # Exclude BOS/EOS
                    embedding = outputs.last_hidden_state[j, 1:seq_len+1].mean(0)
                    embedding = embedding.float().cpu()

        # Synchronize after all GPU work completes
        torch.cuda.synchronize()
        end_time = time.perf_counter()

        run_time = end_time - start_time
        run_times.append(run_time)
        print(f"  Run {run_idx+1}: {run_time:.4f}s")

    # Calculate statistics
    avg_time = sum(run_times) / len(run_times)
    per_seq_time = avg_time / len(sequences)
    seqs_per_sec = len(sequences) / avg_time

    print(f"\nESM2 3B Results:")
    print(f"  Average time: {avg_time:.4f}s")
    print(f"  Per-sequence: {per_seq_time*1000:.2f}ms")
    print(f"  Throughput: {seqs_per_sec:.2f} sequences/second")

    return avg_time, per_seq_time, seqs_per_sec


def generate_markdown_report(fastesm_results, esm2_results, num_sequences, num_runs, report_path):
    """Generate markdown benchmark report."""

    with open(report_path, 'w') as f:
        # Header
        f.write("# FastESM2_650 Speed Benchmark Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}\n\n")
        f.write(f"**Benchmark scope:** Protein embedding extraction only\n")
        f.write(f"**Test sequences:** {num_sequences}\n")
        f.write(f"**Timed runs:** {num_runs} (averaged)\n\n")

        # Results table
        f.write("## Benchmark Results\n\n")

        if esm2_results[0] is not None:
            # Both models benchmarked
            f.write("| Model | Total Time | Per-Sequence | Sequences/Second |\n")
            f.write("|-------|------------|--------------|------------------|\n")
            f.write(f"| FastESM2_650 | {fastesm_results[0]:.4f}s | {fastesm_results[1]*1000:.2f}ms | {fastesm_results[2]:.2f} |\n")
            f.write(f"| ESM2 3B | {esm2_results[0]:.4f}s | {esm2_results[1]*1000:.2f}ms | {esm2_results[2]:.2f} |\n\n")

            # Speedup calculation
            speedup = esm2_results[0] / fastesm_results[0]
            f.write("## Speedup Analysis\n\n")
            f.write(f"**Speedup ratio:** {speedup:.2f}x\n\n")
            f.write(f"FastESM2_650 is **{speedup:.2f}x faster** than ESM2 3B for protein embedding extraction.\n\n")

            # Threshold validation
            f.write("## Threshold Validation\n\n")
            if speedup >= 2.0:
                f.write("**Status:** ✅ PASSED\n\n")
                f.write(f"FastESM2_650 achieves {speedup:.2f}x speedup, meeting the 2x threshold.\n\n")
            else:
                f.write("**Status:** ❌ FAILED\n\n")
                f.write(f"FastESM2_650 achieves {speedup:.2f}x speedup, below the 2x threshold.\n\n")
                f.write("**Note:** Speedup may vary based on GPU architecture:\n")
                f.write("- H100/A100: Expected ~2x\n")
                f.write("- GB10 (sm_121): Expected ~1.29x (see Phase 1 decision)\n\n")

        else:
            # FastESM2 only
            f.write("| Model | Total Time | Per-Sequence | Sequences/Second |\n")
            f.write("|-------|------------|--------------|------------------|\n")
            f.write(f"| FastESM2_650 | {fastesm_results[0]:.4f}s | {fastesm_results[1]*1000:.2f}ms | {fastesm_results[2]:.2f} |\n\n")

            f.write("## ESM2 3B Comparison\n\n")
            f.write("ESM2 3B benchmark was skipped.\n\n")
            f.write("**Reason:** Model failed to load (likely GPU memory < 12GB)\n\n")
            f.write("To run full comparison, use a GPU with at least 12GB VRAM.\n\n")

        # Hardware info
        f.write("## Hardware Information\n\n")
        f.write(f"- GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}\n")
        f.write(f"- CUDA version: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}\n")
        f.write(f"- PyTorch version: {torch.__version__}\n\n")

        # Methodology
        f.write("## Methodology\n\n")
        f.write("- **Timing:** `time.perf_counter()` with `torch.cuda.synchronize()` before/after\n")
        f.write("- **Warmup:** 10 iterations (not counted in results)\n")
        f.write("- **Runs:** 3 timed runs, averaged for statistical significance\n")
        f.write("- **Processing:** Batch size 16, mean pooling positions 1:seq_len+1 (excludes BOS/EOS)\n\n")

        f.write("---\n")
        f.write("_Report generated by scripts/benchmark_speed.py_\n")

    print(f"\nBenchmark report saved to: {report_path}")


def print_summary(fastesm_results, esm2_results, num_sequences):
    """Print terminal summary."""

    print("\n" + "="*70)
    print("SPEED BENCHMARK SUMMARY")
    print("="*70)

    print(f"\nTest configuration:")
    print(f"  Sequences: {num_sequences}")
    print(f"  GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")

    print(f"\nFastESM2_650:")
    print(f"  Total time: {fastesm_results[0]:.4f}s")
    print(f"  Per-sequence: {fastesm_results[1]*1000:.2f}ms")
    print(f"  Throughput: {fastesm_results[2]:.2f} seqs/sec")

    if esm2_results[0] is not None:
        print(f"\nESM2 3B:")
        print(f"  Total time: {esm2_results[0]:.4f}s")
        print(f"  Per-sequence: {esm2_results[1]*1000:.2f}ms")
        print(f"  Throughput: {esm2_results[2]:.2f} seqs/sec")

        speedup = esm2_results[0] / fastesm_results[0]
        print(f"\nSpeedup: {speedup:.2f}x")

        if speedup >= 2.0:
            print("Status: ✅ PASSED (meets 2x threshold)")
        else:
            print(f"Status: ❌ FAILED (below 2x threshold)")
            print("Note: GB10 GPUs expect ~1.29x (see Phase 1 decision)")
    else:
        print("\nESM2 3B: Benchmark skipped (model failed to load)")

    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark FastESM2_650 vs ESM2 3B protein embedding extraction speed"
    )
    parser.add_argument(
        '--num-sequences',
        type=int,
        default=100,
        help='Number of sequences to benchmark (default: 100)'
    )
    parser.add_argument(
        '--num-runs',
        type=int,
        default=3,
        help='Number of timed runs to average (default: 3)'
    )
    parser.add_argument(
        '--warmup',
        type=int,
        default=10,
        help='Number of warmup iterations (default: 10)'
    )
    parser.add_argument(
        '--report-dir',
        type=str,
        default='./reports/',
        help='Output directory for benchmark report (default: ./reports/)'
    )
    parser.add_argument(
        '--fastesm-only',
        action='store_true',
        help='Skip ESM2 3B benchmark (useful for GPUs with <12GB memory)'
    )

    args = parser.parse_args()

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. This benchmark requires a GPU.")
        sys.exit(1)

    print("="*70)
    print("FASTESM2 SPEED BENCHMARK")
    print("="*70)
    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Sequences: {args.num_sequences}")
    print(f"Runs: {args.num_runs} (+ {args.warmup} warmup)")

    # Create report directory
    os.makedirs(args.report_dir, exist_ok=True)
    report_path = os.path.join(args.report_dir, 'speed_benchmark.md')

    # Load sample sequences
    print("\nLoading sample sequences...")
    sequences = load_sample_sequences(args.num_sequences)
    print(f"Loaded {len(sequences)} protein sequences")

    # Benchmark FastESM2
    fastesm_results = benchmark_fastesm(sequences, args.num_runs, args.warmup)

    if fastesm_results[0] is None:
        print("ERROR: FastESM2 benchmark failed")
        sys.exit(1)

    # Benchmark ESM2 3B (unless --fastesm-only)
    if args.fastesm_only:
        print("\n" + "="*70)
        print("ESM2 3B BENCHMARK")
        print("="*70)
        print("Skipped (--fastesm-only flag)")
        esm2_results = (None, None, None)
    else:
        esm2_results = benchmark_esm2_3b(sequences, args.num_runs, args.warmup)

    # Generate report
    generate_markdown_report(
        fastesm_results,
        esm2_results,
        len(sequences),
        args.num_runs,
        report_path
    )

    # Print summary
    print_summary(fastesm_results, esm2_results, len(sequences))
    print(f"Full report: {report_path}\n")


if __name__ == '__main__':
    main()
