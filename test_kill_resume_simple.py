#!/usr/bin/env python3
"""Simplified kill+resume test using Python API."""

import multiprocessing as mp
import signal
import time
import tempfile
from pathlib import Path
import sys
import os


def create_test_fasta(path: Path, num_sequences: int = 50):
    """Create small test FASTA file."""
    with open(path, 'w') as f:
        for i in range(num_sequences):
            seq_id = f"seq_{i:05d}"
            # Varied length sequences (50-200 residues)
            seq_len = 50 + (i % 150)
            sequence = "A" * seq_len
            f.write(f">{seq_id}\n{sequence}\n")
    print(f"✓ Created {num_sequences} sequences")


def run_inference_wrapper(fasta_files, output_dir, checkpoint_dir, world_size, result_queue):
    """Wrapper to run inference in subprocess."""
    try:
        import sys
        import logging

        # Setup logging to see what's happening
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        from virnucpro.pipeline.multi_gpu_inference import run_multi_gpu_inference
        from virnucpro.pipeline.runtime_config import RuntimeConfig

        print(f"[Worker] Starting inference with {world_size} GPUs", file=sys.stderr, flush=True)
        print(f"[Worker] FASTA: {fasta_files}", file=sys.stderr, flush=True)
        print(f"[Worker] Output: {output_dir}", file=sys.stderr, flush=True)
        print(f"[Worker] Checkpoint: {checkpoint_dir}", file=sys.stderr, flush=True)

        # Configure for testing
        model_config = {
            'model_type': 'esm2',
            'model_name': 'esm2_t36_3B_UR50D',
            'dtype': 'float32',  # FP32 for testing
            'batch_size': 4,     # Small batches
        }

        runtime_config = RuntimeConfig(
            enable_checkpointing=True,
            checkpoint_dir=checkpoint_dir,
            checkpoint_seq_threshold=20,  # Checkpoint every 20 sequences
            checkpoint_time_threshold=30, # or every 30 seconds
        )

        print(f"[Worker] Calling run_multi_gpu_inference...", file=sys.stderr, flush=True)
        output_path, failed_ranks = run_multi_gpu_inference(
            fasta_files=fasta_files,
            output_dir=output_dir,
            model_config=model_config,
            world_size=world_size,
            runtime_config=runtime_config,
        )

        print(f"[Worker] Completed! Output: {output_path}", file=sys.stderr, flush=True)
        result_queue.put(('success', output_path, failed_ranks))
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"[Worker] ERROR: {e}", file=sys.stderr, flush=True)
        print(f"[Worker] Traceback:\n{tb}", file=sys.stderr, flush=True)
        result_queue.put(('error', str(e), tb))


def count_checkpoint_files(checkpoint_dir: Path):
    """Count checkpoint .done markers."""
    count = 0
    for shard_dir in checkpoint_dir.glob("shard_*"):
        count += len(list(shard_dir.glob("*.done")))
    return count


def main():
    mp.set_start_method('spawn', force=True)

    print("=" * 70)
    print("Kill+Resume Test (Simplified)")
    print("=" * 70)

    # Check GPU availability
    try:
        import torch
        if not torch.cuda.is_available():
            print("ERROR: No CUDA GPUs available")
            sys.exit(1)
        num_gpus = min(torch.cuda.device_count(), 2)
        print(f"Using {num_gpus} GPU(s)")
    except ImportError:
        print("ERROR: PyTorch not available")
        sys.exit(1)

    # Setup
    test_dir = Path(tempfile.mkdtemp(prefix="checkpoint_test_"))
    print(f"Test directory: {test_dir}")

    input_fasta = test_dir / "input.fasta"
    checkpoint_dir = test_dir / "checkpoints"
    output_dir = test_dir / "output"

    checkpoint_dir.mkdir()
    output_dir.mkdir()

    num_sequences = 50
    create_test_fasta(input_fasta, num_sequences)

    print("\n" + "-" * 70)
    print("PHASE 1: Start inference and kill mid-processing")
    print("-" * 70)

    result_queue = mp.Queue()

    # Start inference in subprocess
    proc = mp.Process(
        target=run_inference_wrapper,
        args=([input_fasta], output_dir, checkpoint_dir, num_gpus, result_queue)
    )
    proc.start()
    print(f"Started inference process (PID={proc.pid})")

    # Wait for checkpoints to appear
    print("Waiting for checkpoints...")
    max_wait = 60
    waited = 0
    checkpoint_found = False

    while waited < max_wait and proc.is_alive():
        time.sleep(3)
        waited += 3

        checkpoint_count = count_checkpoint_files(checkpoint_dir)
        if checkpoint_count > 0:
            checkpoint_found = True
            print(f"✓ Found {checkpoint_count} checkpoint(s) after {waited}s")
            break

        if not proc.is_alive():
            print("Process exited early - checking for errors...")
            try:
                status, *result = result_queue.get(timeout=1)
                if status == 'error':
                    print(f"ERROR from worker: {result[0]}")
                    print(f"Traceback:\n{result[1]}")
                    sys.exit(1)
                else:
                    print("Process completed successfully (unexpectedly)")
                    break
            except:
                print("No error info available from worker")
            break

        print(f"  Waiting... ({waited}s)")

    if not checkpoint_found:
        print(f"ERROR: No checkpoints found after {waited}s")
        if proc.is_alive():
            proc.kill()
            proc.join()
        else:
            # Try to get error from queue
            try:
                status, *result = result_queue.get(timeout=1)
                if status == 'error':
                    print(f"\nWorker error: {result[0]}")
                    print(f"\nTraceback:\n{result[1]}")
            except:
                pass
        sys.exit(1)

    # Wait a bit more then kill
    print("Waiting 5s more then killing...")
    time.sleep(5)

    if proc.is_alive():
        print(f"Sending SIGKILL to PID={proc.pid}")
        os.kill(proc.pid, signal.SIGKILL)
        proc.join(timeout=10)
        print(f"✓ Process killed")
    else:
        print("Process already exited")

    checkpoint_count_after_kill = count_checkpoint_files(checkpoint_dir)
    print(f"Checkpoints after kill: {checkpoint_count_after_kill}")

    print("\n" + "-" * 70)
    print("PHASE 2: Resume from checkpoints")
    print("-" * 70)

    # Create fresh queue
    result_queue2 = mp.Queue()

    # Restart
    proc2 = mp.Process(
        target=run_inference_wrapper,
        args=([input_fasta], output_dir, checkpoint_dir, num_gpus, result_queue2)
    )
    proc2.start()
    print(f"Started resume process (PID={proc2.pid})")

    # Wait for completion
    proc2.join(timeout=180)  # 3 min max

    if proc2.is_alive():
        print("ERROR: Resume process timeout")
        proc2.kill()
        proc2.join()
        sys.exit(1)

    # Get result
    try:
        status, *result = result_queue2.get(timeout=1)
        if status == 'error':
            print(f"ERROR: {result[0]}")
            print(result[1])
            sys.exit(1)
        output_path, failed_ranks = result
        print(f"✓ Resume completed: {output_path}")
        if failed_ranks:
            print(f"  (Failed ranks: {failed_ranks})")
    except:
        print("ERROR: No result from resume process")
        sys.exit(1)

    print("\n" + "-" * 70)
    print("PHASE 3: Verification")
    print("-" * 70)

    # Verify output
    if not output_path.exists():
        print(f"ERROR: Output file not found: {output_path}")
        sys.exit(1)

    import h5py
    with h5py.File(output_path, 'r') as f:
        output_count = len(f['sequence_ids'])
        sequence_ids = [s.decode() if isinstance(s, bytes) else s
                       for s in f['sequence_ids'][:]]

    unique_count = len(set(sequence_ids))

    print(f"Expected sequences: {num_sequences}")
    print(f"Output sequences:   {output_count}")
    print(f"Unique sequences:   {unique_count}")

    checks_passed = 0
    if output_count == num_sequences:
        print("✓ All sequences present")
        checks_passed += 1
    else:
        print("✗ Sequence count mismatch")

    if unique_count == output_count:
        print("✓ No duplicates")
        checks_passed += 1
    else:
        print(f"✗ Found {output_count - unique_count} duplicates")

    final_checkpoints = count_checkpoint_files(checkpoint_dir)
    if final_checkpoints >= checkpoint_count_after_kill:
        print(f"✓ Checkpoints maintained ({final_checkpoints})")
        checks_passed += 1
    else:
        print("✗ Checkpoint count decreased")

    print("\n" + "=" * 70)
    if checks_passed == 3:
        print("SUCCESS: Kill+resume works! ✓")
    else:
        print(f"FAILURE: {checks_passed}/3 checks passed")
    print("=" * 70)

    print(f"\nTest directory: {test_dir}")


if __name__ == "__main__":
    main()
