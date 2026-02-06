#!/usr/bin/env python3
"""End-to-end kill+resume test for checkpoint recovery.

Tests actual process crash recovery by:
1. Starting multi-GPU inference with checkpointing
2. Killing a worker mid-batch with SIGKILL
3. Restarting the pipeline
4. Verifying complete recovery with no duplicates
"""

import subprocess
import signal
import time
import sys
from pathlib import Path
import tempfile
import shutil
import os


def create_test_fasta(path: Path, num_sequences: int = 100):
    """Create test FASTA file with simple sequences."""
    with open(path, 'w') as f:
        for i in range(num_sequences):
            seq_id = f"seq_{i:05d}"
            # Create varied length sequences (100-500 residues)
            seq_len = 100 + (i % 400)
            sequence = "A" * seq_len
            f.write(f">{seq_id}\n{sequence}\n")
    print(f"✓ Created test FASTA with {num_sequences} sequences at {path}")


def find_worker_pids(log_dir: Path, exclude_coordinator: bool = True):
    """Find worker PIDs from log files."""
    pids = []
    for log_file in log_dir.glob("worker_*.log"):
        # Extract PID from worker log filename pattern: worker_<rank>_<pid>.log
        parts = log_file.stem.split('_')
        if len(parts) >= 3:
            try:
                pid = int(parts[2])
                pids.append(pid)
            except ValueError:
                continue
    return pids


def count_checkpoint_files(checkpoint_dir: Path):
    """Count checkpoint .done markers."""
    count = 0
    for shard_dir in checkpoint_dir.glob("shard_*"):
        count += len(list(shard_dir.glob("*.done")))
    return count


def count_sequences_in_hdf5(hdf5_path: Path):
    """Count sequences in HDF5 output."""
    import h5py
    with h5py.File(hdf5_path, 'r') as f:
        return len(f['sequence_ids'])


def main():
    """Run end-to-end kill+resume test."""
    print("=" * 70)
    print("Kill+Resume End-to-End Test")
    print("=" * 70)
    print()

    # Setup test directory
    test_dir = Path(tempfile.mkdtemp(prefix="checkpoint_test_"))
    print(f"Test directory: {test_dir}")

    input_fasta = test_dir / "input.fasta"
    checkpoint_dir = test_dir / "checkpoints"
    output_dir = test_dir / "output"
    log_dir = test_dir / "logs"

    checkpoint_dir.mkdir()
    output_dir.mkdir()
    log_dir.mkdir()

    # Create test data (smaller for faster testing)
    num_sequences = 100
    create_test_fasta(input_fasta, num_sequences)

    print()
    print("-" * 70)
    print("PHASE 1: Initial run with kill mid-batch")
    print("-" * 70)

    # Start multi-GPU inference (use 2 GPUs if available, else 1)
    try:
        import torch
        num_gpus = min(torch.cuda.device_count(), 2)
        if num_gpus == 0:
            print("ERROR: No GPUs available. This test requires GPU.")
            sys.exit(1)
    except ImportError:
        num_gpus = 1

    gpu_list = ",".join(str(i) for i in range(num_gpus))

    # Build command
    cmd = [
        sys.executable, "-m", "virnucpro.pipeline.multi_gpu_inference",
        "--fasta-files", str(input_fasta),
        "--output-dir", str(output_dir),
        "--checkpoint-dir", str(checkpoint_dir),
        "--world-size", str(num_gpus),
        "--model-name", "esm2_t36_3B_UR50D",
    ]

    print(f"Command: {' '.join(cmd)}")
    print(f"Using {num_gpus} GPU(s)")
    print()

    # Start process
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu_list}
    )

    print(f"Started process PID={proc.pid}")

    # Wait for checkpoints to start appearing
    print("Waiting for checkpoints to appear...")
    wait_time = 0
    max_wait = 120  # 2 minutes max
    checkpoint_found = False

    while wait_time < max_wait:
        time.sleep(5)
        wait_time += 5

        checkpoint_count = count_checkpoint_files(checkpoint_dir)
        if checkpoint_count > 0:
            checkpoint_found = True
            print(f"✓ Found {checkpoint_count} checkpoint(s) after {wait_time}s")
            break

        # Check if process is still alive
        if proc.poll() is not None:
            stdout, stderr = proc.communicate()
            print(f"ERROR: Process exited early (returncode={proc.returncode})")
            print("STDOUT:", stdout[-500:] if stdout else "(empty)")
            print("STDERR:", stderr[-500:] if stderr else "(empty)")
            sys.exit(1)

        print(f"  Waiting... ({wait_time}s / {max_wait}s)")

    if not checkpoint_found:
        print(f"ERROR: No checkpoints appeared after {max_wait}s")
        proc.kill()
        proc.wait()
        sys.exit(1)

    # Give it a bit more time to ensure we're mid-batch
    print("Waiting 5s more to ensure mid-batch processing...")
    time.sleep(5)

    # Kill the process with SIGKILL
    print(f"\nKilling process with SIGKILL...")
    try:
        proc.send_signal(signal.SIGKILL)
        proc.wait(timeout=10)
        print(f"✓ Process killed (returncode={proc.returncode})")
    except subprocess.TimeoutExpired:
        print("ERROR: Process did not terminate after SIGKILL")
        sys.exit(1)

    # Check checkpoint state
    checkpoint_count_after_kill = count_checkpoint_files(checkpoint_dir)
    print(f"✓ Checkpoints after kill: {checkpoint_count_after_kill}")

    print()
    print("-" * 70)
    print("PHASE 2: Resume from checkpoints")
    print("-" * 70)

    # Restart pipeline
    proc2 = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu_list}
    )

    print(f"Started resume process PID={proc2.pid}")
    print("Waiting for completion...")

    try:
        stdout, stderr = proc2.communicate(timeout=300)  # 5 min max
        returncode = proc2.returncode

        print(f"Process completed (returncode={returncode})")

        if returncode != 0:
            print("\nERROR: Resume process failed")
            print("STDOUT:", stdout[-1000:] if stdout else "(empty)")
            print("STDERR:", stderr[-1000:] if stderr else "(empty)")
            sys.exit(1)

    except subprocess.TimeoutExpired:
        print("ERROR: Resume process did not complete in 5 minutes")
        proc2.kill()
        proc2.wait()
        sys.exit(1)

    print()
    print("-" * 70)
    print("PHASE 3: Verification")
    print("-" * 70)

    # Find output HDF5 file
    output_files = list(output_dir.glob("*.h5"))
    if not output_files:
        print("ERROR: No HDF5 output file found")
        sys.exit(1)

    output_file = output_files[0]
    print(f"Output file: {output_file}")

    # Count sequences in output
    try:
        output_seq_count = count_sequences_in_hdf5(output_file)
        print(f"Sequences in output: {output_seq_count}")
    except Exception as e:
        print(f"ERROR: Could not read HDF5 file: {e}")
        sys.exit(1)

    # Verification checks
    print("\nVerification:")
    checks_passed = 0
    total_checks = 3

    # Check 1: All sequences processed
    if output_seq_count == num_sequences:
        print(f"  ✓ All {num_sequences} sequences present in output")
        checks_passed += 1
    else:
        print(f"  ✗ Expected {num_sequences} sequences, got {output_seq_count}")

    # Check 2: No duplicates
    import h5py
    with h5py.File(output_file, 'r') as f:
        sequence_ids = [s.decode() if isinstance(s, bytes) else s
                       for s in f['sequence_ids'][:]]
        unique_ids = set(sequence_ids)

        if len(unique_ids) == len(sequence_ids):
            print(f"  ✓ No duplicate sequences (all {len(sequence_ids)} unique)")
            checks_passed += 1
        else:
            duplicates = len(sequence_ids) - len(unique_ids)
            print(f"  ✗ Found {duplicates} duplicate sequences")

    # Check 3: Checkpoints were created during both runs
    final_checkpoint_count = count_checkpoint_files(checkpoint_dir)
    if final_checkpoint_count >= checkpoint_count_after_kill:
        print(f"  ✓ Checkpoints present: {final_checkpoint_count} (increased or maintained)")
        checks_passed += 1
    else:
        print(f"  ✗ Checkpoint count decreased: {checkpoint_count_after_kill} → {final_checkpoint_count}")

    print()
    print("=" * 70)
    if checks_passed == total_checks:
        print(f"SUCCESS: All {total_checks} verification checks passed! ✓")
        print("=" * 70)
        print("\nKill+resume functionality works correctly:")
        print("  • Process killed mid-batch with SIGKILL")
        print("  • Pipeline resumed from checkpoints successfully")
        print("  • All sequences processed exactly once (no duplicates)")
        print("  • Checkpoints persisted correctly across process boundaries")
    else:
        print(f"FAILURE: {checks_passed}/{total_checks} checks passed")
        print("=" * 70)
        sys.exit(1)

    # Cleanup
    print(f"\nTest directory: {test_dir}")
    print("(Not deleted - inspect if needed)")


if __name__ == "__main__":
    main()
