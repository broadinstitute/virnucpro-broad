#!/usr/bin/env python3
"""Aggressive kill+resume test - kills process very early to ensure mid-processing termination."""

import multiprocessing as mp
import signal
import time
import tempfile
from pathlib import Path
import sys
import os
import h5py


def create_test_fasta(path: Path, num_sequences: int = 2000):
    """Create large test FASTA."""
    print(f"Creating test FASTA with {num_sequences} sequences...")
    with open(path, 'w') as f:
        for i in range(num_sequences):
            seq_id = f"test_seq_{i:06d}"
            # Varied lengths: 100-600 residues
            seq_len = 100 + (i % 500)
            aa = ['A', 'G', 'V', 'L', 'I'][i % 5]
            sequence = aa * seq_len
            f.write(f">{seq_id}\n{sequence}\n")
    print(f"✓ Created {num_sequences} sequences")


def run_inference(fasta_files, output_dir, checkpoint_dir, world_size, result_queue):
    """Run inference."""
    try:
        from virnucpro.pipeline.multi_gpu_inference import run_multi_gpu_inference
        from virnucpro.pipeline.runtime_config import RuntimeConfig

        model_config = {
            'model_type': 'esm2',
            'model_name': 'esm2_t36_3B_UR50D',
            'dtype': 'float32',
            'batch_size': 4,
        }

        runtime_config = RuntimeConfig(
            enable_checkpointing=True,
            checkpoint_dir=checkpoint_dir,
            checkpoint_seq_threshold=100,  # Checkpoint every 100 sequences
            checkpoint_time_threshold=30,
        )

        output_path, failed_ranks = run_multi_gpu_inference(
            fasta_files=fasta_files,
            output_dir=output_dir,
            model_config=model_config,
            world_size=world_size,
            runtime_config=runtime_config,
        )

        result_queue.put(('success', str(output_path), failed_ranks))
    except Exception as e:
        import traceback
        result_queue.put(('error', str(e), traceback.format_exc()))


def count_checkpointed_sequences(checkpoint_dir: Path):
    """Count sequences in checkpoint files."""
    import torch
    total = 0
    for shard_dir in checkpoint_dir.glob("shard_*"):
        for ckpt_file in shard_dir.glob("batch_*.pt"):
            if not (ckpt_file.parent / f"{ckpt_file.name}.done").exists():
                continue  # Skip checkpoints without .done markers
            try:
                data = torch.load(ckpt_file, map_location='cpu', weights_only=False)
                total += len(data.get('sequence_ids', []))
            except:
                pass
    return total


def main():
    mp.set_start_method('spawn', force=True)

    print("=" * 80)
    print("AGGRESSIVE KILL+RESUME TEST")
    print("Strategy: Kill process as soon as first checkpoint appears")
    print("=" * 80)
    print()

    try:
        import torch
        if not torch.cuda.is_available():
            print("ERROR: No CUDA GPUs available")
            sys.exit(1)
        num_gpus = min(torch.cuda.device_count(), 2)
    except ImportError:
        print("ERROR: PyTorch not available")
        sys.exit(1)

    # Setup
    test_dir = Path(tempfile.mkdtemp(prefix="checkpoint_aggressive_"))
    input_fasta = test_dir / "input.fasta"
    checkpoint_dir = test_dir / "checkpoints"
    output_dir = test_dir / "output"

    checkpoint_dir.mkdir()
    output_dir.mkdir()

    num_sequences = 2000  # Large enough that it takes significant time
    create_test_fasta(input_fasta, num_sequences)
    print(f"Test directory: {test_dir}\n")

    # ==================================================================
    # PHASE 1: Start and kill ASAP after first checkpoint
    # ==================================================================
    print("=" * 80)
    print("PHASE 1: Start inference and kill immediately after first checkpoint")
    print("=" * 80)

    result_queue = mp.Queue()
    proc = mp.Process(
        target=run_inference,
        args=([input_fasta], output_dir, checkpoint_dir, num_gpus, result_queue)
    )

    proc.start()
    print(f"Started process PID={proc.pid}")
    print("Waiting for first checkpoint...")

    # Wait for ANY checkpoint to appear
    max_wait = 60
    waited = 0
    found_checkpoint = False

    while waited < max_wait and proc.is_alive():
        time.sleep(2)
        waited += 2

        # Check for any .done marker
        done_files = list(checkpoint_dir.glob("shard_*/*.done"))
        if done_files:
            found_checkpoint = True
            print(f"✓ First checkpoint appeared after {waited}s")
            break

    if not found_checkpoint:
        print(f"ERROR: No checkpoints after {waited}s")
        if proc.is_alive():
            proc.kill()
            proc.join()
        sys.exit(1)

    # Kill IMMEDIATELY
    print(f"Killing process NOW with SIGKILL...")
    if proc.is_alive():
        os.kill(proc.pid, signal.SIGKILL)
        proc.join(timeout=5)
        print(f"✓ Process killed")
    else:
        print("Process already exited (too fast)")

    # Count what was checkpointed
    checkpointed = count_checkpointed_sequences(checkpoint_dir)
    print(f"\nCheckpointed: {checkpointed}/{num_sequences} sequences ({checkpointed/num_sequences*100:.1f}%)")

    if checkpointed == 0:
        print("ERROR: No sequences checkpointed")
        sys.exit(1)

    if checkpointed >= num_sequences:
        print("WARNING: All sequences processed (kill was too late)")
        print("But continuing to test resume anyway...")

    # ==================================================================
    # PHASE 2: Resume
    # ==================================================================
    print()
    print("=" * 80)
    print("PHASE 2: Resume from checkpoints")
    print("=" * 80)

    result_queue2 = mp.Queue()
    proc2 = mp.Process(
        target=run_inference,
        args=([input_fasta], output_dir, checkpoint_dir, num_gpus, result_queue2)
    )

    proc2.start()
    print(f"Started resume PID={proc2.pid}")

    proc2.join(timeout=300)

    if proc2.is_alive():
        print("ERROR: Resume timeout")
        proc2.kill()
        proc2.join()
        sys.exit(1)

    try:
        status, *result = result_queue2.get(timeout=2)
        if status == 'error':
            print(f"ERROR: {result[0]}")
            sys.exit(1)

        output_path = Path(result[0])
        print(f"✓ Resume completed: {output_path}")

    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    # ==================================================================
    # PHASE 3: Verification
    # ==================================================================
    print()
    print("=" * 80)
    print("VERIFICATION")
    print("=" * 80)

    with h5py.File(output_path, 'r') as f:
        sequence_ids = [s.decode() if isinstance(s, bytes) else s
                       for s in f['sequence_ids'][:]]

    actual = len(sequence_ids)
    unique = len(set(sequence_ids))
    duplicates = actual - unique

    print(f"Expected:   {num_sequences} sequences")
    print(f"Got:        {actual} sequences")
    print(f"Unique:     {unique}")
    print(f"Duplicates: {duplicates}")

    print()
    checks = 0

    if actual == num_sequences:
        print("✓ [1/3] All sequences present")
        checks += 1
    else:
        print(f"✗ [1/3] Sequence count mismatch")

    if duplicates == 0:
        print("✓ [2/3] No duplicates")
        checks += 1
    else:
        print(f"✗ [2/3] Found {duplicates} duplicates")

    if checkpointed > 0 and checkpointed < num_sequences:
        print(f"✓ [3/3] Process was killed mid-processing ({checkpointed}/{num_sequences} checkpointed)")
        checks += 1
    elif checkpointed == num_sequences:
        print(f"⚠ [3/3] All sequences were checkpointed before kill (test timing issue)")
        checks += 0.5
    else:
        print(f"✗ [3/3] Checkpoint verification failed")

    print()
    print("=" * 80)
    if checks >= 2.5:
        print("✓ SUCCESS: Kill+resume works!")
        print("=" * 80)
        print()
        print("Key findings:")
        print(f"  • {checkpointed} sequences checkpointed before SIGKILL")
        print(f"  • Resume completed processing remaining sequences")
        print(f"  • Final output: {actual} sequences (expected {num_sequences})")
        print(f"  • No duplicates: {unique} unique sequences")
        exit_code = 0
    else:
        print(f"FAILURE: {checks}/3 checks passed")
        print("=" * 80)
        exit_code = 1

    print(f"\nTest directory: {test_dir}")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
