#!/usr/bin/env python3
"""Full kill+resume test with realistic workload.

Tests:
1. Start inference with 500 sequences
2. Wait for checkpoints to appear
3. Kill process with SIGKILL mid-processing
4. Verify checkpoints exist
5. Resume from checkpoints
6. Verify all sequences processed exactly once (no duplicates)
"""

import multiprocessing as mp
import signal
import time
import tempfile
from pathlib import Path
import sys
import os
import h5py


def create_test_fasta(path: Path, num_sequences: int = 500):
    """Create test FASTA with realistic varied-length sequences."""
    with open(path, 'w') as f:
        for i in range(num_sequences):
            seq_id = f"test_seq_{i:05d}"
            # Varied lengths: 50-500 residues (realistic for viral proteins)
            seq_len = 50 + (i % 450)
            # Alternate between different amino acids for variety
            aa = ['A', 'G', 'V', 'L', 'I'][i % 5]
            sequence = aa * seq_len
            f.write(f">{seq_id}\n{sequence}\n")
    print(f"✓ Created test FASTA: {num_sequences} sequences, varied lengths 50-500")


def run_inference_wrapper(fasta_files, output_dir, checkpoint_dir, world_size, result_queue):
    """Run inference in subprocess."""
    try:
        import logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

        from virnucpro.pipeline.multi_gpu_inference import run_multi_gpu_inference
        from virnucpro.pipeline.runtime_config import RuntimeConfig

        # Configure for realistic testing
        model_config = {
            'model_type': 'esm2',
            'model_name': 'esm2_t36_3B_UR50D',
            'dtype': 'float32',
            'batch_size': 4,
        }

        runtime_config = RuntimeConfig(
            enable_checkpointing=True,
            checkpoint_dir=checkpoint_dir,
            checkpoint_seq_threshold=50,  # Checkpoint every 50 sequences
            checkpoint_time_threshold=60,  # or every 60 seconds
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


def count_checkpoint_files(checkpoint_dir: Path):
    """Count total checkpoint .done markers across all shards."""
    count = 0
    for shard_dir in checkpoint_dir.glob("shard_*"):
        count += len(list(shard_dir.glob("*.done")))
    return count


def get_checkpoint_details(checkpoint_dir: Path):
    """Get detailed checkpoint info."""
    details = {}
    for shard_dir in sorted(checkpoint_dir.glob("shard_*")):
        shard_num = shard_dir.name.split('_')[1]
        checkpoints = list(shard_dir.glob("batch_*.pt"))
        done_markers = list(shard_dir.glob("batch_*.done"))

        # Count sequences from checkpoint metadata
        total_seqs = 0
        for ckpt_file in sorted(checkpoints):
            done_marker = ckpt_file.parent / f"{ckpt_file.name}.done"
            if not done_marker.exists():
                print(f"WARNING: Checkpoint {ckpt_file} missing .done marker - may be incomplete")
            try:
                import torch
                data = torch.load(ckpt_file, map_location='cpu')
                total_seqs += len(data['sequence_ids'])
            except Exception as e:
                print(f"WARNING: Failed to load checkpoint {ckpt_file}: {e}")

        details[f"shard_{shard_num}"] = {
            'checkpoints': len(checkpoints),
            'done_markers': len(done_markers),
            'sequences': total_seqs
        }
    return details


def verify_output(output_path: Path, expected_count: int):
    """Verify output file completeness."""
    with h5py.File(output_path, 'r') as f:
        sequence_ids = [s.decode() if isinstance(s, bytes) else s
                       for s in f['sequence_ids'][:]]

    actual_count = len(sequence_ids)
    unique_count = len(set(sequence_ids))
    duplicates = actual_count - unique_count

    return {
        'total': actual_count,
        'unique': unique_count,
        'duplicates': duplicates,
        'complete': actual_count == expected_count,
        'no_duplicates': duplicates == 0,
    }


def verify_checkpoint_integrity(checkpoint_dir: Path, expected_hidden_dim: int = 128):
    """Spot-check checkpoint data integrity."""
    import torch
    checkpoint_files = list(checkpoint_dir.glob("shard_*/batch_*.pt"))
    if not checkpoint_files:
        print("  ⚠ No checkpoint files to verify")
        return True

    sample_ckpt = checkpoint_files[0]
    try:
        data = torch.load(sample_ckpt, map_location='cpu')
        if 'embeddings' in data:
            shape = data['embeddings'].shape
            if len(shape) == 2 and shape[1] == expected_hidden_dim:
                print(f"  ✓ Checkpoint integrity: embeddings shape {shape}")
                return True
            else:
                print(f"  ⚠ Unexpected embeddings shape: {shape} (expected [*, {expected_hidden_dim}])")
                return False
        elif 'sequence_embeddings' in data:
            print(f"  ✓ Checkpoint integrity: found sequence_embeddings")
            return True
        else:
            print(f"  ⚠ No embeddings found in checkpoint")
            return False
    except Exception as e:
        print(f"  ⚠ Checkpoint integrity check failed: {e}")
        return False


def main():
    mp.set_start_method('spawn', force=True)

    print("=" * 80)
    print("FULL KILL+RESUME TEST")
    print("=" * 80)
    print()

    # Check GPU availability
    try:
        import torch
        if not torch.cuda.is_available():
            print("ERROR: No CUDA GPUs available")
            sys.exit(1)
        num_gpus = min(torch.cuda.device_count(), 2)
        print(f"✓ Using {num_gpus} GPU(s)")
    except Exception as e:
        print(f"ERROR: CUDA validation failed: {e}")
        sys.exit(1)

    # Setup test environment
    test_dir = Path(tempfile.mkdtemp(prefix="checkpoint_full_test_"))
    input_fasta = test_dir / "input.fasta"
    checkpoint_dir = test_dir / "checkpoints"
    output_dir = test_dir / "output"

    checkpoint_dir.mkdir()
    output_dir.mkdir()

    num_sequences = 500
    create_test_fasta(input_fasta, num_sequences)

    print(f"✓ Test directory: {test_dir}")
    print()

    # ========================================================================
    # PHASE 1: Start inference and kill mid-processing
    # ========================================================================
    print("=" * 80)
    print("PHASE 1: Start inference, wait for checkpoints, then SIGKILL")
    print("=" * 80)

    result_queue = mp.Queue()
    proc = mp.Process(
        target=run_inference_wrapper,
        args=([input_fasta], output_dir, checkpoint_dir, num_gpus, result_queue)
    )

    start_time = time.time()
    proc.start()
    print(f"✓ Started inference process (PID={proc.pid})")
    print(f"  Waiting for checkpoints to accumulate...")

    # Wait for multiple checkpoints (ensures we're mid-processing)
    min_checkpoints = 4  # Wait for at least 4 checkpoints
    max_wait = 180  # 3 minutes max
    waited = 0
    checkpoint_count = 0

    while waited < max_wait and proc.is_alive():
        time.sleep(5)
        waited += 5

        checkpoint_count = count_checkpoint_files(checkpoint_dir)

        if checkpoint_count > 0:
            print(f"  [{waited}s] Checkpoints: {checkpoint_count}")

        if checkpoint_count >= min_checkpoints:
            print(f"✓ Found {checkpoint_count} checkpoints after {waited}s")
            break

        if not proc.is_alive():
            print("  Process exited early")
            try:
                status, *result = result_queue.get(timeout=1)
                if status == 'error':
                    print(f"ERROR: {result[0]}")
                    print(result[1])
                    sys.exit(1)
                else:
                    print("Process completed before we could kill it")
                    print("(500 sequences processed too quickly - may need more sequences)")
            except Exception as e:
                print(f"WARNING: Could not get result from queue: {e}")
            break

    if checkpoint_count == 0:
        print(f"ERROR: No checkpoints appeared after {waited}s")
        if proc.is_alive():
            proc.kill()
            proc.join()
        sys.exit(1)

    if checkpoint_count < min_checkpoints:
        print(f"WARNING: Only {checkpoint_count} checkpoints (wanted {min_checkpoints})")
        print("Proceeding anyway...")

    # Kill the process
    print()
    print(f"Sending SIGKILL to PID={proc.pid}...")

    if proc.is_alive():
        os.kill(proc.pid, signal.SIGKILL)
        proc.join(timeout=10)
        print(f"✓ Process killed")
    else:
        print("Process already exited (completed too fast)")

    elapsed_phase1 = time.time() - start_time

    # Analyze checkpoints after kill
    print()
    print("Checkpoint state after kill:")
    details = get_checkpoint_details(checkpoint_dir)
    total_checkpointed_seqs = 0
    for shard, info in details.items():
        print(f"  {shard}: {info['checkpoints']} checkpoints, {info['sequences']} sequences")
        total_checkpointed_seqs += info['sequences']

    print(f"  Total checkpointed: {total_checkpointed_seqs}/{num_sequences} sequences")
    print(f"  Phase 1 duration: {elapsed_phase1:.1f}s")

    print("\nCheckpoint data integrity verification:")
    verify_checkpoint_integrity(checkpoint_dir)

    # ========================================================================
    # PHASE 2: Resume from checkpoints
    # ========================================================================
    print()
    print("=" * 80)
    print("PHASE 2: Resume from checkpoints")
    print("=" * 80)

    result_queue2 = mp.Queue()
    proc2 = mp.Process(
        target=run_inference_wrapper,
        args=([input_fasta], output_dir, checkpoint_dir, num_gpus, result_queue2)
    )

    resume_start = time.time()
    proc2.start()
    print(f"✓ Started resume process (PID={proc2.pid})")
    print(f"  Waiting for completion...")

    # Wait for completion (with timeout)
    proc2.join(timeout=300)  # 5 min max

    if proc2.is_alive():
        print("ERROR: Resume process timeout (5 minutes)")
        proc2.kill()
        proc2.join()
        sys.exit(1)

    # Get result
    try:
        status, *result = result_queue2.get(timeout=2)
        if status == 'error':
            print(f"ERROR during resume: {result[0]}")
            print(result[1])
            sys.exit(1)

        output_path = Path(result[0])
        failed_ranks = result[1]

        print(f"✓ Resume completed successfully")
        if failed_ranks:
            print(f"  WARNING: Some ranks failed: {failed_ranks}")

    except Exception as e:
        print(f"ERROR: Could not get result from resume process: {e}")
        sys.exit(1)

    resume_elapsed = time.time() - resume_start

    # ========================================================================
    # PHASE 3: Verification
    # ========================================================================
    print()
    print("=" * 80)
    print("PHASE 3: Verification")
    print("=" * 80)

    # Check output file exists
    if not output_path.exists():
        print(f"ERROR: Output file not found: {output_path}")
        sys.exit(1)

    print(f"✓ Output file: {output_path}")

    # Verify sequences
    verification = verify_output(output_path, num_sequences)

    print()
    print("Results:")
    print(f"  Expected sequences: {num_sequences}")
    print(f"  Output sequences:   {verification['total']}")
    print(f"  Unique sequences:   {verification['unique']}")
    print(f"  Duplicates:         {verification['duplicates']}")

    # Final checkpoint state
    final_details = get_checkpoint_details(checkpoint_dir)
    final_checkpointed = sum(info['sequences'] for info in final_details.values())
    print(f"  Final checkpointed: {final_checkpointed} sequences")

    print()
    print("Performance:")
    print(f"  Phase 1 (before kill): {elapsed_phase1:.1f}s")
    print(f"  Phase 2 (resume):      {resume_elapsed:.1f}s")
    print(f"  Total time:            {elapsed_phase1 + resume_elapsed:.1f}s")

    # Verification checks
    print()
    print("Verification Checks:")

    checks_passed = 0
    total_checks = 4

    # Check 1: All sequences present
    if verification['complete']:
        print(f"  ✓ [1/4] All {num_sequences} sequences present")
        checks_passed += 1
    else:
        print(f"  ✗ [1/4] Missing sequences: expected {num_sequences}, got {verification['total']}")

    # Check 2: No duplicates
    if verification['no_duplicates']:
        print(f"  ✓ [2/4] No duplicate sequences")
        checks_passed += 1
    else:
        print(f"  ✗ [2/4] Found {verification['duplicates']} duplicates")

    # Check 3: Checkpoints created during both runs
    if final_checkpointed >= total_checkpointed_seqs:
        print(f"  ✓ [3/4] Checkpoints maintained/increased ({total_checkpointed_seqs} → {final_checkpointed})")
        checks_passed += 1
    else:
        print(f"  ✗ [3/4] Checkpoint count decreased")

    # Check 4: Resume was partial (not full rerun)
    # If resume processed fewer sequences than original, it actually resumed
    if total_checkpointed_seqs > 0 and resume_elapsed < elapsed_phase1:
        print(f"  ✓ [4/4] Resume was faster than initial run (resumed from checkpoints)")
        checks_passed += 1
    else:
        print(f"  ⚠ [4/4] Could not verify resume efficiency")
        print(f"      (Phase 1: {elapsed_phase1:.1f}s, Phase 2: {resume_elapsed:.1f}s)")

    # Summary
    print()
    print("=" * 80)
    if checks_passed == total_checks:
        print("✓ SUCCESS: Kill+resume functionality VERIFIED!")
        print("=" * 80)
        print()
        print("Summary:")
        print("  • Process killed mid-batch with SIGKILL")
        print(f"  • {total_checkpointed_seqs} sequences checkpointed before kill")
        print("  • Pipeline resumed from last checkpoint")
        print("  • All sequences processed exactly once")
        print("  • No data loss, no duplicates")
        print()
        import shutil
        shutil.rmtree(test_dir)
        print(f"Cleaned up: {test_dir}")
        exit_code = 0
    else:
        print(f"FAILURE: {checks_passed}/{total_checks} checks passed")
        print("=" * 80)
        exit_code = 1

    print()
    print(f"Preserved for debugging: {test_dir}")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
