# Checkpoint Kill+Resume Test Report

**Date:** 2026-02-06
**Test:** End-to-end kill+resume functionality validation

## Summary

Attempted to validate kill+resume functionality with actual process termination. **Test partially successful** - identified checkpoint path bug that needs fixing.

## Test Results

### ✓ What Works

1. **Multi-GPU inference runs successfully**
   - 2 GPUs processed 50 sequences in ~20 seconds
   - Final output created: `embeddings.h5` with all 50 sequences
   - No duplicates, complete coverage

2. **Checkpoints ARE created**
   - Found checkpoint files: `batch_00000.pt`, `batch_00001.pt`
   - `.done` markers present
   - Checkpoint data appears valid (284KB and 72KB files)

3. **Worker coordination works**
   - Both GPU workers spawned correctly
   - Async monitoring functional
   - Shard aggregation successful

### ✗ Issues Found

#### 1. Checkpoint Path Bug (CRITICAL)

**Problem:** Checkpoint files created in doubly-nested directory structure

**Expected path:**
```
checkpoints/
  shard_0/
    batch_00000.pt
    batch_00000.pt.done
```

**Actual path:**
```
checkpoints/
  shard_0/
    shard_0/    <-- EXTRA NESTING
      batch_00000.pt
      batch_00000.pt.done
```

**Error logs:**
```
ERROR - Failed to write checkpoint batch_00000.pt: Checkpoint batch_00000.pt not found at expected path /tmp/checkpoint_test_z691jq8c/checkpoints/shard_0/batch_00000.pt, cannot update manifest
```

**Impact:**
- Checkpoints are created but manifest can't find them
- Resume functionality will fail (can't locate checkpoints)
- This is a regression introduced in Phase 9 integration

**Root cause:** Likely in `gpu_worker.py` or `AsyncInferenceRunner` where `checkpoint_dir` is passed. The worker is probably adding `shard_{rank}` when it's already in the path.

#### 2. Test Timing Issue

**Problem:** With only 50 sequences, inference completes in ~20 seconds before we can kill the process mid-batch.

**Solution:** Need 500-1000 sequences for realistic kill testing.

## Detailed Logs

### Successful Run (50 sequences, 2 GPUs)

- Start: 10:11:22
- Workers spawned: 10:11:22 (ranks 0 and 1)
- Model loaded: ~10:11:38 (16s for ESM-2 3B loading)
- Processing: 10:11:38 - 10:11:40 (2s for 50 sequences!)
- Aggregation: 10:11:42
- **Total time: 20 seconds**

### Checkpoint Errors

Both workers reported:
```
10:11:40 - ERROR - Failed to write checkpoint batch_00000.pt:
  Checkpoint batch_00000.pt not found at expected path
  /tmp/.../checkpoints/shard_0/batch_00000.pt,
  cannot update manifest
```

But checkpoints actually exist at:
```
/tmp/.../checkpoints/shard_0/shard_0/batch_00000.pt  ✓ (284 KB)
/tmp/.../checkpoints/shard_0/shard_0/batch_00001.pt  ✓ (72 KB)
```

### Packing Warnings

Many warnings about low packing efficiency (1-2%), but this is expected with:
- Only 50 total sequences
- Small batch sizes (batch_size=4)
- No sequence packing enabled in test config

## Recommendations

### Immediate (Blocking Phase 9 completion)

1. **Fix checkpoint path bug**
   - Check `gpu_worker.py` lines ~184-294 where `checkpoint_dir` is used
   - Verify `AsyncInferenceRunner` initialization doesn't double-add `shard_{rank}`
   - The path should be: `checkpoint_dir / f"shard_{rank}"` (constructed ONCE, not twice)

2. **Add integration test with actual checkpoints**
   - Modify `test_checkpoint_integration.py` to verify actual file paths
   - Add assertion that checkpoints exist at expected locations (not nested)

### For Kill+Resume Manual Testing

After fixing the path bug:

1. Use larger dataset (500-1000 sequences)
2. Run multi-GPU inference with checkpointing
3. After ~30 seconds, send `SIGKILL` to one worker PID
4. Restart inference
5. Verify:
   - Resume from checkpoints (no reprocessing)
   - All sequences in final output
   - No duplicates

## Files for Investigation

- `/home/unix/carze/projects/virnucpro-broad-2.0/virnucpro/pipeline/gpu_worker.py:184-294`
- `/home/unix/carze/projects/virnucpro-broad-2.0/virnucpro/pipeline/async_inference.py:180-195` (checkpoint_dir initialization)
- `/home/unix/carze/projects/virnucpro-broad-2.0/virnucpro/pipeline/checkpoint_writer.py` (manifest update logic)

## Test Artifacts

- Test script: `test_kill_resume_simple.py` ✓
- Test directory (last run): `/tmp/checkpoint_test_z691jq8c` ✓
- Full logs: `/tmp/kill_resume_test.log` ✓

## Next Steps

1. Fix checkpoint path double-nesting bug
2. Run kill+resume test with larger dataset
3. Add regression test to prevent path nesting
4. Mark Phase 9 VERIFICATION.md truth #3 as VERIFIED after manual test passes
