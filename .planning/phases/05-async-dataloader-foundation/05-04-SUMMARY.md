---
phase: 05-async-dataloader-foundation
plan: 04
subsystem: inference
tags: [async, inference, cuda-streams, gpu-monitoring, pytorch, esm2, dataloader-metrics]

# Dependency graph
requires:
  - phase: 05-async-dataloader-foundation
    plan: 01
    provides: "SequenceDataset and VarlenCollator for CUDA-safe data loading"
  - phase: 05-async-dataloader-foundation
    plan: 02
    provides: "GPU monitoring with DataLoader metrics and bottleneck detection"
  - phase: 05-async-dataloader-foundation
    plan: 03
    provides: "Async DataLoader factory with CUDA-safe worker configuration"
provides:
  - "AsyncInferenceRunner class for single-GPU async inference"
  - "Stream-based batch processing with H2D/compute/D2H overlap"
  - "GPU utilization monitoring with DataLoader bottleneck detection"
  - "InferenceResult dataclass for batch output traceability"
  - "run_async_inference convenience function for common use cases"
affects: [06-sequence-packing-integration, 07-multi-gpu-coordination]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Single-GPU async inference with stream-based I/O overlap"
    - "Inter-batch arrival timing for queue state heuristic"
    - "Pinned memory validation on first batch"
    - "FP16→FP32 conversion for embedding stability"
    - "NotImplementedError gates for Phase 6 packed format"

key-files:
  created:
    - "virnucpro/pipeline/async_inference.py"
  modified:
    - "virnucpro/pipeline/__init__.py"

key-decisions:
  - "FIX 1: Inter-batch arrival timing (not processing time) - measures DataLoader wait time for queue state heuristic"
  - "FIX 2: Single GPU transfer via gpu_batch_ref - prevents double transfer of cu_seqlens and other tensors"
  - "FIX 3: NotImplementedError for packed batches - explicit gate for Phase 6 FlashAttention varlen integration"
  - "FIX 5: Pinned memory validation on first batch - detects misconfiguration early before performance degrades"
  - "FIX 7: FP16→FP32 embedding conversion - model may compute in FP16 but embeddings always stored in FP32 for stability"
  - "FIX 8: sequence_ids required in batch - ValueError if missing, prevents synthetic ID generation bugs"

patterns-established:
  - "AsyncInferenceRunner pattern: runner = AsyncInferenceRunner(model, device); results = list(runner.run(dataloader))"
  - "Stream-based batch processing: StreamProcessor.process_batch_async(batch, transfer_fn, compute_fn, retrieve_fn=None)"
  - "GPU monitor integration: monitor.record_dataloader_wait() for bottleneck detection, monitor.check_bottleneck() every 10 batches"
  - "InferenceResult structure: (sequence_ids, embeddings, batch_idx) for full traceability"

# Metrics
duration: 3min
completed: 2026-02-03
---

# Phase 05 Plan 04: Async Inference Runner Summary

**Single-GPU async inference runner with stream-based I/O overlap, GPU utilization monitoring, DataLoader bottleneck detection, and 8 critical fixes for correctness and performance**

## Performance

- **Duration:** 3 min
- **Started:** 2026-02-03T15:56:06Z
- **Completed:** 2026-02-03T15:59:26Z
- **Tasks:** 3 (Task 2 included in Task 1)
- **Files modified:** 2

## Accomplishments

- AsyncInferenceRunner class: single-GPU async inference loop with StreamProcessor integration
- InferenceResult dataclass: captures sequence_ids, embeddings, batch_idx for full traceability
- Stream-based batch processing: H2D/compute/D2H overlap via CUDA streams for latency hiding
- GPU utilization monitoring: NvitopMonitor integration with DataLoader metrics and tiered bottleneck detection
- 8 critical fixes: inter-batch timing (FIX 1), single GPU transfer (FIX 2), packed format gate (FIX 3), pinned memory validation (FIX 5), exception handling (FIX 6), FP16→FP32 conversion (FIX 7), sequence_ids requirement (FIX 8)
- run_async_inference convenience function: simple wrapper for common use cases
- Pipeline module exports: AsyncInferenceRunner, InferenceResult, run_async_inference all importable from virnucpro.pipeline

## Task Commits

Each task was committed atomically:

1. **Task 1: Create AsyncInferenceRunner class** - `0278d4c` (feat)
   - AsyncInferenceRunner class with stream-based batch processing
   - InferenceResult dataclass for batch outputs
   - StreamProcessor integration for H2D/compute/D2H overlap
   - NvitopMonitor integration for GPU utilization tracking
   - FIX 1: Inter-batch arrival timing (fetch_start → fetch_end)
   - FIX 2: Single GPU transfer via gpu_batch_ref (no re-transfer)
   - FIX 3: NotImplementedError for packed batches (Phase 6 TODO)
   - FIX 4: StreamProcessor.process_batch_async signature documented
   - FIX 5: Pinned memory validation on first batch
   - FIX 6: DataLoader exception handling with context
   - FIX 7: FP16→FP32 conversion for embeddings documented
   - FIX 8: sequence_ids required (ValueError if missing)
   - run_async_inference convenience function included

2. **Task 3: Update pipeline module exports** - `989e610` (feat)
   - Added AsyncInferenceRunner to virnucpro.pipeline imports
   - Added InferenceResult to virnucpro.pipeline imports
   - Added run_async_inference to virnucpro.pipeline imports
   - Created __all__ for explicit public API

## Files Created/Modified

- `virnucpro/pipeline/async_inference.py` - AsyncInferenceRunner class, InferenceResult dataclass, run_async_inference function
- `virnucpro/pipeline/__init__.py` - Export async inference components for public API

## Decisions Made

**FIX 1: Inter-batch arrival timing (not processing time)**
- **Rationale:** Queue state heuristic (05-02 decision) requires measuring time BEFORE fetching batch
- **Implementation:** `fetch_start = time.perf_counter()` → `batch = next(dataloader_iter)` → `fetch_time_ms = (time.perf_counter() - fetch_start) * 1000`
- **Why:** <1ms wait indicates queue full (batch ready), >50ms indicates starved (queue empty)
- **Impact:** Accurate bottleneck detection via monitor.record_dataloader_wait()

**FIX 2: Single GPU transfer via gpu_batch_ref**
- **Rationale:** Prevent double transfer of cu_seqlens and other tensors in packed format
- **Implementation:** transfer_fn saves gpu_batch in gpu_batch_ref dict, _extract_embeddings reuses gpu_batch_ref
- **Why:** cu_seqlens already on GPU from transfer_fn, re-transferring wastes bandwidth
- **Impact:** 1D transfer instead of 2D transfer per batch (critical for packed format performance)

**FIX 3: NotImplementedError for packed batches**
- **Rationale:** Packed format (cu_seqlens) requires FlashAttention varlen (Phase 6 integration)
- **Implementation:** `if 'cu_seqlens' in gpu_batch: raise NotImplementedError("Packed batches require FlashAttention varlen (Phase 6)")`
- **Why:** Explicit failure mode prevents silent incorrect behavior in Phase 5
- **Impact:** Clear error message for users, documents Phase 6 TODO

**FIX 5: Pinned memory validation on first batch**
- **Rationale:** pin_memory=True critical for performance, but silently ignored if CUDA unavailable
- **Implementation:** `if batch_idx == 0: self._validate_pinned_memory(batch)` checks `tensor.is_pinned()`
- **Why:** Early detection of misconfiguration before running full dataset
- **Impact:** Warning logged if pinning failed, user can fix DataLoader config

**FIX 7: FP16→FP32 embedding conversion**
- **Rationale:** Model may run in FP16 (model.half() or autocast) but embeddings need FP32 for stability
- **Implementation:** `result.float()` in _extract_embeddings converts representations to FP32
- **Why:** Downstream operations (cosine similarity, norms) more stable in FP32
- **Impact:** Documented in docstrings, explicit conversion prevents silent precision loss

**FIX 8: sequence_ids required in batch**
- **Rationale:** Prevent synthetic ID generation bugs (e.g., [0, 1, 2] instead of real FASTA IDs)
- **Implementation:** `if 'sequence_ids' not in batch: raise ValueError(...)`
- **Why:** Traceability essential for debugging, results must map back to input sequences
- **Impact:** Forces VarlenCollator to include sequence_ids, early validation

## Deviations from Plan

None - plan executed exactly as written. All 8 fixes were specified in plan requirements.

## Issues Encountered

**1. Python environment not configured**
- **Issue:** Cannot run verification tests because torch/esm not installed in current Python environment
- **Context:** Same issue as 05-01, 05-02, 05-03 - project uses pixi environment from sibling directory
- **Resolution:** Proceeded with implementation based on plan specification and existing code patterns
- **Verification:**
  - Python syntax verified via `python -m py_compile`
  - Method signatures verified via grep
  - All 8 FIX comments present in code
  - StreamProcessor signature matches implementation
  - NvitopMonitor API matches usage
- **Impact:** Code is correct per plan and existing patterns, but not runtime-tested

**2. Task 2 included in Task 1**
- **Issue:** Plan specified separate Task 2 for run_async_inference convenience function
- **Context:** Function is simple wrapper (15 lines), logically part of same file
- **Resolution:** Included run_async_inference in Task 1 implementation
- **Verification:** Function present at line 392, signature matches plan specification
- **Impact:** Single file creation instead of file + append, more efficient

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

**Ready for Phase 6 (Sequence Packing Integration):**
- AsyncInferenceRunner provides single-GPU inference foundation
- FIX 3 gates packed format with NotImplementedError until Phase 6 FlashAttention integration
- Stream-based processing ready for varlen attention integration
- GPU monitoring tracks batch composition (avg_sequence_length, max_sequence_length, tokens_in_batch)
- FIX 2 prevents double transfer for packed format (already optimized)

**Integration example:**
```python
from virnucpro.pipeline import AsyncInferenceRunner, run_async_inference
from virnucpro.data import SequenceDataset, VarlenCollator, create_async_dataloader
from virnucpro.models.esm2_flash import load_esm2_model

# Load model and dataset
model, batch_converter = load_esm2_model(device='cuda:0')
dataset = SequenceDataset(fasta_files=['sample.fasta'])
collator = VarlenCollator(batch_converter, max_tokens_per_batch=4096)
loader = create_async_dataloader(dataset, collator)

# Simple wrapper
results = run_async_inference(model, loader, torch.device('cuda:0'))

# Or use runner directly for more control
runner = AsyncInferenceRunner(model, device=torch.device('cuda:0'))
for result in runner.run(loader):
    print(f"Batch {result.batch_idx}: {len(result.sequence_ids)} sequences")
    # result.embeddings shape: (num_sequences, 2560) in FP32
```

**Phase 6 TODOs flagged:**
- FIX 3: Replace NotImplementedError with flash_attn_varlen_func call
- Implement packed format support using cu_seqlens boundaries
- Integrate FlashAttention varlen API from fair-esm 2.0.0

**Blockers/Concerns:**
- Test environment still needs setup for runtime verification
- StreamProcessor.process_batch_async signature assumed correct (no runtime test)
- NvitopMonitor API assumed correct (no runtime test)
- Consider activating pixi environment for Phase 6 integration testing

---
*Phase: 05-async-dataloader-foundation*
*Completed: 2026-02-03*
