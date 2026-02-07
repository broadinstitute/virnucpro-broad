---
phase: 02-feature-extraction-pipeline
plan: 01
subsystem: feature-extraction
tags: [fastesm2, transformers, pytorch, embeddings, protein-sequences]

# Dependency graph
requires:
  - phase: 01-environment-setup
    provides: Docker-based workflow with GB10 GPU support and PyTorch 2.9.0a0
provides:
  - extract_fast_esm() function with FastESM2_650 integration
  - Dynamic batching with toks_per_batch=2048
  - Mean-pooled 1280-dim embeddings compatible with downstream merge_data()
  - Resume capability and failure logging
affects: [03-model-architecture-update, 04-training-pipeline, 05-validation]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Dynamic batching by token count for efficient GPU utilization"
    - "Sequential processing for GPU inference (CUDA contexts not fork-safe)"
    - "Embedding validation pipeline (dimensions, NaN, Inf detection)"
    - "Resume capability by skipping existing output files"

key-files:
  created: []
  modified:
    - units.py
    - features_extract.py

key-decisions:
  - "Sequential protein processing instead of multiprocessing (CUDA contexts not fork-safe)"
  - "Float16 model precision for FastESM2_650 to reduce memory footprint"
  - "Output format {'proteins': [...], 'data': [...]} maintains compatibility with merge_data()"
  - "Mean pooling positions 1:seq_len+1 (excludes BOS/EOS) matches original ESM2 approach"

patterns-established:
  - "Helper function pattern: get_batch_indices() for batching logic, validate_embeddings() for quality checks"
  - "Failure logging pattern: append to extraction_failures.log with timestamp|file|sequence|error_type|message"
  - "CUDA OOM recovery: batch splitting with recursive retry for multi-sequence batches"

# Metrics
duration: 4min
completed: 2026-02-07
---

# Phase 02 Plan 01: Feature Extraction Pipeline Summary

**FastESM2_650 integration with dynamic batching, mean-pooled 1280-dim embeddings, and ESM2-compatible output format**

## Performance

- **Duration:** 4 min
- **Started:** 2026-02-07T05:55:38Z
- **Completed:** 2026-02-07T05:59:10Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments

- Implemented extract_fast_esm() with get_batch_indices() for dynamic batching and validate_embeddings() for quality checks
- Replaced deprecated ESM2 3B extraction with FastESM2_650 (650M parameters, float16 precision)
- Output format {'proteins': [labels], 'data': [1280-dim tensors]} maintains compatibility with existing merge_data() function
- Sequential protein file processing replaces multiprocessing for GPU-safe CUDA context handling

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement extract_fast_esm() with helpers in units.py** - `1ef93bb` (feat)
2. **Task 2: Update features_extract.py for FastESM2_650 model loading and sequential processing** - `e0e393d` (feat)

## Files Created/Modified

- `units.py` - Added extract_fast_esm() with FastESM2_650 support, get_batch_indices() for dynamic batching, validate_embeddings() for quality checks, failure logging, CUDA OOM handling
- `features_extract.py` - Load FastESM2_650 at module level (float16, GPU), update process_file_pro() to call extract_fast_esm(), replace protein multiprocessing with sequential loops

## Decisions Made

**Sequential processing instead of multiprocessing for protein extraction:**
- CUDA contexts are not fork-safe - multiprocessing.Pool with GPU models causes errors
- DNA processing (DNABERT-S) keeps multiprocessing.Pool(processes=8) because it uses CPU
- Protein processing now uses sequential for-loops with tqdm progress bars

**Float16 precision for FastESM2_650:**
- Reduces memory footprint by 50% vs float32
- Model is designed for float16 (trust_remote_code=True handles this)
- Embeddings converted to float32 on CPU before saving for compatibility

**Mean pooling positions 1:seq_len+1:**
- Position 0 is BOS token, position seq_len+1 is EOS token
- Exactly matches original ESM2 approach: `t[i, 1:truncate_len+1].mean(0)`
- Ensures embedding represents protein sequence only, not special tokens

**Output format compatibility:**
- Returns {'proteins': [...], 'data': [...]} matching ESM2 output structure
- merge_data() expects this format: `for protein, data in zip(ESM_outfile['proteins'], ESM_outfile['data'])`
- Data list contains 1D tensors (not dicts) - merge_data() uses `protein_data_dict[protein] = data` directly

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None - implementation followed research findings and plan specifications without obstacles.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

**Ready for next phase:**
- extract_fast_esm() produces 1280-dim embeddings (vs 2560-dim from ESM2 3B)
- Output files maintain same naming convention (_ESM.pt) and structure
- merge_data() function requires no modifications
- Existing test/validation infrastructure can use these embeddings

**Critical dimension change:**
- Embeddings reduced from 2560 → 1280 dimensions
- Downstream model architecture must be updated before training
- Phase 3 (Model Architecture Update) will handle dimension adjustments

**Next steps:**
- Phase 02-02: Run feature extraction on test dataset
- Phase 02-03: Validate embedding quality and extraction performance
- Phase 03: Update model architecture for 1280-dim protein embeddings

---
*Phase: 02-feature-extraction-pipeline*
*Completed: 2026-02-07*

## Self-Check: PASSED

All commits verified present in git history.
