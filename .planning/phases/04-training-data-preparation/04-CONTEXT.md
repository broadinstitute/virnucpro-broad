# Phase 4: Training Data Preparation - Context

**Gathered:** 2026-02-08
**Status:** Ready for planning

<domain>
## Phase Boundary

Re-extract all training data with FastESM2_650 embeddings to prepare for model retraining. This phase takes the existing training dataset and re-processes it through the new FastESM2_650 extraction pipeline. Scope is re-extraction only - not creating new datasets, not modifying which sequences are included, just converting existing data to the new 1280-dim protein embedding format.

</domain>

<decisions>
## Implementation Decisions

### Extraction workflow
- Single command that processes entire dataset from start to finish
- Resume capability: skip already-extracted files if interrupted
- Auto-discover training data from data/ directory (no explicit paths required)
- Full validation suite before completion: dimension checks + count verification + integrity checks

### Data organization
- Replace old *_ESM.pt files with new embeddings (in-place replacement)
- No automatic backup - user responsible for backing up old data if desired
- Reuse existing DNABERT-S embeddings (*_DNABERT_S.pt) - no re-extraction needed
- Match current directory structure exactly (drop-in replacement)

### Progress tracking
- Summary logging: per-file completion with aggregate stats (not per-sequence verbose)
- Progress bar with ETA for visual feedback
- Checkpoint file (.checkpoint) for reliable resume across restarts
- Full statistics on completion: total sequences, files processed, time taken, avg time/sequence, dimension validation results

### Error handling
- Fail fast: stop immediately on any protein extraction error
- Dimension validation failures are critical - halt and fix
- Missing DNABERT-S embeddings cause error (require all DNA data to exist before starting)
- Detailed debugging context in errors: file paths, tensor shapes, stack traces

### Claude's Discretion
- Checkpoint file format and location
- Progress bar library choice (tqdm, rich, etc.)
- Specific validation checks in "full validation suite"
- Batch size for extraction processing

</decisions>

<specifics>
## Specific Ideas

- Should feel like running a single `python extract_training_data.py` command
- Checkpoint resume should be transparent - just re-run same command if interrupted
- Error messages need enough detail for me to debug dimension mismatches

</specifics>

<deferred>
## Deferred Ideas

None - discussion stayed within phase scope

</deferred>

---

*Phase: 04-training-data-preparation*
*Context gathered: 2026-02-08*
