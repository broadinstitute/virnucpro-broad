# Phase 2: Feature Extraction Pipeline - Context

**Gathered:** 2026-02-07
**Status:** Ready for planning

<domain>
## Phase Boundary

Implement FastESM2_650 protein embedding extraction using HuggingFace API to produce 1280-dim embeddings. This phase focuses solely on training data preparation - the inference pipeline (prediction.py) is being rewritten separately and is out of scope.

</domain>

<decisions>
## Implementation Decisions

### API Integration Approach
- Load model once at start, keep in GPU memory for entire extraction run
- Process all sequences without reloading model (training workflow optimization)
- Always use SDPA optimization (hardcoded, validated in Phase 1)
- No CPU fallback mode - assume GPU memory sufficient for model + batch

### Batch Processing Strategy
- Maintain original ESM2 parameters for fair comparison: `truncation_seq_length=1024`, `toks_per_batch=2048`
- Support batch file processing (optimize to process multiple files efficiently, not single file per call)
- Use tqdm progress bar (matches existing DNABERT_S extraction style)
- Built-in parallel processing within extract_fast_esm() function

### Error Handling & Validation
- Skip and log failures - continue processing remaining sequences on errors (resilient approach)
- Validate embeddings before saving:
  - Dimension check: verify exactly 1280-dim output
  - NaN/Inf detection: catch invalid values
  - Sequence count match: verify embeddings count matches input sequences
- Informative logging: log start/finish, major milestones (every 100 files), warnings/errors
- Write failures to separate log file (extraction_failures.log)

### File I/O and Data Format
- Output format: `{'proteins': [labels], 'data': [embeddings]}` (exact match to ESM2 structure)
- File naming: keep `_ESM.pt` suffix (maintains compatibility with merge_data() and downstream code)
- Skip existing files: don't re-extract if output .pt file already exists (resume capability)

### Claude's Discretion
- Exact parallel processing implementation (multiprocessing vs threading given GPU constraints)
- Progress bar formatting details
- Retry logic for transient GPU errors
- Failure log format and detail level

</decisions>

<specifics>
## Specific Ideas

- Batch processing optimization is acceptable as long as final outputs (.pt files with embeddings) remain identical to what single-file processing would produce
- Maintain exact same parameters as original ESM2 3B extraction (`truncation_seq_length=1024`, `toks_per_batch=2048`) for fair comparison
- Training-only scope: extract_fast_esm() is for Phase 4 training data re-extraction, not for inference

</specifics>

<deferred>
## Deferred Ideas

None - discussion stayed within phase scope

</deferred>

---

*Phase: 02-feature-extraction-pipeline*
*Context gathered: 2026-02-07*
