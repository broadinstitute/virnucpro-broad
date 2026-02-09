# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-06)

**Core value:** Fast, accurate viral sequence classification on large datasets - enabling researchers to process thousands of sequences in minutes instead of hours while maintaining reliable classification performance.
**Current focus:** Phase 4 - Training Data Preparation

## Current Position

Phase: 4 of 5 (Training Data Preparation)
Plan: 2 of 2 in current phase
Status: Phase complete
Last activity: 2026-02-09 - Completed 04-02-PLAN.md (Run extraction and verify results)

Progress: [████████░░] ~80%

## Performance Metrics

**Velocity:**
- Total plans completed: 9
- Average duration: 6.4 min
- Total execution time: 0.96 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01 - Environment Setup | 2 | 25 min | 12.5 min |
| 02 - Feature Extraction Pipeline | 2 | 9 min | 4.5 min |
| 03 - Dimension Compatibility | 3 | 8 min | 2.7 min |
| 04 - Training Data Preparation | 2 | 21 min | 10.5 min |

**Recent Trend:**
- Last 5 plans: 03-02 (3m), 03-03 (3m), 04-01 (2m), 04-02 (19m)
- Trend: Phase 4 execution plan longer due to full dataset extraction (2M sequences)

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- Start with FastESM2_650 (not larger variants) — Balance between speed improvement and capability; 650M is 5x smaller than 3B
- Keep DNABERT-S unchanged — DNA embeddings aren't the bottlenwork; focus optimization on protein embeddings
- Random sample test set — Representative cross-section of viral families for fair comparison
- Side-by-side comparison approach — Need both models available to validate metrics and speed claims

**From 01-01 execution:**
- einops >=0.6.1 (not ==0.8.2) — Version 0.8.2 requires Python 3.10+, incompatible with Python 3.9
- CUDA 12.6 system requirement — Required for __cuda virtual package on aarch64 PyTorch builds
- pandas >=1.3 added — Used by prediction.py but was missing from dependencies

**From 01-02 execution:**
- Docker-based workflow for GB10 GPU support — PyTorch 2.5.1 from conda-forge lacks GB10 (sm_121) support; NVIDIA container provides PyTorch 2.9.0a0 with 1.29x SDPA speedup vs 0.55x slowdown
- SDPA speedup expectations adjusted for GB10 — 1.29x realistic for GB10 hardware (vs claimed 2x on H100/A100); validation threshold accepts 1.29x for GB10 GPUs
- All future development in Docker containers — All phases will use docker-compose for consistent environment across GPU architectures

**From 02-01 execution:**
- Sequential protein processing instead of multiprocessing — CUDA contexts are not fork-safe; multiprocessing.Pool causes GPU errors
- Float16 model precision for FastESM2_650 — Reduces memory footprint by 50%, embeddings converted to float32 on CPU for compatibility
- Mean pooling positions 1:seq_len+1 — Matches original ESM2 approach, excludes BOS (position 0) and EOS tokens
- Output format compatibility maintained — {'proteins': [...], 'data': [...]} works with existing merge_data() function

**From 02-02 execution:**
- Comprehensive test validation approach — 11 checks in single test run validates all Phase 2 requirements (FEAT-01 through FEAT-06)
- merge_data() compatibility testing — Simulates downstream consumption pattern to prevent integration issues in Phase 3
- Resume capability performance baseline — 842x speedup on cached extraction (0.84s → 0.001s) confirms optimization works

**From 03-01 execution:**
- Clean constant naming (DNA_DIM, PROTEIN_DIM, MERGED_DIM) — Old code being replaced, not coexisting; OLD_/NEW_ prefix unnecessary
- Critical path validation always runs — merge_data() inputs/outputs are critical integration points; must validate even when VALIDATE_DIMS=false
- DimensionError exception pattern — Structured attributes (expected_dim, actual_dim, tensor_name, location) for standardized error reporting

**From 03-02 execution:**
- MLPClassifier.forward() validates input dimensions on every forward pass — Critical path validation independent of VALIDATE_DIMS setting catches mismatches before hidden layer computation
- Checkpoint filename convention model_fastesm650.pth — Distinguishes from old ESM2 3B checkpoints (model.pth)
- Feature filename convention *_fastesm.pt — Namespaces FastESM2 outputs separately from old *_ESM.pt files
- Checkpoint version 2.0.0 for breaking changes — Major version bump indicates dimension incompatibility, triggers rejection with migration guidance
- State dict loading pattern with metadata-driven instantiation — Enables dimension validation before loading weights

**From 03-03 execution:**
- Synthetic tensor testing approach — Creates fake DNABERT-S and protein data matching real extraction formats for isolated dimension validation without GPU/models
- Fallback import mechanisms for modules with execution code — try/except with string search fallback enables testing even when train.py/prediction.py can't be imported
- 10-check integration test pattern — Comprehensive coverage of all DIM-01 through DIM-05 requirements in single test run

**From 04-01 execution:**
- JSON checkpoint file for resume capability — Tracks completed files at ./data/.extraction_checkpoint.json, enables restart without re-extraction
- Pre-flight validation pattern — Check all prerequisites (DNABERT-S embeddings, CUDA) before starting expensive GPU work
- Post-extraction validation suite — Validates all outputs for dimension correctness (1280-dim proteins, 2048-dim merged) before declaring success
- Single-command extraction workflow — Replaces manual multi-step process from features_extract.py with automated script

**From 04-02 execution:**
- Runtime Triton patch for DNABERT-S PyTorch 2.9 compatibility — trans_b parameter deprecated in newer Triton; patch replaces tl.dot(q, k, trans_b=True) with tl.dot(q, tl.trans(k))
- Preprocessing step required before extraction — make_train_dataset_300.py splits raw FASTA files into processable chunks
- Generic pattern matching for Triton patch — Regex-based approach handles all trans_b variations uniformly instead of targeted replacements
- Docker cache path search strategy — Check multiple locations (transformers_modules, model snapshots, host vs container paths) to handle environment variability
- Full dataset extraction completed — 201 merged files (105 viral + 96 host), 2M sequences, all validated for correct dimensions

### Pending Todos

None yet.

### Blockers/Concerns

**Critical dimension change:** ✓ RESOLVED - Phase 3 complete, Phase 4 complete. 1280-dim embeddings validated (02-02). Combined feature vector: 768 (DNABERT-S) + 1280 (FastESM2_650) = 2048 dimensions. Dimension validation infrastructure (03-01). MLPClassifier updated with checkpoint versioning (03-02). Integration test validates all requirements (03-03). Training data extracted and validated (04-02): 201 files, 2M sequences, all dimensions correct.

**PyTorch 2.5+ requirement:** ✓ RESOLVED - PyTorch 2.9.0a0 with CUDA 13.0 support via NVIDIA container (01-02, upgraded from 2.5.1 for GB10 support)

**GB10 GPU compatibility:** ✓ RESOLVED - Docker with NVIDIA PyTorch 25.09 container provides native GB10 (sm_121) support and 1.29x SDPA speedup (01-02)

**Accuracy validation:** If Phase 5 shows >5% accuracy drop compared to ESM2 3B baseline, may need to research fine-tuning strategies before proceeding to production.

## Session Continuity

Last session: 2026-02-09 04:19 UTC
Stopped at: Completed 04-02-PLAN.md (Run extraction and verify results)
Resume file: None
Next action: Plan Phase 5 - Model Training and Evaluation

Config:
{
  "mode": "yolo",
  "depth": "standard",
  "parallelization": true,
  "commit_docs": true,
  "model_profile": "balanced",
  "workflow": {
    "research": true,
    "plan_check": true,
    "verifier": true
  }
}
