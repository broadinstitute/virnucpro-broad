# Roadmap: VirNucPro FastESM2 Migration

## Overview

This roadmap guides the migration from ESM2 3B to FastESM2_650 embeddings for viral sequence classification. The journey starts with environment setup (PyTorch 2.5+ for SDPA optimization), progresses through feature extraction pipeline updates and dimension compatibility changes (2560-dim to 1280-dim embeddings), then re-extracts all training data before retraining the MLP classifier and validating performance against the ESM2 3B baseline. Success is measured by maintaining comparable accuracy while achieving 2x speed improvement.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [x] **Phase 1: Environment Setup** - Establish FastESM2-compatible environment with PyTorch 2.5+ and SDPA support
- [x] **Phase 2: Feature Extraction Pipeline** - Implement FastESM2_650 protein embedding extraction with HuggingFace API
- [x] **Phase 3: Dimension Compatibility** - Update downstream code for 2048-dim merged features (768 DNA + 1280 protein) with validation at merge points
- [ ] **Phase 4: Training Data Preparation** - Re-extract all training data with FastESM2_650 embeddings
- [ ] **Phase 5: Model Training & Validation** - Train new MLP classifier and validate performance against baseline

## Phase Details

### Phase 1: Environment Setup
**Goal**: FastESM2_650 can be loaded and run with optimal SDPA performance on target GPU
**Depends on**: Nothing (first phase)
**Requirements**: ENV-01, ENV-02, ENV-03, ENV-04, ENV-05
**Success Criteria** (what must be TRUE):
  1. PyTorch 2.5+ installed with CUDA support verified working
  2. fair-esm package removed from environment without breaking existing code
  3. FastESM2_650 model loads from HuggingFace Hub with trust_remote_code=True
  4. SDPA functionality validated on target GPU (2x speedup confirmed vs old attention)
  5. transformers library version compatible (4.30.0+) and can tokenize protein sequences
**Plans**: 2 plans

Plans:
- [x] 01-01-PLAN.md -- Configure pixi environment with pinned dependencies and fix fair-esm imports
- [x] 01-02-PLAN.md -- Create validation script (ENV-01 through ENV-05 + SDPA benchmark) and update README

### Phase 2: Feature Extraction Pipeline
**Goal**: New extract_fast_esm() function produces 1280-dim embeddings with batch processing and GPU acceleration
**Depends on**: Phase 1
**Requirements**: FEAT-01, FEAT-02, FEAT-03, FEAT-04, FEAT-05, FEAT-06
**Success Criteria** (what must be TRUE):
  1. extract_fast_esm() function implemented using AutoModel.from_pretrained() API
  2. FastESM2_650 tokenizer correctly processes protein sequences with same truncation behavior as ESM2 3B
  3. Mean-pooled embeddings extracted from last_hidden_state produce exactly 1280-dim output
  4. Batch processing works with configurable batch size and GPU acceleration active
  5. Feature extraction outputs saved to .pt files in same format as existing ESM2 pipeline
  6. Test run on sample sequences completes without errors and produces valid embeddings
**Plans**: 2 plans

Plans:
- [x] 02-01-PLAN.md -- Implement extract_fast_esm() in units.py and update features_extract.py for FastESM2_650
- [x] 02-02-PLAN.md -- End-to-end extraction test with sample sequences and human verification

### Phase 3: Dimension Compatibility
**Goal**: All downstream code updated for 2048-dim merged features (768 DNA + 1280 protein) with validation at merge points
**Depends on**: Phase 2
**Requirements**: DIM-01, DIM-02, DIM-03, DIM-04, DIM-05
**Success Criteria** (what must be TRUE):
  1. merge_data() function produces 2048-dim concatenated features (768 DNA + 1280 protein)
  2. Dimension validation assertions added at merge points catch mismatches before training
  3. MLPClassifier updated with input_dim=2048 parameter
  4. Checkpoint metadata includes embedding model type (fastesm650) and dimensions for version tracking
  5. Old ESM2 3B checkpoints cannot be loaded with FastESM2 pipeline (namespace protection prevents silent failures)
**Plans**: 3 plans

Plans:
- [x] 03-01-PLAN.md -- Add DimensionError, constants, validation functions, and update merge_data() with dimension validation
- [x] 03-02-PLAN.md -- Update MLPClassifier dimensions, checkpoint metadata, prediction pipeline, and namespace protection
- [x] 03-03-PLAN.md -- Integration test validating all DIM-01 through DIM-05 requirements

### Phase 4: Training Data Preparation
**Goal**: All training data re-extracted with FastESM2_650 embeddings and validated for dimension correctness
**Depends on**: Phase 3
**Requirements**: TRAIN-01
**Success Criteria** (what must be TRUE):
  1. All training data re-extracted using extract_fast_esm() function
  2. Extracted .pt files validated to contain exactly 1280-dim protein embeddings
  3. Merged features validated to be 2048-dim (768 + 1280)
  4. Extraction completes without errors or dimension mismatches
  5. Data ready for MLP training with correct input dimensions
**Plans**: TBD

Plans:
- [ ] 04-01: TBD

### Phase 5: Model Training & Validation
**Goal**: New MLP classifier trained on FastESM2 embeddings with performance validated against ESM2 3B baseline
**Depends on**: Phase 4
**Requirements**: TRAIN-02, TRAIN-03, TRAIN-04, TRAIN-05, TEST-01, TEST-02, TEST-03, TEST-04, TEST-05, TEST-06
**Success Criteria** (what must be TRUE):
  1. New MLP classifier trained from scratch on 2048-dim features with training metrics logged
  2. Training pipeline validates input dimensions before starting (prevents silent failures)
  3. New model checkpoint saved with clear naming (model_fastesm650.pth)
  4. Test dataset created from random sample of existing data
  5. Automated metrics calculated (F1, accuracy, precision, recall) for both FastESM2 and ESM2 3B models
  6. FastESM2 model accuracy within acceptable range of ESM2 3B baseline (ideally <5% drop)
  7. Speed benchmark shows 2x improvement on real-world sample sequences
  8. Manual testing workflow validated end-to-end (user can run predictions)
  9. Both ESM2 3B and FastESM2_650 pipelines available for side-by-side comparison
**Plans**: TBD

Plans:
- [ ] 05-01: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 1 -> 2 -> 3 -> 4 -> 5

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Environment Setup | 2/2 | Complete | 2026-02-07 |
| 2. Feature Extraction Pipeline | 2/2 | Complete | 2026-02-07 |
| 3. Dimension Compatibility | 3/3 | Complete | 2026-02-08 |
| 4. Training Data Preparation | 0/TBD | Not started | - |
| 5. Model Training & Validation | 0/TBD | Not started | - |
