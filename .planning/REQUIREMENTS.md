# Requirements: VirNucPro FastESM2 Migration

**Defined:** 2026-02-07
**Core Value:** Fast, accurate viral sequence classification on large datasets - enabling researchers to process thousands of sequences in minutes instead of hours while maintaining reliable classification performance.

## v1 Requirements

Requirements for initial FastESM2_650 migration experiment. Each maps to roadmap phases.

### Environment Setup

- [x] **ENV-01**: PyTorch 2.5+ installed with CUDA support
- [x] **ENV-02**: fair-esm 2.0.0 dependency removed from environment
- [x] **ENV-03**: HuggingFace transformers library verified compatible (≥4.30.0)
- [x] **ENV-04**: FastESM2_650 model can be loaded from HuggingFace Hub with trust_remote_code=True
- [x] **ENV-05**: SDPA (Scaled Dot-Product Attention) functionality validated on target GPU

### Feature Extraction

- [ ] **FEAT-01**: New extract_fast_esm() function implemented using HuggingFace AutoModel API
- [ ] **FEAT-02**: FastESM2_650 tokenizer correctly processes protein sequences
- [ ] **FEAT-03**: Mean-pooled embeddings extracted from last_hidden_state (1280-dim output)
- [ ] **FEAT-04**: Batch processing maintained with configurable batch size
- [ ] **FEAT-05**: GPU acceleration working (model and tensors on correct device)
- [ ] **FEAT-06**: Feature extraction outputs saved to .pt files in same format as ESM2 pipeline

### Dimension Compatibility

- [ ] **DIM-01**: merge_data() function produces 1664-dim concatenated features (384 DNA + 1280 protein)
- [ ] **DIM-02**: Dimension validation assertions added at merge points to catch mismatches
- [ ] **DIM-03**: MLPClassifier updated with input_dim=1664 (changed from 2944)
- [ ] **DIM-04**: Checkpoint metadata includes embedding model type and dimensions
- [ ] **DIM-05**: Old ESM2 3B checkpoints cannot be accidentally loaded with FastESM2 pipeline (namespace protection)

### Model Training

- [ ] **TRAIN-01**: All training data re-extracted with FastESM2_650 embeddings
- [ ] **TRAIN-02**: New MLP classifier trained from scratch on 1664-dim features
- [ ] **TRAIN-03**: Training pipeline validates input dimensions before starting
- [ ] **TRAIN-04**: New model checkpoint saved with clear naming (e.g., model_fastesm650.pth)
- [ ] **TRAIN-05**: Training metrics logged (loss, accuracy per epoch)

### Testing & Validation

- [ ] **TEST-01**: Test dataset created from random sample of existing data
- [ ] **TEST-02**: Automated metrics calculated (F1, accuracy, precision, recall)
- [ ] **TEST-03**: FastESM2_650 model metrics compared against ESM2 3B baseline
- [ ] **TEST-04**: Speed benchmark completed on real-world sample sequences
- [ ] **TEST-05**: Manual testing workflow validated (user can run predictions end-to-end)
- [ ] **TEST-06**: Both ESM2 3B and FastESM2_650 pipelines available for side-by-side comparison

## v2 Requirements

Deferred to future work after initial migration validated.

### Performance Optimization

- **PERF-01**: Batch size optimized for FastESM2_650 memory footprint (target: 3-4x larger than ESM2 3B)
- **PERF-02**: Worker count increased for protein extraction (from 2 to 4-6 processes)
- **PERF-03**: Memory profiling completed to maximize GPU utilization
- **PERF-04**: End-to-end throughput benchmarked on large dataset (1000+ sequences)

### Advanced Features

- **ADV-01**: Support for longer sequences (>1024 tokens) without truncation
- **ADV-02**: Mixed precision (FP16/BF16) inference for additional speedup
- **ADV-03**: Fine-tuning FastESM2_650 on viral protein sequences for improved accuracy
- **ADV-04**: A/B testing framework for comparing multiple models

## Out of Scope

Explicitly excluded from this migration experiment. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| Production deployment | This is validation only - deployment decision comes after testing results |
| Trying other FastESM2 sizes (300M, 1B) | Focused on 650M for this experiment, can explore others later |
| Modifying DNA embedding pipeline (DNABERT-S) | DNA embeddings are not the bottleneck, keep unchanged |
| Changing MLP classifier architecture | Keep existing architecture for fair comparison with ESM2 3B |
| Automated model selection based on metrics | Manual decision after reviewing test results |
| Fine-tuning for production use | Defer to v2 - validate zero-shot embeddings first |
| Multi-GPU distributed training | Single GPU sufficient for experiment scope |
| Attention map visualization | Not required for classification task, SDPA doesn't support it anyway |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| ENV-01 | Phase 1 | Complete |
| ENV-02 | Phase 1 | Complete |
| ENV-03 | Phase 1 | Complete |
| ENV-04 | Phase 1 | Complete |
| ENV-05 | Phase 1 | Complete |
| FEAT-01 | Phase 2 | Pending |
| FEAT-02 | Phase 2 | Pending |
| FEAT-03 | Phase 2 | Pending |
| FEAT-04 | Phase 2 | Pending |
| FEAT-05 | Phase 2 | Pending |
| FEAT-06 | Phase 2 | Pending |
| DIM-01 | Phase 3 | Pending |
| DIM-02 | Phase 3 | Pending |
| DIM-03 | Phase 3 | Pending |
| DIM-04 | Phase 3 | Pending |
| DIM-05 | Phase 3 | Pending |
| TRAIN-01 | Phase 4 | Pending |
| TRAIN-02 | Phase 5 | Pending |
| TRAIN-03 | Phase 5 | Pending |
| TRAIN-04 | Phase 5 | Pending |
| TRAIN-05 | Phase 5 | Pending |
| TEST-01 | Phase 5 | Pending |
| TEST-02 | Phase 5 | Pending |
| TEST-03 | Phase 5 | Pending |
| TEST-04 | Phase 5 | Pending |
| TEST-05 | Phase 5 | Pending |
| TEST-06 | Phase 5 | Pending |

**Coverage:**
- v1 requirements: 26 total
- Mapped to phases: 26
- Unmapped: 0
- Coverage: 100%

---
*Requirements defined: 2026-02-07*
*Last updated: 2026-02-07 after roadmap creation*
