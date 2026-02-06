# VirNucPro FastESM2 Migration

## What This Is

An experimental migration of the VirNucPro viral sequence classification system from ESM2 3B to FastESM2_650 embeddings. This tests whether the speed improvements from FastESM2's optimizations and smaller model size (650M vs 3B parameters) justify any potential accuracy loss on viral classification tasks.

## Core Value

Fast, accurate viral sequence classification on large datasets - enabling researchers to process thousands of sequences in minutes instead of hours while maintaining reliable classification performance.

## Requirements

### Validated

<!-- Existing system capabilities (shipped and working) -->

- ✓ Viral vs non-viral sequence classification using deep learning — existing
- ✓ DNA sequence preprocessing with six-frame translation — existing
- ✓ DNABERT-S DNA embeddings (384-dim) — existing
- ✓ ESM2 3B protein embeddings (2560-dim) — existing
- ✓ MLP classifier training with early stopping — existing
- ✓ Batch prediction pipeline with result aggregation — existing
- ✓ Support for 300bp and 500bp sequence chunks — existing
- ✓ GPU-accelerated feature extraction and inference — existing

### Active

<!-- Migration experiment scope -->

- [ ] FastESM2_650 model integration for protein embeddings
- [ ] Updated feature extraction pipeline using FastESM2_650
- [ ] Retrained MLP classifier on FastESM2 embeddings
- [ ] Test dataset creation (random sample from existing data)
- [ ] Automated metrics comparison (F1, accuracy, precision, recall) against ESM2 3B baseline
- [ ] Speed benchmarking on real-world samples
- [ ] Side-by-side model comparison capability (both ESM2 3B and FastESM2_650 available)

### Out of Scope

- Production deployment — this is validation only, deployment decision comes after testing
- Trying other FastESM2 model sizes (300M, 1B) — focused on 650M for this experiment
- Modifying DNA embedding pipeline — DNABERT-S stays unchanged
- Changing MLP classifier architecture — keep existing architecture for fair comparison
- Automated model selection — manual decision based on test results

## Context

**Current System:**
- Uses ESM2 3B (3 billion parameters) for protein sequence embeddings
- Too slow on large viral sequence datasets - bottleneck for research workflows
- Trained models: `300_model.pth` and `500_model.pth` (6.8 MB each)
- Pipeline: DNA chunking → six-frame translation → DNABERT-S + ESM2 embeddings → MLP classifier

**Migration Rationale:**
- FastESM2 offers architectural optimizations for faster inference
- Smaller model (650M params) reduces memory footprint and compute time
- Synthyra's FastESM2_650 provides good balance of speed and capability
- Need to validate that speed gains don't compromise classification accuracy

**Testing Approach:**
1. Train new model with FastESM2_650 embeddings
2. Create random sample test set from existing data
3. Compare quantitative metrics (F1, accuracy) against ESM2 3B baseline
4. Run manual real-world samples to assess speed improvements
5. Decision point: is the tradeoff worth it?

## Constraints

- **Model Source**: Must use Synthyra/FastESM2_650 from HuggingFace (https://huggingface.co/Synthyra/FastESM2_650)
- **Accuracy Target**: Ideally maintain ESM2 3B accuracy levels, but acceptable to lose some performance if speed gain justifies it
- **Platform**: Linux aarch64, CUDA-capable GPU
- **Compatibility**: Must work with existing DNABERT-S embeddings and MLP classifier architecture
- **Testing**: Must support manual testing workflow for real-world validation

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Start with FastESM2_650 (not larger variants) | Balance between speed improvement and capability; 650M is 5x smaller than 3B | — Pending |
| Keep DNABERT-S unchanged | DNA embeddings aren't the bottleneck; focus optimization on protein embeddings | — Pending |
| Random sample test set | Representative cross-section of viral families for fair comparison | — Pending |
| Side-by-side comparison approach | Need both models available to validate metrics and speed claims | — Pending |

---
*Last updated: 2026-02-06 after initialization*
