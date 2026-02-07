# Project Research Summary

**Project:** VirNucPro - FastESM2_650 Migration
**Domain:** Protein language model integration for viral sequence classification
**Researched:** 2026-02-07
**Confidence:** MEDIUM-HIGH

## Executive Summary

VirNucPro is a viral sequence classification pipeline that combines nucleotide embeddings (DNABERT-S) and protein embeddings (currently ESM2 3B) to classify sequences as viral or non-viral. The migration to FastESM2_650 offers significant performance improvements (2x speed, 80% memory reduction) by replacing the 3B parameter ESM2 model with an optimized 650M parameter variant. However, this is not a simple drop-in replacement due to a critical architectural change: embedding dimensions drop from 2560 to 1280, requiring complete downstream model retraining.

The recommended approach is a phased migration starting with environment validation (PyTorch 2.5+ is mandatory for SDPA optimization), followed by feature extraction pipeline updates, comprehensive validation testing, and finally model retraining with the new embedding dimensions. The key risk is dimension incompatibility - the MLP classifier expects 2944-dimensional input (384 DNABERT-S + 2560 ESM2), but FastESM2 produces 1664-dimensional input (384 + 1280). Any attempt to load old checkpoints or hardcoded dimensions will fail silently or catastrophically.

Critical success factors include: (1) never hardcoding embedding dimensions, (2) implementing strict dimension validation at merge points, (3) completely retraining the MLP classifier rather than attempting transfer learning, and (4) optimizing batch sizes and worker counts to leverage FastESM2's reduced memory footprint. The migration timeline is 8-12 days with proper validation, but cutting corners on dimension compatibility will result in weeks of debugging.

## Key Findings

### Recommended Stack

FastESM2_650 requires a modern PyTorch ecosystem with specific version constraints. The migration moves from Facebook's deprecated `fair-esm` library to HuggingFace's standard `transformers` API, which simplifies deployment but requires API changes throughout the codebase.

**Core technologies:**
- **PyTorch ≥2.5.0**: Critical for SDPA (Scaled Dot-Product Attention) optimization that provides 2x speedup. Older versions fall back to slow attention mechanisms, negating migration benefits.
- **transformers ≥4.30.0, <5.0.0**: HuggingFace library for model loading via AutoModel API. Stay on 4.x series to avoid v5 breaking changes in position embeddings.
- **Remove fair-esm 2.0.0**: Deprecated library (last update Nov 2022) incompatible with HuggingFace API. Must migrate away from `pretrained.load_model_and_alphabet()`.
- **trust_remote_code=True**: Required for FastESM2's custom SDPA implementation when loading from HuggingFace Hub.

**Version constraints validated:**
- Current `transformers==4.30.0` is compatible (no upgrade needed)
- PyTorch version unknown but likely <2.5 (needs verification and upgrade)
- Python 3.9 is compatible (no change needed)

### Expected Features

Migration must maintain all current functionality while gaining performance improvements. The dimension change is the critical breaking change that cascades through the pipeline.

**Must have (table stakes):**
- **Protein embedding extraction** - Core replacement for ESM2 3B, produces 1280-dim embeddings instead of 2560-dim
- **Batch processing support** - Current system handles 10,000 sequences per file with `toks_per_batch=2048` token-based batching
- **Mean-pooled embeddings** - Must compute `outputs.last_hidden_state.mean(dim=1)` to match current `extract_esm()` format
- **GPU acceleration** - Same CUDA workflow as ESM2, critical for performance
- **Compatible output dimensions** - CRITICAL: Total input changes from 2944-dim to 1664-dim, requires MLP retraining
- **Tokenization handling** - New API: `model.tokenizer` instead of `alphabet.get_batch_converter()`
- **File-based caching** - Same `.pt` file format, dimension change transparent to `torch.save()`

**Should have (differentiators):**
- **2x faster inference** - Drop-in benefit from PyTorch 2.5+ SDPA with no code changes once environment updated
- **Lower memory footprint** - 650M params vs 3B = 4.6x smaller, enables larger batch sizes or more parallel workers
- **HuggingFace ecosystem** - Standard transformers API improves integration with ecosystem tools
- **Optimized batch sizing** - Smaller model allows 3-4x larger batches for same memory budget

**Defer (v2+):**
- **Attention map extraction** - Not natively supported with SDPA (requires `output_attentions=True` which is slower)
- **Fine-tuning for virus classification** - May improve beyond zero-shot embeddings but adds complexity
- **Mixed precision (FP16/BF16)** - Further optimization possible but not required for initial migration
- **Multiple model comparison** - A/B testing FastESM2 vs ESM2 3B vs ESM2-650M side-by-side

### Architecture Approach

VirNucPro uses a dual-embedding pipeline: DNA sequences → DNABERT-S (384-dim) and 6-frame translated proteins → ESM2 (2560-dim), concatenated → MLP classifier (2944 → 512 → 2 classes). The architecture is clean with well-defined component boundaries, but the dimension change creates a critical integration point.

**Major components:**
1. **Preprocessing layer** - `split_fasta_chunk()`, `identify_seq()`, `translate_dna()` - No changes needed, independent of embedding models
2. **DNA feature extraction** - `extract_DNABERT_S()` with 8 parallel processes - No changes needed, outputs unchanged 384-dim embeddings
3. **Protein feature extraction** - `extract_esm()` with 2 parallel processes - PRIMARY INTEGRATION POINT: complete rewrite from `pretrained.load_model_and_alphabet()` to `AutoModel.from_pretrained()`, embedding dimension changes from 2560 → 1280
4. **Feature merging** - `merge_data()` concatenates embeddings - DIMENSION COMPATIBILITY CHECK: output changes from 2944-dim to 1664-dim, no code change but validation required
5. **MLP classifier** - `MLPClassifier(input_dim=3328, ...)` in train.py - INPUT DIMENSION UPDATE: must change to `input_dim=1664` and retrain completely
6. **Inference pipeline** - prediction.py end-to-end - ALL DOWNSTREAM CONSUMERS: must validate dimension consistency

**Recommended build order:** Test harness (validate FastESM2 works) → Feature extraction update (`extract_fast_esm()` function) → Dimension updates (MLP input_dim) → Model retraining (new embeddings) → Inference pipeline deployment. Estimated 8-12 days total.

### Critical Pitfalls

Research identified 5 critical pitfalls that will cause migration failure if not addressed proactively. All relate to the dimension change and PyTorch version requirements.

1. **Embedding dimension mismatch breaking downstream MLP** - MLP hardcoded with `input_dim=3328`, but FastESM2 produces 1664-dim merged features. Results in tensor shape errors during load. AVOID: Never hardcode dimensions, use dynamic detection from embedding shapes. Add assertions at merge points.

2. **Model checkpoint state dict key mismatch** - Loading old ESM2 3B checkpoint into FastESM2 pipeline causes shape errors: `[512, 3328]` vs `[512, 1664]`. AVOID: Never reuse old checkpoints, train from scratch. Namespace checkpoints by embedding type: `300_model_fastesm650.pth` not `300_model.pth`.

3. **PyTorch version incompatibility with SDPA** - FastESM2 requires PyTorch 2.5+ for SDPA optimization. Older versions fall back to slow attention or crash. AVOID: Verify PyTorch version before migration, add to pixi.toml dependencies as `pytorch>=2.5,<3.0`.

4. **Batch size and memory assumptions broken** - Code tuned for ESM2 3B uses `processes=2` workers and conservative batch sizes, leaving 60-70% GPU memory unused with FastESM2. AVOID: Profile memory usage, increase batch size 3-4x and workers from 2 to 4-6.

5. **Silent tokenization compatibility assumptions** - Sequence truncation to 1024 tokens was added for ESM2 3B memory limits. FastESM2 can handle 100K tokens but keeping truncation for consistency is safer during migration. AVOID: Document all sequence length constraints, test edge cases with long sequences (2000+ amino acids).

## Implications for Roadmap

Based on research, the migration follows a clear dependency chain: environment setup → feature extraction → validation → retraining → deployment. Each phase addresses specific pitfalls and builds on previous work.

### Phase 0: Environment Validation
**Rationale:** PyTorch 2.5+ is mandatory for SDPA optimization. Without it, FastESM2 runs slower than ESM2 3B, negating the migration. This phase must come first to prevent building on incompatible foundation.
**Delivers:** Verified PyTorch 2.5+ installation, removed fair-esm package, confirmed CUDA compatibility
**Addresses:** Pitfall #3 (PyTorch version incompatibility)
**Avoids:** Building entire pipeline only to discover slow performance due to old PyTorch

**Research flags:** SKIP - environment setup is straightforward, no deep research needed

### Phase 1: Feature Extraction Pipeline Update
**Rationale:** The core integration point is `extract_esm()` function. This must be rewritten to use HuggingFace API before any downstream work. Testing in isolation prevents cascading failures.
**Delivers:** New `extract_fast_esm()` function using `AutoModel.from_pretrained()`, updated tokenization with `model.tokenizer`, dimension validation producing 1280-dim embeddings
**Addresses:** Must-have features (protein embedding extraction, batch processing, mean pooling, GPU acceleration)
**Uses:** transformers ≥4.30.0, PyTorch ≥2.5.0 from Phase 0
**Avoids:** Pitfall #1 (dimension mismatch) via dynamic dimension detection, Pitfall #5 (tokenization edge cases) via explicit truncation documentation

**Research flags:** SKIP - HuggingFace transformers API is well-documented, standard patterns apply

### Phase 2: Dimension Compatibility Updates
**Rationale:** Once feature extraction produces 1280-dim embeddings, all downstream code expecting 2560-dim must be updated. This is mechanical but error-prone if rushed.
**Delivers:** Updated `MLPClassifier` with `input_dim=1664`, dimension assertions in `merge_data()`, checkpoint metadata including embedding dimensions
**Addresses:** Feature merging compatibility check, MLP classifier architecture update
**Avoids:** Pitfall #1 (hardcoded dimensions), Pitfall #2 (checkpoint namespace collision)
**Implements:** Dimension validation pattern from architecture research

**Research flags:** SKIP - straightforward dimension arithmetic, no domain research needed

### Phase 3: Feature Re-extraction for Training Data
**Rationale:** Cannot train new MLP until training dataset embeddings regenerated with FastESM2. This phase is compute-intensive but straightforward.
**Delivers:** All training data re-embedded with FastESM2 (1280-dim protein + 384-dim DNA = 1664-dim merged), validated .pt file dimensions
**Uses:** `extract_fast_esm()` from Phase 1
**Addresses:** Training data preparation for retraining
**Avoids:** Pitfall #4 (batch size underutilization) via memory profiling and worker count optimization

**Research flags:** SKIP - applies existing feature extraction, no new patterns

### Phase 4: Model Retraining and Validation
**Rationale:** New embedding dimensions require complete MLP retraining. This is high-risk phase that determines migration success - if accuracy drops significantly, may need to abort migration.
**Delivers:** New `model.pth` checkpoint trained on 1664-dim features, accuracy metrics compared to ESM2 3B baseline, speed benchmarks validating 2x improvement
**Addresses:** Should-have feature (similar downstream performance)
**Avoids:** Pitfall #2 (checkpoint incompatibility) via strict versioning, "looks done but isn't" trap via comprehensive validation
**Critical validation:** Must achieve comparable accuracy to ESM2 3B model on test set

**Research flags:** POTENTIAL - if accuracy drops >5%, may need research on transfer learning or fine-tuning strategies

### Phase 5: Inference Pipeline Deployment
**Rationale:** Final integration into prediction.py. Low-risk if previous phases validated properly.
**Delivers:** Updated prediction.py using FastESM2, end-to-end validation on test sequences, production-ready deployment
**Addresses:** Inference pipeline compatibility
**Avoids:** Pitfall integration gotchas (device placement, model.eval() mode)
**Implements:** Complete pipeline from architecture research

**Research flags:** SKIP - applies validated components, standard deployment patterns

### Phase Ordering Rationale

- **Environment first (Phase 0):** Prevents building on incompatible foundation. PyTorch 2.5+ is non-negotiable for SDPA performance.
- **Feature extraction before retraining (Phases 1-3):** Must validate embeddings are correct before committing to weeks of training. Isolation testing catches dimension issues early.
- **Validation before deployment (Phase 4):** Accuracy comparison is the go/no-go decision. If FastESM2 underperforms, can abort before production deployment.
- **Sequential dependencies:** Each phase uses outputs from previous. Cannot parallelize without risking rework.

The grouping avoids the common pitfall of attempting "big bang" migration. Breaking into phases allows validation gates at each step, preventing cascading failures from dimension mismatches.

### Research Flags

**Phases likely needing deeper research during planning:**
- **Phase 4 (Model Retraining):** IF accuracy drops >5% from baseline, will need `/gsd:research-phase` on transfer learning strategies, fine-tuning approaches, or hyperparameter optimization for smaller embedding space.

**Phases with standard patterns (skip research-phase):**
- **Phase 0:** Environment setup is standard DevOps, well-documented in PyTorch and HuggingFace docs
- **Phase 1:** HuggingFace transformers API has extensive documentation and examples
- **Phase 2:** Dimension updates are mechanical code changes
- **Phase 3:** Feature extraction is repetitive application of Phase 1 work
- **Phase 5:** Deployment follows standard patterns from existing prediction.py

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | Official HuggingFace model card, PyTorch docs, and Synthyra documentation verified. Version requirements clear. |
| Features | HIGH | Current codebase analyzed, API differences documented in official transformers docs. Dimension change validated via model cards. |
| Architecture | HIGH | Direct codebase inspection (units.py, train.py, prediction.py). Component boundaries clear, integration points identified. |
| Pitfalls | MEDIUM | Based on official docs, GitHub issues, and community forums. Some pitfalls extrapolated from dimension change impact, not directly observed. |

**Overall confidence:** MEDIUM-HIGH

Research is grounded in official documentation and codebase analysis, but migration-specific impacts (particularly accuracy after dimension change) cannot be verified without implementation. The dimension change from 2560 → 1280 is architectural and unavoidable, but impact on downstream classification accuracy is estimated based on research papers showing medium-sized models perform well, not direct measurement with this dataset.

### Gaps to Address

Areas where research was inconclusive or needs validation during implementation:

- **Optimal batch size for FastESM2 on available GPU:** Theoretical calculations suggest 3-4x increase possible, but actual optimal batch size depends on GPU memory (unknown), sequence length distribution in dataset (unknown), and CUDA version compatibility. HANDLE: Profile during Phase 1, incrementally increase batch size until 85-90% memory utilization.

- **Accuracy impact of embedding dimension reduction:** Research shows medium-sized models (650M) perform comparably to large models (3B) on many tasks, particularly with <10K observations (per Nature Scientific Reports 2025). However, viral classification may have unique characteristics. HANDLE: Phase 4 must include rigorous accuracy comparison. If >5% drop, consider fine-tuning FastESM2 on viral protein data.

- **Worker count optimization for parallelization:** Current system uses 2 processes for protein extraction (ESM2 3B constraint), 8 for DNA extraction. FastESM2's reduced memory may allow 4-6 protein workers, but DNABERT-S may become bottleneck. HANDLE: Phase 3 should profile both DNA and protein extraction concurrently, balance worker allocation to maximize throughput.

- **Sequence length truncation necessity:** Current code truncates to 1024 tokens for ESM2 3B memory constraints. FastESM2 can handle much longer sequences, but removing truncation requires MLP retraining with variable-length embeddings. HANDLE: Keep 1024 truncation for Phase 1-4 consistency. Defer long sequence handling to v2 after baseline migration validated.

- **Transfer learning viability:** Could potentially keep MLP output layers and only retrain input layer, but dimension mismatch (2560 → 1280) makes this complex. Research suggests full retraining is cleaner for embedding changes. HANDLE: Default to full retraining (simpler, more reliable). Only explore transfer learning if Phase 4 retraining is unexpectedly slow or data-limited.

## Sources

### Primary (HIGH confidence)
- [Synthyra/FastESM2_650 - HuggingFace](https://huggingface.co/Synthyra/FastESM2_650) - Official model card, architecture specs, usage examples
- [HuggingFace Transformers ESM Documentation](https://huggingface.co/docs/transformers/en/model_doc/esm) - API reference for AutoModel and tokenizer
- [NVIDIA BioNeMo ESM-2 Documentation](https://docs.nvidia.com/bionemo-framework/2.0/models/esm2/) - Architecture specifications for ESM2 variants
- [PyTorch SDPA Tutorial](https://pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html) - Version requirements and performance characteristics
- Current VirNucPro codebase (units.py, features_extract.py, train.py, prediction.py) - Direct analysis

### Secondary (MEDIUM confidence)
- [Synthyra/FastPLMs GitHub](https://github.com/Synthyra/FastPLMs) - Implementation details, benchmarks
- [Medium-sized protein language models perform well - Nature Scientific Reports 2025](https://www.nature.com/articles/s41598-025-05674-x) - Performance validation for 650M models
- [Efficient inference of protein language models - iScience](https://www.cell.com/iscience/fulltext/S2589-0042(25)01756-0) - Memory optimization strategies
- GitHub issues and forums on ESM2 migration, checkpoint loading, dimension mismatches

### Tertiary (LOW confidence)
- Speedup benchmarks (2x faster) from Synthyra documentation - not independently verified, depends on hardware and sequence lengths
- Batch size optimization estimates (3-4x increase) - theoretical based on model size ratio, actual depends on GPU and workload

---
*Research completed: 2026-02-07*
*Ready for roadmap: YES*
