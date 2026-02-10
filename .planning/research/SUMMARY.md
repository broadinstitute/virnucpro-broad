# Project Research Summary

**Project:** VirNucPro v2.5 Model Optimizations Round 2
**Domain:** GPU-accelerated viral nucleotide prediction pipeline
**Researched:** 2026-02-09
**Confidence:** HIGH

## Executive Summary

VirNucPro v2.5 builds on the successful v2.0 async DataLoader architecture to deliver six categories of optimizations: (1) porting DNABERT-S from legacy v1.0 to v2.0 architecture, (2) making ESM-2 model selection configurable, (3) integrating torch.compile for 10-20% speedup, (4) vectorizing hot-path operations, (5) implementing quick wins (env var caching, deque, combined tensor ops), and (6) code quality improvements. The research reveals that **no new dependencies are required** - all improvements leverage existing PyTorch >=2.8.0, fair-esm 2.0.0, and transformers 4.30.0 capabilities.

The recommended approach prioritizes **low-risk, high-impact changes first**: code quality improvements and model flexibility are straightforward refactoring with clear testing patterns, torch.compile integration provides measurable speedup with minimal code changes, and vectorized operations deliver incremental performance gains. **DNABERT-S v2.0 port is the highest-risk feature** (replaces working v1.0 architecture) and should be tackled last after all supporting infrastructure is validated. This port is critical for unlocking multi-GPU scaling - current v1.0 architecture shows 0.96x scaling on 2 GPUs (4% slowdown), while ESM-2's v2.0 achieves 1.87x (93.7% efficiency).

Key risks include: (1) DNABERT-S k-mer tokenization incompatibility with ESM-2's batch_converter requiring custom collator implementation, (2) ESM-2 model swaps producing different embedding dimensions (650M=1280, 3B=2560) breaking checkpoint compatibility, (3) torch.compile recompilation storms with dynamic sequence lengths, and (4) FlashAttention integration differences between ESM-2's custom architecture and DNABERT-S's standard BERT. Mitigation strategies include comprehensive validation phases, checkpoint metadata validation, dynamic shape compilation settings, and dedicated FlashAttention wrapper implementations per model.

## Key Findings

### Recommended Stack

**No stack changes required.** All v2.5 features achievable with current dependencies: PyTorch >=2.8.0 (provides torch.compile, scatter_reduce, cumsum), fair-esm 2.0.0 (supports all ESM-2 model variants 8M-15B), transformers 4.30.0 (DNABERT-S compatible with trust_remote_code), and flash-attn >=2.6.0 (FlashAttention-2 for packed attention). Optional dependency torch-scatter 2.1.1 deferred as LOW priority - native PyTorch operations sufficient for v2.5 scope.

**Core technologies:**
- **PyTorch >=2.8.0**: GPU acceleration, torch.compile (10-20% speedup), vectorized operations (scatter_reduce, cumsum) - already installed, no upgrade needed
- **fair-esm 2.0.0**: ESM-2 model family (650M/3B/15B variants) with unified API - supports model swapping via function name (esm2_t33_650M_UR50D, esm2_t36_3B_UR50D)
- **transformers 4.30.0**: DNABERT-S (zhihan1996/DNABERT-S) with k-mer tokenization - different API than ESM (returns BatchEncoding vs 3-tuple)
- **flash-attn >=2.6.0**: FlashAttention-2 varlen for packed sequences - cannot be compiled by torch.compile (remains as-is, already optimal)

**torch.compile integration details:** Available in PyTorch 2.0+, fully supported in 2.8.0. Recommend "reduce-overhead" mode for inference (targets small batches with CUDA graphs). Expected 10-20% speedup from compiling embeddings/FFN/layer norms (FlashAttention passes through uncompiled). Dynamic shapes (variable cu_seqlens) handled via dynamic=True flag with acceptable recompilation overhead for long runs (1M sequences).

**ESM-2 model variants:** All use same tokenizer (33-token alphabet), same input format, same sequence packing architecture. Key differences: layer count (650M=33, 3B=36, 15B=48), embedding dimension (650M=1280, 3B=2560), memory footprint (650M=3-4GB FP16, 3B=12-16GB FP16), and inference speed (650M ~3-4x faster than 3B). Model selection requires dynamic repr_layers configuration based on layer count.

**DNABERT-S architecture:** 12 transformer layers, 768 hidden dim, 12 attention heads, ~4100 vocab size (k-mer based), 512 max sequence length. Uses BERT-style [CLS]/[SEP] tokens (not ESM's BOS/EOS), attention_mask for padding (not padding_idx), and transformers.AutoTokenizer (not ESM batch_converter). FlashAttention integration requires BERT-specific wrapper extracting from separate Q/K/V Linear layers (vs ESM-2's unified in_proj_weight).

### Expected Features

**Must have (table stakes):**
- **DNABERT-S v2.0 port** - Primary value proposition: fixes 0.96x scaling bottleneck, enables multi-GPU for DNA embeddings matching ESM-2's 1.87x efficiency
- **ESM-2 model selection** - Required for 16GB GPUs (650M fits where 3B cannot), enables speed/quality tradeoffs (650M 3-4x faster for prototyping)
- **Environment variable centralization** - Quality gate: 6 env vars scattered across 6 files, prevents debugging and testing
- **Deque for packed_queue** - Fixes performance anti-pattern: list.pop(0) is O(n), deque.popleft() is O(1)

**Should have (competitive):**
- **torch.compile integration** - 10-20% speedup at minimal risk (single-line change per model), production-ready in PyTorch 2.2+, opt-in via CLI flag
- **Vectorized position IDs** - 5-10x speedup in hot path (negligible overall <0.1% batch time), replaces Python loop with torch.cumsum
- **Vectorized embeddings** - 4-20x speedup removing .item() CUDA syncs, optimizes extraction from packed format
- **Combined .to() calls** - Faster model init, single transfer to(device=device, dtype=torch.float16) vs separate .to(device).half()

**Defer (v2+):**
- **BFD packing** - 2-3% efficiency gain over current FFD, high complexity, FFD achieves 92-94% already sufficient
- **FlashAttention-3** - Requires H100 GPUs, not compatible with RTX 4090 target hardware
- **Dynamic work distribution** - Overkill for current 93.7% multi-GPU efficiency, adds coordinator complexity
- **Tensor pooling** - Profiling shows not a bottleneck, premature optimization
- **torch-scatter library** - Native PyTorch scatter_reduce sufficient, avoid dependency complexity

### Architecture Approach

VirNucPro v2.5 builds on single-process-per-GPU foundation with 90% component reuse between ESM-2 and DNABERT-S. Core architecture: FASTA → DataLoader Workers (CPU I/O) → VarlenCollator (tokenize + pack) → Pinned Memory → CUDA Streams (H2D/compute/D2H overlap) → HDF5 Shards → Aggregator. DNABERT-S integration requires only (1) VarlenCollator tokenization type detection (3 lines), (2) DNABERT-specific FlashAttention wrapper (BERT Q/K/V extraction pattern), and (3) gpu_worker model_type switch (5 lines). Model flexibility requires ESM2Config dataclass with auto-detected repr_layers based on layer count. torch.compile integrates at model loading (wrap entire ESM2WithFlashAttention or DNABERTWithFlashAttention). Vectorized operations replace Python loops in create_position_ids_packed and embedding extraction. Code quality improvements centralize env vars in core/env_config.py (zero imports, lazy evaluation) and extract oversized functions (async_inference.py 526-line run method, gpu_worker.py 414-line function).

**Major components:**
1. **DNABERT-S v2.0 async pipeline** - Reuses IndexBasedDataset, SequenceIndex, GreedyPacker, AsyncInferenceRunner; adds DnabertVarlenCollator and DNABERTWithFlashAttention wrapper
2. **Model configuration system** - ESM2Config dataclass with auto-detection (num_layers → repr_layers), runtime validation (assert max(repr_layers) < model.num_layers), checkpoint metadata (model_name, embedding_dim)
3. **torch.compile integration** - Wraps loaded models with torch.compile(mode='reduce-overhead', dynamic=True, fullgraph=False), warmup phase, recompilation monitoring, VIRNUCPRO_DISABLE_COMPILE rollback
4. **Vectorized operation layer** - create_position_ids_packed_vectorized (torch.cumsum + boundary masking), cu_seqlens_cpu caching (removes .item() syncs), hybrid approach (loop for <threshold, vectorized for large batches)
5. **Environment config module** - core/env_config.py with @lru_cache methods, zero imports, reload() for testing, replaces 17 scattered os.getenv() calls
6. **Function extraction refactoring** - async_inference.run() → _setup_inference, _process_batches, _flush_remaining; gpu_worker → _setup_worker_components, _run_worker_inference, _save_worker_shard

### Critical Pitfalls

1. **DNABERT-S k-mer tokenization incompatible with ESM batch_converter** - transformers returns BatchEncoding {input_ids, attention_mask}, ESM returns (labels, strs, tokens) tuple. VarlenCollator hardcodes ESM interface. Prevention: Create separate DnabertVarlenCollator, verify padding token IDs match, unit test tokenization before integration.

2. **ESM-2 embedding dimension mismatch breaks checkpoints** - 650M produces [seq_len, 1280], 3B produces [seq_len, 2560]. Switching models without checkpoint format update causes shape errors during resume/merge. Prevention: Store model metadata in checkpoint header (model_name, embedding_dim, num_layers), validate compatibility before resume, implement migration tool.

3. **torch.compile dynamic shapes cause recompilation storm** - Variable cu_seqlens lengths trigger recompilation on every unique shape (1000+ kernels × 30-60s = hours). Prevention: Use dynamic=True for symbolic shape tracking, warmup with representative length distribution, monitor cache size, add VIRNUCPRO_DISABLE_COMPILE rollback.

4. **torch.compile + FlashAttention binary compatibility** - Custom CUDA kernels require exact version matching (torch 2.10.0 → CUDA 12.6.3/12.8.1 → triton 3.6.0 → flash-attn 2.8.2). Mismatch causes "undefined symbol" import errors or silent fallback to eager mode. Prevention: Pin exact versions, test single-GPU before multi-GPU, integration test verifying FlashAttention active under compile.

5. **DNABERT-S FlashAttention integration differs from ESM-2** - BERT uses separate query/key/value Linear layers, ESM-2 uses unified in_proj_weight. Direct copy-paste of ESM-2 wrapper raises AttributeError. Prevention: Study transformers.models.bert.modeling_bert.BertSelfAttention structure, create DNABERT-specific wrapper, test packed attention equivalence (cosine >0.999).

6. **repr_layers hardcoded [36] causes invalid layer access** - 650M has 33 layers (valid 0-32), accessing layer 36 raises IndexError or returns None. Prevention: Parameterize repr_layers based on model.num_layers, add validation (assert max(repr_layers) < model.num_layers), centralize in model config.

7. **Vectorized operations break on edge cases** - Empty sequences (cu_seqlens[i] == cu_seqlens[i+1]) cause torch.split ValueError, boundary position IDs have off-by-one errors with incorrect cu_seqlens indexing. Prevention: Filter empty sequences in collator (min_length > 0), add explicit zero-length handling, property-based testing validating vectorized matches loop.

## Implications for Roadmap

Based on research, suggested phase structure prioritizes low-risk foundations before high-risk DNABERT-S port:

### Phase 1: Code Quality Foundations
**Rationale:** Independent refactoring establishes testing patterns and code organization before risky changes. Environment variable centralization enables better feature flagging for later phases. Function extraction reduces cognitive load for subsequent modifications.

**Delivers:**
- core/env_config.py with cached, documented env var access
- Refactored async_inference.run() into <100 line methods
- Refactored gpu_worker into component functions
- Deque replacement for O(1) queue operations

**Addresses:** Environment variable centralization (must-have), code quality improvements (supporting infrastructure)

**Avoids:** Import cycle pitfalls (minimal config module, lazy evaluation), refactoring regressions (1:1 behavior equivalence tests, keep _legacy() for A/B validation)

**Estimated plans:** 3-5 plans

---

### Phase 2: ESM-2 Model Flexibility
**Rationale:** Low-risk configuration change with clear testing (load 650M/3B/15B, verify repr_layers). Required foundation for Phase 4 (checkpoint compatibility) and enables 16GB GPU support.

**Delivers:**
- ESM2Config dataclass with auto-detected repr_layers
- load_esm2_model returns (model, batch_converter, config)
- CLI --esm-model flag (choices: 650M, 3B, 15B)
- Integration tests with multiple model variants

**Uses:** No new stack elements (fair-esm 2.0.0 already supports all variants)

**Implements:** Model configuration system (auto-detection based on model name t33/t36/t48)

**Avoids:** repr_layers hardcoded [36] invalid access (parameterized based on num_layers), backward compatibility (defaults to current esm2_t36_3B_UR50D)

**Estimated plans:** 3-4 plans

---

### Phase 3: Vectorized Operations
**Rationale:** Pure optimization with existing test coverage validates correctness. Independent from other features, provides incremental performance gains without architectural changes.

**Delivers:**
- Vectorized create_position_ids_packed (torch.cumsum + boundary masking)
- Optimized embedding extraction (cu_seqlens_cpu caching removes .item() syncs)
- Hybrid implementation (loop for small batches, vectorized for large)
- Comprehensive edge case tests (empty sequences, single-token, boundaries)

**Uses:** PyTorch >=2.8.0 native operations (torch.cumsum, scatter_reduce) - no new dependencies

**Implements:** Vectorized operation layer replacing Python loops in hot paths

**Avoids:** Empty sequence failures (filter min_length > 0 in collator), boundary off-by-one errors (property-based testing), performance regression on small batches (hybrid threshold approach)

**Estimated plans:** 2-3 plans

---

### Phase 4: Checkpoint Compatibility for Model Swaps
**Rationale:** Depends on Phase 2 (model flexibility). Validates metadata storage before torch.compile phase adds compilation cache complexity.

**Delivers:**
- Checkpoint header with model metadata (model_name, embedding_dim, num_layers)
- Resume validation comparing checkpoint vs current model config
- Embedding dimension migration tool for model swaps
- Integration test: checkpoint with 3B, resume with 650M → clear error

**Uses:** Existing checkpoint infrastructure (AsyncCheckpointWriter, CheckpointTrigger)

**Implements:** Model configuration system (checkpoint metadata validation)

**Avoids:** Embedding dimension mismatch (1280 vs 2560 shapes fail merge), orphaned shards (validate model consistency), silent corruption (explicit validation with actionable errors)

**Estimated plans:** 2-3 plans

---

### Phase 5: torch.compile Integration (Experimental)
**Rationale:** Depends on Phase 2 (test with multiple models) and Phase 4 (checkpoint compatibility validated). Isolated feature with VIRNUCPRO_DISABLE_COMPILE rollback. Default OFF for initial release.

**Delivers:**
- torch.compile wrapper in load_esm2_model (mode='reduce-overhead', dynamic=True)
- CLI --torch-compile flag (default: disabled)
- Warmup phase with representative length distribution
- Recompilation monitoring and cache size tracking
- Integration test verifying FlashAttention remains active

**Uses:** PyTorch >=2.8.0 torch.compile, exact version pinning (torch 2.10.0, triton 3.6.0, flash-attn 2.8.2)

**Implements:** torch.compile integration layer wrapping model after FP16 conversion

**Avoids:** Recompilation storm (dynamic=True, warmup phase, monitor cache), FlashAttention binary incompatibility (pin versions, test before rollout), slowdown if overhead >10% (feature flag OFF by default)

**Estimated plans:** 3-4 plans

---

### Phase 6: DNABERT-S v2.0 Port (HIGH RISK)
**Rationale:** Highest-risk feature requiring separate validation phase. Depends on all infrastructure (env config, model flexibility, vectorization, checkpointing). Tackles primary value proposition (fix 0.96x scaling) after all supporting changes validated.

**Sub-phases:**
- **6.1 Validation prep** - Extract v1.0 reference embeddings (100 sequences), document expected shapes
- **6.2 Tokenization** - Update VarlenCollator with tokenizer type detection, test unpacked equivalence
- **6.3 FlashAttention wrapper** - Implement DNABERTWithFlashAttention.forward_packed(), test packed==unpacked (cosine >0.999)
- **6.4 Integration** - Wire DNABERT-S into gpu_worker model_type='dnabert', end-to-end test (1K sequences)
- **6.5 Validation** - Correctness test v2.0 vs v1.0 (cosine >0.99), performance test 1.8x+ scaling with 2 GPUs

**Delivers:**
- DnabertVarlenCollator with transformers tokenizer
- DNABERTWithFlashAttention with packed attention
- Multi-GPU DNABERT-S pipeline matching ESM-2 scaling
- v1.0 fallback flag (--dnabert-v1-fallback) for rollback

**Addresses:** DNABERT-S v2.0 port (must-have), primary bottleneck (0.96x → 1.8x+ scaling)

**Avoids:** K-mer tokenization incompatibility (separate collator, verify padding IDs), FlashAttention wrapper differences (BERT Q/K/V extraction pattern), packing efficiency drop (profile length distribution, tune buffer_size), silent v1.0 regression (keep parallel_dnabert.py intact)

**Estimated plans:** 6-9 plans

---

### Phase 7: Quick Wins and Final Integration
**Rationale:** Low-hanging fruit after major features validated. Combined .to() calls, final benchmarking with all features enabled.

**Delivers:**
- Combined tensor transfers (single .to(device=device, dtype=torch.float16))
- Full pipeline benchmark on 1M subset (compare v2.0 baseline)
- Multi-GPU scaling validation (2x RTX 4090)
- Performance regression suite

**Addresses:** Quick wins (combined ops), integration validation

**Estimated plans:** 1-2 plans

---

### Phase Ordering Rationale

**Dependency chain:**
1. Code Quality (Phase 1) → enables better feature flagging for all later phases
2. Model Flexibility (Phase 2) → required for Checkpoint Compatibility (Phase 4) and torch.compile testing (Phase 5)
3. Vectorization (Phase 3) → independent, can run parallel to Phase 2
4. Checkpoint Compatibility (Phase 4) → depends on Phase 2, required before Phase 6 (DNABERT-S needs checkpoint support)
5. torch.compile (Phase 5) → depends on Phase 2 (test multiple models), validates before DNABERT-S
6. DNABERT-S (Phase 6) → depends on all infrastructure (Phases 1-5), highest risk saved for last
7. Quick Wins (Phase 7) → final polish after major features validated

**Risk mitigation:** HIGH-risk DNABERT-S port deferred until all LOW/MEDIUM risk features validated. Each phase has rollback mechanism (env vars, feature flags, v1.0 fallback).

**Parallel tracks:** Phases 1-3 can proceed in parallel (independent). Phases 4-6 must be sequential (dependencies).

### Research Flags

**Phases needing deeper research during planning:**
- **Phase 6 (DNABERT-S v2.0):** Complex integration requiring custom FlashAttention wrapper, MosaicBERT architecture less documented than ESM-2, k-mer tokenization edge cases need validation
- **Phase 5 (torch.compile):** Dynamic shape interaction with FlashAttention varlen unknown, may require experimentation to determine optimal compilation scope (full model vs layers only)

**Phases with standard patterns (skip research-phase):**
- **Phase 1 (Code Quality):** Standard refactoring patterns, well-established best practices for env var centralization and function extraction
- **Phase 2 (Model Flexibility):** Configuration pattern established in fair-esm API, straightforward dataclass implementation
- **Phase 3 (Vectorized Ops):** PyTorch vectorization well-documented, existing tests validate correctness
- **Phase 4 (Checkpoint Compatibility):** Extends existing checkpoint system with metadata validation, standard pattern
- **Phase 7 (Quick Wins):** Trivial optimizations, no research needed

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | Verified with PyTorch 2.10 docs, fair-esm GitHub, HuggingFace model cards. All features achievable with existing dependencies. |
| Features | HIGH | Production codebase analysis (PROJECT.md, STATE.md, OPTIMIZATION_REVIEW.md) validates feature landscape. Must-have vs nice-to-have clear from v2.0 experience. |
| Architecture | HIGH | Production code inspection (async_inference.py, gpu_worker.py, parallel_dnabert.py) reveals 90% component reuse between ESM-2 and DNABERT-S. Integration points well-defined. |
| Pitfalls | HIGH | Critical pitfalls verified from official docs (PyTorch dynamic shapes, NeMo ESM-2 dimensions, GitHub FlashAttention compatibility issues). Moderate/minor pitfalls are hypotheses requiring validation. |

**Overall confidence:** HIGH

### Gaps to Address

**DNABERT-S packing efficiency:** Hypothesis that DNA sequence length distribution differs from protein, potentially lowering packing efficiency below 85% threshold. Needs profiling on production data before tuning buffer_size and token_budget parameters. If efficiency drops significantly, may require separate packing configuration for DNABERT-S vs ESM-2.

**torch.compile recompilation overhead:** Unknown how often recompilation triggers with production sequence length distribution. Research shows dynamic shapes "can be quite high overhead" but actual frequency needs measurement. Warmup phase may need tuning based on observed recompilation patterns. If recompilation >10% overhead, disable compile for packed path.

**FlashAttention + torch.compile interaction:** Unclear whether torch.compile optimizes surrounding operations while treating FlashAttention as opaque, or fails to compile at all. Integration testing required to validate (1) FlashAttention remains active, (2) speedup achieved from non-attention operations, (3) no binary compatibility errors with pinned versions.

**Vectorization threshold:** Need profiling to determine batch size threshold where vectorized operations outperform Python loops. Hypothesis: vectorized faster when batch_size >100 sequences, but actual threshold may differ. Hybrid implementation (loop for small, vectorized for large) requires tuning based on measurements.

**Empty sequence handling:** Edge case where tokenization produces zero tokens (sequence shorter than k-mer size, exceeds max_length). Need validation that VarlenCollator filters these correctly before packing. If empty sequences reach extraction stage, vectorized operations will fail with shape errors.

## Sources

### Primary (HIGH confidence)

**PyTorch and torch.compile:**
- [torch.compile — PyTorch 2.10 documentation](https://docs.pytorch.org/docs/stable/generated/torch.compile.html)
- [Dynamic Shapes — PyTorch 2.10 documentation](https://docs.pytorch.org/docs/stable/torch.compiler_dynamic_shapes.html)
- [torch.Tensor.scatter_reduce_ — PyTorch 2.10 documentation](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.scatter_reduce_.html)
- [torch.cumsum — PyTorch 2.10 documentation](https://docs.pytorch.org/docs/stable/generated/torch.cumsum.html)
- [PyTorch 2.1: automatic dynamic shape compilation](https://pytorch.org/blog/pytorch-2-1/)
- [Introduction to torch.compile — PyTorch Tutorials](https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html)

**ESM-2 models:**
- [ESM-2 - BioNeMo Framework 2.0](https://docs.nvidia.com/bionemo-framework/2.0/models/esm2/) - embedding dimensions (650M=1280, 3B=2560)
- [GitHub - facebookresearch/esm](https://github.com/facebookresearch/esm)
- [fair-esm · PyPI](https://pypi.org/project/fair-esm/)
- [facebook/esm2_t33_650M_UR50D · Hugging Face](https://huggingface.co/facebook/esm2_t33_650M_UR50D)

**DNABERT-S:**
- [zhihan1996/DNABERT-S · Hugging Face](https://huggingface.co/zhihan1996/DNABERT-S)
- [GitHub - MAGICS-LAB/DNABERT_S](https://github.com/MAGICS-LAB/DNABERT_S)
- [DNABERT-2: EFFICIENT FOUNDATION MODEL | ICLR 2024](https://openreview.net/pdf?id=oMLQB4EZE1)

**FlashAttention:**
- [GitHub - Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)
- [Definitive Guide to PyTorch, CUDA, Flash Attention Compatibility | Medium](https://medium.com/@vici0549/the-definitive-guide-to-pytorch-cuda-and-flash-attention-compatibility-ebec1161ec10)
- [Binary compatibility issue with flash_attn-2.8.2 | GitHub Issue #1783](https://github.com/Dao-AILab/flash-attention/issues/1783)

**Production codebase analysis:**
- `.planning/PROJECT.md` - v2.0 architecture documentation
- `.planning/STATE.md` - accumulated decisions and context
- `OPTIMIZATION_REVIEW.md` - v2.5 feature catalog
- `virnucpro/pipeline/async_inference.py` (976 lines) - core v2.0 inference loop
- `virnucpro/pipeline/gpu_worker.py` (462 lines) - per-GPU worker implementation
- `virnucpro/pipeline/parallel_dnabert.py` (608 lines) - v1.0 DNABERT-S architecture
- `virnucpro/data/collators.py` (367 lines) - VarlenCollator tokenization + packing
- `virnucpro/models/esm2_flash.py` (668 lines) - ESM-2 FlashAttention integration
- `virnucpro/models/dnabert_flash.py` (438 lines) - DNABERT-S attention patching

### Secondary (MEDIUM confidence)

**torch.compile best practices:**
- [Everything You Need to Know About PyTorch Compile | Medium](https://medium.com/@lambdafluxofficial/everything-you-need-to-know-about-pytorch-compile-3d7fd94ce701)
- [HuggingFace: Optimize inference using torch.compile()](https://huggingface.co/docs/transformers/main/perf_torch_compile)
- [Introduction to torch.compile and How It Works with vLLM | vLLM Blog](https://blog.vllm.ai/2025/08/20/torch-compile.html)
- [State of torch.compile for training (August 2025) | ezyang's blog](https://blog.ezyang.com/2025/08/state-of-torch-compile-august-2025/)

**Vectorization:**
- [Vectorization and Broadcasting with PyTorch | Paperspace](https://blog.paperspace.com/pytorch-vectorization-and-broadcasting/)
- [What happens when you vectorize wide PyTorch expressions? | Marcus Lewis](https://probablymarcus.com/blocks/2023/10/19/vectorizing-wide-pytorch-expressions.html)
- [UX Limitations — PyTorch 2.9 documentation](https://docs.pytorch.org/docs/stable/func.ux_limitations.html)

**Configuration patterns:**
- [Leveraging Environment Variables in Python Programming - Configu](https://configu.com/blog/working-with-python-environment-variables-and-5-best-practices-you-should-know/)
- [Best Practices for Python Env Variables | Dagster](https://dagster.io/blog/python-environment-variables)

**Sequence packing:**
- [Sequence Packing and Dynamic Batching — NeMo-RL](https://docs.nvidia.com/nemo/rl/latest/design-docs/sequence-packing-and-dynamic-batching.html)
- [Dynamic Batching vs. Sequence Packing | Medium](https://medium.com/better-ml/dynamic-batching-vs-sequence-packing-0ef4a3894dad)
- [Efficient LLM Pretraining: Packed Sequences and Masked Attention | HuggingFace](https://huggingface.co/blog/sirluk/llm-sequence-packing)

### Tertiary (LOW confidence - needs validation)

**Refactoring risks:**
- [AI-Driven Refactoring: Identifying and Correcting Data Clumps | MDPI](https://www.mdpi.com/2079-9292/13/9/1644) - general refactoring pitfalls, not specific to this codebase
- [Medium-sized protein language models perform well at transfer learning | Nature](https://www.nature.com/articles/s41598-025-05674-x) - ESM-2 650M vs 3B quality comparisons

**Packing efficiency hypothesis:**
- [DNABERT: pre-trained Bidirectional Encoder Representations | Bioinformatics](https://academic.oup.com/bioinformatics/article/37/15/2112/6128680) - k-mer tokenization details, no packing efficiency data

---

*Research completed: 2026-02-09*
*Ready for roadmap: yes*
