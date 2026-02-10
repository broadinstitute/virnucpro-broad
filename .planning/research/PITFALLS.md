# Pitfalls Research: v2.5 Model Optimizations Round 2

**Domain:** Adding performance optimizations to existing GPU inference pipeline
**Researched:** 2026-02-09
**Confidence:** HIGH

## Executive Summary

This research identifies critical pitfalls when porting DNABERT-S to v2.0 async architecture, making ESM-2 model selection configurable, integrating torch.compile, and vectorizing operations in a production GPU pipeline. The most severe risks involve tokenization mismatches between DNABERT-S k-mer and ESM-2 protein tokenization, checkpoint dimension incompatibility when swapping models, and torch.compile recompilation storms with dynamic shapes.

**Critical risk:** DNABERT-S uses transformers k-mer tokenization (different vocabulary, padding behavior) vs ESM-2 batch_converter - requires custom collator implementation, cannot reuse VarlenCollator directly.

**Integration risk:** Changing ESM-2 model size (650M vs 3B) produces different embedding dimensions (1280 vs 2560) which breaks downstream checkpoint compatibility without migration.

## Critical Pitfalls

### DNABERT-S v2.0 Port

**Pitfall 1: K-mer Tokenization Incompatibility with ESM batch_converter**

**What goes wrong:** DNABERT-S uses Hugging Face transformers tokenizer with k-mer vocabulary (e.g., 3-mer: ATG, TGG, GGC) while ESM-2 uses `alphabet.get_batch_converter()` with amino acid vocabulary. VarlenCollator hardcodes `self.batch_converter` expecting ESM's interface (returns labels, strs, tokens tuple). Attempting to reuse VarlenCollator for DNABERT-S will fail during tokenization with incompatible return format or vocabulary mismatch.

**Why it happens:** Different model architectures use different tokenization APIs. DNABERT-S from transformers library returns `BatchEncoding` objects with `input_ids`, `attention_mask` keys, not ESM's 3-tuple format.

**Consequences:**
- Runtime TypeError when VarlenCollator calls `self.batch_converter(sequences)` expecting ESM format
- Incorrect token IDs if vocabularies accidentally overlap but mean different things
- Padding token ID mismatch (ESM padding_idx=1, transformers may use different ID)

**Prevention:**
1. Create separate `DnabertVarlenCollator` class inheriting structure but using transformers tokenizer
2. Verify padding token ID matches between tokenizer and model expectations
3. Test tokenization output format in unit tests before integration
4. Check for special token handling differences (CLS, SEP, MASK vs BOS, EOS)

**Detection:**
- Unit test fails when collator receives DNABERT sequences
- Shape mismatch errors in packed attention (wrong vocab size)
- ValueError about unexpected token IDs during model forward pass
- Attention mask format errors (2D vs 1D expectations)

**Severity:** CRITICAL
**Phase to address:** Phase 1 (DNABERT-S Collator Implementation)
**Confidence:** HIGH (verified from VarlenCollator code and transformers vs ESM API differences)

---

**Pitfall 2: DNABERT-S Packing Efficiency Lower Than ESM-2**

**What goes wrong:** DNA sequences (nucleotides) are shorter and more uniform in length than protein sequences (amino acids) after translation. DNABERT-S processes nucleotide sequences directly, which can be 3x longer than their translated protein equivalents (codon → amino acid mapping). However, k-mer tokenization with overlap creates token sequences where length ≈ `sequence_length - k + 1` (for k=3: 1000bp → ~998 tokens). This different length distribution may cause GreedyPacker FFD to achieve lower efficiency than ESM-2's protein sequences.

**Why it happens:** FFD packing efficiency depends on length distribution variance. ESM-2 processes translated protein sequences (variable lengths from six-frame translation). DNABERT-S processes raw nucleotides with k-mer overlap, creating different length clustering patterns.

**Consequences:**
- Buffer size tuned for ESM-2 (2000 sequences) may be suboptimal for DNABERT-S
- Packing efficiency drops below 85% warning threshold
- More batches generated = increased H2D transfer overhead
- GPU underutilization from smaller effective batch sizes

**Prevention:**
1. Profile DNABERT-S length distribution on production data before tuning
2. Parameterize buffer_size in create_async_dataloader for DNABERT-S
3. Consider separate token_budget for DNABERT-S vs ESM-2 (different model memory footprint)
4. Add packing efficiency monitoring specific to DNABERT-S runs

**Detection:**
- Packing efficiency logs showing <85% for DNABERT-S (06-06 two-tier thresholds)
- Significantly more batches per GPU than ESM-2 on same sequence count
- Lower throughput (sequences/sec) than expected from ESM-2 benchmarks
- GPU utilization drops during DNABERT-S processing

**Severity:** CRITICAL
**Phase to address:** Phase 1 (DNABERT-S port), Phase 4 (parameter tuning)
**Confidence:** MEDIUM (hypothesis based on length distribution differences, needs profiling)

---

**Pitfall 3: FlashAttention Integration Differs from ESM-2**

**What goes wrong:** DNABERT-S uses standard BERT architecture from transformers (BertModel) with different attention implementation than ESM-2's custom architecture. ESM-2's FlashAttention integration in `esm2_flash.py` uses layer-level Q/K/V extraction from `in_proj_weight` and wraps with `flash_attn_varlen_wrapper`. DNABERT-S BertAttention stores Q/K/V as separate `nn.Linear` layers (query, key, value attributes), requiring different wrapper approach.

**Why it happens:** fair-esm and transformers have different internal attention implementations. Direct copy-paste of ESM-2's FlashAttention wrapper will fail AttributeError accessing non-existent weight tensors.

**Consequences:**
- AttributeError when trying to access `in_proj_weight` on DNABERT-S attention layers
- Fallback to standard attention silently (no speedup from FlashAttention)
- Incorrect Q/K/V extraction if weight shapes assumed incorrectly
- Packed attention fails if cu_seqlens not passed through correctly

**Prevention:**
1. Study `transformers.models.bert.modeling_bert.BertSelfAttention` structure before implementing
2. Create DNABERT-specific FlashAttention wrapper extracting from separate Q/K/V Linear layers
3. Test packed attention equivalence with same rigor as ESM-2 (06-05 cosine similarity tests)
4. Verify FlashAttention activation with integration test checking for fallback warnings

**Detection:**
- Missing FlashAttention in benchmark logs (no "Using FlashAttention-2" message)
- Performance not improving despite packed format (attention still O(n²) not O(n))
- Integration test fails comparing packed vs unpacked attention outputs
- dtype validation errors (FlashAttention requires FP16/BF16)

**Severity:** CRITICAL
**Phase to address:** Phase 2 (FlashAttention wrapper for DNABERT-S)
**Confidence:** HIGH (verified from esm2_flash.py implementation and transformers architecture)

---

### ESM-2 Model Flexibility

**Pitfall 4: Embedding Dimension Mismatch Breaks Checkpoints**

**What goes wrong:** ESM-2 650M produces embeddings with shape `[seq_len, 1280]` (33 layers, hidden_dim=1280). ESM-2 3B produces `[seq_len, 2560]` (36 layers, hidden_dim=2560). The existing checkpoint system stores embeddings in HDF5 with shape `[num_sequences, max_seq_len, hidden_dim]`. Switching between models without updating checkpoint format causes shape mismatches when loading/resuming, and downstream merge stage expects consistent dimensions.

**Why it happens:** Checkpoint format was hardcoded assuming single model size. Different ESM-2 variants have different architectures (not just parameter counts). Research shows embedding dimension differences: 650M=1280, 3B=2560 (NeMo docs, 2026).

**Consequences:**
- RuntimeError when loading checkpoint with wrong embedding dimensions
- ShapeError in merge stage concatenating embeddings from different model sizes
- Downstream prediction model fails (trained on 2560-dim, receives 1280-dim)
- Silent corruption if dimensions accidentally match but semantics differ

**Prevention:**
1. Store model metadata in checkpoint header (model_name, embedding_dim, num_layers)
2. Validate checkpoint compatibility before resume (compare metadata)
3. Implement embedding dimension migration tool for model swaps
4. Add integration test mixing 650M/3B embeddings to catch incompatibility
5. Document model compatibility matrix in migration guide

**Detection:**
- ValueError during checkpoint loading about shape mismatch
- Merge stage error concatenating incompatible shapes
- Prediction accuracy drops to random (wrong dimensionality to classifier)
- Checkpoint resume fails with "corrupted checkpoint" when actually dimension mismatch

**Severity:** CRITICAL
**Phase to address:** Phase 3 (model selection implementation), Phase 5 (checkpoint migration)
**Confidence:** HIGH (verified from BioNeMo docs showing 650M=1280, 3B=2560 dimensions)

---

**Pitfall 5: repr_layers Hardcoded to [36] Causes Invalid Layer Access**

**What goes wrong:** Code currently hardcoded `repr_layers=[36]` in 6 locations (grep results show PROJECT.md, models, pipeline). ESM-2 650M has 33 layers (valid indices 0-32). Accessing layer 36 on 650M model raises IndexError or returns None/empty tensor. ESM-2 3B has 36 layers (valid 0-35), so layer 36 access succeeds. Swapping to 650M without updating repr_layers causes runtime errors or silent failures.

**Why it happens:** v2.0 development focused exclusively on ESM-2 3B. repr_layers validation not enforced by ESM model (may silently ignore invalid indices or raise uncaught exception).

**Consequences:**
- IndexError or KeyError when extracting embeddings from layer 36 on 650M
- Empty embeddings if ESM silently ignores invalid layer index
- Incorrect embeddings if ESM clamps to valid range (layer 32 instead of 36)
- Integration tests pass on 3B, fail mysteriously on 650M

**Prevention:**
1. Parameterize repr_layers based on model size: `model.num_layers - 1` for final layer
2. Add validation in model loading: `assert max(repr_layers) < model.num_layers`
3. Centralize repr_layers logic in model configuration
4. Add unit test for invalid repr_layers access
5. Document layer count mapping: {650M: 33, 3B: 36, custom: detect from model}

**Detection:**
- IndexError during forward pass accessing representations dict
- Empty embedding output (num_tokens=0 in packed format)
- Warnings about invalid layer indices (if ESM provides them)
- Integration test comparing 650M vs 3B embeddings catches difference

**Severity:** CRITICAL
**Phase to address:** Phase 3 (model selection implementation)
**Confidence:** HIGH (verified from NeMo docs and grep showing hardcoded [36])

---

### torch.compile Integration

**Pitfall 6: Dynamic Shapes Cause Recompilation Storm**

**What goes wrong:** Variable-length DNA sequences produce batches with different `max_seqlen` and `cu_seqlens` lengths on every batch. torch.compile with default settings triggers recompilation whenever input shapes change. With 1000+ unique sequence length combinations in production data, first epoch compiles 1000+ kernels, each taking 30-60 seconds (triton codegen overhead). Total compilation time exceeds inference time, negating speedup.

**Why it happens:** PyTorch 2.1+ automatic dynamic shapes detection requires repeated recompilations to learn shape patterns. FlashAttention varlen interface uses dynamic `cu_seqlens` tensor lengths (N+1 for N sequences, where N varies per batch). Research shows recompilation overhead "can be quite high for workloads with dynamic shapes" (vLLM blog, 2025).

**Consequences:**
- First run takes 10x longer than uncompiled (compilation overhead)
- Subsequent runs still slow if length distribution not seen during warmup
- Memory bloat from cached compiled kernels (100+ GB cache directory)
- Spot instance preemption during compilation wastes progress (no checkpoint)

**Prevention:**
1. Use `torch.compile(dynamic=True)` to enable symbolic shape tracking
2. Add warmup phase with representative length distribution before benchmarking
3. Disable compilation for variable-length stages, compile only fixed-shape stages
4. Consider ahead-of-time compilation for common length buckets
5. Monitor compilation cache size and clear periodically
6. Add `VIRNUCPRO_DISABLE_COMPILE` env var for emergency rollback

**Detection:**
- Log messages: "Recompiling function X due to shape change"
- First batch takes 60+ seconds vs <1s subsequent batches
- `~/.triton/cache` directory grows to 10+ GB
- nvidia-smi shows GPU idle during compilation (CPU pegged)
- Throughput significantly slower than uncompiled baseline

**Severity:** CRITICAL
**Phase to address:** Phase 6 (torch.compile integration)
**Confidence:** HIGH (verified from PyTorch docs and vLLM blog on dynamic shapes)

---

**Pitfall 7: torch.compile + FlashAttention Custom CUDA Kernels Incompatibility**

**What goes wrong:** FlashAttention uses custom CUDA kernels (flash_attn_varlen_func) that may not be compatible with torch.compile's graph tracing. torch.compile uses TorchDynamo to capture computation graph and Triton to generate kernels. When encountering pre-compiled CUDA kernels like FlashAttention, it may: (1) fail to trace through, causing compilation error, (2) treat as opaque operation and not optimize, (3) trigger fallback to eager mode silently.

**Why it happens:** Custom CUDA extensions bypass PyTorch's autograd and aren't visible to TorchDynamo's Python frame evaluation. Research shows "binary compatibility issue with flash_attn-2.8.2 wheel" and version-specific problems (GitHub issues, 2026). torch.compile requires precise version matching: PyTorch 2.10.0 → CUDA 12.6.3/12.8.1 → Triton 3.6.0 → flash-attn 2.8.2+.

**Consequences:**
- CompilationError when torch.compile tries to trace through FlashAttention
- Silent fallback to standard attention (no speedup, wrong path tested)
- Version conflicts between torch, triton, and flash-attn wheels
- Import errors: "undefined symbol: _ZN2at..." (binary incompatibility)
- Different behavior in compiled vs eager mode (wrong code path validated)

**Prevention:**
1. Test torch.compile on single GPU before multi-GPU rollout
2. Pin exact versions: torch 2.10.0, triton 3.6.0, flash-attn 2.8.2
3. Use `torch.compiler.disable()` context manager around FlashAttention calls if needed
4. Add integration test verifying FlashAttention active under torch.compile
5. Implement version compatibility check at startup (log warnings for mismatches)
6. Document known-good version combinations in README

**Detection:**
- ImportError during flash-attn import (undefined symbols)
- Performance regression (compiled slower than eager)
- Log warnings: "Skipping optimization due to unsupported operation"
- Integration test comparing eager vs compiled FlashAttention outputs diverges
- Binary compatibility errors at runtime despite installation succeeding

**Severity:** CRITICAL
**Phase to address:** Phase 6 (torch.compile integration)
**Confidence:** HIGH (verified from Medium guide, GitHub issues showing version conflicts)

---

## Moderate Pitfalls

### Vectorized Operations

**Pitfall 8: Vectorized Position ID Generation Breaks at Boundaries**

**What goes wrong:** Replacing Python loop in `create_position_ids_packed()` with vectorized torch operations requires careful handling of cumulative boundary resets. Current implementation (packed_attention.py) uses loop to insert zeros at each sequence boundary. Vectorized approach using scatter/masked operations may produce off-by-one errors if boundary indices calculated incorrectly, causing position IDs to not reset at cu_seqlens boundaries.

**Why it happens:** cu_seqlens has N+1 elements for N sequences. Boundary indices are at `cu_seqlens[1:]` (excluding first 0). Vectorized operations require correct indexing accounting for this offset. PyTorch vmap and vectorized ops have edge case issues with empty tensors and boundary conditions (PyTorch docs UX limitations).

**Consequences:**
- Position IDs don't reset at sequence boundaries (wrong positional embeddings)
- Off-by-one errors cause position overflow (position > max_position_embeddings)
- Empty sequence edge case (single sequence with len=1) produces wrong shape
- Packed attention output silently incorrect (no error, just wrong values)

**Prevention:**
1. Keep vectorized implementation for common case (>1 token per sequence)
2. Add explicit edge case handling for single-token sequences
3. Comprehensive unit tests covering: empty batch, single sequence, boundary positions
4. Validate output matches loop-based version with property-based testing (hypothesis)
5. Add assertion checking position IDs reset at boundaries (max position < max_pos_embed)

**Detection:**
- Unit test fails on edge cases (empty, single-token, single-sequence)
- Assertion error: position IDs exceed maximum (>1022 for ESM-2)
- Packed attention output diverges from expected (cosine similarity <0.99)
- Boundary sequences have wrong embeddings (position info corrupted)

**Severity:** MODERATE
**Phase to address:** Phase 7 (vectorization)
**Confidence:** MEDIUM (hypothesis based on PyTorch vmap limitations, needs validation)

---

**Pitfall 9: Embedding Extraction Vectorization Fails on Empty Sequences**

**What goes wrong:** Current embedding extraction loops through cu_seqlens boundaries to slice packed embeddings. Vectorizing with advanced indexing (torch.index_select, torch.split) fails when a sequence has zero tokens (cu_seqlens[i] == cu_seqlens[i+1]). torch.split with size 0 in split list raises ValueError. torch.index_select with empty index tensor produces unexpected shapes.

**Why it happens:** Empty sequences can occur from: tokenization producing only padding (stripped in VarlenCollator), sequences shorter than k-mer size in DNABERT-S, or sequences exceeding max_length and truncated to zero. PyTorch vectorized operations don't gracefully handle zero-length splits.

**Consequences:**
- ValueError: "split expects at least a 1-dimensional tensor" during extraction
- Shape mismatch when stacking embeddings (empty vs non-empty tensors)
- Downstream HDF5 writing fails (can't write variable-length arrays without ragged support)
- Silent data loss if empty sequences skipped without tracking

**Prevention:**
1. Filter empty sequences before packing (validate min_length > 0 in collator)
2. Add zero-length check in vectorized extraction with explicit empty tensor handling
3. Maintain parallel implementation: vectorized for normal, loop for edge cases
4. Unit test specifically for empty sequence handling at extraction stage
5. Log warning when empty sequences detected (indicates upstream issue)

**Detection:**
- ValueError during embedding extraction in AsyncInferenceRunner
- Shape assertion fails: embeddings list contains tensors with incompatible shapes
- HDF5 dataset creation fails with size mismatch
- Sequence count mismatch: input N sequences, output N-k embeddings (k empty)

**Severity:** MODERATE
**Phase to address:** Phase 7 (vectorization)
**Confidence:** MEDIUM (hypothesis based on edge case testing requirements)

---

### Code Quality Refactoring

**Pitfall 10: Environment Variable Centralization Breaks Import Cycles**

**What goes wrong:** Moving environment variable reads from inline `os.getenv()` calls throughout codebase to centralized config module creates import dependency. If config module imports from modules that check env vars, circular import occurs. Example: `config.py` imports `utils.precision` to validate FP16 settings, but `precision.py` imports `config.get()` to check `VIRNUCPRO_DISABLE_FP16` → circular dependency.

**Why it happens:** Centralization best practice collides with existing import graph structure. Research recommends "centralize environment variable access in single configuration module" (Configu blog, 2026), but retrofitting to existing codebase requires careful import ordering.

**Consequences:**
- ImportError: "cannot import name 'get' from partially initialized module 'config'"
- Module attributes undefined during import (AttributeError accessing config vars)
- Tests fail to mock env vars if config imports happen before test setup
- Startup failures in production that don't reproduce locally (import order differences)

**Prevention:**
1. Create minimal config module with zero imports (only os, typing)
2. Defer validation to runtime (lazy evaluation) not import time
3. Use `TYPE_CHECKING` for type-only imports to break cycles
4. Refactor in stages: centralize reads first, defer validation refactor separately
5. Add import cycle detection test in CI (using importlab or similar)

**Detection:**
- ImportError during pytest collection phase
- ModuleNotFoundError in worker processes (different import order than main)
- AttributeError accessing config attributes that "should" be defined
- Tests pass individually but fail when run together (import order dependent)

**Severity:** MODERATE
**Phase to address:** Phase 8 (code quality cleanup)
**Confidence:** MEDIUM (common refactoring issue, best practices from Configu blog)

---

**Pitfall 11: Refactoring Large Functions Introduces Regressions**

**What goes wrong:** Splitting 200+ line functions (like `process_dnabert_files_worker`) into smaller components risks introducing behavioral changes. Edge cases handled implicitly in original monolithic code (error handling order, state management, resource cleanup) get lost during extraction. Original function may have relied on local variable sequencing or exception propagation that breaks when split across functions.

**Why it happens:** Large functions accumulate implicit dependencies over time. Research shows "risk of generating faulty refactorings" with AI-assisted code changes, and "adopting human-in-the-loop approach can help mitigate risks" (MDPI AI refactoring study, 2024). Production code has subtle correctness properties not captured in docstrings.

**Consequences:**
- Regression bugs that don't appear until production workload (timing/concurrency)
- Resource leaks if cleanup code moved to wrong scope (file handles not closed)
- Changed error handling order masks real exceptions with cleanup exceptions
- Performance regression from extra function call overhead in hot path
- Tests pass but behavior subtly different (missing edge case coverage)

**Prevention:**
1. Extract pure functions first (no side effects, fully tested)
2. Maintain 1:1 behavior equivalence test (input→output unchanged)
3. Refactor in small PRs with extensive before/after testing
4. Use characterization tests capturing current behavior before changes
5. Keep original function as `_legacy()` for A/B validation
6. Add integration test comparing refactored vs original on production data sample

**Detection:**
- Integration test divergence (refactored output != original output)
- Production errors that don't reproduce in test (missing edge case)
- Memory usage increase (resource leak from cleanup order change)
- Timing regression (10% slower due to function call overhead)
- Error logs showing different exception types than before

**Severity:** MODERATE
**Phase to address:** Phase 8 (code quality cleanup)
**Confidence:** MEDIUM (general refactoring risk, MDPI paper on AI refactoring pitfalls)

---

## Minor Pitfalls

### Quick Wins

**Pitfall 12: Env Var Caching Without Invalidation**

**What goes wrong:** Caching `os.getenv()` results at module import time for performance (avoid repeated syscalls) prevents runtime configuration changes. Tests that mock environment variables after module import see cached stale values. Production systems can't toggle features via env vars without restart.

**Why it happens:** Performance optimization (caching) trades flexibility for speed. `os.getenv()` is relatively expensive (~1µs) but called in hot paths.

**Consequences:**
- Tests set `VIRNUCPRO_DISABLE_PACKING=true` but code sees cached False value
- Cannot toggle FP16 without process restart (defeats emergency rollback pattern)
- Debugging harder (can't change log level without restart)

**Prevention:**
1. Cache only truly static values (installation paths, version info)
2. Provide cache invalidation function for testing: `config.reload()`
3. Document which vars are cached vs dynamic in configuration guide
4. Use properties for dynamic lookups, constants for static

**Detection:**
- Test sets env var but code behavior doesn't change
- Log level changes ignored until restart
- Feature flags don't toggle as expected

**Severity:** MINOR
**Phase to address:** Phase 8 (env var centralization)
**Confidence:** HIGH (common caching pitfall)

---

**Pitfall 13: Deque Replace Breaks Compatibility**

**What goes wrong:** Replacing list-based queues with `collections.deque` for O(1) popleft operations changes iteration behavior and API surface. Deque doesn't support slicing (`queue[1:3]` raises TypeError), doesn't support `+` concatenation, and iteration order differs after rotate operations.

**Why it happens:** Deque and list are not drop-in replacements despite similar interface. Code may have implicit list-isms.

**Consequences:**
- TypeError when existing code tries to slice queue
- Changed behavior if code relies on index access beyond [0] and [-1]
- Subtle bugs if queue rotation used (iteration order changes)

**Prevention:**
1. Search codebase for queue slicing before replacement
2. Add unit tests for queue operations before refactoring
3. Replace only in proven hot paths (profiling shows list.pop(0) is bottleneck)
4. Keep list for small queues (<100 items) where performance doesn't matter

**Detection:**
- TypeError: 'deque' object does not support item assignment/slicing
- Behavioral change in queue processing order
- Tests fail after deque introduction

**Severity:** MINOR
**Phase to address:** Phase 8 (quick wins)
**Confidence:** HIGH (well-known API difference)

---

## Integration Pitfalls (Cross-Feature)

**Pitfall 14: DNABERT-S + torch.compile Multiplies Recompilation**

**What goes wrong:** DNABERT-S k-mer tokenization produces different length distributions than ESM-2 protein sequences. If torch.compile enabled for both models, compilation cache contains ESM-2 length patterns. When switching to DNABERT-S, entirely new set of compilations triggered for different shape space. Multi-model pipeline doubles compilation overhead.

**Why it happens:** torch.compile caches based on input shapes. Two models with non-overlapping length distributions create disjoint cache entries.

**Consequences:**
- First DNABERT-S run after ESM-2 recompiles everything (no cache hits)
- Cache directory doubles in size (ESM-2 kernels + DNABERT-S kernels)
- Warmup time increases proportionally to number of models
- Pipeline switching between models thrashes cache

**Prevention:**
1. Use separate cache directories per model (TORCH_COMPILE_CACHE_DIR env var)
2. Disable compilation for whichever model benefits less (profile to determine)
3. Share compilation across runs (persistent cache directory, not /tmp)
4. Consider compiling only attention layers, not full model

**Detection:**
- Cache directory >50GB (excessive compiled kernels)
- Recompilation messages when switching between DNABERT-S and ESM-2 stages
- First-run overhead doubles when both models compiled
- Memory pressure from large cache

**Severity:** MODERATE
**Phase to address:** Phase 6 (torch.compile integration)
**Confidence:** MEDIUM (hypothesis based on shape distribution differences)

---

**Pitfall 15: Model Selection + Checkpointing Creates Orphaned Shards**

**What goes wrong:** User runs partial dataset with ESM-2 3B, creates checkpoint shards with 2560-dim embeddings. User stops, switches to 650M for faster iteration, resumes from checkpoint. Checkpoint resume loads partial results with wrong dimensions, merge stage fails concatenating 2560-dim (old) + 1280-dim (new) embeddings.

**Why it happens:** Checkpoint system stores partial results. Model selection changes embedding dimension. No validation enforcing model consistency across checkpoint/resume.

**Consequences:**
- ShapeError during shard aggregation (incompatible dimensions)
- Corrupted output HDF5 with mixed dimensions
- User confusion about why resume fails ("I didn't change the checkpoint")
- Data loss if orphaned shards not detected and reprocessed

**Prevention:**
1. Store model identifier in checkpoint manifest metadata
2. Validate model match during resume (error if mismatch)
3. Provide migration tool: reprocess orphaned shards with new model
4. Clear documentation: "changing model requires --force-reprocess"
5. Add checkpoint compatibility check in CLI with actionable error message

**Detection:**
- ValueError during resume: "checkpoint model mismatch (expected 650M, found 3B)"
- Shard aggregation error: incompatible shapes
- Integration test: checkpoint with 3B, resume with 650M → should error clearly

**Severity:** MODERATE
**Phase to address:** Phase 5 (checkpoint compatibility validation)
**Confidence:** HIGH (direct consequence of combining model selection + checkpointing)

---

**Pitfall 16: Vectorization + Packing Efficiency Tradeoff**

**What goes wrong:** Vectorized position ID generation and embedding extraction may introduce overhead for small batches. Packing efficiency depends on buffer size (2000 sequences for 92-94%). Small buffers produce small batches where vectorization overhead exceeds loop savings. Tradeoff: larger buffers improve packing but delay first batch output.

**Why it happens:** Vectorized ops have fixed kernel launch overhead. PyTorch GPU kernel launch ~5-10µs. For batch with 10 sequences, loop might be faster than launching vectorized kernel. Research shows "vectorization introduces extra work through tensor permutations" (probablymarcus.com, 2023).

**Consequences:**
- Slower throughput on small batches (vectorized slower than loop)
- Increased latency if buffer size increased to amortize vectorization overhead
- Parameter coupling: buffer_size tuning affects both packing and vectorization efficiency
- Different optimal parameters for development (small data) vs production (large data)

**Prevention:**
1. Implement hybrid approach: loop for small batches (<threshold), vectorized for large
2. Profile threshold: "vectorized faster when batch_size > N sequences"
3. Make threshold configurable (VIRNUCPRO_VECTORIZATION_THRESHOLD)
4. Document tradeoff in tuning guide: buffer size affects both packing and vectorization
5. Add telemetry tracking vectorized vs loop code path usage

**Detection:**
- Throughput regression on small datasets after vectorization
- Profiling shows vectorized kernel launch overhead dominates for small batches
- A/B test: vectorized slower than original loop on <100 sequence workloads

**Severity:** MINOR
**Phase to address:** Phase 7 (vectorization tuning)
**Confidence:** LOW (need profiling to validate threshold exists)

---

## Prevention Checklist

Use this checklist when implementing v2.5 features to avoid documented pitfalls:

### DNABERT-S v2.0 Port
- [ ] Create separate DnabertVarlenCollator (don't reuse ESM collator)
- [ ] Verify padding token ID compatibility between tokenizer and model
- [ ] Unit test tokenization output format before integration
- [ ] Profile DNABERT-S sequence length distribution on production data
- [ ] Tune buffer_size separately from ESM-2 based on profiling
- [ ] Implement DNABERT-specific FlashAttention wrapper (separate Q/K/V Linear layers)
- [ ] Test packed attention equivalence (cosine similarity >0.99)
- [ ] Verify FlashAttention activation (no fallback warnings in logs)

### ESM-2 Model Selection
- [ ] Store model metadata in checkpoint header (model_name, embedding_dim, num_layers)
- [ ] Validate checkpoint compatibility before resume (compare metadata)
- [ ] Implement embedding dimension migration tool for model swaps
- [ ] Parameterize repr_layers based on model size: `model.num_layers - 1`
- [ ] Add validation: `assert max(repr_layers) < model.num_layers`
- [ ] Document layer count mapping: {650M: 33, 3B: 36}
- [ ] Integration test mixing 650M/3B embeddings to catch incompatibility
- [ ] Integration test: checkpoint with 3B, resume with 650M → clear error

### torch.compile Integration
- [ ] Use `torch.compile(dynamic=True)` for symbolic shape tracking
- [ ] Add warmup phase with representative length distribution
- [ ] Test torch.compile on single GPU before multi-GPU rollout
- [ ] Pin exact versions: torch 2.10.0, triton 3.6.0, flash-attn 2.8.2
- [ ] Integration test verifying FlashAttention active under torch.compile
- [ ] Add `VIRNUCPRO_DISABLE_COMPILE` env var for emergency rollback
- [ ] Monitor compilation cache size (alert if >20GB)
- [ ] Document known-good version combinations in README

### Vectorized Operations
- [ ] Keep loop-based implementation for edge cases (empty sequences, single-token)
- [ ] Comprehensive unit tests: empty batch, single sequence, boundary positions
- [ ] Property-based testing (hypothesis) validating vectorized matches loop
- [ ] Filter empty sequences before packing (validate min_length > 0)
- [ ] Add zero-length check in vectorized extraction with explicit handling
- [ ] Profile vectorization threshold (when vectorized > loop performance)
- [ ] Implement hybrid: loop for small batches, vectorized for large

### Code Quality Refactoring
- [ ] Create minimal config module with zero imports (only os, typing)
- [ ] Defer validation to runtime (lazy evaluation) not import time
- [ ] Add import cycle detection test in CI
- [ ] Extract pure functions first (no side effects, fully tested)
- [ ] Maintain 1:1 behavior equivalence test (input→output unchanged)
- [ ] Keep original function as `_legacy()` for A/B validation
- [ ] Cache only truly static env vars (provide reload() for testing)
- [ ] Search codebase for queue slicing before deque replacement

### Integration Testing
- [ ] Test DNABERT-S + ESM-2 pipeline end-to-end (both models in one run)
- [ ] Test checkpoint resume after model switch (should error clearly)
- [ ] Test torch.compile cache behavior with both models (measure size/time)
- [ ] Test vectorization on production data sample (validate no regressions)
- [ ] Benchmark comparison: v2.0 baseline vs v2.5 with each feature enabled incrementally

---

## Sources

### Tokenization and Sequence Packing
- [DNABERT: pre-trained Bidirectional Encoder Representations from Transformers model for DNA-language in genome | Bioinformatics](https://academic.oup.com/bioinformatics/article/37/15/2112/6128680)
- [Model Decides How to Tokenize: Adaptive DNA Sequence Tokenization with MxDNA](https://proceedings.neurips.cc/paper_files/paper/2024/file/79af547fa22cdcb0facd0b31dcd4bdb0-Paper-Conference.pdf)
- [DNABERT-2: Efficient Foundation Model and Benchmark For Multi-Species Genomes](https://arxiv.org/html/2306.15006v2)
- [Tokenizer, Dataset, and "collate_fn" - Yu.Z's Personal Site](https://yuzhu.run/tokenizer-location/)
- [Tokenizer stalls / hangs when used in DataLoader (multiprocessing issue) · Issue #258 · huggingface/tokenizers](https://github.com/huggingface/tokenizers/issues/258)
- [Sequence Packing and Dynamic Batching — NeMo-RL](https://docs.nvidia.com/nemo/rl/latest/design-docs/sequence-packing-and-dynamic-batching.html)
- [Dynamic Batching vs. Sequence Packing | by Jaideep Ray | Better ML](https://medium.com/better-ml/dynamic-batching-vs-sequence-packing-0ef4a3894dad)
- [Efficient LLM Pretraining: Packed Sequences and Masked Attention](https://huggingface.co/blog/sirluk/llm-sequence-packing)

### torch.compile and Dynamic Shapes
- [PyTorch 2.1: automatic dynamic shape compilation, distributed checkpointing](https://pytorch.org/blog/pytorch-2-1/)
- [Dynamic Shapes Core Concepts — PyTorch main documentation](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/compile/dynamic_shapes_core_concepts.html)
- [State of torch.compile for training (August 2025) : ezyang's blog](https://blog.ezyang.com/2025/08/state-of-torch-compile-august-2025/)
- [Introduction to torch.compile and How It Works with vLLM | vLLM Blog](https://blog.vllm.ai/2025/08/20/torch-compile.html)
- [Does torch.compile use FlashAttention？ - torch._inductor - PyTorch Forums](https://discuss.pytorch.org/t/does-torch-compile-use-flashattention/176599)

### FlashAttention and CUDA Compatibility
- [Definitive Guide to PyTorch, CUDA, Flash Attention, Xformers, Triton, and Bitsandbytes Compatibility | by vici0549 | Jan, 2026](https://medium.com/@vici0549/the-definitive-guide-to-pytorch-cuda-and-flash-attention-compatibility-ebec1161ec10)
- [GitHub - Dao-AILab/flash-attention: Fast and memory-efficient exact attention](https://github.com/Dao-AILab/flash-attention)
- [Binary compatibility issue with flash_attn-2.8.2 wheel for PyTorch 2.6.0+cu124 · Issue #1783 · Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention/issues/1783)
- [PyTorch 2.7 Release](https://pytorch.org/blog/pytorch-2-7/)

### ESM-2 Model Specifications
- [ESM-2 - BioNeMo Framework](https://docs.nvidia.com/bionemo-framework/2.1/models/esm2/)
- [Upgrading from ESM2-650M to ESM2-3B · Issue #1 · matsengrp/dasm-experiments](https://github.com/matsengrp/dasm-experiments/issues/1)
- [ESM-2 650M | BioLM](https://biolm.ai/models/esm2-650m/)
- [Medium-sized protein language models perform well at transfer learning on realistic datasets](https://www.nature.com/articles/s41598-025-05674-x)

### Checkpoint Compatibility
- [PyTorch-BigGraph I/O format](https://torchbiggraph.readthedocs.io/en/latest/input_output.html)
- [Size Mismatch error for LLM checkpoint of PEFT model with a resized token embeddings - Hugging Face Forums](https://discuss.huggingface.co/t/size-mismatch-error-for-llm-checkpoint-of-peft-model-with-a-resized-token-embeddings/104157)
- [Python TensorFlow Model Checkpoint in 2 Files: 2026 Guide](https://copyprogramming.com/howto/python-tensorflow-model-checkpoint-in-2-files)

### Vectorization and PyTorch Operations
- [What happens when you vectorize wide PyTorch expressions? | Marcus Lewis](https://probablymarcus.com/blocks/2023/10/19/vectorizing-wide-pytorch-expressions.html)
- [UX Limitations — PyTorch 2.9 documentation](https://docs.pytorch.org/docs/stable/func.ux_limitations.html)
- [Unleashing the Power of PyTorch vmap](https://www.codegenes.net/blog/pytorch-vmap/)

### Refactoring Best Practices
- [FlexPipe: Adapting Dynamic LLM Serving](https://arxiv.org/pdf/2510.11938)
- [Demystifying Production Inference Serving for Large Language Models in 2026 | by Shankar Jayaratnam | Jan, 2026](https://medium.com/@jsshankar/demystifying-production-inference-serving-for-large-language-models-in-2026-7cfeea701b53)
- [AI Risk Mitigation: Tools and Strategies for 2026](https://www.sentinelone.com/cybersecurity-101/data-and-ai/ai-risk-mitigation/)
- [AI-Driven Refactoring: A Pipeline for Identifying and Correcting Data Clumps in Git Repositories](https://www.mdpi.com/2079-9292/13/9/1644)

### Environment Variables and Configuration
- [Best Practices for Python Env Variables](https://dagster.io/blog/python-environment-variables)
- [Leveraging Environment Variables in Python Programming - Configu](https://configu.com/blog/working-with-python-environment-variables-and-5-best-practices-you-should-know/)
- [How to Work with Environment Variables in Python](https://www.freecodecamp.org/news/how-to-work-with-environment-variables-in-python/)
- [Refactoring Python Flask Environment Variables with Environs module](https://medium.com/@aswens0276/refactoring-python-flask-environment-variables-with-environs-module-d0e1850c89eb)
