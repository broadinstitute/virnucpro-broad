# Project Research Summary

**Project:** VirNucPro v2.0 Async Architecture + Sequence Packing
**Domain:** High-throughput protein/DNA transformer inference optimization
**Researched:** 2026-02-02
**Confidence:** HIGH

## Executive Summary

VirNucPro v2.0 aims to replace the v1.0 multi-worker-per-GPU architecture with a modern async DataLoader + sequence packing pipeline to achieve 4.5× speedup (45h → <10h target). Research confirms this is the industry-standard pattern for inference workloads: one process per GPU using native PyTorch DataLoader with CPU workers for async I/O, eliminating v1.0's critical bottlenecks (N×11GB memory overhead from replicated models, pickle serialization tax, and GPU starvation from small batches).

The recommended approach uses **native PyTorch capabilities only** (DataLoader, torch.amp.autocast, FlashAttention-2 varlen kernels) with **zero new dependencies**. Three optimizations stack multiplicatively: (1) async DataLoader with prefetching (1.2-1.5× speedup by overlapping I/O with compute), (2) sequence packing via custom greedy algorithm (2-3× speedup by eliminating 50-70% padding waste), and (3) FP16 mixed precision (1.5-2× speedup on tensor cores). Combined with existing 4-GPU parallelization, this delivers 4-6× total speedup, comfortably exceeding the <10h target with conservative estimates yielding 7.5-11h.

Key risks are **CUDA worker safety** (workers must never touch CUDA tensors), **packing correctness** (position IDs must reset per sequence, attention masks must prevent cross-contamination), and **memory leaks** from persistent workers with high prefetch factors. Mitigation is straightforward: spawn context for workers, cu_seqlens validation for packing, and conservative prefetch_factor=2 for inference workloads. Research shows these patterns are well-established with extensive PyTorch documentation and production validation from ESM-2 inference papers achieving 9.4× speedup with identical techniques.

## Key Findings

### Recommended Stack

All required technologies are native to PyTorch ≥2.8.0, eliminating external dependency risk. The async DataLoader pattern (num_workers=4-8, prefetch_factor=2, pin_memory=True) overlaps CPU-bound I/O (FASTA parsing, tokenization) with GPU inference, preventing GPU starvation. FP16 mixed precision via torch.amp.autocast provides 2× memory reduction and 1.5-2× speedup; ESM-2 was trained in FP16 making this particularly safe (norm difference <1e-3). Sequence packing requires custom implementation since torchtune's PackedDataset is training-focused; a simple greedy algorithm (sort by length, pack into token budget) suffices for 2-3× gains.

**Core technologies:**
- **torch.utils.data.DataLoader** (PyTorch ≥2.8.0): Async data loading with CPU workers for I/O/compute overlap. Best practices: num_workers=4-8, prefetch_factor=2, persistent_workers=True, pin_memory=True. Zero new dependencies.
- **torch.amp.autocast** (PyTorch ≥2.8.0): FP16 mixed precision for 2× memory reduction and 1.5-2× speedup. ESM-2 trained in FP16, making this ideal. Use torch.float16 (not bfloat16) for closest match to training.
- **Manual greedy packing** (custom implementation): Pack variable-length sequences (100-3000nt, 30-1000aa) into fixed-size batches to minimize padding waste. No suitable off-the-shelf library exists for inference-only workloads. Sort sequences by length, greedily pack into max_tokens budget.
- **torch.cuda.Stream** (PyTorch ≥2.8.0): CUDA stream management for I/O-compute overlap. Existing stream_manager.py already implements this pattern. Enables 20-40% latency reduction by overlapping data movement with inference.
- **torch.compile** (PyTorch ≥2.8.0): JIT compilation for kernel optimization. Provides up to 2× inference speedup (1.35-2× geomean) with single-line code addition. Use mode="default" or "reduce-overhead".

### Expected Features

**Must have (table stakes):**
- Async data loading with CPU workers — Essential for I/O/compute overlap. Eliminates GPU idle time waiting for data.
- Sequence concatenation for packing — Pack multiple sequences into single tensor with cu_seqlens boundaries. Core packing primitive.
- Attention masking for packed sequences — Prevent cross-contamination via flash_attn_varlen_func with cu_seqlens. Critical for correctness.
- Token budget enforcement — Respect GPU memory limits (toks_per_batch=2048-4096). Prevents OOM.
- GPU memory pinning — Enable asynchronous CPU→GPU transfers via pin_memory=True. Required for non_blocking.
- Non-blocking GPU transfers — Overlap data transfer with kernel execution via .to(device, non_blocking=True).
- FP16 precision support — 2× memory reduction and speedup. Requires accuracy validation (cosine similarity >0.99).

**Should have (competitive advantage):**
- Intelligent packing algorithm — First-Fit Decreasing (FFD) or Longest-Pack-First-Highest-Priority (LPFHP). Maximize GPU utilization by minimizing padding waste. Research shows 1.5-2× throughput gains.
- Dynamic batch sizing — Adapt batch size based on sequence lengths to maintain constant token budget. Small sequences → larger batches, long sequences → smaller batches.
- Multi-stream GPU execution — Overlap data transfer and kernel execution using CUDA streams. Provides 20-30% speedup but requires careful synchronization.
- Zero-copy data path — Direct tensor passing from DataLoader → GPU → inference. Avoid intermediate CPU copies.

**Defer (v2+):**
- Continuous batching — Incompatible with current file-based work distribution. Requires major architecture refactor. Only valuable if batch completion time variance is high.
- BF16 precision — Better dynamic range than FP16, but requires Ampere+ GPUs (A100, H100). Check deployment hardware first.
- Custom CUDA kernels for packing — PyTorch native operations fast enough initially. Only optimize if packing overhead >10% of total time.
- Adaptive prefetching — Dynamically tune num_workers based on GPU utilization. Complex monitoring/tuning, marginal benefit for uniform workloads.

### Architecture Approach

VirNucPro v2.0 replaces the multi-worker-per-GPU pattern (4 workers × 11GB model per GPU = 44GB overhead) with single-process-per-GPU where one model loads once per GPU and uses DataLoader's CPU workers for async I/O. This eliminates model replication (1×11GB per GPU instead of 4×11GB), removes pickle serialization overhead, and enables continuous GPU utilization through prefetching. Sequence packing integrates naturally into the collate_fn: when DataLoader assembles a batch, PackingCollator concatenates short sequences into one "packed" sequence up to max_tokens limit, reducing padding waste from 40-60% to <10%.

**Major components:**
1. **SequenceDataset (IterableDataset)** — Streams sequences from FASTA files with file-level sharding for multi-GPU. Each GPU process gets deterministic file shard (rank % world_size). DataLoader workers within a process read files round-robin.
2. **PackingCollator (custom collate_fn)** — Packs sequences into dense batches with FlashAttention-2 variable-length support. Receives list of samples, concatenates into packed tensor, computes cu_seqlens for unpacking, prevents cross-sequence attention.
3. **GPUProcessCoordinator** — Spawns one process per GPU, assigns file shards, aggregates outputs. Replaces v1.0's multiprocessing.Pool with explicit process spawning. Each GPU process is independent (no shared state during processing).
4. **CheckpointIntegrator** — Extends existing atomic write pattern to stream-based processing. File-level resume logic unchanged; GPU process rank included in progress metadata for crash recovery.

### Critical Pitfalls

1. **CUDA tensors in DataLoader workers cause silent corruption** — Workers create CUDA tensors, causing GPUs to receive empty embeddings with no exceptions. CUDA runtime is not fork-safe and CUDA tensors cannot be safely shared between processes. **Avoid:** Keep all Dataset/collate operations on CPU, use spawn context, move to CUDA only in main process after DataLoader yields batch.

2. **Concurrent model loading causes HuggingFace cache race** — Multiple workers call AutoModel.from_pretrained() simultaneously with persistent_workers=True, corrupting cache and producing empty embeddings. **Avoid:** Stagger loading with worker_id × 1s delay, or pre-load in main process before forking workers, or use filelock around from_pretrained().

3. **FlashAttention dtype mismatch breaks packing silently** — flash_attn_varlen_func requires FP16/BF16 but FP32 attention masks cause silent fallback to standard attention without packing-aware masking, causing cross-contamination. **Avoid:** Load model with explicit torch_dtype=torch.bfloat16, ensure attention masks match model dtype, validate before packing.

4. **Position IDs off-by-one corrupts positional embeddings** — Naive packing generates sequential position IDs [0,1,2,3,4,5] instead of per-sequence [0,1,0,1,2,3], corrupting transformers' positional understanding. **Avoid:** Reset position IDs to 0 at each cu_seqlens boundary, validate with roundtrip test.

5. **Attention mask cross-contamination between packed sequences** — Standard attention masks allow tokens from seq1 to attend to seq2 in same pack, mixing information between independent sequences. **Avoid:** Use flash_attn_varlen_func with cu_seqlens for automatic masking, or create block-diagonal mask for standard attention fallback.

6. **Persistent worker memory leaks with prefetching** — persistent_workers=True + prefetch_factor>2 causes gradual CPU RAM accumulation (5-10GB per worker) that's never released. **Avoid:** Use prefetch_factor=2 for inference (not training), cap num_workers at 8, monitor worker memory growth.

## Implications for Roadmap

Based on research, suggested phase structure for incremental migration from v1.0 to v2.0:

### Phase 1: Foundation — Single-GPU Async DataLoader
**Rationale:** Prove async DataLoader pattern works without multi-GPU complexity. Establish safe worker patterns before adding packing sophistication. This is the highest-impact change (eliminates N×11GB overhead and GPU starvation).

**Delivers:** Single-GPU throughput improvement from async prefetching (1.2-1.5× speedup). Validation that DataLoader workers handle FASTA I/O correctly without CUDA corruption.

**Addresses:** Async data loading with CPU workers (table stakes), GPU memory pinning (table stakes), non-blocking GPU transfers (table stakes), prefetch buffer management (table stakes).

**Avoids:** CUDA tensors in workers (Pitfall #1), HuggingFace cache race (Pitfall #2), persistent worker memory leaks (Pitfall #6).

**Implementation:** Create SequenceDataset (IterableDataset) for single file, create basic PackingCollator without FlashAttention integration yet, modify extract_esm_features() to use DataLoader with num_workers=4. Benchmark: compare DataLoader prefetching vs current sequential loading. Validate output .pt files identical to v1.0.

### Phase 2: Packing Integration — Sequence Packing + FlashAttention
**Rationale:** Add packing after async pattern is validated. Packing is the second-highest impact optimization (2-3× throughput gain) but has the most correctness risks (position IDs, attention masks, unpacking). Must validate extensively before multi-GPU scaling.

**Delivers:** 2-3× throughput improvement on variable-length sequences by eliminating 50-70% padding waste. Validation that packed sequences produce identical embeddings to unpacked.

**Addresses:** Intelligent packing algorithm (FFD, competitive advantage), sequence concatenation (table stakes), attention masking for packed sequences (table stakes), token budget enforcement (table stakes).

**Avoids:** FlashAttention dtype mismatch (Pitfall #3), position IDs off-by-one (Pitfall #4), attention mask cross-contamination (Pitfall #5), unpacking corruption from misaligned cu_seqlens (Pitfall #8).

**Implementation:** Implement full PackingCollator with greedy packing algorithm, add cu_seqlens metadata for FlashAttention-2, modify model forward pass to handle packed format, add unpacking logic after inference. Validate: batch density >90%, throughput 2-3× improvement, output embeddings match v1.0 per-sequence mean pooling. Add validation test: packed output == non-packed output for same sequences.

### Phase 3: Multi-GPU Coordinator — Scale to 4 GPUs
**Rationale:** Once async + packing works correctly on single GPU, scale to 4 GPUs with deterministic file sharding. This is low-risk because it extends existing spawn pattern from v1.0.

**Delivers:** 4× throughput improvement from multi-GPU parallelization (3.8× with 95% efficiency). Combined with Phase 1-2: 1.3 × 2.5 × 3.8 = 12× total speedup (45h → 3.5h).

**Addresses:** Multi-GPU sequence distribution (replaced with file-based sharding to avoid coordination complexity anti-pattern).

**Avoids:** Spawn context conflicts (Pitfall #9 — migration-specific).

**Implementation:** Create GPUProcessCoordinator class, implement file sharding logic (rank % world_size), spawn processes with multiprocessing.spawn, aggregate outputs from all GPUs. Validate: 4 GPU processes run independently, file sharding deterministic, output aggregation correct (all sequences present, no duplicates).

### Phase 4: FP16 Precision Validation — Memory + Speed Optimization
**Rationale:** FP16 provides 1.5-2× additional speedup but requires accuracy validation for embeddings. Defer until core architecture is working to isolate any accuracy degradation issues.

**Delivers:** 1.5-2× additional speedup from FP16 tensor cores + 2× memory reduction (11GB → 6GB per model). Enables larger batch sizes (64-128 instead of 32-64). Combined total: 1.3 × 2.5 × 3.8 × 1.75 = 21× speedup (45h → 2.1h).

**Addresses:** FP16 precision support (table stakes, pending validation).

**Avoids:** Accuracy degradation from FP16 (LayerNorm has limited dynamic range in FP16).

**Implementation:** Enable torch.autocast('cuda', dtype=torch.float16) for inference, extract embeddings in both FP32 and FP16, compute cosine similarity, validate similarity >0.99 for random sample of sequences. Update esm2_flash.py to remove "EXPERIMENTAL: Forcing FP32" (line 79-80). Keep embeddings in FP32 for storage (convert after mean pooling).

### Phase 5: Checkpoint Integration — Robustness
**Rationale:** After performance is validated, add robust crash recovery for production use. File-level checkpointing already exists in v1.0, just needs adaptation to stream-based processing.

**Delivers:** Resume from partial completion with stream-based processing. GPU process crash recovery without losing completed work.

**Addresses:** Resumability (v1.0 parity).

**Avoids:** Loss of work from mid-batch crashes.

**Implementation:** Extend file-level .done markers to GPU process outputs, add checkpoint validation (file exists, non-empty, .done marker), implement resume logic (skip completed files), test crash recovery (kill GPU process mid-batch, resume).

### Phase 6 (Optional): Performance Tuning — CUDA Streams + torch.compile
**Rationale:** Phases 1-4 already deliver 12-21× speedup, exceeding <10h target. Streams and torch.compile provide incremental 10-20% gains each but add complexity (compilation cache, stream synchronization). Only pursue if profiling shows specific bottlenecks.

**Delivers:** Additional 1.1-1.2× speedup each from streams and torch.compile. Combined potential: 1.3 × 2.5 × 3.8 × 1.75 × 1.1 × 1.2 = 27× speedup (45h → 1.7h).

**Addresses:** Multi-stream GPU execution (competitive advantage), torch.compile (stack recommendation).

**Implementation:** Integrate stream_manager.py with DataLoader loop (H2D/compute/D2H overlap), wrap model with torch.compile(model, mode="reduce-overhead"), profile with torch.profiler to verify overlap and kernel optimization.

### Phase Ordering Rationale

- **Foundation first (Phase 1)** because async DataLoader is prerequisite for packing and eliminates the biggest bottleneck (N×11GB overhead). Establishing safe worker patterns (spawn context, CPU-only operations) prevents CUDA corruption in all later phases.
- **Packing second (Phase 2)** because it has the highest correctness risk (position IDs, attention masking). Must validate thoroughly on single GPU before multi-GPU scaling amplifies any bugs.
- **Multi-GPU third (Phase 3)** because it's low-risk once single-GPU async + packing works. File sharding extends v1.0 pattern. Enables validation of full pipeline before FP16 adds numerical precision concerns.
- **FP16 fourth (Phase 4)** to isolate accuracy degradation from other changes. If embeddings degrade, we know it's FP16, not packing or DataLoader issues.
- **Checkpointing fifth (Phase 5)** because performance must be validated before adding resume complexity. No point in robust crash recovery if the pipeline doesn't hit performance targets.
- **Optional tuning last (Phase 6)** because Phases 1-4 already deliver 12-21× speedup. Streams and torch.compile are "nice to have" optimizations with diminishing returns.

This ordering follows dependency chains discovered in research: async enables packing (workers handle I/O), packing enables efficiency (reduces padding), multi-GPU enables scale (multiplies throughput), FP16 enables memory headroom (larger batches), checkpointing enables robustness (production deployment).

### Research Flags

**Phases needing deeper research during planning:**
- **Phase 2 (Packing Integration)** — FlashAttention varlen API integration may need model architecture research if ESM-2/DNABERT don't natively support cu_seqlens parameter. Research shows most HuggingFace transformers support this via `attn_implementation="flash_attention_2"` but verification needed for fair-esm library.
- **Phase 4 (FP16 Validation)** — ESM-2 embedding degradation in FP16 may require layer-specific precision research. Some sources mention LayerNorm instability in FP16; may need selective FP32 for specific layers while keeping rest in FP16.

**Phases with standard patterns (skip research-phase):**
- **Phase 1 (Foundation)** — PyTorch DataLoader is extensively documented with established best practices. num_workers tuning, pin_memory, prefetch_factor are well-understood parameters.
- **Phase 3 (Multi-GPU Coordinator)** — File-based sharding and multiprocessing.spawn are v1.0-validated patterns. Just extending existing code.
- **Phase 5 (Checkpoint Integration)** — File-level .done markers already implemented in v1.0. Stream-based processing changes are minimal (accumulate in dict instead of list).
- **Phase 6 (Streams/compile)** — stream_manager.py already exists from v1.0 Phase 4. torch.compile is single-line change with well-documented modes.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | All technologies native to PyTorch ≥2.8.0 with official documentation. ESM-2 FP16 compatibility verified in GitHub discussions and FastESM2 model card. No external dependencies eliminates integration risk. |
| Features | HIGH | Table stakes features (async loading, packing, attention masking) are well-documented in PyTorch and research papers. Competitive features (dynamic batching, multi-stream) have production validation from LLM inference papers. Anti-features (multi-GPU distribution, KV cache) are confirmed not applicable for inference-only workloads. |
| Architecture | HIGH | Single-process-per-GPU + DataLoader pattern is industry standard for inference (validated in PyTorch docs, NVIDIA NeMo, HuggingFace transformers). IterableDataset for streaming is established pattern. PackingCollator as custom collate_fn aligns with PyTorch design. Component boundaries (Dataset, Collator, Coordinator, Checkpoint) are clean and testable. |
| Pitfalls | HIGH | All 10 pitfalls documented in PyTorch issues, research papers, or VirNucPro v1.0 debugging logs. CUDA worker corruption is well-known PyTorch multiprocessing issue. HuggingFace cache race observed in v1.0 empty-files-race-condition.md. FlashAttention dtype requirements from official flash-attn repo issues. Position ID bugs documented in LLaMA-Factory PR #7754. |

**Overall confidence:** HIGH

Research is grounded in official PyTorch documentation, peer-reviewed papers (Efficient Inference for Protein Language Models achieving 9.4× speedup with identical techniques), and VirNucPro v1.0 production experience. All recommended technologies are mature (PyTorch 2.x APIs, FlashAttention-2) with extensive community validation. Risk is primarily in integration complexity (packing correctness, worker safety) rather than technology uncertainty, and mitigation strategies are well-established.

### Gaps to Address

**FP16 numerical precision for ESM-2 embeddings** — Research confirms ESM-2 was trained in FP16 (norm difference <1e-3 vs FP32), but this needs empirical validation with VirNucPro's specific embedding extraction and mean pooling. Plan: Phase 4 includes validation test suite comparing FP16 vs FP32 embeddings on representative sample. Accept FP16 only if cosine similarity >0.99.

**FlashAttention-2 integration with fair-esm library** — Research shows HuggingFace transformers support flash_attn_varlen_func via `attn_implementation="flash_attention_2"` parameter, but fair-esm 2.0.0 (used for ESM-2) may require manual integration. Plan: Phase 2 includes investigation of esm2_flash.py wrapper layer. May need to replace ESM model's attention layers with FlashAttention varlen kernels, or migrate to HuggingFace's ESM implementation.

**Optimal num_workers and prefetch_factor for VirNucPro workload** — Research recommends num_workers=4-8 and prefetch_factor=2, but optimal values depend on FASTA file size distribution, tokenization speed, and GPU inference speed. Plan: Phase 1 includes tuning phase with GPU utilization monitoring. Start conservative (num_workers=4, prefetch_factor=2), increase incrementally while monitoring memory.

**Sequence length distribution impact on packing efficiency** — Research shows packing provides 1.5-2× gains for variable-length sequences, but actual gain depends on length variance. VirNucPro has 100-3000nt DNA (30× range) and 30-1000aa protein (33× range), suggesting high packing potential. Plan: Phase 2 includes efficiency monitoring (log actual packing density, padding waste per batch). If efficiency <85%, implement more sophisticated bin-packing (LPFHP instead of FFD).

## Sources

### Primary (HIGH confidence)
- [PyTorch DataLoader Official Documentation](https://pytorch.org/docs/stable/data.html) — num_workers, pin_memory, prefetch_factor parameters, IterableDataset patterns
- [PyTorch AMP Documentation](https://pytorch.org/docs/stable/amp.html) — torch.amp.autocast API, FP16 mixed precision
- [PyTorch FlexAttention Documentation](https://pytorch.org/docs/stable/nn.attention.flex_attention.html) — document masking for packed sequences
- [PyTorch Multiprocessing Best Practices](https://pytorch.org/docs/stable/notes/multiprocessing.html) — spawn context, CUDA initialization timing
- [FlashAttention-2 GitHub](https://github.com/Dao-AILab/flash-attention) — flash_attn_varlen_func API, cu_seqlens format
- [ESM-2 FP16 Discussion (GitHub #684)](https://github.com/facebookresearch/esm/discussions/684) — ESM-2 trained in FP16, norm difference <1e-3
- [Efficient Inference for Protein Language Models (PMC12481099)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12481099/) — 9.4× speedup with FlashAttention-2 + packing for ESM-2

### Secondary (MEDIUM confidence)
- [8 PyTorch DataLoader Tactics](https://medium.com/@Modexa/8-pytorch-dataloader-tactics-to-max-out-your-gpu-22270f6f3fa8) — num_workers best practices (2× CPU cores per GPU)
- [Dynamic Batching vs. Sequence Packing](https://medium.com/better-ml/dynamic-batching-vs-sequence-packing-0ef4a3894dad) — performance comparison (1.5-2× speedup)
- [NVIDIA NeMo Sequence Packing](https://docs.nvidia.com/nemo/rl/latest/design-docs/sequence-packing-and-dynamic-batching.html) — implementation patterns
- [HuggingFace Blog: Packing with Flash Attention 2](https://huggingface.co/blog/packing-with-FA2) — cu_seqlens format and usage, cross-contamination prevention
- [PyTorch num_workers Guide (Forums)](https://discuss.pytorch.org/t/guidelines-for-assigning-num-workers-to-dataloader/813) — worker count tuning

### Tertiary (MEDIUM-HIGH confidence - project-specific)
- VirNucPro v1.0 debugging logs — empty-files-race-condition.md (HuggingFace cache race pattern), flashattention-not-integrated.md (wrapper integration gap)
- VirNucPro v1.0 test.txt — N×11GB memory overhead, pickle serialization tax, GPU starvation from small batches (documented v1.0 bottlenecks)
- VirNucPro Phase 4 research — 04-RESEARCH.md (FlashAttention patterns, DataLoader configuration, stream_manager.py implementation)

---
*Research completed: 2026-02-02*
*Ready for roadmap: yes*
