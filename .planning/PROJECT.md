# VirNucPro GPU Optimization

## What This Is

A performance optimization project for VirNucPro, a viral nucleotide prediction pipeline. The pipeline processes DNA sequences through DNABERT-S and ESM-2 embeddings before classification. The v2.0 async architecture delivers 6.2x speedup over v1.0 (3.5 hours to 34 minutes on 2x RTX 4090) through single-process-per-GPU design, async DataLoader, sequence packing with FlashAttention varlen, FP16 precision, and fault-tolerant checkpointing.

## Core Value

Embedding steps (DNABERT-S and ESM-2) efficiently utilize all available GPUs with async DataLoader and sequence packing, delivering 6.2x speedup over v1.0 baseline.

## Current State

**Shipped:** v2.0 Async Architecture + Sequence Packing (2026-02-09)

**Architecture:**
- Single-process-per-GPU with async DataLoader (4-8 CPU workers for I/O)
- Sequence packing via GreedyPacker FFD algorithm (~92-94% efficiency)
- FlashAttention varlen for packed attention (no cross-sequence contamination)
- FP16 precision with NaN/Inf detection (<1ms overhead)
- Stride-based multi-GPU index distribution [rank::world_size]
- Fault-tolerant checkpointing with SIGTERM handling and elastic redistribution

**Performance (validated on 1M subset, 2x RTX 4090):**
- v1.0 baseline: 3.5 hours
- v2.0 result: 33.7 minutes (6.2x speedup)
- ESM-2 scaling: 1.87x with 2 GPUs (93.7% efficiency)
- Correctness: 99.87% consensus agreement with v1.0
- Packing efficiency: ~358% token utilization

**Codebase:**
- 18,846 lines of production Python (61 files)
- 27,951 lines of test Python (62 files)
- 93 plans executed across 13 phases (v1.0 + v2.0)

## Requirements

### Validated

**Existing capabilities from production codebase:**

- ✓ **CLI-01**: Users run predictions via Click-based CLI (`virnucpro predict`) — existing
- ✓ **PIPELINE-01**: 9-stage pipeline (chunking → translation → splitting → DNABERT-S → ESM-2 → merging → prediction → consensus → output) — existing
- ✓ **CHECKPOINT-01**: Each pipeline stage saves checkpoints for resumable execution — existing
- ✓ **DNABERT-01**: DNABERT-S extracts features from nucleotide sequences — existing
- ✓ **ESM-01**: ESM-2 extracts features from protein sequences — existing
- ✓ **PARALLEL-01**: DNABERT-S supports multi-GPU via parallel_feature_extraction.py — existing
- ✓ **IO-01**: Reads FASTA input, writes CSV/TXT predictions — existing
- ✓ **CONFIG-01**: YAML-based configuration for chunk sizes, model paths — existing

**Delivered in v1.0 (Phases 1-4.1):**

- ✓ **ESM-OPT-01**: ESM-2 parallelizes across multiple GPUs — v1.0
- ✓ **ESM-OPT-02**: ESM-2 automatically queues and processes multiple batches per GPU — v1.0
- ✓ **DNABERT-OPT-01**: DNABERT-S optimized batch sizes with BF16 (3072 tokens on Ampere+) — v1.0
- ✓ **DNABERT-OPT-02**: DNABERT-S automatically queues and processes multiple batches per GPU — v1.0
- ✓ **SCALE-02**: Works with variable GPU counts (1, 4, 8, or any number) with auto-detection — v1.0
- ✓ **COMPAT-01**: Maintains backward compatibility with existing CLI interface — v1.0
- ✓ **COMPAT-02**: Resumes checkpoints from pre-optimization runs with atomic writes — v1.0
- ✓ **FLASH-01**: FlashAttention-2 integration for ESM-2 and DNABERT-S — v1.0
- ✓ **PERSIST-01**: Persistent model loading eliminates re-loading overhead — v1.0
- ✓ **MEM-01**: Memory management with expandable segments and periodic cache clearing — v1.0

**Delivered in v2.0 (Phases 5-10):**

- ✓ **ARCH-01 to ARCH-11**: Async DataLoader foundation + sequence packing architecture — v2.0
- ✓ **SAFE-01 to SAFE-05**: CUDA safety (spawn context, CPU-only workers, deferred init) — v2.0
- ✓ **PACK-01 to PACK-06a**: FlashAttention varlen packing with FFD algorithm — v2.0
- ✓ **PREC-01, PREC-02**: FP16 precision with >0.99 cosine similarity to FP32 — v2.0
- ✓ **GPU-01 to GPU-04**: Multi-GPU coordination with stride-based sharding — v2.0
- ✓ **CKPT-01 to CKPT-06**: Fault-tolerant checkpointing with crash recovery — v2.0
- ✓ **PERF-01**: Pipeline completes 1M subset in 53:20 on 1x RTX 4090 (<1h target) — v2.0
- ✓ **PERF-04, PERF-05**: Throughput telemetry (321 seq/s, 16.5K tokens/s) — v2.0
- ⚠ **PERF-02**: GPU utilization 60-80% (target >80% — measurement methodology questionable) — v2.0
- ⚠ **PERF-03**: Overall scaling 1.58x (ESM-2: 1.87x excellent; DNABERT-S v1.0 drag) — v2.0

### Active

**Future enhancements:**

- [ ] **LOAD-01**: Work-stealing queue for dynamic load balancing
- [ ] **SEC-01**: Upgrade transformers to 4.53.0+ (address 12 CVEs including 4 RCE vulnerabilities)
- [ ] **DNABERT-V2**: Port DNABERT-S to v2.0 async architecture (currently v1.0 bin-packing)
- [ ] **ESM-MODEL**: Configurable ESM-2 model selection (support 650M, 3B, custom models via CLI)

### Out of Scope

- Distributed multi-node processing — single-machine multi-GPU only
- CPU-only optimization — GPUs are the target environment
- Optimizing non-embedding pipeline stages — embeddings are the bottleneck

## Context

**Technical Environment:**
- Python 3.9, PyTorch >=2.8.0
- Transformers 4.30.0 (DNABERT-S: `zhihan1996/DNABERT-S`)
- ESM 2.0.0 (ESM-2 3B parameter model, `esm2_t36_3B_UR50D`)
- flash-attn >=2.6.0 (FlashAttention-2 for packed attention)
- RTX 4090 (24GB VRAM) — primary development/benchmark hardware

**Known Technical Debt:**
- DNABERT-S v1.0 bin-packing causes 4% slowdown on 2 GPUs
- repr_layers hardcoded to [36] in 6 locations (blocks model swaps)
- ESM-2 model name hardcoded (not configurable via CLI)
- transformers 4.30.0 has 12 CVEs including 4 RCE

## Constraints

- **GPU Variability**: Must work across different GPU counts (1, 4, 8+) without code changes
- **Memory**: ESM-2 3B model is large; single copy per GPU enables larger batch sizes
- **Precision**: FP16 mixed precision for optimal ESM-2 performance
- **Checkpoint Compatibility**: Must support resuming from existing checkpoint files

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Focus on embeddings only | ESM-2 is 45-hour bottleneck, biggest ROI | ✓ Good - delivered parallelization |
| Target <10 hours for one sample | 4.5x speedup from current 45 hours | ✓ Good - achieved 6.2x (34 min on 2x 4090) |
| Maintain CLI interface | Users have existing workflows and scripts | ✓ Good - zero breaking changes for users |
| File-level data parallelism | Simpler than tensor parallelism, ESM-2 fits on single GPU | ✓ Good - reliable scaling pattern |
| BF16 on Ampere+ GPUs (v1.0) | 2x speedup with minimal accuracy impact | ✓ Good - significant performance gain |
| Persistent model loading | Eliminate re-loading overhead | ✓ Good - opt-in feature working |
| FlashAttention-2 integration | 2-4x attention speedup | ✓ Good - simpler maintenance |
| Async architecture for v2.0 | Multi-worker-per-GPU causes N×11GB memory overhead | ✓ Good - 6.2x speedup achieved |
| FP16 over BF16 (v2.0) | ESM-2 trained in FP16, optimal precision for this model | ✓ Good - stable with >0.99 cosine similarity |
| FFD packing algorithm | First-Fit Decreasing for 92-94% packing efficiency | ✓ Good - ~358% efficiency on production data |
| Stride-based index sharding | [rank::world_size] on length-sorted index for balanced GPU load | ✓ Good - 93.7% scaling efficiency on ESM-2 |
| Hybrid v1.0/v2.0 architecture | ESM-2 uses v2.0, DNABERT-S stays v1.0 (staged validation) | ⚠ Revisit - DNABERT-S drags overall scaling to 1.58x |
| FlashAttention divergence is cosmetic | 100% label agreement despite minor embedding differences | ✓ Good - v1_compatible fallback available |

---
*Last updated: 2026-02-09 after v2.0 milestone*
