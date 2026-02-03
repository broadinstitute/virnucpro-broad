# VirNucPro GPU Optimization

## What This Is

A performance optimization project for VirNucPro, a viral nucleotide prediction pipeline. The pipeline processes thousands of DNA sequences through DNABERT-S and ESM-2 embeddings before classification. Currently, the embedding steps take 45+ hours for one sample; this project optimizes them to under 10 hours through multi-GPU parallelization and intelligent batch processing.

## Core Value

Embedding steps (DNABERT-S and ESM-2) efficiently utilize all available GPUs and automatically queue batches, reducing sample processing time from 45+ hours to under 10 hours.

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
- ✓ **FLASH-01**: FlashAttention-2 integration for ESM-2 and DNABERT-S (2-4x attention speedup) — v1.0
- ✓ **PERSIST-01**: Persistent model loading eliminates re-loading overhead — v1.0
- ✓ **MEM-01**: Memory management with expandable segments and periodic cache clearing — v1.0

### Active

**Deferred from v1.0 (needs validation):**

- [ ] **PERF-01**: Pipeline completes one sample in under 10 hours (needs end-to-end benchmarking)
- [ ] **PERF-02**: GPU utilization >80% during embedding steps (needs validation with nvitop)
- [ ] **SCALE-01**: Linear GPU scaling verified (2x GPUs = ~2x faster, needs scaling tests)

**Future enhancements:**

- [ ] **LOAD-01**: Work-stealing queue for dynamic load balancing
- [ ] **SEC-01**: Upgrade transformers to 4.53.0+ (address 12 CVEs including 4 RCE vulnerabilities)

### Out of Scope

- Optimizing non-embedding pipeline stages (chunking, translation, prediction) — benchmark these later, focus is embeddings only
- Changing CLI interface for users — must remain `virnucpro predict --input X --output-dir Y`
- Distributed multi-node processing — single-machine multi-GPU only
- CPU-only optimization — GPUs are the target environment

## Context

**Current Performance:**
- ESM-2 embedding: 45 hours for one sample on single GPU
- DNABERT-S embedding: Multi-GPU capable but underutilized (one file per GPU)
- Sample size: Thousands of sequences, chunked into batches of 10k sequences

**Current Architecture:**
- Layered pipeline: CLI → Core → Pipeline → Utils
- Feature extraction in `virnucpro/pipeline/feature_extraction.py` (244 lines)
- Parallel extraction in `virnucpro/pipeline/parallel_feature_extraction.py` (113 lines)
- Multiprocessing with spawn context for CUDA compatibility
- Checkpointing via `CheckpointManager` in `virnucpro/core/checkpointing.py`

**Technical Environment:**
- Python 3.9, PyTorch >=2.8.0
- Transformers 4.30.0 (DNABERT-S: `zhihan1996/DNABERT-S`)
- ESM 2.0.0 (ESM-2 3B parameter model)
- Variable GPU configurations across users (4-8 GPUs typical)

**Known Issues from CONCERNS.md:**
- ESM-2 extraction not parallelized (single GPU bottleneck)
- File I/O in tight loops (many small FASTA writes)
- Redundant sequence parsing between stages
- Checkpoint resume fragility with empty/corrupt files

## Constraints

- **CLI Compatibility**: Must not break existing `virnucpro predict` interface — users should not need to change commands
- **Checkpoint Compatibility**: Must support resuming from existing pre-optimization checkpoint files
- **GPU Variability**: Must work across different GPU counts (1, 4, 8+) without code changes
- **Dependencies**: Open to new dependencies (DeepSpeed, Ray, etc.) if they provide significant speedup
- **Memory**: ESM-2 3B model is large; GPU memory limits batch sizes

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Focus on embeddings only | ESM-2 is 45-hour bottleneck, biggest ROI | ✓ Good - delivered parallelization |
| Target <10 hours for one sample | 4.5x speedup from current 45 hours | ⚠️ Revisit - not validated (benchmarking incomplete) |
| Maintain CLI interface | Users have existing workflows and scripts | ✓ Good - zero breaking changes |
| Use file-level data parallelism | Simpler than tensor parallelism, ESM-2 fits on single GPU | ✓ Good - reliable scaling pattern |
| BF16 on Ampere+ GPUs | 2x speedup with minimal accuracy impact | ✓ Good - significant performance gain |
| Persistent model loading | Eliminate re-loading overhead | ✓ Good - opt-in feature working |
| FlashAttention-2 via PyTorch SDPA | Native integration vs separate flash-attn package | ✓ Good - simpler maintenance |

---
*Last updated: 2026-02-02 after v1.0 milestone*
