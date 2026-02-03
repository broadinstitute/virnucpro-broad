# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-02)

**Core value:** Embedding steps (DNABERT-S and ESM-2) efficiently utilize all available GPUs and automatically queue batches, reducing sample processing time from 45+ hours to under 10 hours.

**Current focus:** Phase 5 - Async DataLoader Foundation

## Current Position

Phase: 5 of 10 (Async DataLoader Foundation)
Plan: 0 of TBD in current phase
Status: Ready to plan
Last activity: 2026-02-03 — v2.0 roadmap created

Progress: [████░░░░░░] 34/TBD plans (v1.0: 34/34 complete, v2.0: 0/TBD)

## Performance Metrics

**Velocity:**
- Total plans completed: 34 (v1.0)
- Average duration: 3.4 min
- Total execution time: 2.2 hours

**By Phase (v1.0):**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 1 | 5 | 17 min | 3.4 min |
| 2 | 7 | 24 min | 3.4 min |
| 3 | 7 | 24 min | 3.4 min |
| 4 | 12 | 41 min | 3.4 min |
| 4.1 | 3 | 10 min | 3.3 min |

**Recent Trend:**
- Last 5 plans (v1.0): ~3.4 min average
- Trend: Stable

**v2.0 Note:** Phase planning will determine plan counts and reset velocity tracking for this milestone.

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- **v2.0 Async architecture**: Multi-worker-per-GPU causes N×11GB memory overhead, serialization tax, GPU starvation - replacing with single-process-per-GPU + async DataLoader (major refactor, breaking changes acceptable)
- **FP16 over BF16**: Research shows ESM-2 trained in FP16 (norm difference <1e-3 vs FP32) - FP16 is optimal for this model
- **FlashAttention varlen**: Required for sequence packing to prevent cross-sequence attention contamination
- **File-level sharding**: Multi-GPU coordination uses deterministic file assignment (rank % world_size) instead of dynamic work-stealing

### Pending Todos

None yet.

### Blockers/Concerns

**Phase 5 (Async DataLoader Foundation):**
- CUDA worker safety: Workers must never initialize CUDA (fork copies CUDA context causing corruption) - use spawn context and validate torch.cuda.is_available() returns False in Dataset.__init__
- HuggingFace cache race: Multiple workers calling AutoModel.from_pretrained() simultaneously can corrupt cache - stagger loading with worker_id delay or use filelock
- Persistent worker memory leaks: prefetch_factor>2 causes gradual CPU RAM accumulation - use conservative prefetch_factor=2 for inference workloads

**Phase 6 (Sequence Packing Integration):**
- FlashAttention varlen API: fair-esm 2.0.0 may require manual integration of flash_attn_varlen_func - investigate esm2_flash.py wrapper layer
- Position ID off-by-one: Must reset position IDs to 0 at each cu_seqlens boundary (not sequential [0,1,2,3,4,5])
- Packing correctness: Need extensive validation (packed output == unpacked output for same sequences)

**Phase 8 (FP16 Precision Validation):**
- Numerical precision: LayerNorm may have limited dynamic range in FP16 - may need selective FP32 for specific layers while keeping rest in FP16

## Session Continuity

Last session: 2026-02-03 (roadmap creation)
Stopped at: v2.0 roadmap created with 6 phases (5-10) covering 42 requirements
Resume file: None

**Next step:** `/gsd:plan-phase 5` to decompose Async DataLoader Foundation into executable plans
