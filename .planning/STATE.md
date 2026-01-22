# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-01-22)

**Core value:** Embedding steps (DNABERT-S and ESM-2) efficiently utilize all available GPUs and automatically queue batches, reducing sample processing time from 45+ hours to under 10 hours.
**Current focus:** Phase 1 - ESM-2 Multi-GPU Foundation

## Current Position

Phase: 1 of 6 (ESM-2 Multi-GPU Foundation)
Plan: 1 of 4 (ESM-2 Multi-GPU Foundation)
Status: In progress
Last activity: 2026-01-22 — Completed 01-01-PLAN.md (ESM-2 Multi-GPU Foundation)

Progress: [█░░░░░░░░░] 6.7% (1/15 plans)

## Performance Metrics

**Velocity:**
- Total plans completed: 1
- Average duration: 2.9 minutes
- Total execution time: 0.05 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 1     | 1     | 2.9m  | 2.9m     |

**Recent Trend:**
- Last 5 plans: 01-01 (2.9m)
- Trend: First plan (baseline established)

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- Focus on embeddings only: ESM-2 is 45-hour bottleneck, biggest ROI
- Target <10 hours for one sample: 4.5x speedup from current 45 hours
- Maintain CLI interface: Users have existing workflows and scripts
- Open to new dependencies: Willing to add libraries (DeepSpeed, Ray) if they accelerate significantly

**From 01-01 execution:**
- bf16-auto-detect: Automatically detect GPU compute capability and enable BF16 on Ampere+ GPUs
- spawn-context-default: Use spawn context by default for multiprocessing (prevents CUDA re-init errors)
- batch-size-increase-with-bf16: Increase toks_per_batch from 2048 to 3072 when BF16 enabled

### Pending Todos

None yet.

### Blockers/Concerns

**Phase 1 Critical:**
- ✓ CUDA context initialization: Resolved via spawn context and deferred GPU initialization in workers (01-01)
- Batch size variability can fragment memory even with available VRAM — implement sequence sorting and expandable segments from start
- Checkpoint corruption from partial writes — use atomic temp-then-rename pattern

**Phase 4 Research:**
- FlashAttention-2 compatibility with fair-esm library (ESM-2) needs verification during planning — DNABERT-S integration is straightforward

## Session Continuity

Last session: 2026-01-22 22:53 UTC
Stopped at: Completed 01-01-PLAN.md execution
Resume file: None
