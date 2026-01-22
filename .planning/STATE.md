# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-01-22)

**Core value:** Embedding steps (DNABERT-S and ESM-2) efficiently utilize all available GPUs and automatically queue batches, reducing sample processing time from 45+ hours to under 10 hours.
**Current focus:** Phase 1 - ESM-2 Multi-GPU Foundation

## Current Position

Phase: 1 of 6 (ESM-2 Multi-GPU Foundation)
Plan: 3 of 4 (ESM-2 Multi-GPU Foundation)
Status: In progress
Last activity: 2026-01-22 — Completed 01-03-PLAN.md (Pipeline Integration & CLI)

Progress: [██░░░░░░░░] 20.0% (3/15 plans)

## Performance Metrics

**Velocity:**
- Total plans completed: 3
- Average duration: 3.0 minutes
- Total execution time: 0.15 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 1     | 3     | 9.1m  | 3.0m     |

**Recent Trend:**
- Last 5 plans: 01-01 (2.9m), 01-02 (2.4m), 01-03 (4.0m)
- Trend: Consistent velocity (~2.5-4.0 minutes per plan)

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

**From 01-02 execution:**
- cuda-api-direct-access: Use torch.cuda.mem_get_info() directly instead of parsing nvidia-smi for lower overhead
- rich-with-fallback: Use Rich library for live progress display with automatic TTY detection and logging fallback
- background-monitoring-thread: Implement daemon thread for periodic memory logging without blocking GPU workers

**From 01-03 execution:**
- esm-batch-size-separate-flag: Use --esm-batch-size CLI flag instead of --batch-size to avoid confusion with prediction batch size
- pipeline-exit-codes: Return exit codes from pipeline (0: success, 1: failure, 2: partial) to signal partial failures
- failed-files-pipe-delimited: Log failed files to failed_files.txt with format {file_path}|ESM-2|{error_message}

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

Last session: 2026-01-22 23:07 UTC
Stopped at: Completed 01-03-PLAN.md execution
Resume file: None
