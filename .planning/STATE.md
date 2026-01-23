# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-01-22)

**Core value:** Embedding steps (DNABERT-S and ESM-2) efficiently utilize all available GPUs and automatically queue batches, reducing sample processing time from 45+ hours to under 10 hours.
**Current focus:** Phase 1 - ESM-2 Multi-GPU Foundation

## Current Position

Phase: 1 of 6 (ESM-2 Multi-GPU Foundation)
Plan: 6 of 6 (Gap Closure - BF16 Logging Visibility)
Status: In progress
Last activity: 2026-01-23 — Completed 01-06-PLAN.md (BF16 Logging Visibility)

Progress: [████░░░░░░] 40.0% (6/15 plans)

## Performance Metrics

**Velocity:**
- Total plans completed: 6
- Average duration: 3.8 minutes
- Total execution time: 0.38 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 1     | 6     | 22.5m | 3.8m     |

**Recent Trend:**
- Last 5 plans: 01-02 (2.4m), 01-03 (4.0m), 01-04 (12.4m), 01-05 (0m), 01-06 (1m)
- Trend: Gap closure plans (01-05, 01-06) very fast, focused fixes

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

**From 01-04 execution:**
- auto-enable-parallel-from-gpus: Auto-enable parallel processing when --gpus flag contains multiple GPU IDs (improves UX)
- python39-type-compatibility: Use Optional[Tuple[...]] instead of | operator for Python 3.9 compatibility
- integration-test-subprocess: Use subprocess calls in integration tests to test exact CLI interface users invoke

**From 01-06 execution:**
- worker-logging-init: Initialize logging at worker function start with setup_worker_logging()
- log-config-via-kwargs: Pass log_level and log_format to workers through BatchQueueManager kwargs
- gpu-capability-main-log: Log GPU capabilities and BF16 status in main process before spawning workers

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

Last session: 2026-01-23 12:08 UTC
Stopped at: Completed 01-06-PLAN.md execution (BF16 Logging Visibility gap closure)
Resume file: None
