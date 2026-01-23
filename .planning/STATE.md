# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-01-22)

**Core value:** Embedding steps (DNABERT-S and ESM-2) efficiently utilize all available GPUs and automatically queue batches, reducing sample processing time from 45+ hours to under 10 hours.
**Current focus:** Phase 2 - DNABERT-S Optimization

## Current Position

Phase: 2 of 6 (DNABERT-S Optimization)
Plan: 4 of 4 (Batch Size Profiling Utilities)
Status: Phase complete
Last activity: 2026-01-23 — Completed 02-04-PLAN.md

Progress: [█████████░] 93.3% (14/15 plans)

## Performance Metrics

**Velocity:**
- Total plans completed: 14
- Average duration: 3.8 minutes
- Total execution time: 0.90 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 1     | 7     | 33.6m | 4.8m     |
| 1.1   | 3     | 10.3m | 3.4m     |
| 2     | 4     | 11.5m | 2.9m     |

**Recent Trend:**
- Last 5 plans: 01.1-03 (6.4m), 02-01 (2.9m), 02-02 (3.3m), 02-03 (2.7m), 02-04 (2.6m)
- Trend: Phase 2 complete; consistently fast execution averaging 2.9m/plan

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

**From 01-05 execution:**
- gpu-auto-detection: Auto-detect available GPUs and enable parallel mode without user flags
- early-cli-detection: Detect GPUs early in CLI before pipeline execution to set flags
- zero-config-multi-gpu: Multi-GPU systems work without --gpus or --parallel flags (zero configuration)
- preserve-user-control: Explicit --gpus flag overrides auto-detection (user control preserved)

**From 01-07 execution:**
- progress-queue-pattern: Workers report progress via multiprocessing Queue, monitor thread updates dashboard
- bin-packing-by-sequences: Distribute files by sequence count (not file count) for balanced GPU utilization
- unified-worker-interface: Both DNABERT and ESM workers return (processed, failed) tuple for consistency
- dashboard-auto-tty-detect: Dashboard automatically uses Rich Live in TTY, logging fallback in non-TTY

**From 01.1-01 execution:**
- batch-processing-100x-reduction: Use batch worker to reduce serialization from 22M to 220K operations (100x improvement)
- spawn-context-consistency: Use spawn context matching GPU worker pattern for consistency and safety
- imap-for-memory: Use Pool.imap() instead of Pool.map() for lazy evaluation with 22M sequences
- optimal-settings-helper: Provide get_optimal_settings() to calculate num_workers, batch_size, chunksize based on data characteristics

**From 01.1-02 execution:**
- auto-parallel-multicore: Automatically use parallel translation when multiple cores available (num_workers > 1)
- sequential-fallback: Maintain sequential translation as fallback for single-core systems and parallel failures
- performance-metrics-logging: Log sequences/sec, total time, and ORF count for observability

**From 01.1-03 execution:**
- test-output-equivalence: Compare parallel and sequential outputs byte-for-byte to verify correctness
- performance-threshold-1.5x: Expect minimum 1.5x speedup with 4 workers (conservative for testing)
- mock-progress-at-module: Mock ProgressReporter at utils.progress level (imported inside function)

**From 02-01 execution:**
- base-worker-abstraction: Create BaseEmbeddingWorker abstract class shared by DNABERT-S and ESM-2 for unified interface
- token-abstraction-dna: Treat DNA sequence length as token count (1 base ≈ 1 token) to abstract k-mer complexity
- shared-utilities-location: Place shared utilities (count_sequences, assign_files_by_sequences) in base_worker.py as single source of truth

**From 02-03 execution:**
- dnabert-batch-size-default-2048: Default DNABERT-S batch size to 2048 tokens (not 256) matching token-based batching pattern

**From 02-04 execution:**
- profiling-80-percent-headroom: Recommend 80% of maximum batch size as optimal to leave headroom for sequence length variation
- synthetic-test-sequences: Generate synthetic test sequences if no file provided for zero-config profiling

### Pending Todos

None yet.

### Roadmap Evolution

- **Phase 1.1 inserted after Phase 1** (2026-01-23): Parallel Translation (URGENT)
  - Reason: Six-frame translation taking >10 minutes on 22M sequences is a bottleneck
  - Scope: Add CPU multiprocessing to translation step, add --threads CLI parameter
  - Impact: Reduces non-GPU bottleneck before proceeding to Phase 2 DNABERT optimization

### Blockers/Concerns

**Phase 1 Critical:**
- ✓ CUDA context initialization: Resolved via spawn context and deferred GPU initialization in workers (01-01)
- Batch size variability can fragment memory even with available VRAM — implement sequence sorting and expandable segments from start
- Checkpoint corruption from partial writes — use atomic temp-then-rename pattern

**Phase 4 Research:**
- FlashAttention-2 compatibility with fair-esm library (ESM-2) needs verification during planning — DNABERT-S integration is straightforward

## Session Continuity

Last session: 2026-01-23 16:07 UTC
Stopped at: Completed 02-04-PLAN.md execution (Batch Size Profiling Utilities) - Phase 2 complete
Resume file: None
