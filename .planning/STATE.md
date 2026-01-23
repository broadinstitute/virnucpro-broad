# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-01-22)

**Core value:** Embedding steps (DNABERT-S and ESM-2) efficiently utilize all available GPUs and automatically queue batches, reducing sample processing time from 45+ hours to under 10 hours.
**Current focus:** Phase 2 - DNABERT-S Optimization

## Current Position

Phase: 2.1 of 6 (Parallel Embedding Merge)
Plan: 2 of 2 (Pipeline Integration)
Status: Phase complete
Last activity: 2026-01-23 — Completed 02.1-02-PLAN.md

Progress: [███████████] 100% (17/17 plans)

## Performance Metrics

**Velocity:**
- Total plans completed: 17
- Average duration: 3.8 minutes
- Total execution time: 1.13 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 1     | 7     | 33.6m | 4.8m     |
| 1.1   | 3     | 10.3m | 3.4m     |
| 2     | 5     | 19.5m | 3.9m     |
| 2.1   | 2     | 6.2m  | 3.1m     |

**Recent Trend:**
- Last 5 plans: 02-03 (2.7m), 02-04 (2.6m), 02-05 (8.0m), 02.1-01 (3.7m), 02.1-02 (2.5m)
- Trend: Phase 2.1 complete; CPU parallelization for merge operations integrated

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

**From 02-05 execution:**
- single-file-auto-split: Automatically split large files (>10K sequences) into num_gpus * 2 files for balanced multi-GPU distribution
- batch-size-bf16-3072: DNABERT-S batch size is 3072 tokens with BF16 on Ampere+ GPUs (not 2048)
- no-multi-file-requirement: Parallel processing works with single large file via auto-splitting (no minimum file count)

**From 02.1-01 execution:**
- spawn-context-merge: Use spawn context for merge workers matching Phase 1.1 pattern for consistency and Python 3.14 compatibility
- chunksize-dynamic: Calculate chunksize dynamically (max(1, num_files // (workers * 4))) to balance overhead vs parallelism
- checkpoint-and-atomic: Add checkpoint skip and atomic write to merge_features() for resume capability and corruption prevention
- batch-size-one: Default batch_size=1 for merge (each file pair is substantial work, batching available for 100+ files)

**From 02.1-02 execution:**
- auto-parallel-merge: Auto-detect CPU count and use parallel merge when >1 core and >1 file for zero-config optimization
- sequential-fallback-merge: Maintain sequential merge path for single file or single core to avoid overhead when not beneficial
- merge-threads-cli: Add --merge-threads CLI parameter for user control of merge parallelism (default: auto-detect)

### Pending Todos

None yet.

### Roadmap Evolution

- **Phase 1.1 inserted after Phase 1** (2026-01-23): Parallel Translation (URGENT)
  - Reason: Six-frame translation taking >10 minutes on 22M sequences is a bottleneck
  - Scope: Add CPU multiprocessing to translation step, add --threads CLI parameter
  - Impact: Reduces non-GPU bottleneck before proceeding to Phase 2 DNABERT optimization

- **Phase 2.1 inserted after Phase 2** (2026-01-23): Parallel Embedding Merge (URGENT)
  - Reason: Embedding merge step is a bottleneck after GPU-optimized extraction
  - Scope: Add multi-processing/multi-threading to embedding merge operations
  - Impact: Eliminates post-extraction bottleneck before checkpoint robustness work

### Blockers/Concerns

**Phase 1 Critical:**
- ✓ CUDA context initialization: Resolved via spawn context and deferred GPU initialization in workers (01-01)
- Batch size variability can fragment memory even with available VRAM — implement sequence sorting and expandable segments from start
- Checkpoint corruption from partial writes — use atomic temp-then-rename pattern

**Phase 4 Research:**
- FlashAttention-2 compatibility with fair-esm library (ESM-2) needs verification during planning — DNABERT-S integration is straightforward

## Session Continuity

Last session: 2026-01-23 17:48 UTC
Stopped at: Completed 02.1-02-PLAN.md execution (Pipeline Integration)
Resume file: None

## Phase 2 Complete

**DNABERT-S Optimization - Complete**

All 5 plans executed successfully:
- 02-01: BaseEmbeddingWorker foundation and parallel processing
- 02-02: BF16 optimization and performance validation
- 02-03: Pipeline integration and CLI support
- 02-04: Batch size profiling utilities
- 02-05: Integration tests and documentation

**Key achievements:**
- Multi-GPU DNABERT-S processing with 3-4x speedup
- BF16 mixed precision on Ampere+ GPUs
- Token-based dynamic batching
- Auto-splitting for load balancing
- Comprehensive integration tests
- User-facing optimization guide

**Phase duration:** 19.5 minutes (5 plans)
**Average per plan:** 3.9 minutes

## Phase 2.1 Complete

**Parallel Embedding Merge - Complete**

All 2 plans executed successfully:
- 02.1-01: Parallel merge worker functions
- 02.1-02: Pipeline integration and CLI control

**Key achievements:**
- CPU multiprocessing for embedding merge with 6-7x expected speedup
- Auto-detection of CPU count for zero-config optimization
- User control via --merge-threads CLI parameter
- Sequential fallback for single file or single core
- Checkpoint support for resume capability
- Atomic writes for corruption prevention

**Phase duration:** 6.2 minutes (2 plans)
**Average per plan:** 3.1 minutes

Ready to proceed to next phase when defined.
