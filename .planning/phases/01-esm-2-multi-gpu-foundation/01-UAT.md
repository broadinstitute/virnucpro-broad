---
status: diagnosed
phase: 01-esm-2-multi-gpu-foundation
source:
  - 01-01-SUMMARY.md
  - 01-02-SUMMARY.md
  - 01-03-SUMMARY.md
  - 01-04-SUMMARY.md
started: 2026-01-23T00:31:00Z
updated: 2026-01-23T00:42:00Z
---

## Current Test

[testing complete]

## Tests

### 1. Multi-GPU Auto-Detection
expected: Running virnucpro predict on a multi-GPU system without specifying GPU flags automatically uses all available GPUs for ESM-2 extraction (verifiable in logs showing GPU worker initialization for each GPU).
result: issue
reported: "I ran on a 2x GPU system (this system) without specifying GPU and only one GPU is being used"
severity: major

### 2. Single-GPU Fallback
expected: Running virnucpro predict on a single-GPU system works without code changes, processing continues normally without parallel mode (verifiable in logs showing single-GPU mode).
result: pass

### 3. GPU Selection via CLI
expected: Using --gpus flag (e.g., --gpus 0,2,3) restricts processing to specified GPUs only, and parallel mode is auto-enabled (logs show only specified GPUs being used).
result: pass

### 4. BF16 Auto-Optimization
expected: On Ampere+ GPUs (A100, RTX 3090, etc.), BF16 mixed precision is automatically enabled for ESM-2 inference (logs show "BF16 enabled" and larger batch size 3072 vs 2048).
result: issue
reported: "Running on a 4090 I don't see BF16 mixed precision and see batch size at 256"
severity: major

### 5. ESM Batch Size Tuning
expected: Using --esm-batch-size flag adjusts tokens-per-batch for ESM-2 processing (not confused with prediction batch size), allowing manual performance tuning.
result: skipped

### 6. Failed File Logging
expected: When some files fail during ESM-2 processing, pipeline completes successfully for valid files and logs failures to failed_files.txt with format {file_path}|ESM-2|{error_message}.
result: skipped

### 7. Multi-GPU Output Equivalence
expected: Processing the same dataset with single GPU vs multiple GPUs produces identical prediction results (within floating-point tolerance), confirming correctness.
result: pass

### 8. GPU Progress Visibility
expected: During multi-GPU processing, live progress dashboard shows per-GPU progress bars (in TTY environments) or periodic logging (in non-TTY environments like CI/CD).
result: issue
reported: "During DNABERT step no progress bars are shown. Both GPUs are being used but utilization is not consistent."
severity: major

### 9. Memory Monitoring
expected: GPU memory usage is monitored during processing, with logs showing memory stats and warnings if memory pressure is detected.
result: skipped

### 10. Round-Robin Work Distribution
expected: Files are distributed evenly across GPU workers using round-robin assignment (verifiable in logs showing file count per worker, difference ≤1 file).
result: skipped

## Summary

total: 10
passed: 3
issues: 3
pending: 0
skipped: 4

## Gaps

- truth: "Running virnucpro predict on a multi-GPU system without specifying GPU flags automatically uses all available GPUs for ESM-2 extraction"
  status: failed
  reason: "User reported: I ran on a 2x GPU system (this system) without specifying GPU and only one GPU is being used"
  severity: major
  test: 1
  root_cause: "The --parallel flag defaults to False, preventing automatic multi-GPU detection. GPU detection logic only executes when parallel=True. Auto-enable only triggers if --gpus contains multiple GPU IDs."
  artifacts:
    - path: "virnucpro/cli/predict.py"
      issue: "Lines 54-60: --parallel flag is opt-in (default False); Lines 104-112: only auto-enable if --gpus specifies multiple GPUs"
    - path: "virnucpro/pipeline/prediction.py"
      issue: "Lines 222-233 (DNABERT) and 326-330 (ESM-2) require parallel=True to enable multi-GPU mode"
  missing:
    - "Auto-detect available GPUs and set parallel=True when multiple GPUs detected and --gpus not specified"
  debug_session: ".planning/debug/multi-gpu-auto-detection-failure.md"

- truth: "On Ampere+ GPUs (A100, RTX 3090, etc.), BF16 mixed precision is automatically enabled for ESM-2 inference (logs show BF16 enabled and larger batch size 3072 vs 2048)"
  status: failed
  reason: "User reported: Running on a 4090 I don't see BF16 mixed precision and see batch size at 256"
  severity: major
  test: 4
  root_cause: "BF16 IS enabled but log messages are invisible. Spawn context creates workers without inheriting logging config. Workers use module-level loggers defaulting to WARNING level, silently discarding INFO-level BF16 messages."
  artifacts:
    - path: "virnucpro/pipeline/parallel_esm.py"
      issue: "Worker function lacks logging configuration initialization"
    - path: "virnucpro/pipeline/features.py"
      issue: "BF16 log messages at INFO level invisible in workers"
    - path: "virnucpro/core/logging_setup.py"
      issue: "Logging setup not designed for multiprocessing workers"
    - path: "virnucpro/pipeline/work_queue.py"
      issue: "Spawns workers without passing logging config"
  missing:
    - "Add logging configuration to worker initialization (pass level/format as kwargs)"
    - "Add main process log showing GPU capabilities and BF16 status before spawning workers"
  debug_session: ".planning/debug/bf16-not-enabled-rtx4090.md"

- truth: "During multi-GPU processing, live progress dashboard shows per-GPU progress bars (in TTY environments) or periodic logging (in non-TTY environments)"
  status: failed
  reason: "User reported: During DNABERT step no progress bars are shown. Both GPUs are being used but utilization is not consistent."
  severity: major
  test: 8
  root_cause: "Dashboard infrastructure exists but not integrated into workers. Workers use Pool.starmap with no progress callback mechanism. Round-robin assignment distributes files by count not size, causing uneven GPU utilization."
  artifacts:
    - path: "virnucpro/pipeline/prediction.py"
      issue: "Missing dashboard integration in ESM-2 (lines 329-351) and DNABERT (lines 234-266) multi-GPU paths"
    - path: "virnucpro/pipeline/work_queue.py"
      issue: "BatchQueueManager lacks progress reporting infrastructure (no callback, no progress queue)"
    - path: "virnucpro/pipeline/parallel.py"
      issue: "process_dnabert_files_worker has no progress update mechanism"
    - path: "virnucpro/pipeline/parallel_esm.py"
      issue: "process_esm_files_worker has no progress update mechanism; assign_files_round_robin distributes by count not size"
  missing:
    - "Add multiprocessing.Queue for progress events from workers"
    - "Modify workers to send progress updates after each file"
    - "Add monitoring thread to consume queue and update dashboard"
    - "Modify assign_files_round_robin to sort by sequence count and use bin-packing for balanced load"
  debug_session: ".planning/debug/multi-gpu-dnabert-progress-missing.md"
