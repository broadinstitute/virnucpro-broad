---
status: complete
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
  root_cause: ""
  artifacts: []
  missing: []
  debug_session: ""

- truth: "On Ampere+ GPUs (A100, RTX 3090, etc.), BF16 mixed precision is automatically enabled for ESM-2 inference (logs show BF16 enabled and larger batch size 3072 vs 2048)"
  status: failed
  reason: "User reported: Running on a 4090 I don't see BF16 mixed precision and see batch size at 256"
  severity: major
  test: 4
  root_cause: ""
  artifacts: []
  missing: []
  debug_session: ""

- truth: "During multi-GPU processing, live progress dashboard shows per-GPU progress bars (in TTY environments) or periodic logging (in non-TTY environments)"
  status: failed
  reason: "User reported: During DNABERT step no progress bars are shown. Both GPUs are being used but utilization is not consistent."
  severity: major
  test: 8
  root_cause: ""
  artifacts: []
  missing: []
  debug_session: ""
