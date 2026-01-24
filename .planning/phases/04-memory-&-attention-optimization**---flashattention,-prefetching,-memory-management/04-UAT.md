---
status: complete
phase: 04-memory-attention
source: [04-01-SUMMARY.md, 04-02-SUMMARY.md, 04-03-SUMMARY.md, 04-04-SUMMARY.md]
started: 2026-01-24T01:22:00Z
updated: 2026-01-24T01:27:00Z
---

## Current Test

[testing complete]

## Tests

### 1. CLI memory optimization flags visible
expected: Running `python -m virnucpro predict --help` shows 5 new memory optimization flags (--dataloader-workers, --pin-memory, --expandable-segments, --cache-clear-interval, --cuda-streams)
result: pass

### 2. FlashAttention-2 auto-detection logging
expected: When running prediction with CUDA available, logs clearly show "FlashAttention-2: enabled" on Ampere+ GPUs or "Using standard attention" on older GPUs
result: issue
reported: "I ran into this error right at the end of the DNABERT step: 2026-01-23 20:56:51 - virnucpro.pipeline.prediction - ERROR - Pipeline failed\nTraceback (most recent call last):\n  File \"/home/unix/carze/projects/virnucpro-broad/virnucpro/pipeline/prediction.py\", line 459, in run_prediction\n    if memory_manager and memory_manager.should_clear_cache():\nTypeError: should_clear_cache() missing 1 required positional argument: 'batch_num'"
severity: blocker

### 3. Memory optimization with default settings
expected: Running `virnucpro predict` without memory flags works unchanged (backward compatible), auto-detects optimal settings
result: issue
reported: "Again failed with the same error:\n\n2026-01-23 21:00:09 - virnucpro.pipeline.prediction - ERROR - Pipeline failed\nTraceback (most recent call last):\n  File \"/home/unix/carze/projects/virnucpro-broad/virnucpro/pipeline/prediction.py\", line 459, in run_prediction\n    if memory_manager and memory_manager.should_clear_cache():\nTypeError: should_clear_cache() missing 1 required positional argument: 'batch_num'"
severity: blocker

### 4. DataLoader worker count auto-detection
expected: Pipeline auto-detects CPU cores and GPUs, logs message like "DataLoader workers: 4 (auto-detected from 16 CPUs / 4 GPUs)"
result: skipped
reason: Pipeline crashes with should_clear_cache() error before reaching this stage

### 5. CUDA streams enabled by default
expected: When running with GPUs, logs show "CUDA streams: enabled" unless --no-cuda-streams is used
result: skipped
reason: Pipeline crashes with should_clear_cache() error before reaching this stage

### 6. Expandable segments configuration
expected: Using --expandable-segments flag logs "Expandable CUDA segments: enabled" and sets PYTORCH_CUDA_ALLOC_CONF environment variable
result: skipped
reason: Pipeline crashes with should_clear_cache() error before reaching this stage

### 7. OOM error handling with diagnostics
expected: On OOM error, pipeline exits with code 4, logs memory diagnostics (allocated/reserved/free), and suggests: "Try reducing batch size, enabling --expandable-segments, or increasing --cache-clear-interval"
result: skipped
reason: Pipeline crashes with should_clear_cache() error before reaching this stage

### 8. Memory tracking at stage boundaries
expected: With verbose logging enabled, pipeline logs memory stats after major stages: "Memory after DNABERT-S: X.XGB allocated, Y.YGB reserved, Z.ZGB free"
result: skipped
reason: Pipeline crashes with should_clear_cache() error before reaching this stage

## Summary

total: 8
passed: 1
issues: 2
pending: 0
skipped: 5

## Gaps

- truth: "Pipeline completes DNABERT-S step without crashing"
  status: failed
  reason: "User reported: I ran into this error right at the end of the DNABERT step: TypeError: should_clear_cache() missing 1 required positional argument: 'batch_num'"
  severity: blocker
  test: 2
  root_cause: ""
  artifacts: []
  missing: []
  debug_session: ""
