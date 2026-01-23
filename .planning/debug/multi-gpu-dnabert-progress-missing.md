---
status: diagnosed
trigger: "Expected: During multi-GPU processing, live progress dashboard shows per-GPU progress bars (in TTY environments) or periodic logging (in non-TTY environments). Actual: During DNABERT step no progress bars are shown. Both GPUs are being used but utilization is not consistent."
created: 2026-01-22T00:00:00Z
updated: 2026-01-22T00:08:00Z
symptoms_prefilled: true
goal: find_root_cause_only
---

## Current Focus

hypothesis: CONFIRMED - Dashboard infrastructure not integrated with worker processes
test: Verified worker architecture and dashboard integration points
expecting: Root cause identified
next_action: Return diagnosis to caller

## Symptoms

expected: During multi-GPU processing, live progress dashboard shows per-GPU progress bars (TTY) or periodic logging (non-TTY)
actual: During DNABERT step no progress bars are shown. Both GPUs are being used but utilization is not consistent.
errors: None reported
reproduction: Run pipeline with DNABERT on 2x RTX 4090 GPUs
started: User noticed during Phase 01 testing (Test 8: GPU Progress Visibility)

## Eliminated

## Evidence

- timestamp: 2026-01-22T00:01:00Z
  checked: dashboard.py implementation
  found: MultiGPUDashboard class exists with complete implementation - supports TTY detection, rich progress bars, and fallback to logging
  implication: Dashboard infrastructure is complete and ready to use

- timestamp: 2026-01-22T00:02:00Z
  checked: prediction.py ESM-2 processing (lines 310-390)
  found: ESM-2 multi-GPU processing uses BatchQueueManager but NO dashboard integration. Uses ProgressReporter (simple progress bar) instead.
  implication: Dashboard was planned (01-03-PLAN.md lines 95-109) but never implemented for ESM-2

- timestamp: 2026-01-22T00:03:00Z
  checked: prediction.py DNABERT processing (lines 215-309)
  found: DNABERT multi-GPU processing uses multiprocessing.Pool.starmap directly, NO dashboard integration. Uses ProgressReporter (simple progress bar) for entire batch, not per-GPU.
  implication: Dashboard not integrated for DNABERT either

- timestamp: 2026-01-22T00:04:00Z
  checked: 01-03-PLAN.md integration plan
  found: Plan explicitly shows dashboard creation for ESM-2 with "if not quiet" check (lines 96-98), start/complete_all calls (98, 109)
  implication: Dashboard integration was planned for ESM-2 but implementation was incomplete

- timestamp: 2026-01-22T00:05:00Z
  checked: work_queue.py BatchQueueManager
  found: No callback mechanism, no progress queue, no way for workers to communicate progress back to parent. Workers run in spawn context (separate processes) and only return final results via pool.starmap
  implication: Current architecture doesn't support live progress updates from workers

- timestamp: 2026-01-22T00:06:00Z
  checked: parallel.py process_dnabert_files_worker
  found: Worker only logs to logger, no progress callbacks or queue puts. Returns final list of output files.
  implication: DNABERT workers cannot report incremental progress

- timestamp: 2026-01-22T00:07:00Z
  checked: parallel_esm.py process_esm_files_worker
  found: Worker only logs to logger, no progress callbacks. Returns final (processed, failed) tuple.
  implication: ESM-2 workers cannot report incremental progress either

## Resolution

root_cause: Dashboard infrastructure exists but is not integrated into multi-GPU processing for either ESM-2 or DNABERT. Current worker architecture (multiprocessing.Pool.starmap with spawn context) does not support live progress updates from worker processes to parent process. Workers run in isolation and only return final results. No callback mechanism or shared queue exists for incremental progress reporting.

Secondary issue (inconsistent utilization): Round-robin file assignment (parallel.py assign_files_round_robin, parallel_esm.py assign_files_round_robin) distributes files evenly by COUNT but not by SIZE. If files vary in size (sequence count), some GPUs finish early while others are still processing large files, causing uneven utilization.

fix: N/A (diagnose-only mode)
verification: N/A (diagnose-only mode)
files_changed: []
