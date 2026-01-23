---
status: diagnosed
trigger: "BF16 mixed precision not enabled on RTX 4090 (Ampere+ GPU), batch size 256 instead of expected 2048/3072"
created: 2026-01-22T00:00:00Z
updated: 2026-01-22T00:08:30Z
symptoms_prefilled: true
goal: find_root_cause_only
---

## Current Focus

hypothesis: ROOT CAUSE FOUND - Multiprocessing spawn workers have no logging configuration, BF16 logs are invisible (but feature IS working). Also need to clarify batch_size vs toks_per_batch confusion.
test: final verification of findings
expecting: BF16 works but logs lost, user confused about batch size metrics
next_action: write final root cause summary

## Symptoms

expected: RTX 4090 (Ampere+, compute 8.9) should auto-enable BF16 with logs showing "BF16 enabled" and batch size 3072 or 2048
actual: No BF16 enabled, batch size shows 256
errors: None reported
reproduction: Run ESM-2 inference on RTX 4090
started: Current issue (Test 4 of Phase 01)

## Eliminated

## Evidence

- timestamp: 2026-01-22T00:01:00Z
  checked: features.py lines 119-130
  found: BF16 detection logic exists - checks device capability >= 8 (Ampere+), logs "Using BF16 mixed precision", increases toks_per_batch from 2048 to 3072
  implication: BF16 logic is correctly implemented in extract_esm_features function

- timestamp: 2026-01-22T00:02:00Z
  checked: parallel_esm.py lines 70-120
  found: process_esm_files_worker calls extract_esm_features with toks_per_batch parameter (line 119), passes device correctly (line 118)
  implication: Worker function passes parameters correctly, should enable BF16 detection

- timestamp: 2026-01-22T00:03:00Z
  checked: prediction.py lines 310-380 (ESM-2 feature extraction)
  found: Multi-GPU path (lines 331-350) passes toks_per_batch to queue_manager.process_files() (line 345). Single-GPU path (lines 359-380) passes toks_per_batch to extract_esm_features() (line 373).
  implication: toks_per_batch is correctly passed in both parallel and single-GPU paths

- timestamp: 2026-01-22T00:04:00Z
  checked: features.py lines 119-130 BF16 logging
  found: Logger is 'virnucpro.features' (line 10), BF16 log message at line 125: "Using BF16 mixed precision for memory efficiency"
  implication: BF16 logging uses module-level logger that may need proper configuration to be visible

- timestamp: 2026-01-22T00:05:00Z
  checked: parallel_esm.py worker context
  found: Worker uses logger 'virnucpro.parallel_esm' (line 10), logs "Worker {device_id}: Initializing..." at line 106. extract_esm_features is called inside worker at line 115.
  implication: BF16 log message from features.py would only appear if virnucpro.features logger is configured in worker process

- timestamp: 2026-01-22T00:06:00Z
  checked: logging_setup.py and multiprocessing spawn behavior
  found: setup_logging() configures root 'virnucpro' logger (line 39) in MAIN process. Spawn context creates NEW Python interpreter for workers. Module-level loggers in worker are unconfigured (default WARNING level).
  implication: BF16 log at INFO level (features.py:125) is silently discarded in worker process

- timestamp: 2026-01-22T00:07:00Z
  checked: process_esm_files_worker in parallel_esm.py
  found: Worker function does NOT call setup_logging() - directly creates logger at module level (line 10). Worker logs at INFO level (line 106, 114, 123) will be lost.
  implication: ROOT CAUSE CONFIRMED - workers have no logging configuration, all INFO logs invisible

- timestamp: 2026-01-22T00:08:00Z
  checked: CLI batch size parameters (predict.py)
  found: Three different batch size parameters: --batch-size (DataLoader, default 256 from config), --dnabert-batch-size (DNABERT batching), --esm-batch-size (ESM-2 toks_per_batch)
  implication: User saw "batch size 256" from DataLoader config logging (line 168), NOT ESM-2 toks_per_batch. These are different parameters for different purposes.

## Resolution

root_cause: Multiprocessing spawn context creates worker processes without logging configuration. Workers use module-level loggers (e.g., virnucpro.features, virnucpro.parallel_esm) which inherit Python's default WARNING level. BF16 detection IS working on RTX 4090, but INFO-level log messages ("Using BF16 mixed precision", "Increased toks_per_batch to 3072") are silently discarded. User only sees main process logs, which don't show worker-internal behavior. Secondary issue: user confused "batch size 256" (DataLoader batching for final prediction) with ESM-2 toks_per_batch (default 2048, increased to 3072 with BF16).

fix:
verification:
files_changed: []
