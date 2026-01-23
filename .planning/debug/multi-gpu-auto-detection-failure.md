---
status: diagnosed
trigger: "Expected: Running virnucpro predict on a multi-GPU system without specifying GPU flags automatically uses all available GPUs for ESM-2 extraction. Actual: User ran on a 2x GPU system without specifying GPU flags and only one GPU is being used."
created: 2026-01-22T00:00:00Z
updated: 2026-01-22T00:03:30Z
---

## Current Focus

hypothesis: CONFIRMED - parallel flag defaults to False, blocking multi-GPU auto-detection
test: Checking if both DNABERT and ESM-2 have same issue
expecting: Both require parallel=True flag to use multiple GPUs
next_action: Check DNABERT logic at prediction.py lines 222-233

## Symptoms

expected: On 2x GPU system without GPU flags, both GPUs used for ESM-2 extraction
actual: Only 1 GPU used
errors: None reported
reproduction: Run `virnucpro predict` on 2x RTX 4090 system without --gpus or --parallel flags
started: Current behavior (Phase 01 multi-GPU testing)

## Eliminated

## Evidence

- timestamp: 2026-01-22T00:01:00Z
  checked: CLI predict.py lines 54-60, 104-112
  found: `--parallel` is a flag (default False), `--gpus` is optional string (default None)
  implication: Parallel mode is opt-in, not automatic. Even with multiple GPUs available, user must explicitly pass --parallel flag.

- timestamp: 2026-01-22T00:01:30Z
  checked: CLI predict.py lines 109-112
  found: Auto-enables parallel only if `--gpus` specifies multiple GPUs (contains comma)
  implication: Without --gpus flag, parallel stays False even on multi-GPU systems

- timestamp: 2026-01-22T00:02:00Z
  checked: prediction.py lines 326-330
  found: `use_parallel = num_gpus > 1 and len(protein_files) > 1 and parallel`
  implication: ESM-2 uses parallel only if: (1) multiple GPUs detected, (2) multiple files, AND (3) parallel flag is True

- timestamp: 2026-01-22T00:03:00Z
  checked: prediction.py lines 222-233
  found: DNABERT has same logic - `use_parallel = False; if parallel: ... if len(available_gpus) > 1: use_parallel = True`
  implication: Both DNABERT and ESM-2 require explicit --parallel flag. Auto-detection happens ONLY if parallel flag is set.

## Resolution

root_cause: The `--parallel` flag defaults to False in CLI, preventing automatic multi-GPU detection. Even when multiple GPUs are available, the pipeline requires explicit `--parallel` flag to enable multi-GPU mode. Auto-detection logic exists in prediction.py (detect_cuda_devices), but is only executed when parallel=True.
fix: Change --parallel flag default from False to auto-detect based on available GPUs, OR auto-enable parallel when multiple GPUs detected without user specification
verification: Run on 2x GPU system without --parallel flag and verify both GPUs are utilized
files_changed:
  - virnucpro/cli/predict.py: CLI argument parsing and parallel flag handling
  - virnucpro/pipeline/prediction.py: Parallel mode logic for DNABERT and ESM-2
