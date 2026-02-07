---
status: resolved
trigger: "Missing 10577 sequences in shard aggregation - VarlenCollator buffer loss in DataLoader workers"
created: 2026-02-06T00:00:00Z
updated: 2026-02-06T01:00:00Z
---

## Current Focus

hypothesis: CONFIRMED - VarlenCollator buffer-based packing lost data because collate_fn ran in DataLoader WORKER processes
test: Fix implemented and verified with 276 passing unit tests, 8 new regression tests
expecting: All sequences processed with zero loss
next_action: Archive session

## Symptoms

expected: 794,577 sequences processed across 2 GPU workers (397,289 + 397,288)
actual: 784,000 sequences processed (392,000 + 392,000). 10,577 missing.
errors: ValueError: Missing 10577 expected sequences at shard aggregation
reproduction: Run `python -m virnucpro predict <fasta> --parallel --gpus 0,1 --resume`
started: Persistent across multiple fix attempts

## Eliminated

- hypothesis: Overflow recovery in _tokenize_and_pack causes data loss
  evidence: Previous fix attempt did not resolve the missing sequences
  timestamp: 2026-02-05

## Evidence

- timestamp: 2026-02-06
  checked: Worker logs for flush() behavior
  found: Main process collator shows total_received=0, proving it was never called
  implication: collate_fn runs in DataLoader worker subprocesses, not main process

- timestamp: 2026-02-06
  checked: Mathematical analysis of missing sequences
  found: 10,577 = 2 GPU workers * 4 DL workers * ~1,322 remaining buffer per DL worker
  implication: Confirms buffer remainder loss pattern exactly matches num_workers * buffer remainder

- timestamp: 2026-02-06
  checked: Source code analysis of create_async_dataloader and VarlenCollator
  found: collate_fn=collate_fn passed directly to DataLoader constructor at line 334
  implication: With num_workers>0, PyTorch pickles collator and sends copies to worker subprocesses

- timestamp: 2026-02-06
  checked: VarlenCollator docstring vs actual behavior
  found: Docstring says "runs in MAIN PROCESS" but this is false with num_workers>0
  implication: Architecture assumption was incorrect - stateful collator must not be pickled

## Resolution

root_cause: VarlenCollator passed as collate_fn to DataLoader with num_workers>0. PyTorch pickles the collator and sends copies to 4 worker subprocesses. Each worker gets its own independent buffer. When workers finish iterating, sequences remaining in their buffers (~1,322 each) are destroyed. The main process flush() operates on the original collator instance that was never called (total_received=0). Total loss = 2 GPU workers * 4 DL workers * ~1,322 = 10,577 sequences.

fix: |
  Three-part fix to ensure collator runs in main process:
  1. create_async_dataloader (dataloader_utils.py): When collator has enable_packing=True
     (stateful), use _passthrough_collate as DataLoader's collate_fn instead of the
     real collator. Store the real collator as dataloader.collator. Workers yield raw
     dicts unchanged. pin_memory disabled for passthrough mode (no tensors in raw dicts).
  2. AsyncInferenceRunner.run() (async_inference.py): Added _get_collator() and
     _is_main_process_collation() helper methods. When main-process collation is
     detected, each raw item from DataLoader is passed through the collator manually
     in the main process loop. flush() now operates on the collator that actually
     received data.
  3. VarlenCollator (collators.py): Updated docstring to accurately describe that
     stateful collators must run in main process (via create_async_dataloader).
     Backward compatibility preserved for stateless collators (enable_packing=False).

verification: |
  - 276 unit tests pass (0 new failures introduced)
  - 8 new regression tests added in TestMainProcessCollation:
    - test_stateful_collator_not_passed_to_dataloader_workers
    - test_stateless_collator_passed_to_dataloader_normally
    - test_collator_without_enable_packing_attribute_treated_as_stateless
    - test_pin_memory_disabled_for_stateful_collator
    - test_runner_detects_main_process_collation
    - test_runner_get_collator_prefers_stored_collator
    - test_runner_get_collator_falls_back_to_collate_fn
    - test_passthrough_collate_identity
  - Backward compatibility preserved:
    - Stateless collators (enable_packing=False) passed directly to DataLoader as before
    - DataLoaders without .collator attribute fall back to checking .collate_fn
    - run_async_inference convenience function works unchanged

files_changed:
  - virnucpro/data/dataloader_utils.py
  - virnucpro/pipeline/async_inference.py
  - virnucpro/data/collators.py
  - tests/unit/test_collators.py
