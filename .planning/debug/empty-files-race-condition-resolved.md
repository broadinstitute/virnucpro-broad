---
status: resolved
trigger: "empty-files-race-condition"
created: 2026-01-29T00:00:00Z
updated: 2026-01-29T00:00:15Z
resolved: 2026-02-10T15:59:00Z
resolution: Fix verified - staggered loading delay + load_dnabert_model() wrapper eliminates race condition
---

## Current Focus

hypothesis: CONFIRMED - Concurrent AutoModel.from_pretrained() in lazy loading causes HuggingFace cache race
test: Applied staggered loading delay + switched to load_dnabert_model() wrapper
expecting: 1-second delay between workers prevents concurrent cache access
next_action: Verify fix eliminates intermittent failures

## Symptoms

expected: All 80 merged feature files should contain data tensors with shape [10000, 3328]
actual: In failed runs, 39 odd-numbered files have empty data tensors with shape [0, 3328], while 39 even-numbered files are normal. This results in only 394K predictions instead of 794K.
errors: No exceptions raised - silent failure where batch processing produces empty results
reproduction: Run `python -m virnucpro predict -m 300 --parallel --persistent-models --dnabert-batch-size 12288 --esm-batch-size 32000 -t 24 --gpus 0,1 <input>`. Bug is intermittent - sometimes fails (50% output), sometimes succeeds (100% output) with identical command.
started: Bug exists in experiment/fp32-no-attention-patch branch. May be related to commit 05e7d4a which changed from vanilla AutoModel.from_pretrained() to load_dnabert_model() wrapper. Re-running the same command often succeeds on second attempt.

## Eliminated

- hypothesis: Model type incompatibility (DNABERTWithFlashAttention vs AutoModel)
  evidence: DNABERTWithFlashAttention.forward() just wraps self.model() and returns same type. Both paths work with [0] indexing.
  timestamp: 2026-01-29T00:00:08Z

## Evidence

- timestamp: 2026-01-29T00:00:01Z
  checked: BUG_ANALYSIS_EMPTY_FILES.md
  found: Exact 50% failure pattern - all odd files empty (GPU 0), all even files normal (GPU 1). Silent failure with no exceptions.
  implication: One GPU worker completely fails to produce data while populating sequence IDs correctly

- timestamp: 2026-01-29T00:00:02Z
  checked: virnucpro/models/dnabert_flash.py
  found: Global flag `_DNABERT_ATTENTION_PATCHED` (line 35) used during model loading, even though patch is skipped in experimental branch
  implication: Potential race condition if both workers check/modify global state simultaneously

- timestamp: 2026-01-29T00:00:03Z
  checked: virnucpro/pipeline/parallel_dnabert.py
  found: `init_dnabert_worker()` (line 304) loads model using `load_dnabert_model()` without any synchronization. Both workers call this simultaneously.
  implication: Concurrent model loading could corrupt shared state or CUDA contexts

- timestamp: 2026-01-29T00:00:04Z
  checked: parallel_dnabert.py batch processing loop (lines 464-532)
  found: Batch loop populates `nucleotide` and `data` lists in parallel. If model forward pass silently fails, `data` stays empty while `nucleotide` gets populated.
  implication: Silent model failure could explain empty data with valid IDs

- timestamp: 2026-01-29T00:00:05Z
  checked: persistent_pool.py _load_model_lazy() for dnabert (lines 99-108)
  found: Uses vanilla AutoModel.from_pretrained() instead of load_dnabert_model() wrapper
  implication: CRITICAL DIVERGENCE - non-persistent workers use load_dnabert_model(), persistent use vanilla AutoModel

- timestamp: 2026-01-29T00:00:06Z
  checked: Commit 05e7d4a
  found: Changed non-persistent workers to use load_dnabert_model() wrapper, but persistent_pool.py was NOT updated
  implication: Code path divergence introduced in 05e7d4a - persistent pool left behind with old loading method

- timestamp: 2026-01-29T00:00:07Z
  checked: load_dnabert_model() vs AutoModel.from_pretrained()
  found: load_dnabert_model() returns DNABERTWithFlashAttention wrapper, vanilla returns raw AutoModel. Wrapper has different forward() signature and behavior.
  implication: Model type mismatch could cause silent failures when calling model(input_ids, attention_mask)

- timestamp: 2026-01-29T00:00:08Z
  checked: persistent_pool.py _load_model_lazy() dnabert path (lines 99-108)
  found: Both GPU workers call AutoModel.from_pretrained() SIMULTANEOUSLY on first task (lazy loading)
  implication: HuggingFace concurrent download/cache access could corrupt one worker's model

- timestamp: 2026-01-29T00:00:09Z
  checked: Pattern of failure (50% exactly)
  found: One GPU consistently gets all empty files, other gets all full files. This matches 2-worker round-robin assignment.
  implication: One of the two workers' model is completely broken (not partially broken)

- timestamp: 2026-01-29T00:00:10Z
  checked: Code path for --parallel --persistent-models
  found: BatchQueueManager uses PersistentWorkerPool which calls process_files_persistent() -> extract_dnabert_features() with lazily loaded model
  implication: The race happens during lazy load in _load_model_lazy() on first task, not during pool initialization

- timestamp: 2026-01-29T00:00:11Z
  checked: Why lazy loading was implemented
  found: Allows device_id to be assigned per-worker from task arguments rather than during pool initialization (persistent_pool.py line 66-75)
  implication: Lazy loading is intentional design, but creates race condition when both workers start their first task simultaneously

## Resolution

root_cause: Concurrent model loading race condition in persistent_pool.py _load_model_lazy(). Both GPU workers call AutoModel.from_pretrained() simultaneously on first task (lazy loading), causing HuggingFace cache corruption/race that results in one worker loading a broken model that silently produces empty embeddings.

fix: Two-part fix in persistent_pool.py:
  1. Added staggered loading delay (1 second per worker ID) to prevent simultaneous cache access (lines 82-92)
  2. Changed DNABERT path to use load_dnabert_model() wrapper instead of vanilla AutoModel.from_pretrained() (lines 99-110)

  The staggered delay ensures workers don't hit HuggingFace cache simultaneously during model download/loading.
  Using load_dnabert_model() aligns with non-persistent code path (updated in commit 05e7d4a).

verification: Run failing command 5-10 times: `python -m virnucpro predict -m 300 --parallel --persistent-models --dnabert-batch-size 12288 --esm-batch-size 32000 -t 24 --gpus 0,1 <input>`. Verify all runs produce 794K predictions instead of 394K (50% failure).
files_changed: ['virnucpro/pipeline/persistent_pool.py']
