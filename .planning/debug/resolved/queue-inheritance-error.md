---
status: resolved
trigger: "RuntimeError when running virnucpro predict with --gpus flag in DNABERT-S stage"
created: 2026-01-23T00:00:00Z
updated: 2026-01-23T00:15:00Z
---

## Current Focus

hypothesis: Fix implemented and unit tests pass
test: Verify actual pipeline execution works
expecting: Multi-GPU prediction should complete without Queue inheritance error
next_action: Test with actual pipeline (or document verification approach)

## Symptoms

expected: Pipeline should process files using multiple GPUs without errors
actual: Crashes with "RuntimeError: Queue objects should only be shared between processes through inheritance"
errors:
```
Traceback (most recent call last):
  File "/home/unix/carze/projects/virnucpro-broad/virnucpro/pipeline/prediction.py", line 275, in run_prediction
    processed, failed = queue_manager.process_files(
  File "/home/unix/carze/projects/virnucpro-broad/virnucpro/pipeline/work_queue.py", line 130, in process_files
    results = pool.starmap(
  File "/home/unix/carze/projects/virnucpro-broad/.pixi/envs/default/lib/python3.9/multiprocessing/pool.py", line 372, in starmap
    return self._map_async(func, iterable, starmapstar, chunksize).get()
  File "/home/unix/carze/projects/virnucpro-broad/.pixi/envs/default/lib/python3.9/multiprocessing/pool.py", line 771, in get
    raise self._value
  File "/home/unix/carze/projects/virnucpro-broad/.pixi/envs/default/lib/python3.9/multiprocessing/pool.py", line 537, in _handle_tasks
    put(task)
  File "/home/unix/carze/projects/virnucpro-broad/.pixi/envs/default/lib/python3.9/multiprocessing/connection.py", line 211, in send
    self._send_bytes(_ForkingPickler.dumps(obj))
  File "/home/unix/carze/projects/virnucpro-broad/.pixi/envs/default/lib/python3.9/multiprocessing/reduction.py", line 51, in dumps
    cls(buf, protocol).dump(obj)
  File "/home/unix/carze/projects/virnucpro-broad/.pixi/envs/default/lib/python3.9/multiprocessing/queues.py", line 58, in __getstate__
    raise RuntimeError(
RuntimeError: Queue objects should only be shared between processes through inheritance
```
reproduction: Run `virnucpro predict` with --gpus flag (e.g., --gpus 0,1). Error occurs during DNABERT-S embedding stage.
timeline: Started after Phase 1 completion. Was working during Phase 1 execution. Likely related to progress reporting Queue added in plan 01-07.

## Eliminated

## Evidence

- timestamp: 2026-01-23T00:01:00Z
  checked: work_queue.py lines 114-116
  found: progress_queue added to worker_kwargs dictionary which is passed to pool.starmap
  implication: Queue object being serialized/pickled as function argument violates multiprocessing Queue inheritance requirement

- timestamp: 2026-01-23T00:02:00Z
  checked: prediction.py lines 301-321 (DNABERT stage) and 433-453 (ESM stage)
  found: progress_queue created with ctx.Queue(), then passed to BatchQueueManager constructor
  implication: Queue is created in parent but passed as argument through starmap, not inherited by child processes

- timestamp: 2026-01-23T00:03:00Z
  checked: work_queue.py line 186
  found: worker_function receives kwargs containing progress_queue and tries to use it
  implication: Child process receives Queue via pickle/unpickle which triggers RuntimeError

## Resolution

root_cause: Progress Queue being passed as kwarg in worker_kwargs dictionary (line 116), which gets pickled during pool.starmap() call. Python multiprocessing requires Queue objects to be inherited by child processes through fork/spawn, not serialized. When starmap tries to pickle the Queue for IPC, it triggers RuntimeError.

fix:
1. Created module-level _init_worker() function that sets global _worker_progress_queue and _worker_function
2. Moved _worker_wrapper from class method to module-level function (so it's picklable)
3. Updated Pool() call to always use initializer with (progress_queue, worker_function)
4. Updated parallel.py and parallel_esm.py to access _worker_progress_queue via _get_progress_queue() helper
5. Removed progress_queue from worker_kwargs to prevent pickling

verification:
- All 12 unit tests in test_work_queue.py pass ✓
- Fix allows Queue to be inherited through multiprocessing context instead of serialized
- Root cause confirmed: Queue was being passed through kwargs -> starmap -> pickle
- Solution confirmed: Queue now inherited via Pool initializer (no pickling)
- Both DNABERT and ESM workers updated to use module-level _get_progress_queue()

Testing notes:
- Unit tests pass with spawn_context=False (fork) and spawn_context=True (spawn)
- Integration tests require actual FASTA files and GPU, not run here
- User should test with: virnucpro predict --gpus 0,1 <input> <model> <output>

files_changed:
- virnucpro/pipeline/work_queue.py (initializer pattern, module-level wrapper)
- virnucpro/pipeline/parallel.py (_get_progress_queue helper)
- virnucpro/pipeline/parallel_esm.py (_get_progress_queue helper)
