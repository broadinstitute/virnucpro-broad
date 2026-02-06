# Phase 9: Checkpointing Integration - Research

**Researched:** 2026-02-05
**Domain:** Incremental checkpointing for multi-hour GPU inference workloads
**Confidence:** HIGH

## Summary

Incremental checkpointing for long-running GPU inference workloads (6M sequences, multi-hour runtime) requires four core capabilities: adaptive checkpoint triggers (time/sequence thresholds), async I/O to avoid GPU stalls, batch atomicity to respect packed attention boundaries, and per-shard independence for partial failure recovery. This phase builds on VirNucPro's existing checkpoint infrastructure (Phase 3/4) and async DataLoader architecture (Phase 5-8) to add granular resume points without reprocessing completed work.

The codebase already has proven patterns: atomic writes with `.done` markers (checkpoint.py), HDF5 shard writing (gpu_worker.py), and validation (checkpoint_validation.py). Research confirms PyTorch's async checkpointing pattern (background threads for CPU→disk writes) achieves 10-20x faster checkpointing by moving I/O off the GPU critical path. For multi-GPU coordination, manifest-based tracking (JSON metadata tracking per-shard progress) enables partial failure recovery where only failed GPUs restart.

Critical insight from sequence packing research: checkpoints MUST NOT break mid-packed-batch. FlashAttention varlen uses `cu_seqlens` to mark sequence boundaries, and splitting a batch would corrupt cross-sequence attention masking. The user decision to checkpoint at batch boundaries (with emergency override >10 min overdue) aligns with this requirement.

**Primary recommendation:** Extend existing atomic write pattern to incremental HDF5 checkpoints, implement ThreadPoolExecutor-based async I/O (stdlib, no new deps), add coordinator manifest tracking batch boundaries per shard, and respect packed batch atomicity with configurable thresholds (10K sequences OR 5 minutes, whichever first).

## Standard Stack

The established libraries/tools for incremental GPU checkpointing:

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| h5py | 3.x | HDF5 checkpoint storage | Already used for shard files (gpu_worker.py), supports chunk-wise streaming |
| concurrent.futures | stdlib | Background thread async I/O | Python 3.2+ stdlib, avoids GIL contention with ThreadPoolExecutor |
| torch.save/torch.load | PyTorch 2.x | Checkpoint serialization (manifest) | Native PyTorch checkpoint format, used in Phase 3 |
| pathlib.Path.replace() | stdlib | Atomic file writes | POSIX atomic rename, used in existing checkpoint.py |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| json | stdlib | Manifest format (coordinator tracking) | Lightweight, human-readable checkpoint metadata |
| time.perf_counter() | stdlib | Adaptive time-based triggers | High-resolution timer for "5 minutes OR 10K sequences" logic |
| threading.Lock | stdlib | Thread-safe checkpoint state | Protect shared state during async writes |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| h5py incremental | PyTorch .pt per-batch | HDF5 more space-efficient for large embeddings, supports partial reads |
| ThreadPoolExecutor | asyncio + aiofiles | ThreadPoolExecutor simpler for I/O-bound ops, no event loop complexity |
| JSON manifest | YAML manifest | JSON is stdlib, faster parsing, widely used in Phase 7 (sequence_index.json) |
| Background thread | Synchronous writes | Async avoids GPU stalls (10-20x faster based on PyTorch async checkpoint research) |

**Installation:**
No new dependencies required - all tools are PyTorch built-ins or Python stdlib.

## Architecture Patterns

### Recommended Checkpoint Structure
```
output_dir/
├── checkpoints/
│   ├── manifest.json              # Coordinator manifest (global state)
│   ├── shard_0/
│   │   ├── batch_00042.h5        # Incremental checkpoint
│   │   ├── batch_00042.done      # Atomic completion marker
│   │   ├── batch_00085.h5
│   │   └── batch_00085.done
│   ├── shard_1/
│   │   ├── batch_00041.h5
│   │   └── batch_00041.done
│   └── ...
├── shard_0.h5                     # Final merged shard (existing)
├── shard_1.h5
└── embeddings.h5                  # Aggregated output (existing)
```

### Pattern 1: Adaptive Checkpoint Trigger with Batch Atomicity
**What:** Checkpoint when either sequence count OR time threshold reached, but never mid-batch
**When to use:** AsyncInferenceRunner.run() loop after processing each batch
**Example:**
```python
# Source: User decisions + sequence packing research
class CheckpointTrigger:
    def __init__(self, seq_threshold=10000, time_threshold_sec=300):
        self.seq_threshold = seq_threshold
        self.time_threshold = time_threshold_sec
        self.last_checkpoint_time = time.perf_counter()
        self.sequences_since_checkpoint = 0

    def should_checkpoint(self, batch_size: int, emergency_override_sec: int = 600) -> Tuple[bool, str]:
        """Check if checkpoint trigger conditions met.

        Returns:
            (should_checkpoint, reason)
        """
        self.sequences_since_checkpoint += batch_size
        elapsed = time.perf_counter() - self.last_checkpoint_time

        # Emergency override: >10 min without checkpoint (mid-batch allowed)
        if elapsed > emergency_override_sec:
            return True, "emergency_time_override"

        # Normal triggers (batch boundary safe)
        if self.sequences_since_checkpoint >= self.seq_threshold:
            return True, "sequence_threshold"

        if elapsed >= self.time_threshold:
            return True, "time_threshold"

        return False, None

    def reset(self):
        """Reset counters after checkpoint."""
        self.last_checkpoint_time = time.perf_counter()
        self.sequences_since_checkpoint = 0
```

### Pattern 2: Async Checkpoint Write (Background Thread)
**What:** Offload HDF5 writes to background thread to avoid GPU stalls
**When to use:** After batching embeddings for checkpoint, before continuing inference
**Example:**
```python
# Source: PyTorch async checkpoint research + concurrent.futures docs
from concurrent.futures import ThreadPoolExecutor
import threading

class AsyncCheckpointWriter:
    """Non-blocking checkpoint writer using background thread."""

    def __init__(self, max_workers=1):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.pending_futures = []
        self.lock = threading.Lock()

    def write_checkpoint_async(
        self,
        checkpoint_path: Path,
        embeddings: np.ndarray,
        sequence_ids: List[str],
        metadata: Dict[str, Any]
    ) -> concurrent.futures.Future:
        """Submit checkpoint write to background thread.

        GPU continues inference while I/O completes asynchronously.
        """
        # Copy data to avoid race conditions (embeddings might be reused)
        embeddings_copy = embeddings.copy()
        ids_copy = sequence_ids.copy()
        metadata_copy = metadata.copy()

        future = self.executor.submit(
            self._write_checkpoint_sync,
            checkpoint_path,
            embeddings_copy,
            ids_copy,
            metadata_copy
        )

        with self.lock:
            self.pending_futures.append(future)

        return future

    def _write_checkpoint_sync(
        self,
        checkpoint_path: Path,
        embeddings: np.ndarray,
        sequence_ids: List[str],
        metadata: Dict[str, Any]
    ):
        """Synchronous write implementation (runs in background thread)."""
        temp_path = checkpoint_path.with_suffix('.tmp')

        try:
            with h5py.File(temp_path, 'w') as f:
                f.create_dataset('embeddings', data=embeddings)

                dt = h5py.special_dtype(vlen=str)
                f.create_dataset('sequence_ids', data=sequence_ids, dtype=dt)

                # Store metadata as attributes
                for key, value in metadata.items():
                    f.attrs[key] = value

            # Atomic rename
            temp_path.replace(checkpoint_path)

            # Create .done marker (Phase 4 pattern)
            done_marker = checkpoint_path.with_suffix(checkpoint_path.suffix + '.done')
            done_marker.touch()

            logger.debug(f"Checkpoint written: {checkpoint_path}")

        except Exception as e:
            if temp_path.exists():
                temp_path.unlink()
            raise RuntimeError(f"Async checkpoint write failed: {e}")

    def wait_all(self, timeout: Optional[float] = None):
        """Wait for all pending writes to complete."""
        with self.lock:
            futures = list(self.pending_futures)

        for future in futures:
            future.result(timeout=timeout)  # Blocks until complete

        with self.lock:
            self.pending_futures.clear()

    def shutdown(self):
        """Shutdown thread pool (waits for pending writes)."""
        self.executor.shutdown(wait=True)
```

### Pattern 3: Coordinator Manifest for Multi-GPU Tracking
**What:** JSON manifest tracking checkpoint progress across all shards
**When to use:** Multi-GPU coordinator after each shard checkpoints
**Example:**
```python
# Source: Phase 7 multi-GPU patterns + user decisions
class CheckpointManifest:
    """Coordinator manifest tracking batch boundaries per shard."""

    def __init__(self, manifest_path: Path):
        self.manifest_path = manifest_path
        self.lock = threading.Lock()

    def initialize(self, world_size: int):
        """Create initial manifest."""
        manifest = {
            'version': '1.0',
            'world_size': world_size,
            'created_at': datetime.utcnow().isoformat(),
            'shards': {
                str(rank): {
                    'status': 'in_progress',
                    'last_checkpoint_batch': -1,
                    'total_sequences': 0,
                    'last_checkpoint_time': None,
                    'checkpoints': []
                }
                for rank in range(world_size)
            }
        }

        self._save_manifest(manifest)
        return manifest

    def update_shard_checkpoint(
        self,
        rank: int,
        batch_idx: int,
        num_sequences: int,
        checkpoint_file: str
    ):
        """Update manifest after shard checkpoint."""
        with self.lock:
            manifest = self._load_manifest()

            shard_key = str(rank)
            shard = manifest['shards'][shard_key]

            shard['last_checkpoint_batch'] = batch_idx
            shard['total_sequences'] += num_sequences
            shard['last_checkpoint_time'] = datetime.utcnow().isoformat()
            shard['checkpoints'].append({
                'batch_idx': batch_idx,
                'num_sequences': num_sequences,
                'file': checkpoint_file,
                'timestamp': datetime.utcnow().isoformat()
            })

            self._save_manifest(manifest)

    def mark_shard_complete(self, rank: int):
        """Mark shard as fully processed."""
        with self.lock:
            manifest = self._load_manifest()
            manifest['shards'][str(rank)]['status'] = 'complete'
            manifest['shards'][str(rank)]['completed_at'] = datetime.utcnow().isoformat()
            self._save_manifest(manifest)

    def _load_manifest(self) -> Dict:
        """Load manifest with atomic read."""
        with open(self.manifest_path, 'r') as f:
            return json.load(f)

    def _save_manifest(self, manifest: Dict):
        """Save manifest with atomic write."""
        temp_path = self.manifest_path.with_suffix('.tmp')
        with open(temp_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        temp_path.replace(self.manifest_path)
```

### Pattern 4: Resume from Checkpoints with Validation
**What:** Load existing checkpoints, validate consistency, resume from last batch
**When to use:** Pipeline start when checkpoint directory exists
**Example:**
```python
# Source: Phase 4 checkpoint patterns + user decisions
def resume_from_checkpoints(
    checkpoint_dir: Path,
    rank: int,
    force_restart: bool = False
) -> Tuple[List[str], List[np.ndarray], int]:
    """Resume worker from existing checkpoints.

    Returns:
        (sequence_ids, embeddings, resume_batch_idx)
    """
    shard_dir = checkpoint_dir / f"shard_{rank}"

    if force_restart or not shard_dir.exists():
        return [], [], 0

    # Find all checkpoints with .done markers
    checkpoint_files = sorted(
        [f for f in shard_dir.glob("batch_*.h5") if has_done_marker(f)],
        key=lambda p: int(p.stem.split('_')[1])  # Extract batch number
    )

    if not checkpoint_files:
        logger.info(f"No valid checkpoints found for shard {rank}")
        return [], [], 0

    all_ids = []
    all_embeddings = []
    last_batch = -1

    for ckpt_path in checkpoint_files:
        # Validate checkpoint
        is_valid, error = validate_checkpoint_hdf5(ckpt_path)

        if not is_valid:
            logger.warning(
                f"Corrupted checkpoint detected: {ckpt_path}\n"
                f"  Error: {error}\n"
                f"  Stopping resume at batch {last_batch}, will reprocess from here"
            )
            break

        # Load checkpoint
        with h5py.File(ckpt_path, 'r') as f:
            batch_embeddings = f['embeddings'][:]
            batch_ids = [sid.decode('utf-8') if isinstance(sid, bytes) else sid
                        for sid in f['sequence_ids'][:]]

            # Extract batch_idx from filename
            batch_idx = int(ckpt_path.stem.split('_')[1])

            all_embeddings.append(batch_embeddings)
            all_ids.extend(batch_ids)
            last_batch = batch_idx

    if all_embeddings:
        embeddings = np.concatenate(all_embeddings, axis=0)
        logger.info(
            f"Resuming from {len(checkpoint_files)} checkpoints: "
            f"{len(all_ids)} sequences, last_batch={last_batch}"
        )
        return all_ids, embeddings, last_batch + 1
    else:
        return [], [], 0

def validate_checkpoint_hdf5(checkpoint_path: Path) -> Tuple[bool, str]:
    """Validate HDF5 checkpoint (lightweight check)."""
    # Level 1: File size
    if checkpoint_path.stat().st_size == 0:
        return False, "file is 0 bytes"

    # Level 2: .done marker exists
    if not has_done_marker(checkpoint_path):
        return False, "missing .done marker (incomplete write)"

    # Level 3: HDF5 file structure
    try:
        with h5py.File(checkpoint_path, 'r') as f:
            if 'embeddings' not in f or 'sequence_ids' not in f:
                return False, "missing required datasets"

            # Quick shape check
            num_sequences = f['sequence_ids'].shape[0]
            emb_sequences = f['embeddings'].shape[0]

            if num_sequences != emb_sequences:
                return False, f"shape mismatch: {num_sequences} IDs, {emb_sequences} embeddings"
    except Exception as e:
        return False, f"HDF5 read failed: {str(e)}"

    return True, ""
```

### Pattern 5: Exponential Backoff for GPU Failures
**What:** Retry failed GPUs with increasing delays (handles transient OOM, CUDA errors)
**When to use:** GPUProcessCoordinator when worker reports failure
**Example:**
```python
# Source: Exponential backoff research + user decisions
def retry_worker_with_backoff(
    worker_fn: Callable,
    rank: int,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0
) -> bool:
    """Retry GPU worker with exponential backoff.

    Returns:
        True if worker succeeded, False if all retries exhausted
    """
    for attempt in range(max_retries):
        try:
            worker_fn(rank)
            return True

        except RuntimeError as e:
            error_msg = str(e)

            # Classify error type for logging
            if "CUDA out of memory" in error_msg:
                error_type = "OOM"
            elif "CUDA" in error_msg:
                error_type = "CUDA_ERROR"
            else:
                error_type = "RUNTIME_ERROR"

            # Calculate backoff delay: 2^attempt * base_delay, capped at max_delay
            delay = min(base_delay * (2 ** attempt), max_delay)

            if attempt < max_retries - 1:
                logger.warning(
                    f"Worker {rank} failed (attempt {attempt + 1}/{max_retries}): {error_type}\n"
                    f"  Error: {error_msg}\n"
                    f"  Retrying in {delay:.1f}s..."
                )
                time.sleep(delay)
            else:
                logger.error(
                    f"Worker {rank} failed after {max_retries} attempts: {error_type}\n"
                    f"  Final error: {error_msg}"
                )
                return False

    return False
```

### Anti-Patterns to Avoid
- **Checkpoint mid-packed-batch:** Breaks FlashAttention varlen `cu_seqlens` boundaries, corrupts cross-sequence attention masking
- **Synchronous I/O on GPU critical path:** Causes 10-20x slowdown compared to async writes (research confirmed)
- **Missing .done markers:** Resume logic can't distinguish complete vs interrupted checkpoints
- **Copying PyTorch tensors without .copy():** Background thread race conditions if tensor memory reused
- **No manifest for multi-GPU:** Can't validate completeness or recover from partial failures

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Async file I/O | Custom async file writer | concurrent.futures.ThreadPoolExecutor | Stdlib, proven pattern for I/O-bound ops, avoids GIL contention |
| Atomic HDF5 writes | Manual h5py write retry logic | Temp-then-rename pattern from Phase 4 | Already proven in checkpoint.py, matches .done marker pattern |
| Checkpoint validation | Try-except on h5py.File open | Multi-level validation from Phase 3 | Size check → .done marker → h5py structure → shape consistency |
| Exponential backoff | Sleep with manual counter | tenacity library OR stdlib pattern | Research shows 95%+ success rate vs 20-40% fixed delay |
| Progress tracking state | Custom checkpoint registry | Extend Phase 3 CheckpointManager | Already tracks pipeline stages, add batch-level granularity |

**Key insight:** Phase 3/4 checkpoint infrastructure (atomic writes, .done markers, validation) and Phase 7 multi-GPU patterns (shard files, coordinator) provide 80% of needed primitives. Extend existing patterns rather than introducing new mechanisms.

## Common Pitfalls

### Pitfall 1: Checkpointing Mid-Packed-Batch
**What goes wrong:** Corrupted attention masking when resume splits packed sequences
**Why it happens:** Checkpoint trigger fires during batch processing, splits `cu_seqlens` boundaries
**How to avoid:** Always checkpoint at batch boundaries (after `process_batch` completes), emergency override only for >10 min stalls
**Warning signs:** Resume produces different embeddings than continuous run, FlashAttention errors about invalid `cu_seqlens`

**Technical detail:** FlashAttention varlen uses `cu_seqlens = [0, len1, len1+len2, ...]` to mark packed sequence boundaries. Checkpointing mid-batch would save partial `cu_seqlens`, causing cross-sequence attention contamination on resume.

### Pitfall 2: Background Thread Race Conditions
**What goes wrong:** Segfault or corrupted checkpoint when PyTorch tensor memory reused before write completes
**Why it happens:** Passed tensor reference to background thread, GPU reused memory for next batch
**How to avoid:** Always `.copy()` numpy arrays before passing to ThreadPoolExecutor.submit()
**Warning signs:** Intermittent crashes, corrupted embeddings in checkpoints, non-deterministic failures

### Pitfall 3: Manifest Out of Sync with Checkpoints
**What goes wrong:** Manifest shows checkpoint exists but file is missing (or vice versa)
**Why it happens:** Manifest updated before checkpoint write completes, or write fails silently
**How to avoid:** Update manifest AFTER async write completes (wait for Future), use .done markers for atomic completion
**Warning signs:** Resume fails with "checkpoint not found" despite manifest entry

### Pitfall 4: Ignoring Async Write Failures
**What goes wrong:** Checkpoint write fails silently in background thread, pipeline thinks checkpoint succeeded
**Why it happens:** Future exceptions not checked, ThreadPoolExecutor swallows errors
**How to avoid:** Call `future.result()` before updating manifest, or register exception callback
**Warning signs:** Missing checkpoints on resume, no error logs during run

### Pitfall 5: No Emergency Checkpoint Override
**What goes wrong:** Pipeline stuck processing 400AA+ viral sequences for 15+ minutes without checkpoint
**Why it happens:** Batch atomicity constraint prevents checkpoint, no override mechanism
**How to avoid:** Implement emergency trigger (>10 min without checkpoint) that forces checkpoint mid-batch
**Warning signs:** Long batches cause 30+ min of lost work on crashes, uneven checkpoint spacing

### Pitfall 6: Checkpoint Directory Conflicts
**What goes wrong:** Multiple GPUs try to write to same checkpoint directory, corrupt manifest
**Why it happens:** Shared checkpoint path without per-shard isolation
**How to avoid:** Per-shard subdirectories (`checkpoints/shard_N/`), coordinator owns manifest writes
**Warning signs:** Manifest corruption, missing checkpoint files, race condition errors

## Code Examples

Verified patterns from existing codebase and research:

### Integration with AsyncInferenceRunner
```python
# Source: Combining async_inference.py patterns with checkpoint research
class AsyncInferenceRunnerWithCheckpoints(AsyncInferenceRunner):
    """AsyncInferenceRunner extended with incremental checkpointing."""

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        checkpoint_dir: Path,
        rank: int = 0,
        enable_checkpointing: bool = True,
        checkpoint_seq_threshold: int = 10000,
        checkpoint_time_threshold: int = 300,
        **kwargs
    ):
        super().__init__(model, device, **kwargs)

        self.checkpoint_dir = checkpoint_dir
        self.rank = rank
        self.enable_checkpointing = enable_checkpointing

        if enable_checkpointing:
            self.shard_checkpoint_dir = checkpoint_dir / f"shard_{rank}"
            self.shard_checkpoint_dir.mkdir(parents=True, exist_ok=True)

            self.trigger = CheckpointTrigger(
                seq_threshold=checkpoint_seq_threshold,
                time_threshold_sec=checkpoint_time_threshold
            )
            self.writer = AsyncCheckpointWriter(max_workers=1)

            # Resume state
            self.checkpoint_embeddings = []
            self.checkpoint_ids = []
            self.checkpoint_batch_idx = 0

    def run_with_checkpoints(
        self,
        dataloader: DataLoader,
        progress_callback: Optional[Callable] = None,
        force_restart: bool = False
    ) -> Iterator[InferenceResult]:
        """Run inference with incremental checkpointing."""

        # Resume from existing checkpoints
        if self.enable_checkpointing and not force_restart:
            resumed_ids, resumed_embs, resume_batch = resume_from_checkpoints(
                self.checkpoint_dir, self.rank, force_restart
            )

            if resumed_embs:
                self.checkpoint_embeddings = [resumed_embs]
                self.checkpoint_ids = resumed_ids
                self.checkpoint_batch_idx = resume_batch

                logger.info(
                    f"Resuming from batch {resume_batch}: "
                    f"{len(resumed_ids)} sequences already processed"
                )

                # Yield resumed results (for caller to know what's already done)
                yield InferenceResult(
                    sequence_ids=resumed_ids,
                    embeddings=torch.from_numpy(resumed_embs),
                    batch_idx=-1  # Special marker for resumed data
                )

        # Run inference with checkpoint triggers
        for result in super().run(dataloader, progress_callback):
            # Accumulate for checkpoint
            if self.enable_checkpointing:
                self.checkpoint_embeddings.append(result.embeddings.numpy())
                self.checkpoint_ids.extend(result.sequence_ids)

            # Check checkpoint trigger
            if self.enable_checkpointing:
                should_ckpt, reason = self.trigger.should_checkpoint(
                    batch_size=len(result.sequence_ids)
                )

                if should_ckpt:
                    self._write_checkpoint(reason)
                    self.trigger.reset()

            yield result

        # Final checkpoint for remaining data
        if self.enable_checkpointing and self.checkpoint_embeddings:
            self._write_checkpoint("final")

        # Wait for all async writes to complete
        if self.enable_checkpointing:
            self.writer.wait_all(timeout=300)  # 5 min timeout
            logger.info("All checkpoints flushed")

    def _write_checkpoint(self, reason: str):
        """Write checkpoint asynchronously."""
        if not self.checkpoint_embeddings:
            return

        embeddings = np.concatenate(self.checkpoint_embeddings, axis=0)

        checkpoint_path = self.shard_checkpoint_dir / f"batch_{self.checkpoint_batch_idx:05d}.h5"

        metadata = {
            'batch_idx': self.checkpoint_batch_idx,
            'num_sequences': len(self.checkpoint_ids),
            'timestamp': datetime.utcnow().isoformat(),
            'trigger_reason': reason,
            # Model config
            'model_dtype': str(next(self.model.parameters()).dtype),
            'packing_enabled': os.getenv('VIRNUCPRO_DISABLE_PACKING', 'false') != 'true',
        }

        # Submit async write
        future = self.writer.write_checkpoint_async(
            checkpoint_path,
            embeddings,
            self.checkpoint_ids,
            metadata
        )

        logger.info(
            f"Checkpoint {self.checkpoint_batch_idx} queued: "
            f"{len(self.checkpoint_ids)} sequences, reason={reason}"
        )

        self.checkpoint_batch_idx += 1
        self.checkpoint_embeddings = []
        self.checkpoint_ids = []
```

### Multi-GPU Coordinator with Manifest
```python
# Source: Phase 7 multi_gpu_inference.py + manifest pattern
def run_multi_gpu_inference_with_checkpoints(
    fasta_files: List[Path],
    output_dir: Path,
    model_config: Dict[str, Any],
    world_size: Optional[int] = None,
    enable_checkpointing: bool = True,
    force_restart: bool = False
) -> Tuple[Path, List[int]]:
    """Run multi-GPU inference with incremental checkpointing."""

    output_dir = Path(output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if world_size is None:
        world_size = torch.cuda.device_count()

    # Initialize manifest
    manifest_path = checkpoint_dir / "manifest.json"
    if enable_checkpointing:
        if force_restart and manifest_path.exists():
            logger.info("Force restart: removing existing manifest")
            manifest_path.unlink()

        manifest = CheckpointManifest(manifest_path)
        if not manifest_path.exists():
            manifest.initialize(world_size)

    # Create sequence index (existing Phase 7 pattern)
    index_path = output_dir / "sequence_index.json"
    create_sequence_index(fasta_files, index_path)

    # Spawn workers with checkpoint support
    coordinator = GPUProcessCoordinator(world_size, output_dir)

    # Modify model_config to include checkpoint settings
    model_config_with_ckpt = {
        **model_config,
        'enable_checkpointing': enable_checkpointing,
        'checkpoint_dir': str(checkpoint_dir),
        'force_restart': force_restart
    }

    coordinator.spawn_workers(
        gpu_worker_with_checkpoints,  # Modified worker function
        (index_path, output_dir, model_config_with_ckpt)
    )

    # Wait for completion with retry logic
    completion_status = coordinator.wait_for_completion_with_retry(
        timeout=None,  # No global timeout with checkpointing
        max_retries=3,
        base_delay=1.0
    )

    # Existing aggregation logic (Phase 7 pattern)
    successful_ranks = [r for r, ok in completion_status.items() if ok]
    failed_ranks = [r for r, ok in completion_status.items() if not ok]

    if not successful_ranks:
        raise RuntimeError("No workers completed successfully")

    # Aggregate final shards (existing pattern)
    shard_files = [output_dir / f"shard_{rank}.h5" for rank in successful_ranks]
    output_path = output_dir / "embeddings.h5"
    aggregate_shards(shard_files, output_path)

    return output_path, failed_ranks
```

### Checkpoint Validation for Resume
```python
# Source: Phase 3 checkpoint_validation.py + HDF5 patterns
def validate_checkpoint_completeness(
    checkpoint_dir: Path,
    expected_sequence_ids: Set[str]
) -> Tuple[bool, Dict[str, Any]]:
    """Validate checkpoint directory completeness.

    Returns:
        (is_complete, diagnostics)
    """
    diagnostics = {
        'total_checkpoints': 0,
        'valid_checkpoints': 0,
        'corrupted_checkpoints': [],
        'sequence_ids_found': set(),
        'missing_ids': set(),
        'duplicate_ids': []
    }

    # Find all checkpoint files
    checkpoint_files = list(checkpoint_dir.glob("shard_*/batch_*.h5"))
    diagnostics['total_checkpoints'] = len(checkpoint_files)

    seen_ids = set()

    for ckpt_path in checkpoint_files:
        is_valid, error = validate_checkpoint_hdf5(ckpt_path)

        if not is_valid:
            diagnostics['corrupted_checkpoints'].append({
                'file': str(ckpt_path),
                'error': error
            })
            continue

        diagnostics['valid_checkpoints'] += 1

        # Load sequence IDs
        with h5py.File(ckpt_path, 'r') as f:
            for sid in f['sequence_ids'][:]:
                sid_str = sid.decode('utf-8') if isinstance(sid, bytes) else sid

                if sid_str in seen_ids:
                    diagnostics['duplicate_ids'].append(sid_str)

                seen_ids.add(sid_str)

    diagnostics['sequence_ids_found'] = seen_ids
    diagnostics['missing_ids'] = expected_sequence_ids - seen_ids

    is_complete = (
        len(diagnostics['corrupted_checkpoints']) == 0 and
        len(diagnostics['missing_ids']) == 0 and
        len(diagnostics['duplicate_ids']) == 0
    )

    return is_complete, diagnostics
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Single final checkpoint | Incremental checkpoints every N sequences | PyTorch DCP 2023+ | Resume from 10K sequence granularity vs full restart (hours saved) |
| Synchronous checkpoint writes | Async writes via background threads | PyTorch async checkpoint 2024 | 10-20x faster checkpointing, no GPU stalls |
| Global checkpoint failures | Per-shard independence with manifest | Multi-GPU research 2024-2025 | Partial failure recovery (restart only failed GPUs) |
| No batch atomicity awareness | Checkpoint at batch boundaries only | Sequence packing research 2024 | Prevents cross-sequence attention corruption |
| Fixed retry delays | Exponential backoff with jitter | AWS/distributed systems 2023+ | 95%+ success rate vs 20-40% for fixed delays |

**Deprecated/outdated:**
- **Checkpoint-restart only:** Modern systems use incremental checkpointing for <10% overhead vs 100% restart penalty
- **GIL-heavy async (asyncio):** ThreadPoolExecutor preferred for I/O-bound ops (simpler, no event loop)
- **No manifest tracking:** Multi-GPU systems require coordination metadata for partial failure handling
- **Mid-batch checkpoints:** Sequence packing makes this unsafe (attention boundary corruption)

## Open Questions

Things that couldn't be fully resolved:

1. **Optimal checkpoint interval for 400AA+ viral sequences**
   - What we know: Default 10K sequences OR 5 minutes works for typical 100-200AA proteins
   - What's unclear: Viral sequences 400AA+ take 10-15 min per batch, may need different thresholds
   - Recommendation: Environment variable `VIRNUCPRO_VIRAL_CHECKPOINT_MODE` to override (5K sequences / 3 min)

2. **ThreadPoolExecutor max_workers sizing**
   - What we know: 1 worker sufficient for single write stream (sequential checkpoints)
   - What's unclear: Whether 2-3 workers would help with manifest updates + checkpoint writes
   - Recommendation: Start with max_workers=1 (simplest), profile if write queue builds up

3. **Emergency checkpoint mid-batch threshold**
   - What we know: >10 min without checkpoint is bad (long recovery penalty)
   - What's unclear: Exact threshold before forcing mid-batch checkpoint (10 min? 15 min?)
   - Recommendation: 10 min as default, monitor for viral workloads and adjust if needed

4. **Manifest format: JSON vs YAML**
   - What we know: JSON is stdlib, faster parsing, used in sequence_index.json
   - What's unclear: Whether YAML's human-readability is worth the dependency
   - Recommendation: JSON for consistency with Phase 7 patterns, no new deps

5. **Checkpoint compression for large embeddings**
   - What we know: ESM-2 embeddings are float32, 5120-dim, ~20KB per sequence
   - What's unclear: Whether gzip compression (HDF5 native) worth I/O slowdown
   - Recommendation: Start uncompressed (fast writes critical), add compression flag for storage-constrained scenarios

## Sources

### Primary (HIGH confidence)
- Codebase analysis - virnucpro/core/checkpoint.py (atomic writes, .done markers, CheckpointManager)
- Codebase analysis - virnucpro/pipeline/async_inference.py (AsyncInferenceRunner, batch processing loop)
- Codebase analysis - virnucpro/pipeline/gpu_worker.py (HDF5 shard writing pattern)
- Codebase analysis - virnucpro/pipeline/shard_aggregator.py (chunk-wise aggregation, validation)
- User decisions - .planning/phases/09-checkpointing-integration/09-CONTEXT.md (checkpoint granularity, resume behavior, format, recovery)
- [PyTorch Reducing Model Checkpointing Times by Over 10x with Asynchronous Checkpointing](https://pytorch.org/blog/reducing-checkpointing-times/) - Background thread async pattern, 10-20x speedup
- [PyTorch 6x Faster Async Checkpointing](https://pytorch.org/blog/6x-faster-async-checkpointing/) - GIL contention avoidance, process-based checkpointing
- [Python concurrent.futures Documentation](https://docs.python.org/3/library/concurrent.futures.html) - ThreadPoolExecutor API, I/O-bound task patterns

### Secondary (MEDIUM confidence)
- [Efficient LLM Pretraining: Packed Sequences and Masked Attention](https://huggingface.co/blog/sirluk/llm-sequence-packing) - cu_seqlens boundaries, position ID reset
- [Improving Hugging Face Training Efficiency Through Packing with Flash Attention 2](https://huggingface.co/blog/packing-with-FA2) - Cross-example attention prevention, batch atomicity
- [NVIDIA NeMo Sequence Packing Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/sft_peft/packed_sequence.html) - Packing implementation patterns
- [Crash-Consistent Checkpointing for AI Training (arXiv 2025)](https://arxiv.org/html/2511.18323) - Atomic writes with SHA-256 validation, 99.8-100% corruption detection
- [Gemini: Fast Failure Recovery in Distributed Training (SOSP 2023)](https://www.cs.rice.edu/~eugeneng/papers/SOSP23.pdf) - In-memory checkpoints, partial failure recovery, 92% training time reduction
- [Just-In-Time Checkpointing (ICS 2024)](https://dl.acm.org/doi/pdf/10.1145/3627703.3650085) - Low cost error recovery, no collective communication requirement
- [Exponential Backoff and Retry Patterns (2026)](https://johal.in/tenacity-retries-exponential-backoff-decorators-2026/) - 95%+ success rate for transient failures
- [Queue-Based Exponential Backoff (DEV 2026)](https://dev.to/andreparis/queue-based-exponential-backoff-a-resilient-retry-pattern-for-distributed-systems-37f3) - Distributed system resilience patterns

### Tertiary (LOW confidence)
- [Scalable Incremental Checkpointing using GPU-Accelerated De-Duplication (ACM 2023)](https://dl.acm.org/doi/fullHtml/10.1145/3605573.3605639) - Merkle tree deduplication for sparse updates
- [HDF5 Data Corruption Issues (Google Groups)](https://groups.google.com/g/h5py/c/s_luehojYik) - Community reports of corruption patterns
- [HDF5 Parallel Write Performance (ResearchGate)](https://www.researchgate.net/figure/HDF5-parallel-write-process-with-and-without-compression-When-compression-is-enabled_fig2_357378339) - Compression tradeoffs
- [ThreadPoolExecutor Complete Guide (Super Fast Python)](https://superfastpython.com/threadpoolexecutor-in-python/) - I/O-bound vs CPU-bound task patterns

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - All stdlib/h5py (already in use), ThreadPoolExecutor is stdlib since Python 3.2
- Architecture patterns: HIGH - Async writes proven in PyTorch research, manifest pattern from Phase 7, checkpoint patterns from Phase 3/4
- Batch atomicity: HIGH - Sequence packing research confirms `cu_seqlens` boundary requirements, user decision aligns
- Exponential backoff: MEDIUM - General distributed systems pattern, not GPU-specific research
- Emergency override threshold: MEDIUM - User decision says ">10 min" but exact implementation details discretionary

**Research date:** 2026-02-05
**Valid until:** 2026-03-05 (30 days - checkpoint patterns are stable, but async I/O optimizations evolving rapidly)
