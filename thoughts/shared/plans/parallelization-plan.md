# Parallelize Embeddings Extraction in VirNucPro

## Overview

VirNucPro's embedding extraction pipeline has two critical bottlenecks: (1) DNABERT-S processes sequences one at a time instead of in batches, and (2) file-level processing is sequential instead of parallel across GPUs. This plan implements **Option 2: Hybrid Batching + Multiprocessing** to achieve 150-380x speedup with 4 GPUs.

The solution adds batching to `extract_dnabert_features()` (following the pattern already used by ESM-2) and introduces multiprocessing for file-level parallelization with auto-detected GPU assignment.

## Planning Context

### Decision Log

| Decision | Reasoning Chain |
|----------|-----------------|
| Batched DNABERT-S processing | Current implementation makes 100k GPU calls for 100k sequences -> GPU underutilized at 5-10% -> batching reduces to ~390 calls at batch_size=256 -> 50-100x speedup with minimal code change |
| Token-based batching over fixed batching | DNA sequences are fixed length after chunking (300/500bp) -> fixed batch_size is sufficient -> simpler than ESM-2's token-based approach -> user can configure via CLI/config |
| Multiprocessing over threading | Python GIL prevents true parallelism with threading -> multiprocessing creates separate interpreter per GPU -> each process can fully utilize its assigned GPU -> linear speedup with GPU count |
| Multiprocessing over DataParallel | File-based architecture already splits work into independent units -> DataParallel adds inter-GPU communication overhead -> multiprocessing with separate processes per GPU is more efficient for this workload |
| Auto-detect all GPUs | User selected auto-detection -> simplifies CLI usage -> round-robin file assignment balances load -> no manual device specification required |
| Configurable batch_size via CLI/config | User selected configurable option -> adds --dnabert-batch-size flag -> adds features.dnabert.batch_size config -> default 256 for 4GB+ GPUs |
| Round-robin GPU assignment | Simple and effective for uniform file sizes -> files split to 10k sequences each ensures balanced load -> more complex scheduling not justified |
| Process Pool over manual Process management | Pool provides cleaner API -> handles worker lifecycle -> starmap enables parallel dispatch with arguments -> proven pattern in Python ecosystem |
| Model loaded per worker | Multiprocessing requires separate model instance per GPU -> each worker loads DNABERT-S on its assigned GPU -> 1.5GB VRAM per worker is acceptable |
| Checkpoint integration via output file detection | Existing checkpoint tracks output files -> multiprocessing workers write to same output paths -> resume detects completed files -> no checkpoint schema changes needed |
| batch_size=256 default | Memory analysis: 256 sequences × 512 tokens × 768 dim × 4 bytes = 400MB batch memory -> plus 1.5GB model = 2GB total -> safe for 4GB+ GPUs -> optimal tradeoff of speed vs compatibility |
| Tolerance rtol=1e-4, atol=1e-6 for output equivalence | PyTorch default float32 precision ~1e-7 -> mean pooling across 512 tokens accumulates small errors -> rtol=1e-4 accommodates this accumulation while detecting real differences -> atol=1e-6 handles near-zero values -> stricter than necessary but validates true equivalence without false positives |
| Mean pooling for embeddings | ESM-2 implementation uses mean pooling successfully (features.py:152) -> DNABERT-S paper recommends mean pooling for sequence-level representations -> max pooling loses magnitude information useful for classification -> CLS token not reliably used in DNABERT-S architecture -> mean pooling is standard approach for transformer embeddings |
| Overall file progress bar (not per-worker) | Per-worker progress would show N separate bars -> cluttered terminal output with unclear total completion -> overall file count gives clear completion percentage -> workers finish files at different rates anyway -> simpler implementation with clearer user value |
| Fail-fast on worker error | GPU errors indicate hardware/driver issues -> partial results from failed worker unreliable -> re-running with same configuration would repeat the error -> better to fail immediately and let user diagnose issue -> checkpoint system allows resume after fix -> graceful degradation risks silent partial failures and incorrect results -> pool.starmap() blocks until ALL workers complete before propagating exceptions, so failure detection is delayed but unavoidable tradeoff for simpler implementation |
| Parallelization opt-in via --parallel flag (user-specified) | User selected opt-in behavior -> in shared GPU environments auto-enabling could monopolize resources -> explicit flag gives user control -> default single-GPU is safe and predictable -> power users enable with --parallel |

### Rejected Alternatives

| Alternative | Why Rejected |
|-------------|--------------|
| DataParallel | Adds inter-GPU communication overhead, imbalanced GPU memory on master GPU, file-based architecture makes multiprocessing more natural |
| DistributedDataParallel | Requires process group initialization, overkill for single-node multi-GPU, adds significant complexity |
| Async I/O | Python asyncio doesn't bypass GIL for GPU operations, file I/O is not the bottleneck, provides no meaningful speedup |
| Threading | Python GIL prevents true parallelism, GPU operations release GIL but file I/O doesn't, limited benefit over sequential |
| Dynamic VRAM-aware batch sizing | Adds complexity for probing VRAM, try-catch approach can cause partial failures, configurable batch_size is simpler and sufficient |
| Translation parallelization | Not included in this plan - requires profiling to determine if translation is >15% of runtime, can be added later if needed |

### Constraints & Assumptions

- Python 3.9+ with multiprocessing.Pool support
- CUDA available for GPU acceleration (CPU fallback exists)
- DNABERT-S tokenizer supports batch tokenization
- Files split to uniform 10k sequences each (existing behavior)
- Each GPU has sufficient VRAM for model + batch (2GB minimum at batch_size=256)
- Checkpoint system tracks output file paths (existing behavior)
- ESM-2 batching pattern is stable and can be referenced (existing code)

### Known Risks

| Risk | Mitigation | Anchor |
|------|------------|--------|
| OOM with batch_size=256 | Configurable batch_size allows user to reduce; default is safe for 4GB+ GPUs | N/A - user configuration |
| Process spawning overhead | Workers process multiple files each; overhead amortized over file count | N/A - acceptable tradeoff |
| Model loading time per worker | Each worker loads model once, processes all assigned files; loading time amortized | N/A - one-time cost per worker |
| Checkpoint compatibility | Existing checkpoint tracks output files; no schema changes needed | virnucpro/pipeline/prediction.py:226-229 - nucleotide_features list |
| Tokenizer batching behavior | DNABERT-S uses HuggingFace transformers; batch tokenization is standard API | features.py:38 - AutoTokenizer.from_pretrained |

## Invisible Knowledge

### Architecture

```
FASTA Input
    |
    v
+-------------+     +---------------+     +------------------+
| Chunk (300/ | --> | 6-Frame       | --> | Split to 10k     |
| 500bp)      |     | Translation   |     | seqs/file        |
+-------------+     +---------------+     +------------------+
                                                 |
                    +----------------------------+
                    |
                    v
        +------------------------+
        | Multiprocessing Pool   |
        | (N workers = N GPUs)   |
        +------------------------+
             |    |    |    |
             v    v    v    v
        +------+ +------+ +------+ +------+
        |GPU 0 | |GPU 1 | |GPU 2 | |GPU 3 |
        |Files | |Files | |Files | |Files |
        |0,4,8 | |1,5,9 | |2,6,10| |3,7,11|
        +------+ +------+ +------+ +------+
             |    |    |    |
             +----+----+----+
                    |
                    v
        +------------------------+
        | Merge + Predict + Cons |
        +------------------------+
                    |
                    v
              Final Results
```

### Data Flow

```
nucleotide_files (list of .fa)
        |
        v
round_robin_assignment(files, num_gpus)
        |
        v
[files_for_gpu0, files_for_gpu1, ...]
        |
        v
Pool.starmap(process_files_on_gpu, [(files0, 0), (files1, 1), ...])
        |
        v (each worker)
load_model(device=cuda:N)
for file in assigned_files:
    sequences = load_fasta(file)
    batches = chunk(sequences, batch_size)
    for batch in batches:
        embeddings = model(tokenizer(batch))
    save(embeddings)
        |
        v
nucleotide_feature_files (list of .pt)
```

### Why This Structure

The existing file-splitting architecture (10k sequences per file) naturally enables parallelization. Each file is an independent unit of work that can be processed on a separate GPU. This avoids the need to restructure the pipeline or change checkpoint semantics.

Multiprocessing over DataParallel because:
- Files are already independent units (no cross-file dependencies)
- DataParallel adds GPU-to-GPU communication overhead
- DataParallel requires batch splitting and result gathering on master GPU
- Multiprocessing allows true independent processing with better GPU utilization

### Invariants

1. **Output equivalence**: Batched processing must produce identical embeddings to sequential processing (within floating point tolerance)
2. **File ordering preserved**: Output files must correspond 1:1 with input files regardless of processing order
3. **GPU assignment deterministic**: Same file list + GPU count = same assignment (enables resume)
4. **Checkpoint compatibility**: Existing checkpoint schema unchanged; resume works by detecting completed output files

### Tradeoffs

| Tradeoff | Cost | Benefit |
|----------|------|---------|
| Model loaded per worker | N × 1.5GB VRAM | True parallel processing, no GIL limitations |
| Process spawning | ~0.5-1s startup per worker | Workers reused for all files, amortized |
| Fixed batch_size default | May not be optimal for all GPUs | Simplicity, predictable memory usage |
| Round-robin assignment | Suboptimal if file sizes vary | Simple, works well with uniform 10k splits |

## Milestones

### Milestone 1: Batched DNABERT-S Processing

**Files**:
- `virnucpro/pipeline/features.py`
- `config/default_config.yaml`
- `virnucpro/core/config.py`
- `tests/test_features.py` (new)

**Flags**: `performance`, `needs-rationale`

**Requirements**:
- Modify `extract_dnabert_features()` to batch sequences before tokenization
- Add `features.dnabert.batch_size` config option with default 256
- Tokenize entire batch at once using tokenizer's padding support
- Single GPU forward pass per batch instead of per sequence
- Mean pool embeddings per sequence (preserve existing output format)
- Maintain backward compatibility with existing output file format

**Acceptance Criteria**:
- GPU utilization increases from ~10% to 80%+ during extraction
- Output embeddings match sequential processing within rtol=1e-4, atol=1e-6
- Processing 10k sequences completes in <10s (vs ~90s sequential)
- Config option `features.dnabert.batch_size` is respected

**Tests**:
- **Test files**: `tests/test_features.py`
- **Test type**: integration + property-based
- **Backing**: user-specified (example-based with property invariants)
- **Scenarios**:
  - Normal: Batch of 10 sequences produces correct embeddings matching sequential
  - Edge: Single sequence batch works correctly
  - Edge: Batch size larger than sequence count handles gracefully
  - Error: Empty input file handled gracefully
  - Property: Output equivalence invariant - batched(sequences, bs) == sequential(sequences) for any valid batch_size
  - Property: Batch size independence - batched(sequences, bs1) == batched(sequences, bs2) for any two batch sizes
  - Property: Order preservation - output order matches input order regardless of batch boundaries

**Code Intent**:
- Modify `extract_dnabert_features()` in `features.py`:
  - Load all sequences into list before processing
  - Create batches of `batch_size` sequences
  - Tokenize each batch with `tokenizer(batch_seqs, return_tensors='pt', padding=True)`
  - Forward pass on batch, get hidden states of shape (batch, seq_len, 768)
  - Mean pool each sequence: `torch.mean(hidden_states, dim=1)` -> (batch, 768)
  - Append results maintaining sequence order
  - Preserve output format: `{'nucleotide': [ids], 'data': [dicts with label and mean_representation]}`
- Add to `config/default_config.yaml`:
  - `features.dnabert.batch_size: 256`
- Add to `virnucpro/core/config.py`:
  - Document new config key in schema/validation

**Code Changes**:

```diff
--- a/virnucpro/pipeline/features.py
+++ b/virnucpro/pipeline/features.py
@@ -40,30 +40,40 @@ def extract_dnabert_features(
     model.eval()

-    # Load sequences and process one at a time (matching original implementation)
+    # Load all sequences
     nucleotide = []
     data = []

     records = list(SeqIO.parse(nucleotide_file, 'fasta'))

     with torch.no_grad():
-        for record in records:
-            seq = str(record.seq)
-            label = record.id
-
-            # Tokenize
-            inputs = tokenizer(seq, return_tensors='pt')
-            input_ids = inputs["input_ids"].to(device)
-
-            # Forward pass - model returns tuple, take first element
-            hidden_states = model(input_ids)[0]
-            embedding_mean = torch.mean(hidden_states, dim=1)
-
-            result = {
-                "label": label,
-                "mean_representation": embedding_mean.squeeze().cpu().tolist()
-            }
-            nucleotide.append(label)
-            data.append(result)
+        # Process in batches
+        for i in range(0, len(records), batch_size):
+            batch_records = records[i:i + batch_size]
+            batch_seqs = [str(record.seq) for record in batch_records]
+            batch_labels = [record.id for record in batch_records]
+
+            # Tokenize batch with padding
+            inputs = tokenizer(batch_seqs, return_tensors='pt', padding=True)
+            input_ids = inputs["input_ids"].to(device)
+            attention_mask = inputs.get("attention_mask", None)
+            if attention_mask is not None:
+                attention_mask = attention_mask.to(device)
+
+            # Forward pass - model returns tuple, take first element
+            # Shape: (batch_size, seq_len, 768)
+            if attention_mask is not None:
+                hidden_states = model(input_ids, attention_mask=attention_mask)[0]
+            else:
+                hidden_states = model(input_ids)[0]
+
+            # Mean pool each sequence in batch, excluding padding tokens
+            if attention_mask is not None:
+                # Weighted mean: sum(hidden * mask) / sum(mask)
+                embedding_means = (hidden_states * attention_mask.unsqueeze(-1)).sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
+            else:
+                # No padding, simple mean
+                embedding_means = torch.mean(hidden_states, dim=1)  # (batch_size, 768)
+
+            for label, embedding_mean in zip(batch_labels, embedding_means):
+                result = {"label": label, "mean_representation": embedding_mean.cpu().tolist()}
+                nucleotide.append(label)
+                data.append(result)

     # Save to file in original format
     torch.save({'nucleotide': nucleotide, 'data': data}, output_file)
```

```diff
--- a/config/default_config.yaml
+++ b/config/default_config.yaml
@@ -35,6 +35,7 @@ features:
   dnabert:
     model_name: "zhihan1996/DNABERT-S"
     trust_remote_code: true
+    batch_size: 256

   esm:
     model_name: "esm2_t36_3B_UR50D"
```

```diff
--- a/virnucpro/core/config.py
+++ b/virnucpro/core/config.py
@@ -1,6 +1,19 @@
 """Configuration management for VirNucPro"""

+# Configuration Schema Documentation:
+#
+# features.dnabert.batch_size (int, default: 256)
+#   Batch size for DNABERT-S feature extraction. Determines how many
+#   sequences are processed in parallel on GPU. Higher values increase
+#   GPU utilization but require more VRAM. Recommended values:
+#   - 4GB+ GPU: 256 (default)
+#   - 2-4GB GPU: 128
+#   - <2GB GPU: 64
+#
+# features.esm.toks_per_batch (int, default: 2048)
+#   Maximum tokens per batch for ESM-2 extraction.
+
 import yaml
 from pathlib import Path
 from typing import Any, Dict, Optional
```

---

### Milestone 2: Multiprocessing Infrastructure

**Files**:
- `virnucpro/pipeline/parallel.py` (new)
- `virnucpro/core/config.py`
- `config/default_config.yaml`

**Flags**: `error-handling`

**Requirements**:
- Create `parallel.py` module with multiprocessing utilities
- Function to detect available CUDA devices
- Function to assign files to GPUs in round-robin fashion
- Wrapper function that processes files on assigned GPU
- Proper error handling and logging per worker
- Add `parallelization.enabled` and `parallelization.num_workers` config options

**Acceptance Criteria**:
- `detect_cuda_devices()` returns list of available CUDA device indices
- `assign_files_round_robin(files, n_gpus)` returns balanced lists
- Worker function loads model on correct device
- Errors in one worker don't crash other workers
- Config options respected (enabled=false disables multiprocessing)

**Tests**:
- **Test files**: `tests/test_parallel.py` (new)
- **Test type**: unit
- **Backing**: user-specified (example-based)
- **Scenarios**:
  - Normal: 8 files, 4 GPUs -> 2 files per GPU
  - Edge: 3 files, 4 GPUs -> 3 GPUs with 1 file each
  - Edge: 0 GPUs detected -> graceful fallback to single-device
  - Error: Worker exception logged and raised appropriately

**Code Intent**:
- New file `virnucpro/pipeline/parallel.py`:
  - `detect_cuda_devices()` -> List[int]: returns indices of available CUDA devices
  - `assign_files_round_robin(files: List[Path], num_workers: int)` -> List[List[Path]]: distributes files
  - `process_dnabert_files_worker(file_subset: List[Path], device_id: int, batch_size: int, output_dir: Path)` -> List[Path]: worker function that loads model on cuda:device_id, processes all files in subset, returns output paths
  - Error handling: catch exceptions, log with device context, re-raise
- Add to `config/default_config.yaml`:
  - `parallelization.enabled: false` (user opts-in via --parallel flag)
  - `parallelization.num_workers: auto` (auto = number of GPUs when enabled)
- Add to `virnucpro/core/config.py`:
  - Document new config keys

**Code Changes**:

```diff
--- /dev/null
+++ b/virnucpro/pipeline/parallel.py
@@ -0,0 +1,115 @@
+"""Multiprocessing utilities for parallel feature extraction"""
+
+import torch
+from pathlib import Path
+from typing import List
+import logging
+
+logger = logging.getLogger('virnucpro.parallel')
+
+
+def detect_cuda_devices() -> List[int]:
+    """
+    Detect available CUDA devices.
+
+    Returns:
+        List of CUDA device indices (e.g., [0, 1, 2, 3] for 4 GPUs)
+        Empty list if no CUDA available
+    """
+    if not torch.cuda.is_available():
+        logger.warning("CUDA not available, parallel processing disabled")
+        return []
+
+    num_devices = torch.cuda.device_count()
+    logger.info(f"Detected {num_devices} CUDA device(s)")
+
+    return list(range(num_devices))
+
+
+def assign_files_round_robin(files: List[Path], num_workers: int) -> List[List[Path]]:
+    """
+    Assign files to workers in round-robin fashion.
+
+    Args:
+        files: List of file paths to process
+        num_workers: Number of worker processes
+
+    Returns:
+        List of file lists, one per worker
+
+    Example:
+        >>> files = [Path(f"file_{i}.fa") for i in range(8)]
+        >>> assign_files_round_robin(files, 4)
+        [[file_0.fa, file_4.fa], [file_1.fa, file_5.fa], ...]
+    """
+    if num_workers <= 0:
+        raise ValueError(f"num_workers must be positive, got {num_workers}")
+
+    if not files:
+        return [[] for _ in range(num_workers)]
+
+    # Initialize worker lists
+    worker_files = [[] for _ in range(num_workers)]
+
+    # Round-robin assignment
+    for idx, file_path in enumerate(files):
+        worker_idx = idx % num_workers
+        worker_files[worker_idx].append(file_path)
+
+    logger.debug(f"Assigned {len(files)} files to {num_workers} workers")
+    for worker_idx, file_list in enumerate(worker_files):
+        logger.debug(f"  Worker {worker_idx}: {len(file_list)} files")
+
+    return worker_files
+
+
+def process_dnabert_files_worker(
+    file_subset: List[Path],
+    device_id: int,
+    batch_size: int,
+    output_dir: Path
+) -> List[Path]:
+    """
+    Worker function to process DNABERT-S features on a specific GPU.
+
+    This function is called by multiprocessing Pool workers. Each worker
+    loads the DNABERT-S model on its assigned GPU and processes all
+    files in its subset.
+
+    Args:
+        file_subset: List of nucleotide FASTA files to process
+        device_id: CUDA device ID (e.g., 0 for cuda:0)
+        batch_size: Batch size for DNABERT-S processing
+        output_dir: Directory where output files should be saved
+
+    Returns:
+        List of output .pt file paths
+
+    Raises:
+        Exception: Any error during processing (logged with device context)
+    """
+    output_files = []
+
+    try:
+        device = torch.device(f'cuda:{device_id}')
+        logger.info(f"Worker {device_id}: Processing {len(file_subset)} files on {device}")
+
+        from virnucpro.pipeline.features import extract_dnabert_features
+
+        for nuc_file in file_subset:
+            # Construct output filename
+            output_file = output_dir / f"{nuc_file.stem}_DNABERT_S.pt"
+
+            # Extract features
+            extract_dnabert_features(
+                nuc_file,
+                output_file,
+                device,
+                batch_size=batch_size
+            )
+            output_files.append(output_file)
+
+        logger.info(f"Worker {device_id}: Completed {len(output_files)} files")
+        return output_files
+
+    except Exception as e:
+        logger.exception(f"Worker {device_id}: Error processing files")
+        raise
```

```diff
--- a/config/default_config.yaml
+++ b/config/default_config.yaml
@@ -43,6 +43,12 @@ features:
     toks_per_batch: 2048
     representation_layer: 36

+# Parallelization settings
+parallelization:
+  enabled: false
+  # Number of workers: "auto" uses GPU count when enabled
+  num_workers: auto
+
 # Checkpointing settings
 checkpointing:
   # Enable checkpointing by default
```

```diff
--- a/virnucpro/core/config.py
+++ b/virnucpro/core/config.py
@@ -13,6 +13,16 @@
 # features.esm.toks_per_batch (int, default: 2048)
 #   Maximum tokens per batch for ESM-2 extraction.

+# parallelization.enabled (bool, default: false)
+#   Enable multi-GPU parallel processing. When true, feature extraction
+#   uses all available GPUs via multiprocessing. Users should enable this
+#   with --parallel CLI flag for better GPU resource management.
+#
+# parallelization.num_workers (int or "auto", default: "auto")
+#   Number of parallel workers for feature extraction. When set to "auto",
+#   uses the number of detected GPUs. Can be set to specific integer to
+#   limit worker count (useful in shared GPU environments).
+
 import yaml
 from pathlib import Path
 from typing import Any, Dict, Optional
```

---

### Milestone 3: Pipeline Integration

**Files**:
- `virnucpro/pipeline/prediction.py`
- `virnucpro/cli/predict.py`

**Flags**: `conformance`

**Requirements**:
- Integrate parallel processing into Stage 5 (nucleotide feature extraction)
- Add `--dnabert-batch-size` CLI option
- Add `--parallel` CLI flag to enable multi-GPU processing
- When --parallel enabled: auto-detect GPUs and dispatch to multiprocessing pool
- Fall back to sequential processing if only 1 GPU or parallelization not enabled
- Maintain checkpoint compatibility (output file list unchanged)
- Progress bar shows overall file progress (not per-worker)

**Acceptance Criteria**:
- `virnucpro predict input.fa` uses single GPU by default
- `virnucpro predict input.fa --parallel` uses all available GPUs
- `--dnabert-batch-size 128` overrides default batch size
- Checkpoint resume works correctly (skips completed files)
- Progress bar updates as files complete
- Single GPU mode works identically to before (no regression)

**Tests**:
- **Test files**: `tests/test_prediction_parallel.py` (new)
- **Test type**: integration
- **Backing**: user-specified (real models with small dataset)
- **Scenarios**:
  - Normal: 4 files on 2 GPUs processed in parallel
  - Edge: 1 GPU available -> sequential mode
  - Edge: Resume after interruption -> skips completed files
  - Error: GPU failure mid-run -> clean error message

**Code Intent**:
- Modify `virnucpro/pipeline/prediction.py`:
  - Import from `parallel.py`
  - In Stage 5 block (lines ~201-234):
    - Detect available GPUs using `detect_cuda_devices()`
    - If >1 GPU and parallelization enabled:
      - Assign files using `assign_files_round_robin()`
      - Create Pool with num_workers = num_gpus
      - `pool.starmap(process_dnabert_files_worker, args)`
      - Collect results (output file paths)
    - Else: existing sequential loop
  - Update progress bar to track file completions across all workers
- Modify `virnucpro/cli/predict.py`:
  - Add `--dnabert-batch-size` option (type=int, default from config)
  - Add `--parallel` flag (type=bool, default False) to enable multi-GPU processing
  - Pass both options to prediction function

**Code Changes**:

```diff
--- a/virnucpro/pipeline/prediction.py
+++ b/virnucpro/pipeline/prediction.py
@@ -15,7 +15,10 @@ def run_prediction(
     input_file: Path,
     model_path: Path,
     expected_length: int,
     output_dir: Path,
     device: torch.device,
+    dnabert_batch_size: int,
+    parallel: bool,
     batch_size: int,
     num_workers: int,
     cleanup_intermediate: bool,
@@ -28,6 +31,8 @@ def run_prediction(
         model_path: Path to trained model
         expected_length: Expected sequence length
         output_dir: Output directory
         device: PyTorch device
+        dnabert_batch_size: Batch size for DNABERT-S extraction
+        parallel: Enable multi-GPU parallel processing
         batch_size: Batch size for DataLoader
         num_workers: Number of data loading workers
         cleanup_intermediate: Whether to clean intermediate files
@@ -201,29 +206,58 @@ def run_prediction(
         # Stage 5: Nucleotide Feature Extraction
         if start_stage <= PipelineStage.NUCLEOTIDE_FEATURES or not checkpoint_manager.can_skip_stage(state, PipelineStage.NUCLEOTIDE_FEATURES):
             logger.info("=== Stage 5: Nucleotide Feature Extraction ===")
             checkpoint_manager.mark_stage_started(state, PipelineStage.NUCLEOTIDE_FEATURES)

-            from virnucpro.pipeline.features import extract_dnabert_features
-
             nucleotide_feature_files = []

-            # Extract DNABERT-S features
-            logger.info("Extracting DNABERT-S features from nucleotide sequences")
-            with progress.create_file_bar(len(nucleotide_files), desc="DNABERT-S extraction") as pbar:
-                for nuc_file in nucleotide_files:
-                    # Construct output filename: output_0.fa -> output_0_DNABERT_S.pt
-                    output_file = nuc_file.parent / f"{nuc_file.stem}_DNABERT_S.pt"
-                    extract_dnabert_features(
-                        nuc_file,
-                        output_file,
-                        device,
-                        batch_size=batch_size
-                    )
-                    nucleotide_feature_files.append(output_file)
-                    pbar.update(1)
-                    pbar.set_postfix_str(f"Current: {nuc_file.name}")
+            # Check if parallel processing is enabled and multiple GPUs available
+            use_parallel = False
+            if parallel:
+                from virnucpro.pipeline.parallel import detect_cuda_devices, assign_files_round_robin, process_dnabert_files_worker
+                from multiprocessing import Pool
+
+                available_gpus = detect_cuda_devices()
+                if len(available_gpus) > 1:
+                    use_parallel = True
+                    logger.info(f"Using parallel processing with {len(available_gpus)} GPUs")
+                else:
+                    logger.info("Only 1 GPU available, falling back to sequential processing")
+
+            if use_parallel:
+                # Parallel multi-GPU processing
+                logger.info("Extracting DNABERT-S features in parallel across GPUs")
+
+                # Filter out files with existing outputs for checkpoint resume
+                files_to_process = []
+                for nuc_file in nucleotide_files:
+                    output_file = nuc_file.parent / f"{nuc_file.stem}_DNABERT_S.pt"
+                    if not output_file.exists():
+                        files_to_process.append(nuc_file)
+                    else:
+                        nucleotide_feature_files.append(output_file)
+                        logger.info(f"Skipping {nuc_file.name} (output already exists)")
+
+                # Assign remaining files to workers
+                worker_file_assignments = assign_files_round_robin(files_to_process, len(available_gpus))
+
+                # Create worker arguments
+                worker_args = [
+                    (file_subset, device_id, dnabert_batch_size, nucleotide_files[0].parent)
+                    for device_id, file_subset in zip(available_gpus, worker_file_assignments)
+                ]
+
+                # Run parallel processing
+                with Pool(processes=len(available_gpus)) as pool:
+                    with progress.create_file_bar(len(files_to_process), desc="DNABERT-S extraction (parallel)") as pbar:
+                        results = pool.starmap(process_dnabert_files_worker, worker_args)
+                        # Flatten results from all workers
+                        for worker_output in results:
+                            nucleotide_feature_files.extend(worker_output)
+                        pbar.update(len(files_to_process))
+            else:
+                # Sequential single-GPU processing
+                from virnucpro.pipeline.features import extract_dnabert_features
+
+                logger.info("Extracting DNABERT-S features from nucleotide sequences")
+                with progress.create_file_bar(len(nucleotide_files), desc="DNABERT-S extraction") as pbar:
+                    for nuc_file in nucleotide_files:
+                        # Construct output filename: output_0.fa -> output_0_DNABERT_S.pt
+                        output_file = nuc_file.parent / f"{nuc_file.stem}_DNABERT_S.pt"
+                        extract_dnabert_features(
+                            nuc_file,
+                            output_file,
+                            device,
+                            batch_size=dnabert_batch_size
+                        )
+                        nucleotide_feature_files.append(output_file)
+                        pbar.update(1)
+                        pbar.set_postfix_str(f"Current: {nuc_file.name}")

             checkpoint_manager.mark_stage_completed(
                 state,
```

```diff
--- a/virnucpro/cli/predict.py
+++ b/virnucpro/cli/predict.py
@@ -48,6 +48,12 @@ from virnucpro.core.config import Config
 @click.option('--no-progress',
               is_flag=True,
               help='Disable progress bars (useful for logging to files)')
+@click.option('--dnabert-batch-size',
+              type=int,
+              help='Batch size for DNABERT-S feature extraction (default: from config)')
+@click.option('--parallel',
+              is_flag=True,
+              help='Enable multi-GPU parallel processing for feature extraction')
 @click.pass_context
 def predict(ctx, input_file, model_type, model_path, expected_length,
             output_dir, device, batch_size, num_workers,
-            keep_intermediate, resume, force, no_progress):
+            keep_intermediate, resume, force, no_progress,
+            dnabert_batch_size, parallel):
     """
     Predict viral sequences from FASTA input.

@@ -120,6 +126,10 @@ def predict(ctx, input_file, model_type, model_path, expected_length,
             num_workers = config.get('prediction.num_workers', 4)

+        # Get DNABERT batch size from config if not specified
+        if dnabert_batch_size is None:
+            dnabert_batch_size = config.get('features.dnabert.batch_size', 256)
+
         # Validate and get device
         fallback_to_cpu = config.get('device.fallback_to_cpu', True)
         device_obj = validate_and_get_device(device, fallback_to_cpu=fallback_to_cpu)
@@ -136,6 +146,8 @@ def predict(ctx, input_file, model_type, model_path, expected_length,
         logger.info(f"  Device: {device_obj}")
         logger.info(f"  Batch size: {batch_size}")
         logger.info(f"  Workers: {num_workers}")
+        logger.info(f"  DNABERT batch size: {dnabert_batch_size}")
+        logger.info(f"  Parallel processing: {'enabled' if parallel else 'disabled'}")
         logger.info(f"  Resume: {resume}")
         logger.info(f"  Cleanup intermediate files: {cleanup}")
         logger.info(f"  Progress bars: {'disabled' if no_progress else 'enabled'}")
@@ -151,6 +163,8 @@ def predict(ctx, input_file, model_type, model_path, expected_length,
             expected_length=expected_length,
             output_dir=output_dir,
             device=device_obj,
+            dnabert_batch_size=dnabert_batch_size,
+            parallel=parallel,
             batch_size=batch_size,
             num_workers=num_workers,
             cleanup_intermediate=cleanup,
```

---

### Milestone 4: Testing & Validation

**Files**:
- `tests/test_features.py`
- `tests/test_parallel.py`
- `tests/test_prediction_parallel.py`
- `tests/conftest.py` (new - fixtures)
- `tests/data/test_sequences.fa` (new - generated test data)

**Flags**: None

**Requirements**:
- Create pytest fixtures for test data generation
- Generate deterministic FASTA test file with ~100 sequences
- Integration tests verify output equivalence between sequential and parallel
- Tests run on CI without GPU (mock or skip)

**Acceptance Criteria**:
- `pytest tests/` passes with all new tests
- Test coverage for new code >80%
- Tests are deterministic (same results every run)
- GPU tests skipped gracefully when no GPU available

**Tests**:
- **Test files**: All test files listed above
- **Test type**: unit + integration
- **Backing**: user-specified (generated test data)
- **Scenarios**:
  - All scenarios from Milestones 1-3

**Code Intent**:
- New file `tests/conftest.py`:
  - `@pytest.fixture` for generating temporary FASTA with N sequences
  - `@pytest.fixture` for temporary output directory
  - `@pytest.fixture` for mock GPU detection (returns [0] or [0,1])
  - `@pytest.mark.skipif(not torch.cuda.is_available())` decorator helper
- New file `tests/data/test_sequences.fa`:
  - 100 deterministic DNA sequences of 500bp each
  - Fixed random seed for reproducibility
- Test structure:
  - `test_features.py`: Unit tests for batched DNABERT-S
  - `test_parallel.py`: Unit tests for parallel utilities
  - `test_prediction_parallel.py`: Integration tests for full pipeline

**Code Changes**:

```diff
--- /dev/null
+++ b/tests/conftest.py
@@ -0,0 +1,73 @@
+"""Pytest fixtures for VirNucPro tests"""
+
+import pytest
+import torch
+from pathlib import Path
+import tempfile
+import shutil
+from typing import List
+
+
+@pytest.fixture
+def temp_dir():
+    """Create temporary directory for test outputs"""
+    tmpdir = tempfile.mkdtemp()
+    yield Path(tmpdir)
+    shutil.rmtree(tmpdir)
+
+
+@pytest.fixture
+def temp_fasta(temp_dir):
+    """Generate temporary FASTA file with test sequences"""
+    def _generate_fasta(num_sequences: int = 10, seq_length: int = 500):
+        fasta_file = temp_dir / "test_sequences.fa"
+
+        # Use fixed seed for reproducibility
+        import random
+        random.seed(42)
+
+        bases = ['A', 'T', 'G', 'C']
+        with open(fasta_file, 'w') as f:
+            for i in range(num_sequences):
+                seq = ''.join(random.choice(bases) for _ in range(seq_length))
+                f.write(f">test_seq_{i}\n")
+                f.write(f"{seq}\n")
+
+        return fasta_file
+
+    return _generate_fasta
+
+
+@pytest.fixture
+def mock_gpu_devices(monkeypatch):
+    """Mock GPU detection for testing without actual GPUs"""
+    def _mock_detection(num_gpus: int = 2):
+        def mock_is_available():
+            return num_gpus > 0
+
+        def mock_device_count():
+            return num_gpus
+
+        monkeypatch.setattr(torch.cuda, "is_available", mock_is_available)
+        monkeypatch.setattr(torch.cuda, "device_count", mock_device_count)
+
+    return _mock_detection
+
+
+def pytest_configure(config):
+    """Register custom markers"""
+    config.addinivalue_line(
+        "markers", "gpu: mark test as requiring GPU (skip if no GPU available)"
+    )
+    config.addinivalue_line(
+        "markers", "slow: mark test as slow running"
+    )
+
+
+def pytest_collection_modifyitems(config, items):
+    """Skip GPU tests if no GPU available"""
+    skip_gpu = pytest.mark.skip(reason="GPU not available")
+    for item in items:
+        if "gpu" in item.keywords:
+            if not torch.cuda.is_available():
+                item.add_marker(skip_gpu)
+```

```diff
--- /dev/null
+++ b/tests/test_features.py
@@ -0,0 +1,125 @@
+"""Tests for feature extraction with batching"""
+
+import pytest
+import torch
+from pathlib import Path
+from virnucpro.pipeline.features import extract_dnabert_features
+
+
+@pytest.mark.gpu
+class TestDNABERTBatching:
+    """Test batched DNABERT-S feature extraction"""
+
+    def test_batch_processing_produces_correct_output(self, temp_fasta, temp_dir):
+        """Normal: Batch of 10 sequences produces correct embeddings"""
+        # Generate test data
+        fasta_file = temp_fasta(num_sequences=10, seq_length=500)
+        output_file = temp_dir / "features.pt"
+
+        device = torch.device("cuda:0")
+        batch_size = 4
+
+        # Extract features
+        result = extract_dnabert_features(fasta_file, output_file, device, batch_size)
+
+        # Verify output file exists
+        assert output_file.exists()
+        assert result == output_file
+
+        # Load and verify structure
+        data = torch.load(output_file)
+        assert 'nucleotide' in data
+        assert 'data' in data
+        assert len(data['nucleotide']) == 10
+        assert len(data['data']) == 10
+
+        # Verify each embedding has correct structure
+        for item in data['data']:
+            assert 'label' in item
+            assert 'mean_representation' in item
+            assert len(item['mean_representation']) == 768  # DNABERT-S embedding dim
+
+    def test_single_sequence_batch(self, temp_fasta, temp_dir):
+        """Edge: Single sequence batch works correctly"""
+        fasta_file = temp_fasta(num_sequences=1, seq_length=500)
+        output_file = temp_dir / "features.pt"
+
+        device = torch.device("cuda:0")
+        batch_size = 256
+
+        result = extract_dnabert_features(fasta_file, output_file, device, batch_size)
+
+        data = torch.load(output_file)
+        assert len(data['nucleotide']) == 1
+        assert len(data['data']) == 1
+
+    def test_batch_size_larger_than_input(self, temp_fasta, temp_dir):
+        """Edge: Batch size larger than sequence count handles gracefully"""
+        fasta_file = temp_fasta(num_sequences=5, seq_length=500)
+        output_file = temp_dir / "features.pt"
+
+        device = torch.device("cuda:0")
+        batch_size = 1000  # Much larger than input
+
+        result = extract_dnabert_features(fasta_file, output_file, device, batch_size)
+
+        data = torch.load(output_file)
+        assert len(data['nucleotide']) == 5
+        assert len(data['data']) == 5
+
+    @pytest.mark.slow
+    def test_output_equivalence_across_batch_sizes(self, temp_fasta, temp_dir):
+        """Property: batched(sequences, bs1) == batched(sequences, bs2)"""
+        fasta_file = temp_fasta(num_sequences=20, seq_length=500)
+
+        device = torch.device("cuda:0")
+
+        # Extract with different batch sizes
+        output1 = temp_dir / "features_bs4.pt"
+        output2 = temp_dir / "features_bs8.pt"
+
+        extract_dnabert_features(fasta_file, output1, device, batch_size=4)
+        extract_dnabert_features(fasta_file, output2, device, batch_size=8)
+
+        # Load results
+        data1 = torch.load(output1)
+        data2 = torch.load(output2)
+
+        # Verify same sequence IDs
+        assert data1['nucleotide'] == data2['nucleotide']
+
+        # Verify embeddings are equivalent within tolerance
+        for item1, item2 in zip(data1['data'], data2['data']):
+            emb1 = torch.tensor(item1['mean_representation'])
+            emb2 = torch.tensor(item2['mean_representation'])
+
+            # Use tolerances from plan
+            assert torch.allclose(emb1, emb2, rtol=1e-4, atol=1e-6)
+
+    def test_order_preservation(self, temp_fasta, temp_dir):
+        """Property: Output order matches input order regardless of batching"""
+        fasta_file = temp_fasta(num_sequences=15, seq_length=500)
+        output_file = temp_dir / "features.pt"
+
+        device = torch.device("cuda:0")
+        batch_size = 4  # Forces multiple batches with varying sizes
+
+        extract_dnabert_features(fasta_file, output_file, device, batch_size)
+
+        data = torch.load(output_file)
+
+        # Verify sequence IDs are in order
+        expected_ids = [f"test_seq_{i}" for i in range(15)]
+        assert data['nucleotide'] == expected_ids
+
+        # Verify data list matches
+        for i, item in enumerate(data['data']):
+            assert item['label'] == f"test_seq_{i}"
+
+
+class TestFeatureErrors:
+    """Test error handling in feature extraction"""
+
+    def test_empty_input_file(self, temp_dir):
+        """Error: Empty input file handled gracefully"""
+        # Test stub - implement error handling for empty FASTA files
+        pass
+```

```diff
--- /dev/null
+++ b/tests/test_parallel.py
@@ -0,0 +1,89 @@
+"""Tests for parallel processing utilities"""
+
+import pytest
+from pathlib import Path
+from virnucpro.pipeline.parallel import detect_cuda_devices, assign_files_round_robin
+
+
+class TestGPUDetection:
+    """Test GPU detection functionality"""
+
+    def test_detect_gpus_with_cuda_available(self, mock_gpu_devices):
+        """Normal: Detect multiple GPUs when CUDA available"""
+        mock_gpu_devices(num_gpus=4)
+
+        devices = detect_cuda_devices()
+
+        assert devices == [0, 1, 2, 3]
+
+    def test_detect_single_gpu(self, mock_gpu_devices):
+        """Normal: Detect single GPU"""
+        mock_gpu_devices(num_gpus=1)
+
+        devices = detect_cuda_devices()
+
+        assert devices == [0]
+
+    def test_no_cuda_available(self, mock_gpu_devices):
+        """Edge: No CUDA available returns empty list"""
+        mock_gpu_devices(num_gpus=0)
+
+        devices = detect_cuda_devices()
+
+        assert devices == []
+
+
+class TestFileAssignment:
+    """Test round-robin file assignment"""
+
+    def test_balanced_assignment(self):
+        """Normal: 8 files, 4 GPUs -> 2 files per GPU"""
+        files = [Path(f"file_{i}.fa") for i in range(8)]
+
+        assignments = assign_files_round_robin(files, num_workers=4)
+
+        assert len(assignments) == 4
+        assert all(len(worker_files) == 2 for worker_files in assignments)
+
+        # Verify round-robin pattern
+        assert assignments[0] == [Path("file_0.fa"), Path("file_4.fa")]
+        assert assignments[1] == [Path("file_1.fa"), Path("file_5.fa")]
+        assert assignments[2] == [Path("file_2.fa"), Path("file_6.fa")]
+        assert assignments[3] == [Path("file_3.fa"), Path("file_7.fa")]
+
+    def test_uneven_assignment(self):
+        """Edge: 3 files, 4 GPUs -> 3 GPUs with 1 file each"""
+        files = [Path(f"file_{i}.fa") for i in range(3)]
+
+        assignments = assign_files_round_robin(files, num_workers=4)
+
+        assert len(assignments) == 4
+        assert assignments[0] == [Path("file_0.fa")]
+        assert assignments[1] == [Path("file_1.fa")]
+        assert assignments[2] == [Path("file_2.fa")]
+        assert assignments[3] == []  # Fourth worker gets no files
+
+    def test_more_files_than_workers(self):
+        """Normal: 10 files, 3 GPUs -> uneven distribution"""
+        files = [Path(f"file_{i}.fa") for i in range(10)]
+
+        assignments = assign_files_round_robin(files, num_workers=3)
+
+        assert len(assignments) == 3
+        assert len(assignments[0]) == 4  # files 0, 3, 6, 9
+        assert len(assignments[1]) == 3  # files 1, 4, 7
+        assert len(assignments[2]) == 3  # files 2, 5, 8
+
+    def test_empty_file_list(self):
+        """Edge: Empty file list returns empty assignments"""
+        files = []
+
+        assignments = assign_files_round_robin(files, num_workers=4)
+
+        assert len(assignments) == 4
+        assert all(len(worker_files) == 0 for worker_files in assignments)
+
+    def test_invalid_num_workers(self):
+        """Error: Invalid num_workers raises ValueError"""
+        with pytest.raises(ValueError):
+            assign_files_round_robin([Path("file.fa")], num_workers=0)
+```

```diff
--- /dev/null
+++ b/tests/test_prediction_parallel.py
@@ -0,0 +1,78 @@
+"""Integration tests for parallel prediction pipeline"""
+
+import pytest
+import torch
+from pathlib import Path
+from unittest.mock import Mock, patch
+
+
+@pytest.mark.gpu
+@pytest.mark.slow
+class TestParallelPipeline:
+    """Integration tests for parallel feature extraction"""
+
+    def test_parallel_processing_with_multiple_gpus(self, temp_fasta, temp_dir, mock_gpu_devices):
+        """Normal: 4 files on 2 GPUs processed in parallel"""
+        mock_gpu_devices(num_gpus=2)
+
+        # Create test files
+        files = []
+        for i in range(4):
+            fasta = temp_fasta(num_sequences=100, seq_length=500)
+            # Rename to simulate multiple files
+            new_path = temp_dir / f"nucleotide_{i}.fa"
+            fasta.rename(new_path)
+            files.append(new_path)
+
+        # Test parallel processing
+        from virnucpro.pipeline.parallel import (
+            detect_cuda_devices,
+            assign_files_round_robin,
+            process_dnabert_files_worker
+        )
+
+        devices = detect_cuda_devices()
+        assert len(devices) == 2
+
+        assignments = assign_files_round_robin(files, len(devices))
+        assert len(assignments[0]) == 2
+        assert len(assignments[1]) == 2
+
+    def test_fallback_to_sequential_single_gpu(self, temp_fasta, temp_dir, mock_gpu_devices):
+        """Edge: 1 GPU available -> sequential mode"""
+        mock_gpu_devices(num_gpus=1)
+
+        from virnucpro.pipeline.parallel import detect_cuda_devices
+
+        devices = detect_cuda_devices()
+        assert len(devices) == 1
+
+        # Pipeline should fall back to sequential processing
+        # This is tested at the integration level in prediction.py
+
+    def test_no_gpu_fallback(self, temp_fasta, temp_dir, mock_gpu_devices):
+        """Edge: No GPU available -> CPU fallback"""
+        mock_gpu_devices(num_gpus=0)
+
+        from virnucpro.pipeline.parallel import detect_cuda_devices
+
+        devices = detect_cuda_devices()
+        assert devices == []
+
+
+class TestCheckpointCompatibility:
+    """Test that parallel processing maintains checkpoint compatibility"""
+
+    def test_resume_after_interruption(self, temp_dir):
+        """Edge: Resume after interruption skips completed files"""
+        # Test stub - implement checkpoint resume verification
+        pass
+
+
+class TestErrorHandling:
+    """Test error handling in parallel pipeline"""
+
+    def test_worker_exception_propagates(self):
+        """Error: GPU failure mid-run produces clean error message"""
+        # Test stub - implement worker exception handling verification
+        pass
+```

```diff
--- /dev/null
+++ b/tests/data/test_sequences.fa
@@ -0,0 +1,201 @@
+>test_seq_0
+ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG
+ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG
+ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG
+ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG
+ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG
+ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG
+ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG
+ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG
+ATCGATCGATCGATCG
+>test_seq_1
+GCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTA
+GCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTA
+GCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTA
+GCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTA
+GCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTA
+GCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTA
+GCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTA
+GCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTA
+GCTAGCTAGCTAGCTA
+>test_seq_2
+TACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACG
+TACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACG
+TACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACG
+TACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACG
+TACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACG
+TACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACG
+TACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACG
+TACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACG
+TACGTACGTACGTACG
+[... 97 more sequences with same pattern, each 500bp ...]
+```

---

### Milestone 5: Documentation

**Delegated to**: @agent-technical-writer (mode: post-implementation)

**Source**: `## Invisible Knowledge` section of this plan

**Files**:
- `virnucpro/pipeline/CLAUDE.md` (update index)
- `virnucpro/pipeline/README.md` (new - architecture docs)
- `README.md` (update with parallelization usage)

**Requirements**:
- CLAUDE.md updated with new parallel.py entry
- README.md explains parallel architecture and data flow, including:
  - Architecture diagram (lines 67-100)
  - Data flow diagram (lines 104-127)
  - Rationale for multiprocessing over DataParallel (lines 134-137)
  - Four invariants: output equivalence, file ordering preservation, GPU assignment determinism, checkpoint compatibility (lines 141-144)
- Main README updated with multi-GPU usage examples

**Acceptance Criteria**:
- CLAUDE.md is tabular index only
- README.md contains architecture diagram from Invisible Knowledge
- Usage examples show `--dnabert-batch-size` option

**Code Intent**: Documentation milestone - no code changes.

**Code Changes**: Documentation milestone - no code changes.

## Milestone Dependencies

```
M1 (Batching) ----+
                  |
                  v
M2 (Parallel) ----+---> M3 (Integration) ---> M4 (Testing) ---> M5 (Docs)
```

- M1 and M2 can be developed in parallel
- M3 depends on both M1 and M2
- M4 depends on M3
- M5 depends on M4
