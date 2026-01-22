# GPU Optimization Pitfalls

Research for multi-GPU optimization of VirNucPro's embedding pipeline (ESM-2 3B + DNABERT-S).

## Critical Pitfalls

### 1. CUDA Context Initialization in Multiprocessing

**What Goes Wrong:**
When using `multiprocessing` with CUDA, initialization of CUDA in the parent process before spawning workers causes "Cannot re-initialize CUDA in forked subprocess" errors, even when using `spawn` context.

**Why It Happens:**
- `torch.cuda.is_available()` counts as CUDA initialization
- `torch.manual_seed()` can trigger CUDA context creation
- Model loading or device checks in parent process contaminate workers
- Fork context is completely incompatible with CUDA (but spawn can still fail)

**Warning Signs:**
```python
RuntimeError: Cannot re-initialize CUDA in forked subprocess
CUDA error: initialization error (multiprocessing)
```

**VirNucPro Specifics:**
- Current code in `parallel.py` uses `spawn` correctly (line 245)
- Risk: Device detection in `detect_cuda_devices()` might initialize context
- Risk: Any `device` parameter validation before multiprocessing starts

**Prevention Strategy:**
1. Move ALL CUDA operations inside worker functions
2. Defer device detection until after spawn (or make it CUDA-free)
3. Pass device IDs as integers, not torch.device objects
4. Avoid `torch.cuda.is_available()` in parent process
5. Use `if __name__ == '__main__':` guard

**Code Pattern to Avoid:**
```python
# BAD - in parent process before spawn
device = torch.device('cuda:0')  # Initializes CUDA!
torch.cuda.is_available()  # Initializes CUDA!
model.to(device)  # Initializes CUDA!

# Start multiprocessing...
```

**Code Pattern to Use:**
```python
# GOOD - parent process stays CUDA-free
available_gpus = list(range(torch.cuda.device_count()))  # Uses C API, safe

# Worker function initializes CUDA
def worker(device_id):
    device = torch.device(f'cuda:{device_id}')  # CUDA init happens here
    model = load_model().to(device)
```

**Which Phase Should Address:**
- Phase 1 (ESM-2 parallelization): Critical to get right from start
- Refactoring existing DNABERT-S code if issues detected

**References:**
- [PyTorch Issue #40403: Cannot re-initialize CUDA in forked subprocess](https://github.com/pytorch/pytorch/issues/40403)
- [PyTorch Forums: spawn start method](https://discuss.pytorch.org/t/unable-to-fix-runtimeerror-cannot-re-initialize-cuda-in-forked-subprocess-to-use-cuda-with-multiprocessing-you-must-use-the-spawn-start-method/208718)

---

### 2. Batch Size Variability Causing Memory Fragmentation

**What Goes Wrong:**
When batch sizes vary across iterations (common in inference with variable-length sequences), CUDA memory becomes fragmented even if total memory usage seems fine. This leads to OOM errors despite having "enough" VRAM.

**Why It Happens:**
- PyTorch's caching allocator creates fixed-size blocks
- Varying allocations create "holes" in memory
- Allocator can't coalesce fragmented regions efficiently
- Exception handling keeps references to stack frames, preventing memory release

**Warning Signs:**
```python
RuntimeError: CUDA out of memory (but nvidia-smi shows available memory)
# Memory usage pattern: 7.2GB → 6.8GB → 7.5GB → 8.1GB → OOM
# Fragment count increases over time
```

**VirNucPro Specifics:**
- ESM-2 uses dynamic batching by token count (`toks_per_batch=2048`)
- Protein sequences vary widely in length (post-ORF detection)
- Each batch has different size: 1 long sequence OR 50 short sequences
- This creates worst-case fragmentation scenario

**Prevention Strategy:**
1. **Enable expandable segments** (PyTorch 2.0+):
   ```python
   os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
   ```
2. **Sort sequences by length** before batching to reduce variability
3. **Wrap inference in `torch.no_grad()`** (already done, verify)
4. **Explicit cache clearing** between file batches:
   ```python
   torch.cuda.empty_cache()  # After processing each file
   ```
5. **Move OOM recovery OUTSIDE except clause** to prevent tensor references

**Code Pattern to Avoid:**
```python
# BAD - variable batching without safeguards
for batch in dynamic_batches:  # Sizes: 512, 1024, 256, 2048, ...
    output = model(batch)  # Fragmentation accumulates
    # No cleanup
```

**Code Pattern to Use:**
```python
# GOOD - sorted batching with cleanup
sequences.sort(key=lambda x: len(x[1]))  # Sort by length
for i, batch in enumerate(dynamic_batches):
    with torch.no_grad():
        output = model(batch)

    if i % 10 == 0:  # Periodic cleanup
        torch.cuda.empty_cache()
```

**Which Phase Should Address:**
- Phase 1: ESM-2 sequence sorting before batching
- Phase 2: Environment variable setting in initialization
- Phase 3: Monitoring/logging to detect fragmentation

**References:**
- [Saturn Cloud: CUDA out of memory solutions](https://saturncloud.io/blog/how-to-solve-cuda-out-of-memory-in-pytorch/)
- [PyTorch Docs: CUDA semantics](https://docs.pytorch.org/docs/stable/notes/cuda.html)

---

### 3. Checkpoint Corruption on Multi-GPU Resume

**What Goes Wrong:**
When resuming from checkpoints in multi-GPU inference, file existence checks pass but files are empty or partially written. Loading corrupt checkpoints crashes without useful error messages. In distributed scenarios, rank-0 saves checkpoint but other ranks attempt to load before write completes.

**Why It Happens:**
- Process interruption during `torch.save()` leaves 0-byte files
- No atomic writes by default (partial file exists)
- No file size or content validation before loading
- Race conditions: multiple processes checking same file
- Distributed: ranks don't synchronize on checkpoint writes

**Warning Signs:**
```python
RuntimeError: [Errno 9] Bad file descriptor  # Empty .pt file
EOFError: Ran out of input  # Truncated checkpoint
IndexError: list index out of range  # Expected data missing
```

**VirNucPro Specifics:**
- Current `CheckpointManager` checks file existence only (line 83-102)
- DNABERT-S parallel workers skip files if output exists (prediction.py:231-234)
- No file size validation → 0-byte files treated as valid
- ESM-2 not yet parallelized → will inherit this risk

**Prevention Strategy:**
1. **Validate file size before loading**:
   ```python
   if path.exists() and path.stat().st_size > MIN_EXPECTED_SIZE:
       try:
           data = torch.load(path)
       except Exception as e:
           logger.warning(f"Corrupt checkpoint {path}, regenerating: {e}")
           path.unlink()  # Delete corrupt file
           data = None
   ```

2. **Atomic writes with temp file + rename**:
   ```python
   temp_path = output_file.with_suffix('.tmp')
   torch.save(data, temp_path)
   temp_path.replace(output_file)  # Atomic on POSIX
   ```

3. **Checkpoint metadata with hash validation**:
   ```python
   checkpoint = {
       'data': features,
       'checksum': hashlib.sha256(pickle.dumps(features)).hexdigest(),
       'size': len(features)
   }
   ```

4. **Distributed: Only rank 0 writes, all ranks synchronize**:
   ```python
   if rank == 0:
       torch.save(checkpoint, path)
   torch.distributed.barrier()  # Wait for write completion
   ```

**Code Pattern Currently Used:**
```python
# RISKY - from prediction.py:268
if not output_file.exists() or output_file.stat().st_size == 0:
    files_to_process.append(nuc_file)
```
This is already better than most, but still vulnerable to truncated files > 0 bytes.

**Code Pattern to Use:**
```python
# ROBUST
def is_valid_checkpoint(path, expected_keys=None):
    if not path.exists():
        return False
    if path.stat().st_size < 100:  # Too small to be valid
        return False
    try:
        data = torch.load(path)
        if expected_keys:
            return all(k in data for k in expected_keys)
        return True
    except Exception as e:
        logger.warning(f"Corrupt checkpoint {path}: {e}")
        return False
```

**Which Phase Should Address:**
- Phase 1: Add validation to ESM-2 parallelization from start
- Phase 2: Retrofit DNABERT-S worker with atomic writes
- Phase 3: Add checkpoint metadata and hash validation

**References:**
- [PyTorch DDP checkpointing best practices](https://discuss.pytorch.org/t/what-is-the-proper-way-to-checkpoint-during-training-when-using-distributed-data-parallel-ddp-in-pytorch/139575)
- [Distributed checkpoint fault tolerance](https://docs.pytorch.org/tutorials/beginner/ddp_series_fault_tolerance.html)

---

### 4. ESM-2 3B Model Size vs. Data Parallelism Strategy

**What Goes Wrong:**
Attempting naive data parallelism (DataParallel or DDP) with ESM-2 3B model fails because:
- Model replication requires 6GB × N_GPUs VRAM (4 GPUs = 24GB just for models)
- Gradient synchronization overhead (even with `requires_grad=False`, some frameworks sync)
- Master GPU bottleneck in DataParallel (all results gathered to GPU 0)

**Why It Happens:**
- ESM-2 3B model is 5.7GB in FP32 (3.2GB in FP16)
- Each GPU needs full model copy in data parallelism
- Inference still allocates gradient buffers in some scenarios
- DataParallel synchronizes even when unnecessary

**Warning Signs:**
```python
RuntimeError: CUDA out of memory (during model.to(device))
# OR
# GPU 0: 7.8GB, GPU 1: 7.8GB, GPU 2: 7.8GB → OOM on 8GB cards
# Only 1 GPU showing activity (others idle)
```

**VirNucPro Specifics:**
- ESM-2 3B model loaded via `esm.pretrained.esm2_t36_3B_UR50D()`
- Currently single-GPU only (features.py:113-115)
- DNABERT-S (1.5GB model) uses data parallelism successfully
- Users have 4-8 GPUs with varying VRAM (typically 16GB or 32GB)

**Prevention Strategy:**

**Option A: Tensor Parallelism (best for ESM-2 3B)**
- Split model across GPUs, not data
- Use Fairscale FSDP (ESM repo provides example: `esm2_infer_fairscale_fsdp_cpu_offloading.py`)
- Memory: 6GB / N_GPUs + batch memory
- Tradeoff: Complex implementation, requires NVLink for bandwidth

**Option B: File-Level Data Parallelism (simpler, VirNucPro's current approach)**
- Each GPU loads full model, processes different files
- Memory: 6GB per GPU (constant)
- Tradeoff: Doesn't help with large single files

**Option C: Pipeline Parallelism**
- Different layers on different GPUs
- Requires model surgery, not practical for ESM-2

**Recommendation for VirNucPro:**
1. **Phase 1**: Use file-level parallelism (like DNABERT-S) for ESM-2
   - Pro: Simple, reuses existing pattern, works with variable GPU counts
   - Con: Doesn't help if single protein file is huge
2. **Phase 2+**: Consider FSDP if files are too large for memory

**Code Pattern to Use:**
```python
# File-level parallelism (like DNABERT-S)
def process_esm_files_worker(file_subset, device_id, output_dir):
    device = torch.device(f'cuda:{device_id}')
    model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
    model = model.to(device)
    model.eval()

    for protein_file in file_subset:
        extract_esm_features(protein_file, output_file, device)
    return output_files

# Distribute files round-robin to GPUs
worker_args = [(files, device_id, output_dir)
               for device_id, files in enumerate(gpu_file_assignments)]
with multiprocessing.Pool() as pool:
    pool.starmap(process_esm_files_worker, worker_args)
```

**Which Phase Should Address:**
- Phase 1: Implement file-level parallelism for ESM-2
- Phase 3+: Investigate FSDP if users report memory issues

**References:**
- [ESM GitHub: FSDP CPU offloading example](https://github.com/facebookresearch/esm)
- [HuggingFace: Parallelism methods](https://huggingface.co/docs/transformers/main/perf_train_gpu_many)
- [vLLM: Tensor vs Pipeline parallelism](https://docs.vllm.ai/en/stable/serving/parallelism_scaling/)

---

### 5. Unbalanced GPU Work Distribution with Variable File Sizes

**What Goes Wrong:**
Round-robin file assignment to GPUs assumes files have similar processing times. When file sizes vary significantly (common after splitting sequences), some GPUs finish early and sit idle while others process large files, wasting GPU time.

**Why It Happens:**
- File splitting creates unequal sequence counts (last file often smaller)
- ORF detection creates variable sequence counts per file
- Round-robin doesn't account for file size: `[10k seqs, 10k seqs, 10k seqs, 2k seqs]`
- GPU 0 gets files 0,4 (12k seqs), GPU 1 gets files 1,5 (10k seqs)

**Warning Signs:**
```bash
# nvidia-smi output over time:
GPU 0: 95% | 95% | 95% | 95% | 5%   # Still working
GPU 1: 95% | 95% | 5%  | 5%  | 5%   # Finished early
GPU 2: 95% | 95% | 95% | 5%  | 5%
GPU 3: 95% | 5%  | 5%  | 5%  | 5%

# Total time determined by slowest GPU
```

**VirNucPro Specifics:**
- `assign_files_round_robin()` doesn't consider file sizes (parallel.py:29-61)
- Sequences split into 10k chunks, but last chunk varies
- After 6-frame translation, sequence counts become unpredictable
- 100k sequences → ~600k ORFs, but distribution varies per original sequence

**Prevention Strategy:**

1. **Load-balanced assignment** by file size:
   ```python
   def assign_files_balanced(files, num_workers):
       # Get file sizes (sequence counts)
       file_sizes = [(f, count_sequences(f)) for f in files]
       file_sizes.sort(key=lambda x: x[1], reverse=True)

       # Greedy assignment: assign largest file to least-loaded GPU
       worker_loads = [0] * num_workers
       worker_files = [[] for _ in range(num_workers)]

       for file, size in file_sizes:
           min_idx = worker_loads.index(min(worker_loads))
           worker_files[min_idx].append(file)
           worker_loads[min_idx] += size

       return worker_files
   ```

2. **Dynamic work stealing** (advanced):
   - Use multiprocessing.Queue with all files
   - Workers pull next file when done (no pre-assignment)
   - Tradeoff: More complex, but perfect load balancing

3. **Padding small batches** to equalize:
   - Not recommended: wastes compute on dummy sequences

**Code Pattern Currently Used:**
```python
# parallel.py:29 - Simple but unbalanced
def assign_files_round_robin(files, num_workers):
    worker_files = [[] for _ in range(num_workers)]
    for idx, file_path in enumerate(files):
        worker_idx = idx % num_workers
        worker_files[worker_idx].append(file_path)
```

**Code Pattern to Use (Phase 2+):**
```python
def assign_files_by_size(files, num_workers):
    """Greedy bin packing for load balancing"""
    file_sizes = []
    for f in files:
        # Count sequences (cached or fast)
        count = sum(1 for _ in SeqIO.parse(f, 'fasta'))
        file_sizes.append((f, count))

    # Sort descending
    file_sizes.sort(key=lambda x: x[1], reverse=True)

    # Greedy assignment
    bins = [{'files': [], 'size': 0} for _ in range(num_workers)]
    for file, size in file_sizes:
        min_bin = min(bins, key=lambda b: b['size'])
        min_bin['files'].append(file)
        min_bin['size'] += size

    return [b['files'] for b in bins]
```

**Which Phase Should Address:**
- Phase 1: Keep round-robin (simple, good enough for 10k uniform splits)
- Phase 2: Implement size-based balancing if monitoring shows >20% imbalance
- Phase 3: Consider work queue if users process highly variable inputs

**References:**
- Greedy bin packing algorithm (computer science fundamentals)
- VirNucPro's own monitoring: compare GPU idle time across workers

---

### 6. `torch.no_grad()` Omission in Inference

**What Goes Wrong:**
Forgetting `torch.no_grad()` during inference causes PyTorch to:
- Allocate memory for gradient computation (doubles memory usage)
- Build computation graph (slower inference)
- Not release tensor memory properly (accumulates over batches)

**Why It Happens:**
- Training code has gradients enabled by default
- Copy-paste from training code to inference
- Nested functions where outer has `no_grad()` but inner doesn't
- Worker functions bypass outer `no_grad()` context

**Warning Signs:**
```python
# Memory usage doubles unexpectedly
# Expected: 4GB model + 2GB batch = 6GB
# Actual: 4GB model + 2GB batch + 2GB gradients + 1GB graph = 9GB

RuntimeError: CUDA out of memory (but shouldn't based on calculation)
```

**VirNucPro Specifics:**
- DNABERT-S extraction uses `torch.no_grad()` correctly (features.py:48)
- ESM-2 extraction uses `torch.no_grad()` correctly (features.py:149)
- Risk: Worker functions in parallel.py might nest incorrectly

**Prevention Strategy:**

1. **Explicit `model.eval()` + `torch.no_grad()`** at worker level:
   ```python
   def worker(files, device_id):
       model = load_model().to(device)
       model.eval()  # Disable dropout, batch norm updates

       with torch.no_grad():  # Disable gradient computation
           for file in files:
               process(file, model)
   ```

2. **Decorator for inference functions**:
   ```python
   @torch.no_grad()
   def extract_features(inputs, model):
       return model(inputs)
   ```

3. **Audit all inference paths** - grep for model forward passes without `no_grad()`

**Code Pattern to Avoid:**
```python
# BAD
def worker(files, device_id):
    model = load_model().to(device)
    for file in files:
        output = model(input)  # Gradients enabled!
```

**Code Pattern to Use:**
```python
# GOOD
def worker(files, device_id):
    model = load_model().to(device)
    model.eval()

    with torch.no_grad():
        for file in files:
            output = model(input)
```

**Which Phase Should Address:**
- Phase 1: Verify in code review (already correct)
- Ongoing: CI test to detect gradient tracking in inference

**References:**
- [PyTorch FAQ: Memory management](https://docs.pytorch.org/docs/stable/notes/faq.html)
- [GeeksforGeeks: Avoid CUDA OOM](https://www.geeksforgeeks.org/deep-learning/how-to-avoid-cuda-out-of-memory-in-pytorch/)

---

### 7. Inconsistent Checkpoint Schema Across Versions

**What Goes Wrong:**
When optimizing code, checkpoint format changes (e.g., adding metadata, changing keys) break resume from old checkpoints. Users lose hours of computation if they can't resume.

**Why It Happens:**
- Refactoring changes data structure keys
- Adding validation fields to checkpoints
- Changing from list to dict format
- Version upgrades (PyTorch, transformers) change serialization

**Warning Signs:**
```python
KeyError: 'nucleotide_features'  # Expected key missing
ValueError: Invalid checkpoint version
# User reports: "Can't resume after update"
```

**VirNucPro Specifics:**
- Checkpoint format defined in checkpoint.py (line 105-125)
- Feature files saved with specific keys: `{'nucleotide': ..., 'data': ...}`
- Risk: Optimized ESM-2 code changes key names
- Risk: Adding batch processing changes checkpoint structure

**Prevention Strategy:**

1. **Version checkpoint format**:
   ```python
   checkpoint = {
       'version': '2.0',  # Increment when format changes
       'data': features,
       'metadata': {...}
   }

   # On load
   if checkpoint.get('version', '1.0') == '1.0':
       data = migrate_v1_to_v2(checkpoint)
   ```

2. **Backward compatibility loader**:
   ```python
   def load_checkpoint(path):
       data = torch.load(path)

       # Handle old format (no version field)
       if 'version' not in data:
           return migrate_legacy(data)

       # Handle versioned formats
       if data['version'] == '1.0':
           return data['data']
       elif data['version'] == '2.0':
           return data['features']
   ```

3. **Schema validation**:
   ```python
   EXPECTED_KEYS = {'nucleotide', 'data'}

   def validate_checkpoint(data):
       if not all(k in data for k in EXPECTED_KEYS):
           raise ValueError(f"Checkpoint missing keys: {EXPECTED_KEYS - data.keys()}")
   ```

4. **Don't change format unless necessary** - backward compatibility > optimization

**Code Pattern to Avoid:**
```python
# BAD - changing keys breaks old checkpoints
# Old: {'nucleotide': [...], 'data': [...]}
# New: {'sequences': [...], 'features': [...]}  # BREAKING CHANGE
torch.save({'sequences': seqs, 'features': feats}, path)
```

**Code Pattern to Use:**
```python
# GOOD - version-aware saving
checkpoint = {
    'version': '2.0',
    'sequences': seqs,  # New key name
    'features': feats,
    # Legacy keys for backward compat
    'nucleotide': seqs,
    'data': feats
}
torch.save(checkpoint, path)
```

**Which Phase Should Address:**
- Phase 1: Add version field to new ESM-2 checkpoints
- Phase 2: Implement migration for old checkpoints
- Ongoing: Never change keys without version bump

**References:**
- VirNucPro CONCERNS.md: Checkpoint resume fragility (line 324-338)
- Software versioning best practices

---

## Medium-Risk Pitfalls

### 8. Memory Leaks from Persistent CUDA Tensors in Workers

**What Goes Wrong:**
Long-lived worker processes accumulate tensor references that aren't garbage collected, causing gradual memory growth and eventual OOM.

**Why It Happens:**
- Python's garbage collector doesn't immediately free GPU memory
- Circular references prevent collection
- Tensors stored in class attributes without cleanup
- Logger formatters hold tensor references

**Warning Signs:**
```bash
# nvidia-smi shows growing memory over time:
Iteration 1: 4.2GB
Iteration 10: 4.8GB
Iteration 50: 6.3GB
Iteration 100: OOM
```

**Prevention Strategy:**

1. **Explicit deletion in worker loops**:
   ```python
   for file in files:
       features = extract(file, model)
       save(features)
       del features  # Explicit delete
       torch.cuda.empty_cache()  # Optional: force cleanup
   ```

2. **Avoid tensor logging**:
   ```python
   # BAD
   logger.info(f"Features: {features}")  # Tensor in log keeps reference

   # GOOD
   logger.info(f"Features shape: {features.shape}")  # Only metadata
   ```

3. **Process restart after N files** (nuclear option):
   ```python
   # Worker processes 100 files then exits, new worker spawned
   ```

**Which Phase Should Address:**
- Phase 2: Add explicit cleanup if memory growth detected

---

### 9. Incorrect Assumption of GPU Homogeneity

**What Goes Wrong:**
Assuming all GPUs have same VRAM/compute capacity causes crashes when users have mixed GPU setups (e.g., 1× 32GB A100 + 3× 16GB V100).

**Why It Happens:**
- Code sets batch_size globally
- Round-robin assigns same workload to all GPUs
- No per-GPU memory detection

**Warning Signs:**
```python
# GPU 0: 32GB → no problem
# GPU 1: 16GB → OOM on same batch size
RuntimeError: CUDA out of memory (only on some GPUs)
```

**Prevention Strategy:**

1. **Per-GPU batch size calculation**:
   ```python
   def get_batch_size_for_gpu(device_id):
       props = torch.cuda.get_device_properties(device_id)
       vram_gb = props.total_memory / 1e9

       if vram_gb >= 32:
           return 256
       elif vram_gb >= 16:
           return 128
       else:
           return 64
   ```

2. **Document homogeneous GPU requirement** (simpler)

**Which Phase Should Address:**
- Phase 3: Add per-GPU configuration if users report issues

---

### 10. DataLoader Worker Inefficiency in Multi-GPU Context

**What Goes Wrong:**
Using PyTorch DataLoader with `num_workers > 0` inside multiprocessing workers creates excessive processes and CPU contention.

**Why It Happens:**
- 4 GPU workers × 4 DataLoader workers = 16 processes competing for CPU
- Context switching overhead
- Memory duplication

**Warning Signs:**
```bash
# htop shows 16+ Python processes
# CPU usage: 1600% (all cores saturated)
# GPU utilization: 60% (waiting for data)
```

**Prevention Strategy:**

1. **Disable DataLoader workers in GPU workers**:
   ```python
   # In multiprocessing worker
   dataloader = DataLoader(dataset, num_workers=0)  # Single-threaded
   ```

2. **OR use single DataLoader with pin_memory** before multiprocessing

**Which Phase Should Address:**
- Phase 1: Set `num_workers=0` in worker processes

---

## Edge Cases

### 11. Empty File Handling in Parallel Processing

**What Goes Wrong:**
Empty FASTA files after translation (no valid ORFs) cause workers to crash or produce 0-byte checkpoints that break resume.

**Prevention Strategy:**
```python
if len(sequences) == 0:
    logger.warning(f"Empty file {input_file}, skipping")
    # Save empty checkpoint with metadata
    torch.save({'sequences': [], 'features': [], 'empty': True}, output)
    return
```

**Which Phase Should Address:**
- Phase 1: Add to ESM-2 worker

---

### 12. CUDA Visible Devices Environment Variable Conflicts

**What Goes Wrong:**
User sets `CUDA_VISIBLE_DEVICES=0,2` but code assumes devices 0,1,2,3 are available.

**Prevention Strategy:**
```python
# Don't use: range(torch.cuda.device_count())  # Returns 0,1 (remapped!)
# Use: Parse actual device IDs or respect CUDA_VISIBLE_DEVICES mapping
```

**Which Phase Should Address:**
- Phase 2: Document that CUDA_VISIBLE_DEVICES is respected

---

## Summary: Prioritized Pitfall Mitigation

### Phase 1 (ESM-2 Parallelization)
- ✓ CUDA context initialization (Pitfall #1) - Critical
- ✓ Checkpoint validation (Pitfall #3) - Critical
- ✓ File-level parallelism for ESM-2 (Pitfall #4) - Critical
- ✓ `torch.no_grad()` verification (Pitfall #6) - Quick win
- ✓ Empty file handling (Pitfall #11) - Edge case
- ✓ DataLoader workers (Pitfall #10) - Quick fix

### Phase 2 (Optimization)
- ✓ Memory fragmentation mitigation (Pitfall #2) - High impact
- ✓ Load balancing (Pitfall #5) - If monitoring shows need
- ✓ Checkpoint versioning (Pitfall #7) - Good practice
- ✓ Memory leak monitoring (Pitfall #8) - If detected

### Phase 3+ (Polish)
- ✓ Heterogeneous GPU support (Pitfall #9) - If users request
- ✓ CUDA_VISIBLE_DEVICES handling (Pitfall #12) - Documentation

---

## Research Sources

### Multi-GPU Inference
- [Running Inference on multiple GPUs - PyTorch Forums](https://discuss.pytorch.org/t/running-inference-on-multiple-gpus/163095)
- [Multi-GPU Inference Discussion - Lightning-AI](https://github.com/Lightning-AI/pytorch-lightning/discussions/9259)
- [Distributed inference - HuggingFace](https://huggingface.co/docs/diffusers/training/distributed_inference)

### CUDA Memory Management
- [How to Solve CUDA out of memory - Saturn Cloud](https://saturncloud.io/blog/how-to-solve-cuda-out-of-memory-in-pytorch/)
- [CUDA semantics - PyTorch Docs](https://docs.pytorch.org/docs/stable/notes/cuda.html)
- [PyTorch Memory Management - DigitalOcean](https://blog.paperspace.com/pytorch-memory-multi-gpu-debugging/)
- [Avoiding CUDA OOM - GeeksforGeeks](https://www.geeksforgeeks.org/deep-learning/how-to-avoid-cuda-out-of-memory-in-pytorch/)

### Multiprocessing & CUDA
- [Cannot re-initialize CUDA in forked subprocess - Issue #40403](https://github.com/pytorch/pytorch/issues/40403)
- [RuntimeError: Cannot re-initialize CUDA - PyTorch Forums](https://discuss.pytorch.org/t/runtimeerror-cannot-re-initialize-cuda-in-forked-subprocess-to-use-cuda-with-multiprocessing-you-must-use-the-spawn-start-method/14083)

### Large Model Parallelism
- [Parallelism methods - HuggingFace](https://huggingface.co/docs/transformers/main/perf_train_gpu_many)
- [ESM GitHub Repository](https://github.com/facebookresearch/esm)
- [Parallelism and Scaling - vLLM](https://docs.vllm.ai/en/stable/serving/parallelism_scaling/)

### Checkpointing
- [Distributed Checkpoint - PyTorch Forums](https://discuss.pytorch.org/t/what-is-the-proper-way-to-checkpoint-during-training-when-using-distributed-data-parallel-ddp-in-pytorch/139575)
- [Reducing Checkpointing Times - PyTorch Blog](https://pytorch.org/blog/reducing-checkpointing-times/)
- [Fault-tolerant Training - PyTorch Tutorials](https://docs.pytorch.org/tutorials/beginner/ddp_series_fault_tolerance.html)
