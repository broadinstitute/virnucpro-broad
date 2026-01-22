# Multi-GPU Inference Architecture for Batch Processing

**Research Question**: How are multi-GPU inference systems typically structured? What are major components for batch processing and work distribution?

**Context**: VirNucPro needs to optimize ESM-2 (3B parameter model) and DNABERT-S feature extraction across multiple GPUs. Current bottleneck is ESM-2 at 45 hours per sample on single GPU. Target: <10 hours with 4-8 GPUs.

---

## Common Multi-GPU Inference Patterns

Multi-GPU inference systems for transformer models typically follow one of three architectural patterns:

### 1. **Data Parallel (Most Common for Inference)**
- **Structure**: Clone model on each GPU, distribute data batches
- **Best for**: Models that fit in single GPU memory (ESM-2 3B may be tight but feasible)
- **Speedup**: Linear with GPU count (2x GPUs = 2x throughput)
- **Complexity**: Low

### 2. **Pipeline Parallel**
- **Structure**: Split model layers across GPUs, stream batches through pipeline
- **Best for**: Very large models that don't fit on one GPU
- **Speedup**: Sublinear (pipeline bubbles reduce efficiency)
- **Complexity**: High

### 3. **Tensor Parallel**
- **Structure**: Split individual layer computations across GPUs
- **Best for**: Extremely large models (>10B parameters)
- **Speedup**: Limited by communication overhead
- **Complexity**: Very high

**Recommendation for VirNucPro**: Data parallel architecture - ESM-2 3B fits on modern GPUs (A100 40GB, H100 80GB), and data parallel provides best throughput for batch inference.

---

## Core Components for Multi-GPU Batch Processing

### Component 1: Batch Queue Manager

**Purpose**: Central coordinator that manages work distribution across GPUs.

**Responsibilities**:
- Maintain global queue of sequences/files to process
- Track which batches are assigned to which GPUs
- Handle dynamic load balancing (GPU speed differences, memory constraints)
- Coordinate checkpoint updates

**Key Abstractions**:
```python
class BatchQueue:
    def __init__(self, input_files: List[Path], batch_size_per_gpu: int)
    def get_next_batch(self, gpu_id: int) -> Optional[Batch]
    def mark_completed(self, batch_id: str, outputs: Dict)
    def get_progress() -> ProgressSummary
```

**Design Considerations**:
- **Queue Granularity**: File-level vs sequence-level batching
  - *File-level*: Simple, works well with existing checkpoint system (VirNucPro splits into 10k-seq files)
  - *Sequence-level*: Better load balancing but requires more coordination
  - **Recommendation**: Start with file-level, matches VirNucPro's current split-file pattern

- **Synchronization**: Shared memory (multiprocessing.Manager) vs disk-based (checkpoint files)
  - *Shared memory*: Faster but lost on crash
  - *Disk-based*: Slower but resume-friendly
  - **Recommendation**: Hybrid - shared memory queue with periodic checkpoint flushes

**Data Flow**:
```
Input Files → Queue Manager → GPU Workers → Results Aggregator → Checkpoint
```

---

### Component 2: GPU Worker Pool

**Purpose**: Independent processes that pull batches from queue and run inference.

**Responsibilities**:
- Load model on assigned GPU (one model per GPU)
- Pull batches from queue manager
- Run inference (forward pass only, no gradients)
- Return results to queue/checkpoint

**Key Abstractions**:
```python
class GPUWorker:
    def __init__(self, gpu_id: int, model_config: Dict)
    def _load_model(self) -> torch.nn.Module
    def process_batch(self, batch: Batch) -> Results
    def run_loop(self, queue: BatchQueue)  # Main worker loop
```

**Design Considerations**:
- **Process vs Thread**: Must use multiprocessing with 'spawn' (CUDA incompatible with fork)
  - VirNucPro already uses spawn context in `parallel.py` - extend this pattern

- **Model Loading**: Load once per worker at startup
  - ESM-2 3B takes ~30s to load, amortize across many batches
  - DNABERT-S loads faster (~10s)

- **Batch Size Tuning**: Per-GPU batch size affects memory and throughput
  - ESM-2: Limited by memory (3B params + activations), typically 4-16 sequences depending on length
  - DNABERT-S: Can handle larger batches (256 currently in VirNucPro)
  - **Recommendation**: Make configurable per model, auto-tune based on GPU memory

**Worker Lifecycle**:
```
Start → Load Model → Poll Queue → Process Batch → Save Results → (repeat) → Shutdown
```

---

### Component 3: Checkpoint Integration

**Purpose**: Track batch-level progress for resume capability.

**Responsibilities**:
- Record which batches/files completed on which GPU
- Enable resume from partial completion
- Validate outputs exist and are non-corrupt

**Key Abstractions**:
```python
class BatchCheckpointManager:
    def mark_batch_started(self, batch_id: str, gpu_id: int)
    def mark_batch_completed(self, batch_id: str, output_files: List[Path])
    def get_pending_batches() -> List[Batch]
    def validate_checkpoint_outputs() -> bool
```

**Design Considerations**:
- **Integration with Existing**: VirNucPro has `CheckpointManager` for stages, `FileProgressTracker` for files
  - Extend `FileProgressTracker` to track GPU assignment
  - Add per-file atomic completion markers (e.g., `.done` files alongside `.pt` outputs)

- **Granularity Trade-off**:
  - *Fine-grained* (per-batch): Better resume, more I/O overhead
  - *Coarse-grained* (per-stage): Less resume granularity, simpler
  - **Recommendation**: Per-file checkpointing (VirNucPro already splits to 10k-seq files, good granularity)

**Checkpoint Data Flow**:
```
GPU Worker Completes Batch → Update Checkpoint → Verify Output Files → Mark Available for Next Stage
```

---

### Component 4: GPU Utilization Monitor (Optional but Recommended)

**Purpose**: Real-time monitoring of GPU usage to validate optimization effectiveness.

**Responsibilities**:
- Track GPU memory utilization per device
- Track GPU compute utilization (SM occupancy)
- Log throughput metrics (sequences/sec per GPU)
- Detect stalled workers or imbalanced load

**Key Abstractions**:
```python
class GPUMonitor:
    def start_monitoring(self, gpu_ids: List[int])
    def get_current_stats() -> Dict[int, GPUStats]
    def log_stats(self, interval_seconds: int = 10)
```

**Design Considerations**:
- **Monitoring Library**: `nvidia-ml-py3` (pynvml) for NVIDIA GPUs
- **Overhead**: Minimal (<1% CPU), run in separate thread
- **Logging**: Periodic logs to validate >80% GPU utilization (project requirement PERF-02)

**Monitoring Metrics**:
- GPU utilization % (target: >80% during embedding stages)
- Memory usage MB (detect OOM before crash)
- Throughput (sequences/sec per GPU, total pipeline throughput)

---

## Component Boundaries and Interfaces

### Boundary 1: Queue Manager ↔ GPU Workers

**Interface**: Multiprocessing queue or shared manager dictionary

```python
# Queue Manager provides
def get_next_batch(gpu_id: int) -> Optional[BatchDescriptor]
    # Returns: {'batch_id': str, 'input_files': [Path], 'output_dir': Path}

# GPU Worker calls
def mark_completed(batch_id: str, outputs: Dict[str, Any])
    # Sends: {'batch_id': str, 'output_files': [Path], 'metadata': {...}}
```

**Data Flow**: Queue → Worker (batch descriptor), Worker → Queue (completion signal)

**Error Handling**: Worker crashes must not lose batch assignment (timeout + requeue pattern)

---

### Boundary 2: GPU Workers ↔ Checkpoint Manager

**Interface**: Direct file system I/O (workers write .pt files, manager reads completion)

```python
# Worker writes
output_file = output_dir / f"{input_file.stem}_ESM.pt"
torch.save(features, output_file)
marker_file = output_file.with_suffix('.done')
marker_file.touch()  # Atomic completion signal

# Checkpoint Manager validates
def validate_batch_output(batch_id: str) -> bool:
    return all(
        output_file.exists() and
        output_file.with_suffix('.done').exists() and
        output_file.stat().st_size > 0
        for output_file in expected_outputs
    )
```

**Data Flow**: Worker → Filesystem → Checkpoint Manager

**Resume Logic**: On resume, skip any batch with valid `.done` marker + non-empty output file

---

### Boundary 3: Main Orchestrator ↔ All Components

**Interface**: Main prediction pipeline (`run_prediction()`) creates and manages lifecycle

```python
# In run_prediction() for Stage 6: Protein Feature Extraction
if use_multi_gpu:
    # Create components
    queue = BatchQueue(protein_files, batch_size_per_gpu=4)
    checkpoint = BatchCheckpointManager(checkpoint_dir, stage='PROTEIN_FEATURES')
    monitor = GPUMonitor(available_gpus)

    # Start workers
    workers = [
        GPUWorker(gpu_id, model='esm2_t36_3B_UR50D')
        for gpu_id in available_gpus
    ]

    # Run parallel processing
    with ProcessPool(workers) as pool:
        pool.map(lambda w: w.run_loop(queue), workers)

    # Aggregate results (already written to disk by workers)
    protein_feature_files = checkpoint.get_completed_outputs()
```

**Data Flow**: Orchestrator controls component initialization → Components interact → Orchestrator validates results

---

## Data Flow: Complete Multi-GPU Batch Processing

### High-Level Flow
```
Input Files (10k sequences each)
    ↓
[Queue Manager] Distributes files to workers
    ↓
[GPU Worker 0] ← File 0, 4, 8...  [Model loaded on cuda:0]
[GPU Worker 1] ← File 1, 5, 9...  [Model loaded on cuda:1]
[GPU Worker 2] ← File 2, 6, 10... [Model loaded on cuda:2]
[GPU Worker 3] ← File 3, 7, 11... [Model loaded on cuda:3]
    ↓
Each worker: Load sequences → Tokenize → Batch inference → Save .pt
    ↓
[Checkpoint Manager] Validates all outputs complete
    ↓
[Next Pipeline Stage] Uses aggregated feature files
```

### Detailed Worker Flow (ESM-2 Example)
```
GPU Worker receives batch descriptor {'file': protein_0.fa, 'output': output_dir}
    ↓
Load sequences from protein_0.fa (BioPython SeqIO)
    ↓
Batch sequences by token count (current: 2048 tokens/batch)
    ↓
For each micro-batch:
    - Tokenize with ESM alphabet
    - Move to GPU (cuda:N)
    - Forward pass (no gradients)
    - Mean pool representations
    - Move to CPU, store in list
    ↓
Save all features: torch.save({'proteins': ids, 'data': features}, output_0_ESM.pt)
    ↓
Create completion marker: output_0_ESM.pt.done
    ↓
Return completion signal to queue manager
```

### Checkpoint Resume Flow
```
User runs `virnucpro predict --resume`
    ↓
Load checkpoint state (CheckpointManager.load_state())
    ↓
For Stage 6 (PROTEIN_FEATURES):
    - Find all expected output files (protein_split_dir/*_ESM.pt)
    - Check which have valid .done markers
    - Build queue of remaining files
    ↓
If queue empty → Skip stage
If queue has files → Run multi-GPU processing for remaining files only
    ↓
Aggregate all outputs (pre-existing + newly computed)
    ↓
Mark stage completed
```

---

## Build Order and Dependencies

### Phase 1: Extend Existing Single-GPU (Foundation)
**Goal**: Refactor current ESM-2 code to be worker-compatible

**Tasks**:
1. Extract ESM-2 loading into reusable `ESMWorker` class
2. Add batch-level checkpointing (`.done` marker pattern)
3. Validate single-GPU performance matches current implementation

**Dependencies**: None (refactor existing code)

**Success Criteria**:
- Single-GPU ESM-2 extraction works with new worker pattern
- Can resume from partial file completion
- No performance regression

---

### Phase 2: Add Queue Manager (Coordination)
**Goal**: Central work distribution component

**Tasks**:
1. Implement `BatchQueue` with multiprocessing.Manager
2. Round-robin file assignment to N workers
3. Add completion tracking and progress reporting

**Dependencies**: Phase 1 (needs worker interface)

**Success Criteria**:
- Multiple workers can pull from shared queue
- No duplicate work (each file assigned once)
- Progress tracking shows files remaining

---

### Phase 3: Multi-GPU Workers (Parallelization)
**Goal**: Run multiple workers in parallel processes

**Tasks**:
1. Create worker pool with spawn context (extend existing `parallel.py`)
2. Load ESM-2 on each GPU (one model per GPU)
3. Coordinate via queue manager from Phase 2

**Dependencies**: Phase 1 + Phase 2

**Success Criteria**:
- 4 GPUs process 4 files concurrently
- Linear speedup (4 GPUs = ~4x throughput)
- No CUDA errors from process forking

---

### Phase 4: Checkpoint Integration (Robustness)
**Goal**: Resume from partial completion across GPUs

**Tasks**:
1. Extend `FileProgressTracker` to store GPU assignment
2. Add output validation (file exists, non-empty, has .done marker)
3. Test resume after simulated crash mid-stage

**Dependencies**: Phase 3

**Success Criteria**:
- Can resume after killing worker mid-batch
- Skips completed files, reprocesses incomplete
- Checkpoint state matches actual filesystem state

---

### Phase 5: Monitoring and Tuning (Optimization)
**Goal**: Validate GPU utilization, tune batch sizes

**Tasks**:
1. Add `GPUMonitor` with pynvml
2. Log GPU utilization every 10s during processing
3. Auto-tune batch size based on GPU memory (optional)

**Dependencies**: Phase 3 (needs parallel workers)

**Success Criteria**:
- Logs show >80% GPU utilization during embedding
- Can detect and report underutilization or imbalance
- Metrics validate performance improvement

---

## Suggested Build Order Summary

1. **Phase 1** (Refactor): Single-GPU worker pattern → Foundation for multi-GPU
2. **Phase 2** (Coordinate): Queue manager → Work distribution logic
3. **Phase 3** (Parallelize): Multi-GPU workers → Core performance gain
4. **Phase 4** (Robustness): Checkpoint integration → Production reliability
5. **Phase 5** (Validate): Monitoring → Performance validation

**Critical Path**: Phase 1 → Phase 2 → Phase 3 (speedup achieved here) → Phase 4 (resume capability)

**Optional**: Phase 5 (helpful for debugging, validating PERF-02 requirement)

---

## VirNucPro-Specific Considerations

### Integration with Existing Architecture

**Current State**:
- `virnucpro/pipeline/features.py`: Single-GPU ESM-2 extraction
- `virnucpro/pipeline/parallel.py`: Multi-GPU DNABERT-S (file-level parallelism)
- `virnucpro/core/checkpoint.py`: Stage-level checkpointing

**Proposed Extensions**:
- Create `virnucpro/pipeline/gpu_pool.py`: GPU worker pool manager
- Create `virnucpro/pipeline/batch_queue.py`: Batch queue with checkpointing
- Extend `virnucpro/core/checkpoint.py`: Add `BatchCheckpointManager` class
- Extend `virnucpro/pipeline/parallel.py`: Add ESM-2 worker (currently DNABERT-S only)

**Backward Compatibility**:
- Keep `extract_esm_features()` function in `features.py` for single-GPU path
- Add `extract_esm_features_parallel()` function for multi-GPU
- CLI flag `--parallel` (existing) enables multi-GPU
- Single-GPU remains default for small datasets

---

### Memory and Batch Size Constraints

**ESM-2 3B Parameter Model**:
- Model weights: ~12 GB (FP32) or ~6 GB (FP16)
- Per-sequence activation memory: ~500 MB - 2 GB depending on length (1024 max)
- Typical GPU memory: A100 40GB, A6000 48GB, H100 80GB

**Batch Size Recommendations**:
- **Conservative**: 4 sequences/batch (fits on 24GB GPUs)
- **Moderate**: 8 sequences/batch (fits on 40GB GPUs)
- **Aggressive**: 16 sequences/batch (requires 80GB GPUs, truncation to 512 tokens)

**VirNucPro Current**: `toks_per_batch=2048` (adaptive batching by token count)
- Keep this pattern, it's memory-efficient
- Expose as config parameter for tuning per GPU type

**DNABERT-S**:
- Smaller model (~100M params), current batch_size=256 is fine
- Can increase to 512-1024 on modern GPUs for even better throughput

---

### Checkpointing Strategy

**Current VirNucPro Pattern**:
- 10,000 sequences per file (configurable via `sequences_per_file`)
- Each file produces one `.pt` feature file
- Stage completes when all `.pt` files exist

**Proposed Enhancement**:
```python
# In gpu_pool.py worker
def process_file(file_path: Path, output_dir: Path, gpu_id: int):
    output_file = output_dir / f"{file_path.stem}_ESM.pt"
    done_marker = output_file.with_suffix('.pt.done')

    # Skip if already completed
    if done_marker.exists() and output_file.exists():
        logger.info(f"Skipping {file_path.name} (already processed)")
        return output_file

    # Process file
    features = extract_esm_features_batch(file_path, gpu_id)

    # Atomic save: write to temp, then rename
    temp_file = output_file.with_suffix('.pt.tmp')
    torch.save(features, temp_file)
    temp_file.replace(output_file)

    # Mark complete
    done_marker.touch()

    return output_file
```

**Benefits**:
- Resume from any point (file-level granularity)
- Atomic writes prevent corrupt partial files
- `.done` marker distinguishes complete vs in-progress

---

## Performance Expectations

### Theoretical Speedup
- **Linear scaling**: N GPUs → N× throughput (data parallel ideal)
- **Reality**: 90-95% efficiency due to:
  - Queue coordination overhead (~1-2%)
  - Load imbalance (different file sizes) (~3-5%)
  - I/O contention (multiple workers writing) (~1-2%)

### VirNucPro Projected Performance
**Current**: 45 hours ESM-2 on 1 GPU

**With 4 GPUs**: 45h / 4 / 0.93 efficiency = **12.1 hours**

**With 8 GPUs**: 45h / 8 / 0.90 efficiency = **6.25 hours** ✓ Meets <10h requirement

**Factors**:
- Assumes GPUs same type (heterogeneous GPUs reduce efficiency)
- Assumes sufficient CPU/RAM for data loading (bottleneck if underpowered)
- Assumes fast storage (NVMe SSD recommended, HDD will bottleneck I/O)

### Validation Metrics (PERF-02 Requirement)
Target: >80% GPU utilization during embedding stages

**How to Measure**:
```python
import pynvml
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
print(f"GPU {gpu_id}: {utilization.gpu}% compute, {utilization.memory}% memory")
```

**Expected**:
- During batch processing: 85-95% compute utilization
- During I/O (loading next batch): 10-30% utilization (brief dips normal)
- **Average over stage**: Should exceed 80%

---

## Alternative Architectures Considered

### Option A: Ray for Distributed Inference
**Pros**: Built-in task scheduling, fault tolerance, monitoring dashboard

**Cons**:
- Heavy dependency (entire framework)
- Overkill for single-machine multi-GPU
- Learning curve for team

**Verdict**: Not recommended unless expanding to multi-node in future

---

### Option B: DeepSpeed Inference
**Pros**: Optimized for transformer inference, kernel fusion, model parallelism

**Cons**:
- Primarily for >10B models (ESM-2 3B doesn't benefit much)
- Requires model rewrites for DeepSpeed API
- Complex integration with BioPython/ESM library

**Verdict**: Consider for future if moving to ESM-3 (100B+ params), not needed now

---

### Option C: Simple multiprocessing.Pool (Chosen)
**Pros**:
- Minimal dependencies (stdlib + existing code)
- VirNucPro already uses this for DNABERT-S
- Easy to understand and maintain

**Cons**:
- Manual queue management
- No built-in monitoring (need custom)

**Verdict**: Best fit for VirNucPro - simple, proven pattern, extends existing code

---

## Key Takeaways for Implementation

1. **Component Boundaries**:
   - `BatchQueue`: File assignment and progress tracking
   - `GPUWorker`: Model loading and batch inference
   - `BatchCheckpointManager`: Resume capability
   - `GPUMonitor`: Utilization validation (optional)

2. **Data Flow**:
   - Files → Queue → Workers (parallel) → Outputs → Checkpoint validation

3. **Build Order**:
   - Start with single-GPU worker refactor (Phase 1)
   - Add queue coordination (Phase 2)
   - Parallelize workers (Phase 3) ← Core speedup here
   - Add checkpoint integration (Phase 4)
   - Add monitoring (Phase 5)

4. **VirNucPro Integration**:
   - Extend existing `parallel.py` pattern to ESM-2
   - Keep file-level granularity (10k sequences/file)
   - Use `.done` markers for atomic completion
   - Expose batch size as config parameter

5. **Performance Target**:
   - 4 GPUs: ~12 hours (good)
   - 8 GPUs: ~6 hours (exceeds <10h requirement)
   - Validate with >80% GPU utilization metrics

---

**Next Steps for Planning**:
- Break Phase 1-5 into detailed task lists in PLAN.md
- Identify specific files to modify in each phase
- Define test cases for each phase (unit tests for queue, integration tests for multi-GPU)
