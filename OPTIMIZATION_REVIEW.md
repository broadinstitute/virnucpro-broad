# VirNucPro GPU Optimization Review

**Date**: February 8, 2026  
**Scope**: Viral nucleotide prediction pipeline GPU optimization analysis  
**Objective**: Identify opportunities for further performance improvements beyond current v2.0 architecture

---

## Executive Summary

The VirNucPro v2.0 refactoring has achieved remarkable success, reducing embedding extraction time from **45+ hours to under 10 hours** through sophisticated GPU optimization techniques. This review identifies additional opportunities for **30-50% further throughput improvements** with targeted optimizations.

### Key Findings

| Metric | Current | Potential |
|--------|---------|-----------|
| Throughput | Baseline | +30-50% |
| Packing Efficiency | 92-94% | 95-97% (BFD algorithm) |
| Multi-GPU Scaling | Sub-linear after 4 GPUs | Near-linear to 8 GPUs |
| Memory Fragmentation | Moderate | Minimal (with pooling) |

---

## Architecture Overview (Current v2.0)

### High-Level Data Flow

```
Input FASTA Files
       ↓
┌─────────────────────────────────────────────────────────────────┐
│                     Multi-GPU Coordinator                        │
│              (GPUProcessCoordinator - spawn context)             │
└─────────────────────────────────────────────────────────────────┘
       ↓                              ↓                              ↓
┌──────────────┐              ┌──────────────┐              ┌──────────────┐
│  GPU Worker 0 │              │  GPU Worker 1 │              │  GPU Worker N │
│  (cuda:0)     │              │  (cuda:1)     │              │  (cuda:N)     │
└──────┬───────┘              └──────┬───────┘              └──────┬───────┘
       ↓                              ↓                              ↓
┌──────────────┐              ┌──────────────┐              ┌──────────────┐
│ AsyncInference│              │ AsyncInference│              │ AsyncInference│
│   Runner      │              │   Runner      │              │   Runner      │
└──────┬───────┘              └──────┬───────┘              └──────┬───────┘
       ↓                              ↓                              ↓
┌──────────────┐              ┌──────────────┐              ┌──────────────┐
│  IndexBased   │              │  IndexBased   │              │  IndexBased   │
│   Dataset     │              │   Dataset     │              │   Dataset     │
│ (byte-offset  │              │ (byte-offset  │              │ (byte-offset  │
│   reading)    │              │   reading)    │              │   reading)    │
└──────┬───────┘              └──────┬───────┘              └──────┬───────┘
       ↓                              ↓                              ↓
┌──────────────┐              ┌──────────────┐              ┌──────────────┐
│  DataLoader   │              │  DataLoader   │              │  DataLoader   │
│   Workers     │              │   Workers     │              │   Workers     │
│ (CPU-only I/O)│              │ (CPU-only I/O)│              │ (CPU-only I/O)│
└──────┬───────┘              └──────┬───────┘              └──────┬───────┘
       ↓                              ↓                              ↓
┌──────────────┐              ┌──────────────┐              ┌──────────────┐
│ VarlenCollator│              │ VarlenCollator│              │ VarlenCollator│
│ (Main Process)│              │ (Main Process)│              │ (Main Process)│
│  - Tokenize   │              │  - Tokenize   │              │  - Tokenize   │
│  - Pack (FFD) │              │  - Pack (FFD) │              │  - Pack (FFD) │
└──────┬───────┘              └──────┬───────┘              └──────┬───────┘
       ↓                              ↓                              ↓
┌──────────────┐              ┌──────────────┐              ┌──────────────┐
│ StreamProcessor│            │ StreamProcessor│            │ StreamProcessor│
│  - H2D Stream │              │  - H2D Stream │              │  - H2D Stream │
│  - Compute    │              │  - Compute    │              │  - Compute    │
│  - D2H Stream │              │  - D2H Stream │              │  - D2H Stream │
└──────┬───────┘              └──────┬───────┘              └──────┬───────┘
       ↓                              ↓                              ↓
┌──────────────┐              ┌──────────────┐              ┌──────────────┐
│ ESM2/DNABERT  │              │ ESM2/DNABERT  │              │ ESM2/DNABERT  │
│  with Flash   │              │  with Flash   │              │  with Flash   │
│  Attention    │              │  Attention    │              │  Attention    │
└──────┬───────┘              └──────┬───────┘              └──────┬───────┘
       ↓                              ↓                              ↓
┌──────────────┐              ┌──────────────┐              ┌──────────────┐
│  shard_0.h5   │              │  shard_1.h5   │              │  shard_N.h5   │
└──────────────┘              └──────────────┘              └──────────────┘
       ↓                              ↓                              ↓
                    ┌──────────────────────┐
                    │   Shard Aggregator    │
                    │   (embeddings.h5)     │
                    └──────────────────────┘
```

### Key Optimization Components

| Component | Implementation | Performance Impact |
|-----------|----------------|-------------------|
| **Sequence Packing** | FFD algorithm with 2000-seq buffer | 92-94% token utilization vs 50-60% padded |
| **FlashAttention-2** | `flash_attn_varlen_func` for packed sequences | 2-4x attention speedup |
| **CUDA Streams** | 3-stream pipeline (H2D/Compute/D2H) | 20-40% latency hiding |
| **Pinned Memory** | `pin_memory=True` in DataLoader | Faster async GPU transfers |
| **FP16 Inference** | Model.half() with FP32 output | 2x memory, faster compute |
| **Byte-Offset Reading** | `IndexBasedDataset` with seek() | Efficient random access |
| **Spawn Multiprocessing** | `ctx = multiprocessing.get_context('spawn')` | Safe CUDA isolation |

---

## Detailed Findings

### 1. Packing Algorithm Bottlenecks

#### 1.1 FFD vs BFD Efficiency Gap
**Location**: `virnucpro/data/packing.py:135-171`

**Current**: First-Fit Decreasing (FFD) greedy algorithm
**Issue**: FFD leaves significant gaps compared to Best-Fit Decreasing (BFD)

```python
# Current (line 156-166):
if current_tokens + tokenized_len <= self.max_tokens_per_batch:
    current_batch.append(seq_dict)
    current_tokens += tokenized_len
else:
    # Starts new batch immediately - doesn't check other batches
```

**Impact**: 2-3% packing efficiency improvement possible with BFD

**Recommendation**: Implement BFD with gap tracking:
```python
# BFD tracks best-fit batch instead of first-fit
best_batch_idx = None
best_remaining = float('inf')
for i, (batch, tokens) in enumerate(batches):
    remaining = self.max_tokens_per_batch - tokens
    if tokenized_len <= remaining < best_remaining:
        best_batch_idx = i
        best_remaining = remaining
```

#### 1.2 Python Loop Overhead in Sorting
**Location**: `virnucpro/data/packing.py:82-100`

**Issue**: Pure Python sort with lambda for 1000-5000 items

**Optimization**: Use `operator.itemgetter`:
```python
from operator import itemgetter

def sort_by_length(self, sequences):
    with_lengths = [(len(s['sequence']), s['id'], s) for s in sequences]
    with_lengths.sort(key=lambda x: (-x[0], x[1]))
    return [s for _, _, s in with_lengths]
```

#### 1.3 Repeated Tokenized Length Calculation
**Location**: `virnucpro/data/packing.py:154, 214-216`

**Issue**: `seq_len + 2` calculated repeatedly; method call overhead

**Optimization**: Inline or cache during initial sort

---

### 2. Collator Performance Issues

#### 2.1 Redundant Token Budget Check
**Location**: `virnucpro/data/collators.py:185-203`

**Issue**: GreedyPacker already checked token budget, but collator checks again

```python
# Lines 185-203 - redundant after packer filtering
if i > 0 and cu_seqlens[-1] + len(seq_tokens) > self.max_tokens_per_batch:
    break
```

**Recommendation**: Remove when `enable_packing=True`

#### 2.2 Inefficient Buffer List Operations
**Location**: `virnucpro/data/collators.py:259-293`

**Issue**: `list.pop(0)` is O(n) for lists

```python
# Current (line 285):
batch_to_return = self.packed_queue.pop(0)  # O(n)
```

**Optimization**: Use `collections.deque`:
```python
from collections import deque

class VarlenCollator:
    def __init__(self, ...):
        self.packed_queue = deque()  # O(1) popleft
```

#### 2.3 Sequential Tensor Construction
**Location**: `virnucpro/data/collators.py:157-206`

**Issue**: Building Python list then converting to tensor

**Optimization**: Preallocate tensor:
```python
# Pre-allocate approach
max_possible_tokens = min(batch_token_estimate, self.max_tokens_per_batch)
input_ids = torch.empty(max_possible_tokens, dtype=torch.long)
pos = 0
for seq_tokens in sequences:
    seq_tensor = torch.tensor(seq_tokens)
    input_ids[pos:pos+len(seq_tokens)] = seq_tensor
    pos += len(seq_tokens)
input_ids = input_ids[:pos]  # Trim to actual size
```

---

### 3. CUDA Stream Management Gaps

#### 3.1 No True Pipeline Parallelism
**Location**: `virnucpro/cuda/stream_manager.py:315-380`

**Current**: Processes one batch at a time with sequential stages

```python
for i, batch in enumerate(batches):
    with self.stream_manager.stream_context('h2d'):
        gpu_data = transfer_fn(batch)  # Batch i
    self.stream_manager.wait_for_stream('compute', 'h2d')
    with self.stream_manager.stream_context('compute'):
        result = compute_fn(gpu_data)   # Batch i (after H2D)
```

**Issue**: Does NOT overlap H2D(batch i+1) with Compute(batch i)

**Optimization**: Implement triple buffering:
```python
def process_batches_pipelined(self, batches, ...):
    # Triple buffer: H2D[i+2], Compute[i+1], D2H[i]
    gpu_data = [None] * 3
    results = [None] * 3
    
    for i, batch in enumerate(batches):
        buf_idx = i % 3
        
        # Stage 1: H2D for current batch
        with self.stream_manager.stream_context('h2d'):
            gpu_data[buf_idx] = transfer_fn(batch)
        
        # Stage 2: Compute for previous batch
        prev_idx = (i - 1) % 3
        if i > 0:
            self.stream_manager.wait_for_stream('compute', 'h2d')
            with self.stream_manager.stream_context('compute'):
                results[prev_idx] = compute_fn(gpu_data[prev_idx])
        
        # Stage 3: D2H for batch i-2
        if i > 1:
            self.stream_manager.wait_for_stream('d2h', 'compute')
            with self.stream_manager.stream_context('d2h'):
                results[prev_idx] = retrieve_fn(results[prev_idx])
```

**Expected Gain**: 15-25% throughput improvement

#### 3.2 Excessive Synchronization
**Location**: `virnucpro/cuda/stream_manager.py:370-371`

**Issue**: `check_error()` calls `synchronize()` every 10 batches

**Optimization**: Use events for non-blocking error checking:
```python
if i % 10 == 0:
    event = torch.cuda.Event()
    event.record(self.compute_stream)
    if event.query():  # Non-blocking check
        pass
```

---

### 4. Async Inference Bottlenecks

#### 4.1 Embedding Extraction Loop
**Location**: `virnucpro/pipeline/async_inference.py:371-410`

**Issue**: Python loop over sequences with individual mean operations:
```python
for i in range(len(sequence_ids)):
    start = cu_seqlens[i].item()  # Individual .item() calls
    end = cu_seqlens[i + 1].item()
    seq_repr = representations[start + 1:end - 1].mean(dim=0)
    embeddings.append(seq_repr)
```

**Optimization**: Use `torch.segment_reduce` or `scatter_add`:
```python
# Vectorized approach using scatter_add
counts = cu_seqlens[1:] - cu_seqlens[:-1]
# Compute cumulative sums per segment
```

#### 4.2 Synchronization Before Embedding Extraction
**Location**: `virnucpro/pipeline/async_inference.py:468`

**Issue**: Full synchronization blocks CPU

**Optimization**: Use events:
```python
event = torch.cuda.Event()
event.record(torch.cuda.current_stream())
event.wait(torch.cuda.default_stream())
```

#### 4.3 Checkpoint Memory Growth
**Location**: `virnucpro/pipeline/async_inference.py:778-804`

**Issue**: Embeddings accumulated as list then concatenated

**Optimization**: Preallocate:
```python
self._ckpt_embeddings = np.empty((checkpoint_seq_threshold, embedding_dim), dtype=np.float32)
self._ckpt_pos = 0
```

#### 4.4 Environment Variable in Hot Path
**Location**: `virnucpro/pipeline/async_inference.py:301-302`

**Issue**: `os.getenv()` called every batch

**Optimization**: Cache at initialization:
```python
def __init__(self, ...):
    self._disable_packing = os.getenv('VIRNUCPRO_DISABLE_PACKING', 'false').lower() == 'true'
```

---

### 5. Multi-GPU Scaling Issues

#### 5.1 Process Spawning Overhead
**Location**: `virnucpro/pipeline/gpu_coordinator.py:119-127`

**Issue**: Sequential process creation, no process pooling

```python
for rank in range(self.world_size):
    p = self.ctx.Process(target=_worker_wrapper, ...)
    p.start()  # Sequential startup
```

**Impact**: ~0.5-1s per worker startup, overhead repeated for every workload

**Recommendations**:
1. Implement process pooling (use existing `persistent_pool.py`)
2. Parallelize spawning with thread pool
3. Pre-spawn workers between workloads

#### 5.2 Duplicate Model Loading
**Location**: `virnucpro/pipeline/gpu_worker.py:246-285`

**Issue**: Each worker loads model independently (~5GB per GPU)

```python
# Each worker loads in isolation
model, batch_converter = load_esm2_model(
    model_name=model_config.get('model_name', 'esm2_t36_3B_UR50D'),
    device=str(device),
    enable_fp16=enable_fp16
)
```

**Impact**: 8 GPUs × 5GB = 40GB disk reads, sequential latency

**Recommendations**:
1. Load model once in coordinator, share via shared memory
2. Use `torch.distributed` broadcast
3. Stagger initialization by 2-3s to avoid disk contention

#### 5.3 Static Load Balancing
**Location**: `virnucpro/pipeline/shard_index.py:224`

**Issue**: Stride distribution causes stragglers

```python
indices = list(range(rank, total_sequences, world_size))
```

**Example Imbalance**:
```
GPU 0: [0, 4, 8, ...]  # Longest sequences
GPU 1: [1, 5, 9, ...]  # Different complexity
```

**If GPU 2 encounters more memory-intensive sequences, GPUs 0,1,3 will idle**

**Recommendations**:
1. Dynamic work distribution with work queue
2. Smaller work units (10K sequences chunks)
3. Work stealing for idle GPUs
4. Performance-based redistribution

#### 5.4 Sequential Shard Aggregation
**Location**: `virnucpro/pipeline/shard_aggregator.py:154`

**Issue**: Single-threaded shard processing

```python
for shard_idx, shard_path in enumerate(shard_files):
    # ... sequential I/O
```

**Optimization**: Use ThreadPoolExecutor:
```python
with ThreadPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(read_shard, path) for path in shard_files]
```

**Expected**: 2-4x reduction in aggregation time

---

### 6. FlashAttention Integration Issues

#### 6.1 Sequential Position ID Generation
**Location**: `virnucpro/models/packed_attention.py:88-97`

**Issue**: Python loop creates CPU-GPU sync points

```python
for i in range(num_sequences):
    start = cu_seqlens[i].item()
    end = cu_seqlens[i + 1].item()
    seq_len = end - start
    position_ids[start:end] = torch.arange(seq_len, device=cu_seqlens.device)
```

**Optimization**: Use `torch.repeat_interleave`:
```python
seq_lengths = cu_seqlens[1:] - cu_seqlens[:-1]
position_ids = torch.cat([
    torch.arange(length, device=cu_seqlens.device) 
    for length in seq_lengths
])
```

#### 6.2 Missing FlashAttention-3 Support
**Location**: `virnucpro/models/packed_attention.py:23-41`

**Issue**: Only checks for flash-attn >= 2.6.0

**FlashAttention-3 benefits** (Hopper/H100):
- 1.5-2x speedup via warp group cluster
- FP8 support for inference
- Improved tile scheduling

**Recommendation**: Add FA3 detection:
```python
fa3_available = version.parse(flash_attn.__version__) >= version.parse("3.0.0")
if fa3_available and torch.cuda.get_device_capability()[0] >= 9:
    from flash_attn import flash_attn_varlen_func_v3
```

#### 6.3 RoPE Computation Overhead
**Location**: `virnucpro/models/esm2_flash.py:406-443`

**Issue**: Explicit FP32 conversion per layer (72 layers for ESM-2 3B)

```python
q_rot_fp32 = q_rot.float()
k_rot_fp32 = k_rot.float()
q_rot = ((q_rot_fp32 * cos) + (rotate_half(q_rot_fp32) * sin)).to(original_dtype)
```

**Impact**: ~3-5% overhead

**Recommendation**: Add `fast_rope` option:
```python
if self.fast_rope:
    q_rot = (q_rot * cos) + (rotate_half(q_rot) * sin)
else:
    # Current FP32 path for precision
```

#### 6.4 Missing torch.compile Integration

**Issue**: No PyTorch 2.x `torch.compile()` usage

**Recommendation**: Add optional compilation:
```python
if os.environ.get('VIRNUCPRO_COMPILE_MODEL', '').lower() == 'true':
    self.model = torch.compile(self.model, mode='reduce-overhead')
```

**Expected**: 10-20% inference speedup

---

### 7. Memory Management Issues

#### 7.1 No Tensor Pooling
**Location**: Throughout async_inference.py and collators.py

**Issue**: New tensors allocated for every batch → fragmentation

**Recommendation**: Implement tensor pooling:
```python
class TensorPool:
    def __init__(self):
        self._pools = defaultdict(list)
    
    def get(self, shape, dtype, device):
        key = (shape, dtype, device)
        if self._pools[key]:
            return self._pools[key].pop()
        return torch.empty(shape, dtype=dtype, device=device)
```

#### 7.2 Combined Device/Dtype Transfer
**Location**: `virnucpro/models/esm2_flash.py:78,90`

**Issue**: Two separate transfers

```python
self.model = self.model.to(device)   # Transfer 1
self.model = self.model.half()       # Transfer 2 (copies again)
```

**Optimization**: Single transfer:
```python
self.model = self.model.to(device=device, dtype=torch.float16)
```

#### 7.3 IndexBasedDataset Memory Footprint
**Location**: `virnucpro/data/sequence_dataset.py:387-412`

**Issue**: Stores all sequences before yielding

**Optimization**: Use generator pattern with `heapq.merge`

---

### 8. Code Quality & Maintainability

#### 8.1 Code Duplication

| Function | Files | Lines |
|----------|-------|-------|
| `_get_progress_queue()` | parallel_esm.py, parallel_dnabert.py, parallel.py | 29-40, 41-52, 11-22 |
| `_validate_cuda_isolation()` | sequence_dataset.py (2 classes) | 83-125, 245-287 |

**Recommendation**: Extract to shared utilities

#### 8.2 Oversized Functions

| Function | File | Lines | Complexity |
|----------|------|-------|------------|
| `run()` | async_inference.py | ~526 | Too high |
| `gpu_worker()` | gpu_worker.py | ~414 | Too high |
| `run_prediction()` | prediction.py | ~900 | Too high |

**Recommendation**: Apply extract method pattern

#### 8.3 Hardcoded Magic Numbers

| Value | Location | Recommendation |
|-------|----------|----------------|
| `checkpoint_seq_threshold=10000` | async_inference.py:127 | Move to constants |
| `buffer_size=2000` | collators.py:81 | Make configurable |
| `CRITICAL_THRESHOLD=0.80` | packing.py:176 | Document assumptions |
| Progress logging intervals | async_inference.py:727 | Configurable intervals |

#### 8.4 Environment Variable Sprawl

**Current**: Scattered across 4+ files:
- `VIRNUCPRO_DISABLE_PACKING`
- `VIRNUCPRO_DISABLE_FP16`
- `VIRNUCPRO_V1_ATTENTION`
- `VIRNUCPRO_VIRAL_CHECKPOINT_MODE`

**Recommendation**: Centralize:
```python
# virnucpro/core/env_config.py
@dataclass(frozen=True)
class EnvConfig:
    disable_packing: bool = False
    disable_fp16: bool = False
    # ... etc
```

#### 8.5 Type Safety Issues

**Missing Return Type Annotations**:
- `get_statistics()` returns `Dict[str, Any]` → Should use TypedDict
- `input_fingerprint: str = ""` → Should be `Optional[str]`

**Bare Except Clauses**:
- `prediction.py:756-759`: Catches SystemExit, KeyboardInterrupt

#### 8.6 Concurrency Issues

**AsyncCheckpointWriter Unbounded Queue**:
- `checkpoint_writer.py:193-196`: Comment acknowledges risk
- **Fix**: Add semaphore or bounded queue

**SIGTERM Handler Race Condition**:
- `gpu_worker.py:317-325`: Accesses `runner` before initialization
- **Fix**: Use atomic state flags

---

## Prioritized Recommendations

### Phase 1: Quick Wins (1-2 days, 10-15% improvement)

| Priority | Issue | File | Effort | Impact |
|----------|-------|------|--------|--------|
| 1 | Cache environment variables | async_inference.py | Low | High |
| 2 | Use deque for packed_queue | collators.py | Low | High |
| 3 | Combine .to() and .half() | esm2_flash.py | Low | Medium |
| 4 | Remove redundant token check | collators.py | Low | Medium |
| 5 | Centralize env vars | env_config.py | Low | High |

### Phase 2: Stream Optimization (3-5 days, 15-25% improvement)

| Priority | Issue | File | Effort | Impact |
|----------|-------|------|--------|--------|
| 1 | True pipeline parallelism | stream_manager.py | High | High |
| 2 | Vectorize position IDs | packed_attention.py | Low | Medium |
| 3 | Optimize embedding extraction | async_inference.py | Medium | High |
| 4 | Add torch.compile option | esm2_flash.py | Low | Medium |

### Phase 3: Multi-GPU Scaling (5-7 days, 20-30% improvement)

| Priority | Issue | File | Effort | Impact |
|----------|-------|------|--------|--------|
| 1 | Dynamic work distribution | shard_index.py | High | High |
| 2 | Parallel shard aggregation | shard_aggregator.py | Medium | High |
| 3 | Process pooling | gpu_coordinator.py | High | Medium |
| 4 | Model weight sharing | gpu_worker.py | High | Medium |

### Phase 4: Algorithm & Memory (7-10 days, 10-15% improvement)

| Priority | Issue | File | Effort | Impact |
|----------|-------|------|--------|--------|
| 1 | Implement BFD packing | packing.py | Medium | Medium |
| 2 | Add tensor pooling | async_inference.py | High | Medium |
| 3 | FlashAttention-3 support | packed_attention.py | Medium | Medium |
| 4 | Streaming dataset | sequence_dataset.py | Medium | Low |

### Phase 5: Code Quality (Ongoing)

| Priority | Issue | Scope | Effort | Impact |
|----------|-------|-------|--------|--------|
| 1 | Break down oversized functions | pipeline/ | High | Low |
| 2 | Extract duplicate code | multiple files | Low | Low |
| 3 | Fix race conditions | gpu_worker.py | Medium | High |
| 4 | Standardize logging | all files | Low | Low |

---

## Implementation Roadmap

### Immediate (Next Sprint)
- [ ] Cache environment variable lookups
- [ ] Replace list with deque for packed_queue
- [ ] Combine device/dtype transfers
- [ ] Centralize environment configuration

### Short-term (Next Month)
- [ ] Implement true pipeline parallelism with triple buffering
- [ ] Vectorize position ID generation
- [ ] Add torch.compile() integration
- [ ] Parallelize shard aggregation

### Medium-term (Next Quarter)
- [ ] Dynamic work distribution with work stealing
- [ ] Process pooling implementation
- [ ] BFD packing algorithm
- [ ] Tensor pooling for embeddings

### Long-term (Future Releases)
- [ ] FlashAttention-3 integration
- [ ] Model weight sharing across GPUs
- [ ] Streaming dataset without full materialization
- [ ] Comprehensive code refactoring

---

## Conclusion

The VirNucPro v2.0 architecture represents a **sophisticated production-grade GPU optimization pipeline** with excellent fault tolerance, checkpointing, and performance characteristics. The current implementation successfully achieves its primary goal of dramatically reducing embedding extraction time.

**Key strengths**:
- Well-designed async DataLoader architecture
- Proper CUDA isolation with spawn context
- Comprehensive checkpointing and fault tolerance
- Clean separation of concerns

**Primary opportunities**:
1. **Stream pipelining**: Current sequential stream usage leaves 15-25% performance on the table
2. **Multi-GPU load balancing**: Static distribution causes stragglers at scale
3. **Hot-path micro-optimizations**: Environment variable caching, deque usage, tensor pooling
4. **Algorithm improvements**: BFD packing, vectorized operations

**Expected Total Improvement**: **30-50% additional throughput** with all optimizations implemented, bringing the pipeline from ~10 hours to potentially **5-7 hours** for large workloads.

The codebase is well-positioned for these enhancements, with clear module boundaries and good test coverage providing confidence for iterative improvements.

---

## Appendix: File Index

### Core Pipeline Files
- `virnucpro/pipeline/async_inference.py` - Single-GPU inference runner
- `virnucpro/pipeline/multi_gpu_inference.py` - Multi-GPU entry point
- `virnucpro/pipeline/gpu_coordinator.py` - Process coordination
- `virnucpro/pipeline/gpu_worker.py` - Per-GPU worker
- `virnucpro/pipeline/shard_aggregator.py` - HDF5 merging

### Data Loading
- `virnucpro/data/sequence_dataset.py` - FASTA reading
- `virnucpro/data/collators.py` - VarlenCollator
- `virnucpro/data/packing.py` - GreedyPacker
- `virnucpro/data/dataloader_utils.py` - Async DataLoader factory
- `virnucpro/data/shard_index.py` - Length-sorted indexing

### CUDA & Models
- `virnucpro/cuda/stream_manager.py` - 3-stream orchestration
- `virnucpro/models/esm2_flash.py` - ESM-2 with FlashAttention
- `virnucpro/models/packed_attention.py` - Varlen utilities
- `virnucpro/models/dnabert_flash.py` - DNABERT-S patching

### Supporting Files
- `virnucpro/pipeline/checkpoint_writer.py` - Async checkpointing
- `virnucpro/pipeline/prediction.py` - Main pipeline orchestration
- `virnucpro/pipeline/parallel_esm.py` - ESM parallel processing
- `virnucpro/pipeline/parallel_dnabert.py` - DNABERT parallel processing
