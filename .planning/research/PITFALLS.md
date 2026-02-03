# Pitfalls Research: Async DataLoader + Sequence Packing

**Domain:** PyTorch async DataLoader for GPU inference + transformer sequence packing
**Researched:** 2026-02-02
**Confidence:** HIGH

## Critical Pitfalls

### Pitfall 1: CUDA Tensors in DataLoader Workers Cause Silent Corruption

**What goes wrong:**
DataLoader workers create CUDA tensors in worker processes, causing one or more GPUs to receive corrupted/empty embeddings with no exceptions raised. Output files appear valid (correct sequence IDs, correct shape) but contain empty data tensors.

**Why it happens:**
CUDA runtime is not fork-safe. When worker processes access CUDA after forking, the CUDA context can become corrupted. Additionally, CUDA tensors cannot be safely shared between processes like CPU tensors - the worker may successfully create the tensor, but when transferred to the main process, the data is lost or corrupted.

**How to avoid:**
```python
# WRONG: Creating CUDA tensors in worker
def collate_fn(batch):
    tokens = tokenizer(batch)
    return tokens.to('cuda')  # BAD - CUDA in worker

# CORRECT: Move to CUDA in main process after DataLoader
for batch in dataloader:  # batch is CPU tensor
    batch = batch.to(device)  # Move to CUDA in main process
    with torch.no_grad():
        output = model(batch)
```

**Additional safeguards:**
- Use `multiprocessing_context='spawn'` (not fork) for DataLoader workers
- Set `pin_memory=True` and use `.to(device, non_blocking=True)` for async transfer
- Never instantiate CUDA models in worker `__init__` or `collate_fn`
- Keep all Dataset/collate operations on CPU, defer CUDA to main loop

**Warning signs:**
- Intermittent failures (succeeds sometimes, fails sometimes with identical command)
- Empty tensors with correct shape (e.g., `[0, 768]` instead of `[32, 768]`)
- One GPU consistently produces empty results while others work (round-robin failure)
- No exceptions or error messages (silent data corruption)
- Valid metadata (sequence IDs, counts) but zero predictions

**Phase to address:** Phase 1 (Foundation) - establish DataLoader patterns before adding complexity

---

### Pitfall 2: Concurrent Model Loading in Workers Causes HuggingFace Cache Race

**What goes wrong:**
When using `persistent_workers=True` with lazy model loading, multiple workers call `AutoModel.from_pretrained()` simultaneously on first batch, causing HuggingFace cache corruption. One worker loads a broken model that silently produces empty embeddings for all subsequent batches.

**Why it happens:**
HuggingFace's model cache is not designed for concurrent access from multiple processes. When workers start processing their first task simultaneously, they both attempt to download/read from `~/.cache/huggingface/`, causing file system race conditions. The corrupted worker doesn't crash - it loads malformed weights and produces plausible-looking but empty outputs.

**How to avoid:**
```python
# Strategy 1: Staggered loading with delay
def _load_model_lazy(self, worker_id, device_id):
    if self.model is None:
        # Stagger by worker ID to prevent cache contention
        if worker_id > 0:
            time.sleep(worker_id * 1.0)  # 1 second per worker

        self.model = AutoModel.from_pretrained(model_name)

# Strategy 2: Pre-load in main process before forking workers
# (Only works with 'spawn' context - fork copies CUDA context)
model = AutoModel.from_pretrained(model_name)  # Cache populated
# Now workers can load from cache safely

# Strategy 3: File locking for concurrent access
import filelock
lock_file = Path.home() / ".cache/huggingface/model.lock"
with filelock.FileLock(lock_file):
    model = AutoModel.from_pretrained(model_name)
```

**Warning signs:**
- 50% output on 2-GPU runs, 33% output on 3-GPU runs (failure rate = 1/num_workers)
- Consistent per-worker failure pattern (GPU 0 always fails, or GPU 1 always fails)
- Succeeds on second run without code changes (cache already populated)
- No stack traces or error messages
- Different behavior between `--persistent-models` and non-persistent modes

**Phase to address:** Phase 1 (Foundation) - critical for multi-GPU reliability

---

### Pitfall 3: FlashAttention Only Supports FP16/BF16, Silent Dtype Mismatch Breaks Packing

**What goes wrong:**
FlashAttention's `flash_attn_varlen_func` requires FP16 or BF16 inputs. When sequence packing uses FP32 attention masks or position IDs, FlashAttention either crashes with "only supports fp16 and bf16" error, or worse, silently falls back to standard attention without the packing-aware masking, causing cross-contamination between packed sequences.

**Why it happens:**
Sequence packing requires variable-length attention kernels (`flash_attn_varlen_func`) which enforce dtype restrictions. Regular model loading may default to FP32, and creating attention masks/position IDs often defaults to `torch.long` or `torch.float32`. The dtype mismatch is not always caught at model load time - it only manifests when the first packed batch is processed.

**How to avoid:**
```python
# Explicit dtype alignment for packing inputs
def create_packed_batch(sequences, model_dtype=torch.bfloat16):
    # Pack sequences and compute cumulative lengths
    packed_input_ids, cu_seqlens = pack_sequences(sequences)

    # CRITICAL: Match model dtype exactly
    packed_input_ids = packed_input_ids.to(dtype=torch.long)  # IDs are long
    attention_mask = create_packing_mask(cu_seqlens).to(dtype=model_dtype)
    position_ids = create_position_ids(cu_seqlens).to(dtype=torch.long)

    return {
        'input_ids': packed_input_ids,
        'attention_mask': attention_mask,  # Must match model dtype
        'position_ids': position_ids,
        'cu_seqlens': cu_seqlens.to(dtype=torch.int32)  # Flash requires int32
    }

# Load model with explicit dtype
model = AutoModel.from_pretrained(
    model_name,
    use_flash_attention_2=True,
    torch_dtype=torch.bfloat16,  # Explicit dtype, not auto
    attn_implementation="flash_attention_2"
)

# Verify dtype before packing
assert model.dtype in [torch.float16, torch.bfloat16], \
    f"FlashAttention requires FP16/BF16, got {model.dtype}"
```

**Warning signs:**
- Error: "FlashAttention only supports fp16 and bf16 data type"
- Error: "expected attention_mask dtype to be bool or match query dtype"
- Unexpected speedup loss (Flash falls back to standard attention silently)
- Cross-contamination in outputs (sequence 1 attention leaks into sequence 2)
- Works with single sequences but fails with packed batches

**Phase to address:** Phase 2 (Packing Integration) - must validate dtype compatibility before enabling packing

---

### Pitfall 4: Sequence Packing Position IDs Off-By-One Bug Corrupts Positional Embeddings

**What goes wrong:**
When packing multiple sequences into one tensor with `cu_seqlens = [0, 2, 6, 7]`, position IDs for the second and third sequences start from the cumulative offset instead of 0, causing transformers to see position [2, 3, 4, 5] for the second sequence instead of [0, 1, 2, 3]. This corrupts positional embeddings and degrades model accuracy silently.

**Why it happens:**
Naive packing concatenates sequences and generates position IDs sequentially `[0, 1, 2, 3, 4, 5, 6]` for the packed tensor. However, each sequence needs position IDs relative to its own start, not the packed tensor's start. The `cu_seqlens` array defines sequence boundaries, but position ID generation must reset at each boundary.

**How to avoid:**
```python
def create_position_ids_for_packing(cu_seqlens):
    """
    Generate per-sequence position IDs for packed input.

    Example:
        cu_seqlens = [0, 2, 6, 7]  # 3 sequences: len 2, 4, 1
        Returns: [0, 1, 0, 1, 2, 3, 0]  # Reset per sequence
    """
    position_ids = []

    for i in range(len(cu_seqlens) - 1):
        seq_start = cu_seqlens[i]
        seq_end = cu_seqlens[i + 1]
        seq_len = seq_end - seq_start

        # Position IDs reset to 0 for each sequence
        position_ids.append(torch.arange(seq_len, dtype=torch.long))

    return torch.cat(position_ids)

# WRONG - sequential position IDs
total_len = cu_seqlens[-1]
position_ids = torch.arange(total_len)  # [0,1,2,3,4,5,6] - WRONG

# CORRECT - per-sequence position IDs
position_ids = create_position_ids_for_packing(cu_seqlens)  # [0,1,0,1,2,3,0]
```

**Validation test:**
```python
def test_position_ids_reset():
    cu_seqlens = torch.tensor([0, 2, 6, 7])
    position_ids = create_position_ids_for_packing(cu_seqlens)

    # Each sequence should start at position 0
    assert position_ids[0] == 0  # Seq 1 starts at 0
    assert position_ids[2] == 0  # Seq 2 starts at 0
    assert position_ids[6] == 0  # Seq 3 starts at 0

    # Check max position per sequence
    assert position_ids[1] == 1  # Seq 1 max (len 2)
    assert position_ids[5] == 3  # Seq 2 max (len 4)
```

**Warning signs:**
- Packing works (no crashes) but model accuracy degrades compared to non-packed
- Longer sequences in pack show higher accuracy degradation
- Positional embedding visualization shows discontinuities
- Attention patterns show unexpected boundary artifacts

**Phase to address:** Phase 2 (Packing Integration) - validation tests before integration

---

### Pitfall 5: Attention Mask Cross-Contamination Between Packed Sequences

**What goes wrong:**
Standard attention masks allow tokens from sequence 1 to attend to tokens from sequence 2 in the same packed batch, causing the model to mix information between independent sequences. Outputs appear valid but contain contaminated predictions that mix features from multiple unrelated sequences.

**Why it happens:**
Packing creates `input_ids = [seq1_tokens, seq2_tokens, padding]` with shape `[total_len]`. A standard attention mask `[1,1,1,1,1,0,0]` (1=attend, 0=ignore padding) doesn't prevent seq1 tokens from attending to seq2 tokens. FlashAttention's varlen kernel requires a 1D `cu_seqlens` array to define boundaries, but if not provided, all non-padding tokens attend to each other.

**How to avoid:**
```python
# For FlashAttention with packing (varlen API)
from flash_attn import flash_attn_varlen_func

def forward_packed_batch(model, packed_inputs, cu_seqlens, max_seqlen):
    """
    Process packed batch with proper masking.

    Args:
        packed_inputs: [total_len, hidden_dim] - concatenated sequences
        cu_seqlens: [num_seqs + 1] - cumulative sequence lengths [0, len1, len1+len2, ...]
        max_seqlen: int - max sequence length in pack
    """
    # FlashAttention varlen automatically masks between sequences
    # based on cu_seqlens - no cross-contamination
    output = flash_attn_varlen_func(
        q=packed_inputs,
        k=packed_inputs,
        v=packed_inputs,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        dropout_p=0.0,
        causal=False  # For BERT-like models
    )
    return output

# For standard attention (fallback when Flash unavailable)
def create_block_diagonal_mask(cu_seqlens, device):
    """
    Create 2D block-diagonal mask preventing cross-sequence attention.

    Example for cu_seqlens=[0,2,5]:
        [[1,1,0,0,0],   # Seq1 token 0 attends only to seq1
         [1,1,0,0,0],   # Seq1 token 1 attends only to seq1
         [0,0,1,1,1],   # Seq2 token 0 attends only to seq2
         [0,0,1,1,1],   # Seq2 token 1 attends only to seq2
         [0,0,1,1,1]]   # Seq2 token 2 attends only to seq2
    """
    total_len = cu_seqlens[-1]
    mask = torch.zeros(total_len, total_len, device=device)

    for i in range(len(cu_seqlens) - 1):
        start = cu_seqlens[i]
        end = cu_seqlens[i + 1]
        # Allow attention within sequence boundaries only
        mask[start:end, start:end] = 1

    return mask.bool()
```

**Validation test:**
```python
def test_no_cross_contamination():
    """Verify sequences don't contaminate each other."""
    seq1 = "ACGT" * 10  # DNA sequence 1
    seq2 = "TTTT" * 10  # DNA sequence 2 (different composition)

    # Process separately
    emb1_solo = model(seq1)
    emb2_solo = model(seq2)

    # Process packed together
    packed_input, cu_seqlens = pack_sequences([seq1, seq2])
    packed_output = model_with_packing(packed_input, cu_seqlens)
    emb1_packed = packed_output[:len(seq1)]
    emb2_packed = packed_output[len(seq1):len(seq1)+len(seq2)]

    # Embeddings should be identical (within numerical precision)
    assert torch.allclose(emb1_solo, emb1_packed, atol=1e-5)
    assert torch.allclose(emb2_solo, emb2_packed, atol=1e-5)
```

**Warning signs:**
- Packed outputs differ from non-packed outputs for same sequences
- Validation accuracy drops when packing is enabled
- Sequences with distinct characteristics (e.g., GC-rich vs AT-rich DNA) show similar embeddings when packed together
- Attention visualization shows off-diagonal blocks (cross-sequence attention)

**Phase to address:** Phase 2 (Packing Integration) - correctness validation before performance testing

---

### Pitfall 6: DataLoader Persistent Workers Memory Leak with Prefetching

**What goes wrong:**
Using `persistent_workers=True` with `prefetch_factor > 2` causes gradual memory accumulation on CPU RAM. With 8 workers and prefetch_factor=16, each worker can consume 5-10GB of host memory, leading to OOM on systems with less than 128GB RAM. Memory is not released between batches or even between epochs.

**Why it happens:**
Each worker pre-fetches `prefetch_factor` batches ahead of consumption. With persistent workers, these batches stay in worker memory even after being consumed by the main process. The issue is exacerbated by `pin_memory=True`, which allocates additional pinned (non-pageable) memory via `cudaHostAlloc`. The pinned memory is not released until workers terminate.

**How to avoid:**
```python
# Conservative configuration for inference
def create_inference_dataloader(dataset, batch_size, num_gpus):
    cpu_count = multiprocessing.cpu_count()
    num_workers = min(cpu_count // num_gpus, 8)  # Cap at 8

    # Conservative prefetch for inference (not training)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2,  # Default, don't increase for inference
        persistent_workers=True if num_workers > 0 else False,
        multiprocessing_context='spawn'
    )
    return dataloader

# Monitor memory usage
def log_worker_memory():
    import psutil
    process = psutil.Process()
    children = process.children(recursive=True)
    for child in children:
        mem_mb = child.memory_info().rss / 1024**2
        if mem_mb > 1000:  # Warn if worker uses >1GB
            logger.warning(f"Worker {child.pid} using {mem_mb:.0f}MB")
```

**Detection script:**
```python
# Add to DataLoader loop for debugging
for i, batch in enumerate(dataloader):
    process_batch(batch)

    if i % 100 == 0:  # Check every 100 batches
        # Log memory growth
        allocated = torch.cuda.memory_allocated() / 1024**3
        import psutil
        ram_gb = psutil.virtual_memory().used / 1024**3
        logger.info(f"Batch {i}: GPU={allocated:.2f}GB, RAM={ram_gb:.2f}GB")
```

**Warning signs:**
- Host RAM usage steadily increases during inference
- `htop` shows DataLoader worker processes growing over time
- System swap usage increases during long runs
- Workers take >60 seconds to terminate after DataLoader finishes
- Out-of-memory on CPU (not GPU) after processing many batches

**Phase to address:** Phase 1 (Foundation) - establish safe defaults before scaling

---

### Pitfall 7: Sequence Packing Efficiency Loss from Improper Batch Construction

**What goes wrong:**
Packing random-length sequences into fixed-size batches results in high padding overhead (40-60% wasted computation) because short and long sequences are mixed in the same pack. The theoretical 2-3x speedup from packing degrades to 1.2x or less in practice.

**Why it happens:**
Transformers have quadratic attention complexity O(n²). Packing [100nt, 3000nt, 150nt] creates a batch with max_len=3000, so the 100nt and 150nt sequences still compute attention over 3000 positions worth of padding. Without sorting, random packing frequently creates these worst-case scenarios.

**How to avoid:**
```python
def create_efficient_packs(sequences, max_pack_len=4096, pack_tolerance=0.9):
    """
    Create efficient sequence packs using bin-packing algorithm.

    Args:
        sequences: List of sequences with varying lengths
        max_pack_len: Maximum total length per pack
        pack_tolerance: Minimum packing efficiency (0.9 = 90% utilization)
    """
    # Sort by length descending (greedy bin packing works better)
    sorted_seqs = sorted(sequences, key=len, reverse=True)

    packs = []
    current_pack = []
    current_len = 0

    for seq in sorted_seqs:
        seq_len = len(seq)

        # Check if sequence fits in current pack
        if current_len + seq_len <= max_pack_len:
            current_pack.append(seq)
            current_len += seq_len
        else:
            # Current pack is full, start new pack
            if current_pack:
                efficiency = current_len / max_pack_len
                if efficiency < pack_tolerance:
                    logger.warning(f"Low pack efficiency: {efficiency:.2%}")
                packs.append(current_pack)

            current_pack = [seq]
            current_len = seq_len

    # Add final pack
    if current_pack:
        packs.append(current_pack)

    # Log efficiency statistics
    efficiencies = [sum(len(s) for s in pack) / max_pack_len for pack in packs]
    avg_efficiency = sum(efficiencies) / len(efficiencies)
    logger.info(f"Created {len(packs)} packs with {avg_efficiency:.2%} avg efficiency")

    return packs

# WRONG: Random packing
packs = [sequences[i:i+4] for i in range(0, len(sequences), 4)]

# CORRECT: Sorted bin-packing
packs = create_efficient_packs(sequences, max_pack_len=4096)
```

**Efficiency calculation:**
```python
def measure_packing_efficiency(packed_batch, cu_seqlens):
    """
    Measure wasted computation from padding in packed batch.

    Efficiency = actual_tokens / (num_sequences * max_seq_len)
    """
    num_sequences = len(cu_seqlens) - 1
    total_tokens = cu_seqlens[-1].item()  # Actual tokens

    # Compute max sequence length in pack
    seq_lengths = [cu_seqlens[i+1] - cu_seqlens[i] for i in range(num_sequences)]
    max_seq_len = max(seq_lengths)

    # Wasted computation (quadratic in max_len)
    theoretical_ops = num_sequences * max_seq_len * max_seq_len
    actual_ops = sum(l * l for l in seq_lengths)

    efficiency = actual_ops / theoretical_ops
    padding_waste = 1 - (total_tokens / (num_sequences * max_seq_len))

    logger.debug(f"Pack efficiency: {efficiency:.2%}, padding waste: {padding_waste:.2%}")
    return efficiency
```

**Warning signs:**
- Packing speedup is 1.1-1.3x instead of expected 2-3x
- High variance in batch processing time (some batches 10x slower)
- Memory usage close to non-packed batches
- Profiling shows high padding percentage (>40%)
- GPU utilization varies widely between batches

**Phase to address:** Phase 3 (Optimization) - after correctness is validated

---

### Pitfall 8: Unpacking Corruption from Misaligned cu_seqlens

**What goes wrong:**
After processing a packed batch through the model, unpacking the output using incorrect `cu_seqlens` offsets produces corrupted embeddings. Sequence boundaries are off by 1-2 positions, causing each unpacked sequence to include tokens from its neighbor or miss its own final tokens.

**Why it happens:**
`cu_seqlens` must be **cumulative** offsets starting at 0. If computed as lengths `[2, 4, 1]` instead of cumulative `[0, 2, 6, 7]`, unpacking slices the wrong regions. Off-by-one errors also occur when adding padding tokens to `cu_seqlens` calculation but not to the actual packed tensor.

**How to avoid:**
```python
def unpack_sequences(packed_output, cu_seqlens, sequence_ids):
    """
    Unpack model output back to individual sequences.

    Args:
        packed_output: [total_len, hidden_dim] - packed model output
        cu_seqlens: [num_seqs + 1] - cumulative sequence lengths
        sequence_ids: List of original sequence IDs for validation

    Returns:
        Dict mapping sequence_id -> embedding tensor
    """
    unpacked = {}

    # Validate cu_seqlens format
    assert cu_seqlens[0] == 0, "cu_seqlens must start with 0"
    assert len(cu_seqlens) == len(sequence_ids) + 1, \
        f"cu_seqlens length {len(cu_seqlens)} != num_sequences {len(sequence_ids)} + 1"

    for i, seq_id in enumerate(sequence_ids):
        start = cu_seqlens[i].item()
        end = cu_seqlens[i + 1].item()

        # Extract sequence embedding
        seq_embedding = packed_output[start:end]

        # Validation: Check expected length
        expected_len = end - start
        assert seq_embedding.shape[0] == expected_len, \
            f"Sequence {seq_id}: extracted {seq_embedding.shape[0]} tokens, expected {expected_len}"

        unpacked[seq_id] = seq_embedding

    return unpacked

# WRONG: Using lengths instead of cumulative
seq_lengths = [len(seq) for seq in sequences]
cu_seqlens = torch.tensor(seq_lengths)  # [2, 4, 1] - WRONG

# CORRECT: Cumulative sum
seq_lengths = [len(seq) for seq in sequences]
cu_seqlens = torch.tensor([0] + seq_lengths).cumsum(0)  # [0, 2, 6, 7] - CORRECT
```

**Validation test:**
```python
def test_pack_unpack_roundtrip():
    """Verify packing and unpacking preserves sequence identity."""
    sequences = [
        torch.randn(10, 768),  # Seq 0: 10 tokens
        torch.randn(25, 768),  # Seq 1: 25 tokens
        torch.randn(5, 768),   # Seq 2: 5 tokens
    ]

    # Pack sequences
    packed, cu_seqlens = pack_sequences(sequences)
    assert packed.shape[0] == 10 + 25 + 5  # Total length
    assert cu_seqlens.tolist() == [0, 10, 35, 40]

    # Unpack sequences
    unpacked = unpack_sequences(packed, cu_seqlens, range(3))

    # Verify identity
    for i, original_seq in enumerate(sequences):
        assert torch.allclose(unpacked[i], original_seq), \
            f"Sequence {i} corrupted in pack/unpack roundtrip"
```

**Warning signs:**
- Assertion errors during unpacking: "extracted X tokens, expected Y"
- Embeddings for short sequences contain data from long sequences
- Final tokens of sequences are missing (clipped)
- First tokens of sequences are duplicated from previous sequence
- Index out of bounds errors during unpacking

**Phase to address:** Phase 2 (Packing Integration) - unit tests before integration

---

## Technical Debt Patterns

Shortcuts that seem reasonable but create long-term problems.

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Skip dtype validation for packing | Faster development iteration | Silent accuracy degradation, hard-to-debug contamination | Never - always validate dtype compatibility |
| Use high prefetch_factor (>4) for faster loading | Better GPU utilization in training | Memory leaks in inference, OOM on long runs | Only in training with worker restart between epochs |
| Random sequence packing without sorting | Simple implementation | 40-60% efficiency loss, negates packing benefits | Early prototyping only, must optimize before production |
| Reuse training DataLoader config for inference | Code reuse | Persistent worker memory leaks, excessive resource usage | Never - inference needs different config |
| Skip cross-contamination validation | Trust library implementation | Silent correctness bugs that corrupt predictions | Never - always validate with reference implementation |
| Use fork context for faster worker startup | 2-3x faster initialization | CUDA context corruption, intermittent failures | Never with CUDA - always use spawn |

## Integration Gotchas

Common mistakes when connecting async DataLoader to GPU inference pipeline.

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| DataLoader → CUDA model | Creating CUDA tensors in worker collate_fn | Keep workers CPU-only, move to CUDA in main process after DataLoader |
| Persistent workers + lazy model loading | All workers load models simultaneously (cache race) | Stagger loading with worker_id * delay or pre-populate cache before workers start |
| FlashAttention + sequence packing | Using standard attention mask format (1D) with packed sequences | Use cu_seqlens with flash_attn_varlen_func or create 2D block-diagonal mask |
| Variable-length batching | Pack sequences in random order | Sort by length and use bin-packing to minimize padding waste |
| Multi-GPU + DataLoader workers | num_workers = cpu_count (ignores GPU count) | num_workers = min(cpu_count // num_gpus, 8) to balance resources |
| Pin memory for GPU transfer | Enable without checking available RAM | Check system RAM, disable if <32GB or provide --pin-memory flag |

## Performance Traps

Patterns that work at small scale but fail as usage grows.

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| Linear scaling of num_workers | Memory explosion, swap usage | Cap at min(cpu_count//num_gpus, 8) regardless of CPU count | >16 workers or <8GB RAM per worker |
| Persistent workers without cleanup | Gradual memory leak over hours | Use persistent_workers=False for inference or implement periodic worker restart | Long-running jobs (>10K batches) |
| High prefetch_factor for throughput | 5-10GB RAM per worker, pinned memory exhaustion | Use prefetch_factor=2 for inference, 4 max for training | >4 workers or <64GB RAM |
| Packing without efficiency monitoring | 40-60% wasted computation, no speedup | Log packing efficiency, warn if <90% utilization | Variable-length sequences (100-3000nt range) |
| Assuming FlashAttention always faster | Slowdown on small batches or short sequences | Measure and compare, fall back for batch_size<8 or seq_len<128 | Small-scale inference (batch_size=1-4) |

## "Looks Done But Isn't" Checklist

Things that appear complete but are missing critical pieces.

- [ ] **Sequence packing:** Often missing per-sequence position ID reset — verify position_ids[cu_seqlens[i]] == 0 for all sequences
- [ ] **FlashAttention integration:** Often missing dtype validation — verify model.dtype in [torch.float16, torch.bfloat16] before packing
- [ ] **DataLoader workers:** Often missing spawn context specification — verify multiprocessing_context='spawn' explicitly set
- [ ] **Persistent workers:** Often missing memory monitoring — verify worker memory stays constant over 1000+ batches
- [ ] **Cross-contamination prevention:** Often missing validation test — verify packed output == non-packed output for same sequences
- [ ] **Unpacking logic:** Often missing cu_seqlens validation — verify cu_seqlens[0]==0 and lengths match sequence count
- [ ] **Concurrent model loading:** Often missing stagger/lock mechanism — verify only one worker loads from HF cache at a time
- [ ] **Packing efficiency:** Often missing bin-packing algorithm — verify average pack utilization >85%

## Recovery Strategies

When pitfalls occur despite prevention, how to recover.

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| CUDA corruption in workers | MEDIUM | Add multiprocessing_context='spawn', move all .to(device) calls to main process, restart job |
| HuggingFace cache race | LOW | Add 1-second stagger delay in worker lazy loading or use filelock around from_pretrained() |
| FlashAttention dtype mismatch | LOW | Add explicit torch_dtype=torch.bfloat16 to model loading, validate before processing |
| Position ID corruption | MEDIUM | Implement create_position_ids_for_packing() with per-sequence reset, add validation test |
| Cross-contamination | HIGH | Implement block-diagonal masking or migrate to flash_attn_varlen_func with cu_seqlens |
| Memory leak from prefetch | LOW | Reduce prefetch_factor to 2, disable persistent_workers for inference |
| Poor packing efficiency | MEDIUM | Implement bin-packing with length sorting, monitor and log efficiency per batch |
| Unpacking corruption | LOW | Fix cu_seqlens calculation (must be cumulative starting at 0), add roundtrip test |

## Pitfall-to-Phase Mapping

How roadmap phases should address these pitfalls.

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| CUDA tensors in workers | Phase 1 (Foundation) | Unit test: worker returns CPU tensor, main process moves to CUDA |
| HuggingFace cache race | Phase 1 (Foundation) | Integration test: 10 runs with --persistent-models all succeed |
| FlashAttention dtype mismatch | Phase 2 (Packing Integration) | Startup check: assert model.dtype in [fp16, bf16] before packing |
| Position ID corruption | Phase 2 (Packing Integration) | Unit test: position_ids reset per sequence in cu_seqlens |
| Cross-contamination | Phase 2 (Packing Integration) | Validation test: packed output matches non-packed for same sequences |
| Prefetch memory leak | Phase 1 (Foundation) | Stress test: 1000 batches with stable worker memory |
| Packing efficiency loss | Phase 3 (Optimization) | Monitoring: log pack efficiency, warn if <85% |
| Unpacking corruption | Phase 2 (Packing Integration) | Unit test: pack/unpack roundtrip preserves sequence identity |

## Migration-Specific Pitfalls (v1.0 → v2.0)

### Pitfall 9: Spawn Context Already Established Prevents Multi-Worker DataLoader

**What goes wrong:**
V1.0 uses `multiprocessing.set_start_method('spawn')` globally for GPU workers. When v2.0 adds DataLoader with `multiprocessing_context='spawn'`, Python raises "context has already been set" error, preventing DataLoader creation.

**Why it happens:**
`set_start_method()` can only be called once per process. V1.0's global call locks the context before DataLoader tries to set it via the `multiprocessing_context` parameter.

**How to avoid:**
```python
# V1.0 pattern (global set)
if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)  # Forces override
    # ... rest of code

# V2.0 pattern (per-DataLoader context)
# Remove global set_start_method() call
# Pass context to DataLoader explicitly
dataloader = DataLoader(
    dataset,
    num_workers=4,
    multiprocessing_context='spawn'  # Works if not set globally
)

# Migration compatibility pattern
import multiprocessing as mp
try:
    mp.set_start_method('spawn', force=False)
except RuntimeError:
    pass  # Already set, DataLoader will use existing context
```

**Phase to address:** Phase 1 (Foundation) - resolve before adding async DataLoader

---

### Pitfall 10: Persistent Model Loading Incompatible with DataLoader Worker Lifecycle

**What goes wrong:**
V1.0's persistent model pattern loads models once and reuses them across batches. DataLoader workers are forked/spawned per-epoch, so models must be reloaded every epoch, negating the persistent model optimization and causing 10-20 second overhead per epoch.

**Why it happens:**
V1.0 uses long-lived worker processes (via Pool) that persist for the entire run. DataLoader workers have a different lifecycle - they're created per DataLoader instantiation and destroyed when the DataLoader is exhausted. Loading 3B parameter models in worker `__init__` happens every epoch.

**How to avoid:**
```python
# Strategy 1: Load models in main process (single-GPU only)
model = load_model().to(device)
for batch in dataloader:  # Workers provide data only
    batch = batch.to(device)
    with torch.no_grad():
        output = model(batch)

# Strategy 2: Use persistent_workers=True (keeps workers alive)
dataloader = DataLoader(
    dataset,
    num_workers=4,
    persistent_workers=True  # Workers persist across epochs
)

# Workers load model once in __init__
class InferenceDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences
        self.model = None  # Lazy load in worker

    def __getitem__(self, idx):
        if self.model is None:
            self.model = load_model()  # Load once per worker
        # Process with model...
```

**Warning signs:**
- Model loading logs appear every epoch/DataLoader instantiation
- 10-20 second delay at start of each epoch
- GPU memory shows model being loaded and unloaded repeatedly

**Phase to address:** Phase 1 (Foundation) - critical for performance parity with v1.0

---

## Sources

### Primary (HIGH confidence)
- [PyTorch CUDA semantics documentation](https://docs.pytorch.org/docs/stable/notes/cuda.html) - CUDA multiprocessing safety, tensor sharing restrictions
- [PyTorch Multiprocessing best practices](https://docs.pytorch.org/docs/stable/notes/multiprocessing.html) - Fork vs spawn, CUDA initialization timing
- [PyTorch DataLoader documentation](https://docs.pytorch.org/docs/stable/data.html) - Worker configuration, pin_memory, prefetch_factor
- [FlashAttention only supports fp16 and bf16 - Issue #822](https://github.com/Dao-AILab/flash-attention/issues/822) - Dtype restrictions and error patterns
- [Packing with Flash Attention 2 - Hugging Face Blog](https://huggingface.co/blog/packing-with-FA2) - cu_seqlens format and usage
- [Position IDs bug with packed sequences - PR #7754](https://github.com/hiyouga/LLaMA-Factory/pull/7754) - Documented position ID corruption pattern

### Secondary (MEDIUM confidence)
- [Guidelines for assigning num_workers - PyTorch Forums](https://discuss.pytorch.org/t/guidelines-for-assigning-num-workers-to-dataloader/813) - Worker count recommendations
- [DataLoader memory leak with prefetch_factor - Issue #97432](https://github.com/pytorch/pytorch/issues/97432) - Pinned memory leak pattern
- [Efficient LLM Pretraining: Packed Sequences - Hugging Face Blog](https://huggingface.co/blog/sirluk/llm-sequence-packing) - Cross-contamination prevention
- [Dynamic Batching vs Sequence Packing - Medium](https://medium.com/better-ml/dynamic-batching-vs-sequence-packing-0ef4a3894dad) - Packing efficiency tradeoffs
- [PhoenixOS: GPU Checkpoint with Validated Speculation](https://arxiv.org/abs/2405.12079) - GPU checkpoint race conditions

### Tertiary (MEDIUM-HIGH confidence - project-specific)
- VirNucPro v1.0 debugging logs - empty-files-race-condition.md (HuggingFace cache race pattern)
- VirNucPro v1.0 debugging logs - flashattention-not-integrated.md (wrapper integration gap)
- VirNucPro Phase 4 research - 04-RESEARCH.md (FlashAttention patterns, DataLoader configuration)

---
*Pitfalls research for: VirNucPro v2.0 async DataLoader + sequence packing migration*
*Researched: 2026-02-02*
*Focus: Migration risks from v1.0 multi-worker to v2.0 async DataLoader + sequence packing*
