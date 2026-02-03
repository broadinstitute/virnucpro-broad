# Phase 6: Sequence Packing Integration - Research

**Researched:** 2026-02-03
**Domain:** FlashAttention varlen, sequence packing, greedy bin-packing algorithms
**Confidence:** HIGH (existing codebase has Phase 5 foundation, prior research extensive, FlashAttention docs verified)

## Summary

Phase 6 implements sequence packing integration to maximize GPU utilization when processing variable-length protein sequences. The core work is replacing the Phase 5 `NotImplementedError` in `AsyncInferenceRunner._run_inference()` with a FlashAttention varlen forward pass that processes packed batches using the `cu_seqlens` metadata already produced by `VarlenCollator`.

The existing codebase already has:
- `VarlenCollator` producing packed format with `cu_seqlens` (Phase 5)
- FlashAttention context managers in `esm2_flash.py` (via PyTorch SDPA)
- GPU monitoring infrastructure for packing efficiency metrics

The new work requires:
1. Integrating `flash_attn_varlen_func` for packed sequence attention
2. Implementing position ID reset at sequence boundaries
3. Adding greedy First-Fit Decreasing packing algorithm
4. Validation tests comparing packed vs unpacked outputs
5. Dynamic token budget calculation based on GPU memory

**Primary recommendation:** Use `flash_attn_varlen_func` with `unpad_input`/`pad_input` helpers from `flash_attn.bert_padding` to handle the packed format. ESM-2 uses fair-esm (not HuggingFace), so FlashAttention varlen must be integrated at the forward pass level, not via `attn_implementation` config. The approach is to wrap the ESM attention computation with FlashAttention varlen kernels.

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| flash-attn | 2.6+ | `flash_attn_varlen_func` for packed sequences | Official FlashAttention API for variable-length attention |
| PyTorch | 2.2+ | SDPA with FlashAttention backend | Native FlashAttention support via `torch.nn.attention.sdpa_kernel` |
| fair-esm | 2.0.0 | ESM-2 model implementation | Current ESM-2 model in codebase |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| flash_attn.bert_padding | 2.6+ | `unpad_input`, `pad_input`, `index_first_axis` helpers | Converting between padded and packed formats |
| nvitop | 1.3+ | GPU memory monitoring for token budget | Dynamic memory-based batch sizing |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| flash_attn_varlen_func | PyTorch SDPA with manual masking | SDPA doesn't natively support cu_seqlens - must create block-diagonal mask (O(n^2) memory) |
| flash_attn.bert_padding helpers | Manual implementation | Helpers are well-tested, handle edge cases |
| Modifying ESM layers | Wrapper approach | Wrapper avoids modifying vendored esm code |

**Installation:**
```bash
pip install flash-attn --no-build-isolation
```

## Architecture Patterns

### Recommended Project Structure
```
virnucpro/
  data/
    collators.py              # EXISTING: VarlenCollator (no changes needed)
    packing.py                # NEW: GreedyPacker for FFD algorithm + validation
  models/
    esm2_flash.py            # MODIFY: Add packed forward path with flash_attn_varlen_func
  pipeline/
    async_inference.py       # MODIFY: Replace NotImplementedError with packed path
  cuda/
    attention_utils.py       # EXISTING: FlashAttention detection (no changes)
```

### Pattern 1: Packed Forward Pass with flash_attn_varlen_func

**What:** Process packed batches through ESM-2 using FlashAttention varlen attention
**When to use:** When `cu_seqlens` is present in batch (packed format from VarlenCollator)
**Example:**
```python
# Source: flash-attn documentation + existing esm2_flash.py patterns
from flash_attn import flash_attn_varlen_func
from flash_attn.bert_padding import unpad_input, pad_input

def forward_packed(self, input_ids: torch.Tensor, cu_seqlens: torch.Tensor,
                   max_seqlen: int) -> dict:
    """
    Forward pass for packed sequences.

    Args:
        input_ids: 1D tensor of concatenated token IDs [total_tokens]
        cu_seqlens: Cumulative sequence lengths [num_sequences + 1]
                    Format: [0, len1, len1+len2, ...], dtype=int32
        max_seqlen: Maximum sequence length in batch

    Returns:
        dict with 'representations' containing packed embeddings
    """
    batch_size = len(cu_seqlens) - 1

    # Create position IDs that reset at each sequence boundary
    position_ids = self._create_position_ids_packed(cu_seqlens)

    # Get embeddings (shape: [total_tokens, hidden_dim])
    embeddings = self.model.embed_tokens(input_ids)

    # Add position embeddings
    embeddings = embeddings + self.model.embed_positions(position_ids)

    # Process through transformer layers
    hidden_states = embeddings
    for layer in self.model.layers:
        hidden_states = self._layer_forward_packed(
            layer, hidden_states, cu_seqlens, max_seqlen
        )

    # Final layer norm
    hidden_states = self.model.emb_layer_norm_after(hidden_states)

    return {'representations': {36: hidden_states}}  # For compatibility
```

### Pattern 2: Position ID Reset at Sequence Boundaries

**What:** Generate per-sequence position IDs that reset to 0 at each cu_seqlens boundary
**When to use:** Always when packing sequences - position IDs must be relative to sequence start
**Example:**
```python
# Source: PITFALLS.md research + HuggingFace packing-with-FA2 blog
def create_position_ids_packed(cu_seqlens: torch.Tensor) -> torch.Tensor:
    """
    Create position IDs that reset at each sequence boundary.

    For cu_seqlens = [0, 3, 7, 10]:
        Returns: [0, 1, 2, 0, 1, 2, 3, 0, 1, 2]

    NOT: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (WRONG - sequential)
    """
    total_len = cu_seqlens[-1].item()
    position_ids = torch.zeros(total_len, dtype=torch.long, device=cu_seqlens.device)

    num_sequences = len(cu_seqlens) - 1
    for i in range(num_sequences):
        start = cu_seqlens[i].item()
        end = cu_seqlens[i + 1].item()
        seq_len = end - start
        position_ids[start:end] = torch.arange(seq_len, device=cu_seqlens.device)

    return position_ids
```

### Pattern 3: FlashAttention Varlen Layer Forward

**What:** Replace standard attention with flash_attn_varlen_func
**When to use:** In each transformer layer when processing packed batches
**Example:**
```python
# Source: flash-attn GitHub, bert.py model implementation
def layer_forward_packed(self, layer, hidden_states, cu_seqlens, max_seqlen):
    """
    Single layer forward with FlashAttention varlen.

    Args:
        hidden_states: [total_tokens, hidden_dim]
        cu_seqlens: [batch_size + 1] cumulative lengths
        max_seqlen: int maximum sequence length
    """
    # Self-attention with residual
    residual = hidden_states
    hidden_states = layer.self_attn_layer_norm(hidden_states)

    # Compute Q, K, V
    # Shape: [total_tokens, num_heads, head_dim]
    q, k, v = self._compute_qkv(layer, hidden_states)

    # FlashAttention varlen - automatically prevents cross-sequence attention
    attn_output = flash_attn_varlen_func(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        dropout_p=0.0,  # No dropout for inference
        causal=False,   # Bidirectional for BERT/ESM
    )

    # Reshape and project
    attn_output = attn_output.reshape(-1, self.hidden_dim)
    hidden_states = layer.self_attn.out_proj(attn_output)
    hidden_states = residual + hidden_states

    # Feed-forward with residual
    residual = hidden_states
    hidden_states = layer.final_layer_norm(hidden_states)
    hidden_states = layer.fc1(hidden_states)
    hidden_states = F.gelu(hidden_states)
    hidden_states = layer.fc2(hidden_states)
    hidden_states = residual + hidden_states

    return hidden_states
```

### Pattern 4: First-Fit Decreasing Packing Algorithm

**What:** Pack sequences into batches using greedy bin-packing with length sorting
**When to use:** Before creating batches - sort and pack for optimal efficiency
**Example:**
```python
# Source: CONTEXT.md decisions + FEATURES.md research
class GreedyPacker:
    """
    First-Fit Decreasing (FFD) packing algorithm.

    Sorts sequences by length descending, then greedily packs into bins
    until token budget is reached. Achieves ~90-95% packing efficiency.
    """

    def __init__(self, max_tokens_per_batch: int, max_sequence_length: int):
        self.max_tokens = max_tokens_per_batch
        self.max_length = max_sequence_length

    def pack_sequences(self, sequences: List[Dict]) -> List[List[Dict]]:
        """
        Pack sequences into efficient batches.

        Args:
            sequences: List of dicts with 'id', 'sequence', 'file' keys

        Returns:
            List of batches, each a list of sequence dicts
        """
        # Sort by length descending (FFD algorithm)
        sorted_seqs = sorted(sequences, key=lambda x: len(x['sequence']), reverse=True)

        # Deterministic tie-breaking by ID
        sorted_seqs = sorted(sorted_seqs, key=lambda x: (-len(x['sequence']), x['id']))

        batches = []
        current_batch = []
        current_tokens = 0

        for seq_dict in sorted_seqs:
            seq_len = len(seq_dict['sequence'])

            # Handle oversized sequences
            if seq_len > self.max_length:
                # Truncate with warning
                logger.warning(
                    f"Sequence {seq_dict['id']} exceeds max_length "
                    f"({seq_len} > {self.max_length}), truncating"
                )
                seq_dict['sequence'] = seq_dict['sequence'][:self.max_length]
                seq_dict['truncated'] = True
                seq_len = self.max_length

            # Check if fits in current batch
            # +2 for BOS/EOS tokens added during tokenization
            tokenized_len = seq_len + 2

            if current_tokens + tokenized_len <= self.max_tokens:
                current_batch.append(seq_dict)
                current_tokens += tokenized_len
            else:
                # Start new batch
                if current_batch:
                    batches.append(current_batch)
                current_batch = [seq_dict]
                current_tokens = tokenized_len

        # Add final batch
        if current_batch:
            batches.append(current_batch)

        return batches

    def compute_efficiency(self, batches: List[List[Dict]]) -> float:
        """Compute packing efficiency across all batches."""
        total_tokens = sum(
            sum(len(s['sequence']) + 2 for s in batch)
            for batch in batches
        )
        total_capacity = len(batches) * self.max_tokens
        return total_tokens / total_capacity if total_capacity > 0 else 0.0
```

### Pattern 5: Packed Output Embedding Extraction

**What:** Extract per-sequence embeddings from packed model output using cu_seqlens
**When to use:** After model forward to get individual sequence embeddings
**Example:**
```python
# Source: existing async_inference.py _extract_embeddings pattern
def extract_embeddings_packed(
    packed_output: torch.Tensor,
    cu_seqlens: torch.Tensor,
    sequence_ids: List[str]
) -> Dict[str, torch.Tensor]:
    """
    Extract per-sequence mean-pooled embeddings from packed output.

    Args:
        packed_output: [total_tokens, hidden_dim] from model
        cu_seqlens: [num_sequences + 1] cumulative lengths
        sequence_ids: List of sequence identifiers

    Returns:
        Dict mapping sequence_id -> embedding tensor [hidden_dim]
    """
    # Validate cu_seqlens format
    assert cu_seqlens[0] == 0, "cu_seqlens must start with 0"
    assert len(cu_seqlens) == len(sequence_ids) + 1, (
        f"cu_seqlens length {len(cu_seqlens)} != {len(sequence_ids)} + 1"
    )

    embeddings = {}
    for i, seq_id in enumerate(sequence_ids):
        start = cu_seqlens[i].item()
        end = cu_seqlens[i + 1].item()

        # Skip BOS token (position 0), mean-pool rest
        # ESM-2 adds BOS at start of each sequence
        if end - start > 1:
            seq_embedding = packed_output[start + 1:end].mean(dim=0)
        else:
            seq_embedding = packed_output[start:end].mean(dim=0)

        embeddings[seq_id] = seq_embedding.float()  # Convert to FP32

    return embeddings
```

### Anti-Patterns to Avoid

- **Sequential position IDs across batch:** Position IDs [0,1,2,3,4,5] across packed sequences corrupts positional understanding. Must reset at each cu_seqlens boundary
- **Using standard attention mask with packing:** 1D mask [1,1,1,0,0] doesn't prevent cross-sequence attention. Use cu_seqlens with flash_attn_varlen_func
- **Modifying fair-esm source:** Wrap with custom forward pass instead of editing vendored code
- **Creating cu_seqlens on GPU:** Compute cu_seqlens on CPU in collator, transfer to GPU is fast (small tensor)
- **FP32 inputs to flash_attn_varlen_func:** FlashAttention requires FP16/BF16. Ensure model dtype matches

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Packed sequence attention | Block-diagonal mask construction | `flash_attn_varlen_func` | Native cu_seqlens support, memory-efficient |
| Unpadding/padding conversions | Manual tensor slicing | `flash_attn.bert_padding.unpad_input/pad_input` | Handles edge cases, well-tested |
| Token budget calculation | Fixed estimation | nvitop memory monitoring | Dynamic GPU memory awareness |
| Packing efficiency metrics | Custom calculations | Existing `NvitopMonitor.get_dataloader_statistics()` | Already has packing_efficiency tracking |
| Position ID generation for packing | Manual index arithmetic | Vectorized implementation with arange | Less error-prone, faster |

**Key insight:** The codebase already has VarlenCollator producing packed format and GPU monitoring. Phase 6 fills the gap between data preparation and model inference.

## Common Pitfalls

### Pitfall 1: Position IDs Off-By-One at Sequence Boundaries

**What goes wrong:** Packed sequences use sequential [0,1,2,3,4,5] position IDs instead of resetting at each boundary [0,1,0,1,2,3]
**Why it happens:** Naive implementation generates position IDs for entire packed tensor, not per-sequence
**How to avoid:**
- Iterate over cu_seqlens boundaries
- Generate position IDs per sequence starting at 0
- Validation: `assert position_ids[cu_seqlens[i]] == 0` for all i
**Warning signs:** Model accuracy degrades with packing enabled vs disabled

### Pitfall 2: Cross-Sequence Attention Contamination

**What goes wrong:** Tokens from sequence A attend to tokens from sequence B in same pack
**Why it happens:** Using standard attention mask instead of flash_attn_varlen_func with cu_seqlens
**How to avoid:**
- Use flash_attn_varlen_func which enforces sequence boundaries via cu_seqlens
- Validation test: packed output must match unpacked output (cosine sim >0.999)
**Warning signs:** Packed embeddings differ significantly from unpacked; sequences with distinct characteristics show similar embeddings when packed

### Pitfall 3: FlashAttention Dtype Mismatch

**What goes wrong:** flash_attn_varlen_func crashes or silently falls back to standard attention
**Why it happens:** Model produces FP32 but FlashAttention requires FP16/BF16
**How to avoid:**
- Ensure model is in FP16: `model.half()` or autocast context
- cu_seqlens must be int32 (not int64)
- Validate dtype before first packed batch
**Warning signs:** "FlashAttention only supports fp16 and bf16" error; unexpected speedup loss

### Pitfall 4: cu_seqlens Format Errors

**What goes wrong:** Unpacking produces corrupted embeddings - wrong sequence boundaries
**Why it happens:** cu_seqlens computed as lengths [5,7,3] instead of cumulative [0,5,12,15]
**How to avoid:**
- cu_seqlens must start with 0
- Use cumsum: `cu_seqlens = torch.tensor([0] + lengths).cumsum(0)`
- len(cu_seqlens) == num_sequences + 1
- Validate in collator before GPU transfer
**Warning signs:** Index out of bounds during unpacking; embeddings contain wrong tokens

### Pitfall 5: Oversized Sequences Break Token Budget

**What goes wrong:** Single sequence exceeds token budget, causes OOM or batch rejection
**Why it happens:** No handling for sequences longer than max_tokens_per_batch
**How to avoid:**
- Truncate sequences >max_length with warning (CONTEXT.md decision)
- Log truncation statistics per batch
- Track which sequences were truncated for downstream analysis
**Warning signs:** Batches with single sequence; empty batches; OOM errors

## Code Examples

### Complete Integration Point (async_inference.py)

Replace the NotImplementedError in `_run_inference`:

```python
# Source: Phase 5 async_inference.py + flash_attn API
def _run_inference(self, gpu_batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Run model inference on GPU batch."""
    with torch.no_grad():
        input_ids = gpu_batch['input_ids']

        if 'cu_seqlens' in gpu_batch:
            # PACKED FORMAT: Use FlashAttention varlen
            cu_seqlens = gpu_batch['cu_seqlens']
            max_seqlen = gpu_batch['max_seqlen']

            # Forward pass with packed sequences
            outputs = self.model.forward_packed(
                input_ids=input_ids,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen
            )
            representations = outputs['representations'][36]
        else:
            # Standard padded format (fallback)
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
            outputs = self.model(input_ids, repr_layers=[36])
            representations = outputs['representations'][36]

        return representations
```

### Validation Test Pattern

```python
# Source: PACK-04 requirement + PITFALLS.md
def test_packed_unpacked_equivalence():
    """Verify packed output matches unpacked output."""
    sequences = [
        ('seq1', 'MKTAYIAKQR'),
        ('seq2', 'VLSPADKTNVKAAWGKV'),
        ('seq3', 'MVHLT'),
    ]

    # Process unpacked (baseline)
    unpacked_embeddings = {}
    for seq_id, seq in sequences:
        with torch.no_grad():
            tokens = tokenize([(seq_id, seq)])
            output = model(tokens)
            # Mean pool, skip BOS
            emb = output['representations'][36][0, 1:len(seq)+1].mean(dim=0)
            unpacked_embeddings[seq_id] = emb.float()

    # Process packed
    packed_batch = collator([{'id': s[0], 'sequence': s[1]} for s in sequences])
    packed_batch = {k: v.to(device) for k, v in packed_batch.items()}

    with torch.no_grad():
        packed_output = model.forward_packed(
            input_ids=packed_batch['input_ids'],
            cu_seqlens=packed_batch['cu_seqlens'],
            max_seqlen=packed_batch['max_seqlen']
        )

    packed_embeddings = extract_embeddings_packed(
        packed_output['representations'][36],
        packed_batch['cu_seqlens'],
        packed_batch['sequence_ids']
    )

    # Compare (PACK-04: cosine similarity >0.999)
    for seq_id in unpacked_embeddings:
        cos_sim = F.cosine_similarity(
            unpacked_embeddings[seq_id].unsqueeze(0),
            packed_embeddings[seq_id].unsqueeze(0)
        ).item()
        assert cos_sim > 0.999, f"{seq_id}: cosine_sim={cos_sim:.6f} < 0.999"
```

### Dynamic Token Budget Calculation

```python
# Source: PACK-03 requirement + nvitop patterns
def calculate_token_budget(device_id: int, model_memory_gb: float) -> int:
    """
    Calculate token budget based on available GPU memory.

    Args:
        device_id: CUDA device index
        model_memory_gb: Estimated model memory usage in GB

    Returns:
        Maximum tokens per batch
    """
    from nvitop import Device

    device = Device(device_id)
    total_memory_gb = device.memory_total() / (1024**3)

    # Reserve memory for: model + activations + safety margin
    available_memory_gb = total_memory_gb - model_memory_gb - 2.0  # 2GB safety

    # ESM-2 3B: ~2.5KB per token for intermediate activations
    # Conservative estimate: 4KB per token with overhead
    bytes_per_token = 4 * 1024

    max_tokens = int((available_memory_gb * 1024**3) / bytes_per_token)

    # Clamp to reasonable range
    max_tokens = max(1024, min(max_tokens, 16384))

    logger.info(f"Token budget: {max_tokens} (GPU {device_id}: "
                f"{available_memory_gb:.1f}GB available)")

    return max_tokens
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Padded batches | Packed with cu_seqlens | FlashAttention 2.0+ | 2-3x throughput, linear memory |
| Standard attention mask | flash_attn_varlen_func | FlashAttention varlen API | Native boundary enforcement |
| Fixed batch size | Dynamic token budget | Best practice 2024+ | Optimal GPU memory utilization |
| Sequential position IDs | Per-sequence reset | Required for packing | Correct positional embeddings |

**Deprecated/outdated:**
- `torch.backends.cuda.sdp_kernel()`: Deprecated in PyTorch 2.5+, use `torch.nn.attention.sdpa_kernel()` (already handled in codebase)
- Fixed-size batching: Token-budget batching is superior for variable-length sequences

## Open Questions

1. **ESM-2 attention layer access pattern**
   - What we know: fair-esm uses standard PyTorch attention
   - What's unclear: Exact interface for replacing attention computation
   - Recommendation: Inspect ESM-2 layer structure, may need to hook into attention call or wrap layer forward

2. **flash-attn version compatibility**
   - What we know: Version 2.6+ required for flash_attn_varlen_func
   - What's unclear: Exact version on GPU server
   - Recommendation: Check installed version, document minimum required

3. **Optimal token budget for ESM-2 3B**
   - What we know: 4096 used in Phase 5 collator
   - What's unclear: Whether this is optimal for 3B model on available GPUs
   - Recommendation: Start with 4096, adjust based on GPU memory monitoring

## Sources

### Primary (HIGH confidence)
- [FlashAttention GitHub](https://github.com/Dao-AILab/flash-attention) - flash_attn_varlen_func API, cu_seqlens format
- [PyTorch Variable Length Attention Tutorial](https://docs.pytorch.org/tutorials/intermediate/variable_length_attention_tutorial.html) - varlen_attn API
- Existing codebase: `virnucpro/data/collators.py` - VarlenCollator implementation
- Existing codebase: `virnucpro/models/esm2_flash.py` - Current FlashAttention integration
- `.planning/research/PITFALLS.md` - Comprehensive pitfall documentation
- `.planning/research/FEATURES.md` - Feature landscape and patterns

### Secondary (MEDIUM confidence)
- [HuggingFace Blog: Packing with Flash Attention 2](https://huggingface.co/blog/packing-with-FA2) - cu_seqlens format, position ID handling
- [Hacking FlashAttention for Variable-Length Inputs](https://gdewael.github.io/blog/flashattnvarlen/) - ESM-specific varlen patterns
- [flash_attn.bert_padding](https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/bert_padding.py) - unpad_input/pad_input helpers
- [ESME: Efficient ESM Implementation](https://www.biorxiv.org/content/10.1101/2024.10.22.619563v1.full.pdf) - ESM + FlashAttention integration patterns

### Tertiary (LOW confidence)
- [PyTorch Forums: Flash Attention with variable-length sequences](https://discuss.pytorch.org/t/flash-attention-with-variable-length-sequences/200901) - Community patterns
- [Bin Packing Algorithms](https://www.geeksforgeeks.org/dsa/bin-packing-problem-minimize-number-of-used-bins/) - FFD algorithm reference

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - flash-attn is canonical, well-documented
- Architecture patterns: HIGH - Based on existing codebase patterns + official examples
- Position ID handling: HIGH - Multiple sources confirm reset pattern
- FlashAttention varlen API: HIGH - Official API, verified signature
- ESM-2 integration: MEDIUM - May require layer inspection for exact hook points
- Packing efficiency: HIGH - FFD algorithm well-established

**Research date:** 2026-02-03
**Valid until:** 60 days (flash-attn API stable, core patterns established)

---

## Appendix: Phase 5 Integration Points

Key Phase 5 files requiring modification:

1. **`virnucpro/pipeline/async_inference.py`** (lines 141-149)
   - Replace `NotImplementedError` with packed forward call
   - Use existing `gpu_batch['cu_seqlens']` and `gpu_batch['max_seqlen']`

2. **`virnucpro/models/esm2_flash.py`**
   - Add `forward_packed()` method
   - Integrate flash_attn_varlen_func for packed attention

3. **`virnucpro/data/collators.py`** (VarlenCollator)
   - Already produces correct packed format
   - Consider adding packing efficiency logging

4. **New file: `virnucpro/data/packing.py`**
   - GreedyPacker class for FFD algorithm
   - Packing efficiency metrics
   - Validation utilities
