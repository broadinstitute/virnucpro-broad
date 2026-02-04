# Debug: Packed vs Unpacked Equivalence Issue

## Status
**10/11 tests passing** - Only `test_mixed_lengths` fails due to the "large" (400 AA) sequence having 0.9911 cosine similarity instead of >0.995.

## Problem Summary

| Sequence | Length | Cosine Similarity | Status |
|----------|--------|-------------------|--------|
| tiny     | 3 AA   | 0.9999+           | PASS   |
| small    | 8 AA   | 0.9999+           | PASS   |
| medium   | 160 AA | 0.9999+           | PASS   |
| large    | 400 AA | 0.9911            | FAIL   |

The similarity degrades specifically for the longest sequence. Shorter sequences (up to 160 AA) work perfectly.

## What We've Tried

### 1. FP32 RoPE Rotation (commit 7deb1d3)
**Hypothesis:** Precision loss from BF16 sin/cos accumulating over 36 layers.
**Fix:** Keep sin/cos in FP32 during rotation, cast only after.
**Result:** No improvement - values unchanged.

### 2. Same Precision for Both Paths (commit 8c48d50)
**Hypothesis:** Unpacked runs in FP32, packed auto-converts to BF16.
**Fix:** Convert model to BF16 at start of validation.
**Result:** No improvement - values unchanged.

## Root Cause Hypotheses (Remaining)

### Hypothesis A: RoPE Implementation Difference
Our manual RoPE in `_apply_rotary_embeddings` may differ from ESM-2's native `rot_emb.forward()`.

**To test:**
```python
# Compare our RoPE output to ESM-2's native RoPE for same input
# Check if rotate_half formula matches
# Check if frequency computation matches
```

### Hypothesis B: FlashAttention vs Standard Attention
FlashAttention varlen may have numerical differences from standard attention for longer sequences.

**To test:**
```python
# Run large sequence ALONE through both paths
# If similarity is >0.999, issue is with packing multiple sequences
# If similarity is still 0.991, issue is in our attention/RoPE
```

### Hypothesis C: Sequence Position in Packed Batch
The "large" sequence is 4th in the packed batch. Something about later positions may cause issues.

**To test:**
```python
# Reorder sequences so "large" is first
# Check if similarity improves
```

### Hypothesis D: Attention Scaling or Softmax Differences
Different numerical paths through attention computation.

## Key Files

| File | Purpose |
|------|---------|
| `virnucpro/models/esm2_flash.py` | ESM-2 wrapper with forward_packed |
| `virnucpro/models/packed_attention.py` | Position ID generation, FlashAttention wrapper |
| `virnucpro/data/packing.py` | validate_packed_equivalence function |
| `tests/integration/test_packed_equivalence.py` | Failing test |

## Debug Script to Run on GPU Server

```python
"""Debug large sequence similarity issue."""
import torch
import torch.nn.functional as F

from virnucpro.models.esm2_flash import load_esm2_model
from virnucpro.data import VarlenCollator

# Load model
model, batch_converter = load_esm2_model(
    model_name="esm2_t36_3B_UR50D",
    device="cuda:0"
)
model.eval()

# Convert to BF16 first to ensure same precision
model.model = model.model.to(dtype=torch.bfloat16)

num_layers = len(model.model.layers)

# Test ONLY the large sequence
large_seq = ("large", "MKTAYIAK" * 50)  # 400 AA

with torch.no_grad():
    # === UNPACKED FORWARD ===
    labels, strs, tokens = batch_converter([large_seq])
    tokens = tokens.to("cuda:0")
    unpacked_output = model(tokens, repr_layers=[num_layers])
    unpacked_emb = unpacked_output['representations'][num_layers][0, 1:401].mean(dim=0).float().cpu()

    # === PACKED FORWARD (single sequence) ===
    collator = VarlenCollator(batch_converter, max_tokens_per_batch=16384, enable_packing=False)
    batch = collator([{'id': 'large', 'sequence': 'MKTAYIAK' * 50}])

    packed_output = model.forward_packed(
        input_ids=batch['input_ids'].to("cuda:0"),
        cu_seqlens=batch['cu_seqlens'].to("cuda:0"),
        max_seqlen=batch['max_seqlen'],
        repr_layers=[num_layers],
    )

    # Extract packed embedding
    cu_seqlens = batch['cu_seqlens']
    packed_repr = packed_output['representations'][num_layers]
    start = cu_seqlens[0].item()
    end = cu_seqlens[1].item()
    packed_emb = packed_repr[start + 1:end - 1].mean(dim=0).float().cpu()

    # === COMPARE ===
    cos_sim = F.cosine_similarity(unpacked_emb.unsqueeze(0), packed_emb.unsqueeze(0)).item()

    print(f"Large sequence ALONE (400 AA):")
    print(f"  Cosine similarity: {cos_sim:.6f}")
    print(f"  Pass (>0.999): {cos_sim > 0.999}")

    if cos_sim < 0.999:
        print("\n  Issue is in forward_packed implementation (RoPE or attention)")
    else:
        print("\n  Issue is with mixing multiple sequences in packed batch")
```

## Debug Script 2: Compare RoPE Outputs

```python
"""Compare our RoPE to ESM-2's native RoPE."""
import torch
from virnucpro.models.esm2_flash import load_esm2_model

model, batch_converter = load_esm2_model(model_name="esm2_t36_3B_UR50D", device="cuda:0")
model.model = model.model.to(dtype=torch.bfloat16)

# Get first layer's RoPE module
layer = model.model.layers[0]
rot_emb = layer.self_attn.rot_emb

# Create test Q/K tensors
batch_size = 1
seq_len = 400
num_heads = 40  # ESM-2 3B
head_dim = 64   # 2560 / 40

q = torch.randn(seq_len, num_heads, head_dim, device="cuda:0", dtype=torch.bfloat16)
k = torch.randn(seq_len, num_heads, head_dim, device="cuda:0", dtype=torch.bfloat16)

# Our RoPE application
from virnucpro.models.packed_attention import create_position_ids_packed
cu_seqlens = torch.tensor([0, seq_len], dtype=torch.int32, device="cuda:0")
position_ids = create_position_ids_packed(cu_seqlens)

q_ours, k_ours = model._apply_rotary_embeddings(q.clone(), k.clone(), position_ids, rot_emb)

# ESM-2's native RoPE (expects [seq_len, batch, num_heads, head_dim] or similar)
# Need to check exact input format for rot_emb.forward()
print(f"rot_emb forward signature: {rot_emb.forward.__code__.co_varnames[:5]}")

# Try calling native RoPE
q_native, k_native = rot_emb(q.clone().transpose(0, 1).unsqueeze(0), k.clone().transpose(0, 1).unsqueeze(0))

# Compare
q_sim = torch.nn.functional.cosine_similarity(q_ours.flatten(), q_native.flatten(), dim=0)
k_sim = torch.nn.functional.cosine_similarity(k_ours.flatten(), k_native.flatten(), dim=0)

print(f"Q similarity: {q_sim.item():.6f}")
print(f"K similarity: {k_sim.item():.6f}")
```

## Debug Script 3: Reorder Sequences

```python
"""Test if sequence order in packed batch affects similarity."""
import torch
from virnucpro.data.packing import validate_packed_equivalence
from virnucpro.models.esm2_flash import load_esm2_model

model, batch_converter = load_esm2_model(model_name="esm2_t36_3B_UR50D", device="cuda:0")

# Original order (large is last)
sequences_orig = [
    ("tiny", "MKT"),
    ("small", "MKTAYIAK"),
    ("medium", "MKTAYIAK" * 20),
    ("large", "MKTAYIAK" * 50),
]

# Reversed order (large is first)
sequences_rev = [
    ("large", "MKTAYIAK" * 50),
    ("medium", "MKTAYIAK" * 20),
    ("small", "MKTAYIAK"),
    ("tiny", "MKT"),
]

for name, seqs in [("original", sequences_orig), ("reversed", sequences_rev)]:
    passed, details = validate_packed_equivalence(
        model, batch_converter, seqs, torch.device("cuda:0")
    )
    print(f"\n{name.upper()} order:")
    for seq_id, sim in details['per_sequence'].items():
        status = "PASS" if sim > 0.995 else "FAIL"
        print(f"  {seq_id}: {sim:.6f} [{status}]")
```

## Next Steps

1. Run Debug Script 1 to isolate whether issue is forward_packed or packing
2. If forward_packed issue: Run Debug Script 2 to compare RoPE
3. If packing issue: Run Debug Script 3 to check sequence order effects

## ESM-2 RoPE Details (from research)

- ESM-2 uses **partial rotary embeddings**: only first `rotary_dim` dimensions get rotation
- `rotary_dim` for ESM-2 3B appears to be 7 (out of head_dim=64)
- Uses `inv_freq` buffer to compute sin/cos on-the-fly
- rotate_half formula: `(-x2, x1)` split at midpoint

## Key Code Locations

### Our RoPE Implementation
`virnucpro/models/esm2_flash.py:329-407` - `_apply_rotary_embeddings`

### Position ID Creation
`virnucpro/models/packed_attention.py:44-110` - `create_position_ids_packed`

### Layer Forward (packed)
`virnucpro/models/esm2_flash.py:248-327` - `_layer_forward_packed`

### Validation Function
`virnucpro/data/packing.py:356-512` - `validate_packed_equivalence`
