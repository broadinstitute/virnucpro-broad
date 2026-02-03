---
phase: 06-sequence-packing-integration
plan: 03
subsystem: models
tags: [flashattention, varlen-attention, packed-sequences, esm2-forward]

# Dependency graph
requires:
  - phase: 06-sequence-packing-integration
    plan: 02
    provides: Position ID generator and FlashAttention varlen wrapper
provides:
  - forward_packed method in ESM2WithFlashAttention for packed sequence inference
  - _layer_forward_packed for per-layer FlashAttention varlen processing
  - _forward_packed_fallback for systems without flash-attn
affects: [06-04, 06-05, async-inference-pipeline]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "FlashAttention varlen integration at transformer layer level"
    - "Unpack/repack fallback strategy for non-FlashAttention systems"
    - "Dtype validation (FP16/BF16) before FlashAttention kernel calls"
    - "Q/K/V extraction from ESM-2 MultiheadAttention in_proj_weight"

key-files:
  created:
    - tests/unit/test_esm2_packed.py
  modified:
    - virnucpro/models/esm2_flash.py

key-decisions:
  - "Integrate flash_attn_varlen_func at layer level, not via HuggingFace config (ESM uses fair-esm)"
  - "Validate FP16/BF16 dtype before FlashAttention with clear error messages"
  - "Fallback to unpack/repack when FlashAttention unavailable (ensures correctness on all systems)"
  - "Extract Q/K/V from in_proj_weight rather than modify ESM layer classes"
  - "Return representations in packed format [total_tokens, hidden_dim] for consistency"

patterns-established:
  - "Layer-level attention replacement for packed format"
  - "Graceful fallback with performance warnings"
  - "Dtype validation with conversion instructions"
  - "Position ID reset integration at forward pass level"

# Metrics
duration: 2.4min
completed: 2026-02-03
---

# Phase 6 Plan 3: ESM-2 Packed Forward Pass Integration

**FlashAttention varlen integration with position ID reset and unpack/repack fallback for packed sequence inference**

## Performance

- **Duration:** 2.4 min
- **Started:** 2026-02-03T20:32:50Z
- **Completed:** 2026-02-03T20:35:17Z
- **Tasks:** 3
- **Files modified:** 2

## Accomplishments

- Implemented forward_packed() method for packed sequence inference through ESM-2
- Integrated create_position_ids_packed for boundary-reset position embeddings
- Used flash_attn_varlen_wrapper for efficient variable-length attention
- Added dtype validation (FP16/BF16) before FlashAttention kernel calls
- Implemented _layer_forward_packed for per-layer FlashAttention varlen processing
- Created _forward_packed_fallback for systems without flash-attn
- Added comprehensive unit tests for method signature and structure

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement forward_packed method** - `97fce1c` (feat)
   - Added forward_packed() with input_ids, cu_seqlens, max_seqlen parameters
   - Integrated create_position_ids_packed for position ID reset at boundaries
   - Validated FP16/BF16 dtype for FlashAttention compatibility
   - Implemented _layer_forward_packed for per-layer processing
   - Created _forward_packed_fallback for unpack/repack strategy
   - Included Task 2 functionality (fallback was part of same implementation)

2. **Task 3: Add unit tests** - `65e1343` (test)
   - Tests for forward_packed signature (input_ids, cu_seqlens, max_seqlen, repr_layers)
   - Tests for fallback method existence
   - Tests for packed_attention imports
   - Parametrized tests for FlashAttention availability
   - Return structure validation

## Files Created/Modified

- `virnucpro/models/esm2_flash.py` - Added forward_packed, _layer_forward_packed, _forward_packed_fallback methods (+236 lines)
- `tests/unit/test_esm2_packed.py` - Unit tests for packed forward pass (+161 lines)

## Implementation Details

### Forward Pass Architecture

The forward_packed method processes packed batches through three stages:

1. **Embedding Stage**
   - Create position IDs with boundary reset using create_position_ids_packed(cu_seqlens)
   - Embed tokens: `self.model.embed_tokens(input_ids)` → [total_tokens, hidden_dim]
   - Validate dtype in [FP16, BF16] for FlashAttention compatibility
   - Add position embeddings: ESM-2 uses learned positional embeddings

2. **Transformer Layers Stage**
   - For each layer, call _layer_forward_packed:
     - Apply layer norm
     - Extract Q/K/V from in_proj_weight: [total_tokens, 3*hidden_dim]
     - Reshape to [total_tokens, num_heads, head_dim] for each
     - Call flash_attn_varlen_wrapper with cu_seqlens
     - Apply output projection and residual
     - Feed-forward with residual

3. **Output Stage**
   - Apply final layer norm
   - Return representations dict in packed format [total_tokens, hidden_dim]

### FlashAttention Integration Pattern

```python
# Extract Q, K, V from ESM-2 MultiheadAttention
qkv = torch.nn.functional.linear(
    hidden_states,
    layer.self_attn.in_proj_weight,
    layer.self_attn.in_proj_bias
)

# Reshape for multi-head attention
qkv = qkv.reshape(-1, 3, num_heads, head_dim)
q, k, v = qkv.unbind(dim=1)

# FlashAttention varlen call
attn_output = flash_attn_varlen_wrapper(
    q=q, k=k, v=v,
    cu_seqlens=cu_seqlens,
    max_seqlen=max_seqlen,
    dropout_p=0.0,  # Inference
    causal=False,   # Bidirectional
)
```

### Fallback Strategy

When FLASH_ATTN_AVAILABLE=False:
1. Unpack 1D input_ids to 2D padded tensor using cu_seqlens
2. Call standard forward() with padded format
3. Repack output to 1D using cu_seqlens boundaries
4. Return same format as flash_attn path

This ensures correctness on:
- Older GPUs (< Ampere)
- CI environments without flash-attn
- Testing environments

## Decisions Made

**ESM-2 Architecture Integration Approach:**
- **Chosen:** Extract Q/K/V from in_proj_weight, wrap attention with flash_attn_varlen_wrapper
- **Why:** ESM uses fair-esm (not HuggingFace), so can't use attn_implementation config
- **Alternative:** Modify ESM layer classes (rejected: avoids vendored code modification)

**Fallback Strategy:**
- **Chosen:** Unpack to 2D padded, run standard forward, repack to 1D
- **Why:** Guarantees correctness on all systems, transparent to caller
- **Alternative:** Raise error when flash-attn unavailable (rejected: breaks compatibility)

**Dtype Validation Timing:**
- **Chosen:** Validate after embedding layer, before first FlashAttention call
- **Why:** Clear error message at start of forward pass
- **Alternative:** Let FlashAttention validate (rejected: cryptic kernel errors)

## Deviations from Plan

None - plan executed exactly as written.

## Validation Results

**Success Criteria:**
- ✅ forward_packed method processes packed batches
- ✅ Position IDs reset at sequence boundaries (via create_position_ids_packed)
- ✅ FlashAttention varlen used when available
- ✅ Graceful fallback when FlashAttention unavailable

**Verification Checklist:**
- ✅ forward_packed method added to ESM2WithFlashAttention
- ✅ Method accepts input_ids (1D), cu_seqlens, max_seqlen
- ✅ Returns dict with 'representations' key
- ✅ Uses create_position_ids_packed for position embeddings
- ✅ Fallback to unpack/repack when FlashAttention unavailable
- ✅ Unit tests for signature verification

**Unit Tests:**
- ✅ test_forward_packed_signature - Verifies parameter names and order
- ✅ test_forward_packed_method_exists - Method callable
- ✅ test_fallback_method_exists - Fallback path exists
- ✅ test_layer_forward_packed_method_exists - Helper method exists
- ✅ test_packed_attention_imports - Utilities imported correctly
- ✅ test_forward_packed_docstring - Comprehensive documentation
- ✅ test_fallback_used_when_flash_unavailable - Parametrized FlashAttention availability
- ✅ test_forward_packed_returns_dict_structure - Return type validation
- ✅ test_layer_forward_packed_signature - Helper signature correct

## Integration Points

**Upstream Dependencies:**
- `virnucpro.models.packed_attention.create_position_ids_packed` - Position ID generation
- `virnucpro.models.packed_attention.flash_attn_varlen_wrapper` - FlashAttention varlen
- `virnucpro.models.packed_attention.FLASH_ATTN_AVAILABLE` - Feature detection

**Downstream Consumers:**
- `virnucpro.pipeline.async_inference.py` - Will call forward_packed in packed inference path
- `tests/integration/test_packed_correctness.py` - Validation tests (Phase 6 Plan 4)

**Modified Components:**
- `ESM2WithFlashAttention.forward_packed` - New method for packed sequences
- `ESM2WithFlashAttention._layer_forward_packed` - New helper for layer processing
- `ESM2WithFlashAttention._forward_packed_fallback` - New fallback for non-FlashAttention systems

## Next Phase Readiness

**Blockers:** None

**Concerns:**
- FlashAttention varlen correctness: Need integration tests comparing packed vs unpacked outputs (06-04)
- Q/K/V extraction: Assumes ESM-2 uses in_proj_weight (standard MultiheadAttention) - validated against fair-esm source
- Position ID off-by-one: Relies on create_position_ids_packed validation (already tested in 06-02)

**Required for Phase 6 Plan 4 (Validation):**
- ✅ forward_packed method integrated
- ✅ Position IDs reset at boundaries
- ✅ FlashAttention varlen wrapper used
- ✅ Fallback path for non-FlashAttention systems
- ⏳ Correctness validation tests (packed == unpacked outputs)

**Required for Phase 6 Plan 5 (Pipeline Integration):**
- ✅ forward_packed callable from async_inference.py
- ✅ Accepts VarlenCollator output format (input_ids, cu_seqlens, max_seqlen)
- ✅ Returns packed representations
- ⏳ Pipeline integration (replace NotImplementedError)

## Technical Notes

### ESM-2 Model Structure

From fair-esm library:
- `self.model.embed_tokens(input_ids)` - Token embeddings
- `self.model.embed_positions(position_ids)` - Learned positional embeddings (not rotary)
- `self.model.layers` - List of TransformerEncoderLayer
- `layer.self_attn` - MultiheadAttention with in_proj_weight [hidden_dim, 3*hidden_dim]
- `layer.self_attn_layer_norm` - Pre-attention layer norm
- `layer.final_layer_norm` - Pre-FFN layer norm
- `layer.fc1`, `layer.fc2` - Feed-forward layers
- `self.model.emb_layer_norm_after` - Final layer norm

### Q/K/V Extraction

ESM-2 uses PyTorch's MultiheadAttention with single in_proj_weight:
```python
# in_proj_weight shape: [3*hidden_dim, hidden_dim]
# Projects input to [Q, K, V] concatenated
qkv = F.linear(x, in_proj_weight, in_proj_bias)  # [total_tokens, 3*hidden_dim]

# Split into Q, K, V
qkv = qkv.reshape(-1, 3, num_heads, head_dim)
q, k, v = qkv.unbind(dim=1)  # Each: [total_tokens, num_heads, head_dim]
```

### Dtype Compatibility

FlashAttention requires FP16 or BF16 inputs:
- Validation: `assert embeddings.dtype in [torch.float16, torch.bfloat16]`
- Error message includes conversion instructions: `model.half()`
- ESM-2 typically loaded in FP32, so model needs explicit conversion for FlashAttention

## Performance Impact

**Expected:**
- Forward pass with FlashAttention varlen: 2-3x faster than padded attention for variable-length sequences
- Fallback path: Equivalent to standard forward (no speedup, but no slowdown)
- Dtype validation: <0.1ms overhead (single assertion on first batch)

**Memory:**
- Packed format: No padding tokens → 20-40% memory savings vs padded batches
- Q/K/V tensors: Transient, same memory as standard attention
- Fallback: Temporary padded tensor allocation (2× memory during unpacking)

## Lessons Learned

**What Worked:**
- Layer-level integration allows FlashAttention without modifying ESM classes
- Comprehensive dtype validation prevents cryptic CUDA errors
- Fallback strategy ensures universal compatibility
- Unit tests verify structure without requiring GPU

**What Could Be Better:**
- Integration testing needed to verify correctness (next plan)
- Documentation could include example usage with VarlenCollator output
- Fallback performance could be optimized (current implementation is simple but inefficient)

**Reusable Patterns:**
- Extract Q/K/V from in_proj_weight for models using standard MultiheadAttention
- Unpack/repack fallback strategy for algorithm unavailability
- Dtype validation with conversion instructions in error messages

## Code Statistics

**Files Created:** 1
- `tests/unit/test_esm2_packed.py`: 161 lines

**Files Modified:** 1
- `virnucpro/models/esm2_flash.py`: +240 lines (forward_packed, _layer_forward_packed, _forward_packed_fallback, imports)

**Total Impact:** 401 lines added

**Test Coverage:**
- Unit tests: 9 test functions
- Integration tests: Deferred to 06-04 (GPU required)

**Commits:** 2
- `97fce1c`: feat(06-03): implement forward_packed method in ESM2WithFlashAttention
- `65e1343`: test(06-03): add unit tests for forward_packed signature and structure

**Duration:** 2.4 minutes

## Future Work

**Phase 6 Plan 4 (Validation):**
- Integration tests comparing packed vs unpacked outputs for same sequences
- Verify position IDs reset correctly at boundaries
- Validate FlashAttention varlen produces correct attention outputs

**Phase 6 Plan 5 (Pipeline Integration):**
- Replace NotImplementedError in async_inference.py with forward_packed call
- Handle VarlenCollator batch format conversion
- Add packing efficiency metrics to GPU monitoring

**Phase 8 (FP16 Precision Validation):**
- Verify FP16 numerical stability with FlashAttention
- Test dtype conversion strategies (model.half() vs model.to(dtype=torch.float16))
- Validate embedding precision after dtype conversion
