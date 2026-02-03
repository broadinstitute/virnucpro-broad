---
phase: 06-sequence-packing-integration
plan: 02
subsystem: models
completed: 2026-02-03
duration: 2.6 min

tags:
  - flashattention
  - sequence-packing
  - position-ids
  - varlen-attention

requires:
  - "05-05: Async DataLoader foundation with VarlenCollator producing cu_seqlens"

provides:
  - "Position ID generator that resets at sequence boundaries"
  - "FlashAttention varlen wrapper with validation"
  - "FLASH_ATTN_AVAILABLE feature detection constant"

affects:
  - "06-03: Forward pass will use these utilities for packed sequences"
  - "06-04: Validation will test position ID correctness"

tech-stack:
  added:
    - "flash-attn >= 2.6.0 (optional dependency with graceful fallback)"
  patterns:
    - "Position ID reset at cu_seqlens boundaries for packed format"
    - "Input validation before FlashAttention kernel calls"
    - "Version checking with helpful upgrade messages"

key-files:
  created:
    - path: virnucpro/models/packed_attention.py
      purpose: "Position ID generation and FlashAttention varlen utilities"
      exports: ["create_position_ids_packed", "flash_attn_varlen_wrapper", "FLASH_ATTN_AVAILABLE"]
  modified:
    - path: virnucpro/models/__init__.py
      change: "Added packed_attention exports"

decisions:
  - id: PACK-02-1
    choice: "Reset position IDs to 0 at each cu_seqlens boundary"
    rationale: "Position IDs must be relative to sequence start, not batch start, for correct positional embeddings in packed format"
    alternatives: ["Sequential position IDs across batch (incorrect)", "No position IDs (missing critical info)"]

  - id: PACK-02-2
    choice: "Validate cu_seqlens and dtype before FlashAttention call"
    rationale: "Early validation provides clear error messages; FlashAttention dtype errors are cryptic"
    alternatives: ["Let FlashAttention validate (poor error messages)", "Skip validation (crashes)"]

  - id: PACK-02-3
    choice: "FLASH_ATTN_AVAILABLE as module-level constant"
    rationale: "Allows runtime feature detection; consumers can check availability before attempting to use varlen"
    alternatives: ["Try/except at usage site (scattered error handling)", "Require flash-attn (limits compatibility)"]

  - id: PACK-02-4
    choice: "Version check for flash-attn >= 2.6.0 with warning"
    rationale: "2.6.0 has critical bug fixes for varlen; warn users but don't block (Gap 11)"
    alternatives: ["Hard requirement (breaks older installs)", "No check (silent bugs)"]
---

# Phase 6 Plan 2: Position ID Generator and FlashAttention Varlen Wrapper Summary

**One-liner:** Created position ID generator with boundary reset validation and FlashAttention varlen wrapper with comprehensive dtype/format checks

## What Was Built

Created `virnucpro/models/packed_attention.py` module providing:

1. **Position ID Generator** (`create_position_ids_packed`)
   - Resets position IDs to 0 at each cu_seqlens boundary
   - Validates cu_seqlens format: starts with 0, monotonically increasing
   - Validates output: position_ids[cu_seqlens[i]] == 0 for all boundaries
   - Example: cu_seqlens [0,3,7,10] → position_ids [0,1,2,0,1,2,3,0,1,2]

2. **FlashAttention Varlen Wrapper** (`flash_attn_varlen_wrapper`)
   - Validates FP16/BF16 dtype requirement (FlashAttention constraint)
   - Validates cu_seqlens int32 dtype requirement
   - Validates cu_seqlens format (starts with 0)
   - Graceful ImportError handling with install instructions
   - Version check for flash-attn >= 2.6.0 with upgrade warning

3. **Feature Detection** (`FLASH_ATTN_AVAILABLE`)
   - Module-level constant for runtime availability check
   - Allows consumers to conditionally use packed attention
   - Enables graceful fallback to padded batches

## Architecture

**Module Structure:**
```
virnucpro/models/packed_attention.py
├── FLASH_ATTN_AVAILABLE (constant)
├── create_position_ids_packed(cu_seqlens) → position_ids
└── flash_attn_varlen_wrapper(q, k, v, cu_seqlens, max_seqlen) → output
```

**Position ID Generation Pattern:**
```python
# For packed sequences [seq1, seq2, seq3]:
cu_seqlens = [0, 3, 7, 10]  # Cumulative boundaries

# Position IDs reset at each boundary:
position_ids = [0, 1, 2,    # seq1
                0, 1, 2, 3,  # seq2
                0, 1, 2]     # seq3

# NOT sequential: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (WRONG)
```

**Validation Flow:**
```
flash_attn_varlen_wrapper()
  1. Check FLASH_ATTN_AVAILABLE
  2. Check CUDA available
  3. Validate cu_seqlens dtype (int32)
  4. Validate q/k/v dtype (FP16/BF16)
  5. Validate cu_seqlens format (starts with 0)
  6. Call flash_attn_varlen_func()
```

## Key Technical Details

**Position ID Correctness:**
- Validates cu_seqlens[0] == 0 (format requirement)
- Validates monotonically increasing (detects construction errors)
- Validates position_ids[cu_seqlens[i]] == 0 for all i (detects reset failures)
- Uses torch.arange() per sequence for efficiency

**FlashAttention Requirements:**
- Inputs must be FP16 or BF16 (validated with clear error message)
- cu_seqlens must be int32 (validated with conversion hint)
- cu_seqlens format: [0, len1, len1+len2, ...] with N+1 elements
- Both cu_seqlens_q and cu_seqlens_k use same tensor (self-attention)

**Version Compatibility (Gap 11):**
- Checks flash-attn version >= 2.6.0
- Logs warning if older version detected
- Provides install command in warning
- Doesn't block execution (allows testing with older versions)

**Error Handling:**
- ImportError: flash-attn not installed → helpful install message
- RuntimeError: CUDA not available → clear requirement message
- ValueError: dtype mismatch → conversion instructions
- AssertionError: cu_seqlens format → format explanation

## Deviations from Plan

None - plan executed exactly as written.

## Validation Results

**Position ID Generation:**
- ✅ Resets to 0 at each cu_seqlens boundary
- ✅ Validation: cu_seqlens[0] == 0 assertion present
- ✅ Validation: position_ids[cu_seqlens[i]] == 0 assertion present
- ✅ Example test case: [0,3,7,10] → [0,1,2,0,1,2,3,0,1,2]

**FlashAttention Wrapper:**
- ✅ FLASH_ATTN_AVAILABLE constant for feature detection
- ✅ Validates dtype (FP16/BF16) with clear error messages
- ✅ Validates cu_seqlens dtype (int32) with conversion hint
- ✅ Graceful ImportError handling with install instructions
- ✅ Version check for >= 2.6.0 with upgrade warning

**Module Exports:**
- ✅ All exports work from virnucpro.models (syntactically verified)
- ✅ __all__ list updated with new exports
- ✅ Existing ESM2 exports maintained

## Performance Impact

**Expected:**
- Position ID generation: O(N) where N = total_tokens (minimal overhead)
- Validation overhead: <0.1ms per batch (CPU operations on small tensors)
- FlashAttention varlen: 2-3x faster than padded attention for variable-length sequences

**Memory:**
- Position IDs: int64 tensor [total_tokens] ~8 bytes per token
- No additional memory overhead (validations use existing tensors)

## Integration Points

**Downstream Consumers:**
- `virnucpro/models/esm2_flash.py`: Will use position IDs in forward_packed()
- `virnucpro/pipeline/async_inference.py`: Will call wrapper in packed inference path
- `tests/integration/test_packed_attention.py`: Validation tests for correctness

**Upstream Dependencies:**
- `flash_attn >= 2.6.0`: Optional dependency (graceful fallback if missing)
- `packaging`: For version comparison
- PyTorch: For tensor operations and dtype validation

## Testing Strategy

**Unit Tests (Future 06-04):**
- Position ID reset at boundaries for various cu_seqlens patterns
- Position ID validation catches format errors
- Wrapper validates dtype mismatches
- Wrapper validates cu_seqlens format errors
- FLASH_ATTN_AVAILABLE correctly reflects availability

**Integration Tests (Future 06-04):**
- Packed forward pass uses correct position IDs
- FlashAttention varlen produces correct outputs
- Fallback works when flash-attn unavailable

## Next Phase Readiness

**Blockers:** None

**Concerns:**
- FlashAttention version compatibility: Warning added, but runtime issues possible with <2.6.0
- Position ID off-by-one errors: Comprehensive validation should catch, but testing critical

**Required for Phase 6 Plan 3:**
- ✅ Position ID generator with validation
- ✅ FlashAttention varlen wrapper
- ✅ Feature detection constant
- ⏳ Integration with ESM-2 forward pass (next plan)

## Lessons Learned

**What Worked:**
- Comprehensive validation with clear error messages prevents debugging pain
- Module-level feature detection enables conditional usage
- Version checking with warnings balances compatibility and safety

**What Could Be Better:**
- Testing: Need actual GPU environment to test FlashAttention integration
- Documentation: Examples in docstrings help but integration guide needed

**Reusable Patterns:**
- Early validation with helpful error messages (dtype, format)
- Feature detection constants for optional dependencies
- Version checking with warnings (not hard failures)

## Code Statistics

**Files Created:** 1
- `virnucpro/models/packed_attention.py`: 200 lines

**Files Modified:** 1
- `virnucpro/models/__init__.py`: +9 lines

**Total Impact:** 209 lines added

**Commits:** 2
- `7c1597f`: Create packed attention utilities module
- `3939611`: Export packed attention utilities from models module

**Duration:** 2.6 minutes
