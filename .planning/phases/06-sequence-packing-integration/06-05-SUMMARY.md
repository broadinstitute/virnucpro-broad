---
phase: 06-sequence-packing-integration
plan: 05
subsystem: validation
tags: [testing, equivalence, flashattention, quality-assurance]

requires:
  - phase: 06
    plan: 04
    provides: packed inference path
  - phase: 06
    plan: 03
    provides: forward_packed method
  - phase: 06
    plan: 02
    provides: position ID generation

provides:
  - validate_packed_equivalence function for correctness validation
  - integration tests for packed vs unpacked equivalence
  - cross-contamination detection tests
  - position ID reset verification

affects:
  - phase: 06
    plan: 06
    why: validation tests ensure correctness before production use

tech-stack:
  added: []
  patterns:
    - cosine similarity for embedding comparison
    - two-tier threshold validation (strict 0.999, lenient 0.995)
    - per-sequence metrics collection

key-files:
  created:
    - tests/integration/test_packed_equivalence.py
  modified:
    - virnucpro/data/packing.py
    - virnucpro/data/__init__.py

decisions:
  - id: VAL-01
    what: Cosine similarity threshold 0.999 with 1% lenient allowance
    why: FP16 precision can cause minor numerical differences, strict 0.999 for 99% sequences with 0.995 floor for 1%
    impact: Catches real errors while allowing expected FP16 precision variations

metrics:
  duration: 2.9 min
  completed: 2026-02-03
---

# Phase 6 Plan 5: Packed Equivalence Validation Summary

**One-liner:** Cosine similarity validation (>0.999) for packed vs unpacked equivalence with comprehensive integration tests

## What Was Built

Implemented validation infrastructure to verify packed sequences produce identical embeddings to unpacked sequences, proving FlashAttention varlen integration is correct.

**Key Components:**

1. **validate_packed_equivalence function** (virnucpro/data/packing.py)
   - Processes sequences individually (unpacked baseline)
   - Packs all sequences using VarlenCollator
   - Runs model.forward_packed
   - Extracts embeddings using cu_seqlens boundaries
   - Compares with F.cosine_similarity
   - Returns pass/fail with detailed per-sequence metrics

2. **Integration tests** (tests/integration/test_packed_equivalence.py)
   - TestPackedEquivalence: short, medium, mixed length sequences
   - TestCrossContamination: verify sequences remain distinct
   - TestPositionIDReset: verify position IDs reset at boundaries
   - TestEdgeCases: single sequence, very short, empty list

3. **Public API export**
   - Added validate_packed_equivalence to virnucpro.data exports
   - Available for production validation and debugging

## How It Works

```python
from virnucpro.data import validate_packed_equivalence

# Validate packed vs unpacked equivalence
sequences = [
    ("seq1", "MKTAYIAK"),
    ("seq2", "VLSPADKTNV"),
]

passed, details = validate_packed_equivalence(
    model, batch_converter, sequences, device
)

if not passed:
    print(f"Failed sequences: {details['failed_sequences']}")
    print(f"Min similarity: {details['min_similarity']}")
```

**Validation algorithm:**

1. For each sequence individually:
   - Tokenize and run standard forward pass
   - Extract mean-pooled embedding (skip BOS/EOS)
   - Store as unpacked baseline

2. Pack all sequences together:
   - Use VarlenCollator to create packed batch
   - Run forward_packed with cu_seqlens
   - Extract per-sequence embeddings using boundaries

3. Compare with cosine similarity:
   - Strict threshold: 0.999 (99% of sequences must pass)
   - Lenient threshold: 0.995 (1% allowed in this range)
   - Fail if any sequence < 0.995

4. Return detailed metrics:
   - Per-sequence similarity scores
   - Strict pass rate
   - Min/max similarity
   - Failed sequence IDs

## Decisions Made

**VAL-01: Two-tier threshold system**
- **Decision:** Strict 0.999 for 99% sequences, lenient 0.995 for 1%
- **Rationale:** FP16 precision can cause minor numerical differences (~0.998) that don't indicate actual errors
- **Impact:** Catches real implementation bugs while allowing expected precision variations
- **Alternative considered:** Single strict 1.0 threshold (too brittle, FP16 will fail)

## Test Coverage

Integration tests verify:

1. **Correctness:**
   - Short sequences (<50 aa) match with >0.999 similarity
   - Medium sequences (50-200 aa) match
   - Mixed length sequences in same batch match
   - Many sequences (>10) all match

2. **Cross-contamination prevention:**
   - Distinct sequences remain distinct (hydrophobic vs charged)
   - Repeated identical sequences produce identical embeddings

3. **Position ID reset:**
   - Position IDs reset to 0 at each cu_seqlens boundary
   - Position IDs continuous within each sequence
   - Model runs without errors on packed format

4. **Edge cases:**
   - Single sequence (no packing needed)
   - Very short sequences (3-5 aa)
   - Empty sequences list

**All tests require GPU** (skip if CUDA unavailable)

## Integration Points

**Consumes:**
- model.forward_packed (from 06-03)
- VarlenCollator (from 06-08, already implemented)
- create_position_ids_packed (from 06-02)

**Produces:**
- validate_packed_equivalence for production validation
- Integration test suite for CI/CD quality gates

**Used by:**
- Phase 06 Plan 06: Production rollout validation
- Debugging when packed inference produces unexpected results

## Files Changed

**Created:**
- `tests/integration/test_packed_equivalence.py` (268 lines)
  - 5 test classes, 12 test methods
  - Comprehensive coverage of equivalence, contamination, position IDs, edge cases

**Modified:**
- `virnucpro/data/packing.py` (+154 lines)
  - Added validate_packed_equivalence function
  - Import F.cosine_similarity for comparison
- `virnucpro/data/__init__.py` (+1 line)
  - Exported validate_packed_equivalence

## Deviations from Plan

None - plan executed exactly as written.

## Validation Results

**Verification criteria met:**

- ✅ validate_packed_equivalence function in packing.py
- ✅ Function returns (passed, details) tuple
- ✅ Strict threshold 0.999, lenient 0.995 with 1% allowance
- ✅ Integration tests cover short, medium, mixed length sequences
- ✅ Cross-contamination test verifies distinct sequences remain distinct
- ✅ All exports work from virnucpro.data

**Success criteria met:**

1. ✅ Validation function compares packed vs unpacked embeddings
2. ✅ Returns detailed metrics (per-sequence similarity, pass rate, min/max)
3. ✅ Integration tests verify equivalence on GPU
4. ✅ Threshold system allows small FP16 precision variations

## Performance Impact

**Testing overhead:**
- Validation requires 2x forward passes (unpacked + packed)
- Not intended for production hot path (debugging/CI only)
- Integration tests run on smaller 650M model for speed

**Benefits:**
- Gold standard test for FlashAttention varlen correctness
- Catches position ID bugs, cross-contamination, packing errors
- Per-sequence granularity helps isolate issues

## Next Phase Readiness

**Phase 06 Plan 06 (Production rollout) unblocked:**
- ✅ Validation function available for pre-deployment checks
- ✅ Integration tests can be added to CI pipeline
- ✅ Per-sequence metrics enable debugging production issues

**No blockers or concerns.**

## Knowledge for Future Phases

**When debugging packed inference:**
1. Run validate_packed_equivalence on failing sequences
2. Check details['per_sequence'] for which sequences fail
3. If min_similarity < 0.99, likely position ID or cu_seqlens bug
4. If specific sequences fail, check their length/content for edge cases

**Expected behavior:**
- Most sequences should have similarity >0.9995 (nearly perfect)
- Occasional 0.998-0.999 from FP16 precision is normal
- Anything <0.995 indicates implementation bug

**Integration test usage:**
```bash
# Run packed equivalence tests (requires GPU)
pytest tests/integration/test_packed_equivalence.py -v

# Skip if no GPU
pytest tests/integration/test_packed_equivalence.py -v --skip-gpu
```
