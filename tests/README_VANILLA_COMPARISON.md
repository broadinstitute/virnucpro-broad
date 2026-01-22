# Vanilla vs Refactored Comparison Testing

This directory contains tests that verify the refactored VirNucPro implementation produces mathematically equivalent results to the vanilla `prediction.py` implementation.

## Overview

The comparison testing validates:

1. **Embeddings** (.pt files):
   - DNABERT-S nucleotide features (768 dimensions)
   - ESM-2 protein features (2560 dimensions)
   - Merged features (3328 dimensions)

2. **Predictions**:
   - Raw predictions (`prediction_results.txt`)
   - Consensus results (`prediction_results_highestscore.csv`)

## Quick Start

### 1. Generate Vanilla Reference Outputs

First, generate the reference "golden" outputs from the vanilla implementation:

```bash
# From project root
./tests/generate_vanilla_reference.sh
```

This script:
- Runs vanilla `prediction.py` on `tests/data/test_with_orfs.fa`
- Saves embeddings and predictions to `tests/data/reference_vanilla_output/`
- These become the reference for all future comparisons

**Prerequisites**:
- `prediction.py` must exist in project root
- `500_model.pth` checkpoint must be available
- `tests/data/test_with_orfs.fa` must exist (synthetic sequences with valid ORFs)

### 2. Run Comparison Tests

Once reference outputs are generated, run the comparison tests:

```bash
# Run just vanilla comparison tests
pytest tests/test_vanilla_comparison.py -v

# Run with detailed output
pytest tests/test_vanilla_comparison.py -v -s

# Run specific test
pytest tests/test_vanilla_comparison.py::TestVanillaEquivalence::test_embeddings_equivalence -v
```

## Test Files

### `test_vanilla_comparison.py`

Main test suite with three test methods:

1. **`test_embeddings_equivalence`**: Compares all `.pt` embedding files
   - Verifies sequence IDs match
   - Checks embeddings within tolerance (rtol=1e-4, atol=1e-6)
   - Provides detailed mismatch reporting

2. **`test_prediction_output_equivalence`**: Compares prediction files
   - Validates `prediction_results.txt` (raw predictions)
   - Validates `prediction_results_highestscore.csv` (consensus)
   - Checks scores within floating-point tolerance

3. **`test_full_pipeline_equivalence`**: End-to-end test
   - Runs both embedding and prediction comparisons
   - Comprehensive validation of entire pipeline

### `compare_vanilla_embeddings.py`

Utility script for standalone embedding comparison:

```bash
python tests/compare_vanilla_embeddings.py \
    tests/data/reference_vanilla_output \
    tests/data/test_with_orfs_output/test_with_orfs_nucleotide
```

Can be used independently for debugging or manual verification.

### `generate_vanilla_reference.sh`

Helper script to automate vanilla reference generation:
- Runs vanilla pipeline
- Saves outputs to reference directory
- Validates outputs were created

## Test Data

### `test_with_orfs.fa`

Synthetic sequences designed with valid open reading frames (ORFs):
- 2 sequences, each ~175bp
- Each produces 2 valid ORFs (~57-58 amino acids)
- Ensures vanilla implementation generates features (doesn't filter all sequences)

**Why synthetic?** The original `test_sequences.fa` (100 random sequences) produced no valid ORFs, so vanilla filtered out all sequences during `identify_seq()`.

## Success Criteria

### ✅ Pass

- All embeddings match within tolerance:
  - Relative tolerance: 1e-4 (0.01%)
  - Absolute tolerance: 1e-6
- Predictions identical (sequence IDs, labels, scores within tolerance)

### ⚠️ Acceptable Variance

- Floating-point precision differences up to 1e-6
- Minor GPU/CPU numerical variations
- Differences due to batching order (should not occur with deterministic seeds)

### ❌ Fail

- Embeddings differ by >1e-4 relative tolerance
- Prediction labels differ
- Sequence IDs mismatch

## Common Issues

### Reference Directory Empty

**Symptom**: Test skipped with message "Vanilla reference outputs not found"

**Solution**: Run `./tests/generate_vanilla_reference.sh` first

### No .pt Files Generated

**Symptom**: Vanilla runs but no embedding files created

**Cause**: Input sequences don't produce valid ORFs

**Solution**:
- Use `test_with_orfs.fa` (synthetic sequences with known ORFs)
- Or use real viral genomes (see Alternative Test Data below)

### Vanilla Crashes

**Known Issues**:
- `UnboundLocalError` on line 173 if merge loop doesn't execute
- No error handling for empty inputs, GPU OOM, etc.

**Solution**: Document the error and rely on refactored test suite (30 passing tests)

## Alternative Test Data

If synthetic sequences are insufficient, use real viral genomes:

```bash
# Lambda phage (48.5kb - many ORFs)
wget -O tests/data/lambda_phage.fa \
  "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nuccore&id=NC_001416&rettype=fasta"

# SARS-CoV-2 reference (29.9kb)
wget -O tests/data/sars_cov2.fa \
  "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nuccore&id=NC_045512&rettype=fasta"

# Update generate_vanilla_reference.sh to use new input
```

## Regression Testing

Once vanilla comparison passes, the refactored outputs become the "golden reference" for future regression testing:

```bash
# Save golden reference (after vanilla comparison passes)
mkdir -p tests/data/golden_reference
cp -r tests/data/test_with_orfs_output/* tests/data/golden_reference/

# Future regression test (after code changes)
virnucpro predict tests/data/test_with_orfs.fa \
  --output-dir /tmp/test_output

python tests/compare_vanilla_embeddings.py \
  tests/data/golden_reference/test_with_orfs_nucleotide \
  /tmp/test_output/test_with_orfs_nucleotide
```

Add to CI pipeline:
```yaml
# .github/workflows/test.yml
- name: Regression test
  run: |
    pytest tests/test_vanilla_comparison.py -v
```

## Interpreting Results

### Perfect Match
```
✅ All embeddings match! Refactored implementation is mathematically equivalent.
🎉 Full pipeline equivalence verified!
```

**Meaning**: Refactored implementation produces identical results to vanilla (within tolerance)

### Partial Match
```
⚠️  Some embeddings differ. Review mismatches above.
Match rate: 95.00%
```

**Action**:
1. Check mismatch details for patterns
2. Verify tolerance settings are appropriate
3. Investigate large differences (>1e-3)

### Complete Mismatch
```
❌ Mismatches: 100/100 sequences
```

**Action**:
1. Verify input files are identical
2. Check model checkpoint is the same
3. Review refactored code for logic errors
4. Compare intermediate outputs step-by-step

## Expected Differences (Not Failures)

The refactored implementation **improves** upon vanilla:

1. **Attention masking**: Proper attention masks for mean pooling
2. **Error handling**: Validates inputs, handles GPU OOM gracefully
3. **Checkpoint resume**: Skips completed files on restart
4. **Batching**: Processes multiple sequences per GPU call
5. **Parallelization**: Multi-GPU support

These improvements should **not change embeddings** for fixed-length sequences (post-chunking all are 500bp).

## Tolerance Tuning

Default tolerances are conservative for production validation:
- `rtol=1e-4` (0.01% relative)
- `atol=1e-6` (absolute)

To adjust (e.g., for debugging):

```python
# In test_vanilla_comparison.py
tolerance_rtol = 1e-3  # More lenient
tolerance_atol = 1e-5

# Or in compare_vanilla_embeddings.py command line
python tests/compare_vanilla_embeddings.py \
    --rtol 1e-3 --atol 1e-5 \
    vanilla_dir refactored_dir
```

## Debugging Workflow

If tests fail:

1. **Run comparison script directly** for detailed output:
   ```bash
   python tests/compare_vanilla_embeddings.py \
       tests/data/reference_vanilla_output \
       tests/data/test_with_orfs_output/test_with_orfs_nucleotide
   ```

2. **Inspect specific embedding files** in Python:
   ```python
   import torch
   vanilla = torch.load('tests/data/reference_vanilla_output/output_0_DNABERT_S.pt')
   refactored = torch.load('tests/data/test_with_orfs_output/.../output_0_DNABERT_S.pt')

   # Compare specific sequence
   v_emb = vanilla['data'][0]['mean_representation']
   r_emb = refactored['data'][0]['mean_representation']
   print(torch.tensor(v_emb) - torch.tensor(r_emb))
   ```

3. **Check intermediate files** are being generated:
   ```bash
   ls -lh tests/data/test_with_orfs_nucleotide/
   ls -lh tests/data/test_with_orfs_output/test_with_orfs_nucleotide/
   ```

4. **Run with verbose logging**:
   ```bash
   pytest tests/test_vanilla_comparison.py -v -s --log-cli-level=DEBUG
   ```

## Maintenance

### Updating Test Data

When adding new test sequences:

1. Create new FASTA in `tests/data/`
2. Generate vanilla reference: `./tests/generate_vanilla_reference.sh`
3. Update test fixtures in `test_vanilla_comparison.py`
4. Run tests: `pytest tests/test_vanilla_comparison.py -v`

### Updating Tolerance

If legitimate code changes cause small numerical differences:

1. Investigate root cause of differences
2. Document reason in CHANGELOG or commit message
3. Update tolerance in test if justified
4. Regenerate golden reference if needed

## Related Documentation

- `/thoughts/shared/plans/vanilla-comparison-testing.md` - Detailed testing plan
- `tests/conftest.py` - Shared pytest fixtures
- `virnucpro/pipeline/predictor.py` - Refactored predictor implementation
