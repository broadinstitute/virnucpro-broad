# Testing Patterns

**Analysis Date:** 2026-02-06

## Test Framework

**Runner:**
- Not detected - no pytest.ini, tox.ini, or test runner configuration found
- No testing framework installed in requirements.txt or pixi.toml
- Testing infrastructure not implemented in this codebase

**Assertion Library:**
- Not applicable - no test framework present

**Run Commands:**
- Not defined - no test commands configured
- No Makefile or test runner scripts present

## Test File Organization

**Location:**
- No test files detected in codebase
- No `tests/` directory or `test_*` files present
- No `*_test.py` files found

**Naming:**
- Not applicable - no test files exist

**Structure:**
- Not applicable - no test structure implemented

## Test Structure

**Suite Organization:**
- Not implemented

**Patterns:**
- Not applicable - no tests present in codebase

## Mocking

**Framework:**
- Not implemented - no mock library usage detected
- No unittest.mock imports found

**Patterns:**
- Not applicable - no mocking infrastructure present

**What to Mock:**
- Recommendation: External API calls (torch.load, SeqIO.parse, requests.get)
- Recommendation: Model inference (DNABERT-S, ESM models)
- Recommendation: File I/O operations for unit tests

**What NOT to Mock:**
- Recommendation: Core data transformation logic in units.py
- Recommendation: Dataset class indexing and aggregation
- Recommendation: Model training loops

## Fixtures and Factories

**Test Data:**
- Not implemented

**Location:**
- Not applicable - no fixtures/factories present

## Coverage

**Requirements:**
- Not enforced - no coverage configuration found
- No coverage targets defined

**View Coverage:**
- Not configured

## Test Types

**Unit Tests:**
- Not implemented
- Recommended: Test individual functions in units.py (translate_dna, reverse_complement, identify_seq)
- Recommended: Test Dataset classes (FileBatchDataset, PredictDataBatchDataset) with synthetic data
- Recommended: Test data validation and error handling

**Integration Tests:**
- Not implemented
- Recommended: Test data pipeline (chunk → identify → extract features → merge)
- Recommended: Test model training workflow (dataset → model → evaluation metrics)
- Recommended: Test prediction pipeline (model → postprocessing → output)

**E2E Tests:**
- Not implemented
- Recommended: Could test complete viral/non-viral classification workflow

## Manual Testing Evidence

While no automated tests exist, manual validation is observable in code patterns:

**Validation Patterns in Code:**
- Checkpoint verification: `if os.path.exists(output_file) or os.path.exists(merged_file_path):`
- File integrity checks: `if local_size == remote_size:` in `download_data.py` (lines 60-68)
- Data existence validation: `if "*" not in item:` in `units.py` (line 109, 130)
- Exception handling for data parsing: Try-except in `drew_fig.py` (lines 59-64)

**Test-like Assertions in Runtime:**
- Sequence validation: `ambiguous_bases = {'N', 'R', 'Y', ...}` check in `units.py` (lines 84-86)
- Count validation: `if sum(1 for _ in SeqIO.parse(...)) == sequences_per_file:` in `features_extract.py` (line 51)
- Data shape validation implicit in DataLoader operations

## Testable Components

**High Priority for Testing:**
1. `units.py:identify_seq()` - Core biological sequence processing logic
   - Complex conditionals based on ambiguous bases, stop codons
   - Multiple frame translations

2. `units.py:translate_dna()` - DNA to protein translation
   - Requires synthetic DNA sequences as test fixtures
   - Multiple reading frames tested

3. Dataset classes (`train.py`, `prediction.py`)
   - `FileBatchDataset.__getitem__()` - Index mapping across files
   - `PredictDataBatchDataset.__getitem__()` - Same indexing logic

4. `units.py:reverse_complement()` - Sequence reversal
   - Simple logic, high test coverage potential

**Medium Priority:**
1. File I/O functions: `split_fasta_chunk()`, `split_fasta_file()`, `merge_data()`
2. Feature extraction: `extract_DNABERT_S()`, `extract_esm()` (mocked models)
3. Metrics calculation: `train.py:test_model()` with synthetic predictions

**Lower Priority (Integration):**
1. `make_predictdata()` - Full prediction pipeline
2. `train_model()` - Full training loop with mock dataloaders

## Recommended Testing Approach

**Framework Selection:**
- Recommend: pytest for simplicity and fixtures
- Alternative: unittest (Python standard library)

**Mock Strategy:**
- Mock large model loading (DNABERT-S, ESM2) to reduce test time
- Use small fixture sequences for DNA/protein transformations
- Mock file I/O for unit tests of data processing logic

**Test Data Requirements:**
- Small FASTA files (10-20 sequences) for integration tests
- Synthetic DNA sequences (codon combinations) for translation tests
- Pre-computed embedding arrays for merge operations

**Example Test Structure (Pseudocode):**
```python
# Unit test for translate_dna
def test_translate_dna_forward_frames():
    sequence = "ATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGTAA"  # AUG...AUG...STOP
    frames = translate_dna(sequence)
    assert len(frames) == 6  # 3 forward + 3 reverse
    assert frames[0].startswith('M')  # Start codon

# Unit test for reverse_complement
def test_reverse_complement():
    sequence = "ACGT"
    result = reverse_complement(sequence)
    assert result == "ACGT"  # Palindrome

# Integration test for identify_seq
def test_identify_seq_viral():
    sequence = "ATGATGATGATGATGATGATGATGATGTAA"
    results = identify_seq("test_seq", sequence)
    assert isinstance(results, list)
    assert all('seqid' in r for r in results)
```

## Coverage Gaps

**Untested Areas:**
- `download_data.py` - Network I/O, no mocking infrastructure
- `drew_fig.py` - Matplotlib visualization, requires image comparison
- `features_extract.py` - Full data pipeline with model loading
- `make_train_dataset_300.py`, `make_train_dataset_500.py` - Data preprocessing pipelines
- `train.py` - Model training and validation loops
- `prediction.py` - Model inference and post-processing

**Risk Assessment:**
- HIGH: Core sequence identification logic lacks tests (units.py identify_seq)
- MEDIUM: Dataset indexing across files not tested (potential for off-by-one errors)
- MEDIUM: Model I/O not tested (corrupt model loading not caught)
- LOW: Utility functions (reverse_complement, translate_dna) relatively simple

---

*Testing analysis: 2026-02-06*
