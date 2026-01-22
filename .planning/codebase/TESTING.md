# Testing

## Testing Framework

**pytest >=8.4.1**

**Configuration:** No explicit `pytest.ini` or `pyproject.toml` pytest section detected

**Test Discovery:** Automatic via pytest conventions (`test_*.py`)

## Test Structure

### Test Directory Layout

```
tests/
├── __init__.py
├── conftest.py                           # Shared fixtures
├── test_prediction_vanilla_comparison.py # Main test suite
└── data/                                 # Test fixtures
    ├── test_sequences.fa
    ├── test_sequences_small.fa
    ├── test_fixed_500bp_*/
    ├── test_with_orfs_*/
    └── ...
```

**Pattern:** Co-located test directory at repository root

**Test Data:** Extensive test fixtures including intermediate pipeline outputs

### Test Files

**Main Test File:**
- `tests/test_prediction_vanilla_comparison.py`

**Purpose:** Validate refactored implementation against original vanilla scripts

**Strategy:** End-to-end comparison testing

### Test Data Organization

**Test Fixtures:**
1. **Input FASTA files:**
   - `test_sequences.fa` - Main test input
   - `test_sequences_small.fa` - Small input for quick tests

2. **Intermediate outputs (for comparison):**
   - `test_fixed_500bp_chunked500.fa` - Chunking stage output
   - `test_fixed_500bp_identified_nucleotide.fa` - Nucleotide sequences
   - `test_fixed_500bp_identified_protein.faa` - Protein sequences
   - `test_fixed_500bp_merged/` - Merged features
   - `test_fixed_500bp_refactored/` - Refactored output
   - Similar structure for `test_with_orfs_*`

**Pattern:** Each pipeline stage has corresponding test data

## Test Naming

### Test Functions

**Pattern: `test_<functionality>`**

**Examples:**
```python
def test_chunking_matches_vanilla():
    ...

def test_feature_extraction_nucleotide():
    ...

def test_end_to_end_pipeline():
    ...
```

**Descriptive names explaining what is being tested**

### Test Classes

**No test classes detected** - using function-based tests

## Fixtures

### Fixture Definition

**Location:** `tests/conftest.py`

**Purpose:** Shared test setup and teardown

**Common Fixtures (inferred):**
- Input file paths
- Expected output paths
- Temporary directories
- Configuration objects

**Example pattern:**
```python
@pytest.fixture
def test_input_fasta():
    return Path("tests/data/test_sequences.fa")

@pytest.fixture
def temp_output_dir(tmp_path):
    return tmp_path / "output"
```

### Fixture Scope

**Default scope:** Function-level (recreated per test)

**Shared fixtures:** Likely session or module-scoped for expensive setup (model loading)

## Test Markers

### Custom Markers

**GPU Tests:**
```python
@pytest.mark.gpu
def test_feature_extraction_gpu():
    ...
```

**Purpose:** Skip GPU tests when CUDA unavailable

**Slow Tests:**
```python
@pytest.mark.slow
def test_full_pipeline_large_dataset():
    ...
```

**Purpose:** Skip slow tests for quick test runs

### Running Marked Tests

**Run only GPU tests:**
```bash
pytest -m gpu
```

**Skip slow tests:**
```bash
pytest -m "not slow"
```

**Skip GPU tests:**
```bash
pytest -m "not gpu"
```

## Mocking Patterns

### Framework

**unittest.mock** (Python standard library)

**Also available:** pytest-mock plugin

### Common Mocking Patterns

**Mock external model downloads:**
```python
from unittest.mock import patch

@patch('transformers.AutoModel.from_pretrained')
def test_feature_extraction_without_download(mock_model):
    mock_model.return_value = MockModel()
    ...
```

**Mock file I/O:**
```python
@patch('builtins.open')
def test_config_loading(mock_open):
    ...
```

**Mock torch.cuda availability:**
```python
@patch('torch.cuda.is_available', return_value=False)
def test_cpu_fallback(mock_cuda):
    ...
```

### Mocking Strategy

**Pattern: Mock external dependencies, not internal code**

- Mock HuggingFace model downloads
- Mock ESM model downloads
- Mock CUDA availability for CPU-only CI
- Don't mock internal pipeline functions (test them directly)

## Test Patterns

### Vanilla Comparison Testing

**Primary pattern:** Compare refactored vs original implementation

**Approach:**
1. Run vanilla implementation (original `prediction.py`)
2. Run refactored implementation (new `virnucpro` package)
3. Compare outputs at each pipeline stage
4. Assert exact matches or within tolerance

**Example:**
```python
def test_chunking_vanilla_comparison():
    # Run vanilla chunking
    vanilla_output = run_vanilla_chunking("tests/data/test_sequences.fa")

    # Run refactored chunking
    refactored_output = chunk_sequences(load_fasta("tests/data/test_sequences.fa"))

    # Compare
    assert vanilla_output == refactored_output
```

### End-to-End Testing

**Pattern: Full pipeline execution**

**Approach:**
1. Provide input FASTA
2. Run entire pipeline
3. Verify final outputs exist and are valid
4. Compare against known-good outputs

**Example:**
```python
def test_end_to_end_pipeline():
    result = run_prediction(
        input_file="tests/data/test_sequences_small.fa",
        output_dir="temp/output",
        config="config/default_config.yaml"
    )

    assert Path("temp/output/predictions.csv").exists()
    assert Path("temp/output/consensus.txt").exists()
```

### Unit Testing

**Limited pure unit tests**

**Focus:** Integration and end-to-end testing over unit tests

**Reason:** Pipeline stages are tightly integrated

### Assertion Patterns

**File existence:**
```python
assert output_file.exists()
```

**Value equality:**
```python
assert result == expected
```

**Tensor equality:**
```python
assert torch.allclose(tensor1, tensor2, rtol=1e-5)
```

**List/dict comparison:**
```python
assert len(results) == expected_count
assert results[0]['prediction'] == 'viral'
```

## Test Data Management

### Test Data Location

**Directory:** `tests/data/`

**Size:** Extensive (multiple intermediate stage outputs)

**Version Control:** Test data committed to repository

### Test Data Naming

**Pattern: `test_<variant>_<stage>.<ext>`**

**Examples:**
- `test_sequences.fa` - Input
- `test_fixed_500bp_chunked500.fa` - After chunking
- `test_with_orfs_identified_protein.faa` - Protein sequences

### Temporary Test Data

**Pattern: Use pytest's `tmp_path` fixture**

```python
def test_output_generation(tmp_path):
    output_dir = tmp_path / "output"
    run_prediction(output_dir=output_dir)
    assert (output_dir / "predictions.csv").exists()
```

**Automatic cleanup:** pytest cleans up `tmp_path` after test

## Running Tests

### Run All Tests

```bash
pytest
```

### Run Specific Test File

```bash
pytest tests/test_prediction_vanilla_comparison.py
```

### Run Specific Test

```bash
pytest tests/test_prediction_vanilla_comparison.py::test_end_to_end
```

### Run with Verbosity

```bash
pytest -v
```

### Run with Output

```bash
pytest -s  # Show print statements
```

### Run Parallel Tests

**Not configured** - no pytest-xdist detected

## Coverage

### Coverage Tool

**No explicit coverage configuration detected**

**Can add:** pytest-cov plugin

**Usage:**
```bash
pytest --cov=virnucpro --cov-report=html
```

### Coverage Gaps

**Areas likely lacking tests:**
- Error handling paths
- Edge cases (empty inputs, malformed FASTA)
- GPU-specific code paths (if CI is CPU-only)
- Checkpoint resume functionality
- Parallel processing edge cases

**See CONCERNS.md for detailed coverage gaps**

## Test Isolation

### Independent Tests

**Pattern: Each test is independent**

- No test depends on another test's output
- Use fixtures for setup
- Use `tmp_path` for outputs

### Cleanup

**Automatic via pytest:**
- `tmp_path` cleaned up after test
- Fixtures with `yield` for teardown

**Manual cleanup (if needed):**
```python
@pytest.fixture
def temp_output(tmp_path):
    output_dir = tmp_path / "output"
    yield output_dir
    # Cleanup if needed
    shutil.rmtree(output_dir)
```

## CI/CD Integration

**No CI/CD configuration detected**

**Tests run locally:**
```bash
pytest
```

**Missing:**
- GitHub Actions
- GitLab CI
- CircleCI
- Automated test runs on push/PR

## Test Performance

### Test Speed

**Slow tests marked:** `@pytest.mark.slow`

**Reason:** Full pipeline with model loading and feature extraction

### Optimization Strategies

**Potential improvements:**
1. Mock model downloads
2. Use smaller test datasets for quick tests
3. Cache model loading across tests (session-scoped fixture)
4. Parallelize tests with pytest-xdist

## Test Quality Metrics

### Current State

**Strengths:**
- Comprehensive vanilla comparison
- Real-world test data
- End-to-end coverage

**Weaknesses:**
- Limited unit tests
- No coverage metrics
- No automated CI
- Missing edge case tests

### Recommended Additions

1. **Error handling tests**
2. **Edge case tests** (empty inputs, malformed data)
3. **Performance regression tests**
4. **GPU-specific tests** (with skip on CPU)
5. **Checkpoint resume tests**
6. **Parallel processing tests**
