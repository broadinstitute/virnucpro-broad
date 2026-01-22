# Coding Conventions

## Naming Patterns

### Files

**Modules:** Snake case
- `sequence_utils.py`
- `parallel_feature_extraction.py`
- `checkpointing.py`

**Packages:** Lowercase, single-word preferred
- `cli/`
- `core/`
- `pipeline/`
- `utils/`

**Tests:** `test_<functionality>.py`
- `test_prediction_vanilla_comparison.py`

**Config:** `<purpose>_config.yaml`
- `default_config.yaml`

### Functions

**Pattern:** Snake case, verb-based for actions

**Examples:**
- `run_prediction()`
- `chunk_sequences()`
- `load_config()`
- `extract_features()`
- `translate_sequence()`

**CLI Commands:** Verb-based
- `predict` - `virnucpro/cli/predict.py:predict()`

### Variables

**Pattern:** Snake case

**Examples:**
- `chunk_size`
- `output_dir`
- `device_ids`
- `feature_tensors`
- `nucleotide_sequences`

**Constants:** UPPER_SNAKE_CASE (when used)
- Limited usage detected in codebase

### Classes

**Pattern:** PascalCase (limited class usage)

**Examples:**
- `CheckpointManager` - `virnucpro/core/checkpointing.py`
- Custom model classes in `virnucpro/pipeline/models.py`

### Module-Level Variables

**Pattern:** Snake case
- `logger` - Module-level logger instances
- `device` - Device context

## Code Style

### Indentation

**4 spaces** (Python standard)

### Line Length

**No strict limit detected**
- Some lines exceed 100 characters
- No Black or isort configuration found

### Quotes

**Inconsistent:**
- Both single (`'`) and double (`"`) quotes used
- No enforced standard via formatter

### Imports

**Pattern: Grouped and ordered**

**Typical Order:**
1. Standard library imports
2. Third-party imports (PyTorch, BioPython, etc.)
3. Local package imports

**Example from `virnucpro/pipeline/prediction.py`:**
```python
import logging
import os
from pathlib import Path

import torch
from Bio import SeqIO

from virnucpro.core.checkpointing import CheckpointManager
from virnucpro.pipeline.feature_extraction import extract_features
from virnucpro.utils.sequence_utils import chunk_sequences
```

**Note:** Some files have imports inside functions (non-standard pattern)

### Docstrings

**Minimal docstring usage**

**Pattern (when present):** Standard Python docstrings
- Some functions have brief descriptions
- No consistent docstring format (Google, NumPy, etc.)
- Many functions lack docstrings

**Example:**
```python
def chunk_sequences(sequences, chunk_size):
    """Split sequences into fixed-size chunks."""
    ...
```

### Type Hints

**Minimal type hint usage**

- No consistent type annotation pattern
- Some function signatures have hints, many don't
- No mypy configuration detected

**Example (type hints present in some places):**
```python
def load_config(config_path: str) -> dict:
    ...
```

## Error Handling

### Exception Types

**Preferred Exceptions:**
- `ValueError` - For validation failures
- `RuntimeError` - For pipeline/execution failures
- `FileNotFoundError` - For missing files

**Pattern: Raise with descriptive messages**

**Example:**
```python
if not os.path.exists(input_file):
    raise FileNotFoundError(f"Input file not found: {input_file}")
```

### Error Messages

**Pattern: Descriptive with context**

**Examples:**
- Include file paths in file-related errors
- Include parameter values in validation errors
- Prefix with context (e.g., "Failed to load model: ...")

### No Custom Exception Hierarchy

- Uses standard Python exceptions
- No domain-specific exception classes

## Logging

### Framework

**Python Standard Library `logging`**

**Setup:** `virnucpro/core/logging.py` (76 lines)

### Logger Creation

**Pattern: Module-level logger**

```python
import logging

logger = logging.getLogger(__name__)
```

**Used consistently across modules**

### Log Levels

**Usage patterns:**
- `logger.info()` - Pipeline stage progress, major operations
- `logger.warning()` - Non-critical issues, fallbacks
- `logger.error()` - Failures, exceptions
- `logger.debug()` - Detailed debugging (less common)

**Examples:**
```python
logger.info(f"Starting prediction for {len(sequences)} sequences")
logger.warning(f"Checkpoint found, resuming from {stage}")
logger.error(f"Failed to load model: {e}")
```

### Log Message Format

**Pattern: Descriptive with f-strings**

- Use f-strings for interpolation
- Include relevant values (counts, paths, etc.)
- Present tense for ongoing operations
- Past tense for completed operations

## Comments

### Inline Comments

**Minimal usage**

**Pattern (when used):** Explain "why", not "what"

**Examples:**
- Explaining workarounds
- Clarifying non-obvious logic
- TODOs (rare)

### Block Comments

**Limited usage**

- Some files have section headers
- No extensive documentation comments

### TODO Comments

**Rare**

**No consistent TODO pattern detected**

## Function Design

### Function Length

**Variable:**
- Many functions < 50 lines
- Some orchestration functions 100-200 lines (e.g., `run_prediction()` at 462 lines includes multiple stages)

### Function Parameters

**Pattern: Explicit parameters**

- Positional parameters for required args
- Keyword arguments for optional config
- Some functions accept `**kwargs` for flexibility

**Example:**
```python
def extract_features(sequences, model_name, device="cpu", batch_size=8):
    ...
```

### Return Values

**Pattern: Explicit returns**

- Return meaningful values (tensors, lists, dicts)
- `None` for side-effect functions
- Tuples for multiple values

**Example:**
```python
def chunk_sequences(sequences, chunk_size):
    return chunked_sequences, chunk_metadata
```

## Module Design

### Module Size

**Small to medium:**
- Most modules 100-300 lines
- Largest: `virnucpro/core/checkpointing.py` (347 lines)
- Focused modules with single responsibility

### Module Organization

**Pattern: Related functions grouped**

- Sequence operations in `sequence_utils.py`
- Progress tracking in `progress.py`
- File operations in `file_utils.py`

### Package Structure

**Clear layering:**
- CLI layer (`cli/`)
- Core infrastructure (`core/`)
- Business logic (`pipeline/`)
- Utilities (`utils/`)

**No circular dependencies**

## Import Management

### Absolute Imports

**Preferred pattern:**
```python
from virnucpro.core.config import load_config
from virnucpro.pipeline.prediction import run_prediction
```

### Relative Imports

**Not used** - all imports are absolute from package root

### Import Inside Functions

**Anti-pattern present in some files:**
```python
def some_function():
    import torch  # Import inside function
    ...
```

**Location:** `virnucpro/pipeline/prediction.py` has multiple cases

**Why:** Likely to delay heavy imports (PyTorch, Transformers) until needed

## Configuration

### Configuration Format

**YAML-based:** `config/default_config.yaml`

**Pattern: Nested dictionaries**
```yaml
chunking:
  chunk_size: 500

models:
  model_300: "300_model.pth"
  model_500: "500_model.pth"
```

### Configuration Loading

**Via OmegaConf:**
- `virnucpro/core/config.py`
- Supports nested access via dot notation
- Type validation

## Testing Patterns

**Pattern: Vanilla comparison testing**

- Compare refactored implementation against original scripts
- End-to-end pipeline tests
- Test data in `tests/data/`

**See TESTING.md for detailed testing conventions**

## Git Commit Patterns

**Recent commits show:**
- Conventional commit style emerging (e.g., "Update README to reflect production-ready status")
- Descriptive messages
- No strict conventional commit enforcement

## Code Formatting

### No Automated Formatting Detected

**Missing:**
- No Black configuration
- No isort configuration
- No flake8/pylint configuration

**Result:**
- Inconsistent formatting in places
- Manual code style maintenance

## Performance Patterns

### GPU Usage

**Pattern: Explicit device management**
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
```

### Batch Processing

**Pattern: Process in batches for efficiency**
- Feature extraction uses batching
- Configurable batch sizes

### Checkpointing

**Pattern: Save intermediate results**
- Every pipeline stage saves checkpoint
- Enables resume after interruption
- Trade-off: disk space for reliability

## Security Patterns

### File Path Handling

**Pattern: Path validation in `virnucpro/utils/validation.py`**
- Check file existence
- Validate file extensions
- Basic sanitization

### Model Loading

**Potential issue:** `torch.load(..., weights_only=False)`
- Allows arbitrary code execution
- Standard for loading custom models
- See CONCERNS.md for security considerations
