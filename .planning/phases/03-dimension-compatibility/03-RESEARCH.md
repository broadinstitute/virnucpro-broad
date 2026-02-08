# Phase 3: Dimension Compatibility - Research

**Researched:** 2026-02-08
**Domain:** PyTorch tensor dimension validation, checkpoint versioning, backward compatibility
**Confidence:** HIGH

## Summary

Phase 3 updates all downstream code to handle the dimension change from ESM2 3B (2560-dim protein embeddings) to FastESM2_650 (1280-dim protein embeddings). The merged feature vector changes from 3328-dim to 2048-dim (768 DNA + 1280 protein). This phase ensures merge_data(), MLPClassifier, and checkpoint handling work correctly with the new dimensions.

This is fundamentally a **dimension migration and validation** problem, not a library integration problem. The core technology stack (PyTorch, standard Python) is already in place. The critical challenge is preventing silent failures when dimensions mismatch, establishing checkpoint versioning to prevent loading incompatible models, and providing clear error messages when old checkpoints are encountered.

Standard practice in PyTorch dimension compatibility updates: (1) add dimension validation with assertions at critical points, (2) use semantic versioning for checkpoints to detect incompatibility, (3) embed metadata in saved tensors to track model provenance, (4) define custom exception classes for dimension errors. User decisions from CONTEXT.md lock in specific validation strategy (fail-fast with assertions), backward compatibility approach (hard incompatibility, single model only), and metadata structure.

**Primary recommendation:** Implement fail-fast dimension validation with assert statements at merge points and model input, use semantic versioning v2.0.0 for checkpoints to signal breaking change, embed comprehensive metadata in feature files and checkpoints, and create custom DimensionError exception class for standardized error reporting.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

#### Validation strategy
- Comprehensive checks: Validate dimensions at extraction output, merge points, model input, and checkpoint loading
- Fail fast with assertions: Use assert statements that crash immediately on dimension mismatch
- Basic error messages: State expected vs actual dimensions (e.g., "Expected 1280-dim protein embeddings, got 2560-dim")
- Configurable toggle: Environment variable or config option (VALIDATE_DIMS) to enable/disable validation

#### Backward compatibility
- Hard incompatibility: Old ESM2 3B checkpoints cannot load with new code - namespace protection prevents silent failures
- Single model only: Code supports FastESM2_650 only after Phase 3 - clean migration, no dual pipeline support
- Clear error with migration guide: Detect old checkpoints and show message like "This checkpoint uses ESM2 3B (2560-dim). Re-extract features with FastESM2_650 and retrain."
- Filename convention: Use different naming patterns (*_fastesm.pt vs *_esm2.pt) to distinguish old vs new feature files

#### Metadata and versioning
- Track embedding model info: Model name (fastesm650), dimensions (1280), HuggingFace model ID
- Track feature dimensions: DNA dim (768), protein dim (1280), merged dim (2048)
- Track training metadata: Training date, dataset version, hyperparameters used
- Semantic versioning: checkpoint_version: '2.0.0' - major version changes break compatibility
- Embed metadata in feature files: Store {'embeddings': tensor, 'model': 'fastesm650', 'dim': 1280, 'extraction_date': ...} in .pt files

#### Error handling
- Immediate failure on mismatch: Stop execution when dimension mismatch detected during prediction
- Technical error details: Show expected dim, actual dim, tensor shape, code location - for developers
- Custom DimensionError class: Define custom exception with standardized attributes for all dimension validation failures
- Always validate critical paths: Some checks (like model input dims) always run even when VALIDATE_DIMS is disabled

### Claude's Discretion
- Checkpoint metadata structure (top-level keys vs nested dict)
- Exact wording of error messages beyond basic dimension info
- Location and frequency of validation checks within "comprehensive" approach

### Deferred Ideas (OUT OF SCOPE)
None - discussion stayed within phase scope

</user_constraints>

## Standard Stack

Phase 3 works entirely with existing dependencies. No new libraries required.

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| PyTorch | 2.9.0a0+ | Tensor operations, checkpoint save/load | Universal deep learning framework, already in use |
| Python | 3.10+ | Standard library (os, logging, datetime) | Language runtime |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| os | stdlib | Environment variable (VALIDATE_DIMS) | Configuration toggle |
| logging | stdlib | Dimension validation warnings | Debug output |
| datetime | stdlib | Timestamp metadata | Feature file provenance tracking |

### No Installation Required
All functionality uses existing project dependencies. Phase 3 is pure code modification.

## Architecture Patterns

### Dimension Constants Module
Create centralized dimension constants to prevent magic numbers scattered throughout code.

```python
# Source: PyTorch best practices (codebases define constants in separate module)
# Location: units.py or new constants.py

# OLD dimensions (ESM2 3B)
OLD_DNA_DIM = 768
OLD_PROTEIN_DIM = 2560
OLD_MERGED_DIM = 3328  # 768 + 2560

# NEW dimensions (FastESM2_650)
NEW_DNA_DIM = 768  # unchanged
NEW_PROTEIN_DIM = 1280
NEW_MERGED_DIM = 2048  # 768 + 1280

# Checkpoint versioning
CHECKPOINT_VERSION_OLD = "1.0.0"  # ESM2 3B
CHECKPOINT_VERSION_NEW = "2.0.0"  # FastESM2_650 (breaking change)
```

### Custom Exception Pattern
Define DimensionError exception class with standardized attributes for all dimension validation failures.

```python
# Source: Python best practices, 2026 standards emphasize structured error attributes
# https://www.geeksforgeeks.org/python/define-custom-exceptions-in-python/
# https://towardsdatascience.com/how-to-define-custom-exception-classes-in-python-bfa346629bca/

class DimensionError(Exception):
    """Raised when tensor dimensions don't match expected values."""

    def __init__(self, expected_dim, actual_dim, tensor_name=None, location=None):
        self.expected_dim = expected_dim
        self.actual_dim = actual_dim
        self.tensor_name = tensor_name or "unknown"
        self.location = location or "unknown"

        message = (
            f"Dimension mismatch at {self.location}: "
            f"Expected {self.expected_dim}-dim {self.tensor_name}, "
            f"got {self.actual_dim}-dim"
        )
        super().__init__(message)
```

### Fail-Fast Validation Pattern
Use assert statements at critical points to crash immediately on dimension mismatch.

```python
# Source: User decision from CONTEXT.md + PyTorch dimension validation best practices
# https://discuss.pytorch.org/t/canonical-way-to-assert-tensor-shape/56649

import os

# Configurable toggle via environment variable
VALIDATE_DIMS = os.getenv('VALIDATE_DIMS', 'true').lower() == 'true'

def validate_protein_embedding(embedding, protein_id):
    """Validate protein embedding dimensions."""
    if VALIDATE_DIMS:
        expected_dim = NEW_PROTEIN_DIM  # 1280
        actual_dim = embedding.shape[0] if embedding.dim() == 1 else embedding.shape[-1]

        assert embedding.shape == (expected_dim,), (
            f"Expected {expected_dim}-dim protein embedding for {protein_id}, "
            f"got {embedding.shape}"
        )

def validate_merged_features(merged_tensor, seq_id):
    """Validate merged feature dimensions (always runs, even if VALIDATE_DIMS=false)."""
    expected_dim = NEW_MERGED_DIM  # 2048
    actual_dim = merged_tensor.shape[-1]

    # Critical path validation - always runs
    if actual_dim != expected_dim:
        raise DimensionError(
            expected_dim=expected_dim,
            actual_dim=actual_dim,
            tensor_name="merged_features",
            location=f"merge_data() for {seq_id}"
        )
```

### Checkpoint Metadata Pattern
Embed comprehensive metadata in all saved checkpoints and feature files.

```python
# Source: PyTorch Lightning checkpointing best practices 2026
# https://lightning.ai/docs/pytorch/stable/common/checkpointing_basic.html
# https://medium.com/@piyushkashyap045/how-to-save-and-load-checkpoints-for-training-a-cnn-with-pytorch-e17395cdbd3d

import datetime

def save_feature_file_with_metadata(proteins, data, out_file, model_name="fastesm650"):
    """Save protein embeddings with metadata."""
    metadata = {
        'model_name': model_name,
        'model_id': 'Synthyra/FastESM2_650',
        'embedding_dim': NEW_PROTEIN_DIM,  # 1280
        'extraction_date': datetime.datetime.now().isoformat(),
        'checkpoint_version': CHECKPOINT_VERSION_NEW,  # "2.0.0"
    }

    checkpoint = {
        'proteins': proteins,
        'data': data,
        'metadata': metadata
    }

    torch.save(checkpoint, out_file)

def save_training_checkpoint(model, optimizer, epoch, best_loss):
    """Save training checkpoint with full metadata."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_loss': best_loss,
        'metadata': {
            'checkpoint_version': CHECKPOINT_VERSION_NEW,  # "2.0.0"
            'dna_dim': NEW_DNA_DIM,  # 768
            'protein_dim': NEW_PROTEIN_DIM,  # 1280
            'merged_dim': NEW_MERGED_DIM,  # 2048
            'model_type': 'fastesm650',
            'training_date': datetime.datetime.now().isoformat(),
            'input_dim': NEW_MERGED_DIM,  # for MLPClassifier
            'hidden_dim': 512,
            'num_class': 2
        }
    }
    torch.save(checkpoint, 'model_fastesm650.pth')
```

### Checkpoint Version Detection Pattern
Detect old checkpoints and provide clear migration guidance.

```python
# Source: Semantic versioning for ML models best practices 2026
# https://gerben-oostra.medium.com/semantic-versioning-for-ml-models-8315d03907bf
# https://www.deepchecks.com/model-versioning-for-ml-models/

def load_checkpoint_with_validation(checkpoint_path):
    """Load checkpoint and validate version compatibility."""
    checkpoint = torch.load(checkpoint_path, weights_only=False)

    # Check if metadata exists (new format)
    if 'metadata' not in checkpoint:
        raise ValueError(
            f"This checkpoint is from ESM2 3B pipeline (no metadata found). "
            f"Re-extract features with FastESM2_650 and retrain. "
            f"Old checkpoint: {checkpoint_path}"
        )

    # Check version
    checkpoint_version = checkpoint['metadata'].get('checkpoint_version', '1.0.0')
    if checkpoint_version.startswith('1.'):
        raise ValueError(
            f"This checkpoint uses ESM2 3B (2560-dim, version {checkpoint_version}). "
            f"Re-extract features with FastESM2_650 and retrain. "
            f"Incompatible checkpoint: {checkpoint_path}"
        )

    # Validate dimensions match expected
    metadata = checkpoint['metadata']
    expected_input_dim = NEW_MERGED_DIM  # 2048
    actual_input_dim = metadata.get('input_dim', metadata.get('merged_dim'))

    if actual_input_dim != expected_input_dim:
        raise DimensionError(
            expected_dim=expected_input_dim,
            actual_dim=actual_input_dim,
            tensor_name="model_input",
            location=f"load_checkpoint({checkpoint_path})"
        )

    return checkpoint
```

### MLPClassifier Update Pattern
Update model initialization to use new dimensions with validation.

```python
# Source: Current codebase pattern (train.py line 105)
# Modified to use constants and add validation

# OLD (train.py):
# input_dim = 3328

# NEW:
input_dim = NEW_MERGED_DIM  # 2048
hidden_dim = 512
num_class = 2

mlp_model = MLPClassifier(input_dim, hidden_dim, num_class)

# Add validation during forward pass (in MLPClassifier.forward)
def forward(self, x):
    # Critical path validation - always runs
    if x.shape[-1] != self.hidden_layer.in_features:
        raise DimensionError(
            expected_dim=self.hidden_layer.in_features,
            actual_dim=x.shape[-1],
            tensor_name="model_input",
            location="MLPClassifier.forward()"
        )

    x = self.hidden_layer(x)
    x = self.bn1(x)
    x = F.relu(x)
    x = self.dropout(x)
    x = self.output_layer(x)
    return x
```

### Anti-Patterns to Avoid

- **Hardcoded dimension numbers scattered throughout code:** Leads to inconsistency and difficult maintenance. Use centralized constants instead.

- **Silent dimension mismatches:** PyTorch may broadcast or truncate tensors in unexpected ways. Always validate explicitly with assertions.

- **Checkpoint loading without version checks:** Loading incompatible checkpoints can produce nonsensical results instead of failing fast. Always check metadata.

- **Optional validation everywhere:** Critical paths (model input, checkpoint loading) must always validate, even when VALIDATE_DIMS=false. Only disable non-critical checks.

- **Generic error messages:** "Shape mismatch" without context is useless. Include expected vs actual dimensions, location, and tensor name.

## Don't Hand-Roll

Problems that look simple but have existing solutions or patterns:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Checkpoint versioning scheme | Custom version string format | Semantic versioning (MAJOR.MINOR.PATCH) | Standard, widely understood, tools exist for parsing |
| Dimension validation functions | Per-tensor custom checks | Centralized validation functions + DimensionError class | Consistency, reusability, standardized errors |
| Metadata timestamps | Custom date string format | datetime.datetime.now().isoformat() | ISO 8601 standard, sortable, parseable |
| Environment variable parsing | Manual os.getenv() + string comparison everywhere | Single VALIDATE_DIMS constant defined once | DRY principle, single source of truth |

**Key insight:** Dimension compatibility is about **preventing silent failures**, not adding features. Simple, standard patterns (assertions, semantic versioning, ISO timestamps) are preferable to clever custom solutions. The goal is code that fails loudly and helpfully when dimensions mismatch.

## Common Pitfalls

### Pitfall 1: Forgetting to Update All Dimension References
**What goes wrong:** Code has hardcoded `3328` in multiple places (train.py line 105, prediction.py model loading, etc.). Updating one location but missing others causes subtle bugs.

**Why it happens:** Magic numbers scattered throughout codebase instead of centralized constants.

**How to avoid:**
1. Search entire codebase for old dimension numbers: `grep -r "3328\|2560" .`
2. Replace with named constants: `NEW_MERGED_DIM`, `NEW_PROTEIN_DIM`
3. Add assertion at each usage point to catch remaining hardcoded values

**Warning signs:** Training runs but validation accuracy is 50% (random guessing) - likely dimension mismatch silently breaking model.

### Pitfall 2: Assertion Only in Debug Mode
**What goes wrong:** Python's `assert` statements are disabled when running with `python -O` (optimized mode). Critical dimension checks silently disappear in production.

**Why it happens:** Misunderstanding that `assert` is for testing only, not production validation.

**How to avoid:**
- Use `assert` for optional checks (when VALIDATE_DIMS=true)
- Use `if` + `raise DimensionError` for critical paths (always run)
- Document which validations are critical vs optional

**Warning signs:** Tests pass, production crashes with cryptic PyTorch errors deep in matmul operations.

### Pitfall 3: Incomplete Checkpoint Metadata
**What goes wrong:** Checkpoint saved with partial metadata (e.g., missing `protein_dim`). Later code assumes metadata is complete and crashes when accessing missing keys.

**Why it happens:** Adding metadata fields incrementally without ensuring all save locations are updated.

**How to avoid:**
- Define metadata structure in single function (save_feature_file_with_metadata, save_training_checkpoint)
- All checkpoint saves must use these functions
- Add validation on load: check all required keys exist

**Warning signs:** KeyError when loading checkpoint that was saved during Phase 3 development.

### Pitfall 4: Not Testing Backward Incompatibility
**What goes wrong:** Code intended to reject old checkpoints actually loads them silently, then crashes mysteriously during training.

**Why it happens:** Version check logic is incorrect (e.g., checking `checkpoint_version == '1.0.0'` but old checkpoints have no version field).

**How to avoid:**
- Test with actual old checkpoint file from before Phase 3
- Verify error message is helpful and mentions migration path
- Test both "no metadata" case and "version 1.x" case

**Warning signs:** Old checkpoint loads successfully but training immediately crashes with dimension mismatch.

### Pitfall 5: Merge Dimension Mismatch Silent Failure
**What goes wrong:** merge_data() concatenates 768-dim DNA + 2560-dim protein (old), producing 3328-dim output instead of expected 2048-dim. Code continues silently until model forward pass.

**Why it happens:** Feature extraction updated to FastESM2 but old protein embeddings still cached on disk. merge_data() loads cached file instead of re-extracting.

**How to avoid:**
- Validate protein embedding dimensions immediately after loading in merge_data()
- Use filename convention (*_fastesm.pt vs *_esm2.pt) to distinguish old vs new
- Add assertion before torch.cat() to check both tensor dimensions

**Warning signs:** merge_data() completes successfully but merged file has wrong shape. Model loading crashes.

## Code Examples

Verified patterns for Phase 3 implementation.

### Complete Dimension Validation Function
```python
# Source: User decisions from CONTEXT.md + PyTorch validation best practices
# Location: units.py

import os
import torch

# Configuration
VALIDATE_DIMS = os.getenv('VALIDATE_DIMS', 'true').lower() == 'true'

# Dimension constants
NEW_DNA_DIM = 768
NEW_PROTEIN_DIM = 1280
NEW_MERGED_DIM = 2048

class DimensionError(Exception):
    """Raised when tensor dimensions don't match expected values."""

    def __init__(self, expected_dim, actual_dim, tensor_name=None, location=None):
        self.expected_dim = expected_dim
        self.actual_dim = actual_dim
        self.tensor_name = tensor_name or "unknown"
        self.location = location or "unknown"

        message = (
            f"Dimension mismatch at {self.location}: "
            f"Expected {self.expected_dim}-dim {self.tensor_name}, "
            f"got {self.actual_dim}-dim"
        )
        super().__init__(message)

def validate_protein_embeddings(proteins, data):
    """
    Validate protein embedding dimensions after extraction.
    Optional check (respects VALIDATE_DIMS setting).
    """
    if not VALIDATE_DIMS:
        return

    for protein, embedding in zip(proteins, data):
        if embedding.shape != (NEW_PROTEIN_DIM,):
            raise DimensionError(
                expected_dim=NEW_PROTEIN_DIM,
                actual_dim=embedding.shape[0] if embedding.dim() == 1 else embedding.shape[-1],
                tensor_name=f"protein_embedding[{protein}]",
                location="extract_fast_esm() output"
            )

def validate_merge_inputs(nucleotide_tensor, protein_tensor, seq_id):
    """
    Validate merge_data() input dimensions.
    Critical path - always runs regardless of VALIDATE_DIMS.
    """
    # DNA dimension check
    if nucleotide_tensor.shape != (NEW_DNA_DIM,):
        raise DimensionError(
            expected_dim=NEW_DNA_DIM,
            actual_dim=nucleotide_tensor.shape[0],
            tensor_name=f"dna_embedding[{seq_id}]",
            location="merge_data() input validation"
        )

    # Protein dimension check
    if protein_tensor.shape != (NEW_PROTEIN_DIM,):
        raise DimensionError(
            expected_dim=NEW_PROTEIN_DIM,
            actual_dim=protein_tensor.shape[0],
            tensor_name=f"protein_embedding[{seq_id}]",
            location="merge_data() input validation"
        )

def validate_merged_output(merged_tensor, seq_id):
    """
    Validate merge_data() output dimensions.
    Critical path - always runs.
    """
    if merged_tensor.shape != (NEW_MERGED_DIM,):
        raise DimensionError(
            expected_dim=NEW_MERGED_DIM,
            actual_dim=merged_tensor.shape[0],
            tensor_name=f"merged_features[{seq_id}]",
            location="merge_data() output"
        )
```

### Updated merge_data() with Validation
```python
# Source: Current implementation (units.py line 475) + added validation
# Location: units.py

def merge_data(DNABERT_S_data, ESM_data, merged_file, data_type=None):
    """Merge DNA and protein embeddings with dimension validation."""

    DNABERT_S_outfile = torch.load(DNABERT_S_data)
    ESM_outfile = torch.load(ESM_data)

    nucleotide_data_dict = {}
    protein_data_dict = {}
    merged_data = []

    # Build dictionaries
    for nucleotide, data in zip(DNABERT_S_outfile['nucleotide'], DNABERT_S_outfile['data']):
        nucleotide_data_dict[nucleotide] = torch.tensor(data['mean_representation'])

    for protein, data in zip(ESM_outfile['proteins'], ESM_outfile['data']):
        protein_data_dict[protein] = data

    # Merge with validation
    for item in DNABERT_S_outfile['nucleotide']:
        if item in protein_data_dict and item in nucleotide_data_dict:
            protein_data = protein_data_dict[item]
            nucleotide_data = nucleotide_data_dict[item]

            # Critical path validation - always runs
            validate_merge_inputs(nucleotide_data, protein_data, item)

            # Concatenate
            merged_feature = torch.cat((nucleotide_data, protein_data), dim=-1)

            # Validate output
            validate_merged_output(merged_feature, item)

            merged_data.append(merged_feature)
        else:
            print(f"Warning: {item} not found in both datasets")

    # Stack and save
    merged_data = torch.stack(merged_data)

    if data_type == 'host':
        merged_torch = {'ids': DNABERT_S_outfile['nucleotide'], 'data': merged_data, 'labels': [0]}
    elif data_type == 'viral':
        merged_torch = {'ids': DNABERT_S_outfile['nucleotide'], 'data': merged_data, 'labels': [1]}
    elif data_type is None:
        merged_torch = {'ids': DNABERT_S_outfile['nucleotide'], 'data': merged_data}

    torch.save(merged_torch, merged_file)
```

### Updated MLPClassifier with Input Validation
```python
# Source: Current implementation (train.py line 80) + added validation
# Location: train.py

class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_class):
        super(MLPClassifier, self).__init__()

        self.input_dim = input_dim  # Store for validation
        self.hidden_layer = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, num_class)
        self.dropout = nn.Dropout(0.5)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.hidden_layer.weight)
        nn.init.xavier_uniform_(self.output_layer.weight)

    def forward(self, x):
        # Critical path validation - always runs
        if x.shape[-1] != self.input_dim:
            raise DimensionError(
                expected_dim=self.input_dim,
                actual_dim=x.shape[-1],
                tensor_name="model_input",
                location="MLPClassifier.forward()"
            )

        x = self.hidden_layer(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.output_layer(x)
        return x

# Model initialization with new dimensions
input_dim = NEW_MERGED_DIM  # 2048 (was 3328)
hidden_dim = 512
num_class = 2
mlp_model = MLPClassifier(input_dim, hidden_dim, num_class)
```

### Checkpoint Save with Metadata
```python
# Source: User decisions from CONTEXT.md + PyTorch Lightning patterns
# Location: train.py (modify train_model function)

import datetime

CHECKPOINT_VERSION = "2.0.0"  # Semantic version - breaking change from 1.x

def save_checkpoint_with_metadata(model, optimizer, epoch, best_loss, filepath='model_fastesm650.pth'):
    """Save training checkpoint with comprehensive metadata."""
    checkpoint = {
        # Model state
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_loss': best_loss,

        # Metadata for version tracking and dimension validation
        'metadata': {
            'checkpoint_version': CHECKPOINT_VERSION,
            'model_type': 'fastesm650',
            'huggingface_model_id': 'Synthyra/FastESM2_650',

            # Dimension tracking
            'dna_dim': NEW_DNA_DIM,  # 768
            'protein_dim': NEW_PROTEIN_DIM,  # 1280
            'merged_dim': NEW_MERGED_DIM,  # 2048
            'input_dim': NEW_MERGED_DIM,  # MLPClassifier input
            'hidden_dim': 512,
            'num_class': 2,

            # Training metadata
            'training_date': datetime.datetime.now().isoformat(),
            'dataset_version': '1.0',  # Update if dataset changes
            'pytorch_version': torch.__version__
        }
    }

    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath} (version {CHECKPOINT_VERSION})")
```

### Checkpoint Load with Version Validation
```python
# Source: Semantic versioning best practices + user decisions
# Location: prediction.py (modify model loading)

def load_checkpoint_with_validation(checkpoint_path):
    """
    Load checkpoint and validate version compatibility.
    Prevents loading old ESM2 3B checkpoints with helpful error message.
    """
    checkpoint = torch.load(checkpoint_path, weights_only=False)

    # Check if metadata exists (distinguishes new from old format)
    if 'metadata' not in checkpoint:
        raise ValueError(
            f"This checkpoint is from ESM2 3B pipeline (no metadata found).\n"
            f"Re-extract features with FastESM2_650 and retrain.\n"
            f"Old checkpoint: {checkpoint_path}\n"
            f"Expected checkpoint version: 2.x"
        )

    metadata = checkpoint['metadata']
    checkpoint_version = metadata.get('checkpoint_version', '1.0.0')

    # Check major version (v1.x is incompatible with v2.x)
    major_version = int(checkpoint_version.split('.')[0])
    if major_version < 2:
        raise ValueError(
            f"This checkpoint uses ESM2 3B (2560-dim, version {checkpoint_version}).\n"
            f"Re-extract features with FastESM2_650 and retrain.\n"
            f"Incompatible checkpoint: {checkpoint_path}\n"
            f"Required version: 2.x"
        )

    # Validate dimensions
    expected_merged_dim = NEW_MERGED_DIM  # 2048
    actual_merged_dim = metadata.get('merged_dim', metadata.get('input_dim'))

    if actual_merged_dim != expected_merged_dim:
        raise DimensionError(
            expected_dim=expected_merged_dim,
            actual_dim=actual_merged_dim,
            tensor_name="checkpoint_merged_dim",
            location=f"load_checkpoint({checkpoint_path})"
        )

    print(f"Loaded checkpoint version {checkpoint_version}")
    print(f"  Model type: {metadata.get('model_type')}")
    print(f"  Dimensions: {metadata.get('dna_dim')} DNA + {metadata.get('protein_dim')} protein = {metadata.get('merged_dim')} merged")
    print(f"  Trained: {metadata.get('training_date')}")

    return checkpoint

# Usage in prediction.py:
# OLD: mlp_model = torch.load(model_path, weights_only=False)
# NEW:
checkpoint = load_checkpoint_with_validation(model_path)
mlp_model = MLPClassifier(
    input_dim=checkpoint['metadata']['input_dim'],
    hidden_dim=checkpoint['metadata']['hidden_dim'],
    num_class=checkpoint['metadata']['num_class']
)
mlp_model.load_state_dict(checkpoint['model_state_dict'])
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Hardcoded `input_dim = 3328` in train.py | `input_dim = NEW_MERGED_DIM` constant | Phase 3 | Centralized dimension constants, easier to update |
| No dimension validation | Assert statements at merge points and model input | Phase 3 | Fail fast on dimension mismatch instead of silent failure |
| torch.save(model, 'model.pth') without metadata | Checkpoint with version, dimensions, timestamps | Phase 3 | Track model provenance, prevent loading incompatible checkpoints |
| torch.load() without version check | load_checkpoint_with_validation() | Phase 3 | Clear error message with migration guidance for old checkpoints |
| Generic Python Exception for errors | Custom DimensionError class | Phase 3 | Standardized error attributes for programmatic handling |
| Manual dimension checking per-function | Centralized validate_*() functions | Phase 3 | Consistent validation logic, reusable across codebase |

**Deprecated/outdated:**
- **Hardcoded dimension numbers (3328, 2560):** Replace with constants NEW_MERGED_DIM, NEW_PROTEIN_DIM
- **Checkpoints without metadata:** All new checkpoints must include version and dimension metadata
- **torch.load(model_path, weights_only=False) without validation:** Must validate version and dimensions

## Open Questions

None. All critical aspects of dimension compatibility are well-understood:

1. **Dimension values confirmed:** 768 DNA + 1280 protein = 2048 merged (from CONTEXT.md and Phase 2 validation)
2. **Validation strategy locked:** Fail-fast with assertions, configurable toggle via VALIDATE_DIMS (from CONTEXT.md)
3. **Backward compatibility approach locked:** Hard incompatibility, single model only (from CONTEXT.md)
4. **Metadata structure flexible:** Claude has discretion on exact structure (top-level vs nested dict)

## Sources

### Primary (HIGH confidence)
- Current codebase: train.py, units.py, prediction.py, features_extract.py (direct inspection)
- Phase 2 validation results: scripts/test_extraction.py confirms 1280-dim embeddings
- User decisions: .planning/phases/03-dimension-compatibility/03-CONTEXT.md

### Secondary (MEDIUM confidence)
- [PyTorch dimension validation best practices](https://discuss.pytorch.org/t/canonical-way-to-assert-tensor-shape/56649) - Community discussion on assert patterns
- [PyTorch Lightning checkpointing](https://lightning.ai/docs/pytorch/stable/common/checkpointing_basic.html) - Metadata patterns
- [How to Save and Load Checkpoints for Training a CNN with PyTorch](https://medium.com/@piyushkashyap045/how-to-save-and-load-checkpoints-for-training-a-cnn-with-pytorch-e17395cdbd3d) - Checkpoint metadata examples
- [Define Custom Exceptions in Python](https://www.geeksforgeeks.org/python/define-custom-exceptions-in-python/) - Exception class patterns
- [How to Define Custom Exception Classes in Python](https://towardsdatascience.com/how-to-define-custom-exception-classes-in-python-bfa346629bca/) - 2026 best practices
- [Semantic Versioning for ML models](https://gerben-oostra.medium.com/semantic-versioning-for-ml-models-8315d03907bf) - Versioning schema for breaking changes
- [Model Versioning for ML Models: A Comprehensive Guide](https://www.deepchecks.com/model-versioning-for-ml-models/) - Checkpoint versioning best practices

### Tertiary (LOW confidence)
- WebSearch results on PyTorch 2.6 torch.load weights_only breaking change - mentioned for awareness but not directly applicable to Phase 3

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - No new dependencies, uses existing PyTorch and Python stdlib
- Architecture patterns: HIGH - Based on current codebase patterns + verified PyTorch/Python best practices
- Pitfalls: HIGH - Derived from actual codebase inspection and common PyTorch dimension mismatch patterns
- Code examples: HIGH - All examples adapted from current codebase (train.py, units.py, prediction.py)

**Research date:** 2026-02-08
**Valid until:** 60 days (stable - dimension compatibility patterns don't change rapidly)

**Critical correction:** REQUIREMENTS.md lists DIM-01 as "1664-dim" but CONTEXT.md and codebase analysis confirm correct value is 2048-dim (768 + 1280). REQUIREMENTS.md has an error that should be corrected.
