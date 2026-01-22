# Architecture

## Architectural Pattern

**Layered Pipeline Architecture**

The codebase follows a layered architecture with a multi-stage pipeline for viral nucleotide prediction:

```
CLI Layer (User Interface)
    ↓
Core Layer (Cross-cutting concerns)
    ↓
Pipeline Layer (Business logic)
    ↓
Utils Layer (Shared utilities)
```

## System Layers

### 1. CLI Layer

**Location:** `virnucpro/cli/`

**Purpose:** User interface and command routing

**Key Files:**
- `virnucpro/__main__.py` - Entry point (`python -m virnucpro`)
- `virnucpro/cli/main.py:cli()` - Click CLI group
- `virnucpro/cli/predict.py:predict()` - Main predict command (183 lines)

**Responsibilities:**
- Parse command-line arguments
- Validate user inputs
- Route to pipeline layer
- Display results and progress

**Data Flow:** CLI args → Pipeline configuration → Pipeline execution

### 2. Core Layer

**Location:** `virnucpro/core/`

**Purpose:** Cross-cutting infrastructure concerns

**Key Files:**
- `virnucpro/core/config.py` (143 lines) - Configuration management
- `virnucpro/core/checkpointing.py` (347 lines) - State persistence
- `virnucpro/core/device.py` (149 lines) - GPU/CPU device management
- `virnucpro/core/logging.py` (76 lines) - Logging setup

**Responsibilities:**
- Load and validate YAML configuration
- Manage checkpoint files for resumable execution
- Device selection (CPU vs GPU, multi-GPU)
- Logging configuration

**Key Abstractions:**
- `CheckpointManager` - Saves/loads pipeline state
- Device context management

### 3. Pipeline Layer

**Location:** `virnucpro/pipeline/`

**Purpose:** Core business logic for viral prediction

**Key Files:**
- `virnucpro/pipeline/prediction.py:run_prediction()` (462 lines) - Main orchestrator
- `virnucpro/pipeline/feature_extraction.py` (244 lines) - DNABERT-S and ESM-2 feature extraction
- `virnucpro/pipeline/parallel_feature_extraction.py` (113 lines) - Multi-GPU feature extraction
- `virnucpro/pipeline/models.py` (110 lines) - Model loading and prediction
- `virnucpro/pipeline/predictor.py` (165 lines) - Prediction logic

**Responsibilities:**
- Orchestrate 9-stage prediction pipeline
- Extract features from DNA/protein sequences
- Load and run prediction models
- Generate predictions and consensus

**9-Stage Pipeline:**
1. **Chunking** - Split long sequences into fixed-size chunks
2. **Translation** - Translate DNA to all 6 reading frames (ORF detection)
3. **Splitting** - Split sequences by type (nucleotide vs protein)
4. **Feature Extraction (Nucleotide)** - DNABERT-S embeddings
5. **Feature Extraction (Protein)** - ESM-2 embeddings
6. **Merging Features** - Combine nucleotide and protein features
7. **Prediction** - Run trained model on features
8. **Consensus** - Aggregate predictions from multiple frames
9. **Output** - Write results to CSV/TXT

**Checkpointing:** Each stage can be resumed from checkpoint

### 4. Utils Layer

**Location:** `virnucpro/utils/`

**Purpose:** Shared utility functions

**Key Files:**
- `virnucpro/utils/sequence_utils.py` (294 lines) - FASTA I/O, chunking, translation
- `virnucpro/utils/validation.py` (94 lines) - Input validation
- `virnucpro/utils/progress.py` (185 lines) - Progress bar utilities
- `virnucpro/utils/file_utils.py` (85 lines) - File operations

**Responsibilities:**
- Sequence manipulation (chunking, translation, reverse complement)
- File path validation
- Progress tracking
- Generic file utilities

## Data Flow

### High-Level Flow

```
User FASTA Input
    ↓
CLI Validation
    ↓
Configuration Loading
    ↓
Pipeline Orchestration (9 stages with checkpoints)
    ↓
CSV/TXT Output
```

### Detailed Pipeline Flow

```
Input FASTA
    ↓
[1] Chunking (500bp or 300bp chunks)
    ├─> Checkpoint: chunked_sequences.pt
    ↓
[2] Translation (6 reading frames)
    ├─> Checkpoint: translated_sequences/
    ↓
[3] Splitting (nucleotide vs protein)
    ├─> Checkpoint: nucleotide/, protein/
    ↓
[4] DNABERT-S Feature Extraction
    ├─> Checkpoint: nucleotide_features.pt
    ↓
[5] ESM-2 Feature Extraction
    ├─> Checkpoint: protein_features.pt
    ↓
[6] Feature Merging
    ├─> Checkpoint: merged_features.pt
    ↓
[7] Model Prediction
    ├─> Checkpoint: predictions.pt
    ↓
[8] Consensus Aggregation
    ├─> Checkpoint: consensus.pt
    ↓
[9] Output Writing
    └─> predictions.csv, consensus.txt
```

## Key Entry Points

**Primary Entry Point:**
```python
virnucpro/__main__.py
    → virnucpro/cli/main.py:cli()
    → virnucpro/cli/predict.py:predict()
    → virnucpro/pipeline/prediction.py:run_prediction()
```

**Alternative Entry (Legacy):**
```python
prediction.py  # Root-level vanilla script (legacy)
```

## Abstraction Patterns

### Checkpointing

**Pattern:** State persistence at each pipeline stage

**Implementation:**
- `virnucpro/core/checkpointing.py:CheckpointManager`
- Each stage checks for existing checkpoint before running
- Enables resume after interruption
- Stored as `.pt` files (PyTorch tensors) or FASTA files

### Device Management

**Pattern:** Abstract GPU/CPU selection

**Implementation:**
- `virnucpro/core/device.py`
- Automatic CUDA detection
- Multi-GPU support for DNABERT-S
- Falls back to CPU if no GPU

### Parallel Processing

**Pattern:** Multi-GPU feature extraction

**Implementation:**
- `virnucpro/pipeline/parallel_feature_extraction.py`
- Splits sequences across GPUs
- Only for DNABERT-S (ESM-2 is single-GPU)
- Uses `multiprocessing` with spawn context

### Configuration

**Pattern:** YAML-based configuration with defaults

**Implementation:**
- `virnucpro/core/config.py`
- `config/default_config.yaml` as base
- OmegaConf for parsing
- Nested configuration structure

## Error Handling

**Strategy:** Fail-fast with logging

**Patterns:**
- ValueError for validation failures
- RuntimeError for pipeline failures
- Logging before exceptions
- No custom exception hierarchy detected

## Concurrency Model

**Multi-GPU:**
- `multiprocessing` for parallel feature extraction
- Spawn context for CUDA compatibility

**Single-Threaded:**
- Main pipeline is sequential
- No async/await
- No threading

## Testing Strategy

**Location:** `tests/`

**Approach:**
- Vanilla comparison testing (comparing refactored vs original implementation)
- End-to-end pipeline tests
- Test data: Small FASTA files in `tests/data/`

**Test Files:**
- `tests/test_prediction_vanilla_comparison.py` - Main comparison suite
- `tests/conftest.py` - Shared fixtures

## Module Dependencies

```
CLI Layer
    ↓ depends on
Pipeline Layer
    ↓ depends on
Core Layer + Utils Layer
    ↓ depends on
External Libraries (PyTorch, Transformers, BioPython)
```

**No Circular Dependencies**

## Legacy Code

**Root-Level Scripts:**
- `prediction.py` - Original prediction script (preserved for reference)
- `units.py` - Original utility functions (preserved for reference)

**Status:** Not imported by refactored code; kept for validation testing
