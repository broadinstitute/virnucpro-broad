# VirNucPro CLI Refactoring Implementation Plan

**Date**: 2025-11-10
**Author**: Claude
**Scope**: Phase 1 - Prediction Pipeline Refactoring

## Overview

Refactor VirNucPro from a collection of standalone scripts with basic `sys.argv` parsing into a production-ready bioinformatics tool with a proper Click-based CLI interface, comprehensive error handling, GPU selection, checkpointing/resume capability, and YAML configuration support. This plan focuses on Phase 1: the prediction pipeline only.

## Current State Analysis

**Existing Structure** (from `thoughts/shared/research/2025-11-10-virnucpro-codebase-structure.md`):
- 8 Python scripts in flat directory structure
- Only 2 scripts (`prediction.py`, `drew_fig.py`) have CLI via basic `sys.argv`
- `prediction.py:193-195`: Takes 3 positional arguments (fasta_file, expected_length, model_path)
- No error handling, parameter validation, or logging infrastructure
- Hardcoded paths, batch sizes, worker counts throughout
- No package structure or configuration support

**Core Functionality** (`prediction.py:98-197`):
1. Sequence chunking (`prediction.py:110`)
2. Six-frame translation (`prediction.py:112-127`)
3. File splitting for parallel processing (`prediction.py:129-151`)
4. DNABERT-S feature extraction (`prediction.py:137`)
5. ESM-2 feature extraction (`prediction.py:149`)
6. Feature merging (`prediction.py:160-164`)
7. MLP prediction (`prediction.py:166-181`)
8. Consensus scoring (`prediction.py:183-191`)

**Key Dependencies**:
- `units.py`: 11 utility functions for sequence processing and feature extraction
- Pre-trained models: `300_model.pth`, `500_model.pth` (6.8MB each)
- External models: DNABERT-S, ESM2-3B (loaded from HuggingFace/ESM)

## Desired End State

A production-ready viral sequence prediction tool with:

✓ **Modular package structure**: `virnucpro/` with proper submodules
✓ **Click-based CLI**: `python -m virnucpro predict input.fasta`
✓ **GPU selection**: `--device` flag with validation and `--list-devices`
✓ **Checkpointing**: Hash-based resume from failed stages
✓ **YAML config**: External configuration with CLI overrides
✓ **Comprehensive logging**: Structured logging with levels
✓ **Progress reporting**: Auto-refreshing progress bars with tqdm
✓ **Error handling**: Validation and graceful failures
✓ **Clean defaults**: Auto-cleanup intermediate files after success

### Success Verification

After implementation:
```bash
# Basic prediction
python -m virnucpro predict input.fasta

# With options
python -m virnucpro predict input.fasta --model-type 300 --device cuda:1 --output-dir results/

# Resume from checkpoint
python -m virnucpro predict input.fasta --resume

# List devices
python -m virnucpro utils list-devices

# Validate input
python -m virnucpro utils validate input.fasta

# Generate config
python -m virnucpro utils generate-config -o config.yaml
```

## What We're NOT Doing

**Out of Scope for Phase 1**:
- Training pipeline refactoring (`train.py`, `download_data.py`, `make_train_dataset_*.py`, `features_extract.py`)
- Visualization script refactoring (`drew_fig.py`)
- Pip/setuptools packaging (direct execution only)
- Distributed/multi-node execution
- API/web interface
- Database integration for tracking
- Performance optimizations beyond existing implementation
- Modification of ML model architecture or training

## Implementation Approach

**Strategy**: Incremental refactoring with minimal disruption to core logic.

1. **Create new package structure** alongside existing files
2. **Extract and modularize** existing code into logical components
3. **Add new infrastructure** (logging, config, checkpointing) around existing logic
4. **Preserve existing functionality** - only add features, don't change algorithms
5. **Test incrementally** at each phase before proceeding

**Key Principles**:
- Keep existing prediction logic intact
- Add validation and error handling around existing code
- Make refactored code importable and testable
- Maintain backward-compatible intermediate file formats

---

## Phase 1: Project Structure & Infrastructure

### Overview
Create the foundational package structure, logging system, and configuration framework that all subsequent phases will build upon.

### Changes Required

#### 1.1 Package Structure

**Create**: New directory structure

```
VirNucPro/
├── virnucpro/
│   ├── __init__.py
│   ├── __main__.py              # Entry point for python -m virnucpro
│   ├── cli/
│   │   ├── __init__.py
│   │   ├── main.py              # Main Click group
│   │   ├── predict.py           # Predict command
│   │   └── utils.py             # Utility commands
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py            # Configuration management
│   │   ├── logging_setup.py     # Logging configuration
│   │   ├── device.py            # GPU/device management
│   │   └── checkpoint.py        # Checkpointing system
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── prediction.py        # Refactored prediction workflow
│   │   ├── chunking.py          # Sequence chunking
│   │   ├── translation.py       # Six-frame translation
│   │   ├── features.py          # Feature extraction
│   │   └── models.py            # Model classes (MLPClassifier, Dataset)
│   └── utils/
│       ├── __init__.py
│       ├── sequence.py          # Sequence utilities (from units.py)
│       ├── file_utils.py        # File operations
│       └── validation.py        # Input validation
├── config/
│   └── default_config.yaml      # Default configuration
├── tests/                       # Future: test suite
│   └── __init__.py
├── prediction.py                # Keep temporarily for reference
├── units.py                     # Keep temporarily for reference
├── 300_model.pth               # Existing model files
├── 500_model.pth
├── requirements.txt
├── pixi.toml
└── README.md
```

**File**: `virnucpro/__init__.py`
```python
"""
VirNucPro - Viral Nucleotide and Protein Identifier

A tool for identifying viral sequences using six-frame translation
and deep learning models (DNABERT-S and ESM-2).
"""

__version__ = "2.0.0"
__author__ = "VirNucPro Team"

from virnucpro.core.config import Config
from virnucpro.core.logging_setup import setup_logging

__all__ = ["Config", "setup_logging", "__version__"]
```

**File**: `virnucpro/__main__.py`
```python
"""
Entry point for VirNucPro CLI.
Allows execution via: python -m virnucpro
"""

import sys
from virnucpro.cli.main import cli

if __name__ == "__main__":
    sys.exit(cli())
```

#### 1.2 Logging System

**File**: `virnucpro/core/logging_setup.py`

```python
"""Centralized logging configuration for VirNucPro"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


def setup_logging(
    verbose: bool = False,
    log_file: Optional[Path] = None,
    quiet: bool = False
) -> logging.Logger:
    """
    Configure logging for the application.

    Args:
        verbose: Enable DEBUG level logging
        log_file: Optional file path for log output
        quiet: Suppress console output (file logging only)

    Returns:
        Configured logger instance
    """
    # Determine log level
    if verbose:
        log_level = logging.DEBUG
    elif quiet:
        log_level = logging.WARNING
    else:
        log_level = logging.INFO

    # Create formatter
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(log_format, datefmt='%Y-%m-%d %H:%M:%S')

    # Configure root logger
    logger = logging.getLogger('virnucpro')
    logger.setLevel(log_level)
    logger.handlers = []  # Clear any existing handlers

    # Console handler (unless quiet)
    if not quiet:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)  # Always debug level to file
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        if not quiet:
            logger.info(f"Logging to file: {log_file}")

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module.

    Args:
        name: Module name (typically __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(f'virnucpro.{name}')
```

#### 1.3 Configuration System

**File**: `config/default_config.yaml`

```yaml
# VirNucPro Default Configuration

# Version of this config schema
version: "2.0.0"

# Prediction settings
prediction:
  # Batch size for DataLoader
  batch_size: 256

  # Number of worker processes for data loading
  num_workers: 4

  # Number of sequences per intermediate file
  sequences_per_file: 10000

  # Default model type if not specified
  default_model_type: "500"

  # Model file paths (relative to project root)
  models:
    "300": "300_model.pth"
    "500": "500_model.pth"

# Device settings
device:
  # Default device: "auto", "cpu", "cuda", or "cuda:N"
  default: "auto"

  # Fallback to CPU if requested GPU unavailable
  fallback_to_cpu: true

# Feature extraction settings
features:
  dnabert:
    model_name: "zhihan1996/DNABERT-S"
    trust_remote_code: true

  esm:
    model_name: "esm2_t36_3B_UR50D"
    truncation_seq_length: 1024
    toks_per_batch: 2048
    representation_layer: 36

# Checkpointing settings
checkpointing:
  # Enable checkpointing by default
  enabled: true

  # Checkpoint directory name (relative to output dir)
  checkpoint_dir: ".checkpoints"

  # Keep checkpoint history for debugging
  keep_history: 5

# File management
files:
  # Clean intermediate files after successful completion
  auto_cleanup: true

  # Always keep these file types
  keep_files:
    - "prediction_results.txt"
    - "prediction_results_highestscore.csv"

# Logging settings
logging:
  # Log level: DEBUG, INFO, WARNING, ERROR
  default_level: "INFO"

  # Log format
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

  # Auto-create log file in output directory
  auto_log_file: true

# Progress reporting settings
progress:
  # Show progress bars by default
  enabled: true

  # Progress bar width (in characters)
  bar_width: 100

  # Update interval (in seconds) for progress bars
  update_interval: 0.1

# Validation settings
validation:
  # Check for ambiguous bases and warn
  warn_ambiguous_bases: true

  # Maximum ambiguous base percentage before warning
  max_ambiguous_percent: 5.0

  # Check for duplicate sequence IDs
  check_duplicate_ids: true
```

**File**: `virnucpro/core/config.py`

```python
"""Configuration management for VirNucPro"""

import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, field
import logging

logger = logging.getLogger('virnucpro.config')


@dataclass
class Config:
    """
    Application configuration container.

    Loads configuration from YAML file with support for
    CLI overrides and validation.
    """

    # Raw configuration dictionary
    _config: Dict[str, Any] = field(default_factory=dict)

    # Configuration file path
    config_file: Optional[Path] = None

    @classmethod
    def load(cls, config_file: Optional[Path] = None) -> 'Config':
        """
        Load configuration from file.

        Args:
            config_file: Path to YAML config file. If None, uses default.

        Returns:
            Config instance
        """
        # Determine config file
        if config_file is None:
            # Use default config
            default_config = Path(__file__).parent.parent.parent / "config" / "default_config.yaml"
            config_file = default_config
        else:
            config_file = Path(config_file)

        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_file}")

        # Load YAML
        with open(config_file, 'r') as f:
            config_dict = yaml.safe_load(f)

        logger.debug(f"Loaded configuration from {config_file}")

        return cls(_config=config_dict, config_file=config_file)

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.

        Args:
            key_path: Dot-separated path (e.g., "prediction.batch_size")
            default: Default value if key not found

        Returns:
            Configuration value

        Example:
            >>> config.get("prediction.batch_size")
            256
        """
        keys = key_path.split('.')
        value = self._config

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default

        return value

    def set(self, key_path: str, value: Any):
        """
        Set configuration value using dot notation.

        Args:
            key_path: Dot-separated path
            value: Value to set
        """
        keys = key_path.split('.')
        config = self._config

        # Navigate to parent
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]

        # Set value
        config[keys[-1]] = value

    def to_dict(self) -> Dict[str, Any]:
        """Return configuration as dictionary"""
        return self._config.copy()

    def save(self, output_file: Path):
        """
        Save configuration to YAML file.

        Args:
            output_file: Output file path
        """
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            yaml.dump(self._config, f, default_flow_style=False, indent=2)

        logger.info(f"Saved configuration to {output_file}")
```

#### 1.4 Device Management

**File**: `virnucpro/core/device.py`

```python
"""GPU/device management utilities"""

import torch
import sys
import logging
from typing import Optional

logger = logging.getLogger('virnucpro.device')


def validate_and_get_device(device_str: str, fallback_to_cpu: bool = True) -> torch.device:
    """
    Validate device string and return torch.device object.

    Args:
        device_str: Device specification ("cpu", "cuda", "cuda:N", or "N")
        fallback_to_cpu: If True, fallback to CPU on errors (with warning)

    Returns:
        torch.device object

    Raises:
        ValueError: If device is invalid and fallback_to_cpu is False
    """
    # Handle auto-detection
    if device_str.lower() == 'auto':
        device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Auto-detected device: {device_str}")

    # Handle CPU
    if device_str.lower() == 'cpu':
        logger.info("Using CPU")
        return torch.device('cpu')

    # Check CUDA availability
    if not torch.cuda.is_available():
        error_msg = (
            "CUDA/GPU support is not available on this system.\n"
            "Possible causes:\n"
            "  - PyTorch was installed without CUDA support\n"
            "  - No NVIDIA GPU detected\n"
            "  - CUDA drivers not installed"
        )

        if fallback_to_cpu:
            logger.warning(f"{error_msg}\nFalling back to CPU")
            return torch.device('cpu')
        else:
            raise ValueError(f"{error_msg}\nPlease use --device cpu or install GPU support")

    # Parse device specification
    if device_str.lower() == 'cuda':
        device_id = 0
    elif device_str.startswith('cuda:'):
        try:
            device_id = int(device_str.split(':')[1])
        except (IndexError, ValueError):
            raise ValueError(f"Invalid device format: {device_str}")
    elif device_str.isdigit():
        device_id = int(device_str)
    else:
        raise ValueError(
            f"Invalid device specification: {device_str}\n"
            f"Valid formats: 'auto', 'cpu', 'cuda', 'cuda:N', or 'N'"
        )

    # Validate device ID is in range
    num_gpus = torch.cuda.device_count()
    if device_id >= num_gpus:
        error_msg = (
            f"GPU cuda:{device_id} is not available.\n"
            f"This system has {num_gpus} GPU(s): "
            f"{', '.join(f'cuda:{i}' for i in range(num_gpus))}\n"
            f"Use --list-devices to see available devices"
        )

        if fallback_to_cpu:
            logger.warning(f"{error_msg}\nFalling back to CPU")
            return torch.device('cpu')
        else:
            raise ValueError(error_msg)

    device = torch.device(f'cuda:{device_id}')

    # Log device info
    props = torch.cuda.get_device_properties(device_id)
    logger.info(f"Using device: {device}")
    logger.info(f"  GPU: {torch.cuda.get_device_name(device_id)}")
    logger.info(f"  Memory: {props.total_memory / 1e9:.2f} GB")

    return device


def list_available_devices():
    """
    Print information about available compute devices.

    Used by the --list-devices CLI option.
    """
    print("Available compute devices:\n")

    # CPU (always available)
    print("  CPU:")
    print("    Device: cpu")
    print("    Available: Yes")

    # GPUs
    if torch.cuda.is_available():
        print(f"\n  GPUs ({torch.cuda.device_count()} detected):")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"    Device: cuda:{i}")
            print(f"      Name: {torch.cuda.get_device_name(i)}")
            print(f"      Memory: {props.total_memory / 1e9:.2f} GB")
            print(f"      Compute Capability: {props.major}.{props.minor}")

            # Check if device is actually usable
            try:
                test = torch.zeros(1, device=f'cuda:{i}')
                del test
                print(f"      Status: Ready")
            except RuntimeError as e:
                print(f"      Status: Error - {e}")
    else:
        print("\n  GPUs:")
        print("    CUDA not available")
        print("    To enable GPU support:")
        print("      1. Install NVIDIA CUDA drivers")
        print("      2. Install PyTorch with CUDA support:")
        print("         pip install torch --index-url https://download.pytorch.org/whl/cu118")


def test_device(device: torch.device) -> bool:
    """
    Test if device is usable by allocating a small tensor.

    Args:
        device: Device to test

    Returns:
        True if device works, False otherwise
    """
    try:
        test_tensor = torch.zeros(1, device=device)
        del test_tensor
        return True
    except RuntimeError as e:
        logger.error(f"Device test failed for {device}: {e}")
        return False
```

#### 1.5 Progress Reporting Module

**File**: `virnucpro/utils/progress.py`

```python
"""Progress reporting utilities using tqdm"""

from tqdm import tqdm
from typing import Optional, Iterable, Any
import logging
import sys

logger = logging.getLogger('virnucpro.progress')


class ProgressReporter:
    """
    Wrapper for tqdm progress bars that integrates with logging.

    Ensures progress bars don't interfere with log messages and
    provides consistent styling across the application.
    """

    def __init__(self, disable: bool = False):
        """
        Initialize progress reporter.

        Args:
            disable: If True, disable all progress bars (for quiet mode or CI)
        """
        self.disable = disable

    def create_bar(
        self,
        iterable: Optional[Iterable] = None,
        total: Optional[int] = None,
        desc: Optional[str] = None,
        unit: str = 'it',
        leave: bool = True,
        **kwargs
    ) -> tqdm:
        """
        Create a tqdm progress bar.

        Args:
            iterable: Optional iterable to wrap
            total: Total number of iterations (if iterable is None)
            desc: Description prefix for progress bar
            unit: Unit name (default: 'it')
            leave: Keep progress bar after completion
            **kwargs: Additional tqdm arguments

        Returns:
            tqdm progress bar object

        Example:
            >>> reporter = ProgressReporter()
            >>> for item in reporter.create_bar(items, desc="Processing"):
            ...     process(item)
        """
        # Configure tqdm to work with logging
        return tqdm(
            iterable=iterable,
            total=total,
            desc=desc,
            unit=unit,
            leave=leave,
            disable=self.disable,
            file=sys.stdout,
            ncols=100,  # Fixed width for consistent display
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
            **kwargs
        )

    def create_file_bar(
        self,
        total_files: int,
        desc: str = "Processing files",
        **kwargs
    ) -> tqdm:
        """
        Create a progress bar specifically for file processing.

        Args:
            total_files: Total number of files to process
            desc: Description
            **kwargs: Additional tqdm arguments

        Returns:
            tqdm progress bar
        """
        return self.create_bar(
            total=total_files,
            desc=desc,
            unit='file',
            **kwargs
        )

    def create_sequence_bar(
        self,
        total_sequences: int,
        desc: str = "Processing sequences",
        **kwargs
    ) -> tqdm:
        """
        Create a progress bar for sequence processing.

        Args:
            total_sequences: Total number of sequences
            desc: Description
            **kwargs: Additional tqdm arguments

        Returns:
            tqdm progress bar
        """
        return self.create_bar(
            total=total_sequences,
            desc=desc,
            unit='seq',
            **kwargs
        )

    def create_stage_bar(
        self,
        stages: int = 1,
        desc: str = "Pipeline stages",
        **kwargs
    ) -> tqdm:
        """
        Create a progress bar for pipeline stages.

        Args:
            stages: Number of stages
            desc: Description
            **kwargs: Additional tqdm arguments

        Returns:
            tqdm progress bar
        """
        return self.create_bar(
            total=stages,
            desc=desc,
            unit='stage',
            **kwargs
        )

    @staticmethod
    def write_above_bar(message: str):
        """
        Write a message above the current progress bar.

        Uses tqdm.write() to ensure message appears above progress bar
        rather than interfering with it.

        Args:
            message: Message to display
        """
        tqdm.write(message)


# Example usage in pipeline stages
def example_usage():
    """Example of how to use ProgressReporter in pipeline code"""

    from pathlib import Path
    from Bio import SeqIO

    reporter = ProgressReporter()

    # Example 1: Processing files with progress
    files = [Path(f"file_{i}.fa") for i in range(100)]

    with reporter.create_file_bar(len(files), desc="Extracting features") as pbar:
        for file in files:
            # Process file
            process_file(file)
            pbar.update(1)

            # Log something while maintaining progress bar
            if file.stat().st_size > 1e6:
                reporter.write_above_bar(f"Large file detected: {file}")

    # Example 2: Processing sequences with nested progress
    fasta_file = Path("input.fasta")
    records = list(SeqIO.parse(fasta_file, 'fasta'))

    with reporter.create_sequence_bar(len(records), desc="Translating sequences") as pbar:
        for record in records:
            # Process sequence
            result = translate_sequence(record)
            pbar.update(1)

            # Update description with current sequence
            pbar.set_postfix_str(f"Current: {record.id[:20]}")

    # Example 3: Stage-level progress
    stages = ['Chunking', 'Translation', 'Features', 'Prediction']

    with reporter.create_stage_bar(len(stages), desc="Pipeline") as pbar:
        for stage in stages:
            pbar.set_description(f"Stage: {stage}")

            # Run stage
            run_stage(stage)

            pbar.update(1)


# Helper functions for common progress patterns
def process_with_progress(
    items: Iterable[Any],
    process_fn: callable,
    desc: str = "Processing",
    unit: str = "it",
    disable: bool = False
) -> list:
    """
    Process items with automatic progress bar.

    Args:
        items: Items to process
        process_fn: Function to apply to each item
        desc: Progress bar description
        unit: Unit name
        disable: Disable progress bar

    Returns:
        List of processed results
    """
    reporter = ProgressReporter(disable=disable)
    results = []

    with reporter.create_bar(items, desc=desc, unit=unit) as pbar:
        for item in pbar:
            result = process_fn(item)
            results.append(result)

    return results
```

**Integration with logging**: The progress bars use `tqdm.write()` which ensures log messages appear above the progress bar without disrupting it. Example integration:

```python
import logging
from virnucpro.utils.progress import ProgressReporter

logger = logging.getLogger('virnucpro.pipeline')
reporter = ProgressReporter()

# Progress bar with logging
with reporter.create_file_bar(100, desc="Processing") as pbar:
    for i in range(100):
        # Work
        process_item(i)

        # Log without disrupting progress bar
        if i % 10 == 0:
            logger.info(f"Checkpoint at item {i}")

        pbar.update(1)
```

### Success Criteria

#### Automated Verification:
- [ ] Package structure created: `test -d virnucpro/cli && test -d virnucpro/core && test -d virnucpro/pipeline && test -d virnucpro/utils`
- [ ] Can import package: `python -c "import virnucpro; print(virnucpro.__version__)"`
- [ ] Config loads successfully: `python -c "from virnucpro.core.config import Config; c = Config.load(); print(c.get('prediction.batch_size'))"`
- [ ] Logging works: `python -c "from virnucpro.core.logging_setup import setup_logging; setup_logging(verbose=True)"`
- [ ] Device validation works: `python -c "from virnucpro.core.device import validate_and_get_device; print(validate_and_get_device('cpu'))"`
- [ ] Progress reporter imports: `python -c "from virnucpro.utils.progress import ProgressReporter; p = ProgressReporter()"`

#### Manual Verification:
- [ ] Default config file is readable and well-documented
- [ ] Logging output is clear and informative
- [ ] Device selection logic handles edge cases (no GPU, invalid device ID)
- [ ] Progress bars display correctly without interfering with logs
- [ ] Directory structure follows Python package conventions

---

## Phase 2: Core Pipeline Refactoring

### Overview
Extract and modularize the prediction pipeline logic from `prediction.py` into well-organized modules while preserving all existing functionality.

### Changes Required

#### 2.1 Model Classes Module

**File**: `virnucpro/pipeline/models.py`

Extract classes from `prediction.py:20-71`:

```python
"""PyTorch model and dataset classes"""

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from typing import List
from pathlib import Path
import logging

logger = logging.getLogger('virnucpro.models')


class MLPClassifier(nn.Module):
    """
    Multi-layer perceptron classifier for viral sequence classification.

    Architecture: Input → Linear → BatchNorm → ReLU → Dropout → Linear → Output

    Based on prediction.py:48-71
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_class: int):
        """
        Initialize MLP classifier.

        Args:
            input_dim: Input feature dimension (default: 3328 for DNABERT-S + ESM-2)
            hidden_dim: Hidden layer dimension
            num_class: Number of output classes (2 for binary classification)
        """
        super(MLPClassifier, self).__init__()

        self.hidden_layer = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, num_class)
        self.dropout = nn.Dropout(0.5)

        self.init_weights()

    def init_weights(self):
        """Initialize weights using Xavier uniform initialization"""
        nn.init.xavier_uniform_(self.hidden_layer.weight)
        nn.init.xavier_uniform_(self.output_layer.weight)

    def forward(self, x):
        """Forward pass through the network"""
        x = self.hidden_layer(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.output_layer(x)
        return x


class PredictDataBatchDataset(Dataset):
    """
    PyTorch Dataset for loading merged feature tensors.

    Loads multiple .pt files containing merged DNABERT-S + ESM-2 features
    and provides index-based access across all files.

    Based on prediction.py:20-45
    """

    def __init__(self, file_list: List[Path]):
        """
        Initialize dataset from list of merged feature files.

        Args:
            file_list: List of paths to .pt files containing merged features
        """
        self.file_list = [Path(f) for f in file_list]
        self.ids = []
        self.data = []
        self._load_all_data()

        logger.info(f"Loaded {len(self)} sequences from {len(self.file_list)} files")

    def _load_all_data(self):
        """Load all data from .pt files into memory"""
        for file_path in self.file_list:
            logger.debug(f"Loading {file_path}")
            data_dict = torch.load(file_path)
            data = data_dict['data']
            self.data.append(data)
            ids = data_dict['ids']
            self.ids.extend(ids)

    def __len__(self):
        """Return total number of sequences across all files"""
        return sum(d.size(0) for d in self.data)

    def __getitem__(self, idx):
        """
        Get item by index across all loaded files.

        Args:
            idx: Global index

        Returns:
            Tuple of (data_tensor, sequence_id)
        """
        cumulative_size = 0
        for data in self.data:
            if cumulative_size + data.size(0) > idx:
                index_in_file = idx - cumulative_size
                return data[index_in_file], self.ids[cumulative_size + index_in_file]
            cumulative_size += data.size(0)
        raise IndexError("Index out of range")
```

#### 2.2 Sequence Utilities Module

**File**: `virnucpro/utils/sequence.py`

Extract sequence processing functions from `units.py`:

```python
"""Sequence processing utilities"""

from Bio import SeqIO
from Bio.Seq import Seq
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import logging

logger = logging.getLogger('virnucpro.sequence')


# Genetic code codon table
CODON_TABLE = {
    'TCA': 'S', 'TCC': 'S', 'TCG': 'S', 'TCT': 'S',
    'TTC': 'F', 'TTT': 'F', 'TTA': 'L', 'TTG': 'L',
    'TAC': 'Y', 'TAT': 'Y', 'TAA': '*', 'TAG': '*',
    'TGC': 'C', 'TGT': 'C', 'TGA': '*', 'TGG': 'W',
    'CTA': 'L', 'CTC': 'L', 'CTG': 'L', 'CTT': 'L',
    'CCA': 'P', 'CCC': 'P', 'CCG': 'P', 'CCT': 'P',
    'CAC': 'H', 'CAT': 'H', 'CAA': 'Q', 'CAG': 'Q',
    'CGA': 'R', 'CGC': 'R', 'CGG': 'R', 'CGT': 'R',
    'ATA': 'I', 'ATC': 'I', 'ATT': 'I', 'ATG': 'M',
    'ACA': 'T', 'ACC': 'T', 'ACG': 'T', 'ACT': 'T',
    'AAC': 'N', 'AAT': 'N', 'AAA': 'K', 'AAG': 'K',
    'AGC': 'S', 'AGT': 'S', 'AGA': 'R', 'AGG': 'R',
    'GTA': 'V', 'GTC': 'V', 'GTG': 'V', 'GTT': 'V',
    'GCA': 'A', 'GCC': 'A', 'GCG': 'A', 'GCT': 'A',
    'GAC': 'D', 'GAT': 'D', 'GAA': 'E', 'GAG': 'E',
    'GGA': 'G', 'GGC': 'G', 'GGG': 'G', 'GGT': 'G',
}


def reverse_complement(sequence: str) -> str:
    """
    Generate reverse complement of DNA sequence.

    Based on units.py:50-52

    Args:
        sequence: DNA sequence string

    Returns:
        Reverse complement sequence
    """
    complement = str.maketrans('ACGTacgt', 'TGCAtgca')
    return sequence.translate(complement)[::-1]


def translate_dna(sequence: str) -> List[str]:
    """
    Translate DNA sequence in all 6 reading frames.

    Returns 6 protein sequences: 3 forward frames + 3 reverse complement frames.

    Based on units.py:54-73

    Args:
        sequence: DNA sequence string

    Returns:
        List of 6 protein sequences
    """
    def translate_frame(seq: str, frame: int) -> str:
        """Translate a single reading frame"""
        return ''.join(
            CODON_TABLE.get(seq[i:i+3], '')
            for i in range(frame, len(seq) - 2, 3)
        )

    # Translate forward strand (3 frames)
    frames = [translate_frame(sequence, frame) for frame in range(3)]

    # Translate reverse complement (3 frames)
    rev_comp_sequence = reverse_complement(sequence)
    frames += [translate_frame(rev_comp_sequence, frame) for frame in range(3)]

    return frames


def identify_seq(seqid: str, sequence: str) -> List[Dict[str, str]]:
    """
    Identify valid protein-coding regions from six-frame translation.

    Translates sequence in all 6 frames and returns valid ORFs
    (those without stop codons).

    Based on units.py:81-146 (prediction mode, istraindata=False)

    Args:
        seqid: Sequence identifier
        sequence: DNA sequence string

    Returns:
        List of dictionaries with keys: 'seqid', 'nucleotide', 'protein'
        Each dict represents a valid ORF with frame indicator (F1-F3, R1-R3)
    """
    final_list = []
    proteins_list = translate_dna(sequence)

    for num, protein in enumerate(proteins_list, start=1):
        # Skip frames with stop codons
        if "*" in protein:
            continue

        # Calculate sequence lengths
        pro_len = len(protein)
        seq_len = pro_len * 3

        # Determine strand
        strand = "+" if num <= 3 else "-"

        if strand == "+":
            # Forward strand
            selected_seq = sequence[num-1:num-1 + seq_len]
            seqid_final = seqid + 'F' + str(num)
        else:
            # Reverse strand
            selected_seq = reverse_complement(sequence)[num-3-1:num-3-1 + seq_len]
            seqid_final = seqid + 'R' + str(num - 3)

        final_list.append({
            'seqid': seqid_final,
            'nucleotide': selected_seq,
            'protein': protein
        })

    return final_list


def split_fasta_chunk(input_file: Path, output_file: Path, chunk_size: int):
    """
    Split FASTA sequences into overlapping chunks of specified size.

    Uses overlapping strategy to ensure all sequence information is captured
    in fixed-size chunks.

    Based on units.py:9-36

    Args:
        input_file: Input FASTA file path
        output_file: Output FASTA file path
        chunk_size: Target size for each chunk
    """
    logger.info(f"Chunking sequences to {chunk_size}bp: {input_file} → {output_file}")

    total_sequences = 0
    total_chunks = 0

    with open(output_file, 'w') as out_handle:
        for record in SeqIO.parse(input_file, 'fasta'):
            sequence = record.seq
            seq_length = len(sequence)

            # Calculate number of chunks needed
            num_chunks = -(-seq_length // chunk_size)  # Ceiling division

            # Calculate overlap distribution
            total_chunk_length = num_chunks * chunk_size
            repeat_length = total_chunk_length - seq_length
            repeat_region = repeat_length / num_chunks
            lower_int = int(repeat_region)
            upper_int = lower_int + 1

            # Distribute overlap across chunks
            low_up_numbers = [lower_int] * (num_chunks - 1)
            total_low_up_numbers = sum(low_up_numbers)
            need_to_add_up_nums = repeat_length - total_low_up_numbers
            final_low_up_numbers = (
                [upper_int] * need_to_add_up_nums +
                [lower_int] * (num_chunks - 1 - need_to_add_up_nums) +
                [0]
            )

            # Generate chunks
            move_step = 0
            for a, b in zip(range(0, seq_length, chunk_size), final_low_up_numbers):
                if a > 1:
                    chunk = record[a-move_step:a-move_step + chunk_size]
                else:
                    chunk = record[a:a + chunk_size]

                new_record = chunk
                new_record.id = f"{record.id}_chunk_{a // chunk_size + 1}"
                new_record.description = ""

                SeqIO.write(new_record, out_handle, 'fasta')
                move_step += b
                total_chunks += 1

            total_sequences += 1

    logger.info(f"Created {total_chunks} chunks from {total_sequences} sequences")
```

**Continue in next phase...**

### Success Criteria

#### Automated Verification:
- [ ] Models module imports successfully: `python -c "from virnucpro.pipeline.models import MLPClassifier, PredictDataBatchDataset"`
- [ ] Sequence utils import successfully: `python -c "from virnucpro.utils.sequence import translate_dna, identify_seq"`
- [ ] Reverse complement works: `python -c "from virnucpro.utils.sequence import reverse_complement; assert reverse_complement('ATCG') == 'CGAT'"`
- [ ] Translation produces 6 frames: `python -c "from virnucpro.utils.sequence import translate_dna; assert len(translate_dna('ATGATGATGATG')) == 6"`

#### Manual Verification:
- [ ] Refactored code matches original behavior exactly
- [ ] All functions have proper docstrings
- [ ] Logging is added at appropriate points
- [ ] No circular imports between modules

**Implementation Note**: After completing this phase and all automated verification passes, pause here for manual confirmation that the refactored modules work correctly before proceeding to Phase 3.

---

## Phase 3: CLI Implementation with Click

### Overview
Implement the Click-based CLI with the `predict` command, device selection, configuration support, and validation.

### Changes Required

#### 3.1 Main CLI Group

**File**: `virnucpro/cli/main.py`

```python
"""Main Click CLI group and global options"""

import click
import sys
from pathlib import Path
from virnucpro.core.logging_setup import setup_logging
from virnucpro.core.config import Config
from virnucpro import __version__

# Import command modules
from virnucpro.cli import predict
from virnucpro.cli import utils


@click.group(context_settings={'help_option_names': ['-h', '--help']})
@click.version_option(version=__version__, prog_name='VirNucPro')
@click.option('--verbose', '-v', is_flag=True,
              help='Enable verbose (DEBUG level) logging')
@click.option('--quiet', '-q', is_flag=True,
              help='Suppress console output (errors only)')
@click.option('--log-file', '-l', type=click.Path(),
              help='Path to log file')
@click.option('--config', '-c', type=click.Path(exists=True),
              help='Path to YAML configuration file')
@click.pass_context
def cli(ctx, verbose, quiet, log_file, config):
    """
    VirNucPro - Viral Nucleotide and Protein Identifier

    A production-ready tool for identifying viral sequences using
    six-frame translation and large language models (DNABERT-S and ESM-2).

    Examples:

      # Basic prediction with 500bp model
      python -m virnucpro predict input.fasta

      # Use 300bp model with custom output
      python -m virnucpro predict input.fasta --model-type 300 -o results/

      # Use specific GPU and resume from checkpoint
      python -m virnucpro predict input.fasta --device cuda:1 --resume

      # List available compute devices
      python -m virnucpro utils list-devices

    For detailed help on a command:
      python -m virnucpro COMMAND --help
    """
    # Initialize context object
    ctx.ensure_object(dict)

    # Setup logging
    log_file_path = Path(log_file) if log_file else None
    logger = setup_logging(verbose=verbose, log_file=log_file_path, quiet=quiet)
    ctx.obj['logger'] = logger

    # Load configuration
    try:
        if config:
            cfg = Config.load(Path(config))
            logger.info(f"Loaded configuration from {config}")
        else:
            cfg = Config.load()  # Load default
            logger.debug("Using default configuration")

        ctx.obj['config'] = cfg
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)


# Register commands
cli.add_command(predict.predict)
cli.add_command(utils.utils)


if __name__ == '__main__':
    cli()
```

#### 3.2 Predict Command

**File**: `virnucpro/cli/predict.py`

```python
"""Predict command implementation"""

import click
import sys
from pathlib import Path
import logging

from virnucpro.core.device import validate_and_get_device
from virnucpro.core.config import Config
from virnucpro.pipeline.prediction import run_prediction

logger = logging.getLogger('virnucpro.cli.predict')


@click.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.option('--model-type', '-m',
              type=click.Choice(['300', '500', 'custom']),
              default='500',
              help='Model type to use (300bp, 500bp, or custom)')
@click.option('--model-path', '-p',
              type=click.Path(exists=True),
              help='Path to custom model file (required if model-type is custom)')
@click.option('--expected-length', '-e',
              type=int,
              help='Expected sequence length (default: matches model type)')
@click.option('--output-dir', '-o',
              type=click.Path(),
              help='Output directory for results (default: {input}_predictions)')
@click.option('--device', '-d',
              type=str,
              default='auto',
              help='Device: "auto", "cpu", "cuda", "cuda:N", or "N" (default: auto)')
@click.option('--batch-size', '-b',
              type=int,
              help='Batch size for prediction (default: from config)')
@click.option('--num-workers', '-w',
              type=int,
              help='Number of data loading workers (default: from config)')
@click.option('--keep-intermediate', '-k',
              is_flag=True,
              help='Keep intermediate files after completion')
@click.option('--resume',
              is_flag=True,
              help='Resume from checkpoint if available')
@click.option('--force', '-f',
              is_flag=True,
              help='Overwrite existing output directory')
@click.option('--no-progress',
              is_flag=True,
              help='Disable progress bars (useful for logging to files)')
@click.pass_context
def predict(ctx, input_file, model_type, model_path, expected_length,
            output_dir, device, batch_size, num_workers,
            keep_intermediate, resume, force, no_progress):
    """
    Predict viral sequences from FASTA input.

    This command processes input sequences through the VirNucPro pipeline:

      1. Chunk sequences to expected length
      2. Six-frame translation to identify ORFs
      3. Extract features using DNABERT-S (DNA) and ESM-2 (protein)
      4. Merge features and predict using MLP classifier
      5. Generate consensus predictions across reading frames

    Examples:

      # Basic prediction
      python -m virnucpro predict sequences.fasta

      # Use 300bp model with GPU 1
      python -m virnucpro predict sequences.fasta -m 300 -d cuda:1

      # Resume interrupted run
      python -m virnucpro predict sequences.fasta --resume

      # Custom model with CPU
      python -m virnucpro predict sequences.fasta -m custom -p my_model.pth -d cpu
    """
    logger = ctx.obj['logger']
    config = ctx.obj['config']

    logger.info(f"VirNucPro Prediction - Input: {input_file}")

    # Validate and prepare parameters
    try:
        # Validate model parameters
        if model_type == 'custom' and not model_path:
            raise click.BadParameter(
                "--model-path is required when using --model-type custom"
            )

        # Set defaults from config
        if not expected_length:
            expected_length = 300 if model_type == '300' else 500

        if not model_path:
            model_path = config.get(f'prediction.models.{model_type}')
            if not model_path or not Path(model_path).exists():
                raise click.FileError(
                    model_path or f"{model_type}_model.pth",
                    "Model file not found. Ensure model file exists in project root."
                )

        # Set output directory
        if not output_dir:
            input_base = Path(input_file).stem
            output_dir = f"{input_base}_predictions"
        output_dir = Path(output_dir)

        # Check output directory
        if output_dir.exists() and not force and not resume:
            if not click.confirm(f"Output directory {output_dir} exists. Overwrite?"):
                logger.info("Prediction cancelled by user")
                sys.exit(0)

        # Get batch size and workers from config if not specified
        if batch_size is None:
            batch_size = config.get('prediction.batch_size', 256)
        if num_workers is None:
            num_workers = config.get('prediction.num_workers', 4)

        # Validate and get device
        fallback_to_cpu = config.get('device.fallback_to_cpu', True)
        device_obj = validate_and_get_device(device, fallback_to_cpu=fallback_to_cpu)

        # Determine cleanup behavior
        auto_cleanup = config.get('files.auto_cleanup', True)
        cleanup = auto_cleanup and not keep_intermediate

        # Log configuration
        logger.info("Configuration:")
        logger.info(f"  Model: {model_path}")
        logger.info(f"  Expected length: {expected_length}bp")
        logger.info(f"  Output directory: {output_dir}")
        logger.info(f"  Device: {device_obj}")
        logger.info(f"  Batch size: {batch_size}")
        logger.info(f"  Workers: {num_workers}")
        logger.info(f"  Resume: {resume}")
        logger.info(f"  Cleanup intermediate files: {cleanup}")
        logger.info(f"  Progress bars: {'disabled' if no_progress else 'enabled'}")

        # Run prediction pipeline
        run_prediction(
            input_file=Path(input_file),
            model_path=Path(model_path),
            expected_length=expected_length,
            output_dir=output_dir,
            device=device_obj,
            batch_size=batch_size,
            num_workers=num_workers,
            cleanup_intermediate=cleanup,
            resume=resume,
            show_progress=not no_progress,
            config=config
        )

        logger.info("Prediction completed successfully!")
        logger.info(f"Results saved to {output_dir}/")

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        if ctx.obj['logger'].level == logging.DEBUG:
            logger.exception("Detailed error traceback:")
        sys.exit(1)
```

#### 3.3 Utility Commands

**File**: `virnucpro/cli/utils.py`

```python
"""Utility CLI commands"""

import click
import sys
from pathlib import Path
import logging

from virnucpro.core.device import list_available_devices
from virnucpro.utils.validation import validate_fasta_file
from virnucpro.core.config import Config

logger = logging.getLogger('virnucpro.cli.utils')


@click.group(name='utils')
def utils():
    """Utility commands for VirNucPro"""
    pass


@utils.command(name='list-devices')
def list_devices_cmd():
    """
    List available compute devices.

    Shows CPU and GPU information including device names,
    memory, and availability status.
    """
    list_available_devices()


@utils.command(name='validate')
@click.argument('input_file', type=click.Path(exists=True))
@click.option('--max-errors', '-e', type=int, default=10,
              help='Maximum errors to display')
def validate_cmd(input_file, max_errors):
    """
    Validate FASTA input file.

    Checks for common issues:
      - Duplicate sequence IDs
      - Invalid characters in sequences
      - Ambiguous bases
      - Empty sequences

    Example:
      python -m virnucpro utils validate sequences.fasta
    """
    logger.info(f"Validating {input_file}")

    try:
        is_valid, errors, warnings, stats = validate_fasta_file(
            Path(input_file),
            max_errors=max_errors
        )

        # Print statistics
        print(f"\nFile Statistics:")
        print(f"  Total sequences: {stats['total_sequences']}")
        print(f"  Length range: {stats['min_length']}-{stats['max_length']} bp")
        print(f"  Average length: {stats['avg_length']:.1f} bp")
        print(f"  Sequences with ambiguous bases: {stats['ambiguous_count']}")
        print(f"  Duplicate IDs: {len(stats['duplicate_ids'])}")

        # Print errors
        if errors:
            print(f"\nErrors found ({len(errors)}):")
            for error in errors[:max_errors]:
                print(f"  ✗ {error}")
            if len(errors) > max_errors:
                print(f"  ... and {len(errors) - max_errors} more")

        # Print warnings
        if warnings:
            print(f"\nWarnings ({len(warnings)}):")
            for warning in warnings[:max_errors]:
                print(f"  ⚠ {warning}")
            if len(warnings) > max_errors:
                print(f"  ... and {len(warnings) - max_errors} more")

        if is_valid:
            print("\n✓ File is valid and ready for processing")
            sys.exit(0)
        else:
            print("\n✗ File validation failed")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Validation failed: {e}")
        sys.exit(1)


@utils.command(name='generate-config')
@click.option('--output', '-o', type=click.Path(),
              default='virnucpro_config.yaml',
              help='Output configuration file path')
def generate_config_cmd(output):
    """
    Generate a configuration file template.

    Creates a YAML configuration file with default values
    that can be customized and used with --config.

    Example:
      python -m virnucpro utils generate-config -o my_config.yaml
      python -m virnucpro predict input.fasta --config my_config.yaml
    """
    try:
        # Load default config
        config = Config.load()

        # Save to specified location
        output_path = Path(output)
        config.save(output_path)

        print(f"Configuration template saved to {output_path}")
        print(f"\nEdit this file to customize VirNucPro settings,")
        print(f"then use it with: --config {output_path}")

    except Exception as e:
        logger.error(f"Failed to generate config: {e}")
        sys.exit(1)
```

#### 3.4 Input Validation Module

**File**: `virnucpro/utils/validation.py`

```python
"""Input validation utilities"""

from Bio import SeqIO
from pathlib import Path
from typing import Dict, List, Tuple, Set
import re
import logging

logger = logging.getLogger('virnucpro.validation')


def validate_fasta_file(
    fasta_file: Path,
    max_errors: int = 10
) -> Tuple[bool, List[str], List[str], Dict]:
    """
    Validate FASTA file for common issues.

    Args:
        fasta_file: Path to FASTA file
        max_errors: Maximum errors to collect

    Returns:
        Tuple of (is_valid, errors, warnings, statistics)
    """
    errors = []
    warnings = []

    stats = {
        'total_sequences': 0,
        'min_length': float('inf'),
        'max_length': 0,
        'avg_length': 0,
        'ambiguous_count': 0,
        'duplicate_ids': set(),
        'empty_sequences': 0
    }

    seen_ids = set()
    total_length = 0
    ambiguous_bases = {'N', 'R', 'Y', 'M', 'K', 'S', 'W', 'H', 'B', 'V', 'D'}
    valid_bases = re.compile(r'^[ATGCNRYKMSWBDHV]+$', re.IGNORECASE)

    try:
        for record in SeqIO.parse(fasta_file, 'fasta'):
            stats['total_sequences'] += 1

            # Check for duplicate IDs
            if record.id in seen_ids:
                stats['duplicate_ids'].add(record.id)
                if len(errors) < max_errors:
                    errors.append(f"Duplicate ID: {record.id}")
            seen_ids.add(record.id)

            # Check sequence length
            seq_len = len(record.seq)
            if seq_len == 0:
                stats['empty_sequences'] += 1
                if len(errors) < max_errors:
                    errors.append(f"Empty sequence: {record.id}")
                continue

            stats['min_length'] = min(stats['min_length'], seq_len)
            stats['max_length'] = max(stats['max_length'], seq_len)
            total_length += seq_len

            # Check for ambiguous bases
            seq_str = str(record.seq).upper()
            if any(base in seq_str for base in ambiguous_bases):
                stats['ambiguous_count'] += 1
                if len(warnings) < max_errors:
                    warnings.append(f"Ambiguous bases in {record.id}")

            # Check for invalid characters
            if not valid_bases.match(seq_str):
                if len(errors) < max_errors:
                    invalid_chars = set(seq_str) - set('ATGCNRYKMSWBDHV')
                    errors.append(
                        f"Invalid characters in {record.id}: {invalid_chars}"
                    )

        # Calculate average length
        if stats['total_sequences'] > 0:
            stats['avg_length'] = total_length / stats['total_sequences']

        # Determine if valid
        is_valid = len(errors) == 0 and stats['total_sequences'] > 0

        return is_valid, errors, warnings, stats

    except Exception as e:
        logger.error(f"Failed to parse FASTA file: {e}")
        errors.append(f"File parsing error: {e}")
        return False, errors, warnings, stats
```

### Success Criteria

#### Automated Verification:
- [ ] CLI imports successfully: `python -c "from virnucpro.cli.main import cli"`
- [ ] Help text displays: `python -m virnucpro --help`
- [ ] Predict help displays: `python -m virnucpro predict --help`
- [ ] Utils help displays: `python -m virnucpro utils --help`
- [ ] List devices works: `python -m virnucpro utils list-devices`
- [ ] Generate config works: `python -m virnucpro utils generate-config -o test_config.yaml && test -f test_config.yaml`

#### Manual Verification:
- [ ] CLI help text is clear and complete
- [ ] All options are properly documented
- [ ] Device selection shows appropriate error messages
- [ ] Config generation produces valid YAML
- [ ] Validation command provides useful feedback

**Implementation Note**: After completing this phase and all automated verification passes, test the CLI with a small sample file to ensure the interface works as expected before proceeding to Phase 4.

---

## Phase 4: Checkpointing System

### Overview
Implement hash-based checkpointing with stage and file-level tracking to enable resume from failures.

### Changes Required

#### 4.1 Checkpoint Manager

**File**: `virnucpro/core/checkpoint.py`

```python
"""Checkpointing system for pipeline resume capability"""

import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum
import logging

logger = logging.getLogger('virnucpro.checkpoint')


class StageStatus(Enum):
    """Pipeline stage status"""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class PipelineStage(Enum):
    """Pipeline stages for prediction"""
    CHUNKING = "chunking"
    TRANSLATION = "translation"
    NUCLEOTIDE_SPLITTING = "nucleotide_splitting"
    PROTEIN_SPLITTING = "protein_splitting"
    NUCLEOTIDE_FEATURES = "nucleotide_features"
    PROTEIN_FEATURES = "protein_features"
    FEATURE_MERGING = "feature_merging"
    PREDICTION = "prediction"
    CONSENSUS = "consensus"


class CheckpointManager:
    """
    Manages pipeline checkpoints for resume capability.

    Implements hash-based validation of inputs and parameters
    to detect when cached results can be reused.
    """

    def __init__(self, checkpoint_dir: Path, pipeline_config: Dict):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to store checkpoint data
            pipeline_config: Configuration dictionary for validation
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.pipeline_config = pipeline_config
        self.config_hash = self._compute_config_hash(pipeline_config)
        self.state_file = self.checkpoint_dir / "pipeline_state.json"

        logger.debug(f"Checkpoint manager initialized: {self.checkpoint_dir}")

    def _compute_config_hash(self, config: Dict) -> str:
        """Compute SHA256 hash of configuration"""
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()

    def _compute_file_checksum(self, filepath: Path) -> str:
        """Compute SHA256 checksum of a file"""
        sha256_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return f"sha256:{sha256_hash.hexdigest()}"

    def load_state(self) -> Dict:
        """
        Load pipeline state from checkpoint.

        Returns:
            State dictionary

        Raises:
            ValueError: If checkpoint config doesn't match current config
        """
        if self.state_file.exists():
            logger.info(f"Loading checkpoint from {self.state_file}")

            with open(self.state_file, 'r') as f:
                state = json.load(f)

            # Validate config compatibility
            checkpoint_hash = state.get('pipeline_config', {}).get('config_hash')
            if checkpoint_hash != self.config_hash:
                logger.warning(
                    f"Configuration changed since checkpoint was created.\n"
                    f"Checkpoint will be ignored and pipeline will run from scratch."
                )
                return self._create_initial_state()

            logger.info("Checkpoint loaded successfully")
            return state
        else:
            logger.debug("No checkpoint found, creating new state")
            return self._create_initial_state()

    def _create_initial_state(self) -> Dict:
        """Create initial pipeline state"""
        return {
            "created_at": datetime.utcnow().isoformat(),
            "pipeline_config": {
                "config_hash": self.config_hash,
                "expected_length": self.pipeline_config.get('expected_length'),
                "model_path": str(self.pipeline_config.get('model_path'))
            },
            "stages": {
                stage.value: {
                    "status": StageStatus.NOT_STARTED.value,
                    "started_at": None,
                    "completed_at": None,
                    "duration_seconds": None,
                    "inputs": {},
                    "outputs": {},
                    "error": None
                }
                for stage in PipelineStage
            }
        }

    def save_state(self, state: Dict):
        """Save pipeline state to checkpoint"""
        state['updated_at'] = datetime.utcnow().isoformat()

        # Save to temp file first, then atomic rename
        temp_file = self.state_file.with_suffix('.tmp')
        with open(temp_file, 'w') as f:
            json.dump(state, f, indent=2)

        temp_file.replace(self.state_file)
        logger.debug("Checkpoint saved")

    def find_resume_stage(self, state: Dict) -> Optional[PipelineStage]:
        """
        Determine which stage to resume from.

        Args:
            state: Current pipeline state

        Returns:
            Stage to resume from, or None if all complete
        """
        stages = list(PipelineStage)

        for stage in stages:
            stage_state = state['stages'][stage.value]
            status = StageStatus(stage_state['status'])

            # Resume from failed or in-progress stages
            if status in [StageStatus.FAILED, StageStatus.IN_PROGRESS]:
                logger.info(f"Resuming from stage: {stage.value} (status: {status.value})")
                return stage

            # Resume from first not-started stage
            if status == StageStatus.NOT_STARTED:
                logger.info(f"Starting from stage: {stage.value}")
                return stage

        # All stages completed
        logger.info("All stages already completed")
        return None

    def mark_stage_started(self, state: Dict, stage: PipelineStage):
        """Mark a stage as started"""
        stage_state = state['stages'][stage.value]
        stage_state['status'] = StageStatus.IN_PROGRESS.value
        stage_state['started_at'] = datetime.utcnow().isoformat()
        self.save_state(state)

        logger.info(f"Stage started: {stage.value}")

    def mark_stage_completed(
        self,
        state: Dict,
        stage: PipelineStage,
        outputs: Dict[str, Any]
    ):
        """
        Mark a stage as completed.

        Args:
            state: Pipeline state
            stage: Completed stage
            outputs: Output files/data from stage
        """
        stage_state = state['stages'][stage.value]
        started_at = stage_state.get('started_at')

        stage_state['status'] = StageStatus.COMPLETED.value
        stage_state['completed_at'] = datetime.utcnow().isoformat()

        # Calculate duration
        if started_at:
            start_dt = datetime.fromisoformat(started_at)
            end_dt = datetime.utcnow()
            duration = (end_dt - start_dt).total_seconds()
            stage_state['duration_seconds'] = duration

        # Save outputs
        stage_state['outputs'] = outputs

        self.save_state(state)

        logger.info(f"Stage completed: {stage.value}")
        if stage_state.get('duration_seconds'):
            logger.info(f"  Duration: {stage_state['duration_seconds']:.1f}s")

    def mark_stage_failed(self, state: Dict, stage: PipelineStage, error: str):
        """Mark a stage as failed"""
        stage_state = state['stages'][stage.value]
        stage_state['status'] = StageStatus.FAILED.value
        stage_state['failed_at'] = datetime.utcnow().isoformat()
        stage_state['error'] = error

        self.save_state(state)

        logger.error(f"Stage failed: {stage.value} - {error}")

    def can_skip_stage(self, state: Dict, stage: PipelineStage) -> bool:
        """
        Determine if a stage can be skipped based on checkpoint.

        Args:
            state: Pipeline state
            stage: Stage to check

        Returns:
            True if stage can be skipped
        """
        stage_state = state['stages'][stage.value]
        status = StageStatus(stage_state['status'])

        # Can only skip completed stages
        if status != StageStatus.COMPLETED:
            return False

        # Validate outputs still exist
        outputs = stage_state.get('outputs', {})
        if 'files' in outputs:
            for file_path in outputs['files']:
                if not Path(file_path).exists():
                    logger.warning(
                        f"Output file missing for {stage.value}: {file_path}"
                    )
                    return False

        logger.info(f"Skipping completed stage: {stage.value}")
        return True


class FileProgressTracker:
    """
    Track processing progress for individual files within a stage.

    Enables resume when only some files in a batch have been processed.
    """

    def __init__(self, stage_name: str, checkpoint_dir: Path):
        """
        Initialize file progress tracker.

        Args:
            stage_name: Name of the pipeline stage
            checkpoint_dir: Checkpoint directory
        """
        self.stage_name = stage_name
        self.progress_file = checkpoint_dir / f"{stage_name}_files.json"

    def load_progress(self, input_files: List[Path]) -> Dict:
        """
        Load or initialize file processing progress.

        Args:
            input_files: List of files to process

        Returns:
            Progress dictionary
        """
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                progress = json.load(f)
            logger.debug(f"Loaded file progress for {self.stage_name}")
        else:
            progress = {
                'stage': self.stage_name,
                'total_files': len(input_files),
                'files': {}
            }

            for filepath in input_files:
                progress['files'][str(filepath)] = {
                    'status': 'pending',
                    'started_at': None,
                    'completed_at': None,
                    'output': None,
                    'error': None
                }

            self._save_progress(progress)

        return progress

    def _save_progress(self, progress: Dict):
        """Save file progress to disk"""
        with open(self.progress_file, 'w') as f:
            json.dump(progress, f, indent=2)

    def get_pending_files(self, progress: Dict) -> List[str]:
        """Get list of files that still need processing"""
        return [
            filepath
            for filepath, info in progress['files'].items()
            if info['status'] in ['pending', 'failed']
        ]

    def mark_file_completed(self, progress: Dict, filepath: str, output: str):
        """Mark a file as successfully processed"""
        progress['files'][filepath]['status'] = 'completed'
        progress['files'][filepath]['completed_at'] = datetime.utcnow().isoformat()
        progress['files'][filepath]['output'] = output
        self._save_progress(progress)

    def mark_file_failed(self, progress: Dict, filepath: str, error: str):
        """Mark a file as failed"""
        progress['files'][filepath]['status'] = 'failed'
        progress['files'][filepath]['error'] = error
        self._save_progress(progress)

    def get_summary(self, progress: Dict) -> Dict:
        """Get processing progress summary"""
        statuses = [f['status'] for f in progress['files'].values()]
        total = len(statuses)
        completed = statuses.count('completed')

        return {
            'total': total,
            'completed': completed,
            'pending': statuses.count('pending'),
            'failed': statuses.count('failed'),
            'percentage': (completed / total * 100) if total > 0 else 0
        }
```

#### 4.2 Integration with Prediction Pipeline

**File**: `virnucpro/pipeline/prediction.py` (skeleton - main orchestration)

```python
"""Main prediction pipeline orchestration with checkpointing"""

from pathlib import Path
from typing import Optional
import logging
import torch

from virnucpro.core.checkpoint import CheckpointManager, PipelineStage
from virnucpro.core.config import Config
# Import other pipeline components as they're refactored

logger = logging.getLogger('virnucpro.pipeline.prediction')


def run_prediction(
    input_file: Path,
    model_path: Path,
    expected_length: int,
    output_dir: Path,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    cleanup_intermediate: bool,
    resume: bool,
    show_progress: bool,
    config: Config
):
    """
    Main prediction pipeline orchestration.

    Args:
        input_file: Input FASTA file
        model_path: Path to trained model
        expected_length: Expected sequence length
        output_dir: Output directory
        device: PyTorch device
        batch_size: Batch size for DataLoader
        num_workers: Number of data loading workers
        cleanup_intermediate: Whether to clean intermediate files
        resume: Whether to resume from checkpoint
        show_progress: Whether to show progress bars
        config: Configuration object
    """
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize progress reporter
    from virnucpro.utils.progress import ProgressReporter
    progress = ProgressReporter(disable=not show_progress)

    # Initialize checkpointing
    checkpoint_dir = output_dir / config.get('checkpointing.checkpoint_dir', '.checkpoints')

    pipeline_config = {
        'expected_length': expected_length,
        'model_path': str(model_path),
        'batch_size': batch_size,
        'num_workers': num_workers
    }

    checkpoint_manager = CheckpointManager(checkpoint_dir, pipeline_config)

    # Load state (or create new)
    if resume:
        state = checkpoint_manager.load_state()
        start_stage = checkpoint_manager.find_resume_stage(state)

        if start_stage is None:
            logger.info("All stages already completed!")
            return
    else:
        state = checkpoint_manager._create_initial_state()
        start_stage = PipelineStage.CHUNKING

    # Define intermediate paths
    chunked_file = output_dir / f"{input_file.stem}_chunked{expected_length}.fa"
    nucleotide_file = output_dir / f"{input_file.stem}_nucleotide.fa"
    protein_file = output_dir / f"{input_file.stem}_protein.faa"

    try:
        # Stage 1: Chunking
        if start_stage == PipelineStage.CHUNKING or not checkpoint_manager.can_skip_stage(state, PipelineStage.CHUNKING):
            logger.info("=== Stage 1: Sequence Chunking ===")
            checkpoint_manager.mark_stage_started(state, PipelineStage.CHUNKING)

            # Chunk sequences with progress bar
            from virnucpro.utils.sequence import split_fasta_chunk
            from Bio import SeqIO

            # Count sequences for progress bar
            num_sequences = sum(1 for _ in SeqIO.parse(input_file, 'fasta'))

            # Show progress during chunking
            with progress.create_sequence_bar(num_sequences, desc="Chunking sequences") as pbar:
                # split_fasta_chunk would be modified to accept progress callback
                split_fasta_chunk(input_file, chunked_file, expected_length)
                pbar.update(num_sequences)

            checkpoint_manager.mark_stage_completed(
                state,
                PipelineStage.CHUNKING,
                {'files': [str(chunked_file)]}
            )

        # Stage 2: Translation
        # Example with file-level progress:
        # with progress.create_file_bar(len(files), desc="Extracting features") as pbar:
        #     for file in files:
        #         extract_features(file)
        #         pbar.update(1)
        #         pbar.set_postfix_str(f"Current: {file.name}")

        # TODO: Implement remaining stages with checkpointing and progress bars

        # Final: Cleanup if requested
        if cleanup_intermediate:
            logger.info("Cleaning up intermediate files...")
            # TODO: Implement cleanup logic

        logger.info("Pipeline completed successfully!")

    except Exception as e:
        # Mark current stage as failed
        # (determine current stage from state)
        logger.exception("Pipeline failed")
        raise
```

### Success Criteria

#### Automated Verification:
- [ ] Checkpoint module imports: `python -c "from virnucpro.core.checkpoint import CheckpointManager, PipelineStage"`
- [ ] Can create checkpoint: `python -c "from virnucpro.core.checkpoint import CheckpointManager; import tempfile; from pathlib import Path; cm = CheckpointManager(Path(tempfile.mkdtemp()), {})"`
- [ ] State persistence works: Create checkpoint, save state, load state, verify match

#### Manual Verification:
- [ ] Checkpoint files are created in expected location
- [ ] State JSON is valid and human-readable
- [ ] Resume correctly skips completed stages
- [ ] Failed stages can be retried
- [ ] Configuration changes invalidate checkpoint

**Implementation Note**: After completing this phase, test the resume functionality by manually interrupting a prediction run and resuming it to verify checkpointing works correctly.

---

## Phase 5: Testing & Documentation

### Overview
Add comprehensive testing, finalize documentation, and perform end-to-end validation.

### Changes Required

#### 5.1 Integration Testing

Create test script: `tests/test_integration.py`

```python
"""Integration tests for VirNucPro CLI"""

import subprocess
import tempfile
from pathlib import Path
import shutil


def test_cli_help():
    """Test CLI help displays correctly"""
    result = subprocess.run(
        ['python', '-m', 'virnucpro', '--help'],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0
    assert 'VirNucPro' in result.stdout
    assert 'predict' in result.stdout


def test_list_devices():
    """Test device listing"""
    result = subprocess.run(
        ['python', '-m', 'virnucpro', 'utils', 'list-devices'],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0
    assert 'CPU' in result.stdout


def test_generate_config():
    """Test config generation"""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_file = Path(tmpdir) / 'test_config.yaml'
        result = subprocess.run(
            ['python', '-m', 'virnucpro', 'utils', 'generate-config', '-o', str(config_file)],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        assert config_file.exists()


# Add more integration tests...
```

#### 5.2 Update README

**File**: `README.md`

```markdown
# VirNucPro - Viral Nucleotide and Protein Identifier

A production-ready tool for identifying viral sequences using six-frame translation and deep learning models (DNABERT-S and ESM-2).

## Features

- ✅ Click-based CLI with comprehensive help
- ✅ GPU selection and validation
- ✅ Automatic checkpointing and resume
- ✅ YAML configuration support
- ✅ Input validation
- ✅ Comprehensive logging
- ✅ Automatic cleanup of intermediate files

## Installation

### Requirements
- Python 3.9+
- CUDA 11.8+ (optional, for GPU acceleration)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/Li-Jing-1997/VirNucPro.git
cd VirNucPro
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Test installation:
```bash
python -m virnucpro --help
```

## Quick Start

### Basic Prediction

```bash
python -m virnucpro predict input.fasta
```

### With Options

```bash
# Use 300bp model with GPU 1
python -m virnucpro predict input.fasta --model-type 300 --device cuda:1

# Custom output directory
python -m virnucpro predict input.fasta --output-dir results/

# Keep intermediate files
python -m virnucpro predict input.fasta --keep-intermediate
```

### Resume from Checkpoint

If prediction is interrupted, resume with:

```bash
python -m virnucpro predict input.fasta --resume
```

### Using Configuration File

```bash
# Generate config template
python -m virnucpro utils generate-config -o my_config.yaml

# Edit my_config.yaml, then use it
python -m virnucpro predict input.fasta --config my_config.yaml
```

## CLI Reference

### Prediction Command

```bash
python -m virnucpro predict [OPTIONS] INPUT_FILE

Options:
  -m, --model-type [300|500|custom]  Model type (default: 500)
  -p, --model-path PATH              Path to custom model
  -e, --expected-length INT          Expected sequence length
  -o, --output-dir PATH              Output directory
  -d, --device TEXT                  Device: auto, cpu, cuda, cuda:N, or N
  -b, --batch-size INT               Batch size
  -w, --num-workers INT              Data loading workers
  -k, --keep-intermediate            Keep intermediate files
  --resume                           Resume from checkpoint
  -f, --force                        Overwrite existing output
  --help                             Show help message
```

### Utility Commands

```bash
# List available devices
python -m virnucpro utils list-devices

# Validate FASTA file
python -m virnucpro utils validate input.fasta

# Generate config template
python -m virnucpro utils generate-config -o config.yaml
```

## Output Files

Prediction generates the following files in the output directory:

- `prediction_results.txt` - Per-frame predictions with scores
- `prediction_results_highestscore.csv` - Consensus predictions
- `.checkpoints/` - Checkpoint data for resume (auto-cleaned on success)

## Advanced Usage

### GPU Selection

```bash
# Auto-detect (use GPU if available)
python -m virnucpro predict input.fasta --device auto

# Use specific GPU
python -m virnucpro predict input.fasta --device cuda:1

# Force CPU
python -m virnucpro predict input.fasta --device cpu

# List available devices
python -m virnucpro utils list-devices
```

### Configuration File

Create a `config.yaml`:

```yaml
prediction:
  batch_size: 512
  num_workers: 8
  default_model_type: "500"

device:
  default: "cuda:0"
  fallback_to_cpu: true

logging:
  default_level: "DEBUG"
```

Use it:
```bash
python -m virnucpro predict input.fasta --config config.yaml
```

## Troubleshooting

### CUDA/GPU Issues

If you get CUDA errors:
1. Check GPU availability: `python -m virnucpro utils list-devices`
2. Verify CUDA installation: `nvidia-smi`
3. Use CPU if needed: `--device cpu`

### Resume Not Working

If `--resume` doesn't work:
1. Ensure output directory hasn't been moved/renamed
2. Check that `.checkpoints/` directory exists
3. Configuration must match original run

## Citation

If you use VirNucPro in your research, please cite:

```
[Citation information]
```

## License

[License information]
```

### Success Criteria

#### Automated Verification:
- [ ] README renders correctly on GitHub
- [ ] All example commands in README are valid
- [ ] Integration tests pass: `python -m pytest tests/`
- [ ] Package imports cleanly: `python -c "import virnucpro; from virnucpro.cli.main import cli"`

#### Manual Verification:
- [ ] End-to-end prediction works on sample data
- [ ] Resume functionality works correctly
- [ ] Device selection handles all edge cases
- [ ] Error messages are clear and actionable
- [ ] Documentation is complete and accurate
- [ ] Config file comments are helpful

---

## Testing Strategy

### Unit Tests
- Configuration loading and validation
- Device selection logic
- Checkpoint state management
- File validation utilities
- Sequence processing functions

### Integration Tests
- CLI help and version display
- Config generation
- Device listing
- Input validation
- Full prediction pipeline (small test file)
- Resume from checkpoint

### Manual Testing Steps
1. Run prediction with default settings
2. Interrupt prediction and resume
3. Test with invalid input file
4. Test with non-existent GPU
5. Test configuration file override
6. Verify cleanup of intermediate files
7. Test on sequences of different lengths

## Performance Considerations

- Checkpoint I/O is minimal (JSON files only)
- File-level progress tracking adds small overhead
- Hash computation only on checkpoint save/load
- No performance impact when checkpointing disabled

## Migration Notes

**For existing users**:
- Old command: `python prediction.py input.fasta 500 500_model.pth`
- New command: `python -m virnucpro predict input.fasta --model-type 500`
- Output location changes from `input_merged/` to `input_predictions/`
- Old scripts (`prediction.py`, `units.py`) remain for reference

## References

- Research document: `thoughts/shared/research/2025-11-10-virnucpro-codebase-structure.md`
- GPU selection research: Web search on PyTorch device patterns
- Checkpointing research: Nextflow, Snakemake patterns
- Original repository: https://github.com/Li-Jing-1997/VirNucPro
