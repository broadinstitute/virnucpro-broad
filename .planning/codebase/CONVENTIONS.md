# Coding Conventions

**Analysis Date:** 2026-02-06

## Naming Patterns

**Files:**
- Lowercase with underscores: `train.py`, `features_extract.py`, `download_data.py`
- Primary script files use domain names: `train.py`, `prediction.py`, `units.py`
- Configuration/utility files: `units.py` (imported utility module)

**Functions:**
- snake_case for all function definitions
- Descriptive names: `identify_seq()`, `translate_dna()`, `extract_DNABERT_S()`, `split_fasta_chunk()`
- Action-oriented verbs: `process_record()`, `create_refseq_pro_list()`, `merge_data()`

**Variables:**
- snake_case for all local and module-level variables
- Long descriptive names acceptable for clarity: `nucleotide_file_list`, `protein_file_list`, `random_selected_other_nucleotide_files`
- Abbreviated names for temporary/loop variables: `f`, `x`, `y`, `item`
- Constants in UPPERCASE with underscores: `sequences_per_file = 10000`, `chunk_size = 300`

**Types/Classes:**
- PascalCase for all classes: `FileBatchDataset`, `MLPClassifier`, `PredictDataBatchDataset`, `EarlyStopping`
- Descriptive compound names: `MLPClassifier`, `EarlyStopping`

## Code Style

**Formatting:**
- No explicit formatter detected (black/autopep8 not configured)
- Line lengths vary: typically 80-120 characters
- 4-space indentation used consistently
- Blank lines used to separate logical sections within functions
- Double blank lines between function/class definitions

**Linting:**
- No linting configuration detected (no .eslintrc, .flake8, etc.)
- Code follows general Python conventions informally
- Some violations observed: inconsistent spacing, occasional long lines (80+ chars)

## Import Organization

**Order:**
1. Standard library imports: `import torch`, `from torch import nn`, `import os`, `import sys`, `from concurrent.futures import ...`
2. Third-party library imports: `from transformers import ...`, `from Bio import SeqIO`, `import pandas as pd`, `from sklearn.metrics import ...`
3. Local imports: `from units import *` (wildcard imports used frequently)

**Path Aliases:**
- No path aliases configured in pixi.toml or setup.cfg
- Relative paths used for data: `./data/`, `./data/data_merge/`

**Import Patterns:**
- Wildcard imports used: `from units import *` in `prediction.py`, `features_extract.py`, `make_train_dataset_300.py`
- Direct function imports used in training: `from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score`

## Error Handling

**Patterns:**
- Try-except blocks used for data parsing and file operations in `drew_fig.py` (lines 59-64):
  ```python
  try:
      chunk_num = int(chunk)
      chunk_numbers.append(chunk_num)
  except (IndexError, ValueError):
      print(f"Warning: Unexpected chunk format '{chunk}'. Skipping...")
      continue
  ```
- IndexError raised explicitly for dataset access in `train.py` (line 41) and `prediction.py` (line 45)
- Conditional checks for file existence and size matching in `download_data.py` (lines 60-68)
- torch.no_grad() context manager for inference: used in `train.py`, `prediction.py`, `units.py`

**Patterns for Dataset Classes:**
- Custom exceptions raised with descriptive messages: `raise IndexError("Index out of range")`
- File existence checks before processing: `if os.path.exists(...)` pattern repeated throughout

## Logging

**Framework:** console output only - `print()` statements

**Patterns:**
- Progress bars with tqdm: `tqdm(records, total=len(records))` in `units.py` (line 181), `train.py` (line 150)
- Descriptive print messages with context:
  ```python
  print(f'saved to: {output_file}')
  print(f"Predictions saved to {output_file}")
  print("Early stopping triggered")
  ```
- File logging in training via `write_log()` function in `train.py`:
  ```python
  def write_log(filename, message):
      with open(filename, 'a') as f:
          f.write(message + '\n')
  ```
- Status messages during data processing: `print(device)`, `print(log_message)`

## Comments

**When to Comment:**
- Minimal comment use observed throughout codebase
- Chinese comments appear occasionally (line 1 in `train.py`): `# 对每个种类单独训练效果还可以`
- Comments for section separation: `# viral_data` in `features_extract.py`
- Inline comments rare; when used, explain complex logic

**JSDoc/TSDoc:**
- No docstrings detected for functions
- No type hints used throughout codebase
- Function behavior must be inferred from implementation

## Function Design

**Size:**
- Range from 6-100+ lines for main functions
- Larger functions (50-100 lines) for complex data processing: `identify_seq()` (81-146 lines), `process_file()` in `make_train_dataset_300.py` (13-45 lines)
- Smaller utility functions (10-30 lines): `reverse_complement()`, `translate_frame()`

**Parameters:**
- 1-3 parameters typical for utility functions
- Up to 6-7 parameters for feature extraction functions: `extract_esm()` has 6 parameters with defaults
- Default parameters used: `extract_DNABERT_S(..., model_loaded=False, tokenizer=None, model=None)`

**Return Values:**
- Multiple returns common: `return proteins, data` in `extract_esm()`
- Dictionary returns for structured data: `{'seqid': ..., 'nucleotide': ..., 'protein': ...}`
- List returns for batch operations: `return all_seqids, all_predictions, all_probabilities` in `prediction.py`
- Early returns for error/skip conditions: `return` or `return None`

## Module Design

**Exports:**
- No explicit `__all__` definitions found
- Module-level variables exposed for import: `sequences_per_file`, `nucleotide_input_file_list`, `file_list`
- wildcard imports assume all non-private names are exported from `units.py`

**Barrel Files:**
- `units.py` serves as central utility module containing shared functions
- Functions in `units.py` used by: `prediction.py`, `features_extract.py`, `make_train_dataset_300.py`, `make_train_dataset_500.py`
- Not a traditional barrel export pattern; more monolithic utility module

**Module-level Code:**
- Executable code at module scope observed in:
  - `train.py`: model initialization, dataset creation, training loop executed (lines 43-220)
  - `features_extract.py`: file listing and processing loop (lines 35-85)
  - `make_train_dataset_300.py`: similar pattern with data pipeline setup
- Conditional main block: `if __name__ == "__main__":` used in `make_train_dataset_300.py` and `prediction.py`

## Common Patterns

**Torch Usage:**
- Consistent use of `torch.device()` for GPU/CPU detection (all files with torch)
- `.to(device)` for tensor movement in inference and training
- `torch.load()` for model/data loading
- `torch.no_grad()` context manager for inference

**Pandas Operations:**
- GroupBy and apply patterns: `df.groupby("Modified_ID", group_keys=False).apply(determine_virus)` in `prediction.py`
- DataFrame creation from results for CSV export
- Column access via string indexing: `df["score1"]`, `df["Sequence_ID"]`

**File I/O Pattern:**
- Context managers with `with` statement used consistently for file operations
- Path manipulation via string operations: `.split()`, `.replace()`, string formatting
- No pathlib usage; using os.path instead

---

*Convention analysis: 2026-02-06*
