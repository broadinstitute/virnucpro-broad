---
phase: 05-model-training-validation
plan: 01
subsystem: model-training
tags: [fastesm2, mlp, training, test-split, early-stopping, sklearn, pytorch, stratification]

# Dependency graph
requires:
  - phase: 04-training-data-preparation
    provides: Complete training dataset with FastESM2_650 embeddings (201 merged files, 2M sequences)
  - phase: 03-dimension-compatibility
    provides: MLPClassifier with 2048-dim input validation
  - phase: 03-dimension-compatibility
    provides: save_checkpoint_with_metadata() with version 2.0.0
provides:
  - Test set splitting script with stratified 10% split (create_test_set.py)
  - Enhanced training script with early stopping and logging (train_fastesm.py)
  - Reproducible train/test separation methodology
  - Training infrastructure ready for model retraining
affects: [05-model-training-validation-02, 05-model-training-validation-03]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - Stratified test set split with sklearn.model_selection.train_test_split
    - Symlink-based test set organization to save disk space
    - JSON metadata persistence for train/test file lists
    - Comprehensive reproducibility: all seeds, worker_init_fn, DataLoader generator
    - Validation split from training data (nested split)
    - Per-epoch detailed logging with time tracking

key-files:
  created:
    - scripts/create_test_set.py
    - scripts/train_fastesm.py
  modified: []

key-decisions:
  - "Test set: 10% stratified split with fixed seed 42 for reproducibility"
  - "Symlinks instead of file copies in test_set directory to save disk space"
  - "Validation split: 10% of training data for early stopping during training"
  - "Early stopping patience: 7 epochs (slightly higher than original 5 to allow exploration)"
  - "Existing hyperparameters preserved: SGD lr=0.0002, momentum=0.9, StepLR step_size=10 gamma=0.85"
  - "Checkpoint filename: model_fastesm650.pth with version 2.0.0 metadata"
  - "Duplicated classes from train.py to avoid module-level execution on import"

patterns-established:
  - "Test set creation before training ensures both FastESM2 and baseline use identical test data"
  - "Metadata-driven training: script reads train/test file lists from JSON, not directories"
  - "Nested data splits: train/test for evaluation, train→train/val for early stopping"
  - "Comprehensive logging: per-epoch metrics, final test evaluation, training summary"

# Metrics
duration: 3min
completed: 2026-02-09
---

# Phase 05 Plan 01: Test Set Splitting and Training Script Summary

**Created stratified test set splitting script (10% split, fixed seed 42) and enhanced training script with early stopping, detailed logging, and checkpoint versioning ready for FastESM2 MLP retraining**

## Performance

- **Duration:** 3 min
- **Started:** 2026-02-09T14:54:12Z
- **Completed:** 2026-02-09T14:57:09Z
- **Tasks:** 2 (both auto execution)
- **Scripts created:** 2 (269 + 579 lines)
- **Deviations:** 0 (plan executed exactly as written)

## Accomplishments

- Created stratified test set splitting script with sklearn.model_selection.train_test_split
- Implemented 10% test split with fixed seed 42 for reproducible train/test separation
- Test files organized via symlinks in data/test_set/viral and data/test_set/non_viral directories
- Metadata persistence in test_metadata.json with train/test file lists and distribution stats
- Enhanced training script duplicating patterns from train.py (MLPClassifier, EarlyStopping, save_checkpoint_with_metadata)
- Comprehensive reproducibility: set all seeds (random, numpy, torch, cuda), worker_init_fn, DataLoader generator
- Validation split creation: 10% of training data for early stopping (nested split)
- Early stopping with patience=7 (higher than original 5 to allow more exploration)
- Detailed per-epoch logging: train loss, val loss, accuracy, F1, precision, recall, AUC, LR, time
- Dimension validation before training starts
- Checkpoint saving as model_fastesm650.pth with version 2.0.0 metadata
- Existing hyperparameters preserved for apples-to-apples comparison: SGD lr=0.0002, momentum=0.9, StepLR step_size=10 gamma=0.85

## Task Commits

1. **Task 1: Create stratified test set splitting script** - `441772f` (feat)
2. **Task 2: Create enhanced training script for FastESM2 MLP** - `cea964c` (feat)

## Files Created/Modified

**Created:**
- `scripts/create_test_set.py` - 269 lines (exceeds 80-line minimum)
  - Discovers all merged .pt files from data_merge directory
  - Classifies files as viral (label=1) or non-viral (label=0) based on directory name
  - Performs stratified split using sklearn with test_size=0.1, stratify=labels, random_state=42
  - Creates symlinks in data/test_set/viral and data/test_set/non_viral
  - Saves metadata to test_metadata.json with train/test file lists, distribution stats, sequence counts
  - Reports distribution: viral ratio in train vs test, number of files, sequence counts
  - Comprehensive seed setting: random.seed(), np.random.seed(), torch.manual_seed(), torch.cuda.manual_seed_all()

- `scripts/train_fastesm.py` - 579 lines (exceeds 150-line minimum)
  - Loads train/test split from test_metadata.json
  - Uses only training files (excludes test set completely)
  - Creates 10% validation split from training data for early stopping
  - Duplicates FileBatchDataset, MLPClassifier, EarlyStopping from train.py to avoid module-level execution
  - Imports DimensionError, MERGED_DIM, CHECKPOINT_VERSION from units
  - Uses existing hyperparameters: input_dim=2048, hidden_dim=512, batch_size=32, lr=0.0002, momentum=0.9, StepLR step_size=10 gamma=0.85
  - Early stopping with patience=7 (default), tracks best model weights
  - Detailed logging to training_log_fastesm650.txt: per-epoch metrics, training summary, final test evaluation
  - Dimension validation before training: loads one batch, checks shape matches MERGED_DIM
  - Checkpoint saving with save_checkpoint_with_metadata() and version 2.0.0
  - Comprehensive reproducibility: set_all_seeds(), worker_init_fn(), torch.Generator for DataLoader shuffle
  - Full argparse interface: --metadata, --epochs, --patience, --batch-size, --lr, --output, --save-best, --log-file, --seed, --num-workers, --val-split

**Modified:**
- None

## Decisions Made

**Test set split ratio and seed:** Used 10% test split with fixed seed 42 per user decision matching paper methodology. This ensures reproducible train/test separation that can be used for both FastESM2 and baseline evaluation.

**Symlinks instead of file copies:** Test set directory uses symlinks to original merged .pt files instead of copying them. This saves disk space (201 files × ~several MB each) while maintaining clean separation between train and test data.

**Validation split strategy:** Created 10% validation split from training data (nested split). This is separate from the test set - validation is used for early stopping during training, test set is for final unbiased evaluation only.

**Early stopping patience:** Set default to 7 epochs (slightly higher than original 5 in train.py). FastESM2 embeddings are different from ESM2 3B, so giving the optimizer more exploration time before stopping could help find better convergence.

**Duplicate classes from train.py:** Importing train.py directly would trigger module-level execution (it runs training at import time). Duplicated FileBatchDataset, MLPClassifier, EarlyStopping, save_checkpoint_with_metadata to train_fastesm.py to avoid this issue.

**Metadata-driven training:** Training script reads train/test file lists from test_metadata.json instead of scanning directories. This guarantees exact reproducibility - the same files are used every time, regardless of directory state.

**Comprehensive reproducibility:** Set all random seeds (random, numpy, torch, cuda), enabled cudnn.deterministic=True and cudnn.benchmark=False, used worker_init_fn for DataLoader workers, and passed torch.Generator to DataLoader shuffle. This ensures bit-identical results across runs with same seed.

**Logging verbosity:** Per-epoch logging includes train loss, val loss, val accuracy, val F1, val precision, val recall, val AUC, current learning rate, epoch time, and total time. This provides complete visibility into training dynamics for debugging and analysis.

## Deviations from Plan

None - plan executed exactly as written.

All requirements fulfilled:
- Test set creation script with stratified split (TEST-01)
- Training script with 2048-dim features (TRAIN-02)
- Dimension validation before training (TRAIN-03)
- Checkpoint saved as model_fastesm650.pth with metadata (TRAIN-04)
- Detailed per-epoch logging (TRAIN-05)

## Issues Encountered

None - no blocking issues or bugs discovered during execution.

## User Setup Required

None - scripts are ready to run in Docker environment with existing dependencies (torch, sklearn, tqdm, numpy).

## Next Phase Readiness

**Ready for Phase 5 Plan 2 (model training execution):**
- Test set splitting script ready to run (create_test_set.py)
- Training script ready to run in Docker with GPU (train_fastesm.py)
- Stratified 10% test split methodology established
- Reproducible training infrastructure with comprehensive seed management
- Detailed logging for training analysis and debugging
- Checkpoint versioning ensures compatibility validation

**Usage workflow:**
1. Run `python scripts/create_test_set.py` to create test set and metadata
2. Run `python scripts/train_fastesm.py` in Docker to train model
3. Model saves to model_fastesm650.pth with version 2.0.0 metadata
4. Training log saves to training_log_fastesm650.txt with all metrics
5. Test set metadata in data/test_set/test_metadata.json contains train/test file lists for reproducibility

**Quality metrics:**
- Both scripts exceed minimum line requirements (269 > 80, 579 > 150)
- All must_haves verified:
  - ✓ Test set: 10% stratified split, separate directory, fixed seed
  - ✓ Training uses only train split, not full dataset
  - ✓ Early stopping and detailed per-epoch logging
  - ✓ Checkpoint saved as model_fastesm650.pth with version 2.0.0
- All key_links verified:
  - ✓ create_test_set.py reads from data/data_merge/
  - ✓ train_fastesm.py reads test_metadata.json to exclude test files
  - ✓ train_fastesm.py reuses MLPClassifier, EarlyStopping, save_checkpoint patterns
- All verification criteria met:
  - ✓ Both scripts run --help without error (syntax valid)
  - ✓ Both scripts import MERGED_DIM from units
  - ✓ Training script reads test_metadata.json
  - ✓ Training script uses existing hyperparameters (SGD lr=0.0002, etc)
  - ✓ Early stopping patience=7
  - ✓ Checkpoint uses save_checkpoint_with_metadata() with version 2.0.0

**Blockers:**
None - all must_haves and success criteria met.

**Next plan (05-02) should:**
- Run create_test_set.py to generate test split
- Run train_fastesm.py in Docker with GPU to train FastESM2 MLP
- Monitor training logs for convergence and early stopping behavior
- Validate checkpoint saved with correct metadata
- Prepare for evaluation and comparison against ESM2 3B baseline

---
*Phase: 05-model-training-validation*
*Completed: 2026-02-09*

## Self-Check: PASSED

Created files verified:
- scripts/create_test_set.py: FOUND (269 lines)
- scripts/train_fastesm.py: FOUND (579 lines)

All commits verified:
- 441772f: FOUND (Task 1)
- cea964c: FOUND (Task 2)
