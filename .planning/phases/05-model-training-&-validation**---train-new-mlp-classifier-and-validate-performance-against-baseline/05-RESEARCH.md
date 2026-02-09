# Phase 5: Model Training & Validation - Research

**Researched:** 2026-02-09
**Domain:** PyTorch deep learning model training, validation, and benchmarking
**Confidence:** HIGH

## Summary

This phase involves training a new MLP classifier on FastESM2_650 embeddings (2048-dim features), creating a proper test set, validating performance against the ESM2 3B baseline, and benchmarking the speed improvement. The research focuses on PyTorch training best practices, stratified test set creation for imbalanced datasets, comprehensive classification metrics, and GPU inference benchmarking.

The standard approach uses PyTorch's native training loops with early stopping, scikit-learn's stratified sampling for test set creation, comprehensive classification metrics (F1, Accuracy, Precision, Recall, ROC-AUC, confusion matrix) for validation, and GPU-synchronized benchmarking with warmup runs. The existing codebase already implements most core patterns correctly but needs enhancement for test set creation, comprehensive reporting, and proper benchmarking.

**Primary recommendation:** Extend existing train.py patterns with stratified test set creation using sklearn, add comprehensive metrics reporting with markdown output, implement GPU-synchronized benchmarking with proper warmup, and ensure full reproducibility through proper random seed management across all libraries.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Training workflow:**
- Use existing hyperparameters from train.py (epochs, batch size, learning rate) for apples-to-apples comparison
- Checkpointing: Default to saving final trained model, with optional parameter to save best validation model instead
- Early stopping: Yes, with configurable patience (stop if validation loss doesn't improve for N epochs)
- Logging: Detailed progress including batch progress, learning rate, time per epoch, loss curves

**Test set creation:**
- Sampling approach: 10% random sample from both viral and non-viral datasets (matching paper methodology)
- Reproducibility: Fixed random seed for consistent test set across runs
- Storage: Separate test files in dedicated directory (not mixed with training data)
- Validation: Check and report that test set has representative distribution (viral families, sequence lengths, etc.)

**Performance validation:**
- Acceptable accuracy drop: <5% from ESM2 3B baseline (strict threshold)
- Metrics to report: F1, Accuracy, Precision, Recall, ROC-AUC, and confusion matrix
- Result presentation: Both terminal summary and detailed markdown report file
- Failure handling: If <5% threshold not met, report clearly and halt deployment with suggested next steps

**Speed benchmarking:**
- Benchmark scope: Just embedding extraction time (FastESM2 vs ESM2 3B protein embeddings)
- Success criteria: FastESM2 takes ≤50% of ESM2 3B time (strict 2x speedup)
- Measurement protocol: 3 runs averaged to account for timing variability

### Claude's Discretion
- Choice of benchmark sample for speed testing (representative of real-world usage)
- Specific patience value for early stopping
- Exact format and structure of markdown report file
- Additional diagnostic metrics if helpful

### Deferred Ideas (OUT OF SCOPE)
- Hyperparameter optimization for FastESM2 embeddings — future milestone
- Per-class performance breakdown by viral family — could be added if needed, but not required for phase success

</user_constraints>

## Standard Stack

The established libraries/tools for PyTorch model training and validation:

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| torch | 2.x | Deep learning framework | Industry standard for model training, native support for GPU acceleration |
| scikit-learn | 1.8.0+ | Classification metrics & splitting | Standard for train/test split with stratification, comprehensive metrics suite |
| tqdm | latest | Progress bars | De facto standard for training loop progress visualization |
| numpy | latest | Numerical operations | Foundation for scientific computing, required by sklearn |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| matplotlib | 3.x | Visualization | Confusion matrix plotting, training curve visualization |
| pandas | latest | Data manipulation | Tabular metrics formatting, markdown table generation via .to_markdown() |
| seaborn | latest | Statistical visualization | Enhanced confusion matrix heatmaps with annotations |

### Already in Project
The project already has these dependencies installed (from requirements.txt):
- torch (installed)
- scikit-learn (installed)
- numpy (installed)
- matplotlib (installed)

**No additional installations required** - all standard stack libraries are already present.

## Architecture Patterns

### Recommended Project Structure
```
scripts/
├── train_fastesm.py           # New training script for FastESM2 model
├── create_test_set.py         # Stratified test set creation
├── validate_performance.py    # Compare FastESM2 vs ESM2 3B metrics
└── benchmark_speed.py         # GPU-synchronized speed comparison

data/
├── data_merge/               # Existing training data (viral and non-viral)
├── test_set/                 # NEW: Separate test set directory
│   ├── viral/               # Test viral samples
│   └── non_viral/           # Test non-viral samples
└── test_metadata.json       # Test set composition info

models/
├── model_fastesm650.pth      # FastESM2 trained checkpoint
└── model_esm3b.pth           # ESM2 3B baseline (existing)

reports/
└── validation_report_{timestamp}.md  # Performance comparison report
```

### Pattern 1: Stratified Test Set Creation
**What:** Split dataset into train/test while preserving class distribution (10% test, 90% train)
**When to use:** Before training, one-time operation to create reproducible test set
**Example:**
```python
# Source: sklearn.model_selection.train_test_split documentation
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html

import random
import numpy as np
from sklearn.model_selection import train_test_split

# Set seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Collect all file paths and labels
viral_files = [...]  # List of viral .pt files
non_viral_files = [...]  # List of non-viral .pt files

all_files = viral_files + non_viral_files
labels = [1] * len(viral_files) + [0] * len(non_viral_files)

# Stratified split: 10% test, 90% train
train_files, test_files, train_labels, test_labels = train_test_split(
    all_files,
    labels,
    test_size=0.1,
    stratify=labels,  # Preserve class distribution
    random_state=SEED
)

# Save test set to separate directory
# Copy files to data/test_set/{viral,non_viral}/
```

### Pattern 2: PyTorch Training with Early Stopping
**What:** Train model with validation monitoring and early stopping to prevent overfitting
**When to use:** All model training workflows
**Example:**
```python
# Source: PyTorch training best practices and existing train.py
# https://pytorch.org/tutorials/beginner/introyt/trainingyt.html

class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop_counter = 0
        self.best_model_wts = None

    def __call__(self, val_loss, model):
        if self.best_score is None:
            self.best_score = val_loss
            self.best_model_wts = model.state_dict()
        elif val_loss < self.best_score - self.delta:
            self.best_score = val_loss
            self.best_model_wts = model.state_dict()
            self.early_stop_counter = 0
        else:
            self.early_stop_counter += 1
            if self.early_stop_counter >= self.patience:
                print("Early stopping triggered")
                model.load_state_dict(self.best_model_wts)
                return True
        return False

# Training loop with early stopping
early_stopping = EarlyStopping(patience=5)
for epoch in range(num_epochs):
    model.train()
    # ... training code ...

    val_loss = validate(model, val_loader)

    if early_stopping(val_loss, model):
        break
```

### Pattern 3: Reproducible PyTorch Training
**What:** Set all random seeds to ensure reproducible results across runs
**When to use:** Beginning of every training script
**Example:**
```python
# Source: PyTorch Reproducibility documentation
# https://pytorch.org/docs/stable/notes/randomness.html

import random
import numpy as np
import torch

def set_seed(seed=42):
    """Set seeds for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # Multi-GPU

    # Deterministic operations (slower but reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# For DataLoader reproducibility with num_workers > 0
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(42)

train_loader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=12,
    worker_init_fn=seed_worker,
    generator=g,
    shuffle=True
)
```

### Pattern 4: Comprehensive Classification Metrics
**What:** Calculate and report all required metrics for binary classification
**When to use:** Model evaluation on test set
**Example:**
```python
# Source: scikit-learn metrics documentation
# https://scikit-learn.org/stable/modules/model_evaluation.html

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import pandas as pd

def evaluate_model(model, test_loader, device):
    model.eval()
    all_labels = []
    all_predictions = []
    all_probs = []

    with torch.no_grad():
        for batch_data, batch_labels in test_loader:
            batch_data = batch_data.to(device)
            outputs = model(batch_data)
            probs = F.softmax(outputs, dim=1)[:, 1]
            _, predicted = torch.max(outputs, 1)

            all_labels.extend(batch_labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # Calculate all metrics
    metrics = {
        'accuracy': accuracy_score(all_labels, all_predictions),
        'precision': precision_score(all_labels, all_predictions, average='binary'),
        'recall': recall_score(all_labels, all_predictions, average='binary'),
        'f1': f1_score(all_labels, all_predictions, average='binary'),
        'roc_auc': roc_auc_score(all_labels, all_probs)
    }

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)

    return metrics, cm
```

### Pattern 5: GPU-Synchronized Benchmarking
**What:** Accurate GPU inference timing with synchronization and warmup runs
**When to use:** Speed comparison between FastESM2 and ESM2 3B
**Example:**
```python
# Source: PyTorch Benchmark documentation
# https://pytorch.org/tutorials/recipes/recipes/benchmark.html

import torch
import time

def benchmark_inference(model, test_sequences, num_runs=3, warmup=50):
    """
    Benchmark model inference with proper GPU synchronization.

    Args:
        model: Model to benchmark
        test_sequences: List of sequences to process
        num_runs: Number of benchmark runs to average
        warmup: Number of warmup iterations

    Returns:
        Average inference time in seconds
    """
    model.eval()
    device = next(model.parameters()).device

    # Warmup runs (not counted)
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(test_sequences[0].to(device))
            if torch.cuda.is_available():
                torch.cuda.synchronize()

    # Benchmark runs
    times = []
    with torch.no_grad():
        for run in range(num_runs):
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            start = time.perf_counter()

            for seq in test_sequences:
                _ = model(seq.to(device))

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            end = time.perf_counter()
            times.append(end - start)

    return sum(times) / len(times)
```

### Pattern 6: Markdown Report Generation
**What:** Generate formatted markdown report with metrics tables
**When to use:** Final validation reporting
**Example:**
```python
# Source: pandas to_markdown() and sklearn classification_report
# https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_markdown.html

import pandas as pd
from datetime import datetime

def generate_validation_report(fastesm_metrics, esm3b_metrics,
                               fastesm_cm, esm3b_cm, output_path):
    """Generate markdown report comparing FastESM2 vs ESM2 3B."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Metrics comparison table
    comparison = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC'],
        'ESM2 3B': [esm3b_metrics['accuracy'], esm3b_metrics['precision'],
                    esm3b_metrics['recall'], esm3b_metrics['f1'], esm3b_metrics['roc_auc']],
        'FastESM2_650': [fastesm_metrics['accuracy'], fastesm_metrics['precision'],
                        fastesm_metrics['recall'], fastesm_metrics['f1'], fastesm_metrics['roc_auc']],
        'Difference': [
            fastesm_metrics['accuracy'] - esm3b_metrics['accuracy'],
            fastesm_metrics['precision'] - esm3b_metrics['precision'],
            fastesm_metrics['recall'] - esm3b_metrics['recall'],
            fastesm_metrics['f1'] - esm3b_metrics['f1'],
            fastesm_metrics['roc_auc'] - esm3b_metrics['roc_auc']
        ]
    })

    with open(output_path, 'w') as f:
        f.write(f"# Model Validation Report\n\n")
        f.write(f"**Generated:** {timestamp}\n\n")
        f.write("## Performance Comparison\n\n")
        f.write(comparison.to_markdown(index=False, floatfmt=".4f"))
        f.write("\n\n")

        # Add confusion matrices, etc.
```

### Anti-Patterns to Avoid
- **Random splitting without stratification:** For imbalanced datasets, simple random split may not preserve class distribution, leading to unrepresentative test sets
- **No GPU synchronization in benchmarks:** CPU timer may finish before GPU operations complete, producing inaccurate timing results
- **No warmup runs:** First few GPU runs are slower due to initialization; including them skews benchmark results
- **Single timing measurement:** GPU timing has variance; always average multiple runs
- **Forgetting to set all random seeds:** Must set seeds for Python random, NumPy, PyTorch CPU, and PyTorch CUDA for full reproducibility
- **Not saving best model weights:** Early stopping should restore best weights, not final epoch weights

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Train/test split with stratification | Manual class-balanced sampling | `sklearn.model_selection.train_test_split(stratify=y)` | Handles edge cases (minimum class size, reproducibility), tested across millions of use cases |
| Classification metrics calculation | Manual TP/FP/TN/FN counting | `sklearn.metrics.*` (accuracy_score, f1_score, etc.) | Handles multiclass, weighted averaging, zero-division cases correctly |
| Confusion matrix visualization | matplotlib rectangles and text | `sklearn.metrics.ConfusionMatrixDisplay` or `seaborn.heatmap()` | Proper normalization, annotations, color scaling, class labels |
| Early stopping logic | Custom validation tracking | Established EarlyStopping class pattern | Edge cases: what if val loss ties? Delta thresholds? Restore best weights? |
| Random seed management | Setting torch.manual_seed() only | Comprehensive seed function (random, np, torch, torch.cuda, cudnn settings) | Need to set seeds for ALL libraries and configure cudnn for determinism |
| DataLoader worker seeding | Default worker initialization | `worker_init_fn` + generator pattern | Workers can have duplicate random states without proper seeding |
| GPU timing | `time.time()` around GPU operations | `torch.cuda.synchronize()` + `time.perf_counter()` | CPU and GPU run asynchronously; synchronization required for accurate timing |
| Markdown table formatting | String concatenation with pipes | `pandas.DataFrame.to_markdown()` | Handles alignment, escaping, float formatting automatically |

**Key insight:** PyTorch and scikit-learn ecosystems have battle-tested solutions for every common training and validation task. Custom implementations miss edge cases (empty classes in stratified split, CUDA async execution, worker process random state) that took the community years to discover and fix.

## Common Pitfalls

### Pitfall 1: Test Set Created After Training
**What goes wrong:** If test set is created from combined data after training, there's no guarantee it matches the distribution the model was validated on during training
**Why it happens:** Convenient to use existing train.py which does random split, then extract test set later
**How to avoid:** Create test set FIRST from raw merged data, save to separate directory, THEN train using only remaining training data
**Warning signs:** Test metrics vastly different from validation metrics during training

### Pitfall 2: Non-Stratified Sampling on Imbalanced Data
**What goes wrong:** Random 10% sample may not preserve viral/non-viral ratio, leading to unrepresentative test set (e.g., 15% viral in test vs 10% in full dataset)
**Why it happens:** Assuming random.shuffle() is sufficient for "random sampling"
**How to avoid:** Always use `stratify=labels` parameter in train_test_split for classification tasks
**Warning signs:** Class distribution in test set doesn't match training set distribution

### Pitfall 3: Comparing Metrics Across Different Test Sets
**What goes wrong:** If FastESM2 and ESM2 3B models are evaluated on different test sets, performance differences may be due to test set difficulty, not model quality
**Why it happens:** Running separate test creation for each model
**How to avoid:** Create test set ONCE with fixed random seed, use same test files for both models
**Warning signs:** Large variance in metrics across supposedly identical runs

### Pitfall 4: CPU-Only Timing for GPU Operations
**What goes wrong:** GPU operations are asynchronous; CPU timer finishes while GPU is still computing, resulting in 10-100x faster reported times than reality
**Why it happens:** Using `time.time()` around model forward pass without synchronization
**How to avoid:** Always call `torch.cuda.synchronize()` before start time and after end time
**Warning signs:** Benchmark shows impossibly fast times (e.g., 1000 sequences/sec on large model)

### Pitfall 5: No Warmup Runs in Benchmarking
**What goes wrong:** First few GPU operations include kernel compilation and CUDA initialization overhead, making them 2-10x slower than steady-state performance
**Why it happens:** Starting timer immediately without warmup iterations
**How to avoid:** Run 50+ warmup iterations (not counted), then measure actual benchmark runs
**Warning signs:** First run is much slower than subsequent runs; high variance in timing

### Pitfall 6: Incomplete Random Seed Setting
**What goes wrong:** Setting `random.seed(42)` but not `torch.manual_seed(42)` leads to non-reproducible DataLoader shuffling or model initialization
**Why it happens:** Not realizing each library (random, numpy, torch, torch.cuda) has separate RNG state
**How to avoid:** Set seeds for ALL libraries at script start, configure cudnn for determinism
**Warning signs:** Different results on identical runs with same seed; "reproducible" until changing batch size or num_workers

### Pitfall 7: DataLoader Worker Non-Determinism
**What goes wrong:** With `num_workers > 0`, workers may have duplicate random states, causing batch order non-determinism despite fixed seed
**Why it happens:** PyTorch doesn't automatically seed worker processes correctly
**How to avoid:** Use `worker_init_fn=seed_worker` and `generator=g` parameters in DataLoader
**Warning signs:** Different batch order with same seed when num_workers > 0 vs num_workers = 0

### Pitfall 8: Dimension Mismatch Silent Failures
**What goes wrong:** Training with wrong input dimensions (e.g., 3328 instead of 2048) may train successfully but produce garbage results
**Why it happens:** PyTorch auto-broadcasts or model accepts wrong shapes without error
**How to avoid:** Use explicit dimension validation in model forward pass (existing code already has this via DimensionError)
**Warning signs:** Model trains but metrics never improve; loss doesn't decrease

### Pitfall 9: Forgetting to Restore Best Weights
**What goes wrong:** Early stopping triggers but final model has weights from last (worst) epoch, not best epoch
**Why it happens:** Saving `self.best_model_wts` but not loading them back when early stop triggers
**How to avoid:** `model.load_state_dict(self.best_model_wts)` when patience exceeded (existing code already does this)
**Warning signs:** Validation metrics worse at end of training than middle epochs

### Pitfall 10: Benchmark Sample Not Representative
**What goes wrong:** Benchmarking on 100 short sequences when production has 10,000 long sequences gives misleading speedup estimates
**Why it happens:** Using convenient small sample for quick testing
**How to avoid:** Benchmark sample should match real-world sequence length distribution and batch sizes
**Warning signs:** Production performance doesn't match benchmark results

## Code Examples

Verified patterns from official sources and existing codebase:

### Existing Pattern: MLPClassifier with Dimension Validation
```python
# Source: Existing train.py lines 82-113
# Already implements input dimension validation correctly

class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_class):
        super(MLPClassifier, self).__init__()
        self.input_dim = input_dim
        self.hidden_layer = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, num_class)
        self.dropout = nn.Dropout(0.5)
        self.init_weights()

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
```

### Existing Pattern: Checkpoint Saving with Metadata
```python
# Source: Existing train.py lines 155-182
# Already implements version-aware checkpoint saving

def save_checkpoint_with_metadata(model, optimizer, epoch, best_loss,
                                   filepath='model_fastesm650.pth'):
    metadata = {
        'checkpoint_version': CHECKPOINT_VERSION,
        'model_type': 'fastesm650',
        'huggingface_model_id': 'Synthyra/FastESM2_650',
        'dna_dim': DNA_DIM,
        'protein_dim': PROTEIN_DIM,
        'merged_dim': MERGED_DIM,
        'input_dim': MERGED_DIM,
        'hidden_dim': 512,
        'num_class': 2,
        'training_date': datetime.datetime.now().isoformat(),
        'pytorch_version': torch.__version__
    }

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_loss': best_loss,
        'metadata': metadata
    }

    torch.save(checkpoint, filepath)
```

### New Pattern: Test Set Distribution Validation
```python
# Source: Best practices for train/test split validation
# https://encord.com/blog/train-val-test-split/

def validate_test_set_distribution(test_files, test_labels):
    """
    Check that test set has representative distribution.

    Reports:
    - Class balance (viral vs non-viral ratio)
    - Number of samples per class
    - Comparison to expected 10% split
    """
    viral_count = sum(test_labels)
    non_viral_count = len(test_labels) - viral_count
    total = len(test_labels)

    viral_ratio = viral_count / total

    print(f"Test Set Distribution:")
    print(f"  Total samples: {total}")
    print(f"  Viral: {viral_count} ({viral_ratio:.2%})")
    print(f"  Non-viral: {non_viral_count} ({1-viral_ratio:.2%})")

    # Check against expected ratio (should match training data)
    # Exact ratio depends on original dataset composition

    return {
        'total': total,
        'viral': viral_count,
        'non_viral': non_viral_count,
        'viral_ratio': viral_ratio
    }
```

### New Pattern: Confusion Matrix Visualization
```python
# Source: scikit-learn ConfusionMatrixDisplay
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ConfusionMatrixDisplay.html

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def plot_confusion_matrix(y_true, y_pred, model_name, save_path=None):
    """
    Create and save confusion matrix visualization.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        model_name: Model name for title
        save_path: Path to save figure (optional)
    """
    cm = confusion_matrix(y_true, y_pred)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=['Non-Viral', 'Viral']
    )

    disp.plot(cmap='Blues', values_format='d')
    plt.title(f'Confusion Matrix - {model_name}')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.close()

    return cm
```

### New Pattern: Performance Threshold Validation
```python
# Source: User requirement for <5% accuracy drop threshold

def validate_performance_threshold(fastesm_metrics, baseline_metrics,
                                   threshold=0.05):
    """
    Validate that FastESM2 performance is within acceptable range of baseline.

    Args:
        fastesm_metrics: Dict of FastESM2 metrics
        baseline_metrics: Dict of ESM2 3B baseline metrics
        threshold: Maximum acceptable accuracy drop (default 0.05 = 5%)

    Returns:
        (passed: bool, report: str)
    """
    accuracy_drop = baseline_metrics['accuracy'] - fastesm_metrics['accuracy']

    passed = accuracy_drop <= threshold

    report_lines = [
        "## Performance Threshold Validation",
        f"Baseline (ESM2 3B) Accuracy: {baseline_metrics['accuracy']:.4f}",
        f"FastESM2_650 Accuracy: {fastesm_metrics['accuracy']:.4f}",
        f"Accuracy Drop: {accuracy_drop:.4f} ({accuracy_drop*100:.2f}%)",
        f"Threshold: {threshold:.4f} ({threshold*100:.2f}%)",
        "",
        f"**Result: {'PASSED ✓' if passed else 'FAILED ✗'}**"
    ]

    if not passed:
        report_lines.extend([
            "",
            "### Suggested Next Steps:",
            "1. Review training logs for convergence issues",
            "2. Check for dimension mismatches in input data",
            "3. Verify test set distribution matches training set",
            "4. Consider hyperparameter tuning (deferred to future milestone)"
        ])

    return passed, "\n".join(report_lines)
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Manual train/test split with random indices | `sklearn.model_selection.train_test_split(stratify=y)` | sklearn 0.17+ (2015) | Ensures class balance in splits, reproducibility with random_state |
| `time.time()` for GPU timing | `torch.cuda.synchronize()` + `time.perf_counter()` | PyTorch best practices (2020+) | Accurate GPU timing (not CPU-only) |
| `torch.backends.cudnn.benchmark = True` always | Conditional (False for reproducibility, True for production) | PyTorch 1.7+ (2020) | Trade-off: determinism vs speed |
| Single random seed (`random.seed()`) | Multi-library seeding (random, np, torch, cuda) | PyTorch reproducibility guide (2021+) | Full reproducibility across all RNG sources |
| String-based logging | Structured logging with metrics tracking | Modern ML ops (2022+) | Easier to parse, visualize, compare runs |
| No warmup in benchmarks | 50+ warmup iterations standard | GPU benchmarking best practices (2023+) | Excludes kernel compilation overhead from measurements |

**Deprecated/outdated:**
- `torch.no_grad()`: Still works but `torch.inference_mode()` is faster for inference-only code (PyTorch 1.9+)
- Manual confusion matrix plotting: `sklearn.metrics.ConfusionMatrixDisplay` (sklearn 0.22+, 2019) provides better defaults
- Single-run benchmarking: Always average 3+ runs for statistical significance

## Open Questions

Things that couldn't be fully resolved:

1. **What constitutes "representative" benchmark sample for speed testing?**
   - What we know: Should match real-world sequence length distribution and batch patterns
   - What's unclear: Exact composition of production workload (sequence length distribution, batch sizes)
   - Recommendation: Use 100-1000 sequences sampled from actual test set with same length distribution

2. **Optimal early stopping patience value**
   - What we know: Existing code uses patience=5, standard range is 3-10 epochs
   - What's unclear: Whether FastESM2 embeddings converge at different rate than ESM2 3B
   - Recommendation: Start with patience=5 (match baseline), increase to 7-10 if training seems unstable

3. **Should we track additional metrics beyond user requirements?**
   - What we know: Required metrics are F1, Accuracy, Precision, Recall, ROC-AUC, confusion matrix
   - What's unclear: Whether per-class breakdown or loss curves would aid debugging
   - Recommendation: Track training/validation loss curves for debugging, but not required for phase success

4. **Exact speedup expectation accounting for other pipeline components**
   - What we know: Benchmark scope is "just embedding extraction time" with 2x speedup target
   - What's unclear: How to isolate embedding extraction from tokenization overhead
   - Recommendation: Measure tokenization separately, report both tokenization+embedding and embedding-only times

## Sources

### Primary (HIGH confidence)
- PyTorch Official Documentation - Training with PyTorch: https://docs.pytorch.org/tutorials/beginner/introyt/trainingyt.html
- PyTorch Official Documentation - Reproducibility: https://pytorch.org/docs/stable/notes/randomness.html
- PyTorch Official Documentation - Benchmark: https://pytorch.org/tutorials/recipes/recipes/benchmark.html
- scikit-learn 1.8.0 Documentation - train_test_split: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
- scikit-learn 1.8.0 Documentation - Classification Metrics: https://scikit-learn.org/stable/modules/model_evaluation.html
- scikit-learn 1.8.0 Documentation - ConfusionMatrixDisplay: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ConfusionMatrixDisplay.html
- Existing VirNucPro codebase - train.py (dimension validation, early stopping, checkpoint saving patterns)
- Existing VirNucPro codebase - units.py (dimension constants, validation functions)

### Secondary (MEDIUM confidence)
- PyTorch Forums - Stratified Split Discussion: https://discuss.pytorch.org/t/how-to-do-a-stratified-split/62290
- PyTorch Forums - DataLoader Reproducibility: https://discuss.pytorch.org/t/reproducibility-and-number-of-dataloader-workers/163011
- GitHub - early-stopping-pytorch: https://github.com/Bjarten/early-stopping-pytorch
- Machine Learning Mastery - Managing PyTorch Training with Checkpoints and Early Stopping: https://machinelearningmastery.com/managing-a-pytorch-training-process-with-checkpoints-and-early-stopping/
- Real Python - Train Test Split: https://realpython.com/train-test-split-python-data/
- Encord Blog - Train/Val/Test Split Best Practices: https://encord.com/blog/train-val-test-split/

### Tertiary (LOW confidence - general guidance only)
- Medium - PyTorch Training Best Practices (multiple articles, 2025-2026): General training patterns, not authoritative
- GeeksforGeeks - Confusion Matrix Visualization: Code examples, not best practices source
- Various blog posts on GPU benchmarking: Patterns confirmed by official PyTorch docs

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - All libraries verified from official docs, already in project requirements
- Architecture patterns: HIGH - Based on official PyTorch/sklearn documentation and existing codebase patterns
- Pitfalls: HIGH - Documented in official PyTorch reproducibility guide and community forums with confirmed issues
- Code examples: HIGH - Extracted from official documentation and verified existing code

**Research date:** 2026-02-09
**Valid until:** 2026-03-09 (30 days - stable domain, PyTorch 2.x and sklearn 1.8.x are mature)

**Notes:**
- Existing codebase already implements many best practices correctly (dimension validation, early stopping, checkpoint metadata)
- Main gaps are: stratified test set creation, comprehensive metrics reporting, GPU-synchronized benchmarking
- No new dependencies required - all standard libraries already installed
- Phase success criteria well-defined and measurable (<5% accuracy drop, 2x speed improvement)
