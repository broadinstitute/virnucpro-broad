# Phase 5: Model Training & Validation - Context

**Gathered:** 2026-02-09
**Status:** Ready for planning

<domain>
## Phase Boundary

Train a new MLP classifier on FastESM2_650 embeddings (2048-dim features) and validate that it performs comparably to the ESM2 3B baseline while delivering the promised 2x speed improvement. This phase covers training, test set creation, performance validation, and speed benchmarking.

</domain>

<decisions>
## Implementation Decisions

### Training workflow
- Use existing hyperparameters from train.py (epochs, batch size, learning rate) for apples-to-apples comparison
- Checkpointing: Default to saving final trained model, with optional parameter to save best validation model instead
- Early stopping: Yes, with configurable patience (stop if validation loss doesn't improve for N epochs)
- Logging: Detailed progress including batch progress, learning rate, time per epoch, loss curves

### Test set creation
- Sampling approach: 10% random sample from both viral and non-viral datasets (matching paper methodology)
- Reproducibility: Fixed random seed for consistent test set across runs
- Storage: Separate test files in dedicated directory (not mixed with training data)
- Validation: Check and report that test set has representative distribution (viral families, sequence lengths, etc.)

### Performance validation
- Acceptable accuracy drop: <5% from ESM2 3B baseline (strict threshold)
- Metrics to report: F1, Accuracy, Precision, Recall, ROC-AUC, and confusion matrix
- Result presentation: Both terminal summary and detailed markdown report file
- Failure handling: If <5% threshold not met, report clearly and halt deployment with suggested next steps

### Speed benchmarking
- Benchmark scope: Just embedding extraction time (FastESM2 vs ESM2 3B protein embeddings)
- Success criteria: FastESM2 takes ≤50% of ESM2 3B time (strict 2x speedup)
- Measurement protocol: 3 runs averaged to account for timing variability

### Claude's Discretion
- Choice of benchmark sample for speed testing (representative of real-world usage)
- Specific patience value for early stopping
- Exact format and structure of markdown report file
- Additional diagnostic metrics if helpful

</decisions>

<specifics>
## Specific Ideas

- Paper methodology: "After generating the full positive (viral) and negative (non-viral) datasets, 10% of samples were randomly selected from each to form the test set. The remaining 90% was used for training."
- Hyperparameter tuning deferred to future milestone (not in scope for Phase 5)

</specifics>

<deferred>
## Deferred Ideas

- Hyperparameter optimization for FastESM2 embeddings — future milestone
- Per-class performance breakdown by viral family — could be added if needed, but not required for phase success

</deferred>

---

*Phase: 05-model-training-&-validation*
*Context gathered: 2026-02-09*
