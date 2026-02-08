# Phase 3: Dimension Compatibility - Context

**Gathered:** 2026-02-07
**Status:** Ready for planning

<domain>
## Phase Boundary

Update all downstream code to handle the dimension change from ESM2 3B embeddings (2560-dim) to FastESM2_650 embeddings (1280-dim). The merged feature vector changes from 3328-dim (768 DNA + 2560 protein) to 2048-dim (768 DNA + 1280 protein). This phase ensures merge_data(), MLPClassifier, and checkpoint handling work correctly with the new dimensions.

</domain>

<decisions>
## Implementation Decisions

### Validation strategy
- Comprehensive checks: Validate dimensions at extraction output, merge points, model input, and checkpoint loading
- Fail fast with assertions: Use assert statements that crash immediately on dimension mismatch
- Basic error messages: State expected vs actual dimensions (e.g., "Expected 1280-dim protein embeddings, got 2560-dim")
- Configurable toggle: Environment variable or config option (VALIDATE_DIMS) to enable/disable validation

### Backward compatibility
- Hard incompatibility: Old ESM2 3B checkpoints cannot load with new code - namespace protection prevents silent failures
- Single model only: Code supports FastESM2_650 only after Phase 3 - clean migration, no dual pipeline support
- Clear error with migration guide: Detect old checkpoints and show message like "This checkpoint uses ESM2 3B (2560-dim). Re-extract features with FastESM2_650 and retrain."
- Filename convention: Use different naming patterns (*_fastesm.pt vs *_esm2.pt) to distinguish old vs new feature files

### Metadata and versioning
- Track embedding model info: Model name (fastesm650), dimensions (1280), HuggingFace model ID
- Track feature dimensions: DNA dim (768), protein dim (1280), merged dim (2048)
- Track training metadata: Training date, dataset version, hyperparameters used
- Semantic versioning: checkpoint_version: '2.0.0' - major version changes break compatibility
- Embed metadata in feature files: Store {'embeddings': tensor, 'model': 'fastesm650', 'dim': 1280, 'extraction_date': ...} in .pt files

### Error handling
- Immediate failure on mismatch: Stop execution when dimension mismatch detected during prediction
- Technical error details: Show expected dim, actual dim, tensor shape, code location - for developers
- Custom DimensionError class: Define custom exception with standardized attributes for all dimension validation failures
- Always validate critical paths: Some checks (like model input dims) always run even when VALIDATE_DIMS is disabled

### Claude's Discretion
- Checkpoint metadata structure (top-level keys vs nested dict)
- Exact wording of error messages beyond basic dimension info
- Location and frequency of validation checks within "comprehensive" approach

</decisions>

<specifics>
## Specific Ideas

- OLD format: 768 (DNABERT-S) + 2560 (ESM2 3B) = 3328-dim merged features
- NEW format: 768 (DNABERT-S) + 1280 (FastESM2_650) = 2048-dim merged features
- State.md notes critical dimension change already documented in project context
- Phase 5 requires side-by-side comparison, but Phase 3 implements single-model-only approach (comparison will use separate old/new codebases or branches)

</specifics>

<deferred>
## Deferred Ideas

None - discussion stayed within phase scope

</deferred>

---

*Phase: 03-dimension-compatibility*
*Context gathered: 2026-02-07*
