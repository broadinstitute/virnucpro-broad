# Technical Concerns

## Technical Debt

### 1. Legacy Vanilla Scripts Still Present

**Location:** `prediction.py`, `units.py` (repository root)

**Issue:** Code duplication between legacy scripts and refactored `virnucpro/` package

**Impact:**
- Maintenance burden (changes needed in two places)
- Confusion about which code is canonical
- Repository clutter

**Why It Exists:** Preserved for validation testing (vanilla comparison)

**Fix Approach:**
- Archive to `legacy/` or `reference/` directory
- Document that refactored code is canonical
- Keep only for historical reference

**Priority:** Low (functionally correct, just organizational debt)

### 2. Imports Inside Functions

**Location:** `virnucpro/pipeline/prediction.py`

**Issue:** Import statements inside function bodies instead of at module level

**Example:**
```python
def run_prediction(...):
    import torch  # Should be at top of file
    from transformers import AutoModel
    ...
```

**Impact:**
- Non-standard Python pattern
- Harder to identify dependencies
- Potential performance cost (repeated imports)

**Why It Exists:** Delay expensive imports (PyTorch, Transformers) until needed

**Fix Approach:**
- Move all imports to module level
- Accept import time cost (minor)
- Benefits: clearer dependencies, standard pattern

**Priority:** Medium

### 3. Hardcoded Model Names

**Location:** `virnucpro/pipeline/feature_extraction.py`

**Issue:** Model names hardcoded despite config values existing

**Example:**
```python
model = AutoModel.from_pretrained("zhihan1996/DNABERT-S")  # Hardcoded
# Config has model_name field that's ignored
```

**Impact:**
- Can't swap models via configuration
- Must edit code to change models
- Config values misleading

**Fix Approach:**
- Read model names from config
- Use config values throughout
- Remove hardcoded strings

**Priority:** Medium

### 4. Manual Checkpoint State Management

**Location:** `virnucpro/core/checkpointing.py` (347 lines)

**Issue:** Complex manual checkpoint management logic

**Impact:**
- Fragile state tracking
- Easy to miss checkpoint updates
- Hard to add new pipeline stages

**Why It Exists:** No framework-level checkpointing in PyTorch

**Fix Approach:**
- Consider checkpoint framework (e.g., PyTorch Lightning)
- Or simplify current implementation
- Add comprehensive checkpoint tests

**Priority:** Low (works, but fragile)

### 5. No Type Hints

**Location:** Throughout codebase

**Issue:** Minimal type annotations

**Impact:**
- No type checking via mypy
- Harder to understand function contracts
- More runtime errors possible

**Fix Approach:**
- Add type hints incrementally
- Start with public API functions
- Configure mypy for gradual typing

**Priority:** Low (nice to have)

## Known Bugs

### 1. Empty File Check Missing for Resume

**Location:** `virnucpro/pipeline/parallel_feature_extraction.py`

**Symptom:** Resume fails if checkpoint exists but is empty/corrupted

**Reproducer:**
1. Interrupt during feature extraction write
2. Checkpoint file exists but incomplete
3. Resume attempts to load, crashes

**Workaround:** Delete checkpoint files and restart

**Root Cause:** No file size or content validation before `torch.load()`

**Fix:**
```python
if os.path.exists(checkpoint_file) and os.path.getsize(checkpoint_file) > 0:
    try:
        data = torch.load(checkpoint_file)
    except Exception as e:
        logger.warning(f"Corrupt checkpoint, regenerating: {e}")
        data = None
```

**Priority:** Medium

### 2. Multiprocessing Spawn Context Fragility

**Location:** `virnucpro/pipeline/parallel_feature_extraction.py`

**Symptom:** CUDA context errors in some multi-GPU scenarios

**Reproducer:** Run with `--device-ids 0,1,2,3` on certain GPU configurations

**Workaround:** Reduce number of GPUs or use `--device-ids 0`

**Root Cause:** CUDA can't be initialized before spawn in some configurations

**Fix:** Move more initialization into worker processes

**Priority:** Low (rare configuration)

### 3. Poor Error Diagnostic on IndexError

**Location:** Feature extraction and merging stages

**Symptom:** Generic IndexError without context

**Example:**
```
IndexError: list index out of range
```

**Root Cause:** Mismatched sequence counts between stages

**Fix:** Add context to index errors:
```python
try:
    feature = features[idx]
except IndexError:
    raise IndexError(
        f"Feature index {idx} out of range (have {len(features)} features)"
    ) from None
```

**Priority:** Low (improves debugging)

## Security Considerations

### 1. Unsafe Model Loading

**Location:** `virnucpro/pipeline/models.py`

**Issue:** `torch.load(..., weights_only=False)`

**Risk:** Allows arbitrary code execution when loading `.pth` files

**Attack Vector:**
- Malicious `.pth` file could execute code
- User loads untrusted model file

**Severity:** Medium (requires user to load untrusted model)

**Mitigation:**
- Use `weights_only=True` if possible (may break custom models)
- Validate model checksums
- Warn users about trusted models only
- Document security implications

### 2. Trust Remote Code in HuggingFace

**Location:** `virnucpro/pipeline/feature_extraction.py`

**Issue:** `trust_remote_code=True` for model loading

**Risk:** Remote code execution when loading models from HuggingFace

**Attack Vector:**
- Compromised HuggingFace model repository
- Malicious code in model's `modeling.py`

**Severity:** Low (reputable model source, low probability)

**Mitigation:**
- Pin model revisions/commit hashes
- Review model code before use
- Consider `trust_remote_code=False` if model supports it

### 3. No Input Sanitization

**Location:** FASTA file parsing

**Issue:** No validation of FASTA content beyond basic parsing

**Risk:** Maliciously crafted FASTA could cause issues

**Examples:**
- Extremely long sequence IDs (buffer overflow in C extensions)
- Special characters in IDs (injection if used in commands)
- Huge files (DoS via memory exhaustion)

**Severity:** Low (mostly academic concerns)

**Mitigation:**
- Add max sequence length checks
- Validate sequence IDs against safe character set
- Add file size limits

## Performance Bottlenecks

### 1. ESM-2 Extraction Not Parallelized

**Location:** `virnucpro/pipeline/feature_extraction.py`

**Issue:** ESM-2 feature extraction uses single GPU, while DNABERT-S uses multi-GPU

**Impact:**
- ESM-2 stage much slower than DNABERT-S
- Underutilizes available GPUs
- Bottleneck in pipeline

**Why:** ESM-2 model too large to replicate across GPUs easily

**Fix Approach:**
- Implement data parallelism for ESM-2
- Or use model parallelism for large batches
- Or accept single-GPU limitation

**Priority:** Medium (significant performance gain possible)

### 2. File I/O in Tight Loops

**Location:** Translation stage writes many small FASTA files

**Issue:** Each translated sequence written individually

**Impact:**
- Slow on networked filesystems
- Many small file operations
- I/O wait time

**Fix Approach:**
- Batch writes
- Write to single file with partitioning
- Use in-memory intermediate format

**Priority:** Low (works, but slow)

### 3. Redundant Sequence Parsing

**Location:** Between pipeline stages

**Issue:** FASTA files read/written between stages, parsed multiple times

**Impact:**
- Unnecessary I/O
- CPU time parsing same data
- Disk space for intermediate files

**Fix Approach:**
- Keep sequences in memory between stages
- Only checkpoint essentials, not full FASTA
- Use binary format for intermediate data

**Priority:** Low (trade-off: memory vs disk)

### 4. Large Uncompressed Intermediate Files

**Location:** `.pt` checkpoint files

**Issue:** Large uncompressed PyTorch tensor files

**Impact:**
- Disk space usage (GBs for large inputs)
- Slow checkpoint save/load
- Network transfer cost (if on shared storage)

**Fix Approach:**
- Compress tensors (gzip, lz4)
- Reduce precision if acceptable (float32 → float16)
- Incremental checkpointing

**Priority:** Low (disk is cheap)

## Fragile Areas

### 1. Checkpoint Resume Logic

**Location:** `virnucpro/core/checkpointing.py`

**Why Fragile:**
- Complex state machine
- Many edge cases (partial writes, corruption)
- Hard to test all scenarios

**Safe Modification Approach:**
- Add comprehensive checkpoint tests first
- Test resume from each stage
- Test partial file scenarios
- Don't refactor without tests

### 2. Multiprocessing Spawn

**Location:** `virnucpro/pipeline/parallel_feature_extraction.py`

**Why Fragile:**
- CUDA context issues
- Pickle limitations
- Platform-specific behavior (Windows vs Linux)

**Safe Modification Approach:**
- Test on all target platforms
- Minimize state passed to workers
- Add worker error handling
- Test CUDA context initialization

### 3. Feature Tensor Merging

**Location:** Feature merging stage

**Why Fragile:**
- Depends on exact alignment of nucleotide and protein features
- Index matching across 6 reading frames
- Easy to get index offsets wrong

**Safe Modification Approach:**
- Add alignment validation
- Test mismatched feature counts
- Add assertions for tensor shapes

### 4. Reading Frame Translation

**Location:** `virnucpro/utils/sequence_utils.py`

**Why Fragile:**
- Complex logic for 6 reading frames
- Reverse complement handling
- ORF detection interaction

**Safe Modification Approach:**
- Don't modify without vanilla comparison tests
- Test all 6 frames independently
- Test reverse complement specifically

## Scaling Limits

### 1. Single-Machine Limitation

**Current:** Runs on single machine only

**Scaling Limit:** Number of GPUs on one machine (typically 4-8)

**Scaling Path:**
- Distributed training frameworks (PyTorch Distributed)
- Split input across machines
- Aggregate results

### 2. Memory Constraints

**Current:** Loads all features into memory for merging

**Scaling Limit:** RAM size (fails on huge genomes)

**Scaling Path:**
- Streaming feature processing
- Process in batches, not all at once
- Memory-mapped tensors

### 3. No Batch Input

**Current:** Processes one FASTA file at a time

**Scaling Limit:** Manual iteration for multiple files

**Scaling Path:**
- Add batch mode (multiple input files)
- Parallel file processing
- Job queue system

### 4. No Incremental Updates

**Current:** Must reprocess entire file if anything changes

**Scaling Limit:** Slow for iterative analysis

**Scaling Path:**
- Sequence-level checksums
- Skip unchanged sequences
- Incremental prediction updates

## Dependency Risks

### 1. PyTorch Version Lock

**Current:** `>=2.8.0`

**Risk:** 2.8.0 very recent, may have bugs

**Migration Path:**
- Test on multiple PyTorch versions
- Relax constraint if possible
- Pin to known-good version

### 2. Transformers Version Pin

**Current:** `==4.30.0` (exact pin)

**Risk:** Old version, missing security fixes

**Migration Path:**
- Test with newer Transformers
- Update to `>=4.30.0,<5.0.0` range
- May need model code changes

### 3. ESM Library Maintenance

**Current:** `fair-esm==2.0.0`

**Risk:** Facebook Research may stop maintaining

**Migration Path:**
- Watch for deprecation notices
- Consider alternative protein models
- May need to vendor library

### 4. BioPython Breaking Changes

**Current:** No version constraint

**Risk:** Breaking changes in future versions

**Migration Path:**
- Pin BioPython to major version
- Test on BioPython updates
- May need parser updates

## Missing Critical Features

### 1. No Model Versioning

**Issue:** No tracking of which model version produced predictions

**Impact:** Can't reproduce results if models change

**Fix:** Include model checksum in output metadata

### 2. No Provenance Tracking

**Issue:** No record of pipeline parameters in output

**Impact:** Can't reproduce analysis

**Fix:** Write metadata file with full config, versions, checksums

### 3. No Validation Mode

**Issue:** No way to validate pipeline without running full analysis

**Impact:** Can't quickly check if setup works

**Fix:** Add `--validate` flag for quick sanity check

### 4. No Progress Estimation

**Issue:** Progress bars don't show time remaining

**Impact:** Users don't know how long to wait

**Fix:** Add ETA to progress bars

## Test Coverage Gaps

### 1. No Error Handling Tests

**Gap:** Don't test error paths

**Examples:**
- Malformed FASTA
- Corrupt checkpoints
- Missing model files
- Invalid config

**Fix:** Add negative tests

### 2. No GPU Tests in CI

**Gap:** GPU code paths untested in CI (if CI is CPU-only)

**Impact:** GPU bugs not caught early

**Fix:** Add GPU test runner or mock CUDA

### 3. No Checkpoint Resume Tests

**Gap:** Don't test resume from each stage

**Impact:** Resume bugs only found by users

**Fix:** Add resume tests for each stage

### 4. No Parallel Processing Tests

**Gap:** Multi-GPU code lightly tested

**Impact:** Race conditions, context errors not caught

**Fix:** Add multi-GPU test scenarios

## Recommendations Priority

**High Priority:**
1. Fix empty checkpoint file bug
2. Add model versioning to outputs
3. Add provenance tracking

**Medium Priority:**
1. Move imports to module level
2. Use config values for model names
3. Parallelize ESM-2 extraction
4. Add error handling tests

**Low Priority:**
1. Archive legacy scripts
2. Add type hints
3. Optimize file I/O
4. Add progress time estimation
