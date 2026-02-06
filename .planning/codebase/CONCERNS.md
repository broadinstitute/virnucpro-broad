# Codebase Concerns

**Analysis Date:** 2026-02-06

## Tech Debt

**Massive code duplication in sequence processing logic:**
- Issue: `identify_seq()` function in `units.py` contains duplicated logic for refseq-based and non-refseq processing paths (lines 81-146). The two branches differ only slightly but share 90% of their code.
- Files: `units.py` (lines 81-146)
- Impact: Makes maintenance difficult, increases risk of inconsistent behavior between paths, future bug fixes must be applied in multiple places
- Fix approach: Extract the common logic into a shared helper function and use conditional flags to control refseq filtering behavior

**Procedural script architecture instead of modular functions:**
- Issue: `features_extract.py` (241 lines) contains massive duplication of file listing and processing patterns repeated 7 times for different data categories (vertebrate, protozoa, plant, invertebrate, fungi, bacteria, archaea).
- Files: `features_extract.py` (lines 99-202)
- Impact: Extremely difficult to modify processing pipeline, scaling to new categories requires copy-pasting 15+ lines, high risk of inconsistent handling
- Fix approach: Extract the repeated pattern into a factory function that takes category name as parameter

**Undefined variable usage with implicit imports:**
- Issue: `features_extract.py` imports `from units import *` (line 1) creating hidden dependencies on imported symbols like `os`, `torch`, `SeqIO`, `AutoTokenizer`, etc.
- Files: `features_extract.py` (line 1), `prediction.py` (line 1), `make_train_dataset_300.py` (line 1)
- Impact: Makes code hard to understand, IDE cannot provide accurate completions/refactorings, importing new items to `units.py` silently affects all dependents
- Fix approach: Replace wildcard imports with explicit imports or consolidate into proper module namespace

**Hard-coded path patterns throughout codebase:**
- Issue: File path transformations use string replacements scattered across functions: `replace('./data/', './data/data_merge/')`, `replace('.identified_nucleotide', '_merged')`, etc.
- Files: `features_extract.py` (lines 17, 27, 44, 100-102, 114-116, 129-131, 144-146, 160-162, 175-177, 190-192, 230-232), `prediction.py` (lines 100-105, 154-157, 162), `units.py` (lines 302-322)
- Impact: Makes it impossible to change directory structure without hunting through code, fragile to any filename conventions changes, hard to test with different layouts
- Fix approach: Create a configuration object or environment variable-based path management system

**Global state in feature extraction module:**
- Issue: `features_extract.py` loads tokenizer and model globally at module level (lines 9-12), keeping them in memory for entire execution.
- Files: `features_extract.py` (lines 9-12)
- Impact: Memory inefficient, makes testing difficult, cannot process different models in same session, global state persists across function calls
- Fix approach: Pass models as function parameters or use a context manager/class-based approach

**Hardcoded model paths and dimensions:**
- Issue: Model input dimensions and hidden dimensions are hard-coded in multiple files: `input_dim = 3328` in `train.py` (line 105), `hidden_dim = 512` (line 106). DNA-BERT model path and ESM model path hard-coded in `units.py` (lines 164-165, 214).
- Files: `train.py` (lines 105-107), `units.py` (lines 164-165, 214), `prediction.py` (line 168)
- Impact: Cannot easily change model architecture, switching models requires code changes, inconsistent dimensions could cause silent failures
- Fix approach: Move to configuration file (e.g., YAML or JSON) loaded at runtime

**Missing error handling in critical paths:**
- Issue: File I/O operations and model loading lack try-except blocks: `torch.load()` in `train.py` (line 24), `prediction.py` (line 168), `SeqIO.parse()` in multiple files, network requests in `download_data.py`.
- Files: `train.py` (line 24), `prediction.py` (line 168), `units.py` (lines 180, 222), `download_data.py` (lines 21, 37), `make_train_dataset_300.py` (lines 25, 71)
- Impact: Script crashes without helpful error messages, incomplete data processing silently fails, network issues in downloader cause unrecoverable state
- Fix approach: Add comprehensive try-except blocks with informative error messages and cleanup handlers

## Known Bugs

**IndexError in dataset __getitem__ never properly handled:**
- Symptoms: If DataLoader requests index beyond dataset size, IndexError raised but not caught by caller
- Files: `train.py` (line 41), `prediction.py` (line 45)
- Trigger: Setting batch_size too large for remaining data or requesting indices beyond cumulative_size calculation
- Workaround: Ensure batch_size divides evenly into total dataset size

**File handle leak in split_fasta_file:**
- Symptoms: If output file creation fails mid-loop, current_output_file may not be closed, leaking file handles
- Files: `units.py` (lines 267-288)
- Trigger: Disk full or permission error while writing FASTA output
- Workaround: None - will require script restart

**Incorrect strand calculation in DNA translation:**
- Symptoms: Reverse strand sequences may be incorrect due to off-by-one errors in slice indices
- Files: `units.py` (lines 98-101, 117-119, 138-139)
- Trigger: Use reverse complement strands (frames 4-6)
- Workaround: Manually validate reverse strand outputs

**Hardcoded path separators causing cross-platform issues:**
- Symptoms: Scripts fail on Windows due to `\` vs `/` path separators
- Files: `features_extract.py` (lines 44-50, 51, etc.), `prediction.py` (lines 100-105)
- Trigger: Running on Windows with hardcoded `./data/` paths
- Workaround: Use Linux/Mac only

## Security Considerations

**Arbitrary file deserialization with torch.load():**
- Risk: Using `torch.load(..., weights_only=False)` deserializes arbitrary Python objects, potential arbitrary code execution if model file is compromised
- Files: `train.py` (line 24), `prediction.py` (line 168), `units.py` (lines 209, 292-293)
- Current mitigation: Assumes model files are trusted (local only)
- Recommendations: Use `weights_only=True` if possible, validate model file integrity with checksums, document assumption that model files are trusted sources

**Network requests without timeout or validation:**
- Risk: `download_data.py` makes requests to NCBI without timeouts; could hang indefinitely or be vulnerable to redirect attacks
- Files: `download_data.py` (lines 21, 37, 72)
- Current mitigation: None
- Recommendations: Add request timeout, validate URLs, use HTTPS only, optionally check file checksums after download

**No input validation on FASTA files:**
- Risk: Script assumes FASTA files are well-formed; malformed files could cause unexpected behavior or memory exhaustion
- Files: `units.py` (lines 11, 180, 222), `prediction.py` (lines 113, 166), all dataset loading code
- Current mitigation: None
- Recommendations: Add FASTA validation before processing, check file size limits, handle invalid characters gracefully

**Command-line arguments not validated:**
- Risk: `prediction.py` and `train.py` accept file paths and model specifications from command-line without validation
- Files: `prediction.py` (lines 193-195), `train.py` (implicit path construction)
- Current mitigation: None
- Recommendations: Validate file existence before processing, restrict to allowed directories, use pathlib for path handling

## Performance Bottlenecks

**Inefficient file pattern matching in dataset construction:**
- Problem: Repeated `os.listdir()` + filtering + `SeqIO.parse()` count operations on large directories
- Files: `features_extract.py` (lines 45-51, 61-66, 105-110, 120-125, 135-140, 150-155, 166-171, 181-186, 196-201)
- Cause: Counts sequences by parsing entire file just to check if `sequences_per_file` threshold met
- Improvement path: Cache file statistics, use file size heuristics, or count once and store result

**Dataset indexing is O(n) for each access:**
- Problem: `__getitem__` iterates through all data tensors to find correct index (quadratic for dataset iteration)
- Files: `train.py` (lines 33-41), `prediction.py` (lines 38-44)
- Cause: Cumulative iteration instead of pre-computed index mapping
- Improvement path: Build `cumsum()` array once at initialization, use binary search or direct lookup

**Model loaded per-file in feature extraction loop:**
- Problem: Even with `model_loaded=True`, models reloaded for each file due to parameter passing issues
- Files: `features_extract.py` (lines 21, 31) - models initialized once but scope unclear
- Cause: Global model variables created but logic suggests per-file loading path exists
- Improvement path: Clearly document and verify model persistence across loop iterations

**No batching in sequence translation:**
- Problem: DNA translation happens per-record, no vectorization
- Files: `units.py` (lines 69-72, 127-145)
- Cause: String-based translation with dict lookups, not optimized for batch processing
- Improvement path: Pre-compute all translation frames at batch level, use NumPy for vectorization

**Memory-inefficient dataset loading:**
- Problem: All data loaded into memory at once: `self._load_all_data()` loads entire feature tensors into lists
- Files: `train.py` (lines 22-28), `prediction.py` (lines 27-33)
- Cause: No lazy loading or memory mapping
- Improvement path: Use memory-mapped files, implement lazy loading with caching, or stream from disk

## Fragile Areas

**Complex sequence identification logic with duplicated paths:**
- Files: `units.py` (lines 81-146), `prediction.py` (lines 9-13), `features_extract.py` (lines 5-10)
- Why fragile: Six different processing paths (3 forward frames + 3 reverse frames) with manual index arithmetic (e.g., `num-1`, `num-3-1`, `num-3`), strand detection, and conditional slicing all in one function
- Safe modification: Add comprehensive unit tests for each frame/strand combination, extract frame processing into separate testable functions
- Test coverage: No test files exist; identify_seq() function untested

**Early stopping logic in training:**
- Files: `train.py` (lines 120-142, 167)
- Why fragile: EarlyStopping stores model state in memory but comparison logic uses loss value directly without smoothing or patience reset conditions clearly documented
- Safe modification: Add detailed comments explaining patience counter behavior, consider extracting to separate file for reusability
- Test coverage: No test files exist; EarlyStopping logic untested

**Data grouping and virus determination in prediction pipeline:**
- Files: `prediction.py` (lines 15-18, 183-189)
- Why fragile: `determine_virus()` extracts last 2 characters to create Modified_ID through string slicing (`str[:-2]`), assumes consistent ID format
- Safe modification: Add ID format validation before grouping, document assumed format explicitly
- Test coverage: No test files exist; grouping logic untested

**File path construction with string replacement:**
- Files: All main files use patterns like `.replace('identified_nucleotide', 'merged').replace('.fa', '_merged.pt')`
- Why fragile: Multiple sequential replacements are order-dependent and break if any filename has unexpected format
- Safe modification: Use `pathlib.Path` with explicit stem/suffix manipulation instead of string replacements
- Test coverage: No test files exist; path logic untested

## Scaling Limits

**Memory consumption with large datasets:**
- Current capacity: All tensors kept in memory simultaneously; on GPU this limits batch size
- Limit: 16GB+ dataset sizes will cause OOM errors with current architecture
- Scaling path: Implement lazy loading with sequential data access or use PyTorch IterableDataset with data generator

**Multiprocessing pool fixed to hardcoded process counts:**
- Current capacity: 8 processes for nucleotide extraction, 2 for protein extraction (lines 55, 69 in `features_extract.py`)
- Limit: Suboptimal on different hardware; single-core systems will thrash, 64-core systems underutilized
- Scaling path: Make process count configurable, detect CPU count with `multiprocessing.cpu_count()`

**Single model per category in training:**
- Current capacity: One MLP model with fixed 3328 input dimensions serves all sequence lengths
- Limit: Two separate models (300bp and 500bp) are maintained - code cannot adapt to new lengths
- Scaling path: Make model dimensions configurable, support dynamic input feature size detection

**FASTA file chunking hardcoded to 10,000 sequences:**
- Current capacity: `sequences_per_file = 10000` (hardcoded in multiple files)
- Limit: Large datasets with millions of sequences create thousands of intermediate files
- Scaling path: Make chunk size configurable, implement streaming batching

## Dependencies at Risk

**transformers==4.30.0 pinned to old version:**
- Risk: May contain known security vulnerabilities, missing bug fixes, incompatible with newer torch versions
- Files: `requirements.txt` (line 2)
- Impact: Cannot easily upgrade without testing, potential security exposure, incompatibility with CUDA 12.x
- Migration plan: Upgrade to 4.43.0+ (current as of 2025), verify DNABERT-S model compatibility

**fair-esm==2.0.0 hardcoded model location:**
- Risk: Model URL may change, package may be deprecated, model file size (3GB+) not documented
- Files: `units.py` (line 214), `features_extract.py` (line 12)
- Impact: Setup fails silently if model server is down, no fallback or caching strategy
- Migration plan: Document minimum disk space requirements, implement model download verification, consider huggingface_hub caching

**biopython functionality concentrated in units.py:**
- Risk: Heavy dependence on SeqIO and SeqRecord interfaces; version changes could break parsing
- Files: `units.py` (lines 1, 9-52), all files that import from units
- Impact: All scripts fail if biopython version incompatible
- Migration plan: Add version pin and compatibility tests, consider abstracting SeqIO interface

**Missing pinned versions for torch, numpy:**
- Risk: Major version changes (torch 1.13 -> 2.0) change API and behavior
- Files: `requirements.txt` (lines 5, 8)
- Impact: Installation on different dates produces incompatible environments
- Migration plan: Pin all dependencies to specific versions, use pixi.lock (partially done) for reproducibility

## Missing Critical Features

**No logging system:**
- Problem: Debugging failures requires re-running entire pipeline; progress tracking is print-statement-based
- Blocks: Cannot diagnose production failures without re-running, no audit trail, cannot suppress verbose output
- Recommendation: Add logging module with DEBUG/INFO/WARNING/ERROR levels, log to file, make verbosity configurable

**No checkpointing or resumption:**
- Problem: If feature extraction fails at file 500/1000, must restart from file 1
- Blocks: Long-running jobs cannot be interrupted/resumed, recovery from failures is inefficient
- Recommendation: Save processed file list, skip completed files on restart, implement checkpoint system

**No configuration file support:**
- Problem: All parameters (batch size, learning rate, model paths) hardcoded, cannot change without editing code
- Blocks: Cannot run different experiments without file edits, no easy way to document hyperparameters
- Recommendation: Create config.yaml with all parameters, load at startup, document in README

**No model validation or metadata:**
- Problem: No way to know which dataset a model was trained on, what version of features it expects
- Blocks: Cannot verify model/data compatibility, cannot reuse trained models across variants
- Recommendation: Save model metadata (training date, data version, input dims, training config)

**No input data validation:**
- Problem: Silently accepts incomplete data, malformed sequences, missing files
- Blocks: Cannot detect data quality issues early, long pipelines may fail deep in processing
- Recommendation: Add data validation stage at entry point, check file formats, size constraints

## Test Coverage Gaps

**No unit tests for core functions:**
- What's not tested: `translate_dna()`, `identify_seq()`, `reverse_complement()`, all sequence processing logic
- Files: `units.py` (lines 50-146)
- Risk: Critical sequence translation could have off-by-one errors, strand detection bugs, or character handling issues
- Priority: High - this is core to correctness

**No tests for dataset classes:**
- What's not tested: `FileBatchDataset.__getitem__()`, `PredictDataBatchDataset.__getitem__()` index calculations
- Files: `train.py` (lines 15-41), `prediction.py` (lines 20-45)
- Risk: IndexError bugs, off-by-one errors in cumulative indexing, data leakage between indices
- Priority: High - directly affects model correctness

**No tests for model training logic:**
- What's not tested: `train_model()`, `test_model()`, metric calculations, loss tracking
- Files: `train.py` (lines 144-218)
- Risk: Early stopping could trigger prematurely, validation metrics could be incorrectly computed
- Priority: High - affects model quality

**No tests for file I/O operations:**
- What's not tested: `split_fasta_file()`, `split_fasta_chunk()`, `merge_data()` correctness
- Files: `units.py` (lines 9-36, 267-288, 290-323)
- Risk: File handle leaks, incomplete writes, merged data corrupted
- Priority: Medium - affects pipeline robustness

**No integration tests:**
- What's not tested: End-to-end prediction pipeline, feature extraction with real models
- Files: All files in sequence
- Risk: Component tests pass but system fails, cannot catch interface mismatches
- Priority: Medium - important for production reliability

**No tests for edge cases:**
- What's not tested: Empty sequences, single-character sequences, sequences with ambiguous bases, very long sequences
- Risk: Production data triggers unhandled exceptions or silent failures
- Priority: Medium - affects robustness

---

*Concerns audit: 2026-02-06*
