# Skip Chunking for Short Reads in VirNucPro

## Overview

VirNucPro currently chunks all input sequences regardless of their length. For datasets where all reads are below the model threshold (300bp or 500bp), this chunking step is unnecessary and wasteful. This plan implements logic to skip the chunking process when all input reads are shorter than the expected model length.

The chosen approach modifies the `split_fasta_chunk()` function to detect when all sequences are below the chunk size threshold. In such cases, the function copies the input file directly to the output path instead of performing chunking operations. This optimization maintains compatibility with the existing pipeline architecture while improving performance for short-read datasets.

Key decision: Conservative handling for mixed datasets - if ANY read exceeds the threshold, all reads are chunked normally to ensure uniform output format expected by downstream stages.

## Planning Context

### Decision Log

| Decision | Reasoning Chain |
|----------|-----------------|
| Chunking function optimization over orchestrator logic | User selected Option 2 -> Implementation directly in split_fasta_chunk() minimizes changes to pipeline architecture -> Single function modification easier to implement -> Performance benefit of single-pass processing without additional file scans |
| Conservative mixed-dataset handling | Models expect uniform-length inputs -> Mixed chunking (some chunked, some not) would create inconsistent output format -> Downstream stages (translation, feature extraction) assume homogeneous chunk structure -> If ANY sequence exceeds threshold, must chunk all to maintain consistency |
| File copy over symlink | Copy guarantees compatibility across all filesystems -> Symlinks may fail on network filesystems or Windows environments -> Symlinks risk corruption if original file is modified during pipeline -> Small I/O overhead acceptable for reliability guarantee |
| Pre-scan all sequences before chunking | Cannot make skip decision without knowing maximum sequence length -> Single-pass pre-scan provides complete information -> Max length determination is O(n) but unavoidable for informed decision -> Scan cost amortized against chunking cost savings |
| BioPython rewrite over direct copy | Downstream stages expect chunk naming convention with `_chunk_N` suffixes -> 6-frame translation and consensus scoring rely on chunk/frame suffix patterns -> Direct file copy preserves original IDs without suffixes -> Must rewrite sequences with `_chunk_1` suffix to maintain pipeline contract -> BioPython SeqIO parse/write adds minimal overhead compared to chunking cost |
| Extract pre-scan to separate function | Pre-scan logic adds 8-10 lines -> Decision and copy logic adds 10-15 lines -> Total would exceed 50-line god-function limit -> Extracting `_determine_max_sequence_length()` helper maintains single responsibility -> split_fasta_chunk() focuses on transformation decision, helper focuses on analysis |

### Rejected Alternatives

| Alternative | Why Rejected |
|-------------|--------------|
| Pipeline orchestrator integration (Option 1) | User selected Option 2 instead -> Would require modifying prediction.py and checkpoint system -> Higher implementation complexity with checkpoint state management -> Deferred to future refactoring if maintainability issues arise |
| Hybrid pre-scan approach (Option 3) | Sampling adds complexity without clear benefit for VirNucPro use case -> Biological datasets typically homogeneous within single run -> Extra complexity of sampling logic and fallback paths not justified -> Over-engineering for this specific optimization |
| Per-sequence chunking skip | Would create mixed output format (some sequences chunked, others not) -> Downstream stages expect uniform chunk structure -> 6-frame translation assumes consistent chunk naming conventions -> Consensus scoring relies on chunk/frame suffix patterns |
| Symlink instead of copy | Symlinks not universally supported across filesystems -> Windows compatibility issues -> Risk of source file modification during pipeline execution -> Small performance gain not worth compatibility risk |
| Pass original path to downstream stages | Would require modifying all downstream stage interfaces -> Translation stage expects chunked file naming -> Checkpoint system tracks chunked_file path -> Breaking change to pipeline contract |
| Direct file copy without ID rewriting | Would preserve original sequence IDs without `_chunk_1` suffix -> Breaks chunk naming convention expected by downstream stages -> 6-frame translation appends frame suffixes (F1-F3, R1-R3) expecting base chunk ID -> Consensus scoring groups by removing frame suffixes, assuming chunk naming pattern -> Pipeline failure or incorrect results |

### Constraints & Assumptions

**Technical Constraints:**
- Must maintain existing pipeline interface: `split_fasta_chunk(input_file, output_file, chunk_size)`
- Output file must exist at expected path regardless of skip decision
- Downstream stages (translation, feature extraction) expect output_file to exist with chunk naming convention
- Cannot modify checkpoint system or pipeline orchestrator in this change
- Must preserve FASTA format and add `_chunk_1` suffix to sequence IDs for pipeline compatibility

**Assumptions:**
- Input FASTA files are well-formed (BioPython can parse them)
- Chunk size parameter matches model's expected_length (300 or 500)
- Homogeneous datasets (all short or all long) are common use case
- File system supports shutil.copy2 operations
- Sufficient disk space for file copy operation

**Default Conventions Applied:**
- `<default-conventions domain="god-function">`: Function stays under 50 lines with single responsibility
- `<default-conventions domain="file-creation">`: Extend existing file (sequence.py), not creating new utility module
- Testing skipped per user specification (no tests milestone)

### Known Risks

| Risk | Mitigation | Anchor |
|------|------------|--------|
| Large files: copying multi-GB FASTA files is slow | Accepted: Skip optimization targets small short-read datasets where copy is fast. Large files likely have long reads requiring chunking anyway. Future: could add size threshold check. | N/A - hypothetical future scenario |
| Memory usage: pre-scan loads all sequence lengths | Low risk: only tracking integer lengths, not sequence data. BioPython SeqIO.parse() streams records, not loading entire file into memory. Max memory: O(n) integers where n = number of sequences. | virnucpro/utils/sequence.py:L169 - existing code already iterates all sequences |
| Mixed datasets: conservative chunking may be inefficient | Accepted: consistency and correctness prioritized over optimization for edge case. Mixed-length datasets uncommon in typical sequencing workflows. | N/A - design tradeoff |
| File copy failure: disk space or permissions | Existing risk: original chunking also writes output file. Copy operation uses same filesystem as chunking would. Python shutil.copy2 raises clear IOError on failure. No new risk introduced. | virnucpro/utils/sequence.py:L154 - existing code opens output file for writing |

## Invisible Knowledge

### Architecture

The chunking optimization sits at a critical decision point in the VirNucPro pipeline:

```
Input FASTA
    |
    v
split_fasta_chunk()  <-- OPTIMIZATION HERE
    |
    +-- All reads <= chunk_size? --> Copy file directly
    |
    +-- Any read > chunk_size? --> Chunk with overlaps
    |
    v
Chunked FASTA
    |
    v
6-frame Translation --> Feature Extraction --> MLP Prediction
```

The optimization is invisible to the pipeline orchestrator and downstream stages. Both paths (chunking and copying) produce a file at the expected output path with valid FASTA format.

### Data Flow

**Pre-scan phase:**
1. Iterate all sequences to determine max_length
2. Track count for logging purposes

**Decision phase:**
3. If max_length <= chunk_size: enter copy path
4. If max_length > chunk_size: enter chunking path (original logic)

**Copy path:**
5. Use shutil.copy2 to duplicate input → output
6. Preserve all metadata (timestamps, permissions)
7. Log skip decision with statistics

**Chunking path (unchanged):**
8. Calculate overlapping chunks per sequence
9. Write chunks with _chunk_N suffix naming

### Why This Structure

The split_fasta_chunk() function is the natural boundary for this optimization because:

1. **Single Responsibility**: Function's job is "produce chunked output at output_file path" - copying is a valid implementation strategy for that contract
2. **Minimal Ripple Effects**: Pipeline orchestrator doesn't need to know if chunking actually occurred
3. **Checkpoint Compatibility**: Checkpoint system tracks output file existence, not how it was created
4. **Error Handling**: Function already has logging infrastructure for reporting chunking statistics

Alternative structures (orchestrator-level decision, separate validator) would require:
- Modifying checkpoint state machine
- Adding new pipeline stage infrastructure
- Threading skip decision through multiple layers
- Breaking existing stage encapsulation

### Invariants

**Function Contract Invariants (must maintain):**
1. Output file MUST exist at output_file path after function completes
2. Output file MUST contain valid FASTA format
3. Output file MUST be compatible with downstream translation stage
4. Function MUST log statistics about operation performed

**Chunking Logic Invariants (unchanged):**
5. If chunking occurs, chunk naming follows pattern: `{original_id}_chunk_{N}`
6. Overlaps distributed evenly to maintain chunk_size consistency
7. All chunks from same sequence have identical chunk_size (except possibly last)

**New Skip Logic Invariants:**
8. Skip decision based on max_length across ALL sequences
9. Either all sequences skipped OR all sequences chunked (no mixing)
10. Skipped output preserves exact input format (headers, line breaks, ordering)

### Tradeoffs

**Performance vs. Maintainability:**
- **Chose**: Single-function modification (lower maintainability complexity)
- **Over**: Orchestrator-level architecture (better separation of concerns)
- **Because**: User selected Option 2 for lower implementation effort
- **Cost**: Future refactoring may need to move logic to orchestrator layer
- **Benefit**: Faster implementation, no checkpoint system changes needed

**Consistency vs. Per-Sequence Optimization:**
- **Chose**: All-or-nothing chunking (chunk all if any exceeds threshold)
- **Over**: Per-sequence chunking decisions
- **Because**: Downstream stages expect uniform chunk structure
- **Cost**: Mixed datasets get no optimization even if 90% are short reads
- **Benefit**: Guaranteed output format consistency, simpler logic

**Copy vs. Symlink:**
- **Chose**: File copy (slower but safer)
- **Over**: Symlink (faster but fragile)
- **Because**: Compatibility and robustness over performance
- **Cost**: I/O overhead for file copy operation
- **Benefit**: Works on all filesystems, immune to source modifications

**Pre-scan Cost:**
- **Chose**: Full pre-scan before decision
- **Over**: Streaming decision per sequence
- **Because**: Cannot make correct skip decision without complete information
- **Cost**: O(n) pre-scan iteration over all sequences
- **Benefit**: Correct handling of mixed datasets, clear decision logic

## Milestones

### Milestone 1: Implement Skip-Chunking Logic

**Files**:
- virnucpro/utils/sequence.py

**Flags**:
- `needs-rationale`: Threshold decision (chunk_size comparison) needs WHY comment
- `performance`: File copy operation on potentially large files

**Requirements**:
- Extract pre-scan logic to `_determine_max_sequence_length()` helper function
- Pre-scan all input sequences to determine maximum sequence length
- If max_length <= chunk_size: rewrite sequences with `_chunk_1` suffix to output path
- If max_length > chunk_size: execute existing chunking logic
- Log skip decision with sequence count and max length statistics
- Maintain chunk naming convention expected by downstream stages

**Acceptance Criteria**:
- Helper function `_determine_max_sequence_length()` exists and returns correct (max_length, sequence_count)
- Function returns successfully for all-short-reads datasets with output file created
- Function returns successfully for mixed or all-long-reads datasets with chunked output
- Output file exists at output_file path in both skip and chunk cases
- All sequence IDs in skip path have `_chunk_1` suffix appended
- Logged messages distinguish between skip and chunk operations with sequence counts
- Skipped sequences maintain chunk naming convention compatible with downstream stages

**Tests**:
Testing skipped per user specification. No test milestone required.

**Code Intent**:

Modify `virnucpro/utils/sequence.py`:

1. **Add helper function `_determine_max_sequence_length()`** (before split_fasta_chunk):
   - Function signature: `def _determine_max_sequence_length(input_file: Path) -> Tuple[int, int]:`
   - Returns tuple of (max_length, sequence_count)
   - Initialize max_length = 0, sequence_count = 0
   - Wrap SeqIO.parse iteration in try/except to catch BioPython parse errors
   - Iterate all records using SeqIO.parse(input_file, 'fasta')
   - Track max_length = max(max_length, len(record.seq))
   - Track sequence_count += 1
   - On parse exception, log "Pre-scan failed: {error}" before re-raising
   - Return (max_length, sequence_count)
   - Add docstring explaining: "Pre-scan input sequences to determine maximum length for skip-chunking decision"

2. **Modify `split_fasta_chunk()` to call helper and decide**:
   - Call helper: `max_length, sequence_count = _determine_max_sequence_length(input_file)`
   - If max_length <= chunk_size:
     - Log: "Skipping chunking: {sequence_count} sequences, max length {max_length}bp <= {chunk_size}bp threshold"
     - Wrap file rewrite in try/except for error handling
     - Open output_file for writing
     - Iterate records from input_file using SeqIO.parse(input_file, 'fasta')
     - Modify each record.id to append "_chunk_1" suffix: `record.id = f"{record.id}_chunk_1"`
     - Write modified record to output_file using SeqIO.write()
     - On success, log: "Rewrote {sequence_count} sequences with chunk_1 suffix"
     - On exception, log: "Skip failed during file rewrite: {error}" before re-raising
     - Return early (exit function)

3. **Existing chunking logic**:
   - If max_length > chunk_size, continue with existing chunking implementation
   - No changes to chunking algorithm, overlap calculation, or output format
   - Existing logging for chunking statistics remains unchanged

**Key Decisions**:
- Pre-scan extraction: Extract to `_determine_max_sequence_length()` helper (Decision: "Extract pre-scan to separate function")
- Chunk naming preservation: Rewrite with `_chunk_1` suffix using BioPython (Decision: "BioPython rewrite over direct copy")
- Pre-scan cost: Accepted O(n) iteration to make informed decision (Decision: "Pre-scan all sequences before chunking")
- All-or-nothing: chunk all if any exceeds threshold (Decision: "Conservative mixed-dataset handling")

**Code Changes**:

(This section will be filled by Developer agent in step 8 of the planning workflow. Developer will read the actual file and produce unified diffs with context lines.)

## Milestone Dependencies

```
M1 (standalone - no dependencies)
```

Single milestone implementation with no cross-milestone dependencies.
