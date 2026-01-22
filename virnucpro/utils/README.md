# virnucpro/utils

Sequence processing utilities for VirNucPro pipeline.

## Skip-Chunking Optimization

The `split_fasta_chunk()` function implements a performance optimization that skips chunking operations when all input sequences are shorter than the chunk size threshold (300bp or 500bp). This optimization is invisible to downstream pipeline stages.

### Why This Architecture

The optimization sits inside `split_fasta_chunk()` rather than at the pipeline orchestrator level because:

1. **Function Contract**: The function's responsibility is "produce valid output at output_file path" - copying short sequences is a valid implementation of this contract
2. **Minimal Ripple**: Pipeline orchestrator (`prediction.py`) and checkpoint system don't need modification
3. **Stage Encapsulation**: Decision logic stays within the transformation boundary, not leaked to orchestration layer
4. **Logging Infrastructure**: Function already has structured logging for reporting chunking statistics

Alternative architectures (orchestrator-level decision, separate validator stage) would require modifying the checkpoint state machine and threading the skip decision through multiple pipeline layers.

### Pipeline Contract Invariants

Downstream stages (6-frame translation, feature extraction, consensus scoring) depend on these guarantees:

1. **Output Existence**: File MUST exist at `output_file` path after function completes
2. **FASTA Format**: Output MUST be valid FASTA parseable by BioPython
3. **Chunk Naming**: Sequence IDs MUST follow `{original_id}_chunk_{N}` pattern
4. **Translation Compatibility**: 6-frame translation appends frame suffixes (F1-F3, R1-R3) to chunk IDs
5. **Consensus Grouping**: Consensus scoring strips frame suffixes to group by chunk ID
6. **Format Uniformity**: Either ALL sequences chunked OR ALL sequences treated as single chunks (no mixing)

The skip optimization preserves these invariants by:
- Rewriting short sequences with `_chunk_1` suffix (satisfies naming convention)
- Using BioPython SeqIO.write (guarantees valid FASTA format)
- All-or-nothing decision logic (maintains uniformity)

### Decision Logic

`split_fasta_chunk()` orchestrates three helper functions based on skip decision:

**Pre-scan Phase:**
```python
# _determine_max_sequence_length() - streams all records to find max length
max_length, sequence_count = _determine_max_sequence_length(input_file)
```

**Decision Phase:**
```python
if max_length <= chunk_size:
    # Skip path: copy with _chunk_1 suffix
    _copy_with_chunk_suffix(input_file, output_file, sequence_count)
else:
    # Chunk path: overlapping chunking algorithm
    _chunk_with_overlaps(input_file, output_file, chunk_size)
```

**Conservative Handling**: If ANY sequence exceeds `chunk_size`, ALL sequences are chunked to maintain uniform output format. Mixed chunking (some chunked, some not) would break downstream stage assumptions.

**Helper Responsibilities:**
- `_determine_max_sequence_length()`: Single-pass pre-scan for informed skip decision
- `_copy_with_chunk_suffix()`: Atomic rewrite preserving chunk naming contract
- `_chunk_with_overlaps()`: Original overlap distribution algorithm for long sequences

### Performance Tradeoffs

**Pre-scan Cost:**
- **Accepted**: O(n) iteration over all sequences before decision
- **Rationale**: Cannot make correct skip decision without knowing maximum sequence length
- **Mitigation**: BioPython SeqIO.parse() streams records (low memory footprint)

**File Copy vs. Symlink:**
- **Chose**: File copy with BioPython rewrite
- **Over**: Symlink (faster but incompatible with chunk naming convention)
- **Cost**: I/O overhead for copy operation and ID rewriting
- **Benefit**: Preserves chunk naming contract, works on all filesystems

**All-or-Nothing vs. Per-Sequence:**
- **Chose**: All-or-nothing (chunk all if any exceeds threshold)
- **Over**: Per-sequence decisions (better optimization for mixed datasets)
- **Cost**: No optimization for datasets with 90% short reads + 10% long reads
- **Benefit**: Guaranteed output uniformity, simpler logic, maintains pipeline contract

### Known Risks

**Large File Copying**: Copying multi-GB FASTA files is slow, but accepted because:
- Optimization targets small short-read datasets (copy is fast)
- Large files likely contain long reads requiring chunking anyway
- Future enhancement could add file size threshold check

**Memory Usage**: Pre-scan tracks integer lengths for all sequences (O(n) integers). Low risk because BioPython SeqIO.parse() streams records without loading entire file into memory.

**Mixed Datasets**: Conservative chunking may skip optimization for datasets with few long outliers. Accepted tradeoff prioritizing correctness over edge-case performance.
