# Phase 3: Checkpoint Robustness - Context

**Gathered:** 2026-01-23
**Status:** Ready for planning

<domain>
## Phase Boundary

Ensure checkpoint system prevents corruption, validates integrity, supports resume from pre-optimization runs, and maintains backward compatibility. Focus is on data integrity, error recovery, and version compatibility for long-running jobs (8+ hours) that need reliable resume capability.

</domain>

<decisions>
## Implementation Decisions

### Validation Depth
- **Validation level:** Basic structure check (verify PyTorch format + expected keys + tensor existence)
- **Validation timing:** Both write-time and load-time for maximum safety
- **Fast-path option:** `--skip-checkpoint-validation` flag for trusted scenarios
- **Logging:** Detailed diagnostic logs showing exactly what failed (missing keys, tensor info, file paths)
- **Error types:** Distinguish between 'corrupted' (broken file) and 'incompatible' (wrong version/format)
- **Partial data:** Fail validation entirely if any required component is missing (no partial recovery)
- **Dry-run mode:** `virnucpro validate-checkpoints <dir>` command reports status without processing

### Recovery Behavior
- **Corruption handling:** Fail immediately with error when corrupted checkpoint detected (no auto-recovery)
- **Override capability:** `--force-resume` flag to skip bad checkpoints and continue with others
- **Progress tracking:** `failed_checkpoints.txt` log with format: `{checkpoint_path}|{reason}|{timestamp}`
- **Exit code:** Exit code 3 specifically for checkpoint issues (distinct from exit code 2 for embedding failures)

### Backward Compatibility
- **Pre-optimization checkpoints:** Load as-is (read-only) without modifying old format
- **Version field:** Semantic version (1.0, 1.1, etc.) embedded in checkpoint metadata
- **Current version:** 1.0 for optimized checkpoints (pre-optimization = no version field, treated as 0.x)
- **Future versions:** Fail with upgrade message when encountering incompatible future versions (e.g., "Checkpoint v2.0 requires virnucpro >= X.Y.Z")

### Progress Visibility
- **Completion tracking:** Metadata field `status: complete` inside checkpoint file (no external .done markers)
- **Metadata content:** Minimal metadata (version + status only)
- **Resume logging:** Summary at start (e.g., "Found 15/20 checkpoints, resuming 5 files")
- **Status commands:** No separate status command — resume logs provide sufficient visibility

### Claude's Discretion
- Specific validation error messages and formatting
- Internal checkpoint file structure beyond version/status metadata
- Temporary file naming for atomic writes
- Optimization of validation performance for large checkpoint files

</decisions>

<specifics>
## Specific Ideas

- Validation should catch "dimension mismatches from code changes" without being overly strict on exact shapes
- Failed checkpoints log should match the `failed_files.txt` pattern from Phase 1 for consistency
- Exit code scheme should be clear: 0 = success, 1 = generic failure, 2 = partial pipeline success, 3 = checkpoint issue
- Version 1.0 represents the "baseline format" after optimization work, with pre-optimization being implicit 0.x

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 03-checkpoint-robustness*
*Context gathered: 2026-01-23*
