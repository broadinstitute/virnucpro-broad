# Phase 3: Checkpoint Robustness - Research

**Researched:** 2026-01-23
**Domain:** PyTorch checkpoint integrity, atomic file operations, versioning
**Confidence:** HIGH

## Summary

PyTorch checkpoint robustness for long-running ML pipelines requires three core capabilities: atomic writes to prevent corruption, validation to detect incomplete/corrupted files, and versioning to support backward compatibility. The codebase already implements atomic writes in two locations (checkpoint.py and features.py) using the temp-then-rename pattern with `pathlib.Path.replace()`. Research confirms this is the standard approach, with PyTorch-Ignite and PyTorch Lightning both using `.part` or `.tmp` extensions for temporary files.

Validation is critical because PyTorch checkpoints are ZIP archives (using Python's pickle format), and corruption is common in HPC/long-running jobs. Standard validation checks file size >0, verifies ZIP format with `zipfile.is_zipfile()`, and optionally validates state_dict keys. The codebase's current 8+ hour ESM-2 runs make checkpoint integrity essential for resume capability.

Versioning enables backward compatibility by embedding a version field in checkpoint metadata. PyTorch Lightning uses on-the-fly migration when loading older checkpoints. The user decisions specify semantic versioning (1.0, 1.1), with pre-optimization checkpoints treated as 0.x (no version field). Validation should distinguish "corrupted" (broken file) from "incompatible" (wrong version/format).

**Primary recommendation:** Extend existing atomic write pattern to all checkpoint saves, add multi-level validation (file size → ZIP format → keys → tensors), embed version metadata in checkpoint dict, and implement detailed error logging with exit code 3 for checkpoint-specific failures.

## Standard Stack

The established libraries/tools for PyTorch checkpoint management:

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| PyTorch | 2.x | torch.save/torch.load checkpoint I/O | Native checkpoint serialization using pickle/ZIP format |
| pathlib | stdlib | Atomic file operations | Path.replace() provides atomic rename on POSIX |
| zipfile | stdlib | Checkpoint format validation | PyTorch checkpoints are ZIP archives, validates integrity |
| logging | stdlib | Diagnostic checkpoint errors | Structured logging for corruption/validation failures |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| json | stdlib | Version metadata storage | Embedding version/status in checkpoint dict |
| hashlib | stdlib | File integrity checksums | Optional SHA256 validation for critical checkpoints |
| tempfile | stdlib | Temporary file naming | Generate unique temp filenames (alternative to .tmp suffix) |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| pathlib.Path.replace() | os.replace() | Both are atomic on POSIX, Path API preferred for consistency with codebase |
| Manual ZIP validation | torch.load() only | Try-except on torch.load() catches corruption but provides poor diagnostics |
| Custom versioning | PyTorch Lightning migration | PL's auto-migration is powerful but introduces dependency, custom is simpler |

**Installation:**
No new dependencies required - all tools are PyTorch built-ins or Python stdlib.

## Architecture Patterns

### Recommended Checkpoint Structure
```
checkpoints/
├── pipeline_state.json          # Pipeline-level state (existing)
├── NUCLEOTIDE_FEATURES_files.json  # File-level progress (existing)
├── output_0_ESM.pt             # ESM embeddings (feature checkpoint)
├── output_0_DNABERT_S.pt       # DNABERT embeddings (feature checkpoint)
└── output_0_merged.pt          # Merged features (feature checkpoint)
```

### Pattern 1: Atomic Write with Cleanup
**What:** Write to temporary file, rename on success, cleanup on failure
**When to use:** All PyTorch checkpoint saves (torch.save operations)
**Example:**
```python
# Source: Existing codebase virnucpro/pipeline/features.py:264-272
temp_file = output_file.with_suffix('.tmp')
try:
    torch.save(checkpoint_dict, temp_file)
    temp_file.replace(output_file)  # Atomic on POSIX
except Exception as e:
    # Clean up temp file on failure
    if temp_file.exists():
        temp_file.unlink()
    raise
```

### Pattern 2: Multi-Level Checkpoint Validation
**What:** Validate checkpoint integrity at multiple levels before using
**When to use:** Both write-time (after save) and load-time (before resume)
**Example:**
```python
def validate_checkpoint(checkpoint_path: Path) -> Tuple[bool, str]:
    """Validate PyTorch checkpoint file integrity.

    Returns (is_valid, error_message)
    """
    # Level 1: File size check (fast)
    if checkpoint_path.stat().st_size == 0:
        return False, "corrupted: file is 0 bytes"

    # Level 2: ZIP format check (fast)
    if not zipfile.is_zipfile(checkpoint_path):
        return False, "corrupted: not a valid ZIP archive"

    # Level 3: PyTorch load check (slow, optional)
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    except Exception as e:
        return False, f"corrupted: torch.load failed - {str(e)}"

    # Level 4: Structure validation (check expected keys)
    required_keys = {'version', 'status', 'data'}  # Example
    if not all(k in checkpoint for k in required_keys):
        missing = required_keys - set(checkpoint.keys())
        return False, f"incompatible: missing keys {missing}"

    return True, ""
```

### Pattern 3: Versioned Checkpoint Metadata
**What:** Embed version and status fields in checkpoint dict
**When to use:** All checkpoints that may need future migration
**Example:**
```python
# Source: User decisions in CONTEXT.md
checkpoint_dict = {
    'version': '1.0',  # Semantic version
    'status': 'complete',  # or 'in_progress'
    'data': data,  # Actual checkpoint content
    # ... other fields
}
torch.save(checkpoint_dict, checkpoint_file)

# On load: check version compatibility
loaded = torch.load(checkpoint_file)
version = loaded.get('version', '0.x')  # Pre-optimization = 0.x

if version.startswith('2.'):
    raise ValueError(f"Checkpoint v{version} requires virnucpro >= 2.0.0")
```

### Pattern 4: Failed Checkpoint Tracking
**What:** Log checkpoint failures in structured format for debugging
**When to use:** When encountering corrupted/incompatible checkpoints
**Example:**
```python
# Source: User decisions - match failed_files.txt pattern from Phase 1
failed_log = checkpoint_dir / "failed_checkpoints.txt"

# Format: {checkpoint_path}|{reason}|{timestamp}
with open(failed_log, 'a') as f:
    timestamp = datetime.utcnow().isoformat()
    f.write(f"{checkpoint_path}|{error_reason}|{timestamp}\n")
```

### Anti-Patterns to Avoid
- **Direct torch.save without atomic write:** Interruption leaves corrupted checkpoint that blocks resume
- **Silent validation failures:** Log should show exactly what failed (missing keys, file size, etc.)
- **Auto-recovery from corruption:** Better to fail loudly than silently skip corrupted data
- **Modifying old checkpoint formats:** Load pre-optimization checkpoints read-only without conversion

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Atomic file writes | Manual write+rename+error handling | Existing pattern in features.py | Already implements cleanup on failure, tested in production |
| Directory sync after rename | Manual fsync on parent dir | pathlib.Path.replace() | POSIX guarantees atomic rename, no explicit sync needed for most filesystems |
| Checkpoint format detection | Try-except on torch.load | zipfile.is_zipfile() | Provides clear diagnostics before expensive load operation |
| Version string parsing | Regex version extraction | String operations (startswith, split) | Semantic versions are simple, no need for packaging.version complexity |
| Progress tracking across runs | Custom checkpoint registry | Existing FileProgressTracker | Already tracks file-level progress with pending/completed/failed status |

**Key insight:** The codebase already has robust patterns for atomic writes (features.py) and progress tracking (checkpoint.py FileProgressTracker). Extend these patterns rather than introducing new mechanisms.

## Common Pitfalls

### Pitfall 1: Using Path.rename() Instead of Path.replace()
**What goes wrong:** Path.rename() fails if target exists on some systems (Windows), causing checkpoint save to fail
**Why it happens:** rename() semantics vary by OS, replace() guarantees overwrite
**How to avoid:** Always use `temp_file.replace(output_file)` not `temp_file.rename(output_file)`
**Warning signs:** Checkpoint saves succeed initially but fail on resume when file already exists

**Note:** Codebase currently uses both - checkpoint.py uses replace() (line 136), features.py uses rename() (line 267). Features.py should be updated to replace() for consistency.

### Pitfall 2: Validating Too Late
**What goes wrong:** Discover corrupted checkpoint 8 hours into resume attempt when trying to load it
**Why it happens:** Validation only happens at load time, not at write time
**How to avoid:** Validate immediately after torch.save completes, before marking checkpoint complete
**Warning signs:** Jobs that fail hours after starting with "corrupted checkpoint" errors

### Pitfall 3: Incomplete Checkpoint Cleanup
**What goes wrong:** Temporary files accumulate in checkpoint directory, eventually filling disk
**Why it happens:** Exception during save leaves .tmp file, no cleanup mechanism
**How to avoid:** Always wrap torch.save in try-finally block that cleans up temp file
**Warning signs:** Checkpoint directory contains multiple .tmp files, disk space warnings

### Pitfall 4: Over-Strict Validation
**What goes wrong:** Checkpoint validation rejects valid files due to minor version changes
**Why it happens:** Validation checks exact tensor shapes instead of semantic compatibility
**How to avoid:** Validate key existence and tensor presence, not exact dimensions (user decision: "catch dimension mismatches from code changes without being overly strict")
**Warning signs:** Resume fails after minor code changes that don't affect checkpoint format

### Pitfall 5: Missing Status Field
**What goes wrong:** Resume attempts to use checkpoint that was interrupted mid-write
**Why it happens:** No way to distinguish complete vs in-progress checkpoints
**How to avoid:** Write 'status': 'in_progress' before torch.save, update to 'complete' after validation
**Warning signs:** Corrupted checkpoint files that are valid ZIP archives but contain partial data

### Pitfall 6: Ignoring Exit Codes
**What goes wrong:** Checkpoint failures return same exit code as embedding failures, hiding root cause
**Why it happens:** All failures use sys.exit(1) or raise unhandled exceptions
**How to avoid:** Use exit code 3 specifically for checkpoint issues (user decision)
**Warning signs:** Debugging requires parsing logs instead of checking exit code

## Code Examples

Verified patterns from existing codebase and official sources:

### Atomic Checkpoint Save (Production Pattern)
```python
# Source: virnucpro/pipeline/features.py:263-272 (existing implementation)
def save_checkpoint_atomic(checkpoint_dict: dict, output_file: Path):
    """Save checkpoint with atomic write to prevent corruption."""
    temp_file = output_file.with_suffix('.tmp')
    try:
        torch.save(checkpoint_dict, temp_file)
        temp_file.replace(output_file)  # Should use replace(), not rename()
    except Exception as e:
        # Clean up temp file on failure
        if temp_file.exists():
            temp_file.unlink()
        raise
```

### Write-Time Validation
```python
# Source: Research on PyTorch checkpoint validation patterns
def save_and_validate_checkpoint(checkpoint_dict: dict, output_file: Path):
    """Save checkpoint and validate integrity before marking complete."""
    # Mark as in-progress
    checkpoint_dict['status'] = 'in_progress'
    checkpoint_dict['version'] = '1.0'

    # Atomic save
    temp_file = output_file.with_suffix('.tmp')
    try:
        torch.save(checkpoint_dict, temp_file)
        temp_file.replace(output_file)
    except Exception as e:
        if temp_file.exists():
            temp_file.unlink()
        raise RuntimeError(f"Failed to save checkpoint: {e}")

    # Validate immediately after save
    is_valid, error = validate_checkpoint(output_file)
    if not is_valid:
        output_file.unlink()  # Remove corrupted file
        raise RuntimeError(f"Checkpoint validation failed: {error}")

    # Mark as complete (requires reload, edit, save)
    checkpoint_dict['status'] = 'complete'
    temp_file = output_file.with_suffix('.tmp')
    try:
        torch.save(checkpoint_dict, temp_file)
        temp_file.replace(output_file)
    except Exception as e:
        if temp_file.exists():
            temp_file.unlink()
        raise RuntimeError(f"Failed to update checkpoint status: {e}")
```

### Load-Time Validation with Detailed Logging
```python
# Source: Research on PyTorch checkpoint validation + user decisions
def load_checkpoint_with_validation(
    checkpoint_path: Path,
    skip_validation: bool = False,
    logger: logging.Logger = None
) -> dict:
    """Load checkpoint with multi-level validation.

    Args:
        checkpoint_path: Path to checkpoint file
        skip_validation: Skip validation for trusted scenarios (--skip-checkpoint-validation)
        logger: Logger for diagnostic output

    Returns:
        Loaded checkpoint dict

    Raises:
        RuntimeError: If checkpoint is corrupted or incompatible
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    # Fast validation checks
    if not skip_validation:
        # Level 1: File size
        file_size = checkpoint_path.stat().st_size
        if file_size == 0:
            error = f"corrupted: file is 0 bytes"
            logger.error(f"Checkpoint validation failed: {checkpoint_path}")
            logger.error(f"  Reason: {error}")
            raise RuntimeError(f"Corrupted checkpoint: {error}")

        logger.debug(f"Checkpoint size: {file_size} bytes")

        # Level 2: ZIP format
        if not zipfile.is_zipfile(checkpoint_path):
            error = f"corrupted: not a valid ZIP archive"
            logger.error(f"Checkpoint validation failed: {checkpoint_path}")
            logger.error(f"  Reason: {error}")
            raise RuntimeError(f"Corrupted checkpoint: {error}")

    # Load checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    except Exception as e:
        error = f"corrupted: torch.load failed - {str(e)}"
        logger.error(f"Checkpoint load failed: {checkpoint_path}")
        logger.error(f"  Error: {error}")
        raise RuntimeError(f"Corrupted checkpoint: {error}")

    if not skip_validation:
        # Level 3: Version check
        version = checkpoint.get('version', '0.x')  # Pre-optimization = no version
        logger.debug(f"Checkpoint version: {version}")

        if version.startswith('2.'):
            error = f"incompatible: version {version} requires virnucpro >= 2.0.0"
            logger.error(f"Checkpoint incompatible: {checkpoint_path}")
            logger.error(f"  Version: {version}")
            raise RuntimeError(f"Incompatible checkpoint: {error}")

        # Level 4: Status check
        status = checkpoint.get('status', 'unknown')
        if status == 'in_progress':
            logger.warning(f"Checkpoint marked as in-progress (may be incomplete): {checkpoint_path}")

        # Level 5: Key validation (basic structure check)
        if 'data' not in checkpoint:
            error = f"incompatible: missing required key 'data'"
            logger.error(f"Checkpoint validation failed: {checkpoint_path}")
            logger.error(f"  Keys found: {list(checkpoint.keys())}")
            logger.error(f"  Reason: {error}")
            raise RuntimeError(f"Incompatible checkpoint: {error}")

    logger.info(f"Checkpoint loaded successfully: {checkpoint_path}")
    return checkpoint
```

### Resume with Failed Checkpoint Tracking
```python
# Source: User decisions - failed_checkpoints.txt format
def resume_with_checkpoint_tracking(
    checkpoint_files: list[Path],
    checkpoint_dir: Path,
    force_resume: bool = False
) -> tuple[list[Path], list[Path]]:
    """Resume from checkpoints, tracking failures.

    Args:
        checkpoint_files: List of checkpoint files to load
        checkpoint_dir: Directory for failed_checkpoints.txt log
        force_resume: Skip bad checkpoints and continue (--force-resume)

    Returns:
        (valid_checkpoints, failed_checkpoints)
    """
    valid = []
    failed = []
    failed_log = checkpoint_dir / "failed_checkpoints.txt"

    for checkpoint_path in checkpoint_files:
        try:
            # Validate checkpoint
            is_valid, error = validate_checkpoint(checkpoint_path)

            if not is_valid:
                # Log failure
                timestamp = datetime.utcnow().isoformat()
                with open(failed_log, 'a') as f:
                    f.write(f"{checkpoint_path}|{error}|{timestamp}\n")

                failed.append(checkpoint_path)

                if not force_resume:
                    logger.error(f"Checkpoint validation failed: {checkpoint_path}")
                    logger.error(f"  Reason: {error}")
                    logger.error(f"  Use --force-resume to skip bad checkpoints")
                    sys.exit(3)  # Exit code 3 for checkpoint issues
                else:
                    logger.warning(f"Skipping bad checkpoint (--force-resume): {checkpoint_path}")
            else:
                valid.append(checkpoint_path)

        except Exception as e:
            # Log unexpected errors
            timestamp = datetime.utcnow().isoformat()
            error = f"unexpected error: {str(e)}"
            with open(failed_log, 'a') as f:
                f.write(f"{checkpoint_path}|{error}|{timestamp}\n")

            failed.append(checkpoint_path)

            if not force_resume:
                raise

    # Log summary
    logger.info(f"Checkpoint resume summary:")
    logger.info(f"  Valid: {len(valid)}/{len(checkpoint_files)}")
    logger.info(f"  Failed: {len(failed)}/{len(checkpoint_files)}")
    if failed:
        logger.info(f"  Failed checkpoints logged to: {failed_log}")

    return valid, failed
```

### Dry-Run Validation Command
```python
# Source: User decisions - virnucpro validate-checkpoints <dir>
def validate_checkpoints_command(checkpoint_dir: Path):
    """Dry-run validation command: report status without processing.

    virnucpro validate-checkpoints <dir>
    """
    checkpoint_files = list(checkpoint_dir.glob("*.pt"))

    if not checkpoint_files:
        print(f"No checkpoints found in {checkpoint_dir}")
        return 0

    print(f"Validating {len(checkpoint_files)} checkpoints in {checkpoint_dir}\n")

    valid_count = 0
    failed_count = 0

    for checkpoint_path in sorted(checkpoint_files):
        is_valid, error = validate_checkpoint(checkpoint_path)

        if is_valid:
            status = "✓ VALID"
            valid_count += 1
        else:
            status = f"✗ FAILED: {error}"
            failed_count += 1

        print(f"{status:<60} {checkpoint_path.name}")

    print(f"\nSummary: {valid_count} valid, {failed_count} failed")

    return 0 if failed_count == 0 else 3  # Exit code 3 for checkpoint issues
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| torch.save without atomic write | Temp-then-rename pattern | PyTorch Lightning ~2020, Ignite 0.4+ | Prevents corruption from interruptions during save |
| Try-except on torch.load only | Multi-level validation (size → ZIP → load → keys) | HPC checkpointing best practices | Faster failure detection, better diagnostics |
| No checkpoint versioning | Embedded version metadata | PyTorch Lightning 1.0+ | Enables backward compatibility and migration |
| Path.rename() | Path.replace() | Python 3.3+ | Atomic overwrite on all platforms, Windows compatibility |
| Manual fsync calls | OS-level atomicity guarantees | Modern filesystems | Simplified code, relies on POSIX rename atomicity |

**Deprecated/outdated:**
- **Direct torch.save to final path:** Creates corruption risk, all modern frameworks use temp-then-rename
- **Path.rename() for atomic writes:** Fails on Windows if target exists, use Path.replace() instead (existing codebase has inconsistency)
- **No validation after save:** Write corruption goes undetected until resume attempt hours later
- **Generic exit codes:** Exit code 1 for all failures hides root cause, modern CLIs use specific codes (checkpoint = 3)

## Open Questions

Things that couldn't be fully resolved:

1. **Optimal validation depth for performance**
   - What we know: Full validation (size → ZIP → load → keys → tensors) is most robust
   - What's unclear: Performance impact of validating every checkpoint at write time for large files (multi-GB ESM-2 embeddings)
   - Recommendation: Implement `--skip-checkpoint-validation` flag for trusted scenarios, default to full validation

2. **Tensor dimension validation specificity**
   - What we know: User decision says "catch dimension mismatches from code changes without being overly strict on exact shapes"
   - What's unclear: Exact heuristic for "overly strict" - check rank only? Check all dims except batch? Check embedding dim?
   - Recommendation: Start with key existence + tensor presence only, add dimension checks if corruption patterns emerge

3. **Migration strategy for pre-optimization checkpoints**
   - What we know: Pre-optimization checkpoints have no version field, should load read-only
   - What's unclear: Whether to add version='0.x' on load or leave unmodified
   - Recommendation: Treat missing version as 0.x internally, don't modify file (read-only load)

4. **Directory-level fsync necessity**
   - What we know: Some HPC guides recommend fsync on parent directory after rename for NFS resilience
   - What's unclear: Whether virnucpro's deployment environments (SLURM clusters) require explicit directory sync
   - Recommendation: Start without directory fsync (POSIX rename is atomic), add if corruption observed on NFS

5. **Checkpoint version increment policy**
   - What we know: Current version is 1.0, future versions need migration
   - What's unclear: When to bump version (MAJOR.MINOR) - format changes only? Schema changes? Any field addition?
   - Recommendation: Semantic versioning - MAJOR for breaking changes (can't load old format), MINOR for additions (can load old format)

## Sources

### Primary (HIGH confidence)
- Codebase analysis - virnucpro/core/checkpoint.py (CheckpointManager, FileProgressTracker implementation)
- Codebase analysis - virnucpro/pipeline/features.py (atomic write pattern in merge_features)
- User decisions - .planning/phases/03-checkpoint-robustness/03-CONTEXT.md (validation depth, recovery behavior, versioning strategy)

### Secondary (MEDIUM confidence)
- [PyTorch Lightning Checkpointing (GitHub Issue #19970)](https://github.com/Lightning-AI/pytorch-lightning/issues/19970) - Atomic checkpoint saving isn't atomic discussion
- [PyTorch-Ignite Checkpoint Documentation](https://docs.pytorch.org/ignite/generated/ignite.handlers.checkpoint.Checkpoint.html) - atomic=True parameter for checkpoint handler
- [Python os.replace Function](https://zetcode.com/python/os-replace/) - Atomic file replacement guarantees
- [Python Atomic Writes Discussion](https://discuss.python.org/t/adding-atomicwrite-in-stdlib/11899) - Python 3.3+ os.replace and Path.replace() for atomic operations
- [PyTorch Forums - Corrupted Checkpoint Detection](https://discuss.pytorch.org/t/problem-with-my-checkpoint-file-when-using-torch-load/92903) - Using zipfile.is_zipfile() for validation
- [Medium - Fixing PyTorch Loading Errors](https://medium.com/@python-javascript-php-html-css/fixing-pytorch-model-loading-error-pickle-unpicklingerror-invalid-load-key-x1f-db49918a548d) - Checkpoint corruption patterns and recovery
- [PyTorch Forums - Missing/Unexpected Keys](https://discuss.pytorch.org/t/missing-keys-unexpected-keys-in-state-dict-when-loading-self-trained-model/22379) - state_dict validation patterns

### Tertiary (LOW confidence)
- [HPC Checkpointing Best Practices - Northeastern](https://rc-docs.northeastern.edu/en/latest/best-practices/checkpointing.html) - General checkpoint-restart strategies for long-running jobs
- [DMTCP for HPC (arXiv)](https://arxiv.org/html/2407.19117v1) - Distributed checkpointing in HPC environments
- [Exit Code Best Practices](https://chrisdown.name/2013/11/03/exit-code-best-practises.html) - CLI exit code conventions (0=success, 1-255=failure codes)
- [Process Exit Codes in Python](https://superfastpython.com/exit-codes-in-python/) - Python sys.exit() and exit code conventions
- [PyTorch Lightning Versioning Policy](https://lightning.ai/docs/pytorch/stable/versioning.html) - Semantic versioning for ML frameworks

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - All stdlib/PyTorch built-ins, verified in codebase
- Architecture: HIGH - Atomic write pattern exists in codebase, validation patterns verified in PyTorch community
- Pitfalls: MEDIUM - Based on codebase observation (rename vs replace inconsistency) and community reports
- Versioning: MEDIUM - User decisions are clear, implementation patterns from PyTorch Lightning
- Exit codes: MEDIUM - User decision is clear (exit code 3), convention research confirms feasibility

**Research date:** 2026-01-23
**Valid until:** 2026-02-23 (30 days - stable domain, PyTorch checkpoint format is well-established)
