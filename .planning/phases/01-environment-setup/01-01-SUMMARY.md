---
phase: 01-environment-setup
plan: 01
subsystem: infrastructure
tags: [pixi, pytorch, transformers, cuda, dependencies, environment-setup]

dependencies:
  requires: []
  provides:
    - pixi environment with PyTorch 2.5.1 CUDA support
    - transformers 4.45.2 (FastESM2 compatible)
    - importable Python modules without fair-esm
  affects:
    - 01-02 (model download depends on transformers 4.45.2)
    - 02-01 (adapter implementation needs PyTorch 2.5.1)
    - all downstream phases (require this environment)

tech-stack:
  added:
    - pytorch: 2.5.1 (CUDA 12.6, aarch64 build)
    - transformers: 4.45.2
    - einops: ">=0.6.1"
    - networkx: ">=3.2"
    - safetensors: ">=0.4.1"
    - pandas: ">=1.3"
  patterns:
    - pixi for conda-forge exclusive dependency management
    - system-requirements for CUDA virtual package support

files:
  key-files:
    created:
      - pixi.toml
      - pixi.lock
    modified:
      - units.py
      - features_extract.py

decisions:
  - id: einops-version-downgrade
    choice: "Use einops >=0.6.1 instead of ==0.8.2"
    rationale: "einops 0.8.2 requires Python >=3.10, incompatible with project's Python 3.9"
    impact: "Minor version difference, no API changes affecting FastESM2"

  - id: system-requirements-cuda
    choice: "Add [system-requirements] cuda = \"12.6\" to pixi.toml"
    rationale: "PyTorch CUDA builds require __cuda virtual package on aarch64"
    impact: "Enables conda solver to find CUDA-enabled PyTorch builds"

  - id: pandas-addition
    choice: "Add pandas >=1.3 to dependencies"
    rationale: "prediction.py imports pandas but it was missing from old pixi.toml"
    impact: "Prevents import errors in prediction pipeline"

metrics:
  duration: "7 minutes"
  completed: 2026-02-07
  commits: 2
---

# Phase 01 Plan 01: Environment Setup Summary

**One-liner:** PyTorch 2.5.1 with CUDA 12.6 + transformers 4.45.2 environment configured via pixi, fair-esm removed, modules importable

## What Was Built

Configured a reproducible pixi environment with all FastESM2-compatible dependencies and removed fair-esm package conflicts.

**Key Capabilities:**
- PyTorch 2.5.1 with CUDA 12.6 support (aarch64 build with SDPA optimization)
- transformers 4.45.2 (matching FastESM2_650 development version)
- All FastESM2 dependencies (einops, networkx, safetensors)
- Importable Python modules without fair-esm ModuleNotFoundError
- extract_esm() deprecated with clear migration message

**Environment Ready For:**
- Phase 01-02: Download FastESM2_650 model files via transformers
- Phase 02: Implement FastESM2 adapter and feature extraction
- Phase 03+: All downstream work requiring this foundation

## Task Commits

| Task | Description | Commit | Files Modified |
|------|-------------|--------|----------------|
| 1 | Configure pixi environment with pinned dependencies | 6909419 | pixi.toml, pixi.lock |
| 2 | Remove fair-esm imports and guard deprecated code | 15502e9 | units.py, features_extract.py |

## Decisions Made

**1. einops version downgrade (Rule 3 - Blocking)**
- **Issue:** einops ==0.8.2 requires Python >=3.10, but project uses Python 3.9
- **Solution:** Changed to einops >=0.6.1 (compatible with Python 3.9)
- **Impact:** No API changes between 0.6.1 and 0.8.2 affecting FastESM2 usage
- **Rationale:** Blocking issue preventing environment installation

**2. CUDA virtual package configuration**
- **Issue:** PyTorch CUDA builds require __cuda virtual package on aarch64
- **Solution:** Added [system-requirements] cuda = "12.6" to pixi.toml
- **Impact:** Enables conda solver to find CUDA-enabled PyTorch 2.5.1 builds
- **Rationale:** CUDA 12.6 forward-compatible with system CUDA 13.0 driver

**3. pandas dependency addition (Rule 2 - Missing Critical)**
- **Issue:** prediction.py imports pandas but it was missing from old pixi.toml
- **Solution:** Added pandas >=1.3 to dependencies
- **Impact:** Prevents import errors in Phase 5 prediction pipeline
- **Rationale:** Required for correct operation of existing code

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] einops version incompatibility**
- **Found during:** Task 1 - pixi install
- **Issue:** einops ==0.8.2 requires Python >=3.10, project uses Python 3.9
- **Fix:** Changed dependency to einops >=0.6.1
- **Files modified:** pixi.toml
- **Commit:** 6909419

**2. [Rule 3 - Blocking] CUDA virtual package missing**
- **Found during:** Task 1 - pixi install
- **Issue:** PyTorch CUDA builds require __cuda virtual package declaration
- **Fix:** Added [system-requirements] cuda = "12.6" section to pixi.toml
- **Files modified:** pixi.toml
- **Commit:** 6909419

**3. [Rule 2 - Missing Critical] pandas dependency**
- **Found during:** Task 1 - dependency review
- **Issue:** prediction.py uses pandas but it was missing from dependencies
- **Fix:** Added pandas >=1.3 to dependencies
- **Files modified:** pixi.toml
- **Commit:** 6909419

**4. [Rule 3 - Blocking] Corrupted .pixi directory**
- **Found during:** Task 1 - pixi install linking phase
- **Issue:** ATen directory linking conflict from previous installation attempt
- **Fix:** Removed .pixi directory and reinstalled cleanly
- **Files modified:** None (temporary directory cleanup)
- **Commit:** 6909419

## Technical Implementation

### Environment Configuration

**pixi.toml structure:**
```toml
[workspace]
channels = ["conda-forge"]
platforms = ["linux-aarch64"]

[system-requirements]
cuda = "12.6"

[dependencies]
python = "3.9.*"
pytorch = { version = "==2.5.1", build = "*cuda126*" }
transformers = "==4.45.2"
einops = ">=0.6.1"
# ... other dependencies
```

**Key decisions:**
- CUDA 12.6 build specification ensures CUDA-enabled PyTorch on aarch64
- Exact version pins for pytorch and transformers (reproducibility)
- Flexible versioning for supporting packages (compatibility)
- No pip in dependencies (conda-forge exclusive)

### Code Refactoring

**units.py changes:**
- Removed: `from esm import FastaBatchedDataset, pretrained`
- Added: Comment explaining removal and migration path
- Modified: `extract_esm()` body replaced with NotImplementedError
- Preserved: Function signature for API documentation

**features_extract.py changes:**
- Removed: Module-level `pretrained.load_model_and_alphabet('esm2_t36_3B_UR50D')`
- Added: Comment explaining removal
- Set: `ESM_model = None` and `ESM_alphabet = None`
- Preserved: process_file_pro() function (will be rewritten in Phase 2)

## Verification Results

All verification criteria passed:

| Check | Expected | Result | Status |
|-------|----------|--------|--------|
| pixi install | Exit 0 | Success | ✓ |
| PyTorch version | 2.5.1 | 2.5.1 | ✓ |
| CUDA available | True | True | ✓ |
| transformers version | 4.45.2 | 4.45.2 | ✓ |
| einops import | Success | einops OK | ✓ |
| fair-esm import | ModuleNotFoundError | ModuleNotFoundError | ✓ |
| units.py import | Success | units module OK | ✓ |
| extract_esm() behavior | NotImplementedError | NotImplementedError with message | ✓ |

## Known Issues & Limitations

**1. extract_esm() deprecated**
- **Impact:** Legacy feature extraction code won't work until Phase 2
- **Workaround:** None needed - function won't be called until Phase 2 replaces it
- **Resolution:** Phase 02-01 will implement extract_fast_esm()

**2. features_extract.py process_file_pro() incomplete**
- **Impact:** Function exists but ESM_model/ESM_alphabet are None
- **Workaround:** None needed - script won't be run until Phase 2
- **Resolution:** Phase 02-02 will rewrite to use FastESM2_650

**3. einops version range**
- **Impact:** Using >=0.6.1 instead of ==0.8.2 allows newer versions
- **Risk:** Future einops releases could break compatibility
- **Mitigation:** pixi.lock pins exact resolved version (0.6.1)

## Next Phase Readiness

**Phase 01-02 can proceed:**
- ✓ transformers 4.45.2 installed (required for model download)
- ✓ safetensors installed (required for model weight loading)
- ✓ PyTorch 2.5.1 with CUDA available (required for model verification)

**No blockers identified.**

**Recommended next steps:**
1. Execute Plan 01-02: Download and verify FastESM2_650 model
2. Document model file structure and tokenizer configuration
3. Create checkpoint to verify model loads correctly with transformers API

## Lessons Learned

**Environment-specific challenges:**
- aarch64 PyTorch CUDA builds require explicit __cuda virtual package declaration
- pixi's conda solver needs system-requirements section for virtual packages
- Build string constraints ({ build = "*cuda126*" }) necessary for CUDA variant selection

**Dependency management:**
- Always verify Python version compatibility before pinning versions
- Missing dependencies in old configs may be discovered during import verification
- Clean .pixi directory if linking errors occur (not just cache issues)

**Migration planning:**
- Removing fair-esm before adding FastESM2 prevents version conflicts
- Deprecating functions with NotImplementedError maintains API documentation
- Module-level imports that load large models should be deferred to function scope

## Self-Check: PASSED

All created files exist:
- FOUND: /home/carze/projects/VirNucPro/pixi.toml
- FOUND: /home/carze/projects/VirNucPro/pixi.lock

All commits exist:
- FOUND: 6909419
- FOUND: 15502e9
