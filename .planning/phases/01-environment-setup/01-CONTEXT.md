# Phase 1: Environment Setup - Context

**Gathered:** 2026-02-07
**Status:** Ready for planning

<domain>
## Phase Boundary

Establish a FastESM2-compatible Python environment with PyTorch 2.5+ and SDPA optimization support. This phase creates the technical foundation for migration but does not include feature extraction implementation or model code changes.

</domain>

<decisions>
## Implementation Decisions

### Dependency Management
- Use pixi for environment management (existing project standard)
- Pin exact versions for all dependencies (pytorch==2.5.0 style) for maximum reproducibility
- Create fresh environment from scratch - hard break from old setup, no in-place migration
- Document setup process in detailed README with step-by-step instructions and troubleshooting

### PyTorch Installation
- Target CUDA 12.x (installed on target GPU system)
- Prefer conda-forge as primary source for PyTorch installation
- Fallback: Use PyPI (pip) if PyTorch 2.5+ not available in conda-forge for CUDA 12.x
- SDPA validation: Simple smoke test (verify torch.nn.functional.scaled_dot_product_attention exists)

### Package Migration
- Remove fair-esm package completely from environment (no backward compatibility)
- Fix any import errors immediately as part of Phase 1 (multiple files import fair-esm)
- Do not test old ESM2 3B pipeline before migration (fresh start approach)
- Update/remove all fair-esm imports discovered during environment setup

### Validation Approach
- Automated validation script to check all success criteria
- SDPA speedup: Must confirm actual 2x speedup claim (not just availability)
- Model loading: Dry run test - actually load FastESM2_650 from HuggingFace Hub and verify initialization
- Failure handling: Fail loudly - stop immediately with clear error message if any check fails

### Claude's Discretion
- Specific transformers library version (as long as >=4.30.0 compatible)
- Exact structure of validation script
- README organization and formatting
- Order of environment setup steps

</decisions>

<specifics>
## Specific Ideas

- Validation script must demonstrate 2x SDPA speedup with actual benchmark
- README should include troubleshooting section for common issues
- pixi.toml should use exact version pinning throughout

</specifics>

<deferred>
## Deferred Ideas

None - discussion stayed within phase scope

</deferred>

---

*Phase: 01-environment-setup*
*Context gathered: 2026-02-07*
