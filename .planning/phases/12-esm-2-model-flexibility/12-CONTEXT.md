# Phase 12: ESM-2 Model Flexibility - Context

**Gathered:** 2026-02-10
**Status:** Ready for planning

<domain>
## Phase Boundary

Configurable ESM-2 model selection (650M, 3B, custom paths) with automatic detection of model architecture properties (repr_layers, hidden_dim). This phase makes model selection flexible rather than hardcoded, while maintaining backward compatibility with the current default (3B model).

Scope: Model selection and configuration auto-detection only. Does not include model ensembles, custom training, or changing how models are used once configured.

</domain>

<decisions>
## Implementation Decisions

### Model Selection Interface
- **Config-first approach**: Config file is source of truth, CLI flag (`--esm-model`) overrides config
- **Full model names**: Use explicit ESM naming (e.g., `esm2_t36_3B_UR50D`, `esm2_t33_650M_UR50D`) for unambiguous specification
- **Early validation**: Validate custom paths at config load time — check existence AND attempt to load model metadata to confirm it's a compatible ESM-2 model
- **Backward compatible default**: `esm2_t36_3B_UR50D` remains the default when no model is specified (maintains existing behavior)

### Auto-Detection Strategy
- **Auto-detect with overrides**: Automatically detect repr_layers, hidden_dim, and num_layers from model architecture, but allow manual override via config for edge cases (fine-tuned models, experiments)
- **Model-specific mapping with overrides**: Use known optimal layers per model variant (e.g., 3B uses layer 36, 650M uses layer 33) via mapping table, but allow config to override
- **Fail on detection failure**: If auto-detection fails for unrecognized model architecture, stop execution with error requiring user to provide explicit config (no silent fallbacks)
- **Always display detection results**: Print detected configuration (model name, layers, hidden_dim) at startup for transparency and debugging

### Checkpoint Compatibility
- **Full config snapshot**: Store model name, repr_layers, hidden_dim, and all detected properties in checkpoint metadata (no re-detection needed on resume)
- **Hard error on model mismatch**: Refuse to resume from checkpoint if model differs from checkpoint metadata (no --force-model override, no dimension-based compatibility)
- **Reject old checkpoints**: Pre-Phase 12 checkpoints without model metadata are incompatible — require fresh start (no migration, no assumption of default model)
- **Validate at startup**: Check checkpoint metadata before any heavy loading to fail fast if incompatible

### Error Handling & Validation
- **Validate at config load time**: Check model configuration as soon as config is parsed, before any model operations (immediate feedback)
- **Technical details only**: Error messages show exact error (e.g., "repr_layer 40 exceeds model depth 36") without additional guidance — assumes user knowledge
- **All-or-nothing detection**: If any property fails detection, abort completely and require user to provide full config (no partial detection)
- **Add --validate-config flag**: Dry-run mode that loads model, shows detected config, and exits cleanly without running pipeline (useful for testing configurations)

### Claude's Discretion
- Internal structure of ESM2Config dataclass (fields, validation methods)
- How model-specific mapping table is implemented and maintained
- Exact format of checkpoint metadata storage
- Logging verbosity levels for detection steps

</decisions>

<specifics>
## Specific Ideas

- Use full ESM model naming convention (esm2_tXX_XXXM_UR50D) for clarity and consistency with ESM library
- Config-first approach enables per-project model defaults while CLI allows quick overrides for experiments
- Validation at config load time (not lazy) prevents wasted time on large jobs that will fail later
- Reject old checkpoints cleanly — Phase 12 is a breaking change for checkpoint format

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 12-esm-2-model-flexibility*
*Context gathered: 2026-02-10*
