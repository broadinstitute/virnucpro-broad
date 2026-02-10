# Phase 11: Code Quality Foundations - Context

**Gathered:** 2026-02-09
**Status:** Ready for planning

<domain>
## Phase Boundary

Refactor internal code for maintainability: centralize environment variable access into an EnvConfig dataclass, extract large functions (async_inference.run(), gpu_worker(), run_prediction()) into focused helpers, replace list-based queues with collections.deque, and deduplicate shared validation code. No external behavior changes — 1:1 behavior equivalence required.

</domain>

<decisions>
## Implementation Decisions

### EnvConfig Design
- Single flat dataclass (not nested/grouped configs) — all env vars in one place
- Loading strategy: Claude's discretion (eager vs lazy)
- Instance pattern: Claude's discretion (singleton vs per-component)
- Invalid env var values (e.g., `VIRNUCPRO_DISABLE_PACKING=banana`) must raise ValueError — fail fast

### Function Extraction Strategy
- Extraction pattern (methods vs standalone): Claude's discretion based on existing codebase patterns
- Line limit (~100 lines): Claude's discretion — target 100 but prioritize natural function boundaries over arbitrary limits
- Testing approach for extracted helpers: Claude's discretion based on risk level of each extraction
- prediction.run_prediction() treatment: Claude's discretion — assess independently from async_inference.run()

### Shared Utilities Location
- Module location for extracted duplicates: Claude's discretion based on existing codebase structure
- Migration approach (neutral location vs keep-in-original): Claude's discretion — minimize import cycles
- CUDA validation grouping: Claude's discretion — group by what makes semantic sense
- Naming (public vs private prefix): Claude's discretion based on Python conventions

### Backward Compatibility
- Env var migration: all-at-once (single plan replaces every os.getenv with EnvConfig access)
- Env var names: standardize naming for consistency (don't preserve inconsistent old names)
- Old name handling: Claude's discretion on whether to support old names with deprecation warnings or hard-cut
- Validation: full benchmark suite after refactoring to verify no performance regression (not just unit tests)

### Claude's Discretion
- EnvConfig loading strategy (eager vs lazy)
- EnvConfig instance pattern (singleton vs per-component)
- Function extraction style (methods on class vs standalone functions)
- Line limit flexibility around 100-line target
- Whether extracted helpers need new unit tests or existing coverage suffices
- Shared utility module location and organization
- Old env var name deprecation approach

</decisions>

<specifics>
## Specific Ideas

- User wants env var names standardized — current names like `VIRNUCPRO_DISABLE_PACKING` may be inconsistent
- Full Phase 10 benchmarks should be re-run after refactoring to confirm no performance regression
- Invalid env var values must fail loudly (ValueError), not silently fall back to defaults

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 11-code-quality-foundations*
*Context gathered: 2026-02-09*
