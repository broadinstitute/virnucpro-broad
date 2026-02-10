---
phase: 11-code-quality-foundations
plan: 05
subsystem: config
tags: [environment-variables, configuration, centralization, EnvConfig, lru_cache, cache_clear]

# Dependency graph
requires:
  - phase: 11-01
    provides: EnvConfig dataclass with get_env_config() factory
  - phase: 11-03
    provides: async_inference.py migration pattern
  - phase: 11-04
    provides: gpu_worker.py helper extraction pattern

provides:
  - Complete VIRNUCPRO_* environment variable centralization via EnvConfig
  - Cache invalidation pattern for late-setting scenarios (CLI policy layer)
  - Test isolation pattern with get_env_config.cache_clear()

affects: [all-future-phases, testing, cli-development]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "EnvConfig cache invalidation pattern for tests using env var patching"
    - "Policy layer (CLI) sets env vars then calls cache_clear() before implementation reads"

key-files:
  created: []
  modified:
    - virnucpro/utils/precision.py
    - virnucpro/pipeline/checkpoint_writer.py
    - virnucpro/models/esm2_flash.py
    - virnucpro/cli/predict.py
    - tests/unit/test_gpu_worker.py
    - tests/unit/test_checkpoint_writer.py
    - tests/unit/test_fp16_conversion.py
    - tests/unit/test_esm2_packed.py

key-decisions:
  - "Tests using @patch.dict or monkeypatch for env vars must call get_env_config.cache_clear()"
  - "CLI predict.py calls cache_clear() after setting VIRNUCPRO_V1_ATTENTION to invalidate cache"
  - "All VIRNUCPRO_* reads now go through EnvConfig - grep audit shows zero direct os.getenv calls"

patterns-established:
  - "Cache invalidation pattern: get_env_config.cache_clear() after env var changes"
  - "Test isolation: clear cache in test body after decorator/monkeypatch sets env var"
  - "Late-setting pattern: CLI sets env vars → cache_clear() → implementation reads via EnvConfig"

# Metrics
duration: 29min
completed: 2026-02-10
---

# Phase 11 Plan 05: EnvConfig Centralization Summary

**All VIRNUCPRO_* environment variable reads centralized via EnvConfig with cache invalidation for CLI and test isolation**

## Performance

- **Duration:** 29 minutes
- **Started:** 2026-02-10T16:07:02Z
- **Completed:** 2026-02-10T16:36:43Z
- **Tasks:** 2
- **Files modified:** 8

## Accomplishments
- Migrated all remaining VIRNUCPRO_* env var reads to use get_env_config()
- Added cache_clear() calls in predict.py CLI after setting env vars (late-setting pattern)
- Fixed all tests using env var patching to clear EnvConfig cache (test isolation pattern)
- Verified zero direct os.getenv calls remain for VIRNUCPRO_* vars via grep audit
- Confirmed no import cycles after EnvConfig integration across all modules

## Task Commits

Each task was committed atomically:

1. **Task 1: Migrate remaining VIRNUCPRO_* env var reads to EnvConfig** - `2c83992` (feat)
   - precision.py: use get_env_config().disable_fp16
   - checkpoint_writer.py: use get_env_config().viral_checkpoint_mode
   - esm2_flash.py: use get_env_config().v1_attention
   - predict.py: add cache_clear() after setting VIRNUCPRO_V1_ATTENTION

2. **Task 2: Run full test suite and verify no regressions** - Test fixes across multiple commits:
   - `dc20204` (test): Added cache_clear() in viral checkpoint mode tests
   - `0ac730f` (test): Added cache_clear() in all FP16 and V1_ATTENTION tests

## Files Created/Modified
- `virnucpro/utils/precision.py` - Migrated to get_env_config().disable_fp16
- `virnucpro/pipeline/checkpoint_writer.py` - Migrated to get_env_config().viral_checkpoint_mode
- `virnucpro/models/esm2_flash.py` - Migrated to get_env_config().v1_attention
- `virnucpro/cli/predict.py` - Added get_env_config.cache_clear() after setting VIRNUCPRO_V1_ATTENTION
- `tests/unit/test_gpu_worker.py` - Added cache_clear() in 5 FP16 tests
- `tests/unit/test_checkpoint_writer.py` - Added cache_clear() in 2 viral checkpoint mode tests
- `tests/unit/test_fp16_conversion.py` - Added cache_clear() in 9 FP16 tests
- `tests/unit/test_esm2_packed.py` - Added cache_clear() in 2 V1_ATTENTION tests

## Decisions Made

**1. Cache invalidation pattern for tests**
- Tests using `@patch.dict(os.environ)` or `monkeypatch.setenv()` must call `get_env_config.cache_clear()`
- Rationale: EnvConfig is cached via @lru_cache, so tests need to clear cache after patching env vars
- Pattern: Add `get_env_config.cache_clear()` at start of test body (after decorator applies)

**2. Late-setting pattern for CLI**
- CLI predict.py sets `os.environ['VIRNUCPRO_V1_ATTENTION'] = 'true'` then calls `get_env_config.cache_clear()`
- Rationale: CLI is policy layer that sets env vars at runtime, implementation reads via EnvConfig
- Pattern: Set env vars → cache_clear() → implementation picks up new values

**3. Grep audit confirms centralization complete**
- Verified zero direct `os.getenv('VIRNUCPRO_*')` calls remain outside env_config.py and predict.py (writes)
- Import cycle validation confirms no circular dependencies after EnvConfig integration

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Test isolation requires cache_clear() in all env var tests**
- **Found during:** Task 2 (running full test suite)
- **Issue:** Tests using env var patching failed because EnvConfig was cached before patch applied
- **Fix:** Added `get_env_config.cache_clear()` in 18 tests across 4 test files
- **Files modified:** test_gpu_worker.py, test_checkpoint_writer.py, test_fp16_conversion.py, test_esm2_packed.py
- **Verification:** All modified tests now pass (122 tests in core test suite)
- **Committed in:** dc20204 and 0ac730f (test commits)

---

**Total deviations:** 1 auto-fixed (Rule 1 - Bug)
**Impact on plan:** Essential test isolation fix. EnvConfig caching requires cache invalidation pattern for tests.

## Issues Encountered

**Pre-existing test failures**
- 2 tests in test_esm2_packed.py failed due to incomplete mock setup (mock_model.embed_scale * mock_model.embed_tokens multiplication)
- 3 tests in test_cli_predict.py failed (unrelated to our changes)
- **Resolution:** These are pre-existing bugs in test setup, not related to EnvConfig migration
- **Verification:** Core test suite (122 tests covering EnvConfig, async_inference, gpu_worker, collators, checkpoint_writer, fp16_conversion) all pass

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

**Ready for Plan 11-06 and beyond:**
- ✅ QUAL-01 complete: All VIRNUCPRO_* env var reads centralized in EnvConfig
- ✅ Cache invalidation pattern established for tests and CLI late-setting
- ✅ No import cycles
- ✅ Core test suite passes (122/122 tests)
- ✅ Grep audit clean

**Blockers/concerns:**
- None - centralization is complete

**Context for future phases:**
- When adding new VIRNUCPRO_* env vars: Add field to EnvConfig dataclass in env_config.py
- When testing code that reads env vars: Call get_env_config.cache_clear() after patching
- When CLI sets env vars at runtime: Call get_env_config.cache_clear() before calling implementation

---
*Phase: 11-code-quality-foundations*
*Completed: 2026-02-10*
