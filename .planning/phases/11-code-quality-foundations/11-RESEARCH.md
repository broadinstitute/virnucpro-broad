# Phase 11: Code Quality Foundations - Research

**Researched:** 2026-02-09
**Domain:** Python code refactoring and maintainability
**Confidence:** HIGH

## Summary

Phase 11 refactors VirNucPro for maintainability without changing external behavior. Research focused on four domains: (1) environment variable centralization using Python dataclasses with `__post_init__` loading, (2) function extraction patterns from large methods (976-1149 lines), (3) replacing O(n) list queue operations with O(1) deque operations, and (4) duplicate code extraction to shared utilities. The codebase already uses dataclasses in 5 modules (runtime_config.py, shard_index.py, async_inference.py), providing established patterns. Current environment variable access is scattered across 19 locations using `os.getenv()` and `os.environ.get()`. Duplicate CUDA isolation validation exists in two dataset classes (83 lines duplicated). Large functions span 462-1149 lines and lack natural cohesion boundaries. List-based queue operations (.pop(0), .insert(0)) appear in collators.py (packed_queue) with O(n) performance cost. All refactoring must maintain 1:1 behavior equivalence verified by existing 52 test files.

**Primary recommendation:** Use flat dataclass in `core/env_config.py` with `__post_init__` env loading, cached property access via `@lru_cache`, and fail-fast validation for invalid values. Extract functions by natural semantic boundaries (not arbitrary line counts), migrate to `collections.deque` for queue operations, and consolidate duplicate validation into `utils/cuda_validation.py`.

## Standard Stack

The established libraries/tools for this domain:

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| dataclasses | stdlib (3.7+) | Data class with type hints and __post_init__ | Already used in 5 modules, zero dependencies, built-in validation |
| functools.lru_cache | stdlib | Method-level caching for env var access | Prevents repeated os.getenv() syscalls in hot paths |
| collections.deque | stdlib | O(1) queue operations | 40-60k times faster than list.pop(0) for front operations |
| os.getenv | stdlib | Environment variable reading | Current standard in codebase (19 call sites) |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| typing | stdlib | Type hints for dataclass fields | All env var fields (bool, int, str, Optional[Path]) |
| pathlib.Path | stdlib | Path handling in env vars | For CHECKPOINT_DIR, CONFIG_PATH fields |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| dataclasses | Pydantic Settings | Pydantic adds dependencies (pydantic 2.0+), more validation, but overkill for ~10 env vars |
| __post_init__ | envclasses library | External dependency for simple use case, not worth maintenance burden |
| @lru_cache | Manual caching | More code, error-prone, defeats purpose of maintainability refactor |

**Installation:**
No new dependencies required - all stdlib.

## Architecture Patterns

### Recommended Project Structure
```
virnucpro/core/
├── env_config.py        # Centralized environment variable access (NEW)
├── config.py            # Existing YAML configuration
├── device.py            # Existing CUDA device validation
└── checkpoint.py        # Existing checkpoint management

virnucpro/utils/
├── cuda_validation.py   # Shared CUDA isolation validation (NEW)
├── precision.py         # Existing FP16 utilities (uses env vars)
└── validation.py        # Existing FASTA validation
```

### Pattern 1: Flat Dataclass with __post_init__ Loading

**What:** Single dataclass with all environment variables as fields, loaded in `__post_init__`

**When to use:** Centralized env var access with validation (exactly this phase's requirement)

**Example:**
```python
# virnucpro/core/env_config.py
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional
import os
import logging

logger = logging.getLogger('virnucpro.core.env_config')

@dataclass
class EnvConfig:
    """Centralized environment variable configuration.

    All VirNucPro environment variables accessed through this class.
    Values loaded once at instantiation, cached for performance.
    """
    disable_packing: bool = False
    disable_fp16: bool = False
    v1_attention: bool = False
    viral_checkpoint_mode: bool = False
    compile_model: bool = False
    checkpoint_dir: Optional[Path] = None

    def __post_init__(self):
        """Load and validate environment variables."""
        # VIRNUCPRO_DISABLE_PACKING
        raw = os.getenv('VIRNUCPRO_DISABLE_PACKING', 'false').strip().lower()
        if raw in ('true', '1', 'yes'):
            self.disable_packing = True
        elif raw in ('false', '0', 'no', ''):
            self.disable_packing = False
        else:
            raise ValueError(
                f"VIRNUCPRO_DISABLE_PACKING must be 'true/false/1/0/yes/no', got '{raw}'"
            )

        # VIRNUCPRO_DISABLE_FP16
        raw = os.getenv("VIRNUCPRO_DISABLE_FP16", "").strip().lower()
        if raw in ("1", "true", "yes"):
            self.disable_fp16 = True
            logger.warning(
                "FP16 precision DISABLED via VIRNUCPRO_DISABLE_FP16. "
                "Using FP32 (slower, more memory)."
            )
        elif raw in ('', '0', 'false', 'no'):
            self.disable_fp16 = False
        else:
            raise ValueError(
                f"VIRNUCPRO_DISABLE_FP16 must be '1/true/yes/0/false/no', got '{raw}'"
            )

        # VIRNUCPRO_V1_ATTENTION
        raw = os.getenv('VIRNUCPRO_V1_ATTENTION', '').strip().lower()
        if raw in ('true', '1', 'yes'):
            self.v1_attention = True
        elif raw in ('', 'false', '0', 'no'):
            self.v1_attention = False
        else:
            raise ValueError(
                f"VIRNUCPRO_V1_ATTENTION must be 'true/false/1/0/yes/no', got '{raw}'"
            )

        # VIRNUCPRO_VIRAL_CHECKPOINT_MODE
        raw = os.getenv('VIRNUCPRO_VIRAL_CHECKPOINT_MODE', '').strip().lower()
        if raw in ('true', '1', 'yes'):
            self.viral_checkpoint_mode = True
        elif raw in ('', 'false', '0', 'no'):
            self.viral_checkpoint_mode = False
        else:
            raise ValueError(
                f"VIRNUCPRO_VIRAL_CHECKPOINT_MODE must be 'true/false/1/0/yes/no', got '{raw}'"
            )

        # VIRNUCPRO_COMPILE_MODEL
        raw = os.getenv('VIRNUCPRO_COMPILE_MODEL', '').strip().lower()
        if raw in ('true', '1', 'yes'):
            self.compile_model = True
        elif raw in ('', 'false', '0', 'no'):
            self.compile_model = False
        else:
            raise ValueError(
                f"VIRNUCPRO_COMPILE_MODEL must be 'true/false/1/0/yes/no', got '{raw}'"
            )

# Module-level singleton with cached instantiation
@lru_cache(maxsize=1)
def get_env_config() -> EnvConfig:
    """Get cached EnvConfig instance (singleton pattern)."""
    return EnvConfig()
```

**Migration pattern:**
```python
# Before
import os
DISABLE_PACKING = os.getenv('VIRNUCPRO_DISABLE_PACKING', 'false').lower() == 'true'

# After
from virnucpro.core.env_config import get_env_config
env = get_env_config()
if env.disable_packing:
    # ...
```

### Pattern 2: Method Extraction by Semantic Cohesion

**What:** Extract helper functions based on logical groupings, not arbitrary line counts

**When to use:** Functions exceeding 100 lines with clear sub-responsibilities

**Example from AsyncInferenceRunner.run() (976 lines):**
```python
# Before: 976-line run() method with mixed concerns
def run(self, dataloader, progress_callback=None, force_restart=False):
    # 50 lines: setup
    # 100 lines: checkpoint resume logic
    # 600 lines: main inference loop
    # 100 lines: buffer flush
    # 126 lines: cleanup and stats

# After: extracted helpers
def run(self, dataloader, progress_callback=None, force_restart=False):
    """Main inference orchestration."""
    self._setup_inference()

    if self._checkpointing_enabled and not force_restart:
        self._resume_from_checkpoints()

    for result in self._inference_loop(dataloader, progress_callback):
        yield result

    self._flush_buffer_and_cleanup()

def _setup_inference(self):
    """Initialize inference state and monitoring."""
    self.model.eval()
    self.monitor.start_monitoring()
    self.monitor.start_inference_timer()
    self.monitor.set_stage('inference')
    logger.info("Starting async inference loop")

def _resume_from_checkpoints(self):
    """Resume from existing checkpoints if available."""
    resumed_ids, resumed_embs, resume_batch_idx, corrupted_ids = resume_from_checkpoints(...)
    # ... existing resume logic ...

def _inference_loop(self, dataloader, progress_callback):
    """Core inference loop yielding results per batch."""
    # Extracted 600-line loop body

def _flush_buffer_and_cleanup(self):
    """Flush collator buffer and finalize statistics."""
    # Extracted flush and cleanup logic
```

**Boundary criteria:**
- Extract when a block has a clear single responsibility
- Keep related state management together (don't split setup and teardown)
- Prefer fewer large helpers over many tiny helpers (avoid overabstraction)
- Each helper should be testable independently

### Pattern 3: Deque for Queue Operations

**What:** Replace list.pop(0) and list.insert(0, x) with deque.popleft() and appendleft()

**When to use:** Any code using lists as queues (FIFO operations at front)

**Example from VarlenCollator (collators.py:102, 261):**
```python
# Before
self.packed_queue = []  # List used as queue
batch_to_return = self.packed_queue.pop(0)  # O(n) operation

# After
from collections import deque
self.packed_queue = deque()  # Deque optimized for queue ops
batch_to_return = self.packed_queue.popleft()  # O(1) operation
```

**Performance impact:** 40-60k times faster for front operations (per benchmarks below).

### Pattern 4: Shared Validation Utilities

**What:** Extract duplicate validation logic to shared module

**When to use:** Identical validation code duplicated across 2+ locations

**Example - CUDA isolation validation (duplicated in sequence_dataset.py:83-120, 245-282):**
```python
# virnucpro/utils/cuda_validation.py (NEW)
import os
import logging
import torch
from torch.utils.data import get_worker_info

logger = logging.getLogger('virnucpro.utils.cuda_validation')

def validate_cuda_isolation_in_worker():
    """Validate that DataLoader worker has NO CUDA access.

    Ensures workers are CPU-only by checking:
    1. CUDA_VISIBLE_DEVICES is empty (set by worker_init_fn)
    2. torch.cuda.is_available() returns False

    Raises:
        RuntimeError: If worker has CUDA access

    Note:
        Only validates in worker processes (get_worker_info() != None).
        Main process is allowed to have CUDA.
    """
    worker_info = get_worker_info()

    # Only validate in worker process, not main process
    if worker_info is None:
        return

    worker_id = worker_info.id

    # Check CUDA_VISIBLE_DEVICES is hidden
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if cuda_visible != '':
        raise RuntimeError(
            f"Worker {worker_id}: CUDA_VISIBLE_DEVICES='{cuda_visible}' - "
            "expected empty string for CUDA isolation. "
            "Check cuda_safe_worker_init() in dataloader_utils.py"
        )

    # Check torch.cuda.is_available() returns False
    if torch.cuda.is_available():
        raise RuntimeError(
            f"Worker {worker_id}: torch.cuda.is_available() = True - "
            "CUDA should be hidden in worker processes"
        )

    logger.debug(f"Worker {worker_id}: CUDA isolation verified")

# Usage in SequenceDataset and IndexBasedDataset
from virnucpro.utils.cuda_validation import validate_cuda_isolation_in_worker

class SequenceDataset:
    def __iter__(self):
        if not self._validated:
            validate_cuda_isolation_in_worker()
            self._validated = True
        # ... rest of iteration ...
```

### Anti-Patterns to Avoid

- **Eager env var loading at module import time:** Don't call `os.getenv()` at module level - breaks test mocking. Use lazy loading via `@lru_cache` function.
- **Arbitrary line count targets:** Don't extract functions just to hit 100-line limits. Extract when semantic cohesion justifies it.
- **Over-extraction:** Don't create 10-line helper methods that are only called once. Prefer fewer cohesive helpers.
- **List as queue without profile data:** Don't assume all lists need conversion to deque. Only convert when profiling shows it's a bottleneck or when code clearly uses queue semantics (pop(0)/insert(0)).
- **Breaking existing tests:** All refactoring must pass existing 52 test files. Don't change behavior.

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Env var caching | Manual dict cache | functools.lru_cache | Thread-safe, tested, handles edge cases |
| Boolean env var parsing | Custom string comparison | Standardized parser (true/false/1/0/yes/no) | Handles case, whitespace, validation |
| Queue operations | List with pop(0) | collections.deque | 40-60k times faster for front operations |
| Dataclass validation | Manual if/else chains | __post_init__ pattern | Centralized, runs on instantiation, standard pattern |
| Duplicate code | Copy-paste with slight tweaks | Extract to shared utility | Single source of truth, easier testing |

**Key insight:** Environment variable parsing has edge cases (whitespace, case variations, empty strings, invalid values). Centralize parsing logic once with fail-fast validation rather than scattered ad-hoc parsing.

## Common Pitfalls

### Pitfall 1: Eager Module-Level Environment Variable Loading

**What goes wrong:** Loading environment variables at module import time (e.g., `DISABLE_FP16 = os.getenv(...) == 'true'` at top of file) prevents test mocking. Tests that patch `os.environ` after module import see stale cached values.

**Why it happens:** Performance optimization - avoid repeated `os.getenv()` syscalls in hot paths. But caching at import time trades testability for marginal performance gain.

**How to avoid:** Use lazy loading via `@lru_cache` function. First call loads and caches, subsequent calls return cached value. Tests can clear cache via `function.cache_clear()`.

**Warning signs:**
- Test failures when mocking environment variables
- Inability to test different env var configurations in same test run
- Module-level constants derived from `os.getenv()` calls

**Example fix:**
```python
# Bad: Eager loading at module level
DISABLE_FP16 = os.getenv("VIRNUCPRO_DISABLE_FP16", "").lower() in ("1", "true", "yes")

# Good: Lazy loading with cache
from functools import lru_cache

@lru_cache(maxsize=1)
def get_env_config():
    return EnvConfig()  # Loads env vars in __post_init__

# Tests can clear cache
def test_with_fp16_disabled():
    get_env_config.cache_clear()
    with patch.dict(os.environ, {'VIRNUCPRO_DISABLE_FP16': '1'}):
        config = get_env_config()
        assert config.disable_fp16 is True
```

### Pitfall 2: Inconsistent Environment Variable Value Parsing

**What goes wrong:** Different parts of codebase accept different boolean representations ("true", "True", "TRUE", "1", "yes", "YES", "on"). When migrating to centralized config, choosing a single canonical set breaks code expecting other variants.

**Why it happens:** No standard - each `os.getenv()` call site implements its own parsing logic. Over time, different conventions emerge (some accept "yes", others don't).

**How to avoid:** During research phase (this phase), inventory ALL existing env var parsing logic. Document what values each site currently accepts. Choose superset that maintains backward compatibility.

**Warning signs:**
- Tests using different boolean string values ("true" vs "1" vs "yes")
- Documentation inconsistency (README says "true", code checks "1")
- User bug reports about env vars "not working" with certain values

**Example audit:**
```python
# Inventory from codebase:
# async_inference.py:302: os.getenv('VIRNUCPRO_DISABLE_PACKING', 'false').lower() == 'true'
# precision.py:36: os.getenv("VIRNUCPRO_DISABLE_FP16", "").strip().lower() in ("1", "true", "yes")
# esm2_flash.py:209: os.environ.get('VIRNUCPRO_V1_ATTENTION', '').lower() == 'true'

# Standardize to accept superset: ("1", "true", "yes") for True, ("0", "false", "no", "") for False
# Fail on invalid values: raise ValueError for "banana", "maybe", etc.
```

### Pitfall 3: Import Cycles from Shared Utilities

**What goes wrong:** Creating shared utility module (e.g., `utils/cuda_validation.py`) that imports from modules that import it back creates circular dependency. Example: `cuda_validation.py` imports `torch`, `sequence_dataset.py` imports `cuda_validation`, `torch` initialization may trigger imports that circle back.

**Why it happens:** Python allows circular imports if the circular dependency occurs at function call time (not module load time). But it's fragile and breaks when adding new imports.

**How to avoid:**
1. Keep shared utilities dependency-free - only import stdlib and torch (no virnucpro internal imports)
2. Extract pure validation logic - no config access, no logging beyond stdlib
3. Move utilities to `core/` instead of `utils/` to signal "foundation layer"

**Warning signs:**
- ImportError with "cannot import name X from partially initialized module"
- Changing import order fixes the error
- Adding seemingly unrelated import breaks the code

**Example:**
```python
# virnucpro/utils/cuda_validation.py
import os
import torch
from torch.utils.data import get_worker_info
# NO imports from virnucpro.* - avoid cycles

def validate_cuda_isolation_in_worker():
    # Pure validation logic - no config, no internal dependencies
    pass
```

### Pitfall 4: Over-Extraction Creating Method Soup

**What goes wrong:** Extracting every 20-30 line block into separate methods creates "method soup" - navigating code requires jumping through 15 tiny methods to understand a single flow. Cognitive overhead increases despite each method being "simple".

**Why it happens:** Overapplication of Single Responsibility Principle. Misinterpreting "functions should be small" as "functions should be 20 lines max".

**How to avoid:** Extract only when:
1. Method has distinct semantic responsibility (e.g., "resume from checkpoints" is distinct from "inference loop")
2. Method is independently testable and tests would add value
3. Method is called from multiple places (code reuse)
4. Method naturally groups related state mutations

**Warning signs:**
- Methods with names like `_helper1`, `_part2`, `_do_thing_step_a`
- Every method calls 3-5 other private methods
- To understand one public method, must read 10+ private methods
- Git blame shows 10 small methods were extracted from 1 cohesive block

**Example:**
```python
# Bad: Over-extraction
def run(self):
    self._setup_part1()
    self._setup_part2()
    self._setup_part3()
    result = self._main_loop_init()
    result = self._main_loop_body(result)
    result = self._main_loop_cleanup(result)
    self._teardown_part1()
    self._teardown_part2()
    # Now must read 8 methods to understand run()

# Good: Extract by semantic boundaries
def run(self):
    self._setup_inference()  # Groups all setup
    result = self._inference_loop()  # Groups loop logic
    self._cleanup()  # Groups cleanup
    # 3 clear phases, each testable
```

### Pitfall 5: Deque Conversion Breaking Index Access

**What goes wrong:** Converting a list to deque breaks code that uses index access (`queue[0]`, `queue[-1]`) or slicing (`queue[1:5]`). Deque supports indexing but with O(n) complexity, negating performance gains.

**Why it happens:** Lists support both queue operations (pop, insert) AND random access. Deque optimizes queue operations at the cost of slow random access. Converting list→deque without auditing usage breaks assumptions.

**How to avoid:** Before converting list to deque:
1. Search codebase for `<var_name>[` to find index access
2. Search for `<var_name>[1:]` to find slicing
3. If found, either: (a) redesign to avoid index access, or (b) keep as list

**Warning signs:**
- Code reads `first_item = queue[0]` instead of `queue.popleft()`
- Code peeks at queue without removing: `if queue[0] == 'special': ...`
- Code reverses queue: `queue[::-1]`

**Example:**
```python
# Audit packed_queue in collators.py before conversion:
# Line 261: batch_to_return = self.packed_queue.pop(0)  # Queue operation ✓
# Line 263: logger.debug(f"... {len(self.packed_queue)} batches remaining")  # len() works on deque ✓
# Line 284: batch_to_return = self.packed_queue.pop(0)  # Queue operation ✓

# Safe to convert - no index access or slicing found
```

## Code Examples

Verified patterns from official sources and existing codebase:

### Environment Variable Dataclass with Validation
```python
# Source: Existing codebase runtime_config.py + Python dataclass patterns
# Reference: https://docs.python.org/3/library/dataclasses.html
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional
import os

@dataclass
class EnvConfig:
    """Centralized environment variable configuration."""
    disable_packing: bool = False
    disable_fp16: bool = False
    v1_attention: bool = False

    def __post_init__(self):
        """Load and validate environment variables with fail-fast errors."""
        # Parse VIRNUCPRO_DISABLE_PACKING
        raw = os.getenv('VIRNUCPRO_DISABLE_PACKING', 'false').strip().lower()
        if raw in ('true', '1', 'yes'):
            self.disable_packing = True
        elif raw not in ('false', '0', 'no', ''):
            raise ValueError(
                f"VIRNUCPRO_DISABLE_PACKING invalid: '{raw}' "
                f"(expected: true/false/1/0/yes/no)"
            )

        # Parse VIRNUCPRO_DISABLE_FP16
        raw = os.getenv("VIRNUCPRO_DISABLE_FP16", "").strip().lower()
        if raw in ("1", "true", "yes"):
            self.disable_fp16 = True
        elif raw not in ('', '0', 'false', 'no'):
            raise ValueError(
                f"VIRNUCPRO_DISABLE_FP16 invalid: '{raw}' "
                f"(expected: 1/true/yes/0/false/no)"
            )

@lru_cache(maxsize=1)
def get_env_config() -> EnvConfig:
    """Get cached EnvConfig singleton."""
    return EnvConfig()
```

### Deque for Queue Operations
```python
# Source: Python collections.deque documentation
# Reference: https://docs.python.org/3/library/collections.html#collections.deque
from collections import deque

# Before: O(n) list operations
packed_queue = []
packed_queue.append(batch)       # O(1)
batch = packed_queue.pop(0)      # O(n) - moves all elements left

# After: O(1) deque operations
packed_queue = deque()
packed_queue.append(batch)       # O(1)
batch = packed_queue.popleft()   # O(1) - no element movement
```

### Shared CUDA Validation Utility
```python
# Source: Extracted from sequence_dataset.py lines 83-120, 245-282
import os
import torch
from torch.utils.data import get_worker_info

def validate_cuda_isolation_in_worker():
    """Validate DataLoader worker has no CUDA access.

    Raises:
        RuntimeError: If worker has CUDA access
    """
    worker_info = get_worker_info()
    if worker_info is None:
        return  # Main process allowed to have CUDA

    worker_id = worker_info.id
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')

    if cuda_visible != '':
        raise RuntimeError(
            f"Worker {worker_id}: CUDA_VISIBLE_DEVICES='{cuda_visible}' "
            "(expected empty for isolation)"
        )

    if torch.cuda.is_available():
        raise RuntimeError(
            f"Worker {worker_id}: CUDA available but should be hidden"
        )
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Scattered os.getenv() calls | Centralized dataclass config | Python 3.7+ (dataclasses) | Type safety, validation, testability |
| list.pop(0) for queues | collections.deque.popleft() | Python 2.4+ (deque) | 40-60k times faster for front ops |
| Copy-paste validation | Shared utility functions | Always best practice | Single source of truth, easier testing |
| Arbitrary function length limits | Extract by semantic cohesion | Modern refactoring (2020+) | More maintainable than method soup |

**Deprecated/outdated:**
- **Module-level env var constants:** Break test mocking, can't change at runtime. Replaced by lazy-loaded cached functions.
- **String-based env var keys everywhere:** `os.getenv("KEY")` scattered in code. Replaced by typed dataclass properties.
- **Lists for queue operations:** O(n) performance. Replaced by deque for O(1) operations.

## Open Questions

Things that couldn't be fully resolved:

1. **Old environment variable name backward compatibility**
   - What we know: User requested "standardize naming" but didn't specify which names are inconsistent
   - What's unclear: Should we maintain old names with deprecation warnings or hard-cut?
   - Recommendation: Document current names in EnvConfig, plan migration in separate phase if user identifies inconsistencies. Phase 11 keeps existing names unchanged.

2. **Testing strategy for extracted helpers**
   - What we know: Codebase has 52 existing test files covering current code
   - What's unclear: Do extracted private methods need new unit tests, or is existing integration coverage sufficient?
   - Recommendation: Extract helpers as private methods (`_helper_name`). Run full test suite after extraction to verify 1:1 equivalence. Only add new tests if existing coverage drops or if helper has complex logic worth isolating.

3. **EnvConfig instantiation pattern**
   - What we know: Singleton via `@lru_cache` is standard Python pattern
   - What's unclear: Should we use module-level singleton `ENV = get_env_config()` or always call `get_env_config()` at use site?
   - Recommendation: Always call `get_env_config()` at use site for testability. Tests can clear cache and mock env vars between test cases. Avoid module-level singleton.

4. **Scope of queue migration**
   - What we know: collators.py packed_queue uses .pop(0), clear candidate for deque
   - What's unclear: Are there other list-as-queue usages in codebase? Should we audit all .append()/.pop() patterns?
   - Recommendation: Search for `.pop\(0\)` and `.insert\(0,` patterns in phase planning. Only convert proven queue patterns (FIFO semantics). Don't convert lists used for random access.

## Sources

### Primary (HIGH confidence)
- [Python dataclasses documentation](https://docs.python.org/3/library/dataclasses.html) - Official stdlib reference for @dataclass pattern
- [Python collections.deque documentation](https://docs.python.org/3/library/collections.html#collections.deque) - Official reference for deque performance characteristics
- [Real Python: Python's deque: Implement Efficient Queues and Stacks](https://realpython.com/python-deque/) - Published January 12, 2026, current best practices
- Existing codebase: runtime_config.py (lines 12-85) - Established dataclass pattern with __post_init__ validation
- Existing codebase: async_inference.py, gpu_worker.py, prediction.py - Current env var usage patterns (19 call sites)

### Secondary (MEDIUM confidence)
- [Python configuration and dataclasses - Alexandra Zaharia](https://alexandra-zaharia.github.io/posts/python-configuration-and-dataclasses/) - __post_init__ pattern for env loading
- [Python deque vs list performance comparison - DEV Community](https://dev.to/wnleao/python-deque-vs-list-time-comparison-5ch4) - Benchmark data: 40-60k times faster for front operations
- [Settings Management - Pydantic Documentation](https://docs.pydantic.dev/latest/concepts/pydantic_settings/) - Alternative approach (not chosen due to dependency weight)

### Tertiary (LOW confidence)
- [Best practices for configurations in Python pipelines - Micropole](https://belux.micropole.com/blog/python/blog-best-practices-for-configurations-in-python-based-pipelines/) - General configuration patterns (not specific to env vars)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - All stdlib modules, documented in official Python docs, existing usage in codebase
- Architecture: HIGH - Patterns verified in existing codebase (runtime_config.py, sequence_dataset.py), supported by official documentation
- Pitfalls: HIGH - Based on real codebase audit (19 env var sites, 2 duplicate CUDA validations, 976-line functions), cross-referenced with Python best practices

**Research date:** 2026-02-09
**Valid until:** 90 days (stdlib APIs stable, patterns from 2026 sources)
