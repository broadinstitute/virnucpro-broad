---
phase: 06-performance-validation
plan: 01
subsystem: testing
tags: [benchmarks, pytest, gpu-monitoring, nvitop, synthetic-data, infrastructure]

# Dependency graph
requires:
  - phase: 04.1-persistent-model-loading
    plan: 06
    provides: Integration test gap closure and complete Phase 4.1
provides:
  - Benchmark infrastructure with pytest fixtures
  - Synthetic data generator for controlled tests
  - Enhanced GPU monitoring with nvitop support
  - Multi-GPU test configurations (1, 2, 4, 8 GPUs)
affects: [future-performance-tests, ci-pipeline, scaling-validation]

# Tech tracking
tech-stack:
  added:
    - nvitop (optional, with fallback)
    - pytest-benchmark (fixture integration)
  patterns:
    - "BenchmarkTimer wrapping torch.utils.benchmark.Timer for CUDA sync"
    - "GPU monitoring with background threading and log file output"
    - "Synthetic FASTA generation with BioPython for reproducible tests"
    - "Multi-GPU configuration fixtures with parametrization"

key-files:
  created:
    - tests/benchmarks/__init__.py
    - tests/benchmarks/conftest.py
    - tests/benchmarks/data_generator.py
    - tests/benchmarks/utils.py
    - virnucpro/utils/gpu_monitor.py
  modified: []

key-decisions:
  - "Use nvitop Python API for GPU monitoring (fallback to torch.cuda if unavailable)"
  - "Write GPU metrics to logs/gpu_metrics_{timestamp}.log for MON-01 requirement"
  - "Support both real and synthetic data via preset configurations (TINY, SMALL, MEDIUM, LARGE)"
  - "Wrap torch.utils.benchmark.Timer for accurate CUDA timing with automatic synchronization"

patterns-established:
  - "Pattern: Benchmark fixtures provide GPU monitoring, data generation, and multi-GPU configs"
  - "Pattern: Enhanced monitoring tracks per-stage metrics (translation, DNABERT, ESM-2, merge)"
  - "Pattern: Synthetic data generator uses seed for reproducibility and BioPython for compatibility"
  - "Pattern: Result formatting supports both markdown (human) and JSON (machine) output"

# Metrics
duration: 5min
completed: 2026-01-26
---

# Phase 06 Plan 01: Benchmark Infrastructure Summary

**Created benchmark infrastructure with synthetic data generation, GPU monitoring, and multi-GPU test configurations**

## Performance

- **Duration:** 5 min
- **Started:** 2026-01-26T18:38:11Z
- **Completed:** 2026-01-26T18:43:38Z
- **Tasks:** 3
- **Files created:** 5

## Accomplishments

- Created pytest benchmark fixtures for GPU availability detection and configuration
- Implemented BenchmarkTimer wrapper around torch.utils.benchmark.Timer with CUDA synchronization
- Added GPU memory tracking utilities and scaling calculation helpers
- Created multi-GPU test configuration fixtures (1, 2, 4, 8 GPUs) with parametrization
- Implemented synthetic FASTA generation with viral-like DNA sequences
- Added preset configurations (TINY: 10, SMALL: 100, MEDIUM: 1K, LARGE: 10K sequences)
- Created generate_benchmark_dataset() for multi-file test datasets with metadata
- Implemented real_sample_loader() for optional real viral samples
- Enhanced GPU monitoring with NvitopMonitor class using nvitop Python API
- Added tracking for GPU utilization %, memory, temperature, power draw per GPU
- Implemented background monitoring thread with configurable sampling interval
- Created log file output (logs/gpu_metrics_{timestamp}.log) for MON-01 compliance
- Added metric aggregation methods (average, peak, throughput calculations)
- Implemented per-stage performance tracking with stage transitions
- Created fallback to torch.cuda when nvitop unavailable
- Maintained backward compatibility with existing GPUMonitor class
- Added result formatting utilities for markdown and JSON output
- Created comparison utilities for correctness validation with BF16/FP32 tolerance

## Task Commits

Each task was committed atomically:

1. **Task 1: Create benchmark infrastructure and fixtures** - `e2725be` (feat)
   - tests/benchmarks/__init__.py (40 lines)
   - tests/benchmarks/conftest.py (290 lines)
   - tests/benchmarks/utils.py (508 lines)

2. **Task 2: Implement synthetic data generator** - `77c36c2` (feat)
   - tests/benchmarks/data_generator.py (408 lines)

3. **Task 3: Enhance GPU monitoring for benchmarks** - `179b52b` (feat)
   - virnucpro/utils/gpu_monitor.py (567 lines)

## Files Created/Modified

**Created:**
- `tests/benchmarks/__init__.py` - Package docstring and exports for benchmark suite
- `tests/benchmarks/conftest.py` - pytest fixtures for GPU monitoring, data generation, and multi-GPU configs
- `tests/benchmarks/utils.py` - BenchmarkTimer, memory tracking, scaling calculations, result formatting
- `tests/benchmarks/data_generator.py` - Synthetic FASTA generation with presets and real sample loading
- `virnucpro/utils/gpu_monitor.py` - Enhanced GPU monitoring with nvitop integration and logging

## Decisions Made

**nvitop-with-fallback:** Use nvitop Python API for enhanced monitoring (utilization, temperature, power) with automatic fallback to torch.cuda if unavailable. Provides better metrics without breaking systems without nvitop.

**log-file-output:** Write all GPU metrics to logs/gpu_metrics_{timestamp}.log files for MON-01 requirement (GPU utilization and memory must be logged). Enables post-run analysis and compliance.

**preset-configurations:** Define standard test sizes (TINY, SMALL, MEDIUM, LARGE) as presets with documented sequence counts. Ensures consistent benchmarking and clear communication about test scale.

**benchmark-timer-wrapper:** Wrap torch.utils.benchmark.Timer to handle CUDA synchronization automatically. Prevents common pitfall of measuring kernel launch time instead of execution time.

**multi-gpu-parametrization:** Use pytest parametrized fixtures for GPU configurations (1, 2, 4, 8 GPUs). Tests run automatically across all available configurations with proper skipping.

**per-stage-tracking:** Enhanced monitor tracks pipeline stage transitions (translation, DNABERT, ESM-2, merge) for bottleneck identification. Critical for understanding where GPU time is spent.

**backward-compatibility:** Keep GPUMonitor class alongside new NvitopMonitor for existing code. Ensures no breaking changes to Phase 1-4 implementations using gpu_monitor.

## Technical Details

### Benchmark Infrastructure
- **pytest-benchmark integration:** Fixtures provide benchmark configuration (warmup, iterations, timing)
- **GPU availability detection:** Automatic test skipping when GPU/multi-GPU not available
- **Directory fixtures:** Temporary directories for test data, outputs, and reports
- **Multi-GPU parametrization:** Tests run across 1, 2, 4, 8 GPU configurations automatically

### Synthetic Data Generation
- **BioPython compatibility:** Uses SeqRecord format for standard FASTA writing
- **Configurable GC content:** Target GC ratio (default: 50%) for realistic sequences
- **Reproducible generation:** Seed parameter ensures identical datasets across runs
- **Metadata tracking:** JSON files document sequence counts, file sizes, generation parameters
- **Multi-file datasets:** Support for splitting sequences across multiple files for multi-GPU testing

### Enhanced GPU Monitoring
- **nvitop integration:** Direct Python API for GPU metrics (no subprocess overhead)
- **Comprehensive metrics:** Utilization %, memory used/total, temperature, power draw per GPU
- **Background threading:** Non-blocking monitoring with configurable sampling interval (default: 1.0s)
- **Per-stage tracking:** set_stage() method associates metrics with pipeline phases
- **Log file format:** CSV-like format with timestamp, device_id, metrics, stage name
- **Aggregated statistics:** Calculate average/min/max utilization, peak memory, per-stage summaries
- **Fallback mode:** Automatic degradation to torch.cuda metrics when nvitop unavailable
- **Backward compatible:** GPUMonitor class remains for existing pipeline code

### Utilities
- **BenchmarkTimer:** Wraps torch.utils.benchmark.Timer with device parameter and memory tracking
- **Scaling calculations:** ScalingResult dataclass with speedup, efficiency, throughput metrics
- **Result formatting:** Dual output (markdown for humans, JSON for CI automation)
- **Comparison utilities:** compare_outputs() with appropriate BF16/FP32 tolerance (rtol=1e-3)
- **Memory tracking:** track_peak_memory() and get_current_memory() using torch.cuda APIs

## Next Phase Readiness

**Ready for Phase 6 Plan 2:** Benchmark infrastructure complete. Can now:
- Generate synthetic test datasets of varying sizes
- Monitor GPU utilization and memory during benchmarks
- Test multi-GPU configurations (1, 2, 4, 8 GPUs)
- Track per-stage performance metrics
- Export results in markdown and JSON formats
- Validate correctness against vanilla implementation

**Infrastructure supports:**
- Scaling validation benchmarks (linear speedup testing)
- Throughput measurements (sequences/second per stage)
- Memory profiling (peak usage per GPU)
- Bottleneck identification (per-stage utilization)
- CI integration (JSON output, pytest markers)

**Blockers/Concerns:**
- nvitop is optional dependency (not in requirements.txt). Add if enhanced monitoring desired, or rely on fallback.
- pytest-benchmark not in requirements.txt. Add for automated benchmark regression tracking.
- Real viral samples not included in repo. Users must provide or tests use synthetic only.

## Deviations from Plan

None - plan executed exactly as written.

## Files Changed

```
tests/benchmarks/
├── __init__.py          (40 lines, package docs and exports)
├── conftest.py          (290 lines, pytest fixtures)
├── data_generator.py    (408 lines, synthetic FASTA generation)
└── utils.py             (508 lines, timing and formatting utilities)

virnucpro/utils/
└── gpu_monitor.py       (567 lines, enhanced monitoring)

logs/                    (created, gitignored)
```

**Total lines added:** 1,813
**Files created:** 5

## Validation

- ✓ Benchmark infrastructure files compile successfully
- ✓ Synthetic data generator contains all required functions (generate_synthetic_fasta, generate_benchmark_dataset, real_sample_loader)
- ✓ GPU monitor contains all required classes (GPUMonitor, NvitopMonitor, GPUMetrics)
- ✓ All preset configurations present (TINY, SMALL, MEDIUM, LARGE)
- ✓ Backward compatibility maintained (GPUMonitor class preserved)
- ✓ Logging capability implemented (logs/gpu_metrics_*.log)
- ✓ Multi-GPU test configurations (1, 2, 4, 8 GPUs)

Infrastructure ready for scaling and performance tests.
