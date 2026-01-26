# Phase 6: Performance Validation - Context

**Gathered:** 2026-01-26
**Status:** Ready for planning

<domain>
## Phase Boundary

Benchmarking and validating that GPU optimizations deliver <10 hour processing time with linear GPU scaling. This phase creates the test infrastructure, runs benchmarks across 1/2/4/8 GPU configurations, measures performance metrics, and validates correctness against vanilla implementation. Does NOT include implementing new optimizations - those are in earlier phases.

</domain>

<decisions>
## Implementation Decisions

### Benchmark Test Data
- **Both real and synthetic data**: Real viral samples for end-to-end validation, synthetic sequences for controlled scaling tests
- **Full size range**: Cover tiny (10s), small (100s), typical (1000s), and large (10000s+) sequences for comprehensive edge case testing
- **Data management**: Commit small test files to repo, generate synthetic data on-demand, user provides large real samples (uncommitted)
- **Real sample documentation**: Document sequence count/size distribution, expected runtime baseline (current 45+ hours), and source/provenance for all real test data

### Performance Metrics & Reporting
- **Comprehensive metrics**: Track GPU utilization %, memory usage (peak and average), throughput (sequences/sec), and per-stage breakdown (translation, DNABERT, ESM-2, merge, prediction)
- **Dual output format**: Both markdown report (human-readable) and JSON data (machine-readable for CI/automation)
- **Hybrid collection approach**: Lightweight real-time logging during execution + detailed post-run analysis from logs/checkpoints

### Claude's Discretion
- GPU utilization measurement tool choice (nvidia-smi, nvitop, PyTorch profiler, or combination)
- Exact implementation of real-time vs post-run metric collection
- Bottleneck analysis implementation details

### Success Criteria & Thresholds
- **Time target with tolerance**: <10 hours on 4 GPUs is target with ±10% tolerance (9-11 hours acceptable)
- **Qualitative scaling verification**: Verify linear scaling trend (more GPUs = proportionally faster), exact ratios less critical than general linearity
- **Average GPU utilization**: >80% measured as average across entire embedding stage (DNABERT + ESM-2)
- **Threshold violations**: Pass benchmark with warnings when thresholds not met (document shortfall but don't fail)

### Vanilla Comparison Testing
- **Comprehensive comparison**: Compare final predictions, intermediate embeddings, and consensus results between optimized and vanilla pipelines
- **Moderate tolerance**: Relative tolerance of 1e-3 for floating-point comparisons (allows BF16/FP32 precision differences and operation reordering)
- **Dual baseline approach**: Pre-computed reference outputs for CI speed, fresh vanilla runs available for validation
- **Mismatch handling**: Investigate tolerance violations to determine if differences are within acceptable numerical variance (don't hard fail immediately)

</decisions>

<specifics>
## Specific Ideas

- Current baseline: 45+ hours for one sample (ESM-2 embedding is the bottleneck)
- Target: Under 10 hours on 4 GPUs (4.5x speedup minimum)
- Test configurations: 1, 2, 4, 8 GPU setups for scaling validation
- Success criteria from roadmap: near-linear speedup (2 GPUs ≈ 1.8x, 4 GPUs ≈ 3.5x, 8 GPUs ≈ 7x)

</specifics>

<deferred>
## Deferred Ideas

None - discussion stayed within phase scope (benchmarking and validation of existing optimizations).

</deferred>

---

*Phase: 06-performance-validation*
*Context gathered: 2026-01-26*
