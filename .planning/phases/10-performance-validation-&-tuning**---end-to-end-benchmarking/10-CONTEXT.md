# Phase 10: Performance Validation & Tuning - Context

**Gathered:** 2026-02-06
**Status:** Ready for planning

<domain>
## Phase Boundary

End-to-end benchmarking and optimization to ensure the pipeline meets the <10 hour target with >80% GPU utilization. This phase validates that all previous work (async DataLoader, sequence packing, multi-GPU, FP16, checkpointing) delivers the 4.5× speedup target. Focus is on measurement, bottleneck identification, and parameter tuning - NOT adding new capabilities.

</domain>

<decisions>
## Implementation Decisions

### Benchmark Scope and Methodology
- **Primary workload:** Real production sample (actual viral nucleotide data from pipeline)
- **Sample count:** Single full run (6M sequences) for target validation + smaller subset runs for profiling/tuning
- **Profiling depth:** Detailed profiling including DataLoader wait times, packing efficiency, GPU utilization per batch - full diagnostics
- **Baseline comparison:** Yes - run same workload on v1.0 to measure actual speedup achieved (validates 4.5× claim)

### Success Criteria and Thresholds
- **Time target:** <10 hours is a HARD REQUIREMENT - must hit target or keep tuning
- **GPU utilization:** 70%+ is acceptable (some overhead inevitable, strict 80% not required)
- **Packing efficiency:** Must maintain 90%+ efficiency (core to performance, below 90% indicates problem)
- **Multi-GPU scaling:** 2 GPUs = 1.9x+ speedup (95% efficiency) - near-perfect scaling expected
- **All criteria must be met:** Time, GPU utilization, packing efficiency, and scaling all required for phase completion

### Tuning Priorities and Approach
- **Priority order:** Biggest bottleneck first - profile-guided optimization regardless of category
- **Tuning budget:** Moderate - keep tuning parameters until <10h target met
- **Tuning scope:** All parameters in scope:
  - DataLoader parameters (num_workers, prefetch_factor, batch size)
  - Packing parameters (buffer size, token budget, thresholds)
  - Checkpoint parameters (frequency, async write tuning)
  - CUDA parameters (stream config, memory allocation)
- **Code changes:** If hitting target requires code changes (not just parameters), document for follow-up Phase 10.1 - this phase is measurement and parameter tuning only

### Telemetry and Observability
- **Metrics captured:** All categories during production runs:
  - Throughput metrics (sequences/sec, tokens/sec per batch and overall)
  - Packing efficiency (token utilization, padding waste per batch)
  - I/O wait times (DataLoader queue state, batch arrival timing)
  - GPU metrics (utilization, memory usage, stream sync overhead)
- **Reporting format:** Real-time progress bars showing throughput, GPU util, packing efficiency - live feedback during runs
- **Data persistence:** Both detailed JSON logs (per-batch metrics) AND summary reports (aggregated statistics)
- **Metric granularity:** Per-batch metrics in detailed JSON, windowed aggregates in summary report - both captured

### Claude's Discretion
- Specific progress bar layout and formatting
- Exact JSON schema for telemetry logs
- Summary report format (table vs YAML vs other)
- Choice of visualization if plotting metrics post-run

</decisions>

<specifics>
## Specific Ideas

- **Typical hardware:** 2 GPUs (not 4) - scaling tests should focus on 2-GPU efficiency
- **Production workload:** ~6M sequences per sample - use real data for most realistic validation
- **v1.0 comparison important:** Want to validate the 4.5× speedup claim with actual measurements

</specifics>

<deferred>
## Deferred Ideas

None - discussion stayed within phase scope (measurement, profiling, parameter tuning).

</deferred>

---

*Phase: 10-performance-validation-tuning*
*Context gathered: 2026-02-06*
