# Phase 10: Performance Validation & Tuning - Context

**Gathered:** 2026-02-08 (updated from 2026-02-06)
**Status:** Ready for planning

<domain>
## Phase Boundary

End-to-end benchmarking proving the v2.0 pipeline meets performance targets on 2x RTX 4090 hardware, with telemetry logging for production monitoring. This phase validates that all previous work (async DataLoader, sequence packing, multi-GPU, FP16, checkpointing, FlashAttention) delivers the required speedup. Focus is on measurement, correctness validation, and parameter tuning - NOT adding new capabilities.

</domain>

<decisions>
## Implementation Decisions

### Benchmark Workload
- **Dataset:** subset_1M sample (USA-MA-Broad_MGH-22989, ~600K sequences after translation)
- **Full pipeline:** Run end-to-end from FASTA input to prediction_results (translate + DNABERT-S + ESM-2 + predict), not just ESM-2 isolation
- **Attention mode:** FlashAttention (v2.0 default path) — no --v1-attention benchmarks
- **GPU configurations:** Two benchmark runs:
  1. 1x RTX 4090 (single-GPU baseline)
  2. 2x RTX 4090 (multi-GPU scaling validation)
- **GPU auto-detection:** Use whatever GPUs are available — no hardcoded GPU count

### Success Criteria and Thresholds
- **Time targets (HARD):**
  - 1x 4090: <1 hour for the 1M subset
  - 2x 4090: <30 minutes for the 1M subset
- **GPU utilization:** >80% during embedding steps
- **Packing efficiency:** >90% maintained on production workloads
- **Multi-GPU scaling:** 2x GPUs = 1.9x+ speedup (95% efficiency)
- **Speedup over v1.0:** >=4.0x (v1.0 baseline: ~3.5 hours on 2x 4090)
- **Correctness:** >=99% consensus label agreement with v1.0 output, score deviance within acceptable range
- **All criteria must be met** for phase completion

### v1.0 Baseline Method
- **Known timing:** v1.0 takes ~3.5 hours on 2x RTX 4090 for the 1M subset
- **No re-run required:** Use known v1.0 timing for speedup calculation
- **Correctness comparison:** Compare v2.0 predictions against existing v1.0 output files (using compare_virnucpro_outputs.py pattern)

### Telemetry and Observability
- **Metrics captured (standard set):**
  - Throughput: tokens/sec, sequences/sec
  - Packing efficiency: token utilization per batch
  - I/O wait: DataLoader queue state, batch arrival timing
  - GPU: utilization %, memory usage
  - Per-stage timing breakdown
- **Output:** Log file only (no separate JSON file) — integrated with virnucpro logging
- **Live progress:** Rich progress bar (tqdm/rich) showing sequences processed, ETA, tokens/sec
- **Final summary:** Print a clear summary block after pipeline completion with key metrics (wall time, throughput, packing efficiency, GPU util)

### Claude's Discretion
- Specific progress bar layout and formatting
- Exact log format for telemetry lines
- Summary block formatting
- DataLoader/packing parameter sweep methodology
- Which parameters to tune and in what order

</decisions>

<specifics>
## Specific Ideas

- **Hardware:** 2x RTX 4090 (not A100) — benchmarks must work on this hardware
- **v1.0 timing known:** 3.5 hours on 2x 4090 for 1M subset — no need to re-run
- **Correctness already characterized:** Phase 10.2 found 99.87% consensus label agreement, 0.13% mismatches all borderline. Benchmark should re-validate this.
- **Production data paths:**
  - v1.0 output: `/data/Broad_Viral_Respiratory/data/in/raw/full/USA-MA-Broad_MGH-22989-2024.lNDM_F11.HLWLWDRX5.1.hs_depleted.subset_1M_merged/`
  - v2.0 output: `/tmp/new_virnucpro_out/USA-MA-Broad_MGH-22989-2024.lNDM_F11.HLWLWDRX5.1.hs_depleted.subset_1M_merged/`

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 10-performance-validation-tuning*
*Context gathered: 2026-02-08*
