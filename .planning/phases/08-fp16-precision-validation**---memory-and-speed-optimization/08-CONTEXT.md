# Phase 8: FP16 Precision Validation - Context

**Gathered:** 2026-02-05
**Status:** Ready for planning

<domain>
## Phase Boundary

Convert ESM-2 and DNABERT-S models from FP32 to FP16 precision to achieve throughput improvement while maintaining embedding accuracy. Memory reduction is a side benefit, but primary goal is total runtime reduction.

</domain>

<decisions>
## Implementation Decisions

### Precision Strategy
- Start with full FP16 conversion (entire model)
- Move operations back to FP32 only if tests fail (data-driven decisions)
- Switch FlashAttention from BF16 to FP16 (align with model precision)
- Prioritize performance over accuracy if tradeoffs needed (accept >0.99 similarity for full FP16 speedup)

### Validation Approach
- Similarity threshold: >0.99 (matches Phase 6 packing validation)
- Test dataset: 10K sequences (medium sample, representative length distribution)
- Stratified testing: Validate short/medium/long sequences separately to catch length-dependent issues
- Additional validation: Statistical analysis (embedding distribution stats - mean, std, outliers)

### Performance Targets
- Primary metric: Total runtime reduction (end-to-end most meaningful)
- Throughput improvement: 1.8-2x target (optimal range)
- Track both end-to-end runtime AND tokens/second (isolate GPU improvement from I/O)
- GPU utilization: >80% (matches ROADMAP Phase 10 target)

### Fallback Handling
- Rollback triggers: Both similarity <0.99 AND NaN/Inf detection
- Deployment: Feature flag control (VIRNUCPRO_DISABLE_FP16 env var for safety)
- FP32 baseline comparison timing: On-demand diagnostic (--fp32-compare flag for troubleshooting)
- Diagnostic workflow: Manual comparison - user runs flag, inspects similarity metrics

### Claude's Discretion
- Specific layers to move to FP32 if full FP16 fails tests
- Exact statistical metrics to track (beyond mean/std)
- Benchmark methodology details
- Specific NaN/Inf monitoring implementation

</decisions>

<specifics>
## Specific Ideas

**Validation workflow:**
1. Phase 8 validation: One-time thorough FP32 baseline comparison to prove FP16 works
2. Production default: FP16 becomes standard, FP32 available via flag
3. Diagnostics: Users run `--fp32-compare` to re-run same data in FP32 when troubleshooting

**Performance focus:**
- Runtime reduction is primary goal, not memory savings
- Even if memory reduction is modest, throughput improvement justifies FP16

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 08-fp16-precision-validation*
*Context gathered: 2026-02-05*
