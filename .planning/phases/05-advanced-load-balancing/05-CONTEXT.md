# Phase 5: Advanced Load Balancing - Context

**Gathered:** 2026-01-26
**Status:** Ready for planning

<domain>
## Phase Boundary

This phase delivers work stealing and dynamic rebalancing to ensure optimal GPU utilization across heterogeneous hardware. Idle GPUs automatically pull work from busy GPUs' queues, with work distributed proportional to GPU compute capability. Real-time dashboard shows per-GPU utilization, throughput, queue depths, and work stealing events.

**Out of scope:** New optimization techniques beyond load balancing, monitoring systems not directly related to work distribution.

</domain>

<decisions>
## Implementation Decisions

### Work Stealing Mechanism
- **Discovery method:** Queue polling (idle workers periodically check other GPU queues)
- **Steal granularity:** Whole files (leverage existing checkpoint system, simpler than batch-level stealing)
- **Polling frequency:** Moderate (every 2-5 seconds) - balances responsiveness and overhead
- **Contention handling:** Skip and try next GPU when source queue is locked (avoids blocking)

### Load Distribution Strategy
- **Compute estimation:** Sequence count × average length per file (more accurate than count alone, uses existing sequence counting)
- **Rebalance trigger:** Claude's discretion (determine optimal imbalance threshold based on workload characteristics)
- **Auto-split handling:** Treat as independent files (each split assigned independently for maximum load balancing flexibility)
- **Adaptive learning:** No - use static assignment with work stealing for corrections (simpler, proven effective)

### GPU Capability Weighting
- **Weight determination:** Static lookup table with predefined weights for GPU models (simple, fast, no startup overhead)
- **GPU coverage:** Consumer + datacenter GPUs (3090, 4090, 4080, 3080, A100, H100, V100, A6000)
- **Weight application:** Proportional file counts (4090 with weight=1.5 gets 1.5x files vs 3090 with weight=1.0)
- **Unknown GPU handling:** Default weight 1.0 (safe fallback for models not in lookup table)

### Dashboard & Monitoring
- **Core metrics (all required):**
  - Per-GPU utilization % (compute utilization 0-100%)
  - Per-GPU throughput (sequences/sec or tokens/sec)
  - Queue depths (files/batches queued per GPU)
  - Work stealing events (when and how much work stolen between GPUs)
- **Display format:** Compact table using Rich library (one row per GPU with columns for metrics)
- **Update frequency:** Moderate (2-3 seconds) - good balance between responsiveness and overhead
- **Memory display:** No - focus dashboard on utilization (memory already managed by Phase 4)

### Claude's Discretion
- Exact polling implementation details (thread vs process-based)
- Rebalance trigger threshold (imbalance % that triggers work stealing)
- Queue data structure and locking mechanisms
- Dashboard column ordering and visual styling
- Logging verbosity for work stealing events

</decisions>

<specifics>
## Specific Ideas

- "Keep it simple - whole file stealing with existing checkpoints"
- "2-5 second polling feels right for files that take minutes to process"
- "Skip locked queues - don't want workers blocking each other"
- "Static weights are fine - no need to profile at startup"
- "Include datacenter GPUs - this will be used in production environments"
- "Compact table dashboard like existing Rich progress - consistent UX"

</specifics>

<deferred>
## Deferred Ideas

None - discussion stayed within phase scope

</deferred>

---

*Phase: 05-advanced-load-balancing*
*Context gathered: 2026-01-26*
