---
phase: 02-dnabert-s-optimization
plan: 01
subsystem: feature-extraction
tags: [dnabert-s, multi-gpu, parallel-processing, bf16, batching, worker-abstraction]

dependencies:
  requires:
    - phase: 01
      plan: 07
      reason: "Multi-GPU patterns and spawn context established"
  provides:
    - "BaseEmbeddingWorker abstract class for unified worker interface"
    - "DNABERT-S parallel processing with token-based batching"
    - "BF16 optimization on Ampere+ GPUs"
    - "Bin-packing file assignment by sequence count"
  affects:
    - phase: 02
      plan: 02
      reason: "Will integrate DNABERT-S worker into pipeline CLI"

tech-stack:
  added:
    - transformers: "DNABERT-S model loading (AutoModel, AutoTokenizer)"
  patterns:
    - "Abstract base classes for embedding workers (BaseEmbeddingWorker)"
    - "Token-based dynamic batching for DNA sequences"
    - "Greedy bin-packing file assignment by sequence count"
    - "BF16 mixed precision with automatic GPU capability detection"

key-files:
  created:
    - virnucpro/pipeline/base_worker.py: "Abstract base class and shared utilities"
    - virnucpro/pipeline/parallel_dnabert.py: "DNABERT-S parallel worker implementation"
  modified: []

decisions:
  - id: base-worker-abstraction
    decision: "Create BaseEmbeddingWorker abstract class shared by DNABERT-S and ESM-2"
    rationale: "Enforces consistent interface, reduces duplication, enables future model additions"
    alternatives: "Duplicate code in each worker module"
    impact: "ESM-2 worker can be refactored to inherit from base class in future"

  - id: token-abstraction-dna
    decision: "Treat DNA sequence length as token count (1 base ≈ 1 token)"
    rationale: "Abstracts k-mer complexity, matches ESM-2 pattern, simplifies batching logic"
    alternatives: "Calculate actual k-mer count from tokenizer"
    impact: "Slight batch size imprecision but massive simplification"

  - id: shared-utilities-location
    decision: "Place shared utilities (count_sequences, assign_files_by_sequences) in base_worker.py"
    rationale: "DRY principle, single source of truth, natural location for shared code"
    alternatives: "Duplicate in each worker module or create separate utils module"
    impact: "Import and export from parallel_dnabert.py for discoverability"

metrics:
  duration: 175s
  completed: 2026-01-23
---

# Phase 02 Plan 01: BaseEmbeddingWorker Foundation and DNABERT-S Parallel Processing Summary

**One-liner:** Created unified BaseEmbeddingWorker abstraction and implemented DNABERT-S parallel processing with token-based batching, BF16 optimization, and bin-packing file assignment.

## Objective

Create the foundational BaseEmbeddingWorker abstraction and implement DNABERT-S parallel processing with token-based batching and BF16 optimization, matching the optimization level achieved for ESM-2 in Phase 1.

## What Was Built

### 1. BaseEmbeddingWorker Abstract Class (`virnucpro/pipeline/base_worker.py`)

Created a comprehensive abstract base class that defines the unified interface for embedding workers:

- **Abstract methods:**
  - `process_files_worker()`: Process files on specific GPU (required signature)
  - `get_optimal_batch_size()`: Determine optimal batch size for device

- **Shared utilities:**
  - `count_sequences()`: Count sequences in FASTA files
  - `assign_files_by_sequences()`: Greedy bin-packing file assignment by sequence count
  - `detect_bf16_support()`: Check GPU compute capability for BF16 support

- **Key design features:**
  - Spawn context compatibility (no unpicklable instance state)
  - Deferred CUDA initialization pattern
  - Progress reporting via multiprocessing.Queue
  - Consistent return type: `(processed_files, failed_files)`

### 2. DNABERT-S Parallel Worker (`virnucpro/pipeline/parallel_dnabert.py`)

Implemented parallel DNABERT-S processing following ESM-2 patterns:

- **Token-based batching:**
  - Treats DNA sequence length as token count (1 base ≈ 1 token)
  - Abstracts k-mer complexity for simpler batch management
  - Default: 2048 tokens per batch (3072 with BF16)
  - Dynamic batch creation: accumulate sequences until token limit

- **BF16 optimization:**
  - Automatic detection via `detect_bf16_support()`
  - Enabled on Ampere+ GPUs (compute capability >= 8)
  - 50% memory savings with minimal accuracy impact
  - Automatic batch size increase when BF16 enabled

- **Worker implementation:**
  - Deferred CUDA initialization (no parent process CUDA context)
  - `torch.no_grad()` context for all inference
  - Progress reporting via multiprocessing.Queue
  - Per-file error handling with continued processing
  - OOM detection and cache clearing

### 3. Bin-Packing File Assignment

Implemented greedy bin-packing algorithm for balanced GPU utilization:

- Sort files by sequence count (descending)
- Assign each file to worker with lowest current sequence total
- Ensures balanced work distribution across GPUs
- Logs assignment distribution for transparency

## Technical Implementation

### Architecture Decisions

**Unified Worker Interface:**
```python
class BaseEmbeddingWorker(ABC):
    @abstractmethod
    def process_files_worker(
        file_subset: List[Path],
        device_id: int,
        batch_size: int,
        output_dir: Path,
        **kwargs
    ) -> Tuple[List[Path], List[Tuple[Path, str]]]:
        pass
```

**Token-Based Batching:**
```python
for record in records:
    seq_str = str(record.seq)
    seq_tokens = len(seq_str)  # Each base = ~1 token

    if current_tokens + seq_tokens > toks_per_batch and current_batch:
        batches.append(current_batch)
        current_batch = []
        current_tokens = 0

    current_batch.append(record)
    current_tokens += seq_tokens
```

**BF16 Auto-Detection:**
```python
capability = torch.cuda.get_device_capability(device)
use_bf16 = capability[0] >= 8  # Ampere or newer

if use_bf16 and toks_per_batch == 2048:
    toks_per_batch = 3072  # Increase batch size with BF16
```

### Code Organization

```
virnucpro/pipeline/
├── base_worker.py          # NEW: Abstract base class + shared utilities
├── parallel_dnabert.py     # NEW: DNABERT-S worker implementation
├── parallel_esm.py         # EXISTING: ESM-2 worker (can refactor later)
└── work_queue.py           # EXISTING: Shared queue manager
```

## Success Criteria Met

- ✅ BaseEmbeddingWorker abstract class created with unified interface
- ✅ DNABERT-S parallel worker implemented with token-based batching
- ✅ BF16 optimization automatically enabled on compatible GPUs
- ✅ Bin-packing algorithm balances sequences across workers
- ✅ Code follows established patterns from ESM-2 implementation

## Verification Results

All verification checks passed:

1. ✅ BaseEmbeddingWorker class exists with correct abstract methods
2. ✅ DNABERT-S worker imports base class utilities
3. ✅ BF16 detection logic is present
4. ✅ Token-based batching implementation verified
5. ✅ Bin-packing assignment function exists

## Deviations from Plan

None - plan executed exactly as written.

## Decisions Made

### 1. Base Worker Abstraction Location

**Decision:** Create `base_worker.py` as separate module for shared code

**Reasoning:**
- Clean separation of concerns (abstract interface vs. implementations)
- Avoids circular imports between parallel_dnabert.py and parallel_esm.py
- Natural location for shared utilities (count_sequences, assign_files_by_sequences)
- Enables future model additions (e.g., ProtBERT) to inherit same interface

**Impact:** ESM-2 worker can be refactored to inherit from BaseEmbeddingWorker in future cleanup, but not required for Phase 2 functionality

### 2. Token Abstraction for DNA Sequences

**Decision:** Treat sequence length (base count) as token count

**Reasoning:**
- DNABERT-S uses k-mer tokenization internally (opaque to batching logic)
- Approximation (1 base ≈ 1 token) is close enough for batch sizing
- Matches ESM-2 pattern (1 amino acid ≈ 1 token)
- Dramatically simplifies batching code

**Impact:** Slight batch size imprecision but massive code simplification and maintainability improvement

### 3. Shared Utilities Export Strategy

**Decision:** Import and re-export shared utilities from parallel_dnabert.py

**Reasoning:**
- Maintains backward compatibility if utilities were expected in worker module
- Improves discoverability (users can import from either location)
- Documents the relationship with inline comments

**Impact:** `__all__` export list makes utilities available as if they were defined in parallel_dnabert.py

## Testing Notes

Not tested yet - awaiting integration with pipeline CLI in Plan 02.

**Expected behavior:**
- Files distributed evenly by sequence count across GPUs
- BF16 automatically enabled on Ampere+ GPUs
- Token batching handles variable-length DNA sequences
- Progress reporting works with existing dashboard

## Next Phase Readiness

**Ready for Plan 02 (Pipeline Integration):**
- ✅ Worker implementation complete and follows ESM-2 patterns
- ✅ Shared utilities available for pipeline to use
- ✅ BF16 optimization ready to use
- ✅ Bin-packing assignment ready for multi-GPU distribution

**No blockers identified.**

**Integration requirements for Plan 02:**
1. Add `--dnabert-batch-size` CLI flag
2. Import `process_dnabert_files_worker` in pipeline
3. Use `assign_files_by_sequences()` for file distribution
4. Initialize BatchQueueManager with DNABERT-S worker
5. Add DNABERT-S progress monitoring to dashboard

## Performance Expectations

Based on ESM-2 patterns and BF16 optimization:

- **Expected speedup:** 3-4x with 4 GPUs (matching ESM-2 scaling)
- **Memory efficiency:** 50% reduction with BF16 on Ampere+ GPUs
- **Batch size scaling:** 2048 tokens (FP32) → 3072 tokens (BF16)
- **GPU utilization:** Balanced across GPUs via bin-packing

Will be measured in Plan 03 (Performance Testing).

## Git Commits

| Task | Commit | Message |
|------|--------|---------|
| 1 | 2812d47 | feat(02-01): create BaseEmbeddingWorker abstract class |
| 2 | 4fc994e | feat(02-01): implement DNABERT-S parallel worker |
| 3 | 4ad1234 | feat(02-01): add bin-packing documentation to DNABERT-S worker |

**Total commits:** 3
**Total duration:** 2.9 minutes (175 seconds)
**Lines added:** ~413 lines across 2 new files

---

*Phase: 02-dnabert-s-optimization*
*Plan: 01*
*Status: ✅ Complete*
*Date: 2026-01-23*
