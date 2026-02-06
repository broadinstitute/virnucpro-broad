# VirNucPro - Claude Code Reference

## What This Is

A viral nucleotide prediction pipeline that processes DNA sequences through six-frame translation and deep learning models (DNABERT-S and ESM-2). The GPU optimization work (this repo) reduces embedding time from 45+ hours to under 10 hours through async DataLoader architecture, sequence packing, and multi-GPU coordination.

## Quick Start

```bash
# Environment
pixi install          # Uses pixi (Python 3.9, conda-forge)
pip install -r requirements.txt

# Run tests
pytest tests/ -v                    # All tests
pytest tests/ -v -m "not slow"      # Skip slow tests
pytest tests/ -v -m "gpu"           # GPU-only tests

# Run pipeline
python -m virnucpro predict input.fasta --parallel
```

## Project Structure

```
virnucpro/                    # Main package
  cli/                        # Click-based CLI (main.py, predict.py, profile.py, benchmark.py)
  core/                       # Config (YAML dot-notation), device validation, checkpointing
  cuda/                       # Stream manager, attention utils, memory manager
  data/                       # Async DataLoader components (Phase 5+)
    collators.py              #   VarlenCollator with buffer-based packing
    dataloader_utils.py       #   create_async_dataloader() factory
    packing.py                #   GreedyPacker FFD algorithm (~92-94% efficiency)
    sequence_dataset.py       #   IndexBasedDataset for byte-offset reading
    shard_index.py            #   Multi-GPU stride-based index distribution
  models/                     # ESM-2 FlashAttention integration, DNABERT flash, packed attention
  pipeline/                   # Inference orchestration
    async_inference.py        #   AsyncInferenceRunner (single-GPU, Phase 5)
    multi_gpu_inference.py    #   run_multi_gpu_inference() entry point (Phase 7)
    gpu_coordinator.py        #   GPUProcessCoordinator lifecycle management
    gpu_worker.py             #   Per-GPU worker function (module-level for pickle)
    shard_aggregator.py       #   HDF5 shard merging with validation
    worker_logging.py         #   Per-worker log file setup
    features.py               #   Single-GPU feature extraction
    parallel_dnabert.py       #   DNABERT-S multi-GPU
    parallel_esm.py           #   ESM-2 multi-GPU
    prediction.py             #   Full pipeline orchestration
  utils/                      # Sequence processing, validation, GPU monitor, progress bars
tests/
  unit/                       # Component tests (packing, collators, workers, etc.)
  integration/                # Multi-component tests (packed equivalence, multi-GPU)
  benchmarks/                 # Performance and scaling tests
.planning/                    # Phase-based project management
  PROJECT.md                  #   Scope, requirements, key decisions
  STATE.md                    #   Current phase, accumulated context, blockers
  phases/                     #   Per-phase plans and summaries (01-08)
```

## Architecture (v2.0)

Single-process-per-GPU with async DataLoader (replaces v1.0 multi-worker-per-GPU):

```
FASTA → DataLoader Workers (CPU I/O) → VarlenCollator (tokenize + pack)
  → Pinned Memory → CUDA Streams (overlap H2D/compute/D2H) → HDF5 Shards
  → Shard Aggregator → Final Output
```

Multi-GPU: stride distribution `[rank::world_size]` on length-sorted index. Each GPU runs its own `AsyncInferenceRunner` in a spawned process.

## Code Conventions

- **Logging**: `logger = logging.getLogger('virnucpro.module_name')` at module level
- **Type hints**: All function signatures typed. Use `Optional[T]`, `Callable`, `List`, `Dict`
- **Docstrings**: Module-level docstrings with architecture notes and integration context
- **Imports**: stdlib, then third-party, then local (relative within package)
- **Config**: YAML via `Config.get("dotted.key", default=value)`
- **Errors**: Fail fast with descriptive messages including context for debugging
- **Workers**: Must be module-level functions (pickle compatibility with spawn)

## Testing

```bash
pytest tests/ -v                          # All
pytest tests/unit/ -v                     # Unit only
pytest tests/ -v -m "not slow"            # Skip slow
pytest tests/ -v -m "gpu"                 # GPU required
pytest tests/ -v -k "test_packing"        # Pattern match
```

**Markers**: `@pytest.mark.gpu` (skipped when no CUDA), `@pytest.mark.slow`

**Fixtures** (in `tests/conftest.py`):
- `temp_dir` - temporary directory, cleaned up after test
- `temp_fasta(num_sequences, seq_length)` - generates test FASTA files
- `mock_gpu_devices(num_gpus)` - mocks `torch.cuda` for CPU-only testing

**Test isolation**: `tests/unit/conftest.py` has `detect_torch_pollution` autouse fixture that fails tests if `torch` gets replaced by a Mock (catches `sys.modules` replacement bugs).

**Test naming**: `test_{what}_{condition}_{expected_outcome}`

**Mocking**: Use `@patch` decorators, never `sys.modules` replacement (causes pollution).

## Key Dependencies

- `torch` >=2.8.0 (CUDA, DataLoader, streams)
- `transformers` ==4.30.0 (DNABERT-S: `zhihan1996/DNABERT-S`)
- `fair-esm` ==2.0.0 (ESM-2 3B model)
- `biopython` (FASTA parsing)
- `click` >=8.0.0 (CLI)
- `h5py` (shard storage - imported at use site)
- `rich` >=13.0.0, `tqdm` (output formatting)
- **Optional**: `flash-attn` >=2.6.0 (FlashAttention-2 for packed attention)
- **Environment**: pixi with conda-forge (Python 3.9)

## Common Gotchas

1. **CUDA in workers**: Must use `multiprocessing.set_start_method('spawn')` - fork inherits CUDA state
2. **Stream sync**: Call `torch.cuda.current_stream().synchronize()` before extracting embeddings from async compute
3. **DataLoader batch_size**: Set `batch_size=None` when using VarlenCollator (collator controls batching via token budget)
4. **Packed format shape**: `[total_tokens, hidden_dim]` not `[batch, seq, hidden]` - no batch dimension
5. **cu_seqlens**: Cumulative boundaries `[0, len1, len1+len2, ...]` with N+1 elements for N sequences
6. **Position IDs**: Must reset to 0 at each cu_seqlens boundary for packed format
7. **FlashAttention dtype**: Requires FP16 or BF16 inputs - validate before calling kernel
8. **Packing disable**: Set `VIRNUCPRO_DISABLE_PACKING=true` env var for emergency rollback

## Git Conventions

- Branch naming: `phase-N-description`
- Commit format: `type(scope): description` (feat, fix, test, docs, refactor, perf)
- Scope examples: `async_inference`, `gpu_worker`, `collators`, `packing`, `checkpoint`

## Project State

Current: Phase 7 complete, Phase 8 (FP16 Precision Validation) next. See `.planning/STATE.md` for accumulated decisions and `.planning/PROJECT.md` for requirements.

Key decisions documented in `.planning/STATE.md` under "Accumulated Context > Decisions".

## QMD — Local Search Engine

QMD is available as a CLI tool and MCP server for searching indexed files locally.

### How GSD Uses QMD

GSD automatically manages a qmd collection per registered project, named `gsd-{projectname}`, indexed against `.planning/`. GSD agents (researcher, planner, executor) use this internally for semantic search across planning docs, falling back to grep when qmd is unavailable. You do not need to create, update, or embed these collections — GSD handles it.

### Direct QMD Usage (Outside GSD Workflows)

When working outside of GSD commands (ad-hoc sessions, debugging, exploration), you can use qmd directly to search project planning history and any other indexed collections.

**Search modes:**
- `qmd search "query"` — BM25 keyword search. Fast, default choice.
- `qmd vsearch "query"` — Semantic vector search. Slower, understands meaning.
- `qmd query "query"` — Hybrid with LLM reranking. Slowest, best for prose. Avoid for code.

**Common patterns:**
```
# Search this project's planning docs
qmd search "auth decisions" -c gsd-{projectname}

# Search across all indexed collections
qmd search "error handling patterns"

# Retrieve a specific doc from search results
qmd get "#abc123"

# Get structured output
qmd search "validation" --json -n 10 -c gsd-{projectname}
```

**When to use qmd directly:**
- Searching across planning history for past decisions or rationale
- Finding which phase handled a specific feature
- Orienting at the start of an ad-hoc session
- Searching non-GSD collections (external docs, notes, code)

**When NOT to use qmd:**
- During GSD workflows — the agents already search internally
- When you know the exact file — just read it
- For standard GSD files (ROADMAP.md, STATE.md, REQUIREMENTS.md) — read directly

### Additional Collections

The user may have non-GSD collections indexed (notes, external docs, code). Run `qmd status` to see all available collections and their document counts. Use `-c collectionname` to scope searches.

### Maintenance

Never run `qmd collection add`, `qmd embed`, or `qmd update` automatically. These are manual operations managed by the user or by GSD's registration system.