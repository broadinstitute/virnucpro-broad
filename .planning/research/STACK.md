# Technology Stack for GPU Optimization

**Research Date:** 2026-01-22
**Research Focus:** Multi-GPU optimization stack for ESM-2 3B and DNABERT-S embeddings
**Target:** Reduce 45+ hour bottleneck to <10 hours

---

## Executive Summary

This document outlines the recommended technology stack for optimizing VirNucPro's embedding pipeline across multiple GPUs. The primary bottleneck is ESM-2 3B (45 hours, single GPU), with DNABERT-S underutilized despite multi-GPU support. The recommended stack prioritizes **PyTorch DistributedDataParallel (DDP)** over alternatives due to native integration, strong community support, and proven performance with transformer models.

---

## Core Multi-GPU Parallelization

### 1. PyTorch DistributedDataParallel (DDP) — **RECOMMENDED**

**Version:** PyTorch >=2.8.0 (already in dependencies)
**Confidence:** HIGH

**What It Does:**
- Multi-process, multi-GPU data parallelism
- Each GPU runs a separate process with its own model replica
- Gradients synchronized across GPUs (for training) or inference batches distributed independently
- Native NCCL backend for GPU-to-GPU communication

**Why Use This:**
- **Already have PyTorch >=2.8.0** in current stack — zero new dependencies for basic DDP
- Official PyTorch recommendation over DataParallel for all multi-GPU scenarios
- Achieves 96%+ linear scaling efficiency on transformers (ESM-2 achieved 96.85% on 256 A100s)
- Proven with ESM-2 3B models (BioNeMo2 benchmarks show strong multi-GPU performance)
- Minimal code changes from existing single-GPU implementation
- Works seamlessly with spawn context already used in `parallel.py`

**Why NOT DataParallel:**
- DataParallel is single-process, multi-threaded (GIL contention)
- Significantly slower than DDP even on single node
- PyTorch officially recommends DDP over DP
- Poor GPU utilization compared to DDP

**Implementation Pattern:**
```python
# In worker process (similar to existing parallel.py pattern)
import torch.distributed as dist

def worker_fn(rank, world_size, device_id, file_subset):
    # Initialize process group
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)

    # Load model on specific GPU
    device = torch.device(f'cuda:{device_id}')
    model = load_model().to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device_id])

    # Process file subset
    for file in file_subset:
        # Extract features with local model
        features = extract_features(file, model, device)
```

**Rationale:**
- ESM-2 and DNABERT-S are inference-only in this pipeline (no backprop needed)
- DDP's overhead for gradient synchronization is avoided during inference
- Can use DDP pattern without actual gradient syncing for clean multi-GPU orchestration
- Alternatively, simpler manual process spawning (already working for DNABERT-S) is sufficient

**Sources:**
- [PyTorch DDP Tutorial](https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [ESM-2 PMC Article - 96.85% scaling efficiency](https://pmc.ncbi.nlm.nih.gov/articles/PMC12481099/)

---

### 2. DeepSpeed ZeRO-Inference — **NOT RECOMMENDED** (For This Use Case)

**Version:** DeepSpeed 0.18.5 (latest as of 2025)
**Confidence:** MEDIUM (well-tested, but overkill for this scenario)

**What It Does:**
- Offloads model weights to CPU/NVMe memory
- Enables inference of massive models on limited GPU resources
- Parallelizes layer fetches across multiple GPUs
- Layer prefetching to overlap compute and memory transfer

**Why NOT Use This:**
- **ESM-2 3B fits on single GPU** (3B params ≈ 12GB with fp16, well within 24-40GB GPU memory)
- Adds complexity of CPU/GPU memory transfers
- Designed for models that *don't fit* on GPU (>100B params)
- Overhead from memory transfers would slow down rather than speed up inference
- Current bottleneck is compute time, not memory capacity

**When To Reconsider:**
- If upgrading to ESM-2 15B model (requires >60GB GPU memory)
- If targeting inference on lower-memory GPUs (<16GB)

**Sources:**
- [DeepSpeed ZeRO-Inference](https://www.deepspeed.ai/2022/09/09/zero-inference.html)
- [DeepSpeed Releases](https://github.com/deepspeedai/DeepSpeed/releases)

---

### 3. PyTorch FSDP (Fully Sharded Data Parallel) — **NOT RECOMMENDED** (For Inference)

**Version:** PyTorch >=2.8.0 (FSDP2 available)
**Confidence:** MEDIUM

**What It Does:**
- Shards model parameters across GPUs (each GPU holds 1/N of model)
- Designed for training models that don't fit on single GPU
- FSDP2 improves memory management and DTensor sharding

**Why NOT Use This:**
- **Primarily designed for training**, not inference
- ESM-2 3B fits entirely on single GPU — no need for parameter sharding
- Adds communication overhead to gather parameters for each forward pass
- More complex than DDP for inference workloads
- Limited documentation for FSDP inference patterns

**Sources:**
- [PyTorch FSDP Tutorial](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html)

---

## Batch Processing & Queue Management

### 4. Ray (ray[default]) — **RECOMMENDED** for Advanced Queueing

**Version:** Ray 2.53.0
**Confidence:** HIGH

**What It Does:**
- Distributed task queue for Python with GPU-aware scheduling
- Dynamic resource allocation (e.g., 1 task per GPU)
- Fault tolerance and automatic retries
- Object store for sharing data across workers without duplication

**Why Use This:**
- **Best-in-class GPU batch processing** (500 GB/hour vs Dask's 300 in 2025 benchmarks)
- Automatically detects available GPUs and assigns tasks
- Superior fault tolerance (95% automatic recovery vs Dask's 80%)
- Native Python integration (unlike Spark)
- Minimal code changes from multiprocessing approach

**Implementation Pattern:**
```python
import ray

@ray.remote(num_gpus=1)
def process_file_batch(file_subset, device_id):
    # Load model on assigned GPU
    device = torch.device(f'cuda:{device_id}')
    model = load_model().to(device)

    # Process files
    return [extract_features(f, model, device) for f in file_subset]

# Launch tasks
ray.init()
gpu_count = torch.cuda.device_count()
futures = [process_file_batch.remote(files[i::gpu_count], i) for i in range(gpu_count)]
results = ray.get(futures)
```

**When To Use:**
- **Phase 1:** Start with native multiprocessing.Pool (already working for DNABERT-S)
- **Phase 2:** Upgrade to Ray if need dynamic GPU allocation or fault tolerance

**Alternative:** Native `multiprocessing.Pool` (already in use)
- Simpler, no new dependencies
- Works well for static GPU assignments
- Current `parallel.py` already implements this pattern
- **Recommendation:** Start here, migrate to Ray if limitations arise

**Sources:**
- [Ray Batch Processing Optimization](https://johal.in/batch-processing-optimization-with-ray-parallel-python-jobs-for-large-scale-data-engineering/)
- [Ray vs Dask Comparison](https://www.kdnuggets.com/ray-or-dask-a-practical-guide-for-data-scientists)

---

### 5. Dask — **NOT RECOMMENDED**

**Version:** Dask 2025.x
**Confidence:** MEDIUM

**What It Does:**
- Distributed computing framework for NumPy/Pandas/Scikit-learn
- Lazy execution with DAG-based scheduling
- GPU support via RAPIDS integration

**Why NOT Use This:**
- **Lower throughput** than Ray (300 vs 500 GB/hour in benchmarks)
- Better suited for Pandas/NumPy workflows, not PyTorch
- More complex setup for GPU-specific tasks
- Ray outperforms for ML workloads

**Sources:**
- [Dask GPU Tutorial](https://developer.nvidia.com/blog/dask-tutorial-beginners-guide-to-distributed-computing-with-gpus-in-python/)

---

## Inference Optimization Libraries

### 6. HuggingFace Accelerate — **RECOMMENDED**

**Version:** accelerate>=1.10.1 (already in dependencies)
**Confidence:** HIGH

**What It Does:**
- Unified API for multi-GPU training/inference across DDP, FSDP, DeepSpeed
- Automatic mixed precision (fp16, bf16, fp8)
- Device placement and distributed setup abstraction
- Works with transformers library (DNABERT-S)

**Why Use This:**
- **Already in dependencies** (accelerate>=1.10.1)
- Native integration with HuggingFace transformers (DNABERT-S uses this)
- Simplifies DDP initialization and device management
- Supports mixed precision for free speedup (2x with fp16)
- Regional compilation support for DeepSpeed

**Implementation Pattern:**
```python
from accelerate import Accelerator

accelerator = Accelerator(mixed_precision='fp16')
model = AutoModel.from_pretrained("zhihan1996/DNABERT-S")
model = accelerator.prepare(model)

# Automatic device placement and precision
outputs = model(inputs)  # Runs in fp16 if supported
```

**Key Features:**
- **Mixed Precision:** Use fp16 or bf16 for 2x speedup with minimal accuracy loss
- **Multi-GPU Inference:** Automatic model replication across GPUs
- **FSDP2 Support:** Full state dict saving (if needed later)

**Sources:**
- [HuggingFace Accelerate Docs](https://huggingface.co/docs/transformers/en/accelerate)
- [Accelerate GitHub](https://github.com/huggingface/accelerate)

---

### 7. FlashAttention-2 / FlashAttention-3 — **RECOMMENDED** (If Applicable)

**Version:** flash-attn>=2.6.x (FlashAttention-2), flash-attn-3 for H100
**Confidence:** HIGH

**What It Does:**
- IO-aware exact attention algorithm
- Reduces memory reads/writes for attention computation
- 2-4x faster attention with same accuracy
- Memory usage scales linearly (not quadratically) with sequence length

**Why Use This:**
- **9.4x speedup** for ESM-2 models with sequence packing (PMC research)
- 70% of theoretical max FLOPS on A100
- 10-20x memory savings for long sequences (2K-4K tokens)
- Native PyTorch 2.2+ integration via SDPA (Scaled Dot Product Attention)

**Applicability:**
- ESM-2: Likely benefits (check if fair-esm 2.0.0 uses standard attention)
- DNABERT-S: Definitely benefits (transformers 4.30.0 supports SDPA with FlashAttention-2)

**Implementation:**
```python
# For HuggingFace transformers (DNABERT-S)
model = AutoModel.from_pretrained(
    "zhihan1996/DNABERT-S",
    attn_implementation="flash_attention_2"  # Enable FlashAttention-2
)

# PyTorch 2.2+ automatically uses FlashAttention-2 via SDPA
```

**Requirements:**
- FlashAttention-2: NVIDIA A100/A6000/RTX 3090+, CUDA >= 11.8
- FlashAttention-3: H100/H800, CUDA >= 12.3 (recommended: CUDA 12.8)

**When To Skip:**
- If using older GPUs (pre-Ampere architecture)
- If fair-esm doesn't support attention backend customization

**Sources:**
- [FlashAttention GitHub](https://github.com/Dao-AILab/flash-attention)
- [FlashAttention-3 PyTorch Blog](https://pytorch.org/blog/flashattention-3/)
- [ESM-2 9.4x Speedup PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12481099/)

---

### 8. torch.compile — **CONDITIONAL RECOMMENDATION**

**Version:** PyTorch >=2.8.0 (already available)
**Confidence:** MEDIUM

**What It Does:**
- JIT compiles PyTorch models to optimized kernels
- Graph-level optimizations via TorchInductor
- Dynamic shape support for variable batch sizes

**Why Use This:**
- **Zero dependencies** (built into PyTorch 2.x)
- 1.5-2x speedup for transformers in many cases
- `dynamic=True` handles variable batch sizes without recompilation

**Why Be Cautious:**
- Compilation overhead on first run (can be slow)
- May have compatibility issues with fair-esm (older library)
- Stability varies across models (better in PyTorch 2.8 than earlier versions)

**Implementation:**
```python
# Compile model for faster inference
model = torch.compile(model, mode='reduce-overhead', dynamic=True)

# First batch: slow (compilation)
# Subsequent batches: faster (1.5-2x speedup)
```

**Recommendation:**
- **Try it after basic multi-GPU works**
- Benchmark with/without compilation
- Use `mode='reduce-overhead'` for small batches, `mode='max-autotune'` for large batches

**Sources:**
- [torch.compile Tutorial](https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html)
- [State of torch.compile August 2025](https://blog.ezyang.com/2025/08/state-of-torch-compile-august-2025/)

---

### 9. xFormers (Memory-Efficient Attention) — **ALTERNATIVE** to FlashAttention

**Version:** xformers>=0.0.34
**Confidence:** MEDIUM

**What It Does:**
- Memory-efficient attention operators (from Meta Research)
- Similar benefits to FlashAttention-2
- Broader GPU compatibility

**Why Consider:**
- If FlashAttention-2 incompatible with GPUs
- Easier installation on some systems

**Why Prefer FlashAttention-2:**
- FlashAttention-2 is faster (70% FLOPS vs 60% for xFormers)
- Better maintained for latest GPUs
- Native PyTorch 2.2+ integration

**Sources:**
- [xFormers Docs](https://facebookresearch.github.io/xformers/components/ops.html)

---

## GPU Utilization Monitoring

### 10. nvitop — **RECOMMENDED**

**Version:** nvitop>=1.3.x
**Confidence:** HIGH

**What It Does:**
- Interactive GPU process viewer (combines nvidia-smi + gpustat + nvtop)
- Real-time monitoring with colorful TUI
- Process filtering, metrics tracking, tree view
- Python API for programmatic monitoring

**Why Use This:**
- **Best-in-class GPU monitoring** (expert consensus 2025)
- Pure Python (easy pip install)
- Portable (Linux + Windows)
- Useful for verifying >80% GPU utilization target
- Integrates with Python scripts for logging

**Installation:**
```bash
pixi add nvitop
# or
pip install nvitop
```

**Usage:**
```bash
# Interactive monitoring
nvitop

# Python API for logging
from nvitop import Device
devices = Device.all()
for device in devices:
    print(f"GPU {device.index}: {device.gpu_utilization()}% util, {device.memory_used()}/{device.memory_total()} MB")
```

**Alternatives:**
- **gpustat** (simpler, less features, uses pynvml >= 12.535.108)
- **nvidia-smi** (built-in, basic, scriptable)

**Sources:**
- [nvitop GitHub](https://github.com/XuehaiPan/nvitop)
- [GPU Monitoring Tools Comparison](https://lambda.ai/blog/keeping-an-eye-on-your-gpus-2)

---

### 11. gpustat — **ALTERNATIVE**

**Version:** gpustat>=1.2
**Confidence:** HIGH

**What It Does:**
- Simple command-line GPU monitor
- Python-based (pynvml wrapper)
- Real-time status updates

**Why Consider:**
- Simpler than nvitop if only need basic monitoring
- Lightweight

**Why Prefer nvitop:**
- More features (process management, filtering, tree view)
- Better UI/UX

**Sources:**
- [gpustat PyPI](https://pypi.org/project/gpustat/)

---

## Model Optimization Techniques

### 12. Quantization (8-bit / 4-bit) — **NOT RECOMMENDED** (For Now)

**Confidence:** LOW (for this use case)

**What It Does:**
- Reduces model precision from fp32/fp16 to int8 or int4
- 2-3x memory reduction
- Marginal speedup on billion-parameter models

**Why NOT Use This:**
- **Memory is not the bottleneck** (ESM-2 3B fits on single GPU)
- Accuracy loss risk (not validated for viral prediction task)
- Current bottleneck is compute time, not memory
- Quantization primarily helps memory-constrained scenarios

**When To Reconsider:**
- If targeting smaller GPUs (<16GB)
- If upgrading to ESM-2 15B

**Sources:**
- [ESM-2 PMC Article - Quantization Benefits](https://pmc.ncbi.nlm.nih.gov/articles/PMC12481099/)

---

## Recommended Stack Summary

| Component | Library/Version | Priority | Reason |
|-----------|----------------|----------|--------|
| **Multi-GPU Parallelization** | PyTorch DDP (>=2.8.0) | **HIGH** | Already in stack, proven with ESM-2, minimal code changes |
| **Batch Queue Management** | Native multiprocessing.Pool | **HIGH** | Already working (parallel.py), simple |
| **Advanced Queueing** | Ray >=2.53.0 | **MEDIUM** | Upgrade path if need fault tolerance / dynamic allocation |
| **Inference Optimization** | HuggingFace Accelerate >=1.10.1 | **HIGH** | Already in stack, mixed precision, DDP abstraction |
| **Attention Speedup** | FlashAttention-2 (flash-attn>=2.6) | **HIGH** | 9.4x speedup for ESM-2, 10-20x memory savings |
| **Model Compilation** | torch.compile (PyTorch >=2.8.0) | **MEDIUM** | Try after basic multi-GPU works, 1.5-2x speedup |
| **GPU Monitoring** | nvitop >=1.3 | **HIGH** | Verify 80%+ utilization, process tracking |

---

## What NOT to Use (And Why)

| Library | Why Avoid |
|---------|-----------|
| **DataParallel** | Deprecated, slow (GIL contention), use DDP instead |
| **DeepSpeed ZeRO-Inference** | Overkill (ESM-2 3B fits on single GPU), adds CPU/GPU transfer overhead |
| **FSDP** | Designed for training + models that don't fit on single GPU (not applicable here) |
| **Dask** | Lower throughput than Ray (300 vs 500 GB/hr), better for Pandas/NumPy not PyTorch |
| **Quantization (8-bit/4-bit)** | Memory not the bottleneck, accuracy risk, minimal speedup for compute-bound tasks |
| **xFormers** | FlashAttention-2 is faster and better integrated with PyTorch 2.2+ |

---

## Dependency Changes Required

### New Dependencies (Add to pixi.toml)

```toml
[pypi-dependencies]
# GPU monitoring
nvitop = ">=1.3.0, <2"

# Optional: FlashAttention-2 (if compatible with GPUs)
# flash-attn = ">=2.6.0, <3"

# Optional: Ray for advanced queueing (Phase 2)
# ray = { version = ">=2.53.0, <3", extras = ["default"] }
```

### Existing Dependencies (No Changes Needed)

- `torch>=2.8.0` — Provides DDP, torch.compile, mixed precision
- `accelerate>=1.10.1` — Provides multi-GPU abstraction, mixed precision
- `transformers==4.30.0` — DNABERT-S support, FlashAttention-2 compatible

---

## Implementation Phasing Recommendations

### Phase 1: Minimal Changes (Highest ROI)
1. **Enable mixed precision** (fp16/bf16) with Accelerate — 2x speedup, zero code change
2. **Parallelize ESM-2** across GPUs using existing multiprocessing.Pool pattern from `parallel.py`
3. **Increase DNABERT-S batch size** per GPU (current batch_size=256 may be conservative)
4. **Add nvitop monitoring** to measure GPU utilization

**Expected Outcome:** 3-4x speedup (45 hours → 11-15 hours)

### Phase 2: Attention Optimization
1. **Integrate FlashAttention-2** for DNABERT-S (transformers library supports this)
2. **Test FlashAttention** with ESM-2 (may require fair-esm modifications or upgrade)
3. **Benchmark torch.compile** on both models

**Expected Outcome:** Additional 1.5-2x speedup (11-15 hours → 6-10 hours)

### Phase 3: Advanced (If Needed)
1. **Migrate to Ray** for dynamic GPU allocation and fault tolerance
2. **Profile and optimize I/O** (address CONCERNS.md: "File I/O in tight loops")
3. **Batch across multiple samples** simultaneously if applicable

**Expected Outcome:** Fine-tuning to <10 hours consistently

---

## Validation Checklist

- [ ] All versions verified against official documentation (not training data)
- [x] PyTorch DDP: [pytorch.org/tutorials](https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [x] ESM-2 scaling: [PMC Article](https://pmc.ncbi.nlm.nih.gov/articles/PMC12481099/)
- [x] FlashAttention-2: [pytorch.org/blog](https://pytorch.org/blog/flashattention-3/)
- [x] Accelerate: [huggingface.co/docs](https://huggingface.co/docs/transformers/en/accelerate)
- [x] Ray benchmarks: [KDnuggets comparison](https://www.kdnuggets.com/ray-or-dask-a-practical-guide-for-data-scientists)
- [x] nvitop: [GitHub repo](https://github.com/XuehaiPan/nvitop)
- [x] DeepSpeed version: [GitHub releases](https://github.com/deepspeedai/DeepSpeed/releases)
- [x] torch.compile: [PyTorch docs](https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html)

---

## Confidence Levels

| Recommendation | Confidence | Reason |
|----------------|-----------|--------|
| PyTorch DDP | **HIGH** | Industry standard, proven with ESM-2, already have PyTorch >=2.8.0 |
| HuggingFace Accelerate | **HIGH** | Already in dependencies, widely adopted, stable |
| FlashAttention-2 | **HIGH** | Massive speedup (9.4x) validated for ESM-2 in recent research |
| nvitop | **HIGH** | Expert consensus for best monitoring tool 2025 |
| Ray | **MEDIUM** | Excellent for batch processing, but multiprocessing.Pool may suffice |
| torch.compile | **MEDIUM** | Can be unstable, benchmark required, but potential 1.5-2x speedup |
| Skip DeepSpeed | **HIGH** | Not needed for models that fit on single GPU |
| Skip FSDP | **HIGH** | Designed for training, not inference |
| Skip Quantization | **MEDIUM** | Memory not bottleneck, but may help if targeting low-end GPUs |

---

## Sources

- [PyTorch DDP Tutorial](https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [ESM-2 Multi-GPU Performance PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12481099/)
- [NVIDIA BioNeMo ESM-2 Docs](https://docs.nvidia.com/bionemo-framework/2.1/models/esm2/)
- [DeepSpeed ZeRO-Inference](https://www.deepspeed.ai/2022/09/09/zero-inference.html)
- [DeepSpeed Releases](https://github.com/deepspeedai/DeepSpeed/releases)
- [PyTorch FSDP Tutorial](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html)
- [Ray vs Dask Comparison](https://www.kdnuggets.com/ray-or-dask-a-practical-guide-for-data-scientists)
- [Ray Batch Processing](https://johal.in/batch-processing-optimization-with-ray-parallel-python-jobs-for-large-scale-data-engineering/)
- [HuggingFace Accelerate Docs](https://huggingface.co/docs/transformers/en/accelerate)
- [Accelerate GitHub](https://github.com/huggingface/accelerate)
- [FlashAttention GitHub](https://github.com/Dao-AILab/flash-attention)
- [FlashAttention-3 PyTorch Blog](https://pytorch.org/blog/flashattention-3/)
- [DNABERT-2 GitHub](https://github.com/MAGICS-LAB/DNABERT_2)
- [DNABERT-2 ArXiv](https://arxiv.org/html/2306.15006v2)
- [torch.compile Tutorial](https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html)
- [State of torch.compile August 2025](https://blog.ezyang.com/2025/08/state-of-torch-compile-august-2025/)
- [xFormers Docs](https://facebookresearch.github.io/xformers/components/ops.html)
- [nvitop GitHub](https://github.com/XuehaiPan/nvitop)
- [nvitop PyPI](https://pypi.org/project/nvitop/)
- [gpustat PyPI](https://pypi.org/project/gpustat/)
- [GPU Monitoring Tools Comparison](https://lambda.ai/blog/keeping-an-eye-on-your-gpus-2)
- [Transformers v5 Release](https://howaiworks.ai/blog/transformers-v5-release-announcement-2025)
- [HuggingFace Multi-GPU Inference](https://huggingface.co/docs/transformers/v4.48.0/perf_infer_gpu_multi)
