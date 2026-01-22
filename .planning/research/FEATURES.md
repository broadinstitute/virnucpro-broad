# Multi-GPU Inference Optimization Features

## Research Context

**Question**: What optimization techniques exist for multi-GPU inference of large language models like ESM-2 and DNABERT-S?

**Background**: VirNucPro processes thousands of sequences through ESM-2 (3B params, protein) and DNABERT-S (DNA) embeddings. Current issues: ESM-2 single GPU sequential batches take 45 hours; DNABERT-S multi-GPU but one file per GPU (underutilized). Target: multi-GPU parallelization, batch queuing, better memory utilization, linear scaling.

**Date**: 2026-01-22

---

## Table Stakes (Must-Have Optimizations)

Features essential for production multi-GPU inference. If we don't have these, we're not competitive.

### TS-01: Data Parallelism for Multi-GPU Distribution

**What it is**: Replicate the model across multiple GPUs and distribute different input batches to each GPU. PyTorch provides `DataParallel` (simpler, single-process) and `DistributedDataParallel` (faster, multi-process).

**Why table stakes**: The foundation for utilizing multiple GPUs. Without this, additional GPUs sit idle.

**Complexity**: LOW
- `DataParallel`: Wrap model in `nn.DataParallel(model)` (3-5 lines)
- `DistributedDataParallel`: Requires process group initialization and rank management (20-30 lines)

**VirNucPro application**:
- DNABERT-S: Already implemented via multiprocessing (file-per-GPU)
- ESM-2: Not implemented (currently single GPU)
- **Recommendation**: Use DDP for both models (better performance than current file-per-GPU approach)

**Trade-offs**:
- DataParallel: Single-process threading bottleneck limits scaling beyond 2-3 GPUs
- DDP: Requires multi-process coordination but scales linearly

**Dependencies**: None

**References**:
- [PyTorch DDP Tutorial](https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [DataParallel vs DistributedDataParallel](https://medium.com/@kuipasta1121/dataparallel-vs-distributeddataparallel-in-pytorch-whats-the-difference-0af10bb43bc7)

---

### TS-02: Optimized Batch Size for Memory Utilization

**What it is**: Maximize batch size to fully utilize GPU memory without OOM errors. Larger batches = bigger kernels = better GPU utilization.

**Why table stakes**: GPUs are most efficient when executing large batches. Small batches waste compute capacity.

**Complexity**: LOW-MEDIUM
- Binary search for max batch size (manual testing: LOW)
- Automated profiling with `torch.cuda.memory_stats()` (programmatic: MEDIUM)

**VirNucPro application**:
- Current: DNABERT-S batch_size=256, ESM-2 toks_per_batch=2048
- **Recommendation**: Profile each GPU type to find max batch size (likely 2-4x current values for A100/H100)

**Trade-offs**:
- Larger batches increase throughput but reduce scheduling flexibility
- Batch size limited by GPU memory (ESM-2 3B model is memory-intensive)

**Dependencies**: GPU profiling tools (nvidia-smi, PyTorch profiler)

**References**:
- [Finding Optimal Batch Size](https://www.digitalocean.com/community/tutorials/find-optimal-batch-size)
- [PyTorch Performance Optimization](https://medium.com/@ishita.verma178/pytorch-gpu-optimization-step-by-step-guide-9dead5164ca2)

---

### TS-03: DataLoader Prefetching with Multiple Workers

**What it is**: Use PyTorch `DataLoader` with `num_workers > 0` and `prefetch_factor` to load next batch while GPU processes current batch. Set `pin_memory=True` for faster CPU→GPU transfer.

**Why table stakes**: CPU data loading shouldn't block GPU computation. Without prefetching, GPU idles waiting for data.

**Complexity**: LOW
- Add `num_workers=4-8`, `prefetch_factor=2`, `pin_memory=True` to DataLoader (1 line change)

**VirNucPro application**:
- Current: Loads FASTA files, tokenizes, processes batch-by-batch (sequential I/O)
- **Recommendation**: Wrap sequence loading in DataLoader with 4-8 workers

**Trade-offs**:
- More workers = more CPU memory (each worker holds `prefetch_factor` batches)
- Too many workers can cause overhead; sweet spot is typically 4-8

**Dependencies**: None

**References**:
- [8 PyTorch DataLoader Tactics](https://medium.com/@Modexa/8-pytorch-dataloader-tactics-to-max-out-your-gpu-22270f6f3fa8)
- [Data Prefetching in Deep Learning](https://www.jpatrickpark.com/post/prefetcher/)

---

### TS-04: Mixed Precision Inference (FP16/BF16)

**What it is**: Run inference in 16-bit floating point instead of 32-bit. FP16 (5-bit exponent, 10-bit mantissa) for high precision; BF16 (8-bit exponent, 7-bit mantissa) for wider dynamic range.

**Why table stakes**: 2x memory reduction, 2x speedup on modern GPUs with tensor cores. No accuracy loss for most inference tasks.

**Complexity**: LOW
- PyTorch: `with torch.autocast(device_type='cuda', dtype=torch.bfloat16):`
- Hugging Face: `model.half()` or `torch_dtype=torch.float16` in `from_pretrained()`

**VirNucPro application**:
- ESM-2: Currently FP32 (implicit)
- DNABERT-S: Currently FP32 (implicit via transformers)
- **Recommendation**: Use BF16 for both (better range, less overflow risk)

**Trade-offs**:
- FP16: Risk of overflow/underflow; needs loss scaling for training (not inference)
- BF16: Slightly lower precision but safer; requires A100/H100 GPUs

**Dependencies**: GPU with tensor cores (V100+), PyTorch 1.10+

**References**:
- [Mixed Precision Training Guide](https://markaicode.com/transformers-mixed-precision-training-fp16-bf16/)
- [Defeating Training-Inference Mismatch via FP16](https://arxiv.org/html/2510.26788v1)

---

### TS-05: CUDA Streams for Overlapping Computation

**What it is**: Use multiple CUDA streams to overlap memory transfers (H2D, D2H) with kernel execution. Default stream is synchronous; multiple streams enable async operations.

**Why table stakes**: Hide memory transfer latency behind computation. Can improve throughput 20-40% for I/O-heavy workloads.

**Complexity**: MEDIUM
- Create stream: `stream = torch.cuda.Stream()`
- Execute in stream: `with torch.cuda.stream(stream):`
- Synchronize: `stream.synchronize()`
- Requires careful orchestration (30-50 lines)

**VirNucPro application**:
- Current: Sequential load→tokenize→forward→save
- **Recommendation**: Stream 1 for next batch load/tokenize, Stream 2 for current batch forward pass

**Trade-offs**:
- Complexity increases (need to manage dependencies)
- Diminishing returns beyond 2-3 streams (GPU has 32 hardware queues max)

**Dependencies**: CUDA 11.0+

**References**:
- [CUDA Streams Introduction](https://wentao.site/cuda_streams/)
- [Optimizing GPU Performance with CUDA Streams](https://medium.com/@kailashcvm/optimizing-gpu-performance-with-cuda-streams-and-batch-sizes-a1725debf86c)

---

## Differentiators (Advanced Optimizations)

Features that provide extra speedup beyond baseline multi-GPU. These are "nice to have" but not critical for initial version.

### DF-01: Continuous Batching (vLLM-style)

**What it is**: Dynamically schedule new sequences into GPU whenever a running sequence completes, rather than waiting for entire batch to finish. Iteration-level scheduling that fills gaps.

**Why differentiator**: Maximizes GPU utilization for variable-length sequences. vLLM achieves 23x throughput vs naive batching.

**Complexity**: HIGH
- Requires custom scheduler to track sequence completion
- Dynamic memory allocation for varying batch sizes
- Integration with existing pipeline (100-200 lines)

**VirNucPro application**:
- Current: Fixed batch sizes, wait for entire batch
- **Potential benefit**: DNABERT-S/ESM-2 have variable sequence lengths (chunked to 500bp but translated ORFs vary)
- **Recommendation**: Phase 2 optimization (after basic multi-GPU works)

**Trade-offs**:
- Significant engineering complexity
- Requires rethinking batch processing logic
- Best for high-throughput serving, less critical for offline batch inference

**Dependencies**: Custom scheduler or vLLM integration

**References**:
- [vLLM Continuous Batching](https://www.anyscale.com/blog/continuous-batching-llm-inference)
- [Inside vLLM](https://www.aleksagordic.com/blog/vllm)

---

### DF-02: FlashAttention-2 for Memory-Efficient Attention

**What it is**: IO-aware attention algorithm that uses tiling to reduce memory reads/writes between HBM and SRAM. 2x speedup over standard attention, enables 2x longer sequences.

**Why differentiator**: Reduces memory bottleneck for long sequences. Attention is O(n²) in sequence length.

**Complexity**: MEDIUM
- Install: `pip install flash-attn`
- Replace attention: Model-dependent (DNABERT-S/ESM-2 may require modifications)
- For BERT-style models: Monkey-patch attention layers (20-40 lines)

**VirNucPro application**:
- DNABERT-S: 500bp chunks = ~500 tokens (moderate length)
- ESM-2: Truncated to 1024 residues (memory savings possible)
- **Potential benefit**: Could enable larger batch sizes or longer sequences
- **Recommendation**: Test on ESM-2 first (longer sequences)

**Trade-offs**:
- Requires CUDA 11.8+, A100/H100 GPUs for full benefit
- Model architecture compatibility (not all transformers support)
- FlashAttention-3 optimized for H100 (cutting edge)

**Dependencies**: flash-attn library, CUDA 11.8+, Ampere/Hopper GPUs

**References**:
- [FlashAttention Paper](https://arxiv.org/abs/2205.14135)
- [FlashAttention-2 Improvements](https://openreview.net/forum?id=mZn2Xyh9Ec)

---

### DF-03: Model Quantization (INT8/4-bit)

**What it is**: Reduce model weights from FP32/FP16 to INT8 (8-bit integers) or 4-bit. Techniques: GPTQ, AWQ, bitsandbytes.

**Why differentiator**: 4x (INT8) or 8x (4-bit) memory reduction, 2-4x inference speedup. Enables larger batch sizes.

**Complexity**: MEDIUM
- bitsandbytes: `load_in_8bit=True` in `from_pretrained()` (1 line)
- GPTQ/AWQ: Requires calibration dataset and quantization script (50-100 lines)

**VirNucPro application**:
- ESM-2 3B: FP32 = 12GB, INT8 = 3GB, 4-bit = 1.5GB
- **Potential benefit**: Fit more batches in memory, use smaller GPUs
- **Concern**: Accuracy impact on protein/DNA embeddings (needs validation)
- **Recommendation**: Benchmark INT8 first, measure embedding quality

**Trade-offs**:
- Accuracy loss: 1-5% for INT8 (minimal), 5-10% for 4-bit (acceptable for most tasks)
- Calibration overhead: GPTQ/AWQ require representative dataset
- Not all operations are quantized (e.g., layer norms stay FP16)

**Dependencies**: bitsandbytes, transformers 4.30+, or GPTQ/AWQ libraries

**References**:
- [LLM Quantization Guide](https://bentoml.com/llm/getting-started/llm-quantization)
- [8-bit Matrix Multiplication](https://huggingface.co/blog/hf-bitsandbytes-integration)

---

### DF-04: Tensor Parallelism for Model Sharding

**What it is**: Split individual model layers across GPUs (shard attention/MLP within a layer). Each GPU computes partial results, then all-reduce.

**Why differentiator**: Reduces per-GPU memory for very large models. Enables models that don't fit on single GPU.

**Complexity**: HIGH
- Requires model surgery to shard layers (DeepSpeed/Megatron-LM libraries)
- All-reduce communication overhead on every forward pass
- Integration complexity (100-300 lines)

**VirNucPro application**:
- ESM-2 3B: Fits on single A100 (40GB) in FP16
- **Current relevance**: LOW (model fits on single GPU)
- **Future relevance**: If upgrading to ESM-2 15B or larger models
- **Recommendation**: Skip for now; data parallelism is sufficient

**Trade-offs**:
- High communication overhead (reduces efficiency vs data parallelism)
- Best for models that don't fit on single GPU
- Requires fast interconnect (NVLink/Infiniband)

**Dependencies**: DeepSpeed, Megatron-LM, or manual implementation

**References**:
- [Tensor Parallelism Overview](https://www.infracloud.io/blogs/inference-parallelism/)
- [Scaling LLM Inference](https://engineering.fb.com/2025/10/17/ai-research/scaling-llm-inference-innovations-tensor-parallelism-context-parallelism-expert-parallelism/)

---

### DF-05: DeepSpeed ZeRO-Inference for Large Batch Sizes

**What it is**: Offload model parameters to CPU/NVMe, stream layers to GPU on-demand. Enables batch sizes 10-100x larger by freeing GPU memory.

**Why differentiator**: Maximize throughput for long sequences or large batches. Best when computation time dominates weight loading time.

**Complexity**: MEDIUM-HIGH
- Install DeepSpeed: `pip install deepspeed`
- Configure ZeRO stage 3 + inference: JSON config file
- Modify training script: `deepspeed.initialize()` (20-50 lines)

**VirNucPro application**:
- ESM-2: Currently memory-bound (3B params = 6GB in FP16)
- **Potential benefit**: Increase batch size from 2048 toks to 20,000+ toks
- **Trade-off**: Slower per-batch (offload overhead), but higher overall throughput
- **Recommendation**: Test if batch size is primary bottleneck

**Trade-offs**:
- Offload overhead: 10-30% slower per iteration
- Net win if throughput gain (10x batch size) exceeds overhead
- Requires NVMe for best performance (PCIe 4.0+ SSD)

**Dependencies**: DeepSpeed, NVMe storage (optional but recommended)

**References**:
- [ZeRO-Inference](https://www.deepspeed.ai/2022/09/09/zero-inference.html)
- [DeepSpeed Inference Tutorial](https://www.deepspeed.ai/tutorials/inference-tutorial/)

---

### DF-06: Pipeline Parallelism for Layer Distribution

**What it is**: Split model layers across GPUs (e.g., GPU0: layers 1-12, GPU1: layers 13-24). Micro-batches pipelined through stages.

**Why differentiator**: Enables very large models. Lower communication than tensor parallelism.

**Complexity**: HIGH
- Manual: Assign layer ranges to GPUs, manage activations (50-100 lines)
- GPipe/PipeDream: Automated pipelining libraries (integration complexity)

**VirNucPro application**:
- ESM-2 3B: 36 layers, fits on single GPU
- **Current relevance**: LOW (model fits on single GPU)
- **Potential use**: If scaling to ESM-2 15B (54 layers)
- **Recommendation**: Skip; data parallelism preferred for inference

**Trade-offs**:
- Bubble overhead: GPUs idle during pipeline fill/drain (5-15% efficiency loss)
- Best for training (gradient accumulation); less efficient for inference
- Requires micro-batching to keep pipeline full

**Dependencies**: PyTorch Pipeline, GPipe, or manual implementation

**References**:
- [Parallelism Techniques](https://www.genesiscloud.com/blog/top-parallelism-techniques-llm-training)
- [NVIDIA LLM Inference Guide](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/)

---

## Anti-Features (Avoid These)

Optimizations that are either too complex for the benefit or inappropriate for this use case.

### AF-01: Gradient Accumulation for Inference

**What it is**: Accumulate gradients over multiple batches to simulate larger batch size (training technique).

**Why anti-feature**: Gradient accumulation is for TRAINING, not inference. Inference doesn't compute gradients.

**Recommendation**: Do not implement. Use actual larger batch sizes or DeepSpeed ZeRO-Inference instead.

---

### AF-02: Multi-Node Distributed Inference

**What it is**: Distribute inference across multiple machines connected by network (e.g., 4 nodes × 8 GPUs = 32 GPUs).

**Why anti-feature**:
- Out of scope (PROJECT.md specifies "single-machine multi-GPU only")
- High complexity (network communication, job scheduling)
- Diminishing returns: Single node with 8x A100/H100 sufficient for VirNucPro throughput

**Recommendation**: Explicitly avoid. Focus on optimizing single-node 4-8 GPU setup.

---

### AF-03: Custom CUDA Kernels for Embedding Extraction

**What it is**: Write custom CUDA C++ kernels to replace PyTorch/Transformers operations.

**Why anti-feature**:
- Extremely high complexity (100-1000+ lines of CUDA C++)
- Maintenance burden (breaks on model updates)
- Marginal benefit: Transformers library already well-optimized
- FlashAttention and DeepSpeed provide most gains without custom kernels

**Recommendation**: Avoid unless profiling shows specific kernel is bottleneck (unlikely).

---

### AF-04: FP8 or FP4 Precision

**What it is**: 8-bit or 4-bit floating point (newer than INT8, requires H100+ GPUs).

**Why anti-feature**:
- Requires cutting-edge hardware (H100, Ada/Blackwell GPUs)
- Limited library support (TransformerEngine FP8 is new)
- FP4 (NVFP4) still experimental
- BF16 provides sufficient memory savings (2x) with broad compatibility

**Recommendation**: Monitor for future (2027+), but prioritize BF16 for 2026.

---

## Feature Dependencies

```
TS-02 (Batch Size) ──> TS-04 (Mixed Precision)  # Larger batches need memory savings
                   └──> DF-03 (Quantization)     # Alternative for memory savings

TS-03 (DataLoader) ──> TS-05 (CUDA Streams)     # Prefetching complements async streams

TS-01 (Data Parallel) ──> DF-04 (Tensor Parallel)  # Can combine both
                      └──> DF-06 (Pipeline Parallel) # Mutual exclusive

DF-05 (ZeRO-Inference) ──> TS-02 (Batch Size)   # Enables larger batches
```

**Critical path for VirNucPro**:
1. TS-01 (Data Parallelism) + TS-02 (Batch Size) = 80% of speedup
2. TS-04 (Mixed Precision) = Additional 2x memory/speed
3. TS-03 (DataLoader) + TS-05 (CUDA Streams) = Remove CPU bottlenecks

**Phase 2 differentiators** (if critical path doesn't hit 10-hour target):
- DF-02 (FlashAttention) for ESM-2
- DF-03 (INT8 Quantization) if memory-bound

---

## Implementation Priorities

### Phase 1: Critical Path (Target: <10 hours)

1. **TS-01**: Implement DDP for ESM-2 (currently single GPU)
2. **TS-04**: Enable BF16 for both models (2x speedup)
3. **TS-02**: Profile and increase batch sizes (expect 2-4x)
4. **TS-03**: Add DataLoader with prefetching (remove I/O bottleneck)

**Expected outcome**: 4-8x total speedup (45 hours → 6-11 hours)

### Phase 2: Differentiators (if needed)

5. **DF-02**: Integrate FlashAttention-2 for ESM-2
6. **TS-05**: Implement CUDA streams for async I/O

**Expected outcome**: Additional 1.5-2x speedup (6 hours → 3-4 hours)

### Phase 3: Advanced (if aggressive targets)

7. **DF-03**: Test INT8 quantization (validate embedding quality)
8. **DF-05**: Evaluate DeepSpeed ZeRO-Inference for massive batches

---

## Quality Gate Checklist

- [x] Categories are clear (table stakes vs differentiators vs anti-features)
- [x] Complexity noted for each feature (LOW/MEDIUM/HIGH)
- [x] Dependencies between features identified (dependency graph included)
- [x] VirNucPro-specific applications documented
- [x] Trade-offs analyzed for each technique
- [x] Implementation priorities defined (Phase 1/2/3)

---

## Sources

### Table Stakes References
- [Mastering LLM Techniques: Inference Optimization | NVIDIA](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/)
- [PyTorch DDP Tutorial](https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [DataParallel vs DistributedDataParallel](https://medium.com/@kuipasta1121/dataparallel-vs-distributeddataparallel-in-pytorch-whats-the-difference-0af10bb43bc7)
- [Finding Optimal Batch Size | DigitalOcean](https://www.digitalocean.com/community/tutorials/find-optimal-batch-size)
- [8 PyTorch DataLoader Tactics](https://medium.com/@Modexa/8-pytorch-dataloader-tactics-to-max-out-your-gpu-22270f6f3fa8)
- [Data Prefetching in Deep Learning](https://www.jpatrickpark.com/post/prefetcher/)
- [Mixed Precision Training Guide](https://markaicode.com/transformers-mixed-precision-training-fp16-bf16/)
- [CUDA Streams Introduction](https://wentao.site/cuda_streams/)
- [Optimizing GPU Performance with CUDA Streams](https://medium.com/@kailashcvm/optimizing-gpu-performance-with-cuda-streams-and-batch-sizes-a1725debf86c)

### Differentiators References
- [vLLM Continuous Batching](https://www.anyscale.com/blog/continuous-batching-llm-inference)
- [Inside vLLM: Anatomy of a High-Throughput LLM Inference System](https://www.aleksagordic.com/blog/vllm)
- [FlashAttention Paper](https://arxiv.org/abs/2205.14135)
- [FlashAttention-2 Improvements](https://openreview.net/forum?id=mZn2Xyh9Ec)
- [LLM Quantization Guide | BentoML](https://bentoml.com/llm/getting-started/llm-quantization)
- [8-bit Matrix Multiplication | Hugging Face](https://huggingface.co/blog/hf-bitsandbytes-integration)
- [What is Inference Parallelism](https://www.infracloud.io/blogs/inference-parallelism/)
- [Scaling LLM Inference | Meta Engineering](https://engineering.fb.com/2025/10/17/ai-research/scaling-llm-inference-innovations-tensor-parallelism-context-parallelism-expert-parallelism/)
- [ZeRO-Inference | DeepSpeed](https://www.deepspeed.ai/2022/09/09/zero-inference.html)
- [DeepSpeed Inference Tutorial](https://www.deepspeed.ai/tutorials/inference-tutorial/)

### ESM-2 Specific
- [ESM-2 Model Overview | BioNeMo](https://docs.nvidia.com/bionemo-framework/latest/models/ESM-2/)
- [Efficiently Fine-tune ESM-2 | AWS](https://aws.amazon.com/blogs/machine-learning/efficiently-fine-tune-the-esm-2-protein-language-model-with-amazon-sagemaker/)
- [Efficient Inference, Training, and Fine-tuning of Protein Language Models](https://www.biorxiv.org/content/10.1101/2024.10.22.619563v1.full.pdf)

### Additional Resources
- [Defeating Training-Inference Mismatch via FP16](https://arxiv.org/html/2510.26788v1)
- [AI Model Quantization | RunPod](https://www.runpod.io/articles/guides/ai-model-quantization-reducing-memory-usage-without-sacrificing-performance)
- [LLM Quantization: BF16 vs FP8 vs INT4 in 2026](https://research.aimultiple.com/llm-quantization/)
- [Parallelism Techniques for LLM Training | Genesis Cloud](https://www.genesiscloud.com/blog/top-parallelism-techniques-llm-training)
