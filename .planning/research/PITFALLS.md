# Pitfalls Research: ESM2 3B to FastESM2_650 Migration

**Domain:** Protein language model migration for viral sequence classification
**Researched:** 2026-02-07
**Confidence:** MEDIUM

## Critical Pitfalls

### Pitfall 1: Embedding Dimension Mismatch Breaking Downstream MLP

**What goes wrong:**
The MLP classifier is hardcoded with `input_dim = 3328` (768 DNABERT-S + 2560 ESM2 3B). When switching to FastESM2_650, the protein embedding dimension drops from 2560 to 1280, resulting in a total input dimension of 2048 (768 + 1280). Loading the old checkpoint or using the hardcoded value causes a tensor shape mismatch error that crashes during inference or training.

**Why it happens:**
ESM2 model variants have different hidden dimensions based on model size. ESM2 3B has 36 layers, 40 attention heads, and **hidden dimension 2560**. FastESM2_650 maintains ESM2-650M architecture with 33 layers, 20 attention heads, and **hidden dimension 1280**. The dimension change is architectural, not a configuration option - it's baked into the model structure.

**How to avoid:**
1. **Never hardcode input dimensions.** Instead, detect them dynamically from embedding shapes:
   ```python
   # BAD - hardcoded
   input_dim = 3328

   # GOOD - dynamic detection
   sample_dna_emb = dnabert_embeddings[0].shape[-1]  # 768
   sample_protein_emb = esm_embeddings[0].shape[-1]  # 1280 or 2560
   input_dim = sample_dna_emb + sample_protein_emb
   ```

2. **Document embedding dimensions in checkpoint metadata:**
   ```python
   torch.save({
       'model_state_dict': model.state_dict(),
       'dna_emb_dim': 768,
       'protein_emb_dim': 2560,  # or 1280
       'total_input_dim': 3328,  # or 2048
       'esm_model': 'esm2_t36_3B_UR50D',  # or 'FastESM2_650'
   }, checkpoint_path)
   ```

3. **Add dimension validation in data pipeline:**
   ```python
   expected_protein_dim = 1280 if using_fastesm else 2560
   assert protein_embedding.shape[-1] == expected_protein_dim, \
       f"Protein embedding dimension mismatch: expected {expected_protein_dim}, got {protein_embedding.shape[-1]}"
   ```

**Warning signs:**
- RuntimeError: "size mismatch for hidden_layer.weight: copying a param with shape torch.Size([512, 2048]) from checkpoint, where the shape is torch.Size([512, 3328]) in current model"
- ValueError during `torch.cat()`: "Sizes of tensors must match except in dimension -1"
- Silent accuracy degradation (if using wrong slice of features instead of failing)

**Phase to address:**
Phase 1 (Feature Extraction Pipeline Update) - Implement dynamic dimension detection before extracting any FastESM2 features. Phase 2 (Model Retraining) must validate dimensions match training data.

---

### Pitfall 2: Model Checkpoint State Dict Key Mismatch

**What goes wrong:**
When loading a checkpoint trained with ESM2 3B into a pipeline using FastESM2_650 (or vice versa), the MLP classifier fails to load because the `hidden_layer.weight` and `hidden_layer.bias` shapes don't match. The error presents as "unexpected key(s) in state_dict" or "missing key(s) in state_dict" warnings, followed by immediate prediction failure or NaN losses.

**Why it happens:**
PyTorch's `load_state_dict()` performs strict shape matching by default. Even though the MLP architecture is identical (same hidden_dim=512, same num_classes=2), the **input layer shape changes** from [512, 3328] to [512, 2048]. The checkpoint from the old model is incompatible with the new input dimension.

**How to avoid:**
1. **Never reuse old checkpoints after model migration.** Train from scratch or use transfer learning properly:
   ```python
   # BAD - direct loading across model changes
   mlp_model.load_state_dict(torch.load('300_model.pth'))

   # GOOD - conditional loading with validation
   checkpoint = torch.load('300_model.pth')
   if checkpoint.get('total_input_dim') == current_input_dim:
       mlp_model.load_state_dict(checkpoint['model_state_dict'])
   else:
       print(f"Dimension mismatch: checkpoint has {checkpoint['total_input_dim']}, current model has {current_input_dim}. Training from scratch.")
       # Initialize new model
   ```

2. **Use transfer learning for output layers only:**
   ```python
   # Transfer only the output layer if hidden_dim unchanged
   old_checkpoint = torch.load('esm2_3b_model.pth')
   new_model_dict = mlp_model.state_dict()

   # Only load layers that match dimensions
   pretrained_dict = {k: v for k, v in old_checkpoint['model_state_dict'].items()
                      if k in new_model_dict and v.shape == new_model_dict[k].shape}

   new_model_dict.update(pretrained_dict)
   mlp_model.load_state_dict(new_model_dict)
   ```

3. **Namespace checkpoints by embedding configuration:**
   ```python
   # Save with descriptive names
   checkpoint_name = f"{chunk_size}_model_fastesm650.pth"  # not just "300_model.pth"
   ```

**Warning signs:**
- Warning: "Some weights of MLPClassifier were not initialized from the model checkpoint"
- Error: "size mismatch for hidden_layer.weight"
- Checkpoint loads without error but produces random predictions (partial parameter loading)
- Training loss starts high (>0.5) when it should start lower with pretrained weights

**Phase to address:**
Phase 2 (Model Retraining) - Implement checkpoint validation and versioning before training. Phase 3 (Validation Testing) must verify no cross-contamination between ESM2 and FastESM2 checkpoints.

---

### Pitfall 3: PyTorch Version Incompatibility with FastESM2 SDPA

**What goes wrong:**
FastESM2 requires PyTorch 2.5+ for Scaled Dot-Product Attention (SDPA) optimizations. On older PyTorch versions (<2.0), FastESM2 falls back to slower attention or crashes with "AttributeError: module 'torch.nn.functional' has no attribute 'scaled_dot_product_attention'". The speedup benefit of FastESM2 disappears, negating the entire migration rationale.

**Why it happens:**
FastESM2 is built on PyTorch's native SDPA (introduced in PyTorch 2.0, optimized in 2.5+). The architecture assumes this API exists. ESM2 3B uses older attention mechanisms compatible with PyTorch 1.x. Migrating without updating PyTorch means you get FastESM2's smaller model size but lose the speed improvements, resulting in **slower inference than ESM2 3B** due to reduced model capacity without compensation.

**How to avoid:**
1. **Verify PyTorch version before migration:**
   ```python
   import torch
   pytorch_version = tuple(int(x) for x in torch.__version__.split('.')[:2])

   if pytorch_version < (2, 5):
       raise RuntimeError(f"FastESM2 requires PyTorch 2.5+. Current version: {torch.__version__}. "
                         f"Update with: pip install torch>=2.5")
   ```

2. **Check CUDA compatibility for SDPA backend:**
   ```python
   # PyTorch 2.5+ with H100 or newer gets cuDNN SDPA backend
   # Older GPUs still benefit but less dramatically
   if torch.cuda.is_available():
       gpu_name = torch.cuda.get_device_name(0)
       print(f"GPU: {gpu_name}")
       if "H100" in gpu_name or "A100" in gpu_name:
           print("Optimal SDPA performance expected")
   ```

3. **Add PyTorch version to pixi.toml dependencies:**
   ```toml
   [dependencies]
   python = "3.9.*"
   pytorch = ">=2.5,<3.0"  # Explicit version requirement
   ```

4. **Benchmark to confirm speedup:**
   ```python
   # After migration, verify FastESM2 is actually faster than ESM2 3B
   # If not, PyTorch version or GPU may be limiting
   import time
   start = time.time()
   # run inference
   elapsed = time.time() - start

   # FastESM2 should be 2-3x faster on same batch size
   assert elapsed < esm2_baseline_time * 0.6, "FastESM2 not showing expected speedup"
   ```

**Warning signs:**
- FastESM2 runs but is slower than expected (not 2x+ faster than ESM2 3B)
- Import warnings about missing torch.nn.functional functions
- Fallback messages in logs: "SDPA not available, using standard attention"
- flash-attn installation errors (optional but indicates PyTorch version issues)

**Phase to address:**
Phase 0 (Environment Setup) - Validate PyTorch version before starting migration. Include in .planning/DEPENDENCIES.md. Phase 5 (Speed Benchmarking) must verify SDPA is actually being used.

---

### Pitfall 4: Batch Size and Memory Assumptions Broken

**What goes wrong:**
Code tuned for ESM2 3B's memory footprint uses batch_size=32 or processes 2 workers for protein extraction. After migrating to FastESM2_650 (5x smaller model), the same configuration underutilizes GPU memory, leaving 60-70% memory unused and throttling throughput. Conversely, naively increasing batch size to match ESM2 3B throughput can trigger OOM errors if DNABERT-S batching wasn't adjusted proportionally.

**Why it happens:**
ESM2 3B has 3 billion parameters (~12GB model weights). FastESM2_650 has 650 million parameters (~2.5GB weights). Peak memory usage is determined by `model_size + batch_size * (activation_memory + gradient_memory)`. The old configuration `processes=2` in `multiprocessing.Pool` was chosen to avoid OOM with ESM2 3B. With FastESM2, you can safely run `processes=4` or higher, but only if you recalculate the memory budget including DNABERT-S parallel processing.

**How to avoid:**
1. **Profile actual memory usage with both models:**
   ```python
   import torch

   # Before migration (ESM2 3B)
   torch.cuda.reset_peak_memory_stats()
   # run batch
   esm2_peak_mb = torch.cuda.max_memory_allocated() / 1024**2

   # After migration (FastESM2_650)
   torch.cuda.reset_peak_memory_stats()
   # run batch
   fastesm_peak_mb = torch.cuda.max_memory_allocated() / 1024**2

   print(f"Memory reduction: {esm2_peak_mb - fastesm_peak_mb:.0f} MB")
   # Use freed memory to increase batch size or parallelism
   ```

2. **Adjust batch size proportionally to memory savings:**
   ```python
   # Rule of thumb: If model is 5x smaller, you can ~3-4x batch size
   # (not 5x due to activation memory scaling with batch size)

   OLD_BATCH_SIZE = 32  # for ESM2 3B
   MEMORY_RATIO = 0.2  # FastESM2 uses ~20% of ESM2 memory

   new_batch_size = int(OLD_BATCH_SIZE / MEMORY_RATIO * 0.7)  # Conservative 3.5x
   # Test for OOM, then increase incrementally
   ```

3. **Increase multiprocessing workers for protein extraction:**
   ```python
   # OLD (ESM2 3B)
   with multiprocessing.Pool(processes=2) as pool:
       results = pool.map(process_file_pro, viral_protein_files)

   # NEW (FastESM2_650) - more parallelism
   with multiprocessing.Pool(processes=6) as pool:  # 3x workers
       results = pool.map(process_file_pro, viral_protein_files)
   ```

4. **Watch for DNABERT-S bottleneck:**
   ```python
   # DNABERT-S was already using 8 workers - may now be the bottleneck
   # Monitor both processes to ensure balanced throughput
   import psutil
   # Log GPU utilization during extraction to identify bottleneck
   ```

**Warning signs:**
- GPU memory utilization drops from 90% to 30% after migration
- Protein feature extraction takes nearly the same time despite smaller model
- nvidia-smi shows low GPU utilization (<50%) during inference
- CPU is maxed but GPU idle (indicates multiprocessing bottleneck, not model speed)

**Phase to address:**
Phase 1 (Feature Extraction Pipeline Update) - Profile and optimize batch sizes and worker counts. Phase 5 (Speed Benchmarking) must measure end-to-end throughput, not just model inference time.

---

### Pitfall 5: Silent Tokenization Compatibility Assumptions

**What goes wrong:**
Developers assume FastESM2 and ESM2 use identical tokenization since FastESM2 is marketed as "drop-in replacement". While the tokenization is indeed the same, the **sequence length limits and batch packing behavior** differ. FastESM2 with FlashAttention can handle up to 100,000 tokens without OOM, while ESM2 3B hits memory limits around 10,000-20,000 tokens. Code that truncates sequences to 1024 tokens for ESM2 3B wastes FastESM2's capability, but removing truncation without adjusting downstream code causes shape mismatches in MLP input.

**Why it happens:**
FastESM2 uses FlashAttention and sequence packing to handle long sequences efficiently. The original ESM2 implementation has quadratic memory scaling with sequence length. Most users added truncation as a workaround. During migration, this truncation is forgotten or assumed necessary, resulting in no benefit from FastESM2's long-sequence handling.

**How to avoid:**
1. **Audit all sequence truncation logic:**
   ```python
   # Find all places where sequences are truncated
   # grep -r "[:1024]" *.py
   # grep -r "truncation=True" *.py

   # Current code likely has:
   # sequences = [seq[:1024] for seq in sequences]  # ESM2 3B limitation

   # With FastESM2, this can be removed IF:
   # - You re-train the MLP with longer sequence embeddings
   # - OR you keep truncation for consistency with old model
   ```

2. **Document sequence length assumptions in validation:**
   ```python
   MAX_SEQ_LEN = 1024  # Document why this limit exists
   # For ESM2 3B: memory constraint
   # For FastESM2: kept for compatibility with trained MLP

   # If removing truncation, validate shape consistency:
   assert all(emb.shape[0] <= MAX_SEQ_LEN for emb in embeddings), \
       "Sequence length exceeds trained model's expected input"
   ```

3. **Test with edge-case long sequences:**
   ```python
   # Add test case with 2000+ amino acid sequence
   # Ensure it doesn't break pipeline silently
   long_test_seq = "M" * 2500
   try:
       embedding = extract_embedding(long_test_seq)
       # Verify downstream model can handle it
   except Exception as e:
       print(f"Long sequence handling failed: {e}")
   ```

**Warning signs:**
- Sequences are still being truncated but no one remembers why
- FastESM2 speed benchmarks don't show expected improvement (truncation limits batch efficiency)
- Variable-length sequences cause intermittent shape errors
- Embeddings have consistent shape (e.g., always [1024, 1280]) even for shorter inputs (indicates forced padding/truncation)

**Phase to address:**
Phase 1 (Feature Extraction Pipeline Update) - Document and validate all sequence length constraints. Phase 4 (Comparison Testing) must test edge cases with long sequences to ensure no silent failures.

---

## Technical Debt Patterns

Shortcuts that seem reasonable but create long-term problems.

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Hardcoding `input_dim=3328` in MLP | Faster prototyping | Breaks silently when embedding model changes; requires code search to fix | Never - always compute dynamically |
| Reusing ESM2 checkpoints with FastESM2 embeddings | Avoid retraining time | Silent accuracy degradation or crashes; impossible to debug | Never - dimension mismatch is unrecoverable |
| Skipping PyTorch version check | Faster deployment | FastESM2 runs slowly or crashes; loses migration benefit | Only in tightly controlled environments with pinned dependencies |
| Keeping old batch_size=32 after migration | No code changes needed | Leaves 70% GPU memory unused; 3x slower than possible | Acceptable for MVP validation, must optimize for production |
| Assuming tokenization is identical without testing | Works for most sequences | Edge cases with special characters or long sequences break silently | Acceptable if comprehensive test suite covers edge cases |
| Training from scratch instead of transfer learning output layer | Simpler implementation | Loses domain-specific knowledge from ESM2 3B training | Acceptable if training data is large (>10K samples) |

## Performance Traps

Patterns that work at small scale but fail as usage grows.

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| Single-worker protein extraction with FastESM2 | GPU utilization <30% | Increase `processes=2` to `processes=6` in multiprocessing.Pool | Immediately - GPU idles while CPU processes features sequentially |
| Batch size tuned for ESM2 3B memory limits | Slow inference, wasted GPU memory | Re-profile memory usage and increase batch size 3-4x | Immediately - FastESM2 uses 5x less memory |
| Sequential feature extraction (DNA then protein) | Long wait times for large datasets | Parallelize both with balanced worker counts | >1000 sequences - DNA extraction becomes bottleneck |
| Loading full checkpoint into memory for every worker | Memory usage scales with workers | Use shared memory or load model once in parent process | >4 workers - OOM despite small model size |
| No gradient checkpointing for MLP fine-tuning | Works for small hidden_dim | Cannot increase hidden_dim or add layers | When experimenting with deeper MLPs (3+ layers) |

## Integration Gotchas

Common mistakes when connecting FastESM2 to the existing pipeline.

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| HuggingFace model loading | Using `facebook/esm2_t33_650M_UR50D` instead of `Synthyra/FastESM2_650` | FastESM2 is a separate model with optimized attention; must use exact model ID |
| Feature merging | Assuming `torch.cat([dna_emb, protein_emb], dim=-1)` works unchanged | Works for concatenation, but validate shapes: ESM2 3B gives (N, 2560), FastESM2 gives (N, 1280) |
| Data loader compatibility | Reusing old `.pt` files with 3328-dim features | Old files have wrong dimensions; must regenerate all embeddings |
| Model.eval() mode | Forgetting to set FastESM2 to eval mode | FastESM2 has dropout/batch norm; eval() is critical for reproducible embeddings |
| Device placement | Loading FastESM2 to GPU but DNABERT-S to CPU | Creates device mismatch during torch.cat; both models must be on same device |
| Checkpoint format | Saving full model with `torch.save(model, ...)` | Breaks across PyTorch versions; use `torch.save(model.state_dict(), ...)` |

## "Looks Done But Isn't" Checklist

Things that appear complete but are missing critical pieces.

- [ ] **Feature extraction completes without errors:** Often missing dimension validation - check output shape matches expected [batch, 2048] not [batch, 3328]
- [ ] **MLP trains and loss decreases:** Often missing checkpoint compatibility check - verify trained with FastESM2 embeddings, not ESM2 3B
- [ ] **Test accuracy is reasonable (>0.8):** Often missing comparison to baseline - must compare to ESM2 3B on same test set to detect degradation
- [ ] **Inference runs without crashes:** Often missing edge case testing - test with sequences <50aa, >1000aa, sequences with rare amino acids
- [ ] **Speed is faster than ESM2 3B:** Often missing end-to-end measurement - check wall-clock time for full pipeline, not just embedding extraction
- [ ] **Checkpoints are versioned and labeled:** Often missing metadata - checkpoint file should include model type, dimensions, training date in filename
- [ ] **Documentation updated with new dimensions:** Often missing in README/docstrings - `input_dim=3328` comments must be updated to `input_dim=2048`
- [ ] **PyTorch version pinned in requirements:** Often missing from pixi.toml - must specify `pytorch>=2.5` for SDPA support

## Recovery Strategies

When pitfalls occur despite prevention, how to recover.

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| Trained MLP with wrong input dimension | MEDIUM | 1. Check checkpoint metadata for input_dim. 2. If mismatch, retrain MLP from scratch with correct embeddings. 3. Estimate 2-4 hours for retraining on full dataset. |
| Mixed ESM2 and FastESM2 embeddings in dataset | HIGH | 1. Delete all `.pt` files in data/data_merge/. 2. Regenerate embeddings with consistent model. 3. Verify file count matches before retraining. Estimate 6-12 hours for full dataset re-extraction. |
| Old checkpoint loaded into new model | LOW | 1. Detect via NaN losses or random predictions. 2. Delete checkpoint, train from scratch. 3. If accuracy was good, investigate transfer learning from output layer only. |
| PyTorch version too old for FastESM2 | MEDIUM | 1. Update PyTorch: `pip install torch>=2.5`. 2. Verify CUDA compatibility. 3. Re-run feature extraction (old embeddings may be from slow fallback). 4. Estimate 1-2 hours for environment update + validation. |
| Batch size not optimized | LOW | 1. Profile GPU memory during inference. 2. Increase batch size incrementally until 85-90% memory utilization. 3. Re-benchmark speed. 4. Update constants in code. 5. Estimate 30 minutes. |
| Sequence truncation forgotten | MEDIUM | 1. Audit all sequence processing code. 2. Decide: keep truncation for compatibility or remove for better embeddings. 3. If removing, retrain MLP. 4. Document decision in code comments. |

## Pitfall-to-Phase Mapping

How roadmap phases should address these pitfalls.

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| Embedding dimension mismatch | Phase 1: Feature Extraction | Extract single sample, print shape, verify 2048 not 3328 |
| Checkpoint incompatibility | Phase 2: Model Retraining | Load checkpoint, check state_dict keys match model architecture |
| PyTorch version incompatibility | Phase 0: Environment Setup | Run `assert torch.__version__ >= "2.5"` before importing FastESM2 |
| Batch size underutilization | Phase 5: Speed Benchmarking | Monitor `nvidia-smi` during inference, verify >80% GPU memory usage |
| Tokenization edge cases | Phase 1: Feature Extraction | Test with 10 edge cases: empty seq, 5000aa seq, special chars, etc. |
| State dict key warnings | Phase 2: Model Retraining | Enable strict=True in load_state_dict, catch any warnings as errors |
| DNABERT-S / ESM2 device mismatch | Phase 1: Feature Extraction | Add assertion: `assert dna_emb.device == protein_emb.device` |
| Undocumented dimension changes | Phase 4: Comparison Testing | Code review checklist: all hardcoded 3328 updated to 2048 or dynamic |
| Checkpoint namespace collision | Phase 2: Model Retraining | Rename: `300_model.pth` → `300_model_esm2_3b.pth`, new → `300_model_fastesm650.pth` |
| Missing metadata in checkpoints | Phase 2: Model Retraining | Verify saved checkpoint includes: model_name, input_dim, training_date |

## Sources

### Official Documentation
- [NVIDIA BioNeMo Framework - ESM-2](https://docs.nvidia.com/bionemo-framework/2.0/models/esm2/) - ESM2 architecture specifications
- [Synthyra/FastESM2_650 - Hugging Face](https://huggingface.co/Synthyra/FastESM2_650) - FastESM2 model card and usage
- [PyTorch SDPA Tutorial](https://pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html) - Scaled Dot-Product Attention requirements

### Research Papers and Community Knowledge
- [Efficient inference, training, and fine-tuning of protein language models - iScience](https://www.cell.com/iscience/fulltext/S2589-0042(25)01756-0) - Memory optimization and FlashAttention for protein LMs
- [Scalable embedding fusion with protein language models - Oxford Academic](https://academic.oup.com/bib/article/27/1/bbag014/8444383) - Downstream task compatibility
- [ESM-2 Fine-tuning - BioNeMo Framework](https://docs.nvidia.com/bionemo-framework/2.6.3/main/examples/bionemo-esm2/finetune/) - Checkpoint loading and state_dict key handling

### GitHub Issues and Forums
- [Upgrading from ESM2-650M to ESM2-3B - GitHub Issue](https://github.com/matsengrp/dasm-experiments/issues/1) - Memory and speed considerations
- [Warning when using ESM pre-trained model - Hugging Face Forums](https://discuss.huggingface.co/t/warning-when-using-esm-pre-trained-model/38489) - State dict loading warnings
- [Size mismatch error in PEFT fine tuned model - Hugging Face Forums](https://discuss.huggingface.co/t/size-mismatch-error-in-peft-fine-tuned-model/95040) - Checkpoint dimension mismatch recovery

### Performance and Optimization
- [FlashAttention-3 - PyTorch Blog](https://pytorch.org/blog/flashattention-3/) - SDPA backend performance on H100+
- [PyTorch 2.5 Release Blog](https://pytorch.org/blog/pytorch2-5/) - cuDNN SDPA backend introduction

---
*Pitfalls research for: VirNucPro FastESM2 Migration*
*Researched: 2026-02-07*
*Confidence: MEDIUM - Based on official documentation, research papers, and community reports. Some project-specific pitfalls extrapolated from existing codebase analysis.*
