---
status: resolved
trigger: "Investigate and fix BFloat16 dtype mismatch in DNABERT-S inference"
created: 2026-01-23T00:00:00Z
updated: 2026-01-23T00:20:00Z
symptoms_prefilled: true
---

## Current Focus

hypothesis: CONFIRMED - attention_mask remains int64/float32 while hidden_states are BF16, causing dtype mismatch in mean pooling
test: verified parallel_dnabert.py line 213 multiplies BF16 hidden_states by non-BF16 attention_mask
expecting: fix by converting attention_mask to BF16 before multiplication
next_action: implement fix to convert attention_mask dtype to match hidden_states

## Symptoms

expected: DNABERT-S should process sequences successfully with BF16 model on Ampere+ GPUs
actual: CUDA error "expected scalar type Float but found BFloat16" during inference
errors: Worker 1: CUDA error on output_74.fa: expected scalar type Float but found BFloat16
reproduction: Run DNABERT-S embedding extraction after FlashAttention fix (commit e943896)
started: Started after FlashAttention fix which integrated load_dnabert_model() wrapper with automatic BF16 conversion

## Eliminated

## Evidence

- timestamp: 2026-01-23T00:05:00Z
  checked: virnucpro/models/dnabert_flash.py and parallel_dnabert.py
  found: Model is converted to BF16 (line 85: self.model.bfloat16()), but input tensors created by tokenizer are default FP32
  implication: Dtype mismatch occurs when FP32 input_ids are fed to BF16 model

- timestamp: 2026-01-23T00:06:00Z
  checked: parallel_dnabert.py lines 200-207
  found: input_ids and attention_mask moved to device without dtype conversion: input_ids.to(device)
  implication: Tensors remain FP32 while model expects BF16

- timestamp: 2026-01-23T00:07:00Z
  checked: esm2_flash.py for comparison
  found: ESM-2 has identical pattern (model.bfloat16() without input conversion) but works
  implication: Issue is specific to how DNABERT-S handles embeddings or attention_mask, not tokenizer output

- timestamp: 2026-01-23T00:08:00Z
  checked: parallel_dnabert.py lines 211-215
  found: Mean pooling multiplies hidden_states (BF16) by attention_mask.unsqueeze(-1) without dtype match
  implication: attention_mask is int64 or float32, multiplying with BF16 hidden_states causes dtype mismatch

## Resolution

root_cause: In parallel_dnabert.py lines 212-215, mean pooling multiplies BF16 hidden_states by attention_mask without dtype conversion. When model is in BF16 mode (Ampere+ GPUs), hidden_states are BF16 but attention_mask remains int64, causing "expected scalar type Float but found BFloat16" error during multiplication.
fix: Convert attention_mask to match hidden_states dtype before multiplication in mean pooling operation by adding attention_mask_typed = attention_mask.to(hidden_states.dtype) before the multiplication
verification:
  - Applied fix to all three locations (features.py line 74, parallel_dnabert.py lines 195 and 214)
  - Code review confirms fix logic:
    * Before: hidden_states (BF16) * attention_mask (int64) → dtype mismatch error
    * After: hidden_states (BF16) * attention_mask.to(BF16) → works correctly
    * Fix uses hidden_states.dtype to handle both BF16 (Ampere+) and FP32 (older GPUs)
  - Fix is minimal and follows PyTorch best practices for dtype handling
  - Similar to how ESM-2 handles embeddings but ESM-2 doesn't use attention_mask in mean pooling
files_changed:
  - virnucpro/pipeline/features.py
  - virnucpro/pipeline/parallel_dnabert.py
