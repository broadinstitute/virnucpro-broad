# Phase 2: Feature Extraction Pipeline - Research

**Researched:** 2026-02-07
**Domain:** FastESM2_650 protein embedding extraction via HuggingFace transformers API
**Confidence:** HIGH

## Summary

This phase implements `extract_fast_esm()`, a new function in `units.py` that replaces the deprecated `extract_esm()` to produce 1280-dim protein embeddings using FastESM2_650. The function must produce output files in exactly the same format as the original ESM2 3B pipeline (`{'proteins': [labels], 'data': [tensor_embeddings]}`) so that `merge_data()` and downstream code consume them without modification.

The research analyzed the complete FastESM2_650 model architecture (from the locally cached `modeling_fastesm.py`), the original `extract_esm()` implementation (from git history), the `merge_data()` consumption pattern, and the `features_extract.py` orchestration code. Key findings: (1) The model has a built-in `_embed()` method and `embed_dataset()` method on the `EmbeddingMixin`, but the output format of `embed_dataset()` maps sequences-to-embeddings (dict) rather than preserving FASTA label ordering, so we cannot use it directly; (2) The tokenizer is a class attribute (`model.tokenizer`) not loaded separately; (3) Mean pooling must account for padding tokens via `attention_mask` to produce correct results; (4) The original ESM2 extraction used `truncation_seq_length=1024` to truncate long sequences before mean pooling -- this must be replicated; (5) GPU-based extraction cannot use multiprocessing.Pool for parallelism since only one process can own the GPU -- batch processing within a single process is the correct parallel strategy.

**Primary recommendation:** Implement `extract_fast_esm()` using the model's built-in tokenizer and `forward()` method with manual mean pooling, processing FASTA files in batches via a DataLoader with `toks_per_batch`-based dynamic batching. Use sequential (single-process) GPU extraction with file-level parallelism handled by processing files in a loop with tqdm progress.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

#### API Integration Approach
- Load model once at start, keep in GPU memory for entire extraction run
- Process all sequences without reloading model (training workflow optimization)
- Always use SDPA optimization (hardcoded, validated in Phase 1)
- No CPU fallback mode - assume GPU memory sufficient for model + batch

#### Batch Processing Strategy
- Maintain original ESM2 parameters for fair comparison: `truncation_seq_length=1024`, `toks_per_batch=2048`
- Support batch file processing (optimize to process multiple files efficiently, not single file per call)
- Use tqdm progress bar (matches existing DNABERT_S extraction style)
- Built-in parallel processing within extract_fast_esm() function

#### Error Handling & Validation
- Skip and log failures - continue processing remaining sequences on errors (resilient approach)
- Validate embeddings before saving:
  - Dimension check: verify exactly 1280-dim output
  - NaN/Inf detection: catch invalid values
  - Sequence count match: verify embeddings count matches input sequences
- Informative logging: log start/finish, major milestones (every 100 files), warnings/errors
- Write failures to separate log file (extraction_failures.log)

#### File I/O and Data Format
- Output format: `{'proteins': [labels], 'data': [embeddings]}` (exact match to ESM2 structure)
- File naming: keep `_ESM.pt` suffix (maintains compatibility with merge_data() and downstream code)
- Skip existing files: don't re-extract if output .pt file already exists (resume capability)

### Claude's Discretion
- Exact parallel processing implementation (multiprocessing vs threading given GPU constraints)
- Progress bar formatting details
- Retry logic for transient GPU errors
- Failure log format and detail level

### Deferred Ideas (OUT OF SCOPE)
None - discussion stayed within phase scope
</user_constraints>

## Standard Stack

The established libraries/tools for this phase:

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| transformers | 4.45.2 | HuggingFace model loading (AutoModel) | FastESM2_650 was developed with transformers 4.45.0; provides AutoModel API for loading custom model architecture |
| torch | 2.9.0a0+nv25.09 | GPU inference, tensor operations, model I/O | Provided by NVIDIA container; handles CUDA operations, SDPA, torch.save for .pt output |
| biopython | >=1.80 | FASTA file parsing | SeqIO.parse is the established pattern for reading protein sequences throughout the codebase |
| tqdm | >=4.27 | Progress bars | User decision: match existing DNABERT_S extraction style |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| einops | >=0.6.1 | Tensor rearrangement | Required internally by FastESM2_650 attention layers; not called directly in extract_fast_esm() |
| networkx | >=3.2 | Graph algorithms | Required internally by FastESM2_650 Pooler class; not called directly |
| logging | stdlib | Structured logging | For milestone logging, failure tracking, extraction_failures.log |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Manual forward + mean pooling | `model.embed_dataset()` (built-in) | embed_dataset() maps sequences-to-embeddings as a dict, losing FASTA label ordering and not matching the `{'proteins': [...], 'data': [...]}` output format. Manual approach gives exact control over output format. |
| Sequential GPU processing | multiprocessing.Pool for GPU | Cannot use multiprocessing for GPU work -- CUDA contexts do not fork safely. Batch processing within single process is the correct GPU parallelism strategy. |
| Custom tokenizer | `model.tokenizer` class attribute | No need to load separately; model.tokenizer is EsmTokenizer from "facebook/esm2_t6_8M_UR50D" and handles all amino acid tokenization |

**Installation:**
```bash
# Already installed in Docker environment from Phase 1
# No new dependencies needed for Phase 2
```

## Architecture Patterns

### Where extract_fast_esm() Lives

```
units.py                    # Add extract_fast_esm() function here
  extract_fast_esm()        # NEW: FastESM2_650 embedding extraction
  extract_esm()             # DEPRECATED: raises NotImplementedError
  extract_DNABERT_S()       # UNCHANGED: DNA embedding extraction
  merge_data()              # UNCHANGED: merges DNA + protein embeddings
  split_fasta_file()        # UNCHANGED: splits large FASTA files

features_extract.py         # UPDATE: process_file_pro() to call extract_fast_esm()
  process_file_pro()        # Change: extract_esm() -> extract_fast_esm()
  ESM_model, ESM_alphabet   # Change: load FastESM2_650 model + tokenizer at top

scripts/
  validate_environment.py   # UNCHANGED (from Phase 1)
```

### Pattern 1: Model Loading (Load Once, Use Many)
**What:** Load FastESM2_650 once at module level, pass to extract function
**When to use:** Always -- model loading takes 5-10 seconds, must not repeat per file
**Example:**
```python
# Source: modeling_fastesm.py lines 799-807, config.json
# At module level in features_extract.py:
from transformers import AutoModel
import torch

model = AutoModel.from_pretrained(
    "Synthyra/FastESM2_650",
    trust_remote_code=True,
    torch_dtype=torch.float16
).eval().cuda()
tokenizer = model.tokenizer  # EsmTokenizer class attribute
```

### Pattern 2: Tokenization with Attention Masking
**What:** Tokenize protein sequences with padding and attention masks for batch processing
**When to use:** Every batch of sequences
**Example:**
```python
# Source: modeling_fastesm.py line 641 (build_collator function)
# The tokenizer accepts lists of strings directly:
sequences = ['MPRTEIN', 'MSEQWENCE']
tokenized = tokenizer(
    sequences,
    return_tensors='pt',
    padding='longest',          # Pad to longest in batch
    truncation=True,
    max_length=1024 + 2         # +2 for BOS/EOS tokens to match truncation_seq_length=1024
)
# tokenized has 'input_ids' and 'attention_mask' keys
input_ids = tokenized['input_ids'].to('cuda')
attention_mask = tokenized['attention_mask'].to('cuda')
```

### Pattern 3: Mean Pooling with Attention Mask
**What:** Extract mean-pooled embeddings from last hidden state, excluding padding tokens
**When to use:** After model forward pass
**Example:**
```python
# Source: modeling_fastesm.py lines 559-564 (Pooler.mean_pooling method)
# The model's built-in Pooler class implements this correctly:
with torch.no_grad():
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    last_hidden = outputs.last_hidden_state  # (batch, seq_len, 1280)

# Mean pooling with attention mask (excludes padding):
mask = attention_mask.unsqueeze(-1)  # (batch, seq_len, 1)
embeddings = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1)  # (batch, 1280)
# Move to CPU for saving
embeddings = embeddings.float().cpu()  # Convert fp16 -> fp32 for storage
```

### Pattern 4: Dynamic Batching by Token Count
**What:** Group sequences into batches where total tokens does not exceed `toks_per_batch`
**When to use:** To match original ESM2 batching strategy for fair comparison
**Example:**
```python
# Source: Original extract_esm() used fair-esm's FastaBatchedDataset.get_batch_indices(toks_per_batch=2048)
# Replicate this logic without fair-esm:
def get_batch_indices(sequence_lengths, toks_per_batch=2048):
    """Group sequences into batches by total token count."""
    batches = []
    current_batch = []
    current_toks = 0

    # Sort by length (descending) for efficient packing
    sorted_indices = sorted(range(len(sequence_lengths)),
                          key=lambda i: sequence_lengths[i], reverse=True)

    for idx in sorted_indices:
        seq_len = sequence_lengths[idx] + 2  # +2 for BOS/EOS special tokens
        if current_toks + seq_len > toks_per_batch and current_batch:
            batches.append(current_batch)
            current_batch = []
            current_toks = 0
        current_batch.append(idx)
        current_toks += seq_len

    if current_batch:
        batches.append(current_batch)

    return batches
```

### Pattern 5: Output Format Matching
**What:** Save embeddings in exact format consumed by merge_data()
**When to use:** When writing .pt output files
**Example:**
```python
# Source: Original extract_esm() (git show 0b7e6f1:units.py lines 256-257)
# Original output:
#   torch.save({'proteins': proteins, 'data': data}, out_file)
# Where:
#   proteins = [label1, label2, ...]  (list of FASTA sequence IDs)
#   data = [tensor1, tensor2, ...]    (list of 1D tensors, each 2560-dim for ESM2-3B)
#
# For FastESM2_650, data items will be 1280-dim tensors.
# merge_data() uses:
#   for protein, data in zip(ESM_outfile['proteins'], ESM_outfile['data']):
#       protein_data_dict[protein] = data
# So each 'data' item must be a tensor (not a dict).

proteins = []
data = []
for label, embedding in zip(labels, embeddings_list):
    proteins.append(label)
    data.append(embedding)  # 1D tensor of shape (1280,)

torch.save({'proteins': proteins, 'data': data}, out_file)
```

### Anti-Patterns to Avoid

- **Using multiprocessing.Pool for GPU inference:** CUDA contexts do not fork safely. The existing code uses `multiprocessing.Pool(processes=2)` for `process_file_pro()`, but the GPU model is loaded as a module-level global. With FastESM2, the model MUST be loaded once and used from a single process. File-level "parallelism" should instead be sequential iteration with efficient GPU batching.
- **Forgetting attention_mask in mean pooling:** Without masking out padding tokens, the mean will be diluted by zero embeddings from padding positions. This is a silent correctness bug that produces wrong embeddings.
- **Storing embeddings in fp16:** The original ESM2 pipeline stored tensors in float32 (they were computed from the model's default float32). While FastESM2_650 runs in fp16 for inference speed, the output embeddings should be converted to float32 before saving to maintain numerical precision for downstream training. The `merge_data()` function does `torch.cat((nucleotide_data, protein_data), dim=-1)` and the training code expects float32.
- **Using model.embed_dataset() directly:** While convenient, this returns a dict mapping sequences to embeddings, which loses FASTA label ordering and does not match the required `{'proteins': [...], 'data': [...]}` format.
- **Ignoring max_position_embeddings=1026:** The config says `max_position_embeddings=1026` which means 1024 actual residue positions + 2 special tokens. Sequences truncated to `truncation_seq_length=1024` residues will have 1026 tokens after tokenization (with BOS and EOS), exactly at the limit.

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Protein tokenization | Custom amino acid encoding | `model.tokenizer` (EsmTokenizer) | Handles BOS/EOS tokens, padding, truncation, special characters. 33-token vocab shared across all ESM2 models. |
| SDPA attention optimization | Manual attention with flash attention | Default `output_attentions=False` path | FastESM2 uses `F.scaled_dot_product_attention` automatically. No configuration needed. |
| FASTA file parsing | Custom file reader | `Bio.SeqIO.parse(file, 'fasta')` | Already used throughout codebase. Handles edge cases in FASTA format. |
| Mean pooling with mask | Simple `.mean(dim=1)` | Attention-mask-weighted mean (see Pattern 3) | Simple mean includes padding token embeddings, producing incorrect results for variable-length batches. |
| Progress bars | Custom print statements | `tqdm` | User decision: match existing DNABERT_S extraction style. |

**Key insight:** The FastESM2_650 model code already contains a complete embedding pipeline (`embed_dataset()` in `EmbeddingMixin`), but its output format (dict mapping sequences to embeddings) does not match what this project needs. The core model operations (tokenize, forward, pool) should be used directly, but the file I/O and output formatting must be custom to match the `{'proteins': [...], 'data': [...]}` format.

## Common Pitfalls

### Pitfall 1: Multiprocessing with CUDA
**What goes wrong:** Using `multiprocessing.Pool` to parallelize GPU inference causes CUDA initialization errors or silent data corruption.
**Why it happens:** CUDA contexts are not fork-safe. The existing `features_extract.py` uses `multiprocessing.Pool(processes=2)` for `process_file_pro()`, which worked with the old ESM2-3B because `esm.pretrained` loaded models independently in each worker. With a shared GPU model, this breaks.
**How to avoid:** Process all files sequentially in a single process that owns the GPU. Achieve throughput via efficient batch processing within each file, not across files. The model stays in GPU memory; file I/O is the only sequential overhead.
**Warning signs:** `RuntimeError: CUDA error: initialization error`, `RuntimeError: Cannot re-initialize CUDA in forked subprocess`.

### Pitfall 2: Truncation Mismatch
**What goes wrong:** Embeddings differ from original ESM2 because truncation is applied differently.
**Why it happens:** Original ESM2 used `truncation_seq_length=1024` which truncated at the representation level (after tokenization). HuggingFace tokenizer truncation works at the token level. Must account for BOS/EOS tokens.
**How to avoid:** Set tokenizer `max_length=1024+2` (1026) with `truncation=True`. This gives 1024 actual residue tokens plus 2 special tokens, matching `max_position_embeddings=1026` in the model config. The original ESM2 code did: `truncate_len = min(truncation_seq_length, len(strs[i]))` and then `t[i, 1 : truncate_len + 1].mean(0)` -- this skipped the BOS token (position 0) and took up to 1024 residue positions for mean pooling. We should replicate: mean pool over positions `1:truncate_len+1` (excluding BOS and EOS).
**Warning signs:** Subtle accuracy differences in downstream training that are hard to diagnose.

### Pitfall 3: fp16/fp32 Conversion for Storage
**What goes wrong:** Embeddings saved in fp16 cause numerical issues in downstream training.
**Why it happens:** FastESM2_650 runs in fp16 for speed. If embeddings are saved directly in fp16, the concatenation with DNABERT-S embeddings (which are float32) and subsequent training may lose precision.
**How to avoid:** Convert embeddings to float32 before saving: `embedding.float().cpu()`. The original ESM2 embeddings were stored as float32 tensors.
**Warning signs:** Training loss instability, unexpected NaN values in merged features.

### Pitfall 4: Padding Token Inclusion in Mean Pooling
**What goes wrong:** Mean-pooled embeddings are diluted by padding token representations.
**Why it happens:** When batching sequences of different lengths, shorter sequences are padded. A simple `hidden_states.mean(dim=1)` averages over padding positions too.
**How to avoid:** Use attention-mask-weighted mean pooling (Pattern 3 above). The FastESM2 `EsmEmbeddings.forward()` already masks embedding values: `embeddings = (embeddings * attention_mask.unsqueeze(-1)).to(embeddings.dtype)`, so padding positions have zero embeddings. However, the denominator must still count only non-padding positions.
**Warning signs:** Embeddings for the same sequence differ depending on what other sequences are in the batch.

### Pitfall 5: Missing BOS/EOS Token Exclusion in Mean Pooling
**What goes wrong:** Mean pooling includes BOS and EOS token representations, which were excluded in the original ESM2 pipeline.
**Why it happens:** The original ESM2 code explicitly excluded special tokens: `t[i, 1 : truncate_len + 1].mean(0)` -- this starts at position 1 (skipping BOS at 0) and goes to `truncate_len` (before EOS).
**How to avoid:** After getting `last_hidden_state`, extract positions `1:-1` (or `1:seq_len-1` based on actual non-padding length) for mean pooling to exclude BOS (position 0) and EOS (last non-padding position).
**Warning signs:** Slightly different embedding values compared to original ESM2, which could affect training accuracy.

### Pitfall 6: features_extract.py Multiprocessing Update
**What goes wrong:** The existing `features_extract.py` uses `multiprocessing.Pool(processes=2)` for `process_file_pro()`. If left unchanged, it will try to call `extract_fast_esm()` from forked processes that cannot access the GPU.
**Why it happens:** The file was designed for the old CPU-capable ESM2 extraction with fair-esm.
**How to avoid:** Update `features_extract.py` to process protein files sequentially (single process with GPU batching) instead of using multiprocessing.Pool. The `extract_fast_esm()` function handles internal batching efficiently.
**Warning signs:** CUDA fork errors when running `features_extract.py`.

## Code Examples

Verified patterns from the codebase and model source code:

### Complete extract_fast_esm() Function Structure
```python
# Source: Synthesized from modeling_fastesm.py, original extract_esm() (git:0b7e6f1), merge_data()
import torch
import logging
from Bio import SeqIO
from tqdm import tqdm

logger = logging.getLogger(__name__)

def extract_fast_esm(
    fasta_file,
    out_file=None,
    model=None,
    tokenizer=None,
    truncation_seq_length=1024,
    toks_per_batch=2048,
):
    """Extract 1280-dim protein embeddings using FastESM2_650.

    Produces output in same format as original extract_esm():
        {'proteins': [labels], 'data': [tensor_embeddings]}

    Args:
        fasta_file: Path to input FASTA file
        out_file: Path to output .pt file (None to return without saving)
        model: Pre-loaded FastESM2_650 model on GPU
        tokenizer: Model tokenizer (model.tokenizer)
        truncation_seq_length: Max residues per sequence (default 1024)
        toks_per_batch: Max tokens per batch (default 2048)

    Returns:
        (proteins, data) tuple of labels and embedding tensors
    """
    # Skip if output already exists (resume capability)
    if out_file is not None and os.path.exists(out_file):
        obj = torch.load(out_file, weights_only=False)
        return obj['proteins'], obj['data']

    # Read sequences from FASTA
    records = list(SeqIO.parse(fasta_file, 'fasta'))
    labels = [record.id for record in records]
    sequences = [str(record.seq) for record in records]

    # Compute sequence lengths for dynamic batching
    seq_lengths = [min(len(seq), truncation_seq_length) for seq in sequences]

    # Group into batches by total token count
    batches = get_batch_indices(seq_lengths, toks_per_batch)

    proteins = []
    data = []

    with torch.no_grad():
        for batch_indices in tqdm(batches, desc=f"Extracting {fasta_file}"):
            batch_labels = [labels[i] for i in batch_indices]
            batch_seqs = [sequences[i] for i in batch_indices]

            # Tokenize with truncation
            tokenized = tokenizer(
                batch_seqs,
                return_tensors='pt',
                padding='longest',
                truncation=True,
                max_length=truncation_seq_length + 2  # +2 for BOS/EOS
            )
            input_ids = tokenized['input_ids'].to(model.device)
            attention_mask = tokenized['attention_mask'].to(model.device)

            # Forward pass (SDPA by default)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden = outputs.last_hidden_state  # (batch, seq_len, 1280)

            # Mean pooling excluding BOS (pos 0) and EOS (last non-pad pos)
            for i, label in enumerate(batch_labels):
                seq_len = min(len(batch_seqs[i]), truncation_seq_length)
                # Positions 1 to seq_len (inclusive) are actual residues
                representation = last_hidden[i, 1:seq_len + 1].mean(0)
                embedding = representation.float().cpu()  # fp16 -> fp32

                proteins.append(label)
                data.append(embedding)

    # Validate embeddings
    assert len(proteins) == len(records), \
        f"Count mismatch: {len(proteins)} embeddings for {len(records)} sequences"
    for emb in data:
        assert emb.shape == (1280,), f"Wrong dimension: {emb.shape}"
        assert not torch.isnan(emb).any(), "NaN in embedding"
        assert not torch.isinf(emb).any(), "Inf in embedding"

    # Save
    if out_file is not None:
        torch.save({'proteins': proteins, 'data': data}, out_file)

    return proteins, data
```

### Model Loading at Module Level (features_extract.py pattern)
```python
# Source: Based on existing features_extract.py pattern (lines 9-14) + Phase 1 validation script
import torch
from transformers import AutoModel

# Load FastESM2_650 once at module level
FastESM_model = AutoModel.from_pretrained(
    "Synthyra/FastESM2_650",
    trust_remote_code=True,
    torch_dtype=torch.float16
).eval().cuda()
FastESM_tokenizer = FastESM_model.tokenizer

def process_file_pro(file):
    """Process a single protein FASTA file for embedding extraction."""
    output_file = f'{file.split(".fa")[0]}_ESM.pt'
    merged_file_path = output_file.replace('./data/', './data/data_merge/') \
        .replace('.identified_protein', '_merged') \
        .replace('ESM', 'merged')

    if os.path.exists(output_file) or os.path.exists(merged_file_path):
        return output_file

    extract_fast_esm(
        fasta_file=file,
        out_file=output_file,
        model=FastESM_model,
        tokenizer=FastESM_tokenizer
    )
    print(f'saved to: {output_file}')
    return output_file
```

### merge_data() Consumption (Existing Code - UNCHANGED)
```python
# Source: units.py lines 236-269 (current codebase)
# This is what extract_fast_esm() output must be compatible with:
def merge_data(DNABERT_S_data, ESM_data, merged_file, data_type=None):
    ESM_outfile = torch.load(ESM_data)

    protein_data_dict = {}
    for protein, data in zip(ESM_outfile['proteins'], ESM_outfile['data']):
        protein_data_dict[protein] = data  # data must be a tensor, shape (1280,)

    # ... merges with DNABERT_S data via torch.cat(..., dim=-1)
    # Result shape: 768 (DNABERT_S) + 1280 (FastESM2) = 2048
```

### Embedding Validation Pattern
```python
# Source: User decision from CONTEXT.md
def validate_embeddings(proteins, data, expected_dim=1280):
    """Validate extracted embeddings before saving."""
    errors = []

    # Dimension check
    for i, emb in enumerate(data):
        if emb.shape != (expected_dim,):
            errors.append(f"Seq {proteins[i]}: dim {emb.shape} != ({expected_dim},)")

    # NaN/Inf detection
    for i, emb in enumerate(data):
        if torch.isnan(emb).any():
            errors.append(f"Seq {proteins[i]}: contains NaN")
        if torch.isinf(emb).any():
            errors.append(f"Seq {proteins[i]}: contains Inf")

    # Count match (caller checks against input)
    return errors
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `esm.pretrained.load_model_and_alphabet('esm2_t36_3B_UR50D')` | `AutoModel.from_pretrained('Synthyra/FastESM2_650', trust_remote_code=True)` | Phase 1 | Standard HuggingFace API, automatic SDPA, 5x smaller model |
| `esm.FastaBatchedDataset.from_file(fasta)` | `SeqIO.parse()` + manual tokenization batching | Phase 2 | Remove fair-esm dependency entirely |
| `alphabet.get_batch_converter()` | `tokenizer(seqs, padding='longest', truncation=True)` | Phase 2 | Standard HuggingFace tokenizer API |
| `out["representations"][36]` (layer 36 of ESM2-3B) | `outputs.last_hidden_state` (last layer of FastESM2_650) | Phase 2 | FastESM2 outputs last hidden state directly via HuggingFace API |
| 2560-dim embeddings | 1280-dim embeddings | Phase 2 | Requires downstream dimension updates in Phase 3 |
| `multiprocessing.Pool(processes=2)` for protein extraction | Sequential processing with GPU batching | Phase 2 | CUDA contexts do not fork safely |

**Deprecated/outdated:**
- `extract_esm()`: Already raises NotImplementedError (Phase 1). Replaced by `extract_fast_esm()`.
- `ESM_model, ESM_alphabet` globals in `features_extract.py`: Set to None (Phase 1). Replace with FastESM2_650 model and tokenizer.
- `multiprocessing.Pool(processes=2)` for protein files: Incompatible with GPU model. Replace with sequential loop.

## Critical Technical Details

### Output Format Compatibility Matrix

The exact data flow from extraction through merge to training:

```
extract_fast_esm() output:
  {'proteins': ['seqid1', 'seqid2', ...],
   'data': [tensor(1280), tensor(1280), ...]}
      |
      v
merge_data() reads:
  ESM_outfile['proteins'] -> list of string labels
  ESM_outfile['data'] -> list of tensors (indexed by position)
  protein_data_dict[protein] = data  # tensor must be 1D
      |
      v
merge_data() output:
  merged_feature = torch.cat((nucleotide_data, protein_data), dim=-1)
  # Shape: (768 + 1280) = (2048,) <-- was (768 + 2560) = (3328,)
      |
      v
train.py reads:
  data_dict['data'][:, :]  # expects 2D tensor
  input_dim = 3328  # MUST UPDATE IN PHASE 3 to 2048
```

### FastESM2_650 Model Architecture (from config.json and modeling_fastesm.py)

| Property | Value | Source |
|----------|-------|--------|
| hidden_size | 1280 | config.json line 19 |
| num_hidden_layers | 33 | config.json line 28 |
| num_attention_heads | 20 | config.json line 27 |
| max_position_embeddings | 1026 | config.json line 26 |
| vocab_size | 33 | config.json line 36 |
| pad_token_id | 1 | config.json line 29 |
| position_embedding_type | rotary | config.json line 30 |
| attention implementation | SDPA default | modeling_fastesm.py line 331 |
| tokenizer | EsmTokenizer from facebook/esm2_t6_8M_UR50D | modeling_fastesm.py line 807 |

### Mean Pooling: Original ESM2 vs FastESM2 Implementation

**Original ESM2 (from git history):**
```python
truncate_len = min(truncation_seq_length, len(strs[i]))
result["mean_representations"] = {
    layer: t[i, 1 : truncate_len + 1].mean(0).clone().to('cpu')
    for layer, t in representations.items()
}
# Key: starts at position 1 (skips BOS), goes to truncate_len+1 (excludes EOS)
# Uses layer 36 representation: data.append(result["mean_representations"][36])
```

**FastESM2 equivalent:**
```python
seq_len = min(len(sequence), truncation_seq_length)
# outputs.last_hidden_state[i, 1:seq_len+1] matches the original range
embedding = outputs.last_hidden_state[i, 1:seq_len+1].mean(0).float().cpu()
# Key: same position range, but from last_hidden_state instead of layer dict
```

### Parallel Processing Recommendation (Claude's Discretion)

Given GPU constraints, the recommended approach is:

**Sequential file processing with GPU-batched sequence extraction:**
- Do NOT use `multiprocessing.Pool` for `process_file_pro()` -- CUDA contexts are not fork-safe
- Process files in a simple for-loop
- Within each file, use dynamic batching (`toks_per_batch=2048`) for efficient GPU utilization
- The GPU batch processing within `extract_fast_esm()` IS the parallelism -- multiple sequences are processed simultaneously on the GPU within each batch
- Use tqdm for file-level progress tracking

**Rationale:**
- The original code used `multiprocessing.Pool(processes=2)` because the ESM2-3B model was loaded independently in each worker via fair-esm
- With FastESM2_650 loaded once in GPU memory as a module-level global, forking processes would either fail (CUDA fork error) or require each worker to load its own model copy (wasting GPU memory)
- GPU batch processing (multiple sequences per forward pass) provides far better throughput than process-level parallelism for GPU workloads

### Retry Logic Recommendation (Claude's Discretion)

For transient GPU errors (e.g., CUDA OOM on a single large batch):

```python
# Simple retry with smaller batch on OOM
MAX_RETRIES = 3

def safe_forward(model, input_ids, attention_mask):
    for attempt in range(MAX_RETRIES):
        try:
            return model(input_ids=input_ids, attention_mask=attention_mask)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if attempt < MAX_RETRIES - 1:
                logger.warning(f"OOM on batch size {input_ids.size(0)}, "
                             f"retrying with smaller batch (attempt {attempt+1})")
                # Process sequences one at a time as fallback
                if input_ids.size(0) == 1:
                    raise  # Cannot reduce further
                # Split batch in half and process each half
                mid = input_ids.size(0) // 2
                out1 = safe_forward(model, input_ids[:mid], attention_mask[:mid])
                out2 = safe_forward(model, input_ids[mid:], attention_mask[mid:])
                # Combine results
                combined_hidden = torch.cat([out1.last_hidden_state, out2.last_hidden_state], dim=0)
                return type(out1)(last_hidden_state=combined_hidden)
            raise
```

### Failure Log Format Recommendation (Claude's Discretion)

```
# extraction_failures.log format:
# Timestamp | File | Sequence ID | Error Type | Error Message
2026-02-07 14:30:00 | data/viral/output_1.fa | seq_123 | ValueError | Embedding dim 0, expected 1280
2026-02-07 14:30:05 | data/viral/output_2.fa | ALL | RuntimeError | CUDA OOM after 3 retries
```

## Open Questions

Things that couldn't be fully resolved:

1. **BOS/EOS token handling precision**
   - What we know: Original ESM2 used `t[i, 1 : truncate_len + 1].mean(0)` to exclude BOS at position 0 and stop before EOS. FastESM2 uses the same ESM tokenizer, so token layout should be identical.
   - What's unclear: Whether FastESM2's embeddings at the BOS/EOS positions differ meaningfully from ESM2-3B's, and whether excluding them has the same effect. The model architectures differ (RoPE vs absolute position embeddings, different layer counts).
   - Recommendation: Match the original approach exactly (exclude BOS/EOS in mean pooling) for fair comparison. This is verifiable by checking embedding dimensions and values.

2. **GPU memory under batch processing load**
   - What we know: FastESM2_650 in fp16 needs ~1.3GB for weights. With `toks_per_batch=2048`, a single batch with 1024-length sequences would be 2 sequences. Activation memory for 2 sequences of 1024 tokens in fp16 with 33 layers is approximately 2 * 1024 * 1280 * 2 bytes * 33 layers = ~175MB. Total ~1.5GB.
   - What's unclear: Exact GPU memory capacity on GB10 (nvidia-smi shows "Not Supported" for memory info). Phase 1 validated that single-sequence inference works.
   - Recommendation: The `toks_per_batch=2048` limit should prevent OOM. Include the OOM retry logic as safety net. If GB10 has issues, reduce `toks_per_batch`.

3. **Exact numerical equivalence to original ESM2 mean pooling**
   - What we know: The original ESM2 code used `representations[36]` (the 36th layer) with `.clone().to('cpu')`. FastESM2 uses `outputs.last_hidden_state` which is the output of the final encoder layer with layer norm applied. Both approaches get the mean of the last layer's hidden states.
   - What's unclear: Whether the layer norm after the encoder in FastESM2 (line 484-485 in modeling_fastesm.py: `hidden_states = self.emb_layer_norm_after(hidden_states)`) changes the comparison. The original ESM2-3B may have had a different normalization scheme.
   - Recommendation: This is not a concern for functional correctness since the entire model is different (650M vs 3B, different training). The goal is correct extraction from FastESM2, not replication of ESM2-3B values.

## Sources

### Primary (HIGH confidence)
- `modeling_fastesm.py` (local cache: `~/.cache/huggingface/hub/models--Synthyra--FastESM2_650/`) - Complete model source code including tokenizer, forward pass, attention, pooling, and embed_dataset API
- `config.json` (local cache) - Model configuration (hidden_size=1280, num_hidden_layers=33, vocab_size=33, max_position_embeddings=1026)
- `units.py` (project codebase) - Current state with deprecated extract_esm() and active extract_DNABERT_S(), merge_data()
- `git show 0b7e6f1:units.py` - Original extract_esm() implementation showing exact output format and mean pooling approach
- `features_extract.py` (project codebase) - Orchestration code showing model loading pattern and multiprocessing usage
- `train.py` (project codebase) - Downstream consumer showing input_dim=3328 and data loading pattern
- `prediction.py` (project codebase) - Inference pipeline showing merge_data() and prediction flow
- Phase 1 RESEARCH.md - Environment details, FastESM2 model specifications, SDPA behavior

### Secondary (MEDIUM confidence)
- Phase 1 01-02-SUMMARY.md - Docker workflow, GB10 compatibility, PyTorch 2.9.0a0 details

### Tertiary (LOW confidence)
- GPU memory estimates for GB10 - Calculated from model parameters, but actual GPU memory capacity unknown

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - All libraries verified in Phase 1, model source code inspected directly
- Architecture: HIGH - Based on direct analysis of model source code and existing codebase patterns
- Output format: HIGH - Verified by reading merge_data() consumption pattern and original extract_esm() output
- Mean pooling approach: HIGH - Original code preserved in git history, FastESM2 equivalent verified in model source
- Pitfalls: HIGH - Identified from actual code analysis (CUDA fork, attention mask, fp16/fp32, BOS/EOS)
- Parallel processing: HIGH - CUDA fork limitations are well-established; batch GPU processing is standard practice

**Research date:** 2026-02-07
**Valid until:** 2026-03-07 (30 days -- model and libraries are stable)
