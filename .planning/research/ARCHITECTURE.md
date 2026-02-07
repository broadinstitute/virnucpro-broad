# Architecture Research: FastESM2 Integration

**Domain:** Viral sequence classification pipeline
**Researched:** 2026-02-07
**Confidence:** HIGH

## Current Architecture Overview

VirNucPro uses a dual-embedding approach combining nucleotide and protein representations for viral sequence classification.

```
Input FASTA
    ↓
┌─────────────────────────────────────────────────────────────┐
│                    PREPROCESSING LAYER                       │
├─────────────────────────────────────────────────────────────┤
│  units.py:split_fasta_chunk() → Chunk sequences             │
│  units.py:identify_seq() → 6-frame translation              │
│  units.py:translate_dna() → Generate protein sequences      │
└─────────────────────────────────────────────────────────────┘
         ↓                              ↓
  nucleotide.fa                    protein.faa
         ↓                              ↓
┌─────────────────────────────────────────────────────────────┐
│                   FEATURE EXTRACTION LAYER                   │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────────┐             ┌──────────────────┐      │
│  │ DNABERT-S        │             │ ESM2_t36_3B      │      │
│  │ (DNA embeddings) │             │ (Protein embed)  │      │
│  │ → 384 dims       │             │ → 2560 dims      │      │
│  └──────────────────┘             └──────────────────┘      │
│         ↓                              ↓                     │
│    units.py:                      units.py:                 │
│    extract_DNABERT_S()            extract_esm()             │
│    (8 processes)                  (2 processes)             │
├─────────────────────────────────────────────────────────────┤
│                     MERGING LAYER                            │
│  units.py:merge_data() → Concatenate embeddings             │
│  → 2944 dimensions (384 + 2560)                              │
└─────────────────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────────────────┐
│                   CLASSIFICATION LAYER                       │
├─────────────────────────────────────────────────────────────┤
│  train.py:MLPClassifier                                     │
│  Input: 2944 → Hidden: 512 → Output: 2                      │
│  (BatchNorm, ReLU, Dropout 0.5)                              │
└─────────────────────────────────────────────────────────────┘
         ↓
   Virus / Non-virus
```

## Component Boundaries

| Component | Responsibility | Current Implementation | Integration Surface |
|-----------|----------------|------------------------|---------------------|
| **Preprocessing** | Sequence chunking, 6-frame translation | units.py functions | Independent - no changes needed |
| **DNA Feature Extraction** | DNABERT-S tokenization and embedding | units.py:extract_DNABERT_S() | Independent - no changes needed |
| **Protein Feature Extraction** | ESM2 tokenization and embedding | units.py:extract_esm() | **PRIMARY INTEGRATION POINT** |
| **Feature Merging** | Concatenate DNA + protein embeddings | units.py:merge_data() | **DIMENSION COMPATIBILITY CHECK** |
| **Model Training** | MLP classifier training | train.py:MLPClassifier | **INPUT DIMENSION UPDATE** |
| **Inference Pipeline** | End-to-end prediction | prediction.py | **ALL DOWNSTREAM CONSUMERS** |

## FastESM2_650 Integration Points

### Integration Point 1: Model Loading (units.py:extract_esm)

**Current Code (lines 204-217):**
```python
def extract_esm(fasta_file,
                model_location='esm2_t36_3B_UR50D',
                truncation_seq_length=1024, toks_per_batch=2048,
                out_file=None, model_loaded = False, model = None, alphabet = None):
    if model_loaded == False:
        model, alphabet = pretrained.load_model_and_alphabet(model_location)
    else:
        model = model
        alphabet = alphabet
```

**Required Changes:**
- Replace `from esm import pretrained` with `from transformers import AutoTokenizer, AutoModel`
- Change model loading from `pretrained.load_model_and_alphabet()` to `AutoModel.from_pretrained("Synthyra/FastESM2_650")`
- Update tokenizer from `alphabet` to `AutoTokenizer.from_pretrained("Synthyra/FastESM2_650")`

**Confidence:** HIGH - FastESM2_650 uses standard Hugging Face transformers API

### Integration Point 2: Tokenization and Batching (units.py:extract_esm)

**Current Code (lines 222-228):**
```python
dataset = FastaBatchedDataset.from_file(fasta_file)
batches = dataset.get_batch_indices(toks_per_batch, extra_toks_per_seq=1)
data_loader = torch.utils.data.DataLoader(
    dataset, collate_fn=alphabet.get_batch_converter(), batch_sampler=batches
)
```

**Required Changes:**
- Replace `FastaBatchedDataset` with custom DataLoader using transformers tokenizer
- Remove `alphabet.get_batch_converter()` and use standard tokenizer batching
- Maintain batch size logic for memory efficiency

**Confidence:** MEDIUM - Custom batching may need testing for optimal throughput

### Integration Point 3: Embedding Extraction (units.py:extract_esm)

**Current Code (lines 238-261):**
```python
with torch.no_grad():
    for batch_idx, (labels, strs, toks) in enumerate(data_loader):
        if device:
            toks = toks.to(device, non_blocking=True)
        out = model(toks, repr_layers=repr_layers, return_contacts=return_contacts)
        representations = {
            layer: t.to(device) for layer, t in out["representations"].items()
        }
        for i, label in enumerate(labels):
            result = {"label": label, "mean_representations": {}}
            truncate_len = min(truncation_seq_length, len(strs[i]))
            result["mean_representations"] = {
                layer: t[i, 1 : truncate_len + 1].mean(0).clone().to('cpu')
                for layer, t in representations.items()
            }
            proteins.append(label)
            data.append(result["mean_representations"][36])
```

**Required Changes:**
- Update model forward pass to transformers API: `outputs = model(**inputs)`
- Extract hidden states: `hidden_states = outputs.last_hidden_state`
- Mean pooling: `embedding_mean = torch.mean(hidden_states[:, 1:-1, :], dim=1)` (exclude CLS/SEP tokens)
- Remove layer-specific extraction (layer 36) - FastESM2_650 uses final layer embeddings

**Critical Dimension Question:**
- ESM2_t36_3B outputs **2560 dimensions** (layer 36)
- FastESM2_650 is based on ESM2-650M which outputs **1280 dimensions**
- **This will break dimension compatibility!**

**Confidence:** HIGH - Dimension mismatch is a critical blocker

### Integration Point 4: Dimension Compatibility (units.py:merge_data & train.py:MLPClassifier)

**Current Architecture:**
- DNA embeddings: 384 dimensions (DNABERT-S)
- Protein embeddings: 2560 dimensions (ESM2_t36_3B layer 36)
- Total: 2944 dimensions → MLP input

**FastESM2_650 Architecture:**
- DNA embeddings: 384 dimensions (DNABERT-S) - **unchanged**
- Protein embeddings: 1280 dimensions (FastESM2_650) - **changed**
- Total: 1664 dimensions → MLP input

**Required Changes:**
```python
# train.py line 105
input_dim = 3328  # OLD: 384 + 2944 (incorrect in current code, should be 2944)
input_dim = 1664  # NEW: 384 + 1280
```

**Confidence:** HIGH - This is a straightforward update but requires model retraining

## Recommended Architecture Changes

### Option A: Drop-in Replacement (Requires Retraining)

**Approach:** Replace ESM2_t36_3B with FastESM2_650 completely

**Pros:**
- Faster inference (5-10x speedup based on FastPLMs benchmarks)
- Lower memory footprint (650M vs 3B parameters)
- Simpler deployment (standard Hugging Face API)

**Cons:**
- Requires full model retraining
- Different embedding dimensions (1280 vs 2560)
- Potential accuracy change (smaller model)

**Build Order:**
1. Create test harness for FastESM2_650 embedding extraction
2. Update units.py:extract_esm() for FastESM2_650 API
3. Update merge_data() dimension logic (no code change, just dimension)
4. Update train.py:MLPClassifier input_dim to 1664
5. Retrain model with new embeddings
6. Update prediction.py to use new model

### Option B: Side-by-Side Comparison (Phased Migration)

**Approach:** Add FastESM2_650 as alternative, compare performance before switching

**Pros:**
- De-risks migration with A/B testing
- Allows accuracy validation before full commitment
- Can fall back to ESM2_t36_3B if accuracy drops

**Cons:**
- More complex implementation
- Requires maintaining two code paths temporarily
- Longer migration timeline

**Build Order:**
1. Add FastESM2_650 extraction as separate function (extract_fast_esm)
2. Create dual-model feature extraction pipeline
3. Train two separate MLPClassifier models (2944-dim and 1664-dim)
4. Benchmark accuracy and speed
5. Choose winner and deprecate loser

### Option C: Ensemble Approach (Maximum Accuracy)

**Approach:** Use both models and combine predictions

**Pros:**
- Potentially higher accuracy (ensemble effect)
- Leverages both large model (accuracy) and fast model (speed)

**Cons:**
- Highest computational cost
- Most complex architecture
- Overkill for simple binary classification

**Not Recommended:** Adds complexity without clear benefit for this use case

## Data Flow: FastESM2_650 Integration

### Current Flow (ESM2_t36_3B)
```
protein.faa
    ↓
FastaBatchedDataset.from_file()
    ↓
alphabet.get_batch_converter()
    ↓
model(toks, repr_layers=[36])
    ↓
representations[36][i, 1:truncate_len+1].mean(0)
    ↓
2560-dim tensor
```

### Proposed Flow (FastESM2_650)
```
protein.faa
    ↓
SeqIO.parse() → Custom batching
    ↓
tokenizer(sequences, return_tensors='pt', padding=True)
    ↓
model(**inputs)
    ↓
outputs.last_hidden_state[:, 1:-1, :].mean(1)
    ↓
1280-dim tensor
```

## Parallelization Strategy

**Current Approach:**
- features_extract.py uses multiprocessing.Pool
- DNA extraction: 8 processes
- Protein extraction: 2 processes

**Rationale for 2 processes (ESM2_t36_3B):**
- Large model (3B parameters) → high GPU memory consumption
- Limited by GPU memory, not CPU cores

**FastESM2_650 Opportunity:**
- Smaller model (650M parameters) → lower GPU memory
- Could potentially increase to 4-6 processes
- Faster per-sequence inference → higher throughput

**Recommendation:**
- Start with 2 processes (same as current)
- Benchmark GPU memory usage
- Incrementally increase if memory allows
- Monitor for GPU memory bottleneck vs CPU bottleneck

## Build Order Recommendation

Based on dependency analysis and risk mitigation:

### Phase 1: Test Harness (Low Risk)
**Goal:** Validate FastESM2_650 works as expected

1. Create standalone test script
2. Load FastESM2_650 from Hugging Face
3. Extract embeddings from sample proteins
4. Verify output dimensions (1280)
5. Benchmark inference speed vs ESM2_t36_3B

**Deliverable:** Confidence in FastESM2_650 functionality

### Phase 2: Feature Extraction Update (Medium Risk)
**Goal:** Integrate FastESM2_650 into units.py

1. Create new function: `extract_fast_esm()`
2. Implement tokenizer-based batching
3. Implement embedding extraction with mean pooling
4. Test on small dataset (100 sequences)
5. Compare output format with extract_esm()

**Deliverable:** Drop-in replacement for extract_esm()

### Phase 3: Dimension Update (Low Risk)
**Goal:** Update downstream consumers for 1664-dim input

1. Update train.py:MLPClassifier input_dim
2. Update merge_data() (no code change, just dimensions)
3. Verify tensor shapes throughout pipeline

**Deliverable:** Architecture ready for retraining

### Phase 4: Model Retraining (High Risk)
**Goal:** Train new classifier with FastESM2_650 embeddings

1. Re-extract features for training dataset
2. Train MLPClassifier with new embeddings
3. Validate accuracy on test set
4. Compare metrics with original model
5. Benchmark inference speed improvement

**Deliverable:** New model.pth with FastESM2_650 embeddings

### Phase 5: Inference Pipeline (Medium Risk)
**Goal:** Deploy FastESM2_650 to prediction.py

1. Update prediction.py imports
2. Update model loading to FastESM2_650
3. Test end-to-end prediction
4. Validate output format unchanged

**Deliverable:** Production-ready prediction pipeline

## Anti-Patterns to Avoid

### Anti-Pattern 1: Dimension Mismatch Silently Failing

**What people do:** Change model without updating input_dim, rely on PyTorch to error
**Why it's wrong:** Error happens late in training, wasting compute time
**Do this instead:** Add dimension assertion at merge_data() output

```python
# In merge_data() after concatenation
assert merged_feature.shape[-1] == expected_dim, \
    f"Expected {expected_dim} dims, got {merged_feature.shape[-1]}"
```

### Anti-Pattern 2: Hardcoding Model Paths

**What people do:** Hardcode "esm2_t36_3B_UR50D" string throughout codebase
**Why it's wrong:** Makes switching models error-prone (missed updates)
**Do this instead:** Use configuration file or constants

```python
# At top of units.py
PROTEIN_MODEL_NAME = "Synthyra/FastESM2_650"
PROTEIN_EMBEDDING_DIM = 1280

# Later in code
model = AutoModel.from_pretrained(PROTEIN_MODEL_NAME)
```

### Anti-Pattern 3: Not Validating Embedding Quality

**What people do:** Switch models, retrain, deploy without comparison
**Why it's wrong:** May lose accuracy without realizing it
**Do this instead:** Compare embeddings before full retraining

```python
# Extract embeddings with both models for same sequences
# Compare cosine similarity
# Ensure embeddings are capturing similar information
```

### Anti-Pattern 4: Batch Size Mismatch

**What people do:** Use same batch size as ESM2_t36_3B
**Why it's wrong:** Smaller model can handle larger batches → faster throughput
**Do this instead:** Tune batch size specifically for FastESM2_650

```python
# Benchmark different batch sizes
# Find optimal throughput vs memory tradeoff
# Update toks_per_batch parameter accordingly
```

## Integration Complexity Matrix

| Component | Complexity | Risk | Effort | Dependencies |
|-----------|------------|------|--------|--------------|
| Test harness | Low | Low | 1 day | None |
| extract_fast_esm() | Medium | Medium | 2-3 days | Test harness |
| Dimension updates | Low | Low | 0.5 day | extract_fast_esm() |
| Feature re-extraction | Medium | Low | 1-2 days | extract_fast_esm() |
| Model retraining | Medium | High | 3-5 days | Feature re-extraction |
| Prediction pipeline | Low | Medium | 1 day | Model retraining |

**Total Estimated Effort:** 8-12 days for full migration

## Sources

- [Synthyra/FastESM2_650 Hugging Face Model](https://huggingface.co/Synthyra/FastESM2_650)
- [Synthyra/FastPLMs GitHub Repository](https://github.com/Synthyra/FastPLMs)
- [Facebook Research ESM Repository](https://github.com/facebookresearch/esm)
- [Hugging Face Transformers ESM Documentation](https://huggingface.co/docs/transformers/en/model_doc/esm)
- [Medium-sized protein language models perform well at transfer learning on realistic datasets - Nature Scientific Reports](https://www.nature.com/articles/s41598-025-05674-x)
- Current VirNucPro codebase analysis (units.py, features_extract.py, train.py, prediction.py)

---
*Architecture research for: FastESM2_650 migration to VirNucPro pipeline*
*Researched: 2026-02-07*
