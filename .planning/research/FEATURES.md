# Feature Research: FastESM2_650 Migration

**Domain:** Protein embedding extraction for viral classification
**Researched:** 2026-02-07
**Confidence:** MEDIUM

## Feature Landscape

### Table Stakes (Users Expect These)

Features users assume exist. Missing these = migration fails.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Protein embedding extraction | Core functionality - replaces current ESM2 3B | MEDIUM | Must produce embeddings that can be concatenated with DNABERT-S (384-dim) |
| Batch processing support | Current system processes 10,000 sequences per file | LOW | FastESM2 uses HuggingFace transformers AutoModel with native batch support |
| Mean-pooled embeddings | Current extract_esm() returns mean of token embeddings | LOW | FastESM2 provides .last_hidden_state, requires manual mean pooling |
| GPU acceleration | Current system uses CUDA, critical for performance | LOW | Same as ESM2, uses torch.device("cuda") |
| Compatible output dimensions | Downstream MLP expects specific input size | HIGH | CRITICAL: ESM2 3B = 2560-dim, FastESM2_650 = 1280-dim (different!) |
| Tokenization handling | Protein sequences need proper tokenization | MEDIUM | FastESM2 includes model.tokenizer vs ESM2 uses alphabet.get_batch_converter() |
| Truncation for long sequences | Current system truncates to 1024 tokens | MEDIUM | Must handle sequences >1024 residues with FastESM2 tokenizer |
| File-based caching | Current system saves embeddings as .pt files | LOW | Same torch.save() mechanism works |

### Differentiators (Competitive Advantage)

Features that set FastESM2 apart. Not required, but valuable.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| 2x faster inference | FastESM2 is over twice as fast on longer sequences | LOW | Drop-in benefit from PyTorch 2.5+ SDPA optimization |
| Lower memory footprint | 650M params vs 3B params = ~4.6x smaller | LOW | Enables larger batch sizes or smaller GPU requirements |
| HuggingFace ecosystem | Standard transformers API vs custom ESM library | MEDIUM | Better integration with ecosystem tools, model sharing |
| Similar downstream performance | 650M performs comparably to 3B on many tasks | MEDIUM | Research shows medium models sufficient for <10^4 observations |
| Simpler tokenization | Built-in tokenizer vs custom alphabet converter | LOW | model.tokenizer vs pretrained.load_model_and_alphabet() |
| Model versioning | HuggingFace hub provides versioning, model cards | LOW | Easier to track model versions and provenance |
| Trust remote code support | Access to latest optimizations via trust_remote_code=True | LOW | Required for FastESM2, enables community contributions |

### Anti-Features (Commonly Requested, Often Problematic)

Features that seem good but create problems.

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|-----------------|-------------|
| Keep exact same embedding dimensions | Minimal changes to downstream code | Defeats purpose - can't use 650M model, forces 3B model | Retrain MLP classifier with new input size (1280+384=1664 vs 2560+384=2944) |
| Attention map extraction | Useful for interpretability | Not natively supported with SDPA (FastESM optimization), requires manual calculation | Only extract if needed, use output_attentions=True (slower) |
| Process all sequences individually | Simple implementation | Wastes GPU, slow | Use batch processing with padding |
| Keep using facebook/esm library | Familiar codebase | facebook/esm uses older PyTorch, misses optimizations | Migrate to HuggingFace transformers for FastESM2 |
| Match ESM2 3B exactly | Compatibility concerns | Requires using 3B model, loses speed benefit | Accept dimension change, retrain downstream model |
| Load model for each batch | Avoid GPU memory issues | Massive overhead from repeated loading | Load once, reuse for all batches |
| Use CPU for "safety" | Avoid CUDA errors | 60x slower (60s vs 1.1s for 242 sequences) | Fix CUDA issues, use GPU |

## Feature Dependencies

```
[Tokenizer Setup]
    └──requires──> [Model Loading]
                       └──requires──> [Embedding Extraction]
                                          └──requires──> [Dimension Alignment]
                                                             └──requires──> [MLP Classifier Update]

[Batch Processing] ──enhances──> [Embedding Extraction]

[GPU Acceleration] ──enhances──> [Embedding Extraction]

[Mean Pooling] ──requires──> [Embedding Extraction]

[2560-dim embeddings] ──conflicts with──> [1280-dim embeddings]
    (Cannot use both in same model - pick one embedding model size)

[PyTorch 2.5+] ──enables──> [2x Speed Improvement]
```

### Dependency Notes

- **Tokenizer Setup requires Model Loading:** FastESM2 tokenizer accessed via `model.tokenizer`, must load model first
- **Embedding Extraction requires Dimension Alignment:** Output is 1280-dim not 2560-dim, must account for in merge
- **Dimension Alignment requires MLP Classifier Update:** Downstream MLP input layer must change from 2944 to 1664
- **Batch Processing enhances Embedding Extraction:** GPU efficiency improves with batching (current: toks_per_batch=2048)
- **Mean Pooling requires Embedding Extraction:** FastESM2 outputs .last_hidden_state, must manually compute mean
- **PyTorch 2.5+ enables Speed Improvement:** SDPA optimization requires modern PyTorch version

## MVP Definition

### Launch With (v1)

Minimum viable migration - what's needed to validate FastESM2 works.

- [ ] FastESM2_650 model loading — Essential to replace ESM2 3B
- [ ] Tokenization with model.tokenizer — Different API from ESM2 alphabet
- [ ] Mean-pooled embedding extraction — Must match current extract_esm() output format
- [ ] 1280-dim output handling — Core dimension change
- [ ] Batch processing support — Current system processes 10K sequences/file
- [ ] GPU acceleration — Performance requirement
- [ ] .pt file saving — Must integrate with existing merge_data() workflow

### Add After Validation (v1.x)

Features to add once core migration is working.

- [ ] Performance benchmarking — Validate 2x speed claim on viral sequences
- [ ] MLP classifier retraining — Update input_dim from 2944 to 1664
- [ ] Accuracy comparison — Ensure classification performance maintained
- [ ] Memory profiling — Quantify memory savings vs ESM2 3B
- [ ] Error handling improvements — Handle edge cases in tokenization
- [ ] Batch size optimization — Find optimal batch size for GPU

### Future Consideration (v2+)

Features to defer until migration validated.

- [ ] Attention map extraction — Only if interpretability needed (slower with manual attention)
- [ ] Multiple model comparison — Test FastESM2 vs ESM2 3B vs ESM2 650M side-by-side
- [ ] Dynamic model selection — Allow switching between models based on use case
- [ ] Fine-tuning for virus classification — Potentially improve on zero-shot embeddings
- [ ] Mixed precision (FP16/BF16) — Further speed/memory optimization

## Feature Prioritization Matrix

| Feature | User Value | Implementation Cost | Priority |
|---------|------------|---------------------|----------|
| Model loading (AutoModel) | HIGH | LOW | P1 |
| Tokenization (model.tokenizer) | HIGH | LOW | P1 |
| Mean pooling extraction | HIGH | LOW | P1 |
| Batch processing | HIGH | LOW | P1 |
| GPU acceleration | HIGH | LOW | P1 |
| Dimension handling (1280) | HIGH | LOW | P1 |
| File caching (.pt) | HIGH | LOW | P1 |
| MLP input layer update | HIGH | MEDIUM | P1 |
| MLP retraining | HIGH | HIGH | P2 |
| Performance benchmarking | MEDIUM | LOW | P2 |
| Accuracy validation | HIGH | MEDIUM | P2 |
| Error handling | MEDIUM | MEDIUM | P2 |
| Batch size optimization | MEDIUM | LOW | P2 |
| Memory profiling | LOW | LOW | P3 |
| Attention map extraction | LOW | MEDIUM | P3 |
| Fine-tuning | LOW | HIGH | P3 |
| Mixed precision | MEDIUM | MEDIUM | P3 |

**Priority key:**
- P1: Must have for migration (blocking)
- P2: Should have, add immediately after core works
- P3: Nice to have, future consideration

## Migration-Specific Features

### Critical Differences: ESM2 3B vs FastESM2_650

| Aspect | ESM2 3B (Current) | FastESM2_650 (Target) | Migration Impact |
|--------|-------------------|----------------------|------------------|
| Library | facebook/esm (pretrained) | HuggingFace transformers (AutoModel) | HIGH - Different import structure |
| Model Loading | `pretrained.load_model_and_alphabet()` | `AutoModel.from_pretrained(..., trust_remote_code=True)` | MEDIUM - API change |
| Tokenizer | `alphabet.get_batch_converter()` | `model.tokenizer(sequences, padding=True)` | HIGH - Different tokenization flow |
| Embedding Dim | 2560 | 1280 | CRITICAL - Downstream impact |
| Merged Dim | 2944 (2560+384) | 1664 (1280+384) | CRITICAL - MLP input_dim change |
| Layers | 36 | 33 | LOW - Internal, no API impact |
| Parameters | 3B | 650M | LOW - Performance benefit |
| Speed | Baseline | 2x faster | HIGH - Performance benefit |
| Memory | ~12GB VRAM | ~3GB VRAM | MEDIUM - Enables larger batches |
| Output Access | `out["representations"][layer]` | `model(**inputs).last_hidden_state` | MEDIUM - Different attribute name |
| Batch Format | `DataLoader + collate_fn` | `tokenizer + DataLoader` | MEDIUM - Simpler with HF |
| PyTorch Version | 1.x compatible | Requires 2.5+ for full speed | MEDIUM - Environment dependency |
| Truncation | Manual (truncation_seq_length=1024) | Tokenizer param (max_length=1024) | LOW - Cleaner API |

### API Migration Examples

**Current ESM2 3B:**
```python
from esm import pretrained
model, alphabet = pretrained.load_model_and_alphabet('esm2_t36_3B_UR50D')
dataset = FastaBatchedDataset.from_file(fasta_file)
batches = dataset.get_batch_indices(toks_per_batch, extra_toks_per_seq=1)
data_loader = torch.utils.data.DataLoader(
    dataset, collate_fn=alphabet.get_batch_converter(), batch_sampler=batches
)
for batch_idx, (labels, strs, toks) in enumerate(data_loader):
    out = model(toks, repr_layers=[36], return_contacts=False)
    embeddings = out["representations"][36][i, 1:truncate_len+1].mean(0)
```

**Target FastESM2_650:**
```python
from transformers import AutoModel
model = AutoModel.from_pretrained('Synthyra/FastESM2_650',
                                   dtype=torch.float16,
                                   trust_remote_code=True).eval()
tokenizer = model.tokenizer
sequences = [str(record.seq) for record in SeqIO.parse(fasta_file, 'fasta')]
for i in range(0, len(sequences), batch_size):
    batch = sequences[i:i+batch_size]
    inputs = tokenizer(batch, padding=True, truncation=True,
                      max_length=1024, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs.to(device))
        embeddings = outputs.last_hidden_state.mean(dim=1)  # Mean pooling
```

### Dimension Impact Analysis

**Current Pipeline:**
1. DNA → DNABERT-S → 384-dim
2. Protein → ESM2 3B → 2560-dim
3. Concatenate → 2944-dim
4. MLP(input_dim=2944, hidden_dim=?, num_class=2)

**Migrated Pipeline:**
1. DNA → DNABERT-S → 384-dim (unchanged)
2. Protein → FastESM2_650 → 1280-dim (changed!)
3. Concatenate → 1664-dim (changed!)
4. MLP(input_dim=1664, hidden_dim=?, num_class=2) (must update!)

**MLP Layer Changes Required:**
- `self.hidden_layer = nn.Linear(input_dim, hidden_dim)` where input_dim changes 2944 → 1664
- Must retrain from scratch OR use transfer learning (reinitialize first layer, keep others)
- Training data unchanged, but model weights incompatible

## Implementation Notes

### PyTorch Version Requirement
- FastESM2 requires PyTorch 2.5+ for full SDPA optimization
- Current code likely uses older PyTorch (check requirements.txt)
- Update: `pip install torch>=2.5.0` or via pixi

### Trust Remote Code
- FastESM2 requires `trust_remote_code=True` in AutoModel.from_pretrained()
- This loads custom modeling code from HuggingFace hub
- Security consideration: Only use with trusted model sources (Synthyra is official)

### Tokenizer Access
- Tokenizer is accessed via `model.tokenizer`, not separately loaded
- Must load model before accessing tokenizer
- Tokenizer provides padding, truncation automatically

### Mean Pooling Implementation
- ESM2: Mean computed in extract_esm() as `t[i, 1:truncate_len+1].mean(0)`
- FastESM2: Must compute manually as `outputs.last_hidden_state.mean(dim=1)`
- Same mathematical operation, different API

### Batch Size Considerations
- Current: `toks_per_batch=2048` (token-based batching)
- FastESM2: Use sequence-based batching (e.g., batch_size=16)
- Smaller model → can increase batch size for same memory
- Optimal batch size depends on sequence lengths and GPU memory

### File Structure Compatibility
- Current: `torch.save({'proteins': proteins, 'data': data}, out_file)`
- Migration: Same structure, just data dimensions change
- merge_data() expects tensors, dimension change transparent to concat operation

## Downstream Impact Assessment

### Components Affected
1. **extract_esm() function** - HIGH impact, requires rewrite
2. **merge_data() function** - LOW impact, concat works with any dims
3. **MLPClassifier** - HIGH impact, input_dim parameter change
4. **Trained model weights** - CRITICAL impact, must retrain
5. **Prediction pipeline** - LOW impact, same interface
6. **DNABERT-S pipeline** - ZERO impact, unchanged

### Retraining Strategy
**Option 1: Full Retraining**
- Reinitialize MLP with input_dim=1664
- Retrain from scratch with viral/host dataset
- Pro: Clean slate, guaranteed compatibility
- Con: Requires full training run

**Option 2: Transfer Learning**
- Keep existing model architecture
- Reinitialize only first layer (2944→1664)
- Fine-tune with frozen early layers, then full fine-tune
- Pro: May converge faster, leverage learned features
- Con: Complex, may not work if embedding quality differs

**Recommendation:** Option 1 (Full Retraining) - Simpler, cleaner, more reliable

### Validation Requirements
- Accuracy must match or exceed current ESM2 3B performance
- Benchmark on viral classification test set
- Compare metrics: precision, recall, F1, ROC-AUC
- If accuracy drops significantly, may need ESM2 3B or fine-tuning

## Sources

**Model Information:**
- [ESM-2 650M Model Card](https://biolm.ai/models/esm2-650m/)
- [Synthyra/FastESM2_650 HuggingFace](https://huggingface.co/Synthyra/FastESM2_650)
- [ESM-2 NVIDIA BioNeMo Documentation](https://docs.nvidia.com/bionemo-framework/2.0/models/esm2/)

**Performance Research:**
- [Medium-sized protein language models perform well - Nature Scientific Reports 2025](https://www.nature.com/articles/s41598-025-05674-x)
- [ESM2_AMP framework - Oxford Academic Briefings in Bioinformatics 2025](https://academic.oup.com/bib/article/26/4/bbaf434/8242608)

**API Documentation:**
- [Synthyra/ESM2-650M HuggingFace](https://huggingface.co/Synthyra/ESM2-650M)
- [HuggingFace Transformers ESM Documentation](https://huggingface.co/docs/transformers/main/model_doc/esm)

**Technical Details:**
- [Protein sequence analyses with Transformers - Medium](https://medium.com/@moeinh77/protein-sequence-analyses-with-transformers-using-pytorch-and-huggingface-11a478bcc602)
- [Transfer Learning & Fine-tuning - Keras Documentation](https://keras.io/guides/transfer_learning/)

---
*Feature research for: FastESM2_650 Migration for Viral Classification*
*Researched: 2026-02-07*
*Confidence: MEDIUM (verified with official docs and recent research, but migration-specific impacts estimated)*
