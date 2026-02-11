# Modality Contribution Analysis

## Question

**Which features contribute more to viral classification: DNA (DNABERT-S) or protein (FastESM2-650M)?**

Does this change for short reads (150bp) vs full-length sequences?

---

## Current Gap in Knowledge

We've only tested **combined** features (768 DNA + 1280 protein = 2048-dim). We don't know:

1. How much does each modality contribute individually?
2. Is protein information useful for 150bp reads (only ~50 amino acids)?
3. Should we use DNA-only for short reads?

---

## Analysis Approach

Train three separate classifiers:

1. **DNA-only** (768-dim): DNABERT-S features alone
2. **Protein-only** (1280-dim): FastESM2-650M features alone
3. **Combined** (2048-dim): Both modalities (current approach)

Test each on:
- Full-length sequences (500-3000bp)
- Short reads (150bp)

---

## Workflow

### Step 1: Train Modality-Specific Classifiers

Train separate MLPs for each modality on full-length training data:

```bash
docker compose run --rm virnucpro python scripts/train_modality_classifiers.py \
  --epochs 20 \
  --batch-size 256 \
  --lr 1e-3
```

**Output:**
- `300_model_fastesm650_dna.pth` (DNA-only classifier)
- `300_model_fastesm650_protein.pth` (protein-only classifier)
- `300_model_fastesm650_combined.pth` (combined classifier)
- `reports/modality_comparison.json` (full-length results)

**Runtime:** ~30-60 minutes

**What this tells us:**
- Which modality is more informative on full-length sequences
- Whether combining them provides synergy or if one dominates

---

### Step 2: Evaluate on 150bp Test Set

Test all three models on the 150bp short read test set:

```bash
docker compose run --rm virnucpro python scripts/evaluate_modality_150bp.py
```

**Output:**
- `reports/modality_comparison_150bp.json`

**Runtime:** ~1 minute

**What this tells us:**
- Which modality degrades less on short reads
- Whether DNA-only might actually be BETTER for 150bp
- If protein features help at all with only ~50 amino acids

---

## Expected Outcomes

### Scenario A: Protein Dominates (Unlikely)

**Full-length:**
- DNA-only: 75% accuracy
- Protein-only: 88% accuracy
- Combined: 90% accuracy

**150bp:**
- DNA-only: 60% accuracy
- Protein-only: 50% accuracy
- Combined: 53% accuracy

**Interpretation:**
- Protein is key for full-length but loses discriminative power on short reads
- DNA features more robust to length variation

---

### Scenario B: DNA Dominates (Possible)

**Full-length:**
- DNA-only: 88% accuracy
- Protein-only: 75% accuracy
- Combined: 90% accuracy

**150bp:**
- DNA-only: 70% accuracy
- Protein-only: 45% accuracy
- Combined: 53% accuracy

**Interpretation:**
- DNA features are primary signal
- Protein adds refinement on full-length but confuses model on short reads
- **Recommendation:** Use DNA-only for <300bp reads

---

### Scenario C: Both Contribute Equally (Likely)

**Full-length:**
- DNA-only: 85% accuracy
- Protein-only: 85% accuracy
- Combined: 90% accuracy

**150bp:**
- DNA-only: 52% accuracy
- Protein-only: 51% accuracy
- Combined: 53% accuracy

**Interpretation:**
- Both modalities provide complementary information
- Both degrade equally on short reads
- Problem is fundamental: 150bp insufficient regardless of modality

---

### Scenario D: DNA Better for Short Reads (Very Possible)

**Full-length:**
- DNA-only: 85% accuracy
- Protein-only: 85% accuracy
- Combined: 90% accuracy

**150bp:**
- DNA-only: 68% accuracy
- Protein-only: 45% accuracy
- Combined: 53% accuracy

**Interpretation:**
- DNA k-mer patterns preserved in short reads
- Protein features from 50 amino acids are noisy
- Combining adds noise, degrading DNA-only performance
- **Recommendation:** Use DNA-only for <300bp, combined for ≥300bp

---

## Actionable Insights

### If DNA-only performs better on 150bp:

**Immediate Action:**
1. Deploy DNA-only classifier for sequences <300bp
2. Use combined classifier for sequences ≥300bp
3. Length-based routing in production

**Architecture:**
```python
def predict(sequence):
    if len(sequence) < 300:
        return dna_only_model(extract_dna_features(sequence))
    else:
        return combined_model(extract_dna_and_protein_features(sequence))
```

**Benefits:**
- Better short-read accuracy without retraining
- Simpler feature extraction for short reads (no protein translation)
- Faster (skip ESM2 embedding extraction)

---

### If protein adds value even on 150bp:

**Conclusion:**
- Keep combined approach
- Problem is training data, not modality choice
- Proceed with fine-tuning experiment

---

### If combined is worse than DNA-only on 150bp:

**Critical Finding:**
- Protein features are HARMFUL for short reads
- Model confusion from low-quality protein embeddings
- Architecture should be length-aware

**Redesign:**
```python
class LengthAwareClassifier:
    def __init__(self):
        self.dna_only = DNAClassifier(768)
        self.combined = CombinedClassifier(2048)
        self.length_threshold = 300

    def forward(self, dna_features, protein_features, seq_length):
        if seq_length < self.length_threshold:
            return self.dna_only(dna_features)
        else:
            combined = torch.cat([dna_features, protein_features])
            return self.combined(combined)
```

---

## Metrics to Track

For each modality on each dataset, measure:

1. **Accuracy** - Overall correctness
2. **Precision** - When it says viral, is it viral?
3. **Recall** - Does it catch viral sequences?
4. **F1 Score** - Balanced metric
5. **ROC-AUC** - Confidence calibration

**Key comparisons:**
- DNA vs Protein (which is stronger?)
- Combined vs Max(DNA, Protein) (synergy or dominance?)
- Full-length degradation (which modality is more robust?)

---

## Quick Start

Run the complete analysis:

```bash
# 1. Train modality classifiers (~30-60 min)
docker compose run --rm virnucpro python scripts/train_modality_classifiers.py

# 2. Evaluate on 150bp (~1 min)
docker compose run --rm virnucpro python scripts/evaluate_modality_150bp.py

# 3. View results
cat reports/modality_comparison_150bp.json | python -m json.tool
```

**Total runtime:** ~35-65 minutes

---

## Expected Output

```
MODALITY COMPARISON: 150bp vs Full-Length
======================================================================

Modality     Full-Length Acc    150bp Acc    Degradation
----------------------------------------------------------------------
dna                  0.8500        0.6800      0.1700 (20.0%)
protein              0.8500        0.4500      0.4000 (47.1%)
combined             0.9020        0.5297      0.3723 (41.3%)

======================================================================
KEY FINDINGS
======================================================================

150bp Performance:
  DNA-only:     0.6800
  Protein-only: 0.4500
  Combined:     0.5297

✓ DNA features are MORE informative for 150bp (+0.2300 accuracy)
  → 50 amino acids provide limited discriminative power
  → DNA sequence patterns more reliable for short reads

✗ Combining modalities HURTS performance (-0.1503 vs best single)
  → Consider using DNA-only for 150bp reads
```

---

## Next Steps Based on Results

### If DNA-only is best for 150bp:

1. **Immediate:** Deploy DNA-only for <300bp in production
2. **Architecture:** Add length-based routing
3. **Training:** Train specialized DNA-only model on short reads
4. **Speedup:** Skip ESM2 extraction for short reads (major speedup!)

### If protein is still valuable:

1. **Training:** Fine-tune on 150bp data (original plan)
2. **Keep architecture:** Combined features for all lengths
3. **Optimize:** Focus on protein embedding quality

### If both fail equally:

1. **Fundamental limit:** 150bp may be too short
2. **Alternative approaches:** k-mer methods, traditional ML
3. **Set minimum:** Require 300bp+ for reliable classification

---

## Cost-Benefit

**Cost:** ~1 hour of training + GPU time

**Benefit:**
- Understand which features matter
- Potentially discover DNA-only is better for short reads (huge speedup)
- Inform architecture decisions
- Guide future development

**Return on Investment:** High - this analysis could reveal a simpler, faster solution for short reads.

---

## Files Created

- `scripts/train_modality_classifiers.py` - Train DNA/protein/combined models
- `scripts/evaluate_modality_150bp.py` - Test on 150bp data
- `docs/modality_contribution_analysis.md` - This document

---

## Summary

This analysis answers: **"Should we even be using protein features for short reads?"**

The answer could be:
- Yes (keep combined)
- No (use DNA-only for <300bp)
- Maybe (depends on accuracy requirements)

Run the analysis to find out!
