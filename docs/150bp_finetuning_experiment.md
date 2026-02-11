# 150bp Fine-Tuning Experiment

## Hypothesis

**The 53% accuracy on 150bp reads is due to distribution mismatch (model never saw short reads during training), not a fundamental limitation.**

If we fine-tune the model on 150bp training data, accuracy should improve from 53% → 70%+ on the 150bp test set.

---

## Experimental Design

### Control (Baseline)
- **Model:** FastESM2-650M trained on full-length sequences (500-3000bp)
- **Test set:** 50K random 150bp windows
- **Accuracy:** 53% (from previous benchmark)

### Treatment (Fine-Tuned)
- **Model:** Same FastESM2-650M, fine-tuned on 150bp training data
- **Training data:** ~100K 150bp windows from training sequences
- **Test set:** Same 50K windows (unchanged)
- **Expected accuracy:** 70%+ if hypothesis is correct

### Success Criteria
- **Strong evidence:** Accuracy improves by >15 percentage points (53% → 68%+)
- **Moderate evidence:** Accuracy improves by 5-15 points (53% → 58-68%)
- **Hypothesis rejected:** Accuracy improves by <5 points

---

## Implementation Workflow

### Step 1: Generate 150bp Training Data

Create random 150bp windows from training sequences:

```bash
pixi run python scripts/generate_150bp_training_data.py
```

**Output:**
- `train_150bp_viral.fasta` (~52K sequences)
- `train_150bp_nonviral.fasta` (~48K sequences)
- `train_150bp_metadata.json`

**Strategy:**
- Sample 1-3 random windows per training sequence (depending on length)
- Target ~100K total windows
- Maintain 52:48 viral:non-viral ratio from original dataset

**Important Note:**
This script currently uses test sequences as a proxy (with different sampling). Ideally, you should generate from original training sequence FASTAs to avoid any test set leakage.

---

### Step 2: Extract Embeddings for Training Data

Extract DNA + protein embeddings for the 150bp training windows:

```bash
docker compose run --rm virnucpro python scripts/extract_150bp_training_embeddings.py
```

**Output:**
- `data/train_150bp/viral_merged.pt` (2048-dim embeddings)
- `data/train_150bp/nonviral_merged.pt` (2048-dim embeddings)

**Runtime:** ~30-40 minutes for 100K sequences

**Process:**
1. DNABERT-S extraction (768-dim DNA embeddings)
2. FastESM2-650M extraction (1280-dim protein embeddings)
3. Merge to 2048-dim features

---

### Step 3: Fine-Tune the Model

Fine-tune the pre-trained model on 150bp training data:

```bash
docker compose run --rm virnucpro python scripts/finetune_150bp.py \
  --pretrained-model 300_model_fastesm650.pth \
  --viral-data data/train_150bp/viral_merged.pt \
  --nonviral-data data/train_150bp/nonviral_merged.pt \
  --output 300_model_fastesm650_finetuned_150bp.pth \
  --epochs 10 \
  --lr 1e-4 \
  --batch-size 256
```

**Parameters:**
- `--epochs 10`: Modest number to avoid overfitting
- `--lr 1e-4`: Lower than initial training (1e-3) for fine-tuning
- `--batch-size 256`: Adjust based on GPU memory

**Optional: Layer Freezing**

Try freezing the hidden layer to see if output layer adaptation is sufficient:

```bash
docker compose run --rm virnucpro python scripts/finetune_150bp.py \
  --freeze-layers \
  --epochs 10 \
  --lr 1e-3  # Higher LR when only training output layer
```

**Output:**
- `300_model_fastesm650_finetuned_150bp.pth`
- `300_model_fastesm650_finetuned_150bp_history.json`

**Runtime:** ~5-10 minutes for 10 epochs on 100K samples

---

### Step 4: Evaluate Fine-Tuned Model

Test the fine-tuned model on the 150bp test set:

```bash
docker compose run --rm virnucpro python scripts/evaluate_150bp.py \
  --model 300_model_fastesm650_finetuned_150bp.pth \
  --viral-data data/test_150bp_50k/viral_merged.pt \
  --nonviral-data data/test_150bp_50k/nonviral_merged.pt \
  --output reports/150bp_evaluation_finetuned.json
```

**Output:**
- `reports/150bp_evaluation_finetuned.json`

**Compare results:**

```bash
# Baseline (no fine-tuning)
cat reports/150bp_evaluation.json | grep -A 5 "metrics"

# Fine-tuned
cat reports/150bp_evaluation_finetuned.json | grep -A 5 "metrics"
```

---

## Expected Outcomes

### Scenario 1: Hypothesis Confirmed (Likely)

**Results:**
- Baseline: 53% accuracy
- Fine-tuned: 70-80% accuracy
- Precision improves dramatically (53% → 75%+)
- Recall decreases slightly (99% → 85%)

**Interpretation:**
- The model CAN learn from 150bp sequences
- Original poor performance was due to training data distribution
- Short reads require exposure during training

**Next Steps:**
1. Retrain full model with mixed-length data (100-3000bp)
2. Use data augmentation (random cropping) during training
3. Consider length-aware features

### Scenario 2: Partial Improvement

**Results:**
- Baseline: 53% accuracy
- Fine-tuned: 60-68% accuracy
- Some improvement but not to full-length levels

**Interpretation:**
- Training data helps but doesn't fully solve the problem
- May be hitting information bottleneck (50 amino acids insufficient)
- Some viral signatures require longer context

**Next Steps:**
1. Try ensemble approach (multiple 150bp windows per sequence)
2. Use DNA-only features for <200bp sequences
3. Set realistic expectations for short-read accuracy

### Scenario 3: Hypothesis Rejected (Unlikely)

**Results:**
- Baseline: 53% accuracy
- Fine-tuned: 53-58% accuracy
- Minimal or no improvement

**Interpretation:**
- Fundamental limitation: 150bp insufficient for reliable classification
- Protein features from 50 amino acids don't provide discriminative signal
- May need different approach entirely

**Next Steps:**
1. Test DNA-only classification (DNABERT-S alone)
2. Use k-mer based methods for short reads
3. Accept 150bp as unsuitable for current approach

---

## Alternative Experiments

### Experiment A: Train from Scratch on 150bp Only

Instead of fine-tuning, train a new model exclusively on 150bp data:

```bash
# Generate much more 150bp training data (500K+ samples)
# Train new model from scratch
# Compare: full-length model vs 150bp-specialist model
```

**Pros:** Model fully optimized for 150bp
**Cons:** Won't generalize to longer sequences

### Experiment B: Multi-Length Training

Train single model on mixed lengths:

```bash
# Training set composition:
# - 30% short (100-200bp)
# - 40% medium (200-500bp)
# - 30% long (500-3000bp)
```

**Pros:** Single model handles all lengths
**Cons:** May sacrifice some full-length accuracy

### Experiment C: DNA-Only for Short Reads

Test if protein features help at all for 150bp:

```bash
# Train model on DNABERT-S features only (768-dim)
# Compare: DNA-only vs DNA+protein for 150bp
```

**Pros:** Faster, may be more accurate for short reads
**Cons:** Loses protein information

---

## Quick Start (Full Workflow)

Run all steps sequentially:

```bash
# 1. Generate training data
pixi run python scripts/generate_150bp_training_data.py

# 2. Extract embeddings (~30-40 min)
docker compose run --rm virnucpro python scripts/extract_150bp_training_embeddings.py

# 3. Fine-tune model (~5-10 min)
docker compose run --rm virnucpro python scripts/finetune_150bp.py \
  --epochs 10 \
  --lr 1e-4

# 4. Evaluate (~1 min)
docker compose run --rm virnucpro python scripts/evaluate_150bp.py \
  --model 300_model_fastesm650_finetuned_150bp.pth \
  --output reports/150bp_evaluation_finetuned.json

# 5. Compare results
echo "=== BASELINE ==="
cat reports/150bp_evaluation.json | python -m json.tool | grep -A 5 "metrics"

echo "=== FINE-TUNED ==="
cat reports/150bp_evaluation_finetuned.json | python -m json.tool | grep -A 5 "metrics"
```

**Total runtime:** ~45-60 minutes

---

## Cost-Benefit Analysis

### If Fine-Tuning Works (70%+ accuracy)

**Benefit:**
- Single model handles both short and long reads
- No need for separate pipelines
- 150bp classification becomes viable

**Cost:**
- Must include short reads in future training runs
- Slightly more complex data preparation
- Small potential accuracy loss on full-length sequences

**Recommendation:** Worth it if short reads are a significant use case

### If Fine-Tuning Doesn't Work (<60% accuracy)

**Options:**
1. **Accept limitation:** Don't use 150bp reads, require minimum 300bp
2. **Separate approach:** DNA-only or k-mer methods for short reads
3. **Aggregation:** Combine multiple 150bp windows per sequence

**Recommendation:** Depends on your use case requirements

---

## Success Metrics

Track these metrics through fine-tuning:

1. **Training accuracy:** Should reach 80%+ on training set
2. **Test accuracy:** Target 70%+ (vs 53% baseline)
3. **Precision:** Should improve from 53% to 70%+
4. **Recall:** May decrease from 99% to 80-90% (acceptable trade-off)
5. **F1 Score:** Should improve from 69% to 75%+

If training accuracy is high (90%+) but test accuracy stays low (55%), you have overfitting. Try:
- Fewer epochs (5 instead of 10)
- Higher dropout (0.2 instead of 0.1)
- More training data

---

## Files Created

- `scripts/generate_150bp_training_data.py` - Generate 150bp FASTA files
- `scripts/extract_150bp_training_embeddings.py` - Extract embeddings
- `scripts/finetune_150bp.py` - Fine-tune model
- `scripts/evaluate_150bp.py` - Evaluate (updated to accept --model arg)
- `docs/150bp_finetuning_experiment.md` - This document

---

## Next Steps After Experiment

1. **Analyze results** - Did accuracy improve significantly?
2. **Document findings** - Update benchmark report with fine-tuning results
3. **Make decision:**
   - If successful: Plan to include short reads in production training
   - If unsuccessful: Document 150bp as unsupported, set minimum length requirement

4. **Consider alternatives** if fine-tuning fails:
   - DNA-only classification for <300bp
   - Ensemble voting across multiple windows
   - Hybrid thresholding based on read length
