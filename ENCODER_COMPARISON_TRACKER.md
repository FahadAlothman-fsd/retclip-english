# RET-CLIP Text Encoder Comparison Tracker

## Training Progress

| Text Encoder | Status | Training Time | Final Loss | Training Acc | Zero-Shot Acc | Notes |
|-------------|--------|---------------|------------|--------------|---------------|-------|
| **PubMedBERT** | ✅ COMPLETED | ~2.5 hours | 1.80 | 82% | **12.19%** | Medical abstracts + full-text |
| **BERT-base** | ⏳ PENDING | - | - | - | - | General English baseline |
| **BioBERT** | ⏳ PENDING | - | - | - | - | PubMed + PMC articles |

## Model Specifications

### 1. PubMedBERT
- **HuggingFace**: `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext`
- **Training Corpus**: PubMed abstracts + PMC full-text (biomedical)
- **Vocabulary**: 30,522 tokens (uncased)
- **Architecture**: BERT-base (12 layers, 768 hidden, 12 heads)
- **Domain**: Clinical/medical text
- **Expected Performance**: Best on medical terminology

### 2. BERT-base-uncased
- **HuggingFace**: `bert-base-uncased`
- **Training Corpus**: Wikipedia + BooksCorpus (general English)
- **Vocabulary**: 30,522 tokens (uncased)
- **Architecture**: BERT-base (12 layers, 768 hidden, 12 heads)
- **Domain**: General language
- **Expected Performance**: Baseline (likely lowest due to domain gap)

### 3. BioBERT
- **HuggingFace**: `dmis-lab/biobert-base-cased-v1.1`
- **Training Corpus**: PubMed abstracts + PMC (biomedical)
- **Vocabulary**: 28,996 tokens (cased)
- **Architecture**: BERT-base (12 layers, 768 hidden, 12 heads)
- **Domain**: Biomedical research
- **Expected Performance**: Comparable to PubMedBERT

## Unified Training Configuration

All three models use identical hyperparameters (from RET-CLIP paper):

```python
LEARNING_RATE = 0.00003      # 3e-5
BATCH_SIZE = 128             # Half of paper's 256
NUM_EPOCHS = 20              # Extended from paper's 10
WARMUP_STEPS = 50
VISION_MODEL = "ViT-B-16"    # Same for all three
IMAGE_SIZE = 224
CONTEXT_LENGTH = 100

# Training flags:
--max-grad-norm 1.0          # Gradient clipping
# NO --skip-scheduler         # Use cosine LR decay
```

## Hypothesis & Expected Results

### Performance Ranking (Predicted)

**Best → Worst (Zero-Shot Accuracy)**

1. **PubMedBERT** (12.19% - confirmed): Most comprehensive medical corpus
2. **BioBERT** (10-14% predicted): Similar biomedical domain
3. **BERT-base** (8-12% predicted): General domain, largest gap

### Why These Predictions?

**PubMedBERT advantage:**
- Trained on both abstracts AND full-text papers
- Larger medical corpus
- Uncased (better for medical acronyms)

**BioBERT competitive:**
- Similar biomedical training data
- May be slightly worse due to cased tokenization
- Less full-text exposure (more abstracts)

**BERT-base disadvantage:**
- No medical domain exposure
- General Wikipedia/books knowledge
- Larger domain gap to retinal imaging

### Potential Surprise Scenarios

**If BERT-base outperforms medical models:**
- Suggests task relies more on basic visual-language alignment than medical knowledge
- Retinal disease descriptions use common English effectively
- Overfitting to medical jargon hurts generalization

**If BioBERT outperforms PubMedBERT:**
- Cased tokenization preserves important medical distinctions
- PubMed-only training is more focused than abstracts+fulltext

**If all three perform similarly (±2%):**
- Dataset too small to leverage medical knowledge
- CLIP pretraining dominates over text encoder differences
- Task is vision-dominated, not language-dominated

## Evaluation Metrics to Compare

### Primary Metrics
- **Accuracy**: Overall classification accuracy (main metric)
- **F1 Macro**: Unweighted average (important for class imbalance)
- **F1 Weighted**: Weighted by support (clinical relevance)

### Secondary Analysis
- **Per-class accuracy**: Which diseases does each model excel at?
- **Confusion matrices**: Do medical models make more "medically plausible" errors?
- **Confidence scores**: Are medical models more confident on correct predictions?

### Example Per-Class Analysis

| Disease | PubMedBERT | BERT-base | BioBERT | Winner |
|---------|-----------|-----------|---------|--------|
| Normal (N) | ? | ? | ? | ? |
| Diabetes (D) | ? | ? | ? | ? |
| Glaucoma (G) | ? | ? | ? | ? |
| Cataract (C) | ? | ? | ? | ? |
| AMD (A) | ? | ? | ? | ? |
| Hypertension (H) | ? | ? | ? | ? |
| Myopia (M) | ? | ? | ? | ? |
| Other (O) | ? | ? | ? | ? |

Fill this table after running zero-shot evaluation on all three models.

## Training Checklist

### BERT-base-uncased Training

- [ ] Set `TEXT_MODEL = "bert-base-uncased"` in Cell 1.5
- [ ] Verify config: LR=0.00003, batch=128, epochs=20
- [ ] Run training cells (6.1-6.4)
- [ ] Monitor convergence: loss ~14 → ~2, acc 1% → 75-85%
- [ ] Verify checkpoint saved to `/content/drive/MyDrive/RET-CLIP-ODIR/checkpoints/retclip_odir_bertbase/`
- [ ] Record final metrics: loss, training accuracy, training time

### BioBERT Training

- [ ] Set `TEXT_MODEL = "dmis-lab-biobert-base-cased-v1.1"` in Cell 1.5
- [ ] Clear GPU memory: `torch.cuda.empty_cache()`
- [ ] Verify config: LR=0.00003, batch=128, epochs=20
- [ ] Run training cells (6.1-6.4)
- [ ] Monitor convergence: loss ~14 → ~2, acc 1% → 75-85%
- [ ] Verify checkpoint saved to `/content/drive/MyDrive/RET-CLIP-ODIR/checkpoints/retclip_odir_biobertbasecasedv11/`
- [ ] Record final metrics: loss, training accuracy, training time

### Zero-Shot Evaluation (All Three)

- [ ] Set `RUN_TEXT_ENCODER_COMPARISON = True` in Cell 1.5
- [ ] Run evaluation cells (7.1-7.3)
- [ ] Record zero-shot accuracy for each model
- [ ] Compare F1 scores (macro and weighted)
- [ ] Analyze per-class performance
- [ ] Generate comparison plots/tables

## Thesis Analysis Questions

After training all three models, analyze:

1. **Does medical pretraining help?**
   - Compare PubMedBERT/BioBERT vs BERT-base
   - Quantify domain advantage
   - Statistical significance testing

2. **Which medical encoder is better?**
   - PubMedBERT vs BioBERT head-to-head
   - Explain differences (corpus, casing, architecture)

3. **Where does each model excel?**
   - Per-disease analysis
   - Which diseases benefit most from medical knowledge?
   - Error analysis: medically plausible vs implausible mistakes

4. **Is the improvement worth it?**
   - Training cost (time, compute)
   - Performance gain
   - Practical clinical significance

## Next Steps After Comparison

Based on results, you can:

1. **Best performer → Fine-tune**: Take the best model and add supervised fine-tuning
2. **Ensemble**: Combine predictions from all three (voting/averaging)
3. **Analysis**: Deep dive into why certain models excel on certain diseases
4. **Ablation**: Try different vision encoders (ViT-L-14) with best text encoder

## Quick Commands Reference

### Start BERT-base Training
```python
# In Colab Cell 1.5:
TEXT_MODEL = "bert-base-uncased"
LEARNING_RATE = 0.00003
BATCH_SIZE = 128
NUM_EPOCHS = 20

# Then run cells 6.1-6.4
```

### Start BioBERT Training
```python
# In Colab Cell 1.5:
TEXT_MODEL = "dmis-lab-biobert-base-cased-v1.1"
LEARNING_RATE = 0.00003
BATCH_SIZE = 128
NUM_EPOCHS = 20

# Clear GPU first:
import torch
torch.cuda.empty_cache()

# Then run cells 6.1-6.4
```

### Run Comparison Evaluation
```python
# In Colab Cell 1.5:
RUN_TEXT_ENCODER_COMPARISON = True

# Then run cells 7.1-7.3
```

---

## Training Log Template

Use this template to track training progress:

### BERT-base-uncased
```
Start time: ___________
End time: ___________
Duration: ___________

Initial loss (epoch 1): ___________
Final loss (epoch 20): ___________
Initial training acc: ___________
Final training acc: ___________

Zero-shot accuracy: ___________
F1 Macro: ___________
F1 Weighted: ___________

Notes:
-
```

### BioBERT
```
Start time: ___________
End time: ___________
Duration: ___________

Initial loss (epoch 1): ___________
Final loss (epoch 20): ___________
Initial training acc: ___________
Final training acc: ___________

Zero-shot accuracy: ___________
F1 Macro: ___________
F1 Weighted: ___________

Notes:
-
```
