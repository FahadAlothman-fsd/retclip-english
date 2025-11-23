# Training Remaining Text Encoders for RET-CLIP

## Overview

You've successfully trained **PubMedBERT** and achieved 12.19% zero-shot accuracy. Now you need to train the remaining two text encoders for comparison:

1. ✅ **PubMedBERT** - COMPLETED (12.19% accuracy)
2. ⏳ **BERT-base-uncased** - General English baseline
3. ⏳ **BioBERT** - Biomedical domain model

## Successful Hyperparameters (from PubMedBERT training)

These are the proven hyperparameters that achieved proper convergence:

```python
LEARNING_RATE = 0.00003      # 3e-5 (from RET-CLIP paper)
BATCH_SIZE = 128             # Half of paper's 256 (memory constraint)
NUM_EPOCHS = 20              # Paper uses 10, but we extended to 20
WARMUP_STEPS = 50            # From paper
VISION_MODEL = "ViT-B-16"    # Keep same vision encoder
IMAGE_SIZE = 224             # Standard for ViT-B-16

# Critical flags:
# --max-grad-norm 1.0        # Gradient clipping enabled
# NO --skip-scheduler        # Use cosine LR decay
```

## Training Configuration for BERT-base-uncased

### Model Details
- **Name**: BERT-base-uncased
- **Model ID**: `bert-base-uncased`
- **HuggingFace ID**: `bert-base-uncased`
- **Domain**: General English (non-medical)
- **Purpose**: Baseline to compare against medical-domain models

### Training Command

In your Colab notebook, modify Cell 1.5 configuration:

```python
# Change TEXT_MODEL to BERT-base-uncased
TEXT_MODEL = "bert-base-uncased"

# Keep all other settings the same:
LEARNING_RATE = 0.00003
BATCH_SIZE = 128
NUM_EPOCHS = 20
WARMUP_STEPS = 50
```

### Expected Checkpoints Location
```
/content/drive/MyDrive/RET-CLIP-ODIR/checkpoints/retclip_odir_bertbase/
```

### Expected Training Behavior
- **Initial loss**: ~14-16 (same as PubMedBERT)
- **Final loss**: ~1.5-2.5 (similar convergence)
- **Training accuracy**: 75-85% (similar to PubMedBERT)
- **Zero-shot accuracy**: Expected 8-15% (may be lower than PubMedBERT due to domain gap)

BERT-base is trained on general English text (Wikipedia, BooksCorpus), not medical literature. This domain gap may result in:
- Lower accuracy on medical terminology
- Potentially competitive on basic anatomical terms
- Useful baseline to quantify medical domain advantage

---

## Training Configuration for BioBERT

### Model Details
- **Name**: BioBERT
- **Model ID**: `dmis-lab-biobert-base-cased-v1.1`
- **HuggingFace ID**: `dmis-lab/biobert-base-cased-v1.1`
- **Domain**: Biomedical (PubMed + PMC)
- **Purpose**: Compare biomedical vs clinical text pretraining

### Training Command

In your Colab notebook, modify Cell 1.5 configuration:

```python
# Change TEXT_MODEL to BioBERT
TEXT_MODEL = "dmis-lab-biobert-base-cased-v1.1"

# Keep all other settings the same:
LEARNING_RATE = 0.00003
BATCH_SIZE = 128
NUM_EPOCHS = 20
WARMUP_STEPS = 50
```

### Expected Checkpoints Location
```
/content/drive/MyDrive/RET-CLIP-ODIR/checkpoints/retclip_odir_biobertbasecasedv11/
```

### Expected Training Behavior
- **Initial loss**: ~14-16 (same as PubMedBERT)
- **Final loss**: ~1.5-2.5 (similar convergence)
- **Training accuracy**: 75-85% (similar to PubMedBERT)
- **Zero-shot accuracy**: Expected 10-14% (comparable to PubMedBERT)

BioBERT is trained on PubMed abstracts + PMC full-text articles. Expected to perform similarly to PubMedBERT since both are biomedical domain models.

---

## Training Workflow

### Step 1: Train BERT-base-uncased

1. Open your Colab notebook: `ODIR_RETCLIP_UNIFIED_PIPELINE.ipynb`
2. Modify Cell 1.5:
   ```python
   TEXT_MODEL = "bert-base-uncased"
   ```
3. Run cells in order:
   - Cell 1.1-1.5: Configuration
   - Cell 2.1-2.3: ODIR setup (skip if already done)
   - Cell 3.1-3.6: Prompts (skip if already done)
   - Cell 4.1-4.5: LMDB creation (skip if already done)
   - **Cell 6.1-6.4: Training** ← Main training cells
4. Monitor training logs for convergence (loss should drop from ~14 to ~2)
5. Training time: ~2-3 hours on Colab T4 GPU

### Step 2: Train BioBERT

1. After BERT-base completes, modify Cell 1.5:
   ```python
   TEXT_MODEL = "dmis-lab-biobert-base-cased-v1.1"
   ```
2. **Important**: Restart runtime or clear GPU memory before training:
   ```python
   import torch
   torch.cuda.empty_cache()
   ```
3. Run Cell 6.1-6.4 again for BioBERT training
4. Monitor training logs for convergence
5. Training time: ~2-3 hours on Colab T4 GPU

### Step 3: Zero-Shot Evaluation

After training both models, run zero-shot evaluation for all three:

1. Set `RUN_TEXT_ENCODER_COMPARISON = True` in Cell 1.5
2. Run Cell 7.1-7.3 (Zero-Shot Evaluation)
3. This will evaluate all three models and generate comparison results

---

## What to Monitor During Training

### Good Training Signs ✅
- Loss decreases from ~14 to ~2 over 20 epochs
- Training accuracy increases from ~1% to 75-85%
- Logit scales remain < 6.9078 (not hitting ceiling)
- No gradient explosions (loss stays smooth)
- Learning rate decays with cosine schedule

### Bad Training Signs ❌
- Loss stuck at same value for 3+ epochs
- Training accuracy stays at ~1% (random chance)
- Logit scales hit 6.9078 ceiling
- Loss explodes to >100 (gradient explosion)
- Loss oscillates wildly

If you see bad signs, stop training and report the logs.

---

## Expected Comparison Results

### Hypothesis

Based on pretraining domains:

1. **PubMedBERT** (12.19% baseline): Medical abstracts + full-text
2. **BioBERT** (predicted 10-14%): PubMed + PMC (similar domain)
3. **BERT-base** (predicted 8-12%): General English (largest domain gap)

### Comparison Metrics

The evaluation will report:
- **Accuracy**: Overall classification accuracy
- **F1 Macro**: Unweighted average F1 across classes
- **F1 Weighted**: Weighted by class support
- **Per-class performance**: Accuracy for each of the 8 ODIR diseases

---

## Troubleshooting

### Issue: "Model config not found"
**Solution**: Make sure the model_id matches the JSON filename in `retclip/RET_CLIP/clip/model_configs/`

### Issue: "CUDA out of memory"
**Solution**: Reduce BATCH_SIZE from 128 to 64 or 32

### Issue: Training loss stuck
**Solution**:
1. Verify `--skip-scheduler` is NOT in the training command
2. Verify `--max-grad-norm 1.0` is present
3. Check learning rate is 0.00003 (not 0.001 or higher)

### Issue: Pretrained weights not loading
**Solution**: The models will auto-download from HuggingFace on first run. Ensure internet connection is active in Colab.

---

## Files Modified During Training

The following files were modified to fix training issues (already committed):

1. [training/params.py:54-57](retclip/RET_CLIP/training/params.py#L54-L57) - Added `--max-grad-norm` parameter
2. [training/train.py:341-346](retclip/RET_CLIP/training/train.py#L341-L346) - Added gradient clipping (AMP path)
3. [training/train.py:427-431](retclip/RET_CLIP/training/train.py#L427-L431) - Increased logit scale ceiling to ln(1000)
4. [training/main.py:180-198](retclip/RET_CLIP/training/main.py#L180-L198) - Fixed `--skip-scheduler` flag

These fixes are already in your codebase and will apply to all three model trainings.

---

## Next Steps After Training All Three

1. ✅ Compare zero-shot accuracies across all three models
2. ✅ Analyze which diseases each model performs best on
3. ✅ Document findings for thesis
4. Optionally: Try supervised fine-tuning on the best-performing model
5. Optionally: Ensemble predictions from all three models

---

## Quick Reference: Training Commands

### PubMedBERT (already completed)
```python
TEXT_MODEL = "microsoft-BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
# Result: 12.19% accuracy
```

### BERT-base-uncased (to train next)
```python
TEXT_MODEL = "bert-base-uncased"
# Expected: 8-12% accuracy (baseline)
```

### BioBERT (to train last)
```python
TEXT_MODEL = "dmis-lab-biobert-base-cased-v1.1"
# Expected: 10-14% accuracy
```

All use same hyperparameters: LR=0.00003, batch=128, epochs=20, gradient_clip=1.0, cosine_scheduler=enabled
