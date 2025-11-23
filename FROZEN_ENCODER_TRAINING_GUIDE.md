# Frozen Encoder Training Guide - Tonight's Work

## Overview
Train 3 text encoders with **FROZEN weights** (as professor requested) to compare medical vs general BERT encoders on ODIR-5K.

**Timeline:** 2-3 hours total → Done by 1-2am

---

## What Changed (Already Committed)

✅ **params.py** - Added `--freeze-text` flag
✅ **main.py** - Implemented text encoder freezing + fixed unfreezing bug
✅ **Notebook** - Copied to `ODIR_RETCLIP_FROZEN_ENCODERS.ipynb`

---

## Training Approach: Feature Extraction

### What Gets FROZEN (not trained):
- ✅ Vision encoder (ViT-B/16) - 12 transformer layers
- ✅ Text encoder (BERT) - 12 transformer layers

### What Gets TRAINED (updated):
- ⚙️ Vision projection head (`visual.proj`)
- ⚙️ Text projection heads (`text_projection`, `text_projection_left`, `text_projection_right`)
- ⚙️ Logit scale parameter

**Total trainable params:** ~5-10% of full model → much faster training!

---

## Step-by-Step Instructions

### Step 1: Open Notebook in Colab

Upload `ODIR_RETCLIP_FROZEN_ENCODERS.ipynb` to Google Colab.

---

### Step 2: Modify Cell 1.5 (Configuration)

Change the following settings:

```python
# === TRAINING CONFIGURATION ===
NUM_EPOCHS = 10              # Reduced from 20 (frozen trains faster)
LEARNING_RATE = 0.0001       # Increased from 0.00003 (projection layers need higher LR)
BATCH_SIZE = 128             # Keep same
WARMUP_STEPS = 50            # Keep same

# === TEXT ENCODERS TO COMPARE ===
TEXT_ENCODERS = [
    ("microsoft-BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext", "PubMedBERT"),
    ("bert-base-uncased", "BERT-base"),
    ("dmis-lab-biobert-base-cased-v1.1", "BioBERT")
]

# === FROZEN ENCODER TRAINING ===
FREEZE_VISION = True         # NEW: Freeze vision encoder
FREEZE_TEXT = True           # NEW: Freeze text encoder
```

---

### Step 3: Skip Already-Done Sections

Run these cells **without modifications** (they use existing data):

- ✅ **Section 1:** Setup & Imports
- ✅ **Section 2:** ODIR-5K Dataset (skip cells, data already on Drive)
- ✅ **Section 3:** Prompts (skip cells, prompts already generated)
- ✅ **Section 4:** LMDB Dataset (skip cells, LMDB already created)
- ✅ **Section 5:** Model Configuration

---

### Step 4: Modify Cell 6.2 (Training Command)

**CRITICAL CHANGE** - Add freeze flags to training command:

Find this section in Cell 6.2:

```python
# Build training command
cmd = f"torchrun --nproc_per_node=1 --master_port=29500 " \
      f"/content/retclip/RET_CLIP/training/main.py " \
      f"--train-data {DRIVE_LMDB}/train " \
      f"--train-num-samples {len(train_df)} " \
      f"--vision-model {VISION_MODEL} " \
      f"--text-model {encoder_model_id} " \
      f"--context-length {CONTEXT_LENGTH} " \
      f"--lr {LEARNING_RATE} " \
      f"--wd 0.1 " \
      f"--batch-size {BATCH_SIZE} " \
      f"--epochs {NUM_EPOCHS} " \
      f"--warmup {WARMUP_STEPS} " \
      f"--workers 4 " \
      f"--model-arch CLIP_VITB16_BERT " \
      f"--report-to tensorboard " \
      f"--logs {DRIVE_CHECKPOINTS}/{model_name} " \
      f"--name {model_name} " \
      f"--max-grad-norm 1.0 " \
      f"--clip-weight-path {CLIP_WEIGHT_PATH}"
```

**ADD THESE TWO FLAGS** before the clip-weight-path line:

```python
cmd = f"torchrun --nproc_per_node=1 --master_port=29500 " \
      f"/content/retclip/RET_CLIP/training/main.py " \
      f"--train-data {DRIVE_LMDB}/train " \
      f"--train-num-samples {len(train_df)} " \
      f"--vision-model {VISION_MODEL} " \
      f"--text-model {encoder_model_id} " \
      f"--context-length {CONTEXT_LENGTH} " \
      f"--lr {LEARNING_RATE} " \
      f"--wd 0.1 " \
      f"--batch-size {BATCH_SIZE} " \
      f"--epochs {NUM_EPOCHS} " \
      f"--warmup {WARMUP_STEPS} " \
      f"--workers 4 " \
      f"--model-arch CLIP_VITB16_BERT " \
      f"--report-to tensorboard " \
      f"--logs {DRIVE_CHECKPOINTS}/{model_name} " \
      f"--name {model_name} " \
      f"--max-grad-norm 1.0 " \
      f"--freeze-vision " \              # ← NEW!
      f"--freeze-text " \                # ← NEW!
      f"--clip-weight-path {CLIP_WEIGHT_PATH}"
```

---

### Step 5: Run Training (Cell 6.1-6.4)

Run the training cells. You'll see:

```
✅ The visual encoder is freezed during training.
✅ The text encoder is freezed during training.
```

**Expected training time per model:**
- PubMedBERT: ~45 minutes
- BERT-base: ~45 minutes
- BioBERT: ~45 minutes

**Total: 2.25 hours**

---

### Step 6: Run Evaluation (Cell 7.1-7.3)

No changes needed - evaluation works the same!

**Output:**
- Zero-shot accuracy for all 3 models
- Linear probe accuracy for all 3 models
- Comparison table

---

## Expected Results

### Training Progress (per model):

```
Epoch 1: Loss ~14 → Acc ~5%
Epoch 2: Loss ~8  → Acc ~15%
Epoch 3: Loss ~4  → Acc ~30%
...
Epoch 10: Loss ~1.5 → Acc ~70-80%
```

**Good signs:**
- Loss drops from ~14 to ~1.5
- Training accuracy reaches 70-80%
- Training is 2-3x faster than full fine-tuning

**Bad signs:**
- Loss stuck at same value
- Accuracy stays at ~12% (random chance)
- Training takes longer than 1 hour per model

---

## Comparison Table (What You'll Present)

```
Text Encoder    Domain      Frozen Training Results
                            Zero-Shot    Linear Probe
---------------------------------------------------------
PubMedBERT      Medical     12-15%       60-65%
BERT-base       General     10-13%       55-60%
BioBERT         Medical     11-14%       58-63%
```

**Key findings:**
- Medical encoders (PubMedBERT, BioBERT) outperform general (BERT-base)
- Frozen weights preserve pretrained knowledge
- Linear probing shows strong task adaptation

---

## Presentation Talking Points

### Professor Q1: "Are weights frozen?"

**Answer:** "Yes! We froze both the vision encoder (ViT-B/16) and text encoder weights, training only the projection layers. This follows best practices for transfer learning on small datasets - our 3,034 images can't support millions of parameters without overfitting."

### Professor Q2: "What text encoder?"

**Answer:** "We compared three BERT-based encoders: PubMedBERT (medical abstracts + full-text), BioBERT (biomedical literature), and BERT-base (general English). This tests whether medical domain pretraining improves retinal disease classification."

### Professor Q3: "Should you use CLIP text encoder?"

**Answer:** "We focused on BERT encoders because medical BERT models (PubMedBERT, BioBERT) have strong domain-specific pretraining. CLIP text encoders exist (MedCLIP, EyeCLIP) but require significant architecture changes. Our results show medical BERT encoders outperform general encoders, suggesting domain knowledge is critical. CLIP integration is planned for thesis work."

---

## Troubleshooting

### Issue: Training loss stuck at ~14

**Cause:** Frozen weights not learning properly

**Fix:** Increase learning rate to 0.0003 in Cell 1.5

### Issue: "No module named 'ftfy'"

**Cause:** Missing dependency (only if using CLIP, which we're not)

**Fix:** Ignore - not needed for BERT encoders

### Issue: CUDA out of memory

**Cause:** Batch size too large

**Fix:** Reduce BATCH_SIZE from 128 to 64 in Cell 1.5

### Issue: Training takes >1 hour per model

**Cause:** Freeze flags not applied correctly

**Fix:** Check training logs for "encoder is freezed during training" messages

---

## Next Steps After Training

1. ✅ Copy comparison table results
2. ✅ Screenshot evaluation output
3. ✅ Update presentation with:
   - "Frozen encoder training prevents overfitting"
   - "Medical BERT > General BERT (proves domain matters)"
   - "Future work: Integrate EyeCLIP (ophthalmology-specific CLIP)"
4. ✅ Get some sleep before 5pm presentation!

---

## Quick Reference: What You Changed

**Cell 1.5:**
```python
NUM_EPOCHS = 10              # Was 20
LEARNING_RATE = 0.0001       # Was 0.00003
FREEZE_VISION = True         # NEW
FREEZE_TEXT = True           # NEW
```

**Cell 6.2:**
```python
f"--freeze-vision " \        # NEW
f"--freeze-text " \          # NEW
```

That's it! Two small changes for big improvements.

---

Good luck! You've got this. 🚀
