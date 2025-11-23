# 🚀 RET-CLIP Training with Pretrained Weights

## Problem Identified

Your model was training from **random initialization** without pretrained weights, which is why:
- Loss plateaued at 16.64 after epoch 2
- Zero-shot accuracy was terrible (9.88%, barely above random)
- Model completely failed to learn

Training CLIP from scratch requires millions of samples. With only 2,427 patients, pretrained weights are **essential**.

---

## Solution: Add Pretrained Weight Initialization

### Option 1: OpenAI CLIP + HuggingFace BERT (Recommended) ✅

This uses official OpenAI CLIP vision weights + medical domain BERT.

**Add this NEW cell BEFORE your training cell (Section 6):**

```python
print("\n" + "="*80)
print("DOWNLOADING AND PREPARING PRETRAINED WEIGHTS")
print("="*80)

import torch
import clip as openai_clip  # OpenAI's official CLIP
from transformers import AutoModel

# Create weights directory
DRIVE_WEIGHTS = f"{DRIVE_BASE}/pretrained_weights"
os.makedirs(DRIVE_WEIGHTS, exist_ok=True)

# ============================================================================
# STEP 1: Download OpenAI CLIP ViT-B/16 Weights
# ============================================================================
print("\n📥 Downloading OpenAI CLIP ViT-B/16...")

# Download using OpenAI's official CLIP library
openai_model, preprocess = openai_clip.load("ViT-B/16", device="cpu")

# Extract vision encoder weights
clip_state_dict = openai_model.state_dict()

# Save in RET-CLIP compatible format
CLIP_WEIGHT_PATH = f"{DRIVE_WEIGHTS}/openai_vit_b_16.pt"
torch.save(clip_state_dict, CLIP_WEIGHT_PATH)

print(f"✅ Saved OpenAI CLIP weights to: {CLIP_WEIGHT_PATH}")
print(f"   Vision encoder: ViT-B/16")
print(f"   Size: {os.path.getsize(CLIP_WEIGHT_PATH) / 1024 / 1024:.1f} MB")

del openai_model, clip_state_dict
torch.cuda.empty_cache()

# ============================================================================
# STEP 2: Download PubMedBERT Weights from HuggingFace
# ============================================================================
print(f"\n📥 Downloading PubMedBERT from HuggingFace...")

# Download the exact model you're using for text encoding
bert_model_name = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
bert_model = AutoModel.from_pretrained(bert_model_name)

# Extract BERT state dict
bert_state_dict = bert_model.state_dict()

# Add 'bert.' prefix to match RET-CLIP's expected format
bert_state_dict_prefixed = {}
for k, v in bert_state_dict.items():
    # Skip pooler layer (RET-CLIP doesn't use it)
    if "pooler" not in k:
        bert_state_dict_prefixed[f"bert.{k}"] = v

# Save in RET-CLIP compatible format
BERT_WEIGHT_PATH = f"{DRIVE_WEIGHTS}/pubmedbert_base.pt"
torch.save(bert_state_dict_prefixed, BERT_WEIGHT_PATH)

print(f"✅ Saved PubMedBERT weights to: {BERT_WEIGHT_PATH}")
print(f"   Text encoder: {bert_model_name}")
print(f"   Size: {os.path.getsize(BERT_WEIGHT_PATH) / 1024 / 1024:.1f} MB")

del bert_model, bert_state_dict, bert_state_dict_prefixed
torch.cuda.empty_cache()

# ============================================================================
# STEP 3: Verify Weight Files
# ============================================================================
print(f"\n" + "="*80)
print("VERIFICATION")
print("="*80)

print(f"\n✅ Pretrained weights ready:")
print(f"   CLIP:  {CLIP_WEIGHT_PATH}")
print(f"   BERT:  {BERT_WEIGHT_PATH}")

# Test loading
print(f"\n🔍 Testing weight loading...")
clip_test = torch.load(CLIP_WEIGHT_PATH, map_location="cpu", weights_only=False)
bert_test = torch.load(BERT_WEIGHT_PATH, map_location="cpu", weights_only=False)

print(f"   CLIP keys: {len(clip_test)} layers")
print(f"   BERT keys: {len(bert_test)} layers")
print(f"   Sample CLIP key: {list(clip_test.keys())[0]}")
print(f"   Sample BERT key: {list(bert_test.keys())[0]}")

del clip_test, bert_test

print(f"\n✅ All checks passed! Ready for training with pretrained weights.")
print(f"\n" + "="*80)
```

---

### Updated Training Cell (Section 6)

**REPLACE your current training cell with this version that includes pretrained weight paths:**

```python
print("\n" + "="*80)
print(f"SECTION 6: TRAIN RET-CLIP WITH PRETRAINED WEIGHTS")
print("="*80)

# Training configuration (same as before)
EXPERIMENT_NAME = f"retclip_odir_{'test' if TEST_MODE else 'full'}"
NUM_GPUS = 1  # Colab has 1 GPU

# Checkpoint directory
CHECKPOINT_DIR = f"{DRIVE_CHECKPOINTS}/{EXPERIMENT_NAME}"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

print(f"\n📦 Experiment: {EXPERIMENT_NAME}")
print(f"📂 Checkpoints: {CHECKPOINT_DIR}")
print(f"🔧 Configuration:")
print(f"   Vision Model: {VISION_MODEL}")
print(f"   Text Model: {TEXT_MODEL}")
print(f"   Image Size: {IMAGE_SIZE}")
print(f"   Batch Size: {BATCH_SIZE}")
print(f"   Learning Rate: {LEARNING_RATE}")
print(f"   Warmup Steps: {WARMUP_STEPS}")
print(f"   Epochs: {NUM_EPOCHS}")
print(f"   ⭐ CLIP Pretrained: {CLIP_WEIGHT_PATH}")
print(f"   ⭐ BERT Pretrained: {BERT_WEIGHT_PATH}")

# Build training command with PRETRAINED WEIGHTS
training_command = f"""
torchrun --nproc_per_node={NUM_GPUS} \\
  --master_port=29500 \\
  /content/retclip/RET_CLIP/training/main.py \\
  --train-data {DRIVE_LMDB}/train/pairs \\
  --train-img {DRIVE_LMDB}/train/imgs \\
  --val-data {DRIVE_LMDB}/test/pairs \\
  --val-img {DRIVE_LMDB}/test/imgs \\
  --vision-model {VISION_MODEL} \\
  --text-model {TEXT_MODEL} \\
  --clip-weight-path {CLIP_WEIGHT_PATH} \\
  --bert-weight-path {BERT_WEIGHT_PATH} \\
  --warmup {WARMUP_STEPS} \\
  --batch-size {BATCH_SIZE} \\
  --lr {LEARNING_RATE} \\
  --wd 0.001 \\
  --max-epochs {NUM_EPOCHS} \\
  --workers 4 \\
  --valid-epoch-interval 1 \\
  --save-epoch-frequency 1 \\
  --report-training-batch-acc \\
  --context-length 77 \\
  --input-resolution {IMAGE_SIZE} \\
  --logs {CHECKPOINT_DIR} \\
  --name checkpoints \\
  --use-augment \\
  --gather-with-grad
""".strip()

print(f"\n🚀 Starting training...")
print(f"📝 Command:\n{training_command}\n")

# Execute training
!{training_command}

print(f"\n✅ Training complete!")
print(f"📂 Checkpoints saved to: {CHECKPOINT_DIR}/checkpoints/")
```

---

### Key Changes:

1. **Added `--clip-weight-path {CLIP_WEIGHT_PATH}`** - Initializes vision encoder with OpenAI CLIP
2. **Added `--bert-weight-path {BERT_WEIGHT_PATH}`** - Initializes text encoder with PubMedBERT
3. Both paths point to the weights downloaded in the previous cell

---

## Expected Improvements

With pretrained weights, you should see:

### Before (Random Init):
- Initial loss: ~17.0
- Loss after 20 epochs: ~16.64 (stuck!)
- Zero-shot accuracy: 9.88% (barely above random)
- Model failed to learn

### After (Pretrained):
- Initial loss: ~5.0-8.0 (much lower!)
- Loss after 20 epochs: <3.0 (actual learning!)
- Zero-shot accuracy: >30% (meaningful performance)
- Continued improvement each epoch

---

## Training Timeline

With pretrained weights on 2,427 samples:

- **Epochs 1-5**: Rapid improvement (loss ~8 → ~4)
- **Epochs 6-15**: Steady improvement (loss ~4 → ~2.5)
- **Epochs 16-20**: Fine-tuning (loss ~2.5 → ~2.0)

**Final zero-shot accuracy estimate: 35-50%** (vs 9.88% without pretraining)

---

## Troubleshooting

### Issue: "ImportError: No module named 'clip'"

**Solution:**
```python
!pip install git+https://github.com/openai/CLIP.git
```

### Issue: "CUDA out of memory" during weight download

**Solution:** Weights download on CPU, shouldn't cause OOM. If it does:
```python
# Clear GPU cache before downloading
import gc
gc.collect()
torch.cuda.empty_cache()
```

### Issue: Weight files not found during training

**Solution:** Verify paths:
```python
print(f"CLIP exists: {os.path.exists(CLIP_WEIGHT_PATH)}")
print(f"BERT exists: {os.path.exists(BERT_WEIGHT_PATH)}")
```

---

## Next Steps

1. **Add the weight download cell** to your notebook (before Section 6)
2. **Update the training cell** with the new command
3. **Re-run training** (will take ~1-2 hours for 20 epochs)
4. **Re-run zero-shot evaluation** (Cell 7.1-7.2)
5. **Compare results** - should see massive improvement!

---

## Why This Works

**Transfer Learning Benefits:**

1. **Vision Encoder**: OpenAI CLIP was trained on 400M image-text pairs, so it already understands visual concepts
2. **Text Encoder**: PubMedBERT was trained on 14M PubMed abstracts, so it understands medical language
3. **Fine-tuning**: Your 2,427 samples are now used to **adapt** these pretrained representations to retinal images, not learn from scratch
4. **Faster Convergence**: Model starts from good initialization, reaches better performance in fewer epochs
5. **Better Generalization**: Pretrained features help with zero-shot transfer to unseen diseases

**This is the standard approach for CLIP-style models and should dramatically improve your results!**

---

## References

- OpenAI CLIP: https://github.com/openai/CLIP
- PubMedBERT: https://huggingface.co/microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext
- RET-CLIP Paper: https://arxiv.org/pdf/2405.14137
