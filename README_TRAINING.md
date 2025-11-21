# RET-CLIP Training Pipeline - Complete Guide

This repository contains everything you need to train RET-CLIP on Google Colab with A100 GPU using your HuggingFace dataset.

---

## 📚 Documentation Files

### 🚀 Getting Started
1. **`QUICK_START.md`** - 5-minute overview, start here!
2. **`COLAB_TRAINING_GUIDE.md`** - Main guide with copy-paste Colab cells

### 🔬 Evaluation
3. **`TEST_EVALUATION_GUIDE.md`** - How to evaluate on test set after training

### 💻 Scripts
4. **`scripts/prepare_retclip_data_streaming.py`** - Standalone preprocessing script
5. **`colab_retclip_training.py`** - Automated Python script

---

## ⚡ Quick Answer to Your Question

### "Are we converting the test set?"

**YES!** ✅ Here's what happens:

```
Training Pipeline:
├── Cell 6: Convert BOTH train + test from CSV → TSV/JSONL
├── Cell 7: Build LMDB for BOTH train + test
├── Cell 9: Training uses test set for validation every epoch
└── After training: Run TEST_EVALUATION_GUIDE.md for full metrics
```

### What You Get:

**During Training (Automatic):**
- ✅ Test set converted to LMDB
- ✅ Validation loss computed every epoch
- ✅ Basic retrieval accuracy in logs

**After Training (Manual - see TEST_EVALUATION_GUIDE.md):**
- ✅ Extract image & text features
- ✅ Compute Recall@1, Recall@5, Recall@10
- ✅ Per-class performance analysis
- ✅ Visualizations of retrievals

---

## 📊 Your Data Pipeline

```
CSV Files (Your prompts)
  ├── retclip_prompts_full.csv (12,989 samples) → TRAIN
  └── retclip_prompts_test.csv (3,253 samples)  → TEST
                    ↓
         HuggingFace Dataset
  (Peacein/color-fundus-eye - Streaming mode + HF_TOKEN)
                    ↓
              TSV + JSONL
    ├── train_imgs.tsv + train_texts.jsonl
    └── test_imgs.tsv + test_texts.jsonl
                    ↓
                  LMDB
    ├── /lmdb/train/ (for training)
    └── /lmdb/test/  (for validation)
                    ↓
              TRAINING
    (Uses train for training, test for validation)
                    ↓
           CHECKPOINTS
    (epoch2.pt, epoch4.pt, ..., epoch10.pt)
                    ↓
          EVALUATION (see TEST_EVALUATION_GUIDE.md)
    (Extract features, compute metrics, visualize)
```

---

## 🎯 Step-by-Step Usage

### Option 1: Copy-Paste into Colab (Recommended)
1. Open `COLAB_TRAINING_GUIDE.md`
2. Copy each cell into Google Colab
3. Update Cell 3 with your HF_TOKEN
4. Run all cells

### Option 2: Run Standalone Script
1. Upload `scripts/prepare_retclip_data_streaming.py` to Colab
2. Run preprocessing manually
3. Run training manually

---

## 📁 File Structure After Training

```
/content/
├── data/                         # Intermediate files
│   ├── train_imgs.tsv
│   ├── train_texts.jsonl
│   ├── test_imgs.tsv
│   └── test_texts.jsonl
├── lmdb/                         # Training data
│   ├── train/
│   │   ├── imgs/
│   │   └── pairs/
│   └── test/
│       ├── imgs/
│       └── pairs/
├── logs/                         # Training outputs
│   └── retclip_training/
│       ├── checkpoints/
│       │   ├── epoch2.pt
│       │   ├── epoch4.pt
│       │   ├── ...
│       │   └── epoch_latest.pt
│       ├── out_*.log
│       └── params_*.txt
└── evaluation/                   # After running TEST_EVALUATION_GUIDE
    ├── test_imgs_features.pt
    ├── test_texts_features.pt
    └── evaluation_results.json
```

---

## 🔑 Key Features

### ✅ Handles HuggingFace Rate Limits
- **Streaming mode**: Downloads images one at a time
- **HF_TOKEN authentication**: Just like your prompt generation notebook
- **Automatic delays**: Configurable sleep between requests

### ✅ Resumable Pipeline
- **Checkpoint support** at every stage:
  - Data preprocessing (can resume from interruptions)
  - LMDB building (won't rebuild if exists)
  - Training (auto-saves every 2 epochs)

### ✅ Dual-Eye Architecture
- RET-CLIP expects left + right eye images
- Currently uses same image for both eyes
- **Future**: Modify if you get paired left/right images

---

## ⏱️ Expected Timeline

### Full Dataset (12,989 train + 3,253 test):
```
1. Data preprocessing:    2-3 hours
2. LMDB building:         10-15 minutes
3. Training (10 epochs):  8-12 hours
4. Evaluation:            5-10 minutes
────────────────────────────────────
Total:                    ~12-15 hours
```

### Small Test (100 samples):
```
1. Data preprocessing:    5-10 minutes
2. LMDB building:         1 minute
3. Training (10 epochs):  30-45 minutes
4. Evaluation:            1 minute
────────────────────────────────────
Total:                    ~45-60 minutes
```

---

## 🎓 What You're Training

### Model Architecture:
- **Vision Encoder**: ViT-B-16 (Visual Transformer)
- **Text Encoder**: Chinese RoBERTa
- **Training**: Contrastive learning (CLIP-style)
- **Input**: Dual fundus images + clinical text prompts

### Training Objective:
Align vision and text embeddings so that:
- Matching image-text pairs have high similarity
- Non-matching pairs have low similarity

### Use Cases After Training:
1. **Zero-shot classification**: New disease categories without retraining
2. **Image retrieval**: Find images by text description
3. **Text retrieval**: Generate descriptions for images
4. **Feature extraction**: Use embeddings for downstream tasks

---

## 🔧 Configuration Options

All configurable in Cell 3 of `COLAB_TRAINING_GUIDE.md`:

```python
# Model Selection
VISION_MODEL = "ViT-B-16"  # or "ViT-L-14" for better performance
BATCH_SIZE = 64            # Reduce to 32 if OOM

# Training Duration
MAX_EPOCHS = 10            # More epochs = better performance

# Performance Optimizations
USE_FLASH_ATTN = True      # 2x faster on A100
```

---

## 🐛 Troubleshooting

### Problem: Rate limited by HuggingFace
**Solution**: Set `HF_TOKEN` in Cell 3

### Problem: Out of memory during training
**Solution**: Reduce `BATCH_SIZE` to 32 or 16

### Problem: Preprocessing interrupted
**Solution**: Rerun Cell 6 with `--resume` flag (automatically handled)

### Problem: Different number of images and texts
**Solution**: Check your CSV `dataset_index` column matches HuggingFace indices

---

## 📊 Expected Results (Baselines)

### From RET-CLIP Paper:
- Image→Text R@1: ~80-85%
- Text→Image R@1: ~75-80%

### Your Results Will Depend On:
- ✅ Training epochs (more = better)
- ✅ Prompt quality (your prompts are excellent!)
- ✅ Dataset size (12,989 is good)
- ✅ Model size (ViT-L-14 > ViT-B-16)

---

## 📖 Citation

If you use this training pipeline, please cite the original RET-CLIP paper:

```bibtex
@inproceedings{retclip2024,
  title={RET-CLIP: A Retinal Image Foundation Model Pre-trained with Clinical Diagnostic Reports},
  booktitle={MICCAI},
  year={2024}
}
```

---

## 🤝 Support

Need help?
1. Check `QUICK_START.md` for common issues
2. Review `COLAB_TRAINING_GUIDE.md` for detailed steps
3. See `TEST_EVALUATION_GUIDE.md` for evaluation help

---

## ✨ Summary

You now have:
- ✅ Complete training pipeline with streaming support
- ✅ HuggingFace authentication to avoid rate limits
- ✅ Automatic test set conversion and validation
- ✅ Full evaluation suite with retrieval metrics
- ✅ Per-class analysis and visualizations

Everything is ready for your master's thesis! 🎓🚀
