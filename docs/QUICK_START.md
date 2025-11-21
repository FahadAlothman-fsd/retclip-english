# RET-CLIP Training - Quick Start

## 🚀 What I Created For You

I've prepared a complete pipeline to train RET-CLIP on Google Colab with your data:

### Files Created:
1. **`scripts/prepare_retclip_data_streaming.py`** - Standalone preprocessing script with HF authentication
2. **`colab_retclip_training.py`** - Complete automated Colab training script
3. **`COLAB_TRAINING_GUIDE.md`** - Step-by-step Colab notebook cells (RECOMMENDED)

---

## ✅ Recommended Approach: Use the Training Guide

**Open `COLAB_TRAINING_GUIDE.md`** and copy each cell into a new Google Colab notebook.

### Why this approach?
- ✅ **Streaming support** - Avoids rate limits like your notebook
- ✅ **HuggingFace authentication** - Uses your HF_TOKEN
- ✅ **Checkpoint support** - Resume if interrupted
- ✅ **Step-by-step** - Easy to debug each stage

---

## 📋 Prerequisites

Before starting, make sure you have:

1. **Google Colab Pro/Pro+** (for A100 GPU)
2. **HuggingFace Token**: https://huggingface.co/settings/tokens
3. **Your CSV files** on Google Drive:
   - `content/retclip_prompts_full.csv` (12,989 samples)
   - `content/retclip_prompts_test.csv` (3,253 samples)

---

## 🎯 Quick Start (5 Minutes)

### Step 1: Open Google Colab
Go to https://colab.research.google.com/

### Step 2: Select A100 GPU
- Click **Runtime** → **Change runtime type**
- Select **A100 GPU** (requires Colab Pro+)

### Step 3: Copy the Training Guide
- Open `COLAB_TRAINING_GUIDE.md`
- Copy each cell into your Colab notebook
- **Update Cell 3** with your HF_TOKEN and file paths

### Step 4: Run All Cells
- Click **Runtime** → **Run all**
- Training will start automatically!

---

## 📊 What Happens

```
Cell 1:  Install dependencies (~2-3 min)
Cell 2:  Mount Google Drive
Cell 3:  Set configuration (UPDATE THIS!)
Cell 4:  Clone repository
Cell 5:  Create preprocessing script
Cell 6:  Download images from HuggingFace (STREAMING) (~2-3 hours for full dataset)
Cell 7:  Build LMDB dataset (~10-15 min)
Cell 8:  Download pretrained weights (~2 min)
Cell 9:  Start training (~8-12 hours for full dataset)
Cell 10: Monitor logs (optional)
Cell 11: Download trained model
```

---

## ⚙️ Key Configuration Options

In **Cell 3** of the guide, you can adjust:

```python
# Training settings
BATCH_SIZE = 64      # Reduce to 32 if OOM errors
MAX_EPOCHS = 10      # Number of training epochs
USE_FLASH_ATTN = True  # Speeds up training ~2x on A100
```

---

## 🔍 Important Notes

### About Your Data
- Your CSV files map to HuggingFace dataset indices
- The script uses **streaming** to avoid rate limits (like your prompt generation notebook)
- **Authenticates with HF_TOKEN** before downloading
- Uses same image for both eyes (since dataset has single fundus images)

### Training Architecture
- **Vision**: ViT-B-16 (can change to ViT-L-14 for better performance)
- **Text**: Chinese RoBERTa
- **Dual eye inputs**: Left and right (currently using same image)
- **CLIP-style contrastive learning**

### Expected Results
- **Checkpoints**: Saved every 2 epochs
- **Validation**: Runs every epoch
- **Logs**: Real-time training metrics

---

## 📁 Output Structure

After training completes:

```
/content/logs/retclip_training/
├── checkpoints/
│   ├── epoch2.pt
│   ├── epoch4.pt
│   ├── ...
│   └── epoch_latest.pt
├── out_YYYY-MM-DD-HH-MM-SS.log
└── params_YYYY-MM-DD-HH-MM-SS.txt
```

---

## 🐛 Common Issues

### 1. Rate Limiting from HuggingFace
**Solution**: Make sure `HF_TOKEN` is set in Cell 3

### 2. Out of Memory
**Solution**: Reduce `BATCH_SIZE` to 32 or 16

### 3. Dataset Index Mismatch
**Solution**: Your CSV `dataset_index` column maps directly to HuggingFace dataset indices

### 4. Connection Timeout
**Solution**: The preprocessing script has checkpoint support - rerun Cell 6 to resume

---

## 📝 Example: Updating Cell 3

```python
# ============================================================================
# CONFIGURATION - UPDATE THESE!
# ============================================================================

# Your HuggingFace token (from https://huggingface.co/settings/tokens)
HF_TOKEN = "hf_YOUR_TOKEN_HERE"  # Replace with your actual token

# Paths to your CSV files on Google Drive
CSV_TRAIN = "/content/drive/MyDrive/retclip_data/retclip_prompts_full.csv"
CSV_TEST = "/content/drive/MyDrive/retclip_data/retclip_prompts_test.csv"

# HuggingFace dataset (same as your notebook)
HF_DATASET = "Peacein/color-fundus-eye"

# Training settings
BATCH_SIZE = 64  # A100 can handle this
MAX_EPOCHS = 10
USE_FLASH_ATTN = True
```

---

## 🎓 Next Steps After Training

1. **Download your model**:
   ```python
   !cp -r /content/logs /content/drive/MyDrive/retclip_results/
   ```

2. **Test zero-shot classification** (see original RET-CLIP paper)

3. **Fine-tune on downstream tasks**

---

## ❓ Need Help?

Check these files:
- **Full details**: `COLAB_TRAINING_GUIDE.md`
- **Standalone script**: `scripts/prepare_retclip_data_streaming.py`
- **Training params**: `retclip/RET_CLIP/training/params.py`

---

## 🔗 Key Differences from Original RET-CLIP

1. **Single image per sample**: Original expects separate left/right eye images
   - Solution: We duplicate the image for both eyes
   - For production: Modify dataset to have actual paired images

2. **CSV-based indexing**: Original uses TSV/JSONL directly
   - Solution: Our preprocessing converts CSV → TSV/JSONL → LMDB

3. **Streaming download**: Avoids loading entire dataset in memory
   - Essential for Colab's limited RAM

---

Good luck with your training! 🚀
