# RET-CLIP Deployment Guide

## ✅ Setup Complete!

All fixes have been applied to `temp_retclip/` and the Colab notebook is ready to use.

---

## 🚀 How to Run in Google Colab

### Step 1: Push Changes to GitHub

First, commit and push all the fixes to your GitHub repo:

```bash
cd /home/gondilf/Desktop/projects/masters/retclip

# Add all files
git add .

# Commit
git commit -m "Add complete RET-CLIP pipeline with English BERT support

- Fixed RET-CLIP in temp_retclip/ with all necessary modifications
- Added 3 English BERT configs (PubMedBERT, BERT-base, BioBERT)
- Updated choices arrays in training and eval scripts
- Fixed DDP checkpoint loading
- Complete Colab notebook with text encoder comparison
- Validation scripts to verify all fixes"

# Push
git push origin main
```

### Step 2: Upload Notebook to Colab

1. Go to https://colab.research.google.com/
2. Click **File → Upload notebook**
3. Upload: `RETCLIP_COMPLETE_PIPELINE.ipynb`

### Step 3: Upload CSV Files to Google Drive

Upload these files to your Google Drive:
- `retclip_prompts_full.csv` (training data)
- `retclip_prompts_test.csv` (test data)

Place them in: `Google Drive/RET-CLIP/`

### Step 4: Set HuggingFace Token

In cell-10, set your HuggingFace token:
```python
HF_TOKEN = "hf_your_token_here"  # Get from https://huggingface.co/settings/tokens
```

### Step 5: Select GPU

In Colab:
1. Click **Runtime → Change runtime type**
2. Select **A100 GPU**
3. Click **Save**

### Step 6: Run the Pipeline!

Execute cells sequentially:
- **Cells 1-8**: Setup (5 min)
- **Cells 9-19**: Data preprocessing (2-3 hours)
- **Cells 20-23**: LMDB building (30 min)
- **Cells 24-27**: RET-CLIP training (6-8 hours)
- **Cells 28-39**: Evaluation (1 hour)
- **Cells 40-49**: Final report

**Total time: ~9-12 hours on A100 GPU**

---

## 📋 What Gets Cloned in Colab

When cell-7 runs, it:
1. Clones `https://github.com/FahadAlothman-fsd/retclip-english.git`
2. Copies `temp_retclip/` → `/content/retclip`
3. Uses the fixed RET-CLIP code for training

---

## 🔧 Applied Fixes

The `temp_retclip/` directory contains:

### 1. English BERT Configurations
- `RET_CLIP/clip/model_configs/microsoft-BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext.json`
- `RET_CLIP/clip/model_configs/bert-base-uncased.json`
- `RET_CLIP/clip/model_configs/dmis-lab-biobert-base-cased-v1.1.json`

### 2. Updated Python Files
- `RET_CLIP/training/params.py` - Added English BERT to choices
- `RET_CLIP/training/main.py` - Fixed DDP checkpoint loading
- `RET_CLIP/eval/extract_features_onnx.py` - Added English BERT to choices
- `RET_CLIP/eval/extract_features_tensorrt.py` - Added English BERT to choices

---

## 🧪 Local Testing (Optional)

Before running in Colab, validate locally:

```bash
cd /home/gondilf/Desktop/projects/masters/retclip
python3 validate_fixes.py
```

Expected output: **14/14 checks passed ✅**

---

## 📊 Text Encoder Comparison (Optional)

To compare different BERT models:

1. In cell-31, set:
   ```python
   RUN_TEXT_ENCODER_COMPARISON = True
   ```

2. This will train 3 separate models:
   - PubMedBERT (medical domain)
   - BERT-base (general English)
   - BioBERT (biomedical domain)

3. Runtime: ~18-24 hours total

4. Results saved to: `Google Drive/RET-CLIP/results/encoder_comparison_table.csv`

---

## 📁 Output Artifacts

All saved to: `Google Drive/RET-CLIP/`

```
RET-CLIP/
├── data/
│   ├── train_imgs.tsv
│   ├── train_texts.jsonl
│   ├── test_imgs.tsv
│   └── test_texts.jsonl
├── lmdb/
│   ├── train/
│   └── test/
├── checkpoints/
│   └── epoch_10.pt
└── results/
    ├── zeroshot_metrics.json
    ├── zeroshot_confusion_matrix.png
    ├── linear_probe_metrics.json
    ├── linear_probe_confusion_matrix.png
    ├── linear_probe_classifier.pkl
    ├── train_features.npy
    ├── test_features.npy
    ├── metrics_comparison.csv
    └── final_report.txt
```

---

## ⚠️ Important Notes

1. **Runtime**: Plan for 9-12 hours on A100 GPU
2. **Costs**: Colab A100 requires Colab Pro (~$10/month)
3. **Checkpoints**: Everything auto-saves to Google Drive
4. **Resume**: If interrupted, most steps can resume from checkpoints
5. **HF Token**: Required to download images from HuggingFace

---

## 🐛 Troubleshooting

### "No module named 'RET_CLIP'"
- Make sure cell-7 ran successfully
- Check `/content/retclip` exists
- Verify `sys.path` includes `/content/retclip`

### "Checkpoint not found"
- Check Google Drive mounted: `/content/drive/MyDrive/`
- Verify path: `/content/drive/MyDrive/RET-CLIP/checkpoints/`
- Training completed all epochs

### "HuggingFace rate limit"
- Set `HF_TOKEN` in cell-10
- Authenticate: https://huggingface.co/settings/tokens

### "CUDA out of memory"
- Reduce `BATCH_SIZE` in cell-25 (try 64 or 32)
- Use smaller model or fewer epochs

---

## ✅ Success Criteria

After completion, you should have:
- ✅ Trained RET-CLIP model (`epoch_10.pt`)
- ✅ Zero-shot evaluation results
- ✅ Linear probe classifier
- ✅ Confusion matrices
- ✅ Final report with metrics

---

**Ready to deploy!** 🚀

Just push to GitHub and run in Colab!
