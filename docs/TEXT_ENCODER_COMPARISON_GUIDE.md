# Text Encoder Comparison Guide

This guide explains how to compare different text encoders in both TEST and FULL training modes.

---

## 🎯 Overview

The pipeline automatically compares **3 text encoders** when `RUN_TEXT_ENCODER_COMPARISON = True`:

1. **PubMedBERT** - Medical domain-specific (trained on PubMed abstracts)
2. **BERT-base** - General English (Wikipedia + BookCorpus)
3. **BioBERT** - Biomedical domain (PubMed + PMC articles)

This helps answer: **Which text encoder works best for retinal disease classification?**

---

## ⚙️ Configuration (cell-31)

```python
RUN_TEXT_ENCODER_COMPARISON = True  # Set to True to enable comparison
```

**Default:** `True` (comparison enabled by default)

---

## 🧪 TEST MODE (Quick Validation)

### Configuration:
```python
TEST_MODE = True  # In cell-10
RUN_TEXT_ENCODER_COMPARISON = True  # In cell-31
```

### What Happens:
- **Trains 3 separate RET-CLIP models** (one per encoder)
- Each model trains on **100 samples for 2 epochs**
- Each model is evaluated using **zero-shot classification**

### Runtime (A100 GPU):
| Step | Time per Encoder | Total Time (3 encoders) |
|------|-----------------|------------------------|
| Training | ~30 min | ~1.5 hours |
| Evaluation | ~5 min | ~15 min |
| **Total** | **~35 min** | **~2 hours** |

### Output:
```
Text Encoder Comparison Summary
================================================================================
| Text Encoder | Accuracy | Macro F1 | Weighted F1 |
|--------------|----------|----------|-------------|
| PubMedBERT   | 85.23%   | 83.45%   | 84.92%     |
| BERT-base    | 78.34%   | 76.21%   | 77.89%     |
| BioBERT      | 82.11%   | 80.34%   | 81.56%     |

🏆 Best Text Encoder: PubMedBERT
   Accuracy: 85.23%
```

---

## 🚀 FULL MODE (Production Training)

### Configuration:
```python
TEST_MODE = False  # In cell-10
RUN_TEXT_ENCODER_COMPARISON = True  # In cell-31 (already set!)
```

### What Happens:
- **Trains 3 separate RET-CLIP models** (one per encoder)
- Each model trains on **12,989 samples for 10 epochs**
- Each model is evaluated using **zero-shot classification**

### Runtime (A100 GPU):
| Step | Time per Encoder | Total Time (3 encoders) |
|------|-----------------|------------------------|
| Training | ~6-8 hours | ~18-24 hours |
| Evaluation | ~15 min | ~45 min |
| **Total** | **~6-8 hours** | **~18-25 hours** |

### Output:
Same comparison table as TEST_MODE, but with **production-quality results** from full training.

---

## 📂 Saved Artifacts

### Checkpoints (per encoder):
```
/content/drive/MyDrive/RET-CLIP/checkpoints/
├── pubmedbert/
│   ├── epoch_1.pt
│   ├── epoch_2.pt
│   └── epoch_{NUM_EPOCHS}.pt
├── bert_base/
│   ├── epoch_1.pt
│   └── epoch_{NUM_EPOCHS}.pt
└── biobert/
    ├── epoch_1.pt
    └── epoch_{NUM_EPOCHS}.pt
```

### Results:
```
/content/drive/MyDrive/RET-CLIP/results/
├── encoder_training_results.json        # Training status for all encoders
├── encoder_comparison_metrics.json      # Detailed metrics for all encoders
├── encoder_comparison_table.csv         # Summary table (CSV)
└── [per-encoder evaluation files]
```

---

## 🔬 How It Works

### Training Phase (cell-30):
```python
for encoder_config in TEXT_ENCODERS:
    encoder_name = encoder_config['name']
    encoder_model_id = encoder_config['model_id']

    # Train RET-CLIP with this text encoder
    !python /content/retclip/RET_CLIP/training/main.py \
        --text-model {encoder_model_id} \
        --epochs {NUM_EPOCHS} \
        --batch-size {BATCH_SIZE} \
        ...

    # Save checkpoint
    encoder_results[encoder_name] = {
        'checkpoint': f"{DRIVE_CHECKPOINTS}/{encoder_name}/epoch_{NUM_EPOCHS}.pt",
        'model_id': encoder_model_id,
        'status': 'success'
    }
```

### Evaluation Phase (cell-28):
```python
for encoder_name, encoder_info in encoder_results.items():
    # Load model with this text encoder
    model = load_retclip_model(encoder_info['checkpoint'])

    # Perform zero-shot evaluation
    accuracy, f1_scores = evaluate_zero_shot(model, test_df)

    # Store results
    encoder_comparison_metrics[encoder_name] = {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1
    }
```

---

## 📊 Interpreting Results

### What to Look For:

1. **Accuracy**: Overall correctness (higher is better)
2. **Macro F1**: Average F1 across all classes (handles class imbalance)
3. **Weighted F1**: F1 weighted by class support

### Expected Findings:

- **PubMedBERT** should perform best (medical domain-specific)
- **BioBERT** should be second (biomedical domain)
- **BERT-base** should be third (general English, no medical knowledge)

### Research Value:

This quantifies the benefit of **domain-specific language models** for medical vision-language tasks!

---

## 🎛️ Disabling Comparison

If you want to **skip the comparison** and only train with PubMedBERT:

```python
RUN_TEXT_ENCODER_COMPARISON = False  # In cell-31
```

This will:
- Train only 1 model (PubMedBERT)
- Save **~1 hour in TEST_MODE**
- Save **~12-16 hours in FULL_MODE**

---

## ✅ Summary

| Mode | RUN_TEXT_ENCODER_COMPARISON | Models Trained | Runtime (A100) |
|------|----------------------------|----------------|----------------|
| **TEST** | `True` | 3 encoders | ~2 hours |
| **TEST** | `False` | 1 encoder (PubMedBERT) | ~30 min |
| **FULL** | `True` | 3 encoders | ~18-25 hours |
| **FULL** | `False` | 1 encoder (PubMedBERT) | ~6-8 hours |

**Recommendation:**
- Run **TEST_MODE** with comparison first (~2 hours) to validate setup
- Then run **FULL_MODE** with comparison (~20-25 hours) for production results

---

**Ready to compare text encoders!** 🚀
