# ODIR-5K RET-CLIP Unified Pipeline - Completion Guide

## Status

### ✅ Completed Sections (Fully Implemented)

The notebook `ODIR_RETCLIP_UNIFIED_PIPELINE.ipynb` includes:

- **Section 1: Setup & Configuration** - Complete (Cells 1-13)
- **Section 2: Load ODIR-5K Dataset** - Complete (Cells 14-22)
- **Section 3: Generate Clinical Prompts** - Complete (Cells 23-33)
  - ✅ OdirPromptSignature (DSPy signature for metadata)
  - ✅ OdirPromptGenerator (3-prompt generation module)
  - ✅ Test generator on single patient
  - ✅ Retry logic with backoff
  - ✅ **Main generation loop (3 prompts per patient)**
- **Section 4: Preprocess for RET-CLIP** - Core cells complete (Cells 34-40)
  - ✅ Image encoding helper (URL-safe base64)
  - ✅ TSV creation (real binocular pairs)
  - ✅ **JSONL creation with eye_side annotations**

### 📝 Remaining Sections (To Be Added)

The following sections need to be added by adapting code from `RETCLIP_COMPLETE_PIPELINE.ipynb`:

- **Section 4 (continuation)**: Train/test split
- **Section 5**: Build LMDB Database
- **Section 6**: Train RET-CLIP
- **Section 7**: Zero-Shot Evaluation
- **Section 8**: Linear Probing Evaluation
- **Section 9**: Final Report

---

## How to Complete the Remaining Sections

### Section 4 (Continuation): Train/Test Split

**Add after Cell 40**

#### Cell 4.4: Split Data into Train/Test

```python
from sklearn.model_selection import train_test_split

# Split prompts DataFrame (80/20 split)
train_df, test_df = train_test_split(
    prompts_df,
    test_size=0.2,
    random_state=42,
    stratify=prompts_df['left_keywords']  # Stratify by disease if possible
)

print(f"Train patients: {len(train_df)}")
print(f"Test patients: {len(test_df)}")

# Save splits
train_df.to_csv(f"{DRIVE_DATA}/train_patients.csv", index=False)
test_df.to_csv(f"{DRIVE_DATA}/test_patients.csv", index=False)
```

#### Cell 4.5: Create Train TSV/JSONL

**Copy Cell 4.2 and 4.3 code, but:**
- Change `prompts_df` to `train_df`
- Change output paths to `odir_train_imgs.tsv` and `odir_train_texts.jsonl`

#### Cell 4.6: Create Test TSV/JSONL

**Copy Cell 4.2 and 4.3 code, but:**
- Change `prompts_df` to `test_df`
- Change output paths to `odir_test_imgs.tsv` and `odir_test_texts.jsonl`

---

### Section 5: Build LMDB Database

**Source**: Cells 20-23 from `RETCLIP_COMPLETE_PIPELINE.ipynb`

#### Cell 5.1: Build Train LMDB

```python
# Build LMDB for train set
print("="*80)
print("Building LMDB for TRAIN set")
print("="*80)

!python /content/retclip/RET_CLIP/preprocess/build_lmdb_dataset_for_RET-CLIP.py \
    --data_dir {DRIVE_DATA} \
    --splits train \
    --lmdb_dir {DRIVE_LMDB}
```

#### Cell 5.2: Build Test LMDB

```python
# Build LMDB for test set
print("\n" + "="*80)
print("Building LMDB for TEST set")
print("="*80)

!python /content/retclip/RET_CLIP/preprocess/build_lmdb_dataset_for_RET-CLIP.py \
    --data_dir {DRIVE_DATA} \
    --splits test \
    --lmdb_dir {DRIVE_LMDB}
```

#### Cell 5.3: Validate LMDB

**Copy from** Cell 23 in `RETCLIP_COMPLETE_PIPELINE.ipynb` - validates LMDB by reading samples

---

### Section 6: Train RET-CLIP

**Source**: Cells 24-27 from `RETCLIP_COMPLETE_PIPELINE.ipynb`

**Key Adaptations**:
1. Use the ODIR-5K dataset paths
2. Modify text encoder to use `PubMedBERT` instead of Chinese BERT
3. Add tripartite loss handling for eye_side annotations (if custom training code required)

#### Cell 6.1: Training Configuration

**Copy from** Cell 25 in `RETCLIP_COMPLETE_PIPELINE.ipynb`

Already configured in Section 1, just reference:
```python
print("Training Configuration:")
print(f"  Vision Model: {VISION_MODEL}")
print(f"  Text Model: {TEXT_MODEL}")
print(f"  Batch Size: {BATCH_SIZE}")
print(f"  Epochs: {NUM_EPOCHS}")
print(f"  Learning Rate: {LEARNING_RATE}")
```

#### Cell 6.2: Run Training

**Copy from** Cell 26 in `RETCLIP_COMPLETE_PIPELINE.ipynb`

```python
# Set PYTHONPATH for subprocess
import os
os.environ['PYTHONPATH'] = '/content/retclip'

print("\n" + "="*80)
print("Starting RET-CLIP Training")
print("="*80)

!torchrun --nproc_per_node=1 --master_port=29500 \
    /content/retclip/RET_CLIP/training/main.py \
    --train-data {DRIVE_LMDB}/train \
    --batch-size {BATCH_SIZE} \
    --max-epochs {NUM_EPOCHS} \
    --lr {LEARNING_RATE} \
    --warmup {WARMUP_STEPS} \
    --vision-model {VISION_MODEL} \
    --text-model {TEXT_MODEL} \
    --logs {DRIVE_CHECKPOINTS} \
    --name retclip_odir \
    --save-epoch-frequency 1 \
    --skip-aggregate
```

#### Cell 6.3: List Checkpoints

**Copy from** Cell 27 in `RETCLIP_COMPLETE_PIPELINE.ipynb`

---

### Section 7: Zero-Shot Evaluation

**Source**: Cells 32-38 from `RETCLIP_COMPLETE_PIPELINE.ipynb`

**Key Adaptations**:
1. Use ODIR-5K disease classes instead of Peacein classes
2. Use test set prompts from `test_df`

#### Cell 7.1: Load Model

**Copy from** Cell 33, adapt checkpoint path to `retclip_odir`

#### Cell 7.2: Prepare Zero-Shot Prompts

**Adapt from** Cell 34:
```python
# Get unique disease keywords from test set
disease_classes = []
for keywords_str in pd.concat([test_df['left_keywords'], test_df['right_keywords']]).dropna():
    for keyword in str(keywords_str).split(','):
        disease = keyword.strip()
        if disease and disease not in disease_classes:
            disease_classes.append(disease)

print(f"Disease classes: {len(disease_classes)}")
print(disease_classes)

# Create zero-shot prompts (one per disease)
zero_shot_prompts = {}
for disease in disease_classes:
    # Find a good example prompt for this disease
    matching = test_df[
        (test_df['left_keywords'].str.contains(disease, na=False)) |
        (test_df['right_keywords'].str.contains(disease, na=False))
    ]
    if len(matching) > 0:
        zero_shot_prompts[disease] = matching.iloc[0]['prompt_patient']
```

#### Cell 7.3-7.5: Encode Text, Perform Inference, Compute Metrics

**Copy from** Cells 35-37 (minimal adaptation needed)

#### Cell 7.6: Confusion Matrix

**Copy from** Cell 38

---

### Section 8: Linear Probing Evaluation

**Source**: Cells 39-44 from `RETCLIP_COMPLETE_PIPELINE.ipynb`

**No major adaptations needed** - copy cells 40-44 directly:
- Extract train features
- Extract test features
- Train logistic regression
- Evaluate and plot confusion matrix

---

### Section 9: Final Report

**Source**: Cells 45-48 from `RETCLIP_COMPLETE_PIPELINE.ipynb`

#### Cell 9.1: Comparison Table

**Copy from** Cell 46

#### Cell 9.2: Generate Final Report

**Adapt from** Cell 47 - update to mention ODIR-5K dataset and research contribution

```python
import datetime

final_report = f"""
{'='*80}
ODIR-5K RET-CLIP TRAINING & EVALUATION REPORT
{'='*80}

Date: {datetime.datetime.now()}

RESEARCH CONTRIBUTION
{'-'*80}
"English BERT Embeddings for Binocular Retinal Image-Text Alignment"

- First validation of RET-CLIP architecture on English clinical text
- Cross-lingual transfer validation (original: Chinese → our work: English)
- Real binocular fundus images from ODIR-5K (not duplicated monocular)
- Three-prompt generation strategy for tripartite contrastive learning

DATASET STATISTICS
{'-'*80}
Dataset: ODIR-5K (Ocular Disease Intelligent Recognition)
Train samples: {len(train_df)} patients ({len(train_df) * 3} text-image pairs)
Test samples: {len(test_df)} patients ({len(test_df) * 3} text-image pairs)
Disease keywords: {len(disease_classes)}

TRAINING CONFIGURATION
{'-'*80}
Vision Model: {VISION_MODEL}
Text Model: {TEXT_MODEL} (PubMedBERT - Medical Domain)
Batch Size: {BATCH_SIZE}
Epochs: {NUM_EPOCHS}
Learning Rate: {LEARNING_RATE}
Image Size: {IMAGE_SIZE}x{IMAGE_SIZE}

EVALUATION RESULTS
{'-'*80}

1. Zero-Shot Evaluation (Vision-Language Alignment)
   Accuracy: {zeroshot_acc * 100:.2f}%
   Macro F1: {report['macro avg']['f1-score'] * 100:.2f}%
   Weighted F1: {report['weighted avg']['f1-score'] * 100:.2f}%

2. Linear Probing (Feature Quality)
   Accuracy: {linear_probe_acc * 100:.2f}%
   Macro F1: {linear_report['macro avg']['f1-score'] * 100:.2f}%
   Weighted F1: {linear_report['weighted avg']['f1-score'] * 100:.2f}%

ARTIFACTS SAVED TO GOOGLE DRIVE
{'-'*80}
{DRIVE_BASE}/
├── prompts/
│   └── odir_retclip_prompts.csv (3 prompts per patient)
├── data/
│   ├── odir_train_imgs.tsv (real binocular pairs)
│   ├── odir_test_imgs.tsv
│   ├── odir_train_texts.jsonl (with eye_side annotations)
│   └── odir_test_texts.jsonl
├── lmdb/
│   ├── train/
│   └── test/
├── checkpoints/
│   └── retclip_odir/checkpoints/epoch_{NUM_EPOCHS}.pt
└── results/
    ├── odir_dataset_statistics.png
    ├── zeroshot_metrics.json
    ├── zeroshot_confusion_matrix.png
    ├── linear_probe_metrics.json
    ├── linear_probe_confusion_matrix.png
    └── final_report.txt

{'='*80}
END OF REPORT
{'='*80}
"""

print(final_report)

# Save report
with open(f"{DRIVE_RESULTS}/final_report.txt", 'w') as f:
    f.write(final_report)

print(f"\n✅ Final report saved to {DRIVE_RESULTS}/final_report.txt")
```

#### Cell 9.3: List All Artifacts

**Copy from** Cell 48

---

## Quick Copy-Paste Workflow

### Option 1: Manual Cell Addition

1. Open both notebooks side-by-side:
   - `ODIR_RETCLIP_UNIFIED_PIPELINE.ipynb` (destination)
   - `RETCLIP_COMPLETE_PIPELINE.ipynb` (source)

2. For each remaining section (5-9):
   - Copy cells from source notebook
   - Paste into destination notebook
   - Apply adaptations listed above (mostly path changes)

### Option 2: Programmatic Completion

**Create a Python script to merge notebooks:**

```python
import json

# Load both notebooks
with open('ODIR_RETCLIP_UNIFIED_PIPELINE.ipynb', 'r') as f:
    unified_nb = json.load(f)

with open('RETCLIP_COMPLETE_PIPELINE.ipynb', 'r') as f:
    pipeline_nb = json.load(f)

# Copy cells 20-48 from pipeline notebook (Sections 5-9)
cells_to_copy = pipeline_nb['cells'][20:48]

# Adapt cell content (replace paths/variables)
adaptations = {
    'DRIVE_CHECKPOINTS}/retclip_fundus': 'DRIVE_CHECKPOINTS}/retclip_odir',
    'retclip_fundus': 'retclip_odir',
    'Peacein': 'ODIR-5K',
    # Add more substitutions as needed
}

for cell in cells_to_copy:
    if cell['cell_type'] == 'code':
        source = cell['source']
        if isinstance(source, list):
            source = ''.join(source)
        for old, new in adaptations.items():
            source = source.replace(old, new)
        cell['source'] = [source]
    unified_nb['cells'].append(cell)

# Save completed notebook
with open('ODIR_RETCLIP_UNIFIED_PIPELINE_COMPLETE.ipynb', 'w') as f:
    json.dump(unified_nb, f, indent=1)

print("✅ Complete notebook created!")
```

---

## Key Differences from Original Pipeline

### What's New/Different in ODIR Unified Pipeline:

| Aspect | Original (RETCLIP_COMPLETE_PIPELINE.ipynb) | ODIR Unified Pipeline |
|--------|-------------------------------------------|----------------------|
| **Dataset** | Peacein (single eye, duplicated for binocular) | ODIR-5K (real paired left/right eyes) |
| **Prompts** | Pre-generated CSV (single prompt/image) | **Generated in-pipeline (3 prompts/patient)** |
| **Prompt Source** | Generic retinal disease descriptions | **Metadata-aware (age, sex, keywords)** |
| **Text Encoder** | Chinese BERT models | **English BERT (PubMedBERT, BERT-base, BioBERT)** |
| **JSONL Format** | Standard | **Extended with eye_side field** |
| **TSV Images** | Duplicated monocular | **Real binocular pairs** |
| **Sections 1-4** | Separate notebook required | **Integrated prompt generation** |

### What Stays the Same:

- ✅ LMDB building process (Section 5)
- ✅ Training loop structure (Section 6)
- ✅ Zero-shot evaluation logic (Section 7)
- ✅ Linear probing workflow (Section 8)
- ✅ Report generation (Section 9, with text updates)

---

## Validation Checklist

After completing all sections, verify:

- [ ] TSV files have 3 columns (patient_id, left_b64, right_b64)
- [ ] JSONL files have eye_side field with values: "left", "right", "both"
- [ ] Each patient has exactly 3 JSONL entries
- [ ] LMDB can be opened and read
- [ ] Training completes without errors
- [ ] Model checkpoint exists at expected path
- [ ] Zero-shot evaluation runs on test set
- [ ] Linear probing trains and evaluates
- [ ] Final report includes all metrics

---

## Estimated Completion Time

- **Manual cell addition**: ~1-2 hours (copying + adapting)
- **Programmatic merge**: ~30 minutes (script + validation)
- **Testing in TEST_MODE**: ~2-3 hours runtime
- **Full pipeline (FULL_MODE)**: ~18-24 hours runtime

---

## Support

If you encounter issues:

1. Check [FIX_CHINESE_BERT_ISSUE.md](FIX_CHINESE_BERT_ISSUE.md) for BERT configuration problems
2. Verify ODIR-5K images are downloaded and extracted correctly
3. Ensure API keys (HF_TOKEN, OPENROUTER_API_KEY) are set in Colab secrets
4. Review [QUICK_START.md](QUICK_START.md) for general troubleshooting

---

## Next Steps

1. Complete remaining sections (5-9) using this guide
2. Run notebook in TEST_MODE (100 patients, 2 epochs) to validate
3. Review metrics and debug any issues
4. Switch to FULL_MODE for complete training on all 5,000 patients
5. Document results and create research paper draft

Good luck! 🚀
