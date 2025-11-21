# RET-CLIP Training on Google Colab with A100

Complete step-by-step guide to train RET-CLIP on Google Colab.

---

## Prerequisites

1. **Google Colab Pro/Pro+** (for A100 GPU access)
2. **HuggingFace Token**: Get from https://huggingface.co/settings/tokens
3. **Your CSV files** uploaded to Google Drive:
   - `retclip_prompts_full.csv` (training data)
   - `retclip_prompts_test.csv` (test data)

---

## Step-by-Step Instructions

### Cell 1: Check GPU and Install Dependencies

```python
# Check GPU
!nvidia-smi

# Install PyTorch for CUDA 11.8
!pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
!pip install -q transformers datasets huggingface_hub pandas tqdm lmdb==1.3.0 timm ftfy regex einops scikit-image scikit-learn

# Optional: Install flash-attention for A100 (speeds up training ~2x)
!pip install -q flash-attn --no-build-isolation

print("✅ Setup complete!")
```

---

### Cell 2: Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

---

### Cell 3: Set Configuration

```python
# ============================================================================
# CONFIGURATION - UPDATE THESE!
# ============================================================================

# Your HuggingFace token
HF_TOKEN = "hf_YOUR_TOKEN_HERE"

# Paths to your CSV files on Google Drive
CSV_TRAIN = "/content/drive/MyDrive/retclip_data/retclip_prompts_full.csv"
CSV_TEST = "/content/drive/MyDrive/retclip_data/retclip_prompts_test.csv"

# HuggingFace dataset
HF_DATASET = "Peacein/color-fundus-eye"

# Training settings
BATCH_SIZE = 64  # A100 can handle this
MAX_EPOCHS = 10
USE_FLASH_ATTN = True  # Set False if flash-attn install failed

print("✅ Configuration set!")
```

---

### Cell 4: Clone Repository

```python
import os

# Clone your RET-CLIP repository
if not os.path.exists("/content/retclip"):
    !git clone https://github.com/YOUR_USERNAME/retclip.git /content/retclip
    print("✅ Repository cloned")
else:
    print("✅ Repository already exists")

%cd /content/retclip
```

---

### Cell 5: Data Preprocessing Script

```python
# Save the preprocessing script
preprocessing_code = """
import os
import pandas as pd
import base64
from io import BytesIO
from PIL import Image
from tqdm import tqdm
import json
import time
from huggingface_hub import login
from datasets import load_dataset

def image_to_base64(image, size=224):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize((size, size), Image.BICUBIC)
    buffered = BytesIO()
    image.save(buffered, format="JPEG", quality=95)
    return base64.urlsafe_b64encode(buffered.getvalue()).decode()

def prepare_data(csv_path, split, output_dir, hf_token, hf_dataset):
    # Authenticate
    login(token=hf_token)
    print(f"✅ Authenticated with HuggingFace")

    # Load CSV
    df = pd.read_csv(csv_path)
    print(f"📊 Loaded {len(df)} samples from CSV")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    tsv_path = os.path.join(output_dir, f"{split}_imgs.tsv")
    jsonl_path = os.path.join(output_dir, f"{split}_texts.jsonl")

    # Load HuggingFace dataset in STREAMING mode
    print(f"📥 Loading dataset in streaming mode...")
    dset = load_dataset(hf_dataset, split=split, streaming=True)
    dataset_iter = iter(dset)

    # Open output files
    tsv_file = open(tsv_path, 'w', encoding='utf-8')
    jsonl_file = open(jsonl_path, 'w', encoding='utf-8')

    current_idx = 0
    unique_patients = set()

    # Process each sample
    for csv_idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {split}"):
        dataset_index = int(row['dataset_index'])
        prompt = row['prompt']
        patient_id = f"patient_{dataset_index:06d}"

        try:
            # Stream forward to correct index
            while current_idx < dataset_index:
                next(dataset_iter)
                current_idx += 1

            # Get the sample
            sample = next(dataset_iter)
            current_idx += 1

            # Process image
            if 'image' in sample:
                image = sample['image']
                img_b64 = image_to_base64(image, 224)

                # Write to TSV (use same image for both eyes)
                if patient_id not in unique_patients:
                    tsv_file.write(f"{patient_id}\\t{img_b64}\\t{img_b64}\\n")
                    unique_patients.add(patient_id)

                # Write to JSONL
                annotation = {
                    "text_id": csv_idx,
                    "text": prompt,
                    "image_ids": [patient_id]
                }
                jsonl_file.write(json.dumps(annotation, ensure_ascii=False) + '\\n')

                # Periodic flush and delay
                if (csv_idx + 1) % 100 == 0:
                    tsv_file.flush()
                    jsonl_file.flush()

                time.sleep(0.05)  # Small delay to avoid rate limits

        except Exception as e:
            print(f"Error at index {dataset_index}: {e}")
            continue

    tsv_file.close()
    jsonl_file.close()

    print(f"✅ Created {tsv_path} ({len(unique_patients)} patients)")
    print(f"✅ Created {jsonl_path} ({len(df)} text-image pairs)")

if __name__ == "__main__":
    import sys
    prepare_data(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
"""

with open("/content/prepare_data.py", "w") as f:
    f.write(preprocessing_code)

print("✅ Preprocessing script created!")
```

---

### Cell 6: Run Data Preprocessing

```python
# Process training data
print("="*80)
print("Processing TRAIN split...")
print("="*80)
!python /content/prepare_data.py "{CSV_TRAIN}" "train" "/content/data" "{HF_TOKEN}" "{HF_DATASET}"

# Process test data
print("\n" + "="*80)
print("Processing TEST split...")
print("="*80)
!python /content/prepare_data.py "{CSV_TEST}" "test" "/content/data" "{HF_TOKEN}" "{HF_DATASET}"

print("\n✅ Data preprocessing complete!")
```

---

### Cell 7: Build LMDB Dataset

```python
# Convert TSV/JSONL to LMDB format
!python retclip/RET_CLIP/preprocess/build_lmdb_dataset_for_RET-CLIP.py \
    --data_dir /content/data \
    --splits train,test \
    --lmdb_dir /content/lmdb

print("✅ LMDB dataset created!")
```

---

### Cell 8: Download Pretrained Weights

```python
# Download OpenAI CLIP ViT-B-16 weights
print("📥 Downloading CLIP weights...")
!wget -q https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt \
    -O /content/ViT-B-16.pt

# Download Chinese RoBERTa weights
print("📥 Downloading Chinese RoBERTa...")
!git clone -q https://huggingface.co/hfl/chinese-roberta-wwm-ext /content/chinese-roberta-wwm-ext

print("✅ Pretrained weights downloaded!")
```

---

### Cell 9: Start Training

```python
# Build training command
cmd = f"""
python -m torch.distributed.launch \
    --nproc_per_node=1 \
    --master_port=12345 \
    retclip/RET_CLIP/training/main.py \
    --train-data /content/lmdb/train \
    --val-data /content/lmdb/test \
    --vision-model ViT-B-16 \
    --text-model RoBERTa-wwm-ext-base-chinese \
    --clip-weight-path /content/ViT-B-16.pt \
    --bert-weight-path /content/chinese-roberta-wwm-ext \
    --batch-size {BATCH_SIZE} \
    --valid-batch-size {BATCH_SIZE} \
    --max-epochs {MAX_EPOCHS} \
    --warmup 100 \
    --lr 5e-4 \
    --wd 0.2 \
    --context-length 77 \
    --precision amp \
    --use-augment \
    --logs /content/logs \
    --name retclip_training \
    --save-epoch-frequency 2 \
    --valid-epoch-interval 1 \
    --num-workers 2 \
    --valid-num-workers 1 \
    --log-interval 10 \
    {"--use-flash-attention" if USE_FLASH_ATTN else ""}
"""

print("🚀 Starting training...")
print("="*80)
!{cmd}
```

---

### Cell 10: Monitor Training (Optional)

```python
# View training logs in real-time
!tail -f /content/logs/retclip_training/out_*.log
```

---

### Cell 11: Download Trained Model

```python
# After training completes, download your model
from google.colab import files

# Find latest checkpoint
import glob
checkpoints = sorted(glob.glob("/content/logs/retclip_training/checkpoints/epoch*.pt"))
latest_checkpoint = checkpoints[-1] if checkpoints else None

if latest_checkpoint:
    print(f"Downloading: {latest_checkpoint}")
    files.download(latest_checkpoint)
else:
    print("No checkpoints found!")
```

---

## Expected Training Time

- **100 samples**: ~5-10 minutes
- **1,000 samples**: ~1 hour
- **12,989 samples (full dataset)**: ~8-12 hours

---

## Troubleshooting

### Out of Memory (OOM)
- Reduce `BATCH_SIZE` to 32 or 16
- Use `ViT-B-16` instead of `ViT-L-14`

### Rate Limiting from HuggingFace
- Increase delay in Cell 5: change `time.sleep(0.05)` to `time.sleep(0.2)`
- Ensure `HF_TOKEN` is set correctly

### Flash Attention Install Fails
- Set `USE_FLASH_ATTN = False` in Cell 3
- Training will be slower but will work

### Connection Timeout
- Add `--resume` flag to training command to resume from checkpoints

---

## Key Files

After training, you'll have:
- **Checkpoints**: `/content/logs/retclip_training/checkpoints/epoch{N}.pt`
- **Logs**: `/content/logs/retclip_training/out_*.log`
- **Params**: `/content/logs/retclip_training/params_*.txt`

Download these to Google Drive:
```python
!cp -r /content/logs /content/drive/MyDrive/retclip_training_results/
```

---

## Next Steps

After training:
1. **Evaluate** your model on test set
2. **Extract features** using `eval/extract_features.py`
3. **Zero-shot evaluation** using `eval/zeroshot_evaluation.py`

Good luck with your training! 🚀
