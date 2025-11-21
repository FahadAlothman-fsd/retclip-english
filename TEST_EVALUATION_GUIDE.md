# RET-CLIP Test Set Evaluation Guide

After training your RET-CLIP model, here's how to properly evaluate it on your test set.

---

## What Happens During Training vs After Training

### ✅ During Training (Automatic):
- **Validation loss** is computed every epoch using `--val-data /content/lmdb/test`
- This gives you basic metrics like:
  - Contrastive loss on test set
  - Image-to-text retrieval accuracy
  - Text-to-image retrieval accuracy

### 🎯 After Training (Manual Evaluation):
You need to run separate evaluation scripts to get:
- **Feature extraction** (image and text embeddings)
- **Retrieval metrics** (Recall@1, Recall@5, Recall@10)
- **Classification accuracy** (if doing zero-shot classification)

---

## Evaluation Pipeline (3 Steps)

### Step 1: Extract Features

Extract image and text features from your trained model.

```python
# Cell 1: Extract Image Features
!python retclip/RET_CLIP/eval/extract_features.py \
    --extract-image-feats \
    --image-data /content/lmdb/test/imgs \
    --image-feat-output-path /content/test_imgs_features.pt \
    --resume /content/logs/retclip_training/checkpoints/epoch10.pt \
    --vision-model ViT-B-16 \
    --text-model RoBERTa-wwm-ext-base-chinese \
    --img-batch-size 64 \
    --precision amp

print("✅ Image features extracted!")
```

```python
# Cell 2: Extract Text Features
!python retclip/RET_CLIP/eval/extract_features.py \
    --extract-text-feats \
    --text-data /content/data/test_texts.jsonl \
    --text-feat-output-path /content/test_texts_features.pt \
    --resume /content/logs/retclip_training/checkpoints/epoch10.pt \
    --vision-model ViT-B-16 \
    --text-model RoBERTa-wwm-ext-base-chinese \
    --text-batch-size 64 \
    --context-length 77 \
    --precision amp

print("✅ Text features extracted!")
```

---

### Step 2: Compute Retrieval Metrics

Now compute image-to-text and text-to-image retrieval performance.

```python
# Cell 3: Evaluate Retrieval Performance
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load features
img_feats = torch.load('/content/test_imgs_features.pt')
txt_feats = torch.load('/content/test_texts_features.pt')

# Convert to numpy
img_feats_np = img_feats.cpu().numpy()
txt_feats_np = txt_feats.cpu().numpy()

# Compute similarity matrix
similarity = cosine_similarity(img_feats_np, txt_feats_np)

print(f"Similarity matrix shape: {similarity.shape}")
print(f"  {similarity.shape[0]} images x {similarity.shape[1]} texts")

# Image-to-Text Retrieval
def recall_at_k(similarity, k_vals=[1, 5, 10]):
    """Compute Recall@K for image-to-text retrieval"""
    num_images = similarity.shape[0]

    # Get top-k predictions for each image
    top_k_indices = np.argsort(-similarity, axis=1)

    results = {}
    for k in k_vals:
        # For each image, check if ground truth is in top-k
        correct = 0
        for i in range(num_images):
            # Assuming diagonal is ground truth (image i matches text i)
            if i in top_k_indices[i, :k]:
                correct += 1

        recall = correct / num_images * 100
        results[f'R@{k}'] = recall
        print(f"  Image→Text Recall@{k}: {recall:.2f}%")

    return results

# Text-to-Image Retrieval
def recall_at_k_text_to_image(similarity, k_vals=[1, 5, 10]):
    """Compute Recall@K for text-to-image retrieval"""
    num_texts = similarity.shape[1]

    # Transpose for text-to-image
    similarity_t = similarity.T
    top_k_indices = np.argsort(-similarity_t, axis=1)

    results = {}
    for k in k_vals:
        correct = 0
        for i in range(num_texts):
            if i in top_k_indices[i, :k]:
                correct += 1

        recall = correct / num_texts * 100
        results[f'R@{k}'] = recall
        print(f"  Text→Image Recall@{k}: {recall:.2f}%")

    return results

print("\n" + "="*60)
print("IMAGE-TO-TEXT RETRIEVAL")
print("="*60)
i2t_results = recall_at_k(similarity)

print("\n" + "="*60)
print("TEXT-TO-IMAGE RETRIEVAL")
print("="*60)
t2i_results = recall_at_k_text_to_image(similarity)

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Image→Text: R@1={i2t_results['R@1']:.2f}%, R@5={i2t_results['R@5']:.2f}%, R@10={i2t_results['R@10']:.2f}%")
print(f"Text→Image: R@1={t2i_results['R@1']:.2f}%, R@5={t2i_results['R@5']:.2f}%, R@10={t2i_results['R@10']:.2f}%")
```

---

### Step 3: Visualize Results (Optional)

```python
# Cell 4: Visualize Top-K Retrievals
import matplotlib.pyplot as plt
from datasets import load_dataset

# Load test dataset to get images
test_dataset = load_dataset("Peacein/color-fundus-eye", split="test", streaming=True)

# Load your prompts
import pandas as pd
test_df = pd.read_csv(CSV_TEST)

def visualize_retrieval(image_idx, top_k=5):
    """Show top-K text matches for a given image"""

    # Get top-k text indices for this image
    top_text_indices = np.argsort(-similarity[image_idx])[:top_k]

    # Get the image
    dataset_iter = iter(test_dataset)
    for _ in range(image_idx):
        next(dataset_iter)
    sample = next(dataset_iter)
    image = sample['image']

    # Display
    fig, axes = plt.subplots(1, top_k + 1, figsize=(20, 4))

    # Show image
    axes[0].imshow(image)
    axes[0].set_title(f"Query Image {image_idx}", fontsize=10, fontweight='bold')
    axes[0].axis('off')

    # Show top-k matching texts
    for i, text_idx in enumerate(top_text_indices):
        prompt = test_df.iloc[text_idx]['prompt']
        label = test_df.iloc[text_idx]['label']
        sim_score = similarity[image_idx, text_idx]

        axes[i+1].text(0.5, 0.5,
                       f"Rank {i+1}\n\nSimilarity: {sim_score:.3f}\n\nLabel: {label}\n\n{prompt[:100]}...",
                       ha='center', va='center',
                       fontsize=8, wrap=True)
        axes[i+1].axis('off')

        # Highlight correct match
        if text_idx == image_idx:
            axes[i+1].set_facecolor('lightgreen')

    plt.tight_layout()
    plt.show()

# Example: Visualize retrieval for first 3 test images
for idx in range(3):
    print(f"\n{'='*80}")
    print(f"Image Index {idx}")
    print(f"{'='*80}")
    visualize_retrieval(idx, top_k=5)
```

---

## Expected Results

### Good Performance Indicators:
- **R@1 > 70%**: Model correctly retrieves ground truth in top-1
- **R@5 > 90%**: Ground truth in top-5
- **R@10 > 95%**: Ground truth in top-10

### Baseline Comparison (from RET-CLIP paper):
- Original RET-CLIP on medical datasets: R@1 ~80-85%
- Your results will depend on:
  - Training epochs
  - Dataset quality
  - Prompt diversity

---

## Save Results

```python
# Cell 5: Save Evaluation Results
results = {
    'checkpoint': '/content/logs/retclip_training/checkpoints/epoch10.pt',
    'test_samples': len(img_feats),
    'image_to_text': i2t_results,
    'text_to_image': t2i_results,
    'similarity_matrix_shape': similarity.shape,
}

import json
with open('/content/evaluation_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("✅ Results saved to /content/evaluation_results.json")

# Copy to Google Drive
!cp /content/evaluation_results.json /content/drive/MyDrive/retclip_results/
!cp /content/test_imgs_features.pt /content/drive/MyDrive/retclip_results/
!cp /content/test_texts_features.pt /content/drive/MyDrive/retclip_results/
```

---

## Understanding the Metrics

### Recall@K (Image-to-Text)
**Question**: Given an image, is the correct text description in the top-K retrieved texts?

**Example** (R@5):
- You have 3,253 test images
- For each image, model retrieves top-5 most similar texts
- If 2,900 images have their correct text in top-5
- **R@5 = 2,900 / 3,253 = 89.1%**

### Recall@K (Text-to-Image)
**Question**: Given a text prompt, is the correct image in the top-K retrieved images?

---

## Per-Class Analysis (Advanced)

```python
# Cell 6: Per-Class Retrieval Performance
print("="*80)
print("PER-CLASS RETRIEVAL PERFORMANCE")
print("="*80)

# Group by disease class
class_results = {}
for class_name in test_df['label'].unique():
    class_indices = test_df[test_df['label'] == class_name].index.tolist()

    # Filter similarity matrix for this class
    class_similarity = similarity[class_indices][:, class_indices]

    # Compute R@1 for this class
    top_1 = np.argmax(class_similarity, axis=1)
    correct = sum([i == top_1[i] for i in range(len(class_indices))])
    r1 = correct / len(class_indices) * 100

    class_results[class_name] = {
        'samples': len(class_indices),
        'R@1': r1
    }

    print(f"\n{class_name}")
    print(f"  Samples: {len(class_indices)}")
    print(f"  R@1: {r1:.2f}%")

# Find best and worst performing classes
sorted_classes = sorted(class_results.items(), key=lambda x: x[1]['R@1'], reverse=True)
print(f"\n{'='*80}")
print(f"Best performing: {sorted_classes[0][0]} (R@1={sorted_classes[0][1]['R@1']:.2f}%)")
print(f"Worst performing: {sorted_classes[-1][0]} (R@1={sorted_classes[-1][1]['R@1']:.2f}%)")
```

---

## Add to Your Colab Notebook

Add these cells **AFTER** Cell 11 (Download Trained Model) in your main training notebook.

This gives you comprehensive evaluation of your trained RET-CLIP model!

---

## Quick Command Reference

```bash
# Extract image features
python retclip/RET_CLIP/eval/extract_features.py \
  --extract-image-feats \
  --image-data /content/lmdb/test/imgs \
  --image-feat-output-path /content/test_imgs.pt \
  --resume /content/logs/retclip_training/checkpoints/epoch10.pt

# Extract text features
python retclip/RET_CLIP/eval/extract_features.py \
  --extract-text-feats \
  --text-data /content/data/test_texts.jsonl \
  --text-feat-output-path /content/test_texts.pt \
  --resume /content/logs/retclip_training/checkpoints/epoch10.pt
```

---

## Summary

✅ **Test set IS converted** during preprocessing (Cell 6-7)
✅ **Validation runs automatically** during training
✅ **Full evaluation requires** these manual steps after training

This gives you complete metrics for your paper/thesis! 📊
