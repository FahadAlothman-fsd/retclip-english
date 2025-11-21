# RET-CLIP Hyperparameter Tuning Guide

This guide covers hyperparameters worth tuning for RET-CLIP and how to implement grid/random search.

---

## 🎯 Key Hyperparameters

### 1. **Learning Rate** (`--lr`)
**Current:** `1e-5`
**Search Range:** `[1e-6, 5e-6, 1e-5, 5e-5, 1e-4]`

**Impact:** Controls how fast the model learns
- Too high → unstable training, poor convergence
- Too low → slow convergence, may not reach optimal performance

**Priority:** ⭐⭐⭐⭐⭐ (HIGHEST)

---

### 2. **Batch Size** (`--batch-size`)
**Current:** `128`
**Search Range:** `[32, 64, 128, 256]`

**Impact:** Affects gradient estimation and training stability
- Larger → more stable gradients, faster training, more memory
- Smaller → noisier gradients, slower training, less memory

**Constraint:** Limited by GPU memory (A100 can handle 256 for ViT-B-16)

**Priority:** ⭐⭐⭐⭐

---

### 3. **Warmup Steps** (`--warmup`)
**Current:** `500`
**Search Range:** `[100, 300, 500, 1000, 2000]`

**Impact:** Number of steps to linearly increase LR from 0 to target
- Too short → unstable early training
- Too long → wasted compute on suboptimal LR

**Rule of Thumb:** 5-10% of total training steps

**Priority:** ⭐⭐⭐

---

### 4. **Temperature** (InfoNCE loss) (`--temp` / `--learnable-temp`)
**Current:** Likely `0.07` (CLIP default)
**Search Range:** `[0.01, 0.05, 0.07, 0.1, 0.2]`

**Impact:** Controls sharpness of similarity distribution in contrastive loss
- Lower → sharper distinctions (more confident predictions)
- Higher → softer distinctions (more conservative)

**Alternative:** Use `--learnable-temp` to let the model learn this

**Priority:** ⭐⭐⭐⭐

---

### 5. **Weight Decay** (`--wd`)
**Current:** Likely `0.2` (CLIP default)
**Search Range:** `[0.01, 0.05, 0.1, 0.2, 0.5]`

**Impact:** L2 regularization strength
- Higher → more regularization, less overfitting
- Lower → less regularization, may overfit on small datasets

**Priority:** ⭐⭐⭐

---

### 6. **Number of Epochs** (`--epochs`)
**Current:** `10`
**Search Range:** `[5, 10, 15, 20]`

**Impact:** Total training duration
- Too few → underfitting
- Too many → overfitting, diminishing returns

**Note:** Monitor validation metrics to determine optimal stopping point

**Priority:** ⭐⭐

---

### 7. **Learning Rate Schedule** (`--lr-scheduler`)
**Current:** Cosine annealing (default)
**Options:** `["cosine", "linear", "constant"]`

**Impact:** How LR changes over training
- Cosine → smooth decay (CLIP default, usually best)
- Linear → linear decay
- Constant → no decay

**Priority:** ⭐⭐

---

### 8. **Optimizer** (`--optimizer`)
**Current:** AdamW (default)
**Options:** `["adamw", "adam", "sgd"]`

**Impact:** Optimization algorithm
- AdamW → adaptive LR, good default (CLIP uses this)
- SGD → simpler, may need different LR

**Priority:** ⭐

---

## 📊 Recommended Search Strategy

### Phase 1: Quick Search (TEST_MODE)

**Goal:** Find reasonable ranges quickly

**Hyperparameters:**
```python
search_space = {
    'learning_rate': [1e-6, 5e-6, 1e-5, 5e-5],
    'batch_size': [64, 128],
    'warmup_steps': [300, 500],
}
```

**Strategy:** Grid search (4 × 2 × 2 = 16 runs)

**Runtime:** ~30 min per run × 16 = **~8 hours on A100**

**Dataset:** 100 train samples, 50 test samples, 2 epochs

---

### Phase 2: Fine-Grained Search (FULL_MODE)

**Goal:** Optimize best hyperparameters from Phase 1

**Hyperparameters:**
```python
# Start from best config in Phase 1, then search nearby
best_lr = 5e-6  # Example from Phase 1

search_space = {
    'learning_rate': [1e-6, 3e-6, 5e-6, 7e-6, 1e-5],
    'batch_size': [128],  # Fix based on Phase 1
    'warmup_steps': [500, 1000],
    'weight_decay': [0.1, 0.2, 0.3],
    'temperature': [0.05, 0.07, 0.1],
}
```

**Strategy:** Random search (sample 20 configs)

**Runtime:** ~6 hours per run × 20 = **~120 hours (~5 days) on A100**

**Dataset:** Full dataset (12,989 train, 3,253 test), 10 epochs

---

## 🔧 Implementation

### Option 1: Manual Grid Search (Simple)

Update notebook cell-25 to loop over hyperparameters:

```python
# Define search space
learning_rates = [1e-6, 5e-6, 1e-5, 5e-5]
batch_sizes = [64, 128]
warmup_steps = [300, 500]

results = []

for lr in learning_rates:
    for bs in batch_sizes:
        for ws in warmup_steps:
            print(f"\n{'='*80}")
            print(f"Training: lr={lr}, batch_size={bs}, warmup={ws}")
            print(f"{'='*80}")

            # Train model
            !python /content/retclip/RET_CLIP/training/main.py \
                --train-data {DRIVE_LMDB}/train \
                --train-num-samples {len(train_df)} \
                --batch-size {bs} \
                --epochs {NUM_EPOCHS} \
                --lr {lr} \
                --warmup {ws} \
                --vision-model {VISION_MODEL} \
                --text-model {TEXT_MODEL} \
                --logs {DRIVE_CHECKPOINTS}/lr{lr}_bs{bs}_ws{ws} \
                --name retclip_lr{lr}_bs{bs}_ws{ws} \
                --save-frequency 1

            # Evaluate and store results
            checkpoint = f"{DRIVE_CHECKPOINTS}/lr{lr}_bs{bs}_ws{ws}/epoch_{NUM_EPOCHS}.pt"
            accuracy = evaluate_checkpoint(checkpoint, test_df)

            results.append({
                'lr': lr,
                'batch_size': bs,
                'warmup': ws,
                'accuracy': accuracy
            })

            print(f"✅ Accuracy: {accuracy * 100:.2f}%")

# Save results
pd.DataFrame(results).to_csv(f"{DRIVE_RESULTS}/grid_search_results.csv", index=False)

# Find best config
best_config = max(results, key=lambda x: x['accuracy'])
print(f"\n🏆 Best Configuration:")
print(f"   LR: {best_config['lr']}")
print(f"   Batch Size: {best_config['batch_size']}")
print(f"   Warmup: {best_config['warmup']}")
print(f"   Accuracy: {best_config['accuracy'] * 100:.2f}%")
```

---

### Option 2: Optuna (Advanced)

Use Optuna for Bayesian optimization:

```python
import optuna

def objective(trial):
    # Suggest hyperparameters
    lr = trial.suggest_float('lr', 1e-6, 1e-4, log=True)
    batch_size = trial.suggest_categorical('batch_size', [64, 128, 256])
    warmup = trial.suggest_int('warmup', 100, 2000)
    weight_decay = trial.suggest_float('weight_decay', 0.01, 0.5)

    # Train model
    checkpoint_dir = f"{DRIVE_CHECKPOINTS}/trial_{trial.number}"

    !python /content/retclip/RET_CLIP/training/main.py \
        --train-data {DRIVE_LMDB}/train \
        --batch-size {batch_size} \
        --lr {lr} \
        --warmup {warmup} \
        --wd {weight_decay} \
        --logs {checkpoint_dir} \
        ...

    # Evaluate
    checkpoint = f"{checkpoint_dir}/epoch_{NUM_EPOCHS}.pt"
    accuracy = evaluate_checkpoint(checkpoint, test_df)

    return accuracy

# Run optimization
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20)

print(f"Best trial: {study.best_trial.params}")
print(f"Best accuracy: {study.best_value * 100:.2f}%")
```

---

### Option 3: Weights & Biases Sweeps (Best)

Use W&B for tracking and hyperparameter search:

```python
import wandb

# Define sweep config
sweep_config = {
    'method': 'bayes',  # or 'grid', 'random'
    'metric': {
        'name': 'val_accuracy',
        'goal': 'maximize'
    },
    'parameters': {
        'learning_rate': {
            'distribution': 'log_uniform_values',
            'min': 1e-6,
            'max': 1e-4
        },
        'batch_size': {
            'values': [64, 128, 256]
        },
        'warmup_steps': {
            'distribution': 'int_uniform',
            'min': 100,
            'max': 2000
        },
        'weight_decay': {
            'distribution': 'uniform',
            'min': 0.01,
            'max': 0.5
        },
        'temperature': {
            'distribution': 'uniform',
            'min': 0.01,
            'max': 0.2
        }
    }
}

# Create sweep
sweep_id = wandb.sweep(sweep_config, project="retclip-tuning")

# Run sweep
wandb.agent(sweep_id, function=train_and_evaluate, count=20)
```

---

## 📈 Monitoring & Early Stopping

### What to Track:

1. **Training Loss** (should decrease steadily)
2. **Validation Accuracy** (should increase, then plateau)
3. **Zero-Shot Accuracy** (final metric)

### Early Stopping:

Stop training if validation accuracy doesn't improve for N epochs:

```python
best_val_acc = 0
patience = 3
no_improve_count = 0

for epoch in range(num_epochs):
    train_loss = train_one_epoch()
    val_acc = evaluate(val_set)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        no_improve_count = 0
        save_checkpoint()
    else:
        no_improve_count += 1

    if no_improve_count >= patience:
        print(f"Early stopping at epoch {epoch}")
        break
```

---

## 💡 Best Practices

### 1. Start Small
- Use TEST_MODE for initial search
- Only run FULL_MODE on promising configs

### 2. Fix Some Hyperparameters
- Fix vision model (ViT-B-16 is good)
- Focus on LR, batch size, warmup first

### 3. Use Learning Rate Finder
Before grid search, run LR finder:

```python
from torch_lr_finder import LRFinder

model = ...
optimizer = ...
criterion = ...

lr_finder = LRFinder(model, optimizer, criterion)
lr_finder.range_test(train_loader, end_lr=1e-3, num_iter=100)
lr_finder.plot()  # Shows optimal LR range
```

### 4. Monitor GPU Utilization
```bash
watch -n 1 nvidia-smi
```

Ensure GPU is at ~95-100% utilization for efficient training.

### 5. Save Everything
Save all configs and metrics to compare later:

```python
config = {
    'lr': lr,
    'batch_size': batch_size,
    'warmup': warmup,
    'weight_decay': weight_decay
}

results = {
    'config': config,
    'train_loss': train_loss,
    'val_accuracy': val_accuracy,
    'test_accuracy': test_accuracy
}

with open(f"{DRIVE_RESULTS}/run_{timestamp}.json", 'w') as f:
    json.dump(results, f)
```

---

## 🎯 Expected Improvements

Based on CLIP papers and medical imaging literature:

| Hyperparameter | Default | Optimized | Expected Gain |
|----------------|---------|-----------|---------------|
| Learning Rate | 1e-5 | ~5e-6 | +2-5% accuracy |
| Batch Size | 128 | ~256 | +1-3% accuracy |
| Warmup | 500 | ~1000 | +1-2% accuracy |
| Temperature | 0.07 | ~0.05 | +1-2% accuracy |
| **Total** | - | - | **+5-12% accuracy** |

---

## ✅ Recommended Workflow

### For TEST_MODE (quick validation):

1. **Run default config** → Get baseline (~30 min)
2. **Grid search LR** → Find best LR range (~2-3 hours)
3. **Grid search batch size & warmup** → Fine-tune (~2-3 hours)
4. **Validate best config** → Confirm results (~30 min)

**Total: ~6-8 hours**

### For FULL_MODE (production):

1. **Use best config from TEST_MODE** as starting point
2. **Random/Bayesian search** around that config (20 runs × 6 hours = 5 days)
3. **Train final model** with best config (6-8 hours)
4. **Compare all 3 text encoders** with best config (18-24 hours)

**Total: ~6-7 days**

---

## 📚 References

- [CLIP Paper](https://arxiv.org/abs/2103.00020) - Original CLIP hyperparameters
- [Optuna](https://optuna.org/) - Hyperparameter optimization framework
- [Weights & Biases](https://wandb.ai/) - Experiment tracking and sweeps

---

**Ready to optimize RET-CLIP!** 🚀
