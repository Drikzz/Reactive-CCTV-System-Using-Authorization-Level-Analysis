# MobileNetV2 Training & Diagnosis Guide

## 🎯 Quick Diagnostic Workflow

### Step 1: Run Diagnostics on Your Current Model

```bash
python diagnose_model.py --model models/mobilenet/mobilenet_best.pth --data datasets
```

This will check:
- ✅ **Class balance** in train/valid/test sets
- ✅ **Label order consistency** between model and dataset
- ✅ **Prediction accuracy** on test samples
- ✅ **Model architecture** (frozen vs trainable layers)
- ✅ **Actionable suggestions** for improvements

**Output includes:**
- Class distribution analysis
- Sample predictions with probabilities
- Visual grid of predictions (saved as `prediction_grid.png`)
- Specific suggestions for your model

---

## 🔧 Common Issues & Solutions

### Issue 1: Class Imbalance (e.g., 120 vs 80 samples)

**Symptom:** Model always predicts the more frequent class

**Check:**
```python
from collections import Counter
Counter(train_dataset.targets)
# Example output: {0: 120, 1: 80}  ← Imbalanced!
```

**Solution:** Use the enhanced training script with class balancing:

```bash
python train_mobilenet_enhanced.py --data datasets --epochs 20
```

This uses `WeightedRandomSampler` to ensure balanced batches during training.

---

### Issue 2: Wrong Class Names (Class_0 instead of opening-cabinet)

**Symptom:** Inference shows `Class_0` and `Class_1` instead of actual names

**Quick Fix (without retraining):**
```bash
python update_model_classes.py \
    --model models/mobilenet/mobilenet_best.pth \
    --classes opening-cabinet using-computer
```

**Permanent Fix:** Retrain with updated scripts (they now save class names automatically)

---

### Issue 3: Model Always Predicts Same Class

**Possible causes:**
1. **Class imbalance** → Use `train_mobilenet_enhanced.py` with balancing
2. **Frozen backbone** → Model can't adapt to your specific domain
3. **Label mismatch** → Classes in wrong order

**Diagnosis:**
```bash
python diagnose_model.py --model your_model.pth --data datasets
```

**Solutions:**

#### A. Enable Class Balancing
```bash
python train_mobilenet_enhanced.py --data datasets
```

#### B. Fine-tune with Partially Unfrozen Backbone
```bash
python train_mobilenet_enhanced.py \
    --data datasets \
    --freeze-backbone \
    --unfreeze-last 3 \
    --lr 0.0001 \
    --epochs 30
```

This keeps most layers frozen but allows the last 3 feature blocks to adapt.

#### C. Fully Trainable (Best for Small Datasets)
```bash
python train_mobilenet_enhanced.py \
    --data datasets \
    --lr 0.0001 \
    --epochs 30
```

---

### Issue 4: Model Looks at Background Instead of Action

**Diagnosis:** Need to visualize what the model focuses on

**Check predictions visually:**
```bash
python diagnose_model.py --model models/mobilenet/mobilenet_best.pth --data datasets
```

This creates `prediction_grid.png` showing correct/incorrect predictions.

**Solution:** More aggressive data augmentation

The enhanced training script includes:
- Random horizontal flips
- Random rotation (±10°)
- Color jitter
- Random affine transforms (translation)

---

## 📊 Training Comparison Table

| Feature | `train_mobilenet.py` | `train_mobilenet_enhanced.py` |
|---------|---------------------|-------------------------------|
| Class Balancing | ❌ | ✅ WeightedRandomSampler |
| Saves Class Names | ✅ | ✅ |
| Flexible Freezing | ❌ | ✅ Last N blocks |
| LR Scheduling | ❌ | ✅ ReduceLROnPlateau |
| Per-Class Stats | ✅ | ✅ |
| Augmentation | Basic | Enhanced |

---

## 🚀 Recommended Workflow

### For First-Time Training:

```bash
# 1. Diagnose your dataset
python diagnose_model.py --model models/mobilenet/mobilenet_best.pth --data datasets

# 2. Train with class balancing and partial fine-tuning
python train_mobilenet_enhanced.py \
    --data datasets \
    --epochs 30 \
    --batch-size 16 \
    --lr 0.0001 \
    --freeze-backbone \
    --unfreeze-last 3

# 3. Test inference
python inference_mobilenet.py --image test.jpg --no-display --save output.jpg
```

### For Improving Existing Model:

```bash
# 1. Add class names to existing model
python update_model_classes.py \
    --model models/mobilenet/mobilenet_best.pth \
    --classes opening-cabinet using-computer

# 2. Diagnose issues
python diagnose_model.py --model models/mobilenet/mobilenet_best.pth --data datasets

# 3. Retrain if needed with recommended settings from diagnostics
```

---

## 🎓 Training Strategy Guide

### Strategy 1: Small Dataset (< 200 images per class)

```bash
python train_mobilenet_enhanced.py \
    --data datasets \
    --epochs 40 \
    --batch-size 8 \
    --lr 0.0001 \
    --freeze-backbone \
    --unfreeze-last 4
```

**Why:**
- Smaller batch size for better generalization
- Partially frozen backbone prevents overfitting
- More epochs to fully converge
- Lower LR for stable learning

### Strategy 2: Imbalanced Dataset (ratio > 1.5:1)

```bash
python train_mobilenet_enhanced.py \
    --data datasets \
    --epochs 30 \
    --batch-size 16 \
    --lr 0.0001
    # Class balancing is ON by default
```

**Why:**
- WeightedRandomSampler ensures minority class is seen
- No need to manually oversample

### Strategy 3: Large Balanced Dataset (> 500 images per class)

```bash
python train_mobilenet_enhanced.py \
    --data datasets \
    --epochs 20 \
    --batch-size 32 \
    --lr 0.001 \
    --no-balancing
    # All layers trainable by default
```

**Why:**
- Enough data to train full network
- Higher learning rate for faster convergence
- No balancing needed if already balanced

---

## 📋 Command Reference

### Diagnostics
```bash
# Full diagnostics
python diagnose_model.py --model <path> --data <path>

# Skip visualization (faster, for Colab)
python diagnose_model.py --model <path> --data <path> --skip-viz
```

### Training
```bash
# Basic (with class balancing)
python train_mobilenet_enhanced.py

# Custom settings
python train_mobilenet_enhanced.py \
    --data datasets \
    --epochs 30 \
    --batch-size 16 \
    --lr 0.0001 \
    --freeze-backbone \
    --unfreeze-last 3 \
    --save-dir models/mobilenet \
    --model-name mobilenet_v2_enhanced.pth

# Disable class balancing (if already balanced)
python train_mobilenet_enhanced.py --no-balancing
```

### Inference
```bash
# Single image
python inference_mobilenet.py --image test.jpg --no-display --save output.jpg

# With custom class names (override model)
python inference_mobilenet.py \
    --image test.jpg \
    --classes opening-cabinet using-computer

# Video
python inference_mobilenet.py --video input.mp4 --save output.mp4

# Webcam
python inference_mobilenet.py --webcam
```

### Update Model
```bash
# Add class names to existing model
python update_model_classes.py \
    --model models/mobilenet/mobilenet_best.pth \
    --classes opening-cabinet using-computer
```

---

## 🐛 Troubleshooting

### "Model always predicts Class_0"

1. Check class balance:
   ```bash
   python diagnose_model.py --model <model> --data datasets
   ```

2. Look for "CLASS IMBALANCE DETECTED" in output

3. Retrain with balancing:
   ```bash
   python train_mobilenet_enhanced.py --data datasets
   ```

### "Inference shows wrong class names"

```bash
python update_model_classes.py \
    --model models/mobilenet/mobilenet_best.pth \
    --classes opening-cabinet using-computer
```

### "Model accuracy stuck at ~50-60%"

1. Unfreeze more layers:
   ```bash
   python train_mobilenet_enhanced.py \
       --freeze-backbone \
       --unfreeze-last 5 \
       --lr 0.00005
   ```

2. Or train fully:
   ```bash
   python train_mobilenet_enhanced.py --lr 0.0001
   ```

### "Training is too slow"

- Reduce batch size if GPU memory is full
- Use `--freeze-backbone` to freeze most layers
- Reduce `--unfreeze-last` to fewer blocks

---

## 📊 Expected Performance

With proper configuration:

| Metric | Expected Range | Action if Lower |
|--------|---------------|-----------------|
| Train Accuracy | 85-95% | More epochs, lower LR |
| Validation Accuracy | 80-90% | Check overfitting, add augmentation |
| Test Accuracy | 75-85% | Check class balance, unfreeze layers |
| Train-Val Gap | < 10% | Reduce overfitting, more data |

---

## 🎯 Quick Start for Your Case

Based on your symptoms (always predicting one class):

```bash
# 1. Diagnose
python diagnose_model.py --model models/mobilenet/mobilenet_best.pth --data datasets

# 2. Fix class names
python update_model_classes.py \
    --model models/mobilenet/mobilenet_best.pth \
    --classes opening-cabinet using-computer

# 3. Retrain with balancing and partial fine-tuning
python train_mobilenet_enhanced.py \
    --data datasets \
    --epochs 30 \
    --batch-size 16 \
    --lr 0.0001 \
    --freeze-backbone \
    --unfreeze-last 3

# 4. Test
python inference_mobilenet.py --image test.jpg --no-display --save result.jpg
```

This should resolve the "always predicting same class" issue! 🎉
