# Combined Visualization Output Guide

## New Features in evaluate_models.py

The evaluation script now automatically generates combined visualizations when evaluating multiple models!

## Generated Files

### When Running: `--model all --mode both` (6 models total)

You get **3 types of combined visualizations**:

---

## 1️⃣ ALL MODELS COMBINED (6 models)

### `all_models_confusion_matrices.png`
- **Layout**: 2 rows × 3 columns grid
- **Shows**: All 6 confusion matrices side-by-side
- **Models displayed**:
  - Row 1: MobileNetV2 (FE), MobileNetV3 (FE), ShuffleNetV2 (FE)
  - Row 2: MobileNetV2 (TL), MobileNetV3 (TL), ShuffleNetV2 (TL)
- **Perfect for**: Overall comparison in thesis

### `all_models_metrics_comparison.png`
- **Type**: Grouped bar chart
- **Shows**: Accuracy, Precision, Recall, F1-Score for all 6 models
- **X-axis**: Model names
- **Y-axis**: Score (%)
- **Perfect for**: Quick performance comparison

---

## 2️⃣ FEATURE EXTRACTION ONLY (3 models)

### `feature_extraction_confusion_matrices.png`
- **Layout**: 1 row × 3 columns
- **Shows**: Only feature extraction models
- **Models displayed**:
  - MobileNetV2 (Feature Extraction)
  - MobileNetV3-Small (Feature Extraction)
  - ShuffleNetV2 (Feature Extraction)
- **Perfect for**: Comparing frozen backbone approaches

### `feature_extraction_metrics_comparison.png`
- **Type**: Grouped bar chart
- **Shows**: Metrics for 3 feature extraction models only
- **Perfect for**: Analyzing which architecture + FE performs best

---

## 3️⃣ TRANSFER LEARNING ONLY (3 models)

### `transfer_learning_confusion_matrices.png`
- **Layout**: 1 row × 3 columns
- **Shows**: Only transfer learning models
- **Models displayed**:
  - MobileNetV2 (Transfer Learning)
  - MobileNetV3-Small (Transfer Learning)
  - ShuffleNetV2 (Transfer Learning)
- **Perfect for**: Comparing fine-tuned backbone approaches

### `transfer_learning_metrics_comparison.png`
- **Type**: Grouped bar chart
- **Shows**: Metrics for 3 transfer learning models only
- **Perfect for**: Analyzing which architecture + TL performs best

---

## Complete Output Structure

```
evaluation_results/
├── # Individual model files (existing)
├── mobilenetv2_feature_extraction_confusion_matrix.png
├── mobilenetv2_feature_extraction_metrics.json
├── mobilenetv2_feature_extraction_report.txt
├── mobilenetv2_transfer_confusion_matrix.png
├── mobilenetv2_transfer_metrics.json
├── mobilenetv2_transfer_report.txt
├── ... (same for mobilenetv3 and shufflenetv2)
│
├── # NEW: Combined visualizations
├── all_models_confusion_matrices.png              ← 6 confusion matrices
├── all_models_metrics_comparison.png              ← 6 models bar chart
├── feature_extraction_confusion_matrices.png      ← 3 FE confusion matrices
├── feature_extraction_metrics_comparison.png      ← 3 FE models bar chart
├── transfer_learning_confusion_matrices.png       ← 3 TL confusion matrices
└── transfer_learning_metrics_comparison.png       ← 3 TL models bar chart
```

---

## Usage Examples

### Evaluate All Models (Generates All Combined Visualizations)

```bash
python behavior_recognition/evaluate_models.py --model all --mode both
```

**Generates**:
- ✅ 6 individual confusion matrices
- ✅ 6 individual reports
- ✅ 1 combined "all models" confusion matrix (6 subplots)
- ✅ 1 combined "all models" metrics chart
- ✅ 1 combined "feature extraction" confusion matrix (3 subplots)
- ✅ 1 combined "feature extraction" metrics chart
- ✅ 1 combined "transfer learning" confusion matrix (3 subplots)
- ✅ 1 combined "transfer learning" metrics chart

**Total**: 20+ files!

---

### Evaluate Only Feature Extraction (Generates FE Combined Visualization)

```bash
python behavior_recognition/evaluate_models.py --model all --mode feature_extraction
```

**Generates**:
- ✅ 3 individual confusion matrices (MNV2, MNV3, SNV2 - all FE)
- ✅ 3 individual reports
- ✅ 1 combined "all models" confusion matrix (3 subplots)
- ✅ 1 combined "all models" metrics chart
- ✅ 1 combined "feature extraction" confusion matrix (3 subplots)
- ✅ 1 combined "feature extraction" metrics chart

**Total**: 11 files

---

### Evaluate Only Transfer Learning (Generates TL Combined Visualization)

```bash
python behavior_recognition/evaluate_models.py --model all --mode transfer
```

**Generates**:
- ✅ 3 individual confusion matrices (MNV2, MNV3, SNV2 - all TL)
- ✅ 3 individual reports
- ✅ 1 combined "all models" confusion matrix (3 subplots)
- ✅ 1 combined "all models" metrics chart
- ✅ 1 combined "transfer learning" confusion matrix (3 subplots)
- ✅ 1 combined "transfer learning" metrics chart

**Total**: 11 files

---

### Evaluate Single Model (No Combined Visualization)

```bash
python behavior_recognition/evaluate_models.py --model mobilenetv2 --mode feature_extraction
```

**Generates**:
- ✅ 1 confusion matrix
- ✅ 1 metrics JSON
- ✅ 1 text report

**Total**: 3 files (no combined visualizations since only 1 model)

---

## Visual Layout Examples

### All Models (6 models) - 2×3 Grid

```
┌─────────────┬─────────────┬─────────────┐
│ MobileNetV2 │ MobileNetV3 │ ShuffleNetV2│
│  (Feature)  │  (Feature)  │  (Feature)  │
│             │             │             │
│ [CM Matrix] │ [CM Matrix] │ [CM Matrix] │
├─────────────┼─────────────┼─────────────┤
│ MobileNetV2 │ MobileNetV3 │ ShuffleNetV2│
│ (Transfer)  │ (Transfer)  │ (Transfer)  │
│             │             │             │
│ [CM Matrix] │ [CM Matrix] │ [CM Matrix] │
└─────────────┴─────────────┴─────────────┘
```

### Feature Extraction (3 models) - 1×3 Grid

```
┌─────────────┬─────────────┬─────────────┐
│ MobileNetV2 │ MobileNetV3 │ ShuffleNetV2│
│  (Feature)  │  (Feature)  │  (Feature)  │
│             │             │             │
│ [CM Matrix] │ [CM Matrix] │ [CM Matrix] │
└─────────────┴─────────────┴─────────────┘
```

### Metrics Comparison Bar Chart

```
      │
 100% ├─┐ ┌─┐ ┌─┐ ┌─┐     Each model has 4 bars:
      │ │ │ │ │ │ │ │     🔵 Accuracy
  90% │ │ │ │ │ │ │ │     🟣 Precision
      │ │ │ │ │ │ │ │     🟠 Recall
  80% │ │ │ │ │ │ │ │     🔴 F1-Score
      │ │ │ │ │ │ │ │
  70% └─┴─┴─┴─┴─┴─┴─┴─
       MNV2 MNV3 SNV2 ...
```

---

## Benefits for Your Thesis

### 1. Professional Presentation
- ✅ Side-by-side comparison makes differences obvious
- ✅ Consistent formatting across all models
- ✅ High-resolution (300 DPI) for publication quality

### 2. Easy Analysis
- ✅ Quickly spot which model performs best
- ✅ Compare training modes at a glance
- ✅ Identify patterns in misclassifications

### 3. Thesis-Ready Figures
- ✅ Direct insertion into Results chapter
- ✅ Clear labels and titles
- ✅ Color-coded for easy reading

### 4. Time Saver
- ✅ Automatic generation (no manual plotting)
- ✅ All combinations created in one run
- ✅ Consistent styling across all figures

---

## Tips for Thesis

### Figure Captions (Examples)

**For all_models_confusion_matrices.png:**
> Figure X: Confusion matrices for all six trained models. Top row shows feature extraction models (frozen backbone), bottom row shows transfer learning models (fine-tuned backbone). MobileNetV2, MobileNetV3-Small, and ShuffleNetV2 architectures are compared across both training paradigms.

**For feature_extraction_metrics_comparison.png:**
> Figure Y: Performance comparison of feature extraction models. All three architectures (MobileNetV2, MobileNetV3-Small, ShuffleNetV2) achieve >90% accuracy, with MobileNetV3-Small showing the best F1-score (94.5%).

**For transfer_learning_confusion_matrices.png:**
> Figure Z: Confusion matrices for transfer learning models. These models show higher accuracy than feature extraction but with increased risk of overfitting on the small training dataset.

---

## Console Output Example

```bash
$ python behavior_recognition/evaluate_models.py --model all --mode both

# ... individual evaluations run ...

====================================================================================================
EVALUATION COMPLETE
====================================================================================================

Results saved to: evaluation_results/
Total evaluations: 6

📊 Generating combined visualizations...
✓ Combined confusion matrices saved to: evaluation_results/all_models_confusion_matrices.png
✓ Metrics comparison saved to: evaluation_results/all_models_metrics_comparison.png
✓ Combined confusion matrices saved to: evaluation_results/feature_extraction_confusion_matrices.png
✓ Metrics comparison saved to: evaluation_results/feature_extraction_metrics_comparison.png
✓ Combined confusion matrices saved to: evaluation_results/transfer_learning_confusion_matrices.png
✓ Metrics comparison saved to: evaluation_results/transfer_learning_metrics_comparison.png

📊 COMPARISON SUMMARY:
----------------------------------------------------------------------
Model                          | Accuracy   | F1-Macro  
----------------------------------------------------------------------
mobilenetv2_feature_extraction |     94.50% |     94.17%
mobilenetv3_feature_extraction |     95.20% |     95.05%
shufflenetv2_feature_extraction|     93.80% |     93.45%
mobilenetv2_transfer           |     96.10% |     95.89%
mobilenetv3_transfer           |     96.50% |     96.32%
shufflenetv2_transfer          |     95.70% |     95.44%
----------------------------------------------------------------------

✓ All evaluations completed successfully!
```

---

## Summary

**What's New:**
1. ✅ Combined confusion matrices (multiple models in one image)
2. ✅ Metrics comparison bar charts
3. ✅ Automatic generation for different model groupings
4. ✅ High-quality visualizations for thesis

**What Stayed the Same:**
1. ✅ Individual model reports still generated
2. ✅ JSON metrics files still created
3. ✅ All original functionality preserved

**Result:**
More comprehensive analysis with less manual work! Perfect for your thesis! 🎓
