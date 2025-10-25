# Model Evaluation Guide

## Overview
The `evaluate_models.py` script evaluates trained models on the test dataset and generates:
1. **Confusion Matrix** - Visual heatmap showing true vs predicted labels
2. **Performance Metrics** - Precision, Recall, F1-Score, Accuracy
3. **Per-Class Analysis** - Breakdown of metrics for each behavior class
4. **Detailed Reports** - Text and JSON files with all results

## Usage

### Evaluate a Single Model

```bash
# Evaluate MobileNetV2 with feature extraction
python behavior_recognition/evaluate_models.py --model mobilenetv2 --mode feature_extraction

# Evaluate MobileNetV3 with transfer learning
python behavior_recognition/evaluate_models.py --model mobilenetv3 --mode transfer

# Evaluate ShuffleNetV2 with both modes
python behavior_recognition/evaluate_models.py --model shufflenetv2 --mode both
```

### Evaluate All Models

```bash
# Evaluate all models (MobileNetV2, MobileNetV3, ShuffleNetV2) with feature extraction
python behavior_recognition/evaluate_models.py --model all --mode feature_extraction

# Evaluate all models with both training modes
python behavior_recognition/evaluate_models.py --model all --mode both
```

### Save Results to Custom Directory

```bash
python behavior_recognition/evaluate_models.py \
    --model all \
    --mode both \
    --save-dir thesis_evaluation_results
```

## Output Files

For each model evaluation, the script generates:

1. **Confusion Matrix (PNG)**
   - `{model}_{mode}_confusion_matrix.png`
   - Example: `mobilenetv2_feature_extraction_confusion_matrix.png`
   - Beautiful heatmap visualization

2. **Metrics (JSON)**
   - `{model}_{mode}_metrics.json`
   - Example: `mobilenetv2_transfer_metrics.json`
   - Machine-readable format with all metrics

3. **Report (TXT)**
   - `{model}_{mode}_report.txt`
   - Example: `shufflenetv2_feature_extraction_report.txt`
   - Human-readable detailed report

## Metrics Explained

### Overall Metrics

**Accuracy**: Proportion of correct predictions
- Formula: `Accuracy = (TP + TN) / (TP + TN + FP + FN)`
- Range: 0-100%
- Higher is better

**Precision (Macro)**: Average precision across all classes
- Formula: `Precision = TP / (TP + FP)`
- Measures: How many predicted positives were actually positive
- Range: 0-100%
- Higher is better

**Recall (Macro)**: Average recall across all classes
- Formula: `Recall = TP / (TP + FN)`
- Measures: How many actual positives were correctly identified
- Range: 0-100%
- Higher is better

**F1-Score (Macro)**: Harmonic mean of precision and recall
- Formula: `F1 = 2TP / (2TP + FP + FN)`
- Measures: Balanced evaluation combining precision and recall
- Range: 0-100%
- Higher is better

### Weighted vs Macro Averages

- **Macro**: Simple average of per-class metrics (treats all classes equally)
- **Weighted**: Average weighted by number of samples per class (accounts for class imbalance)

### Per-Class Metrics

Each behavior class gets individual:
- Precision
- Recall
- F1-Score

Useful for identifying which behaviors are easier/harder to classify.

## Example Output

```
====================================================================================================
EVALUATION REPORT: MOBILENETV2 (feature_extraction)
====================================================================================================

📊 OVERALL METRICS:
----------------------------------------------------------------------------------------------------
Accuracy:              94.50%
Precision (Macro):     94.23%
Recall (Macro):        94.12%
F1-Score (Macro):      94.17%

Precision (Weighted):  94.45%
Recall (Weighted):     94.50%
F1-Score (Weighted):   94.47%

📋 PER-CLASS METRICS:
----------------------------------------------------------------------------------------------------
Class                     | Precision  | Recall     | F1-Score  
----------------------------------------------------------------------------------------------------
opening-cabinet           |     95.20% |     96.10% |     95.65%
using-computer            |     93.80% |     92.50% |     93.14%
opening-door              |     94.10% |     94.80% |     94.45%
holding-object            |     93.90% |     93.10% |     93.50%

🔢 CONFUSION MATRIX:
----------------------------------------------------------------------------------------------------
True \ Pred          opening-cabinet using-computer  opening-door  holding-object
----------------------------------------------------------------------------------------------------
opening-cabinet                  123              2             1               0
using-computer                     3            118             4               3
opening-door                       1              2           121               3
holding-object                     0              4             2             119
====================================================================================================
```

## Confusion Matrix Interpretation

The confusion matrix shows:
- **Rows**: True labels (actual behavior)
- **Columns**: Predicted labels (model's prediction)
- **Diagonal**: Correct predictions (e.g., opening-cabinet predicted as opening-cabinet)
- **Off-diagonal**: Misclassifications (e.g., using-computer predicted as opening-door)

### Reading the Matrix

```
              Predicted
              A    B    C
True    A   [100]  5    2     ← Class A: 100 correct, 5 misclassified as B, 2 as C
        B     3  [95]   4     ← Class B: 95 correct, 3 misclassified as A, 4 as C
        C     1    2  [98]    ← Class C: 98 correct, 1 misclassified as A, 2 as B
```

Higher numbers on the diagonal = better performance!

## Comparing Transfer vs Feature Extraction

Run both modes to compare:

```bash
python behavior_recognition/evaluate_models.py --model mobilenetv2 --mode both
```

This helps you determine:
- Which training approach generalizes better
- Which has less overfitting (compare with training accuracy)
- Which is more suitable for your deployment scenario

## For Your Thesis

The generated files are thesis-ready:
- **PNG confusion matrices** → Insert directly into your document
- **JSON metrics** → Import into Excel/Python for further analysis
- **TXT reports** → Copy-paste into appendix or results section

## Troubleshooting

### Model Not Found Error
```
FileNotFoundError: Model not found: models/mobilenetv2/mobilenet_feature_extraction.pth
```

**Solution**: Make sure you've trained the model first using the training scripts:
```bash
python behavior_recognition/MobileNetV2/train_feature_extraction.py --epochs 50
```

### CUDA Out of Memory
```
RuntimeError: CUDA out of memory
```

**Solution**: Force CPU evaluation:
```bash
python behavior_recognition/evaluate_models.py --model mobilenetv2 --mode feature_extraction --device cpu
```

### Import Errors
```
ModuleNotFoundError: No module named 'sklearn'
```

**Solution**: Install required packages:
```bash
pip install scikit-learn seaborn matplotlib
```

## Advanced Usage

### Programmatic Access

You can also use the evaluator in your own scripts:

```python
from behavior_recognition.evaluate_models import ModelEvaluator

# Create evaluator
evaluator = ModelEvaluator(
    model_name='mobilenetv2',
    mode='feature_extraction',
    device='cuda'
)

# Run evaluation
results = evaluator.evaluate()

# Access metrics
accuracy = results['metrics']['accuracy']
confusion_matrix = results['confusion_matrix']
precision_per_class = results['metrics']['per_class']['precision']

# Generate plots
evaluator.plot_confusion_matrix(
    results['confusion_matrix'],
    save_path='my_confusion_matrix.png',
    title='My Custom Title'
)

# Print report
evaluator.print_report(results)
```

## Next Steps

After evaluation:
1. Compare metrics across different models
2. Identify which behaviors are hardest to classify (lowest per-class metrics)
3. Analyze confusion matrix to understand misclassification patterns
4. Use insights to improve training data or model architecture
5. Include results in your thesis results/discussion sections
