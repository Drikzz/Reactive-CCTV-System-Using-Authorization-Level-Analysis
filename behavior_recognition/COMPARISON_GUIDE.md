# Model Comparison Scripts

This folder contains scripts to compare training metrics and tracking performance across MobileNetV2, MobileNetV3-Small, and ShuffleNetV2 models.

---

## 📁 Comparison Scripts

### 1. **compare_training_metrics.py**
Compares training summary files from all three models and generates comprehensive comparison tables.

### 2. **compare_trackers.py**
Runs all three trackers on the same video and compares their real-time performance.

---

## 🚀 Quick Start

### Compare Training Metrics

After training all three models, compare their metrics:

```bash
cd behavior_recognition
python compare_training_metrics.py
```

This will:
- ✅ Read training summary files from `models/mobilenetv2/`, `models/mobilenetv3-small/`, and `models/shufflenetv2/`
- ✅ Parse all metrics (accuracy, loss, training time, per-class results)
- ✅ Generate comparison tables
- ✅ Save reports as both text and markdown

**Output files:**
- `training_comparison.txt` - Text report with tables
- `training_comparison.md` - Markdown report for thesis

---

### Compare Tracker Performance

Run all three trackers on the same video:

```bash
python compare_trackers.py --video ../path/to/test_video.mp4
```

This will:
- ✅ Run each tracker sequentially on the same video
- ✅ Capture FPS, detection counts, classification times
- ✅ Generate performance comparison tables
- ✅ Save reports as both text and markdown

**Optional: Save output videos**
```bash
python compare_trackers.py --video ../test.mp4 --save-outputs
```

**Output files:**
- `tracker_comparison.txt` - Text report with performance metrics
- `tracker_comparison.md` - Markdown report for thesis
- `outputs/comparison_mobilenetv2.mp4` - Output from MobileNetV2 (if --save-outputs)
- `outputs/comparison_mobilenetv3small.mp4` - Output from MobileNetV3-Small (if --save-outputs)
- `outputs/comparison_shufflenetv2.mp4` - Output from ShuffleNetV2 (if --save-outputs)

---

## 📊 Training Metrics Comparison

### Usage

```bash
python compare_training_metrics.py [OPTIONS]
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--mobilenetv2-summary` | `models/mobilenetv2/mobilenet_transfer_training_summary.txt` | Path to MobileNetV2 summary |
| `--mobilenetv3-summary` | `models/mobilenetv3-small/mobilenet_v3_small_transfer_training_summary.txt` | Path to MobileNetV3-Small summary |
| `--shufflenet-summary` | `models/shufflenetv2/shufflenet_v2_transfer_training_summary.txt` | Path to ShuffleNetV2 summary |
| `--save` | `training_comparison.md` | Markdown output filename |
| `--save-txt` | `training_comparison.txt` | Text output filename |
| `--save-json` | None | Optional JSON output for raw metrics |

### Examples

**Basic usage (use default paths):**
```bash
python compare_training_metrics.py
```

**Custom summary files:**
```bash
python compare_training_metrics.py \
    --mobilenetv2-summary path/to/mnv2_summary.txt \
    --mobilenetv3-summary path/to/mnv3_summary.txt \
    --shufflenet-summary path/to/shuffle_summary.txt
```

**Save to custom files:**
```bash
python compare_training_metrics.py \
    --save my_comparison.md \
    --save-txt my_comparison.txt \
    --save-json metrics.json
```

### Output Metrics

The comparison includes:
- ✅ Total epochs
- ✅ Best validation accuracy & epoch
- ✅ Final train/validation accuracy
- ✅ Train-val gap (overfitting indicator)
- ✅ Overall test accuracy
- ✅ Final train/validation loss
- ✅ Training time (minutes)
- ✅ Overfitting status
- ✅ Per-class test accuracy
- ✅ Rankings by accuracy and speed

---

## ⚡ Tracker Performance Comparison

### Usage

```bash
python compare_trackers.py --video <path> [OPTIONS]
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--video` | Required | Path to input video file |
| `--webcam` | - | Use webcam (not recommended for comparison) |
| `--yolo-model` | `models/YOLOv8/yolov8m.pt` | Path to YOLOv8 model |
| `--conf` | 0.5 | YOLO confidence threshold |
| `--iou` | 0.7 | YOLO IOU threshold |
| `--save-outputs` | False | Save output videos for each model |
| `--save-report` | `tracker_comparison.md` | Markdown report filename |
| `--save-txt` | `tracker_comparison.txt` | Text report filename |
| `--save-json` | None | Optional JSON output for raw metrics |

### Examples

**Basic comparison:**
```bash
python compare_trackers.py --video ../Mp4TESTING/test_video.mp4
```

**Save output videos:**
```bash
python compare_trackers.py --video ../test.mp4 --save-outputs
```

**Custom YOLO settings:**
```bash
python compare_trackers.py --video ../test.mp4 --conf 0.6 --iou 0.75
```

**Custom output files:**
```bash
python compare_trackers.py --video ../test.mp4 \
    --save-report my_tracker_comparison.md \
    --save-txt my_tracker_comparison.txt \
    --save-json tracker_metrics.json
```

### Output Metrics

The comparison includes:
- ✅ Total frames processed
- ✅ Total person detections
- ✅ Total classifications
- ✅ Classification rate (%)
- ✅ Average classification time (ms)
- ✅ Final processing FPS
- ✅ Final classification interval
- ✅ Unique tracks seen
- ✅ Total execution time (seconds)
- ✅ FPS rankings
- ✅ Speed improvement percentages

---

## 📈 Example Workflow

### 1. Train All Models

```bash
# Train MobileNetV2
cd MobileNetV2
python train_transfer_learning.py --data ../../datasets --epochs 30

# Train MobileNetV3-Small
cd ../mobilenetv3-small
python train_transfer_learning_mnv3small.py --data ../../datasets --epochs 30

# Train ShuffleNetV2
cd ../shufflenetv2
python train_transfer_learning_shufflenetv2.py --data ../../datasets --epochs 30
```

### 2. Compare Training Metrics

```bash
cd behavior_recognition
python compare_training_metrics.py
```

Review `training_comparison.md` for accuracy comparison.

### 3. Compare Tracker Performance

```bash
python compare_trackers.py --video ../Mp4TESTING/test_video.mp4 --save-outputs
```

Review `tracker_comparison.md` for speed comparison.

### 4. Analyze Results

Compare both reports to understand the accuracy vs speed tradeoff:
- `training_comparison.md` - Shows which model is most accurate
- `tracker_comparison.md` - Shows which model is fastest

---

## 📊 Understanding the Results

### Training Comparison

**Key Metrics:**
- **Best Val Acc:** Highest validation accuracy achieved during training
- **Overall Test Acc:** Final accuracy on unseen test data (most important)
- **Train-Val Gap:** Difference between train and val accuracy (overfitting indicator)
  - < 5%: Excellent generalization
  - 5-10%: Good generalization
  - 10-15%: Moderate overfitting
  - > 15%: High overfitting
- **Training Time:** Total time to train the model

**What to Look For:**
- ✅ Higher test accuracy = better model
- ✅ Lower train-val gap = better generalization
- ✅ Shorter training time = more efficient training

### Tracker Comparison

**Key Metrics:**
- **Final Processing FPS:** Frames per second (higher = faster)
- **Avg Classification Time:** Time to classify one crop (lower = faster)
- **Total Execution Time:** Time to process entire video (lower = faster)
- **Classification Rate:** % of frames where classification occurred

**What to Look For:**
- ✅ Higher FPS = smoother real-time tracking
- ✅ Lower classification time = faster inference
- ✅ Consistent classification rate across models = fair comparison

---

## 🎯 Expected Results Pattern

Based on architecture characteristics:

### Training Accuracy (Highest to Lowest)
1. **MobileNetV2** - Most parameters, best accuracy
2. **MobileNetV3-Small** - Balanced accuracy
3. **ShuffleNetV2** - Fastest, slightly lower accuracy

### Inference Speed (Fastest to Slowest)
1. **ShuffleNetV2** - Optimized for speed (~35% faster than MobileNetV2)
2. **MobileNetV3-Small** - Good speed (~30% faster than MobileNetV2)
3. **MobileNetV2** - Best accuracy, baseline speed

### Model Size (Smallest to Largest)
1. **ShuffleNetV2** - ~2.3M parameters
2. **MobileNetV3-Small** - ~2.5M parameters
3. **MobileNetV2** - ~3.5M parameters

---

## 🔧 Troubleshooting

### Training Comparison Issues

**Error: "No training summary files found"**
- Make sure you've trained all three models first
- Check that the models saved to the default directories
- Verify file paths with `--mobilenetv2-summary`, etc.

**Warning: "Missing summaries for: ..."**
- Only some models were trained
- The comparison will proceed with available models
- Train missing models to get complete comparison

### Tracker Comparison Issues

**Error: "Model not found"**
- Ensure models are trained and checkpoints exist
- Check default paths: `models/mobilenetv2/`, `models/mobilenetv3-small/`, `models/shufflenetv2/`

**Error: "Script not found"**
- Ensure you're running from `behavior_recognition/` directory
- Check that all tracker scripts exist in their respective folders

**Timeout after 10 minutes**
- Video might be too long
- Try with a shorter video clip
- Reduce video resolution

**Low/inconsistent FPS**
- Use GPU if available
- Close other applications
- Use same hardware for all runs to ensure fair comparison

---

## 📝 Thesis Integration

### Using Comparison Results in Thesis

**Training Comparison Table:**
- Include `training_comparison.md` markdown table directly in thesis
- Shows model accuracy comparison
- Demonstrates training efficiency

**Tracker Comparison Table:**
- Include `tracker_comparison.md` markdown table in thesis
- Shows inference speed comparison
- Demonstrates real-time performance

**Combined Analysis:**
Create a scatter plot showing:
- X-axis: Inference speed (FPS)
- Y-axis: Test accuracy (%)
- Points: Each model
- Shows accuracy vs efficiency tradeoff

---

## 💡 Tips for Best Results

### For Fair Comparison

1. **Training:**
   - Use same dataset splits (train/val/test)
   - Use same number of epochs
   - Use same batch size and learning rate
   - Train on same hardware

2. **Tracking:**
   - Use same video file
   - Use same YOLO model
   - Use same confidence/IOU thresholds
   - Run on same hardware
   - Close other applications during comparison

3. **Reporting:**
   - Run comparison multiple times and average results
   - Note hardware specifications in thesis
   - Include any special conditions (GPU model, CPU, RAM, etc.)

---

## 🎓 Example Thesis Sections

### Results Chapter

```markdown
## Model Comparison Results

### Training Performance

Table X shows the training performance comparison across three efficient 
architectures: MobileNetV2, MobileNetV3-Small, and ShuffleNetV2.

[Insert training_comparison.md table here]

### Inference Performance

Table Y presents the real-time tracking performance comparison on a 
standard test video.

[Insert tracker_comparison.md table here]

### Analysis

The results demonstrate a clear accuracy-efficiency tradeoff:
- MobileNetV2 achieved the highest accuracy (XX.XX%) but slowest FPS (XX.X)
- ShuffleNetV2 achieved the fastest FPS (XX.X) with acceptable accuracy (XX.XX%)
- MobileNetV3-Small provided a balanced middle ground
```

---

## 🚀 Advanced Usage

### Automated Comparison Script

Create a shell script to run complete comparison:

```bash
#!/bin/bash
# complete_comparison.sh

echo "Comparing training metrics..."
python compare_training_metrics.py

echo "Comparing tracker performance..."
python compare_trackers.py --video ../test_video.mp4 --save-outputs

echo "Comparison complete!"
echo "Check training_comparison.md and tracker_comparison.md"
```

Run with:
```bash
chmod +x complete_comparison.sh
./complete_comparison.sh
```

---

## ✅ Checklist Before Running

- [ ] All three models trained successfully
- [ ] Training summary files exist in `models/` directories
- [ ] Test video available (recommended: 30-60 seconds, 1-3 people)
- [ ] YOLOv8 model downloaded
- [ ] Sufficient disk space for output videos (if using `--save-outputs`)
- [ ] GPU available and working (for faster comparison)
- [ ] No other heavy applications running

---

**Ready to compare your models! 🎯**

These scripts will help you generate comprehensive comparison data for your thesis, showing both accuracy and efficiency tradeoffs across three state-of-the-art efficient architectures.
