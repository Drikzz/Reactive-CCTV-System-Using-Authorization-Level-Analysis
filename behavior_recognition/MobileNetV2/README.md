# 🎯 Behavior Recognition System

## 📚 Current Active Files (Use These)

### **Training**
- **`train_transfer_learning.py`** ✅ - Main training script using MobileNetV2 transfer learning
  - Pretrained ImageNet weights, fully fine-tuned
  - Focused augmentation to reduce background bias
  - Automatic plotting and summary generation
  - Per-class accuracy tracking
  - Best checkpoint saving

### **Inference & Tracking**
- **`yolo_mobilenet_tracker.py`** ✅ - Integrated YOLOv8 + MobileNetV2 system
  - Real-time person tracking and behavior classification
  - Adaptive classification intervals based on FPS
  - Temporal smoothing to reduce prediction flicker
  - Dynamic confidence threshold (adjustable at runtime)
  - Per-track caching for efficiency

### **Documentation**
- **`YOLO_MOBILENET_TRACKER.md`** - Complete tracker documentation
- **`YOLO_MOBILENET_IMPLEMENTATION.md`** - Implementation details
- **`COLAB_GUIDE.md`** - Google Colab setup and usage
- **`AI_IMAGE_GENERATION_PROMPTS.md`** - Dataset augmentation prompts

---

## 🚀 Quick Start

### **1. Train Your Model**
```powershell
cd behavior_recognition/MobileNetV2

# Train with transfer learning (recommended)
python train_transfer_learning.py \
    --data ../../datasets \
    --epochs 30 \
    --batch-size 16 \
    --lr 1e-4 \
    --model-name mobilenet_transfer.pth
```

**Outputs:**
- `../../models/mobilenet/mobilenet_transfer.pth` (trained model)
- `../../models/mobilenet/mobilenet_transfer_training_curves.png` (graphs for thesis)
- `../../models/mobilenet/mobilenet_transfer_training_summary.txt` (summary)

### **2. Run Tracker on Video**
```powershell
python yolo_mobilenet_tracker.py \
    --video ../../Mp4TESTING/opening-cabinet/vid1.mp4 \
    --mobilenet-model ../../models/mobilenet/mobilenet_transfer.pth \
    --save tracked_output.mp4 \
    --min-class-conf 0.6 \
    --smooth-window 7
```

### **3. Adjust Settings at Runtime**
While the tracker is running:
- Press `[` to decrease confidence threshold (more sensitive)
- Press `]` to increase confidence threshold (less sensitive)
- Press `q` to quit

---

## 🗑️ Deprecated Files (Can Be Removed)

These files are no longer needed if you're using `train_transfer_learning.py` + `yolo_mobilenet_tracker.py`:

### **Training Scripts (Old)**
- ❌ `train_focused_augmentation.py` - Replaced by `train_transfer_learning.py`
- ❌ `train_mobilenet_enhanced.py` - Replaced by `train_transfer_learning.py`

### **Inference Scripts (Old)**
- ❌ `inference_mobilenet.py` - Functionality integrated into `yolo_mobilenet_tracker.py`

### **Debugging/Analysis Tools**
- ❌ `analyze_dataset_diversity.py` - Dataset analysis (use if needed, otherwise remove)
- ❌ `debug_predictions.py` - Debug helper (use if needed, otherwise remove)

### **Documentation (Old)**
- ❌ `TRAINING_GRAPHS_GUIDE.md` - Basic guide, features now in `train_transfer_learning.py`
- ❌ `README_inference.md` - Old inference docs
- ❌ `DIAGNOSTIC_GUIDE.md` - Old troubleshooting guide

---

## 📊 Training Output Explanation

After training completes, you'll have:

### **1. Model Checkpoint** (`.pth`)
Your trained model - use this with the tracker:
```powershell
--mobilenet-model models/mobilenet/mobilenet_transfer.pth
```

### **2. Training Curves** (`.png`)
Two-panel graph showing:
- **Left:** Loss over epochs (lower is better)
- **Right:** Accuracy over epochs (higher is better)

**Use this in your thesis!** 300 DPI, publication-ready.

### **3. Training Summary** (`.txt`)
Text file with:
- Best validation accuracy and epoch
- Final train/val loss and accuracy
- Train-val gap (overfitting check)
- Per-class test accuracy
- Total training time

**Perfect for thesis methodology section!**

---

## 🎬 Tracker Features

### **Temporal Smoothing**
Averages predictions over recent frames to reduce flicker:
```
Before: walking → walking → grabbing → walking → grabbing
After:  walking → walking → walking → walking → walking
```

Controlled by `--smooth-window` (default: 7 frames).

### **Dynamic Confidence Threshold**
- Set initial threshold: `--min-class-conf 0.6`
- Adjust at runtime: `[` (decrease) / `]` (increase)
- Low-confidence predictions labeled "Neutral"
- Current threshold shown on overlay

### **Adaptive Classification**
Automatically adjusts re-classification frequency based on FPS:
- High FPS (>25): Classify every 5 frames (max accuracy)
- Medium FPS (15-25): Classify every 10 frames (balanced)
- Low FPS (<10): Classify every 30 frames (smooth playback)

---

## 🔧 System Requirements

**Minimum:**
- Python 3.8+
- 4GB RAM
- CPU (slow but works)

**Recommended:**
- Python 3.10+
- 8GB RAM
- NVIDIA GPU with 4GB+ VRAM
- CUDA 11.8+

**Dependencies:**
```powershell
pip install ultralytics torch torchvision opencv-python pillow numpy matplotlib
```

---

## 📁 Directory Structure

```
behavior_recognition/
├── MobileNetV2/                        ✅ Main training & tracking folder
│   ├── train_transfer_learning.py      ✅ Main training script
│   ├── yolo_mobilenet_tracker.py       ✅ Main inference/tracking script
│   ├── YOLO_MOBILENET_TRACKER.md       ✅ Tracker documentation
│   ├── YOLO_MOBILENET_IMPLEMENTATION.md ✅ Implementation guide
│   ├── COLAB_GUIDE.md                  ✅ Colab setup guide
│   ├── AI_IMAGE_GENERATION_PROMPTS.md  ✅ Dataset prompts
│   ├── TRAINING_GRAPHS_GUIDE.md        ✅ Training graphs guide
│   └── README.md                       ✅ This file
│
└── [Deprecated files - can be removed]
    ├── train_focused_augmentation.py
    ├── train_mobilenet_enhanced.py
    ├── inference_mobilenet.py
    ├── analyze_dataset_diversity.py
    ├── debug_predictions.py
    └── DIAGNOSTIC_GUIDE.md
```

---

## 💡 Common Commands

### **Train on your dataset:**
```powershell
python train_transfer_learning.py --data ../../datasets --epochs 30
```

### **Track people in video:**
```powershell
python yolo_mobilenet_tracker.py --video test.mp4 --save output.mp4
```

### **Use webcam:**
```powershell
python yolo_mobilenet_tracker.py --webcam
```

### **Headless mode (Colab/server):**
```powershell
python yolo_mobilenet_tracker.py --video test.mp4 --no-display --save output.mp4
```

### **High accuracy mode:**
```powershell
python yolo_mobilenet_tracker.py \
    --video ../../test.mp4 \
    --save output.mp4 \
    --reclass-interval 5 \
    --min-class-conf 0.7 \
    --smooth-window 10 \
    --conf 0.6
```

### **Fast/efficient mode:**
```powershell
python yolo_mobilenet_tracker.py \
    --video ../../test.mp4 \
    --save output.mp4 \
    --yolo-model ../../models/YOLOv8/yolov8n.pt \
    --reclass-interval 15 \
    --smooth-window 5 \
    --conf 0.4
```

---

## 🎓 For Your Thesis

### **Methods Section:**
```
The behavior recognition system uses a two-stage approach:
1. YOLOv8 for person detection and tracking (ByteTrack algorithm)
2. MobileNetV2 for behavior classification (transfer learning from ImageNet)

Temporal smoothing (7-frame window) reduces prediction flicker.
Dynamic confidence thresholding (0.6) filters low-confidence predictions.
```

### **Results Section:**
Include:
- Training curves PNG (`mobilenet_transfer_training_curves.png`)
- Final test accuracy from summary file
- Per-class accuracy breakdown
- Example tracker output screenshots/video clips

---

## 🆘 Need Help?

1. **Training issues?** Check the training summary `.txt` file for overfitting indicators
2. **Tracker not detecting?** Lower `--conf` threshold or use a larger YOLO model
3. **Predictions flickering?** Increase `--smooth-window` (try 10-15)
4. **Too many false positives?** Increase `--min-class-conf` at runtime with `]` key
5. **Slow performance?** Increase `--reclass-interval` or use YOLOv8 Nano

---

## 📝 Citation

If using this system in your thesis, cite:
- YOLOv8: [Ultralytics](https://github.com/ultralytics/ultralytics)
- MobileNetV2: Sandler et al. (2018) - "MobileNetV2: Inverted Residuals and Linear Bottlenecks"
- ByteTrack: Zhang et al. (2022) - "ByteTrack: Multi-Object Tracking by Associating Every Detection Box"

---

**Last Updated:** October 23, 2025  
**Maintainer:** Your thesis system  
**Status:** ✅ Production Ready
