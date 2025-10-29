# ShuffleNetV2 Transfer Learning for Behavior Recognition

This folder contains transfer learning implementation using **ShuffleNetV2 x1.0** for behavior classification, integrated with YOLOv8 tracking.

## 📁 Contents

- `train_transfer_learning_shufflenetv2.py` - Training script for ShuffleNetV2
- `yolo_mobilenet_tracker_shufflenetv2.py` - YOLOv8 + ShuffleNetV2 integrated tracker
- `README.md` - This file

---

## 🚀 Quick Start

### 1. Train ShuffleNetV2 Model

```bash
cd behavior_recognition/shufflenetv2
python train_transfer_learning_shufflenetv2.py --data ../../datasets --epochs 30
```

**Arguments:**
- `--data`: Dataset root directory (default: `datasets`)
- `--epochs`: Number of training epochs (default: 30)
- `--batch-size`: Batch size (default: 16)
- `--lr`: Learning rate (default: 1e-4)
- `--save-dir`: Save directory (default: `models/shufflenetv2`)
- `--model-name`: Model filename (default: `shufflenet_v2_transfer.pth`)
- `--device`: Device to use (`cuda` or `cpu`, default: auto-detect)

### 2. Run Integrated Tracker

```bash
python yolo_mobilenet_tracker_shufflenetv2.py --video ../../path/to/video.mp4
```

**Arguments:**
- `--video`: Path to input video
- `--webcam`: Use webcam input
- `--yolo-model`: YOLOv8 model path (default: `models/YOLOv8/yolov8m.pt`)
- `--mobilenet-model`: ShuffleNetV2 model path (default: `models/shufflenetv2/shufflenet_v2_transfer.pth`)
- `--tracker`: Tracker config (default: `bytetrack.yaml`)
- `--reclass-interval`: Re-classification interval (default: 10 frames)
- `--conf`: YOLO confidence threshold (default: 0.5)
- `--iou`: YOLO IOU threshold (default: 0.7)
- `--min-class-conf`: Min classification confidence (default: 0.6)
- `--smooth-window`: Temporal smoothing window (default: 7 frames)
- `--no-display`: Don't display output window
- `--save`: Save output video path
- `--device`: Device (`cuda` or `cpu`)
- `--classes`: Class names (optional, auto-detected)

---

## 🏗️ Model Architecture

### ShuffleNetV2 x1.0
- **Parameters:** ~2.3M (smallest among the three models)
- **Architecture:** Efficient channel shuffle operations
- **Classifier:** Single fully connected layer (`fc`)
- **Pretrained:** ImageNet1K_V1
- **Input:** 224x224 RGB images

### Training Configuration
- **Optimizer:** Adam (lr=1e-4)
- **Scheduler:** ReduceLROnPlateau (patience=5, factor=0.5)
- **Loss:** CrossEntropyLoss
- **Augmentation:** Focused augmentation (see below)

### Augmentation Pipeline
```python
- Resize: 256x256
- RandomResizedCrop: 224x224 (scale: 0.8-1.0)
- RandomHorizontalFlip: p=0.5
- RandomRotation: ±15°
- ColorJitter: brightness/contrast/saturation=0.3, hue=0.1
- RandomAffine: ±10°, translate=0.08
- Normalize: ImageNet mean/std
- RandomErasing: p=0.12 (scale: 0.02-0.08)
```

---

## 🎯 Features

### Training Script
- ✅ Full fine-tuning (all layers unfrozen)
- ✅ Per-class accuracy tracking
- ✅ Best model checkpoint saving
- ✅ Training curves (Loss & Accuracy plots)
- ✅ Training summary report
- ✅ Test set evaluation
- ✅ Clean console logs for thesis

### Tracker Script
- ✅ YOLOv8 person detection + ByteTrack
- ✅ Per-track classification caching
- ✅ Adaptive re-classification intervals (FPS-based)
- ✅ Temporal smoothing (probability averaging)
- ✅ Dynamic confidence threshold (adjustable with '[' and ']' keys)
- ✅ Real-time FPS display
- ✅ Video output saving
- ✅ Multi-person tracking with color-coded IDs

---

## 📊 Model Comparison

| Model | Parameters | Classifier Layer | Speed | Accuracy | Best For |
|-------|-----------|------------------|-------|----------|----------|
| **MobileNetV2** | ~3.5M | `classifier[1]` | Baseline | Highest | Balanced accuracy/speed |
| **MobileNetV3-Small** | ~2.5M | `classifier[3]` | ~30% faster | Good | Mobile/edge devices |
| **ShuffleNetV2** | ~2.3M | `fc` | Fastest | Good | Real-time applications |

### When to Use ShuffleNetV2
- ✅ Need **maximum inference speed**
- ✅ **Smallest model size** requirement
- ✅ **Real-time** tracking applications
- ✅ **Resource-constrained** environments (embedded systems)
- ✅ Comparative analysis showing speed vs accuracy tradeoffs

### When to Use Others
- **MobileNetV2:** When you need best accuracy with reasonable speed
- **MobileNetV3-Small:** When you need a balance between ShuffleNetV2 speed and MobileNetV2 accuracy

---

## 📈 Output Files

### Training Outputs (saved to `models/shufflenetv2/`)
- `shufflenet_v2_transfer.pth` - Best model checkpoint
- `shufflenet_v2_transfer_training_curves.png` - Loss & accuracy plots
- `shufflenet_v2_transfer_training_summary.txt` - Training summary report

### Checkpoint Contents
```python
{
    'model_state_dict': state_dict,
    'optimizer_state_dict': optimizer_state_dict,
    'num_classes': int,
    'class_names': list,
    'best_val_acc': float,
    'history': dict,
    'epoch': int,
    'model_type': 'shufflenet_v2_x1_0'
}
```

---

## 🔧 Advanced Usage

### Custom Dataset Structure
```
datasets/
├── train/
│   ├── class1/
│   └── class2/
├── valid/
│   ├── class1/
│   └── class2/
└── test/
    ├── class1/
    └── class2/
```

### Training with Custom Settings
```bash
python train_transfer_learning_shufflenetv2.py \
    --data ../../datasets \
    --epochs 50 \
    --batch-size 32 \
    --lr 5e-5 \
    --save-dir ../../models/shufflenetv2_custom \
    --model-name shufflenet_v2_custom.pth
```

### Tracker with Custom Model
```bash
python yolo_mobilenet_tracker_shufflenetv2.py \
    --video ../../test_video.mp4 \
    --mobilenet-model ../../models/shufflenetv2_custom/shufflenet_v2_custom.pth \
    --reclass-interval 15 \
    --smooth-window 10 \
    --save tracked_output.mp4
```

---

## 🧪 Adaptive Classification

The tracker automatically adjusts classification intervals based on processing FPS:

- **>25 FPS:** Classify every 5 frames (high frequency)
- **15-25 FPS:** Classify every 10 frames (balanced)
- **10-15 FPS:** Classify every 20 frames (reduced)
- **<10 FPS:** Classify every 30 frames (minimum)

This ensures smooth tracking performance regardless of hardware capabilities.

---

## 🎮 Runtime Controls

While tracker is running:
- Press `q` - Quit
- Press `[` - Decrease minimum classification confidence
- Press `]` - Increase minimum classification confidence

---

## 📝 Notes

### Optional Enhancements
The training script includes notes for optional techniques:
- **Label Smoothing:** Helps with small datasets
- **Dropout:** Reduces overfitting
- **Mixup:** Improves generalization
- **Weighted Sampling:** Handles class imbalance

### Thesis Integration
All outputs are designed for thesis inclusion:
- High-resolution plots (300 DPI)
- Clean console logs
- Structured summary reports
- Per-class metrics

---

## 🔗 Related Files

- `../MobileNetV2/` - MobileNetV2 implementation (baseline)
- `../mobilenetv3-small/` - MobileNetV3-Small implementation
- `../../models/shufflenetv2/` - Trained model checkpoints
- `../../datasets/` - Training/validation/test data

---

## 📚 Architecture Details

### ShuffleNetV2 Key Features
1. **Channel Shuffle:** Efficient information flow between groups
2. **Channel Split:** Reduces computation by splitting feature maps
3. **Depthwise Convolutions:** Lightweight spatial feature extraction
4. **Linear Bottlenecks:** Preserves information flow
5. **Efficient Design:** Optimized for speed on mobile/embedded devices

### Comparison to MobileNets
- **ShuffleNetV2:** Uses channel shuffle + split operations (fastest)
- **MobileNetV2:** Uses inverted residuals + linear bottlenecks (balanced)
- **MobileNetV3-Small:** Uses SE modules + hard-swish (efficient + accurate)

---

## ⚡ Performance Tips

1. **Batch Size:** Increase for faster training (if GPU memory allows)
2. **Workers:** Adjust `num_workers` in DataLoader for your CPU
3. **Mixed Precision:** Consider using AMP for faster training
4. **TensorRT:** Convert model to TensorRT for maximum inference speed
5. **ONNX Export:** Export to ONNX for cross-platform deployment

---

## 🐛 Troubleshooting

### Model not found
- Ensure you've trained the model first
- Check the path in `--mobilenet-model` argument

### Low FPS during tracking
- Use smaller YOLO model (e.g., `yolov8n.pt`)
- Increase `--reclass-interval`
- Reduce video resolution
- Use GPU if available

### Overfitting
- Reduce learning rate
- Increase augmentation strength
- Add dropout before classifier
- Use label smoothing

---

## 📖 Citation

If you use ShuffleNetV2 in your research, please cite:
```
@inproceedings{ma2018shufflenet,
  title={Shufflenet v2: Practical guidelines for efficient cnn architecture design},
  author={Ma, Ningning and Zhang, Xiangyu and Zheng, Hai-Tao and Sun, Jian},
  booktitle={Proceedings of the European conference on computer vision (ECCV)},
  pages={116--131},
  year={2018}
}
```

---

## ✅ Verification Checklist

Before running experiments:
- [ ] Dataset properly structured (train/valid/test splits)
- [ ] GPU available and CUDA working (if using GPU)
- [ ] Required packages installed (torch, torchvision, ultralytics, opencv, etc.)
- [ ] YOLOv8 model downloaded
- [ ] Sufficient disk space for checkpoints and outputs

---

**Ready to train your ShuffleNetV2 model for behavior recognition! 🚀**
