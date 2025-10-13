# MobileNetV2 + LSTM Behavior Recognition Pipeline

Complete PyTorch-based pipeline for video behavior recognition using MobileNetV2 CNN + LSTM with person tracking.

## 🎯 Features

- **Person Tracking**: Uses YOLOv8 to detect and track individual persons across frames
- **CNN-LSTM Architecture**: MobileNetV2 for spatial features + LSTM for temporal modeling
- **Multi-Person Support**: Handles videos with 1-2 actors, creates separate sequences per person
- **End-to-End Pipeline**: Frame extraction → Training → Inference
- **Optimized**: Uses pre-trained MobileNetV2, class weighting, and data augmentation

## 📁 Directory Structure

```
behavior_recognition/
└── lstm_mn/
    ├── config.py                    # Configuration settings
    ├── extract_frames.py            # Step 1: Extract & track persons
    ├── train_mobilenet_lstm.py      # Step 2: Train CNN-LSTM model
    └── README.md                    # This file

datasets/
├── behavior_clips/                  # Input videos
│   ├── suspicious-actions/
│   │   ├── assault-fighting/
│   │   │   └── video1.mp4
│   │   └── stealing/
│   │       └── video2.mp4
│   └── neutral-actions/
│       ├── walking/
│       │   └── video3.mp4
│       └── sitting/
│           └── video4.mp4
└── behavior_frames/                 # Extracted frames (auto-generated)
    ├── suspicious-actions/
    │   └── assault-fighting/
    │       └── video1/
    │           ├── person_1/
    │           │   ├── frame0001.png
    │           │   └── ...
    │           └── person_2/
    │               └── ...
    └── neutral-actions/
        └── ...

models/
└── lstm_mn/
    └── best_mobilenet_lstm.pth      # Trained model
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install torch torchvision opencv-python ultralytics tqdm scikit-learn pillow
```

### 2. Prepare Your Dataset

Place your videos in the following structure:
```
datasets/behavior_clips/
├── suspicious-actions/
│   ├── class1/
│   │   └── *.mp4
│   └── class2/
│       └── *.mp4
└── neutral-actions/
    ├── class3/
    │   └── *.mp4
    └── class4/
        └── *.mp4
```

### 3. Extract Frames with Person Tracking

```bash
cd behavior_recognition/lstm_mn
python extract_frames.py
```

**What it does:**
- Detects and tracks persons using YOLOv8
- Crops each person individually per frame
- Resizes to 224x224 for MobileNetV2
- Saves sequences in `datasets/behavior_frames/`

**Output:**
```
============================================================
EXTRACTION SUMMARY
============================================================
assault-fighting         | Videos:  50 | Failed:   0 | Persons: 75
walking                  | Videos: 100 | Failed:   2 | Persons: 102
...
------------------------------------------------------------
TOTAL                    | Videos: 300 | Failed:   5 | Persons: 450
```

### 4. Train MobileNetV2 + LSTM Model

```bash
python train_mobilenet_lstm.py
```

**What it does:**
- Loads frame sequences (30 frames @ 30fps = 1 second)
- Extracts spatial features using pre-trained MobileNetV2
- Models temporal dynamics using LSTM
- Trains with class weighting and data augmentation

**Output:**
```
Epoch 1/50
----------------------------------------------------------
Train Loss: 1.5234 | Train Acc: 42.12%
Val Loss:   1.4123 | Val Acc:   45.67%
✓ Best model saved! (Val Acc: 45.67%)

...

FINAL EVALUATION ON TEST SET
🎯 Test Accuracy: 78.45%
```

## ⚙️ Configuration

Edit `config.py` to customize:

```python
# Frame extraction
TARGET_FPS = 30                      # Extract at 30fps
FRAME_WIDTH = 224                    # MobileNetV2 input size
FRAME_HEIGHT = 224
SEQUENCE_LENGTH = 30                 # 1 second of video
ENABLE_PERSON_TRACKING = True        # Use YOLO tracking
MAX_PERSONS_PER_VIDEO = 2            # Track up to 2 people

# Model architecture
LSTM_HIDDEN_DIM = 256                # LSTM hidden state size
LSTM_NUM_LAYERS = 2                  # Number of LSTM layers
LSTM_DROPOUT = 0.3                   # Dropout rate

# Training
BATCH_SIZE = 8                       # Batch size
LEARNING_RATE = 1e-4                 # Adam learning rate
NUM_EPOCHS = 50                      # Max epochs
EARLY_STOPPING_PATIENCE = 10         # Early stopping patience
```

## 📊 Model Architecture

```
Input: [batch, 30, 3, 224, 224]
   ↓
MobileNetV2 Feature Extractor (per frame)
   ↓
Features: [batch, 30, 1280]
   ↓
Bidirectional LSTM (2 layers, 256 hidden)
   ↓
Last Hidden State: [batch, 256]
   ↓
FC Layer → ReLU → Dropout
   ↓
FC Layer → Softmax
   ↓
Output: [batch, num_classes]
```

## 🎯 Key Features Explained

### Person Tracking

- **YOLOv8 Tracking**: Maintains consistent person IDs across frames
- **Multi-Person**: Each person gets a separate training sequence
- **Smart Cropping**: Focuses model on individual actors, not background

### CNN-LSTM Design

- **MobileNetV2**: Lightweight, pre-trained on ImageNet
- **Frozen Backbone**: Faster training, prevents overfitting
- **LSTM**: Captures temporal patterns (punch sequences, walking gait, etc.)

### Training Optimizations

- **Class Weighting**: Handles imbalanced datasets
- **Data Augmentation**: Random flips, color jitter
- **ReduceLROnPlateau**: Adaptive learning rate
- **Early Stopping**: Prevents overfitting

## 📈 Expected Performance

| Dataset Size | Expected Accuracy |
|--------------|-------------------|
| 500 sequences | 60-70% |
| 1000 sequences | 70-80% |
| 2000+ sequences | 80-90% |

## 🔧 Troubleshooting

### "No valid persons detected"
- Lower `YOLO_CONF_THRESHOLD` in config (default: 0.3)
- Check video quality - ensure people are clearly visible

### Low accuracy
- Increase `SEQUENCE_LENGTH` (capture more context)
- Add more training data
- Fine-tune MobileNetV2 (set `requires_grad=True`)

### Out of memory
- Reduce `BATCH_SIZE`
- Reduce `SEQUENCE_LENGTH`
- Use smaller LSTM hidden dim

## 📝 Next Steps

1. **Add More Classes**: Just add videos to `behavior_clips/suspicious-actions/new_class/`
2. **Fine-Tune MobileNetV2**: Enable gradients for better accuracy
3. **Real-Time Inference**: Process webcam or CCTV streams

## 🎓 References

- **MobileNetV2**: [Paper](https://arxiv.org/abs/1801.04381)
- **LSTM**: [Paper](http://www.bioinf.jku.at/publications/older/2604.pdf)
- **YOLOv8**: [Ultralytics](https://github.com/ultralytics/ultralytics)

---

**Built with PyTorch** 🔥
