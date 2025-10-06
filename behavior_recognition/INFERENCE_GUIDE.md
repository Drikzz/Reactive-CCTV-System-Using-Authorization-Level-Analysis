# 🎯 Behavior Recognition Inference Guide

Real-time behavior classification with person tracking using YOLOv8 + LSTM.

## 🌟 Features

✅ **Multi-Person Tracking** - Track and classify multiple people simultaneously  
✅ **Real-time Processing** - Smooth inference on webcam or video files  
✅ **Temporal Windowing** - Uses 60-frame sequences for accurate classification  
✅ **Prediction Smoothing** - Majority voting over 5 predictions for stability  
✅ **Visual Feedback** - Skeleton overlay, bounding boxes, and behavior labels  
✅ **Auto Cleanup** - Removes inactive tracks automatically  

---

## 📋 Prerequisites

Before running inference, ensure you have:

1. **Trained LSTM Model**
   - Location: `models/lstm_mn/best_model.pth`
   - Train using: `python run_training.py`

2. **YOLOv8-Pose Model**
   - Location: `models/YOLOv8/yolov8m-pose.pt`
   - Should already be downloaded

3. **Required Libraries**
   ```powershell
   pip install ultralytics opencv-python torch tqdm
   ```

---

## 🚀 Usage

### 1. Webcam Inference (Default)

```powershell
cd behavior_recognition
python run_inference.py --webcam
```

**Controls:**
- Press `q` to quit
- Press `s` to save screenshot

### 2. Webcam with Specific Camera

```powershell
# Use camera 0 (default)
python run_inference.py --webcam --camera-id 0

# Use camera 1 (external webcam)
python run_inference.py --webcam --camera-id 1
```

### 3. Video File Inference

```powershell
# Process and display
python run_inference.py --video path/to/video.mp4

# Process and save output
python run_inference.py --video input.mp4 --output runs/inference/output.mp4
```

### 4. CPU Mode (No GPU)

```powershell
python run_inference.py --webcam --device cpu
```

---

## ⚙️ Command-Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--webcam` | Use webcam as input source | - |
| `--video` | Path to input video file | - |
| `--camera-id` | Camera device ID (0, 1, 2...) | 0 |
| `--output` | Path to save output video | None |
| `--model` | Path to LSTM model | `models/lstm_mn/best_model.pth` |
| `--pose-model` | Path to YOLOv8-pose | `models/YOLOv8/yolov8m-pose.pt` |
| `--device` | Device to use (`cuda` or `cpu`) | `cuda` |

---

## 📊 How It Works

### 1. **Person Detection & Tracking**
```
Frame → YOLOv8-pose → Detect persons → Track IDs
                    ↓
              Extract 17 keypoints per person
```

### 2. **Temporal Windowing**
```
Person ID: 1
├── Frame 1: [17x2 keypoints]
├── Frame 2: [17x2 keypoints]
├── ...
└── Frame 60: [17x2 keypoints] → Ready for classification!
```

### 3. **Behavior Classification**
```
Sequence [60, 17, 2] → Flatten → [60, 34]
                                    ↓
                              LSTM Classifier
                                    ↓
                          Predicted Behavior + Confidence
```

### 4. **Prediction Smoothing**
```
Last 5 predictions: [0, 0, 1, 0, 0]
Majority vote: 0 (assault-fighting)
Display: "assault-fighting (0.92)"
```

---

## 🎨 Visualization

The inference displays:

1. **Bounding Boxes** - Color-coded per person
2. **Person ID** - Unique tracking ID
3. **Pose Skeleton** - 17 COCO keypoints connected
4. **Behavior Label** - Current predicted behavior
5. **Confidence** - Prediction confidence percentage
6. **Statistics Panel**:
   - FPS (Frames Per Second)
   - Tracked Persons count
   - Sequence window size

---

## 📈 Performance Tips

### For Better FPS:

1. **Use GPU** (CUDA)
   ```powershell
   python run_inference.py --webcam --device cuda
   ```

2. **Lower Resolution**
   - Webcam auto-set to 1280x720
   - Modify in code if needed

3. **Reduce Sequence Length**
   - Edit `config.py`: `FIXED_SEQUENCE_LENGTH = 30` (faster but less accurate)

### For Better Accuracy:

1. **Increase Smoothing Window**
   - Modify `smoothing_window=5` to `smoothing_window=10` in `inference_behavior.py`

2. **Longer Sequences**
   - Keep `FIXED_SEQUENCE_LENGTH = 60` or increase to 100

---

## 🛠️ Troubleshooting

### Issue: "LSTM model not found"
**Solution:**
```powershell
# Train the model first
python run_training.py
```

### Issue: "Cannot open camera"
**Solution:**
```powershell
# Try different camera IDs
python run_inference.py --webcam --camera-id 1
python run_inference.py --webcam --camera-id 2
```

### Issue: Low FPS
**Solutions:**
1. Use GPU: `--device cuda`
2. Lower confidence threshold in `config.py`
3. Reduce sequence length

### Issue: Flickering predictions
**Solution:**
- Increase smoothing window in `PersonTracker.__init__(smoothing_window=10)`

---

## 💡 Examples

### Example 1: Monitor Security Camera
```powershell
# Use external USB camera with output saving
python run_inference.py --webcam --camera-id 1 --output security_footage.mp4
```

### Example 2: Analyze Existing Footage
```powershell
# Process CCTV recording
python run_inference.py --video cctv_recording.mp4 --output analyzed_output.mp4
```

### Example 3: Quick Test on Laptop
```powershell
# Use built-in webcam with CPU
python run_inference.py --webcam --device cpu
```

---

## 📁 Output Structure

When saving output:
```
runs/
└── inference/
    ├── output.mp4          # Annotated video
    └── screenshot_0000.jpg # Screenshots (press 's')
```

---

## 🎯 Expected Results

**Display shows:**
```
┌────────────────────────────────────┐
│ FPS: 28.5                         │
│ Tracked Persons: 2                │
│ Window: 60 frames                 │
└────────────────────────────────────┘

Person 1: [Bounding Box]
├── ID: 1 | stealing (87.3%)
└── [Skeleton overlay]

Person 2: [Bounding Box]
├── ID: 2 | Collecting...
└── [Skeleton overlay]
```

---

## 🚀 Quick Start Checklist

- [ ] Train LSTM model (`python run_training.py`)
- [ ] Verify model exists (`models/lstm_mn/best_model.pth`)
- [ ] Connect webcam or prepare video file
- [ ] Run inference (`python run_inference.py --webcam`)
- [ ] Press `q` to quit, `s` to screenshot

---

## 📞 Support

**Common Issues:**
- Model not found → Train first
- No camera → Check `--camera-id`
- Low FPS → Use `--device cuda`
- Unstable predictions → Increase smoothing

**File Locations:**
- Inference script: `behavior_recognition/inference_behavior.py`
- Launcher: `behavior_recognition/run_inference.py`
- Config: `behavior_recognition/config.py`

---

## ✨ Features Summary

| Feature | Description |
|---------|-------------|
| **Multi-Person** | Track and classify multiple people |
| **Real-time** | 25-30 FPS on GPU |
| **Temporal** | 60-frame sequence window |
| **Smoothing** | 5-frame majority voting |
| **Auto-cleanup** | Remove inactive tracks |
| **Visualization** | Skeletons + boxes + labels |
| **Flexible Input** | Webcam or video file |
| **Save Output** | Optional video saving |

Ready to classify behaviors in real-time! 🎉
