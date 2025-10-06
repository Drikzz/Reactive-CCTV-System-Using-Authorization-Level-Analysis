# 🎓 Google Colab Inference Guide

Run behavior recognition inference in Google Colab with automatic output in the same folder.

## 🚀 Quick Start (Google Colab)

### 1. Setup Environment

```python
# Mount Google Drive (optional - for persistent storage)
from google.colab import drive
drive.mount('/content/drive')

# Clone repository
!git clone https://github.com/Drikzz/Reactive-CCTV-System-Using-Authorization-Level-Analysis.git
%cd Reactive-CCTV-System-Using-Authorization-Level-Analysis

# Install dependencies
!pip install ultralytics opencv-python torch tqdm scikit-learn

# Check GPU
import torch
print(f"GPU Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
```

### 2. Upload Video

```python
from google.colab import files

# Upload video file
print("📤 Upload your video file:")
uploaded = files.upload()

# Get filename
video_file = list(uploaded.keys())[0]
print(f"✓ Uploaded: {video_file}")
```

### 3. Run Inference

```python
# Run inference (output in same folder with _annotated suffix)
!python behavior_recognition/inference_colab.py --video {video_file}

# Or specify custom output
# !python behavior_recognition/inference_colab.py --video {video_file} --output result.mp4
```

### 4. Download Result

```python
import glob

# Find annotated video
annotated_video = glob.glob("*_annotated.*")[0]
print(f"📥 Downloading: {annotated_video}")

# Download
files.download(annotated_video)
```

---

## 📋 Complete Colab Notebook Example

```python
# ===============================
# CELL 1: Setup
# ===============================
!git clone https://github.com/Drikzz/Reactive-CCTV-System-Using-Authorization-Level-Analysis.git
%cd Reactive-CCTV-System-Using-Authorization-Level-Analysis
!pip install -q ultralytics opencv-python torch tqdm scikit-learn

# ===============================
# CELL 2: Check GPU
# ===============================
import torch
print(f"🎮 GPU: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   Device: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ===============================
# CELL 3: Upload Video
# ===============================
from google.colab import files
uploaded = files.upload()
video_file = list(uploaded.keys())[0]
print(f"✓ Video: {video_file}")

# ===============================
# CELL 4: Run Inference
# ===============================
!python behavior_recognition/inference_colab.py --video {video_file} --device cuda

# ===============================
# CELL 5: Download Result
# ===============================
import glob
result = glob.glob("*_annotated.*")[0]
print(f"📥 Result: {result}")
files.download(result)
```

---

## 🎯 Features

✅ **Automatic Output Naming** - Saves as `input_annotated.mp4` in same folder  
✅ **Notebook Progress Bars** - Uses `tqdm.notebook` for clean display  
✅ **GPU Acceleration** - Automatic CUDA detection and usage  
✅ **Statistics Summary** - Shows detected behaviors and counts  
✅ **No Display Required** - Optimized for headless Colab environment  
✅ **Easy Download** - Simple one-line download command  

---

## ⚙️ Command-Line Options

```bash
# Basic usage (output: video_annotated.mp4)
!python behavior_recognition/inference_colab.py --video video.mp4

# Custom output path
!python behavior_recognition/inference_colab.py --video video.mp4 --output result.mp4

# Use CPU instead of GPU
!python behavior_recognition/inference_colab.py --video video.mp4 --device cpu

# Custom models
!python behavior_recognition/inference_colab.py \
    --video video.mp4 \
    --model path/to/lstm.pth \
    --pose-model path/to/yolo.pt

# Disable notebook progress bar
!python behavior_recognition/inference_colab.py --video video.mp4 --no-notebook
```

---

## 📊 Output Statistics

After processing, you'll see:

```
============================================================
PROCESSING COMPLETE
============================================================
✓ Output saved: video_annotated.mp4
✓ Processed: 1500 frames in 45.3s
✓ Average FPS: 33.1
✓ Total detections: 1247

📊 Detected Behaviors:
   normal: 892 instances
   assault-fighting: 234 instances
   stealing: 121 instances
============================================================
```

---

## 🎬 Output Format

**Input:** `my_video.mp4`  
**Output:** `my_video_annotated.mp4` (same folder)

The output video includes:
- ✅ Person tracking boxes (color-coded)
- ✅ Person IDs
- ✅ Pose skeletons (17 keypoints)
- ✅ Behavior labels with confidence
- ✅ Info panel (frame count, tracked persons, window size)

---

## 💡 Tips for Colab

### 1. Use GPU Runtime
```
Runtime → Change runtime type → Hardware accelerator → GPU (T4)
```

### 2. Keep Session Alive
```python
# Ping to prevent disconnect
import time
from IPython.display import display, Javascript

def keep_alive():
    display(Javascript('function KeepClicking(){console.log("Clicking");document.querySelector("colab-toolbar-button").click()}setInterval(KeepClicking,60000)'))

keep_alive()
```

### 3. Save to Google Drive
```python
# Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Save output to Drive
!python behavior_recognition/inference_colab.py \
    --video video.mp4 \
    --output /content/drive/MyDrive/results/output.mp4
```

### 4. Process Multiple Videos
```python
import glob
from google.colab import files

# Upload multiple videos
uploaded = files.upload()

# Process each
for video in uploaded.keys():
    print(f"\n🎬 Processing {video}...")
    !python behavior_recognition/inference_colab.py --video {video}

# Download all results
for result in glob.glob("*_annotated.*"):
    files.download(result)
```

---

## 🔧 Troubleshooting

### Issue: "Model not found"
```python
# Check if models exist
!ls models/lstm_mn/
!ls models/YOLOv8/

# If missing, train first or download
# Training:
# !python behavior_recognition/run_training.py
```

### Issue: "Out of memory"
```bash
# Use CPU instead
!python behavior_recognition/inference_colab.py --video video.mp4 --device cpu

# Or reduce batch processing (modify config.py)
```

### Issue: "Video upload failed"
```python
# Use wget for large files
!wget https://example.com/video.mp4

# Or use gdown for Google Drive links
!pip install gdown
!gdown --id YOUR_FILE_ID
```

---

## 📱 Kaggle Support

Works on Kaggle notebooks too!

```python
# Kaggle setup
import os
os.chdir('/kaggle/working')

# Upload via Kaggle dataset or URL
!wget YOUR_VIDEO_URL

# Run inference
!python inference_colab.py --video video.mp4

# Files available in Output tab
```

---

## 🎨 Visualization in Colab

### Display Video in Notebook
```python
from IPython.display import Video

# Show original
Video("video.mp4", width=640)

# Show result
Video("video_annotated.mp4", width=640)
```

### Show Sample Frames
```python
import cv2
from matplotlib import pyplot as plt

cap = cv2.VideoCapture("video_annotated.mp4")
ret, frame = cap.read()
cap.release()

plt.figure(figsize=(12, 8))
plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title('Sample Frame - Behavior Recognition')
plt.show()
```

---

## 📁 File Structure

```
Reactive-CCTV-System-Using-Authorization-Level-Analysis/
├── behavior_recognition/
│   ├── inference_colab.py       ← Use this in Colab
│   ├── config.py
│   └── train_lstm_classifier.py
├── models/
│   ├── lstm_mn/
│   │   └── best_model.pth       ← LSTM model
│   └── YOLOv8/
│       └── yolov8m-pose.pt      ← YOLO model
├── video.mp4                     ← Your input
└── video_annotated.mp4           ← Generated output
```

---

## ✨ Differences from Standard Inference

| Feature | Standard | Colab |
|---------|----------|-------|
| Progress Bar | Console | Notebook-optimized |
| Output Path | Custom | Same folder (auto) |
| Display | Real-time CV2 | Headless (no display) |
| Download | Manual | One-line command |
| Stats | Terminal | Clean summary |

---

## 🚀 Quick Commands

```bash
# Minimal (output: input_annotated.mp4)
!python behavior_recognition/inference_colab.py --video vid.mp4

# With custom output
!python behavior_recognition/inference_colab.py --video vid.mp4 --output result.mp4

# CPU mode
!python behavior_recognition/inference_colab.py --video vid.mp4 --device cpu
```

---

## 📞 Support

**Colab-Specific Issues:**
- GPU not detected → Check runtime type
- Upload timeout → Use wget/gdown for large files
- Session disconnect → Use keep-alive script
- Out of memory → Use CPU mode

**Ready for Google Colab inference!** 🎉
