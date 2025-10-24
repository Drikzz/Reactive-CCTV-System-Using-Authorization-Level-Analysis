# 🚀 Google Colab Guide - YOLOv8 + MobileNetV2 Tracker

## ⚠️ Important: Headless Environment

Google Colab is a **headless server** (no display), so you MUST use the `--no-display` flag!

---

## 📋 Quick Setup

### **Step 1: Clone Repository**
```python
# In Colab cell
!git clone https://github.com/Drikzz/Reactive-CCTV-System-Using-Authorization-Level-Analysis.git
%cd Reactive-CCTV-System-Using-Authorization-Level-Analysis
```

### **Step 2: Install Dependencies**
```python
# In Colab cell
!pip install ultralytics torch torchvision opencv-python-headless pillow numpy matplotlib
```

**Note:** Use `opencv-python-headless` for Colab (no GUI dependencies)

### **Step 3: Verify GPU**
```python
# Check CUDA availability
import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

---

## 🎯 Running the Tracker (CORRECT WAY)

### **✅ CORRECT: With `--no-display`**
```python
# In Colab cell
!python behavior_recognition/MobileNetV2/yolo_mobilenet_tracker.py \
    --video Mp4TESTING/opening-cabinet/vid1.mp4 \
    --no-display \
    --save output.mp4 \
    --device cuda
```

### **❌ WRONG: Without `--no-display`**
```python
# This will FAIL with Qt display error!
!python behavior_recognition/MobileNetV2/yolo_mobilenet_tracker.py \
    --video test.mp4 \
    --save output.mp4
```

**Error you'll get:**
```
qt.qpa.xcb: could not connect to display 
This application failed to start because no Qt platform plugin could be initialized.
```

---

## 📊 Complete Colab Workflow

### **Notebook Cell 1: Setup**
```python
# Clone repo
!git clone https://github.com/Drikzz/Reactive-CCTV-System-Using-Authorization-Level-Analysis.git
%cd Reactive-CCTV-System-Using-Authorization-Level-Analysis

# Install dependencies
!pip install ultralytics opencv-python-headless -q

# Verify GPU
import torch
print(f"✓ CUDA: {torch.cuda.is_available()}")
print(f"✓ GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
```

### **Notebook Cell 2: Upload Test Video (Optional)**
```python
from google.colab import files
uploaded = files.upload()

# Move to appropriate folder
!mkdir -p test_videos
!mv *.mp4 test_videos/
```

### **Notebook Cell 3: Run Tracker**
```python
# Process video with YOLOv8 + MobileNetV2 (with temporal smoothing)
!python behavior_recognition/MobileNetV2/yolo_mobilenet_tracker.py \
    --video Mp4TESTING/opening-cabinet/vid1.mp4 \
    --no-display \
    --save outputs/tracked_output.mp4 \
    --mobilenet-model models/mobilenetv2/mobilenet_transfer.pth \
    --device cuda \
    --reclass-interval 10 \
    --min-class-conf 0.6 \
    --smooth-window 7 \
    --conf 0.5
```

### **Notebook Cell 4: Check Output**
```python
# List output files
!ls -lh outputs/

# Get file info
import os
output_path = 'outputs/tracked_output.mp4'
if os.path.exists(output_path):
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"✓ Output created: {output_path}")
    print(f"  Size: {size_mb:.1f} MB")
else:
    print("❌ Output not found!")
```

### **Notebook Cell 5: Download Output**
```python
from google.colab import files

# Download the processed video
files.download('outputs/tracked_output.mp4')
```

---

## 🎨 Preview Output in Colab

Since Colab is headless, you can't use `cv2.imshow()`, but you can:

### **Option 1: Display Sample Frames**
```python
import cv2
from IPython.display import Image, display
import matplotlib.pyplot as plt

# Read a few frames
cap = cv2.VideoCapture('outputs/tracked_output.mp4')

for i in range(3):  # Show 3 sample frames
    cap.set(cv2.CAP_PROP_POS_FRAMES, i * 100)  # Every 100 frames
    ret, frame = cap.read()
    if ret:
        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        plt.title(f"Frame {i * 100}")
        plt.axis('off')
        plt.show()

cap.release()
```

### **Option 2: Create GIF Preview**
```python
from PIL import Image
import cv2

# Extract frames
cap = cv2.VideoCapture('outputs/tracked_output.mp4')
frames = []

for i in range(0, 300, 10):  # First 300 frames, every 10th
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    ret, frame = cap.read()
    if ret:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame_rgb))

cap.release()

# Save as GIF
if frames:
    frames[0].save('preview.gif', save_all=True, append_images=frames[1:], 
                   duration=100, loop=0)
    print("✓ Preview GIF created!")
    from IPython.display import Image as IPImage
    display(IPImage('preview.gif'))
```

### **Option 3: Upload to Google Drive**
```python
from google.colab import drive
drive.mount('/content/drive')

# Copy output to Drive
!cp outputs/tracked_output.mp4 /content/drive/MyDrive/

print("✓ Video saved to Google Drive!")
```

---

## 🔧 Common Colab Commands

### **Process Multiple Videos**
```python
import os
from pathlib import Path

# Process all videos in a folder
video_folder = 'Mp4TESTING/opening-cabinet'
output_folder = 'outputs'

for video_file in Path(video_folder).glob('*.mp4'):
    output_name = f"tracked_{video_file.name}"
    
    !python behavior_recognition/yolo_mobilenet_tracker.py \
        --video {video_file} \
        --no-display \
        --save {output_folder}/{output_name} \
        --device cuda \
        --reclass-interval 10
    
    print(f"✓ Processed: {video_file.name}")
```

### **Webcam Mode (Won't Work in Colab)**
```python
# ❌ This will FAIL in Colab (no webcam access)
!python behavior_recognition/MobileNetV2/yolo_mobilenet_tracker.py \
    --webcam \
    --no-display \
    --save webcam.mp4
```

---

## 📊 Performance Tips for Colab

### **1. Use Smaller YOLO Model for Speed**
```python
# Faster processing with YOLOv8 Nano
!python behavior_recognition/yolo_mobilenet_tracker.py \
    --video test.mp4 \
    --yolo-model models/YOLOv8/yolov8n.pt \
    --no-display \
    --save output.mp4 \
    --device cuda
```

### **2. Increase Classification Interval**
```python
# Less frequent classification = faster processing
!python behavior_recognition/yolo_mobilenet_tracker.py \
    --video test.mp4 \
    --reclass-interval 20 \
    --no-display \
    --save output.mp4 \
    --device cuda
```

### **3. Lower Detection Thresholds**
```python
# Fewer detections = faster processing
!python behavior_recognition/yolo_mobilenet_tracker.py \
    --video test.mp4 \
    --conf 0.6 \
    --no-display \
    --save output.mp4 \
    --device cuda
```

---

## 🐛 Troubleshooting

### **Error: "qt.qpa.xcb: could not connect to display"**
**Solution:** Add `--no-display` flag!
```python
!python behavior_recognition/yolo_mobilenet_tracker.py \
    --video test.mp4 \
    --no-display \  # ← ADD THIS!
    --save output.mp4
```

### **Error: "No module named 'ultralytics'"**
**Solution:** Install ultralytics
```python
!pip install ultralytics
```

### **Error: "CUDA out of memory"**
**Solution:** Use smaller model or increase interval
```python
!python behavior_recognition/yolo_mobilenet_tracker.py \
    --video test.mp4 \
    --yolo-model models/YOLOv8/yolov8n.pt \  # Smaller model
    --reclass-interval 20 \  # Less frequent classification
    --no-display \
    --save output.mp4
```

### **Error: "Model not found"**
**Solution:** Check model paths
```python
# Verify models exist
!ls -lh models/YOLOv8/
!ls -lh models/mobilenetv2/
```

---

## 📝 Complete Colab Notebook Template

```python
# ========================================
# CELL 1: Setup
# ========================================
!git clone https://github.com/Drikzz/Reactive-CCTV-System-Using-Authorization-Level-Analysis.git
%cd Reactive-CCTV-System-Using-Authorization-Level-Analysis
!pip install ultralytics opencv-python-headless -q

import torch
print(f"✓ CUDA: {torch.cuda.is_available()}")
print(f"✓ GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

# ========================================
# CELL 2: Run Tracker
# ========================================
!python behavior_recognition/yolo_mobilenet_tracker.py \
    --video Mp4TESTING/opening-cabinet/vid1.mp4 \
    --no-display \
    --save outputs/tracked_output.mp4 \
    --device cuda \
    --reclass-interval 10 \
    --conf 0.5

# ========================================
# CELL 3: Preview Output
# ========================================
import cv2
import matplotlib.pyplot as plt

cap = cv2.VideoCapture('outputs/tracked_output.mp4')
cap.set(cv2.CAP_PROP_POS_FRAMES, 100)
ret, frame = cap.read()

if ret:
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.title("Sample Frame from Tracked Video")
    plt.axis('off')
    plt.show()

cap.release()

# ========================================
# CELL 4: Download Output
# ========================================
from google.colab import files
files.download('outputs/tracked_output.mp4')
```

---

## ✅ Your Fixed Command

Replace this:
```python
# ❌ WRONG
!python behavior_recognition/yolo_mobilenet_tracker.py \
    --video ../Mp4TESTING/opening-cabinet/vid1.mp4 \
    --save output.mp4
```

With this:
```python
# ✅ CORRECT
!python behavior_recognition/MobileNetV2/yolo_mobilenet_tracker.py \
    --video Mp4TESTING/opening-cabinet/vid1.mp4 \
    --no-display \
    --save tracked_output.mp4 \
    --device cuda \
    --min-class-conf 0.6 \
    --smooth-window 7
```

---

## 🎓 Training Your Model in Colab

If you need to train a new MobileNetV2 model:

```python
# Train with transfer learning
!python behavior_recognition/MobileNetV2/train_transfer_learning.py \
    --data datasets \
    --epochs 30 \
    --batch-size 16 \
    --lr 1e-4 \
    --model-name mobilenet_transfer.pth \
    --device cuda

# Check outputs
!ls -lh models/mobilenetv2/
# You'll see:
# - mobilenet_transfer.pth (trained model)
# - mobilenet_transfer_training_curves.png (graphs)
# - mobilenet_transfer_training_summary.txt (summary)

# Download trained model
from google.colab import files
files.download('models/mobilenetv2/mobilenet_transfer.pth')
files.download('models/mobilenetv2/mobilenet_transfer_training_curves.png')
files.download('models/mobilenetv2/mobilenet_transfer_training_summary.txt')
```

---

## 🎉 Summary

**Always use in Colab:**
- ✅ `--no-display` flag (REQUIRED for headless)
- ✅ `--save` to save output video
- ✅ `--device cuda` to use GPU
- ✅ `opencv-python-headless` package
- ✅ Download output with `files.download()`

**Never use in Colab:**
- ❌ Without `--no-display` (will fail)
- ❌ `--webcam` mode (no webcam in Colab)
- ❌ `cv2.imshow()` (use matplotlib instead)

---

**Date:** October 14, 2025  
**Status:** ✅ Colab Ready
