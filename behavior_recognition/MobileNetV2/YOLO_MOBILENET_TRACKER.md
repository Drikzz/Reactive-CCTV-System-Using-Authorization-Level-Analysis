# 🎯 YOLOv8 + MobileNetV2 Integrated Tracker

## Overview

This system combines **YOLOv8 object tracking** with **MobileNetV2 behavior classification** to detect, track, and classify multiple people in real-time video streams.

---

## 🌟 Features

### ✅ **YOLOv8 Tracking**
- Uses ByteTrack algorithm for robust multi-person tracking
- Assigns unique track IDs to each person
- Handles occlusions and re-identifications
- Configurable confidence and IOU thresholds

### ✅ **MobileNetV2 Classification**
- Crops each tracked person bounding box
- Classifies behavior (e.g., "opening-cabinet", "using-computer")
- Uses trained MobileNetV2 model

### ✅ **Smart Caching System**
- Caches last classification per `track_id`
- Re-classifies only every N frames (configurable)
- Reduces computational overhead by 80-90%

### ✅ **Temporal Smoothing** 🎬
- Averages class probabilities over a sliding window
- Reduces prediction flicker (e.g., walking → grabbing → walking)
- Configurable window size (default: 7 frames)
- Makes frame-based models look more stable

### ✅ **Dynamic Confidence Threshold**
- Runtime-adjustable minimum classification confidence
- Press `[` to decrease, `]` to increase (0.05 steps)
- Low-confidence predictions labeled as "Neutral"
- Threshold displayed on overlay

### ✅ **Adaptive Classification Intervals** 🧠
- Automatically adjusts re-classification frequency based on FPS
- High FPS (>25): Classify every 5 frames
- Medium FPS (15-25): Baseline interval
- Low FPS (<10): Classify every 30-40 frames
- Maintains smooth performance across hardware

### ✅ **Rich Annotations**
- Displays track ID, behavior class, and confidence
- Color-coded bounding boxes per track
- Real-time FPS and interval information
- Tracks count and statistics

---

## 📋 Requirements

```bash
# Install dependencies
pip install ultralytics torch torchvision opencv-python pillow numpy matplotlib
```

**Required Models:**
- YOLOv8 model: `models/YOLOv8/yolov8m.pt` (or any YOLOv8 variant)
- MobileNetV2 model: `models/mobilenetv2/mobilenet_transfer.pth` (trained with transfer learning)

---

## 🚀 Quick Start

### **Basic Usage**

```powershell
# Process a video file
python yolo_mobilenet_tracker.py --video test.mp4 --save output.mp4

# Use webcam
python yolo_mobilenet_tracker.py --webcam --save webcam_output.mp4

# Headless mode (for servers/Colab)
python yolo_mobilenet_tracker.py --video test.mp4 --no-display --save output.mp4
```

---

## 🎛️ Command-Line Options

### **Input Source**
```powershell
--video PATH          # Path to video file
--webcam              # Use webcam (camera index 0)
```

### **Model Configuration**
```powershell
--yolo-model PATH     # YOLOv8 model path (default: models/YOLOv8/yolov8m.pt)
--mobilenet-model PATH # MobileNetV2 model path (default: models/mobilenetv2/mobilenet_transfer.pth)
--tracker CONFIG      # Tracker config (default: bytetrack.yaml)
```

### **Classification Settings**
```powershell
--reclass-interval N  # Re-classify every N frames (default: 10)
                      # Adaptive system will adjust this based on FPS
--min-class-conf FLOAT # Minimum classification confidence 0-1 (default: 0.6)
                      # Predictions below this are labeled "Neutral"
                      # Adjust at runtime with [ and ] keys
--smooth-window N     # Temporal smoothing window size (default: 7)
                      # Averages predictions over N frames to reduce flicker
--classes CLASS1 CLASS2 ...  # Behavior class names (optional, auto-detected)
```

### **Detection Thresholds**
```powershell
--conf FLOAT         # YOLO confidence threshold (default: 0.5)
--iou FLOAT          # YOLO IOU threshold (default: 0.7)
```

### **Output Options**
```powershell
--save PATH          # Save output video (auto-creates outputs/ folder)
--no-display         # Don't show video window (for headless)
--device cuda/cpu    # Force device (default: auto-detect)
```

---

## 📊 Example Commands

### **1. High-Quality Tracking**
```powershell
python yolo_mobilenet_tracker.py \
    --video surveillance.mp4 \
    --yolo-model models/YOLOv8/yolov8x.pt \
    --conf 0.6 \
    --iou 0.8 \
    --reclass-interval 5 \
    --save high_quality_output.mp4
```

### **2. Fast Processing (Low-End GPU)**
```powershell
python yolo_mobilenet_tracker.py \
    --video test.mp4 \
    --yolo-model models/YOLOv8/yolov8n.pt \
    --conf 0.4 \
    --reclass-interval 20 \
    --save fast_output.mp4
```

### **3. Real-Time Webcam**
```powershell
python yolo_mobilenet_tracker.py \
    --webcam \
    --reclass-interval 8 \
    --min-class-conf 0.7 \
    --smooth-window 5
```

### **4. Google Colab / Headless Server**
```powershell
python yolo_mobilenet_tracker.py \
    --video /content/test.mp4 \
    --no-display \
    --save output.mp4 \
    --device cuda
```

---

## 🎨 Output Visualization

### **Bounding Box Annotations**
```
┌─────────────────────────────────────────┐
│ ID:1 | using-computer (87.3%)          │ ← Track ID + Classification
├─────────────────────────────────────────┤
│                                         │
│         [Person tracked here]           │
│                                         │
└─────────────────────────────────────────┘
```

### **Global Info Panel**
```
┌─────────────────────────────────────────┐
│ Tracked People: 3                       │
│ Classification Interval: every 10 frames│
│ Processing FPS: 22.3                    │
│ Min Class Conf: 0.60  Smoothing Win: 7 │ ← Bottom right
└─────────────────────────────────────────┘
```

---

## 🧠 How It Works

### **Processing Pipeline**

```
Frame Input
    ↓
┌─────────────────────────────────────┐
│ 1. YOLOv8 Detection + Tracking      │
│    - Detect people (class 0)        │
│    - Assign/update track IDs        │
│    - Output: bboxes + track_ids     │
└──────────────┬──────────────────────┘
               ↓
┌─────────────────────────────────────┐
│ 2. For Each Track:                  │
│    - Check cache & interval         │
│    - Crop person bbox               │
└──────────────┬──────────────────────┘
               ↓
        Need to classify?
               ↓
         YES ↙   ↘ NO
┌──────────────┐  ┌─────────────────┐
│ 3. Classify  │  │ Use Cached      │
│    - Resize  │  │ Classification  │
│    - MobileNet│  └─────────────────┘
│    - Update  │
│      cache   │
└──────────────┘
               ↓
┌─────────────────────────────────────┐
│ 4. Annotate Frame                   │
│    - Draw bbox + track ID           │
│    - Show class + confidence        │
│    - Add global info                │
└──────────────┬──────────────────────┘
               ↓
         Display/Save
```

---

## 📈 Performance Optimization

### **Caching Efficiency**

Without caching:
```
- 30 FPS video, 3 people tracked
- Classifications per second: 3 × 30 = 90
- GPU load: Very High
```

With caching (interval=10):
```
- 30 FPS video, 3 people tracked
- Classifications per second: 3 × 3 = 9
- GPU load: 90% reduction ✅
```

### **Adaptive Interval Benefits**

| Hardware | FPS | Auto Interval | Result |
|----------|-----|---------------|--------|
| RTX 4090 | 45 | 5 frames | Max responsiveness |
| RTX 3060 | 25 | 10 frames | Balanced |
| GTX 1660 | 18 | 10 frames | Stable |
| CPU only | 8 | 30 frames | Smooth playback |

---

## 🔧 Configuration

### **Tracker Selection**

Available trackers:
```python
--tracker bytetrack.yaml    # Default: Best for crowded scenes
--tracker botsort.yaml      # Better re-identification
```

### **Re-classification Interval**

Choose based on your needs:
```python
--reclass-interval 5    # High accuracy, more GPU usage
--reclass-interval 10   # Balanced (default)
--reclass-interval 20   # Efficient, lower GPU usage
--reclass-interval 30   # Very efficient, good for tracking
```

**Note:** Adaptive system will adjust this automatically!

---

## 📊 Output Statistics

After processing, you'll see:

```
======================================================================
PROCESSING SUMMARY
======================================================================
Total Frames Processed: 500
Total Person Detections: 1,247
Total Classifications: 152
Classification Rate: 30.4% of frames

Average Classification Time: 12.3ms

🧠 Adaptive Classification Summary:
   - Final Processing FPS: 22.3
   - Final Classification Interval: every 10 frames
   - Unique Tracks Seen: 5
======================================================================
```

**Key Metrics:**
- **Total Detections:** Number of person bboxes across all frames
- **Total Classifications:** Actual MobileNetV2 inferences run
- **Classification Rate:** Percentage of frames where classification occurred
- **Unique Tracks:** Different people tracked throughout video

---

## 🎯 Use Cases

### **1. Security Surveillance**
Monitor multiple people and classify suspicious behaviors:
```powershell
python yolo_mobilenet_tracker.py \
    --video surveillance_feed.mp4 \
    --conf 0.6 \
    --reclass-interval 15 \
    --save monitored_output.mp4
```

### **2. Behavior Analysis**
Study patterns in public spaces:
```powershell
python yolo_mobilenet_tracker.py \
    --video public_space.mp4 \
    --reclass-interval 10 \
    --save behavior_analysis.mp4
```

### **3. Real-Time Monitoring**
Live webcam monitoring:
```powershell
python yolo_mobilenet_tracker.py \
    --webcam \
    --conf 0.5 \
    --reclass-interval 8
```

---

## 🐛 Troubleshooting

### **Issue: Low FPS**
**Solutions:**
1. Use smaller YOLO model: `--yolo-model models/YOLOv8/yolov8n.pt`
2. Increase interval: `--reclass-interval 20`
3. Lower confidence: `--conf 0.4`
4. Adaptive system will help automatically!

### **Issue: Missing Detections**
**Solutions:**
1. Lower confidence: `--conf 0.3`
2. Adjust IOU: `--iou 0.5`
3. Use larger YOLO model: `--yolo-model models/YOLOv8/yolov8x.pt`

### **Issue: Frequent Re-IDs**
**Solutions:**
1. Switch tracker: `--tracker botsort.yaml`
2. Increase re-classification interval: `--reclass-interval 20`

### **Issue: Choppy Video Output**
**Solutions:**
1. Check FPS in console output
2. Adaptive interval will adjust automatically
3. Manually increase: `--reclass-interval 30`

---

## 📝 Technical Details

### **Classification Cache Structure**
```python
classification_cache = {
    track_id_1: {
        'class_name': 'using-computer',
        'confidence': 0.87,
        'last_frame': 150
    },
    track_id_2: {
        'class_name': 'opening-cabinet',
        'confidence': 0.93,
        'last_frame': 148
    }
}
```

### **Re-classification Logic**
```python
def _should_reclassify(track_id, current_frame):
    if track_id not in cache:
        return True  # First time seeing this track
    
    frames_since_last = current_frame - cache[track_id]['last_frame']
    return frames_since_last >= reclass_interval
```

### **Adaptive Interval Logic**
```python
if current_fps > 25:
    interval = max(5, initial_interval // 2)
elif current_fps >= 15:
    interval = initial_interval
elif current_fps >= 10:
    interval = min(30, initial_interval * 2)
else:
    interval = min(40, initial_interval * 3)
```

---

## 🔮 Future Enhancements

- [ ] Multi-class object tracking (not just people)
- [ ] Confidence-based re-classification triggers
- [ ] Track history visualization (trajectory trails)
- [ ] Export tracking data to CSV/JSON
- [ ] Integration with face recognition (authorization levels)
- [ ] Real-time alerts for specific behaviors
- [ ] Web-based dashboard for monitoring

---

## 📖 Related Documentation

- `ADAPTIVE_CLASSIFICATION.md` - Detailed adaptive interval explanation
- `ADAPTIVE_QUICK_START.md` - Quick reference guide
- `inference_mobilenet.py` - Standalone MobileNetV2 inference
- `train_focused_augmentation.py` - Model training script

---

## ✨ Summary

**This integrated system provides:**
- ✅ Robust multi-person tracking (YOLOv8 + ByteTrack)
- ✅ Behavior classification (MobileNetV2)
- ✅ Smart caching (80-90% efficiency gain)
- ✅ Adaptive intervals (auto-optimization)
- ✅ Rich visualizations
- ✅ Production-ready performance

**Perfect for:**
- Security surveillance
- Behavior analysis
- Activity monitoring
- Research applications

---

**Version:** 1.0  
**Last Updated:** October 14, 2025  
**Status:** ✅ Production Ready
