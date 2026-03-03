# 📊 Unauthorized Action Counting & Behavior Analysis System

## 🎯 Overview

This system implements two advanced features requested by your professor:

1. **Per-Frame Unauthorized Action Counting** - Quantitative security metrics
2. **Repeated Behavior Detection with Smoothing** - Spam/abnormal behavior flagging

---

## 1️⃣ Per-Frame Unauthorized Action Counting

### What It Does

Counts and logs unauthorized interactions in **each video frame** for quantitative security analysis.

### How It Works

#### Detection Rules

The system classifies interactions as "unauthorized" based on:

```python
# Rule 1: Cell phone use is ALWAYS unauthorized
if object_type == "cell phone":
    → UNAUTHORIZED

# Rule 2: Unauthorized/Partial persons cannot use work equipment
if person_authorization in ["Unauthorized", "Partially Authorized"]:
    if object_type in ["laptop", "keyboard", "mouse"]:
        → UNAUTHORIZED

# Rule 3: Authorized persons can use laptops freely
if person_authorization == "Authorized":
    if object_type in ["laptop", "keyboard", "mouse"]:
        → ALLOWED
```

#### Per-Frame Logging

For each frame processed:

```json
{
    "frame_id": 456,
    "timestamp": "2026-03-02 15:30:45",
    "unauthorized_count": 2,
    "details": [
        {
            "person_id": 7,
            "identity": "John Doe",
            "authorization": "Unauthorized",
            "action": "STATUS: INTERACTING WITH CELL PHONE",
            "object": "cell phone"
        },
        {
            "person_id": 9,
            "identity": "Jane Smith",
            "authorization": "Partially Authorized",
            "action": "STATUS: INTERACTING WITH LAPTOP",
            "object": "laptop"
        }
    ]
}
```

### Output Files

#### 1. JSON Analysis File

**Location:** `logs/MM-DD-YYYY/unauthorized_actions_SOURCE_TIMESTAMP.json`

**Structure:**

```json
{
  "summary": {
    "total_frames_with_unauthorized": 145,
    "total_unauthorized_actions": 298,
    "avg_per_frame": 2.06,
    "max_in_single_frame": 5,
    "by_object_type": {
      "cell phone": 180,
      "laptop": 95,
      "keyboard": 23
    },
    "by_person": {
      "7": 120,
      "9": 95,
      "12": 83
    }
  },
  "logs": [
    {
      "frame_id": 123,
      "timestamp": "2026-03-02 15:30:45",
      "unauthorized_count": 2,
      "details": [...]
    },
    ...
  ]
}
```

#### 2. Session Log File

**Location:** `logs/MM-DD-YYYY/session_SOURCE_TIMESTAMP.txt`

**Includes summary at end:**

```
============================================================
UNAUTHORIZED ACTIONS SUMMARY
============================================================
Total frames with unauthorized actions: 145
Total unauthorized actions: 298
Average per frame: 2.06
Max in single frame: 5

Breakdown by object type:
  cell phone: 180 violations
  laptop: 95 violations
  keyboard: 23 violations

Breakdown by person ID:
  Person 7: 120 violations
  Person 9: 95 violations
  Person 12: 83 violations
============================================================
```

### Real-Time Logging

During monitoring, unauthorized actions are logged immediately:

```
🚫 UNAUTHORIZED: 2 violation(s) detected - John Doe (ID 7) - cell phone, Jane Smith (ID 9) - laptop
```

### Benefits

✅ **Quantitative Metrics** - Count violations, not just binary yes/no
✅ **Severity Measurement** - Know how many violations in one moment
✅ **Post-Analysis** - Generate reports, charts, heatmaps
✅ **Alert Thresholds** - Trigger alerts if >N violations in single frame
✅ **Trend Analysis** - Track violations over time (hourly, daily, weekly)
✅ **Evidence Correlation** - Match counts to specific timestamps/screenshots

---

## 2️⃣ Repeated Behavior Detection with Smoothing

### What It Does

Detects when a person **repeatedly** performs the same interaction, flagging it as **spam** or **abnormal behavior**.

### How It Works

#### Sliding Window Tracking

For each person (tracked ID), maintains a history of last 50 frames:

```python
Person ID 5:
  Frame 100: "INTERACTING WITH CELL PHONE"
  Frame 101: "INTERACTING WITH CELL PHONE"
  Frame 102: "INTERACTING WITH CELL PHONE"
  ...
  Frame 149: "INTERACTING WITH CELL PHONE"

  Repetition Rate: 45/50 frames = 90%
```

#### Smoothing Mechanism

Uses **sliding window + threshold** to avoid false positives:

```python
# Parameters (configurable in __init__)
behavior_window_size = 50          # Track last 50 frames
behavior_spam_threshold = 0.80     # 80% repetition = abnormal
behavior_alert_cooldown = 150      # Wait 150 frames before re-alerting (5 sec at 30fps)
```

#### Flagging Logic

```python
if interaction (not "NO INTERACTION"):
    if repetition_rate >= 80%:
        if cooldown_passed:
            → FLAG AS ABNORMAL/SPAM
```

#### Alert Suppression

- **First detection:** Alert immediately
- **Subsequent detections:** Wait 150 frames (5 seconds) before alerting again
- **Counter:** Tracks how many times each person has been flagged

### Output

#### Detection Object Enhancement

Each detection gets additional fields:

```python
{
  "track_id": 5,
  "identity": "John Doe",
  "behavior_status": "STATUS: INTERACTING WITH CELL PHONE [REPEATED ABNORMALLY x3]",
  "repetition_detected": True,
  "repetition_rate": 0.90,  # 90%
  "behavior_flagged": True
}
```

#### Real-Time Logging

```
⚠️ ABNORMAL BEHAVIOR: John Doe (ID 5) - INTERACTING WITH CELL PHONE repeated in 90% of frames (Alert #3)
```

#### Session Summary

At end of session, logs repeated behavior stats:

```
============================================================
REPEATED BEHAVIOR ALERTS (SPAM/ABNORMAL)
============================================================
Person 5: STATUS: INTERACTING WITH CELL PHONE
  Alert count: 3 times
  Last flagged at frame: 1245

Person 7: STATUS: CARRYING BACKPACK
  Alert count: 1 times
  Last flagged at frame: 890
============================================================
```

### Benefits

✅ **Spam Prevention** - Avoid flooding with repetitive alerts
✅ **Abnormal Detection** - Catch suspicious looping behavior
✅ **Smoothing** - Noise-resistant (requires sustained repetition)
✅ **Glitch Detection** - Identify false-positive loops
✅ **Intelligence** - System learns normal vs abnormal patterns

---

## 🚀 Usage

### In Your Code

The system works **automatically** when behavior detection is enabled. No additional configuration needed!

```python
# Initialize pipeline (already in streamlit_app.py)
pipeline = CombinedYOLOFaceNetBehavior(
    yolo_model_path="models/YOLOv8/yolov8n.pt",
    facenet_main_path="face_recognition/Facenet/facenet_main.py",
    enable_behavior=True,  # ← Must be enabled
    ...
)

# Process frames (automatic counting & detection)
annotated, detections = pipeline.process_frame(frame)

# Detections automatically include:
# - unauthorized_count per frame (logged internally)
# - repetition_detected, repetition_rate, behavior_flagged (in detection dict)
```

### Access Analysis Data

```python
# Get all unauthorized logs
logs = pipeline.get_unauthorized_logs()
# Returns: List[Dict] with all frame-by-frame logs

# Get summary statistics
summary = pipeline.get_unauthorized_summary()
# Returns: {total_frames, total_actions, avg_per_frame, max_in_single_frame, by_object, by_person}

# Get repeated behavior summary
spam_summary = pipeline.get_behavior_spam_summary()
# Returns: {track_id: {alert_count, last_behavior, last_alert_frame}}

# Save to JSON for analysis
pipeline.save_unauthorized_logs_to_file("analysis.json")
```

### Customize Rules

Edit `combined_yolo_facenet_behavior.py` in the `count_unauthorized_actions()` method:

```python
# Example: Make laptop use always unauthorized
if object_type == "laptop":
    is_unauthorized_action = True

# Example: Allow partial authorized to use keyboards
if object_type == "keyboard" and auth_level == "Partially Authorized":
    is_unauthorized_action = False

# Example: Unauthorized only during specific hours
import datetime
current_hour = datetime.datetime.now().hour
if current_hour >= 22 or current_hour < 6:  # After 10PM or before 6AM
    is_unauthorized_action = True
```

### Adjust Sensitivity

Edit `__init__` parameters in `combined_yolo_facenet_behavior.py`:

```python
# More sensitive (flag at 60% repetition)
self.behavior_spam_threshold = 0.60

# Less sensitive (flag at 90% repetition)
self.behavior_spam_threshold = 0.90

# Longer history (track last 100 frames)
self.behavior_window_size = 100

# Faster re-alerting (alert every 3 seconds)
self.behavior_alert_cooldown = 90  # at 30fps
```

---

## 📊 Analysis Examples

### Example 1: Peak Violation Times

```python
import json
with open("logs/03-02-2026/unauthorized_actions_rtsp_20260302_150000.json") as f:
    data = json.load(f)

# Group by hour
violations_by_hour = {}
for log in data["logs"]:
    hour = log["timestamp"].split()[1].split(":")[0]
    violations_by_hour[hour] = violations_by_hour.get(hour, 0) + log["unauthorized_count"]

print("Peak violation hours:", violations_by_hour)
# Output: {'15': 45, '16': 78, '17': 23}  # 4PM was worst
```

### Example 2: Top Violators

```python
summary = data["summary"]
top_violators = sorted(summary["by_person"].items(), key=lambda x: x[1], reverse=True)
print("Top 3 violators:")
for person_id, count in top_violators[:3]:
    print(f"  Person {person_id}: {count} violations")
```

### Example 3: Violation Heatmap

```python
import matplotlib.pyplot as plt
import numpy as np

# Extract frame IDs and counts
frames = [log["frame_id"] for log in data["logs"]]
counts = [log["unauthorized_count"] for log in data["logs"]]

plt.figure(figsize=(12, 4))
plt.bar(frames, counts, color='red', alpha=0.6)
plt.xlabel("Frame")
plt.ylabel("Unauthorized Actions")
plt.title("Violation Intensity Over Time")
plt.show()
```

---

## ✅ Testing

```bash
# Start the system
python -m streamlit run .\scripts\streamlit_app.py

# Enable behavior detection
✅ Enable HOI Detection

# Monitor and interact
# - Use cell phone → Should log as unauthorized
# - Repeat action 40+ times → Should flag as abnormal

# After stopping, check:
# 1. Session log: logs/MM-DD-YYYY/session_*.txt
# 2. JSON analysis: logs/MM-DD-YYYY/unauthorized_actions_*.json
```

---

## 🎓 Academic Benefits

### For Your Professor

1. **Quantitative Metrics** - Real numbers, not just "detected/not detected"
2. **Statistical Analysis** - Can generate charts, graphs, trends
3. **Severity Assessment** - Know how bad a situation is (1 vs 5 violations)
4. **Alert Thresholds** - Can set rules like "alert if >3 violations in one frame"
5. **Spam Prevention** - Intelligent filtering reduces false alarms
6. **Abnormal Detection** - Catches suspicious repetitive patterns

### Research Applications

- Security incident reports with quantitative data
- Time-series analysis of violations
- Behavioral pattern recognition
- Automated alert escalation systems
- Employee productivity monitoring (ethically!)

---

## 🎉 Summary

✅ **Feature 1 Implemented:** Per-frame unauthorized action counting with detailed JSON logs
✅ **Feature 2 Implemented:** Repeated behavior detection with smoothing and spam prevention
✅ **Real-Time Logging:** See violations as they happen
✅ **Post-Analysis:** JSON files for statistical analysis
✅ **Customizable:** Adjust rules and thresholds easily
✅ **Production-Ready:** Integrated into your existing CCTV system

Your system now provides **quantitative security metrics** instead of just binary detection! 📊✨
