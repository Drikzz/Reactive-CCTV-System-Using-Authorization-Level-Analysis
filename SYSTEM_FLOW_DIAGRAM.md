# 📊 System Flow Diagram

## 🎥 Video Frame Processing Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    CAMERA INPUT                              │
│              (Webcam / RTSP / Video File)                    │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              YOLO + ByteTrack Detection                      │
│  • Detect persons (track_id assigned)                       │
│  • Detect objects (laptop, phone, backpack, etc.)           │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                FaceNet Recognition                           │
│  • Recognize face → Identity + Confidence                   │
│  • Authorization level (Authorized/Partial/Unauthorized)    │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              Behavior Detection (HOI)                        │
│  • Calculate person-object overlap (IoU)                    │
│  • Detect movement (CARRYING vs INTERACTING WITH)           │
│  • Hysteresis filtering (smooth transitions)                │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│    ⭐ NEW FEATURE 1: Count Unauthorized Actions             │
│                                                              │
│  For each detection in this frame:                          │
│    ├─ Is it an interaction? (not "NO INTERACTION")          │
│    ├─ Is person unauthorized/partial?                       │
│    ├─ Is object forbidden? (e.g., cell phone)               │
│    └─ If YES → Count as unauthorized                        │
│                                                              │
│  Output:                                                     │
│    {                                                         │
│      "frame_id": 456,                                        │
│      "unauthorized_count": 2,                                │
│      "details": [...]                                        │
│    }                                                         │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│    ⭐ NEW FEATURE 2: Detect Repeated Behavior               │
│                                                              │
│  For each person (track_id):                                │
│    ├─ Maintain sliding window (last 50 frames)              │
│    ├─ Store current behavior in window                      │
│    ├─ Calculate repetition rate (% same behavior)           │
│    ├─ Apply threshold (80% = abnormal)                      │
│    ├─ Check cooldown (5 seconds between alerts)             │
│    └─ If abnormal → FLAG and add "[REPEATED...]" tag        │
│                                                              │
│  Output:                                                     │
│    {                                                         │
│      "repetition_detected": True,                            │
│      "repetition_rate": 0.90,                                │
│      "behavior_flagged": True                                │
│    }                                                         │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                  Real-Time Logging                           │
│                                                              │
│  • Display in Streamlit UI                                  │
│  • Log to session text file                                 │
│  • Store in internal buffers                                │
│                                                              │
│  Examples:                                                   │
│    🚫 UNAUTHORIZED: 2 violation(s) detected                 │
│    ⚠️ ABNORMAL BEHAVIOR: John - repeated 90%                │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              Annotated Frame Output                          │
│                                                              │
│  • Draw bounding boxes (color by authorization)             │
│  • Draw behavior labels (with [REPEATED...] if flagged)     │
│  • Display in video feed                                    │
│  • Save to recording (if enabled)                           │
│  • Save as evidence screenshot (for interactions)           │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
                   [NEXT FRAME]
```

---

## 📝 Session End Analysis Flow

```
┌─────────────────────────────────────────────────────────────┐
│                  SESSION ENDS                                │
│              (User clicks Stop)                              │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│         Calculate Summary Statistics                         │
│                                                              │
│  From unauthorized_logs:                                     │
│    • Total frames with violations                           │
│    • Total violations across all frames                     │
│    • Average per frame                                      │
│    • Max in single frame                                    │
│    • Breakdown by object type                               │
│    • Breakdown by person ID                                 │
│                                                              │
│  From behavior_history:                                      │
│    • Which persons had abnormal behavior                    │
│    • How many times flagged                                 │
│    • Last behavior detected                                 │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              Write to Session Log                            │
│                                                              │
│  logs/MM-DD-YYYY/session_SOURCE_TIMESTAMP.txt               │
│                                                              │
│  Includes:                                                   │
│    • Session header (start time, source, etc.)              │
│    • Real-time event logs (person entries, behaviors)       │
│    • Unauthorized actions summary                           │
│    • Repeated behavior alerts summary                       │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│             Export JSON Analysis File                        │
│                                                              │
│  logs/MM-DD-YYYY/unauthorized_actions_SOURCE_TIMESTAMP.json │
│                                                              │
│  Structure:                                                  │
│    {                                                         │
│      "summary": {statistics...},                             │
│      "logs": [                                               │
│        {frame_id, timestamp, count, details},                │
│        ...                                                   │
│      ]                                                       │
│    }                                                         │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              Display in UI                                   │
│                                                              │
│  📊 Analysis: 298 unauthorized actions detected             │
│  ⚠️ 2 person(s) with repeated abnormal behavior             │
└──────────────────────┴──────────────────────────────────────┘
```

---

## 🔄 Repeated Behavior Detection Detail

```
Person ID: 5 (John Doe)
═══════════════════════════════════════════════════════════

Frame Window (Last 50 frames):
┌──────────────────────────────────────────────────────────┐
│ Frame 100: INTERACTING WITH CELL PHONE                   │
│ Frame 101: INTERACTING WITH CELL PHONE                   │
│ Frame 102: INTERACTING WITH CELL PHONE                   │
│ Frame 103: INTERACTING WITH CELL PHONE                   │
│ ...                                                       │
│ Frame 145: INTERACTING WITH CELL PHONE                   │
│ Frame 146: NO INTERACTION                                │
│ Frame 147: INTERACTING WITH CELL PHONE                   │
│ Frame 148: INTERACTING WITH CELL PHONE                   │
│ Frame 149: INTERACTING WITH CELL PHONE                   │
└──────────────────────────────────────────────────────────┘

Analysis:
┌──────────────────────────────────────────────────────────┐
│ "INTERACTING WITH CELL PHONE" count: 45 out of 50       │
│ Repetition Rate: 45/50 = 90%                            │
│ Threshold: 80%                                           │
│ Exceeds Threshold: YES (90% > 80%)                      │
│ Cooldown Check: Last alert at frame 50 (99 frames ago)  │
│ Cooldown Required: 150 frames                           │
│ Cooldown Passed: NO (need 51 more frames)               │
│                                                          │
│ Result: DETECTED but NOT FLAGGED (cooldown not passed)  │
└──────────────────────────────────────────────────────────┘

After 51 more frames...
┌──────────────────────────────────────────────────────────┐
│ Cooldown Check: Last alert at frame 50 (150 frames ago) │
│ Cooldown Passed: YES                                    │
│                                                          │
│ Result: FLAGGED! ⚠️                                      │
│ Alert: "ABNORMAL BEHAVIOR: John Doe - repeated 90%"     │
│ Behavior: "INTERACTING WITH CELL PHONE [REPEATED x2]"   │
└──────────────────────────────────────────────────────────┘
```

---

## 📊 Per-Frame Unauthorized Counting Example

```
Frame 456 @ 2026-03-02 15:30:45
═══════════════════════════════════════════════════════════

Detections in this frame:
┌────────────────────────────────────────────────────────────┐
│ Person 5 (John Doe) - Authorized                           │
│   Behavior: INTERACTING WITH LAPTOP                        │
│   └─> Check: Authorized person + laptop = ALLOWED ✅       │
│                                                             │
│ Person 7 (Unknown) - Unauthorized                          │
│   Behavior: INTERACTING WITH CELL PHONE                    │
│   └─> Check: Cell phone = ALWAYS unauthorized ❌           │
│                                                             │
│ Person 9 (Jane Smith) - Partially Authorized               │
│   Behavior: INTERACTING WITH LAPTOP                        │
│   └─> Check: Partial person + work equipment = ❌          │
│                                                             │
│ Person 12 (Bob Lee) - Authorized                           │
│   Behavior: NO INTERACTION                                 │
│   └─> Check: No interaction = N/A                          │
└────────────────────────────────────────────────────────────┘

Unauthorized Count: 2
┌────────────────────────────────────────────────────────────┐
│ Detail 1:                                                   │
│   person_id: 7                                              │
│   identity: Unknown                                         │
│   authorization: Unauthorized                               │
│   action: INTERACTING WITH CELL PHONE                      │
│   object: cell phone                                        │
│                                                             │
│ Detail 2:                                                   │
│   person_id: 9                                              │
│   identity: Jane Smith                                      │
│   authorization: Partially Authorized                       │
│   action: INTERACTING WITH LAPTOP                          │
│   object: laptop                                            │
└────────────────────────────────────────────────────────────┘

Log Entry Created:
{
  "frame_id": 456,
  "timestamp": "2026-03-02 15:30:45",
  "unauthorized_count": 2,
  "details": [...]
}

Real-Time Alert:
🚫 UNAUTHORIZED: 2 violation(s) detected - Unknown (ID 7) - cell phone, Jane Smith (ID 9) - laptop
```

---

## 🎯 Key Algorithms

### 1. Unauthorized Action Detection

```python
is_unauthorized = False

if object == "cell phone":
    is_unauthorized = True  # Always

elif object in ["laptop", "keyboard", "mouse"]:
    if authorization != "Authorized":
        is_unauthorized = True  # Only unauthorized/partial
```

### 2. Repeated Behavior Smoothing

```python
# Sliding window
window = deque(maxlen=50)  # Last 50 frames
window.append(current_behavior)

# Calculate repetition
count = window.count(current_behavior)
rate = count / len(window)

# Apply threshold
if rate >= 0.80:  # 80%
    if (current_frame - last_alert_frame) >= 150:  # 5 sec cooldown
        FLAG_AS_ABNORMAL()
```

---

## 📈 Data Flow Summary

```
Camera → YOLO → FaceNet → Behavior → ┌─ Count Unauthorized
                                      │
                                      └─ Detect Repeated → Logs → Analysis
                                                                    │
                                                                    ├─ JSON File
                                                                    │
                                                                    └─ Session Log
```

---

## 🎉 Result

Your system now:

- ✅ Counts violations **quantitatively** (not just yes/no)
- ✅ Detects spam/abnormal behavior **intelligently** (smoothing)
- ✅ Provides **real-time alerts** and **post-analysis data**
- ✅ Exports to **JSON for statistical analysis**
- ✅ Includes **comprehensive summaries** in logs

Perfect for academic research and production security systems! 📊🔒✨
