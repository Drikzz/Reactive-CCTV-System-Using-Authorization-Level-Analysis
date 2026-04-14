# Repeated Behavior Detection - Temporal Smoothing Explained

## Overview

The system uses **temporal smoothing with a sliding window** to detect when someone is repeating the same action abnormally (e.g., continuously using a laptop for an extended period). This prevents spam alerts while identifying genuine repeated patterns.

---

## The Algorithm: Sliding Window with Temporal Smoothing

### **What is Temporal Smoothing?**

Instead of analyzing each frame independently (which would cause flickering alerts), the system looks at **behavior over time** by:

1. Keeping a **history window** of the last N frames
2. Calculating the **percentage** of frames showing the same behavior
3. Only flagging if the percentage exceeds a threshold **AND** enough time has passed since last alert

This "smooths out" noise and focuses on sustained patterns rather than momentary actions.

---

## Step-by-Step Process

### **Step 1: Initialize Tracking (Lines 358-367)**

When a person is first detected, create a tracking record:

```python
self.behavior_history[track_id] = {
    "behavior_window": deque(maxlen=50),  # Sliding window: last 50 frames
    "last_behavior": None,                # What they're currently doing
    "last_alert_frame": -9999,            # When we last alerted (start far in past)
    "alert_count": 0                      # How many times we've flagged them
}
```

**Key Component: `deque(maxlen=50)`**

- A **deque** (double-ended queue) is a data structure that automatically removes old items when full
- `maxlen=50` means it keeps only the **last 50 frames** of behavior
- Old data automatically "slides out" as new data comes in → **Sliding Window**

---

### **Step 2: Add Current Frame to Window (Lines 370-371)**

Every frame, record what the person is doing:

```python
track_hist["behavior_window"].append(behavior_status)
track_hist["last_behavior"] = behavior_status
```

**Example Timeline:**

```
Frame 1:  [INTERACTING WITH LAPTOP]
Frame 2:  [INTERACTING WITH LAPTOP, INTERACTING WITH LAPTOP]
Frame 3:  [INTERACTING WITH LAPTOP, INTERACTING WITH LAPTOP, INTERACTING WITH LAPTOP]
...
Frame 50: [50 frames of history, oldest frame dropped when frame 51 comes]
Frame 51: [Frame 2-51 kept, Frame 1 dropped] ← Sliding!
```

This is **temporal smoothing** - we're smoothing the behavior analysis over time instead of instant-by-instant.

---

### **Step 3: Wait for Enough Data (Lines 374-380)**

Don't analyze until we have at least 20 frames:

```python
if len(track_hist["behavior_window"]) < 20:
    return {
        "is_repeated": False,
        "repetition_rate": 0.0,
        "flagged": False
    }
```

**Why?**

- Prevents false positives from too little data
- 20 frames ≈ 0.67 seconds at 30 FPS
- Ensures we have a meaningful sample size for smoothing

---

### **Step 4: Calculate Repetition Rate (Lines 383-384)**

Count how often the current behavior appears in the window:

```python
behavior_count = sum(1 for b in track_hist["behavior_window"] if b == behavior_status)
repetition_rate = behavior_count / len(track_hist["behavior_window"])
```

**Example Calculation:**

```
Window size: 50 frames
Current behavior: "STATUS: INTERACTING WITH LAPTOP"
Count in window: 44 frames show this behavior
Repetition rate: 44/50 = 0.88 = 88%
```

This is the **smoothing formula** - it averages the behavior over the entire window.

---

### **Step 5: Apply Three-Part Test (Lines 387-395)**

#### **Test 1: Is it an actual interaction?**

```python
is_interaction = behavior_status != "STATUS: NO INTERACTION"
```

- Don't flag people for just standing around
- Only flag actual interactions (using objects)

#### **Test 2: Does it exceed the threshold?**

```python
exceeds_threshold = repetition_rate >= self.behavior_spam_threshold
```

- Default threshold: **0.8 (80%)**
- If 80% or more of last 50 frames show same behavior → Repeated pattern

#### **Test 3: Has cooldown passed?**

```python
cooldown_passed = (self._frame_idx - track_hist["last_alert_frame"]) >= self.behavior_alert_cooldown
```

- Default cooldown: **150 frames (5 seconds at 30 FPS)**
- Prevents spamming the same alert every frame
- Only re-alert after enough time has passed

**Combined Logic:**

```python
is_repeated = is_interaction and exceeds_threshold
should_flag = is_repeated and cooldown_passed
```

---

### **Step 6: Flag and Update (Lines 405-407)**

If all tests pass, flag the behavior and record it:

```python
if should_flag:
    track_hist["last_alert_frame"] = self._frame_idx  # Remember when we alerted
    track_hist["alert_count"] += 1                     # Increment alert counter
```

This updates the cooldown timer so we won't alert again for another 150 frames.

---

## Visual Example: Sliding Window Over Time

```
Time →
Frames: [1][2][3][4]...[48][49][50][51][52]...[100]

Window at Frame 50:
┌─────────────────────────────────────────────────┐
│ Frame 1: LAPTOP                                 │
│ Frame 2: LAPTOP                                 │
│ Frame 3: LAPTOP                                 │
│ ...                                             │
│ Frame 50: LAPTOP                                │
└─────────────────────────────────────────────────┘
Analysis: 50/50 = 100% repetition → FLAGGED (Alert #1)
Cooldown starts: Won't alert again until frame 200

Window at Frame 51 (slides forward):
┌─────────────────────────────────────────────────┐
│ Frame 2: LAPTOP   ← Frame 1 dropped             │
│ Frame 3: LAPTOP                                 │
│ Frame 4: LAPTOP                                 │
│ ...                                             │
│ Frame 51: LAPTOP  ← New frame added             │
└─────────────────────────────────────────────────┘
Analysis: 50/50 = 100% repetition
Cooldown active: 51 < 200, so NO alert

Window at Frame 200:
┌─────────────────────────────────────────────────┐
│ Frame 151: LAPTOP                               │
│ Frame 152: LAPTOP                               │
│ ...                                             │
│ Frame 200: LAPTOP                               │
└─────────────────────────────────────────────────┘
Analysis: 50/50 = 100% repetition
Cooldown passed: 200 - 50 = 150 ≥ 150 → FLAGGED (Alert #2)
```

---

## Smoothing Benefits

### **1. Noise Reduction**

Without smoothing (per-frame analysis):

```
Frame 100: Using laptop → ALERT!
Frame 101: Using laptop → ALERT!
Frame 102: Using laptop → ALERT!
Frame 103: Using laptop → ALERT!
... (100 alerts in 3 seconds!)
```

With smoothing (sliding window):

```
Frames 1-50: Building history...
Frame 50: 80% laptop use → ALERT! (Alert #1)
Frames 51-199: Cooldown active, no spam
Frame 200: Still 80% laptop use → ALERT! (Alert #2)
```

### **2. Robust to Brief Interruptions**

Person using laptop continuously but looks away for 2 frames:

```
Window: [L L L L L L L L L L L L L L ... L L N N L L L L L]
        ← 46 laptop frames + 2 "no interaction" + 2 laptop frames
Repetition rate: 48/50 = 96% → Still flagged!
```

The smoothing "absorbs" brief interruptions - focuses on overall pattern, not momentary changes.

### **3. Adapts to Behavior Changes**

Person switches from laptop to phone:

```
Frames 1-50:   [LAPTOP LAPTOP LAPTOP...] → 100% laptop
Frames 51-100: [LAPTOP PHONE PHONE PHONE...] → Window now has mix
               Old laptop frames slide out, new phone frames slide in
Frames 100+:   [PHONE PHONE PHONE...] → Now tracking phone pattern
```

The window "forgets" old behavior as it slides out - adapts to new patterns naturally.

---

## Configurable Parameters

These are set in the constructor (Lines 147-150):

```python
self.behavior_window_size = 50        # How many frames to analyze
self.behavior_spam_threshold = 0.8    # 80% repetition required
self.behavior_alert_cooldown = 150    # Frames between alerts
```

### **Tuning Guide:**

| Want to...               | Adjust                 | Example                      |
| ------------------------ | ---------------------- | ---------------------------- |
| Catch shorter patterns   | Decrease `window_size` | `30` = 1 second at 30 FPS    |
| Require more consistency | Increase `threshold`   | `0.9` = 90% required         |
| Alert more frequently    | Decrease `cooldown`    | `60` = 2 seconds at 30 FPS   |
| Catch longer patterns    | Increase `window_size` | `90` = 3 seconds at 30 FPS   |
| Be more lenient          | Decrease `threshold`   | `0.7` = 70% required         |
| Reduce alert spam        | Increase `cooldown`    | `300` = 10 seconds at 30 FPS |

---

## Real-World Example

**Scenario:** Art (Partially Authorized) using laptop for 10 seconds

```
Assumptions:
- 30 FPS video
- 10 seconds = 300 frames
- Art uses laptop continuously

Timeline:
─────────────────────────────────────────────────────────
Frame 1-19:   Building history (< 20 frames)
              → No analysis yet

Frame 20-49:  Building history (< 50 frames)
              Window: [L L L L L ... L L L L] (20-49 frames)
              → Not enough data for full window yet

Frame 50:     Window full: [L L L L L ... L L L L] (50 frames)
              Repetition: 50/50 = 100%
              Exceeds 80% threshold? YES
              Cooldown passed? YES (first time)
              → 🔁 FLAGGED (Alert #1)
              → "Art (ID 1) - STATUS: INTERACTING WITH LAPTOP repeated in 100% of frames (Alert #1)"

Frame 51-199: Window: [L L L L L ... L L L L] (still 100%)
              Cooldown active: (frame - 50) < 150
              → No new alert (prevents spam)

Frame 200:    Window: [L L L L L ... L L L L] (frames 151-200)
              Repetition: 50/50 = 100%
              Cooldown passed? 200 - 50 = 150 ≥ 150? YES
              → 🔁 FLAGGED (Alert #2)
              → "Art (ID 1) - STATUS: INTERACTING WITH LAPTOP repeated in 100% of frames (Alert #2)"

Frame 201-299: Cooldown active again
              → No alert

Frame 300:    Video ends
```

**Result:** Only 2 alerts in 10 seconds instead of 300 alerts!

---

## How It Prevents Spam

### **Problem Without Smoothing:**

```python
# Naive approach - check each frame independently
if person_using_laptop:
    alert("Person using laptop!")  # Triggers every single frame!
```

At 30 FPS, this would generate **30 alerts per second** for continuous laptop use!

### **Solution With Smoothing:**

1. **Sliding Window** - Only analyze last 50 frames, not entire history
    - Keeps memory usage constant
    - Focuses on recent behavior
    - Old data naturally "ages out"

2. **Threshold-Based** - Require 80% repetition, not 100%
    - Tolerates brief interruptions
    - Robust to noisy detections
    - Focuses on sustained patterns

3. **Cooldown Period** - Wait 150 frames between alerts
    - Prevents alert spam
    - Gives meaningful time gaps
    - Still catches persistent behavior

**Mathematical Proof:**

```
Alert frequency = (Total frames) / (Window size + Cooldown)
                = 300 / (50 + 150)
                = 300 / 200
                = 1.5 alerts per 10 seconds
```

Much better than 300 alerts per second!

---

## Code Flow Summary

```
1. Person detected
   ↓
2. Create sliding window (deque maxlen=50)
   ↓
3. Every frame: Add current behavior to window
   ↓
4. Old behavior automatically slides out
   ↓
5. Calculate repetition rate = (count of behavior) / (window size)
   ↓
6. Check three conditions:
   - Is it an interaction? (not just standing)
   - Exceeds 80% threshold?
   - Cooldown period passed?
   ↓
7. If all YES → Flag and alert
   If any NO → Don't alert
   ↓
8. Update cooldown timer
   ↓
9. Repeat for next frame
```

---

## JSON Output Example

When repeated behavior is detected, it appears in logs:

```json
{
    "behavior_spam_summary": {
        "1": {
            "alert_count": 2,
            "last_behavior": "STATUS: INTERACTING WITH LAPTOP",
            "last_alert_frame": 200
        }
    }
}
```

And in session log:

```
REPEATED BEHAVIOR ALERTS (SPAM/ABNORMAL)
Person 1: STATUS: INTERACTING WITH LAPTOP
  Alert count: 2 times
  Last flagged at frame: 200
```

---

## Conclusion

The system uses **temporal smoothing** through:

- ✅ **Sliding window** (deque) - Analyzes last N frames
- ✅ **Percentage-based threshold** - Requires sustained pattern (80%)
- ✅ **Cooldown mechanism** - Prevents alert spam (150 frames)

This approach balances:

- **Sensitivity** - Catches genuine repeated patterns
- **Robustness** - Tolerates noise and brief interruptions
- **User Experience** - Avoids flooding logs with redundant alerts

The temporal smoothing happens automatically through the sliding window - as new frames come in, old frames slide out, creating a "moving average" of behavior over time.

---

**End of Explanation**
