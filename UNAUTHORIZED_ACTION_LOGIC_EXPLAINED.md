# Unauthorized Action Detection - Complete Logic Explanation

## Overview

The system counts unauthorized actions per frame by analyzing person-object interactions and applying authorization rules based on who is using what object.

---

## What Changed

### **BEFORE** (Lines 305-319)

```python
is_unauthorized_action = False

if object_type == "cell phone":
    # Cell phone use is always unauthorized
    is_unauthorized_action = True
    print(f"[DEBUG] UNAUTHORIZED: {identity} using cell phone")
elif is_unauthorized_person and object_type in ["laptop", "keyboard", "mouse"]:
    # Unauthorized/partial persons cannot use work equipment
    is_unauthorized_action = True
    print(f"[DEBUG] UNAUTHORIZED: {identity} ({auth_level}) using {object_type}")
```

**Problem:** Only laptop, keyboard, and mouse counted as unauthorized for partial/unauthorized users. **Backpacks and handbags were ignored!**

---

### **AFTER** (Lines 305-319)

```python
is_unauthorized_action = False

if object_type == "cell phone":
    # Cell phone use is always unauthorized for everyone
    is_unauthorized_action = True
    print(f"[DEBUG] UNAUTHORIZED: {identity} using cell phone")
elif is_unauthorized_person and object_type is not None:
    # Unauthorized/Partially Authorized persons cannot use ANY detected objects
    is_unauthorized_action = True
    print(f"[DEBUG] UNAUTHORIZED: {identity} ({auth_level}) using {object_type}")
```

**Fix:** Changed from checking specific list `["laptop", "keyboard", "mouse"]` to checking `object_type is not None`, which means **ALL detected objects** now count as unauthorized for partial/unauthorized users.

---

## Complete Logic Flow

### **Step 1: Frame Processing**

```
Frame comes in → YOLO detects persons → ByteTrack assigns IDs → FaceNet identifies faces
```

### **Step 2: Detection Loop** (Line 263)

For each person detected in the frame:

```python
for det in detections:
    identity = det.get("identity", "Unknown")          # e.g., "Art", "Aldrikz", "Unknown"
    auth_level = det.get("authorization", "Unauthorized")  # "Authorized", "Partially Authorized", "Unauthorized"
    behavior = det.get("behavior_status", "STATUS: NO INTERACTION")  # e.g., "STATUS: INTERACTING WITH LAPTOP"
    track_id = det.get("track_id", -1)                # Person's tracking ID
```

### **Step 3: Check if Interaction Exists** (Line 273)

```python
is_interaction = behavior != "STATUS: NO INTERACTION"
```

- If person is just standing/walking → No interaction → Skip
- If person is "INTERACTING WITH" or "CARRYING" something → Continue

### **Step 4: Determine Authorization Status** (Line 274)

```python
is_unauthorized_person = auth_level in ["Unauthorized", "Partially Authorized"]
```

- **Authorized** persons: Can use most objects freely
- **Partially Authorized** persons: Limited access, restricted from objects
- **Unauthorized** persons: No access, all object use is flagged

### **Step 5: Extract Object Type** (Lines 279-293)

```python
behavior_upper = behavior.upper()  # Convert to uppercase for matching

if "LAPTOP" in behavior_upper:
    object_type = "laptop"
elif "CELL PHONE" in behavior_upper or "PHONE" in behavior_upper:
    object_type = "cell phone"
elif "KEYBOARD" in behavior_upper:
    object_type = "keyboard"
elif "MOUSE" in behavior_upper:
    object_type = "mouse"
elif "BACKPACK" in behavior_upper:
    object_type = "backpack"
elif "HANDBAG" in behavior_upper:
    object_type = "handbag"
```

**Example:**

- Input: `"STATUS: INTERACTING WITH LAPTOP"` → Output: `object_type = "laptop"`
- Input: `"STATUS: CARRYING BACKPACK"` → Output: `object_type = "backpack"`

### **Step 6: Apply Authorization Rules** (Lines 305-315)

#### **Rule 1: Cell Phone (Always Unauthorized)**

```python
if object_type == "cell phone":
    is_unauthorized_action = True  # ⚠️ EVERYONE flagged (even Authorized persons)
```

**Example:**

- Art (Partial) uses phone → ❌ UNAUTHORIZED
- Aldrikz (Authorized) uses phone → ❌ UNAUTHORIZED
- Unknown (Unauthorized) uses phone → ❌ UNAUTHORIZED

#### **Rule 2: All Other Objects (Conditional)**

```python
elif is_unauthorized_person and object_type is not None:
    is_unauthorized_action = True  # ⚠️ Only if Unauthorized or Partially Authorized
```

**Examples:**

| Person  | Auth Level           | Object   | Result          |
| ------- | -------------------- | -------- | --------------- |
| Art     | Partially Authorized | Laptop   | ❌ UNAUTHORIZED |
| Art     | Partially Authorized | Backpack | ❌ UNAUTHORIZED |
| Aldrikz | Authorized           | Laptop   | ✅ ALLOWED      |
| Aldrikz | Authorized           | Backpack | ✅ ALLOWED      |
| Unknown | Unauthorized         | Mouse    | ❌ UNAUTHORIZED |
| Unknown | Unauthorized         | Handbag  | ❌ UNAUTHORIZED |

### **Step 7: Record Unauthorized Action** (Lines 317-325)

```python
if is_unauthorized_action:
    action_detail = {
        "person_id": track_id,       # e.g., 5
        "identity": identity,         # e.g., "Art"
        "authorization": auth_level,  # e.g., "Partially Authorized"
        "action": behavior,           # e.g., "STATUS: INTERACTING WITH LAPTOP"
        "object": object_type         # e.g., "laptop"
    }
    unauthorized_actions.append(action_detail)
```

### **Step 8: Create Frame Log Entry** (Lines 328-340)

```python
log_entry = {
    "frame_id": frame_idx,                      # e.g., 280
    "timestamp": timestamp,                     # e.g., "2026-03-02 23:55:16"
    "unauthorized_count": len(unauthorized_actions),  # e.g., 1
    "details": unauthorized_actions             # List of all violations in this frame
}

# Only save to history if there were violations
if len(unauthorized_actions) > 0:
    self.unauthorized_logs.append(log_entry)
```

### **Step 9: Session End Analysis**

When monitoring stops, the system calculates:

```python
summary = {
    "total_frames_with_unauthorized": 38,     # How many frames had violations
    "total_unauthorized_actions": 38,         # Total count across all frames
    "avg_per_frame": 1.0,                     # Average per frame
    "max_in_single_frame": 1,                 # Highest count in any frame
    "by_object_type": {
        "laptop": 38                          # Breakdown by object
    },
    "by_person": {
        "5": 38                               # Breakdown by person ID
    }
}
```

---

## Key Changes Summary

### **Change 1: Object Detection Scope**

```diff
- elif is_unauthorized_person and object_type in ["laptop", "keyboard", "mouse"]:
+ elif is_unauthorized_person and object_type is not None:
```

**Impact:** Now detects ALL objects (laptop, keyboard, mouse, backpack, handbag) instead of just work equipment.

### **Change 2: Counting Before Filtering** (Line 735)

```python
# Count unauthorized actions BEFORE filtering
unauthorized_log = self.count_unauthorized_actions(detections, self._frame_idx, current_timestamp)

# Then filter out Partially Authorized if Authorized person present
has_authorized = any(d["authorization"] == "Authorized" for d in detections)
if has_authorized:
    detections = [d for d in detections if d["authorization"] != "Partially Authorized"]
```

**Why:** If we counted AFTER filtering, partially authorized people would be removed when an authorized person is present, and their violations wouldn't be recorded.

---

## Why It Wasn't Working Before

### **Problem 1: Wrong Object List**

Your test video had someone using a **backpack**, but the code only checked:

```python
object_type in ["laptop", "keyboard", "mouse"]  # ❌ "backpack" not in list!
```

### **Problem 2: Filtering Before Counting**

Originally at line 733:

```python
# This happened FIRST (wrong order!)
if has_authorized:
    detections = [d for d in detections if d["authorization"] != "Partially Authorized"]

# Then counting happened (but detections were already filtered!)
unauthorized_log = self.count_unauthorized_actions(detections, ...)
```

If Aldrikz (Authorized) was in the room, Art (Partially Authorized) would be filtered out **before** counting, so Art's backpack use was never recorded.

---

## Authorization Rules Summary

| Object Type | Authorized   | Partially Authorized | Unauthorized |
| ----------- | ------------ | -------------------- | ------------ |
| Cell Phone  | ❌ VIOLATION | ❌ VIOLATION         | ❌ VIOLATION |
| Laptop      | ✅ Allowed   | ❌ VIOLATION         | ❌ VIOLATION |
| Keyboard    | ✅ Allowed   | ❌ VIOLATION         | ❌ VIOLATION |
| Mouse       | ✅ Allowed   | ❌ VIOLATION         | ❌ VIOLATION |
| Backpack    | ✅ Allowed   | ❌ VIOLATION         | ❌ VIOLATION |
| Handbag     | ✅ Allowed   | ❌ VIOLATION         | ❌ VIOLATION |

---

## Example Output

### JSON Log Entry

```json
{
    "frame_id": 280,
    "timestamp": "2026-03-02 23:55:16",
    "unauthorized_count": 1,
    "details": [
        {
            "person_id": 5,
            "identity": "Art",
            "authorization": "Partially Authorized",
            "action": "STATUS: INTERACTING WITH BACKPACK",
            "object": "backpack"
        }
    ]
}
```

### Session Summary

```
Total frames with unauthorized actions: 38
Total unauthorized actions: 38
Average per frame: 1.0
Max in single frame: 1

Breakdown by object type:
  backpack: 38

Breakdown by person:
  Person 5: 38 violations
```

---

## Debug Output

When running, you'll see in PowerShell:

```
[DEBUG-COUNT] Frame 280: Checking 1 detections
[DEBUG-COUNT]   - Art (Partially Authorized): STATUS: INTERACTING WITH BACKPACK
[DEBUG] Frame 280: Art (Partially Authorized) - STATUS: INTERACTING WITH BACKPACK -> object: backpack
[DEBUG] UNAUTHORIZED: Art (Partially Authorized) using backpack
[DEBUG] Frame 280: 1 unauthorized action(s) logged
```

---

## Files Modified

1. **combined_yolo_facenet_behavior.py** (Lines 305-315)
    - Changed object detection rule from specific list to "any object"
    - Updated comments to reflect new logic

2. **combined_yolo_facenet_behavior.py** (Line 735)
    - Moved `count_unauthorized_actions()` call BEFORE filtering
    - Added debug print statements

3. **combined_yolo_facenet_behavior.py** (Lines 814-855)
    - Improved `save_unauthorized_logs_to_file()` with atomic writes
    - Added string key conversion for JSON compatibility
    - Added fsync for immediate disk write

---

## Testing Recommendation

Test with different scenarios:

1. **Authorized person + laptop** → Should NOT count ✅
2. **Partially Authorized person + laptop** → Should count ❌
3. **Unauthorized person + laptop** → Should count ❌
4. **Anyone + cell phone** → Should ALWAYS count ❌
5. **Authorized person + backpack** → Should NOT count ✅
6. **Partially Authorized person + backpack** → Should count ❌

---

**End of Explanation**
