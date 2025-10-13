# Authorization Level System Guide

## 📋 Overview

Your FaceNet system now supports **2-level authorization**:

| Level | Access | Color | Symbol | Usage |
|-------|--------|-------|--------|-------|
| **Authorized** | Full access | 🟢 Green | ✓ | High-level personnel |
| **Partial** | Limited access | 🟡 Yellow | ◐ | Regular staff |
| **Unauthorized** | No access | 🔴 Red | ✗ | Unknown/blocked |

---

## 🎯 How It Works

### **Your Current Setup:**

```
datasets/faces/
├── Aldrikz/     → Authorized ✓ (Green)
├── Art/         → Authorized ✓ (Green)
├── Jude/        → Partial ◐ (Yellow)
├── Myke/        → Partial ◐ (Yellow)
└── Speed/       → Partial ◐ (Yellow)
```

**No need to reorganize folders!** The mapping is handled in `authorization_config.py`.

---

## ⚙️ Configuration

### **1. Edit Authorization Levels**

Open [`face_recognition/Facenet/authorization_config.py`](authorization_config.py):

```python
# Line 29-39: Define who has what access
AUTHORIZATION_MAP = {
    # Authorized personnel (full access) - GREEN
    "Aldrikz": AUTHORIZED,
    "Art": AUTHORIZED,
    
    # Partial access personnel (limited access) - YELLOW
    "Jude": PARTIAL,
    "Myke": PARTIAL,
    "Speed": PARTIAL,
    
    # Add new people here:
    # "John_Doe": AUTHORIZED,      # Full access
    # "Jane_Smith": PARTIAL,       # Limited access
}
```

### **2. Test Configuration**

```powershell
cd face_recognition\Facenet
python authorization_config.py
```

**Expected output:**
```
============================================================
AUTHORIZATION CONFIGURATION
============================================================

✓ AUTHORIZED: 2 person(s)
   - Aldrikz
   - Art

◐ PARTIAL: 3 person(s)
   - Jude
   - Myke
   - Speed

✗ UNAUTHORIZED: 0 person(s)
   (none)

Default level for unmapped people: Partial
Unknown faces treated as: Unauthorized
============================================================
```

---

## 🚀 Usage

### **Run Face Recognition**

```powershell
python facenet_main.py
```

**You'll see:**
```
============================================================
CAMERA CONFIGURATION
============================================================
Mode: webcam
...

============================================================
AUTHORIZATION CONFIGURATION
============================================================
✓ AUTHORIZED: 2 person(s)
   - Aldrikz
   - Art
...

[INFO] Starting recognition...
```

### **Visual Display**

**On-screen labels:**
- `✓ ID:1 Aldrikz [Authorized] (0.95)` - Green box, thick border
- `◐ ID:2 Jude [Partial] (0.87)` - Yellow box, medium border
- `✗ ID:3 Unknown [Unauthorized]` - Red box, medium border

---

## 📁 File Organization

### **Logs are automatically separated by level:**

```
logs/FaceNet/
├── authorized/       # Aldrikz, Art detections
│   ├── Aldrikz_1234_153045.jpg
│   └── Art_1235_153046.jpg
├── partial/          # Jude, Myke, Speed detections
│   ├── Jude_1236_153047.jpg
│   └── Myke_1237_153048.jpg
└── unauthorized/     # Unknown faces
    └── Unknown_1238_153049.jpg

annotated_frames/FaceNet/
├── authorized/       # Full frame captures of authorized people
├── partial/          # Full frame captures of partial access people
└── unauthorized/     # Full frame captures of unknown people
```

---

## 🎨 Customization

### **Change Colors**

```python
# In authorization_config.py, line 51-55:
LEVEL_COLORS = {
    AUTHORIZED: (0, 255, 0),      # BGR format
    PARTIAL: (0, 255, 255),       # Change to (255, 165, 0) for orange
    UNAUTHORIZED: (0, 0, 255),    # Change to (128, 0, 128) for purple
}
```

### **Change Display Symbols**

```python
# Line 63-67:
LEVEL_PREFIX = {
    AUTHORIZED: "✓",     # Change to "★" for star
    PARTIAL: "◐",        # Change to "⚠" for warning
    UNAUTHORIZED: "✗",   # Change to "⛔" for blocked
}
```

### **Change Default Level**

```python
# Line 42: What level for people NOT in the map?
DEFAULT_LEVEL = PARTIAL  # Change to AUTHORIZED or UNAUTHORIZED
```

---

## 🔧 Adding New People

### **Method 1: Capture New Face**

```powershell
python facenet_capture.py
# Enter name: "John_Doe"
```

Then add to `authorization_config.py`:
```python
AUTHORIZATION_MAP = {
    # ... existing people ...
    "John_Doe": AUTHORIZED,  # Add this line
}
```

### **Method 2: Update Existing Person**

Just change their level in `authorization_config.py`:
```python
# Promote Jude from Partial to Authorized
"Jude": AUTHORIZED,  # Changed from PARTIAL
```

**No need to retrain the model!** Just restart `facenet_main.py`.

---

## 📊 Integration Examples

### **Access Control System**

```python
from authorization_config import get_authorization_level, AUTHORIZED, PARTIAL

def grant_access(person_name):
    level = get_authorization_level(person_name)
    
    if level == AUTHORIZED:
        # Open all doors
        unlock_main_door()
        unlock_secure_area()
        log_access(person_name, "FULL_ACCESS")
        
    elif level == PARTIAL:
        # Limited access
        unlock_main_door()
        # Keep secure_area locked
        log_access(person_name, "PARTIAL_ACCESS")
        
    else:  # UNAUTHORIZED
        # No access
        trigger_alert(f"Unauthorized person detected: {person_name}")
        log_access(person_name, "DENIED")
```

### **Check Authorization in Code**

```python
from authorization_config import is_authorized, has_partial_access

# Quick checks
if is_authorized("Aldrikz"):
    print("Full access granted!")

if has_partial_access("Jude"):
    print("Limited access granted!")
```

---

## 🔍 Troubleshooting

### **Problem: Person shows wrong color**

**Solution:** Check `authorization_config.py` mapping:
```powershell
python authorization_config.py
# Look for the person in the output
```

### **Problem: New person shows as "Unknown"**

**Solutions:**
1. Make sure you captured their face: `python facenet_capture.py`
2. Retrain the model: `python facenet_train.py`
3. Add them to `authorization_config.py`

### **Problem: Colors not changing**

**Solution:** Restart `facenet_main.py` after editing `authorization_config.py`

---

## 📝 Quick Reference

| Task | Command/File |
|------|--------------|
| **Set authorization levels** | Edit `authorization_config.py` |
| **Test configuration** | `python authorization_config.py` |
| **Capture new face** | `python facenet_capture.py` |
| **Train model** | `python facenet_train.py` |
| **Run recognition** | `python facenet_main.py` |
| **Check person's level** | `get_authorization_level("Name")` |

---

## ✅ Summary

- ✅ **2 authorization levels** (Authorized, Partial)
- ✅ **No folder reorganization needed** - mapping in config file
- ✅ **Color-coded display** - Green (full), Yellow (limited), Red (none)
- ✅ **Automatic log separation** - files organized by access level
- ✅ **Easy to update** - just edit `authorization_config.py`
- ✅ **Backward compatible** - works with existing faces

Your dataset stays the same, authorization is managed separately! 🎉
