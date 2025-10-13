# Camera Configuration Guide

## 📋 Overview

The FaceNet system now supports **3 camera modes**:
1. **Webcam** - Local USB/built-in camera
2. **Video File** - Pre-recorded MP4/AVI files
3. **RTSP Camera** - Tapo C200 IP camera

All camera settings are managed in **`camera_config.py`**.

---

## 🚀 Quick Start

### 1️⃣ **Configure Your Camera**

Edit `face_recognition/Facenet/camera_config.py`:

```python
# Change this line to select mode:
CAMERA_MODE = "rtsp"  # Options: "webcam", "video", "rtsp"

# For RTSP (Tapo C200):
RTSP_CAMERAS = {
    "tapo_c200_main": {
        "ip": "192.168.1.244",      # Your camera IP
        "username": "reactivecctv101",  # Camera username
        "password": "reactivecctv101",  # Camera password
        "stream": "stream1",        # stream1=1080p, stream2=640x480
        "enabled": True
    }
}

ACTIVE_RTSP_CAMERA = "tapo_c200_main"  # Which camera to use
```

### 2️⃣ **Test Your Camera**

```bash
cd face_recognition/Facenet
python setup_camera.py
```

**Expected output:**
```
============================================================
CAMERA CONFIGURATION
============================================================
Mode: rtsp
Camera: tapo_c200_main
IP: 192.168.1.244
Stream: stream1 (1080p)
URL: rtsp://reactivecctv101:reactivecctv101@192.168.1.244:554/stream1
Auto-reconnect: True
Processing width: 720px
Frame skip: 2
============================================================

[OK] Camera opened successfully!
[INFO] Resolution: 1920x1080
[INFO] FPS: 15.0
[INFO] Capturing frames... Press 'q' to quit
```

### 3️⃣ **Run FaceNet Recognition**

```bash
# With configured camera
python face_recognition/Facenet/facenet_main.py

# Or capture faces
python face_recognition/Facenet/facenet_capture.py
```

---

## ⚙️ Configuration Options

### **Webcam Mode**
```python
CAMERA_MODE = "webcam"
WEBCAM_ID = 0  # 0 = default, 1 = second camera
```

### **Video File Mode**
```python
CAMERA_MODE = "video"
VIDEO_FILE_PATH = r"C:\path\to\video.mp4"
```

### **RTSP Camera Mode**
```python
CAMERA_MODE = "rtsp"

# Add multiple cameras
RTSP_CAMERAS = {
    "camera1": {
        "ip": "192.168.1.244",
        "port": "554",
        "username": "user",
        "password": "pass",
        "stream": "stream1",  # 1080p
        "enabled": True
    },
    "camera2": {
        "ip": "192.168.1.245",
        "stream": "stream2",  # 640x480 (faster)
        "enabled": False
    }
}

ACTIVE_RTSP_CAMERA = "camera1"  # Switch between cameras
```

---

## 🔧 RTSP Settings

### **Connection Settings**
```python
RTSP_PROTOCOL = "rtsp"  # or "rtspt" for TCP mode (more stable)
RTSP_TIMEOUT = 10  # Connection timeout in seconds
RTSP_BUFFER_SIZE = 1  # 1=minimal latency, higher=more buffering
```

### **Auto-Reconnection**
```python
RTSP_AUTO_RECONNECT = True  # Auto-reconnect on connection loss
RTSP_MAX_RECONNECT_ATTEMPTS = 5  # Max retry attempts
RTSP_RECONNECT_DELAY = 3  # Seconds between retries
```

---

## 🎯 Stream Quality

### **Tapo C200 Streams:**
- **stream1** = 1080p (1920×1080) - High quality, slower
- **stream2** = 640×480 - Lower quality, faster

### **Choose based on your needs:**
```python
# High quality for recognition
"stream": "stream1"  

# Fast processing (recommended for real-time)
"stream": "stream2"  
```

---

## 🐛 Troubleshooting

### **Problem: "Failed to open camera"**

**Solution 1: Check IP**
```bash
# Ping your camera
ping 192.168.1.244
```

**Solution 2: Test in VLC**
1. Open VLC Media Player
2. Media → Open Network Stream
3. Paste: `rtsp://reactivecctv101:reactivecctv101@192.168.1.244:554/stream1`
4. If it works in VLC but not Python, check firewall

**Solution 3: Check credentials**
- Verify username/password in Tapo app
- Try logging into camera web interface

---

### **Problem: "RTSP connection lost" (keeps reconnecting)**

**Solution:**
```python
# In camera_config.py
RTSP_PROTOCOL = "rtspt"  # Use TCP instead of UDP
RTSP_BUFFER_SIZE = 3  # Increase buffer
```

---

### **Problem: High latency/delay**

**Solution:**
```python
# Use lower resolution stream
"stream": "stream2"  # 640x480 instead of 1080p

# Reduce buffer
RTSP_BUFFER_SIZE = 1

# Process fewer frames
FRAME_SKIP = 3  # Process every 3rd frame
```

---

## 📝 Quick Commands Reference

### **Switch Camera Modes (in Python)**
```python
import camera_config

# Switch to webcam
camera_config.use_webcam()

# Switch to video file
camera_config.use_video_file("path/to/video.mp4")

# Switch to RTSP camera
camera_config.use_rtsp_camera("tapo_c200_main")

# Get current source
source = camera_config.get_camera_source()
```

---

## 🎓 Examples

### **Example 1: Single Tapo Camera**
```python
CAMERA_MODE = "rtsp"
ACTIVE_RTSP_CAMERA = "tapo_c200_main"
```

### **Example 2: Testing with Video File**
```python
CAMERA_MODE = "video"
VIDEO_FILE_PATH = r"C:\Users\test\video.mp4"
```

### **Example 3: Switch Between Webcam and RTSP**
```python
# In your script:
import camera_config

# Use webcam for testing
camera_config.use_webcam()
# ... test code ...

# Switch to RTSP for production
camera_config.use_rtsp_camera("tapo_c200_main")
# ... production code ...
```

---

## ✅ Files Modified

1. **`camera_config.py`** - NEW: Camera configuration
2. **`setup_camera.py`** - NEW: Camera testing tool
3. **`facenet_main.py`** - Updated to use camera_config
4. **`facenet_capture.py`** - Updated to use camera_config

---

## 🚀 Next Steps

1. ✅ Configure your camera in `camera_config.py`
2. ✅ Test with `setup_camera.py`
3. ✅ Run `facenet_main.py` for recognition
4. ✅ Use `facenet_capture.py` to add new people

---

## 📞 Support

**Camera not working?**
1. Run `python setup_camera.py` for diagnostics
2. Check network connectivity
3. Verify camera settings in Tapo app
4. Try different stream quality

**Need help?**
- Check error messages in terminal
- Ensure camera is on same network as computer
- Try lower stream quality first (stream2)
