# Dual Camera Separate Recording Feature

## 🎥 What Changed?

The system now saves **two separate video files** when using dual camera mode, instead of one stitched video.

## ✅ Benefits

1. **Better Quality** - Each camera saves at its native resolution (640x480)
2. **Independent Playback** - Review each camera separately or side-by-side
3. **No Detection Issues** - Each camera processes at optimal size for better behavior detection
4. **Easier Evidence Review** - Focus on one camera at a time
5. **Smaller File Sizes** - Two smaller files instead of one wide file

## 📁 Folder Structure

### Single Camera Mode

```
recordings/
└── rtsp_20260213_153631/
    └── recording_20260213_153631.mp4
```

### Dual Camera Mode (NEW!)

```
recordings/
└── rtsp_20260213_153631/
    ├── recording_primary_20260213_153631.mp4    ← Primary camera
    └── recording_secondary_20260213_153631.mp4  ← Secondary camera
```

## 🔄 How It Works

### Display

- **Live view**: Shows both cameras **side-by-side** (stitched) in the Streamlit UI
- **Behavior detection**: Works independently on each camera feed
- **Evidence screenshots**: Saved per camera with `[Primary]` or `[Secondary]` labels

### Recording

- **Primary camera**: Saved as `recording_primary_TIMESTAMP.mp4`
- **Secondary camera**: Saved as `recording_secondary_TIMESTAMP.mp4`
- **Both files**: Stored in the same session folder
- **Raw frames**: Each camera records its **original unprocessed frames** (not annotated)

### Evidence Screenshots

- Saved as before: `alert_TIMESTAMP_object_person_ID.jpg`
- Same folder: `office_evidence/rtsp_TIMESTAMP/`

## 🎯 Use Cases

### Review Both Cameras Together

Open both videos side-by-side in VLC or any video player:

```bash
# Primary camera view
recordings/rtsp_20260213_153631/recording_primary_20260213_153631.mp4

# Secondary camera view
recordings/rtsp_20260213_153631/recording_secondary_20260213_153631.mp4
```

### Review One Camera Only

Focus on specific camera angle:

- Primary: Usually the main entrance/area
- Secondary: Usually the side or back area

### Sync with Evidence

Evidence screenshots show which camera detected the behavior:

```
office_evidence/rtsp_20260213_153631/
├── alert_20260213-153720_laptop_Art_ID1.jpg  → Check timestamp
└── session_log.txt                           → See which camera detected it
```

## 🚀 Testing Your Dual Camera Setup

1. **Start the app**:

    ```bash
    python -m streamlit run .\scripts\streamlit_app.py
    ```

2. **Configure dual cameras**:
    - Primary Camera: RTSP → tapo_c200_main
    - ✅ Enable Dual Camera
    - Secondary Camera: RTSP → tapo_c200_secondary

3. **Enable recording**:
    - ✅ Enable Recording

4. **Enable behavior detection**:
    - ✅ Enable HOI Detection (auto-checked)

5. **Start monitoring**:
    - Click **▶ Start**

6. **Check the logs**:

    ```
    📁 Evidence folder: rtsp_TIMESTAMP/
    🔗 Connecting to secondary camera (rtsp)...
    ✅ Dual camera mode active! Monitoring both feeds side-by-side.
    📹 Recording to: rtsp_TIMESTAMP/ (2 cameras)
    ```

7. **After stopping**, check the recordings folder:
    ```
    recordings/
    └── rtsp_TIMESTAMP/
        ├── recording_primary_TIMESTAMP.mp4
        └── recording_secondary_TIMESTAMP.mp4
    ```

## 📊 File Sizes (Approximate)

### Single Camera (640x480)

- **1 minute**: ~5 MB
- **10 minutes**: ~50 MB
- **1 hour**: ~300 MB

### Dual Camera (2 separate files)

- **1 minute**: ~10 MB (5 MB × 2)
- **10 minutes**: ~100 MB (50 MB × 2)
- **1 hour**: ~600 MB (300 MB × 2)

## 💡 Tips

1. **Disk Space**: Monitor available disk space for long recordings
2. **Evidence Review**: Use `session_log.txt` to correlate timestamps between cameras
3. **Playback**: Use VLC's "Tools → Synchronize" to play both videos in sync
4. **Backup**: Both files are in the same folder for easy backup/archive

## 🔧 Technical Details

### Recording Logic

```python
# Dual camera mode
if enable_dual_cam and recorder2 is not None:
    recorder.write(frame)   # Primary camera (original frame)
    recorder2.write(frame2) # Secondary camera (original frame)

# Single camera mode
else:
    recorder.write(annotated)  # Single annotated frame
```

### Why Separate Files?

**Problem with stitched recording:**

- Stitched frame: 1280x480 (double width)
- Objects appear ~50% smaller
- Harder to review each camera independently
- Single point of failure

**Solution with separate files:**

- Each file: 640x480 (native resolution)
- Objects at original size
- Independent review possible
- Better for evidence/analysis

## ✅ Conclusion

This update makes dual camera recording more practical and useful:

- **Same live view** (stitched side-by-side)
- **Separate recordings** (better quality and flexibility)
- **Independent evidence** (per-camera detection and screenshots)

Enjoy your improved dual camera CCTV system! 🎥✨
