# 📄 Logs Folder Fix - Date Organized

## ✅ What Was Fixed

Session logs are now properly saved in the **`logs/` folder** organized by date, instead of being mixed with evidence files.

## 📁 New Folder Structure

### Before (Wrong)

```
office_evidence/
└── rtsp_20260219_153631/
    ├── session_log.txt              ← Log mixed with evidence
    ├── alert_20260219-153720_laptop_Art_ID1.jpg
    └── alert_20260219-153725_backpack_Art_ID1.jpg
```

### After (Fixed) ✅

```
office_evidence/
└── rtsp_20260219_153631/
    ├── alert_20260219-153720_laptop_Art_ID1.jpg
    └── alert_20260219-153725_backpack_Art_ID1.jpg

logs/
└── 02-19-2026/                       ← Organized by date!
    ├── session_rtsp_20260219_153631.txt
    ├── session_rtsp_20260219_160000.txt
    └── session_webcam_0_20260219_170500.txt
```

## 🎯 Log File Naming Format

```
logs/{MM-DD-YYYY}/session_{SOURCE}_{TIMESTAMP}.txt
```

Examples:

- `logs/02-19-2026/session_rtsp_20260219_153631.txt` - RTSP camera (Feb 19)
- `logs/02-19-2026/session_webcam_0_20260219_154520.txt` - Webcam (Feb 19)
- `logs/02-20-2026/session_laptop_20260220_090000.txt` - Video file (Feb 20)
- `logs/10-10-2026/session_rtsp_20261010_120000.txt` - RTSP camera (Oct 10)

## 📋 What's Logged

### Session Header

```
============================================================
[February 19, 2026 at 03:36:31 PM] CCTV Monitoring Session Started
[February 19, 2026 at 03:36:31 PM] Source: rtsp
[February 19, 2026 at 03:36:31 PM] Evidence Folder: rtsp_20260219_153631
[February 19, 2026 at 03:36:31 PM] Log File: 02-19-2026/session_rtsp_20260219_153631.txt
============================================================
```

### Person Detection Events

```
[February 19, 2026 at 03:36:45 PM] [PRIMARY] Person detected: Art (Authorized)
[February 19, 2026 at 03:36:52 PM] [SECONDARY] Person detected: Unknown (Unauthorized)
```

### Behavior Events

```
[February 19, 2026 at 03:37:10 PM] [PRIMARY] Art (ID: 1) - INTERACTING WITH LAPTOP
[February 19, 2026 at 03:37:25 PM] [SECONDARY] Art (ID: 1) - CARRYING BACKPACK
```

### Session End

```
============================================================
[February 19, 2026 at 03:45:00 PM] CCTV Monitoring Session Ended
============================================================
```

## 🔍 UI Changes

When you start monitoring, you'll now see:

```
📁 Evidence folder: rtsp_20260219_153631/
📄 Log file: logs/02-19-2026/session_rtsp_20260219_153631.txt
```

## 🚀 How to Use

### 1. Start Monitoring

- Run the app and click **▶ Start**
- Check the System Log for the log file name (includes date folder)

### 2. Access Logs

Three ways:

**A. Click Button in UI (Recommended):**

- Click **📂 Open Logs Folder** in the System Log section
- Opens **today's date folder** (e.g., `logs/02-19-2026/`) in File Explorer

**B. Browse by Date:**

```bash
cd logs
dir  # List all date folders (02-19-2026, 02-20-2026, etc.)
cd 02-19-2026  # Open specific date
dir  # List all session logs for that date
```

**C. Navigate Directly:**

```bash
cd logs\02-19-2026
dir  # List today's session logs
```

### 3. View Log File

```bash
# Windows
notepad logs\02-19-2026\session_rtsp_20260219_153631.txt

# Or double-click the file in File Explorer
```

## 📊 Benefits

1. **Date Organization** - Easy to find logs by day (logs/02-19-2026/)
2. **Organized Structure** - Logs separate from evidence files
3. **Clean Folders** - Each day's logs in separate folder
4. **Clear Naming** - Date folder + timestamp and source in filename
5. **Better Management** - Easy to archive or clean up old dates
6. **Evidence Integrity** - Evidence folders only contain screenshots

## 💡 Tips

### Find Logs by Date

Logs are organized by date folders, so you can easily browse by day:

```
logs/
├── 02-19-2026/
│   ├── session_rtsp_20260219_153631.txt  ← 3:36 PM session
│   ├── session_rtsp_20260219_160000.txt  ← 4:00 PM session
│   └── session_webcam_0_20260219_170500.txt
├── 02-20-2026/
│   └── session_rtsp_20260220_090000.txt
└── 10-10-2026/
    └── session_rtsp_20261010_120000.txt
```

### Match Log to Evidence

The log file contains the evidence folder name:

```
Evidence Folder: rtsp_20260219_153631
```

Then check:

```
office_evidence/rtsp_20260219_153631/
```

### Clean Up Old Logs

Easy to delete entire date folders:

```bash
# Windows - Delete old date folders
cd logs
dir  # See all date folders
rmdir /s 02-10-2026  # Delete specific date folder

# Or keep only recent dates (e.g., last 30 days)
# Manually delete older date folders
```

### Archive Logs by Date

```bash
# Zip entire date folder for archiving
tar -czf logs_02-19-2026.tar.gz logs/02-19-2026/

# Or use Windows compression
# Right-click date folder → Send to → Compressed (zipped) folder
```

### Search Logs

Use `findstr` (Windows) or `grep` (Linux/Mac) to search across dates:

```bash
# Find all unauthorized detections (search all dates)
findstr /s "Unauthorized" logs\*\*.txt

# Find all unauthorized detections (specific date)
findstr "Unauthorized" logs\02-19-2026\*.txt

# Find specific person across all dates
findstr /s "Art" logs\*\*.txt

# Find specific person (specific date)
findstr "Art" logs\02-19-2026\session_rtsp_20260219_153631.txt
```

## 🔧 Technical Details

### Log File Location

```python
# Date folder format
current_date = datetime.now().strftime("%m-%d-%Y")  # 02-19-2026
logs_date_dir = REPO_ROOT / "logs" / current_date

# Full log path
log_file_path = logs_date_dir / f"session_{source_label}_{session_timestamp}.txt"
# Example: logs/02-19-2026/session_rtsp_20260219_153631.txt
```

### Date Folder Format

```python
current_date = datetime.now().strftime("%m-%d-%Y")
# Format: MM-DD-YYYY
# Examples: 02-19-2026, 10-10-2026, 12-31-2026
```

### Timestamp Format

```python
session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# Example: 20260219_153631 (Year-Month-Day_Hour-Minute-Second)
```

### Log Entry Format

```python
timestamp = datetime.now().strftime("%B %d, %Y at %I:%M:%S %p")
# Example: February 19, 2026 at 03:36:31 PM
```

## ✅ Testing

1. **Start the app:**

    ```bash
    python -m streamlit run .\scripts\streamlit_app.py
    ```

2. **Check UI logs:**
    - Should see: `📄 Log file: logs/02-19-2026/session_...txt` (with today's date)

3. **Click "📂 Open Logs Folder"**
    - Should open **today's date folder** (e.g., `logs/02-19-2026/`)

4. **Verify date folder exists:**

    ```bash
    ls logs/
    # Should see: 02-19-2026/

    ls logs/02-19-2026/
    # Should see: session_rtsp_20260219_153631.txt
    ```

    ```bash
    ls logs/
    ```

5. **Check log contents:**
    - Should contain session header with timestamp and date folder path
    - Should log person detections and behaviors

## 🎉 Conclusion

Your logs are now properly organized in date-based folders (`logs/02-19-2026/`) with clear naming and timestamps. This makes it easy to:

- **Browse by date** - Find all logs for a specific day
- **Review session activity** - Each session in its own log file
- **Correlate with evidence** - Match timestamps with evidence folders
- **Archive old logs** - Delete or compress entire date folders
- **Search efficiently** - Search within specific dates or across all dates
- **Clean management** - Easy to identify and remove old date folders

Enjoy your date-organized CCTV system! 📄✨
