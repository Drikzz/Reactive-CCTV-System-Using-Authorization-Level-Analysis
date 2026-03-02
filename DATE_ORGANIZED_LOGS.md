# 📅 Date-Organized Logs - Quick Reference

## 📁 Folder Structure

```
logs/
├── 02-19-2026/                    ← Today's logs
│   ├── session_rtsp_20260219_153631.txt
│   ├── session_rtsp_20260219_160000.txt
│   └── session_webcam_0_20260219_170500.txt
├── 02-20-2026/                    ← Tomorrow's logs
│   └── session_rtsp_20260220_090000.txt
├── 10-10-2026/                    ← Future date
│   └── session_rtsp_20261010_120000.txt
└── 12-31-2026/                    ← Year-end logs
    └── session_rtsp_20261231_235959.txt
```

## 🎯 Key Features

✅ **Date Folders** - Each day gets its own folder (MM-DD-YYYY format)
✅ **Easy Browsing** - Click "📂 Open Logs Folder" opens TODAY's folder
✅ **Clean Organization** - All sessions for a day in one place
✅ **Easy Cleanup** - Delete entire date folders for old logs
✅ **Simple Archive** - Zip date folders for backup

## 📋 UI Display

When you start monitoring:

```
📄 Log file: logs/02-19-2026/session_rtsp_20260219_153631.txt
```

## 🚀 Quick Commands

### View Today's Logs

```bash
cd logs\02-19-2026
dir
```

### Browse All Dates

```bash
cd logs
dir  # Shows: 02-19-2026, 02-20-2026, etc.
```

### Delete Old Logs

```bash
# Delete specific date
rmdir /s logs\01-01-2026

# Keep only last 30 days (delete older folders manually)
```

### Search Across All Dates

```bash
# Find in all dates
findstr /s "Unauthorized" logs\*\*.txt

# Find in specific date
findstr "Art" logs\02-19-2026\*.txt
```

## 💡 Benefits

1. **Date Navigation** - Jump to any date instantly
2. **No File Clutter** - Each date isolated
3. **Easy Archiving** - Zip entire date folders
4. **Fast Cleanup** - Remove old date folders
5. **Better Search** - Search by date or across all

Enjoy! 📅✨
