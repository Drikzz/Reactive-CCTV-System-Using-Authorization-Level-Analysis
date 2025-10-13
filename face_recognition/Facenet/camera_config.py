"""
Camera Configuration for FaceNet Recognition System
Supports webcam, video files, and RTSP IP cameras (Tapo C200)
"""

# ==================== CAMERA SOURCE SETTINGS ====================

# Camera Mode: "webcam", "video", or "rtsp"
CAMERA_MODE = "webcam"  # Change to "rtsp" for Tapo camera

# Webcam Settings
WEBCAM_ID = 0  # Usually 0 for default webcam

# Video File Settings
VIDEO_FILE_PATH = r"C:\Users\Alexa\OneDrive\Documents\Thesis\Reactive-CCTV-System-Using-Authorization-Level-Analysis\Mp4TESTING\ThesisMP4TEST3.mp4"

# RTSP Camera Settings (Tapo C200)
RTSP_CAMERAS = {
    "tapo_c200_main": {
        "ip": "192.168.1.245",
        "port": "554",
        "username": "reactivecctv101",
        "password": "reactivecctv101",
        "stream": "stream1",  # stream1 = 1080p, stream2 = 640x480
        "enabled": True
    },
    # Add more cameras here if needed
    # "tapo_c200_backup": {
    #     "ip": "192.168.1.245",
    #     "port": "554",
    #     "username": "reactivecctv101",
    #     "password": "reactivecctv101",
    #     "stream": "stream2",
    #     "enabled": False
    # }
}

# Select which RTSP camera to use (key from RTSP_CAMERAS)
ACTIVE_RTSP_CAMERA = "tapo_c200_main"

# ==================== RTSP CONNECTION SETTINGS ====================

# RTSP protocol: "rtsp" or "rtspt" (TCP mode for more stability)
RTSP_PROTOCOL = "rtsp"

# RTSP connection timeout (seconds)
RTSP_TIMEOUT = 10

# RTSP reconnection settings
RTSP_AUTO_RECONNECT = True
RTSP_MAX_RECONNECT_ATTEMPTS = 5
RTSP_RECONNECT_DELAY = 3  # seconds between reconnection attempts

# Buffer size (1 = minimal latency, higher = more stable but delayed)
RTSP_BUFFER_SIZE = 1

# ==================== VIDEO PROCESSING SETTINGS ====================

# Frame skip for processing (process every Nth frame)
FRAME_SKIP = 2

# Resize width for processing (None = use original)
PROCESS_WIDTH = 720  # Reduce for better performance

# Display settings
DISPLAY_WIDTH = 1280  # Maximum width for display window
SHOW_FPS = True
SHOW_STREAM_INFO = True

# ==================== HELPER FUNCTIONS ====================

def get_rtsp_url(camera_key=None):
    """
    Generate RTSP URL from camera configuration
    
    Args:
        camera_key: Key from RTSP_CAMERAS dict. If None, uses ACTIVE_RTSP_CAMERA
    
    Returns:
        RTSP URL string or None if camera not found/disabled
    """
    if camera_key is None:
        camera_key = ACTIVE_RTSP_CAMERA
    
    camera = RTSP_CAMERAS.get(camera_key)
    
    if camera is None:
        print(f"[ERROR] Camera '{camera_key}' not found in configuration")
        return None
    
    if not camera.get("enabled", False):
        print(f"[WARN] Camera '{camera_key}' is disabled")
        return None
    
    ip = camera["ip"]
    port = camera["port"]
    username = camera["username"]
    password = camera["password"]
    stream = camera["stream"]
    
    # Construct RTSP URL
    url = f"{RTSP_PROTOCOL}://{username}:{password}@{ip}:{port}/{stream}"
    
    return url


def get_camera_source():
    """
    Get the appropriate camera source based on CAMERA_MODE
    
    Returns:
        Camera source (int for webcam, str for video/RTSP) or None if error
    """
    if CAMERA_MODE == "webcam":
        return WEBCAM_ID
    
    elif CAMERA_MODE == "video":
        return VIDEO_FILE_PATH
    
    elif CAMERA_MODE == "rtsp":
        return get_rtsp_url()
    
    else:
        print(f"[ERROR] Invalid CAMERA_MODE: {CAMERA_MODE}")
        return None


def print_camera_info():
    """Print current camera configuration"""
    print("\n" + "="*60)
    print("CAMERA CONFIGURATION")
    print("="*60)
    print(f"Mode: {CAMERA_MODE}")
    
    if CAMERA_MODE == "webcam":
        print(f"Webcam ID: {WEBCAM_ID}")
    
    elif CAMERA_MODE == "video":
        print(f"Video File: {VIDEO_FILE_PATH}")
    
    elif CAMERA_MODE == "rtsp":
        camera = RTSP_CAMERAS.get(ACTIVE_RTSP_CAMERA)
        if camera:
            print(f"Camera: {ACTIVE_RTSP_CAMERA}")
            print(f"IP: {camera['ip']}")
            print(f"Stream: {camera['stream']} ({'1080p' if camera['stream'] == 'stream1' else '640x480'})")
            print(f"URL: {get_rtsp_url()}")
            print(f"Auto-reconnect: {RTSP_AUTO_RECONNECT}")
        else:
            print(f"[ERROR] Active camera '{ACTIVE_RTSP_CAMERA}' not found!")
    
    print(f"Processing width: {PROCESS_WIDTH}px")
    print(f"Frame skip: {FRAME_SKIP}")
    print("="*60 + "\n")


# Quick configuration presets
def use_webcam():
    """Quick switch to webcam"""
    global CAMERA_MODE
    CAMERA_MODE = "webcam"

def use_video_file(path=None):
    """Quick switch to video file"""
    global CAMERA_MODE, VIDEO_FILE_PATH
    CAMERA_MODE = "video"
    if path:
        VIDEO_FILE_PATH = path

def use_rtsp_camera(camera_key=None):
    """Quick switch to RTSP camera"""
    global CAMERA_MODE, ACTIVE_RTSP_CAMERA
    CAMERA_MODE = "rtsp"
    if camera_key:
        ACTIVE_RTSP_CAMERA = camera_key


if __name__ == "__main__":
    # Test configuration
    print_camera_info()
    
    source = get_camera_source()
    if source:
        print(f"[OK] Camera source: {source}")
    else:
        print("[ERROR] Failed to get camera source")
