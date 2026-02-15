"""
Camera Configuration for Streamlit CCTV System
Supports webcam, video files, and RTSP IP cameras (Tapo C200)
"""

# ==================== CAMERA SOURCE SETTINGS ====================

# Camera Mode: "webcam", "video", or "rtsp"
CAMERA_MODE = "webcam"  # Default to webcam for Streamlit

# Webcam Settings
WEBCAM_ID = 0  # Usually 0 for default webcam
WEBCAM_ID_SECONDARY = 1  # Secondary webcam ID

# Video File Settings
VIDEO_FILE_PATH = r"C:\Users\rikzs\Desktop\Aldrikz\Code\thesis_system\videos\sample.mp4"

# RTSP Camera Settings (Tapo C200)
RTSP_CAMERAS = {
    "tapo_c200_main": {
        "ip": "192.168.0.110",
        "port": "554",
        "username": "reactivecctv101",
        "password": "reactivecctv101",
        "stream": "stream2",  # stream1 = 1080p, stream2 = 640x480
        "enabled": True
    },
    "tapo_c200_secondary": {
        "ip": "192.168.0.104",
        "port": "554",
        "username": "CakeyFudo",
        "password": "CakeyFudo",
        "stream": "stream2",  # Lower resolution for better performance
        "enabled": True
    }
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
FRAME_SKIP = 1

# Resize factor for processing (0.5 = half size, 1.0 = original)
RESIZE_FACTOR = 0.4

# Display settings
DISPLAY_WIDTH = 640  # Width for Streamlit display
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


def get_camera_source(mode=None):
    """
    Get the appropriate camera source based on CAMERA_MODE or provided mode
    
    Args:
        mode: Override CAMERA_MODE. Can be "webcam", "video", or "rtsp"
    
    Returns:
        Camera source (int for webcam, str for video/RTSP) or None if error
    """
    current_mode = mode if mode else CAMERA_MODE
    
    if current_mode == "webcam":
        return WEBCAM_ID
    
    elif current_mode == "video":
        return VIDEO_FILE_PATH
    
    elif current_mode == "rtsp":
        return get_rtsp_url()
    
    else:
        print(f"[ERROR] Invalid camera mode: {current_mode}")
        return None


def get_all_rtsp_cameras():
    """
    Get list of all available RTSP cameras
    
    Returns:
        List of tuples (camera_key, camera_name, is_enabled)
    """
    cameras = []
    for key, config in RTSP_CAMERAS.items():
        name = f"{config['ip']} ({config['stream']})"
        enabled = config.get("enabled", False)
        cameras.append((key, name, enabled))
    return cameras


def print_camera_info():
    """Print current camera configuration"""
    print("\n" + "="*60)
    print("STREAMLIT CAMERA CONFIGURATION")
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
    
    print(f"Display width: {DISPLAY_WIDTH}px")
    print(f"Frame skip: {FRAME_SKIP}")
    print(f"Resize factor: {RESIZE_FACTOR}")
    print("="*60 + "\n")


# Quick configuration presets
def use_webcam(webcam_id=None):
    """Quick switch to webcam"""
    global CAMERA_MODE, WEBCAM_ID
    CAMERA_MODE = "webcam"
    if webcam_id is not None:
        WEBCAM_ID = webcam_id


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


def validate_camera_source():
    """
    Validate if the current camera configuration is valid
    
    Returns:
        Tuple (is_valid, error_message)
    """
    if CAMERA_MODE == "webcam":
        if not isinstance(WEBCAM_ID, int) or WEBCAM_ID < 0:
            return False, f"Invalid webcam ID: {WEBCAM_ID}"
        return True, None
    
    elif CAMERA_MODE == "video":
        from pathlib import Path
        if not VIDEO_FILE_PATH:
            return False, "Video file path is empty"
        if not Path(VIDEO_FILE_PATH).exists():
            return False, f"Video file not found: {VIDEO_FILE_PATH}"
        return True, None
    
    elif CAMERA_MODE == "rtsp":
        url = get_rtsp_url()
        if not url:
            return False, f"Failed to generate RTSP URL for camera: {ACTIVE_RTSP_CAMERA}"
        return True, None
    
    else:
        return False, f"Invalid camera mode: {CAMERA_MODE}"


if __name__ == "__main__":
    # Test configuration
    print_camera_info()
    
    # Validate
    is_valid, error = validate_camera_source()
    if is_valid:
        source = get_camera_source()
        print(f"[OK] Camera source: {source}")
    else:
        print(f"[ERROR] {error}")
    
    # Show all RTSP cameras
    print("\nAvailable RTSP Cameras:")
    for key, name, enabled in get_all_rtsp_cameras():
        status = "✓" if enabled else "✗"
        print(f"  {status} {key}: {name}")