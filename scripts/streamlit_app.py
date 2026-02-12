import sys
import threading
import queue
import time
import logging
import traceback
from pathlib import Path
from datetime import datetime
import base64
from io import BytesIO
from PIL import Image

import cv2
import numpy as np
import streamlit as st


# Reduce noisy MediaFileHandler tracebacks (they can still appear on stop/refresh
# if the browser requests an older media id). This matches the old app's behavior.
logging.getLogger("streamlit.web.server.media_file_handler").setLevel(logging.ERROR)


# NOTE: We keep Streamlit media handler logs enabled while stabilizing frame rendering.

# Repo root + imports
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import camera_config_streamlit as cam_config


# -----------------------------
# UI styling (kept old look)
# -----------------------------
st.set_page_config(
    page_title="CCTV Security System",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .status-box {
        padding: 0.8rem;
        border-radius: 0.5rem;
        margin: 0.3rem 0;
        font-size: 0.9rem;
        font-weight: 500;
    }
    .status-box strong {
        font-size: 1.1rem;
    }
    .authorized {
        background-color: #d4edda;
        border: 2px solid #28a745;
        color: #155724 !important;
    }
    .authorized strong { color: #0d4017 !important; }
    .partial {
        background-color: #fff3cd;
        border: 2px solid #ffc107;
        color: #856404 !important;
    }
    .partial strong { color: #533f03 !important; }
    .unauthorized {
        background-color: #f8d7da;
        border: 2px solid #dc3545;
        color: #721c24 !important;
    }
    .unauthorized strong { color: #491217 !important; }

    /* Alert/Notification styles */
    .alert-container {
        position: fixed;
        top: 80px;
        right: 20px;
        z-index: 9999;
        display: flex;
        flex-direction: column;
        gap: 10px;
        max-width: 400px;
        pointer-events: none;
    }
    .alert-notification {
        position: relative;
        padding: 1rem 1.5rem;
        border-radius: 0.5rem;
        font-weight: 600;
        font-size: 0.95rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        animation: slideIn 0.3s ease-out;
        width: 100%;
        pointer-events: auto;
    }
    .alert-unauthorized {
        background-color: #dc3545;
        color: white;
        border: 2px solid #a71d2a;
    }
    .alert-partial {
        background-color: #ffc107;
        color: #000;
        border: 2px solid #d39e00;
    }
    @keyframes slideIn {
        from { transform: translateX(400px); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
</style>
""",
    unsafe_allow_html=True,
)


# -----------------------------
# Session state
# -----------------------------
def _init_state(key, default):
    if key not in st.session_state:
        st.session_state[key] = default


_init_state("running", False)
_init_state("frame_queue", queue.Queue(maxsize=2))
_init_state("log_queue", queue.Queue())
_init_state("current_detections", [])
_init_state("processing_thread", None)
_init_state("stop_flag", threading.Event())
_init_state("recording_enabled", False)
_init_state("recording_path", None)
_init_state("enable_logging", False)
_init_state("all_logs", [])  # Store all logs for viewing
_init_state("logged_people", set())  # Track logged person entries to avoid spam
_init_state("logged_behaviors", set())  # Track logged behaviors to avoid spam
_init_state("session_log_file", None)  # Store current session log file path

# Alerts
_init_state("alert_cooldown", 5)
_init_state("alert_duration", 10)
_init_state("active_alerts", {})
_init_state("last_alert", {})


# -----------------------------
# Authorization (Face-only)
# -----------------------------
AUTHORIZATION_MAP = {
    "myke": "Partially Authorized",
    "dean": "Authorized",
    "art": "Partially Authorized",
    "aldrikz": "Partially Authorized",
}


def get_authorization_level(identity_name: str) -> str:
    if not identity_name or identity_name == "Unknown":
        return "Unauthorized"
    return AUTHORIZATION_MAP.get(str(identity_name).lower(), "Partially Authorized")


def get_authorization_color(auth_level: str):
    color_map = {
        "Authorized": (0, 255, 0),
        "Partially Authorized": (0, 165, 255),
        "Unauthorized": (0, 0, 255),
    }
    return color_map.get(auth_level, (128, 128, 128))


# -----------------------------
# UI helper functions
# -----------------------------
def _safe_put(q: queue.Queue, payload):
    try:
        # Add timestamp if it's a log message
        if isinstance(payload, dict) and 'message' in payload and 'timestamp' not in payload:
            payload['timestamp'] = datetime.now().strftime("%H:%M:%S")
        q.put(payload)
    except Exception:
        pass


def frame_to_base64(frame_bgr):
    """Convert OpenCV frame to base64 data URI"""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)
    buffer = BytesIO()
    pil_img.save(buffer, format="JPEG", quality=85)
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/jpeg;base64,{img_str}"


def display_loop(
    video_placeholder,
    detections_placeholder,
    total_placeholder,
    auth_placeholder,
    unauth_placeholder,
    log_placeholder,
    status_placeholder,
    alert_placeholder,
    *,
    frame_sleep_s: float = 0.01,
) -> None:
    """Run a tight UI loop while `running` is True.

    This matches the old app’s approach and avoids `st.rerun()` churn,
    which is the most common trigger for Streamlit MemoryMediaFileStorage
    "Bad filename ...jpg" errors during rapid frame updates.
    """

    # We update placeholders in-place; Streamlit will stream deltas.
    # Important: do NOT call st.rerun() here. The old app stays stable by
    # avoiding rerun churn entirely during live display.
    while st.session_state.running:
        # If stop was requested from sidebar, honor it ASAP.
        if st.session_state.stop_requested:
            break

        # --- Frame
        frame = None
        try:
            if not st.session_state.frame_queue.empty():
                frame = st.session_state.frame_queue.get_nowait()
                st.session_state.last_frame = frame
        except queue.Empty:
            frame = None
        except Exception:
            frame = None

        frame_to_show = st.session_state.last_frame
        if frame_to_show is not None:
            try:
                frame_b64 = frame_to_base64(frame_to_show)
                video_placeholder.markdown(
                    f'<img src="{frame_b64}" width="640">',
                    unsafe_allow_html=True
                )
            except Exception as e:
                # The old app simply ignores MediaFileStorage-related display issues.
                if "MediaFileStorage" not in str(type(e).__name__):
                    print(f"Display error: {e}")

        # --- Log queue (also carries detections)
        log_messages = []
        while not st.session_state.log_queue.empty():
            try:
                msg = st.session_state.log_queue.get_nowait()
                if msg.get("type") == "detections":
                    st.session_state.current_detections = msg.get("data", [])
                elif msg.get("type") == "recording_path":
                    st.session_state.recording_path = msg.get("path")
                else:
                    log_messages.append(msg)
            except queue.Empty:
                break
            except Exception:
                break

        # --- Detections + alerts
        try:
            with detections_placeholder.container():
                if st.session_state.current_detections:
                    for detection in st.session_state.current_detections:
                        auth_class = "authorized" if detection["authorization"] == "Authorized" else (
                            "partial" if detection["authorization"] == "Partially Authorized" else "unauthorized"
                        )
                        
                        # Build detection HTML with enhanced interaction display
                        behavior_info = ""
                        behavior_emoji = ""
                        if "behavior_status" in detection and detection["behavior_status"] != "STATUS: NO INTERACTION":
                            behavior_status = detection["behavior_status"].replace("STATUS: ", "")
                            
                            # Add emoji based on interaction type
                            if "CARRYING" in behavior_status:
                                behavior_emoji = "🎒"  # Backpack emoji
                                if "LAPTOP" in behavior_status:
                                    behavior_emoji = "💻"
                                elif "HANDBAG" in behavior_status:
                                    behavior_emoji = "👜"
                                elif "CELL PHONE" in behavior_status:
                                    behavior_emoji = "📱"
                            elif "INTERACTING WITH" in behavior_status:
                                behavior_emoji = "🖐️"  # Hand emoji
                                if "LAPTOP" in behavior_status:
                                    behavior_emoji = "💻"
                                elif "KEYBOARD" in behavior_status:
                                    behavior_emoji = "⌨️"
                                elif "MOUSE" in behavior_status:
                                    behavior_emoji = "🖱️"
                            
                            behavior_info = f"<br><span style='font-size: 0.95em; font-weight: 600; color: #ff6b35;'>{behavior_emoji} {behavior_status}</span>"
                        
                        # Track ID info
                        tid = detection.get("track_id", -1)
                        tid_info = f" [ID: {tid}]" if tid != -1 else ""
                        
                        st.markdown(
                            f"""
                        <div class=\"status-box {auth_class}\">
                            <strong>[{detection.get('camera', 'Primary')}] {detection['identity']}{tid_info}</strong><br>
                            {detection['authorization']}{behavior_info}
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )
                        if detection["authorization"] in ["Unauthorized", "Partially Authorized"]:
                            show_alert(detection["authorization"], detection["identity"])
                else:
                    st.info("No detections")

            alerts_html = get_active_alerts()
            if alerts_html:
                alert_placeholder.markdown(alerts_html, unsafe_allow_html=True)
            else:
                alert_placeholder.empty()
        except Exception:
            pass

        # --- Stats
        try:
            total = len(st.session_state.current_detections)
            auth_count = sum(1 for d in st.session_state.current_detections if d["authorization"] == "Authorized")
            unauth_count = sum(1 for d in st.session_state.current_detections if d["authorization"] == "Unauthorized")
            
            # Count active interactions
            interact_count = sum(1 for d in st.session_state.current_detections 
                                if d.get("behavior_status", "STATUS: NO INTERACTION") != "STATUS: NO INTERACTION")
            
            total_placeholder.metric("Total Persons", total)
            auth_placeholder.metric("Authorized", auth_count)
            unauth_placeholder.metric("Unauthorized", unauth_count)
            
            # Show interaction count if behavior detection enabled
            if interact_count > 0:
                status_placeholder.metric("🎯 Active Interactions", interact_count)
        except Exception:
            pass

        # --- Logs UI
        if log_messages:
            try:
                with log_placeholder.container():
                    for msg in log_messages[-5:]:
                        msg_type = msg.get("type", "info")
                        message = msg.get("message", "")
                        if msg_type == "error":
                            st.error(f"❌ {message}")
                        elif msg_type == "warning":
                            st.warning(f"⚠️ {message}")
                        elif msg_type == "success":
                            st.success(f"✅ {message}")
                        else:
                            st.info(f"ℹ️ {message}")
            except Exception:
                pass

        # --- Stop if worker died
        thread = st.session_state.processing_thread
        if thread and not thread.is_alive():
            st.session_state.running = False
            break

        time.sleep(float(frame_sleep_s))


def show_alert(auth_level: str, identity_name: str) -> None:
    key = f"{auth_level}:{identity_name}"
    now = time.time()
    last = st.session_state.last_alert.get(key, 0.0)
    if now - last < float(st.session_state.alert_cooldown):
        return

    if auth_level == "Unauthorized":
        msg = f"🚨 Unauthorized person detected: {identity_name}"
        typ = "unauthorized"
    else:
        msg = f"⚠️ Partially authorized person detected: {identity_name}"
        typ = "partial"

    st.session_state.active_alerts[key] = {
        "message": msg,
        "type": typ,
        "timestamp": now,
    }
    st.session_state.last_alert[key] = now


def get_active_alerts() -> str:
    now = time.time()
    duration = float(st.session_state.alert_duration)

    to_remove = []
    for k, v in st.session_state.active_alerts.items():
        if now - float(v.get("timestamp", 0)) > duration:
            to_remove.append(k)
    for k in to_remove:
        st.session_state.active_alerts.pop(k, None)

    if not st.session_state.active_alerts:
        return ""

    items = []
    for v in st.session_state.active_alerts.values():
        cls = "alert-unauthorized" if v.get("type") == "unauthorized" else "alert-partial"
        items.append(f"<div class=\"alert-notification {cls}\">{v.get('message','')}</div>")
    return f"<div class=\"alert-container\">{''.join(items)}</div>"


def format_source_label(source, source_mode: str) -> str:
    if source_mode == "webcam":
        return f"webcam_{source}"
    if source_mode == "rtsp":
        return "rtsp"
    try:
        p = Path(str(source))
        return p.stem
    except Exception:
        return "video"


def open_video_capture(source_mode: str, video_source):
    if source_mode == "webcam":
        cap = cv2.VideoCapture(int(video_source), cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap
    if source_mode == "rtsp":
        cap = cv2.VideoCapture(str(video_source))
        cap.set(cv2.CAP_PROP_BUFFERSIZE, cam_config.RTSP_BUFFER_SIZE)
        for _ in range(10):
            cap.grab()
        return cap
    return cv2.VideoCapture(str(video_source))


def video_processing_thread(video_source, config, frame_queue, log_queue, stop_flag):
    """YOLO + ByteTrack tracking + FaceNet recognition + optional behavior detection."""

    cap = None
    recorder = None
    pipeline = None
    session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Evidence saving for interactions - organized by session
    last_saved = {}  # track_id -> last save timestamp
    SAVE_COOLDOWN = 3.0  # seconds between saves per track ID
    
    # Create session-specific evidence folder
    source_mode = config.get("source_mode", "webcam")
    source_label = format_source_label(video_source, source_mode)
    evidence_dir = REPO_ROOT / "office_evidence" / f"{source_label}_{session_timestamp}"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    
    # Create date-based log folder structure: logs/MM-DD-YYYY/
    current_date = datetime.now().strftime("%m-%d-%Y")
    logs_date_dir = REPO_ROOT / "logs" / current_date
    logs_date_dir.mkdir(parents=True, exist_ok=True)
    
    # Create log file for this session in the date folder
    log_file_path = logs_date_dir / f"session_log_{session_timestamp}.txt"
    
    def write_to_log_file(message: str):
        """Write a log entry to the session log file"""
        try:
            timestamp = datetime.now().strftime("%B %d, %Y at %I:%M:%S %p")
            with open(log_file_path, 'a', encoding='utf-8') as f:
                f.write(f"[{timestamp}] {message}\n")
        except Exception:
            pass
    
    # Write session header
    write_to_log_file("="*60)
    write_to_log_file(f"CCTV Monitoring Session Started")
    write_to_log_file(f"Source: {source_label}")
    write_to_log_file(f"Evidence Folder: {evidence_dir.name}")
    write_to_log_file("="*60)
    
    # Send log file path to UI
    _safe_put(log_queue, {"type": "log_file_path", "path": str(log_file_path)})
    
    # Check if logging is enabled
    enable_logging = config.get("enable_logging", False)

    try:
        # Simplified startup - no technical logs
        # Only log critical errors or user-facing events
        
        # Import heavy dependencies inside thread
        if config.get("force_cpu"):
            import os
            os.environ["CUDA_VISIBLE_DEVICES"] = ""

        # Choose pipeline based on behavior detection setting
        enable_behavior = config.get("enable_behavior", False)
        if enable_behavior:
            from combined_yolo_facenet_behavior import CombinedYOLOFaceNetBehavior as PipelineClass
        else:
            from combined_yolo_facenet_only import CombinedYOLOFaceOnly as PipelineClass

        # Choose which FaceNet module file to load (current vs old) if provided.
        facenet_main_path = config.get("facenet_main") or (REPO_ROOT / "face_recognition" / "Facenet" / "facenet_main.py")

        # YOLO model path
        yolo_path = Path(config.get("yolo_model", ""))
        if not yolo_path.is_absolute():
            yolo_path = (REPO_ROOT / yolo_path).resolve()

        if not yolo_path.exists():
            _safe_put(log_queue, {"type": "error", "message": "❌ System Error: Camera model not found. Please contact administrator."})
            return

        # Build pipeline with behavior parameters if enabled
        pipeline_kwargs = {
            "yolo_model_path": str(yolo_path),
            "facenet_main_path": str(facenet_main_path),
            "authorization_map": AUTHORIZATION_MAP,
            "conf_threshold": float(config.get("conf_threshold", 0.45)),
            "resize_factor": float(config.get("resize_factor", 1.0)),
            "frame_skip": int(config.get("frame_skip", 1)),
            "recog_interval": int(config.get("recog_interval", 10)),
            "device": ("cpu" if bool(config.get("force_cpu")) else None),
        }

        if enable_behavior:
            pipeline_kwargs.update({
                "enable_behavior": True,
                "coverage_thresh": float(config.get("coverage_thresh", 0.18)),
                "move_px_thresh": float(config.get("move_px_thresh", 8.0)),
                "stationary_frames_required": int(config.get("stationary_frames_required", 6)),
                "status_on_frames_required": int(config.get("status_on_frames_required", 6)),
                "status_off_frames_required": int(config.get("status_off_frames_required", 12)),
                "object_hold_frames": int(config.get("object_hold_frames", 8)),
            })

        pipeline = PipelineClass(**pipeline_kwargs)

        source_mode = config.get("source_mode", "webcam")
        cap = open_video_capture(source_mode, video_source)
        if cap is None or not cap.isOpened():
            _safe_put(log_queue, {"type": "error", "message": "❌ Failed to connect to camera. Please check camera connection."})
            return

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
        fps = max(int(cap.get(cv2.CAP_PROP_FPS) or 25), 1)
        
        # No startup message - system is ready, logs will show when people are detected

        # Recording
        if bool(config.get("recording_enabled")):
            source_label = format_source_label(video_source, source_mode)
            rec_dir = REPO_ROOT / "recordings" / source_label
            rec_dir.mkdir(parents=True, exist_ok=True)
            output_path = rec_dir / f"recording_{session_timestamp}.mp4"
            writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
            if writer.isOpened():
                recorder = writer
                _safe_put(log_queue, {"type": "recording_path", "path": str(output_path)})
                # No logging for recording start - backend operation
            else:
                writer.release()
                # Silent failure for recording issues

        while not stop_flag.is_set():
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.05)
                continue

            annotated, detections = pipeline.process_frame(frame)
            _safe_put(log_queue, {"type": "detections", "data": detections})
            
            # Track current status for each person
            current_people = {}  # track_id -> {name, auth, behavior}
            
            # Log meaningful events - person detection and activities
            if enable_logging and detections:
                for detection in detections:
                    identity = detection.get("identity", "Unknown")
                    authorization = detection.get("authorization", "Unauthorized")
                    behavior = detection.get("behavior_status", "STATUS: NO INTERACTION")
                    track_id = detection.get("track_id", -1)
                    
                    # Store current status
                    current_people[track_id] = {
                        'name': identity,
                        'auth': authorization,
                        'behavior': behavior
                    }
                    
                    # Track which people we've already logged to avoid spam
                    log_key = f"person_{track_id}_{identity}"
                    
                    # First detection of this person
                    if log_key not in st.session_state.get("logged_people", set()):
                        if "logged_people" not in st.session_state:
                            st.session_state.logged_people = set()
                        st.session_state.logged_people.add(log_key)
                        
                        # Entry log
                        if identity != "Unknown":
                            auth_emoji = "✅" if authorization == "Authorized" else "⚠️" if authorization == "Partially Authorized" else "🚨"
                            msg = f"{auth_emoji} {identity} entered the room - {authorization}"
                            _safe_put(log_queue, {
                                "type": "success" if authorization == "Authorized" else "warning" if authorization == "Partially Authorized" else "error",
                                "message": msg
                            })
                            write_to_log_file(msg)
                        else:
                            msg = f"🚨 Unidentified person detected - Unauthorized"
                            _safe_put(log_queue, {
                                "type": "warning",
                                "message": msg
                            })
                            write_to_log_file(msg)
                    
                    # Log behavior changes (only significant ones)
                    if behavior != "STATUS: NO INTERACTION":
                        behavior_key = f"behavior_{track_id}_{behavior}"
                        if behavior_key not in st.session_state.get("logged_behaviors", set()):
                            if "logged_behaviors" not in st.session_state:
                                st.session_state.logged_behaviors = set()
                            st.session_state.logged_behaviors.add(behavior_key)
                            
                            # Clean behavior text
                            if "CARRYING" in behavior:
                                item = behavior.replace("STATUS: CARRYING ", "")
                                msg = f"👜 {identity} is carrying {item.lower()}"
                                _safe_put(log_queue, {
                                    "type": "info",
                                    "message": msg
                                })
                                write_to_log_file(msg)
                            elif "INTERACTING WITH" in behavior:
                                item = behavior.replace("STATUS: INTERACTING WITH ", "")
                                msg = f"💻 {identity} is interacting with {item.lower()}"
                                _safe_put(log_queue, {
                                    "type": "info",
                                    "message": msg
                                })
                                write_to_log_file(msg)
                
                # Generate status summary every few seconds
                if not hasattr(video_processing_thread, 'last_status_log'):
                    video_processing_thread.last_status_log = time.time()
                
                current_time = time.time()
                if current_time - video_processing_thread.last_status_log >= 10.0:  # Every 10 seconds
                    video_processing_thread.last_status_log = current_time
                    
                    if current_people:
                        status_lines = ["� Current Status:"]
                        for tid, info in current_people.items():
                            name = info['name']
                            behavior = info['behavior']
                            
                            if behavior == "STATUS: NO INTERACTION":
                                status_lines.append(f"  • {name} is present in the room")
                            elif "CARRYING" in behavior:
                                item = behavior.replace("STATUS: CARRYING ", "").lower()
                                status_lines.append(f"  • {name} is carrying {item}")
                            elif "INTERACTING WITH" in behavior:
                                item = behavior.replace("STATUS: INTERACTING WITH ", "").lower()
                                status_lines.append(f"  • {name} is interacting with {item}")
                        
                        status_msg = "\n".join(status_lines)
                        _safe_put(log_queue, {
                            "type": "info",
                            "message": status_msg
                        })
                        write_to_log_file(status_msg.replace("\n", " | "))
            
            # Save evidence for interactions (similar to main.py)
            if config.get("enable_behavior", False):
                current_time = time.time()
                for detection in detections:
                    tid = detection.get("track_id", -1)
                    behavior = detection.get("behavior_status", "STATUS: NO INTERACTION")
                    
                    if tid != -1 and behavior != "STATUS: NO INTERACTION":
                        last_save_time = last_saved.get(tid, 0)
                        
                        if current_time - last_save_time >= SAVE_COOLDOWN:
                            # Extract object name from behavior status
                            if "CARRYING" in behavior:
                                obj_name = behavior.replace("STATUS: CARRYING ", "").lower().replace(" ", "_")
                            elif "INTERACTING WITH" in behavior:
                                obj_name = behavior.replace("STATUS: INTERACTING WITH ", "").lower().replace(" ", "_")
                            else:
                                obj_name = "unknown"
                            
                            timestamp_str = datetime.now().strftime("%Y%m%d-%H%M%S")
                            identity = detection.get("identity", "unknown")
                            filename = evidence_dir / f"alert_{timestamp_str}_{obj_name}_{identity}_ID{tid}.jpg"
                            
                            try:
                                cv2.imwrite(str(filename), annotated)
                                last_saved[tid] = current_time
                                # No logging for evidence saving - backend operation
                            except Exception as e:
                                # Silent failure - no need to spam user with backend errors
                                pass

            if recorder is not None:
                if annotated.shape[1] != width or annotated.shape[0] != height:
                    recorder.write(cv2.resize(annotated, (width, height)))
                else:
                    recorder.write(annotated)

            try:
                while not frame_queue.empty():
                    try:
                        frame_queue.get_nowait()
                    except queue.Empty:
                        break
                frame_queue.put(annotated)
            except Exception:
                pass

            time.sleep(0.001)

        # Write session end to log file
        write_to_log_file("="*60)
        write_to_log_file("CCTV Monitoring Session Ended")
        write_to_log_file("="*60)

    except Exception as e:
        # Only log critical user-facing errors
        error_msg = "⚠️ System encountered an error. Please restart."
        _safe_put(log_queue, {"type": "error", "message": error_msg})
        write_to_log_file(f"ERROR: {error_msg}")

    finally:
        if cap is not None:
            cap.release()
            # No camera release logging - backend operation
        if recorder is not None:
            recorder.release()
            # No recording save logging - backend operation


def main():
    st.set_page_config(page_title="CCTV Security System", page_icon="🎥", layout="wide", initial_sidebar_state="expanded")
    st.markdown('<div class="main-header">🎥 CCTV Monitoring System</div>', unsafe_allow_html=True)

    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Video Source
        st.subheader("Video Source")
        source_type = st.radio("Select Source", ["Webcam", "Video File", "RTSP Camera"], key="source_type")
        
        # Initialize variables
        video_source = None
        source_mode = "webcam"
        webcam_index = 0  # Default value
        
        if source_type == "Webcam":
            webcam_index = st.number_input("Webcam Index", min_value=0, max_value=5, value=cam_config.WEBCAM_ID)
            video_source = int(webcam_index)  # Ensure it's an integer
            source_mode = "webcam"
        
        elif source_type == "Video File":
            st.info("💡 For videos > 200MB, use 'File Path' method")
            
            # Get available videos from Mp4TESTING folder
            test_videos_dir = REPO_ROOT / "Tapo"
            available_videos = []
            if test_videos_dir.exists():
                available_videos = sorted([
                    f for f in test_videos_dir.glob("*") 
                    if f.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']
                ])
            
            # Method selection
            upload_method = st.radio(
                "Select Method", 
                ["Browse Test Videos", "Upload File", "Custom Path"], 
                horizontal=True
            )
            if upload_method == "Browse Test Videos":
                if available_videos:
                    video_options = {str(v): v.name for v in available_videos}
                    selected_video = st.selectbox("Select Test Video", options=list(video_options.keys()), format_func=lambda x: video_options[x])
                    if selected_video:
                        video_source = selected_video
                        file_size_mb = Path(selected_video).stat().st_size / (1024 * 1024)
                        st.success(f"✅ {Path(selected_video).name} ({file_size_mb:.1f} MB)")
                else:
                    st.warning(f"No videos found in {test_videos_dir.relative_to(REPO_ROOT)}")
            
            elif upload_method == "Upload File":
                video_file = st.file_uploader("Upload Video (Max 200MB)", type=['mp4', 'avi', 'mov'])
                if video_file:
                    temp_path = REPO_ROOT / "temp" / video_file.name
                    temp_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(temp_path, 'wb') as f:
                        f.write(video_file.read())
                    video_source = str(temp_path)
                    file_size_mb = temp_path.stat().st_size / (1024 * 1024)
                    st.success(f"✅ Uploaded: {video_file.name} ({file_size_mb:.1f} MB)")
            
            else:  # Custom Path
                default_path = cam_config.VIDEO_FILE_PATH if Path(cam_config.VIDEO_FILE_PATH).exists() else ""
                file_path_input = st.text_input("Video File Path", value=default_path, placeholder="C:/Videos/my.mp4")
                if file_path_input:
                    file_path = Path(file_path_input)
                    if not file_path.is_absolute():
                        file_path = REPO_ROOT / file_path_input
                    if file_path.exists():
                        video_source = str(file_path)
                        file_size_mb = file_path.stat().st_size / (1024 * 1024)
                        st.success(f"✅ {file_path.name} ({file_size_mb:.1f} MB)")
                    else:
                        st.error(f"❌ File not found: {file_path}")
            
            source_mode = "video"
        
        else:  # RTSP Camera
            # Get available cameras
            rtsp_cameras = cam_config.get_all_rtsp_cameras()
            camera_options = {key: f"{key} - {name}" for key, name, enabled in rtsp_cameras if enabled}
            if camera_options:
                selected_camera = st.selectbox("Select RTSP Camera", options=list(camera_options.keys()), format_func=lambda x: camera_options[x])
                video_source = cam_config.get_rtsp_url(selected_camera)
                camera_info = cam_config.RTSP_CAMERAS[selected_camera]
                st.text(f"IP: {camera_info['ip']}")
                st.text(f"Stream: {camera_info['stream']}")
                source_mode = "rtsp"
            else:
                st.error("No RTSP cameras configured or enabled")
                video_source = None
            
            source_mode = "rtsp"
        
        st.divider()
        
        # Model Configuration
        st.subheader("Model Settings")
        use_gpu = st.checkbox("Use GPU", value=False)
        device = "cuda" if use_gpu else "cpu"
        
        yolo_model = st.text_input("YOLO Model", value="models/YOLOv8/yolov8n.pt")
        facenet_main = st.text_input("FaceNet Main", value=str(REPO_ROOT / "face_recognition" / "Facenet" / "facenet_main.py"))
        
        st.divider()
        
        # Behavior Detection Settings
        st.subheader("🎯 Behavior Detection")
        enable_behavior = st.checkbox("Enable HOI Detection", value=False,
                                     help="Detect human-object interactions (CARRYING/INTERACTING)")
        
        if enable_behavior:
            with st.expander("Behavior Settings"):
                coverage_thresh = st.slider("Coverage Threshold", 0.05, 0.5, 0.18, 
                                           help="IoA threshold for object interaction")
                move_px_thresh = st.slider("Movement Threshold (px)", 1.0, 20.0, 8.0,
                                          help="Pixels/frame to detect movement")
                stationary_frames = st.slider("Stationary Frames", 3, 15, 6,
                                             help="Frames to confirm stationary")
                status_on_frames = st.slider("Status ON Frames", 3, 15, 6,
                                            help="Frames to confirm new status")
                status_off_frames = st.slider("Status OFF Frames", 6, 30, 12,
                                             help="Frames to clear status")
                object_hold_frames = st.slider("Object Hold Frames", 3, 20, 8,
                                              help="Frames to cache objects")
            
            # Evidence saving info
            st.info("💾 Interaction evidence auto-saved to session-specific folders in: `office_evidence/`")
        else:
            coverage_thresh = 0.18
            move_px_thresh = 8.0
            stationary_frames = 6
            status_on_frames = 6
            status_off_frames = 12
            object_hold_frames = 8
        
        st.divider()
        
        # Performance Settings
        st.subheader("Performance")
        frame_skip = st.slider("Frame Skip", 1, 10, 1)
        resize_factor = st.slider("Resize", 0.1, 1.0, 1.0)
        recog_interval = st.slider("Recognition Interval", 10, 60, 30)
        
        st.divider()
        
        # Recording Settings
        st.subheader("📹 Recording")
        enable_recording = st.checkbox("Enable Recording", value=True, 
                                       help="Save annotated video with all detections")
        
        if enable_recording:
            st.info("📁 Recordings saved to: `recordings/`")
            if st.session_state.recording_path:
                st.info(f"Last: {Path(st.session_state.recording_path).name}")
        
        st.divider()
        
        enable_logging = st.checkbox("Enable Detailed Logging", value=True, 
                                     help="Generate detailed system logs (initialization, frame processing, file saves, etc.)")
        st.session_state.enable_logging = enable_logging
        st.session_state.alert_sounds_enabled = st.checkbox("Enable Alert Sounds", value=True)
        
        st.divider()
        
        # Control Buttons
        col1, col2 = st.columns(2)
        with col1:
            start_button = st.button("▶️ Start", use_container_width=True, type="primary", disabled=st.session_state.running)
        with col2:
            stop_button = st.button("⏹️ Stop", use_container_width=True, disabled=not st.session_state.running)
    
    # Handle start/stop
    if start_button and not st.session_state.running:
        if video_source is not None:
            st.session_state.running = True
            st.session_state.stop_flag.clear()
            st.session_state.recording_enabled = enable_recording
            
            config = {
                'yolo_model': yolo_model,
                'facenet_main': facenet_main,
                'device': device,
                'recog_interval': recog_interval,
                'frame_skip': frame_skip,
                'resize_factor': resize_factor,
                'enable_logging': enable_logging,
                'webcam_index': int(webcam_index) if 'webcam_index' in locals() else 0,
                'conf_threshold': 0.5,
                'iou_threshold': 0.7,
                'source_mode': source_mode,
                'recording_enabled': enable_recording,
                'enable_behavior': enable_behavior,
                'coverage_thresh': coverage_thresh,
                'move_px_thresh': move_px_thresh,
                'stationary_frames_required': stationary_frames,
                'status_on_frames_required': status_on_frames,
                'status_off_frames_required': status_off_frames,
                'object_hold_frames': object_hold_frames,
            }
            
            # Clear queues
            while not st.session_state.frame_queue.empty():
                try:
                    st.session_state.frame_queue.get_nowait()
                except:
                    break
            
            while not st.session_state.log_queue.empty():
                try:
                    st.session_state.log_queue.get_nowait()
                except:
                    break
            
            # Clear tracking sets for new session
            st.session_state.logged_people = set()
            st.session_state.logged_behaviors = set()
            
            # Start processing thread
            thread = threading.Thread(
                target=video_processing_thread,
                args=(video_source, config, st.session_state.frame_queue, st.session_state.log_queue, st.session_state.stop_flag),
                daemon=True
            )
            thread.start()
            st.session_state.processing_thread = thread
            
            time.sleep(0.1)
            st.rerun()
        else:
            st.error("❌ Select a video source first")
    
    if stop_button and st.session_state.running:
        st.session_state.running = False
        st.session_state.stop_flag.set()
        time.sleep(0.1)
        st.rerun()

    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📹 Live Feed")
        video_placeholder = st.empty()
        status_placeholder = st.empty()
        alert_placeholder = st.empty()
    
    with col2:
        st.subheader("👥 Detections")
        detections_placeholder = st.empty()
        
        st.subheader("📊 Stats")
        stats_cols = st.columns(3)
        with stats_cols[0]:
            total_placeholder = st.empty()
        with stats_cols[1]:
            auth_placeholder = st.empty()
        with stats_cols[2]:
            unauth_placeholder = st.empty()

    # Always show log section in UI
    st.subheader("📋 System Log")
    
    # Add "View All Logs" and "Open Logs Folder" buttons
    col_log1, col_log2, col_log3 = st.columns([2, 1, 1])
    with col_log2:
        if st.button("📜 View All Logs", use_container_width=True):
            st.session_state.show_all_logs = not st.session_state.get("show_all_logs", False)
    with col_log3:
        if st.button("📂 Open Logs Folder", use_container_width=True):
            # Open the date-based logs folder
            current_date = datetime.now().strftime("%m-%d-%Y")
            logs_date_dir = REPO_ROOT / "logs" / current_date
            
            # Create folder if it doesn't exist
            if not logs_date_dir.exists():
                logs_date_dir.mkdir(parents=True, exist_ok=True)
            
            # Open folder in file explorer
            import subprocess
            import platform
            
            try:
                if platform.system() == "Windows":
                    subprocess.run(['explorer', str(logs_date_dir)])
                elif platform.system() == "Darwin":  # macOS
                    subprocess.run(['open', str(logs_date_dir)])
                else:  # Linux
                    subprocess.run(['xdg-open', str(logs_date_dir)])
            except Exception:
                pass
    
    log_placeholder = st.empty()
    
    # Expandable section for all logs
    if st.session_state.get("show_all_logs", False):
        with st.expander("📋 Complete Log History", expanded=True):
            if st.session_state.all_logs:
                # Show log file info if available
                if st.session_state.session_log_file and Path(st.session_state.session_log_file).exists():
                    log_file = Path(st.session_state.session_log_file)
                    st.info(f"📄 Session log file: `{log_file.name}`")
                    
                    # Download button for log file
                    try:
                        with open(log_file, 'r', encoding='utf-8') as f:
                            log_content = f.read()
                        st.download_button(
                            label="💾 Download Log File",
                            data=log_content,
                            file_name=log_file.name,
                            mime="text/plain",
                            use_container_width=True
                        )
                    except Exception:
                        pass
                
                st.divider()
                
                # Show all logs with timestamps
                for log_entry in st.session_state.all_logs[-100:]:  # Show last 100 logs
                    msg_type = log_entry.get('type', 'info')
                    message = log_entry.get('message', '')
                    timestamp = log_entry.get('timestamp', '')
                    
                    if msg_type == 'error':
                        st.error(f"[{timestamp}] ❌ {message}")
                    elif msg_type == 'warning':
                        st.warning(f"[{timestamp}] ⚠️ {message}")
                    elif msg_type == 'success':
                        st.success(f"[{timestamp}] ✅ {message}")
                    else:
                        st.info(f"[{timestamp}] ℹ️ {message}")
                
                # Clear logs button
                if st.button("🗑️ Clear Log History"):
                    st.session_state.all_logs = []
                    st.rerun()
            else:
                st.info("No logs yet. Start the system to see logs.")
    
    # Display loop
    if st.session_state.running:
        status_placeholder.success("🟢 Running")
        
        # Show recording status
        if st.session_state.recording_enabled:
            status_placeholder.success("🟢 Running | 🔴 Recording")
        
        while st.session_state.running:
            # Get frame
            try:
                if not st.session_state.frame_queue.empty():
                    frame = st.session_state.frame_queue.get_nowait()
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    video_placeholder.image(frame_rgb, channels="RGB", width=640)
                else:
                    time.sleep(0.01)
                    continue
            except queue.Empty:
                time.sleep(0.01)
                continue
            except Exception as e:
                if "MediaFileStorageError" not in str(type(e).__name__):
                    print(f"Display error: {e}")
                time.sleep(0.01)
                continue
            
            # Update detections
            try:
                with detections_placeholder.container():
                    if st.session_state.current_detections:
                        for detection in st.session_state.current_detections:
                            auth_class = "authorized" if detection['authorization'] == "Authorized" else \
                                        "partial" if detection['authorization'] == "Partially Authorized" else "unauthorized"
                            
                            st.markdown(f"""
                            <div class="status-box {auth_class}">
                                <strong>[{detection.get('camera', 'Primary')}] {detection['identity']}</strong><br>
                                {detection['authorization']}
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Generate alerts for unauthorized/partial
                            if detection['authorization'] in ["Unauthorized", "Partially Authorized"]:
                                show_alert(detection['authorization'], detection['identity'])
                    else:
                        st.info("No detections")
                
                # Display all active alerts (with expiration)
                active_alerts_html = get_active_alerts()
                if active_alerts_html:
                    alert_placeholder.markdown(active_alerts_html, unsafe_allow_html=True)
                else:
                    alert_placeholder.empty()
                    
            except:
                pass
            
            # Update stats
            try:
                total = len(st.session_state.current_detections)
                auth = sum(1 for d in st.session_state.current_detections if d['authorization'] == "Authorized")
                unauth = sum(1 for d in st.session_state.current_detections if d['authorization'] == "Unauthorized")
                
                total_placeholder.metric("Total", total)
                auth_placeholder.metric("Auth", auth)
                unauth_placeholder.metric("Unauth", unauth)
            except:
                pass
            
            # Update logs
            log_messages = []
            while not st.session_state.log_queue.empty():
                try:
                    msg = st.session_state.log_queue.get_nowait()
                    if msg.get('type') == 'detections':
                        st.session_state.current_detections = msg.get('data', [])
                    elif msg.get('type') == 'recording_path':
                        st.session_state.recording_path = msg.get('path')
                    elif msg.get('type') == 'log_file_path':
                        st.session_state.session_log_file = msg.get('path')
                    else:
                        # Add timestamp if not present
                        if 'timestamp' not in msg:
                            msg['timestamp'] = datetime.now().strftime("%H:%M:%S")
                        
                        # Store in all_logs for history
                        st.session_state.all_logs.append(msg)
                        log_messages.append(msg)
                except queue.Empty:
                    break
            
            # Always display logs in UI (even if logging is disabled in backend)
            if log_messages:
                try:
                    with log_placeholder.container():
                        for msg in log_messages[-5:]:
                            msg_type = msg.get('type', 'info')
                            message = msg.get('message', '')
                            timestamp = msg.get('timestamp', '')
                            
                            if msg_type == 'error':
                                st.error(f"[{timestamp}] ❌ {message}")
                            elif msg_type == 'warning':
                                st.warning(f"[{timestamp}] ⚠️ {message}")
                            elif msg_type == 'success':
                                st.success(f"[{timestamp}] ✅ {message}")
                            else:
                                st.info(f"[{timestamp}] ℹ️ {message}")
                except:
                    pass
            
            time.sleep(0.01)
            
            if not st.session_state.running:
                break
    else:
        status_placeholder.warning("🔴 Stopped")
        video_placeholder.empty()
        
        # Show download button for last recording
        if st.session_state.recording_path and Path(st.session_state.recording_path).exists():
            st.divider()
            st.subheader("📥 Last Recording")
            
            recording_file = Path(st.session_state.recording_path)
            file_size_mb = recording_file.stat().st_size / (1024 * 1024)
            
            st.info(f"**{recording_file.name}** ({file_size_mb:.1f} MB)")
            
            with open(recording_file, 'rb') as f:
                st.download_button(
                    label="⬇️ Download Recording",
                    data=f.read(),
                    file_name=recording_file.name,
                    mime="video/mp4",
                    use_container_width=True
                )

if __name__ == "__main__":
    main()