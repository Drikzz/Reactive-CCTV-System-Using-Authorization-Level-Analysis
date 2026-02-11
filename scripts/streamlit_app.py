import sys
import threading
import queue
import time
import logging
import traceback
from pathlib import Path
from datetime import datetime
from collections import deque

import cv2
import numpy as np
import streamlit as st
from streamlit.runtime.scriptrunner import RerunData, RerunException

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from face_recognition.Facenet import facenet_main
import camera_config_streamlit as cam_config

logging.getLogger('streamlit.web.server.media_file_handler').setLevel(logging.ERROR)

st.markdown("""
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
    .authorized strong {
        color: #0d4017 !important;
    }
    .partial {
        background-color: #fff3cd;
        border: 2px solid #ffc107;
        color: #856404 !important;
    }
    .partial strong {
        color: #533f03 !important;
    }
    .unauthorized {
        background-color: #f8d7da;
        border: 2px solid #dc3545;
        color: #721c24 !important;
    }
    .unauthorized strong {
        color: #491217 !important;
    }
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
        from {
            transform: translateX(400px);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
</style>
""", unsafe_allow_html=True)

if 'running' not in st.session_state:
    st.session_state.running = False
if 'frame_queue' not in st.session_state:
    st.session_state.frame_queue = queue.Queue(maxsize=2)
if 'log_queue' not in st.session_state:
    st.session_state.log_queue = queue.Queue()
if 'current_detections' not in st.session_state:
    st.session_state.current_detections = []
if 'processing_thread' not in st.session_state:
    st.session_state.processing_thread = None
if 'stop_flag' not in st.session_state:
    st.session_state.stop_flag = threading.Event()
if 'last_alert' not in st.session_state:
    st.session_state.last_alert = {}
if 'alert_cooldown' not in st.session_state:
    st.session_state.alert_cooldown = 5
if 'active_alerts' not in st.session_state:
    st.session_state.active_alerts = {}
if 'alert_duration' not in st.session_state:
    st.session_state.alert_duration = 10
if 'alert_sounds_enabled' not in st.session_state:
    st.session_state.alert_sounds_enabled = True
if 'last_sound_played' not in st.session_state:
    st.session_state.last_sound_played = {}
if 'sound_cooldown' not in st.session_state:
    st.session_state.sound_cooldown = 3
if 'recording_enabled' not in st.session_state:
    st.session_state.recording_enabled = False
if 'recording_path' not in st.session_state:
    st.session_state.recording_path = None

AUTHORIZATION_MAP = {
    "myke": "Partially Authorized",
    "dean": "Authorized",
    "art": "Partially Authorized",
    "aldrikz": "Partially Authorized"
}

def get_authorization_level(identity_name):
    if not identity_name or identity_name == "Unknown":
        return "Unauthorized"
    return AUTHORIZATION_MAP.get(identity_name.lower(), "Partially Authorized")

def get_authorization_color(auth_level):
    color_map = {
        "Authorized": (0, 255, 0),
        "Partially Authorized": (0, 165, 255),
        "Unauthorized": (0, 0, 255)
    }
    return color_map.get(auth_level, (128, 128, 128))

def play_alert_sound(sound_type="unauthorized"):
    current_time = time.time()
    if sound_type in st.session_state.last_sound_played:
        if current_time - st.session_state.last_sound_played[sound_type] < st.session_state.sound_cooldown:
            return ""
    st.session_state.last_sound_played[sound_type] = current_time
    if sound_type == "unauthorized":
        return """
        <script>
        (function() {
            if (!window.audioContext) {
                window.audioContext = new (window.AudioContext || window.webkitAudioContext)();
            }
            var ctx = window.audioContext;
            var beepCount = 0;
            function beep() {
                if (beepCount >= 3) return;
                var oscillator = ctx.createOscillator();
                var gainNode = ctx.createGain();
                oscillator.connect(gainNode);
                gainNode.connect(ctx.destination);
                gainNode.gain.value = 0.3;
                oscillator.frequency.value = 1000;
                oscillator.type = 'sine';
                oscillator.start(ctx.currentTime);
                oscillator.stop(ctx.currentTime + 0.15);
                beepCount++;
                if (beepCount < 3) {
                    setTimeout(beep, 200);
                }
            }
            beep();
        })();
        </script>
        """
    return """
        <script>
        (function() {
            if (!window.audioContext) {
                window.audioContext = new (window.AudioContext || window.webkitAudioContext)();
            }
            var ctx = window.audioContext;
            var beepCount = 0;
            function beep() {
                if (beepCount >= 2) return;
                var oscillator = ctx.createOscillator();
                var gainNode = ctx.createGain();
                oscillator.connect(gainNode);
                gainNode.connect(ctx.destination);
                gainNode.gain.value = 0.3;
                oscillator.frequency.value = 600;
                oscillator.type = 'sine';
                oscillator.start(ctx.currentTime);
                oscillator.stop(ctx.currentTime + 0.2);
                beepCount++;
                if (beepCount < 2) {
                    setTimeout(beep, 300);
                }
            }
            beep();
        })();
        </script>
        """

def show_alert(auth_level, identity):
    current_time = time.time()
    alert_key = f"{identity}_{auth_level}"
    if alert_key in st.session_state.last_alert:
        if current_time - st.session_state.last_alert[alert_key] < st.session_state.alert_cooldown:
            return
    st.session_state.last_alert[alert_key] = current_time
    if auth_level == "Unauthorized":
        alert_class = "alert-unauthorized"
        icon = "🚨"
        message = f"UNAUTHORIZED ACCESS: {identity}"
        sound_type = "unauthorized"
    elif auth_level == "Partially Authorized":
        alert_class = "alert-partial"
        icon = "⚠️"
        message = f"RESTRICTED ACCESS: {identity}"
        sound_type = "partial"
    else:
        return
    st.session_state.active_alerts[alert_key] = {
        'html': f"""
        <div class=\"alert-notification {alert_class}\">
            <strong>{icon} ALERT</strong><br>
            {message}
        </div>
        """,
        'timestamp': current_time,
        'sound_type': sound_type
    }

def get_active_alerts():
    current_time = time.time()
    active = []
    expired = []
    sound_html = ""
    for key, alert_data in st.session_state.active_alerts.items():
        age = current_time - alert_data['timestamp']
        if age < st.session_state.alert_duration:
            active.append(alert_data['html'])
            if age < 0.5 and st.session_state.alert_sounds_enabled:
                sound_html = play_alert_sound(alert_data.get('sound_type', 'unauthorized'))
        else:
            expired.append(key)
    for key in expired:
        del st.session_state.active_alerts[key]
    if active:
        return f"""
        {sound_html}
        <div class=\"alert-container\">
            {''.join(active)}
        </div>
        """
    return ""

def format_source_label(video_source, source_mode):
    if source_mode == "webcam":
        return f"webcam{video_source}"
    if source_mode == "rtsp":
        return "rtsp"
    try:
        return Path(video_source).stem
    except Exception:
        return "video"

def open_video_capture(source_mode, video_source):
    if source_mode == "webcam":
        return cv2.VideoCapture(int(video_source), cv2.CAP_DSHOW)
    if source_mode == "rtsp":
        cap = cv2.VideoCapture(str(video_source))
        cap.set(cv2.CAP_PROP_BUFFERSIZE, cam_config.RTSP_BUFFER_SIZE)
        return cap
    return cv2.VideoCapture(str(video_source))

def video_processing_thread(video_source, config, frame_queue, log_queue, stop_flag):
    cap = None
    recorder = None
    detector = None
    session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    try:
        log_queue.put({"type": "info", "message": "Starting FaceNet pipeline..."})
        source_label = format_source_label(video_source, config['source_mode'])
        cap = open_video_capture(config['source_mode'], video_source)
        if cap is None or not cap.isOpened():
            log_queue.put({"type": "error", "message": "Failed to open video source"})
            return
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
        fps = max(int(cap.get(cv2.CAP_PROP_FPS) or 25), 1)
        log_queue.put({"type": "info", "message": f"Source ready: {width}x{height} @ {fps}fps"})
        yolo_path = Path(config.get('yolo_model', facenet_main.YOLO_MODEL_PATH))
        if not yolo_path.exists():
            log_queue.put({"type": "warning", "message": f"YOLO model not found at {yolo_path}. Falling back to default."})
            yolo_path = Path(facenet_main.YOLO_MODEL_PATH)
        detector = facenet_main.YOLO(str(yolo_path))
        log_queue.put({"type": "success", "message": f"Loaded YOLO: {yolo_path.name}"})

        frame_skip = max(int(config.get('frame_skip', 1)), 1)
        resize_factor = float(config.get('resize_factor', 1.0))
        conf_threshold = float(config.get('conf_threshold', 0.45))
        recording_enabled = bool(config.get('recording_enabled', False))
        fps_window = deque(maxlen=30)
        frame_idx = 0

        while not stop_flag.is_set():
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)
                continue
            frame_idx += 1
            if frame_idx % frame_skip != 0:
                continue
            start_time = time.time()
            if resize_factor < 1.0:
                proc_w = max(1, int(frame.shape[1] * resize_factor))
                proc_h = max(1, int(frame.shape[0] * resize_factor))
                proc_frame = cv2.resize(frame, (proc_w, proc_h), interpolation=cv2.INTER_LINEAR)
                scale_x = frame.shape[1] / proc_w
                scale_y = frame.shape[0] / proc_h
            else:
                proc_frame = frame
                scale_x = scale_y = 1.0

            results = detector(proc_frame, conf=conf_threshold, classes=[0], verbose=False)
            annotated = frame.copy()
            tracks_data = []

            if results and results[0].boxes is not None:
                boxes = results[0].boxes
                for idx in range(len(boxes)):
                    if boxes.cls is not None and int(boxes.cls[idx]) != 0:
                        continue
                    x1 = int(round(float(boxes.xyxy[idx][0]) * scale_x))
                    y1 = int(round(float(boxes.xyxy[idx][1]) * scale_y))
                    x2 = int(round(float(boxes.xyxy[idx][2]) * scale_x))
                    y2 = int(round(float(boxes.xyxy[idx][3]) * scale_y))
                    x1 = max(0, min(frame.shape[1] - 1, x1))
                    y1 = max(0, min(frame.shape[0] - 1, y1))
                    x2 = max(x1 + 1, min(frame.shape[1], x2))
                    y2 = max(y1 + 1, min(frame.shape[0], y2))
                    person_crop = frame[y1:y2, x1:x2]
                    if person_crop.size == 0:
                        continue
                    identity_name = "Unknown"
                    identity_conf = 0.0
                    try:
                        face_result = facenet_main.recognize_face_in_crop(person_crop, frame, (x1, y1, x2, y2))
                        if face_result:
                            identity_name = face_result.get('name', 'Unknown') or 'Unknown'
                            identity_conf = float(face_result.get('confidence', 0.0) or 0.0)
                    except Exception as exc:
                        log_queue.put({"type": "warning", "message": f"Face recognition error: {str(exc)}"})
                    auth_level = get_authorization_level(identity_name)
                    color = get_authorization_color(auth_level)
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                    label = f"{identity_name} ({identity_conf:.2f})"
                    cv2.putText(annotated, label, (x1 + 2, max(20, y1 - 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.putText(annotated, auth_level, (x1 + 2, y2 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
                    tracks_data.append({
                        "identity": identity_name,
                        "identity_conf": identity_conf,
                        "authorization": auth_level,
                        "bbox": (x1, y1, x2, y2),
                        "camera": "Primary",
                        "timestamp": datetime.now().strftime("%H:%M:%S")
                    })

            elapsed = time.time() - start_time
            if elapsed > 0:
                fps_window.append(1.0 / elapsed)
            if fps_window:
                avg_fps = sum(fps_window) / len(fps_window)
                cv2.putText(annotated, f"FPS: {avg_fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if recording_enabled and recorder is None:
                rec_dir = REPO_ROOT / "recordings" / source_label
                rec_dir.mkdir(parents=True, exist_ok=True)
                output_path = rec_dir / f"recording_{session_timestamp}.mp4"
                writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
                if writer.isOpened():
                    recorder = writer
                    log_queue.put({"type": "recording_path", "path": str(output_path)})
                else:
                    log_queue.put({"type": "error", "message": "Failed to start recording"})
                    writer.release()
                    recorder = None

            if recorder is not None:
                if annotated.shape[1] != width or annotated.shape[0] != height:
                    recorder.write(cv2.resize(annotated, (width, height)))
                else:
                    recorder.write(annotated)

            log_queue.put({"type": "detections", "data": tracks_data})

            try:
                while not frame_queue.empty():
                    try:
                        frame_queue.get_nowait()
                    except queue.Empty:
                        break
                frame_queue.put(annotated.copy())
            except Exception as exc:
                log_queue.put({"type": "error", "message": f"Frame queue error: {str(exc)}"})

            time.sleep(0.001)

    except Exception as exc:
        tb = traceback.format_exc()
        log_queue.put({"type": "error", "message": f"Pipeline error: {str(exc)}\n{tb}"})
    finally:
        if cap is not None:
            cap.release()
        if recorder is not None:
            recorder.release()
            log_queue.put({"type": "success", "message": "Recording saved"})
        log_queue.put({"type": "info", "message": "FaceNet processing stopped"})

def request_rerun():
    raise RerunException(RerunData())


def main():
    st.set_page_config(page_title="CCTV Security System", page_icon="🎥", layout="wide", initial_sidebar_state="expanded")
    st.markdown('<div class="main-header">🎥 CCTV Monitoring System</div>', unsafe_allow_html=True)

    with st.sidebar:
        st.header("⚙️ Configuration")
        source_type = st.radio("Select Source", ["Webcam", "Video File", "RTSP Camera"], key="source_type")
        video_source = None
        source_mode = "webcam"


        if source_type == "Webcam":
            webcam_index = st.number_input("Webcam Index", min_value=0, max_value=5, value=cam_config.WEBCAM_ID)
            video_source = int(webcam_index)
            source_mode = "webcam"
        elif source_type == "Video File":
            st.info("💡 Choose from tracked recordings or upload a video file")
            test_videos_dir = REPO_ROOT / "Mp4TESTING"
            available_videos = []
            if test_videos_dir.exists():
                available_videos = sorted([f for f in test_videos_dir.glob("*") if f.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']])
            upload_method = st.radio("Select Method", ["Browse Test Videos", "Upload File", "Custom Path"], horizontal=True)
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
            else:
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
        else:
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

        st.divider()
        st.subheader("Model & Detection")
        yolo_model = st.text_input("YOLO Model", value=str(facenet_main.YOLO_MODEL_PATH))
        frame_skip = st.slider("Frame Skip", 1, 10, 1)
        resize_factor = st.slider("Resize Factor", 0.3, 1.0, 0.75)
        conf_threshold = st.slider("Detection Confidence", 0.1, 0.9, 0.45)

        st.divider()
        enable_recording = st.checkbox("Enable Recording", value=True)
        st.info("Recordings are saved under recordings/<source>/recording_<timestamp>.mp4")
        st.session_state.alert_sounds_enabled = st.checkbox("Enable Alert Sounds", value=True)

        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            start_button = st.button("▶️ Start", use_container_width=True, disabled=st.session_state.running)
        with col2:
            stop_button = st.button("⏹️ Stop", use_container_width=True, disabled=not st.session_state.running)

    if start_button and not st.session_state.running:
        if video_source is not None:
            st.session_state.running = True
            st.session_state.stop_flag.clear()
            st.session_state.recording_enabled = enable_recording
            st.session_state.recording_path = None
            st.session_state.current_detections = []
            while not st.session_state.frame_queue.empty():
                try:
                    st.session_state.frame_queue.get_nowait()
                except queue.Empty:
                    break
            while not st.session_state.log_queue.empty():
                try:
                    st.session_state.log_queue.get_nowait()
                except queue.Empty:
                    break
            config = {
                'yolo_model': yolo_model,
                'frame_skip': frame_skip,
                'resize_factor': resize_factor,
                'conf_threshold': conf_threshold,
                'recording_enabled': enable_recording,
                'source_mode': source_mode
            }
            thread = threading.Thread(
                target=video_processing_thread,
                args=(video_source, config, st.session_state.frame_queue, st.session_state.log_queue, st.session_state.stop_flag),
                daemon=True
            )
            thread.start()
            st.session_state.processing_thread = thread
            time.sleep(0.1)
        else:
            st.error("❌ Select a valid video source first")

    if stop_button and st.session_state.running:
        st.session_state.running = False
        st.session_state.stop_flag.set()
        time.sleep(0.1)

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
        total_col, auth_col, unauth_col = st.columns(3)
        total_placeholder = total_col.empty()
        auth_placeholder = auth_col.empty()
        unauth_placeholder = unauth_col.empty()

    st.subheader("📋 System Log")
    log_placeholder = st.empty()

    if st.session_state.running:
        status_placeholder.success("🟢 Running")
        if st.session_state.recording_enabled:
            status_placeholder.info("🟢 Recording")

        frame = None
        try:
            if not st.session_state.frame_queue.empty():
                frame = st.session_state.frame_queue.get_nowait()
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(frame_rgb, channels="RGB", width=640)
        except queue.Empty:
            pass

        log_messages = []
        while not st.session_state.log_queue.empty():
            try:
                msg = st.session_state.log_queue.get_nowait()
                if msg.get('type') == 'detections':
                    st.session_state.current_detections = msg.get('data', [])
                elif msg.get('type') == 'recording_path':
                    st.session_state.recording_path = msg.get('path')
                else:
                    log_messages.append(msg)
            except queue.Empty:
                break

        if log_messages:
            with log_placeholder.container():
                for msg in log_messages[-5:]:
                    msg_type = msg.get('type', 'info')
                    message = msg.get('message', '')
                    if msg_type == 'error':
                        st.error(f"❌ {message}")
                    elif msg_type == 'warning':
                        st.warning(f"⚠️ {message}")
                    elif msg_type == 'success':
                        st.success(f"✅ {message}")
                    else:
                        st.info(f"ℹ️ {message}")

        if st.session_state.current_detections:
            with detections_placeholder.container():
                for detection in st.session_state.current_detections:
                    auth_class = "authorized" if detection['authorization'] == "Authorized" else \
                        "partial" if detection['authorization'] == "Partially Authorized" else "unauthorized"
                    st.markdown(f"""
                    <div class=\"status-box {auth_class}\">
                        <strong>[{detection.get('camera', 'Primary')}] {detection['identity']}</strong><br>
                        {detection['authorization']}
                    </div>
                    """, unsafe_allow_html=True)
                    if detection['authorization'] in ["Unauthorized", "Partially Authorized"]:
                        show_alert(detection['authorization'], detection['identity'])
        else:
            detections_placeholder.info("No detections")

        total = len(st.session_state.current_detections)
        auth_count = sum(1 for d in st.session_state.current_detections if d['authorization'] == "Authorized")
        unauth_count = sum(1 for d in st.session_state.current_detections if d['authorization'] == "Unauthorized")
        total_placeholder.metric("Total", total)
        auth_placeholder.metric("Auth", auth_count)
        unauth_placeholder.metric("Unauth", unauth_count)

        alerts_html = get_active_alerts()
        if alerts_html:
            alert_placeholder.markdown(alerts_html, unsafe_allow_html=True)
        else:
            alert_placeholder.empty()

        thread = st.session_state.processing_thread
        if thread and not thread.is_alive():
            st.session_state.running = False
            request_rerun()
        elif not st.session_state.stop_flag.is_set():
            request_rerun()
    else:
        status_placeholder.warning("🔴 Stopped")
        video_placeholder.empty()
        if st.session_state.recording_path and Path(st.session_state.recording_path).exists():
            st.divider()
            st.subheader("📥 Last Recording")
            rec_file = Path(st.session_state.recording_path)
            file_size_mb = rec_file.stat().st_size / (1024 * 1024)
            st.info(f"**{rec_file.name}** ({file_size_mb:.1f} MB)")
            with open(rec_file, 'rb') as f:
                st.download_button("⬇️ Download Recording", data=f.read(), file_name=rec_file.name, mime="video/mp4", use_container_width=True)

if __name__ == "__main__":
    main()