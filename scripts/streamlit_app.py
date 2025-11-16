import streamlit as st
import cv2
import numpy as np
from pathlib import Path
import sys
import threading
import queue
from datetime import datetime
import time
import warnings
import logging

# Suppress Streamlit media file warnings
logging.getLogger('streamlit.web.server.media_file_handler').setLevel(logging.ERROR)
warnings.filterwarnings('ignore', category=UserWarning, module='streamlit')

# Add parent directory to path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from combined_yolo_facenet_mnv3 import CombinedYOLOFaceBehavior
import camera_config_streamlit as cam_config

# Page configuration
st.set_page_config(
    page_title="CCTV Security System",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
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
    .log-entry {
        padding: 0.5rem;
        margin: 0.3rem 0;
        border-left: 3px solid #007bff;
        background-color: #f8f9fa;
    }
    
    /* FIXED: Stats display with high contrast */
    div[data-testid="stMetric"] {
        background-color: #ffffff !important;
        padding: 1rem !important;
        border-radius: 0.5rem !important;
        margin: 0.5rem 0 !important;
        border: 2px solid #1f77b4 !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
    }
    div[data-testid="stMetric"] label {
        font-size: 0.9rem !important;
        font-weight: 700 !important;
        color: #1f77b4 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        font-weight: 800 !important;
        color: #000000 !important;
    }
    div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
        font-size: 0.85rem !important;
        font-weight: 600 !important;
    }
    
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
    .alert-info {
        background-color: #17a2b8;
        color: white;
        border: 2px solid #117a8b;
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

# Initialize session state
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
# ✅ NEW: Recording state
if 'recording_enabled' not in st.session_state:
    st.session_state.recording_enabled = False
if 'recording_path' not in st.session_state:
    st.session_state.recording_path = None
if 'video_writer' not in st.session_state:
    st.session_state.video_writer = None

def play_alert_sound(sound_type="unauthorized"):
    """Generate JavaScript to play alert sound"""
    current_time = time.time()
    
    # Check sound cooldown
    if sound_type in st.session_state.last_sound_played:
        if current_time - st.session_state.last_sound_played[sound_type] < st.session_state.sound_cooldown:
            return ""
    
    st.session_state.last_sound_played[sound_type] = current_time
    
    if sound_type == "unauthorized":
        # High-pitched urgent beeps (frequency: 1000Hz, 3 beeps)
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
    else:  # partial
        # Medium-pitched warning beeps (frequency: 600Hz, 2 beeps)
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

def show_alert(auth_level, identity, behavior=None):
    """Show alert notification for unauthorized/partial authorized persons"""
    current_time = time.time()
    alert_key = f"{identity}_{auth_level}"
    
    # Check cooldown for NEW alerts
    if alert_key in st.session_state.last_alert:
        if current_time - st.session_state.last_alert[alert_key] < st.session_state.alert_cooldown:
            return
    
    # Create new alert
    st.session_state.last_alert[alert_key] = current_time
    
    if auth_level == "Unauthorized":
        alert_class = "alert-unauthorized"
        icon = "🚨"
        message = f"UNAUTHORIZED ACCESS: {identity}"
        sound_type = "unauthorized"
    elif auth_level == "Partially Authorized":
        alert_class = "alert-partial"
        icon = "⚠️"
        behavior_text = f" - {behavior}" if behavior and behavior != "N/A" else ""
        message = f"RESTRICTED ACCESS: {identity}{behavior_text}"
        sound_type = "partial"
    else:
        return
    
    # Add to active alerts
    st.session_state.active_alerts[alert_key] = {
        'html': f"""
        <div class="alert-notification {alert_class}">
            <strong>{icon} ALERT</strong><br>
            {message}
        </div>
        """,
        'timestamp': current_time,
        'sound_type': sound_type
    }

def get_active_alerts():
    """Get currently active alerts (not expired) wrapped in container"""
    current_time = time.time()
    active = []
    expired_keys = []
    sound_html = ""
    
    for key, alert_data in st.session_state.active_alerts.items():
        age = current_time - alert_data['timestamp']
        if age < st.session_state.alert_duration:
            active.append(alert_data['html'])
            # Play sound for new alerts (within first 0.5 seconds)
            if age < 0.5 and st.session_state.alert_sounds_enabled:
                sound_html = play_alert_sound(alert_data.get('sound_type', 'unauthorized'))
        else:
            expired_keys.append(key)
    
    # Remove expired alerts
    for key in expired_keys:
        del st.session_state.active_alerts[key]
    
    # Wrap all alerts in a container for proper stacking
    if active:
        return f"""
        {sound_html}
        <div class="alert-container">
            {''.join(active)}
        </div>
        """
    return ""

def video_processing_thread(video_source, config, frame_queue, log_queue, stop_flag):
    """Background thread for video processing with full detection pipeline"""
    cap = None
    out_writer = None  # ✅ Add video writer
    
    try:
        log_queue.put({"type": "info", "message": "Initializing system..."})
        
        # Initialize the combined system
        comb = CombinedYOLOFaceBehavior(
            yolo_path=config['yolo_model'],
            mobilenet_path=config['mobilenet_model'],
            facenet_main_path=config['facenet_main'],
            device=config['device'],
            min_class_conf=config['min_class_conf'],
            recog_interval=config['recog_interval'],
            frame_skip=config['frame_skip'],
            resize_factor=config['resize_factor'],
            enable_logging=config['enable_logging'],
            smooth_window=config['smooth_window']
        )
        
        log_queue.put({"type": "success", "message": "Models loaded successfully"})
        
        # Open video source
        log_queue.put({"type": "info", "message": f"Opening video source..."})

        if config['source_mode'] == "webcam":
            cap = cv2.VideoCapture(video_source, cv2.CAP_DSHOW)
        elif config['source_mode'] == "rtsp":
            cap = cv2.VideoCapture(video_source)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            for _ in range(10):
                cap.grab()
        else:
            cap = cv2.VideoCapture(str(video_source))
        
        if not cap.isOpened():
            log_queue.put({"type": "error", "message": "Failed to open video source"})
            return
        
        if config['source_mode'] == "webcam":
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
        fps = int(cap.get(cv2.CAP_PROP_FPS) or 30)
        
        # ✅ SETUP VIDEO RECORDING
        if config.get('recording_enabled', False):
            recordings_dir = REPO_ROOT / "recordings"
            recordings_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            source_name = config['source_mode']
            output_filename = f"recording_{source_name}_{timestamp}.mp4"
            output_path = recordings_dir / output_filename
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            
            if out_writer.isOpened():
                log_queue.put({"type": "success", "message": f"Recording started: {output_filename}"})
                log_queue.put({"type": "recording_path", "path": str(output_path)})
            else:
                log_queue.put({"type": "error", "message": "Failed to start recording"})
                out_writer = None
        
        log_queue.put({"type": "success", "message": "Video source opened"})
        
        frame_idx = 0
        consecutive_failures = 0
        max_failures = 10
        
        if config['resize_factor'] < 1.0:
            process_width = int(width * config['resize_factor'])
            process_height = int(height * config['resize_factor'])
            scale_x = width / process_width
            scale_y = height / process_height
        else:
            process_width, process_height = width, height
            scale_x = scale_y = 1.0
        
        while not stop_flag.is_set():
            if config['source_mode'] == "rtsp":
                cap.grab()
                cap.grab()
                ret, frame = cap.read()
            else:
                ret, frame = cap.read()
            
            if not ret:
                consecutive_failures += 1
                if consecutive_failures >= max_failures:
                    log_queue.put({"type": "error", "message": "Video stream ended"})
                    break
                time.sleep(0.1)
                continue
            
            consecutive_failures = 0
            frame_idx += 1
            
            if frame_idx % config['frame_skip'] != 0:
                continue
            
            if config['resize_factor'] < 1.0:
                process_frame = cv2.resize(frame, (process_width, process_height), interpolation=cv2.INTER_LINEAR)
            else:
                process_frame = frame
            
            results = comb.tracker.yolo.track(
                process_frame,
                persist=True,
                tracker=comb.tracker.tracker_name,
                classes=[0],
                conf=config.get('conf_threshold', 0.5),
                iou=config.get('iou_threshold', 0.7),
                verbose=False,
                half=comb.use_half_precision,
                device=comb.tracker.device
            )
            
            tracks_data = []
            annotated_frame = frame.copy()
            
            if results and results[0].boxes is not None:
                boxes = results[0].boxes
                if boxes.id is not None:
                    track_ids = boxes.id.cpu().numpy().astype(int)
                    bboxes = boxes.xyxy.cpu().numpy().astype(int)
                    
                    for bbox, track_id in zip(bboxes, track_ids):
                        x1, y1, x2, y2 = bbox
                        
                        x1, y1 = int(x1 * scale_x), int(y1 * scale_y)
                        x2, y2 = int(x2 * scale_x), int(y2 * scale_y)
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(width, x2), min(height, y2)
                        
                        if x2 <= x1 or y2 <= y1:
                            continue
                        
                        person_crop = frame[y1:y2, x1:x2]
                        
                        body_features = comb._extract_body_features(person_crop)
                        if body_features is not None:
                            comb.track_body_features[int(track_id)].append(body_features)
                        
                        persistent = comb.track_persistent_identity.get(int(track_id))
                        if persistent:
                            cached_name = persistent["name"]
                            cached_conf = persistent["confidence"]
                        else:
                            cached_name, cached_conf = comb.identity_cache.get(int(track_id), ("Unknown", 0.0))
                        
                        current_auth = comb.get_authorization_level(cached_name)
                        do_recog = comb._should_run_recognition(int(track_id), frame_idx, current_auth)
                        
                        has_face_detection = False
                        identity_name = cached_name
                        identity_conf = cached_conf
                        
                        if do_recog:
                            try:
                                face_result = comb.recognize_face_fn(person_crop, frame, (x1, y1, x2, y2))
                                name = face_result.get("name", "Unknown")
                                conf = float(face_result.get("confidence", 0.0) or 0.0)
                                
                                has_face_detection = (name != "Unknown" and conf > 0.0)
                                
                                comb.track_recognition_history[int(track_id)].append((name, conf))
                                consensus_name, consensus_conf = comb._get_consensus_identity(int(track_id))
                                
                                comb.identity_cache[int(track_id)] = (consensus_name, consensus_conf)
                                comb.track_last_recog_frame[int(track_id)] = frame_idx
                                
                                identity_name, identity_conf = comb._update_persistent_identity(
                                    int(track_id), consensus_name, consensus_conf, has_face_detection, frame_idx, person_crop
                                )
                            except Exception as e:
                                log_queue.put({"type": "warning", "message": f"Face recognition error: {str(e)}"})
                        else:
                            identity_name, identity_conf = comb._update_persistent_identity(
                                int(track_id), cached_name, cached_conf, False, frame_idx, person_crop
                            )
                        
                        auth_level = comb.get_authorization_level(identity_name)
                        
                        behavior_name = "N/A"
                        behavior_conf = 0.0
                        
                        if auth_level == "Partially Authorized":
                            do_classify = comb.tracker._should_reclassify(track_id, frame_idx)
                            if do_classify:
                                try:
                                    class_res = comb.tracker._classify_crop(person_crop)
                                    comb.tracker.prob_history[track_id].append(class_res['probs'])
                                    hist = comb.tracker.prob_history[track_id]
                                    
                                    if len(hist) == 1:
                                        smoothed_probs = hist[0]
                                    else:
                                        smoothed_probs = np.mean(np.stack(hist, axis=0), axis=0)
                                    
                                    smoothed_class_id = int(np.argmax(smoothed_probs))
                                    smoothed_confidence = float(smoothed_probs[smoothed_class_id])
                                    
                                    if smoothed_confidence < comb.tracker.min_class_conf:
                                        behavior_name = "Neutral"
                                        behavior_conf = smoothed_confidence
                                    else:
                                        behavior_name = comb.tracker.class_names[smoothed_class_id]
                                        behavior_conf = smoothed_confidence
                                    
                                    comb.tracker.classification_cache[track_id] = {
                                        "class_name": behavior_name,
                                        "confidence": behavior_conf,
                                        "last_frame": frame_idx
                                    }
                                except Exception as e:
                                    log_queue.put({"type": "warning", "message": f"Behavior classification error: {str(e)}"})
                            else:
                                cached_beh = comb.tracker.classification_cache.get(track_id, {"class_name": "Neutral", "confidence": 0.0})
                                behavior_name = cached_beh["class_name"]
                                behavior_conf = cached_beh["confidence"]
                        
                        color = comb.get_authorization_color(auth_level)
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                        
                        persistent = comb.track_persistent_identity.get(int(track_id))
                        lock_indicator = " [LOCKED]" if persistent and persistent.get("locked", False) else ""
                        label = f"ID:{track_id} {identity_name}{lock_indicator}"
                        
                        cv2.putText(annotated_frame, label, (x1+2, max(20, y1-20)), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                        
                        if auth_level == "Partially Authorized":
                            auth_label = f"{auth_level} | {behavior_name}"
                        else:
                            auth_label = f"{auth_level}"
                        
                        cv2.putText(annotated_frame, auth_label, (x1+2, max(35, y1-6)), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        
                        tracks_data.append({
                            "track_id": int(track_id),
                            "identity": identity_name,
                            "authorization": auth_level,
                            "behavior": behavior_name,
                            "behavior_conf": behavior_conf,
                            "identity_conf": identity_conf
                        })
            
            # ✅ ADD TIMESTAMP TO RECORDING
            if out_writer is not None:
                timestamp_text = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                cv2.putText(annotated_frame, timestamp_text, (10, height - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Add recording indicator (red dot)
                cv2.circle(annotated_frame, (width - 30, 30), 10, (0, 0, 255), -1)
                cv2.putText(annotated_frame, "REC", (width - 70, 35),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # Update event logger
            if comb.enable_logging:
                comb.event_logger.update_tracking(tracks_data, frame_idx)
            
            # Send detections to UI
            log_queue.put({"type": "detections", "data": tracks_data})
            
            # ✅ WRITE TO RECORDING
            if out_writer is not None and out_writer.isOpened():
                out_writer.write(annotated_frame)
            
            # Send annotated frame to display queue
            try:
                while not frame_queue.empty():
                    try:
                        frame_queue.get_nowait()
                    except queue.Empty:
                        break
                
                frame_queue.put(annotated_frame.copy())
            except Exception as e:
                log_queue.put({"type": "error", "message": f"Queue error: {str(e)}"})
            
            time.sleep(0.001)
        
        log_queue.put({"type": "info", "message": "Processing stopped"})
        
    except Exception as e:
        import traceback
        error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
        log_queue.put({"type": "error", "message": error_msg})
        print(error_msg)
    
    finally:
        if cap is not None:
            cap.release()
            log_queue.put({"type": "info", "message": "Camera released"})
        
        # ✅ CLOSE VIDEO WRITER
        if out_writer is not None:
            out_writer.release()
            log_queue.put({"type": "success", "message": "Recording saved successfully"})

def main():
    # Header
    st.markdown('<div class="main-header">🎥 CCTV Monitoring System</div>', unsafe_allow_html=True)
    
    # Sidebar - Configuration
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
            test_videos_dir = REPO_ROOT / "Mp4TESTING"
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
                # Option 1: Select from Mp4TESTING folder
                if available_videos:
                    video_options = {str(v): v.name for v in available_videos}
                    selected_video = st.selectbox(
                        "Select Test Video",
                        options=list(video_options.keys()),
                        format_func=lambda x: video_options[x]
                    )
                    
                    if selected_video:
                        video_source = selected_video
                        file_size_mb = Path(selected_video).stat().st_size / (1024 * 1024)
                        st.success(f"✅ {Path(selected_video).name} ({file_size_mb:.1f} MB)")
                    else:
                        video_source = None
                else:
                    st.warning(f"⚠️ No videos found in `{test_videos_dir.relative_to(REPO_ROOT)}`")
                    st.info("Add .mp4/.avi/.mov files to the Mp4TESTING folder")
                    video_source = None
            
            elif upload_method == "Upload File":
                # Option 2: Upload file (limited to 200MB)
                video_file = st.file_uploader(
                    "Upload Video (Max 200MB)", 
                    type=['mp4', 'avi', 'mov'],
                    help="Streamlit limits uploads to 200MB. Use 'Browse Test Videos' or 'Custom Path' for larger files."
                )
                
                if video_file:
                    # Show file size warning
                    file_size_mb = len(video_file.getvalue()) / (1024 * 1024)
                    if file_size_mb > 190:
                        st.warning(f"⚠️ File is {file_size_mb:.1f} MB (close to 200MB limit)")
                    
                    # Save to temp folder
                    temp_path = REPO_ROOT / "temp" / video_file.name
                    temp_path.parent.mkdir(exist_ok=True)
                    with open(temp_path, 'wb') as f:
                        f.write(video_file.read())
                    video_source = str(temp_path)
                    st.success(f"✅ Uploaded: {video_file.name} ({file_size_mb:.1f} MB)")
                else:
                    video_source = None
            
            else:  # Custom Path
                # Option 3: Manual file path entry
                default_path = cam_config.VIDEO_FILE_PATH if Path(cam_config.VIDEO_FILE_PATH).exists() else ""
                file_path_input = st.text_input(
                    "Video File Path",
                    value=default_path,
                    placeholder="e.g., C:/Videos/recording.mp4 or Mp4TESTING/video.mp4",
                    help="Enter full path or relative path from project root"
                )
                
                if file_path_input:
                    # Try absolute path first
                    file_path = Path(file_path_input)
                    
                    # If not absolute, try relative to REPO_ROOT
                    if not file_path.is_absolute():
                        file_path = REPO_ROOT / file_path_input
                    
                    if file_path.exists():
                        video_source = str(file_path)
                        file_size_mb = file_path.stat().st_size / (1024 * 1024)
                        st.success(f"✅ {file_path.name} ({file_size_mb:.1f} MB)")
                    else:
                        st.error(f"❌ File not found: {file_path}")
                        video_source = None
                else:
                    video_source = None
            
            source_mode = "video"
        
        else:  # RTSP Camera
            # Get available cameras
            rtsp_cameras = cam_config.get_all_rtsp_cameras()
            camera_options = {key: f"{key} - {name}" for key, name, enabled in rtsp_cameras if enabled}
            
            if camera_options:
                selected_camera = st.selectbox(
                    "Select RTSP Camera",
                    options=list(camera_options.keys()),
                    format_func=lambda x: camera_options[x],
                    index=list(camera_options.keys()).index(cam_config.ACTIVE_RTSP_CAMERA) if cam_config.ACTIVE_RTSP_CAMERA in camera_options else 0
                )
                
                # Show camera details
                camera_info = cam_config.RTSP_CAMERAS[selected_camera]
                st.text(f"IP: {camera_info['ip']}")
                st.text(f"Stream: {camera_info['stream']}")
                
                # Get RTSP URL
                video_source = cam_config.get_rtsp_url(selected_camera)
                
                # Show connection settings
                with st.expander("RTSP Settings"):
                    st.text(f"Protocol: {cam_config.RTSP_PROTOCOL}")
                    st.text(f"Timeout: {cam_config.RTSP_TIMEOUT}s")
                    st.text(f"Auto-reconnect: {cam_config.RTSP_AUTO_RECONNECT}")
                    st.text(f"Buffer size: {cam_config.RTSP_BUFFER_SIZE}")
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
        mobilenet_model = st.text_input("MobileNet Model", value="models/mobilenetv2/mobilenet_feature_extraction.pth")
        facenet_main = st.text_input("FaceNet Main", value=str(REPO_ROOT / "face_recognition" / "Facenet" / "facenet_main.py"))
        
        st.divider()
        
        # Performance Settings
        st.subheader("Performance")
        frame_skip = st.slider("Frame Skip", 1, 10, 1)
        resize_factor = st.slider("Resize", 0.1, 1.0, 1.0)
        min_class_conf = st.slider("Min Confidence", 0.0, 1.0, 0.7)
        recog_interval = st.slider("Recognition Interval", 10, 60, 30)
        smooth_window = st.slider("Smoothing Window", 1, 15, 7, 
                                  help="Temporal smoothing for behavior predictions")
        
        st.divider()
        
        # ✅ RECORDING SETTINGS
        st.subheader("📹 Recording")
        enable_recording = st.checkbox("Enable Recording", value=True, 
                                       help="Save annotated video with all detections")
        
        if enable_recording:
            st.info("📁 Recordings saved to: `recordings/`")
            if st.session_state.recording_path:
                st.success(f"Current: {Path(st.session_state.recording_path).name}")
        
        st.divider()
        
        enable_logging = st.checkbox("Enable Logging", value=True)
        st.session_state.alert_sounds_enabled = st.checkbox("Enable Alert Sounds", value=True)
        
        st.divider()
        
        # Control Buttons
        col1, col2 = st.columns(2)
        with col1:
            start_button = st.button("▶️ Start", use_container_width=True, type="primary", disabled=st.session_state.running)
        with col2:
            stop_button = st.button("⏹️ Stop", use_container_width=True, disabled=not st.session_state.running)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📹 Live Feed")
        video_placeholder = st.empty()
        status_placeholder = st.empty()
        alert_placeholder = st.empty()  # Add this line
    
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

    st.subheader("📋 System Log")
    log_placeholder = st.empty()
    
    # Handle start/stop
    if start_button and not st.session_state.running:
        if video_source is not None:
            st.session_state.running = True
            st.session_state.stop_flag.clear()
            st.session_state.recording_enabled = enable_recording  # ✅ Store recording state
            
            config = {
                'yolo_model': yolo_model,
                'mobilenet_model': mobilenet_model,
                'facenet_main': facenet_main,
                'device': device,
                'min_class_conf': min_class_conf,
                'recog_interval': recog_interval,
                'frame_skip': frame_skip,
                'resize_factor': resize_factor,
                'enable_logging': enable_logging,
                'webcam_index': int(webcam_index) if 'webcam_index' in locals() else 0,
                'conf_threshold': 0.5,
                'iou_threshold': 0.7,
                'source_mode': source_mode,
                'smooth_window': smooth_window,
                'recording_enabled': enable_recording,  # ✅ Pass to thread
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
    
    # Display loop
    if st.session_state.running:
        status_placeholder.success("🟢 Running")
        
        # ✅ SHOW RECORDING STATUS
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
                            
                            behavior_text = f" | {detection['behavior']}" if detection['authorization'] == "Partially Authorized" and detection['behavior'] != "N/A" else ""
                            
                            st.markdown(f"""
                            <div class="status-box {auth_class}">
                                <strong>{detection['identity']}</strong><br>
                                {detection['authorization']}{behavior_text}
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Generate alerts for unauthorized/partial
                            if detection['authorization'] in ["Unauthorized", "Partially Authorized"]:
                                show_alert(
                                    detection['authorization'], 
                                    detection['identity'], 
                                    detection.get('behavior')
                                )
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
                    else:
                        log_messages.append(msg)
                except queue.Empty:
                    break
            
            if log_messages:
                try:
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
                except:
                    pass
            
            time.sleep(0.01)
            
            if not st.session_state.running:
                break
    else:
        status_placeholder.warning("🔴 Stopped")
        video_placeholder.empty()
        
        # ✅ SHOW DOWNLOAD BUTTON FOR LAST RECORDING
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