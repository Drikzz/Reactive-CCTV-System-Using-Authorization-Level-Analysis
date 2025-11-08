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
    }
    .authorized { background-color: #d4edda; border: 2px solid #28a745; }
    .partial { background-color: #fff3cd; border: 2px solid #ffc107; }
    .unauthorized { background-color: #f8d7da; border: 2px solid #dc3545; }
    .log-entry {
        padding: 0.5rem;
        margin: 0.3rem 0;
        border-left: 3px solid #007bff;
        background-color: #f8f9fa;
    }
    /* Minimize stats display */
    div[data-testid="stMetric"] {
        background-color: #f8f9fa;
        padding: 0.3rem;
        border-radius: 0.3rem;
        margin: 0.1rem 0;
    }
    div[data-testid="stMetric"] > label {
        font-size: 0.75rem !important;
    }
    div[data-testid="stMetric"] > div {
        font-size: 1.1rem !important;
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

def video_processing_thread(video_source, config, frame_queue, log_queue, stop_flag):
    """Background thread for video processing with full detection pipeline"""
    cap = None
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
            enable_logging=config['enable_logging']
        )
        
        log_queue.put({"type": "success", "message": "Models loaded successfully"})
        
        # Open video source
        log_queue.put({"type": "info", "message": f"Opening video source..."})

        if video_source == "webcam":
            cap = cv2.VideoCapture(config.get('webcam_index', 0), cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(str(video_source))
        
        if not cap.isOpened():
            log_queue.put({"type": "error", "message": "Failed to open video source"})
            return
        
        # Set camera properties
        if video_source == "webcam":
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
        
        log_queue.put({"type": "success", "message": "Video source opened"})
        
        frame_idx = 0
        consecutive_failures = 0
        max_failures = 10
        
        # Processing dimensions
        if config['resize_factor'] < 1.0:
            process_width = int(width * config['resize_factor'])
            process_height = int(height * config['resize_factor'])
            scale_x = width / process_width
            scale_y = height / process_height
        else:
            process_width, process_height = width, height
            scale_x = scale_y = 1.0
        
        while not stop_flag.is_set():
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
            
            # Skip frames for performance
            if frame_idx % config['frame_skip'] != 0:
                continue
            
            # Resize for processing
            if config['resize_factor'] < 1.0:
                process_frame = cv2.resize(frame, (process_width, process_height), interpolation=cv2.INTER_LINEAR)
            else:
                process_frame = frame
            
            # Run YOLO detection + tracking
            results = comb.tracker.yolo.track(
                process_frame,
                persist=True,
                tracker=comb.tracker.tracker_name,
                classes=[0],  # Person class only
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
                        
                        # Scale back to original frame size
                        x1, y1 = int(x1 * scale_x), int(y1 * scale_y)
                        x2, y2 = int(x2 * scale_x), int(y2 * scale_y)
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(width, x2), min(height, y2)
                        
                        if x2 <= x1 or y2 <= y1:
                            continue
                        
                        # Extract person crop
                        person_crop = frame[y1:y2, x1:x2]
                        
                        # Update body features
                        body_features = comb._extract_body_features(person_crop)
                        if body_features is not None:
                            comb.track_body_features[int(track_id)].append(body_features)
                        
                        # Get cached identity
                        persistent = comb.track_persistent_identity.get(int(track_id))
                        if persistent:
                            cached_name = persistent["name"]
                            cached_conf = persistent["confidence"]
                        else:
                            cached_name, cached_conf = comb.identity_cache.get(int(track_id), ("Unknown", 0.0))
                        
                        current_auth = comb.get_authorization_level(cached_name)
                        
                        # Determine if we should run face recognition
                        do_recog = comb._should_run_recognition(int(track_id), frame_idx, current_auth)
                        
                        has_face_detection = False
                        identity_name = cached_name
                        identity_conf = cached_conf
                        
                        if do_recog:
                            try:
                                # Run face recognition
                                face_result = comb.recognize_face_fn(person_crop, frame, (x1, y1, x2, y2))
                                name = face_result.get("name", "Unknown")
                                conf = float(face_result.get("confidence", 0.0) or 0.0)
                                
                                has_face_detection = (name != "Unknown" and conf > 0.0)
                                
                                # Update recognition history
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
                            # Update persistent identity without new recognition
                            identity_name, identity_conf = comb._update_persistent_identity(
                                int(track_id), cached_name, cached_conf, False, frame_idx, person_crop
                            )
                        
                        # Get authorization level
                        auth_level = comb.get_authorization_level(identity_name)
                        
                        # Behavior classification - ONLY for Partially Authorized
                        behavior_name = "N/A"
                        behavior_conf = 0.0
                        
                        if auth_level == "Partially Authorized":
                            do_classify = comb.tracker._should_reclassify(track_id, frame_idx)
                            if do_classify:
                                try:
                                    # Classify behavior
                                    class_res = comb.tracker._classify_crop(person_crop)
                                    
                                    # Temporal smoothing
                                    comb.tracker.prob_history[track_id].append(class_res['probs'])
                                    hist = comb.tracker.prob_history[track_id]
                                    
                                    if len(hist) == 1:
                                        smoothed_probs = hist[0]
                                    else:
                                        smoothed_probs = np.mean(np.stack(hist, axis=0), axis=0)
                                    
                                    smoothed_class_id = int(np.argmax(smoothed_probs))
                                    smoothed_confidence = float(smoothed_probs[smoothed_class_id])
                                    
                                    # Apply confidence threshold
                                    if smoothed_confidence < comb.tracker.min_class_conf:
                                        behavior_name = "Neutral"
                                        behavior_conf = smoothed_confidence
                                    else:
                                        behavior_name = comb.tracker.class_names[smoothed_class_id]
                                        behavior_conf = smoothed_confidence
                                    
                                    # Update cache
                                    comb.tracker.classification_cache[track_id] = {
                                        "class_name": behavior_name,
                                        "confidence": behavior_conf,
                                        "last_frame": frame_idx
                                    }
                                except Exception as e:
                                    log_queue.put({"type": "warning", "message": f"Behavior classification error: {str(e)}"})
                            else:
                                # Use cached classification
                                cached_beh = comb.tracker.classification_cache.get(track_id, {"class_name": "Neutral", "confidence": 0.0})
                                behavior_name = cached_beh["class_name"]
                                behavior_conf = cached_beh["confidence"]
                        
                        # Draw bounding box and labels
                        color = comb.get_authorization_color(auth_level)
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                        
                        # Identity label
                        persistent = comb.track_persistent_identity.get(int(track_id))
                        lock_indicator = " [LOCKED]" if persistent and persistent.get("locked", False) else ""
                        label = f"ID:{track_id} {identity_name}{lock_indicator}"
                        
                        cv2.putText(annotated_frame, label, (x1+2, max(20, y1-20)), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                        
                        # Authorization + Behavior label
                        if auth_level == "Partially Authorized":
                            auth_label = f"{auth_level} | {behavior_name}"
                        else:
                            auth_label = f"{auth_level}"
                        
                        cv2.putText(annotated_frame, auth_label, (x1+2, max(35, y1-6)), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        
                        # Add to tracks data
                        tracks_data.append({
                            "track_id": int(track_id),
                            "name": identity_name,
                            "auth": auth_level,
                            "behavior": behavior_name,
                            "behavior_conf": behavior_conf,
                            "identity_conf": identity_conf
                        })
            
            # Update event logger
            if comb.enable_logging:
                comb.event_logger.update_tracking(tracks_data, frame_idx)
            
            # Send detections to UI
            log_queue.put({"type": "detections", "data": tracks_data})
            
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

def main():
    # Header
    st.markdown('<div class="main-header">🎥 CCTV Monitoring System</div>', unsafe_allow_html=True)
    
    # Sidebar - Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Video Source
        st.subheader("Video Source")
        source_type = st.radio("Select Source", ["Webcam", "Video File"], key="source_type")
        
        if source_type == "Webcam":
            webcam_index = st.number_input("Webcam Index", min_value=0, max_value=5, value=0)
            video_source = "webcam"
        else:
            video_file = st.file_uploader("Upload Video File", type=['mp4', 'avi', 'mov'])
            if video_file:
                temp_path = REPO_ROOT / "temp" / video_file.name
                temp_path.parent.mkdir(exist_ok=True)
                with open(temp_path, 'wb') as f:
                    f.write(video_file.read())
                video_source = str(temp_path)
            else:
                video_source = None
        
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
        frame_skip = st.slider("Frame Skip", 1, 10, 3)
        resize_factor = st.slider("Resize", 0.1, 1.0, 0.5)
        min_class_conf = st.slider("Min Confidence", 0.0, 1.0, 0.6)
        recog_interval = st.slider("Recognition Interval", 10, 60, 30)
        
        st.divider()
        
        enable_logging = st.checkbox("Enable Logging", value=True)
        
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
        if video_source:
            st.session_state.running = True
            st.session_state.stop_flag.clear()
            
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
                'webcam_index': int(webcam_index) if source_type == "Webcam" else 0,
                'conf_threshold': 0.5,
                'iou_threshold': 0.7
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
                            auth_class = "authorized" if detection['auth'] == "Authorized" else \
                                        "partial" if detection['auth'] == "Partially Authorized" else "unauthorized"
                            
                            behavior_text = f" | {detection['behavior']}" if detection['auth'] == "Partially Authorized" and detection['behavior'] != "N/A" else ""
                            
                            st.markdown(f"""
                            <div class="status-box {auth_class}">
                                <strong>{detection['name']}</strong><br>
                                {detection['auth']}{behavior_text}
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.info("No detections")
            except:
                pass
            
            # Update stats
            try:
                total = len(st.session_state.current_detections)
                auth = sum(1 for d in st.session_state.current_detections if d['auth'] == "Authorized")
                unauth = sum(1 for d in st.session_state.current_detections if d['auth'] == "Unauthorized")
                
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

if __name__ == "__main__":
    main()