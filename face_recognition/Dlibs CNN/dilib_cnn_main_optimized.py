"""
Optimized Dlib CNN Face Recognition System
Based on FaceNet architecture for better performance and tracking
"""
import sys
import os
import cv2
import dlib
import numpy as np
import joblib
from datetime import datetime, timedelta
from threading import Thread, Event
from queue import Queue, Empty
from collections import deque, defaultdict
from ultralytics import YOLO

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)

# Import optimized recognizer
sys.path.append(os.path.join(os.path.dirname(__file__)))
from dlib_face_recognizer import DlibCNNRecognizer

# -------------------- CONFIG --------------------
USE_WEBCAM = False
VIDEO_PATH = r"C:\Users\Alexa\OneDrive\Documents\Thesis\Reactive-CCTV-System-Using-Authorization-Level-Analysis\Mp4TESTING\ThesisMP4TEST3.mp4"
YOLO_MODEL_PATH = "models/YOLOv8/yolov8n.pt"
DLIB_MODELS_DIR = os.path.join("models", "Dlib")

# Optimized directories
LOGS_BASE = os.path.join("logs", "DlibCNN")
ANNOTATED_BASE = os.path.join("annotated_frames", "DlibCNN")
LOGS_KNOWN_DIR = os.path.join(LOGS_BASE, "known")
LOGS_UNKNOWN_DIR = os.path.join(LOGS_BASE, "unknown")
ANNOTATED_KNOWN_DIR = os.path.join(ANNOTATED_BASE, "known")
ANNOTATED_UNKNOWN_DIR = os.path.join(ANNOTATED_BASE, "unknown")

# Performance settings (copied from FaceNet)
SAVE_FACES = True
RESIZE_WIDTH = 720
PROCESS_EVERY_N = 2

# Recognition thresholds
RECOG_THRESHOLD = 0.45
RECOG_MARGIN = 0.08
PERSON_CONF_THRESHOLD = 0.6
MIN_FACE_SIZE = 30
MAX_FACES_PER_FRAME = 12
FACE_QUALITY_THRESHOLD = 0.8

# ByteTrack settings (copied from FaceNet)
BYTETRACK_TRACK_THRESH = 0.6
BYTETRACK_TRACK_BUFFER = 90
BYTETRACK_MATCH_THRESH = 0.7

# Identity persistence settings (copied from FaceNet)
IDENTITY_MEMORY_FRAMES = 90
IDENTITY_CONFIDENCE_DECAY = 0.995
MIN_IDENTITY_CONFIDENCE = 0.15
FACE_LOST_TOLERANCE = 180

# Enhanced back view tracking settings (copied from FaceNet)
BACK_VIEW_TOLERANCE_FRAMES = 600
BODY_MATCH_THRESHOLD = 0.5
EXTENDED_MEMORY_FRAMES = 300
MIN_BODY_MATCH_CONFIDENCE = 0.4
POSE_HISTORY_LENGTH = 30

# Performance settings
CAPTURE_QUEUE_SIZE = 4
DISPLAY_QUEUE_SIZE = 2
KNOWN_SAVE_INTERVAL_MIN = 3

# Preprocessing settings
ENABLE_CLAHE = True
CLAHE_CLIP = 2.0
CLAHE_TILE = 8
ENABLE_GAMMA_CORRECTION = True
GAMMA_TARGET_MEAN = 100.0

# -------------------- DEVICE & MODELS --------------------
print(f"[INFO] Initializing Dlib CNN system...")

# YOLO with ByteTrack
yolo = YOLO(YOLO_MODEL_PATH)

# Dlib CNN Recognizer
face_recognizer = DlibCNNRecognizer(DLIB_MODELS_DIR)

# Ensure output dirs
if SAVE_FACES:
    for p in [LOGS_KNOWN_DIR, LOGS_UNKNOWN_DIR, ANNOTATED_KNOWN_DIR, ANNOTATED_UNKNOWN_DIR]:
        os.makedirs(p, exist_ok=True)

# -------------------- HELPER FUNCTIONS (from FaceNet) --------------------
def detect_person_pose_from_body(person_crop):
    """Detect if person is facing away based on body characteristics"""
    try:
        if person_crop is None or person_crop.size == 0:
            return "unknown", 0.0
        
        h, w = person_crop.shape[:2]
        if h < 80 or w < 40:
            return "frontal", 0.5
        
        gray = cv2.cvtColor(person_crop, cv2.COLOR_BGR2GRAY)
        
        # Divide person into regions
        head_region = gray[:h//3, :]
        torso_region = gray[h//3:2*h//3, :]
        
        # Analyze head region for back-of-head characteristics
        head_edges = cv2.Canny(head_region, 30, 100)
        head_edge_density = np.count_nonzero(head_edges) / max(head_edges.size, 1)
        
        # Analyze symmetry
        left_half = gray[:, :w//2]
        right_half = cv2.flip(gray[:, w//2:], 1)
        
        min_w = min(left_half.shape[1], right_half.shape[1])
        if min_w > 0:
            left_half = left_half[:, :min_w]
            right_half = right_half[:, :min_w]
            
            if left_half.shape == right_half.shape:
                diff = cv2.absdiff(left_half, right_half)
                symmetry_score = 1.0 - (np.mean(diff) / 255.0)
            else:
                symmetry_score = 0.5
        else:
            symmetry_score = 0.5
        
        # Calculate back view score
        back_score = 0.0
        
        if head_edge_density < 0.06:
            back_score += 0.4
        if symmetry_score > 0.7:
            back_score += 0.3
        
        head_brightness = np.mean(head_region)
        torso_brightness = np.mean(torso_region)
        brightness_ratio = head_brightness / max(torso_brightness, 1)
        if 0.8 < brightness_ratio < 1.3:
            back_score += 0.2
        
        head_std = np.std(head_region)
        if head_std < 25:
            back_score += 0.1
        
        if back_score > 0.6:
            return "back_view", back_score
        elif back_score > 0.4:
            return "partial_back", back_score
        elif head_edge_density > 0.12:
            return "frontal", 1.0 - back_score
        else:
            return "profile", 0.5
            
    except Exception:
        return "frontal", 0.5

def calculate_body_similarity(template_crop, current_crop):
    """Calculate body similarity score between two person crops"""
    try:
        if template_crop is None or current_crop is None:
            return 0.0
        if template_crop.size == 0 or current_crop.size == 0:
            return 0.0

        h, w = 128, 64
        tpl = cv2.resize(template_crop, (w, h))
        cur = cv2.resize(current_crop, (w, h))

        tpl_hsv = cv2.cvtColor(tpl, cv2.COLOR_BGR2HSV)
        cur_hsv = cv2.cvtColor(cur, cv2.COLOR_BGR2HSV)

        hist_size = [50, 60]
        hist_ranges = [0, 180, 0, 256]
        tpl_hist = cv2.calcHist([tpl_hsv], [0, 1], None, hist_size, hist_ranges)
        cur_hist = cv2.calcHist([cur_hsv], [0, 1], None, hist_size, hist_ranges)

        cv2.normalize(tpl_hist, tpl_hist)
        cv2.normalize(cur_hist, cur_hist)

        score = cv2.compareHist(tpl_hist, cur_hist, cv2.HISTCMP_CORREL)
        score = max(0.0, min(1.0, (score + 1.0) / 2.0))

        return float(score)
    except Exception:
        return 0.0

def apply_clahe_rgb(rgb_image, clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_TILE):
    """Apply CLAHE to RGB image"""
    if rgb_image is None or rgb_image.size == 0:
        return rgb_image
    
    img = rgb_image
    if img.dtype != 'uint8':
        img = (np.clip(img, 0.0, 1.0) * 255).astype('uint8')
    
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(tileGridSize, tileGridSize))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

def estimate_brightness(gray_img):
    """Return mean brightness of grayscale image"""
    if gray_img is None or gray_img.size == 0:
        return 0.0
    return float(np.mean(gray_img))

def auto_gamma_correction(img, target_mean=GAMMA_TARGET_MEAN, max_gamma=1.8, min_gamma=0.6):
    """Auto gamma correction"""
    if img is None or img.size == 0:
        return img
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean = estimate_brightness(gray)
    if mean <= 0:
        return img
    
    gamma = float(target_mean) / float(mean)
    gamma = max(min(gamma, max_gamma), min_gamma)
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(256)]).astype('uint8')
    return cv2.LUT(img, table)

def recognize_face_in_crop_optimized(person_crop, original_frame, person_bbox):
    """Optimized face recognition using Dlib CNN with preprocessing"""
    if person_crop is None or person_crop.size == 0:
        return {'name': 'Unknown', 'confidence': 0.0, 'face_bbox': None}
    
    try:
        # Apply preprocessing like FaceNet
        processed_crop = person_crop.copy()
        
        if ENABLE_GAMMA_CORRECTION:
            gray_check = cv2.cvtColor(processed_crop, cv2.COLOR_BGR2GRAY)
            mean_brightness = estimate_brightness(gray_check)
            if mean_brightness < GAMMA_TARGET_MEAN * 0.8:
                processed_crop = auto_gamma_correction(processed_crop, GAMMA_TARGET_MEAN)
        
        if ENABLE_CLAHE:
            person_rgb = cv2.cvtColor(processed_crop, cv2.COLOR_BGR2RGB)
            processed_rgb = apply_clahe_rgb(person_rgb, CLAHE_CLIP, CLAHE_TILE)
            processed_crop = cv2.cvtColor(processed_rgb, cv2.COLOR_RGB2BGR)
        
        # Use Dlib CNN recognizer
        result = face_recognizer.recognize_face_in_crop(processed_crop, original_frame, person_bbox)
        return result
        
    except Exception as e:
        print(f"[ERROR] Optimized face recognition failed: {e}")
        return {'name': 'Unknown', 'confidence': 0.0, 'face_bbox': None}

def update_track_identity(track_id, face_result, person_crop, track_identities, track_face_history, track_body_history, frame_num):
    """Enhanced identity tracking with stronger identity locking (copied from FaceNet)"""
    
    name = face_result['name']
    conf = face_result['confidence']
    
    # Detect person pose
    pose, pose_confidence = detect_person_pose_from_body(person_crop)
    
    # Initialize tracking data for new track
    if track_id not in track_identities:
        track_identities[track_id] = {
            'name': name,
            'confidence': conf,
            'last_face_frame': frame_num if name != 'Unknown' else -1,
            'last_seen_frame': frame_num,
            'stable': False,
            'pose_history': deque(maxlen=POSE_HISTORY_LENGTH),
            'body_template': None,
            'consecutive_back_frames': 0,
            'max_confidence_seen': conf,
            'identity_locked': False,
            'lock_confidence': 0.0,
            'frames_since_face_lost': 0,
            'total_face_detections': 0,  # New: count face detections
            'lock_strength': 0.0,        # New: how strong the lock is
            'unlock_threshold': 600      # New: frames before considering unlock (20 seconds at 30fps)
        }
        track_face_history[track_id] = deque(maxlen=EXTENDED_MEMORY_FRAMES)
        track_body_history[track_id] = deque(maxlen=60)
    
    identity = track_identities[track_id]
    identity['pose_history'].append((pose, pose_confidence, frame_num))
    identity['last_seen_frame'] = frame_num
    
    # Store body crop for matching
    if person_crop is not None and person_crop.size > 0:
        track_body_history[track_id].append({
            'crop': person_crop.copy(),
            'frame': frame_num,
            'pose': pose
        })
    
    # Update face history
    track_face_history[track_id].append({
        'name': name,
        'confidence': conf,
        'frame': frame_num,
        'pose': pose,
        'pose_conf': pose_confidence
    })
    
    # Handle different scenarios
    if name != 'Unknown':
        identity['total_face_detections'] += 1
        
        # Strong face detection - lock identity
        if conf > 0.7:  # Lowered threshold for locking
            identity['name'] = name
            identity['confidence'] = conf
            identity['stable'] = True
            identity['last_face_frame'] = frame_num
            identity['max_confidence_seen'] = max(identity['max_confidence_seen'], conf)
            identity['consecutive_back_frames'] = 0
            identity['frames_since_face_lost'] = 0
            
            # Enhanced locking logic
            identity['identity_locked'] = True
            identity['lock_confidence'] = max(identity['lock_confidence'], conf)
            
            # Calculate lock strength based on detection history
            identity['lock_strength'] = min(1.0, identity['total_face_detections'] / 10.0) * conf
            
            # Update unlock threshold based on confidence
            if conf > 0.9:
                identity['unlock_threshold'] = 1800  # 60 seconds for very high confidence
            elif conf > 0.8:
                identity['unlock_threshold'] = 900   # 30 seconds for high confidence
            else:
                identity['unlock_threshold'] = 600   # 20 seconds for medium confidence
            
            # Update body template when we have good face recognition
            if person_crop is not None:
                identity['body_template'] = person_crop.copy()
                print(f"[LOCK] 🔒 Track {track_id}: Locked identity {name} with confidence {conf:.3f}, strength {identity['lock_strength']:.3f}")
                
        elif name == identity['name']:
            # Same person confirmation
            identity['confidence'] = min(1.0, identity['confidence'] * 1.02)
            identity['stable'] = True
            identity['last_face_frame'] = frame_num
            identity['consecutive_back_frames'] = 0
            identity['frames_since_face_lost'] = 0
            
            # Strengthen existing lock
            if identity['identity_locked']:
                identity['lock_strength'] = min(1.0, identity['lock_strength'] * 1.01)
            
    else:
        # No face detected - handle based on pose and lock status
        identity['frames_since_face_lost'] += 1
        
        if pose in ['back_view', 'partial_back']:
            identity['consecutive_back_frames'] += 1
            
            # If identity is locked, maintain it strongly during back view
            if identity['identity_locked'] and identity['stable']:
                # Almost no confidence decay for locked identity in back view
                decay_rate = 0.9995  # Very minimal decay
                
                # Try body matching for additional confidence
                if (identity['body_template'] is not None and person_crop is not None):
                    body_similarity = calculate_body_similarity(identity['body_template'], person_crop)
                    
                    if body_similarity > 0.5:  # Lower threshold for back view
                        # Boost confidence for good body match
                        identity['confidence'] = min(1.0, identity['confidence'] * 1.005)
                        decay_rate = 0.9998  # Even less decay with body match
                        
                        if identity['consecutive_back_frames'] % 60 == 0:  # Log every 2 seconds
                            print(f"[BODY] Track {track_id}: Maintaining locked {identity['name']} via body match ({body_similarity:.3f}) - {identity['consecutive_back_frames']} back frames")
                
                identity['confidence'] *= decay_rate
                
                # Only consider unlocking after extended period AND very low confidence
                if (identity['consecutive_back_frames'] > identity['unlock_threshold'] and 
                    identity['confidence'] < 0.1 and 
                    identity['lock_strength'] < 0.3):
                    
                    identity['identity_locked'] = False
                    print(f"[UNLOCK] 🔓 Track {track_id}: Unlocking {identity['name']} after {identity['consecutive_back_frames']} back frames")
            else:
                # Not locked, decay more quickly
                identity['confidence'] *= 0.985
        else:
            # Not back view but no face - could be profile or temporary occlusion
            if identity['identity_locked'] and identity['frames_since_face_lost'] < identity['unlock_threshold']:
                # Maintain locked identity for reasonable periods without face
                identity['confidence'] *= 0.998  # Very slow decay
                
                if identity['frames_since_face_lost'] % 60 == 0:  # Log every 2 seconds
                    print(f"[MAINTAIN] Track {track_id}: Maintaining locked {identity['name']} during face loss ({identity['frames_since_face_lost']} frames)")
            else:
                # Regular decay or unlock
                if identity['identity_locked'] and identity['frames_since_face_lost'] > identity['unlock_threshold']:
                    identity['identity_locked'] = False
                    print(f"[UNLOCK] 🔓 Track {track_id}: Unlocking {identity['name']} after {identity['frames_since_face_lost']} frames without face")
                
                identity['confidence'] *= IDENTITY_CONFIDENCE_DECAY
                identity['consecutive_back_frames'] = 0
    
    # Prevent confidence from going too low for locked identities
    if identity['identity_locked']:
        min_locked_confidence = max(0.2, identity['lock_strength'] * 0.5)
        identity['confidence'] = max(identity['confidence'], min_locked_confidence)
    
    # Stability check - but don't mark unstable if locked
    if not identity['identity_locked'] and identity['confidence'] < MIN_IDENTITY_CONFIDENCE:
        identity['stable'] = False

def get_consensus_identity(track_id, track_identities, track_face_history, frames_since_face):
    """Enhanced consensus with identity locking and back view support (copied from FaceNet)"""
    if track_id not in track_identities:
        return 'Unknown', 0.0
    
    identity = track_identities[track_id]
    current_name = identity['name']
    current_conf = identity['confidence']
    
    # If identity is locked, be much more conservative about changing it
    if identity.get('identity_locked', False):
        consecutive_back = identity.get('consecutive_back_frames', 0)
        frames_since_face_lost = identity.get('frames_since_face_lost', 0)
        lock_confidence = identity.get('lock_confidence', 0.0)
        
        # Maintain locked identity for reasonable periods
        if frames_since_face_lost < BACK_VIEW_TOLERANCE_FRAMES:
            # Apply minimal penalty for locked identity
            if consecutive_back > 60:  # Extended back view
                penalty = min(0.2, consecutive_back / (BACK_VIEW_TOLERANCE_FRAMES * 2))
            else:
                penalty = min(0.1, frames_since_face_lost / BACK_VIEW_TOLERANCE_FRAMES)
            
            adjusted_conf = current_conf * (1.0 - penalty)
            
            # Keep locked identity if original lock was strong
            if lock_confidence > 0.8 and adjusted_conf > 0.3:
                return current_name, adjusted_conf
            elif lock_confidence > 0.6 and adjusted_conf > 0.4:
                return current_name, adjusted_conf
    
    # Standard consensus logic for unlocked identities
    if not identity['stable'] or current_conf < MIN_IDENTITY_CONFIDENCE:
        return 'Unknown', 0.0
    
    # Use temporal consensus from history with stronger weighting for locked identities
    if track_id in track_face_history:
        history = track_face_history[track_id]
        if len(history) > 3:  # Need some history
            name_scores = defaultdict(float)
            total_weight = 0
            
            # Look at recent history with emphasis on strong detections
            recent_history = list(history)[-90:]  # Longer history for locked tracks
            
            for i, record in enumerate(recent_history):
                if record['name'] != 'Unknown':
                    # Age weight - more recent = higher weight
                    age_weight = (0.98 ** (len(recent_history) - i - 1))
                    
                    # Confidence weight with boost for high confidence
                    conf_weight = record['confidence']
                    if conf_weight > 0.8:
                        conf_weight *= 2.0  # Strong boost for high confidence
                    elif conf_weight > 0.6:
                        conf_weight *= 1.5
                    
                    # Pose weight
                    pose_weight = 1.0
                    if record.get('pose') == 'frontal':
                        pose_weight = 1.2
                    elif record.get('pose') in ['back_view', 'partial_back']:
                        pose_weight = 0.8
                    
                    weight = conf_weight * age_weight * pose_weight
                    name_scores[record['name']] += weight
                    total_weight += weight
            
            if name_scores:
                best_name = max(name_scores, key=name_scores.get)
                consensus_conf = name_scores[best_name] / max(total_weight, 1)
                
                # Boost for locked identity
                if identity.get('identity_locked', False) and best_name == current_name:
                    consensus_conf = min(1.0, consensus_conf * 1.3)
                
                # Apply reasonable penalties only
                if frames_since_face > FACE_LOST_TOLERANCE:
                    penalty = min(0.3, (frames_since_face - FACE_LOST_TOLERANCE) / BACK_VIEW_TOLERANCE_FRAMES)
                    consensus_conf *= (1.0 - penalty)
                
                return best_name, consensus_conf
    
    return current_name, current_conf

def process_frames_with_bytetrack(frame_q, display_q, stop_event):
    """Enhanced frame processing using YOLO ByteTrack (copied from FaceNet)"""
    
    # ByteTrack tracking data
    track_identities = {}
    track_face_history = defaultdict(lambda: deque(maxlen=EXTENDED_MEMORY_FRAMES))
    track_body_history = defaultdict(lambda: deque(maxlen=60))
    track_last_face_frame = {}
    known_last_saved = {}
    
    processed = 0
    frame_times = deque(maxlen=30)
    
    while not stop_event.is_set() or not frame_q.empty():
        try:
            frame_num, ts, frame = frame_q.get(timeout=0.05)
        except Empty:
            continue
        except Exception as e:
            print(f"[ERROR] Frame queue error: {e}")
            continue
        
        try:
            process_start = datetime.now()
            original_frame = frame.copy()
            orig_h, orig_w = original_frame.shape[:2]
            
            # Resize for processing
            if RESIZE_WIDTH and orig_w > RESIZE_WIDTH:
                ratio = RESIZE_WIDTH / orig_w
                process_frame = cv2.resize(original_frame, (RESIZE_WIDTH, int(orig_h * ratio)))
            else:
                process_frame = original_frame
                ratio = 1.0
            
            # Skip frames for performance
            if frame_num % PROCESS_EVERY_N != 0:
                try:
                    display_q.put((frame_num, ts, original_frame), timeout=0.01)
                except:
                    pass
                continue
            
            annotated_frame = original_frame.copy()
            
            # Run YOLO with ByteTrack
            results = yolo.track(
                process_frame,
                persist=True,
                tracker="bytetrack.yaml",
                classes=[0],  # Only persons
                conf=PERSON_CONF_THRESHOLD,
                iou=0.7,
                imgsz=640,
                verbose=False
            )
            
            # Debug output
            if frame_num % 30 == 0:
                if results and len(results) > 0 and results[0].boxes is not None:
                    num_boxes = len(results[0].boxes)
                    if results[0].boxes.id is not None:
                        num_tracks = len(results[0].boxes.id)
                        print(f"[DEBUG] Frame {frame_num}: {num_boxes} boxes, {num_tracks} tracks")
            
            if results and results[0].boxes is not None:
                boxes = results[0].boxes
                
                if boxes.id is not None:
                    track_ids = boxes.id.cpu().numpy().astype(int)
                    bboxes = boxes.xyxy.cpu().numpy()
                    confidences = boxes.conf.cpu().numpy()
                    
                    for track_id, bbox, conf in zip(track_ids, bboxes, confidences):
                        # Convert bbox to original frame coordinates
                        if ratio != 1.0:
                            bbox = bbox / ratio
                        
                        x1, y1, x2, y2 = bbox.astype(int)
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(orig_w, x2), min(orig_h, y2)
                        
                        person_bbox = (x1, y1, x2, y2)
                        person_crop = original_frame[y1:y2, x1:x2]
                        
                        # Perform optimized face recognition
                        face_result = recognize_face_in_crop_optimized(person_crop, original_frame, person_bbox)
                        
                        # Update tracking history
                        update_track_identity(
                            track_id, face_result, person_crop, track_identities, 
                            track_face_history, track_body_history, frame_num
                        )
                        
                        # Update frame numbers
                        if face_result['name'] != 'Unknown':
                            track_last_face_frame[track_id] = frame_num
                        
                        # Get consensus identity
                        frames_since_face = frame_num - track_last_face_frame.get(track_id, frame_num)
                        identity_name, identity_conf = get_consensus_identity(
                            track_id, track_identities, track_face_history, frames_since_face
                        )
                        
                        # Enhanced display logic
                        identity = track_identities.get(track_id, {})
                        is_locked = identity.get('identity_locked', False)
                        consecutive_back = identity.get('consecutive_back_frames', 0)
                        
                        # Choose display color based on identity status
                        if identity_name != 'Unknown':
                            if is_locked:
                                color = (0, 255, 0)  # Bright green for locked identity
                                thickness = 3
                            else:
                                color = (0, 200, 0)  # Green for known
                                thickness = 2
                        else:
                            color = (0, 0, 255)  # Red for unknown
                            thickness = 2
                        
                        # Draw bounding box
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
                        
                        # Draw face box if available
                        if face_result['face_bbox'] is not None:
                            fx1, fy1, fx2, fy2 = face_result['face_bbox']
                            cv2.rectangle(annotated_frame, (fx1, fy1), (fx2, fy2), (255, 255, 0), 1)
                        
                        # Enhanced label with lock and pose information
                        pose_info = ""
                        
                        if consecutive_back > 30:
                            pose_info = f" [BACK:{consecutive_back}f]"
                        elif frames_since_face > 15:
                            pose_info = f" [NO_FACE:{frames_since_face}f]"
                        
                        lock_indicator = "🔒" if is_locked else ""
                        label = f"ID:{track_id} {lock_indicator}{identity_name}"
                        if identity_conf > 0:
                            label += f" ({identity_conf:.2f})"
                        label += pose_info
                        
                        label_y = max(30, y1 - 10)
                        
                        # Draw label background
                        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                        cv2.rectangle(annotated_frame, (x1, label_y - label_h - 5), 
                                    (x1 + label_w + 5, label_y + 5), (0, 0, 0), -1)
                        
                        # Draw label text
                        cv2.putText(annotated_frame, label, (x1 + 2, label_y), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        
                        # Save faces if enabled
                        if SAVE_FACES and identity_name != 'Unknown' and face_result['face_bbox'] is not None:
                            save_face_from_result(face_result, original_frame, identity_name, 
                                                frame_num, ts, known_last_saved)
            
            # Performance monitoring
            process_end = datetime.now()
            frame_times.append(process_end - process_start)
            
            if len(frame_times) > 0:
                avg_time = sum([t.total_seconds() for t in frame_times]) / len(frame_times)
                fps = 1.0 / max(avg_time, 0.001)
            else:
                fps = 0.0
            
            # Count locked vs unlocked tracks
            locked_tracks = sum(1 for t in track_identities.values() if t.get('identity_locked', False))
            total_tracks = len(track_identities)
            
            # Add performance info to frame
            info_text = f"Frame: {frame_num} | FPS: {fps:.1f} | Tracks: {total_tracks} | Locked: {locked_tracks}"
            cv2.putText(annotated_frame, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Send to display
            try:
                display_q.put((frame_num, ts, annotated_frame), timeout=0.01)
            except:
                pass
            
            processed += 1
            
        except Exception as e:
            print(f"[ERROR] Processing frame {frame_num}: {e}")
            try:
                display_q.put((frame_num, ts, original_frame), timeout=0.01)
            except:
                pass
            continue

def save_face_from_result(face_result, original_frame, name, frame_num, ts, known_last_saved):
    """Save detected face to disk"""
    try:
        # Check save interval
        last_saved = known_last_saved.get(name, datetime.min)
        if (ts - last_saved).total_seconds() < KNOWN_SAVE_INTERVAL_MIN * 60:
            return
        
        if face_result['face_bbox'] is not None:
            fx1, fy1, fx2, fy2 = face_result['face_bbox']
            face_crop = original_frame[fy1:fy2, fx1:fx2]
            
            if face_crop.size > 0:
                filename = f"{name}_{frame_num}_{ts.strftime('%H%M%S')}.jpg"
                
                # Save face crop
                face_path = os.path.join(LOGS_KNOWN_DIR, filename)
                cv2.imwrite(face_path, face_crop)
                
                # Save annotated frame
                ann_path = os.path.join(ANNOTATED_KNOWN_DIR, filename)
                cv2.imwrite(ann_path, original_frame)
                
                known_last_saved[name] = ts
                print(f"[INFO] Saved face: {filename}")
                
    except Exception as e:
        print(f"[WARN] Failed to save face for {name}: {e}")

def display_frames(display_q, stop_event):
    """Display processed frames"""
    while not stop_event.is_set() or not display_q.empty():
        try:
            frame_num, ts, annotated_frame = display_q.get(timeout=0.1)
            
            # Resize for display if too large
            display_frame = annotated_frame
            h, w = display_frame.shape[:2]
            if w > 1280:
                scale = 1280 / w
                new_w = 1280
                new_h = int(h * scale)
                display_frame = cv2.resize(display_frame, (new_w, new_h))
            
            cv2.imshow("Optimized Dlib CNN Recognition", display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC
                stop_event.set()
                break
                
        except Empty:
            continue
        except Exception as e:
            print(f"[ERROR] Display error: {e}")
            continue
    
    cv2.destroyAllWindows()

def grab_frames(cap, frame_q, stop_event):
    """Capture frames from VideoCapture"""
    frame_num = 0
    try:
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret or frame is None:
                stop_event.set()
                break

            ts = datetime.now()

            try:
                frame_q.put((frame_num, ts, frame), timeout=0.05)
            except:
                pass

            frame_num += 1

            if frame_q.full():
                stop_event.wait(0.01)
    except Exception as e:
        print(f"[ERROR] Capture thread error: {e}")
        stop_event.set()

# -------------------- MAIN --------------------
def main():
    print("[INFO] Starting Optimized Dlib CNN recognition system...")
    
    # Initialize video capture
    if USE_WEBCAM:
        cap = cv2.VideoCapture(0)
        print("[INFO] Using webcam")
    else:
        if not os.path.exists(VIDEO_PATH):
            print(f"[ERROR] Video file not found: {VIDEO_PATH}")
            return
        cap = cv2.VideoCapture(VIDEO_PATH)
        print(f"[INFO] Using video file: {VIDEO_PATH}")
    
    if not cap.isOpened():
        print("[ERROR] Failed to open video source")
        return
    
    # Set video properties
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    # Get video info
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"[INFO] Video: {width}x{height} @ {fps:.1f}fps, {total_frames} frames")
    
    # Create queues and stop event
    frame_q = Queue(maxsize=CAPTURE_QUEUE_SIZE)
    display_q = Queue(maxsize=DISPLAY_QUEUE_SIZE)
    stop_event = Event()
    
    # Start threads
    capture_thread = Thread(target=grab_frames, args=(cap, frame_q, stop_event))
    process_thread = Thread(target=process_frames_with_bytetrack, args=(frame_q, display_q, stop_event))
    display_thread = Thread(target=display_frames, args=(display_q, stop_event))
    
    capture_thread.daemon = True
    process_thread.daemon = True
    display_thread.daemon = True
    
    try:
        print("[INFO] Starting threads...")
        capture_thread.start()
        process_thread.start()
        display_thread.start()
        
        print("[INFO] System running. Press 'q' or ESC to quit.")
        
        # Wait for threads to complete
        display_thread.join()
        
    except KeyboardInterrupt:
        print("[INFO] Interrupted by user")
    except Exception as e:
        print(f"[ERROR] Main execution error: {e}")
    finally:
        print("[INFO] Stopping system...")
        stop_event.set()
        
        # Clean up
        try:
            capture_thread.join(timeout=2)
            process_thread.join(timeout=2)
            display_thread.join(timeout=2)
        except:
            pass
        
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] System stopped")

if __name__ == "__main__":
    main()
