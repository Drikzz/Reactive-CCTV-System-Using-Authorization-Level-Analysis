import sys
import os
import csv
# Ensure repo root is on sys.path so local package imports work when running this file directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import joblib
import torch
import numpy as np
from datetime import datetime, timedelta
from threading import Thread, Event
from queue import Queue, Empty
from collections import deque, defaultdict
from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization
from ultralytics import YOLO

# add repo root to path for utils if needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# -------------------- CONFIG --------------------
USE_WEBCAM = False
VIDEO_PATH = r"C:\Users\Alexa\OneDrive\Documents\Thesis\Reactive-CCTV-System-Using-Authorization-Level-Analysis\Mp4TESTING\ThesisMP4TEST.mp4"
YOLO_MODEL_PATH = "models/YOLOv8/yolov8n.pt"

LOGS_BASE = os.path.join("logs", "FaceNet")
ANNOTATED_BASE = os.path.join("annotated_frames", "FaceNet")
LOGS_KNOWN_DIR = os.path.join(LOGS_BASE, "known")
LOGS_UNKNOWN_DIR = os.path.join(LOGS_BASE, "unknown")
ANNOTATED_KNOWN_DIR = os.path.join(ANNOTATED_BASE, "known")
ANNOTATED_UNKNOWN_DIR = os.path.join(ANNOTATED_BASE, "unknown")

SAVE_FACES = True
RESIZE_WIDTH = 720
PROCESS_EVERY_N = 2  # Reduced since ByteTrack is more efficient

# Recognition thresholds
RECOG_THRESHOLD = 0.45
RECOG_MARGIN = 0.08
PERSON_CONF_THRESHOLD = 0.6
MIN_FACE_SIZE = 30
MAX_FACES_PER_FRAME = 12
GPU_BATCH_SIZE = 16
FACE_QUALITY_THRESHOLD = 0.8

# ByteTrack specific settings
BYTETRACK_TRACK_THRESH = 0.6    # High threshold for track confirmation
BYTETRACK_TRACK_BUFFER = 90     # Frames to keep lost tracks
BYTETRACK_MATCH_THRESH = 0.7    # Matching threshold

# Identity persistence settings
IDENTITY_MEMORY_FRAMES = 90     # How long to remember identity
IDENTITY_CONFIDENCE_DECAY = 0.98  # Confidence decay per frame
MIN_IDENTITY_CONFIDENCE = 0.3   # Minimum confidence to maintain identity
FACE_LOST_TOLERANCE = 60        # Frames without face before considering unknown

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
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Using device: {device}")

# YOLO with ByteTrack
yolo = YOLO(YOLO_MODEL_PATH)

# MTCNN
mtcnn = MTCNN(
    image_size=160, 
    margin=0, 
    keep_all=True, 
    device=device, 
    post_process=False,
    min_face_size=MIN_FACE_SIZE,
    thresholds=[0.6, 0.7, 0.7],
    factor=0.7
)

# FaceNet embedder
embedder = InceptionResnetV1(pretrained="vggface2").to(device).eval()

# Load classifier
classifier = None
label_encoder = None
centroids = None
dist_threshold = None

MODELS_DIR = os.path.join("models", "FaceNet")
SVM_PATH = os.path.join(MODELS_DIR, "facenet_svm.joblib")
LE_PATH = os.path.join(MODELS_DIR, "label_encoder.joblib")
THR_PATH = os.path.join(MODELS_DIR, "distance_threshold.npy")

try:
    if not os.path.exists(SVM_PATH) or not os.path.exists(LE_PATH):
        raise FileNotFoundError("SVM or label encoder not found in models/FaceNet")
    classifier = joblib.load(SVM_PATH)
    label_encoder = joblib.load(LE_PATH)
    
    # Load centroids
    centroids_path = os.path.join(MODELS_DIR, 'class_centroids.pkl')
    try:
        if os.path.exists(centroids_path):
            centroids = joblib.load(centroids_path)
            for k, v in list(centroids.items()):
                arr = np.asarray(v, dtype=np.float32)
                n = np.linalg.norm(arr) + 1e-10
                centroids[k] = (arr / n)
            print(f"[INFO] Loaded {len(centroids)} class centroids")
    except Exception as e:
        print(f"[WARN] Failed to load centroids: {e}")
        centroids = None
    
    if os.path.exists(THR_PATH):
        dist_threshold = float(np.load(THR_PATH))
        print(f"[INFO] Loaded distance threshold: {dist_threshold:.3f}")
    
    print(f"[INFO] Loaded classifier. Classes: {list(label_encoder.classes_)}")
except Exception as e:
    print(f"[ERROR] Failed to load models: {e}")
    classifier, label_encoder = None, None

# Ensure output dirs
if SAVE_FACES:
    for p in [LOGS_KNOWN_DIR, LOGS_UNKNOWN_DIR, ANNOTATED_KNOWN_DIR, ANNOTATED_UNKNOWN_DIR]:
        os.makedirs(p, exist_ok=True)

# -------------------- HELPER FUNCTIONS --------------------
def append_csv(csv_path, header, row):
    new_file = not os.path.exists(csv_path)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file and header:
            w.writerow(header)
        w.writerow(row)

def compute_embedding_distance(embedding, centroid):
    return float(np.linalg.norm(np.asarray(embedding, dtype=np.float32) - np.asarray(centroid, dtype=np.float32)))

def filter_quality_faces(face_boxes, face_probs, min_size=MIN_FACE_SIZE, min_prob=FACE_QUALITY_THRESHOLD):
    """Filter faces by size and detection confidence"""
    if face_boxes is None or face_probs is None:
        return None, None
    
    filtered_boxes = []
    filtered_probs = []
    
    for box, prob in zip(face_boxes, face_probs):
        face_area = (box[2] - box[0]) * (box[3] - box[1])
        if prob >= min_prob and face_area >= min_size * min_size:
            filtered_boxes.append(box)
            filtered_probs.append(prob)
    
    return (np.array(filtered_boxes) if filtered_boxes else None, 
            np.array(filtered_probs) if filtered_probs else None)

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

def recognize_face_in_crop(person_crop, original_frame, person_bbox):
    """Recognize face within person crop"""
    if person_crop is None or person_crop.size == 0:
        return {'name': 'Unknown', 'confidence': 0.0, 'face_bbox': None}
    
    try:
        # Apply preprocessing
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
        
        # Face detection
        person_rgb = cv2.cvtColor(processed_crop, cv2.COLOR_BGR2RGB)
        face_boxes, face_probs = mtcnn.detect(person_rgb)
        face_boxes, face_probs = filter_quality_faces(face_boxes, face_probs)
        
        if face_boxes is None or len(face_boxes) == 0:
            return {'name': 'Unknown', 'confidence': 0.0, 'face_bbox': None}
        
        # Take the most confident face
        best_idx = np.argmax(face_probs)
        face_box = face_boxes[best_idx]
        face_prob = face_probs[best_idx]
        
        # Extract face region
        fx1, fy1, fx2, fy2 = face_box.astype(int)
        fx1, fy1 = max(0, fx1), max(0, fy1)
        fx2, fy2 = min(person_crop.shape[1], fx2), min(person_crop.shape[0], fy2)
        
        if fx2 <= fx1 or fy2 <= fy1:
            return {'name': 'Unknown', 'confidence': 0.0, 'face_bbox': None}
        
        face_crop = processed_crop[fy1:fy2, fx1:fx2]
        
        # Convert face coordinates back to original frame
        px1, py1, px2, py2 = person_bbox
        face_x1_orig = px1 + fx1
        face_y1_orig = py1 + fy1
        face_x2_orig = px1 + fx2
        face_y2_orig = py1 + fy2
        
        # Face recognition
        if classifier is not None and label_encoder is not None:
            try:
                # Prepare face for recognition
                face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                face_resized = cv2.resize(face_rgb, (160, 160))
                
                # Convert to tensor and get embedding
                tensor = torch.from_numpy(face_resized).permute(2, 0, 1).float()
                tensor = fixed_image_standardization(tensor)
                
                with torch.no_grad():
                    embedding = embedder(tensor.unsqueeze(0).to(device))
                    embedding = embedding.cpu().numpy()[0]
                
                # Normalize embedding
                emb_norm = embedding / (np.linalg.norm(embedding) + 1e-10)
                
                # Classify
                probs = classifier.predict_proba([emb_norm])[0]
                pred = np.argmax(probs)
                confidence = probs[pred]
                
                # Check thresholds
                sorted_probs = np.sort(probs)[::-1]
                top2_prob = sorted_probs[1] if len(sorted_probs) > 1 else 0.0
                
                if confidence >= RECOG_THRESHOLD and (confidence - top2_prob) >= RECOG_MARGIN:
                    candidate = label_encoder.inverse_transform([pred])[0]
                    
                    # Distance check if available
                    if centroids is not None and dist_threshold is not None:
                        centroid = centroids.get(candidate)
                        if centroid is not None:
                            dist = compute_embedding_distance(emb_norm, centroid)
                            if dist <= dist_threshold:
                                return {
                                    'name': candidate,
                                    'confidence': confidence,
                                    'face_bbox': (face_x1_orig, face_y1_orig, face_x2_orig, face_y2_orig)
                                }
                        else:
                            return {
                                'name': candidate,
                                'confidence': confidence,
                                'face_bbox': (face_x1_orig, face_y1_orig, face_x2_orig, face_y2_orig)
                            }
                    else:
                        return {
                            'name': candidate,
                            'confidence': confidence,
                            'face_bbox': (face_x1_orig, face_y1_orig, face_x2_orig, face_y2_orig)
                        }
                
            except Exception as e:
                print(f"[WARN] Face recognition failed: {e}")
        
        return {'name': 'Unknown', 'confidence': 0.0, 'face_bbox': (face_x1_orig, face_y1_orig, face_x2_orig, face_y2_orig)}
        
    except Exception as e:
        print(f"[ERROR] Face recognition in crop failed: {e}")
        return {'name': 'Unknown', 'confidence': 0.0, 'face_bbox': None}

def update_track_identity(track_id, face_result, track_identities, track_face_history):
    """Update track identity using temporal consistency"""
    name = face_result['name']
    conf = face_result['confidence']
    
    # Initialize tracking data for new track
    if track_id not in track_identities:
        track_identities[track_id] = {
            'name': name,
            'confidence': conf,
            'last_face_frame': 0,
            'stable': False
        }
        track_face_history[track_id] = deque(maxlen=IDENTITY_MEMORY_FRAMES)
    
    # Update face history
    track_face_history[track_id].append({
        'name': name,
        'confidence': conf,
        'frame': 0  # Will be updated by caller
    })
    
    # Current identity data
    current_identity = track_identities[track_id]
    
    if name != 'Unknown':
        # Strong recognition - update identity
        if conf > current_identity['confidence'] * 1.1:  # Require significant improvement
            current_identity['name'] = name
            current_identity['confidence'] = conf
            current_identity['stable'] = True
        elif name == current_identity['name']:
            # Same person - boost confidence slightly
            current_identity['confidence'] = min(1.0, current_identity['confidence'] * 1.02)
            current_identity['stable'] = True
        
        current_identity['last_face_frame'] = 0  # Will be updated by caller
    else:
        # No face detected - decay confidence
        current_identity['confidence'] *= IDENTITY_CONFIDENCE_DECAY
        
        # If confidence too low, mark as unstable
        if current_identity['confidence'] < MIN_IDENTITY_CONFIDENCE:
            current_identity['stable'] = False

def get_consensus_identity(track_id, track_identities, track_face_history, frames_since_face):
    """Get consensus identity from track history"""
    if track_id not in track_identities:
        return 'Unknown', 0.0
    
    identity = track_identities[track_id]
    
    # If recently lost face, check if we should maintain identity
    if frames_since_face > FACE_LOST_TOLERANCE:
        if not identity['stable'] or identity['confidence'] < MIN_IDENTITY_CONFIDENCE:
            return 'Unknown', 0.0
    
    # Use temporal consensus from history
    if track_id in track_face_history:
        history = track_face_history[track_id]
        if len(history) > 0:
            # Weight recent detections more heavily
            name_scores = defaultdict(float)
            total_weight = 0
            
            for i, record in enumerate(history):
                if record['name'] != 'Unknown':
                    # Recent detections get higher weight
                    age_weight = (0.95 ** (len(history) - i - 1))
                    weight = record['confidence'] * age_weight
                    name_scores[record['name']] += weight
                    total_weight += weight
            
            if name_scores:
                best_name = max(name_scores, key=name_scores.get)
                consensus_conf = name_scores[best_name] / max(total_weight, 1)
                
                # Apply confidence penalty for lost face
                if frames_since_face > 30:
                    penalty = min(0.5, frames_since_face / FACE_LOST_TOLERANCE)
                    consensus_conf *= (1.0 - penalty)
                
                return best_name, consensus_conf
    
    return identity['name'], identity['confidence']

# -------------------- THREADS --------------------
def grab_frames(cap, frame_q, stop_event):
    frame_num = 0
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            stop_event.set()
            break
        ts = datetime.now()
        try:
            frame_q.put((frame_num, ts, frame), timeout=0.05)
            frame_num += 1
        except:
            # queue full -> drop frame
            pass
    cap.release()

def process_frames_with_bytetrack(frame_q, display_q, stop_event):
    """Enhanced frame processing using YOLO ByteTrack"""
    
    # ByteTrack tracking data
    track_identities = {}  # track_id -> identity info
    track_face_history = defaultdict(lambda: deque(maxlen=IDENTITY_MEMORY_FRAMES))
    track_last_face_frame = {}  # track_id -> last frame with face
    known_last_saved = {}  # For saving faces
    
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
            
            # Resize for processing if needed
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
                imgsz=640
            )
            
            if results and results[0].boxes is not None:
                boxes = results[0].boxes
                
                if boxes.id is not None:
                    track_ids = boxes.id.cpu().numpy().astype(int)
                    bboxes = boxes.xyxy.cpu().numpy()
                    confidences = boxes.conf.cpu().numpy()
                    
                    for track_id, bbox, conf in zip(track_ids, bboxes, confidences):
                        # Convert bbox back to original frame coordinates
                        if ratio != 1.0:
                            bbox = bbox / ratio
                        
                        x1, y1, x2, y2 = bbox.astype(int)
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(orig_w, x2), min(orig_h, y2)
                        
                        person_bbox = (x1, y1, x2, y2)
                        
                        # Extract person crop for face recognition
                        person_crop = original_frame[y1:y2, x1:x2]
                        
                        # Perform face recognition
                        face_result = recognize_face_in_crop(person_crop, original_frame, person_bbox)
                        
                        # Update tracking history
                        update_track_identity(track_id, face_result, track_identities, track_face_history)
                        
                        # Update frame numbers
                        if face_result['name'] != 'Unknown':
                            track_last_face_frame[track_id] = frame_num
                        
                        # Update tracking records with current frame
                        if track_face_history[track_id]:
                            track_face_history[track_id][-1]['frame'] = frame_num
                        
                        if track_id in track_identities:
                            track_identities[track_id]['last_face_frame'] = track_last_face_frame.get(track_id, 0)
                        
                        # Get consensus identity
                        frames_since_face = frame_num - track_last_face_frame.get(track_id, frame_num)
                        identity_name, identity_conf = get_consensus_identity(
                            track_id, track_identities, track_face_history, frames_since_face
                        )
                        
                        # Choose display color
                        if identity_name != 'Unknown':
                            color = (0, 255, 0)  # Green for known
                            thickness = 3
                        else:
                            color = (0, 0, 255)  # Red for unknown
                            thickness = 2
                        
                        # Draw bounding box
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
                        
                        # Draw face box if available
                        if face_result['face_bbox'] is not None:
                            fx1, fy1, fx2, fy2 = face_result['face_bbox']
                            cv2.rectangle(annotated_frame, (fx1, fy1), (fx2, fy2), (255, 255, 0), 1)
                        
                        # Label
                        label = f"ID:{track_id} {identity_name}"
                        if identity_conf > 0:
                            label += f" ({identity_conf:.2f})"
                        
                        # Add face status
                        if frames_since_face > 30:
                            label += f" [{frames_since_face}f]"
                        
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
            
            # Add performance info to frame
            info_text = f"Frame: {frame_num} | FPS: {fps:.1f} | Tracks: {len(track_identities)}"
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
            
            cv2.imshow("ByteTrack FaceNet Recognition", display_frame)
            
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

# -------------------- MAIN --------------------
def main():
    print("[INFO] Starting ByteTrack FaceNet recognition system...")
    
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