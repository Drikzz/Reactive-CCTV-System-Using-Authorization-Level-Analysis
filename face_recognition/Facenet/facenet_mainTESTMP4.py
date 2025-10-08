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
from collections import deque
from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization
from ultralytics import YOLO
from face_recognition.Facenet.facenet_recognition_tracker import RecognitionTracker
from face_recognition.Facenet.facenet_sort_like_tracker import SortLikeTracker

# add repo root to path for utils if needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# -------------------- CONFIG --------------------
USE_WEBCAM = False
VIDEO_PATH = r"C:\Users\Alexa\OneDrive\Documents\Thesis\Reactive-CCTV-System-Using-Authorization-Level-Analysis\Mp4TESTING\ThesisMP4TEST.mp4"
YOLO_MODEL_PATH = "models/YOLOv8/yolov8n.pt"   # change if needed

LOGS_BASE = os.path.join("logs", "FaceNet")
ANNOTATED_BASE = os.path.join("annotated_frames", "FaceNet")
LOGS_KNOWN_DIR = os.path.join(LOGS_BASE, "known")
LOGS_UNKNOWN_DIR = os.path.join(LOGS_BASE, "unknown")
ANNOTATED_KNOWN_DIR = os.path.join(ANNOTATED_BASE, "known")
ANNOTATED_UNKNOWN_DIR = os.path.join(ANNOTATED_BASE, "unknown")

SAVE_FACES = True
RESIZE_WIDTH = 720                 # you said 720p is desired
PROCESS_EVERY_N = 3                # process detection/recognition every N frames (adaptive)
RECOG_THRESHOLD = 0.45             # classifier probability threshold
RECOG_MARGIN = 0.08                # require top-prob - second-prob >= margin to accept a label
RECOG_COSINE_THRESHOLD = 0.60      # stricter cosine similarity for attaching embeddings to existing tracks
FALLBACK_COSINE = 0.65             # fallback cosine threshold to accept centroid match for difficult views
CAPTURE_QUEUE_SIZE = 4
DISPLAY_QUEUE_SIZE = 2
KNOWN_SAVE_INTERVAL_MIN = 3

# Preprocessing / robustness settings
ENABLE_CLAHE = True                # apply CLAHE to person ROI before face detection
CLAHE_CLIP = 2.0
CLAHE_TILE = 8
ENABLE_GAMMA_CORRECTION = True    # auto-adjust gamma when scene is dark
GAMMA_TARGET_MEAN = 100.0         # desired brightness mean (0-255)

# Motion-based skipping (saves CPU when scene is idle)
MOTION_DETECTION = True
MOTION_THRESHOLD = 4000           # number of changed pixels considered 'motion'
MOTION_HISTORY = 5                # smoothing history for motion decisions

# Optical-flow based motion detection inside body crops (for micro-motions)
FLOW_USE_LK = True                # enable Lucas-Kanade per-body optical flow
FLOW_MAX_CORNERS = 50
FLOW_QUALITY = 0.01
FLOW_MIN_DISTANCE = 7
FLOW_REINIT_EVERY = 30            # re-init feature points every N frames
FLOW_MOTION_THRESHOLD = 1.5       # average motion vector length above which we consider motion

# Performance optimization settings
MIN_FACE_SIZE = 30                 # minimum face size to process (pixels)
MAX_FACES_PER_FRAME = 12            # limit faces processed per frame 0.6 
ADAPTIVE_PROCESSING = True          # adjust processing frequency based on load
GPU_BATCH_SIZE = 16                # maximum batch size for GPU processing
FACE_QUALITY_THRESHOLD = 0.8       # minimum MTCNN detection confidence
PERSON_CONF_THRESHOLD = 0.6        # min YOLO confidence to accept a person detection
MIN_PERSON_AREA_FRAC = 0.02        # drop very small person boxes (<2% of frame area)
DRAW_UNKNOWN_GRACE_FRAMES = 12     # draw unknown only if a face was seen in this track within last N frames
# Label-stability tuning: require repeated detections before treating a predicted name as stable
LABEL_HISTORY_LEN = 5
LABEL_CONFIRM_COUNT = 3
# Display persistence settings (frames)
# Increased TTLs to reduce flicker for stationary people (tune as needed)
DISPLAY_TTL = 120   # frames to keep full box after last seen activity (~8s at 30fps)
GHOST_TTL = 0   # extra frames to show a faded/ghost box before removal (~4s)
IOU_KEEP_THRESHOLD = 0.25  # IoU threshold to consider a person detection matching a track
BODY_DISAPPEAR_FRAMES = 120  # frames of consecutive no-person/no-motion before we remove a body box (~4s)
APPEARANCE_MATCH_THRESHOLD = 0.55  # histogram correlation threshold to accept appearance match (0-1)

MODELS_DIR = os.path.join("models", "FaceNet")
SVM_PATH = os.path.join(MODELS_DIR, "facenet_svm.joblib")
LE_PATH = os.path.join(MODELS_DIR, "label_encoder.joblib")
THR_PATH = os.path.join(MODELS_DIR, "distance_threshold.npy")

# Enhanced recognition robustness for pose variations
RECOG_THRESHOLD = 0.45             # Lower threshold for pose variations
RECOG_MARGIN = 0.08                # Smaller margin requirement
RECOG_COSINE_THRESHOLD = 0.60      # More lenient cosine similarity
FALLBACK_COSINE = 0.65             # More lenient fallback

# Profile face detection settings
ENABLE_PROFILE_DETECTION = True    # Enable detection of profile faces
PROFILE_ANGLE_TOLERANCE = 45       # Degrees of head rotation to still attempt recognition

# Temporal consistency for spinning/rotating persons
POSE_MEMORY_FRAMES = 120           # Increased - remember identity longer
CONFIDENCE_DECAY_RATE = 0.99       # Slower decay
MIN_PROFILE_CONFIDENCE = 0.25      # Lowered from 0.35 - more lenient for profiles
BACK_VIEW_MEMORY_FRAMES = 180      # Remember identity longer when facing away
BACK_VIEW_CONFIDENCE_BOOST = 0.2   # Stronger boost
LABEL_PERSIST_FRAMES = 1800        # Much longer persistence (60 seconds at 30fps)

# Enhanced pose detection
ENABLE_BACK_VIEW_TRACKING = True   # Track people even when facing away
BACK_VIEW_CONFIDENCE_BOOST = 0.15  # Boost confidence for recent detections when face not visible

# Multi-angle processing toggles: if True, generate enhanced variants (histogram eq, gamma variants, flips)
# for each detected face to improve recognition under pose/lighting variations.
MULTI_ANGLE_PROCESSING = True

# -------------------- DEVICE & MODELS --------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Using device: {device}")

# YOLO (person detector)
yolo = YOLO(YOLO_MODEL_PATH)

# MTCNN (face detector + extractor) - optimized settings for speed
mtcnn = MTCNN(
    image_size=160, 
    margin=0, 
    keep_all=True, 
    device=device, 
    post_process=False,
    min_face_size=MIN_FACE_SIZE,  # skip very small faces
    thresholds=[0.6, 0.7, 0.7],  # slightly higher thresholds for speed
    factor=0.7  # reduce pyramid levels for speed
)

# Performance monitoring
frame_times = []
last_fps_update = datetime.now()
current_fps = 0.0
processing_load = 0.0

# embedder (InceptionResnetV1) on device
embedder = InceptionResnetV1(pretrained="vggface2").to(device).eval()

# load SVM + encoder + threshold
classifier = None
label_encoder = None
centroids = None
dist_threshold = None

try:
    if not os.path.exists(SVM_PATH) or not os.path.exists(LE_PATH):
        raise FileNotFoundError("SVM or label encoder not found in models/FaceNet")
    classifier = joblib.load(SVM_PATH)
    label_encoder = joblib.load(LE_PATH)
    # Try to load class centroids (optional) to perform an additional distance check
    centroids_path = os.path.join(MODELS_DIR, 'class_centroids.pkl')
    try:
        if os.path.exists(centroids_path):
            centroids = joblib.load(centroids_path)
            # normalize centroids for consistent comparison with normalized embeddings
            for k, v in list(centroids.items()):
                arr = np.asarray(v, dtype=np.float32)
                n = np.linalg.norm(arr) + 1e-10
                centroids[k] = (arr / n)
            print(f"[INFO] Loaded {len(centroids)} class centroids")
        else:
            centroids = None
    except Exception as e:
        print(f"[WARN] Failed to load centroids: {e}")
        centroids = None
    if os.path.exists(THR_PATH):
        dist_threshold = float(np.load(THR_PATH))
        print(f"[INFO] Loaded distance threshold: {dist_threshold:.3f}")
    else:
        dist_threshold = None
        print("[INFO] No distance threshold found; will rely on classifier probability only.")
    print(f"[INFO] Loaded classifier and encoder. classes: {list(label_encoder.classes_)}")
except Exception as e:
    print(f"[ERROR] Failed to load classifier/encoder/threshold: {e}")
    classifier, label_encoder = None, None

# ensure output dirs
if SAVE_FACES:
    for p in [LOGS_KNOWN_DIR, LOGS_UNKNOWN_DIR, ANNOTATED_KNOWN_DIR, ANNOTATED_UNKNOWN_DIR]:
        os.makedirs(p, exist_ok=True)

# helpers
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

def compute_iou(box1, box2):
    """Compute IoU between two bounding boxes (x1,y1,x2,y2)"""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Intersection
    x1_inter = max(x1_1, x1_2)
    y1_inter = max(y1_1, y1_2)
    x2_inter = min(x2_1, x2_2)
    y2_inter = min(y2_1, y2_2)
    
    if x2_inter <= x1_inter or y2_inter <= y1_inter:
        return 0.0
    
    inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y2_2)
    union_area = area1 + area2 - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0

def point_in_box(point, box):
    """Check if a point (x,y) is inside a bounding box (x1,y1,x2,y2)"""
    x, y = point
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2

def get_face_size(bbox):
    """Calculate face area from bounding box"""
    x1, y1, x2, y2 = bbox
    return (x2 - x1) * (y2 - y1)

def filter_quality_faces(face_boxes, face_probs, min_size=MIN_FACE_SIZE, min_prob=FACE_QUALITY_THRESHOLD):
    """Filter faces by size and detection confidence"""
    if face_boxes is None or face_probs is None:
        return None, None
    
    filtered_boxes = []
    filtered_probs = []
    
    for box, prob in zip(face_boxes, face_probs):
        if prob >= min_prob and get_face_size(box) >= min_size * min_size:
            filtered_boxes.append(box)
            filtered_probs.append(prob)
    
    return (np.array(filtered_boxes) if filtered_boxes else None, 
            np.array(filtered_probs) if filtered_probs else None)

def calculate_adaptive_skip(processing_load, base_skip=PROCESS_EVERY_N):
    """Calculate adaptive frame skip based on processing load"""
    if not ADAPTIVE_PROCESSING:
        return base_skip
    
    if processing_load > 0.8:  # High load
        return min(base_skip * 2, 6)
    elif processing_load > 0.6:  # Medium load
        return base_skip + 1
    else:  # Low load
        return max(base_skip - 1, 1)


def apply_clahe_rgb(rgb_image, clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_TILE):
    """Apply CLAHE to an RGB image by operating on the L channel in LAB color space.
    Expects rgb_image (H,W,3) in uint8 or float (0-255). Returns same dtype uint8 RGB.
    """
    if rgb_image is None or rgb_image.size == 0:
        return rgb_image
    # ensure uint8
    img = rgb_image
    if img.dtype != 'uint8':
        img = (np.clip(img, 0.0, 1.0) * 255).astype('uint8')
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(tileGridSize, tileGridSize))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    res = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    return res


def estimate_brightness(gray_img):
    """Return mean brightness of a grayscale image (0-255)."""
    if gray_img is None or gray_img.size == 0:
        return 0.0
    return float(np.mean(gray_img))


def auto_gamma_correction(img, target_mean=GAMMA_TARGET_MEAN, max_gamma=1.8, min_gamma=0.6):
    """Simple gamma correction to move image mean brightness toward target_mean.
    img is BGR uint8. Returns corrected BGR uint8.
    """
    if img is None or img.size == 0:
        return img
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean = estimate_brightness(gray)
    if mean <= 0:
        return img
    # compute gamma as ratio (clamped)
    gamma = float(target_mean) / float(mean)
    gamma = max(min(gamma, max_gamma), min_gamma)
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(256)]).astype('uint8')
    return cv2.LUT(img, table)


def motion_amount(prev_gray, cur_gray):
    """Return number of changed pixels between two grayscale images (after blur).
    Use as a cheap motion detector.
    """
    if prev_gray is None or cur_gray is None:
        return float('inf')
    d = cv2.absdiff(prev_gray, cur_gray)
    _, th = cv2.threshold(d, 25, 255, cv2.THRESH_BINARY)
    return int(np.count_nonzero(th))

# -------------------- ADDITIONAL FUNCTIONS --------------------
def detect_face_pose(face_crop):
    """Enhanced face pose detection with back view handling"""
    try:
        if face_crop is None or face_crop.size == 0:
            return "no_face", 0.0
        
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY) if len(face_crop.shape) == 3 else face_crop
        h, w = gray.shape
        
        if h < 20 or w < 20:  # Too small to analyze
            return "no_face", 0.0
        
        # Simple pose estimation based on brightness distribution
        left_half = gray[:, :w//2]
        right_half = gray[:, w//2:]
        
        left_brightness = np.mean(left_half)
        right_brightness = np.mean(right_half)
        
        brightness_diff = abs(left_brightness - right_brightness)
        brightness_ratio = min(left_brightness, right_brightness) / max(left_brightness, right_brightness, 1)
        
        # Enhanced pose detection
        if brightness_diff > 25:  # Strong asymmetry indicates profile
            if brightness_ratio < 0.7:  # Very asymmetric
                if left_brightness > right_brightness:
                    return "left_profile", brightness_diff
                else:
                    return "right_profile", brightness_diff
            else:  # Moderate asymmetry
                return "partial_profile", brightness_diff
        elif brightness_diff < 10 and brightness_ratio > 0.85:
            # Very symmetric - likely frontal
            return "frontal", brightness_diff
        else:
            # Could be turning or partially visible
            return "turning", brightness_diff
            
    except Exception as e:
        return "unknown", 0.0

def detect_back_view_person(person_crop):
    """Enhanced back view detection"""
    try:
        if person_crop is None or person_crop.size == 0:
            return False, 0.0
        
        gray = cv2.cvtColor(person_crop, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        if h < 50 or w < 30:  # Too small to analyze
            return False, 0.0
        
        # Divide into regions for analysis
        head_region = gray[:h//3, :]  # Top third
        torso_region = gray[h//3:2*h//3, :]  # Middle third
        legs_region = gray[2*h//3:, :]  # Bottom third
        
        back_score = 0.0
        
        # 1. Head region analysis
        head_brightness = np.mean(head_region)
        head_edges = cv2.Canny(head_region, 30, 100)
        head_edge_density = np.count_nonzero(head_edges) / head_edges.size
        
        # Back of head typically has fewer edges than face
        if head_edge_density < 0.08:  # Low edge density
            back_score += 0.3
        
        # 2. Symmetry analysis (faces are less symmetric when viewed from back)
        left_half = gray[:, :w//2]
        right_half = cv2.flip(gray[:, w//2:], 1)  # Flip for comparison
        
        # Resize to match if needed
        min_w = min(left_half.shape[1], right_half.shape[1])
        left_half = left_half[:, :min_w]
        right_half = right_half[:, :min_w]
        
        if left_half.shape == right_half.shape:
            # Calculate structural similarity
            diff = cv2.absdiff(left_half, right_half)
            symmetry_score = 1.0 - (np.mean(diff) / 255.0)
            
            # Back view tends to be more symmetric than front view
            if symmetry_score > 0.6:
                back_score += 0.2
        
        # 3. Clothing/texture analysis
        torso_brightness = np.mean(torso_region)
        
        # Back typically shows more clothing, less skin
        brightness_ratio = head_brightness / max(torso_brightness, 1)
        if 0.9 < brightness_ratio < 1.4:  # Similar brightness (clothing)
            back_score += 0.2
        
        # 4. Overall darkness (clothing darker than face)
        if torso_brightness < head_brightness * 0.8:
            back_score += 0.1
        
        # 5. Hair/head shape analysis
        # Top portion of head region for hair detection
        hair_region = head_region[:h//6, :]  # Top 1/6 of person
        hair_darkness = np.mean(hair_region)
        
        if hair_darkness < head_brightness * 0.7:  # Dark hair region
            back_score += 0.2
        
        is_back_view = back_score > 0.5
        return is_back_view, min(1.0, back_score)
        
    except Exception as e:
        print(f"[WARN] Back view detection failed: {e}")
        return False, 0.0

def get_pose_adjusted_threshold(pose_type):
    """Return an adjusted recognition probability threshold based on detected face pose.
    Uses the global RECOG_THRESHOLD as the base and adjusts it to be more lenient for
    profiles/turning faces and stricter when no face is detected.
    """
    try:
        base = float(RECOG_THRESHOLD)
    except Exception:
        base = 0.45

    if not pose_type:
        return base

    pt = str(pose_type).lower()
    # Frontal faces: keep base threshold
    if pt == "frontal":
        return base
    # Profiles: be more lenient to allow matches from partial/profile views
    if pt in ("left_profile", "right_profile", "partial_profile"):
        return max(0.20, base - 0.10)
    # Turning / partial visibility: slightly more lenient
    if pt == "turning":
        return max(0.18, base - 0.08)
    # No face / unknown: require higher confidence if used (avoid false positives)
    if pt in ("no_face", "unknown"):
        return min(0.95, base + 0.15)
    # Fallback: a slight relaxation
    return max(0.18, base - 0.02)

def enhance_face_for_recognition(face_crop, pose_type):
    """Generate a small set of enhanced face crops for more robust recognition.
    Returns a list of BGR images (at least the original crop). Uses global MULTI_ANGLE_PROCESSING,
    CLAHE_CLIP and CLAHE_TILE settings.
    """
    try:
        if face_crop is None or face_crop.size == 0:
            return []
        
        # keep original
        crops = [face_crop.copy()]
        
        if not MULTI_ANGLE_PROCESSING:
            return crops
        
        # Helper: gamma adjust
        def adjust_gamma(image, gamma):
            invGamma = 1.0 / float(gamma)
            table = (np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(256)])).astype('uint8')
            return cv2.LUT(image, table)
        
        # 1) CLAHE on L channel (if possible)
        try:
            rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=(CLAHE_TILE, CLAHE_TILE))
            cl = clahe.apply(l)
            lab2 = cv2.merge((cl, a, b))
            rgb_eq = cv2.cvtColor(lab2, cv2.COLOR_LAB2RGB)
            bgr_eq = cv2.cvtColor(rgb_eq, cv2.COLOR_RGB2BGR)
            crops.append(bgr_eq)
        except Exception:
            pass
        
        # 2) horizontal flip (helpful for asymmetry / profiles)
        try:
            crops.append(cv2.flip(face_crop, 1))
        except Exception:
            pass
        
        # 3) flipped CLAHE variant
        try:
            if 'bgr_eq' in locals():
                crops.append(cv2.flip(bgr_eq, 1))
        except Exception:
            pass
        
        # 4) small gamma variations to handle lighting
        try:
            crops.append(adjust_gamma(face_crop, 0.85))
            crops.append(adjust_gamma(face_crop, 1.15))
        except Exception:
            pass
        
        # 5) If pose suggests profile, bias towards flipped variants
        try:
            if pose_type in ("left_profile", "right_profile"):
                # duplicate the flip of the crop to increase chance of matching orientation
                flipped = cv2.flip(face_crop, 1)
                crops.append(flipped)
        except Exception:
            pass
        
        # Deduplicate near-identical crops by comparing shapes and simple hashes
        unique = []
        seen = set()
        for c in crops:
            try:
                h, w = c.shape[:2]
                s = (h, w, int(np.mean(c[:, :, 0]) + np.mean(c[:, :, 1]) + np.mean(c[:, :, 2])))
                if s not in seen:
                    seen.add(s)
                    unique.append(c)
            except Exception:
                continue
        
        return unique if unique else [face_crop.copy()]
    except Exception:
        return [face_crop]


# Simple pose memory utilities: track recent labels per display key and compute a decayed consensus.
def update_pose_memory(display_key, label, confidence, pose_memory, frame_num):
    """
    Update an entry in pose_memory for display_key with the observed label and confidence.
    Returns the memory entry for convenience.
    pose_memory is modified in-place and is expected to be a dict.
    """
    try:
        if display_key not in pose_memory:
            pose_memory[display_key] = {
                'counts': {},
                'confidences': {},
                'last_update': frame_num
            }
        mem = pose_memory[display_key]

        # decay existing confidences slightly based on frames since last update
        elapsed = max(0, frame_num - mem.get('last_update', frame_num))
        if elapsed > 0:
            decay_factor = float(CONFIDENCE_DECAY_RATE) ** float(elapsed)
            for k in list(mem.get('confidences', {}).keys()):
                mem['confidences'][k] = mem['confidences'][k] * decay_factor

        # increment counts and update max confidence per label
        mem['counts'][label] = mem['counts'].get(label, 0) + 1
        mem['confidences'][label] = max(mem['confidences'].get(label, 0.0), float(confidence or 0.0))
        mem['last_update'] = frame_num

        return mem
    except Exception:
        # on error, ensure a minimal memory structure
        pose_memory[display_key] = {
            'counts': {label: 1},
            'confidences': {label: float(confidence or 0.0)},
            'last_update': frame_num
        }
        return pose_memory[display_key]


def get_pose_consensus_label(memory, frame_num, allow_back_view=True):
    """Get consensus label from pose memory using weighted voting - FIXED VERSION"""
    if not memory:
        return "Unknown", 0.0
    
    try:
        # Handle the actual memory structure from update_pose_memory_with_back_view
        if 'labels' in memory and hasattr(memory['labels'], '__iter__'):
            labels = list(memory['labels'])
            confidences = list(memory.get('confidences', []))
        else:
            # Fallback for different memory structure
            return "Unknown", 0.0
        
        if not labels:
            return "Unknown", 0.0
        
        # Focus on recent history (last 10 detections)
        recent_count = min(10, len(labels))
        recent_labels = labels[-recent_count:]
        recent_confidences = confidences[-recent_count:] if confidences else [0.5] * recent_count
        
        label_scores = {}
        total_weight = 0
        
        for i, (label, conf) in enumerate(zip(recent_labels, recent_confidences)):
            if label != "Unknown":
                # Recent frames get much higher weight
                age_weight = (CONFIDENCE_DECAY_RATE ** (recent_count - i - 1))
                
                # Boost weight for very recent detections (last 3)
                if i >= recent_count - 3:
                    age_weight *= 1.5
                
                weight = float(conf) * age_weight
                
                if label not in label_scores:
                    label_scores[label] = 0
                label_scores[label] += weight
                total_weight += weight
        
        if not label_scores:
            return "Unknown", 0.0
        
        best_label = max(label_scores, key=label_scores.get)
        best_score = label_scores[best_label] / max(total_weight, 1)
        
        # Boost confidence if label appeared very recently
        if best_label in recent_labels[-2:]:  # Last 2 detections
            best_score = min(1.0, best_score * 1.3)
        
        return best_label, best_score
        
    except Exception as e:
        print(f"[WARN] Consensus calculation failed: {e}")
        return "Unknown", 0.0

def update_pose_memory_with_back_view(display_key, label, confidence, pose_memory_dict, frame_num, has_face=True):
    """Enhanced pose memory that handles back view/no face scenarios"""
    if display_key not in pose_memory_dict:
        pose_memory_dict[display_key] = {
            'labels': deque(maxlen=POSE_MEMORY_FRAMES),
            'confidences': deque(maxlen=POSE_MEMORY_FRAMES),
            'face_visibility': deque(maxlen=POSE_MEMORY_FRAMES),
            'last_update': frame_num,
            'last_face_frame': frame_num if has_face else -1
        }
    
    memory = pose_memory_dict[display_key]
    memory['labels'].append(label)
    memory['confidences'].append(confidence)
    memory['face_visibility'].append(has_face)
    memory['last_update'] = frame_num
    
    if has_face:
        memory['last_face_frame'] = frame_num
    
    return memory

def detect_turning_motion(display_key, current_pose, pose_history_dict, frame_num):
    """Detect if person is turning around based on pose changes"""
    try:
        if display_key not in pose_history_dict:
            pose_history_dict[display_key] = deque(maxlen=10)
        
        pose_history = pose_history_dict[display_key]
        pose_history.append((current_pose, frame_num))
        
        if len(pose_history) < 3:
            return False, 0.0
        
        # Look for pose transitions that indicate turning
        recent_poses = [p[0] for p in list(pose_history)[-5:]]
        
        # Count pose changes
        pose_changes = 0
        for i in range(1, len(recent_poses)):
            if recent_poses[i] != recent_poses[i-1]:
                pose_changes += 1
        
        # High pose change rate indicates turning
        turn_score = pose_changes / max(len(recent_poses) - 1, 1)
        
        # Check for specific turning patterns
        turning_patterns = [
            ['frontal', 'turning', 'left_profile'],
            ['frontal', 'turning', 'right_profile'],
            ['left_profile', 'turning', 'no_face'],
            ['right_profile', 'turning', 'no_face'],
            ['frontal', 'partial_profile', 'no_face']
        ]
        
        for pattern in turning_patterns:
            if len(recent_poses) >= len(pattern):
                if recent_poses[-len(pattern):] == pattern:
                    turn_score += 0.5
                    break
        
        is_turning = turn_score > 0.4
        return is_turning, min(1.0, turn_score)
        
    except Exception as e:
        return False, 0.0

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

def process_frames(frame_q, display_q, stop_event):
    global current_fps, processing_load
    
    processed = 0
    known_last_saved = {}
    
    # Initialize global variables if not set
    current_fps = 0.0
    processing_load = 0.0
    
    # Add missing variables
    name_to_active_key = {}
    
    # Add original frame dimensions (will be updated when first frame is processed)
    orig_w = 640
    orig_h = 480
    
    # Tracking config
    TRACKING_TIMEOUT = 360
    IOU_THRESHOLD = 0.2
    LABEL_PERSIST_FRAMES = TRACKING_TIMEOUT * 4
    
    rt = RecognitionTracker(cosine_threshold=RECOG_COSINE_THRESHOLD, unknown_ttl=TRACKING_TIMEOUT * 2, max_embeddings_per_person=40, iou_threshold=IOU_THRESHOLD)
    person_tracker = SortLikeTracker(iou_threshold=IOU_THRESHOLD, max_age=TRACKING_TIMEOUT)
    
    # Kalman propagation settings
    USE_KALMAN_PROPAGATION = True
    KALMAN_MAX_MISSES = 60
    
    # Performance monitoring
    processing_times = []
    adaptive_skip = PROCESS_EVERY_N
    
    # Motion detection
    prev_gray_full = None
    motion_history = deque(maxlen=MOTION_HISTORY)
    
    # Tracking dictionaries
    track_last_face_frame = {}
    track_last_person_frame = {}
    display_last_activity = {}
    display_last_known = {}
    display_label_history = {}
    display_appearance = {}
    display_prev_crop = {}
    display_flow_state = {}
    display_miss_count = {}
    display_no_motion_count = {}
    unknown_creation_frame = {}
    display_box_cache = {}
    track_kalman = {}
    
    # Pose memory for handling spinning/rotating persons
    pose_memory = {}
    pose_history = {}  # Add this line around line 400 with other tracking dictionaries

    # Drawing helpers
    drawn_boxes = []
    DEDUPE_IOU = 0.80
    UNKNOWN_TTL = 60

    # ---------- drawing helpers ----------
    def smooth_box(prev_box, new_box, alpha=0.95):
        if prev_box is None or new_box is None:
            return new_box
        x1 = int(prev_box[0] * alpha + new_box[0] * (1 - alpha))
        y1 = int(prev_box[1] * alpha + new_box[1] * (1 - alpha))
        x2 = int(prev_box[2] * alpha + new_box[2] * (1 - alpha))
        y2 = int(prev_box[3] * alpha + new_box[3] * (1 - alpha))
        return (x1, y1, x2, y2)

    def draw_text_with_bg(img, text, org, font_scale, color, thickness=2, bg_color=(0,0,0)):
        (w, h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        x, y = org
        cv2.rectangle(img, (x-2, y-h-4), (x+w+2, y+baseline-2), bg_color, -1)
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,0,0), thickness+2, cv2.LINE_AA)
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)

    def draw_stylized_box(img, box, color=(0,200,0), thickness=2, fill_alpha=0.12, corner=10, replace_overlaps=False):
        x1, y1, x2, y2 = box
        x1, y1 = max(0, x1), max(0, y1)
        h, w = img.shape[:2]
        x2, y2 = min(w-1, x2), min(h-1, y2)
        
        if fill_alpha > 0:
            overlay = img.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            cv2.addWeighted(overlay, fill_alpha, img, 1 - fill_alpha, 0, img)

        try:
            box_area = float(max(1, (x2 - x1) * (y2 - y1)))
            for (dx1, dy1, dx2, dy2, dcolor) in list(drawn_boxes):
                iou_v = compute_iou((x1, y1, x2, y2), (dx1, dy1, dx2, dy2))
                if iou_v >= DEDUPE_IOU:
                    if replace_overlaps:
                        try:
                            drawn_boxes.remove((dx1, dy1, dx2, dy2, dcolor))
                        except ValueError:
                            pass
                        continue
                    existing_area = float(max(1, (dx2 - dx1) * (dy2 - dy1)))
                    if existing_area >= box_area:
                        return
                    else:
                        try:
                            drawn_boxes.remove((dx1, dy1, dx2, dy2, dcolor))
                        except ValueError:
                            pass

            tb = compute_box_thickness((x1, y1, x2, y2))
            cv2.rectangle(img, (x1, y1), (x2, y2), color, tb)
            drawn_boxes.append((int(x1), int(y1), int(x2), int(y2), (int(color[0]), int(color[1]), int(color[2]))))
        except Exception:
            pass

    def compute_box_thickness(box, min_th=2, max_th=8):
        try:
            bx1, by1, bx2, by2 = box
            bw = max(1, bx2 - bx1)
            bh = max(1, by2 - by1)
            base = min(bw, bh)
            t = int(max(min_th, min(max_th, max(1, base // 100))))
            return t
        except Exception:
            return min_th

    # Kalman helpers
    def create_kalman_for_box(box):
        try:
            kf = cv2.KalmanFilter(8, 4)
            kf.transitionMatrix = np.eye(8, dtype=np.float32)
            kf.transitionMatrix[0, 4] = 1.0
            kf.transitionMatrix[1, 5] = 1.0
            kf.transitionMatrix[2, 6] = 1.0
            kf.transitionMatrix[3, 7] = 1.0
            kf.measurementMatrix = np.zeros((4, 8), dtype=np.float32)
            kf.measurementMatrix[0, 0] = 1.0
            kf.measurementMatrix[1, 1] = 1.0
            kf.measurementMatrix[2, 2] = 1.0
            kf.measurementMatrix[3, 3] = 1.0
            kf.processNoiseCov = np.eye(8, dtype=np.float32) * 1e-2
            kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 1e-1
            kf.errorCovPost = np.eye(8, dtype=np.float32) * 1.0
            x1, y1, x2, y2 = box
            w = float(max(1.0, x2 - x1))
            h = float(max(1.0, y2 - y1))
            cx = float(x1 + x2) / 2.0
            cy = float(y1 + y2) / 2.0
            state = np.array([[cx], [cy], [w], [h], [0.0], [0.0], [0.0], [0.0]], dtype=np.float32)
            kf.statePost = state
            kf.statePre = state
            return kf
        except Exception:
            return None

    def kalman_predict_box(kf):
        try:
            pred = kf.predict()
            cx = float(pred[0, 0])
            cy = float(pred[1, 0])
            w = float(pred[2, 0])
            h = float(pred[3, 0])
            x1 = int(max(0, cx - w / 2.0))
            y1 = int(max(0, cy - h / 2.0))
            x2 = int(min(orig_w - 1, cx + w / 2.0))
            y2 = int(min(orig_h - 1, cy + h / 2.0))
            return (x1, y1, x2, y2)
        except Exception:
            return None

    def kalman_correct_with_box(kf, box):
        try:
            x1, y1, x2, y2 = box
            w = float(max(1.0, x2 - x1))
            h = float(max(1.0, y2 - y1))
            cx = float(x1 + x2) / 2.0
            cy = float(y1 + y2) / 2.0
            meas = np.array([[cx], [cy], [w], [h]], dtype=np.float32)
            kf.correct(meas)
            return True
        except Exception:
            return False

    # appearance helpers (HSV histogram)
    def compute_hsv_hist(image_bgr, mask=None, bins=(32, 32)):
        try:
            if image_bgr is None or image_bgr.size == 0:
                return None
            hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], mask, bins, [0, 180, 0, 256])
            if hist is None:
                return None
            cv2.normalize(hist, hist)
            return hist.flatten()
        except Exception:
            return None

    def compare_hist(a, b):
        try:
            if a is None or b is None:
                return 0.0
            # use correlation measure (1.0 best)
            return float(cv2.compareHist(a.astype('float32'), b.astype('float32'), cv2.HISTCMP_CORREL))
        except Exception:
            return 0.0
    # MAIN PROCESSING LOOP
    while not stop_event.is_set() or not frame_q.empty():
        try:
            frame_num, ts, frame = frame_q.get(timeout=0.05)
        except Empty:
            continue
        except Exception as e:
            print(f"[ERROR] Frame processing error: {e}")
            continue

        try:
            original_frame = frame.copy()
            orig_h, orig_w = original_frame.shape[:2]
            annotated_frame = original_frame.copy()
            drawn_boxes = []  # Clear per-frame drawn boxes

            process_start = datetime.now()

            # Resize for YOLO speed
            ratio = 1.0
            if RESIZE_WIDTH and orig_w > RESIZE_WIDTH:
                ratio = RESIZE_WIDTH / orig_w
                resized = cv2.resize(original_frame, (RESIZE_WIDTH, int(orig_h * ratio)))
            else:
                resized = original_frame

            # Adaptive frame skipping
            should_process = (frame_num % adaptive_skip == 0)
            
            if not should_process:
                try:
                    display_q.put((frame_num, ts, annotated_frame), timeout=0.01)
                except:
                    pass
                processed += 1
                continue

            # YOLO: detect persons
            people_boxes_resized = []
            try:
                results = yolo(resized)
                if results and getattr(results[0], "boxes", None) is not None:
                    for box in results[0].boxes:
                        cls = int(box.cls[0])
                        if yolo.model.names[cls] == "person":
                            try:
                                conf = float(box.conf[0])
                            except Exception:
                                conf = 1.0
                            if conf < PERSON_CONF_THRESHOLD:
                                continue
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            people_boxes_resized.append((x1, y1, x2, y2))
            except Exception as e:
                print(f"[ERROR] YOLO inference failed: {e}")
                people_boxes_resized = []

            # Map person boxes back to original coords
            people_boxes = []
            min_person_area = int(MIN_PERSON_AREA_FRAC * orig_w * orig_h)
            for (x1, y1, x2, y2) in people_boxes_resized:
                x1o = max(0, int(x1 / ratio))
                y1o = max(0, int(y1 / ratio))
                x2o = min(orig_w - 1, int(x2 / ratio))
                y2o = min(orig_h - 1, int(y2 / ratio))
                if x2o > x1o and y2o > y1o:
                    area = (x2o - x1o) * (y2o - y1o)
                    if area >= min_person_area:
                        people_boxes.append((x1o, y1o, x2o, y2o))

            # Update person tracker
            track_list = person_tracker.update(people_boxes)
            people_boxes = [t["bbox"] for t in track_list]

            # Motion detection
            motion_flag = True
            if MOTION_DETECTION:
                gray_resized = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
                if prev_gray_full is None:
                    motion_flag = True
                else:
                    amt = motion_amount(prev_gray_full, gray_resized)
                    motion_history.append(amt)
                    avg_motion = float(np.mean(motion_history)) if len(motion_history) > 0 else float('inf')
                    motion_flag = (avg_motion >= MOTION_THRESHOLD)
                prev_gray_full = gray_resized

            # Face detection and recognition (enhanced for turning around)
            faces = []
            face_embeddings = []
            faces_by_name = {}
            
            if not MOTION_DETECTION or motion_flag:
                detect_frame = original_frame.copy()
                
                # Apply gamma correction if enabled
                if ENABLE_GAMMA_CORRECTION:
                    gray_check = cv2.cvtColor(detect_frame, cv2.COLOR_BGR2GRAY)
                    mean_brightness = estimate_brightness(gray_check)
                    if mean_brightness < GAMMA_TARGET_MEAN * 0.8:
                        detect_frame = auto_gamma_correction(detect_frame, GAMMA_TARGET_MEAN)
                
                # Enhanced face detection with back view handling
                try:
                    for pb_idx, person_box in enumerate(people_boxes):
                        x1, y1, x2, y2 = person_box
                        
                        # Extract person ROI with padding
                        padding = 20
                        px1 = max(0, x1 - padding)
                        py1 = max(0, y1 - padding)
                        px2 = min(orig_w, x2 + padding)
                        py2 = min(orig_h, y2 + padding)
                        
                        person_crop = detect_frame[py1:py2, px1:px2]
                        if person_crop.size == 0:
                            continue
                        
                        # Check if person might be facing away
                        is_back_view, back_score = detect_back_view_person(person_crop)
                        
                        # Apply CLAHE if enabled
                        if ENABLE_CLAHE:
                            person_crop_rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
                            person_crop_enhanced = apply_clahe_rgb(person_crop_rgb, CLAHE_CLIP, CLAHE_TILE)
                            person_crop = cv2.cvtColor(person_crop_enhanced, cv2.COLOR_RGB2BGR)
                        
                        # Convert to RGB for MTCNN
                        person_rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
                        
                        # Detect faces with more lenient settings for profiles
                        faces_found = False
                        try:
                            face_boxes, face_probs = mtcnn.detect(person_rgb)
                            
                            # If back view detected, be more lenient with face detection
                            if is_back_view and (face_boxes is None or len(face_boxes) == 0):
                                # Try with lower thresholds for back/profile views
                                mtcnn_lenient = MTCNN(
                                    image_size=160, 
                                    margin=0, 
                                    keep_all=True, 
                                    device=device, 
                                    post_process=False,
                                    min_face_size=20,  # Lower minimum face size
                                    thresholds=[0.5, 0.6, 0.6],  # Lower thresholds
                                    factor=0.8
                                )
                                face_boxes, face_probs = mtcnn_lenient.detect(person_rgb)
                            
                            face_boxes, face_probs = filter_quality_faces(face_boxes, face_probs)
                            
                            if face_boxes is not None and len(face_boxes) > 0:
                                num_faces = min(len(face_boxes), MAX_FACES_PER_FRAME)
                                
                                for i in range(num_faces):
                                    face_box = face_boxes[i]
                                    face_prob = face_probs[i]
                                    
                                    # Convert coordinates back to original frame
                                    fx1, fy1, fx2, fy2 = face_box
                                    fx1_orig = int(px1 + fx1)
                                    fy1_orig = int(py1 + fy1)
                                    fx2_orig = int(px1 + fx2)
                                    fy2_orig = int(py1 + fy2)
                                    
                                    # Ensure within bounds
                                    fx1_orig = max(0, min(fx1_orig, orig_w - 1))
                                    fy1_orig = max(0, min(fy1_orig, orig_h - 1))
                                    fx2_orig = max(0, min(fx2_orig, orig_w - 1))
                                    fy2_orig = max(0, min(fy2_orig, orig_h - 1))
                                    
                                    if fx2_orig <= fx1_orig or fy2_orig <= fy1_orig:
                                        continue
                                    
                                    face_info = {
                                        "bbox": (fx1_orig, fy1_orig, fx2_orig, fy2_orig),
                                        "prob": face_prob,
                                        "person_box_idx": pb_idx,
                                        "name": "Unknown",
                                        "is_back_view": is_back_view,
                                        "back_score": back_score
                                    }
                                    faces.append(face_info)
                                    faces_found = True
                                    
                        except Exception as e:
                            print(f"[WARN] Face detection failed for person box {pb_idx}: {e}")
                            continue
                        
                        # If no face found but person detected, still create entry for tracking continuity
                        if not faces_found and ENABLE_BACK_VIEW_TRACKING:
                            # Create a placeholder face entry for back view tracking
                            face_info = {
                                "bbox": (x1, y1, x2, y2),  # Use full person box
                                "prob": 0.5,  # Moderate confidence
                                "person_box_idx": pb_idx,
                                "name": "Unknown",
                                "is_back_view": True,
                                "back_score": back_score if is_back_view else 0.5,
                                "no_face_detected": True
                            }
                            faces.append(face_info)
                        
                except Exception as e:
                    print(f"[ERROR] Enhanced face detection pipeline failed: {e}")
                    faces = []
                
                # Face recognition with pose handling
                if len(faces) > 0 and classifier is not None and label_encoder is not None:
                    try:
                        face_crops = []
                        valid_face_indices = []
                        face_poses = []
                        
                        for i, face_info in enumerate(faces):
                            fx1, fy1, fx2, fy2 = face_info["bbox"]
                            face_crop = original_frame[fy1:fy2, fx1:fx2]
                            
                            if face_crop.size > 0:
                                # Detect pose for this face
                                pose_type, pose_confidence = detect_face_pose(face_crop)
                                face_poses.append((pose_type, pose_confidence))
                                
                                # Generate enhanced versions based on pose
                                enhanced_crops = enhance_face_for_recognition(face_crop, pose_type)
                                
                                for enhanced_crop in enhanced_crops:
                                    face_rgb = cv2.cvtColor(enhanced_crop, cv2.COLOR_BGR2RGB)
                                    face_resized = cv2.resize(face_rgb, (160, 160))
                                    face_crops.append(face_resized)
                                    valid_face_indices.append((i, pose_type))
                        
                        if len(face_crops) > 0:
                            # Convert to tensors
                            face_tensors = []
                            for crop in face_crops:
                                tensor = torch.from_numpy(crop).permute(2, 0, 1).float()
                                tensor = fixed_image_standardization(tensor)
                                face_tensors.append(tensor)
                            
                            # Batch process embeddings
                            batch_size = min(GPU_BATCH_SIZE, len(face_tensors))
                            all_embeddings = []
                            
                            with torch.no_grad():
                                for i in range(0, len(face_tensors), batch_size):
                                    batch = torch.stack(face_tensors[i:i+batch_size]).to(device)
                                    embeddings = embedder(batch)
                                    all_embeddings.extend(embeddings.cpu().numpy())
                            
                            # Process embeddings by original face (combine results from enhanced versions)
                            face_results = {}
                            for idx, embedding in enumerate(all_embeddings):
                                face_idx, pose_type = valid_face_indices[idx]
                                
                                if face_idx not in face_results:
                                    face_results[face_idx] = []
                                
                                # Normalize embedding
                                emb_norm = embedding / (np.linalg.norm(embedding) + 1e-10)
                                
                                # Get classifier prediction
                                probs = classifier.predict_proba([emb_norm])[0]
                                pred = np.argmax(probs)
                                confidence = probs[pred]
                                
                                # Get second highest probability
                                sorted_probs = np.sort(probs)[::-1]
                                top2_prob = sorted_probs[1] if len(sorted_probs) > 1 else 0.0
                                
                                # Adjust threshold based on pose
                                pose_threshold = get_pose_adjusted_threshold(pose_type)
                                pose_margin = RECOG_MARGIN * (0.7 if pose_type != "frontal" else 1.0)
                                
                                # Accept prediction with pose-adjusted thresholds
                                accept_based_on_prob = (confidence >= pose_threshold and (confidence - top2_prob) >= pose_margin)
                                
                                # Enhanced fallback for profile faces
                                borderline_accept = False
                                if not accept_based_on_prob and pose_type != "frontal":
                                    if confidence >= (pose_threshold - 0.15):
                                        borderline_accept = True
                                elif not accept_based_on_prob and confidence >= (pose_threshold - 0.1):
                                    borderline_accept = True
                                
                                candidate_name = "Unknown"
                                if accept_based_on_prob or borderline_accept:
                                    candidate = label_encoder.inverse_transform([pred])[0]
                                    
                                    # Distance check with more lenient thresholds for profiles
                                    if centroids is not None and dist_threshold is not None:
                                        centroid = centroids.get(candidate)
                                        if centroid is not None:
                                            dist = compute_embedding_distance(emb_norm, centroid)
                                            # Adjust distance threshold for pose
                                            adjusted_threshold = dist_threshold * (1.2 if pose_type != "frontal" else 1.0)
                                            
                                            if dist <= adjusted_threshold:
                                                candidate_name = candidate
                                            else:
                                                if frame_num % 30 == 0:
                                                    print(f"[DEBUG] {candidate} ({pose_type}) rejected by distance {dist:.3f} > {adjusted_threshold:.3f}")
                                        else:
                                            candidate_name = candidate
                                    else:
                                        candidate_name = candidate
                                
                                face_results[face_idx].append({
                                    'name': candidate_name,
                                    'confidence': confidence,
                                    'pose_type': pose_type,
                                    'embedding': emb_norm
                                })
                            
                            # Combine results for each original face (take best result)
                            for face_idx, results in face_results.items():
                                if not results:
                                    continue
                                
                                # Sort by confidence and take best non-unknown result
                                known_results = [r for r in results if r['name'] != "Unknown"]
                                if known_results:
                                    best_result = max(known_results, key=lambda x: x['confidence'])
                                else:
                                    best_result = max(results, key=lambda x: x['confidence'])
                                
                                # Update face info with best result
                                faces[face_idx]["name"] = best_result['name']
                                faces[face_idx]["confidence"] = best_result['confidence']
                                faces[face_idx]["pose_type"] = best_result['pose_type']
                                face_embeddings.append(best_result['embedding'])
                                
                                # Group faces by name
                                name = best_result['name']
                                if name not in faces_by_name:
                                    faces_by_name[name] = []
                                faces_by_name[name].append(faces[face_idx])
                                
                                # Debug logging
                                if frame_num % 30 == 0:
                                    pose_info = best_result['pose_type']
                                    print(f"[DEBUG] Frame {frame_num}: Face recognized as {name} ({pose_info}, conf: {best_result['confidence']:.3f})")
                                    
                    except Exception as e:
                        print(f"[ERROR] Face recognition failed: {e}")
                        face_embeddings = []

            # Update Recognition Tracker
            try:
                face_to_person = {}
                for i, face_info in enumerate(faces):
                    pb_idx = face_info.get("person_box_idx")
                    if pb_idx is not None:
                        face_to_person[i] = pb_idx
                
                tracked_persons, unknown_person_boxes, face_display_names = rt.process_frame(
                    people_boxes=people_boxes,
                    faces=faces,
                    embeddings=face_embeddings,
                    face_to_person=face_to_person,
                    frame_num=frame_num,
                )
            except Exception as e:
                print(f"[ERROR] Recognition tracker update failed: {e}")
                tracked_persons = {}
                unknown_person_boxes = set()
                face_display_names = {}
            
            # Draw Results - Enhanced single box per person
            for idx, tr in enumerate(track_list):
                tb = tr.get('bbox')
                tid = tr.get('track_id', None)
                predicted_flag = tr.get('predicted', False)
                
                if tb is None:
                    continue
                
                # Find matched person ID
                matched_pid = None
                best_iou = 0.0
                try:
                    for pid, td in (tracked_persons or {}).items():
                        iou_v = compute_iou(tb, td.get('bbox', (0,0,0,0)))
                        if iou_v > best_iou and iou_v >= IOU_THRESHOLD:
                            best_iou = iou_v
                            matched_pid = pid
                except Exception:
                    matched_pid = None
                
                # Determine display key
                if matched_pid is not None:
                    display_key = f"pid_{matched_pid}"
                elif tid is not None:
                    display_key = f"tid_{tid}"
                else:
                    display_key = f"pb_{idx}"
                
                # Check for person detection activity
                person_detected_now = False
                for pb in people_boxes:
                    if compute_iou(pb, tb) >= IOU_KEEP_THRESHOLD:
                        person_detected_now = True
                        break
                
                if tid is not None and person_detected_now:
                    track_last_person_frame[tid] = frame_num
                
                # Update activity tracking
                if person_detected_now and not predicted_flag:
                    display_last_activity[display_key] = frame_num
                    display_miss_count[display_key] = 0
                else:
                    display_miss_count[display_key] = display_miss_count.get(display_key, 0) + 1
                
                # Enhanced label determination with better turning around support
                label = 'Unknown'
                current_confidence = 0.0
                
                # Check if we have a face detection for this track
                has_face = False
                face_info = None
                for face in faces:
                    if face.get("person_box_idx") == idx:
                        has_face = True
                        face_info = face
                        break
                
                # Get initial label from recognition tracker
                if matched_pid is not None:
                    try:
                        tdata = (tracked_persons or {}).get(matched_pid, {})
                        label = tdata.get('stable_name') or tdata.get('name') or 'Unknown'
                        current_confidence = tdata.get('confidence', 0.0)
                    except Exception:
                        label = 'Unknown'
                else:
                    # Check recent history
                    lk = display_last_known.get(display_key)
                    if lk is not None:
                        last_label, last_seen = lk
                        frames_since_last = frame_num - last_seen
                        if frames_since_last <= LABEL_PERSIST_FRAMES:
                            label = last_label
                            current_confidence = max(0.3, 0.8 - (frames_since_last / LABEL_PERSIST_FRAMES) * 0.5)

                # Enhanced pose memory with turning around support
                memory_updated = False
                
                if label != 'Unknown' and current_confidence > 0:
                    # Update memory with current detection
                    memory = update_pose_memory_with_back_view(
                        display_key, label, current_confidence, pose_memory, frame_num, has_face
                    )
                    memory_updated = True
                    
                    # Get consensus from memory
                    consensus_label, consensus_confidence = get_pose_consensus_label(memory, frame_num, allow_back_view=True)
                    
                    # Use consensus if it's strong enough
                    if consensus_confidence > max(0.4, current_confidence * 0.6):
                        if consensus_label != label:
                            print(f"[DEBUG] Consensus override: {label} -> {consensus_label} (conf: {consensus_confidence:.3f})")
                        label = consensus_label
                        current_confidence = max(current_confidence, consensus_confidence)
                
                elif display_key in pose_memory:
                    # No current detection, but check memory for recent strong detections
                    memory = pose_memory[display_key]
                    consensus_label, consensus_confidence = get_pose_consensus_label(memory, frame_num, allow_back_view=True)
                    
                    frames_since_update = frame_num - memory.get('last_update', frame_num)
                    frames_since_face = frame_num - memory.get('last_face_frame', -999)
                    
                    # Use memory if:
                    # 1. Strong recent consensus
                    # 2. Recent update (within back view memory)
                    # 3. Had face detection not too long ago
                    if (consensus_confidence > 0.4 and 
                        frames_since_update < BACK_VIEW_MEMORY_FRAMES and
                        frames_since_face < BACK_VIEW_MEMORY_FRAMES * 1.5):
                        
                        label = consensus_label
                        current_confidence = consensus_confidence + BACK_VIEW_CONFIDENCE_BOOST
                        print(f"[DEBUG] Using memory consensus for {display_key}: {label} (conf: {consensus_confidence:.3f}, face_age: {frames_since_face})")
                        
                        # Update memory even with no current face
                        memory = update_pose_memory_with_back_view(
                            display_key, label, current_confidence, pose_memory, frame_num, False
                        )
                        memory_updated = True
                
                # Fallback: check if this might be a brief turn-around
                if label == 'Unknown' and display_key in display_last_known:
                    last_label, last_seen = display_last_known[display_key]
                    frames_since_known = frame_num - last_seen
                    
                    # If we recently knew this person and they just turned around, keep the label temporarily
                    if (last_label != 'Unknown' and 
                        frames_since_known < 60 and  # Within last 2 seconds at 30fps
                        person_detected_now):  # Still detecting the person body
                        
                        label = last_label
                        current_confidence = max(0.2, 0.6 - (frames_since_known / 60) * 0.4)
                        print(f"[DEBUG] Brief turn-around detected, keeping {label} for {display_key} (age: {frames_since_known})")
                        
                        # Update memory to maintain continuity
                        if not memory_updated:
                            memory = update_pose_memory_with_back_view(
                                display_key, label, current_confidence, pose_memory, frame_num, False
                            )

                # Update label history
                if display_key not in display_label_history:
                    display_label_history[display_key] = deque(maxlen=LABEL_HISTORY_LEN)
                
                display_label_history[display_key].append(label)
                
                # Use voting for stable labels
                if len(display_label_history[display_key]) >= LABEL_CONFIRM_COUNT:
                    label_counts = {}
                    for lbl in display_label_history[display_key]:
                        if lbl != "Unknown":
                            label_counts[lbl] = label_counts.get(lbl, 0) + 1
                    
                    if label_counts:
                        stable_label = max(label_counts, key=label_counts.get)
                        if label_counts[stable_label] >= LABEL_CONFIRM_COUNT:
                            label = stable_label

                # ========== ENHANCED SINGLE BOX PER PERSON LOGIC ==========
                # Skip drawing if this person already has an active box
                if label != 'Unknown':
                    current_active_key = name_to_active_key.get(label)
                    
                    if current_active_key is not None and current_active_key != display_key:
                        # Check if the current active box is still valid
                        active_box_valid = False
                        active_last_activity = display_last_activity.get(current_active_key, -9999)
                        active_age = frame_num - active_last_activity
                        
                        # Consider active box valid if it was seen recently
                        if active_age <= BODY_DISAPPEAR_FRAMES and current_active_key in display_box_cache:
                            active_box_valid = True
                        
                        if active_box_valid:
                            # Skip drawing this box - the person already has an active one
                            print(f"[DEBUG] Skipping duplicate box for {label}: {display_key} (active: {current_active_key})")
                            continue
                        else:
                            # Old box is stale, transfer to new box
                            print(f"[INFO] {label} transferring from stale {current_active_key} to {display_key}")
                            
                            # Remove old box data
                            for d in (display_last_known, display_appearance, display_box_cache,
                                      display_label_history, display_last_activity, display_miss_count,
                                      display_no_motion_count, display_prev_crop, display_flow_state, 
                                      unknown_creation_frame):
                                d.pop(current_active_key, None)
                            
                            # Clean up Kalman if old key was tid-based
                            if current_active_key.startswith("tid_"):
                                try:
                                    old_tid = int(current_active_key.split("_", 1)[1])
                                    track_kalman.pop(old_tid, None)
                                    track_last_face_frame.pop(old_tid, None)
                                    track_last_person_frame.pop(old_tid, None)
                                except Exception:
                                    pass
                        
                    # Update mapping to this box
                    name_to_active_key[label] = display_key
                    display_last_known[display_key] = (label, frame_num)
                
                # Choose colors
                frames_since_seen = display_miss_count.get(display_key, 0)
                
                if label != 'Unknown':
                    # Known person: green shades
                    if frames_since_seen == 0:
                        person_color = (0, 220, 0)
                        thickness = 3
                    elif frames_since_seen <= 10:
                        person_color = (0, 180, 0)
                        thickness = 2
                    else:
                        person_color = (0, 120, 0)
                        thickness = 2
                else:
                    # Unknown person: red shades
                    if frames_since_seen == 0:
                        person_color = (0, 0, 200)
                        thickness = 2
                    else:
                        person_color = (0, 0, 160)
                        thickness = 2
                    
                    if display_key not in unknown_creation_frame:
                        unknown_creation_frame[display_key] = frame_num
                
                # Draw box
                prev = display_box_cache.get(display_key)
                smoothed_tb = smooth_box(prev, tb, alpha=0.92)
                display_box_cache[display_key] = smoothed_tb
                x1, y1, x2, y2 = smoothed_tb
                
                # Compute visual alpha with stricter ghost box prevention
                last_activity = display_last_activity.get(display_key, -9999)
                age = frame_num - last_activity
                miss_count = display_miss_count.get(display_key, 0)
                
                # Stricter conditions for showing boxes
                if miss_count <= BODY_DISAPPEAR_FRAMES // 4:  # Very recent detection
                    visual_alpha = 1.0
                elif miss_count <= BODY_DISAPPEAR_FRAMES // 2:  # Recent detection
                    visual_alpha = 0.8
                elif miss_count <= BODY_DISAPPEAR_FRAMES:  # Still within disappear threshold
                    visual_alpha = 0.6
                else:
                    # For older boxes, be much more restrictive
                    if label == "Unknown":
                        # Unknown boxes disappear faster
                        if age > UNKNOWN_TTL or miss_count > UNKNOWN_TTL:
                            continue  # Skip drawing completely
                        visual_alpha = 0.3
                    else:
                        # Known person boxes can persist a bit longer but with strict conditions
                        if age > DISPLAY_TTL:
                            if age > DISPLAY_TTL + GHOST_TTL or miss_count > BODY_DISAPPEAR_FRAMES * 2:
                                continue  # Skip drawing completely
                            # Only show ghost if there's some recent activity evidence
                            visual_alpha = max(0.1, 1.0 - float(age - DISPLAY_TTL) / float(max(1, GHOST_TTL)))
                            # Additional check: only show ghost every few frames to make it blink
                            if (frame_num % 5) != 0:
                                continue
                        else:
                            visual_alpha = 0.4
                
                # Don't draw boxes with very low alpha
                if visual_alpha < 0.2:
                    continue

                def scale_color(c, alpha):
                    return (int(c[0] * alpha), int(c[1] * alpha), int(c[2] * alpha))

                try:
                    draw_stylized_box(annotated_frame, (x1, y1, x2, y2), 
                                    scale_color(person_color, visual_alpha), 
                                    thickness=max(1, int(thickness * visual_alpha)), 
                                    fill_alpha=0.0, corner=10)
                except Exception:
                    pass

                # Draw label
                if label != 'Unknown':
                    label_text = f"{label}"
                    if tid is not None:
                        label_text = f"#{tid} {label_text}"
                    label_y = max(0, y1 - 12)
                    draw_text_with_bg(annotated_frame, label_text, (x1, label_y), 0.6, person_color, thickness=2, bg_color=(20,20,20))

            # ========== ENHANCED CLEANUP - IMPROVED ==========
            # Clean up stale mappings and ghost boxes more aggressively
            try:
                current_time = frame_num
                stale_names = []
                ghost_keys_to_remove = []
                
                # 1. Clean up name mappings
                for name, active_key in list(name_to_active_key.items()):
                    last_activity = display_last_activity.get(active_key, -9999)
                    age = current_time - last_activity
                    
                    # Remove mapping if box is very old or doesn't exist
                
                for name in stale_names:
                    old_key = name_to_active_key.pop(name, None)
                    if old_key:
                        print(f"[DEBUG] Removed stale mapping for {name}: {old_key}")
                
                # 2. Identify currently active tracks
                active_track_ids = set()
                active_person_ids = set()
                
                for tr in track_list:
                    tid = tr.get('track_id')
                    if tid is not None:
                        active_track_ids.add(tid)
                
                for pid in (tracked_persons or {}):
                    active_person_ids.add(pid)
                
                # 3. Find ghost boxes to remove
                for key in list(display_box_cache.keys()):
                    should_remove = False
                    last_activity = display_last_activity.get(key, -9999)
                    age = current_time - last_activity
                    
                    if key.startswith("tid_"):
                        try:
                            tid = int(key.split("_", 1)[1])
                            # Remove if track no longer exists and hasn't been seen recently
                            if tid not in active_track_ids and age > BODY_DISAPPEAR_FRAMES // 2:
                                should_remove = True
                        except ValueError:
                            should_remove = True
                    
                    elif key.startswith("pid_"):
                        try:
                            pid = int(key.split("_", 1)[1])
                            # Remove if person no longer exists and hasn't been seen recently
                            if pid not in active_person_ids and age > BODY_DISAPPEAR_FRAMES // 2:
                                should_remove = True
                        except ValueError:
                            should_remove = True
                    
                    elif key.startswith("pb_"):
                        # Remove person box keys that are very old
                        if age > BODY_DISAPPEAR_FRAMES:
                            should_remove = True
                    
                    # Additional check: remove any box that has been inactive for too long
                    if age > BODY_DISAPPEAR_FRAMES * 3:
                        should_remove = True
                    
                    # Special check for unknown boxes - remove them faster
                    stored_label = display_last_known.get(key, ("Unknown", 0))[0]
                    if stored_label == "Unknown" and age > UNKNOWN_TTL:
                        should_remove = True
                    
                    if should_remove:
                        ghost_keys_to_remove.append(key)
                
                # 4. Remove ghost boxes
                for key in ghost_keys_to_remove:
                    print(f"[DEBUG] Removing ghost box: {key} (age: {current_time - display_last_activity.get(key, current_time)})")
                    
                    # Remove from all tracking dictionaries
                    for d in (display_last_known, display_appearance, display_box_cache,
                              display_label_history, display_last_activity, display_miss_count,
                              display_no_motion_count, display_prev_crop, display_flow_state, 
                              unknown_creation_frame, pose_memory):
                        d.pop(key, None)
                    
                    # Clean up Kalman filters for track-based keys
                    if key.startswith("tid_"):
                        try:
                            tid = int(key.split("_", 1)[1])
                            track_kalman.pop(tid, None)
                            track_last_face_frame.pop(tid, None)
                            track_last_person_frame.pop(tid, None)
                        except ValueError:
                            pass
                    
                    # Remove from name mapping if it points to this key
                    for name, mapped_key in list(name_to_active_key.items()):
                        if mapped_key == key:
                            name_to_active_key.pop(name, None)
                            print(f"[DEBUG] Cleaned up name mapping for {name}: {key}")
                
            except Exception as e:
                print(f"[WARN] Enhanced cleanup failed: {e}")

            # Performance monitoring
            process_end = datetime.now()
            processing_times.append(process_end - process_start)
            if len(processing_times) > 30:
                processing_times.pop(0)
            
            avg_process_time = sum([t.total_seconds() for t in processing_times]) / len(processing_times)
            processing_load = min(1.0, avg_process_time * 30)
            adaptive_skip = calculate_adaptive_skip(processing_load)

            # Save faces
            if SAVE_FACES:
                save_detected_faces(faces, original_frame, frame_num, ts, faces_by_name, known_last_saved)

            # Put frame in display queue
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

def save_detected_faces(faces, original_frame, frame_num, ts, faces_by_name, known_last_saved):
    """Save detected faces to disk"""
    try:
        for name, face_list in faces_by_name.items():
            if not name or name == "Unknown":
                continue
                
            # Check if enough time has passed since last save for this person
            last_saved = known_last_saved.get(name, datetime.min)
            if (ts - last_saved).total_seconds() < KNOWN_SAVE_INTERVAL_MIN * 60:
                continue
                
            for face_info in face_list:
                try:
                    x1, y1, x2, y2 = face_info["bbox"]
                    face_crop = original_frame[y1:y2, x1:x2]
                    
                   
                    if face_crop.size > 0:
                        filename = f"{name}_{frame_num}_{ts.strftime('%H%M%S')}.jpg"
                        filepath = os.path.join(LOGS_KNOWN_DIR, filename)
                        cv2.imwrite(filepath, face_crop)
                        
                        # Also save annotated frame
                        ann_filepath = os.path.join(ANNOTATED_KNOWN_DIR, filename)
                        cv2.imwrite(ann_filepath, original_frame)
                        
                        known_last_saved[name] = ts
                        break  # Only save one face per person per interval
                except Exception as e:
                    print(f"[WARN] Failed to save face for {name}: {e}")
                    
        # Save unknown faces
        unknown_faces = [f for f in faces if f.get("name") == "Unknown"]
        for i, face_info in enumerate(unknown_faces):
            try:
                x1, y1, x2, y2 = face_info["bbox"]
                face_crop = original_frame[y1:y2, x1:x2]
                
                if face_crop.size > 0:
                    filename = f"unknown_{frame_num}_{i}_{ts.strftime('%H%M%S')}.jpg"
                    filepath = os.path.join(LOGS_UNKNOWN_DIR, filename)
                    cv2.imwrite(filepath, face_crop)
            except Exception as e:
                print(f"[WARN] Failed to save unknown face: {e}")
                
    except Exception as e:
        print(f"[ERROR] Face saving failed: {e}")

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
            
            # Add frame info
            info_text = f"Frame: {frame_num} | FPS: {current_fps:.1f}"
            cv2.putText(display_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
           
            cv2.imshow("FaceNet Recognition", display_frame)
            
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
    global current_fps, processing_load
    
    print("[INFO] Starting FaceNet recognition system...")
    
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
    
    # Set video properties for better performance
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
    process_thread = Thread(target=process_frames, args=(frame_q, display_q, stop_event))
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