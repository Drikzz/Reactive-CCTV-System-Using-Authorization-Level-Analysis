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
from face_recognition.facenet_recognition_tracker import RecognitionTracker
from face_recognition.facenet_sort_like_tracker import SortLikeTracker

# add repo root to path for utils if needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# -------------------- CONFIG --------------------
USE_WEBCAM = True
VIDEO_PATH = "sample_video.mp4"
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
RECOG_THRESHOLD = 0.65             # classifier probability threshold
RECOG_MARGIN = 0.15                # require top-prob - second-prob >= margin to accept a label
RECOG_COSINE_THRESHOLD = 0.75      # stricter cosine similarity for attaching embeddings to existing tracks
CAPTURE_QUEUE_SIZE = 4
DISPLAY_QUEUE_SIZE = 2
KNOWN_SAVE_INTERVAL_MIN = 5

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

# Performance optimization settings
MIN_FACE_SIZE = 40                 # minimum face size to process (pixels)
MAX_FACES_PER_FRAME = 8            # limit faces processed per frame
ADAPTIVE_PROCESSING = True          # adjust processing frequency based on load
GPU_BATCH_SIZE = 16                # maximum batch size for GPU processing
FACE_QUALITY_THRESHOLD = 0.8       # minimum MTCNN detection confidence
PERSON_CONF_THRESHOLD = 0.6        # min YOLO confidence to accept a person detection
MIN_PERSON_AREA_FRAC = 0.02        # drop very small person boxes (<2% of frame area)
DRAW_UNKNOWN_GRACE_FRAMES = 12     # draw unknown only if a face was seen in this track within last N frames
# Display persistence settings (frames)
DISPLAY_TTL = 90   # frames to keep full box after last seen activity (~3s at 30fps)
GHOST_TTL = 60     # extra frames to show a faded/ghost box before removal (~2s)
IOU_KEEP_THRESHOLD = 0.25  # IoU threshold to consider a person detection matching a track
BODY_DISAPPEAR_FRAMES = 60  # frames of consecutive no-person/no-motion before we remove a body box (~2s)
APPEARANCE_MATCH_THRESHOLD = 0.55  # histogram correlation threshold to accept appearance match (0-1)

MODELS_DIR = os.path.join("models", "FaceNet")
SVM_PATH = os.path.join(MODELS_DIR, "facenet_svm.joblib")
LE_PATH = os.path.join(MODELS_DIR, "label_encoder.joblib")
THR_PATH = os.path.join(MODELS_DIR, "distance_threshold.npy")

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
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
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
    processed = 0
    known_last_saved = {}
    # Tracking config (use RecognitionTracker)
    # Increase timeouts for private-office use so a person remains tracked when their face
    # is briefly occluded (e.g., covered). Be cautious: larger values keep identities longer
    # but may produce stale tracks if the person leaves the scene.
    TRACKING_TIMEOUT = 240  # frames to keep tracking without face detection (~8s at 30fps)
    IOU_THRESHOLD = 0.2     # lower IoU to allow association under partial occlusion/misaligned boxes
    # how long to keep a recognized label for a display_key when face evidence disappears
    LABEL_PERSIST_FRAMES = TRACKING_TIMEOUT * 2
    # Recognition tracker: unknown_ttl controls how long a known identity is kept without face evidence.
    # Set unknown_ttl larger than person tracker max_age so identity survives brief detector dropouts.
    rt = RecognitionTracker(cosine_threshold=RECOG_COSINE_THRESHOLD, unknown_ttl=TRACKING_TIMEOUT * 2, max_embeddings_per_person=40, iou_threshold=IOU_THRESHOLD)
    person_tracker = SortLikeTracker(iou_threshold=IOU_THRESHOLD, max_age=TRACKING_TIMEOUT)
    
    # Performance monitoring
    frame_start_time = datetime.now()
    processing_times = []
    adaptive_skip = PROCESS_EVERY_N
    # motion history
    prev_gray_full = None
    motion_history = deque(maxlen=MOTION_HISTORY)
    # per-frame grayscale for per-track motion
    prev_gray_frame = None
    # track id -> last frame num when any face was inside its bbox
    track_last_face_frame = {}
    # track id -> last frame num when YOLO person detection overlapped this track
    track_last_person_frame = {}
    # display bookkeeping: track_id or pb_idx -> last_activity_frame (face or person seen)
    display_last_activity = {}
    # persistent mapping of display_key -> (label, last_seen_frame) so known names survive brief occlusion
    display_last_known = {}
    # persistent appearance descriptors per display_key (HSV histograms)
    display_appearance = {}
    # per-track previous small grayscale crop for motion detection
    display_prev_crop = {}
    # per-track consecutive miss counter (no face, no person overlap, no motion)
    display_miss_count = {}

    # display smoothing cache for person boxes
    display_box_cache = {}  # track_id -> (x1,y1,x2,y2)

    # ---------- drawing helpers ----------
    def smooth_box(prev_box, new_box, alpha=0.6):
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
        # background rectangle
        cv2.rectangle(img, (x-2, y-h-4), (x+w+2, y+baseline-2), bg_color, -1)
        # text (shadow for readability)
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,0,0), thickness+2, cv2.LINE_AA)
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)

    def draw_stylized_box(img, box, color=(0,200,0), thickness=2, fill_alpha=0.12, corner=10):
        x1, y1, x2, y2 = box
        x1, y1 = max(0, x1), max(0, y1)
        h, w = img.shape[:2]
        x2, y2 = min(w-1, x2), min(h-1, y2)
        # translucent fill
        if fill_alpha > 0:
            overlay = img.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            cv2.addWeighted(overlay, fill_alpha, img, 1 - fill_alpha, 0, img)
        # corner accents
        cl = max(6, corner)
        # top-left
        cv2.line(img, (x1, y1), (x1+cl, y1), color, thickness)
        cv2.line(img, (x1, y1), (x1, y1+cl), color, thickness)
        # top-right
        cv2.line(img, (x2, y1), (x2-cl, y1), color, thickness)
        cv2.line(img, (x2, y1), (x2, y1+cl), color, thickness)
        # bottom-left
        cv2.line(img, (x1, y2), (x1+cl, y2), color, thickness)
        cv2.line(img, (x1, y2), (x1, y2-cl), color, thickness)
        # bottom-right
        cv2.line(img, (x2, y2), (x2-cl, y2), color, thickness)
        cv2.line(img, (x2, y2), (x2, y2-cl), color, thickness)
    
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
    while not stop_event.is_set() or not frame_q.empty():
        try:
            frame_num, ts, frame = frame_q.get(timeout=0.05)
        except Empty:
            continue

        original_frame = frame.copy()
        orig_h, orig_w = original_frame.shape[:2]
        annotated_frame = original_frame.copy()

        # Resize for YOLO speed (we map boxes back to original coords)
        ratio = 1.0
        if RESIZE_WIDTH and orig_w > RESIZE_WIDTH:
            ratio = RESIZE_WIDTH / orig_w
            resized = cv2.resize(original_frame, (RESIZE_WIDTH, int(orig_h * ratio)))
        else:
            resized = original_frame

        # Adaptive frame skipping based on processing load
        should_process = (frame_num % adaptive_skip == 0)
        
        # If not the processing frame, reuse last annotated_frame (no heavy ops)
        if not should_process:
            # still push annotated_frame for display (keeps UI smooth)
            try:
                display_q.put((frame_num, ts, annotated_frame), timeout=0.01)
            except:
                pass
            processed += 1
            continue
        
        # Start timing for this processing frame
        process_start = datetime.now()

        # ------------- YOLO: detect persons on resized image -------------
        people_boxes_resized = []
        try:
            results = yolo(resized)
            if results and getattr(results[0], "boxes", None) is not None:
                for box in results[0].boxes:
                    cls = int(box.cls[0])
                    if yolo.model.names[cls] == "person":
                        # confidence filter
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
        # compute minimum absolute person area based on original frame size
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

        # Remove expired tracked persons
    # tracker handles expiration internally (via rt.process_frame)
        # ---------- stabilize person boxes with simple PersonTracker ----------
        # Update tracker with current detections (people_boxes) and use stabilized boxes
        # Always update the person tracker (even with empty detections) so existing tracks
        # are aged correctly and boxes persist when detections temporarily drop out.
        track_list = person_tracker.update(people_boxes)
        # convert back to list of bboxes preserving order of tracks
        people_boxes = [t["bbox"] for t in track_list]
        # ---------- Motion detection: decide if frame should be processed ----------
        do_motion_check = MOTION_DETECTION
        motion_flag = True
        if do_motion_check:
            # compute grayscale of resized frame for motion detection
            gray_resized = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            if prev_gray_full is None:
                motion_flag = True
            else:
                amt = motion_amount(prev_gray_full, gray_resized)
                motion_history.append(amt)
                avg_motion = float(np.mean(motion_history)) if len(motion_history) > 0 else float('inf')
                motion_flag = (avg_motion >= MOTION_THRESHOLD)
            prev_gray_full = gray_resized

        # If motion detection is enabled and no motion is detected, do a light-weight draw
        # Pass: we still want to show existing tracked persons (stationary people) so
        # call the recognition tracker with no faces/embeddings to refresh tracked_persons
        # and then render a simplified person/body overlay from the tracker state.
        if MOTION_DETECTION and not motion_flag:
            try:
                # Update recognition tracker with no new faces so it can age/update persons
                tracked_persons, unknown_person_boxes, face_display_names = rt.process_frame(
                    people_boxes=people_boxes,
                    faces=[],
                    embeddings=None,
                    face_to_person={},
                    frame_num=frame_num,
                )
            except Exception:
                tracked_persons = {}
                unknown_person_boxes = set()

            # Lightweight draw: iterate current tracks and draw smoothed person/body boxes
            active_tracks = track_list if 'track_list' in locals() and track_list is not None else [{'track_id': None, 'bbox': b} for b in people_boxes]

            for idx, tr in enumerate(active_tracks):
                tb = tr.get('bbox')
                tid = tr.get('track_id', None)

                # find matched pid by IoU (small helper here instead of full function)
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

                # minimal activity checks
                person_detected_now = False
                for pb in people_boxes:
                    if compute_iou(pb, tr.get('bbox', (0,0,0,0))) >= IOU_KEEP_THRESHOLD:
                        person_detected_now = True
                        break
                if tid is not None and person_detected_now:
                    track_last_person_frame[tid] = frame_num

                # compute display_key similarly to full path
                if matched_pid is not None:
                    display_key = f"pid_{matched_pid}"
                elif tid is not None:
                    display_key = f"tid_{tid}"
                else:
                    # try reuse
                    best_key = None
                    best_iou_k = 0.0
                    for k, prev_box in display_box_cache.items():
                        try:
                            iou_v = compute_iou(tb, prev_box)
                        except Exception:
                            iou_v = 0.0
                        if iou_v > best_iou_k:
                            best_iou_k = iou_v
                            best_key = k
                    if best_iou_k >= 0.4 and best_key is not None:
                        display_key = best_key
                    else:
                        display_key = f"pb_{idx}"

                # treat active track as activity to keep visible
                has_active_track = (tid is not None) or (matched_pid is not None) or (best_iou >= 0.4 if 'best_iou' in locals() else False)
                if person_detected_now or has_active_track:
                    display_last_activity[display_key] = frame_num
                    display_miss_count[display_key] = 0
                else:
                    display_miss_count[display_key] = display_miss_count.get(display_key, 0) + 1

                # choose label: prefer tracked_persons (matched_pid) -> stable name/name
                # otherwise fall back to display_last_known cache (for brief occlusions)
                label = 'Unknown'
                if matched_pid is not None:
                    try:
                        tdata = (tracked_persons or {}).get(matched_pid, {})
                        label = tdata.get('stable_name') or tdata.get('name') or 'Unknown'
                    except Exception:
                        label = 'Unknown'
                else:
                    lk = display_last_known.get(display_key)
                    if lk is not None:
                        last_label, last_seen = lk
                        if frame_num - last_seen <= LABEL_PERSIST_FRAMES:
                            label = last_label

                # color selection: known -> green shades, unknown -> neutral cyan fallback
                fill_alpha = 0.0
                if label != 'Unknown':
                    # Known stationary person -> emphasize green
                    person_color = (0, 200, 0)
                    thickness = 2
                else:
                    person_color = (60, 180, 200)  # light cyan-ish for stationary fallback
                    thickness = 2

                # Smooth + cache
                prev = display_box_cache.get(display_key)
                smoothed_tb = smooth_box(prev, tb, alpha=0.8)
                display_box_cache[display_key] = smoothed_tb
                x1, y1, x2, y2 = smoothed_tb

                # visual alpha based on miss_count/age like main flow
                last_activity = display_last_activity.get(display_key, -9999)
                age = frame_num - last_activity
                miss_count = display_miss_count.get(display_key, 0)
                if miss_count <= BODY_DISAPPEAR_FRAMES:
                    visual_alpha = 1.0
                    ghost = False
                else:
                    if age <= DISPLAY_TTL:
                        visual_alpha = 1.0
                        ghost = False
                    elif age <= DISPLAY_TTL + GHOST_TTL:
                        visual_alpha = max(0.15, 1.0 - float(age - DISPLAY_TTL) / float(GHOST_TTL))
                        ghost = True
                    else:
                        if (frame_num % 3) == 0:
                            visual_alpha = 0.15
                            ghost = True
                        else:
                            continue

                def scale_color(c, alpha):
                    return (int(c[0] * alpha), int(c[1] * alpha), int(c[2] * alpha))

                try:
                    pw = x2 - x1
                    ph = y2 - y1
                    pad_w = max(8, int(pw * 0.12))
                    extend_down = max(8, int(ph * 0.6))
                    body_x1 = max(0, x1 - pad_w)
                    body_x2 = min(orig_w - 1, x2 + pad_w)
                    body_y1 = max(0, int(y1 + ph * 0.05))
                    body_y2 = min(orig_h - 1, y2 + extend_down)
                    body_box = (body_x1, body_y1, body_x2, body_y2)
                    body_color = (int(person_color[0]*0.45), int(person_color[1]*0.45), int(person_color[2]*0.45))
                    draw_stylized_box(annotated_frame, body_box, scale_color(body_color, visual_alpha), thickness=1, fill_alpha=0.0, corner=8)
                except Exception:
                    pass

                draw_stylized_box(annotated_frame, (x1, y1, x2, y2), scale_color(person_color, visual_alpha), thickness=max(1, int(thickness * visual_alpha)), fill_alpha=0.0, corner=10)
                label_text = f"{label}"
                if tid is not None:
                    label_text = f"#{tid} {label_text}"
                label_y = max(0, y1 - 12)
                draw_text_with_bg(annotated_frame, label_text, (x1, label_y), 0.6, person_color, thickness=2, bg_color=(20,20,20))

            try:
                display_q.put((frame_num, ts, annotated_frame), timeout=0.01)
            except:
                pass
            processed += 1
            continue
        faces = []                # will hold dicts with bbox/prob/distance/name after classification
        aligned_tensors = []      # list of torch tensors (C,H,W) in [0,1] from mtcnn.extract
        aligned_meta = []         # meta for each aligned crop -> (abs_bbox, roi_offset)

        # ------------- MTCNN inside each person ROI (optimized) -------------
        rgb_full = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
        faces_processed = 0
        
        for (px1, py1, px2, py2) in people_boxes:
            # Early termination if we've processed enough faces
            if faces_processed >= MAX_FACES_PER_FRAME:
                break
                
            roi = original_frame[py1:py2, px1:px2]
            if roi.size == 0:
                continue
                
            # Skip very small person ROIs
            roi_area = (px2 - px1) * (py2 - py1)
            if roi_area < MIN_FACE_SIZE * MIN_FACE_SIZE * 4:  # person should be larger than 4x min face
                continue
                
            rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            # Preprocessing for low-light / contrast issues
            if ENABLE_GAMMA_CORRECTION:
                roi = auto_gamma_correction(roi)
                rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            if ENABLE_CLAHE:
                rgb_roi = apply_clahe_rgb(rgb_roi)

            try:
                face_boxes_roi, face_probs = mtcnn.detect(rgb_roi)
                # Filter faces by quality and size
                face_boxes_roi, face_probs = filter_quality_faces(face_boxes_roi, face_probs)
            except Exception as e:
                # Sometimes detect fails on ROI — skip
                print(f"[WARN] MTCNN.detect failed on ROI: {e}")
                face_boxes_roi = None
                face_probs = None

            if face_boxes_roi is None or len(face_boxes_roi) == 0:
                continue

            try:
                # extract aligned crops for this ROI (returns tensor on CPU of shape (N,3,160,160) in [0,1])
                aligned = mtcnn.extract(rgb_roi, face_boxes_roi, save_path=None)
            except Exception as e:
                print(f"[WARN] MTCNN.extract failed: {e}")
                aligned = None

            if aligned is None or aligned.shape[0] == 0:
                continue

            # For each aligned crop, store its tensor and absolute bbox (mapped to original coords)
            for i, box in enumerate(face_boxes_roi):
                # box coords are relative to roi: [x1,y1,x2,y2]
                x1r, y1r, x2r, y2r = map(int, box)
                abs_x1, abs_y1 = px1 + x1r, py1 + y1r
                abs_x2, abs_y2 = px1 + x2r, py1 + y2r
                aligned_tensors.append(aligned[i])                 # tensor shape (3,160,160), dtype float32 [0,1]
                aligned_meta.append(((abs_x1, abs_y1, abs_x2, abs_y2), (px1, py1)))
                # placeholder for faces entry; will fill after embedding/classification
                faces.append({"name": "Unknown", "bbox": (abs_x1, abs_y1, abs_x2, abs_y2), "prob": 0.0, "distance": None})
                faces_processed += 1
                
                # Break if we've reached max faces per frame
                if len(aligned_tensors) >= MAX_FACES_PER_FRAME:
                    break

        # If we have aligned crops, batch them and run embedder (with chunking for large batches)
        if len(aligned_tensors) > 0:
            all_embeddings = []
            
            # Process in chunks to manage GPU memory
            for chunk_start in range(0, len(aligned_tensors), GPU_BATCH_SIZE):
                chunk_end = min(chunk_start + GPU_BATCH_SIZE, len(aligned_tensors))
                chunk_tensors = aligned_tensors[chunk_start:chunk_end]
                
                # stack on CPU then standardize & transfer to device once
                batch = torch.stack(chunk_tensors, dim=0)              # (chunk_size,3,160,160), values in [0,1]
                batch = fixed_image_standardization(batch.to(device))  # normalize [-1,1] style and move to device
                
                with torch.no_grad():
                    chunk_embeddings = embedder(batch).cpu().numpy()   # (chunk_size,512) on CPU for sklearn
                    all_embeddings.append(chunk_embeddings)
                    
                # Clear GPU cache after each chunk
                if device == "cuda":
                    torch.cuda.empty_cache()
            
            # Concatenate all embeddings
            embeddings = np.concatenate(all_embeddings, axis=0) if all_embeddings else np.array([])
            # classify all embeddings in a loop (fast; embeddings produced in batch)
            for i, emb in enumerate(embeddings):
                # normalize embedding (recommended)
                emb_norm = emb / (np.linalg.norm(emb) + 1e-10)

                name = "Unknown"
                confidence = 0.0

                # classifier -> your loaded SVM (variable in file is `classifier`)
                # label_encoder -> your encoder (variable is `label_encoder`)
                if classifier is not None and label_encoder is not None:
                    try:
                        # many sklearn classifiers expect shape (1, -1)
                        probs = classifier.predict_proba(emb_norm.reshape(1, -1))[0]
                        # check margin between top-2 classes to avoid weak/confusing predictions
                        sorted_idx = np.argsort(probs)[::-1]
                        top1_idx = sorted_idx[0]
                        top2_idx = sorted_idx[1] if len(sorted_idx) > 1 else None
                        top1_prob = float(probs[top1_idx])
                        top2_prob = float(probs[top2_idx]) if top2_idx is not None else 0.0
                        confidence = top1_prob
                        pred = classifier.classes_[top1_idx]
                        # Accept prediction only if it clears absolute threshold and margin
                        accept_based_on_prob = (confidence >= RECOG_THRESHOLD and (confidence - top2_prob) >= RECOG_MARGIN)
                        if accept_based_on_prob:
                            candidate = label_encoder.inverse_transform([pred])[0]
                            # if centroids exist and distance threshold is configured, check embedding distance too
                            if centroids is not None and dist_threshold is not None:
                                # normalize embedding for fair comparison
                                emb_norm = emb_norm / (np.linalg.norm(emb_norm) + 1e-10)
                                centroid = centroids.get(candidate)
                                if centroid is not None:
                                    # centroid already normalized on load
                                    dist = float(np.linalg.norm(emb_norm - centroid))
                                    if dist <= dist_threshold:
                                        name = candidate
                                        confidence = confidence
                                    else:
                                        name = "Unknown"
                                else:
                                    name = candidate
                            else:
                                name = candidate
                        else:
                            name = "Unknown"
                    except AttributeError:
                        # classifier doesn't implement predict_proba (e.g. not trained with probability=True)
                        # fallback: use decision_function if available, or treat as Unknown
                        try:
                            # decision_function fallback: use top-2 difference as a proxy for margin
                            scores = classifier.decision_function(emb_norm.reshape(1, -1))[0]
                            if isinstance(scores, (list, tuple, np.ndarray)) and len(scores) > 1:
                                sorted_idx = np.argsort(scores)[::-1]
                                top1_idx = sorted_idx[0]
                                top2_idx = sorted_idx[1] if len(sorted_idx) > 1 else None
                                top1 = float(scores[top1_idx])
                                top2 = float(scores[top2_idx]) if top2_idx is not None else 0.0
                                confidence = top1
                                pred = classifier.classes_[top1_idx]
                                accept_based_on_prob = (confidence >= RECOG_THRESHOLD and (confidence - top2) >= RECOG_MARGIN)
                                if accept_based_on_prob:
                                    candidate = label_encoder.inverse_transform([pred])[0]
                                    if centroids is not None and dist_threshold is not None:
                                        emb_norm = emb_norm / (np.linalg.norm(emb_norm) + 1e-10)
                                        centroid = centroids.get(candidate)
                                        if centroid is not None:
                                            dist = float(np.linalg.norm(emb_norm - centroid))
                                            if dist <= dist_threshold:
                                                name = candidate
                                            else:
                                                name = "Unknown"
                                        else:
                                            name = candidate
                                    else:
                                        name = candidate
                                else:
                                    name = "Unknown"
                            else:
                                name = "Unknown"
                        except Exception:
                            name = "Unknown"
                            confidence = 0.0
                    except Exception as e:
                        # Keep Unknown on any other error
                        print(f"[WARN] classification error: {e}")
                        name = "Unknown"
                        confidence = 0.0

                # write back into your faces structure
                faces[i]["name"] = name
                faces[i]["prob"] = confidence
                faces[i]["distance"] = None   # compute if using centroids

        # ------------- Use RecognitionTracker to update/maintain trackers -------------
        # Build face_to_person map (which face belongs to which person box)
        face_to_person = {}  # face_idx -> person_box_idx
        for i, face in enumerate(faces):
            face_bbox = face["bbox"]
            face_center = ((face_bbox[0] + face_bbox[2]) // 2, (face_bbox[1] + face_bbox[3]) // 2)
            for j, person_box in enumerate(people_boxes):
                if point_in_box(face_center, person_box):
                    face_to_person[i] = j
                    break

        # Ask the recognition tracker to process this frame (it handles expiration/promotion)
        tracker_embeddings = embeddings if ("embeddings" in locals() and embeddings is not None and len(embeddings) > 0) else None
        tracked_persons, unknown_person_boxes, face_display_names = rt.process_frame(
            people_boxes=people_boxes,
            faces=faces,
            embeddings=tracker_embeddings,
            face_to_person=face_to_person,
            frame_num=frame_num,
        )

        # ---------------- draw results and logging ----------------
        unknown_in_frame = False
        faces_by_name = {}

        # Draw face detections using tracker-provided display names (small face boxes)
        for idx, f in enumerate(faces):
            display_name = face_display_names[idx] if idx < len(face_display_names) else f.get("name", "Unknown")
            prob = f.get("prob", 0.0) or 0.0
            x1, y1, x2, y2 = f.get("bbox")

            # If classifier returned Unknown for this face, try to fall back to the
            # last-known person-level label (display_last_known) by mapping the face
            # to a person_box and then to a stable display_key (pid/tid/pb).
            if display_name == "Unknown":
                fallback_label = None
                try:
                    pb_idx = face_to_person.get(idx)
                    if pb_idx is not None and pb_idx < len(people_boxes):
                        pb_box = people_boxes[pb_idx]
                        # try match to tracked_persons by IoU
                        best_pid = None
                        best_iou = 0.0
                        for pid, td in (tracked_persons or {}).items():
                            try:
                                iou_v = compute_iou(pb_box, td.get('bbox', (0,0,0,0)))
                            except Exception:
                                iou_v = 0.0
                            if iou_v > best_iou and iou_v >= IOU_THRESHOLD:
                                best_iou = iou_v
                                best_pid = pid
                        if best_pid is not None:
                            tdata = tracked_persons.get(best_pid, {})
                            fallback_label = tdata.get('stable_name') or tdata.get('name')
                        else:
                            # try match to track_list (tid)
                            best_tid = None
                            best_iou2 = 0.0
                            for tr in (track_list or []):
                                try:
                                    iou_v = compute_iou(pb_box, tr.get('bbox', (0,0,0,0)))
                                except Exception:
                                    iou_v = 0.0
                                if iou_v > best_iou2 and iou_v >= IOU_KEEP_THRESHOLD:
                                    best_iou2 = iou_v
                                    best_tid = tr.get('track_id')
                            if best_tid is not None:
                                lk = display_last_known.get(f"tid_{best_tid}")
                                if lk is not None and frame_num - lk[1] <= LABEL_PERSIST_FRAMES:
                                    fallback_label = lk[0]
                            else:
                                # lastly try cache keys by box IoU
                                best_key = None
                                best_iou3 = 0.0
                                for k, prev_box in display_box_cache.items():
                                    try:
                                        iou_v = compute_iou(pb_box, prev_box)
                                    except Exception:
                                        iou_v = 0.0
                                    if iou_v > best_iou3:
                                        best_iou3 = iou_v
                                        best_key = k
                                if best_key is not None and best_iou3 >= 0.4:
                                    lk = display_last_known.get(best_key)
                                    if lk is not None and frame_num - lk[1] <= LABEL_PERSIST_FRAMES:
                                        fallback_label = lk[0]
                except Exception:
                    fallback_label = None

                if fallback_label:
                    display_name = fallback_label
                    face_color = (0, 200, 0)
                    faces_by_name.setdefault(display_name, []).append(f)
                else:
                    face_color = (0, 0, 200)
                    unknown_in_frame = True
            else:
                face_color = (0, 200, 0)
                faces_by_name.setdefault(display_name, []).append(f)

            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), face_color, 1)
            face_label = display_name if display_name != "Unknown" else "?"
            cv2.putText(annotated_frame, face_label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, face_color, 1)

        # Persist recognized face names to display_last_known so that person-level labels
        # can survive short occlusions or motionless periods. Use the active_tracks
        # computed below to determine a stable display_key for each face.
        try:
            # Ensure active_tracks exists (fallback to people_boxes if not)
            active_tracks = track_list if 'track_list' in locals() and track_list is not None else [{'track_id': None, 'bbox': b} for b in people_boxes]
            for fi, f in enumerate(faces):
                display_name = face_display_names[fi] if fi < len(face_display_names) else f.get("name", "Unknown")
                if display_name == 'Unknown':
                    continue
                pb_idx = face_to_person.get(fi)
                if pb_idx is None:
                    continue
                # Find the track that overlaps this person_box
                for tr_idx, tr in enumerate(active_tracks):
                    try:
                        if compute_iou(tr.get('bbox', (0,0,0,0)), people_boxes[pb_idx]) >= IOU_KEEP_THRESHOLD:
                            matched_pid = find_best_pid(tr.get('bbox', (0,0,0,0)), tracked_persons, iou_thresh=IOU_THRESHOLD)
                            if matched_pid is not None:
                                display_key = f"pid_{matched_pid}"
                            elif tr.get('track_id') is not None:
                                display_key = f"tid_{tr.get('track_id')}"
                            else:
                                display_key = f"pb_{tr_idx}"
                            display_last_known[display_key] = (display_name, frame_num)
                            break
                    except Exception:
                        continue
        except Exception:
            pass

    # Draw unified person-level boxes using the SORT-like tracker output (track_list)
        # Each track in track_list is {'track_id': id, 'bbox': (x1,y1,x2,y2)}
        def find_best_pid(box, tracked_persons_dict, iou_thresh=IOU_THRESHOLD):
            best_pid = None
            best_iou = 0.0
            for pid, td in tracked_persons_dict.items():
                iou_v = compute_iou(box, td.get('bbox', (0,0,0,0)))
                if iou_v > best_iou and iou_v >= iou_thresh:
                    best_iou = iou_v
                    best_pid = pid
            return best_pid

        # If we had a person tracker update, use that list, else reconstruct from people_boxes
        active_tracks = track_list if 'track_list' in locals() and track_list is not None else [{'track_id': None, 'bbox': b} for b in people_boxes]

        # Compute per-track: did we see any face inside this track this frame?
        faces_in_track_idx = {i for i in range(len(faces))}
        for idx, tr in enumerate(active_tracks):
            tb = tr.get('bbox')
            tid = tr.get('track_id', None)
            # Map to recognition tracker pid if available via IoU
            matched_pid = find_best_pid(tb, tracked_persons, iou_thresh=IOU_THRESHOLD)
            label = 'Unknown'
            frames_since_seen = 9999
            # track has face now?
            has_face_now = False
            for fi in range(len(faces)):
                if face_to_person.get(fi) == idx:
                    has_face_now = True
                    break
            if tid is not None and has_face_now:
                track_last_face_frame[tid] = frame_num
            # also update person detection overlap (YOLO person inside track)
            # Find any person_box detection that overlaps this track with IoU>IOU_KEEP_THRESHOLD
            person_detected_now = False
            for pb in people_boxes:
                if compute_iou(pb, tr.get('bbox', (0,0,0,0))) >= IOU_KEEP_THRESHOLD:
                    person_detected_now = True
                    break
            if tid is not None and person_detected_now:
                track_last_person_frame[tid] = frame_num
            
            # Compute consistent display_key early (reuse from cache if possible)
            if matched_pid is not None:
                display_key = f"pid_{matched_pid}"
            elif tid is not None:
                display_key = f"tid_{tid}"
            else:
                # try to find an existing cache key whose bbox overlaps strongly with current tb
                best_key = None
                best_iou = 0.0
                for k, prev_box in display_box_cache.items():
                    try:
                        iou_v = compute_iou(tb, prev_box)
                    except Exception:
                        iou_v = 0.0
                    if iou_v > best_iou:
                        best_iou = iou_v
                        best_key = k
                # reuse if overlap significant
                if best_iou >= 0.4 and best_key is not None:
                    display_key = best_key
                else:
                    display_key = f"pb_{idx}"
            
            # Per-track motion detection: crop the track bbox, downscale+blur, compare to previous
            motion_detected = False
            try:
                tbx1, tby1, tbx2, tby2 = map(int, tr.get('bbox', (0,0,0,0)))
                tbx1 = max(0, tbx1); tby1 = max(0, tby1)
                tbx2 = min(orig_w-1, tbx2); tby2 = min(orig_h-1, tby2)
                if tbx2 > tbx1 and tby2 > tby1:
                    body_crop = original_frame[tby1:tby2, tbx1:tbx2]
                    gray_crop = cv2.cvtColor(body_crop, cv2.COLOR_BGR2GRAY)
                    small = cv2.resize(gray_crop, (32, 32))
                    small = cv2.GaussianBlur(small, (5,5), 0)
                    prev_small = display_prev_crop.get(display_key)
                    if prev_small is not None:
                        diff = cv2.absdiff(prev_small, small)
                        motion_score = float(np.mean(diff))
                        if motion_score > 6.0:
                            motion_detected = True
                    display_prev_crop[display_key] = small
            except Exception:
                motion_detected = False

            # Update display activity: force activity for any active track to keep stationary people visible
            # Active track means: has tid, has matched_pid, or strong IoU overlap with cached box
            has_active_track = (tid is not None) or (matched_pid is not None) or (best_iou >= 0.4 if 'best_iou' in locals() else False)
            
            if has_face_now or person_detected_now or motion_detected or has_active_track:
                display_last_activity[display_key] = frame_num
                display_miss_count[display_key] = 0
                # If we have a matched pid, update the last-known label for this display_key
                if matched_pid is not None:
                    tdata = tracked_persons.get(matched_pid, {})
                    stable_name = tdata.get('stable_name') or tdata.get('name')
                    if stable_name:
                        display_last_known[display_key] = (stable_name, frame_num)
            else:
                display_miss_count[display_key] = display_miss_count.get(display_key, 0) + 1
            if matched_pid is not None:
                tdata = tracked_persons.get(matched_pid, {})
                # Prefer stable_name then name
                label = tdata.get('stable_name') or tdata.get('name') or 'Unknown'
                frames_since_seen = frame_num - tdata.get('last_seen', frame_num)
            else:
                # if no matched pid but we have a recent last-known label for this display_key, reuse it
                lk = display_last_known.get(display_key)
                if lk is not None:
                    last_label, last_seen = lk
                    if frame_num - last_seen <= LABEL_PERSIST_FRAMES:
                        label = last_label
                # If RecognitionTracker flagged this person_box index as unknown, override to Unknown
                if idx in unknown_person_boxes:
                    label = 'Unknown'

            # Choose color/thickness by known vs unknown and recency
            if label != 'Unknown':
                # Known person: green shades (hollow boxes)
                if frames_since_seen == 0:
                    person_color = (0, 220, 0)
                    thickness = 3
                    fill_alpha = 0.0
                elif frames_since_seen <= 10:
                    person_color = (0, 180, 0)
                    thickness = 2
                    fill_alpha = 0.0
                else:
                    person_color = (0, 120, 0)
                    thickness = 2
                    fill_alpha = 0.0
            else:
                # Unknown person: red shades (hollow boxes)
                if frames_since_seen == 0:
                    person_color = (0, 0, 255)  # bright red (BGR)
                    thickness = 3
                    fill_alpha = 0.0
                elif frames_since_seen <= 10:
                    person_color = (0, 0, 200)  # medium red
                    thickness = 2
                    fill_alpha = 0.0
                else:
                    person_color = (0, 0, 120)  # dim red
                    thickness = 1
                    fill_alpha = 0.0

                # Suppress drawing unknown boxes that haven't seen a face recently (likely false positives)
                grace_ok = False
                if has_face_now:
                    grace_ok = True
                elif tid is not None and track_last_face_frame.get(tid) is not None:
                    grace_ok = (frame_num - track_last_face_frame[tid]) <= DRAW_UNKNOWN_GRACE_FRAMES
                # If no face now and no recent face, skip drawing this unknown box
                if not grace_ok:
                    continue

            # Smooth the displayed box using the consistent display_key computed earlier
            smoothed_tb = tb
            prev = display_box_cache.get(display_key)
            smoothed_tb = smooth_box(prev, tb, alpha=0.8)
            display_box_cache[display_key] = smoothed_tb

            x1, y1, x2, y2 = smoothed_tb
            # Decide whether to draw full, ghosted, or skip based on last activity
            # NOTE: do NOT overwrite display_key here — it was computed above and must
            # remain consistent (e.g., 'pid_X' or 'tid_Y' or 'pb_N') so lookups succeed.
            last_activity = display_last_activity.get(display_key, -9999)
            age = frame_num - last_activity

            # Determine visual alpha multipliers for ghosting
            miss_count = display_miss_count.get(display_key, 0)
            if miss_count <= BODY_DISAPPEAR_FRAMES:
                # keep full-strength display while within disappear tolerance
                visual_alpha = 1.0
                ghost = False
            else:
                # fallback to age-based ghosting after we've missed for enough frames
                if age <= DISPLAY_TTL:
                    visual_alpha = 1.0
                    ghost = False
                elif age <= DISPLAY_TTL + GHOST_TTL:
                    visual_alpha = max(0.15, 1.0 - float(age - DISPLAY_TTL) / float(GHOST_TTL))
                    ghost = True
                else:
                    # too old: normally skip, but draw a faint ghost every 3 frames so
                    # the body box and label reappear periodically for visibility
                    if (frame_num % 3) == 0:
                        visual_alpha = 0.15
                        ghost = True
                    else:
                        continue

            # Apply visual_alpha to thickness and body/person colors (scale down brightness)
            def scale_color(c, alpha):
                return (int(c[0] * alpha), int(c[1] * alpha), int(c[2] * alpha))
            # Use a dimmer version of the person_color so known -> green, unknown -> red
            try:
                pw = x2 - x1
                ph = y2 - y1
                pad_w = max(8, int(pw * 0.12))
                extend_down = max(8, int(ph * 0.6))
                body_x1 = max(0, x1 - pad_w)
                body_x2 = min(orig_w - 1, x2 + pad_w)
                body_y1 = max(0, int(y1 + ph * 0.05))
                body_y2 = min(orig_h - 1, y2 + extend_down)
                body_box = (body_x1, body_y1, body_x2, body_y2)
                # dimmer body color based on person_color
                body_color = (int(person_color[0]*0.45), int(person_color[1]*0.45), int(person_color[2]*0.45))
                # draw the body box first so the person box overlays it (hollow)
                draw_stylized_box(annotated_frame, body_box, scale_color(body_color, visual_alpha), thickness=1, fill_alpha=0.0, corner=8)
                # optional small label for body only for Unknown to emphasize it's unlabeled
                if label == 'Unknown' and not ghost:
                    draw_text_with_bg(annotated_frame, 'Body', (body_x1, min(body_y2 + 14, orig_h-6)), 0.45, scale_color(body_color, visual_alpha), thickness=1, bg_color=(10,10,10))
            except Exception:
                pass

            # draw person box with scaled color/width when ghosting
            draw_stylized_box(annotated_frame, (x1, y1, x2, y2), scale_color(person_color, visual_alpha), thickness=max(1, int(thickness * visual_alpha)), fill_alpha=0.0, corner=10)
            label_text = f"{label}"
            if tid is not None:
                label_text = f"#{tid} {label_text}"
            if matched_pid is not None and frames_since_seen > 0:
                label_text += f" ({frames_since_seen}f ago)"
            label_y = max(0, y1 - 12)
            draw_text_with_bg(annotated_frame, label_text, (x1, label_y), 0.6, person_color, thickness=2, bg_color=(20,20,20))

        # Handle unknown face logging
        if unknown_in_frame:
            ts_str = ts.strftime("%Y%m%d_%H%M%S%f")
            try:
                # Save unknown faces for review
                for f in faces:
                    if f.get("name") == "Unknown":
                        x1, y1, x2, y2 = f.get("bbox")
                        face_path = os.path.join(LOGS_UNKNOWN_DIR, f"unknown_{ts_str}_{x1}_{y1}.jpg")
                        crop = original_frame[max(0, y1):min(orig_h, y2), max(0, x1):min(orig_w, x2)]
                        if crop.size > 0:
                            cv2.imwrite(face_path, crop)
                        append_csv(os.path.join(LOGS_UNKNOWN_DIR, "detections.csv"),
                                   ["timestamp", "frame", "file", "x1", "y1", "x2", "y2", "prob", "distance"],
                                   [ts.isoformat(), frame_num, os.path.basename(face_path), x1, y1, x2, y2, 
                                    f"{f.get('prob', 0.0):.3f}", "" if f.get('distance') is None else f"{f.get('distance'):.4f}"])
            except Exception as e:
                print(f"[WARN] Failed saving unknown crops: {e}")

        # Note: Unknown person boxes are now visualized via unified person-level drawing above.

        # save annotated unknown frame if any unknown faces detected
        if unknown_in_frame:
            ts_str = ts.strftime("%Y%m%d_%H%M%S%f")
            try:
                cv2.imwrite(os.path.join(ANNOTATED_UNKNOWN_DIR, f"frame_{frame_num}_{ts_str}.jpg"), annotated_frame)
            except Exception:
                pass

        # save knowns periodically
        if faces_by_name:
            now = ts
            for name, flist in faces_by_name.items():
                last = known_last_saved.get(name)
                if last is None or now - last >= timedelta(minutes=KNOWN_SAVE_INTERVAL_MIN):
                    known_last_saved[name] = now
                    for f in flist:
                        x1, y1, x2, y2 = f["bbox"]
                        append_csv(os.path.join(LOGS_KNOWN_DIR, "detections.csv"),
                                   ["timestamp", "frame", "name", "x1", "y1", "x2", "y2", "prob", "distance"],
                                   [ts.isoformat(), frame_num, name, x1, y1, x2, y2, f"{f['prob']:.3f}", "" if f["distance"] is None else f"{f['distance']:.4f}"])
                    # save annotated frame
                    ts_str = ts.strftime("%Y%m%d_%H%M%S%f")
                    try:
                        cv2.imwrite(os.path.join(ANNOTATED_KNOWN_DIR, f"frame_{frame_num}_{ts_str}.jpg"), annotated_frame)
                    except Exception:
                        pass

        # Performance monitoring and adaptive adjustment
        process_end = datetime.now()
        process_time = (process_end - process_start).total_seconds()
        processing_times.append(process_time)
        
        # Keep only recent processing times for adaptive calculation
        if len(processing_times) > 10:
            processing_times = processing_times[-10:]
        
        # Calculate processing load and adjust adaptive skip
        avg_process_time = sum(processing_times) / len(processing_times)
        target_fps = 30  # target FPS
        processing_load = avg_process_time * target_fps
        adaptive_skip = calculate_adaptive_skip(processing_load)
        
        # Update FPS calculation
        current_time = datetime.now()
        if (current_time - frame_start_time).total_seconds() >= 1.0:
            current_fps = processed / (current_time - frame_start_time).total_seconds()
            frame_start_time = current_time
            processed = 0
            
            # Print performance stats occasionally
            if frame_num % 300 == 0:  # every 10 seconds at 30fps
                print(f"[PERF] FPS: {current_fps:.1f}, Load: {processing_load:.2f}, Skip: {adaptive_skip}, Faces: {len(aligned_tensors)}")

        processed += 1
        try:
            display_q.put((frame_num, ts, annotated_frame), timeout=0.01)
        except:
            try:
                _ = display_q.get_nowait()
                display_q.put((frame_num, ts, annotated_frame), timeout=0.01)
            except:
                pass

    print(f"[INFO] Processed {processed} frames.")


# -------------------- MAIN --------------------
def main():
    cap = cv2.VideoCapture(0 if USE_WEBCAM else VIDEO_PATH)
    frame_q = Queue(maxsize=CAPTURE_QUEUE_SIZE)
    display_q = Queue(maxsize=DISPLAY_QUEUE_SIZE)
    stop_event = Event()

    grab_t = Thread(target=grab_frames, args=(cap, frame_q, stop_event), daemon=True)
    proc_t = Thread(target=process_frames, args=(frame_q, display_q, stop_event), daemon=True)
    grab_t.start()
    proc_t.start()

    print("[INFO] Running hybrid YOLO -> MTCNN -> FaceNet (press 'q' to quit)")

    try:
        while not stop_event.is_set():
            try:
                frame_num, ts, annotated = display_q.get(timeout=0.05)
            except Empty:
                if not grab_t.is_alive() and not proc_t.is_alive():
                    break
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    stop_event.set()
                continue

            cv2.imshow("Hybrid Face Recognition", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                stop_event.set()
                break
    finally:
        stop_event.set()
        grab_t.join(timeout=1.0)
        proc_t.join(timeout=2.0)
        cv2.destroyAllWindows()
        print("[INFO] Exited.")


if __name__ == "__main__":
    main()
