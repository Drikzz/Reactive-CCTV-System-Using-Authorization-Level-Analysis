"""
ArcFace Main Interface - Complete face recognition system
Provides training, inference, and capture functionality for ArcFace model.
Supports webcam and MP4 video input like FaceNet.
"""

import os
import sys
import cv2
import joblib
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.models as models
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from threading import Thread, Event
from queue import Queue, Empty
from collections import deque, defaultdict
import time
import argparse

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)

# Try to import MTCNN for detection
try:
    from facenet_pytorch import MTCNN, InceptionResnetV1
    HAS_MTCNN = True
    print("[INFO] Using MTCNN for face detection")
except ImportError:
    HAS_MTCNN = False
    print("[WARN] MTCNN not available, using Haar cascade")

# Try to import YOLO for tracking
try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False

# Try to import InsightFace for recognition
try:
    from insightface.app import FaceAnalysis
    HAS_INSIGHTFACE = True
except Exception:
    HAS_INSIGHTFACE = False

# -------------------- SIMPLE ARCFACE INTEGRATION --------------------
# --- Config ---
MODELS_DIR = os.path.join("models", "ArcFace")
CLASSIFIER_PATH = os.path.join(MODELS_DIR, "arcface_svm.joblib")
ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoder.joblib")
THRESHOLD_PATH = os.path.join(MODELS_DIR, "distance_threshold.npy")
BACKBONE_PATH = os.path.join(MODELS_DIR, "resnet50_backbone.pt")
CENTROIDS_PATH = os.path.join(MODELS_DIR, "class_centroids.pkl")   # optional centroids file

IMAGE_SIZE = 112
EMBEDDING_SIZE = 512
RECOG_MARGIN = 0.08   # require a margin between top1 and top2 like FaceNet
device = "cuda" if torch.cuda.is_available() else "cpu"

# Enable cuDNN autotuner for best performance on fixed-size inputs
torch.backends.cudnn.benchmark = True
print(f"[INFO] Using device: {device} (torch.cuda.is_available={torch.cuda.is_available()})")

# --- Simple ArcFace Backbone ---
class ArcFaceBackbone(nn.Module):
    def __init__(self, embedding_size=512):
        super().__init__()
        resnet = models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.embedding = nn.Linear(2048, embedding_size)
        self.bn = nn.BatchNorm1d(embedding_size)
        
    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.embedding(x)
        x = self.bn(x)
        return nn.functional.normalize(x, p=2, dim=1)

class SimpleArcFaceRecognizer:
    def __init__(self):
        self.embedder = None
        self.classifier = None
        self.encoder = None
        self.threshold = None
        self.loaded = False
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.centroids = None
    def load_models(self):
        """Load all trained models."""
        try:
            # Check if files exist
            required_files = [BACKBONE_PATH, CLASSIFIER_PATH, ENCODER_PATH, THRESHOLD_PATH]
            missing_files = [f for f in required_files if not os.path.exists(f)]
            if missing_files:
                print(f"❌ Missing ArcFace model files: {[os.path.basename(f) for f in missing_files]}")
                return False

            # DEBUG: file info
            print("🔎 Loading ArcFace files:")
            for f in required_files:
                st = os.stat(f)
                print(f"   {os.path.basename(f)} — {st.st_size:,} bytes — modified: {time.ctime(st.st_mtime)}")

            # Load backbone (use InceptionResnetV1 pretrained like FaceNet)
            # Optionally use insightface if available
            if HAS_INSIGHTFACE:
                # initialize insightface FaceAnalysis (recognition module)
                self.insight = FaceAnalysis(allowed_modules=['detection', 'recognition'])
                # ctx_id=0 for CUDA, -1 for CPU
                ctx_id = 0 if torch.cuda.is_available() else -1
                self.insight.prepare(ctx_id=ctx_id, det_size=(224,224))
                # mark we will use insightface embeddings
                self.use_insightface = True
                self.embedder = None
            else:
                self.embedder = InceptionResnetV1(pretrained='vggface2').to(device).eval()
                self.use_insightface = False

            # Load classifier and encoder
            self.classifier = joblib.load(CLASSIFIER_PATH)
            self.encoder = joblib.load(ENCODER_PATH)
            self.threshold = np.load(THRESHOLD_PATH)

            # Try to load optional centroids (same as FaceNet centroids)
            try:
                if os.path.exists(CENTROIDS_PATH):
                    self.centroids = joblib.load(CENTROIDS_PATH)
                    # ensure centroids are L2-normalized
                    for k, v in list(self.centroids.items()):
                        arr = np.asarray(v, dtype=np.float32)
                        n = np.linalg.norm(arr) + 1e-10
                        self.centroids[k] = (arr / n)
                    print(f"   Loaded {len(self.centroids)} class centroids")
                else:
                    self.centroids = None
                    print("   No centroids file found (optional)")
            except Exception as e:
                print(f"   Failed to load centroids: {e}")
                self.centroids = None

            # DEBUG: print mappings
            print(f"   Classifier.classes_: {getattr(self.classifier, 'classes_', None)}")
            print(f"   LabelEncoder.classes_: {getattr(self.encoder, 'classes_', None)}")
            print(f"   Threshold: {self.threshold:.4f}")

            self.loaded = True
            return True

        except Exception as e:
            print(f"❌ Failed to load ArcFace models: {e}")
            self.loaded = False
            return False

    def recognize_face(self, face_img: np.ndarray) -> tuple:
        """Robust multi-variant recognition:
        - generate a few enhanced variants (CLAHE / flipped / gamma tweaks)
        - batch through the ArcFace backbone
        - average classifier probabilities across variants
        - use centroid distance on the mean embedding as final check (if available)
        """
        if not self.loaded:
            return "Unknown", 0.0

        try:
            # Build variants (at least the original). Keep small set to avoid GPU overload.
            def make_variants(bgr):
                variants = []
                try:
                    orig = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    variants.append(cv2.resize(orig, (IMAGE_SIZE, IMAGE_SIZE)))

                    # CLAHE variant
                    try:
                        lab = cv2.cvtColor(orig, cv2.COLOR_RGB2LAB)
                        l, a, b = cv2.split(lab)
                        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                        cl = clahe.apply(l)
                        merged = cv2.merge((cl, a, b))
                        clahe_rgb = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
                        variants.append(cv2.resize(clahe_rgb, (IMAGE_SIZE, IMAGE_SIZE)))
                    except Exception:
                        pass

                    # Gamma correction modest adjustments
                    for g in (0.8, 1.2):
                        try:
                            invGamma = 1.0 / g
                            table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(256)]).astype('uint8')
                            gamma_img = cv2.LUT(orig.astype('uint8'), table)
                            variants.append(cv2.resize(gamma_img, (IMAGE_SIZE, IMAGE_SIZE)))
                        except Exception:
                            pass

                    # Horizontal flip (helps profiles)
                    try:
                        flipped = cv2.flip(orig, 1)
                        variants.append(cv2.resize(flipped, (IMAGE_SIZE, IMAGE_SIZE)))
                    except Exception:
                        pass
                except Exception:
                    pass
                # Deduplicate by shape+mean quick hash
                unique = []
                seen = set()
                for v in variants:
                    try:
                        key = (v.shape[0], v.shape[1], int(v.mean()))
                        if key not in seen:
                            seen.add(key)
                            unique.append(v)
                    except Exception:
                        continue
                return unique if unique else [cv2.resize(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB), (IMAGE_SIZE, IMAGE_SIZE))]

            variants = make_variants(face_img)

            # If using insightface, obtain embeddings per-variant via its API
            embs = None
            if getattr(self, 'use_insightface', False):
                try:
                    emb_list = []
                    # variants are RGB, insightface expects BGR numpy
                    for v_rgb in variants:
                        v_bgr = cv2.cvtColor(v_rgb, cv2.COLOR_RGB2BGR)
                        e = self._get_insightface_embedding(v_bgr)
                        if e is not None:
                            emb_list.append(e)
                    if len(emb_list) == 0:
                        # insightface failed to produce embeddings for all variants
                        return "Unknown", 0.0
                    embs = np.stack(emb_list, axis=0)  # (N, D)
                except Exception as ex:
                    print(f"❌ InsightFace embedding failed: {ex}")
                    return "Unknown", 0.0
            else:
                # Convert to tensors and run embeddings in a batch with local embedder
                tensors = []
                for v in variants:
                    t = self.transform(v).unsqueeze(0)
                    tensors.append(t)
                batch = torch.cat(tensors, dim=0).to(device)
                # If the embedder parameters are half precision, make inputs half as well
                try:
                    if self.embedder is not None:
                        emb_dtype = next(self.embedder.parameters()).dtype
                        if emb_dtype == torch.float16:
                            batch = batch.half()
                except Exception:
                    pass
                with torch.no_grad():
                    embs = self.embedder(batch).cpu().numpy()  # (N, D)

            # L2-normalize each embedding and compute mean embedding
            embs_norm = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-10)
            mean_emb = np.mean(embs_norm, axis=0)
            mean_emb = mean_emb / (np.linalg.norm(mean_emb) + 1e-10)

            # Get classifier probabilities for each variant (or fallback to predict)
            if hasattr(self.classifier, "predict_proba"):
                probs_list = []
                for e in embs_norm:
                    try:
                        probs = self.classifier.predict_proba([e])[0]
                    except Exception:
                        probs = np.zeros(len(getattr(self.classifier, "classes_", []))) + 1e-9
                    probs_list.append(probs)
                # Average probabilities across variants
                avg_probs = np.mean(np.stack(probs_list, axis=0), axis=0)
                sorted_idx = np.argsort(avg_probs)[::-1]
                top_idx = int(sorted_idx[0])
                top_prob = float(avg_probs[top_idx])
                second_prob = float(avg_probs[sorted_idx[1]]) if len(sorted_idx) > 1 else 0.0
            else:
                # classifier without probabilities -> use single-variant predict on mean emb
                pred = self.classifier.predict([mean_emb])[0]
                classes = list(self.classifier.classes_)
                top_idx = classes.index(pred)
                top_prob = 1.0
                second_prob = 0.0

            # Resolve predicted label robustly
            pred_value = self.classifier.classes_[top_idx]
            name = None
            try:
                if isinstance(pred_value, (int, np.integer)) or (isinstance(pred_value, str) and pred_value.isdigit()):
                    try:
                        key = int(pred_value) if isinstance(pred_value, str) else pred_value
                        name = self.encoder.inverse_transform([key])[0]
                    except Exception:
                        name = None
                if name is None:
                    try:
                        if pred_value in list(self.encoder.classes_):
                            name = pred_value
                    except Exception:
                        pass
                if name is None:
                    try:
                        idx = int(pred_value)
                        name = list(self.encoder.classes_)[idx]
                    except Exception:
                        name = str(pred_value)
            except Exception:
                name = str(pred_value)

            # Apply probability + margin acceptance
            if top_prob < RECOG_THRESHOLD or (top_prob - second_prob) < RECOG_MARGIN:
                return "Unknown", top_prob

            # If centroids available, require centroid agreement:
            if self.centroids is not None and self.threshold is not None:
                # compute distance to every centroid and find nearest
                nearest_name = None
                nearest_dist = float("inf")
                for k, v in self.centroids.items():
                    try:
                        v_arr = np.asarray(v, dtype=np.float32)
                        # centroids are already normalized in load_models
                        d = float(np.linalg.norm(mean_emb - v_arr))
                        if d < nearest_dist:
                            nearest_dist = d
                            nearest_name = k
                    except Exception:
                        continue

                # Accept only when classifier prediction matches nearest centroid AND within threshold
                if nearest_name is None:
                    return "Unknown", top_prob

                if nearest_name != name:
                    # classifier and centroid disagree -> reject to avoid mislabel
                    return "Unknown", top_prob

                # classifier and centroid agree; enforce distance threshold
                if nearest_dist <= float(self.threshold):
                    return name, top_prob
                else:
                    return "Unknown", top_prob

            # no centroids available: accept classifier decision
            return name, top_prob

        except Exception as e:
            print(f"❌ ArcFace recognition failed: {e}")
            return "Unknown", 0.0

    # ------------------ InsightFace helper (added) ------------------
    def _get_insightface_embedding(self, face_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Return L2-normalized embedding using insightface FaceAnalysis (stored on self.insight), or None."""
        if not HAS_INSIGHTFACE:
            return None
        try:
            if not hasattr(self, 'insight') or self.insight is None:
                return None
            # insightface expects BGR numpy images
            faces = self.insight.get(face_bgr)
            if not faces:
                return None
            emb = np.asarray(faces[0].embedding, dtype=np.float32)
            return emb / (np.linalg.norm(emb) + 1e-10)
        except Exception:
            return None

# Small helper to capture faces from a webcam for training
class ArcFaceCapture:
    def __init__(self, detector=None):
        # detector can be an MTCNN instance or None; we will fallback to Haar cascade if needed
        self.detector = detector

    def capture_from_webcam(self, person_name: str, max_images: int = 100, camera_id: int = 0):
        """Capture face crops from the webcam and save them to dataset/<person_name>/."""
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print("[ERROR] Could not open webcam for capture")
            return False

        out_dir = os.path.join("dataset", person_name)
        os.makedirs(out_dir, exist_ok=True)

        saved = 0
        # Prepare a Haar cascade detector if MTCNN is not available / not provided
        cascade = None
        if not HAS_MTCNN:
            cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # If a detector wasn't provided, try to create one (only if MTCNN is available)
        local_mtcnn = None
        if self.detector is None and HAS_MTCNN:
            try:
                local_mtcnn = MTCNN(image_size=160, margin=0, keep_all=False, device=device, post_process=False)
            except Exception:
                local_mtcnn = None

        print(f"[INFO] Starting capture for '{person_name}' - saving up to {max_images} images to {out_dir}")
        try:
            while saved < max_images:
                ret, frame = cap.read()
                if not ret or frame is None:
                    print("[WARN] Failed to read frame from webcam")
                    break

                face_box = None

                # Try MTCNN first if available
                mtcnn_to_use = self.detector or local_mtcnn
                if mtcnn_to_use is not None and HAS_MTCNN:
                    try:
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        boxes, probs, _ = mtcnn_to_use.detect(rgb, landmarks=False)
                        if boxes is not None and len(boxes) > 0:
                            # Use the first detected face
                            x1, y1, x2, y2 = boxes[0].astype(int)
                            face_box = (max(0, x1), max(0, y1), max(0, x2), max(0, y2))
                    except Exception:
                        face_box = None

                # Fallback to Haar cascade detection
                if face_box is None and cascade is not None:
                    try:
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                                         minSize=(MIN_FACE_SIZE, MIN_FACE_SIZE))
                        if len(faces) > 0:
                            x, y, w, h = faces[0]
                            face_box = (x, y, x + w, y + h)
                    except Exception:
                        face_box = None

                # Save detected face crop
                if face_box is not None:
                    x1, y1, x2, y2 = face_box
                    # Clamp coordinates
                    h, w = frame.shape[:2]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    if x2 > x1 and y2 > y1:
                        face_crop = frame[y1:y2, x1:x2]
                        if face_crop is not None and face_crop.size > 0:
                            filename = os.path.join(out_dir, f"{person_name}_{saved:04d}.jpg")
                            cv2.imwrite(filename, face_crop)
                            saved += 1
                            print(f"[INFO] Saved {filename} ({saved}/{max_images})")

                # Show a preview if GUI is available
                if HAS_GUI:
                    try:
                        preview = frame.copy()
                        if face_box is not None:
                            cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.imshow("Capture Faces - Press 'q' to quit", preview)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            print("[INFO] User requested abort")
                            break
                    except Exception:
                        # If display fails, continue headless
                        pass

        except KeyboardInterrupt:
            print("[INFO] Capture interrupted by user")
        finally:
            cap.release()
            if HAS_GUI:
                try:
                    cv2.destroyAllWindows()
                except Exception:
                    pass

        print(f"[INFO] Capture finished - saved {saved} images to {out_dir}")
        return True

# -------------------- REST OF YOUR EXISTING CONFIG --------------------

USE_WEBCAM = False
VIDEO_PATH = r"C:\Users\Alexa\OneDrive\Documents\Thesis\Reactive-CCTV-System-Using-Authorization-Level-Analysis\Mp4TESTING\ThesisMP4TEST3.mp4"
YOLO_MODEL_PATH = "models/YOLOv8/yolov8n.pt"

# Output directories
LOGS_BASE = os.path.join("logs", "ArcFace")
ANNOTATED_BASE = os.path.join("annotated_frames", "ArcFace")
LOGS_KNOWN_DIR = os.path.join(LOGS_BASE, "known")
LOGS_UNKNOWN_DIR = os.path.join(LOGS_BASE, "unknown")
ANNOTATED_KNOWN_DIR = os.path.join(ANNOTATED_BASE, "known")
ANNOTATED_UNKNOWN_DIR = os.path.join(ANNOTATED_BASE, "unknown")

SAVE_FACES = True
RESIZE_WIDTH = 720
PROCESS_EVERY_N = 2
# tuneable recognition threshold (lower for video tests)
RECOG_THRESHOLD = 0.45  # was 0.60 - lower for live/video testing
PERSON_CONF_THRESHOLD = 0.6
MIN_FACE_SIZE = 30
MAX_FACES_PER_FRAME = 12
BYTETRACK_MATCH_THRESH = 0.7

# New: require a minimum face detection confidence and relative size inside person crop
FACE_DET_CONF_THRESHOLD = 0.55    # require detector confidence >= this
FACE_MIN_REL_SIZE = 0.15          # face height must be >= 15% of person crop height
FACE_IOU_THRESHOLD = 0.25        # require face bbox to overlap person box by this IoU
CONFIRM_FACE_FRAMES = 2          # require detections for N frames before accepting recognition

# Identity persistence settings
IDENTITY_MEMORY_FRAMES = 90
IDENTITY_CONFIDENCE_DECAY = 0.995
MIN_IDENTITY_CONFIDENCE = 0.15
FACE_LOST_TOLERANCE = 180
EXTENDED_MEMORY_FRAMES = 600

# Performance settings
CAPTURE_QUEUE_SIZE = 4
DISPLAY_QUEUE_SIZE = 2
KNOWN_SAVE_INTERVAL_MIN = 3

# Check OpenCV GUI support
def has_opencv_gui():
    """Check if OpenCV was built with GUI support."""
    try:
        test_window = "opencv_gui_test"
        cv2.namedWindow(test_window, cv2.WINDOW_NORMAL)
        cv2.destroyWindow(test_window)
        cv2.waitKey(1)
        return True
    except Exception:
        return False

HAS_GUI = has_opencv_gui()
print(f"[DEBUG] HAS_GUI: {HAS_GUI}")

# -------------------- REPLACE YOUR ARCFACESYSTEM CLASS --------------------
class ArcFaceSystem:
    """Simple ArcFace face recognition system."""
    
    def __init__(self, model_path=None, config_path=None, device=None, use_yolo=True):
        self.detector = None
        self.yolo = None
        
        # Diagnostic flag: when True, always draw predicted name/conf even if final decision is Unknown
        self.always_draw_predictions = False
        
        # Initialize simple ArcFace recognizer
        self.face_recognizer = SimpleArcFaceRecognizer()
        self.face_recognizer_loaded = False
        
        # Initialize detectors
        self._init_detectors()
        
        # Load ArcFace models
        if self.face_recognizer.load_models():
            self.face_recognizer_loaded = True
        else:
            print("⚠️  ArcFace not loaded - will only detect persons")
    
    def _init_detectors(self):
        """Initialize face detector and person tracker."""
        # Initialize MTCNN for face detection
        if HAS_MTCNN:
            self.detector = MTCNN(
                image_size=160,
                margin=0,
                keep_all=True,
                device=device,   # use GPU if available
                post_process=False,
                min_face_size=MIN_FACE_SIZE,
                thresholds=[0.6, 0.7, 0.7]
            )
            print("✅ MTCNN detector initialized")
        else:
            # Fallback to Haar cascade
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.detector = cv2.CascadeClassifier(cascade_path)
            print("✅ Haar cascade detector initialized")
        
        # Initialize YOLO for person tracking
        if HAS_YOLO:
            try:
                self.yolo = YOLO(YOLO_MODEL_PATH)
                print("✅ YOLO tracker initialized")
            except Exception as e:
                print(f"⚠️  YOLO initialization failed: {e}")
                self.yolo = None
    
    def detect_faces(self, image: np.ndarray) -> List[Dict]:
        """Detect faces in image. Returns box, confidence and optional landmarks."""
        detections = []
        
        # Prefer insightface detection if initialized (returns BGR input)
        if getattr(self, 'insight', None) is not None:
            try:
                faces = self.insight.get(image)
                for f in faces:
                    # insightface Face object may expose bbox/kps/score attributes
                    bbox = getattr(f, "bbox", None) or (f.get("bbox") if isinstance(f, dict) else None)
                    score = getattr(f, "det_score", None) or getattr(f, "score", None) or (f.get("score") if isinstance(f, dict) else None) or 1.0
                    kps = getattr(f, "kps", None) or getattr(f, "landmark", None) or (f.get("kps") if isinstance(f, dict) else None)
                    if bbox is None:
                        continue
                    x1, y1, x2, y2 = [int(max(0, v)) for v in bbox]
                    lm = None
                    try:
                        lm = np.asarray(kps).astype(int) if kps is not None else None
                    except Exception:
                        lm = None
                    detections.append({
                        "box": (x1, y1, x2 - x1, y2 - y1),
                        "confidence": float(score),
                        "landmarks": lm
                    })
                return detections
            except Exception:
                # insightface detection failed -> fall back to MTCNN/Haar below
                pass
        

        if HAS_MTCNN and hasattr(self.detector, 'detect'):
            # MTCNN detection
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            try:
                # request landmarks so we can verify face geometry later
                boxes, probs, landmarks = self.detector.detect(image_rgb, landmarks=True)
                
                if boxes is not None:
                    for i, box in enumerate(boxes):
                        x1, y1, x2, y2 = [int(max(0, v)) for v in box]
                        lm = None
                        try:
                            lm = landmarks[i].astype(int) if landmarks is not None else None
                        except Exception:
                            lm = None
                        detections.append({
                            "box": (x1, y1, x2 - x1, y2 - y1),
                            "confidence": float(probs[i]) if probs is not None else 1.0,
                            "landmarks": lm
                        })
            except Exception:
                pass
        else:
            # Haar cascade detection
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.detector.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5,
                minSize=(MIN_FACE_SIZE, MIN_FACE_SIZE)
            )
            
            for (x, y, w, h) in faces:
                detections.append({
                    "box": (x, y, w, h),
                    "confidence": 1.0
                })
        
        return detections
    
    def recognize_face_in_crop_enhanced(self, person_crop: np.ndarray, original_frame: np.ndarray, person_bbox: tuple, frame_num: int) -> Dict:
        """Recognize face within person crop using simple ArcFace.
        Returns {'name','confidence','predicted_name','predicted_confidence','face_bbox'} where
        'name' is the accepted identity (may be 'Unknown') and predicted_* are the raw classifier outputs.
        """
        if not self.face_recognizer_loaded:
            return {'name': 'Unknown', 'confidence': 0.0, 'predicted_name': 'Unknown', 'predicted_confidence': 0.0, 'face_bbox': None}
        
        if person_crop is None or person_crop.size == 0:
            return {'name': 'Unknown', 'confidence': 0.0, 'predicted_name': 'Unknown', 'predicted_confidence': 0.0, 'face_bbox': None}
        
        try:
            # Detect faces in person crop
            detections = self.detect_faces(person_crop)
            debug_save = False

            # If no faces found, try upscaling the crop (faces may be too small for MTCNN)
            if not detections:
                h, w = person_crop.shape[:2]
                if h > 0 and w > 0:
                    scale = 2.0
                    up_h, up_w = min(int(h * scale), 640), min(int(w * scale), 640)
                    try:
                        up = cv2.resize(person_crop, (up_w, up_h), interpolation=cv2.INTER_LINEAR)
                        up_dets = self.detect_faces(up)
                        if up_dets:
                            # scale coordinates back to original crop
                            detections = []
                            for d in up_dets:
                                bx, by, bw, bh = d['box']
                                sx = int(bx / scale); sy = int(by / scale)
                                sw = int(bw / scale); sh = int(bh / scale)
                                detections.append({'box': (sx, sy, sw, sh), 'confidence': d.get('confidence', 1.0)})
                    except Exception:
                        pass

            # If still no detections, try Haar cascade on the person crop as a fallback
            if not detections:
                try:
                    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                    gray = cv2.cvtColor(person_crop, cv2.COLOR_BGR2GRAY)
                    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(20,20))
                    if len(faces) > 0:
                        detections = []
                        for (fx, fy, fw, fh) in faces:
                            detections.append({'box': (fx, fy, fw, fh), 'confidence': 1.0})
                except Exception:
                    pass

            # Debug: if still none, save crop for inspection (one per run)
            # if not detections:
            #     print(f"[DBG] No face detections inside person crop (crop shape={person_crop.shape}). Saving crop for inspection.")
            #     try:
            #         os.makedirs('debug_person_crops', exist_ok=True)
            #         fn = os.path.join('debug_person_crops', f"crop_f{frame_num}_t{int(time.time())}.jpg")
            #         cv2.imwrite(fn, person_crop)
            #     except Exception:
            #         pass
            #     # continue to return Unknown below
            

            if not detections:
                return {'name': 'Unknown', 'confidence': 0.0, 'predicted_name': 'Unknown', 'predicted_confidence': 0.0, 'face_bbox': None}
            
            # Take the best face detection
            best_detection = max(detections, key=lambda d: d['confidence'])
            x, y, w, h = best_detection['box']
            det_conf = float(best_detection.get('confidence', 1.0))
            person_h = person_crop.shape[0] if person_crop.shape[0] > 0 else 1

            # Reject tiny / low-confidence detections
            face_rel_height = h / float(person_h)
            if det_conf < FACE_DET_CONF_THRESHOLD or face_rel_height < FACE_MIN_REL_SIZE:
                return {'name': 'Unknown', 'confidence': 0.0, 'predicted_name': 'Unknown', 'predicted_confidence': 0.0, 'face_bbox': None}

            # Add small margin to include whole face (clamped)
            pad = int(0.15 * max(w, h))
            fx1 = max(0, x - pad); fy1 = max(0, y - pad)
            fx2 = min(person_crop.shape[1], x + w + pad); fy2 = min(person_crop.shape[0], y + h + pad)
            face_crop = person_crop[fy1:fy2, fx1:fx2]
            
            if face_crop.size == 0:
                return {'name': 'Unknown', 'confidence': 0.0, 'predicted_name': 'Unknown', 'predicted_confidence': 0.0, 'face_bbox': None}
            
            # ------------------ Validation to avoid false positives ------------------
            # Accept detection only if it has reasonable facial structure:
            #  - landmark presence from MTCNN (preferred), OR
            #  - reasonable aspect ratio + edge density inside the face crop
            try:
                from utils.common import compute_blur_score, eye_angle_deg
            except Exception:
                compute_blur_score = None
                eye_angle_deg = None

            valid_face = False
            lm = best_detection.get('landmarks', None)

            # If diagnostics mode enabled, skip strict validation
            if getattr(self, 'always_draw_predictions', False):
                valid_face = True
            else:
                # If detector provided landmarks, trust them (eyes/nose/mouth)
                if lm is not None:
                    try:
                        arr = np.asarray(lm)
                        if arr.size >= 4:  # at least eyes available
                            # simple sanity: eye distance non-zero
                            if arr.shape[0] >= 2:
                                valid_face = True
                    except Exception:
                        valid_face = False

                # Fallback structural checks (reject flat/edge-poor blobs like elbows)
                if not valid_face:
                    try:
                        gray_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY) if face_crop.ndim == 3 else face_crop
                        fh, fw = gray_face.shape[:2]
                        ar = float(fh) / (fw + 1e-9)
                        edges = cv2.Canny(gray_face, 50, 150)
                        edge_density = float(np.count_nonzero(edges)) / (edges.size + 1e-9)
                        # reasonable human face aspect ratio filter and edge density
                        if 0.5 < ar < 2.0 and edge_density > 0.02:
                            valid_face = True
                    except Exception:
                        valid_face = False

            # If invalid -> return Unknown (no debug output)
            if not valid_face:
                return {'name': 'Unknown', 'confidence': 0.0, 'predicted_name': 'Unknown', 'predicted_confidence': 0.0, 'face_bbox': None}
            # ------------------------------------------------------------------------
            
            # Recognize face using simple ArcFace (ensure color/size)
            predicted_name, predicted_confidence = self.face_recognizer.recognize_face(face_crop)
            
            # Determine final accepted name based on threshold (keep predicted values separately)
            final_name = predicted_name if predicted_confidence >= RECOG_THRESHOLD else 'Unknown'
            final_confidence = predicted_confidence if predicted_confidence >= RECOG_THRESHOLD else predicted_confidence
            
            # recognition info (no debug)
            
            # Map face bbox to original frame coords
            px1, py1, px2, py2 = person_bbox
            face_bbox_original = (px1 + fx1, py1 + fy1, px1 + fx2, py1 + fy2)
            
            return {
                'name': final_name,
                'confidence': final_confidence,
                'predicted_name': predicted_name,
                'predicted_confidence': predicted_confidence,
                'face_bbox': face_bbox_original
            }
            
        except Exception as e:
            print(f"⚠️  Face recognition failed: {e}")
            return {'name': 'Unknown', 'confidence': 0.0, 'predicted_name': 'Unknown', 'predicted_confidence': 0.0, 'face_bbox': None}
    
    def process_video_with_tracking(
        self,
        video_path: str = None,
        use_webcam: bool = False,
        display: bool = True
    ):
        """Enhanced video processing with FaceNet-style tracking."""
        if not self.face_recognizer_loaded:
            print("⚠️  No ArcFace model loaded. Running detection-only mode...")
            self.run_detection_only()
            return
        
        # Override display if GUI not available
        if display and not HAS_GUI:
            print("[WARN] Display requested but GUI not available. Running in headless mode.")
            display = False
        
        print("🎥 Starting ArcFace video processing with face recognition...")
        print(f"[INFO] Display mode: {'GUI' if display else 'Headless'}")
        
        # Initialize video capture
        if use_webcam:
            cap = cv2.VideoCapture(0)
            print("[INFO] Using webcam")
        else:
            video_path = video_path or VIDEO_PATH
            if not os.path.exists(video_path):
                print(f"[ERROR] Video file not found: {video_path}")
                return
            cap = cv2.VideoCapture(video_path)
            print(f"[INFO] Using video file: {video_path}")
        
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
        print(f"[INFO] Display enabled: {display and HAS_GUI}")
        
        # Enhanced progress reporting for headless mode
        if not display:
            print(f"[INFO] HEADLESS MODE - Progress will be shown every {100} frames")
            print(f"[INFO] Press Ctrl+C to stop processing")
            print(f"[INFO] Results will be saved to: {LOGS_KNOWN_DIR}")
        
        # FaceNet-style tracking variables
        track_identities = {}
        track_face_history = defaultdict(lambda: deque(maxlen=EXTENDED_MEMORY_FRAMES))
        track_body_history = defaultdict(lambda: deque(maxlen=60))
        track_last_face_frame = {}
        known_last_saved = {}
        
        frame_num = 0
        frame_times = deque(maxlen=30)
        
        try:
            while True:
                process_start = time.time()
                
                ret, frame = cap.read()
                if not ret:
                    print("[INFO] End of video reached")
                    break
                
                original_frame = frame.copy()
                orig_h, orig_w = original_frame.shape[:2]
                ts = datetime.now()
                
                # Resize for processing if needed
                if RESIZE_WIDTH and orig_w > RESIZE_WIDTH:
                    ratio = RESIZE_WIDTH / orig_w
                    process_frame = cv2.resize(original_frame, (RESIZE_WIDTH, int(orig_h * ratio)))
                else:
                    process_frame = original_frame
                    ratio = 1.0
                
                # Always create annotated frame
                annotated_frame = original_frame.copy()
                
                # Only do heavy processing every Nth frame
                do_processing = (frame_num % PROCESS_EVERY_N == 0)
                
                if do_processing:
                    # Process with YOLO tracking
                    if self.yolo is not None:
                        try:
                            # Use default ByteTracker without config file
                            results = self.yolo.track(
                                process_frame,
                                persist=True,
                                classes=[0],  # Only persons
                                conf=PERSON_CONF_THRESHOLD,  # Use consistent threshold
                                iou=0.7,  # Use standard IoU
                                verbose=False
                            )
                            
                            if results and len(results) > 0:
                                result = results[0]
                                
                                if hasattr(result, 'boxes') and result.boxes is not None:
                                    boxes = result.boxes
                                    
                                    # Check for tracking IDs
                                    if hasattr(boxes, 'id') and boxes.id is not None:
                                        track_ids = boxes.id.cpu().numpy().astype(int)
                                        bboxes = boxes.xyxy.cpu().numpy()
                                        confidences = boxes.conf.cpu().numpy()
                                        
                                        # Ensure consistent array lengths
                                        min_len = min(len(track_ids), len(bboxes), len(confidences))
                                        track_ids = track_ids[:min_len]
                                        bboxes = bboxes[:min_len]
                                        confidences = confidences[:min_len]
                                        
                                        for i, (track_id, bbox, conf) in enumerate(zip(track_ids, bboxes, confidences)):
                                            # Convert to original frame coordinates
                                            if ratio != 1.0:
                                                bbox = bbox / ratio
                                            
                                            x1, y1, x2, y2 = bbox.astype(int)
                                            x1, y1 = max(0, x1), max(0, y1)
                                            x2, y2 = min(orig_w, x2), min(orig_h, y2)
                                            
                                            person_bbox = (x1, y1, x2, y2)
                                            person_crop = original_frame[y1:y2, x1:x2]
                                            
                                            # Debug: Check if we have a valid person crop
                                            if person_crop is None or person_crop.size == 0:
                                                continue
                                            
                                            # Enhanced face recognition - THIS IS THE KEY PART
                                            face_result = self.recognize_face_in_crop_enhanced(person_crop, original_frame, person_bbox, frame_num)
                                            
                                            # FaceNet-style identity tracking
                                            self.update_track_identity_enhanced(
                                                track_id, face_result, person_crop, track_identities, 
                                                track_face_history, track_body_history, frame_num
                                            )
                                            
                                            # Reflect face-level signals into person-level tracking state:
                                            if frame_num is not None:
                                                # update last face seen
                                                track_last_face_frame[track_id] = frame_num

                                            # store a compact face history entry (used by consensus/decay)
                                            track_face_history[track_id].append({
                                                "frame": frame_num,
                                                "name": face_result.get("name", "Unknown"),
                                                "predicted_name": face_result.get("predicted_name", "Unknown"),
                                                "conf": float(face_result.get("confidence", 0.0)),
                                                "pred_conf": float(face_result.get("predicted_confidence", 0.0)),
                                                "face_bbox": face_result.get("face_bbox")
                                            })

                                            # mark back/profile views to suppress/harden decisions
                                            if face_result.get("is_back_view"):
                                                track_identities.setdefault(track_id, {}).setdefault("consecutive_back_frames", 0)
                                                track_identities[track_id]["consecutive_back_frames"] += 1
                                            else:
                                                if track_id in track_identities:
                                                    track_identities[track_id]["consecutive_back_frames"] = 0

                                            # optionally update an appearance template (simple HSV hist)
                                            try:
                                                bbox = face_result.get("face_bbox")
                                                if bbox is not None:
                                                    fx1, fy1, fx2, fy2 = map(int, bbox)
                                                    face_crop = original_frame[fy1:fy2, fx1:fx2]
                                                    if face_crop.size > 0:
                                                        hsv = cv2.cvtColor(face_crop, cv2.COLOR_BGR2HSV)
                                                        hist = cv2.calcHist([hsv], [0,1], None, [16,16], [0,180,0,256])
                                                        cv2.normalize(hist, hist)
                                                        track_body_history[track_id].append(hist.flatten())
                                            except Exception:
                                                pass
                                            
                                            # Update frame numbers
                                            if face_result['name'] != 'Unknown':
                                                track_last_face_frame[track_id] = frame_num
                                            

                                            # Get consensus identity
                                            frames_since_face = frame_num - track_last_face_frame.get(track_id, frame_num)
                                            identity_name, identity_conf = self.get_consensus_identity_enhanced(
                                                track_id, track_identities, track_face_history, frames_since_face
                                            )
                                            

                                            # Enhanced display
                                            identity = track_identities.get(track_id, {})
                                            is_locked = identity.get('identity_locked', False)
                                            
                                            # Choose display color based on identity
                                            if identity_name != 'Unknown' and identity_name != '':
                                                if is_locked:
                                                    color = (0, 255, 0)  # Bright green for locked
                                                    thickness = 3
                                                else:
                                                    color = (0, 200, 0)  # Green for known
                                                    thickness = 2
                                            else:
                                                color = (0, 0, 255)  # Red for unknown
                                                thickness = 2
                                            
                                            # Draw bounding box for person (existing)
                                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
                                            
                                            # Draw face box if available
                                            if face_result.get('face_bbox') is not None:
                                                fx1, fy1, fx2, fy2 = face_result['face_bbox']
                                                # clamp to frame bounds
                                                fx1, fy1 = int(max(0, fx1)), int(max(0, fy1))
                                                fx2 = int(min(original_frame.shape[1] - 1, fx2))
                                                fy2 = int(min(original_frame.shape[0] - 1, fy2))
                                                # cyan box for face
                                                cv2.rectangle(annotated_frame, (fx1, fy1), (fx2, fy2), (255, 200, 0), 2)
                                                
                                                # Decide which label to draw depending on diagnostic mode
                                                if getattr(self, 'always_draw_predictions', False):
                                                    pred_name = face_result.get('predicted_name', 'Unknown')
                                                    pred_conf = face_result.get('predicted_confidence', 0.0)
                                                    if face_result.get('name') == 'Unknown':
                                                        label = f"Unknown (pred: {pred_name} {pred_conf:.2f})"
                                                    else:
                                                        label = f"{face_result['name']} {face_result['confidence']:.2f} (pred:{pred_name} {pred_conf:.2f})"
                                                else:
                                                    label = f"{face_result['name']} {face_result['confidence']:.2f}" if face_result['name'] != 'Unknown' else "Unknown"
                                                
                                                (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                                                cv2.rectangle(annotated_frame, (fx1, fy1 - lh - 6), (fx1 + lw + 6, fy1), (255, 200, 0), -1)
                                                cv2.putText(annotated_frame, label, (fx1 + 3, fy1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
                                            
                                            # Enhanced label with actual identity
                                            if identity_name != 'Unknown' and identity_name != '':
                                                label = f"ID:{track_id} {identity_name}"
                                                if identity_conf > 0:
                                                    label += f" ({identity_conf:.2f})"
                                                if is_locked:
                                                    label += " LOCKED"
                                            else:
                                                label = f"ID:{track_id} Unknown Person"
                                            
                                            label_y = max(30, y1 - 10)
                                            
                                            # Draw label background
                                            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                                            cv2.rectangle(annotated_frame, (x1, label_y - label_h - 5), 
                                                        (x1 + label_w + 5, label_y + 5), (0, 0, 0), -1)
                                            
                                            # Draw label text
                                            cv2.putText(annotated_frame, label, (x1 + 2, label_y), 
                                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                                            
                                            # Save faces if enabled
                                            if SAVE_FACES and identity_name != 'Unknown' and identity_name != '':
                                                if face_result.get('face_bbox') is not None:
                                                    self.save_face_from_result(face_result, original_frame, identity_name, 
                                                                             frame_num, ts, known_last_saved)
                                        
                        except Exception as e:
                            if not display:  # Only print detailed errors in headless mode
                                print(f"⚠️  YOLO tracking error: {e}")
                            # Fallback: try simple detection without tracking
                            try:
                                simple_results = self.yolo(process_frame, verbose=False, classes=[0])
                                if simple_results and len(simple_results) > 0:
                                    simple_result = simple_results[0]
                                    if hasattr(simple_result, 'boxes') and simple_result.boxes is not None:
                                        simple_boxes = simple_result.boxes
                                        simple_bboxes = simple_boxes.xyxy.cpu().numpy()
                                        simple_confidences = simple_boxes.conf.cpu().numpy()
                                        
                                        for i, (bbox, conf) in enumerate(zip(simple_bboxes, simple_confidences)):
                                            if ratio != 1.0:
                                                bbox = bbox / ratio
                                            
                                            x1, y1, x2, y2 = bbox.astype(int)
                                            x1, y1 = max(0, x1), max(0, y1)
                                            x2, y2 = min(orig_w, x2), min(orig_h, y2)
                                            
                                            # Draw detection box without tracking ID
                                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (128, 128, 128), 2)
                                            cv2.putText(annotated_frame, f"Person {conf:.2f}", (x1, y1-10), 
                                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2)
                            except Exception as e2:
                                print(f"⚠️  Fallback detection failed: {e2}")
                
                # Performance monitoring
                process_time = time.time() - process_start
                frame_times.append(process_time)
                
                if len(frame_times) > 0:
                    avg_time = sum(frame_times) / len(frame_times)
                    fps_current = 1.0 / max(avg_time, 0.001)
                else:
                    fps_current = 0.0
                
                # Count tracks
                total_tracks = len(track_identities)
                
                # Add performance info
                status = "PROCESSING" if do_processing else "SKIPPED"
                info_text = f"Frame: {frame_num} | FPS: {fps_current:.1f} | Tracks: {total_tracks} | {status}"
                cv2.putText(annotated_frame, info_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Show display only if GUI is available and requested
                if display and HAS_GUI:
                    try:
                        display_frame = annotated_frame
                        h, w = display_frame.shape[:2]
                        if w > 1280:
                            scale = 1280 / w
                            new_w = 1280
                            new_h = int(h * scale)
                            display_frame = cv2.resize(display_frame, (new_w, new_h))
                        
                        cv2.imshow("ArcFace Enhanced Recognition", display_frame)
                        
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q') or key == 27:  # 'q' or ESC
                            print("[INFO] User requested quit")
                            break
                    except Exception as e:
                        print(f"[WARN] Display error: {e}")
                        print("[INFO] Continuing in headless mode...")
                        display = False  # Disable display for remaining frames
                
                frame_num += 1
                
                # Enhanced progress reporting for headless mode
                if frame_num % 100 == 0:
                    progress = (frame_num / total_frames) * 100 if total_frames > 0 else 0
                    active_tracks = sum(1 for t in track_identities.values() if t.get('identity') != 'Unknown')
                    
                    if not display:  # Headless mode - more detailed progress
                        print(f"\n[PROGRESS] Frame {frame_num}/{total_frames} ({progress:.1f}%)")
                        print(f"           FPS: {fps_current:.1f} | Active Tracks: {active_tracks}/{total_tracks}")
                        print(f"           Processing: {'ON' if do_processing else 'SKIPPED'}")
                        
                        # Show recent recognitions
                        if track_identities:
                            recent_ids = []
                            for tid, tdata in track_identities.items():
                                if tdata.get('identity') != 'Unknown':
                                    conf = tdata.get('confidence', 0)
                                    recent_ids.append(f"ID{tid}:{tdata['identity']}({conf:.2f})")
                            
                            if recent_ids:
                                print(f"           Recognized: {', '.join(recent_ids[:5])}")
                        print(f"           Time: {ts.strftime('%H:%M:%S')}")
                    else:
                        print(f"[INFO] Progress: {progress:.1f}% | Frame {frame_num}/{total_frames} | FPS: {fps_current:.1f}")
                
                # Show brief status every 10 frames in headless mode
                elif not display and frame_num % 10 == 0:
                    print(f"[INFO] Frame {frame_num} | FPS: {fps_current:.1f} | Tracks: {len(track_identities)}", end='\r')
        
        except KeyboardInterrupt:
            print(f"\n[INFO] Interrupted by user at frame {frame_num}")
            print(f"[INFO] Processed {frame_num}/{total_frames} frames ({(frame_num/total_frames)*100:.1f}%)")
        except Exception as e:
            print(f"[ERROR] Processing error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            cap.release()
            if display and HAS_GUI:
                try:
                    cv2.destroyAllWindows()
                except:
                    pass  # Ignore cleanup errors
            
            # Final summary
            print(f"\n{'='*50}")
            print(f"📊 PROCESSING SUMMARY")
            print(f"{'='*50}")
            print(f"Total frames processed: {frame_num}")
            print(f"Total tracks created: {len(track_identities)}")
            
            if track_identities:
                recognized_tracks = {tid: data for tid, data in track_identities.items() 
                                   if data.get('identity') != 'Unknown'}
                print(f"Recognized identities: {len(recognized_tracks)}")
                
                for tid, data in recognized_tracks.items():
                    identity = data.get('identity', 'Unknown')
                    confidence = data.get('confidence', 0)
                    print(f"  Track {tid}: {identity} (confidence: {confidence:.3f})")
            
            if SAVE_FACES:
                print(f"Face images saved to: {LOGS_KNOWN_DIR}")
            print(f"{'='*50}")

    def run_detection_only(self):
        """Run face detection without recognition."""
        print("[INFO] Starting ArcFace detection-only mode...")
        
        # Check if GUI is available for display
        display_available = HAS_GUI
        if not display_available:
            print("[INFO] Running in headless mode (no GUI display)")
            print("[INFO] Face detection results will be printed to console")
        
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
        
        frame_num = 0
        face_count_total = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Detect faces
                detections = self.detect_faces(frame)
                face_count_total += len(detections)
                
                # Print detection info in headless mode
                if not display_available and len(detections) > 0:
                    print(f"[DETECT] Frame {frame_num}: {len(detections)} faces detected")
                    for i, detection in enumerate(detections):
                        conf = detection['confidence']
                        x, y, w, h = detection['box']
                        print(f"  Face {i+1}: confidence={conf:.3f}, bbox=({x},{y},{w},{h})")
                
                # Draw detections
                for detection in detections:
                    x, y, w, h = detection['box']
                    confidence = detection['confidence']
                    
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                    label = f"Face ({confidence:.2f})"
                    cv2.putText(frame, label, (x, y - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # Show display only if available
                if display_available:
                    try:
                        cv2.imshow("ArcFace Detection", frame)
                        
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q') or key == 27:
                            break
                    except Exception as e:
                        print(f"[WARN] Display error: {e}, continuing in headless mode...")
                        display_available = False
                
                frame_num += 1
                
                # Print progress in headless mode
                if not display_available and frame_num % 100 == 0:
                    avg_faces = face_count_total / frame_num if frame_num > 0 else 0
                    print(f"[PROGRESS] Frame {frame_num} | Total faces: {face_count_total} | Avg: {avg_faces:.1f}/frame")
                
        except KeyboardInterrupt:
            print(f"\n[INFO] Interrupted by user at frame {frame_num}")
        finally:
            cap.release()
            if display_available:
                try:
                    cv2.destroyAllWindows()
                except:
                    pass
            
            print(f"\n{'='*40}")
            print(f"📊 DETECTION SUMMARY")
            print(f"{'='*40}")
            print(f"Frames processed: {frame_num}")
            print(f"Total faces detected: {face_count_total}")
            if frame_num > 0:
                print(f"Average faces per frame: {face_count_total/frame_num:.2f}")
            print(f"{'='*40}")
    
    def capture_faces(self, person_name: str, max_images: int = 100, camera_id: int = 0):
        """Capture faces for training."""
        capture = ArcFaceCapture()
        return capture.capture_from_webcam(person_name, max_images, camera_id)

    def test_video_window(self):
        """Test video window functionality."""
        print("[INFO] Testing video window functionality...")
        
        try:
            # Try to open video source
            if USE_WEBCAM:
                cap = cv2.VideoCapture(0)
                source_name = "webcam"
            else:
                if not os.path.exists(VIDEO_PATH):
                    print(f"[ERROR] Video file not found: {VIDEO_PATH}")
                    return False
                cap = cv2.VideoCapture(VIDEO_PATH)
                source_name = "video file"
            
            if not cap.isOpened():
                print(f"[ERROR] Failed to open {source_name}")
                return False
            
            print(f"[INFO] Successfully opened {source_name}")
            
            # Test reading a few frames
            frame_count = 0
            test_frames = 5
            
            for i in range(test_frames):
                ret, frame = cap.read()
                if not ret:
                    print(f"[WARN] Failed to read frame {i}")
                    break
                frame_count += 1
            
            cap.release()
            
            if frame_count == 0:
                print("[ERROR] No frames could be read")
                return False
            
            print(f"[INFO] Successfully read {frame_count}/{test_frames} frames")
            
            # Test window creation if GUI available
            if HAS_GUI:
                try:
                    # Create a simple test image
                    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(test_image, "Video Test - Press any key", (50, 240), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    # Try to display it
                    cv2.imshow("ArcFace Video Test", test_image)
                    cv2.waitKey(1000)  # Wait 1 second
                    cv2.destroyAllWindows()
                    cv2.waitKey(1)  # Process window events
                    
                    print("[INFO] Video window test completed successfully")
                    return True
                    
                except Exception as e:
                    print(f"[WARN] Video window test failed: {e}")
                    print("[INFO] GUI not working, will run in headless mode")
                    return True  # Still return True since video reading works
            else:
                print("[INFO] GUI not available, will run in headless mode")
                return True
                
        except Exception as e:
            print(f"[ERROR] Video test failed: {e}")
            return False

    def update_track_identity_enhanced(self, track_id, face_result, person_crop, track_identities, 
                                     track_face_history, track_body_history, frame_num):
        """Update track identity with face recognition result."""
        name = face_result.get('name', 'Unknown')
        confidence = face_result.get('confidence', 0.0)
        
        # Initialize track if new
        if track_id not in track_identities:
            track_identities[track_id] = {
                'identity': 'Unknown',
                'confidence': 0.0,
                'identity_locked': False,
                'last_face_frame': frame_num,
                'face_detections': 0,
                'total_detections': 0
            }
        
        track_data = track_identities[track_id]
        track_data['total_detections'] += 1
        
        # Update face history
        if name != 'Unknown':
            track_face_history[track_id].append({
                'name': name,
                'confidence': confidence,
                'frame': frame_num
            })
            track_data['face_detections'] += 1
            track_data['last_face_frame'] = frame_num
        
        # Update identity based on short temporal consensus to avoid single-frame errors
        recent = list(track_face_history[track_id])[-(CONFIRM_FACE_FRAMES * 3):]  # look back a bit
        if recent:
            recent_names = [h['name'] for h in recent if h.get('name') != 'Unknown']
            if recent_names:
                from collections import Counter
                most_common = Counter(recent_names).most_common(1)[0]
                consensus_name = most_common[0]
                consensus_count = most_common[1]

                # require at least CONFIRM_FACE_FRAMES consistent detections to set identity
                if consensus_count >= max(1, CONFIRM_FACE_FRAMES):
                    track_data['identity'] = consensus_name
                    track_data['confidence'] = float(confidence)

                    # stronger lock: require 3x confirmation frames
                    if consensus_count >= max(5, CONFIRM_FACE_FRAMES * 3):
                        track_data['identity_locked'] = True

    def get_consensus_identity_enhanced(self, track_id, track_identities, track_face_history, frames_since_face):
        """Get consensus identity for track."""
        if track_id not in track_identities:
            return 'Unknown', 0.0
        
        track_data = track_identities[track_id]
        identity = track_data.get('identity', 'Unknown')
        confidence = track_data.get('confidence', 0.0)
        
        # Return current identity if face was seen recently
        if frames_since_face <= FACE_LOST_TOLERANCE:
            return identity, confidence
        
        # If face lost for too long, mark as unknown
        if frames_since_face > FACE_LOST_TOLERANCE * 2:
            return 'Unknown', 0.0
        
        return identity, confidence * 0.5  # Reduce confidence for old detections
    
    def save_face_from_result(self, face_result, original_frame, identity_name, frame_num, ts, known_last_saved):
        """Save face from recognition result."""
        try:
            # Check save interval
            save_key = identity_name
            if save_key in known_last_saved:
                last_saved_frame = known_last_saved[save_key]
                if frame_num - last_saved_frame < KNOWN_SAVE_INTERVAL_MIN * 30:  # 30 fps assumption
                    return
            
            # Extract face if bbox available
            face_bbox = face_result.get('face_bbox')
            if face_bbox is not None:
                x1, y1, x2, y2 = face_bbox
                face_crop = original_frame[y1:y2, x1:x2]
                
                if face_crop.size > 0:
                    # Ensure directory exists
                    os.makedirs(LOGS_KNOWN_DIR, exist_ok=True)
                    
                    # Generate filename
                    timestamp = ts.strftime("%H%M%S")
                    filename = f"{identity_name}_{timestamp}_f{frame_num}.jpg"
                    face_path = os.path.join(LOGS_KNOWN_DIR, filename)
                    
                    # Save face
                    cv2.imwrite(face_path, face_crop)
                    known_last_saved[save_key] = frame_num
                    
                    # Print save notification (only in headless mode)
                    if not HAS_GUI:
                        print(f"[SAVE] {filename}")
                        
        except Exception as e:
            print(f"⚠️  Failed to save face: {e}")

def main():
    """Command-line interface for ArcFace system."""
    parser = argparse.ArgumentParser(description='ArcFace Face Recognition System')
    parser.add_argument('--always_draw', action='store_true', help='Always draw predicted face label/confidence (diagnostic)')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Capture command
    capture_parser = subparsers.add_parser('capture', help='Capture faces for training')
    capture_parser.add_argument('--name', type=str, help='Person name')
    capture_parser.add_argument('--max_images', type=int, default=100, help='Max images to capture')
    capture_parser.add_argument('--camera', type=int, default=0, help='Camera ID')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train simple ArcFace model')
    train_parser.add_argument('--dataset', type=str, default='datasets/faces', help='Dataset directory')
    
    # Video command
    video_parser = subparsers.add_parser('video', help='Process video with face recognition')
    video_parser.add_argument('--video', type=str, help='Path to video file')
    video_parser.add_argument('--webcam', action='store_true', help='Use webcam input')
    video_parser.add_argument('--no_display', action='store_true', help='Disable display')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Test video window')
    test_parser.add_argument('--simple', action='store_true', help='Simple video test')
    
    args = parser.parse_args()
    
    # Default behavior: run video processing
    if args.command is None:
        print("🎥 Starting ArcFace Simple Recognition System...")
        print(f"[DEBUG] HAS_GUI status: {HAS_GUI}")
        print(f"[DEBUG] OpenCV version: {cv2.__version__}")
        
        # Inform user about GUI status
        if not HAS_GUI:
            print("\n" + "="*60)
            print("⚠️  GUI DISPLAY NOT AVAILABLE")
            print("="*60)
            print("Your OpenCV installation lacks GUI support.")
            print("The system will run in HEADLESS mode with:")
            print("• Console progress updates every 100 frames")
            print("• Face detection and recognition still working")
            print("• Results saved to logs/ directory")
            print("• Performance metrics in console")
            print("\nTo enable GUI display, install opencv-contrib-python:")
            print("pip uninstall opencv-python")
            print("pip install opencv-contrib-python")
            print("="*60 + "\n")
            
            # Ask user if they want to continue
            try:
                response = input("Continue in headless mode? (y, N): ").strip().lower()
                if response not in ['y', 'yes']:
                    print("Exiting...")
                    return
            except KeyboardInterrupt:
                print("\nExiting...")
                return
        
        # Create system
        system = ArcFaceSystem(use_yolo=True)
        
        # Set diagnostic draw mode from CLI
        system.always_draw_predictions = bool(getattr(args, 'always_draw', False))
        
        print("🧪 Running video window test...")
        if system.test_video_window():
            if HAS_GUI:
                print("✅ Video window test PASSED!")
            else:
                print("✅ Video reading test PASSED! (Running in headless mode)")
            
            # Check for SIMPLE trained models (not the old complex one)
            required_files = [BACKBONE_PATH, CLASSIFIER_PATH, ENCODER_PATH, THRESHOLD_PATH]
            missing_files = [f for f in required_files if not os.path.exists(f)]
            
            if not missing_files:
                print(f"✅ Found simple ArcFace models!")
                print(f"   Backbone: {os.path.basename(BACKBONE_PATH)}")
                print(f"   Classifier: {os.path.basename(CLASSIFIER_PATH)}")
                print(f"   Encoder: {os.path.basename(ENCODER_PATH)}")
                print(f"   Threshold: {os.path.basename(THRESHOLD_PATH)}")
                
                try:
                    # Models are loaded automatically in system.__init__()
                    # Just run video processing
                    system.process_video_with_tracking(
                        video_path=VIDEO_PATH,
                        use_webcam=USE_WEBCAM,
                        display=True  # Will be overridden to False if no GUI
                    )
                    
                except Exception as e:
                    print(f"❌ Video processing failed: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"⚠️  Missing simple ArcFace model files:")
                for f in missing_files:
                    print(f"   ❌ {os.path.basename(f)}")
                print("\n🔧 Please run training first:")
                print("   python face_recognition/ArcFace/arcface_train_simple.py")
                print("\n   OR run detection-only mode:")
                
                try:
                    response = input("Run detection-only mode? (y, N): ").strip().lower()
                    if response in ['y', 'yes']:
                        system.run_detection_only()
                except KeyboardInterrupt:
                    print("\nExiting...")
        else:
            print("❌ Video window test FAILED!")
            print("   Cannot proceed with video processing.")
        
        return
    
    # Handle other commands
    if args.command == 'test':
        system = ArcFaceSystem(use_yolo=True)
        if args.simple:
            system.test_video_window()
        else:
            print("🧪 Testing ArcFace components...")
            print(f"✅ System created successfully")
            print(f"✅ YOLO available: {HAS_YOLO}")
            print(f"✅ MTCNN available: {HAS_MTCNN}")
            print(f"✅ GUI available: {HAS_GUI}")
    
    elif args.command == 'video':
        system = ArcFaceSystem(use_yolo=True)
        system.always_draw_predictions = bool(getattr(args, 'always_draw', False))
        
        # Check for simple models
        required_files = [BACKBONE_PATH, CLASSIFIER_PATH, ENCODER_PATH, THRESHOLD_PATH]
        missing_files = [f for f in required_files if not os.path.exists(f)]
        
        if not missing_files:
            print("✅ Simple ArcFace models found!")
            system.process_video_with_tracking(
                video_path=args.video,
                use_webcam=args.webcam,
                display=not args.no_display
            )
        else:
            print("⚠️  No simple models found, running detection only...")
            system.run_detection_only()
    
    elif args.command == 'capture':
        if not args.name:
            args.name = input("Enter person name: ").strip()
            if not args.name:
                print("❌ Person name required")
                return
        
        system = ArcFaceSystem(use_yolo=True)
        print(f"🎥 Capturing faces for '{args.name}'...")
        system.capture_faces(args.name, args.max_images, args.camera)
    
    elif args.command == 'train':
        print("🔧 To train the simple ArcFace model, run:")
        print("   python face_recognition/ArcFace/arcface_train_simple.py")
        print("\nMake sure you have face images in:")
        print(f"   {args.dataset}/person1/*.jpg")
        print(f"   {args.dataset}/person2/*.jpg")
        print("   etc...")


if __name__ == "__main__":
    main()
