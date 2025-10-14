import os
import sys
import cv2
import joblib
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import time
import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.models as models
from datetime import datetime
from collections import defaultdict, deque
from pathlib import Path
from ultralytics import YOLO

# Try to import face detection libraries
try:
    from facenet_pytorch import MTCNN
    HAS_MTCNN = True
except ImportError:
    HAS_MTCNN = False

try:
    from insightface.app import FaceAnalysis
    HAS_INSIGHTFACE = True
except ImportError:
    HAS_INSIGHTFACE = False

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# ArcFace Model Configuration (from arcface_main.py)
IMAGE_SIZE = 112
EMBEDDING_SIZE = 512
RECOG_MARGIN = 0.08
RECOG_THRESHOLD = 0.45

class ArcFaceBackbone(nn.Module):
    """Simple ArcFace backbone from arcface_main.py"""
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

class ArcFaceVideoAnalyzer:
    def __init__(self, video_path, output_dir="model_comparison_results"):
        self.video_path = video_path
        self.video_name = Path(video_path).stem  # Get video name without extension
        
        # Create organized folder structure
        self.base_output_dir = output_dir
        self.arcface_dir = os.path.join(output_dir, "ArcFace")
        self.graphs_dir = os.path.join(self.arcface_dir, "graphs")
        self.data_dir = os.path.join(self.arcface_dir, "data")
        self.comparisons_dir = os.path.join(output_dir, "Comparisons")
        
        # Ensure all directories exist
        os.makedirs(self.arcface_dir, exist_ok=True)
        os.makedirs(self.graphs_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.comparisons_dir, exist_ok=True)
        
        # Create subdirectories in Comparisons
        os.makedirs(os.path.join(self.comparisons_dir, "summaries"), exist_ok=True)
        os.makedirs(os.path.join(self.comparisons_dir, "detailed"), exist_ok=True)
        os.makedirs(os.path.join(self.comparisons_dir, "graphs"), exist_ok=True)
        
        self.results = {
            'frames': [],
            'recognitions': []
        }
        
        # Enhanced tracking settings (from arcface_main.py and facenet_video_analyzer.py)
        self.BACK_VIEW_TOLERANCE_FRAMES = 600
        self.BODY_MATCH_THRESHOLD = 0.5
        self.EXTENDED_MEMORY_FRAMES = 300
        self.MIN_BODY_MATCH_CONFIDENCE = 0.4
        self.POSE_HISTORY_LENGTH = 30
        self.IDENTITY_MEMORY_FRAMES = 90
        self.IDENTITY_CONFIDENCE_DECAY = 0.995
        self.MIN_IDENTITY_CONFIDENCE = 0.15
        self.FACE_LOST_TOLERANCE = 180
        
        # Identity confirmation settings (from arcface_main.py)
        self.CONFIRM_FACE_FRAMES = 2
        self.FACE_DET_CONF_THRESHOLD = 0.55
        self.FACE_MIN_REL_SIZE = 0.15
        self.FACE_IOU_THRESHOLD = 0.25
        
        # Tracking data structures
        self.track_identities = {}
        self.track_face_history = defaultdict(lambda: deque(maxlen=self.EXTENDED_MEMORY_FRAMES))
        self.track_body_history = defaultdict(lambda: deque(maxlen=60))
        self.track_last_face_frame = {}
        
        # ArcFace specific initialization
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Model placeholders
        self.embedder = None
        self.classifier = None
        self.label_encoder = None
        self.centroids = None
        self.threshold = None
        self.mtcnn = None
        self.yolo = None
        self.insight = None
        
        print(f"[INFO] ArcFace Video Analyzer initialized")
        print(f"[INFO] Video: {video_path}")
        print(f"[INFO] ArcFace Output: {self.arcface_dir}")
        print(f"[INFO] Device: {self.device}")
        print(f"[INFO] Graphs will be saved to: {self.graphs_dir}")
    
    def load_arcface_models(self):
        """Load ArcFace models from arcface_main.py"""
        try:
            MODELS_DIR = os.path.join("models", "ArcFace")
            BACKBONE_PATH = os.path.join(MODELS_DIR, "resnet50_backbone.pt")
            CLASSIFIER_PATH = os.path.join(MODELS_DIR, "arcface_svm.joblib")
            ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoder.joblib")
            THRESHOLD_PATH = os.path.join(MODELS_DIR, "distance_threshold.npy")
            CENTROIDS_PATH = os.path.join(MODELS_DIR, "class_centroids.pkl")
            
            # Check if files exist
            required_files = [BACKBONE_PATH, CLASSIFIER_PATH, ENCODER_PATH, THRESHOLD_PATH]
            missing_files = [f for f in required_files if not os.path.exists(f)]
            if missing_files:
                print(f"❌ Missing ArcFace model files: {[os.path.basename(f) for f in missing_files]}")
                return False
            
            # Load models based on availability
            if HAS_INSIGHTFACE:
                # Use InsightFace if available
                self.insight = FaceAnalysis(allowed_modules=['detection', 'recognition'])
                ctx_id = 0 if torch.cuda.is_available() else -1
                self.insight.prepare(ctx_id=ctx_id, det_size=(224, 224))
                self.use_insightface = True
                print("[INFO] ✅ Using InsightFace for embeddings")
            else:
                # Fall back to custom backbone
                from facenet_pytorch import InceptionResnetV1
                self.embedder = InceptionResnetV1(pretrained='vggface2').to(self.device).eval()
                self.use_insightface = False
                print("[INFO] ✅ Using InceptionResnetV1 backbone")
            
            # Load classifier and encoder
            self.classifier = joblib.load(CLASSIFIER_PATH)
            self.label_encoder = joblib.load(ENCODER_PATH)
            self.threshold = np.load(THRESHOLD_PATH)
            
            # Load centroids if available
            if os.path.exists(CENTROIDS_PATH):
                self.centroids = joblib.load(CENTROIDS_PATH)
                # Normalize centroids
                for k, v in list(self.centroids.items()):
                    arr = np.asarray(v, dtype=np.float32)
                    n = np.linalg.norm(arr) + 1e-10
                    self.centroids[k] = (arr / n)
                print(f"[INFO] ✅ Loaded {len(self.centroids)} class centroids")
            else:
                self.centroids = None
                print("[INFO] No centroids file found")
            
            print(f"[INFO] ✅ Loaded ArcFace classifier. Classes: {list(self.label_encoder.classes_)}")
            print(f"[INFO] Distance threshold: {self.threshold:.3f}")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to load ArcFace models: {e}")
            return False
    
    def load_detection_models(self):
        """Load detection models"""
        try:
            # Load YOLO
            self.yolo = YOLO("models/YOLOv8/yolov8n.pt")
            print("[INFO] ✅ YOLO loaded successfully")
            
            # Load MTCNN if available
            if HAS_MTCNN:
                self.mtcnn = MTCNN(
                    image_size=160,
                    margin=0,
                    keep_all=True,
                    device=self.device,
                    post_process=False,
                    min_face_size=30,
                    thresholds=[0.6, 0.7, 0.7]
                )
                print("[INFO] ✅ MTCNN loaded successfully")
            else:
                print("[WARN] MTCNN not available, using Haar cascade fallback")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to load detection models: {e}")
            return False
    
    def detect_faces(self, image):
        """Detect faces using InsightFace, MTCNN, or Haar cascade"""
        detections = []
        
        # Try InsightFace first if available
        if self.use_insightface and self.insight is not None:
            try:
                faces = self.insight.get(image)
                for f in faces:
                    bbox = getattr(f, "bbox", None)
                    score = getattr(f, "det_score", None) or 1.0
                    if bbox is None:
                        continue
                    x1, y1, x2, y2 = [int(max(0, v)) for v in bbox]
                    detections.append({
                        "box": (x1, y1, x2 - x1, y2 - y1),
                        "confidence": float(score)
                    })
                return detections
            except Exception:
                pass
        
        # Try MTCNN
        if HAS_MTCNN and self.mtcnn is not None:
            try:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                boxes, probs, _ = self.mtcnn.detect(image_rgb, landmarks=False)
                
                if boxes is not None:
                    for i, box in enumerate(boxes):
                        x1, y1, x2, y2 = [int(max(0, v)) for v in box]
                        detections.append({
                            "box": (x1, y1, x2 - x1, y2 - y1),
                            "confidence": float(probs[i]) if probs is not None else 1.0
                        })
            except Exception:
                pass
        
        # Fallback to Haar cascade
        if not detections:
            try:
                cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                
                for (x, y, w, h) in faces:
                    detections.append({
                        "box": (x, y, w, h),
                        "confidence": 1.0
                    })
            except Exception:
                pass
        
        return detections
    
    def get_face_embedding(self, face_crop):
        """Get face embedding using InsightFace or custom backbone"""
        try:
            if self.use_insightface and self.insight is not None:
                # Use InsightFace
                faces = self.insight.get(face_crop)
                if not faces:
                    return None
                emb = np.asarray(faces[0].embedding, dtype=np.float32)
                return emb / (np.linalg.norm(emb) + 1e-10)
            else:
                # Use custom backbone (fallback to InceptionResnetV1)
                face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                face_resized = cv2.resize(face_rgb, (IMAGE_SIZE, IMAGE_SIZE))
                
                # Convert to tensor
                tensor = self.transform(face_resized).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    if self.embedder is not None:
                        embedding = self.embedder(tensor).cpu().numpy()[0]
                    else:
                        return None
                
                # Normalize
                return embedding / (np.linalg.norm(embedding) + 1e-10)
                
        except Exception as e:
            print(f"[WARN] Face embedding failed: {e}")
            return None
    
    def recognize_face_enhanced(self, face_img):
        """Enhanced face recognition with multiple variants (from arcface_main.py)"""
        try:
            # Create image variants for robust recognition
            variants = []
            
            # Original
            orig = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            variants.append(cv2.resize(orig, (IMAGE_SIZE, IMAGE_SIZE)))
            
            # CLAHE variant
            try:
                lab = cv2.cvtColor(orig, cv2.COLOR_RGB2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                cl = clahe.apply(l)
                merged = cv2.merge((cl, a, b))
                clahe_rgb = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
                variants.append(cv2.resize(clahe_rgb, (IMAGE_SIZE, IMAGE_SIZE)))
            except Exception:
                pass
            
            # Gamma correction variants
            for gamma in [0.8, 1.2]:
                try:
                    invGamma = 1.0 / gamma
                    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(256)]).astype('uint8')
                    gamma_img = cv2.LUT(orig.astype('uint8'), table)
                    variants.append(cv2.resize(gamma_img, (IMAGE_SIZE, IMAGE_SIZE)))
                except Exception:
                    pass
            
            # Horizontal flip
            try:
                flipped = cv2.flip(orig, 1)
                variants.append(cv2.resize(flipped, (IMAGE_SIZE, IMAGE_SIZE)))
            except Exception:
                pass
            
            # Get embeddings for all variants
            embeddings = []
            for variant in variants:
                variant_bgr = cv2.cvtColor(variant, cv2.COLOR_RGB2BGR)
                emb = self.get_face_embedding(variant_bgr)
                if emb is not None:
                    embeddings.append(emb)
            
            if not embeddings:
                return "Unknown", 0.0
            
            # Average embeddings
            mean_emb = np.mean(np.stack(embeddings, axis=0), axis=0)
            mean_emb = mean_emb / (np.linalg.norm(mean_emb) + 1e-10)
            
            # Classify
            if hasattr(self.classifier, "predict_proba"):
                probs_list = []
                for emb in embeddings:
                    try:
                        probs = self.classifier.predict_proba([emb])[0]
                        probs_list.append(probs)
                    except Exception:
                        probs = np.zeros(len(self.classifier.classes_)) + 1e-9
                        probs_list.append(probs)
                
                # Average probabilities
                avg_probs = np.mean(np.stack(probs_list, axis=0), axis=0)
                sorted_idx = np.argsort(avg_probs)[::-1]
                top_idx = int(sorted_idx[0])
                top_prob = float(avg_probs[top_idx])
                second_prob = float(avg_probs[sorted_idx[1]]) if len(sorted_idx) > 1 else 0.0
            else:
                # Fallback without probabilities
                pred = self.classifier.predict([mean_emb])[0]
                classes = list(self.classifier.classes_)
                top_idx = classes.index(pred)
                top_prob = 1.0
                second_prob = 0.0
            
            # Get predicted name
            pred_value = self.classifier.classes_[top_idx]
            
            # Resolve name
            name = None
            try:
                if isinstance(pred_value, (int, np.integer)):
                    name = self.label_encoder.inverse_transform([pred_value])[0]
                else:
                    name = str(pred_value)
            except Exception:
                name = str(pred_value)
            
            # Apply thresholds
            if top_prob < RECOG_THRESHOLD or (top_prob - second_prob) < RECOG_MARGIN:
                return "Unknown", top_prob
            
            # Centroid check if available
            if self.centroids is not None and self.threshold is not None:
                if name in self.centroids:
                    centroid = self.centroids[name]
                    dist = float(np.linalg.norm(mean_emb - np.asarray(centroid, dtype=np.float32)))
                    
                    if dist <= self.threshold:
                        return name, top_prob
                    else:
                        return "Unknown", top_prob
            
            return name, top_prob
            
        except Exception as e:
            print(f"[WARN] ArcFace recognition failed: {e}")
            return "Unknown", 0.0
    
    def detect_person_pose_from_body(self, person_crop):
        """Detect person pose from body (from facenet_video_analyzer.py)"""
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
            
            # Analyze head region
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
            
            # Brightness analysis
            head_brightness = np.mean(head_region)
            torso_brightness = np.mean(torso_region)
            brightness_ratio = head_brightness / max(torso_brightness, 1)
            if 0.8 < brightness_ratio < 1.3:
                back_score += 0.2
            
            # Texture analysis
            head_std = np.std(head_region)
            if head_std < 25:
                back_score += 0.1
            
            # Determine pose
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
    
    def calculate_body_similarity(self, template_crop, current_crop):
        """Calculate body similarity (from facenet_video_analyzer.py)"""
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
    
    def recognize_face_in_crop_enhanced(self, person_crop, original_frame, person_bbox, frame_num):
        """Enhanced face recognition in person crop (from arcface_main.py)"""
        if person_crop is None or person_crop.size == 0:
            return {
                'name': 'Unknown', 'confidence': 0.0, 'predicted_name': 'Unknown', 
                'predicted_confidence': 0.0, 'face_bbox': None, 'inference_time': 0.0
            }
        
        try:
            inference_start = time.time()
            
            # Detect faces in person crop
            detections = self.detect_faces(person_crop)
            
            # Try upscaling if no faces found
            if not detections:
                h, w = person_crop.shape[:2]
                if h > 0 and w > 0:
                    scale = 2.0
                    up_h, up_w = min(int(h * scale), 640), min(int(w * scale), 640)
                    try:
                        up = cv2.resize(person_crop, (up_w, up_h), interpolation=cv2.INTER_LINEAR)
                        up_dets = self.detect_faces(up)
                        if up_dets:
                            detections = []
                            for d in up_dets:
                                bx, by, bw, bh = d['box']
                                sx = int(bx / scale); sy = int(by / scale)
                                sw = int(bw / scale); sh = int(bh / scale)
                                detections.append({'box': (sx, sy, sw, sh), 'confidence': d.get('confidence', 1.0)})
                    except Exception:
                        pass
            
            # Haar cascade fallback
            if not detections:
                try:
                    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                    gray = cv2.cvtColor(person_crop, cv2.COLOR_BGR2GRAY)
                    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(20, 20))
                    for (fx, fy, fw, fh) in faces:
                        detections.append({'box': (fx, fy, fw, fh), 'confidence': 1.0})
                except Exception:
                    pass
            
            if not detections:
                inference_time = (time.time() - inference_start) * 1000
                return {
                    'name': 'Unknown', 'confidence': 0.0, 'predicted_name': 'Unknown',
                    'predicted_confidence': 0.0, 'face_bbox': None, 'inference_time': inference_time
                }
            
            # Get best detection
            best_detection = max(detections, key=lambda d: d['confidence'])
            x, y, w, h = best_detection['box']
            det_conf = float(best_detection.get('confidence', 1.0))
            person_h = person_crop.shape[0] if person_crop.shape[0] > 0 else 1
            
            # Quality checks
            face_rel_height = h / float(person_h)
            if det_conf < self.FACE_DET_CONF_THRESHOLD or face_rel_height < self.FACE_MIN_REL_SIZE:
                inference_time = (time.time() - inference_start) * 1000
                return {
                    'name': 'Unknown', 'confidence': 0.0, 'predicted_name': 'Unknown',
                    'predicted_confidence': 0.0, 'face_bbox': None, 'inference_time': inference_time
                }
            
            # Extract face crop
            pad = int(0.15 * max(w, h))
            fx1 = max(0, x - pad); fy1 = max(0, y - pad)
            fx2 = min(person_crop.shape[1], x + w + pad); fy2 = min(person_crop.shape[0], y + h + pad)
            face_crop = person_crop[fy1:fy2, fx1:fx2]
            
            if face_crop.size == 0:
                inference_time = (time.time() - inference_start) * 1000
                return {
                    'name': 'Unknown', 'confidence': 0.0, 'predicted_name': 'Unknown',
                    'predicted_confidence': 0.0, 'face_bbox': None, 'inference_time': inference_time
                }
            
            # Face recognition
            predicted_name, predicted_confidence = self.recognize_face_enhanced(face_crop)
            
            # Determine final name
            final_name = predicted_name if predicted_confidence >= RECOG_THRESHOLD else 'Unknown'
            final_confidence = predicted_confidence if predicted_confidence >= RECOG_THRESHOLD else predicted_confidence
            
            # Map to original frame coordinates
            px1, py1, px2, py2 = person_bbox
            face_bbox_original = (px1 + fx1, py1 + fy1, px1 + fx2, py1 + fy2)
            
            inference_time = (time.time() - inference_start) * 1000
            
            return {
                'name': final_name,
                'confidence': final_confidence,
                'predicted_name': predicted_name,
                'predicted_confidence': predicted_confidence,
                'face_bbox': face_bbox_original,
                'inference_time': inference_time
            }
            
        except Exception as e:
            print(f"[WARN] ArcFace face recognition failed: {e}")
            inference_time = (time.time() - inference_start) * 1000
            return {
                'name': 'Unknown', 'confidence': 0.0, 'predicted_name': 'Unknown',
                'predicted_confidence': 0.0, 'face_bbox': None, 'inference_time': inference_time
            }
    
    def update_track_identity_enhanced(self, track_id, face_result, person_crop, frame_num):
        """Enhanced identity tracking (from facenet_video_analyzer.py and arcface_main.py)"""
        name = face_result.get('name', 'Unknown')
        conf = face_result.get('confidence', 0.0)
        
        # Detect person pose
        pose, pose_confidence = self.detect_person_pose_from_body(person_crop)
        
        # Initialize tracking data for new track
        if track_id not in self.track_identities:
            self.track_identities[track_id] = {
                'name': name,
                'confidence': conf,
                'last_face_frame': frame_num if name != 'Unknown' else -1,
                'last_seen_frame': frame_num,
                'stable': False,
                'pose_history': deque(maxlen=self.POSE_HISTORY_LENGTH),
                'body_template': None,
                'consecutive_back_frames': 0,
                'max_confidence_seen': conf,
                'identity_locked': False,
                'lock_confidence': 0.0,
                'frames_since_face_lost': 0,
                'total_face_detections': 0,
                'lock_strength': 0.0,
                'unlock_threshold': 600
            }
            self.track_face_history[track_id] = deque(maxlen=self.EXTENDED_MEMORY_FRAMES)
            self.track_body_history[track_id] = deque(maxlen=60)
        
        identity = self.track_identities[track_id]
        identity['pose_history'].append((pose, pose_confidence, frame_num))
        identity['last_seen_frame'] = frame_num
        
        # Store body crop
        if person_crop is not None and person_crop.size > 0:
            self.track_body_history[track_id].append({
                'crop': person_crop.copy(),
                'frame': frame_num,
                'pose': pose
            })
        
        # Update face history
        self.track_face_history[track_id].append({
            'name': name,
            'confidence': conf,
            'frame': frame_num,
            'pose': pose,
            'pose_conf': pose_confidence
        })
        
        # Enhanced identity tracking logic
        if name != 'Unknown':
            identity['total_face_detections'] += 1
            
            # Strong detection - lock identity
            if conf > 0.7:
                identity['name'] = name
                identity['confidence'] = conf
                identity['stable'] = True
                identity['last_face_frame'] = frame_num
                identity['max_confidence_seen'] = max(identity['max_confidence_seen'], conf)
                identity['consecutive_back_frames'] = 0
                identity['frames_since_face_lost'] = 0
                
                # Enhanced locking
                identity['identity_locked'] = True
                identity['lock_confidence'] = max(identity['lock_confidence'], conf)
                identity['lock_strength'] = min(1.0, identity['total_face_detections'] / 10.0) * conf
                
                # Update unlock threshold
                if conf > 0.9:
                    identity['unlock_threshold'] = 1800
                elif conf > 0.8:
                    identity['unlock_threshold'] = 900
                else:
                    identity['unlock_threshold'] = 600
                
                # Update body template
                if person_crop is not None:
                    identity['body_template'] = person_crop.copy()
                    
            elif name == identity['name']:
                # Same person confirmation
                identity['confidence'] = min(1.0, identity['confidence'] * 1.02)
                identity['stable'] = True
                identity['last_face_frame'] = frame_num
                identity['consecutive_back_frames'] = 0
                identity['frames_since_face_lost'] = 0
                
                if identity['identity_locked']:
                    identity['lock_strength'] = min(1.0, identity['lock_strength'] * 1.01)
        
        else:
            # No face detected
            identity['frames_since_face_lost'] += 1
            
            if pose in ['back_view', 'partial_back']:
                identity['consecutive_back_frames'] += 1
                
                # Maintain locked identity during back view
                if identity['identity_locked'] and identity['stable']:
                    decay_rate = 0.9995
                    
                    # Body matching for additional confidence
                    if identity['body_template'] is not None and person_crop is not None:
                        body_similarity = self.calculate_body_similarity(identity['body_template'], person_crop)
                        
                        if body_similarity > 0.5:
                            identity['confidence'] = min(1.0, identity['confidence'] * 1.005)
                            decay_rate = 0.9998
                    
                    identity['confidence'] *= decay_rate
                    
                    # Only unlock after extended period and very low confidence
                    if (identity['consecutive_back_frames'] > identity['unlock_threshold'] and 
                        identity['confidence'] < 0.1 and 
                        identity['lock_strength'] < 0.3):
                        identity['identity_locked'] = False
                else:
                    identity['confidence'] *= 0.985
            else:
                # Not back view but no face
                if identity['identity_locked'] and identity['frames_since_face_lost'] < identity['unlock_threshold']:
                    identity['confidence'] *= 0.998
                else:
                    if identity['identity_locked'] and identity['frames_since_face_lost'] > identity['unlock_threshold']:
                        identity['identity_locked'] = False
                    
                    identity['confidence'] *= self.IDENTITY_CONFIDENCE_DECAY
                    identity['consecutive_back_frames'] = 0
        
        # Prevent confidence from going too low for locked identities
        if identity['identity_locked']:
            min_locked_confidence = max(0.2, identity['lock_strength'] * 0.5)
            identity['confidence'] = max(identity['confidence'], min_locked_confidence)
        
        # Stability check
        if not identity['identity_locked'] and identity['confidence'] < self.MIN_IDENTITY_CONFIDENCE:
            identity['stable'] = False
    
    def get_consensus_identity_enhanced(self, track_id, frames_since_face):
        """Get consensus identity with enhanced tracking (from facenet_video_analyzer.py)"""
        if track_id not in self.track_identities:
            return 'Unknown', 0.0
        
        identity = self.track_identities[track_id]
        current_name = identity['name']
        current_conf = identity['confidence']
        
        # Enhanced logic for locked identities
        if identity.get('identity_locked', False):
            consecutive_back = identity.get('consecutive_back_frames', 0)
            frames_since_face_lost = identity.get('frames_since_face_lost', 0)
            lock_confidence = identity.get('lock_confidence', 0.0)
            
            # Maintain locked identity
            if frames_since_face_lost < self.BACK_VIEW_TOLERANCE_FRAMES:
                if consecutive_back > 60:
                    penalty = min(0.2, consecutive_back / (self.BACK_VIEW_TOLERANCE_FRAMES * 2))
                else:
                    penalty = min(0.1, frames_since_face_lost / self.BACK_VIEW_TOLERANCE_FRAMES)
                
                adjusted_conf = current_conf * (1.0 - penalty)
                
                if lock_confidence > 0.8 and adjusted_conf > 0.3:
                    return current_name, adjusted_conf
                elif lock_confidence > 0.6 and adjusted_conf > 0.4:
                    return current_name, adjusted_conf
        
        # Standard consensus logic
        if not identity['stable'] or current_conf < self.MIN_IDENTITY_CONFIDENCE:
            return 'Unknown', 0.0
        
        # Temporal consensus
        if track_id in self.track_face_history:
            history = self.track_face_history[track_id]
            if len(history) > 3:
                name_scores = defaultdict(float)
                total_weight = 0
                
                recent_history = list(history)[-90:]
                
                for i, record in enumerate(recent_history):
                    if record['name'] != 'Unknown':
                        age_weight = (0.98 ** (len(recent_history) - i - 1))
                        conf_weight = record['confidence']
                        
                        if conf_weight > 0.8:
                            conf_weight *= 2.0
                        elif conf_weight > 0.6:
                            conf_weight *= 1.5
                        
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
                    
                    if identity.get('identity_locked', False) and best_name == current_name:
                        consensus_conf = min(1.0, consensus_conf * 1.3)
                    
                    if frames_since_face > self.FACE_LOST_TOLERANCE:
                        penalty = min(0.3, (frames_since_face - self.FACE_LOST_TOLERANCE) / self.BACK_VIEW_TOLERANCE_FRAMES)
                        consensus_conf *= (1.0 - penalty)
                    
                    return best_name, consensus_conf
        
        return current_name, current_conf
    
    def process_video_with_analysis(self, show_video=True):
        """Process video with ArcFace recognition and enhanced tracking"""
        
        print(f"[INFO] Loading ArcFace models...")
        
        # Load models
        if not self.load_arcface_models():
            print("[ERROR] Failed to load ArcFace models")
            return False
        
        if not self.load_detection_models():
            print("[ERROR] Failed to load detection models")
            return False
        
        # Open video
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open video: {self.video_path}")
            return False
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"[INFO] Video properties:")
        print(f"  Resolution: {width}x{height}")
        print(f"  FPS: {fps}")
        print(f"  Total frames: {total_frames}")
        print(f"  Duration: {total_frames/fps:.1f} seconds")
        
        # Processing variables
        frame_count = 0
        all_recognitions = []
        
        # Processing settings
        PERSON_CONF_THRESHOLD = 0.6
        RESIZE_WIDTH = 720
        PROCESS_EVERY_N = 2
        
        print(f"\n[INFO] Processing video with ArcFace enhanced recognition...")
        if show_video:
            print("[INFO] Press 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_start_time = time.time()
            
            # Process every Nth frame
            if frame_count % PROCESS_EVERY_N != 0:
                frame_count += 1
                continue
            
            original_frame = frame.copy()
            display_frame = frame.copy()
            orig_h, orig_w = original_frame.shape[:2]
            
            # Resize for processing
            if RESIZE_WIDTH and orig_w > RESIZE_WIDTH:
                ratio = RESIZE_WIDTH / orig_w
                process_frame = cv2.resize(original_frame, (RESIZE_WIDTH, int(orig_h * ratio)))
            else:
                process_frame = original_frame
                ratio = 1.0
            
            try:
                # YOLO person detection with tracking
                results = self.yolo.track(
                    process_frame,
                    persist=True,
                    classes=[0],  # Only persons
                    conf=PERSON_CONF_THRESHOLD,
                    iou=0.7,
                    imgsz=640,
                    verbose=False
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
                            person_crop = original_frame[y1:y2, x1:x2]
                            
                            # Enhanced face recognition
                            face_result = self.recognize_face_in_crop_enhanced(
                                person_crop, original_frame, person_bbox, frame_count
                            )
                            
                            # Update tracking with enhanced identity management
                            self.update_track_identity_enhanced(track_id, face_result, person_crop, frame_count)
                            
                            # Update frame tracking
                            if face_result['name'] != 'Unknown':
                                self.track_last_face_frame[track_id] = frame_count
                            
                            # Get consensus identity
                            frames_since_face = frame_count - self.track_last_face_frame.get(track_id, frame_count)
                            identity_name, identity_conf = self.get_consensus_identity_enhanced(track_id, frames_since_face)
                            
                            # Get pose and tracking info
                            pose, pose_confidence = self.detect_person_pose_from_body(person_crop)
                            identity = self.track_identities.get(track_id, {})
                            is_locked = identity.get('identity_locked', False)
                            consecutive_back = identity.get('consecutive_back_frames', 0)
                            
                            # Store recognition data
                            recognition_data = {
                                'frame': frame_count,
                                'track_id': int(track_id),  # Ensure int type for JSON compatibility
                                'name': identity_name,
                                'confidence': identity_conf,
                                'inference_time_ms': face_result.get('inference_time', 0.0),
                                'pose': pose,
                                'pose_confidence': pose_confidence,
                                'is_locked': is_locked,
                                'consecutive_back_frames': consecutive_back,
                                'frames_since_face': frames_since_face
                            }
                            
                            all_recognitions.append(recognition_data)
                            
                            # Enhanced display
                            if identity_name != 'Unknown':
                                if is_locked:
                                    color = (0, 255, 0)  # Bright green for locked
                                    thickness = 3
                                else:
                                    color = (0, 200, 0)  # Green for known
                                    thickness = 2
                            else:
                                color = (0, 0, 255)  # Red for unknown
                                thickness = 2
                            
                            # Draw person bounding box
                            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, thickness)
                            
                            # Draw face box if available
                            if face_result.get('face_bbox') is not None:
                                fx1, fy1, fx2, fy2 = face_result['face_bbox']
                                cv2.rectangle(display_frame, (fx1, fy1), (fx2, fy2), (255, 255, 0), 2)
                            
                            # Enhanced label with pose and lock info
                            pose_info = ""
                            if pose == 'back_view':
                                pose_info = f" [BACK:{consecutive_back}f]"
                            elif pose == 'partial_back':
                                pose_info = f" [P_BACK:{consecutive_back}f]"
                            elif frames_since_face > 15:
                                pose_info = f" [NO_FACE:{frames_since_face}f]"
                            
                            lock_info = " 🔒" if is_locked else ""
                            
                            label = f"ID:{track_id} {identity_name}"
                            if identity_conf > 0:
                                label += f" ({identity_conf:.2f})"
                            label += pose_info + lock_info
                            
                            label_y = max(30, y1 - 10)
                            
                            # Draw label background
                            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                            cv2.rectangle(display_frame, (x1, label_y - label_h - 5), 
                                        (x1 + label_w + 5, label_y + 5), (0, 0, 0), -1)
                            
                            # Draw label text
                            cv2.putText(display_frame, label, (x1 + 2, label_y), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                            
                            # Draw pose indicator
                            if pose == 'back_view':
                                cv2.circle(display_frame, (x2 - 15, y1 + 15), 8, (255, 0, 255), -1)
                                cv2.putText(display_frame, "B", (x2 - 20, y1 + 20), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Performance overlay
                frame_time = (time.time() - frame_start_time) * 1000
                fps_actual = 1000.0 / max(frame_time, 1)
                locked_tracks = sum(1 for t in self.track_identities.values() if t.get('identity_locked', False))
                total_tracks = len(self.track_identities)
                back_view_tracks = sum(1 for t in self.track_identities.values() if t.get('consecutive_back_frames', 0) > 0)
                
                info_lines = [
                    f"Frame: {frame_count}/{total_frames} | FPS: {fps_actual:.1f}",
                    f"Tracks: {total_tracks} | Locked: {locked_tracks} | Back View: {back_view_tracks}"
                ]
                
                for i, info in enumerate(info_lines):
                    cv2.putText(display_frame, info, (10, 30 + i * 25), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Progress update
                if frame_count % 100 == 0:
                    progress = (frame_count / total_frames) * 100
                    print(f"  Progress: {progress:.1f}% (Frame {frame_count}/{total_frames}) - Locked: {locked_tracks}, Back View: {back_view_tracks}")
                
            except Exception as e:
                print(f"[ERROR] Frame {frame_count} processing failed: {e}")
            
            # Show frame
            if show_video:
                # Resize for display
                max_display_height = 720
                max_display_width = 1280
                
                h, w = display_frame.shape[:2]
                scale = min(max_display_width/w, max_display_height/h, 1.0)
                
                if scale < 1.0:
                    new_width = int(w * scale)
                    new_height = int(h * scale)
                    display_frame = cv2.resize(display_frame, (new_width, new_height))
                
                cv2.imshow('ArcFace Video Analysis with Enhanced Tracking', display_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n[INFO] User requested quit")
                    break
            
            frame_count += 1
        
        cap.release()
        if show_video:
            cv2.destroyAllWindows()
        
        # Store results
        self.results['recognitions'] = all_recognitions
        
        print(f"\n[INFO] Video processing completed with ArcFace enhanced tracking!")
        print(f"  Total recognitions: {len(all_recognitions)}")
        print(f"  Final locked tracks: {sum(1 for t in self.track_identities.values() if t.get('identity_locked', False))}")
        
        return True
    
    def calculate_simple_metrics(self):
        """Calculate performance metrics"""
        
        if not self.results['recognitions']:
            print("[ERROR] No recognition data to analyze")
            return None
        
        recognitions = self.results['recognitions']
        df = pd.DataFrame(recognitions)
        
        # 1. Average Confidence (only for known faces)
        known_faces = df[df['name'] != 'Unknown']
        if len(known_faces) > 0:
            avg_confidence = known_faces['confidence'].mean()
        else:
            avg_confidence = 0.0
        
        # 2. Average Inference Time
        avg_inference_time = df['inference_time_ms'].mean()
        
        # 3. Average Accuracy (recognition success rate)
        total_detections = len(df)
        successful_recognitions = len(known_faces)
        avg_accuracy = (successful_recognitions / total_detections) if total_detections > 0 else 0.0
        
        # Enhanced tracking stats
        back_view_detections = len(df[df['pose'] == 'back_view'])
        locked_detections = len(df[df['is_locked'] == True])
        
        stats = {
            'model_name': 'ArcFace',
            'video_name': self.video_name,
            'avg_confidence': avg_confidence,
            'avg_inference_time_ms': avg_inference_time,
            'avg_accuracy': avg_accuracy,
            'total_detections': total_detections,
            'successful_recognitions': successful_recognitions,
            'recognition_rate': avg_accuracy,
            'back_view_detections': back_view_detections,
            'back_view_percentage': (back_view_detections / total_detections) * 100 if total_detections > 0 else 0,
            'locked_detections': locked_detections,
            'locked_percentage': (locked_detections / total_detections) * 100 if total_detections > 0 else 0,
            'unique_tracks': len(self.track_identities),
            'locked_tracks': sum(1 for t in self.track_identities.values() if t.get('identity_locked', False))
        }
        
        return stats
    
    def create_simple_graph(self, stats):
        """Create ArcFace performance graph"""
        
        print(f"\n[INFO] Creating ArcFace performance graph...")
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Data for the 3 metrics
        metrics = ['Average\nConfidence', 'Average Inference\nTime (ms)', 'Average\nAccuracy']
        values = [
            stats['avg_confidence'],
            stats['avg_inference_time_ms'],
            stats['avg_accuracy']
        ]
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']  # Red, Teal, Blue
        
        # Create bar chart
        bars = ax.bar(metrics, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Customize the plot
        ax.set_title(f'ArcFace Performance Metrics - {self.video_name}\nwith Enhanced Identity Tracking', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_ylabel('Value', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, values)):
            height = bar.get_height()
            
            if i == 0:  # Confidence
                label_text = f'{value:.3f}'
            elif i == 1:  # Inference time
                label_text = f'{value:.1f}ms'
            else:  # Accuracy
                label_text = f'{value:.1%}'
            
            ax.text(bar.get_x() + bar.get_width()/2., height + max(values) * 0.01,
                   label_text, ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        # Set y-axis limits
        max_val = max(values)
        ax.set_ylim(0, max_val * 1.15)
        
        # Enhanced summary text box
        summary_text = f"""
📊 ARCFACE PERFORMANCE SUMMARY

🎯 CORE METRICS:
• Confidence: {stats['avg_confidence']:.3f}
• Inference Time: {stats['avg_inference_time_ms']:.1f}ms  
• Accuracy: {stats['avg_accuracy']:.1%}

📈 DETECTION STATS:
• Total Detections: {stats['total_detections']:,}
• Successful Recognitions: {stats['successful_recognitions']:,}
• Recognition Rate: {stats['recognition_rate']:.1%}

🔄 ENHANCED TRACKING:
• Back View: {stats['back_view_detections']:,} ({stats['back_view_percentage']:.1f}%)
• Locked Detections: {stats['locked_detections']:,} ({stats['locked_percentage']:.1f}%)
• Unique Tracks: {stats['unique_tracks']:,}
• Final Locked: {stats['locked_tracks']:,}

📁 VIDEO: {stats['video_name']}
⚙️  MODEL: ArcFace Enhanced
🖥️  DEVICE: {self.device.upper()}
        """
        
        # Position text box
        ax.text(0.02, 0.98, summary_text.strip(), transform=ax.transAxes, 
               fontsize=9, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcyan', alpha=0.8))
        
        # Tight layout
        plt.tight_layout()
        
        # Save graphs
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        graph_filename = f"arcface_performance_{self.video_name}_{timestamp}.png"
        graph_path = os.path.join(self.graphs_dir, graph_filename)
        
        plt.savefig(graph_path, dpi=300, bbox_inches='tight')
        print(f"[INFO] ✅ Saved ArcFace graph: {graph_path}")
        
        # Standard named version
        standard_graph_path = os.path.join(self.graphs_dir, f"arcface_metrics_{self.video_name}.png")
        plt.savefig(standard_graph_path, dpi=300, bbox_inches='tight')
        print(f"[INFO] ✅ Saved standard ArcFace graph: {standard_graph_path}")
        
        # Show the graph
        plt.show()
        
        return graph_path, standard_graph_path
    
    def save_results(self, stats):
        """Save results to ArcFace specific folders with enhanced comparison structure"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed recognition data
        if self.results['recognitions']:
            csv_filename = f"arcface_recognition_data_{self.video_name}_{timestamp}.csv"
            csv_path = os.path.join(self.data_dir, csv_filename)
            df = pd.DataFrame(self.results['recognitions'])
            df.to_csv(csv_path, index=False)
            print(f"[INFO] ✅ Saved recognition data: {csv_path}")
            
            # Standard named version
            standard_csv_path = os.path.join(self.data_dir, f"arcface_data_{self.video_name}.csv")
            df.to_csv(standard_csv_path, index=False)
            print(f"[INFO] ✅ Saved standard recognition data: {standard_csv_path}")
        
        # Save statistics
        stats_filename = f"arcface_stats_{self.video_name}_{timestamp}.json"
        stats_path = os.path.join(self.data_dir, stats_filename)
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"[INFO] ✅ Saved statistics: {stats_path}")
        
        # Standard named version
        standard_stats_path = os.path.join(self.data_dir, f"arcface_stats_{self.video_name}.json")
        with open(standard_stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"[INFO] ✅ Saved standard statistics: {standard_stats_path}")
        
        # Save tracking summary with proper key conversion
        tracking_summary = {
            'model_name': 'ArcFace',
            'video_name': self.video_name,
            'timestamp': timestamp,
            'total_tracks': len(self.track_identities),
            'track_details': {}
        }
        
        # Convert track_id keys to strings for JSON compatibility
        for track_id, track_data in self.track_identities.items():
            tracking_summary['track_details'][str(track_id)] = {
                'name': track_data.get('name', 'Unknown'),
                'confidence': track_data.get('confidence', 0.0),
                'identity_locked': track_data.get('identity_locked', False),
                'lock_strength': track_data.get('lock_strength', 0.0),
                'consecutive_back_frames': track_data.get('consecutive_back_frames', 0),
                'total_face_detections': track_data.get('total_face_detections', 0),
                'frames_since_face_lost': track_data.get('frames_since_face_lost', 0)
            }
        
        tracking_filename = f"arcface_tracking_{self.video_name}_{timestamp}.json"
        tracking_path = os.path.join(self.data_dir, tracking_filename)
        with open(tracking_path, 'w') as f:
            json.dump(tracking_summary, f, indent=2)
        print(f"[INFO] ✅ Saved tracking summary: {tracking_path}")
        
        # Enhanced comparison summaries
        # 1. Basic summary for model comparison
        summary_for_comparison = {
            'model': 'ArcFace',
            'video': self.video_name,
            'metrics': {
                'avg_confidence': stats['avg_confidence'],
                'avg_inference_time_ms': stats['avg_inference_time_ms'],
                'avg_accuracy': stats['avg_accuracy']
            },
            'additional_stats': {
                'total_detections': stats['total_detections'],
                'back_view_percentage': stats['back_view_percentage'],
                'locked_percentage': stats['locked_percentage']
            },
            'timestamp': timestamp
        }
        
        # Save to summaries subfolder
        comparison_summary_path = os.path.join(self.comparisons_dir, "summaries", f"arcface_summary_{self.video_name}.json")
        with open(comparison_summary_path, 'w') as f:
            json.dump(summary_for_comparison, f, indent=2)
        print(f"[INFO] ✅ Saved comparison summary: {comparison_summary_path}")
        
        # 2. Detailed comparison data
        detailed_comparison = {
            'model_info': {
                'name': 'ArcFace',
                'full_name': 'ArcFace Enhanced Recognition',
                'version': 'ResNet50 + Custom Enhancement',
                'architecture': 'ArcFace loss with enhanced tracking'
            },
            'video_info': {
                'name': self.video_name,
                'analysis_timestamp': timestamp
            },
            'performance_metrics': {
                'core_metrics': {
                    'average_confidence': stats['avg_confidence'],
                    'average_inference_time_ms': stats['avg_inference_time_ms'],
                    'average_accuracy': stats['avg_accuracy']
                },
                'detection_stats': {
                    'total_detections': stats['total_detections'],
                    'successful_recognitions': stats['successful_recognitions'],
                    'recognition_rate': stats['recognition_rate']
                },
                'advanced_tracking': {
                    'back_view_detections': stats['back_view_detections'],
                    'back_view_percentage': stats['back_view_percentage'],
                    'locked_detections': stats['locked_detections'],
                    'locked_percentage': stats['locked_percentage'],
                    'unique_tracks': stats['unique_tracks'],
                    'final_locked_tracks': stats['locked_tracks']
                }
            },
            'tracking_details': {
                'locking_mechanism': 'Enhanced identity locking with adaptive thresholds',
                'back_view_support': 'Body pose detection with temporal consistency',
                'face_detection': 'Multi-library support (InsightFace, MTCNN, Haar)',
                'embedding_method': 'InsightFace or custom ResNet50 backbone'
            }
        }
        
        # Save to detailed subfolder
        detailed_comparison_path = os.path.join(self.comparisons_dir, "detailed", f"arcface_detailed_{self.video_name}.json")
        with open(detailed_comparison_path, 'w') as f:
            json.dump(detailed_comparison, f, indent=2)
        print(f"[INFO] ✅ Saved detailed comparison: {detailed_comparison_path}")
        
        # 3. Copy graph to comparisons folder
        try:
            import shutil
            source_graph = os.path.join(self.graphs_dir, f"arcface_metrics_{self.video_name}.png")
            comparison_graph = os.path.join(self.comparisons_dir, "graphs", f"arcface_metrics_{self.video_name}.png")
            
            if os.path.exists(source_graph):
                shutil.copy2(source_graph, comparison_graph)
                print(f"[INFO] ✅ Saved comparison graph: {comparison_graph}")
        except Exception as e:
            print(f"[WARN] Could not copy graph to comparisons folder: {e}")
        
        return True
    
    def print_summary(self, stats):
        """Print enhanced summary with folder information"""
        
        print(f"\n" + "=" * 70)
        print("📊 ARCFACE PERFORMANCE SUMMARY WITH ORGANIZED OUTPUT")
        print("=" * 70)
        
        print(f"\n📁 OUTPUT ORGANIZATION:")
        print(f"  🎯 Base Directory: {self.base_output_dir}")
        print(f"  📊 ArcFace Results: {self.arcface_dir}")
        print(f"  📈 Graphs Folder: {self.graphs_dir}")
        print(f"  📋 Data Folder: {self.data_dir}")
        print(f"  🔄 Comparisons Folder: {self.comparisons_dir}")
        
        print(f"\n🎥 VIDEO ANALYSIS:")
        print(f"  📁 Video: {self.video_name}")
        print(f"  🎯 Total Detections: {stats['total_detections']:,}")
        print(f"  ✅ Successful Recognitions: {stats['successful_recognitions']:,}")
        print(f"  📊 Recognition Rate: {stats['recognition_rate']:.1%}")
        
        print(f"\n📈 CORE METRICS:")
        print(f"  💯 Average Confidence: {stats['avg_confidence']:.3f}")
        print(f"  ⚡ Average Inference Time: {stats['avg_inference_time_ms']:.1f}ms")
        print(f"  🎯 Average Accuracy: {stats['avg_accuracy']:.1%}")
        
        print(f"\n🔄 ENHANCED TRACKING:")
        print(f"  👤 Back View Detections: {stats['back_view_detections']:,} ({stats['back_view_percentage']:.1f}%)")
        print(f"  🔒 Identity Locked Detections: {stats['locked_detections']:,} ({stats['locked_percentage']:.1f}%)")
        print(f"  📋 Unique Tracks: {stats['unique_tracks']:,}")
        print(f"  🔐 Final Locked Tracks: {stats['locked_tracks']:,}")
        
        # Show locked tracks details
        locked_identities = {k: v for k, v in self.track_identities.items() if v.get('identity_locked', False)}
        if locked_identities:
            print(f"\n🔒 LOCKED IDENTITY DETAILS:")
            for track_id, track_data in locked_identities.items():
                print(f"  Track {track_id}: {track_data['name']} (strength: {track_data.get('lock_strength', 0):.2f})")
        
        print(f"\n📂 ENHANCED FOLDER STRUCTURE:")
        print(f"  {self.base_output_dir}/")
        print(f"  ├── ArcFace/")
        print(f"  │   ├── graphs/           # ArcFace performance graphs")
        print(f"  │   └── data/             # ArcFace detailed results")
        print(f"  ├── FaceNet/              # FaceNet results (if exists)")
        print(f"  ├── DlibCNN/              # DlibCNN results (if exists)")
        print(f"  └── Comparisons/")
        print(f"      ├── summaries/        # Basic comparison summaries")
        print(f"      ├── detailed/         # Detailed comparison data")
        print(f"      └── graphs/           # Comparison graphs")
        
        print(f"\n🔮 MODEL CAPABILITIES:")
        print(f"  🖥️  Device: {self.device.upper()}")
        print(f"  🧠 Embedding: {'InsightFace' if self.use_insightface else 'Custom ResNet50'}")
        print(f"  👁️  Face Detection: Multi-library support")
        print(f"  🎯 Enhanced Identity Locking ✅")
        print(f"  🔄 Temporal Consistency Tracking ✅")
        
        print("=" * 70)

def main():
    """Main function with ArcFace video analysis"""
    
    parser = argparse.ArgumentParser(description="ArcFace video analysis with enhanced tracking")
    parser.add_argument("--video", "-v", 
                       default=r"C:\Users\Alexa\OneDrive\Documents\Thesis\Reactive-CCTV-System-Using-Authorization-Level-Analysis\Mp4TESTING\ThesisMP4TEST3.mp4",
                       help="Path to video file")
    parser.add_argument("--output", "-o", 
                       default="model_comparison_results",
                       help="Base output directory for organized results")
    parser.add_argument("--no-display", "-n", 
                       action="store_true",
                       help="Run without video display (headless mode)")
    
    args = parser.parse_args()
    
    # Check if video exists
    if not os.path.exists(args.video):
        print(f"[ERROR] Video file not found: {args.video}")
        return
    
    print("🎬 ARCFACE VIDEO ANALYZER - ENHANCED TRACKING")
    print("=" * 50)
    
    if not args.no_display:
        print("[INFO] Video will be displayed with enhanced tracking indicators")
        print("[INFO] Press 'q' to quit")
        print("[INFO] Look for: 🔒 (locked identity), [BACK] (back view), Green (known), Red (unknown)")
    else:
        print("[INFO] Running in headless mode with enhanced tracking")
    
    # Initialize analyzer
    analyzer = ArcFaceVideoAnalyzer(args.video, args.output)
    
    # Process video
    if not analyzer.process_video_with_analysis(show_video=not args.no_display):
        print("[ERROR] Video processing failed")
        return
    
    # Calculate metrics
    stats = analyzer.calculate_simple_metrics()
    if stats is None:
        print("[ERROR] Statistics calculation failed")
        return
    
    # Create graph
    analyzer.create_simple_graph(stats)
    
    # Save results
    analyzer.save_results(stats)
    
    # Print summary
    analyzer.print_summary(stats)
    
    print(f"\n✅ ArcFace analysis complete!")
    print(f"📁 Results organized in: {analyzer.base_output_dir}")
    print(f"📊 ArcFace graphs: {analyzer.graphs_dir}")
    print(f"🔄 Ready to compare with other models!")

if __name__ == "__main__":
    main()
