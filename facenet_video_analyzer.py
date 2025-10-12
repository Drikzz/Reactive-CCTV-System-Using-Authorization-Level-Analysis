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
from datetime import datetime
from collections import defaultdict, deque
from pathlib import Path
from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization
from ultralytics import YOLO

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

class FaceNetVideoAnalyzer:
    def __init__(self, video_path, output_dir="model_comparison_results"):
        self.video_path = video_path
        self.video_name = Path(video_path).stem  # Get video name without extension
        
        # Create organized folder structure
        self.base_output_dir = output_dir
        self.facenet_dir = os.path.join(output_dir, "FaceNet")
        self.graphs_dir = os.path.join(self.facenet_dir, "graphs")
        self.data_dir = os.path.join(self.facenet_dir, "data")
        
        # Ensure all directories exist
        os.makedirs(self.facenet_dir, exist_ok=True)
        os.makedirs(self.graphs_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Also create comparison structure for future use
        os.makedirs(os.path.join(output_dir, "Comparisons"), exist_ok=True)
        
        self.results = {
            'frames': [],
            'recognitions': []
        }
        
        # Enhanced back view tracking settings (from facenet_main.py)
        self.BACK_VIEW_TOLERANCE_FRAMES = 600    # 20 seconds at 30fps
        self.BODY_MATCH_THRESHOLD = 0.5
        self.EXTENDED_MEMORY_FRAMES = 300
        self.MIN_BODY_MATCH_CONFIDENCE = 0.4
        self.POSE_HISTORY_LENGTH = 30
        self.IDENTITY_MEMORY_FRAMES = 90
        self.IDENTITY_CONFIDENCE_DECAY = 0.995
        self.MIN_IDENTITY_CONFIDENCE = 0.15
        self.FACE_LOST_TOLERANCE = 180
        
        # Tracking data structures
        self.track_identities = {}
        self.track_face_history = defaultdict(lambda: deque(maxlen=self.EXTENDED_MEMORY_FRAMES))
        self.track_body_history = defaultdict(lambda: deque(maxlen=60))
        self.track_last_face_frame = {}
        
        print(f"[INFO] FaceNet Video Analyzer initialized")
        print(f"[INFO] Video: {video_path}")
        print(f"[INFO] FaceNet Output: {self.facenet_dir}")
        print(f"[INFO] Graphs will be saved to: {self.graphs_dir}")
    
    def detect_person_pose_from_body(self, person_crop):
        """Detect if person is facing away based on body characteristics (from facenet_main.py)"""
        try:
            if person_crop is None or person_crop.size == 0:
                return "unknown", 0.0
            
            h, w = person_crop.shape[:2]
            if h < 80 or w < 40:
                return "frontal", 0.5
            
            gray = cv2.cvtColor(person_crop, cv2.COLOR_BGR2GRAY)
            
            # Divide person into regions
            head_region = gray[:h//3, :]          # Top 1/3
            torso_region = gray[h//3:2*h//3, :]   # Middle 1/3
            
            # Analyze head region for back-of-head characteristics
            head_edges = cv2.Canny(head_region, 30, 100)
            head_edge_density = np.count_nonzero(head_edges) / max(head_edges.size, 1)
            
            # Analyze symmetry (back view tends to be more symmetric)
            left_half = gray[:, :w//2]
            right_half = cv2.flip(gray[:, w//2:], 1)
            
            # Resize to match if needed
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
            
            # Low edge density in head suggests back of head
            if head_edge_density < 0.06:
                back_score += 0.4
            
            # High symmetry suggests back view
            if symmetry_score > 0.7:
                back_score += 0.3
            
            # Brightness analysis
            head_brightness = np.mean(head_region)
            torso_brightness = np.mean(torso_region)
            
            # Back view: head and torso have similar brightness (clothing/hair)
            brightness_ratio = head_brightness / max(torso_brightness, 1)
            if 0.8 < brightness_ratio < 1.3:
                back_score += 0.2
            
            # Texture analysis - back view has less facial texture
            head_std = np.std(head_region)
            if head_std < 25:  # Low texture variance
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
                
        except Exception as e:
            return "frontal", 0.5

    def calculate_body_similarity(self, template_crop, current_crop):
        """Calculate body similarity score between two person crops (from facenet_main.py)"""
        try:
            if template_crop is None or current_crop is None:
                return 0.0
            if template_crop.size == 0 or current_crop.size == 0:
                return 0.0

            # Resize to a consistent size for histogram comparison
            h, w = 128, 64
            tpl = cv2.resize(template_crop, (w, h))
            cur = cv2.resize(current_crop, (w, h))

            # Convert to HSV and compute 2D histograms on H and S channels
            tpl_hsv = cv2.cvtColor(tpl, cv2.COLOR_BGR2HSV)
            cur_hsv = cv2.cvtColor(cur, cv2.COLOR_BGR2HSV)

            hist_size = [50, 60]
            hist_ranges = [0, 180, 0, 256]
            tpl_hist = cv2.calcHist([tpl_hsv], [0, 1], None, hist_size, hist_ranges)
            cur_hist = cv2.calcHist([cur_hsv], [0, 1], None, hist_size, hist_ranges)

            cv2.normalize(tpl_hist, tpl_hist)
            cv2.normalize(cur_hist, cur_hist)

            # Use correlation which gives 1 for identical histograms
            score = cv2.compareHist(tpl_hist, cur_hist, cv2.HISTCMP_CORREL)
            # Correlation may be in [-1,1]; map to [0,1]
            score = max(0.0, min(1.0, (score + 1.0) / 2.0))

            return float(score)
        except Exception:
            return 0.0

    def update_track_identity(self, track_id, face_result, person_crop, frame_num):
        """Enhanced identity tracking with back view support (from facenet_main.py)"""
        
        name = face_result['name']
        conf = face_result['confidence']
        
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
        
        # Store body crop for matching
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
        
        # Handle different scenarios
        if name != 'Unknown':
            identity['total_face_detections'] += 1
            
            # Strong face detection - lock identity
            if conf > 0.7:
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
                        body_similarity = self.calculate_body_similarity(identity['body_template'], person_crop)
                        
                        if body_similarity > 0.5:  # Lower threshold for back view
                            # Boost confidence for good body match
                            identity['confidence'] = min(1.0, identity['confidence'] * 1.005)
                            decay_rate = 0.9998  # Even less decay with body match
                    
                    identity['confidence'] *= decay_rate
                    
                    # Only consider unlocking after extended period AND very low confidence
                    if (identity['consecutive_back_frames'] > identity['unlock_threshold'] and 
                        identity['confidence'] < 0.1 and 
                        identity['lock_strength'] < 0.3):
                        
                        identity['identity_locked'] = False
                else:
                    # Not locked, decay more quickly
                    identity['confidence'] *= 0.985
            else:
                # Not back view but no face - could be profile or temporary occlusion
                if identity['identity_locked'] and identity['frames_since_face_lost'] < identity['unlock_threshold']:
                    # Maintain locked identity for reasonable periods without face
                    identity['confidence'] *= 0.998  # Very slow decay
                else:
                    # Regular decay or unlock
                    if identity['identity_locked'] and identity['frames_since_face_lost'] > identity['unlock_threshold']:
                        identity['identity_locked'] = False
                    
                    identity['confidence'] *= self.IDENTITY_CONFIDENCE_DECAY
                    identity['consecutive_back_frames'] = 0
        
        # Prevent confidence from going too low for locked identities
        if identity['identity_locked']:
            min_locked_confidence = max(0.2, identity['lock_strength'] * 0.5)
            identity['confidence'] = max(identity['confidence'], min_locked_confidence)
        
        # Stability check - but don't mark unstable if locked
        if not identity['identity_locked'] and identity['confidence'] < self.MIN_IDENTITY_CONFIDENCE:
            identity['stable'] = False

    def get_consensus_identity(self, track_id, frames_since_face):
        """Enhanced consensus with identity locking and back view support (from facenet_main.py)"""
        if track_id not in self.track_identities:
            return 'Unknown', 0.0
        
        identity = self.track_identities[track_id]
        current_name = identity['name']
        current_conf = identity['confidence']
        
        # If identity is locked, be much more conservative about changing it
        if identity.get('identity_locked', False):
            consecutive_back = identity.get('consecutive_back_frames', 0)
            frames_since_face_lost = identity.get('frames_since_face_lost', 0)
            lock_confidence = identity.get('lock_confidence', 0.0)
            
            # Maintain locked identity for reasonable periods
            if frames_since_face_lost < self.BACK_VIEW_TOLERANCE_FRAMES:
                # Apply minimal penalty for locked identity
                if consecutive_back > 60:  # Extended back view
                    penalty = min(0.2, consecutive_back / (self.BACK_VIEW_TOLERANCE_FRAMES * 2))
                else:
                    penalty = min(0.1, frames_since_face_lost / self.BACK_VIEW_TOLERANCE_FRAMES)
                
                adjusted_conf = current_conf * (1.0 - penalty)
                
                # Keep locked identity if original lock was strong
                if lock_confidence > 0.8 and adjusted_conf > 0.3:
                    return current_name, adjusted_conf
                elif lock_confidence > 0.6 and adjusted_conf > 0.4:
                    return current_name, adjusted_conf
        
        # Standard consensus logic for unlocked identities
        if not identity['stable'] or current_conf < self.MIN_IDENTITY_CONFIDENCE:
            return 'Unknown', 0.0
        
        # Use temporal consensus from history with stronger weighting for locked identities
        if track_id in self.track_face_history:
            history = self.track_face_history[track_id]
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
                    if frames_since_face > self.FACE_LOST_TOLERANCE:
                        penalty = min(0.3, (frames_since_face - self.FACE_LOST_TOLERANCE) / self.BACK_VIEW_TOLERANCE_FRAMES)
                        consensus_conf *= (1.0 - penalty)
                    
                    return best_name, consensus_conf
        
        return current_name, current_conf

    def recognize_face_in_crop(self, person_crop, original_frame, person_bbox):
        """Use exact recognition mechanics from facenet_main.py"""
        if person_crop is None or person_crop.size == 0:
            return {'name': 'Unknown', 'confidence': 0.0, 'face_bbox': None, 'inference_time': 0.0}
        
        try:
            inference_start = time.time()
            
            # Apply preprocessing (from facenet_main.py)
            processed_crop = person_crop.copy()
            
            # Gamma correction
            gray_check = cv2.cvtColor(processed_crop, cv2.COLOR_BGR2GRAY)
            mean_brightness = float(np.mean(gray_check))
            if mean_brightness < 80.0:  # GAMMA_TARGET_MEAN * 0.8
                gamma = 100.0 / max(mean_brightness, 1.0)
                gamma = max(min(gamma, 1.8), 0.6)
                invGamma = 1.0 / gamma
                table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(256)]).astype('uint8')
                processed_crop = cv2.LUT(processed_crop, table)
            
            # CLAHE
            person_rgb = cv2.cvtColor(processed_crop, cv2.COLOR_BGR2RGB)
            lab = cv2.cvtColor(person_rgb, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            limg = cv2.merge((cl, a, b))
            processed_rgb = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
            processed_crop = cv2.cvtColor(processed_rgb, cv2.COLOR_RGB2BGR)
            
            # Face detection with filtering (from facenet_main.py)
            person_rgb = cv2.cvtColor(processed_crop, cv2.COLOR_BGR2RGB)
            face_boxes, face_probs = self.mtcnn.detect(person_rgb)
            
            # Filter quality faces
            if face_boxes is not None and face_probs is not None:
                filtered_boxes = []
                filtered_probs = []
                min_size = 30
                min_prob = 0.8
                
                for box, prob in zip(face_boxes, face_probs):
                    face_area = (box[2] - box[0]) * (box[3] - box[1])
                    if prob >= min_prob and face_area >= min_size * min_size:
                        filtered_boxes.append(box)
                        filtered_probs.append(prob)
                
                face_boxes = np.array(filtered_boxes) if filtered_boxes else None
                face_probs = np.array(filtered_probs) if filtered_probs else None
            
            if face_boxes is None or len(face_boxes) == 0:
                inference_time = (time.time() - inference_start) * 1000
                return {'name': 'Unknown', 'confidence': 0.0, 'face_bbox': None, 'inference_time': inference_time}
            
            # Take the most confident face
            best_idx = np.argmax(face_probs)
            face_box = face_boxes[best_idx]
            face_prob = face_probs[best_idx]
            
            # Extract face region
            fx1, fy1, fx2, fy2 = face_box.astype(int)
            fx1, fy1 = max(0, fx1), max(0, fy1)
            fx2, fy2 = min(person_crop.shape[1], fx2), min(person_crop.shape[0], fy2)
            
            if fx2 <= fx1 or fy2 <= fy1:
                inference_time = (time.time() - inference_start) * 1000
                return {'name': 'Unknown', 'confidence': 0.0, 'face_bbox': None, 'inference_time': inference_time}
            
            face_crop = processed_crop[fy1:fy2, fx1:fx2]
            
            # Convert face coordinates back to original frame
            px1, py1, px2, py2 = person_bbox
            face_x1_orig = px1 + fx1
            face_y1_orig = py1 + fy1
            face_x2_orig = px1 + fx2
            face_y2_orig = py1 + fy2
            
            # Face recognition (exact from facenet_main.py)
            if self.classifier is not None and self.label_encoder is not None:
                try:
                    # Prepare face for recognition
                    face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                    face_resized = cv2.resize(face_rgb, (160, 160))
                    
                    # Convert to tensor and get embedding (exact from facenet_main.py)
                    tensor = torch.from_numpy(face_resized).permute(2, 0, 1).float()
                    tensor = fixed_image_standardization(tensor)
                    
                    with torch.no_grad():
                        embedding = self.embedder(tensor.unsqueeze(0).to(self.device))
                        embedding = embedding.cpu().numpy()[0]
                    
                    # Normalize embedding
                    emb_norm = embedding / (np.linalg.norm(embedding) + 1e-10)
                    
                    # Classify (exact thresholds from facenet_main.py)
                    probs = self.classifier.predict_proba([emb_norm])[0]
                    pred = np.argmax(probs)
                    confidence = probs[pred]
                    
                    # Check thresholds (exact from facenet_main.py)
                    RECOG_THRESHOLD = 0.45
                    RECOG_MARGIN = 0.08
                    
                    sorted_probs = np.sort(probs)[::-1]
                    top2_prob = sorted_probs[1] if len(sorted_probs) > 1 else 0.0
                    
                    inference_time = (time.time() - inference_start) * 1000
                    
                    if confidence >= RECOG_THRESHOLD and (confidence - top2_prob) >= RECOG_MARGIN:
                        candidate = self.label_encoder.inverse_transform([pred])[0]
                        
                        # Distance check if available (from facenet_main.py)
                        if self.centroids is not None and self.dist_threshold is not None:
                            centroid = self.centroids.get(candidate)
                            if centroid is not None:
                                dist = float(np.linalg.norm(np.asarray(emb_norm, dtype=np.float32) - np.asarray(centroid, dtype=np.float32)))
                                if dist <= self.dist_threshold:
                                    return {
                                        'name': candidate,
                                        'confidence': confidence,
                                        'face_bbox': (face_x1_orig, face_y1_orig, face_x2_orig, face_y2_orig),
                                        'inference_time': inference_time
                                    }
                            else:
                                return {
                                    'name': candidate,
                                    'confidence': confidence,
                                    'face_bbox': (face_x1_orig, face_y1_orig, face_x2_orig, face_y2_orig),
                                    'inference_time': inference_time
                                }
                        else:
                            return {
                                'name': candidate,
                                'confidence': confidence,
                                'face_bbox': (face_x1_orig, face_y1_orig, face_x2_orig, face_y2_orig),
                                'inference_time': inference_time
                            }
                    
                except Exception as e:
                    print(f"[WARN] Face recognition failed: {e}")
            
            inference_time = (time.time() - inference_start) * 1000
            return {
                'name': 'Unknown', 
                'confidence': 0.0, 
                'face_bbox': (face_x1_orig, face_y1_orig, face_x2_orig, face_y2_orig),
                'inference_time': inference_time
            }
            
        except Exception as e:
            print(f"[ERROR] Face recognition in crop failed: {e}")
            inference_time = (time.time() - inference_start) * 1000
            return {'name': 'Unknown', 'confidence': 0.0, 'face_bbox': None, 'inference_time': inference_time}
    
    def process_video_with_analysis(self, show_video=True):
        """Process video using exact FaceNet mechanics with back view tracking"""
        
        # Initialize models (exact from facenet_main.py)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[INFO] Using device: {self.device}")
        
        # Load models
        self.yolo = YOLO("models/YOLOv8/yolov8n.pt")
        self.mtcnn = MTCNN(
            image_size=160, 
            margin=0, 
            keep_all=True, 
            device=self.device, 
            post_process=False,
            min_face_size=30,
            thresholds=[0.6, 0.7, 0.7],
            factor=0.7
        )
        self.embedder = InceptionResnetV1(pretrained="vggface2").to(self.device).eval()
        
        # Load classifier (exact from facenet_main.py)
        try:
            MODELS_DIR = os.path.join("models", "FaceNet")
            SVM_PATH = os.path.join(MODELS_DIR, "facenet_svm.joblib")
            LE_PATH = os.path.join(MODELS_DIR, "label_encoder.joblib")
            THR_PATH = os.path.join(MODELS_DIR, "distance_threshold.npy")
            
            if not os.path.exists(SVM_PATH) or not os.path.exists(LE_PATH):
                raise FileNotFoundError("SVM or label encoder not found in models/FaceNet")
            
            self.classifier = joblib.load(SVM_PATH)
            self.label_encoder = joblib.load(LE_PATH)
            
            # Load centroids
            centroids_path = os.path.join(MODELS_DIR, 'class_centroids.pkl')
            self.centroids = None
            self.dist_threshold = None
            
            try:
                if os.path.exists(centroids_path):
                    self.centroids = joblib.load(centroids_path)
                    for k, v in list(self.centroids.items()):
                        arr = np.asarray(v, dtype=np.float32)
                        n = np.linalg.norm(arr) + 1e-10
                        self.centroids[k] = (arr / n)
                    print(f"[INFO] Loaded {len(self.centroids)} class centroids")
            except Exception as e:
                print(f"[WARN] Failed to load centroids: {e}")
                self.centroids = None
            
            if os.path.exists(THR_PATH):
                self.dist_threshold = float(np.load(THR_PATH))
                print(f"[INFO] Loaded distance threshold: {self.dist_threshold:.3f}")
            
            print(f"[INFO] Loaded classifier. Classes: {list(self.label_encoder.classes_)}")
            
        except Exception as e:
            print(f"[ERROR] Failed to load models: {e}")
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
        
        # Recognition thresholds (from facenet_main.py)
        PERSON_CONF_THRESHOLD = 0.6
        RESIZE_WIDTH = 720
        PROCESS_EVERY_N = 2
        
        print(f"\n[INFO] Processing video with FaceNet mechanics and back view tracking...")
        if show_video:
            print("[INFO] Press 'q' to quit")
        
        # Colors for tracks
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
            (0, 255, 255), (128, 0, 128), (255, 165, 0), (0, 128, 0), (128, 128, 128)
        ]
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_start_time = time.time()
            
            # Process every Nth frame for performance
            if frame_count % PROCESS_EVERY_N != 0:
                frame_count += 1
                continue
            
            original_frame = frame.copy()
            display_frame = frame.copy()
            orig_h, orig_w = original_frame.shape[:2]
            
            # Resize for processing (from facenet_main.py)
            if RESIZE_WIDTH and orig_w > RESIZE_WIDTH:
                ratio = RESIZE_WIDTH / orig_w
                process_frame = cv2.resize(original_frame, (RESIZE_WIDTH, int(orig_h * ratio)))
            else:
                process_frame = original_frame
                ratio = 1.0
            
            try:
                # YOLO person detection with tracking (exact from facenet_main.py)
                results = self.yolo.track(
                    process_frame,
                    persist=True,
                    tracker="bytetrack.yaml",
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
                            
                            # Face recognition using exact mechanics
                            face_result = self.recognize_face_in_crop(person_crop, original_frame, person_bbox)
                            
                            # Update tracking history with back view support
                            self.update_track_identity(track_id, face_result, person_crop, frame_count)
                            
                            # Update frame numbers
                            if face_result['name'] != 'Unknown':
                                self.track_last_face_frame[track_id] = frame_count
                            
                            # Get consensus identity with back view tracking
                            frames_since_face = frame_count - self.track_last_face_frame.get(track_id, frame_count)
                            identity_name, identity_conf = self.get_consensus_identity(track_id, frames_since_face)
                            
                            # Get pose and tracking info for display
                            pose, pose_confidence = self.detect_person_pose_from_body(person_crop)
                            identity = self.track_identities.get(track_id, {})
                            is_locked = identity.get('identity_locked', False)
                            consecutive_back = identity.get('consecutive_back_frames', 0)
                            
                            # Store recognition data with enhanced tracking info
                            recognition_data = {
                                'frame': frame_count,
                                'track_id': track_id,
                                'name': identity_name,
                                'confidence': identity_conf,
                                'inference_time_ms': face_result['inference_time'],
                                'pose': pose,
                                'pose_confidence': pose_confidence,
                                'is_locked': is_locked,
                                'consecutive_back_frames': consecutive_back,
                                'frames_since_face': frames_since_face
                            }
                            
                            all_recognitions.append(recognition_data)
                            
                            # Enhanced display logic
                            # Choose display color based on identity status and pose
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
                            
                            # Draw person bounding box
                            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, thickness)
                            
                            # Draw face box if available
                            if face_result['face_bbox'] is not None:
                                fx1, fy1, fx2, fy2 = face_result['face_bbox']
                                cv2.rectangle(display_frame, (fx1, fy1), (fx2, fy2), (255, 255, 0), 2)
                            
                            # Enhanced label with pose and tracking information
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
                            
                            # Draw pose indicator on person
                            if pose == 'back_view':
                                cv2.circle(display_frame, (x2 - 15, y1 + 15), 8, (255, 0, 255), -1)
                                cv2.putText(display_frame, "B", (x2 - 20, y1 + 20), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Calculate frame processing time
                frame_time = (time.time() - frame_start_time) * 1000
                
                # Enhanced performance overlay with tracking stats
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
                
                cv2.imshow('FaceNet Video Analysis with Back View Tracking', display_frame)
                
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
        
        print(f"\n[INFO] Video processing completed with back view tracking!")
        print(f"  Total recognitions: {len(all_recognitions)}")
        print(f"  Final locked tracks: {sum(1 for t in self.track_identities.values() if t.get('identity_locked', False))}")
        
        return True
    
    def calculate_simple_metrics(self):
        """Calculate the 3 requested metrics with back view tracking insights"""
        
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
        
        # Additional back view tracking stats
        back_view_detections = len(df[df['pose'] == 'back_view'])
        locked_detections = len(df[df['is_locked'] == True])
        
        stats = {
            'model_name': 'FaceNet',
            'video_name': self.video_name,
            'avg_confidence': avg_confidence,
            'avg_inference_time_ms': avg_inference_time,
            'avg_accuracy': avg_accuracy,
            'total_detections': total_detections,
            'successful_recognitions': successful_recognitions,
            'recognition_rate': avg_accuracy,
            # Enhanced stats with back view tracking
            'back_view_detections': back_view_detections,
            'back_view_percentage': (back_view_detections / total_detections) * 100 if total_detections > 0 else 0,
            'locked_detections': locked_detections,
            'locked_percentage': (locked_detections / total_detections) * 100 if total_detections > 0 else 0,
            'unique_tracks': len(self.track_identities),
            'locked_tracks': sum(1 for t in self.track_identities.values() if t.get('identity_locked', False))
        }
        
        return stats
    
    def create_simple_graph(self, stats):
        """Create simple graph with only 3 metrics - saved to FaceNet specific folder"""
        
        print(f"\n[INFO] Creating FaceNet performance graph...")
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Data for the 3 metrics
        metrics = ['Average\nConfidence', 'Average Inference\nTime (ms)', 'Average\nAccuracy']
        values = [
            stats['avg_confidence'],
            stats['avg_inference_time_ms'],
            stats['avg_accuracy']
        ]
        colors = ['#2E8B57', '#FF6B35', '#4169E1']  # Sea green, Orange red, Royal blue
        
        # Create bar chart
        bars = ax.bar(metrics, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Customize the plot
        ax.set_title(f'FaceNet Performance Metrics - {self.video_name}\nwith Back View Tracking', 
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
        
        # Set y-axis limits to show all values properly
        max_val = max(values)
        ax.set_ylim(0, max_val * 1.15)
        
        # Enhanced summary text box with back view tracking info
        summary_text = f"""
📊 FACENET PERFORMANCE SUMMARY

🎯 CORE METRICS:
• Confidence: {stats['avg_confidence']:.3f}
• Inference Time: {stats['avg_inference_time_ms']:.1f}ms  
• Accuracy: {stats['avg_accuracy']:.1%}

📈 DETECTION STATS:
• Total Detections: {stats['total_detections']:,}
• Successful Recognitions: {stats['successful_recognitions']:,}
• Recognition Rate: {stats['recognition_rate']:.1%}

🔄 BACK VIEW TRACKING:
• Back View: {stats['back_view_detections']:,} ({stats['back_view_percentage']:.1f}%)
• Locked Detections: {stats['locked_detections']:,} ({stats['locked_percentage']:.1f}%)
• Unique Tracks: {stats['unique_tracks']:,}
• Final Locked: {stats['locked_tracks']:,}

📁 VIDEO: {stats['video_name']}
⚙️  DEVICE: {'CUDA' if torch.cuda.is_available() else 'CPU'}
        """
        
        # Position text box
        ax.text(0.02, 0.98, summary_text.strip(), transform=ax.transAxes, 
               fontsize=9, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
        
        # Tight layout
        plt.tight_layout()
        
        # Save the graph to FaceNet specific folder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        graph_filename = f"facenet_performance_{self.video_name}_{timestamp}.png"
        graph_path = os.path.join(self.graphs_dir, graph_filename)
        
        plt.savefig(graph_path, dpi=300, bbox_inches='tight')
        print(f"[INFO] ✅ Saved FaceNet graph: {graph_path}")
        
        # Also save a standard named version for easy comparison
        standard_graph_path = os.path.join(self.graphs_dir, f"facenet_metrics_{self.video_name}.png")
        plt.savefig(standard_graph_path, dpi=300, bbox_inches='tight')
        print(f"[INFO] ✅ Saved standard FaceNet graph: {standard_graph_path}")
        
        # Show the graph
        plt.show()
        
        return graph_path, standard_graph_path
    
    def save_results(self, stats):
        """Save results to FaceNet specific folders"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed recognition data
        if self.results['recognitions']:
            csv_filename = f"facenet_recognition_data_{self.video_name}_{timestamp}.csv"
            csv_path = os.path.join(self.data_dir, csv_filename)
            df = pd.DataFrame(self.results['recognitions'])
            df.to_csv(csv_path, index=False)
            print(f"[INFO] ✅ Saved recognition data: {csv_path}")
            
            # Also save standard named version
            standard_csv_path = os.path.join(self.data_dir, f"facenet_data_{self.video_name}.csv")
            df.to_csv(standard_csv_path, index=False)
            print(f"[INFO] ✅ Saved standard recognition data: {standard_csv_path}")
        
        # Save statistics
        stats_filename = f"facenet_stats_{self.video_name}_{timestamp}.json"
        stats_path = os.path.join(self.data_dir, stats_filename)
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"[INFO] ✅ Saved statistics: {stats_path}")
        
        # Also save standard named version
        standard_stats_path = os.path.join(self.data_dir, f"facenet_stats_{self.video_name}.json")
        with open(standard_stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"[INFO] ✅ Saved standard statistics: {standard_stats_path}")
        
        # Save tracking summary
        tracking_summary = {
            'model_name': 'FaceNet',
            'video_name': self.video_name,
            'timestamp': timestamp,
            'total_tracks': len(self.track_identities),
            'track_details': {}
        }
        
        for track_id, identity in self.track_identities.items():
            tracking_summary['track_details'][track_id] = {
                'name': identity['name'],
                'confidence': identity['confidence'],
                'is_locked': identity.get('identity_locked', False),
                'lock_strength': identity.get('lock_strength', 0.0),
                'consecutive_back_frames': identity.get('consecutive_back_frames', 0),
                'total_face_detections': identity.get('total_face_detections', 0),
                'frames_since_face_lost': identity.get('frames_since_face_lost', 0)
            }
        
        tracking_filename = f"facenet_tracking_{self.video_name}_{timestamp}.json"
        tracking_path = os.path.join(self.data_dir, tracking_filename)
        with open(tracking_path, 'w') as f:
            json.dump(tracking_summary, f, indent=2)
        print(f"[INFO] ✅ Saved tracking summary: {tracking_path}")
        
        # Create summary file for easy comparison
        summary_for_comparison = {
            'model': 'FaceNet',
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
        
        comparison_path = os.path.join(self.base_output_dir, "Comparisons", f"facenet_summary_{self.video_name}.json")
        with open(comparison_path, 'w') as f:
            json.dump(summary_for_comparison, f, indent=2)
        print(f"[INFO] ✅ Saved comparison summary: {comparison_path}")
        
        return True
    
    def print_summary(self, stats):
        """Print enhanced summary with folder information"""
        
        print(f"\n" + "=" * 70)
        print("📊 FACENET PERFORMANCE SUMMARY WITH ORGANIZED OUTPUT")
        print("=" * 70)
        
        print(f"\n📁 OUTPUT ORGANIZATION:")
        print(f"  🎯 Base Directory: {self.base_output_dir}")
        print(f"  📊 FaceNet Results: {self.facenet_dir}")
        print(f"  📈 Graphs Folder: {self.graphs_dir}")
        print(f"  📋 Data Folder: {self.data_dir}")
        
        print(f"\n🎥 VIDEO ANALYSIS:")
        print(f"  📁 Video: {self.video_name}")
        print(f"  🎯 Total Detections: {stats['total_detections']:,}")
        print(f"  ✅ Successful Recognitions: {stats['successful_recognitions']:,}")
        print(f"  📊 Recognition Rate: {stats['recognition_rate']:.1%}")
        
        print(f"\n📈 CORE METRICS:")
        print(f"  💯 Average Confidence: {stats['avg_confidence']:.3f}")
        print(f"  ⚡ Average Inference Time: {stats['avg_inference_time_ms']:.1f}ms")
        print(f"  🎯 Average Accuracy: {stats['avg_accuracy']:.1%}")
        
        print(f"\n🔄 BACK VIEW TRACKING:")
        print(f"  👤 Back View Detections: {stats['back_view_detections']:,} ({stats['back_view_percentage']:.1f}%)")
        print(f"  🔒 Identity Locked Detections: {stats['locked_detections']:,} ({stats['locked_percentage']:.1f}%)")
        print(f"  📋 Unique Tracks: {stats['unique_tracks']:,}")
        print(f"  🔐 Final Locked Tracks: {stats['locked_tracks']:,}")
        
        # Show locked tracks details
        locked_identities = {k: v for k, v in self.track_identities.items() if v.get('identity_locked', False)}
        if locked_identities:
            print(f"\n🔒 LOCKED IDENTITY DETAILS:")
            for track_id, identity in locked_identities.items():
                print(f"  Track {track_id}: {identity['name']} (conf: {identity['confidence']:.3f}, strength: {identity.get('lock_strength', 0):.3f})")
        
        print(f"\n📂 FOLDER STRUCTURE CREATED:")
        print(f"  {self.base_output_dir}/")
        print(f"  ├── FaceNet/")
        print(f"  │   ├── graphs/           # FaceNet performance graphs")
        print(f"  │   └── data/             # FaceNet detailed results")
        print(f"  └── Comparisons/          # Model comparison summaries")
        
        print(f"\n🔮 READY FOR MODEL COMPARISON:")
        print(f"  • Add other model analyzers (OpenFace, ArcFace, etc.)")
        print(f"  • Each model gets its own subfolder")
        print(f"  • Comparison summaries in /Comparisons/")
        print(f"  • Easy to create comparison graphs later")
        
        print("=" * 70)

def main():
    """Main function with organized folder structure"""
    
    parser = argparse.ArgumentParser(description="FaceNet video analysis with organized output structure")
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
    
    print("🎬 FACENET VIDEO ANALYZER - ORGANIZED OUTPUT")
    print("=" * 50)
    
    if not args.no_display:
        print("[INFO] Video will be displayed with back view tracking indicators")
        print("[INFO] Press 'q' to quit")
        print("[INFO] Look for: 🔒 (locked identity), [BACK] (back view), 'B' circle (back pose)")
    else:
        print("[INFO] Running in headless mode with back view tracking")
    
    # Initialize analyzer with organized output
    analyzer = FaceNetVideoAnalyzer(args.video, args.output)
    
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
    
    print(f"\n✅ FaceNet analysis complete!")
    print(f"📁 Results organized in: {analyzer.base_output_dir}")
    print(f"📊 FaceNet graphs: {analyzer.graphs_dir}")
    print(f"🔄 Ready to add other model analyzers!")

if __name__ == "__main__":
    main()

import os
import glob
import json
import time
import matplotlib.pyplot as plt
import numpy as np
from facenet_video_analyzer import FaceNetVideoAnalyzer

def make_dummy_stats(video_name="test_video"):
    return {
        'model_name': 'FaceNet',
        'video_name': video_name,
        'avg_confidence': 0.75,
        'avg_inference_time_ms': 12.5,
        'avg_accuracy': 0.82,
        'total_detections': 100,
        'successful_recognitions': 82,
        'recognition_rate': 0.82,
        'back_view_detections': 5,
        'back_view_percentage': 5.0,
        'locked_detections': 10,
        'locked_percentage': 10.0,
        'unique_tracks': 7,
        'locked_tracks': 3
    }

def test_create_simple_graph_writes_files(tmp_path, monkeypatch):
    # Prevent interactive display from blocking tests
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)

    out_dir = tmp_path / "model_comparison_results"
    video_path = str(tmp_path / "dummy_video.mp4")
    analyzer = FaceNetVideoAnalyzer(video_path, output_dir=str(out_dir))

    stats = make_dummy_stats(video_name=analyzer.video_name)

    graph_path, standard_graph_path = analyzer.create_simple_graph(stats)

    assert isinstance(graph_path, str)
    assert isinstance(standard_graph_path, str)
    assert os.path.exists(graph_path), f"Graph file not created: {graph_path}"
    assert os.path.exists(standard_graph_path), f"Standard graph file not created: {standard_graph_path}"
    assert os.path.getsize(graph_path) > 0
    assert os.path.getsize(standard_graph_path) > 0

def test_create_simple_graph_content_paths_in_expected_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)

    out_dir = tmp_path / "results_folder"
    video_path = str(tmp_path / "my_test_video.mp4")
    analyzer = FaceNetVideoAnalyzer(video_path, output_dir=str(out_dir))

    stats = make_dummy_stats(video_name=analyzer.video_name)

    graph_path, standard_graph_path = analyzer.create_simple_graph(stats)

    graphs_dir = os.path.abspath(analyzer.graphs_dir)
    assert os.path.commonpath([os.path.abspath(graph_path), graphs_dir]) == graphs_dir
    assert os.path.commonpath([os.path.abspath(standard_graph_path), graphs_dir]) == graphs_dir

    assert analyzer.video_name in os.path.basename(graph_path)
    assert analyzer.video_name in os.path.basename(standard_graph_path)

def test_save_results_writes_expected_files(tmp_path, monkeypatch):
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)

    out_dir = tmp_path / "model_comparison_results"
    video_path = str(tmp_path / "dummy_video2.mp4")
    analyzer = FaceNetVideoAnalyzer(video_path, output_dir=str(out_dir))

    # Minimal recognitions entries
    analyzer.results['recognitions'] = [
        {
            'frame': 1,
            'track_id': 1,
            'name': 'Alice',
            'confidence': 0.9,
            'inference_time_ms': 10.0,
            'pose': 'frontal',
            'pose_confidence': 0.9,
            'is_locked': True,
            'consecutive_back_frames': 0,
            'frames_since_face': 0
        },
        {
            'frame': 2,
            'track_id': 2,
            'name': 'Unknown',
            'confidence': 0.0,
            'inference_time_ms': 12.0,
            'pose': 'profile',
            'pose_confidence': 0.5,
            'is_locked': False,
            'consecutive_back_frames': 0,
            'frames_since_face': 10
        }
    ]

    # Use Python int keys (json accepts int keys) to avoid numpy.int64 problem
    analyzer.track_identities = {
        1: {
            'name': 'Alice',
            'confidence': 0.9,
            'identity_locked': True,
            'lock_strength': 0.8,
            'consecutive_back_frames': 0,
            'total_face_detections': 5,
            'frames_since_face_lost': 0
        },
        2: {
            'name': 'Unknown',
            'confidence': 0.0,
            'identity_locked': False,
            'lock_strength': 0.0,
            'consecutive_back_frames': 0,
            'total_face_detections': 0,
            'frames_since_face_lost': 5
        }
    }

    stats = make_dummy_stats(video_name=analyzer.video_name)

    result = analyzer.save_results(stats)
    assert result is True

    # Check data directory contains stats json and tracking json
    data_files = list(Path(analyzer.data_dir).glob("*.json"))
    assert any("facenet_stats_" in p.name for p in data_files), "No stats JSON found"
    assert any("facenet_tracking_" in p.name for p in data_files), "No tracking JSON found"

    # Check Comparisons summary exists
    comparison_file = os.path.join(analyzer.base_output_dir, "Comparisons", f"facenet_summary_{analyzer.video_name}.json")
    assert os.path.exists(comparison_file), f"Comparison summary not created: {comparison_file}"

    # Validate that the saved tracking JSON can be loaded and has expected top-level keys
    tracking_json_paths = [p for p in data_files if "facenet_tracking_" in p.name]
    assert len(tracking_json_paths) > 0
    with open(tracking_json_paths[0], 'r') as fh:
        js = json.load(fh)
    assert js.get('model_name') == 'FaceNet'
    assert 'track_details' in js
    # track_details keys should be strings or ints - ensure conversion for at least one entry
    assert len(js['track_details']) >= 1