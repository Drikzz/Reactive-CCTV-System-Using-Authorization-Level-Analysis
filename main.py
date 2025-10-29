"""
Integrated Face Recognition + Behavior Recognition System

Combines:
- FaceNet face recognition with authorization levels
- MobileNetV3-Small behavior classification
- YOLOv8 person detection and tracking (ByteTrack)
- Conditional alerts based on authorization levels

Authorization Levels:
- AUTHORIZED: Face tracking only, no behavior monitoring, never alerts
- PARTIAL: Face + behavior tracking, alerts only if no authorized person present
- UNAUTHORIZED: Full tracking, immediate alerts on suspicious behavior

Usage:
    # Run with video file
    python main.py --video path/to/video.mp4
    
    # Run with webcam
    python main.py --webcam
    
    # Save outputs
    python main.py --video path/to/video.mp4 --save outputs/result.mp4 --save-logs
"""

import os
import sys
import cv2
import torch
import numpy as np
import argparse
import time
import joblib
from pathlib import Path
from datetime import datetime
from collections import defaultdict, deque
from PIL import Image
from threading import Thread, Event
from queue import Queue, Empty

# Deep learning models
from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision import transforms, models
from torch import nn
from ultralytics import YOLO

# Import authorization configuration
from authorization_config_main import (
    get_authorization_level, should_monitor_behavior, is_suspicious_behavior,
    should_trigger_alert, format_display_name, get_level_color, get_level_thickness,
    get_log_directory, get_annotated_directory, AUTHORIZED, PARTIAL, UNAUTHORIZED,
    SUSPICIOUS_BEHAVIORS, BEHAVIOR_CONFIDENCE_THRESHOLD, print_authorization_summary
)


class IntegratedRecognitionSystem:
    """
    Integrated Face Recognition + Behavior Recognition System
    """
    
    def __init__(self, 
                 yolo_model_path='models/YOLOv8/yolov8m.pt',
                 facenet_model_path='models/FaceNet/inception_resnet_v1.pt',
                 behavior_model_path='models/mobilenetv3-small/mobilenetv3_small_feature_extraction.pth',
                 device=None,
                 save_logs=False,
                 min_class_conf: float = None,
                 smoothing_window: int = 7):
        """
        Initialize integrated system
        
        Args:
            yolo_model_path: Path to YOLOv8 model
            facenet_model_path: Path to FaceNet model
            behavior_model_path: Path to MobileNetV3-Small behavior model
            device: Device to use ('cuda' or 'cpu')
            save_logs: Whether to save logs and frames
            smoothing_window: Temporal smoothing window size (default: 7 frames)
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.save_logs = save_logs
        self.smoothing_window = max(1, int(smoothing_window))
        
        print("\n" + "="*80)
        print("INITIALIZING INTEGRATED RECOGNITION SYSTEM")
        print("="*80)
        print(f"Device: {self.device}")
        print(f"Save Logs: {self.save_logs}")
        
        # Initialize YOLO detector
        print(f"\nLoading YOLOv8 from: {yolo_model_path}")
        self.yolo = YOLO(yolo_model_path)
        
        # Initialize FaceNet components
        print(f"Loading FaceNet components...")
        self._init_facenet(facenet_model_path)
        
        # Initialize Behavior Recognition
        print(f"Loading Behavior model from: {behavior_model_path}")
        self._init_behavior_model(behavior_model_path)
        
        # Tracking state
        self.track_identities = {}  # {track_id: identity_info}
        self.track_behaviors = {}   # {track_id: behavior_info}
        
        # Enhanced face tracking with pose and body matching
        self.track_face_history = defaultdict(lambda: deque(maxlen=90))  # Recent face recognitions
        self.track_body_history = defaultdict(lambda: deque(maxlen=60))  # Body crops for matching
        self.track_last_face_frame = {}  # {track_id: last frame with face detected}
        
        # Identity persistence settings
        self.identity_memory_frames = 90
        self.identity_confidence_decay = 0.995
        self.min_identity_confidence = 0.15
        self.face_lost_tolerance = 180
        self.back_view_tolerance_frames = 600
        self.body_match_threshold = 0.5
        self.min_body_match_confidence = 0.4
        
        # ByteTrack tuning constants
        self.bytetrack_track_thresh = 0.6    # High threshold for track confirmation
        self.bytetrack_track_buffer = 90     # Frames to keep lost tracks
        self.bytetrack_match_thresh = 0.7    # Matching threshold
        
        # Periodic face saving (for known faces)
        self.known_last_saved = {}  # {name: last_save_datetime}
        self.known_save_interval_minutes = 3  # Save known faces every N minutes
        
        # Threading settings
        self.capture_queue_size = 4
        self.display_queue_size = 2
        
        # Alert tracking
        self.alerts_sent = []
        self.alert_cooldown = {}  # {track_id: last_alert_time}
        self.alert_cooldown_seconds = 5.0
        
        # Minimum behavior classification confidence (can be overridden by CLI)
        # If not provided, fall back to config value
        if min_class_conf is not None:
            try:
                self.min_class_conf = float(min_class_conf)
            except Exception:
                self.min_class_conf = BEHAVIOR_CONFIDENCE_THRESHOLD
        else:
            self.min_class_conf = BEHAVIOR_CONFIDENCE_THRESHOLD

        # CSV logging
        self.csv_path = None
        if self.save_logs:
            self._setup_logging()
        
        print("\n✓ System initialized successfully!")
        print("="*80 + "\n")
    
    def _init_facenet(self, model_path):
        """Initialize FaceNet face recognition components"""
        # MTCNN face detector
        self.mtcnn = MTCNN(
            image_size=160,
            margin=0,
            keep_all=True,
            device=self.device
        )
        
        # FaceNet embedder
        self.embedder = InceptionResnetV1(pretrained='vggface2').to(self.device).eval()
        
        # Load classifier
        models_dir = "models/FaceNet"
        svm_path = os.path.join(models_dir, "facenet_svm.joblib")
        le_path = os.path.join(models_dir, "label_encoder.joblib")
        
        if not os.path.exists(svm_path) or not os.path.exists(le_path):
            raise FileNotFoundError(f"FaceNet models not found in {models_dir}")
        
        self.classifier = joblib.load(svm_path)
        self.label_encoder = joblib.load(le_path)
        
        # Load class centroids for distance-based recognition
        centroids_path = os.path.join(models_dir, 'class_centroids.pkl')
        if os.path.exists(centroids_path):
            import pickle
            try:
                with open(centroids_path, 'rb') as f:
                    self.centroids = pickle.load(f)
                print(f"  ✓ Loaded class centroids for {len(self.centroids)} classes")
            except Exception as e:
                print(f"  ⚠ Warning: Could not load centroids ({e}). Using decision function fallback.")
                self.centroids = None
        else:
            self.centroids = None
        
        # Recognition thresholds
        self.face_recognition_threshold = 0.45
        self.face_recognition_margin = 0.08
        
        print(f"  ✓ FaceNet classes: {list(self.label_encoder.classes_)}")
    
    def _init_behavior_model(self, model_path):
        """Initialize MobileNetV3-Small behavior recognition model"""
        checkpoint = torch.load(model_path, map_location=self.device)
        num_classes = checkpoint['num_classes']
        
        # Build MobileNetV3-Small
        self.behavior_model = models.mobilenet_v3_small(weights=None)
        in_features = self.behavior_model.classifier[3].in_features
        self.behavior_model.classifier[3] = nn.Linear(in_features, num_classes)
        
        # Load weights
        self.behavior_model.load_state_dict(checkpoint['model_state_dict'])
        self.behavior_model = self.behavior_model.to(self.device)
        self.behavior_model.eval()
        
        # Get class names
        self.behavior_classes = checkpoint.get('class_names', [f'Class_{i}' for i in range(num_classes)])
        
        # Behavior preprocessing
        self.behavior_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Behavior classification cache
        self.behavior_cache = {}  # {track_id: {'class': name, 'confidence': conf, 'last_frame': num}}
        self.behavior_interval = 10  # Classify every N frames
        
        # Temporal smoothing: store recent probability vectors per track
        self.prob_history = defaultdict(lambda: deque(maxlen=self.smoothing_window))
        
        print(f"  ✓ Behavior classes: {self.behavior_classes}")
        print(f"  ✓ Temporal smoothing window: {self.smoothing_window} frames")
    
    def _setup_logging(self):
        """Setup logging directories and CSV"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create directories
        for level in [AUTHORIZED, PARTIAL, UNAUTHORIZED]:
            log_dir = get_log_directory(level)
            ann_dir = get_annotated_directory(level)
            os.makedirs(log_dir, exist_ok=True)
            os.makedirs(ann_dir, exist_ok=True)
        
        # Alert logs
        os.makedirs("logs/main_system/alerts", exist_ok=True)
        os.makedirs("logs/main_system/suspicious_behavior", exist_ok=True)
        
        # CSV for per-frame results
        self.csv_path = f"logs/main_system/tracking_log_{timestamp}.csv"
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
        
        # Write CSV header
        with open(self.csv_path, 'w') as f:
            f.write("timestamp,frame_num,track_id,name,auth_level,behavior_class,behavior_conf,alert_triggered\n")
        
        print(f"  ✓ CSV logging to: {self.csv_path}")
    
    def detect_person_pose_from_body(self, person_crop):
        """
        Detect if person is facing away based on body characteristics.
        
        Returns:
            tuple: (pose, confidence) where pose is 'front', 'back', or 'side'
        """
        try:
            if person_crop is None or person_crop.size == 0:
                return ('unknown', 0.0)
            
            h, w = person_crop.shape[:2]
            if h < 80 or w < 40:
                return ('unknown', 0.0)
            
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
            left_half = left_half[:, :min_w]
            right_half = right_half[:, :min_w]
            
            # Calculate symmetry score
            symmetry = cv2.matchTemplate(left_half, right_half, cv2.TM_CCOEFF_NORMED)[0][0]
            
            # Heuristic scoring
            back_score = 0.0
            
            # Low edge density in head = smoother back of head
            if head_edge_density < 0.15:
                back_score += 0.3
            
            # High symmetry = likely back view
            if symmetry > 0.7:
                back_score += 0.4
            elif symmetry > 0.5:
                back_score += 0.2
            
            # Determine pose
            if back_score > 0.5:
                return ('back', back_score)
            elif back_score > 0.3:
                return ('side', 0.5)
            else:
                return ('front', 1.0 - back_score)
                
        except Exception as e:
            return ('unknown', 0.0)
    
    def calculate_body_similarity(self, template_crop, current_crop):
        """
        Calculate body similarity score between two person crops.
        Uses HSV color histograms with correlation.
        
        Returns:
            float: Similarity score [0.0, 1.0] where 1.0 means identical
        """
        try:
            if template_crop is None or current_crop is None:
                return 0.0
            
            if template_crop.size == 0 or current_crop.size == 0:
                return 0.0
            
            # Resize both to same size for comparison
            target_size = (64, 128)
            template_resized = cv2.resize(template_crop, target_size)
            current_resized = cv2.resize(current_crop, target_size)
            
            # Convert to HSV
            template_hsv = cv2.cvtColor(template_resized, cv2.COLOR_BGR2HSV)
            current_hsv = cv2.cvtColor(current_resized, cv2.COLOR_BGR2HSV)
            
            # Calculate histograms
            hist_bins = [30, 32]  # H, S bins
            hist_ranges = [0, 180, 0, 256]  # H: 0-180, S: 0-256
            
            template_hist = cv2.calcHist([template_hsv], [0, 1], None, hist_bins, hist_ranges)
            current_hist = cv2.calcHist([current_hsv], [0, 1], None, hist_bins, hist_ranges)
            
            # Normalize
            cv2.normalize(template_hist, template_hist, 0, 1, cv2.NORM_MINMAX)
            cv2.normalize(current_hist, current_hist, 0, 1, cv2.NORM_MINMAX)
            
            # Compare using correlation
            similarity = cv2.compareHist(template_hist, current_hist, cv2.HISTCMP_CORREL)
            
            # Clamp to [0, 1]
            similarity = max(0.0, min(1.0, similarity))
            
            return float(similarity)
            
        except Exception:
            return 0.0
    
    def apply_clahe_rgb(self, rgb_image, clip_limit=2.0, tile_grid_size=8):
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to RGB image.
        Improves face detection in low-light conditions.
        
        Args:
            rgb_image: RGB image (numpy array)
            clip_limit: Threshold for contrast limiting
            tile_grid_size: Size of grid for histogram equalization
            
        Returns:
            Enhanced RGB image
        """
        if rgb_image is None or rgb_image.size == 0:
            return rgb_image
        
        img = rgb_image
        if img.dtype != 'uint8':
            img = (np.clip(img, 0.0, 1.0) * 255).astype('uint8')
        
        # Convert to LAB color space
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
        cl = clahe.apply(l)
        
        # Merge and convert back to RGB
        limg = cv2.merge((cl, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    
    def estimate_brightness(self, gray_img):
        """Return mean brightness of grayscale image"""
        if gray_img is None or gray_img.size == 0:
            return 0.0
        return float(np.mean(gray_img))
    
    def auto_gamma_correction(self, img, target_mean=100.0, max_gamma=1.8, min_gamma=0.6):
        """
        Apply automatic gamma correction based on image brightness.
        Brightens dark images, darkens overly bright images.
        
        Args:
            img: BGR image
            target_mean: Target mean brightness (0-255)
            max_gamma: Maximum gamma value
            min_gamma: Minimum gamma value
            
        Returns:
            Gamma-corrected BGR image
        """
        if img is None or img.size == 0:
            return img
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mean = self.estimate_brightness(gray)
        if mean <= 0:
            return img
        
        gamma = float(target_mean) / float(mean)
        gamma = max(min(gamma, max_gamma), min_gamma)
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(256)]).astype('uint8')
        return cv2.LUT(img, table)
    
    def filter_quality_faces(self, face_boxes, face_probs, min_size=30, min_prob=0.8):
        """
        Filter faces by size and detection confidence.
        
        Args:
            face_boxes: Detected face bounding boxes
            face_probs: Detection probabilities
            min_size: Minimum face size (width or height)
            min_prob: Minimum detection probability
            
        Returns:
            Tuple of (filtered_boxes, filtered_probs) or (None, None)
        """
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
    
    def recognize_face(self, person_crop, original_frame):
        """
        Recognize face within person crop
        
        Returns:
            dict: {'name': str, 'confidence': float} or None if no face found
        """
        if person_crop is None or person_crop.size == 0:
            return None
        
        try:
            # Apply preprocessing for better face detection
            processed_crop = person_crop.copy()
            
            # Auto gamma correction for low-light images
            gray_check = cv2.cvtColor(processed_crop, cv2.COLOR_BGR2GRAY)
            mean_brightness = self.estimate_brightness(gray_check)
            if mean_brightness < 80:  # Image is dark
                processed_crop = self.auto_gamma_correction(processed_crop, target_mean=100.0)
            
            # Convert BGR to RGB
            crop_rgb = cv2.cvtColor(processed_crop, cv2.COLOR_BGR2RGB)
            
            # Apply CLAHE for contrast enhancement
            crop_rgb = self.apply_clahe_rgb(crop_rgb, clip_limit=2.0, tile_grid_size=8)
            
            # Detect faces with MTCNN
            face_boxes, face_probs = self.mtcnn.detect(crop_rgb)
            
            # Filter faces by quality
            face_boxes, face_probs = self.filter_quality_faces(face_boxes, face_probs, min_size=30, min_prob=0.8)
            
            if face_boxes is None or len(face_boxes) == 0:
                return None
            
            # Use highest confidence face
            best_idx = np.argmax(face_probs)
            face_box = face_boxes[best_idx]
            face_prob = face_probs[best_idx]
            
            # Extract face crop
            x1, y1, x2, y2 = [int(c) for c in face_box]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(crop_rgb.shape[1], x2), min(crop_rgb.shape[0], y2)
            
            if x2 <= x1 or y2 <= y1:
                return None
            
            face_crop = crop_rgb[y1:y2, x1:x2]
            
            # Get embedding
            face_pil = Image.fromarray(face_crop)
            face_tensor = transforms.functional.to_tensor(face_pil).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                embedding = self.embedder(face_tensor).cpu().numpy()[0]
            
            # Classify
            prediction = self.classifier.predict([embedding])[0]
            name = self.label_encoder.inverse_transform([prediction])[0]
            
            # Calculate confidence using centroid distance if available
            if self.centroids is not None and name in self.centroids:
                centroid = self.centroids[name]
                distance = np.linalg.norm(embedding - centroid)
                confidence = max(0.0, 1.0 - distance)
            else:
                # Use decision function
                decision = self.classifier.decision_function([embedding])
                confidence = float(np.max(decision))
            
            # Apply threshold
            if confidence < self.face_recognition_threshold:
                name = "Unknown"
                confidence = 1.0 - confidence
            
            return {'name': name, 'confidence': confidence}
            
        except Exception as e:
            print(f"[ERROR] Face recognition failed: {e}")
            return None
    
    def update_track_identity(self, track_id, face_result, person_crop, frame_num):
        """
        Enhanced identity tracking with pose detection and body matching.
        
        Args:
            track_id: Track identifier
            face_result: Result from recognize_face() or None
            person_crop: Current person crop
            frame_num: Current frame number
        """
        # Detect person pose
        pose, pose_confidence = self.detect_person_pose_from_body(person_crop)
        
        # Initialize tracking data for new track
        if track_id not in self.track_identities:
            if face_result:
                self.track_identities[track_id] = {
                    'name': face_result['name'],
                    'confidence': face_result['confidence'],
                    'stable': False,
                    'identity_locked': False,
                    'lock_confidence_threshold': 0.75,
                    'frames_confirmed': 0,
                    'last_seen_frame': frame_num,
                    'pose_history': deque(maxlen=30),
                    'body_template': person_crop.copy() if person_crop is not None else None
                }
            else:
                self.track_identities[track_id] = {
                    'name': 'Unknown',
                    'confidence': 0.0,
                    'stable': False,
                    'identity_locked': False,
                    'lock_confidence_threshold': 0.75,
                    'frames_confirmed': 0,
                    'last_seen_frame': frame_num,
                    'pose_history': deque(maxlen=30),
                    'body_template': person_crop.copy() if person_crop is not None else None
                }
        
        identity = self.track_identities[track_id]
        identity['pose_history'].append((pose, pose_confidence, frame_num))
        identity['last_seen_frame'] = frame_num
        
        # Store body crop for matching
        if person_crop is not None and person_crop.size > 0:
            self.track_body_history[track_id].append({
                'crop': person_crop.copy(),
                'frame': frame_num
            })
        
        # Update face history
        if face_result:
            self.track_face_history[track_id].append({
                'name': face_result['name'],
                'confidence': face_result['confidence'],
                'frame': frame_num,
                'pose': pose,
                'pose_conf': pose_confidence
            })
            self.track_last_face_frame[track_id] = frame_num
        
        # Handle identity updates
        if face_result and face_result['name'] != 'Unknown':
            name = face_result['name']
            conf = face_result['confidence']
            
            # Lock identity if high confidence
            if conf > identity['lock_confidence_threshold'] and not identity['identity_locked']:
                identity['frames_confirmed'] += 1
                if identity['frames_confirmed'] >= 5:
                    identity['identity_locked'] = True
                    identity['stable'] = True
                    print(f"  [LOCKED] Track {track_id} → {name} (conf: {conf:.2f})")
            
            # Update if same person or unlocked
            if identity['name'] == name or not identity['identity_locked']:
                identity['name'] = name
                identity['confidence'] = conf
                identity['body_template'] = person_crop.copy()
        
        else:
            # No face detected - handle based on pose and body matching
            frames_since_face = frame_num - self.track_last_face_frame.get(track_id, frame_num)
            
            # If back view and identity is locked, maintain identity using body matching
            if pose == 'back' and identity.get('identity_locked', False):
                if frames_since_face < self.back_view_tolerance_frames:
                    # Try body matching
                    if identity.get('body_template') is not None:
                        body_sim = self.calculate_body_similarity(identity['body_template'], person_crop)
                        
                        if body_sim > self.body_match_threshold:
                            # Maintain identity with decayed confidence
                            identity['confidence'] *= self.identity_confidence_decay
                            identity['confidence'] = max(identity['confidence'], self.min_body_match_confidence)
                        else:
                            # Body doesn't match - decay faster
                            identity['confidence'] *= 0.95
                    else:
                        # No body template - gentle decay
                        identity['confidence'] *= self.identity_confidence_decay
                else:
                    # Too long without face - mark unstable
                    identity['stable'] = False
                    identity['confidence'] *= 0.9
            else:
                # Front/side view but no face - decay confidence
                if frames_since_face > self.face_lost_tolerance:
                    identity['stable'] = False
                identity['confidence'] *= self.identity_confidence_decay
        
        # Prevent confidence from going too low for locked identities
        if identity['identity_locked']:
            identity['confidence'] = max(identity['confidence'], self.min_identity_confidence)
        
        # Stability check - but don't mark unstable if locked
        if not identity['identity_locked'] and identity['confidence'] < self.min_identity_confidence:
            identity['stable'] = False
    
    def get_track_identity(self, track_id):
        """Get current identity for a track"""
        if track_id in self.track_identities:
            identity = self.track_identities[track_id]
            return {
                'name': identity['name'],
                'confidence': identity['confidence'],
                'stable': identity.get('stable', False),
                'locked': identity.get('identity_locked', False)
            }
        return {'name': 'Unknown', 'confidence': 0.0, 'stable': False, 'locked': False}
    
    def classify_behavior(self, person_crop):
        """
        Classify behavior from person crop
        
        Returns:
            dict: {'class_name': str, 'confidence': float, 'probs': np.ndarray}
        """
        try:
            # Convert BGR to RGB
            crop_rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
            crop_pil = Image.fromarray(crop_rgb)
            
            # Transform
            crop_tensor = self.behavior_transform(crop_pil).unsqueeze(0).to(self.device)
            
            # Inference
            with torch.no_grad():
                outputs = self.behavior_model(crop_tensor)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probs, 1)
                probs_np = probs.squeeze(0).cpu().numpy()
            
            class_id = predicted.item()
            class_name = self.behavior_classes[class_id]
            conf = confidence.item()
            
            # Apply confidence threshold (use instance setting)
            if conf < self.min_class_conf:
                class_name = "Neutral"
                conf = 1.0 - conf
            
            return {
                'class_name': class_name, 
                'confidence': conf,
                'probs': probs_np
            }
            
        except Exception as e:
            print(f"[ERROR] Behavior classification failed: {e}")
            return {
                'class_name': 'Neutral', 
                'confidence': 0.0,
                'probs': np.zeros(len(self.behavior_classes))
            }
    
    def should_reclassify_behavior(self, track_id, current_frame):
        """Check if behavior should be re-classified for this track"""
        if track_id not in self.behavior_cache:
            return True
        
        last_frame = self.behavior_cache[track_id].get('last_frame', 0)
        return (current_frame - last_frame) >= self.behavior_interval
    
    def check_authorized_present(self, frame_identities):
        """
        Check if any authorized person is present in current frame
        
        Args:
            frame_identities: List of identity dicts for current frame
        
        Returns:
            bool: True if at least one authorized person present
        """
        for identity in frame_identities:
            if identity.get('auth_level') == AUTHORIZED:
                return True
        return False
    
    def log_frame_data(self, timestamp, frame_num, track_id, name, auth_level, 
                       behavior_class, behavior_conf, alert_triggered):
        """Log per-frame data to CSV"""
        if not self.save_logs or self.csv_path is None:
            return
        
        try:
            with open(self.csv_path, 'a') as f:
                f.write(f"{timestamp},{frame_num},{track_id},{name},{auth_level},"
                       f"{behavior_class},{behavior_conf:.3f},{alert_triggered}\n")
        except Exception as e:
            print(f"[ERROR] Failed to log to CSV: {e}")
    
    def save_alert_frames(self, frame, person_crop, track_id, name, auth_level, 
                          behavior_class, frame_num):
        """Save full frame and cropped person for alerts"""
        if not self.save_logs:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        
        # Alert directory
        alert_dir = f"logs/main_system/alerts/{auth_level.lower()}"
        os.makedirs(alert_dir, exist_ok=True)
        
        # Save full frame
        full_path = os.path.join(alert_dir, f"alert_full_{timestamp}_t{track_id}_f{frame_num}.jpg")
        cv2.imwrite(full_path, frame)
        
        # Save person crop
        crop_path = os.path.join(alert_dir, f"alert_crop_{timestamp}_t{track_id}_f{frame_num}.jpg")
        cv2.imwrite(crop_path, person_crop)
        
        print(f"  [ALERT SAVED] {name} ({auth_level}) - {behavior_class}")
    
    def can_send_alert(self, track_id):
        """Check if alert cooldown has passed for this track"""
        if track_id not in self.alert_cooldown:
            return True
        
        elapsed = time.time() - self.alert_cooldown[track_id]
        return elapsed >= self.alert_cooldown_seconds
    
    def send_alert(self, track_id, name, auth_level, behavior_class, frame_num):
        """Send alert (placeholder for UI integration)"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        alert_msg = {
            'timestamp': timestamp,
            'frame': frame_num,
            'track_id': track_id,
            'name': name,
            'auth_level': auth_level,
            'behavior': behavior_class
        }
        
        self.alerts_sent.append(alert_msg)
        self.alert_cooldown[track_id] = time.time()
        
        # Print to console
        print(f"\n🚨 ALERT: {name} ({auth_level}) - {behavior_class} [Frame {frame_num}, Track {track_id}]")
        
        # TODO: Send to UI/notification system
    
    def save_known_face(self, frame, person_crop, face_bbox, name, frame_num):
        """
        Periodically save known face detections to disk.
        Only saves if minimum time interval has passed since last save.
        
        Args:
            frame: Full frame
            person_crop: Cropped person image
            face_bbox: Face bounding box (x1, y1, x2, y2) in frame coordinates
            name: Recognized person name
            frame_num: Current frame number
        """
        if not self.save_logs or name == "Unknown":
            return
        
        from datetime import datetime, timedelta
        
        # Check if enough time has passed since last save
        now = datetime.now()
        last_saved = self.known_last_saved.get(name, datetime.min)
        
        if (now - last_saved).total_seconds() < self.known_save_interval_minutes * 60:
            return  # Skip - too soon
        
        # Update last saved time
        self.known_last_saved[name] = now
        
        # Create known faces directory
        known_dir = os.path.join("logs", "main_system", "known_faces", name)
        os.makedirs(known_dir, exist_ok=True)
        
        timestamp = now.strftime("%Y%m%d_%H%M%S_%f")
        
        # Save full frame with annotation
        frame_annotated = frame.copy()
        if face_bbox is not None:
            x1, y1, x2, y2 = [int(c) for c in face_bbox]
            cv2.rectangle(frame_annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame_annotated, name, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        frame_path = os.path.join(known_dir, f"frame_{timestamp}_f{frame_num}.jpg")
        cv2.imwrite(frame_path, frame_annotated)
        
        # Save person crop
        if person_crop is not None and person_crop.size > 0:
            crop_path = os.path.join(known_dir, f"crop_{timestamp}_f{frame_num}.jpg")
            cv2.imwrite(crop_path, person_crop)
    
    def grab_frames(self, cap, frame_q, stop_event):
        """
        Capture frames from video source and put into queue.
        Runs in separate thread.
        """
        frame_num = 0
        try:
            while not stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    break
                
                timestamp = time.time()
                
                # Put frame in queue (blocks if queue is full)
                try:
                    frame_q.put((frame_num, timestamp, frame), timeout=1.0)
                    frame_num += 1
                except:
                    # Queue full, skip frame
                    continue
                    
        except Exception as e:
            print(f"[ERROR] Frame capture error: {e}")
        finally:
            stop_event.set()
    
    def display_frames(self, display_q, stop_event):
        """
        Display processed frames from queue.
        Runs in separate thread.
        """
        while not stop_event.is_set() or not display_q.empty():
            try:
                frame_data = display_q.get(timeout=0.5)
                if frame_data is None:
                    continue
                
                frame_num, annotated_frame, fps_text = frame_data
                
                # Add FPS text
                cv2.putText(annotated_frame, fps_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                # Display
                cv2.imshow('Integrated Recognition System', annotated_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # q or ESC
                    stop_event.set()
                    break
                    
            except Empty:
                continue
            except Exception as e:
                print(f"[ERROR] Display error: {e}")
                break
        
        cv2.destroyAllWindows()
    
    def process_frames(self, frame_q, display_q, stop_event, display_enabled,
                      conf_threshold, iou_threshold):
        """
        Process frames with face + behavior recognition.
        Runs in separate thread (main processing logic).
        """
        processed = 0
        frame_times = deque(maxlen=30)
        
        while not stop_event.is_set() or not frame_q.empty():
            try:
                # Get frame from queue
                frame_data = frame_q.get(timeout=0.5)
                if frame_data is None:
                    continue
                
                frame_num, timestamp, frame = frame_data
                t_start = time.time()
                
                # YOLO person detection with ByteTrack
                results = self.yolo.track(
                    frame,
                    persist=True,
                    conf=conf_threshold,
                    iou=iou_threshold,
                    classes=[0],  # person class
                    verbose=False,
                    # ByteTrack tuning (if supported by ultralytics version):
                    # tracker='bytetrack.yaml',
                    # track_thresh=self.bytetrack_track_thresh,
                    # track_buffer=self.bytetrack_track_buffer,
                    # match_thresh=self.bytetrack_match_thresh
                )
                
                annotated_frame = frame.copy()
                frame_identities = {}  # Track identities in current frame
                
                # Process detections
                if results and len(results) > 0 and results[0].boxes is not None:
                    boxes = results[0].boxes
                    
                    for i, box in enumerate(boxes):
                        # Get bounding box
                        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                        confidence = float(box.conf[0])
                        
                        # Get track ID (ByteTrack)
                        if box.id is not None:
                            track_id = int(box.id[0])
                        else:
                            continue
                        
                        # Extract person crop
                        person_crop = frame[y1:y2, x1:x2]
                        if person_crop.size == 0:
                            continue
                        
                        # Face recognition
                        face_result = self.recognize_face(person_crop, frame)
                        
                        # Update identity tracking with pose and body matching
                        self.update_track_identity(track_id, face_result, person_crop, frame_num)
                        
                        # Get tracked identity (with persistence)
                        name = self.get_track_identity(track_id)
                        
                        # Save known faces periodically
                        if name != "Unknown" and face_result is not None:
                            self.save_known_face(frame, person_crop, None, name, frame_num)
                        
                        # Get authorization level
                        auth_level = get_authorization_level(name)
                        frame_identities[track_id] = {'name': name, 'auth_level': auth_level}
                        
                        # Behavior classification
                        behavior_class = "Neutral"
                        behavior_conf = 0.0
                        
                        if should_monitor_behavior(auth_level):
                            if self.should_reclassify_behavior(track_id, frame_num):
                                result = self.classify_behavior(person_crop)
                                if result:
                                    # Store probability vector in history
                                    probs = result.get('probs')
                                    if probs is not None:
                                        self.prob_history[track_id].append(probs)
                                    
                                    # Apply temporal smoothing if we have history
                                    if len(self.prob_history[track_id]) > 0:
                                        # Average probability vectors across recent frames
                                        avg_probs = np.mean(list(self.prob_history[track_id]), axis=0)
                                        
                                        # Get final class from averaged probabilities
                                        smoothed_class_id = np.argmax(avg_probs)
                                        smoothed_conf = avg_probs[smoothed_class_id]
                                        
                                        # Apply confidence threshold to smoothed result
                                        if smoothed_conf >= self.min_class_conf:
                                            behavior_class = self.behavior_classes[smoothed_class_id]
                                            behavior_conf = smoothed_conf
                                        else:
                                            behavior_class = "Neutral"
                                            behavior_conf = smoothed_conf
                                    else:
                                        # Fallback to direct result
                                        behavior_class = result['class_name']
                                        behavior_conf = result['confidence']
                                    
                                    # Update cache with smoothed result
                                    self.behavior_cache[track_id] = {
                                        'class': behavior_class,
                                        'confidence': behavior_conf,
                                        'last_frame': frame_num
                                    }
                            else:
                                # Use cached classification
                                if track_id in self.behavior_cache:
                                    behavior_class = self.behavior_cache[track_id]['class']
                                    behavior_conf = self.behavior_cache[track_id]['confidence']
                        
                        # Check for alerts
                        alert_triggered = False
                        authorized_present = self.check_authorized_present(frame_identities)
                        
                        if should_trigger_alert(auth_level, behavior_class, authorized_present):
                            if self.can_send_alert(track_id):
                                self.send_alert(track_id, name, auth_level, behavior_class, frame_num)
                                alert_triggered = True
                                
                                if self.save_logs:
                                    self.save_alert_frames(frame, person_crop, track_id, name, 
                                                          auth_level, behavior_class, frame_num)
                        
                        # Log data
                        if self.save_logs:
                            self.log_frame_data(timestamp, frame_num, track_id, name, auth_level,
                                              behavior_class, behavior_conf, alert_triggered)
                        
                        # Draw annotations
                        display_name = format_display_name(name, auth_level)
                        color = get_level_color(auth_level)
                        thickness = get_level_thickness(auth_level)
                        
                        # Bounding box
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
                        
                        # Identity label
                        label = f"{display_name} ({confidence:.2f})"
                        label_y = y1 - 10 if y1 > 30 else y1 + 20
                        cv2.putText(annotated_frame, label, (x1, label_y),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        
                        # Behavior label (if monitored)
                        if should_monitor_behavior(auth_level) and behavior_class != "Neutral":
                            behavior_label = f"{behavior_class} ({behavior_conf:.2f})"
                            behavior_y = y2 + 20
                            
                            # Color code suspicious behavior
                            behavior_color = (0, 0, 255) if is_suspicious_behavior(behavior_class) else (255, 255, 0)
                            cv2.putText(annotated_frame, behavior_label, (x1, behavior_y),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, behavior_color, 2)
                
                # Calculate FPS
                t_end = time.time()
                frame_times.append(t_end - t_start)
                avg_time = sum(frame_times) / len(frame_times)
                fps = 1.0 / avg_time if avg_time > 0 else 0
                fps_text = f"FPS: {fps:.1f}"
                
                processed += 1
                
                # Put result in display queue (if display enabled)
                if display_enabled:
                    try:
                        display_q.put((frame_num, annotated_frame, fps_text), timeout=0.1)
                    except:
                        pass  # Display queue full, skip
                
            except Empty:
                continue
            except Exception as e:
                print(f"[ERROR] Frame processing error: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        print(f"\n[INFO] Processed {processed} frames")
    
    def process_video(self, video_path, display=True, save_output=None,
                     conf_threshold=0.5, iou_threshold=0.7):
        """
        Process video with integrated recognition using multi-threading
        
        Args:
            video_path: Path to video or 'webcam'
            display: Show output window
            save_output: Path to save output video (Note: Not implemented in threaded version)
            conf_threshold: YOLO confidence threshold
            iou_threshold: YOLO IOU threshold
        """
        print("\n[INFO] Starting threaded processing pipeline...")
        
        # Open video
        if video_path == 'webcam':
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer for live camera
        else:
            cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")
        
        # Get video info
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"[INFO] Video: {width}x{height} @ {fps:.1f}fps, {total_frames} frames")
        
        # Create queues and stop event
        frame_q = Queue(maxsize=self.capture_queue_size)
        display_q = Queue(maxsize=self.display_queue_size)
        stop_event = Event()
        
        # Start threads
        capture_thread = Thread(target=self.grab_frames, args=(cap, frame_q, stop_event))
        process_thread = Thread(target=self.process_frames, 
                               args=(frame_q, display_q, stop_event, display, 
                                     conf_threshold, iou_threshold))
        display_thread = Thread(target=self.display_frames, args=(display_q, stop_event))
        
        capture_thread.daemon = True
        process_thread.daemon = True
        display_thread.daemon = True
        
        try:
            print("[INFO] Starting capture thread...")
            capture_thread.start()
            
            print("[INFO] Starting processing thread...")
            process_thread.start()
            
            if display:
                print("[INFO] Starting display thread...")
                display_thread.start()
            
            # Wait for processing to complete
            capture_thread.join()
            process_thread.join()
            
            if display:
                display_thread.join()
            
            print("[INFO] All threads completed")
            
        except KeyboardInterrupt:
            print("\n[INFO] Interrupted by user")
        except Exception as e:
            print(f"[ERROR] Processing error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            stop_event.set()
            cap.release()
            
            # Wait for threads to finish
            if capture_thread.is_alive():
                capture_thread.join(timeout=2.0)
            if process_thread.is_alive():
                process_thread.join(timeout=2.0)
            if display and display_thread.is_alive():
                display_thread.join(timeout=2.0)
        
        print("\n" + "="*80)
        print("PROCESSING COMPLETE")
        print("="*80)
        print(f"Alerts sent: {len(self.alerts_sent)}")
        if self.save_logs:
            print(f"CSV log: {self.csv_path}")
        print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Integrated Face Recognition + Behavior Recognition System'
    )
    
    # Input
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--video', type=str, help='Path to input video')
    input_group.add_argument('--webcam', action='store_true', help='Use webcam input')
    
    # Models
    parser.add_argument('--yolo-model', type=str,
                       default='models/YOLOv8/yolov8m.pt',
                       help='Path to YOLOv8 model')
    parser.add_argument('--facenet-model', type=str,
                       default='models/FaceNet/inception_resnet_v1.pt',
                       help='Path to FaceNet model')
    parser.add_argument('--behavior-model', type=str,
                       default='models/mobilenetv3-small/mobilenetv3_small_feature_extraction.pth',
                       help='Path to MobileNetV3-Small behavior model')
    
    # Configuration
    parser.add_argument('--conf', type=float, default=0.5,
                       help='YOLO confidence threshold (default: 0.5)')
    parser.add_argument('--iou', type=float, default=0.7,
                       help='YOLO IOU threshold (default: 0.7)')
    
    # Output
    parser.add_argument('--no-display', action='store_true',
                       help='Do not display output window')
    parser.add_argument('--save', type=str,
                       help='Save output video to specified path')
    parser.add_argument('--save-logs', action='store_true',
                       help='Save logs and alert frames')
    
    # Device
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'],
                       help='Device to use (default: auto-detect)')
    parser.add_argument('--min-class-conf', type=float, default=None,
                       help='Minimum behavior classification confidence (0-1) to accept behavior; below this will be treated as Neutral')
    parser.add_argument('--smooth-window', type=int, default=7,
                       help='Temporal smoothing window size in frames (default: 7)')
    
    args = parser.parse_args()
    
    # Print authorization configuration
    print_authorization_summary()
    
    # Initialize system
    system = IntegratedRecognitionSystem(
        yolo_model_path=args.yolo_model,
        facenet_model_path=args.facenet_model,
        behavior_model_path=args.behavior_model,
        device=args.device,
        save_logs=args.save_logs,
        min_class_conf=args.min_class_conf,
        smoothing_window=args.smooth_window
    )
    
    # Process video
    video_source = 'webcam' if args.webcam else args.video
    
    system.process_video(
        video_path=video_source,
        display=not args.no_display,
        save_output=args.save,
        conf_threshold=args.conf,
        iou_threshold=args.iou
    )


if __name__ == '__main__':
    main()
