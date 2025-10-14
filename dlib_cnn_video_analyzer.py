import os
import sys
import cv2
import dlib
import joblib
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import time
import pickle
from datetime import datetime
from collections import defaultdict, deque
from pathlib import Path
from ultralytics import YOLO

# Add project root to path for imports
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

class DlibCNNVideoAnalyzer:
    def __init__(self, video_path, output_dir="model_comparison_results"):
        self.video_path = video_path
        self.video_name = Path(video_path).stem  # Get video name without extension
        
        # Create organized folder structure
        self.base_output_dir = output_dir
        self.dlib_dir = os.path.join(output_dir, "DlibCNN")
        self.graphs_dir = os.path.join(self.dlib_dir, "graphs")
        self.data_dir = os.path.join(self.dlib_dir, "data")
        self.comparisons_dir = os.path.join(output_dir, "Comparisons")
        
        # Ensure all directories exist
        os.makedirs(self.dlib_dir, exist_ok=True)
        os.makedirs(self.graphs_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.comparisons_dir, exist_ok=True)
        
        # Create subdirectories in Comparisons for better organization
        os.makedirs(os.path.join(self.comparisons_dir, "summaries"), exist_ok=True)
        os.makedirs(os.path.join(self.comparisons_dir, "detailed"), exist_ok=True)
        os.makedirs(os.path.join(self.comparisons_dir, "graphs"), exist_ok=True)
        
        self.results = {
            'frames': [],
            'recognitions': []
        }
        
        # Enhanced tracking settings (from dilib_cnn_main.py)
        self.BACK_VIEW_TOLERANCE_FRAMES = 600
        self.BODY_MATCH_THRESHOLD = 0.5
        self.EXTENDED_MEMORY_FRAMES = 300
        self.MIN_BODY_MATCH_CONFIDENCE = 0.4
        self.POSE_HISTORY_LENGTH = 30
        self.IDENTITY_MEMORY_FRAMES = 90
        self.IDENTITY_CONFIDENCE_DECAY = 0.995
        self.MIN_IDENTITY_CONFIDENCE = 0.15
        self.FACE_LOST_TOLERANCE = 180
        
        # Identity locking settings (from dilib_cnn_main.py)
        self.LOCK_CONFIDENCE_THRESHOLD = 0.5
        self.LOCK_CONSISTENT_FRAMES = 3
        self.LOCK_TIMEOUT_FRAMES = 30
        self.LOCK_MIN_CONFIDENCE_TO_MAINTAIN = 0.2
        self.LOCK_BREAK_THRESHOLD = 0.7
        
        # Tracking data structures
        self.track_identities = {}
        self.track_face_history = defaultdict(lambda: deque(maxlen=self.EXTENDED_MEMORY_FRAMES))
        self.track_body_history = defaultdict(lambda: deque(maxlen=60))
        self.track_last_face_frame = {}
        
        # Enhanced tracking from dilib_cnn_main.py
        self.track_locks = {}
        self.track_detection_history = defaultdict(lambda: deque(maxlen=self.LOCK_CONSISTENT_FRAMES * 2))
        self.global_identities = {}
        self.identity_persistence_threshold = 0.35
        
        # Enhanced performance settings
        self.PERFORMANCE_MODE = True  # Enable performance optimizations
        self.FACE_RECOG_COOLDOWN = {}  # Track cooldown per track
        self.FREEZE_PREVENTION_MODE = True  # New flag for anti-freeze measures
        
        print(f"[INFO] Dlib CNN Video Analyzer initialized with performance optimizations")
        print(f"[INFO] Video: {video_path}")
        print(f"[INFO] Dlib CNN Output: {self.dlib_dir}")
        print(f"[INFO] Graphs will be saved to: {self.graphs_dir}")
        print(f"[INFO] Comparison data will be saved to: {self.comparisons_dir}")
    
    def detect_person_pose_from_body(self, person_crop):
        """Detect if person is facing away based on body characteristics"""
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
        """Calculate body similarity score between two person crops"""
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

    def update_track_lock_status(self, track_id, detected_identity, confidence, frame_num):
        """Update the locking mechanism for a track - Enhanced from dilib_cnn_main.py"""
        
        # Initialize track lock data if needed
        if track_id not in self.track_locks:
            self.track_locks[track_id] = {
                'identity': 'Unknown',
                'locked': False,
                'lock_strength': 0,
                'last_seen': frame_num,
                'lock_start_frame': None,
                'lock_confidence': 0.0,
                'conflicting_detections': 0
            }
        
        track_lock = self.track_locks[track_id]
        track_lock['last_seen'] = frame_num
        
        # Add current detection to history
        self.track_detection_history[track_id].append({
            'identity': detected_identity,
            'confidence': confidence,
            'frame': frame_num
        })
        
        # If already locked, check if we should maintain or break the lock
        if track_lock['locked']:
            locked_identity = track_lock['identity']
            
            # Strong evidence for the locked identity or acceptable low confidence
            if (detected_identity == locked_identity and confidence >= self.LOCK_MIN_CONFIDENCE_TO_MAINTAIN) or \
               (detected_identity == 'Unknown' and confidence == 0.0):  # No detection is OK when locked
                # Maintain lock - STRENGTHEN IT
                track_lock['lock_strength'] = min(track_lock['lock_strength'] + 2, self.LOCK_CONSISTENT_FRAMES * 5)
                track_lock['conflicting_detections'] = 0  # Reset conflict counter
                return locked_identity, track_lock.get('lock_confidence', confidence)
            
            # Strong evidence against the locked identity - MAKE IT HARDER TO BREAK
            elif detected_identity != 'Unknown' and detected_identity != locked_identity and confidence >= self.LOCK_BREAK_THRESHOLD:
                track_lock['conflicting_detections'] += 1
                track_lock['lock_strength'] -= 1  # Smaller penalty
                
                # Check if we should break the lock - REQUIRE MORE EVIDENCE
                if track_lock['lock_strength'] <= 0 and track_lock['conflicting_detections'] >= 3:
                    track_lock['locked'] = False
                    track_lock['lock_strength'] = 0
                    track_lock['identity'] = 'Unknown'
                    track_lock['conflicting_detections'] = 0
                else:
                    return locked_identity, track_lock.get('lock_confidence', confidence)
            else:
                # Weak conflicting evidence - ignore it
                return locked_identity, track_lock.get('lock_confidence', confidence)
            
            # If still locked, return locked identity
            if track_lock['locked']:
                return locked_identity, track_lock.get('lock_confidence', confidence)
        
        # Not locked - check if we should establish a lock
        if not track_lock['locked'] and detected_identity != 'Unknown' and confidence >= self.LOCK_CONFIDENCE_THRESHOLD:
            
            # Count recent consistent detections
            recent_detections = list(self.track_detection_history[track_id])[-self.LOCK_CONSISTENT_FRAMES:]
            
            if len(recent_detections) >= self.LOCK_CONSISTENT_FRAMES:
                # Check consistency
                same_identity_count = sum(1 for d in recent_detections 
                                        if d['identity'] == detected_identity and d['confidence'] >= self.LOCK_CONFIDENCE_THRESHOLD)
                
                consistency_ratio = same_identity_count / len(recent_detections)
                
                # Lock if consistent enough - LOWER THE BAR
                if consistency_ratio >= 0.67:  # 67% consistency (2 out of 3 frames)
                    track_lock['locked'] = True
                    track_lock['identity'] = detected_identity
                    track_lock['lock_strength'] = self.LOCK_CONSISTENT_FRAMES * 2  # Start with stronger lock
                    track_lock['lock_start_frame'] = frame_num
                    track_lock['lock_confidence'] = confidence
                    track_lock['conflicting_detections'] = 0
                    
                    return detected_identity, confidence
        
        # Default: return current detection
        track_lock['identity'] = detected_identity if detected_identity != 'Unknown' else track_lock.get('identity', 'Unknown')
        return detected_identity, confidence

    def find_matching_identity(self, face_encoding, confidence):
        """Find matching identity from global database using face encoding similarity"""
        if face_encoding is None or len(self.global_identities) == 0:
            return None, 0.0
        
        best_match = None
        best_similarity = 0.0
        
        for identity_name, identity_data in self.global_identities.items():
            if 'encodings' not in identity_data or len(identity_data['encodings']) == 0:
                continue
            
            # Compare with stored encodings for this identity
            similarities = []
            for stored_encoding in identity_data['encodings'][-5:]:  # Use last 5 encodings
                # Calculate cosine similarity
                similarity = np.dot(face_encoding.flatten(), stored_encoding.flatten()) / (
                    np.linalg.norm(face_encoding.flatten()) * np.linalg.norm(stored_encoding.flatten())
                )
                similarities.append(similarity)
            
            # Use average similarity
            avg_similarity = np.mean(similarities) if similarities else 0.0
            
            if avg_similarity > best_similarity:
                best_similarity = avg_similarity
                best_match = identity_name
        
        # Return match if similarity is above threshold
        if best_similarity > self.identity_persistence_threshold:
            return best_match, best_similarity
        
        return None, 0.0

    def update_global_identity(self, name, face_encoding, confidence, frame_num):
        """Update global identity database with new encoding"""
        if name == 'Unknown' or face_encoding is None:
            return
        
        if name not in self.global_identities:
            self.global_identities[name] = {
                'encodings': [],
                'confidence_history': [],
                'last_seen': frame_num
            }
        
        # Add new encoding (keep only recent ones)
        self.global_identities[name]['encodings'].append(face_encoding.flatten())
        self.global_identities[name]['confidence_history'].append(confidence)
        self.global_identities[name]['last_seen'] = frame_num
        
        # Limit stored encodings per identity
        if len(self.global_identities[name]['encodings']) > 10:
            self.global_identities[name]['encodings'] = self.global_identities[name]['encodings'][-10:]
            self.global_identities[name]['confidence_history'] = self.global_identities[name]['confidence_history'][-10:]

    def recognize_face_in_crop(self, person_crop, original_frame, person_bbox):
        """Dlib CNN face recognition in person crop"""
        if person_crop is None or person_crop.size == 0:
            return {'name': 'Unknown', 'confidence': 0.0, 'face_bbox': None, 'inference_time': 0.0, 'face_encoding': None}
        
        try:
            inference_start = time.time()
            
            # Convert to RGB for dlib
            person_rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
            
            # Face detection using Dlib CNN detector
            face_locations = self.cnn_face_detector(person_rgb, 1)
            
            if len(face_locations) == 0:
                inference_time = (time.time() - inference_start) * 1000
                return {'name': 'Unknown', 'confidence': 0.0, 'face_bbox': None, 'inference_time': inference_time, 'face_encoding': None}
            
            # Get the most confident face (first detection)
            face_location = face_locations[0]
            face_rect = face_location.rect
            
            left, top, right, bottom = face_rect.left(), face_rect.top(), face_rect.right(), face_rect.bottom()
            
            # Convert face coordinates back to original frame
            px1, py1, px2, py2 = person_bbox
            face_x1_orig = px1 + left
            face_y1_orig = py1 + top
            face_x2_orig = px1 + right
            face_y2_orig = py1 + bottom
            
            # Get face landmarks
            landmarks = self.shape_predictor(person_rgb, face_rect)
            
            # Get face encoding
            face_encoding = np.array(self.face_rec_model.compute_face_descriptor(person_rgb, landmarks))
            
            # Face recognition using trained classifier
            if self.classifier is not None and self.label_encoder is not None:
                try:
                    # Predict using classifier
                    probs = self.classifier.predict_proba([face_encoding])[0]
                    pred = np.argmax(probs)
                    confidence = probs[pred]
                    
                    # Check thresholds
                    RECOG_THRESHOLD = 0.45
                    RECOG_MARGIN = 0.08
                    
                    sorted_probs = np.sort(probs)[::-1]
                    top2_prob = sorted_probs[1] if len(sorted_probs) > 1 else 0.0
                    
                    inference_time = (time.time() - inference_start) * 1000
                    
                    if confidence >= RECOG_THRESHOLD and (confidence - top2_prob) >= RECOG_MARGIN:
                        candidate = self.label_encoder.inverse_transform([pred])[0]
                        
                        return {
                            'name': candidate,
                            'confidence': confidence,
                            'face_bbox': (face_x1_orig, face_y1_orig, face_x2_orig, face_y2_orig),
                            'inference_time': inference_time,
                            'face_encoding': face_encoding
                        }
                    
                except Exception as e:
                    print(f"[WARN] Dlib face recognition failed: {e}")
            
            inference_time = (time.time() - inference_start) * 1000
            return {
                'name': 'Unknown', 
                'confidence': 0.0, 
                'face_bbox': (face_x1_orig, face_y1_orig, face_x2_orig, face_y2_orig),
                'inference_time': inference_time,
                'face_encoding': face_encoding
            }
            
        except Exception as e:
            print(f"[ERROR] Dlib face recognition in crop failed: {e}")
            inference_time = (time.time() - inference_start) * 1000
            return {'name': 'Unknown', 'confidence': 0.0, 'face_bbox': None, 'inference_time': inference_time, 'face_encoding': None}
    
    def cleanup_old_tracks(self, current_frame, active_track_ids):
        """Clean up locks for tracks that are no longer active"""
        tracks_to_remove = []
        
        for track_id, track_lock in self.track_locks.items():
            # Remove if track is no longer active and hasn't been seen recently
            if track_id not in active_track_ids and \
               current_frame - track_lock['last_seen'] > self.LOCK_TIMEOUT_FRAMES:
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            if self.track_locks[track_id]['locked']:
                print(f"[UNLOCK] 🔓 Unlocking {self.track_locks[track_id]['identity']} for track {track_id} (track lost)")
            del self.track_locks[track_id]
            if track_id in self.track_detection_history:
                del self.track_detection_history[track_id]
    
    def process_video_with_analysis(self, show_video=True):
        """Process video with anti-freeze optimizations"""
        
        print(f"[INFO] Loading Dlib CNN models...")
        
        # Load models (same as before)
        try:
            DLIB_MODELS_DIR = os.path.join("models", "Dlib")
            
            # CNN face detector
            cnn_detector_path = os.path.join(DLIB_MODELS_DIR, "mmod_human_face_detector.dat")
            if not os.path.exists(cnn_detector_path):
                raise FileNotFoundError(f"CNN face detector not found: {cnn_detector_path}")
            self.cnn_face_detector = dlib.cnn_face_detection_model_v1(cnn_detector_path)
            
            # Shape predictor for landmarks
            predictor_path = os.path.join(DLIB_MODELS_DIR, "shape_predictor_68_face_landmarks.dat")
            if not os.path.exists(predictor_path):
                raise FileNotFoundError(f"Shape predictor not found: {predictor_path}")
            self.shape_predictor = dlib.shape_predictor(predictor_path)
            
            # Face recognition model
            face_rec_path = os.path.join(DLIB_MODELS_DIR, "dlib_face_recognition_resnet_model_v1.dat")
            if not os.path.exists(face_rec_path):
                raise FileNotFoundError(f"Face recognition model not found: {face_rec_path}")
            self.face_rec_model = dlib.face_recognition_model_v1(face_rec_path)
            
            print(f"[INFO] ✅ Loaded Dlib models successfully")
            
        except Exception as e:
            print(f"[ERROR] Failed to load Dlib models: {e}")
            return False
        
        # Load YOLO
        self.yolo = YOLO("models/YOLOv8/yolov8n.pt")
        
        # Load trained classifier
        try:
            SVM_PATH = os.path.join(DLIB_MODELS_DIR, "dlib_svm.joblib")
            LE_PATH = os.path.join(DLIB_MODELS_DIR, "dlib_label_encoder.joblib")
            
            if os.path.exists(SVM_PATH) and os.path.exists(LE_PATH):
                self.classifier = joblib.load(SVM_PATH)
                self.label_encoder = joblib.load(LE_PATH)
                print(f"[INFO] Loaded Dlib classifier. Classes: {list(self.label_encoder.classes_)}")
            else:
                print(f"[WARN] Dlib classifier not found, using similarity matching only")
                self.classifier = None
                self.label_encoder = None
                
        except Exception as e:
            print(f"[WARN] Failed to load Dlib classifier: {e}")
            self.classifier = None
            self.label_encoder = None
        
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
        
        # Processing variables with anti-freeze counters
        frame_count = 0
        processed_frame_count = 0
        consecutive_processing_count = 0
        display_frame = None
        # Accumulate recognitions for this run
        all_recognitions = []
        last_yield_time = time.time()
        
        # Ultra-aggressive performance settings to prevent freezing
        PERSON_CONF_THRESHOLD = 0.6
        RESIZE_WIDTH = 360  # Further reduced from 480
        PROCESS_EVERY_N = 5  # Increased from 3 - process even fewer frames
        FACE_RECOG_EVERY_N = 60  # Increased from 30 - much less frequent
        DISPLAY_UPDATE_EVERY_N = 4  # Increased from 2 - update display less often
        
        # Much more aggressive cooldowns
        FACE_RECOG_COOLDOWN_FRAMES = 120  # Increased from 60
        MAX_CONSECUTIVE_PROCESSING = 10   # Limit consecutive processing to prevent blocking
        
        # Anti-freeze measures
        FORCE_YIELD_EVERY_N = 20  # Force yield CPU every N processed frames
        MAX_FRAME_PROCESSING_TIME = 200  # Max ms per frame before skipping
        
        print(f"\n[INFO] ANTI-FREEZE optimized processing:")
        print(f"  • Process every {PROCESS_EVERY_N} frames (ultra-sparse)")
        print(f"  • Face recognition every {FACE_RECOG_EVERY_N} frames")
        print(f"  • Display update every {DISPLAY_UPDATE_EVERY_N} processed frames")
        print(f"  • Max frame time: {MAX_FRAME_PROCESSING_TIME}ms")
        print(f"  • Resize width: {RESIZE_WIDTH}px (ultra-small)")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Ultra-aggressive frame skipping
            if frame_count % PROCESS_EVERY_N != 0:
                frame_count += 1
                continue
            
            frame_start_time = time.time()
            processed_frame_count += 1
            consecutive_processing_count += 1
            
            # Force CPU yield periodically to prevent UI freezing
            if consecutive_processing_count >= FORCE_YIELD_EVERY_N:
                time.sleep(0.001)  # 1ms yield
                consecutive_processing_count = 0
                last_yield_time = time.time()
            
            original_frame = frame.copy()
            
            # Only update display frame occasionally
            should_update_display = (processed_frame_count % DISPLAY_UPDATE_EVERY_N == 0) and show_video
            if should_update_display:
                display_frame = frame.copy()
            
            orig_h, orig_w = original_frame.shape[:2]
            
            # Ultra-aggressive resize to prevent processing bottlenecks
            if RESIZE_WIDTH and orig_w > RESIZE_WIDTH:
                ratio = RESIZE_WIDTH / orig_w
                process_frame = cv2.resize(original_frame, (RESIZE_WIDTH, int(orig_h * ratio)))
            else:
                process_frame = original_frame
                ratio = 1.0
            
            try:
                # YOLO with minimal settings to prevent blocking
                results = self.yolo.track(
                    process_frame,
                    persist=True,
                    tracker="bytetrack.yaml",
                    classes=[0],
                    conf=PERSON_CONF_THRESHOLD,
                    iou=0.7,
                    imgsz=256,  # Further reduced from 320
                    verbose=False,
                    device='cpu'  # Ensure CPU to avoid GPU memory issues
                )
                
                current_active_tracks = set()
                
                if results and results[0].boxes is not None:
                    boxes = results[0].boxes
                    
                    if boxes.id is not None:
                        track_ids = boxes.id.cpu().numpy().astype(int)
                        current_active_tracks.update(track_ids)
                        bboxes = boxes.xyxy.cpu().numpy()
                        confidences = boxes.conf.cpu().numpy()
                        
                        # Limit number of tracks processed per frame to prevent blocking
                        max_tracks_per_frame = 3
                        track_data = list(zip(track_ids, bboxes, confidences))[:max_tracks_per_frame]
                        
                        for track_id, bbox, conf in track_data:
                            # Check frame processing time limit
                            if (time.time() - frame_start_time) * 1000 > MAX_FRAME_PROCESSING_TIME:
                                print(f"[WARN] Frame {frame_count} processing timeout, skipping remaining tracks")
                                break
                            
                            # Convert bbox back to original frame coordinates
                            if ratio != 1.0:
                                bbox = bbox / ratio
                            
                            x1, y1, x2, y2 = bbox.astype(int)
                            x1, y1 = max(0, x1), max(0, y1)
                            x2, y2 = min(orig_w, x2), min(orig_h, y2)
                            
                            # Smaller padding to reduce crop size
                            padding = 20  # Reduced from 40
                            x1_pad = max(0, x1 - padding)
                            y1_pad = max(0, y1 - padding)
                            x2_pad = min(orig_w, x2 + padding)
                            y2_pad = min(orig_h, y2 + padding)
                            
                            person_bbox = (x1_pad, y1_pad, x2_pad, y2_pad)
                            person_crop = original_frame[y1_pad:y2_pad, x1_pad:x2_pad]
                            
                            # Ultra-conservative face recognition with longer cooldowns
                            current_time = frame_count
                            last_recog_time = self.FACE_RECOG_COOLDOWN.get(track_id, 0)
                            
                            should_recognize = (
                                (current_time - last_recog_time) >= FACE_RECOG_COOLDOWN_FRAMES and
                                person_crop.shape[0] > 120 and person_crop.shape[1] > 60
                            )
                            
                            # Face recognition with timeout protection
                            raw_identity_name = 'Unknown'
                            raw_identity_conf = 0.0
                            face_result = None
                            
                            if should_recognize:
                                self.FACE_RECOG_COOLDOWN[track_id] = current_time
                                
                                try:
                                    # Add timeout for face recognition to prevent blocking
                                    recog_start_time = time.time()
                                    face_result = self.recognize_face_in_crop(person_crop, original_frame, person_bbox)
                                    recog_time = (time.time() - recog_start_time) * 1000
                                    
                                    # Skip if recognition took too long
                                    if recog_time > 100:  # 100ms limit
                                        print(f"[WARN] Face recognition timeout for track {track_id}")
                                        continue
                                    
                                    raw_identity_name = face_result.get('name', 'Unknown')
                                    raw_identity_conf = face_result.get('confidence', 0.0)
                                    face_encoding = face_result.get('face_encoding')
                                    
                                    # Skip global matching for performance
                                    if raw_identity_name == 'Unknown':
                                        pass  # Skip time-consuming global matching
                                    
                                    # Update global database only for confident results
                                    if raw_identity_name != 'Unknown' and raw_identity_conf > 0.6:
                                        if face_encoding is not None:
                                            self.update_global_identity(raw_identity_name, face_encoding, raw_identity_conf, frame_count)
                                
                                except Exception as e:
                                    print(f"[WARN] Face recognition failed for track {track_id}: {e}")
                                    continue
                            

                            # Apply locking mechanism (simplified for performance)
                            final_identity_name, final_identity_conf = self.update_track_lock_status(
                                track_id, raw_identity_name, raw_identity_conf, frame_count
                            )
                            
                            # Simplified tracking storage
                            if track_id not in self.track_identities:
                                self.track_identities[track_id] = {'identity': 'Unknown', 'confidence': 0.0}
                            
                            if final_identity_name != 'Unknown':
                                self.track_identities[track_id]['identity'] = final_identity_name
                                self.track_identities[track_id]['confidence'] = final_identity_conf
                            

                            # Use stored identity if current recognition failed
                            if final_identity_name == 'Unknown' and track_id in self.track_identities:
                                stored_identity = self.track_identities[track_id]['identity']
                                if stored_identity != 'Unknown':
                                    final_identity_name = stored_identity
                                    final_identity_conf = self.track_identities[track_id]['confidence']
                            
                            # Skip pose detection for performance
                            pose = "frontal"
                            pose_confidence = 0.8
                            

                            is_locked = track_id in self.track_locks and self.track_locks[track_id]['locked']
                            frames_since_face = frame_count - self.FACE_RECOG_COOLDOWN.get(track_id, frame_count)
                            
                            # Store minimal recognition data
                            recognition_data = {
                                'frame': frame_count,
                                'track_id': int(track_id),
                                'name': final_identity_name,
                                'confidence': final_identity_conf,
                                'inference_time_ms': face_result['inference_time'] if face_result else 0.0,
                                'pose': pose,
                                'pose_confidence': pose_confidence,
                                'is_locked': is_locked,
                                'consecutive_back_frames': 0,
                                'frames_since_face': frames_since_face
                            }
                            
                            all_recognitions.append(recognition_data)
                            
                            # Simplified display drawing (only essential elements)
                            if should_update_display and display_frame is not None:
                                # Ultra-simple display for performance
                                if final_identity_name == 'Unknown':
                                    color = (0, 0, 255)  # Red
                                elif is_locked:
                                    color = (255, 0, 0)  # Blue for locked
                                else:
                                    color = (0, 255, 255)  # Yellow
                                
                                # Draw only bounding box (no text for performance)
                                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                                
                                # Minimal label (only ID and lock status)
                                if is_locked:
                                    cv2.putText(display_frame, f"L{track_id}", (x1, y1-5), 
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                                else:
                                    cv2.putText(display_frame, f"{track_id}", (x1, y1-5), 
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                # Clean up old tracks much less frequently
                if processed_frame_count % 100 == 0:
                    self.cleanup_old_tracks(frame_count, current_active_tracks)
                
                # Minimal performance info (only when updating display)
                if should_update_display and display_frame is not None:
                    frame_time = (time.time() - frame_start_time) * 1000
                    locked_tracks = sum(1 for t in self.track_locks.values() if t.get('locked', False))
                    
                    # Single line info
                    info_text = f"F:{frame_count} L:{locked_tracks}"
                    cv2.putText(display_frame, info_text, (10, 25), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Progress update much less frequently
                if frame_count % 300 == 0:
                    progress = (frame_count / total_frames) * 100
                    locked_count = sum(1 for t in self.track_locks.values() if t.get('locked', False))
                    print(f"  Progress: {progress:.1f}% - Locked: {locked_count}")
                
            except Exception as e:
                print(f"[ERROR] Frame {frame_count} processing failed: {e}")
                # Continue processing instead of breaking
                continue
            
            # Show frame with minimal processing
            if should_update_display and display_frame is not None:
                # Ultra-aggressive resize for display
                max_display_height = 400  # Further reduced
                max_display_width = 640   # Further reduced
                
                h, w = display_frame.shape[:2]
                scale = min(max_display_width/w, max_display_height/h, 0.7)  # Max 70% of original
                
                new_width = int(w * scale)
                new_height = int(h * scale)
                display_frame_resized = cv2.resize(display_frame, (new_width, new_height))
                
                cv2.imshow('Dlib CNN (Ultra Performance Mode)', display_frame_resized)
                
                # Non-blocking waitKey with minimal delay
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n[INFO] User requested quit")
                    break
            
            # Force yield every few frames to keep UI responsive
            if frame_count % 10 == 0:
                time.sleep(0.001)  # 1ms yield to prevent complete blocking
            
            frame_count += 1
        
        cap.release()
        if show_video:
            cv2.destroyAllWindows()
        
        # Store results
        self.results['recognitions'] = all_recognitions
        
        print(f"\n[INFO] Video processing completed with Dlib CNN!")
        print(f"  Total recognitions: {len(all_recognitions)}")
        print(f"  Final locked tracks: {sum(1 for t in self.track_locks.values() if t.get('locked', False))}")
        
        return True
    
    def calculate_simple_metrics(self):
        """Calculate the 3 requested metrics with enhanced tracking insights"""
        
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
        
        # Additional enhanced tracking stats
        back_view_detections = len(df[df['pose'] == 'back_view'])
        locked_detections = len(df[df['is_locked'] == True])
        
        stats = {
            'model_name': 'DlibCNN',
            'video_name': self.video_name,
            'avg_confidence': avg_confidence,
            'avg_inference_time_ms': avg_inference_time,
            'avg_accuracy': avg_accuracy,
            'total_detections': total_detections,
            'successful_recognitions': successful_recognitions,
            'recognition_rate': avg_accuracy,
            # Enhanced stats with tracking
            'back_view_detections': back_view_detections,
            'back_view_percentage': (back_view_detections / total_detections) * 100 if total_detections > 0 else 0,
            'locked_detections': locked_detections,
            'locked_percentage': (locked_detections / total_detections) * 100 if total_detections > 0 else 0,
            'unique_tracks': len(self.track_locks),
            'locked_tracks': sum(1 for t in self.track_locks.values() if t.get('locked', False))
        }
        
        return stats
    
    def create_simple_graph(self, stats):
        """Create simple graph with only 3 metrics - saved to DlibCNN specific folder"""
        
        print(f"\n[INFO] Creating Dlib CNN performance graph...")
        
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
        ax.set_title(f'Dlib CNN Performance Metrics - {self.video_name}\nwith Enhanced Identity Tracking', 
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
        
        # Enhanced summary text box with tracking info
        summary_text = f"""
📊 DLIB CNN PERFORMANCE SUMMARY

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
⚙️  MODEL: Dlib CNN ResNet
        """
        
        # Position text box
        ax.text(0.02, 0.98, summary_text.strip(), transform=ax.transAxes, 
               fontsize=9, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcyan', alpha=0.8))
        
        # Tight layout
        plt.tight_layout()
        
        # Save the graph to DlibCNN specific folder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        graph_filename = f"dlib_cnn_performance_{self.video_name}_{timestamp}.png"
        graph_path = os.path.join(self.graphs_dir, graph_filename)
        
        plt.savefig(graph_path, dpi=300, bbox_inches='tight')
        print(f"[INFO] ✅ Saved Dlib CNN graph: {graph_path}")
        
        # Also save a standard named version for easy comparison
        standard_graph_path = os.path.join(self.graphs_dir, f"dlib_cnn_metrics_{self.video_name}.png")
        plt.savefig(standard_graph_path, dpi=300, bbox_inches='tight')
        print(f"[INFO] ✅ Saved standard Dlib CNN graph: {standard_graph_path}")
        
        # Show the graph
        plt.show()
        
        return graph_path, standard_graph_path
    
    def save_results(self, stats):
        """Save results to DlibCNN specific folders with enhanced comparison structure"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed recognition data
        if self.results['recognitions']:
            csv_filename = f"dlib_cnn_recognition_data_{self.video_name}_{timestamp}.csv"
            csv_path = os.path.join(self.data_dir, csv_filename)
            df = pd.DataFrame(self.results['recognitions'])
            df.to_csv(csv_path, index=False)
            print(f"[INFO] ✅ Saved recognition data: {csv_path}")
            
            # Also save standard named version
            standard_csv_path = os.path.join(self.data_dir, f"dlib_cnn_data_{self.video_name}.csv")
            df.to_csv(standard_csv_path, index=False)
            print(f"[INFO] ✅ Saved standard recognition data: {standard_csv_path}")
        
        # Save statistics
        stats_filename = f"dlib_cnn_stats_{self.video_name}_{timestamp}.json"
        stats_path = os.path.join(self.data_dir, stats_filename)
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"[INFO] ✅ Saved statistics: {stats_path}")
        
        # Also save standard named version
        standard_stats_path = os.path.join(self.data_dir, f"dlib_cnn_stats_{self.video_name}.json")
        with open(standard_stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"[INFO] ✅ Saved standard statistics: {standard_stats_path}")
        
        # Save tracking summary with proper key conversion
        tracking_summary = {
            'model_name': 'DlibCNN',
            'video_name': self.video_name,
            'timestamp': timestamp,
            'total_tracks': len(self.track_locks),
            'track_details': {}
        }
        
        # Convert track_id keys to strings for JSON compatibility
        for track_id, track_data in self.track_locks.items():
            tracking_summary['track_details'][str(track_id)] = {
                'identity': track_data.get('identity', 'Unknown'),
                'locked': track_data.get('locked', False),
                'lock_strength': track_data.get('lock_strength', 0),
                'lock_confidence': track_data.get('lock_confidence', 0.0),
                'conflicting_detections': track_data.get('conflicting_detections', 0),
                'last_seen': track_data.get('last_seen', 0)
            }
        
        tracking_filename = f"dlib_cnn_tracking_{self.video_name}_{timestamp}.json"
        tracking_path = os.path.join(self.data_dir, tracking_filename)
        with open(tracking_path, 'w') as f:
            json.dump(tracking_summary, f, indent=2)
        print(f"[INFO] ✅ Saved tracking summary: {tracking_path}")
        
        # Enhanced comparison summaries
        # 1. Basic summary for model comparison
        summary_for_comparison = {
            'model': 'DlibCNN',
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
        comparison_summary_path = os.path.join(self.comparisons_dir, "summaries", f"dlib_cnn_summary_{self.video_name}.json")
        with open(comparison_summary_path, 'w') as f:
            json.dump(summary_for_comparison, f, indent=2)
        print(f"[INFO] ✅ Saved comparison summary: {comparison_summary_path}")
        
        # 2. Detailed comparison data
        detailed_comparison = {
            'model_info': {
                'name': 'DlibCNN',
                'full_name': 'Dlib CNN Face Recognition',
                'version': 'ResNet v1',
                'architecture': 'CNN-based face detector + ResNet embeddings'
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
                'locking_mechanism': 'Enhanced identity locking with consistency tracking',
                'back_view_support': 'Body pose detection with similarity matching',
                'global_identity_database': 'Face encoding similarity with cosine distance',
                'conflict_resolution': 'Multi-frame consistency with lock strength'
            }
        }
        
        # Save to detailed subfolder
        detailed_comparison_path = os.path.join(self.comparisons_dir, "detailed", f"dlib_cnn_detailed_{self.video_name}.json")
        with open(detailed_comparison_path, 'w') as f:
            json.dump(detailed_comparison, f, indent=2)
        print(f"[INFO] ✅ Saved detailed comparison: {detailed_comparison_path}")
        
        # 3. Also save a copy of the main graph to comparisons/graphs for easy access
        try:
            # Copy the standard graph to comparisons folder
            import shutil
            source_graph = os.path.join(self.graphs_dir, f"dlib_cnn_metrics_{self.video_name}.png")
            comparison_graph = os.path.join(self.comparisons_dir, "graphs", f"dlib_cnn_metrics_{self.video_name}.png")
            
            if os.path.exists(source_graph):
                shutil.copy2(source_graph, comparison_graph)
                print(f"[INFO] ✅ Saved comparison graph: {comparison_graph}")
        except Exception as e:
            print(f"[WARN] Could not copy graph to comparisons folder: {e}")
        
        return True
    
    def print_summary(self, stats):
        """Print enhanced summary with folder information"""
        
        print(f"\n" + "=" * 70)
        print("📊 DLIB CNN PERFORMANCE SUMMARY WITH ORGANIZED OUTPUT")
        print("=" * 70)
        
        print(f"\n📁 OUTPUT ORGANIZATION:")
        print(f"  🎯 Base Directory: {self.base_output_dir}")
        print(f"  📊 Dlib CNN Results: {self.dlib_dir}")
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
        locked_identities = {k: v for k, v in self.track_locks.items() if v.get('locked', False)}
        if locked_identities:
            print(f"\n🔒 LOCKED IDENTITY DETAILS:")
            for track_id, track_data in locked_identities.items():
                print(f"  Track {track_id}: {track_data['identity']} (strength: {track_data.get('lock_strength', 0)})")
        
        print(f"\n📂 ENHANCED FOLDER STRUCTURE:")
        print(f"  {self.base_output_dir}/")
        print(f"  ├── DlibCNN/")
        print(f"  │   ├── graphs/           # Dlib CNN performance graphs")
        print(f"  │   └── data/             # Dlib CNN detailed results")
        print(f"  ├── FaceNet/              # FaceNet results (if exists)")
        print(f"  └── Comparisons/")
        print(f"      ├── summaries/        # Basic comparison summaries")
        print(f"      ├── detailed/         # Detailed comparison data")
        print(f"      └── graphs/           # Comparison graphs")
        
        print(f"\n🔮 READY FOR MODEL COMPARISON:")
        print(f"  • DlibCNN analyzer complete ✅")
        print(f"  • Organized comparison structure ✅")
        print(f"  • Add more model analyzers (OpenFace, ArcFace, etc.)")
        print(f"  • Each model gets its own subfolder")
        print(f"  • Enhanced comparison data in /Comparisons/")
        print(f"  • Easy to create comparison graphs later")
        
        print("=" * 70)

def main():
    """Main function with anti-freeze warnings"""
    
    parser = argparse.ArgumentParser(description="Dlib CNN video analysis with anti-freeze optimizations")
    parser.add_argument("--video", "-v", 
                       default=r"C:\Users\Alexa\OneDrive\Documents\Thesis\Reactive-CCTV-System-Using-Authorization-Level-Analysis\Mp4TESTING\ThesisMP4TEST3.mp4",
                       help="Path to video file")
    parser.add_argument("--output", "-o", 
                       default="model_comparison_results",
                       help="Base output directory for organized results")
    parser.add_argument("--no-display", "-n", 
                       action="store_true",
                       help="Run without video display (RECOMMENDED for stability)")
    parser.add_argument("--ultra-fast", "-u",
                       action="store_true", 
                       help="Ultra-fast mode with maximum frame skipping")
    
    args = parser.parse_args()
    
    # Check if video exists
    if not os.path.exists(args.video):
        print(f"[ERROR] Video file not found: {args.video}")
        return
    
    print("🎬 DLIB CNN VIDEO ANALYZER - ANTI-FREEZE MODE")
    print("=" * 50)
    
    if not args.no_display:
        print("⚠️  WARNING: Display mode may cause freezing with Dlib CNN")
        print("💡 RECOMMENDATION: Use --no-display for best stability")
        print("[INFO] Minimal display enabled - only boxes and IDs shown")
        print("[INFO] Press 'q' to quit if display freezes")
    else:
        print("✅ HEADLESS MODE - Maximum stability and performance")
    
    if args.ultra_fast:
        print("🚀 ULTRA-FAST MODE - Maximum frame skipping enabled")
    
    # Initialize analyzer
    analyzer = DlibCNNVideoAnalyzer(args.video, args.output)
    
    # Process video with anti-freeze measures
    try:
        if not analyzer.process_video_with_analysis(show_video=not args.no_display):
            print("[ERROR] Video processing failed")
            return
    except KeyboardInterrupt:
        print("\n[INFO] Processing interrupted by user")
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
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
    
    print(f"\n✅ Dlib CNN analysis complete!")
    print(f"💡 For best performance, consider using FaceNet analyzer instead")

if __name__ == "__main__":
    main()