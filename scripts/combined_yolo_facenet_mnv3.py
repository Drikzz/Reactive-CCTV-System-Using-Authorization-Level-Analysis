import argparse
import importlib.util
import sys
import time
from pathlib import Path
import cv2
import numpy as np
from collections import defaultdict, deque
import os
from datetime import datetime

# Resolve repo root
REPO_ROOT = Path(__file__).resolve().parents[1]

# -------------------- CONFIGURATION --------------------
# Set this to True to use webcam, False to require --video argument
USE_WEBCAM = True  # Change this to True to use webcam by default
WEBCAM_INDEX = 0   # Default webcam index (0 = first camera, 1 = second, etc.)
# -------------------------------------------------------

def load_module_from_path(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

class EventLogger:
    """Handles logging of security events to file"""
    def __init__(self, log_dir="logs", log_filename="log.txt"):
        self.log_dir = Path(REPO_ROOT) / log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / log_filename
        
        # Track person states to detect entry/exit
        self.person_in_room = {}  # track_id: (name, auth_level, last_seen_frame)
        self.last_logged_behavior = {}  # track_id: behavior_name
        
        # Create log file if it doesn't exist
        if not self.log_file.exists():
            with open(self.log_file, 'w') as f:
                f.write("=== CCTV Security Log ===\n")
                f.write(f"Log started: {self._get_timestamp()}\n")
                f.write("=" * 50 + "\n\n")
        
        print(f"[INFO] Event logger initialized. Log file: {self.log_file}")
    
    def _get_timestamp(self):
        """Get formatted timestamp"""
        return datetime.now().strftime("%m-%d-%y %I:%M%p")
    
    def _write_log(self, message):
        """Write message to log file"""
        try:
            with open(self.log_file, 'a') as f:
                f.write(f"{message}\n")
        except Exception as e:
            print(f"[ERROR] Failed to write to log: {e}")
    
    def log_person_entry(self, track_id, name, auth_level):
        """Log when a person enters the room"""
        timestamp = self._get_timestamp()
        
        if auth_level == "Authorized":
            message = f"{timestamp} {name} entered the room"
        elif auth_level == "Partially Authorized":
            message = f"{timestamp} {name} has entered the room"
        else:  # Unauthorized
            message = f"{timestamp} Unrecognized person has entered the room"
        
        self._write_log(message)
        print(f"[LOG] {message}")
    
    def log_person_exit(self, track_id, name, auth_level):
        """Log when a person exits the room"""
        timestamp = self._get_timestamp()
        
        if auth_level == "Authorized":
            message = f"{timestamp} {name} left the room"
        elif auth_level == "Partially Authorized":
            message = f"{timestamp} {name} has left the room"
        else:  # Unauthorized
            message = f"{timestamp} Unrecognized person has left the room"
        
        self._write_log(message)
        print(f"[LOG] {message}")
    
    def log_behavior(self, track_id, name, behavior):
        """Log behavior for partially authorized persons"""
        timestamp = self._get_timestamp()
        
        # Map behavior names to readable actions
        behavior_actions = {
            "Neutral": "NEUTRAL BEHAVIOR",
            "Suspicious": "SUSPICIOUS BEHAVIOR",
            "holding-object": "HOLDING OBJECT",
            "using-computer": "USING COMPUTER",
            "opening-cabinet": "OPENING CABINET",
            # Add more mappings as needed
        }
        
        action = behavior_actions.get(behavior, behavior.upper())
        message = f"{timestamp} {name} is {action}"
        
        self._write_log(message)
        print(f"[LOG] {message}")
    
    def update_tracking(self, tracks_data, frame_idx, frames_before_exit=90):
        """
        Update person tracking and log entries/exits
        
        Args:
            tracks_data: List of detected tracks with their info
            frame_idx: Current frame index
            frames_before_exit: Number of frames to wait before logging exit
        """
        current_track_ids = set()
        
        # Process current tracks
        for td in tracks_data:
            track_id = td["track_id"]
            name = td["identity"]
            auth_level = td["authorization"]
            behavior = td.get("behavior", "N/A")
            
            current_track_ids.add(track_id)
            
            # Check if this is a new person
            if track_id not in self.person_in_room:
                # Log entry
                self.log_person_entry(track_id, name, auth_level)
                self.person_in_room[track_id] = (name, auth_level, frame_idx)
                self.last_logged_behavior[track_id] = "Neutral"
            else:
                # Update last seen frame
                stored_name, stored_auth, _ = self.person_in_room[track_id]
                self.person_in_room[track_id] = (name, auth_level, frame_idx)
                
                # Log behavior for Partially Authorized (only if behavior changed and not Neutral)
                if auth_level == "Partially Authorized" and behavior != "N/A":
                    last_behavior = self.last_logged_behavior.get(track_id, "Neutral")
                    
                    # Only log if behavior changed and is not Neutral
                    if behavior != "Neutral" and behavior != last_behavior:
                        self.log_behavior(track_id, name, behavior)
                        self.last_logged_behavior[track_id] = behavior
        
        # Check for people who left (not seen recently)
        to_remove = []
        for track_id, (name, auth_level, last_frame) in self.person_in_room.items():
            if track_id not in current_track_ids:
                frames_absent = frame_idx - last_frame
                
                # If absent for enough frames, log exit
                if frames_absent >= frames_before_exit:
                    self.log_person_exit(track_id, name, auth_level)
                    to_remove.append(track_id)
        
        # Remove exited people from tracking
        for track_id in to_remove:
            del self.person_in_room[track_id]
            if track_id in self.last_logged_behavior:
                del self.last_logged_behavior[track_id]

class CombinedYOLOFaceBehavior:
    """
    Combine:
     - YOLOv8 + ByteTrack -> detection + tracking
     - MobileNetV2 -> behavior classification (per-track caching + smoothing) - ONLY for Partially Authorized
     - FaceNet (from facenet_main.py) -> face recognition per track
     - Authorization logic based on recognized identity
     - Event logging system
    """
    def __init__(self, yolo_path, mobilenet_path, facenet_main_path, device=None, tracker_cfg="bytetrack.yaml",
                 reclass_interval=10, smooth_window=7, min_class_conf=0.6, recog_interval=30,
                 iou_update_threshold=0.3, centroid_update_px=80, identity_lock_frames=120, identity_lock_conf=0.75,
                 frame_skip=2, use_half_precision=True, resize_factor=0.5,
                 min_face_size=60, face_quality_threshold=0.3, consensus_window=3,
                 body_memory_frames=300, min_face_detections=2, identity_persistence_frames=900,
                 enable_logging=True):
        
        # Initialize event logger
        self.enable_logging = enable_logging
        if self.enable_logging:
            self.event_logger = EventLogger()
        
        # Check if MobileNetV2 tracker exists, if not create inline
        yolo_mnv_path = Path(REPO_ROOT) / "behavior_recognition" / "MobileNetV2" / "yolo_mobilenet_tracker.py"
        
        if yolo_mnv_path.exists():
            # Load existing tracker
            tracker_mod = load_module_from_path("yolo_mnv_tracker_mod", yolo_mnv_path)
            TrackerClass = getattr(tracker_mod, "YOLOMobileNetTracker")
            
            if Path(yolo_path).exists():
                yolo_model_path = str(yolo_path)
            else:
                yolo_model_path = str(yolo_path)

            if not Path(yolo_model_path).exists():
                print(f"[INFO] YOLO model not found at '{yolo_model_path}'. Falling back to hub model 'yolov8n.pt' (will download automatically).")
                yolo_model_path = "yolov8n.pt"
     
            self.tracker = TrackerClass(
                yolo_model_path=yolo_model_path,
                mobilenet_model_path=str(mobilenet_path),
                class_names=None,
                tracker=tracker_cfg,
                device=device,
                reclass_interval=reclass_interval,
                smoothing_window=smooth_window
            )
            self.tracker.min_class_conf = float(min_class_conf)

            # Load facenet_main module (for recognize_face_in_crop)
            facenet_path = Path(facenet_main_path)
            facenet_mod = load_module_from_path("facenet_main_mod", facenet_path)
            self.recognize_face_fn = getattr(facenet_mod, "recognize_face_in_crop", None)
            if self.recognize_face_fn is None:
                raise RuntimeError("facenet_main module does not expose recognize_face_in_crop()")

            self.display_title = "YOLO + MobileNetV2 + FaceNet (Combined)"

            # add facenet track state
            self.track_identities = {}
            self.track_face_history = defaultdict(lambda: deque(maxlen=300))
            self.track_body_history = defaultdict(lambda: deque(maxlen=60))
            self.track_last_face_frame = {}

            # load facenet module object
            self.facenet_mod = facenet_mod

            # Face recognition control
            self.recog_interval = int(recog_interval)
            self.track_last_recog_frame = {}
            self.identity_cache = {}

            # Continuity tracking
            self.track_last_bbox = {}
            self.track_identity_lock = {}
            self.iou_update_threshold = float(iou_update_threshold)
            self.centroid_update_px = float(centroid_update_px)
            self.identity_lock_frames = int(identity_lock_frames)
            self.identity_lock_conf = float(identity_lock_conf)

            # Authorization mapping
            self.authorization_map = {
                "myke": "Partially Authorized",
                "dean": "Authorized",
                "art": "Authorized",
            }

            # Performance settings
            self.frame_skip = int(frame_skip)
            self.use_half_precision = use_half_precision and device == "cuda"
            self.resize_factor = float(resize_factor)
            
            if self.use_half_precision:
                print("[INFO] Using FP16 (half precision) for YOLO (MobileNet stays FP32)")
            else:
                self.tracker.use_half = False

            # Recognition quality settings
            self.min_face_size = int(min_face_size)
            self.face_quality_threshold = float(face_quality_threshold)
            self.consensus_window = int(consensus_window)
            self.track_recognition_history = defaultdict(lambda: deque(maxlen=consensus_window))

            # Persistent tracking parameters
            self.body_memory_frames = int(body_memory_frames)
            self.min_face_detections = int(min_face_detections)
            self.identity_persistence_frames = int(identity_persistence_frames)
            
            # Track persistent identity state
            self.track_persistent_identity = {}
            self.track_body_features = defaultdict(lambda: deque(maxlen=30))
            
            # Track spatial position history
            self.track_position_history = defaultdict(lambda: deque(maxlen=60))
            self.track_size_history = defaultdict(lambda: deque(maxlen=30))

            # Enhanced back view tracking
            self.back_view_tolerance_frames = 600
            self.body_match_threshold = 0.5
            self.extended_memory_frames = 300
            self.min_body_match_confidence = 0.4
            self.pose_history_length = 30
            
            # Track pose history
            self.track_pose_history = defaultdict(lambda: deque(maxlen=self.pose_history_length))
            
        else:
            # Create inline tracker
            print("[INFO] MobileNetV2 tracker not found, creating inline implementation...")
            self._create_inline_tracker(yolo_path, mobilenet_path, device, tracker_cfg, reclass_interval, smooth_window)
            
            # Initialize other components after inline tracker
            # Load facenet_main module (for recognize_face_in_crop)
            facenet_path = Path(facenet_main_path)
            facenet_mod = load_module_from_path("facenet_main_mod", facenet_path)
            self.recognize_face_fn = getattr(facenet_mod, "recognize_face_in_crop", None)
            if self.recognize_face_fn is None:
                raise RuntimeError("facenet_main module does not expose recognize_face_in_crop()")

            self.display_title = "YOLO + MobileNetV2 + FaceNet (Combined)"

            # add facenet track state
            self.track_identities = {}
            self.track_face_history = defaultdict(lambda: deque(maxlen=300))
            self.track_body_history = defaultdict(lambda: deque(maxlen=60))
            self.track_last_face_frame = {}

            # load facenet module object
            self.facenet_mod = facenet_mod

            # Face recognition control
            self.recog_interval = int(recog_interval)
            self.track_last_recog_frame = {}
            self.identity_cache = {}

            # Continuity tracking
            self.track_last_bbox = {}
            self.track_identity_lock = {}
            self.iou_update_threshold = float(iou_update_threshold)
            self.centroid_update_px = float(centroid_update_px)
            self.identity_lock_frames = int(identity_lock_frames)
            self.identity_lock_conf = float(identity_lock_conf)

            # Authorization mapping
            self.authorization_map = {
                "myke": "Partially Authorized",
                "dean": "Authorized",
                "art": "Authorized",
            }

            # Performance settings
            self.frame_skip = int(frame_skip)
            self.use_half_precision = use_half_precision and device == "cuda"
            self.resize_factor = float(resize_factor)
            
            if self.use_half_precision:
                print("[INFO] Using FP16 (half precision) for YOLO (MobileNet stays FP32)")

            # Recognition quality settings
            self.min_face_size = int(min_face_size)
            self.face_quality_threshold = float(face_quality_threshold)
            self.consensus_window = int(consensus_window)
            self.track_recognition_history = defaultdict(lambda: deque(maxlen=consensus_window))

            # Persistent tracking parameters
            self.body_memory_frames = int(body_memory_frames)
            self.min_face_detections = int(min_face_detections)
            self.identity_persistence_frames = int(identity_persistence_frames)
            
            # Track persistent identity state
            self.track_persistent_identity = {}
            self.track_body_features = defaultdict(lambda: deque(maxlen=30))
            
            # Track spatial position history
            self.track_position_history = defaultdict(lambda: deque(maxlen=60))
            self.track_size_history = defaultdict(lambda: deque(maxlen=30))

            # Enhanced back view tracking
            self.back_view_tolerance_frames = 600
            self.body_match_threshold = 0.5
            self.extended_memory_frames = 300
            self.min_body_match_confidence = 0.4
            self.pose_history_length = 30
            
            # Track pose history
            self.track_pose_history = defaultdict(lambda: deque(maxlen=self.pose_history_length))

    def get_authorization_level(self, identity_name):
        """Determine authorization level based on identity."""
        if identity_name == "Unknown":
            return "Unauthorized"
        
        name_lower = identity_name.lower()
        
        if name_lower in self.authorization_map:
            return self.authorization_map[name_lower]
        else:
            return "Unauthorized"

    def get_authorization_color(self, auth_level):
        """Get color for authorization level visualization."""
        color_map = {
            "Authorized": (0, 255, 0),
            "Partially Authorized": (0, 165, 255),
            "Unauthorized": (0, 0, 255)
        }
        return color_map.get(auth_level, (128, 128, 128))

    def _estimate_blur(self, image):
        """Estimate image blur using Laplacian variance"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    def _get_consensus_identity(self, track_id):
        """Get most frequent identity from recent recognitions"""
        history = self.track_recognition_history[track_id]
        if not history:
            return "Unknown", 0.0
        
        from collections import Counter
        names = [name for name, conf in history if name != "Unknown"]
        if not names:
            return "Unknown", 0.0
        
        name_counts = Counter(names)
        most_common_name = name_counts.most_common(1)[0][0]
        
        confs = [conf for name, conf in history if name == most_common_name]
        avg_conf = sum(confs) / len(confs) if confs else 0.0
        
        return most_common_name, avg_conf

    def _extract_body_features(self, crop):
        """Extract simple color histogram features from body crop"""
        if crop.size == 0:
            return None
        
        crop_resized = cv2.resize(crop, (64, 128))
        
        hsv = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2HSV)
        hist_h = cv2.calcHist([hsv], [0], None, [32], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], None, [32], [0, 256])
        hist_v = cv2.calcHist([hsv], [2], None, [16], [0, 256])
        
        cv2.normalize(hist_h, hist_h)
        cv2.normalize(hist_s, hist_s)
        cv2.normalize(hist_v, hist_v)
        
        features = np.concatenate([hist_h.flatten(), hist_s.flatten(), hist_v.flatten()])
        return features

    def _compare_body_features(self, feat1, feat2):
        """Compare two body feature vectors using correlation"""
        if feat1 is None or feat2 is None:
            return 0.0
        
        correlation = np.corrcoef(feat1, feat2)[0, 1]
        return max(0.0, correlation)

    def detect_person_pose_from_body(self, person_crop):
        """Detect if person is facing away based on body characteristics"""
        try:
            if person_crop is None or person_crop.size == 0:
                return "unknown", 0.0
            
            h, w = person_crop.shape[:2]
            if h < 80 or w < 40:
                return "frontal", 0.5
            
            gray = cv2.cvtColor(person_crop, cv2.COLOR_BGR2GRAY)
            
            head_region = gray[:h//3, :]
            torso_region = gray[h//3:2*h//3, :]
            
            head_edges = cv2.Canny(head_region, 30, 100)
            head_edge_density = np.count_nonzero(head_edges) / max(head_edges.size, 1)
            
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
                
        except Exception as e:
            return "frontal", 0.5

    def calculate_body_similarity(self, template_crop, current_crop):
        """Calculate body similarity using HSV histograms"""
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

    def _update_persistent_identity(self, track_id, identity_name, identity_conf, has_face_detection, frame_idx, person_crop=None):
        """Enhanced update with back-view tracking"""
        track_id = int(track_id)
        
        pose = "frontal"
        pose_confidence = 0.5
        if person_crop is not None and person_crop.size > 0:
            pose, pose_confidence = self.detect_person_pose_from_body(person_crop)
            self.track_pose_history[track_id].append((pose, pose_confidence, frame_idx))
        
        if track_id not in self.track_persistent_identity:
            self.track_persistent_identity[track_id] = {
                "name": identity_name,
                "confidence": identity_conf,
                "frames_without_face": 0 if has_face_detection else 1,
                "total_face_detections": 1 if has_face_detection and identity_name != "Unknown" else 0,
                "last_seen_frame": frame_idx,
                "locked": False,
                "consecutive_back_frames": 0,
                "body_template": None,
                "max_confidence_seen": identity_conf,
                "lock_strength": 0.0,
                "unlock_threshold": 600
            }
            return identity_name, identity_conf
        
        persistent = self.track_persistent_identity[track_id]
        
        if has_face_detection and identity_name != "Unknown":
            persistent["total_face_detections"] += 1
            persistent["frames_without_face"] = 0
            persistent["consecutive_back_frames"] = 0
            
            if persistent["total_face_detections"] >= self.min_face_detections and identity_conf >= self.identity_lock_conf:
                persistent["locked"] = True
                persistent["name"] = identity_name
                persistent["confidence"] = identity_conf
                persistent["last_seen_frame"] = frame_idx
                persistent["max_confidence_seen"] = max(persistent["max_confidence_seen"], identity_conf)
                persistent["lock_strength"] = min(1.0, persistent["total_face_detections"] / 10.0) * identity_conf
                
                if person_crop is not None:
                    persistent["body_template"] = person_crop.copy()
                
                if identity_conf > 0.9:
                    persistent["unlock_threshold"] = 1800
                elif identity_conf > 0.8:
                    persistent["unlock_threshold"] = 900
                else:
                    persistent["unlock_threshold"] = 600
                
                print(f"[LOCK] Track {track_id} locked as '{identity_name}' (conf={identity_conf:.3f}, detections={persistent['total_face_detections']})")
                return identity_name, identity_conf
            
            if identity_name == persistent["name"] or identity_conf > persistent["confidence"]:
                persistent["name"] = identity_name
                persistent["confidence"] = max(identity_conf, persistent["confidence"])
                persistent["last_seen_frame"] = frame_idx
                
                if person_crop is not None:
                    persistent["body_template"] = person_crop.copy()
                
                if persistent["locked"]:
                    persistent["lock_strength"] = min(1.0, persistent["lock_strength"] * 1.01)
                
                return identity_name, identity_conf
        
        else:
            persistent["frames_without_face"] += 1
            
            if pose in ['back_view', 'partial_back']:
                persistent["consecutive_back_frames"] += 1
                
                if persistent["locked"] and persistent["name"] != "Unknown":
                    decay_rate = 0.9995
                    
                    if persistent["body_template"] is not None and person_crop is not None:
                        body_similarity = self.calculate_body_similarity(persistent["body_template"], person_crop)
                        
                        if body_similarity > self.body_match_threshold:
                            persistent["confidence"] = min(1.0, persistent["confidence"] * 1.005)
                            decay_rate = 0.9998
                            
                            if persistent["consecutive_back_frames"] % 60 == 0:
                                print(f"[BODY_MATCH] Track {track_id}: Maintaining {persistent['name']} via body match ({body_similarity:.3f}) - {persistent['consecutive_back_frames']} back frames")
                    
                    persistent["confidence"] *= decay_rate
                    
                    if (persistent["consecutive_back_frames"] > persistent["unlock_threshold"] and 
                        persistent["confidence"] < 0.1 and 
                        persistent["lock_strength"] < 0.3):
                        
                        persistent["locked"] = False
                        print(f"[UNLOCK] Track {track_id} unlocked after {persistent['consecutive_back_frames']} back frames")
                else:
                    persistent["confidence"] *= 0.985
            
            else:
                persistent["consecutive_back_frames"] = 0
                
                if persistent["locked"] and persistent["frames_without_face"] < persistent["unlock_threshold"]:
                    persistent["confidence"] *= 0.998
                    
                    if persistent["frames_without_face"] % 60 == 0:
                        print(f"[NO_FACE] Track {track_id}: Maintaining {persistent['name']} during face loss ({persistent['frames_without_face']} frames)")
                else:
                    if persistent["locked"] and persistent["frames_without_face"] > persistent["unlock_threshold"]:
                        persistent["locked"] = False
                        print(f"[UNLOCK] Track {track_id} unlocked after {persistent['frames_without_face']} frames without face")
                    
                    persistent["confidence"] *= 0.995
        
        if persistent["locked"]:
            min_locked_confidence = max(0.2, persistent["lock_strength"] * 0.5)
            persistent["confidence"] = max(persistent["confidence"], min_locked_confidence)
        
        if person_crop is not None and person_crop.size > 0:
            self.track_body_features[track_id].append(self._extract_body_features(person_crop))
        
        return persistent["name"], persistent["confidence"]

    def _should_run_recognition(self, track_id, frame_idx, auth_level):
        """Adaptive recognition interval based on track stability and authorization"""
        last_recog = self.track_last_recog_frame.get(int(track_id), -999)
        frames_since = frame_idx - last_recog
        
        if last_recog == -999:
            return True
        
        persistent = self.track_persistent_identity.get(int(track_id))
        if persistent and persistent.get("locked", False):
            return frames_since >= self.recog_interval * 5
        
        if auth_level == "Unauthorized":
            return frames_since >= max(15, self.recog_interval // 2)
        
        stability = self.track_stability_count.get(int(track_id), 0)
        if stability > 30:
            return frames_since >= self.recog_interval * 3
        
        return frames_since >= self.recog_interval

    def process_video(self, video_source, display=True, save_output=None, conf_threshold=0.5, iou_threshold=0.7):
        # Open video
        if video_source == "webcam":
            # Try to open the default webcam index first
            cap = cv2.VideoCapture(WEBCAM_INDEX)
            
            # If default fails, try to find any available camera
            if not cap.isOpened():
                print(f"[WARNING] Could not open webcam at index {WEBCAM_INDEX}")
                print("[INFO] Searching for available cameras...")
                
                # Try indices 0-10
                for idx in range(11):
                    if idx == WEBCAM_INDEX:
                        continue  # Already tried this one
                    print(f"[INFO] Trying camera index {idx}...")
                    test_cap = cv2.VideoCapture(idx)
                    if test_cap.isOpened():
                        # Test if we can actually read a frame
                        ret, _ = test_cap.read()
                        if ret:
                            cap = test_cap
                            print(f"[INFO] Successfully opened camera at index {idx}")
                            break
                        else:
                            test_cap.release()
                    else:
                        test_cap.release()
            is_webcam = True
        elif isinstance(video_source, int):
            # Direct camera index specified
            cap = cv2.VideoCapture(video_source)
            is_webcam = True
        else:
            cap = cv2.VideoCapture(str(video_source))
            is_webcam = False

        if not cap.isOpened():
            raise ValueError(f"Could not open video source: {video_source}")

        fps = int(cap.get(cv2.CAP_PROP_FPS) or 25)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        out_writer = None
        if save_output:
            out_path = Path(save_output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out_writer = cv2.VideoWriter(str(out_path), fourcc, max(1, fps), (width, height))

        frame_idx = 0
        fps_window = []
        print(f"Processing {video_source} ({width}x{height} @ {fps}fps)")

        if self.resize_factor < 1.0:
            process_width = int(width * self.resize_factor)
            process_height = int(height * self.resize_factor)
            print(f"[INFO] Processing at {process_width}x{process_height} (resize_factor={self.resize_factor})")
        else:
            process_width, process_height = width, height

        # Adaptive tracking
        self.track_stability_count = defaultdict(int)
        self.track_auth_stable = defaultdict(int)

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_idx += 1
                
                if frame_idx % self.frame_skip != 0:
                    if display and 'annotated' in locals():
                        cv2.imshow(self.display_title, annotated)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break
                    continue
                
                t0 = time.time()
                
                if self.resize_factor < 1.0:
                    process_frame = cv2.resize(frame, (process_width, process_height), interpolation=cv2.INTER_LINEAR)
                    scale_x = width / process_width
                    scale_y = height / process_height
                else:
                    process_frame = frame
                    scale_x = scale_y = 1.0

                results = self.tracker.yolo.track(
                    process_frame,
                    persist=True,
                    tracker=self.tracker.tracker_name,
                    classes=[0],
                    conf=conf_threshold,
                    iou=iou_threshold,
                    verbose=False,
                    half=self.use_half_precision,
                    device=self.tracker.device
                )

                tracks_data = []
                if results and results[0].boxes is not None:
                    boxes = results[0].boxes
                    if boxes.id is not None:
                        track_ids = boxes.id.cpu().numpy().astype(int)
                        bboxes = boxes.xyxy.cpu().numpy().astype(int)
                        
                        for bbox, track_id in zip(bboxes, track_ids):
                            x1, y1, x2, y2 = bbox
                            x1, y1 = int(x1 * scale_x), int(y1 * scale_y)
                            x2, y2 = int(x2 * scale_x), int(y2 * scale_y)
                            x1, y1 = max(0, x1), max(0, y1)
                            x2, y2 = min(width, x2), min(height, y2)
                            
                            if x2 <= x1 or y2 <= y1:
                                continue

                            if self.resize_factor < 1.0:
                                crop_x1, crop_y1 = int(x1 / scale_x), int(y1 / scale_y)
                                crop_x2, crop_y2 = int(x2 / scale_x), int(y2 / scale_y)
                                person_crop = process_frame[crop_y1:crop_y2, crop_x1:crop_x2]
                            else:
                                person_crop = frame[y1:y2, x1:x2]

                            body_features = self._extract_body_features(person_crop)
                            if body_features is not None:
                                self.track_body_features[int(track_id)].append(body_features)
                        
                            centroid = self._centroid((x1, y1, x2, y2))
                            self.track_position_history[int(track_id)].append(centroid)
                            self.track_size_history[int(track_id)].append((x2 - x1, y2 - y1))

                            last_bbox = self.track_last_bbox.get(int(track_id))
                            current_bbox = (x1, y1, x2, y2)
                            
                            if last_bbox is not None:
                                iou_val = self._iou(last_bbox, current_bbox)
                                if iou_val > 0.7:
                                    self.track_stability_count[int(track_id)] += 1
                                else:
                                    self.track_stability_count[int(track_id)] = 0
                        
                            persistent = self.track_persistent_identity.get(int(track_id))
                            if persistent:
                                cached_name = persistent["name"]
                                cached_conf = persistent["confidence"]
                            else:
                                cached_name, cached_conf = self.identity_cache.get(int(track_id), ("Unknown", 0.0))
                            current_auth = self.get_authorization_level(cached_name)
                            
                            do_recog = self._should_run_recognition(int(track_id), frame_idx, current_auth)

                            h, w = person_crop.shape[:2]
                            has_face_detection = False
                            
                            if h < 40 or w < 40:
                                identity_name = cached_name
                                identity_conf = cached_conf
                            elif do_recog:
                                face_result = self.recognize_face_fn(person_crop, process_frame if self.resize_factor < 1.0 else frame, current_bbox)
                                name = face_result.get("name", "Unknown")
                                conf = float(face_result.get("confidence", 0.0) or 0.0)
                                
                                has_face_detection = (name != "Unknown" and conf > 0.0)

                                self.track_recognition_history[int(track_id)].append((name, conf))
                                consensus_name, consensus_conf = self._get_consensus_identity(int(track_id))
                                
                                self.identity_cache[int(track_id)] = (consensus_name, consensus_conf)
                                self.track_last_recog_frame[int(track_id)] = frame_idx
                                
                                identity_name, identity_conf = self._update_persistent_identity(
                                    int(track_id), consensus_name, consensus_conf, has_face_detection, frame_idx, person_crop
                                )
                            else:
                                identity_name, identity_conf = self._update_persistent_identity(
                                    int(track_id), cached_name, cached_conf, False, frame_idx, person_crop
                                )

                            self.track_last_bbox[int(track_id)] = current_bbox

                            try:
                                face_result = {"name": identity_name, "confidence": identity_conf}
                                self.facenet_mod.update_track_identity(
                                    int(track_id), face_result, person_crop,
                                    self.track_identities, self.track_face_history,
                                    self.track_body_history, frame_idx
                                )
                            except Exception:
                                pass

                            auth_level = self.get_authorization_level(identity_name)

                            # BEHAVIOR CLASSIFICATION - ONLY for Partially Authorized
                            # Use the SAME logic as yolo_mobilenet_tracker_mnv3small.py
                            if auth_level == "Partially Authorized":
                                do_classify = self.tracker._should_reclassify(track_id, frame_idx)
                                if do_classify:
                                    # Classify using tracker's method
                                    class_res = self.tracker._classify_crop(person_crop)
                                    
                                    # Temporal smoothing (same as yolo_mobilenet_tracker)
                                    self.tracker.prob_history[track_id].append(class_res['probs'])
                                    hist = self.tracker.prob_history[track_id]
                                    
                                    if len(hist) == 1:
                                        smoothed_probs = hist[0]
                                    else:
                                        smoothed_probs = np.mean(np.stack(hist, axis=0), axis=0)
                                
                                    smoothed_class_id = int(np.argmax(smoothed_probs))
                                    smoothed_confidence = float(smoothed_probs[smoothed_class_id])
                                    
                                    # Apply min_class_conf threshold (same as yolo_mobilenet_tracker)
                                    if smoothed_confidence < self.tracker.min_class_conf:
                                        behavior_name = "Neutral"
                                        behavior_conf = smoothed_confidence
                                    else:
                                        behavior_name = self.tracker.class_names[smoothed_class_id]
                                        behavior_conf = smoothed_confidence
                                
                                    # Update cache
                                    self.tracker.classification_cache[track_id] = {
                                        "class_name": behavior_name,
                                        "confidence": behavior_conf,
                                        "last_frame": frame_idx
                                    }
                                else:
                                    # Use cached classification
                                    cached_beh = self.tracker.classification_cache.get(track_id, {"class_name": "Neutral", "confidence": 0.0})
                                    behavior_name = cached_beh["class_name"]
                                    behavior_conf = cached_beh["confidence"]
                            else:
                                # Skip behavior classification for Authorized/Unauthorized
                                behavior_name = "N/A"
                                behavior_conf = 0.0

                            tracks_data.append({
                                "track_id": int(track_id),
                                "bbox": current_bbox,
                                "behavior": behavior_name,
                                "behavior_conf": behavior_conf,
                                "identity": identity_name,
                                "identity_conf": identity_conf,
                                "authorization": auth_level
                            })

                # Update event logger with current tracks
                if self.enable_logging:
                    self.event_logger.update_tracking(tracks_data, frame_idx)

                # Annotate
                annotated = frame.copy()
                for td in tracks_data:
                    x1, y1, x2, y2 = td["bbox"]
                    tid = td["track_id"]
                    behavior = td["behavior"]
                    bconf = td["behavior_conf"]
                    ident = td["identity"]
                    iconf = td["identity_conf"]
                    auth = td["authorization"]

                    color = self.get_authorization_color(auth)
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                    
                    persistent = self.track_persistent_identity.get(int(tid))
                    lock_indicator = " [LOCKED]" if persistent and persistent.get("locked", False) else ""
                    
                    frames_no_face = persistent.get("frames_without_face", 0) if persistent else 0
                    face_detections = persistent.get("total_face_detections", 0) if persistent else 0
                    consecutive_back = persistent.get("consecutive_back_frames", 0) if persistent else 0
                    
                    pose_info = ""
                    if consecutive_back > 30:
                        pose_info = f" [BACK:{consecutive_back}f]"
                    elif frames_no_face > 15:
                        pose_info = f" [NO_FACE:{frames_no_face}f]"
                    
                    label = f"ID:{tid} {ident}{lock_indicator}{pose_info} ({iconf:.2f})"
                    
                    if auth == "Partially Authorized":
                        auth_label = f"{auth} | {behavior} ({bconf:.2f})"
                    else:
                        auth_label = f"{auth}"
                    
                    debug_label = f"NoFace:{frames_no_face} Det:{face_detections} Back:{consecutive_back}"
                    
                    cv2.putText(annotated, label, (x1+2, max(20, y1-20)), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                    
                    cv2.putText(annotated, auth_label, (x1+2, max(35, y1-6)), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    cv2.putText(annotated, debug_label, (x1+2, max(50, y2+15)), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,0), 1)

                if len(fps_window) > 0:
                    avg_fps = len(fps_window) / sum(fps_window)
                    cv2.putText(annotated, f"FPS: {avg_fps:.1f}", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                if display:
                    cv2.imshow(self.display_title, annotated)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break

                if out_writer:
                    out_writer.write(annotated)

                dt = time.time() - t0
                fps_window.append(dt)
                if len(fps_window) > 30:
                    fps_window.pop(0)

        finally:
            cap.release()
            if out_writer:
                out_writer.release()
            if display:
                cv2.destroyAllWindows()

    # Helper utils
    @staticmethod
    def _iou(boxA, boxB):
        if boxA is None or boxB is None:
            return 0.0
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interW = max(0, xB - xA)
        interH = max(0, yB - yA)
        interArea = interW * interH
        boxAArea = max(0, (boxA[2]-boxA[0])) * max(0, (boxA[3]-boxA[1]))
        boxBArea = max(0, (boxB[2]-boxB[0])) * max(0, (boxB[3]-boxB[1]))
        denom = float(boxAArea + boxBArea - interArea)
        return (interArea / denom) if denom > 0 else 0.0

    @staticmethod
    def _centroid(box):
        x1,y1,x2,y2 = box
        return ((x1+x2)/2.0, (y1+y2)/2.0)

    def _create_inline_tracker(self, yolo_path, mobilenet_path, device, tracker_cfg, reclass_interval, smooth_window):
        """Create inline MobileNetV2 tracker when separate file doesn't exist"""
        import torch
        import torchvision.transforms as transforms
        from torchvision.models import mobilenet_v2
        from ultralytics import YOLO
        
        # Initialize device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        
        # Create a simple tracker object
        class InlineTracker:
            def __init__(self, yolo_path, mobilenet_path, device):
                self.device = device
                self.yolo = YOLO(yolo_path if Path(yolo_path).exists() else "yolov8n.pt")
                self.tracker_name = tracker_cfg
                self.reclass_interval = reclass_interval
                self.smoothing_window = smooth_window
                self.min_class_conf = 0.6
                
                # Load MobileNetV2
                self.mobilenet = mobilenet_v2(pretrained=True)
                self.mobilenet.classifier = torch.nn.Sequential(
                    torch.nn.Dropout(0.2),
                    torch.nn.Linear(self.mobilenet.classifier[1].in_features, 4)
                )
                self.mobilenet.to(device)
                self.mobilenet.eval()
                
                self.num_classes = 4
                self.class_names = ["Normal", "Suspicious", "Aggressive", "Neutral"]
                
                # Tracking state
                self.classification_cache = {}
                self.prob_history = defaultdict(lambda: deque(maxlen=smooth_window))
                self.last_classification_frame = {}
                
                # Transforms
                self.transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            
            def _should_reclassify(self, track_id, frame_idx):
                last_frame = self.last_classification_frame.get(track_id, -999)
                return (frame_idx - last_frame) >= self.reclass_interval
            
            def _classify_crop(self, person_crop):
                try:
                    if person_crop is None or person_crop.size == 0:
                        return {"class_name": "Neutral", "confidence": 0.0, "probs": np.ones(self.num_classes) / self.num_classes}
                    
                    crop_rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
                    input_tensor = self.transform(crop_rgb).unsqueeze(0).to(self.device)
                    
                    with torch.no_grad():
                        outputs = self.mobilenet(input_tensor)
                        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
                    
                    class_id = int(np.argmax(probs))
                    confidence = float(probs[class_id])
                    class_name = self.class_names[class_id]
                    
                    return {"class_name": class_name, "confidence": confidence, "probs": probs}
                except Exception as e:
                    print(f"[ERROR] Classification failed: {e}")
                    return {"class_name": "Neutral", "confidence": 0.0, "probs": np.ones(self.num_classes) / self.num_classes}
        
        self.tracker = InlineTracker(yolo_path, mobilenet_path, device)
        print("[INFO] Inline MobileNetV2 tracker created successfully")

def main():
    parser = argparse.ArgumentParser(description="Combined YOLOv8 + MobileNetV2 + FaceNet pipeline with Authorization and Logging")
    
    # Make video argument optional if USE_WEBCAM is True
    if USE_WEBCAM:
        parser.add_argument("--video", type=str, help="Path to video file (optional, webcam will be used if not provided)", default=None)
    else:
        parser.add_argument("--video", type=str, help="Path to video file (or 'webcam')", required=True)
    
    parser.add_argument("--yolo-model", type=str, default="models/YOLOv8/yolov8n.pt")
    parser.add_argument("--mobilenet-model", type=str, default="models/mobilenetv2/mobilenet_feature_extraction.pth")
    parser.add_argument("--facenet-main", type=str, default=str(REPO_ROOT / "face_recognition" / "Facenet" / "facenet_main.py"))
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--save", type=str, help="Save annotated video to path")    
    parser.add_argument("--device", type=str, choices=["cuda","cpu"], help="Device to use")
    parser.add_argument("--min-class-conf", type=float, default=0.6, help="Minimum behavior classification confidence")
    
    # Performance parameters
    parser.add_argument("--frame-skip", type=int, default=2, help="Process every Nth frame")
    parser.add_argument("--resize-factor", type=float, default=0.5, help="Resize input frames (0.5 = half size)")
    parser.add_argument("--no-half", action="store_true", help="Disable FP16 half precision")
    
    # Recognition quality parameters
    parser.add_argument("--min-face-size", type=int, default=60, help="Minimum face crop size for recognition")
    parser.add_argument("--recog-interval", type=int, default=30, help="Frames between face recognition runs")
    parser.add_argument("--consensus-window", type=int, default=3, help="Number of frames for identity consensus")
    
    # Behavior classification parameters
    parser.add_argument("--reclass-interval", type=int, default=10, help="Re-classification interval in frames")
    parser.add_argument("--smooth-window", type=int, default=7, help="Temporal smoothing window for behavior")
    
    # Persistence parameters
    parser.add_argument("--identity-persistence", type=int, default=900, help="Frames to persist identity without face")
    parser.add_argument("--min-face-detections", type=int, default=2, help="Minimum face detections before locking")
    
    # Webcam parameters
    parser.add_argument("--use-webcam", action="store_true", help="Force use of webcam (overrides USE_WEBCAM setting)")
    parser.add_argument("--webcam-index", type=int, default=None, help="Webcam index to use (0=first camera, 1=second, etc.)")
    
    parser.add_argument("--no-logging", action="store_true", help="Disable event logging")
    
    args = parser.parse_args()

    # Override global webcam index if specified in arguments
    if args.webcam_index is not None:
        global WEBCAM_INDEX
        WEBCAM_INDEX = args.webcam_index
        print(f"[INFO] Using webcam index: {WEBCAM_INDEX}")

    # Determine video source with priority: command line > USE_WEBCAM config
    if args.use_webcam:
        video_src = "webcam"
        print("[INFO] Using webcam (forced by --use-webcam argument)")
    elif USE_WEBCAM and args.video is None:
        video_src = "webcam"
        print("[INFO] Using webcam (set by USE_WEBCAM configuration)")
    elif args.video == "webcam":
        video_src = "webcam"
        print("[INFO] Using webcam (specified in --video argument)")
    elif args.video and args.video.isdigit():
        # Support direct camera index as video argument
        video_src = int(args.video)
        print(f"[INFO] Using camera index: {video_src}")
    elif args.video:
        video_src = args.video
        print(f"[INFO] Using video file: {video_src}")
    else:
        print("[ERROR] No video source specified. Set USE_WEBCAM=True or provide --video argument")
        return

    comb = CombinedYOLOFaceBehavior(
        yolo_path=args.yolo_model,
        mobilenet_path=args.mobilenet_model,
        facenet_main_path=args.facenet_main,
        device=args.device,
        min_class_conf=args.min_class_conf,
        recog_interval=args.recog_interval,
        identity_lock_conf=0.75,
        frame_skip=args.frame_skip,
        use_half_precision=not args.no_half,
        resize_factor=args.resize_factor,
        min_face_size=args.min_face_size,
        consensus_window=args.consensus_window,
        reclass_interval=args.recog_interval,
        smooth_window=args.smooth_window,
        identity_persistence_frames=args.identity_persistence,
        min_face_detections=args.min_face_detections,
        enable_logging=not args.no_logging
    )
    comb.process_video(video_src, display=not args.no_display, save_output=args.save)

if __name__ == "__main__":
    main()